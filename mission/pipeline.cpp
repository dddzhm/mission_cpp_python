#include "pipeline.h"
#include <stdexcept>

Pipeline::Pipeline(const gpt_params &p): params(p), sparams(p.sparams), path_session(p.path_prompt_cache) {
    if (!params.tiktoken_path.empty() && !params.tiktoken_config.empty()) {
        tokenizer = mission::build_tokenizer(params.tiktoken_config, params.tiktoken_path);
    }
#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("pipeline", "log"));
    LOG_TEE("Log start\n");
    llama_log_set(llama_log_callback_logTee, nullptr);
#endif // LOG_DISABLE_LOGS

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_TEE("%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(nullptr);
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);
    LOG("%s: llama backend init\n", __func__);
    llama_backend_init(params.numa);

    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (sparams.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    if (model == nullptr) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return;
    }

    n_ctx_train = llama_n_ctx_train(model);
    n_ctx = (int)llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        LOG_TEE("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("%s\n", get_system_info(params).c_str());
    }

    if (!path_session.empty()) {
        LOG_TEE("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != nullptr) {
            std::fclose(fp);

            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_TEE("%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return;
            }
            session_tokens.resize(n_token_count_out);
            llama_set_rng_seed(ctx, params.seed);

            LOG_TEE("%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            LOG_TEE("%s: session file does not exist, will create\n", __func__);
        }
    }

    add_bos = llama_should_add_bos_token(model);
    LOG("add_bos: %d\n", add_bos);

    if (params.interactive_first || params.instruct || params.chatml || !params.prompt.empty() || session_tokens.empty()) {
        LOG("tokenize the prompt\n");
        if (params.chatml) {
            params.prompt = "<|im_start|>system\n" + params.prompt + "<|im_end|>";
        }
        if (tokenizer != nullptr) {
            embd_inp = tokenizer->encode(params.prompt, params.n_ctx);
        }else{
            embd_inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);
        }
    } else {
        LOG("use session tokens\n");
        embd_inp = session_tokens;
    }

    LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
    }

    if (ctx_guidance) {
        LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(sparams.cfg_negative_prompt));

        guidance_inp = ::llama_tokenize(ctx_guidance, sparams.cfg_negative_prompt, add_bos, true);
        LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_inp).c_str());

        std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);
        LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, original_inp).c_str());

        original_prompt_len = (int)original_inp.size();
        guidance_offset = (int)guidance_inp.size() - original_prompt_len;
        LOG("original_prompt_len: %s", log_tostr(original_prompt_len));
        LOG("guidance_offset:     %s", log_tostr(guidance_offset));
    }

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_TEE("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_TEE("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_TEE("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_TEE("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, (int)n_matching_session_tokens, -1);
    }

    LOGLN(
            "recalculate the cached logits (check): embd_inp.empty() %s, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu, embd_inp.size() %zu",
            log_tostr(embd_inp.empty()), n_matching_session_tokens, embd_inp.size(), session_tokens.size(), embd_inp.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOGLN("recalculate the cached logits (do): session_tokens.resize( %zu )", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct || params.chatml) {
        params.n_keep = (int)embd_inp.size();
    }

    if (tokenizer != nullptr) {
        inp_pfx = tokenizer->encode("\n\n### Instruction:\n\n", params.n_ctx);
        inp_sfx = tokenizer->encode("\n\n### Response:\n\n", params.n_ctx);
    }else{
        inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", add_bos, true);
        inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n", false, true);
    }

    LOG("inp_pfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_pfx).c_str());
    LOG("inp_sfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_sfx).c_str());

    // chatml prefix & suffix
    if (tokenizer != nullptr) {
        cml_pfx = tokenizer->encode("\n<|im_start|>user\n", params.n_ctx);
        cml_sfx = tokenizer->encode("<|im_end|>\n<|im_start|>assistant\n", params.n_ctx);
    }else{
        cml_pfx = ::llama_tokenize(ctx, "\n<|im_start|>user\n", add_bos, true);
        cml_sfx = ::llama_tokenize(ctx, "<|im_end|>\n<|im_start|>assistant\n", false, true);
    }

    LOG("cml_pfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, cml_pfx).c_str());
    LOG("cml_sfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, cml_sfx).c_str());

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive_first = true;
        params.antiprompt.emplace_back("### Instruction:\n\n");
    }
        // similar for chatml mode
    else if (params.chatml) {
        params.interactive_first = true;
        params.antiprompt.emplace_back("<|im_start|>user\n");
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_TEE("\n");
        LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (auto token : embd_inp) {
            LOG_TEE("%6d -> '%s'\n", token, llama_token_to_piece(ctx, token).c_str());
        }

        if (ctx_guidance) {
            LOG_TEE("\n");
            LOG_TEE("%s: negative prompt: '%s'\n", __func__, sparams.cfg_negative_prompt.c_str());
            LOG_TEE("%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
            for (auto token : guidance_inp) {
                LOG_TEE("%6d -> '%s'\n", token, llama_token_to_piece(ctx, token).c_str());
            }
        }

        if (params.n_keep > 0) {
            LOG_TEE("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_TEE("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_TEE("'\n");
        }
        LOG_TEE("\n");
    }
    if (params.interactive) {
        LOG_TEE("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_TEE("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    std::vector<int> tmp;
                    if (tokenizer != nullptr) {
                        tmp = tokenizer->encode(antiprompt, params.n_ctx);
                    }else{
                        tmp = ::llama_tokenize(ctx, antiprompt, false, true);
                    }
                    for (int i : tmp) {
                        LOG_TEE("%6d -> '%s'\n", i, llama_token_to_piece(ctx, i).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_TEE("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_TEE("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                std::vector<int> tmp;
                if (tokenizer != nullptr) {
                    tmp = tokenizer->encode(params.input_prefix, params.n_ctx);
                }else{
                    tmp = ::llama_tokenize(ctx, params.input_prefix, true, true);
                }
                for (int i : tmp) {
                    LOG_TEE("%6d -> '%s'\n", i, llama_token_to_piece(ctx, i).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_TEE("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                std::vector<int> tmp;
                if (tokenizer != nullptr) {
                    tmp = tokenizer->encode(params.input_suffix, params.n_ctx);
                }else{
                    tmp = ::llama_tokenize(ctx, params.input_suffix, false, true);
                }
                for (int i : tmp) {
                    LOG_TEE("%6d -> '%s'\n", i, llama_token_to_piece(ctx, i).c_str());
                }
            }
        }
    }
    LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)

    ga_n = params.grp_attn_n;
    ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
        LOG_TEE("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_TEE("\n\n");

    if (params.interactive) {
        const char *control_message;
        if (params.multiline_input) {
            control_message = " - To return control to LLaMa, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to LLaMa.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_TEE("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_TEE(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_TEE(       "%s\n", control_message);

        is_interacting = params.interactive_first;
    }

    need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
    n_remain             = params.n_predict;
    ctx_sampling         = llama_sampling_init(sparams);

    generator(true);
}

Pipeline::~Pipeline() {
    llama_print_timings(ctx);
    ::write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);

    if (ctx_guidance) { llama_free(ctx_guidance); }

    llama_free(ctx);
    llama_free_model(model);

    llama_sampling_free(ctx_sampling);
    llama_backend_free();

#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log end\n");
#endif // LOG_DISABLE_LOGS
}

void Pipeline::llama_log_callback_logTee(ggml_log_level level, const char *text, void *user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

gpt_params Pipeline::get_params() const {
    return params;
}

std::string Pipeline::generator(const bool& init_flag, const std::string& prompts) {
    if (prompts.empty() && !init_flag){
        return "未获得输入......";
    }
    if (!prompts.empty()){
        tokenize(prompts);
    }
    int init_count = 2;
    std::string result;
    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                std::cout << "\033[3;31m<<input too long: skipped " << skipped_tokens << " token" << (skipped_tokens != 1 ? "s" : "") << ">>\033[0m" << "\n";
            }

            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
                if (n_past + (int) embd.size() + std::max<int>(0, guidance_offset) > n_ctx) {
                    if (params.n_predict == -2) {
                        LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep - 1;
                    const int n_discard = n_left/2;

                    LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                        n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_kv_cache_seq_rm   (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
                    llama_kv_cache_seq_shift(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    if (ctx_guidance) {
                        n_past_guidance -= n_discard;
                    }

                    LOG("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);

                    LOG("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                    LOG("clear session path\n");
                    path_session.clear();
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG("\n");
                    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_kv_cache_seq_shift(ctx, 0, ga_i,                n_past,              ib*bd);
                    llama_kv_cache_seq_div  (ctx, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_kv_cache_seq_shift(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            // evaluate tokens in batches
            // embd is typically prepared beforehand to fit within a batch, but not always
            if (ctx_guidance) {
                int input_size;
                llama_token * input_buf;

                if (n_past_guidance < (int) guidance_inp.size()) {
                    // Guidance context should have the same data with these modifications:
                    //
                    // * Replace the initial prompt
                    // * Shift everything by guidance_offset
                    embd_guidance = guidance_inp;
                    if (embd.begin() + original_prompt_len < embd.end()) {
                        embd_guidance.insert(
                                embd_guidance.end(),
                                embd.begin() + original_prompt_len,
                                embd.end()
                        );
                    }

                    input_buf  = embd_guidance.data();
                    input_size = (int)embd_guidance.size();

                    LOG("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_guidance).c_str());
                } else {
                    input_buf  = embd.data();
                    input_size = (int)embd.size();
                }

                for (int i = 0; i < input_size; i += params.n_batch) {
                    int n_eval = std::min(input_size - i, params.n_batch);
                    if (llama_decode(ctx_guidance, llama_batch_get_one(input_buf + i, n_eval, n_past_guidance, 0))) {
                        LOG_TEE("%s : failed to eval\n", __func__);
                        return "failed to eval";
                    }

                    n_past_guidance += n_eval;
                }
            }
            for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                int n_eval = (int) embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return "failed to eval";
                }

                n_past += n_eval;

                LOG("n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = (int)session_tokens.size();
            }
        }

        embd.clear();
        embd_guidance.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG("saved session to %s\n", path_session.c_str());
            }

            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

            embd.push_back(id);
            embd_list.push_back(id);

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        if (!params.no_streaming){
            // display text
            if (input_echo && display) {
                auto t1 = tokenizer->decode(embd);
                result += t1;
            }
            // reset color to default if there is no pending user input
            if (input_echo && (int) embd_inp.size() == n_consumed) {
                display = true;
            }
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                                              ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                                              : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of text token in interactive mode
            if (llama_sampling_last(ctx_sampling) == llama_token_eos(model)) {
                LOG("found EOS token\n");

                if (params.no_streaming) {
                    // display text
                    if (input_echo && display) {
                        auto logit_string = tokenizer->decode(embd_list);
                        result = logit_string;
                    }
                    // reset color to default if there is no pending user input
                    if (input_echo && (int) embd_inp.size() == n_consumed) {
                        display = true;
                    }
                }

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        std::vector<llama_token> first_antiprompt;
                        if (tokenizer != nullptr) {
                            first_antiprompt = tokenizer->encode(params.antiprompt.front(), params.n_ctx);
                        }else{
                            first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false, true);
                        }
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    is_interacting = true;
                    printf("\n");
                } else if (params.instruct || params.chatml) {
                    is_interacting = true;
                }
                break;
            }

            if (init_flag && --init_count <= 0) {
                break;
            }

            reset_ctx_sampling();
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos(model) && !(params.instruct || params.interactive || params.chatml)) {
            LOG_TEE(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }
    return result;
}

void Pipeline::tokenize(std::basic_string<char> prompts) {
    if (n_past > 0 && is_interacting) {
        if (params.input_prefix_bos) {
            LOG("adding input prefix BOS token\n");
            embd_inp.push_back(llama_token_bos(model));
        }

        if (!params.input_prefix.empty()) {
            LOG("appending input prefix: '%s'\n", params.input_prefix.c_str());
        }

        if (prompts.length() > 1) {
            // append input suffix if any
            if (!params.input_suffix.empty()) {
                LOG("appending input suffix: '%s'\n", params.input_suffix.c_str());
            }

            LOG("buffer: '%s'\n", prompts.c_str());

            const size_t original_size = embd_inp.size();

            // instruct mode: insert instruction prefix
            if (params.instruct && !is_antiprompt) {
                LOG("inserting instruction prefix\n");
                n_consumed = (int)embd_inp.size();
                embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
            }
            // chatml mode: insert user chat prefix
            if (params.chatml && !is_antiprompt) {
                LOG("inserting chatml prefix\n");
                n_consumed = (int)embd_inp.size();
                embd_inp.insert(embd_inp.end(), cml_pfx.begin(), cml_pfx.end());
            }
            if (params.escape) {
                process_escapes(prompts);
            }

            const auto line_pfx = ::llama_tokenize(ctx, params.input_prefix, false, true);
            const auto line_inp = ::llama_tokenize(ctx, prompts,              false, false);
            const auto line_sfx = ::llama_tokenize(ctx, params.input_suffix, false, true);
            LOG("input tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, line_inp).c_str());

            embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
            embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
            embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

            // instruct mode: insert response suffix
            if (params.instruct) {
                LOG("inserting instruction suffix\n");
                embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
            }
            // chatml mode: insert assistant chat suffix
            if (params.chatml) {
                LOG("inserting chatml suffix\n");
                embd_inp.insert(embd_inp.end(), cml_sfx.begin(), cml_sfx.end());
            }

            for (size_t i = original_size; i < embd_inp.size(); ++i) {
                const llama_token token = embd_inp[i];
                output_tokens.push_back(token);
                output_ss << llama_token_to_piece(ctx, token);
            }

            n_remain -= (int)line_inp.size();
            LOG("n_remain: %d\n", n_remain);
        } else {
            LOG("empty line, passing control back\n");
        }

        input_echo = false; // do not echo this again
    }
    reset_ctx_sampling();
}

void Pipeline::reset_ctx_sampling() {
    if (n_past > 0) {
        if (is_interacting) {
            llama_sampling_reset(ctx_sampling);
        }
        is_interacting = false;
    }
}

bool Pipeline::get_is_interacting() const {
    return is_interacting;
}

llama_model * Pipeline::get_model() const {
    return model;
}

llama_context *Pipeline::get_ctx() const {
    return ctx;
}

llama_context *Pipeline::get_ctx_guidance() const {
    return ctx_guidance;
}

llama_sampling_context *Pipeline::get_ctx_sampling() const {
    return ctx_sampling;
}

std::vector<llama_token> &Pipeline::get_input_tokens(){
    return input_tokens;
}

std::vector<llama_token> &Pipeline::get_output_tokens(){
    return output_tokens;
}

std::ostringstream &Pipeline::get_output_ss() {
    return output_ss;
}

gpt_params mission_params_parse(const int& argc, const std::vector<std::string>& argv) {
    gpt_params params;
    try {
        if (!mission_params_parse_ex(argc, argv, params)) {
            mission_print_usage(argc, argv, gpt_params());
            exit(0);
        }
//        mission_print_usage(argc, argv, params);
    }
    catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        mission_print_usage(argc, argv, gpt_params());
        exit(1);
    }
    return params;
}

bool mission_params_parse_ex(const int& argc, const std::vector<std::string>& argv, gpt_params & params) {
    bool invalid_param = false;
    std::string arg;
    const std::string arg_prefix = "--";
    llama_sampling_params & sparams = params.sparams;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.seed = std::stoul(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
            if (params.n_threads <= 0) {
                params.n_threads = std::thread::hardware_concurrency();
            }
        } else if (arg == "-tb" || arg == "--threads-batch") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads_batch = std::stoi(argv[i]);
            if (params.n_threads_batch <= 0) {
                params.n_threads_batch = std::thread::hardware_concurrency();
            }
        } else if (arg == "-td" || arg == "--threads-draft") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads_draft = std::stoi(argv[i]);
            if (params.n_threads_draft <= 0) {
                params.n_threads_draft = std::thread::hardware_concurrency();
            }
        } else if (arg == "-tbd" || arg == "--threads-batch-draft") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads_batch_draft = std::stoi(argv[i]);
            if (params.n_threads_batch_draft <= 0) {
                params.n_threads_batch_draft = std::thread::hardware_concurrency();
            }
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.prompt = argv[i];
        } else if (arg == "-e" || arg == "--escape") {
            params.escape = true;
        } else if (arg == "--prompt-cache") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.path_prompt_cache = argv[i];
        } else if (arg == "--prompt-cache-all") {
            params.prompt_cache_all = true;
        } else if (arg == "--prompt-cache-ro") {
            params.prompt_cache_ro = true;
        } else if (arg == "-bf" || arg == "--binary-file") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i], std::ios::binary);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i].c_str());
                invalid_param = true;
                break;
            }
            // store the external file name in params
            params.prompt_file = argv[i];
            std::ostringstream ss;
            ss << file.rdbuf();
            params.prompt = ss.str();
            fprintf(stderr, "Read %zu bytes from binary file %s\n", params.prompt.size(), argv[i].c_str());
        } else if (arg == "-f" || arg == "--file") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i].c_str());
                invalid_param = true;
                break;
            }
            // store the external file name in params
            params.prompt_file = argv[i];
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (!params.prompt.empty() && params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        } else if (arg == "-n" || arg == "--n-predict") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_predict = std::stoi(argv[i]);
        } else if (arg == "--top-k") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.top_k = std::stoi(argv[i]);
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_ctx = std::stoi(argv[i]);
        } else if (arg == "--grp-attn-n" || arg == "-gan") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }

            params.grp_attn_n = std::stoi(argv[i]);
        } else if (arg == "--grp-attn-w" || arg == "-gaw") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }

            params.grp_attn_w = std::stoi(argv[i]);
        } else if (arg == "--rope-freq-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.rope_freq_base = std::stof(argv[i]);
        } else if (arg == "--rope-freq-scale") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.rope_freq_scale = std::stof(argv[i]);
        } else if (arg == "--rope-scaling") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string value(argv[i]);
            /**/ if (value == "none")   { params.rope_scaling_type = LLAMA_ROPE_SCALING_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_LINEAR; }
            else if (value == "yarn")   { params.rope_scaling_type = LLAMA_ROPE_SCALING_YARN; }
            else { invalid_param = true; break; }
        } else if (arg == "--rope-scale") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.rope_freq_scale = 1.0f/std::stof(argv[i]);
        } else if (arg == "--yarn-orig-ctx") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.yarn_orig_ctx = std::stoi(argv[i]);
        } else if (arg == "--yarn-ext-factor") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.yarn_ext_factor = std::stof(argv[i]);
        } else if (arg == "--yarn-attn-factor") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.yarn_attn_factor = std::stof(argv[i]);
        } else if (arg == "--yarn-beta-fast") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.yarn_beta_fast = std::stof(argv[i]);
        } else if (arg == "--yarn-beta-slow") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.yarn_beta_slow = std::stof(argv[i]);
        } else if (arg == "--samplers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.samplers_sequence = parse_samplers_input(argv[i]);
        } else if (arg == "--sampling-seq") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.samplers_sequence = argv[i];
        } else if (arg == "--top-p") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.top_p = std::stof(argv[i]);
        } else if (arg == "--min-p") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.min_p = std::stof(argv[i]);
        } else if (arg == "--temp") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.temp = std::stof(argv[i]);
            sparams.temp = std::max(sparams.temp, 0.0f);
        } else if (arg == "--tfs") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.tfs_z = std::stof(argv[i]);
        } else if (arg == "--typical") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.typical_p = std::stof(argv[i]);
        } else if (arg == "--repeat-last-n") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.penalty_last_n = std::stoi(argv[i]);
            sparams.n_prev = std::max(sparams.n_prev, sparams.penalty_last_n);
        } else if (arg == "--repeat-penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.penalty_repeat = std::stof(argv[i]);
        } else if (arg == "--frequency-penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.penalty_freq = std::stof(argv[i]);
        } else if (arg == "--presence-penalty") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.penalty_present = std::stof(argv[i]);
        } else if (arg == "--mirostat") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.mirostat = std::stoi(argv[i]);
        } else if (arg == "--mirostat-lr") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.mirostat_eta = std::stof(argv[i]);
        } else if (arg == "--mirostat-ent") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.mirostat_tau = std::stof(argv[i]);
        } else if (arg == "--cfg-negative-prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.cfg_negative_prompt = argv[i];
        } else if (arg == "--cfg-negative-prompt-file") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i].c_str());
                invalid_param = true;
                break;
            }
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(sparams.cfg_negative_prompt));
            if (!sparams.cfg_negative_prompt.empty() && sparams.cfg_negative_prompt.back() == '\n') {
                sparams.cfg_negative_prompt.pop_back();
            }
        } else if (arg == "--cfg-scale") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.cfg_scale = std::stof(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(argv[i]);
        } else if (arg == "--keep") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_keep = std::stoi(argv[i]);
        } else if (arg == "--draft") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_draft = std::stoi(argv[i]);
        } else if (arg == "--chunks") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_chunks = std::stoi(argv[i]);
        } else if (arg == "-np" || arg == "--parallel") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_parallel = std::stoi(argv[i]);
        } else if (arg == "-ns" || arg == "--sequences") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_sequences = std::stoi(argv[i]);
        } else if (arg == "--p-accept" || arg == "-pa") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.p_accept = std::stof(argv[i]);
        } else if (arg == "--p-split" || arg == "-ps") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.p_split = std::stof(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-md" || arg == "--model-draft") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model_draft = argv[i];
        } else if (arg == "-a" || arg == "--alias") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model_alias = argv[i];
        } else if (arg == "-tc" || arg == "--tiktoken_config"){
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.tiktoken_config = argv[i];
        } else if (arg == "-tt" || arg == "--tiktoken"){
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.tiktoken_path = argv[i];
        } else if (arg == "-nsg" || arg == "--no_streaming"){
            params.no_streaming = true;
        } else if (arg == "--lora") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.lora_adapter.push_back(std::make_tuple(argv[i], 1.0f));
            params.use_mmap = false;
        } else if (arg == "--lora-scaled") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            const char * lora_adapter = argv[i].c_str();
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.lora_adapter.push_back(std::make_tuple(lora_adapter, std::stof(argv[i])));
            params.use_mmap = false;
        } else if (arg == "--lora-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.lora_base = argv[i];
        } else if (arg == "--mmproj") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.mmproj = argv[i];
        } else if (arg == "--image") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.image = argv[i];
        } else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        } else if (arg == "--embedding") {
            params.embedding = true;
        } else if (arg == "--interactive-first") {
            params.interactive_first = true;
        } else if (arg == "-ins" || arg == "--instruct") {
            params.instruct = true;
        } else if (arg == "-cml" || arg == "--chatml") {
            params.chatml = true;
        } else if (arg == "--infill") {
            params.infill = true;
        } else if (arg == "-dkvc" || arg == "--dump-kv-cache") {
            params.dump_kv_cache = true;
        } else if (arg == "-nkvo" || arg == "--no-kv-offload") {
            params.no_kv_offload = true;
        } else if (arg == "-ctk" || arg == "--cache-type-k") {
            params.cache_type_k = argv[++i];
        } else if (arg == "-ctv" || arg == "--cache-type-v") {
            params.cache_type_v = argv[++i];
        } else if (arg == "--multiline-input") {
            params.multiline_input = true;
        } else if (arg == "--simple-io") {
            params.simple_io = true;
        } else if (arg == "-cb" || arg == "--cont-batching") {
            params.cont_batching = true;
        } else if (arg == "--color") {
            params.use_color = true;
        } else if (arg == "--mlock") {
            params.use_mlock = true;
        } else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_gpu_layers = std::stoi(argv[i]);
#ifndef LLAMA_SUPPORTS_GPU_OFFLOAD
            fprintf(stderr, "warning: not compiled with GPU offload support, --n-gpu-layers option will be ignored\n");
            fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
#endif
        } else if (arg == "--gpu-layers-draft" || arg == "-ngld" || arg == "--n-gpu-layers-draft") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_gpu_layers_draft = std::stoi(argv[i]);
#ifndef LLAMA_SUPPORTS_GPU_OFFLOAD
            fprintf(stderr, "warning: not compiled with GPU offload support, --n-gpu-layers-draft option will be ignored\n");
            fprintf(stderr, "warning: see main README.md for information on enabling GPU BLAS support\n");
#endif
        } else if (arg == "--main-gpu" || arg == "-mg") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.main_gpu = std::stoi(argv[i]);
#ifndef GGML_USE_CUBLAS_SYCL
            fprintf(stderr, "warning: llama.cpp was compiled without cuBLAS/SYCL. Setting the main GPU has no effect.\n");
#endif // GGML_USE_CUBLAS_SYCL
        } else if (arg == "--split-mode" || arg == "-sm") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string arg_next = argv[i];
            if (arg_next == "none") {
                params.split_mode = LLAMA_SPLIT_NONE;
            } else if (arg_next == "layer") {
                params.split_mode = LLAMA_SPLIT_LAYER;
            } else if (arg_next == "row") {
                params.split_mode = LLAMA_SPLIT_ROW;
            } else {
                invalid_param = true;
                break;
            }
#ifndef GGML_USE_CUBLAS_SYCL
            fprintf(stderr, "warning: llama.cpp was compiled without cuBLAS/SYCL. Setting the split mode has no effect.\n");
#endif // GGML_USE_CUBLAS_SYCL

        } else if (arg == "--tensor-split" || arg == "-ts") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string arg_next = argv[i];

            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            if (split_arg.size() >= LLAMA_MAX_DEVICES) {
                invalid_param = true;
                break;
            }
            for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
                if (i < split_arg.size()) {
                    params.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params.tensor_split[i] = 0.0f;
                }
            }
#ifndef GGML_USE_CUBLAS_SYCL
            fprintf(stderr, "warning: llama.cpp was compiled without cuBLAS/SYCL. Setting a tensor split has no effect.\n");
#endif // GGML_USE_CUBLAS_SYCL
        } else if (arg == "--no-mmap") {
            params.use_mmap = false;
        } else if (arg == "--numa") {
            params.numa = true;
        } else if (arg == "--verbose-prompt") {
            params.verbose_prompt = true;
        } else if (arg == "--no-display-prompt") {
            params.display_prompt = false;
        } else if (arg == "-r" || arg == "--reverse-prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.antiprompt.push_back(argv[i]);
        } else if (arg == "-ld" || arg == "--logdir") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.logdir = argv[i];

            if (params.logdir.back() != DIRECTORY_SEPARATOR) {
                params.logdir += DIRECTORY_SEPARATOR;
            }
        } else if (arg == "--save-all-logits" || arg == "--kl-divergence-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.logits_file = argv[i];
        } else if (arg == "--perplexity" || arg == "--all-logits") {
            params.logits_all = true;
        } else if (arg == "--ppl-stride") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.ppl_stride = std::stoi(argv[i]);
        } else if (arg == "-ptc" || arg == "--print-token-count") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_print = std::stoi(argv[i]);
        } else if (arg == "--ppl-output-type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.ppl_output_type = std::stoi(argv[i]);
        } else if (arg == "--hellaswag") {
            params.hellaswag = true;
        } else if (arg == "--hellaswag-tasks") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.hellaswag_tasks = std::stoi(argv[i]);
        } else if (arg == "--winogrande") {
            params.winogrande = true;
        } else if (arg == "--winogrande-tasks") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.winogrande_tasks = std::stoi(argv[i]);
        } else if (arg == "--multiple-choice") {
            params.multiple_choice = true;
        } else if (arg == "--multiple-choice-tasks") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.multiple_choice_tasks = std::stoi(argv[i]);
        } else if (arg == "--kl-divergence") {
            params.kl_divergence = true;
        } else if (arg == "--ignore-eos") {
            params.ignore_eos = true;
        } else if (arg == "--no-penalize-nl") {
            sparams.penalize_nl = false;
        } else if (arg == "-l" || arg == "--logit-bias") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::stringstream ss(argv[i]);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    sparams.logit_bias[key] = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                } else {
                    throw std::exception();
                }
            } catch (const std::exception&) {
                invalid_param = true;
                break;
            }
        } else if (arg == "-h" || arg == "--help") {
            return false;

        } else if (arg == "--version") {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        } else if (arg == "--random-prompt") {
            params.random_prompt = true;
        } else if (arg == "--in-prefix-bos") {
            params.input_prefix_bos = true;
        } else if (arg == "--in-prefix") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.input_prefix = argv[i];
        } else if (arg == "--in-suffix") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.input_suffix = argv[i];
        } else if (arg == "--grammar") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.grammar = argv[i];
        } else if (arg == "--grammar-file") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i].c_str());
                invalid_param = true;
                break;
            }
            std::copy(
                    std::istreambuf_iterator<char>(file),
                    std::istreambuf_iterator<char>(),
                    std::back_inserter(sparams.grammar)
            );
        } else if (arg == "--override-kv") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            const char *sep = strchr(argv[i].c_str(), '=');
            if (sep == nullptr || sep - argv[i].c_str() >= 128) {
                fprintf(stderr, "error: Malformed KV override: %s\n", argv[i].c_str());
                invalid_param = true;
                break;
            }
            struct llama_model_kv_override kvo;
            std::strncpy(kvo.key, argv[i].c_str(), sep - argv[i].c_str());
            kvo.key[sep - argv[i].c_str()] = 0;
            sep++;
            if (strncmp(sep, "int:", 4) == 0) {
                sep += 4;
                kvo.tag = LLAMA_KV_OVERRIDE_INT;
                kvo.int_value = std::atol(sep);
            } else if (strncmp(sep, "float:", 6) == 0) {
                sep += 6;
                kvo.tag = LLAMA_KV_OVERRIDE_FLOAT;
                kvo.float_value = std::atof(sep);
            } else if (strncmp(sep, "bool:", 5) == 0) {
                sep += 5;
                kvo.tag = LLAMA_KV_OVERRIDE_BOOL;
                if (std::strcmp(sep, "true") == 0) {
                    kvo.bool_value = true;
                } else if (std::strcmp(sep, "false") == 0) {
                    kvo.bool_value = false;
                } else {
                    fprintf(stderr, "error: Invalid boolean value for KV override: %s\n", argv[i].c_str());
                    invalid_param = true;
                    break;
                }
            } else {
                fprintf(stderr, "error: Invalid type for KV override: %s\n", argv[i].c_str());
                invalid_param = true;
                break;
            }
            params.kv_overrides.push_back(kvo);
#ifndef LOG_DISABLE_LOGS
            // Parse args for logging parameters
        } else if ( log_param_single_parse( argv[i] ) ) {
            // Do nothing, log_param_single_parse automatically does it's thing
            //  and returns if a match was found and parsed.
        } else if ( log_param_pair_parse( /*check_but_dont_parse*/ true, argv[i] ) ) {
            // We have a matching known parameter requiring an argument,
            //  now we need to check if there is anything after this argv
            //  and flag invalid_param or parse it.
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            if( !log_param_pair_parse( /*check_but_dont_parse*/ false, argv[i-1], argv[i]) ) {
                invalid_param = true;
                break;
            }
            // End of Parse args for logging parameters
#endif // LOG_DISABLE_LOGS
        } else {
            throw std::invalid_argument("error: unknown argument: " + arg);
        }
    }
    if (invalid_param) {
        throw std::invalid_argument("error: invalid parameter for argument: " + arg);
    }
    if (params.prompt_cache_all &&
        (params.interactive || params.interactive_first ||
         params.instruct)) {

        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    if (params.escape) {
        process_escapes(params.prompt);
        process_escapes(params.input_prefix);
        process_escapes(params.input_suffix);
        process_escapes(sparams.cfg_negative_prompt);
        for (auto & antiprompt : params.antiprompt) {
            process_escapes(antiprompt);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back(llama_model_kv_override());
        params.kv_overrides.back().key[0] = 0;
    }

    return true;
}

void mission_print_usage(int /*argc*/, const std::vector<std::string>& argv, const gpt_params & params) {
    const llama_sampling_params & sparams = params.sparams;

    printf("\n");
    printf("usage: %s [options]\n", argv[0].c_str());
    printf("\n");
    printf("options:\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  --version             show version and build info\n");
    printf("  -i, --interactive     run in interactive mode\n");
    printf("  --interactive-first   run in interactive mode and wait for input right away\n");
    printf("  -ins, --instruct      run in instruction mode (use with Alpaca models)\n");
    printf("  -cml, --chatml        run in chatml mode (use with ChatML-compatible models)\n");
    printf("  --multiline-input     allows you to write or paste multiple lines without ending each in '\\'\n");
    printf("  -r PROMPT, --reverse-prompt PROMPT\n");
    printf("                        halt generation at PROMPT, return control in interactive mode\n");
    printf("                        (can be specified more than once for multiple prompts).\n");
    printf("  --color               colorise output to distinguish prompt and user input from generations\n");
    printf("  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for < 0)\n");
    printf("  -t N, --threads N     number of threads to use during generation (default: %d)\n", params.n_threads);
    printf("  -tb N, --threads-batch N\n");
    printf("                        number of threads to use during batch and prompt processing (default: same as --threads)\n");
    printf("  -td N, --threads-draft N");
    printf("                        number of threads to use during generation (default: same as --threads)");
    printf("  -tbd N, --threads-batch-draft N\n");
    printf("                        number of threads to use during batch and prompt processing (default: same as --threads-draft)\n");
    printf("  -p PROMPT, --prompt PROMPT\n");
    printf("                        prompt to start generation with (default: empty)\n");
    printf("  -e, --escape          process prompt escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\)\n");
    printf("  --prompt-cache FNAME  file to cache prompt state for faster startup (default: none)\n");
    printf("  --prompt-cache-all    if specified, saves user input and generations to cache as well.\n");
    printf("                        not supported with --interactive or other interactive options\n");
    printf("  --prompt-cache-ro     if specified, uses the prompt cache but does not update it.\n");
    printf("  --random-prompt       start with a randomized prompt.\n");
    printf("  --in-prefix-bos       prefix BOS to user inputs, preceding the `--in-prefix` string\n");
    printf("  --in-prefix STRING    string to prefix user inputs with (default: empty)\n");
    printf("  --in-suffix STRING    string to suffix after user inputs with (default: empty)\n");
    printf("  -f FNAME, --file FNAME\n");
    printf("                        prompt file to start generation.\n");
    printf("  -bf FNAME, --binary-file FNAME\n");
    printf("                        binary file containing multiple choice tasks.\n");
    printf("  -n N, --n-predict N   number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)\n", params.n_predict);
    printf("  -c N, --ctx-size N    size of the prompt context (default: %d, 0 = loaded from model)\n", params.n_ctx);
    printf("  -b N, --batch-size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    printf("  --samplers            samplers that will be used for generation in the order, separated by \';\', for example: \"top_k;tfs;typical;top_p;min_p;temp\"\n");
    printf("  --sampling-seq        simplified sequence for samplers that will be used (default: %s)\n", sparams.samplers_sequence.c_str());
    printf("  --top-k N             top-k sampling (default: %d, 0 = disabled)\n", sparams.top_k);
    printf("  --top-p N             top-p sampling (default: %.1f, 1.0 = disabled)\n", (double)sparams.top_p);
    printf("  --min-p N             min-p sampling (default: %.1f, 0.0 = disabled)\n", (double)sparams.min_p);
    printf("  --tfs N               tail free sampling, parameter z (default: %.1f, 1.0 = disabled)\n", (double)sparams.tfs_z);
    printf("  --typical N           locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)\n", (double)sparams.typical_p);
    printf("  --repeat-last-n N     last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)\n", sparams.penalty_last_n);
    printf("  --repeat-penalty N    penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)\n", (double)sparams.penalty_repeat);
    printf("  --presence-penalty N  repeat alpha presence penalty (default: %.1f, 0.0 = disabled)\n", (double)sparams.penalty_present);
    printf("  --frequency-penalty N repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)\n", (double)sparams.penalty_freq);
    printf("  --mirostat N          use Mirostat sampling.\n");
    printf("                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used.\n");
    printf("                        (default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)\n", sparams.mirostat);
    printf("  --mirostat-lr N       Mirostat learning rate, parameter eta (default: %.1f)\n", (double)sparams.mirostat_eta);
    printf("  --mirostat-ent N      Mirostat target entropy, parameter tau (default: %.1f)\n", (double)sparams.mirostat_tau);
    printf("  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS\n");
    printf("                        modifies the likelihood of token appearing in the completion,\n");
    printf("                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n");
    printf("                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'\n");
    printf("  --grammar GRAMMAR     BNF-like grammar to constrain generations (see samples in grammars/ dir)\n");
    printf("  --grammar-file FNAME  file to read grammar from\n");
    printf("  --cfg-negative-prompt PROMPT\n");
    printf("                        negative prompt to use for guidance. (default: empty)\n");
    printf("  --cfg-negative-prompt-file FNAME\n");
    printf("                        negative prompt file to use for guidance. (default: empty)\n");
    printf("  --cfg-scale N         strength of guidance (default: %f, 1.0 = disable)\n", sparams.cfg_scale);
    printf("  --rope-scaling {none,linear,yarn}\n");
    printf("                        RoPE frequency scaling method, defaults to linear unless specified by the model\n");
    printf("  --rope-scale N        RoPE context scaling factor, expands context by a factor of N\n");
    printf("  --rope-freq-base N    RoPE base frequency, used by NTK-aware scaling (default: loaded from model)\n");
    printf("  --rope-freq-scale N   RoPE frequency scaling factor, expands context by a factor of 1/N\n");
    printf("  --yarn-orig-ctx N     YaRN: original context size of model (default: 0 = model training context size)\n");
    printf("  --yarn-ext-factor N   YaRN: extrapolation mix factor (default: 1.0, 0.0 = full interpolation)\n");
    printf("  --yarn-attn-factor N  YaRN: scale sqrt(t) or attention magnitude (default: 1.0)\n");
    printf("  --yarn-beta-slow N    YaRN: high correction dim or alpha (default: %.1f)\n", params.yarn_beta_slow);
    printf("  --yarn-beta-fast N    YaRN: low correction dim or beta (default: %.1f)\n", params.yarn_beta_fast);
    printf("  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)\n");
    printf("  --no-penalize-nl      do not penalize newline token\n");
    printf("  --temp N              temperature (default: %.1f)\n", (double)sparams.temp);
    printf("  --logits-all          return logits for all tokens in the batch (default: disabled)\n");
    printf("  --hellaswag           compute HellaSwag score over random tasks from datafile supplied with -f\n");
    printf("  --hellaswag-tasks N   number of tasks to use when computing the HellaSwag score (default: %zu)\n", params.hellaswag_tasks);
    printf("  --winogrande          compute Winogrande score over random tasks from datafile supplied with -f\n");
    printf("  --winogrande-tasks N  number of tasks to use when computing the Winogrande score (default: %zu)\n", params.winogrande_tasks);
    printf("  --multiple-choice     compute multiple choice score over random tasks from datafile supplied with -f\n");
    printf("  --multiple-choice-tasks N number of tasks to use when computing the multiple choice score (default: %zu)\n", params.winogrande_tasks);
    printf("  --kl-divergence       computes KL-divergence to logits provided via --kl-divergence-base");
    printf("  --keep N              number of tokens to keep from the initial prompt (default: %d, -1 = all)\n", params.n_keep);
    printf("  --draft N             number of tokens to draft for speculative decoding (default: %d)\n", params.n_draft);
    printf("  --chunks N            max number of chunks to process (default: %d, -1 = all)\n", params.n_chunks);
    printf("  -np N, --parallel N   number of parallel sequences to decode (default: %d)\n", params.n_parallel);
    printf("  -ns N, --sequences N  number of sequences to decode (default: %d)\n", params.n_sequences);
    printf("  -pa N, --p-accept N   speculative decoding accept probability (default: %.1f)\n", (double)params.p_accept);
    printf("  -ps N, --p-split N    speculative decoding split probability (default: %.1f)\n", (double)params.p_split);
    printf("  -cb, --cont-batching  enable continuous batching (a.k.a dynamic batching) (default: disabled)\n");
    printf("  --mmproj MMPROJ_FILE  path to a multimodal projector file for LLaVA. see examples/llava/README.md\n");
    printf("  --image IMAGE_FILE    path to an image file. use with multimodal models\n");
    if (llama_mlock_supported()) {
        printf("  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_mmap_supported()) {
        printf("  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
    printf("  --numa                attempt optimizations that help on some NUMA systems\n");
    printf("                        if run without this previously, it is recommended to drop the system page cache before using this\n");
    printf("                        see https://github.com/ggerganov/llama.cpp/issues/1437\n");
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
    printf("  -ngl N, --n-gpu-layers N\n");
    printf("                        number of layers to store in VRAM\n");
    printf("  -ngld N, --n-gpu-layers-draft N\n");
    printf("                        number of layers to store in VRAM for the draft model\n");
    printf("  -sm SPLIT_MODE, --split-mode SPLIT_MODE\n");
    printf("                        how to split the model across multiple GPUs, one of:\n");
    printf("                          - none: use one GPU only\n");
    printf("                          - layer (default): split layers and KV across GPUs\n");
    printf("                          - row: split rows across GPUs\n");
    printf("  -ts SPLIT, --tensor-split SPLIT\n");
    printf("                        fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1\n");
    printf("  -mg i, --main-gpu i   the GPU to use for the model (with split-mode = none),\n");
    printf("                        or for intermediate results and KV (with split-mode = row) (default: %d)\n", params.main_gpu);
#endif // LLAMA_SUPPORTS_GPU_OFFLOAD
    printf("  --verbose-prompt      print a verbose prompt before generation (default: %s)\n", params.verbose_prompt ? "true" : "false");
    printf("  --no-display-prompt   don't print prompt at generation (default: %s)\n", !params.display_prompt ? "true" : "false");
    printf("  -gan N, --grp-attn-n N\n");
    printf("                        group-attention factor (default: %d)\n", params.grp_attn_n);
    printf("  -gaw N, --grp-attn-w N\n");
    printf("                        group-attention width (default: %.1f)\n", (double)params.grp_attn_w);
    printf("  -dkvc, --dump-kv-cache\n");
    printf("                        verbose print of the KV cache\n");
    printf("  -nkvo, --no-kv-offload\n");
    printf("                        disable KV offload\n");
    printf("  -ctk TYPE, --cache-type-k TYPE\n");
    printf("                        KV cache data type for K (default: %s)\n", params.cache_type_k.c_str());
    printf("  -ctv TYPE, --cache-type-v TYPE\n");
    printf("                        KV cache data type for V (default: %s)\n", params.cache_type_v.c_str());
    printf("  --simple-io           use basic IO for better compatibility in subprocesses and limited consoles\n");
    printf("  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
    printf("  --lora-scaled FNAME S apply LoRA adapter with user defined scaling S (implies --no-mmap)\n");
    printf("  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
    printf("  -m FNAME, --model FNAME\n");
    printf("                        model path (default: %s)\n", params.model.c_str());
    printf("  -md FNAME, --model-draft FNAME\n");
    printf("                        draft model for speculative decoding\n");
    printf("  -ld LOGDIR, --logdir LOGDIR\n");
    printf("                        path under which to save YAML logs (no logging if unset)\n");
    printf("  --override-kv KEY=TYPE:VALUE\n");
    printf("                        advanced option to override model metadata by key. may be specified multiple times.\n");
    printf("                        types: int, float, bool. example: --override-kv tokenizer.ggml.add_bos_token=bool:false\n");
    printf("  -ptc N, --print-token-count N\n");
    printf("                        print token count every N tokens (default: %d)\n", params.n_print);
    printf("\n");
#ifndef LOG_DISABLE_LOGS
    log_print_usage();
#endif // LOG_DISABLE_LOGS
}

void Pipeline::write_logfile() {
    ::write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);
}

llama_context           ** g_ctx;
llama_model             ** g_model;
gpt_params               * g_params;
std::vector<llama_token> * g_input_tokens;
std::ostringstream       * g_output_ss;
std::vector<llama_token> * g_output_tokens;
bool is_interacting = false;

void write_logfile(
        const llama_context * ctx, const gpt_params & params, const llama_model * model,
        const std::vector<llama_token> & input_tokens, const std::string & output,
        const std::vector<llama_token> & output_tokens
) {
    if (params.logdir.empty()) {
        return;
    }

    const std::string timestamp = get_sortable_timestamp();

    const bool success = create_directory_with_parents(params.logdir);
    if (!success) {
        fprintf(stderr, "%s: warning: failed to create logdir %s, cannot write logfile\n",
                __func__, params.logdir.c_str());
        return;
    }

    const std::string logfile_path = params.logdir + timestamp + ".yml";
    FILE * logfile = fopen(logfile_path.c_str(), "w");

    if (logfile == nullptr) {
        fprintf(stderr, "%s: failed to open logfile %s\n", __func__, logfile_path.c_str());
        return;
    }

    fprintf(logfile, "binary: main\n");
    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    dump_non_result_info_yaml(logfile, params, ctx, timestamp, input_tokens, model_desc);

    fprintf(logfile, "\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "# Generation Results #\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "\n");

    dump_string_yaml_multiline(logfile, "output", output.c_str());
    dump_vector_int_yaml(logfile, "output_tokens", output_tokens);

    llama_dump_timing_info_yaml(logfile, ctx);
    fclose(logfile);
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        std::cout<<"------------------------\n";
        if (!is_interacting) {
            is_interacting = true;
        } else {
//            console::cleanup();
            printf("\n");
            llama_print_timings(*g_ctx);
            write_logfile(*g_ctx, *g_params, *g_model, *g_input_tokens, g_output_ss->str(), *g_output_tokens);
            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv){
    gpt_params params;

    //todo: delete
    params.model = "../../1_8B_tuned/q4_0.gguf";
    params.tiktoken_config = "../../1_8B_tuned/tiktoken_config.json";
    params.tiktoken_path= "../../1_8B_tuned/qwen.tiktoken";
    params.prompt = "You are a helpful assistant.";
    params.chatml = true;
    params.interactive=true;
    params.no_streaming= true;
    //

    if (!gpt_params_parse(argc, argv, params)){
        return 0;
    }

    Pipeline test(params);
    std::cout<<test.generator(false, "instruction：识别目标和指令（指令包括:search、get、go_to、go_back、rotate、turn_left、turn_right、get_in、go_forward、wait、put、stop）。\\n任务:逆时针旋转负二十度。");
    is_interacting = test.get_is_interacting();
    auto *t1_model = test.get_model();
    auto *t2_ctx = test.get_ctx();

    gpt_params w_params = test.get_params();
    g_ctx               = &t2_ctx;
    g_model             = &t1_model;
    g_params            = &w_params;
    g_input_tokens      = &test.get_input_tokens();
    g_output_ss         = &test.get_output_ss();
    g_output_tokens     = &test.get_output_tokens();

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action{};
    sigint_action.sa_handler = sigint_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, nullptr);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    std::string line;
    while(true){
        if(!std::getline(std::cin, line)) {
            line.clear();
            break;
        }
        std::cout<<line<<"\n";
    }
    return 0;
}
