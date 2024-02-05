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

    std::cout << generator("") << "\n";
}

Pipeline::~Pipeline() {
    llama_print_timings(ctx);
    write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);

    if (ctx_guidance) { llama_free(ctx_guidance); }

    llama_free(ctx);
    llama_free_model(model);

    llama_sampling_free(ctx_sampling);
    llama_backend_free();
}

void Pipeline::llama_log_callback_logTee(ggml_log_level level, const char *text, void *user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

gpt_params Pipeline::get_params() {
    return params;
}

std::string Pipeline::generator(const std::string& prompts) {
    // TODO: add "console" settings and delete all "printf"
    if (!prompts.empty()){
        tokenize(prompts);
    }
    return "";
}

void Pipeline::tokenize(std::basic_string<char> prompts) {
    if (n_past > 0 && is_interacting) {
        // TODO: add "console" settings and delete all "printf"
        if (params.instruct || params.chatml) {
            printf("\n> ");
        }

        if (params.input_prefix_bos) {
            LOG("adding input prefix BOS token\n");
            embd_inp.push_back(llama_token_bos(model));
        }

        if (!params.input_prefix.empty()) {
            LOG("appending input prefix: '%s'\n", params.input_prefix.c_str());
            printf("%s", params.input_prefix.c_str());
        }

        if (prompts.length() > 1) {
            // append input suffix if any
            if (!params.input_suffix.empty()) {
                LOG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                printf("%s", params.input_suffix.c_str());
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
            console::cleanup();
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
    //

    if (!gpt_params_parse(argc, argv, params)){
        return 0;
    }

    Pipeline test(params);
    test.generator("test");
    is_interacting = test.is_interacting;

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
