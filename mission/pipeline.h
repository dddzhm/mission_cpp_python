//
// Created by zhouhongmin on 24-2-5.
//

#ifndef PIPLINE_H
#define PIPLINE_H

#include "common.h"

#include "console.h"
//#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include "mission.h"


#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif
void write_logfile(
        const llama_context * ctx, const gpt_params & params, const llama_model * model,
        const std::vector<llama_token> & input_tokens, const std::string & output,
        const std::vector<llama_token> & output_tokens
);

class Pipeline{
public:
    Pipeline()=delete;
    explicit Pipeline(const gpt_params &params);
    ~Pipeline();

    std::string generator(const std::string& prompts);
    void tokenize(std::basic_string<char> prompts);

    bool is_interacting = false;

    std::ostringstream output_ss;
    std::vector<llama_token> input_tokens;
    std::vector<llama_token> output_tokens;

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context* ctx_guidance = nullptr;
    llama_sampling_context * ctx_sampling = nullptr;
    std::unique_ptr<mission::MissionTokenizer> tokenizer = nullptr;

    gpt_params get_params();

private:
    static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data);

    gpt_params params;
    const llama_sampling_params &sparams;

    int n_ctx_train         = 0;
    int n_ctx               = 0;
    int guidance_offset     = 0;
    int original_prompt_len = 0;
    int n_past              = 0;
    int n_remain            = 0;
    int n_consumed          = 0;
    int n_session_consumed  = 0;
    int n_past_guidance     = 0;
    int ga_i                = 0;
    int ga_n;
    int ga_w;

    bool add_bos;
    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool display              = true;
    bool need_to_save_session = false;

    std::string        path_session;

    std::vector<llama_token> session_tokens;
    std::vector<llama_token> inp_pfx;
    std::vector<llama_token> inp_sfx;
    std::vector<llama_token> cml_pfx;
    std::vector<llama_token> cml_sfx;
    std::vector<llama_token> embd_inp;
    std::vector<llama_token> guidance_inp;
    std::vector<llama_token> embd_list;
};

#endif //PIPLINE_H
