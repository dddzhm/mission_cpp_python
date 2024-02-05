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

class Pipeline{
public:
    Pipeline()=delete;
    explicit Pipeline(const gpt_params &params);
    ~Pipeline();

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context* ctx_guidance = nullptr;
    std::unique_ptr<mission::MissionTokenizer> tokenizer = nullptr;
private:
    static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data);

    gpt_params params;
    const llama_sampling_params &sparams;

    int n_ctx_train;
    int n_ctx;
    int guidance_offset = 0;
    int original_prompt_len = 0;
    bool add_bos;

    std::string path_session;
    std::vector<llama_token> session_tokens;
    std::vector<llama_token> inp_pfx;
    std::vector<llama_token> inp_sfx;

    std::vector<llama_token> cml_pfx;
    std::vector<llama_token> cml_sfx;
    std::vector<llama_token> embd_inp;
    std::vector<llama_token> guidance_inp;
};

#endif //PIPLINE_H
