#include "tiktoken.h"
#include "mission.h"
#include "pipeline.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mission {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_C, m) {
  m.doc() = "mission.cpp python binding";

  py::class_<MissionConfig>(m, "MissionConfig")
    .def(py::init())
    // .def_readonly("dtype", &MissionConfig::dtype)
    .def_readwrite("vocab_size", &MissionConfig::vocab_size)
    .def_readwrite("hidden_size", &MissionConfig::hidden_size)
    .def_readwrite("num_attention_heads", &MissionConfig::num_attention_heads)
    .def_readwrite("num_kv_heads", &MissionConfig::num_kv_heads)
    .def_readwrite("num_hidden_layers", &MissionConfig::num_hidden_layers)
    .def_readwrite("intermediate_size", &MissionConfig::intermediate_size)
    .def_readwrite("max_length", &MissionConfig::max_length)
    .def_readwrite("eos_token_id", &MissionConfig::eos_token_id)
    .def_readwrite("pad_token_id", &MissionConfig::pad_token_id)
    .def_readwrite("im_start_id", &MissionConfig::im_start_id)
    .def_readwrite("im_end_id", &MissionConfig::im_end_id);

  py::class_<tiktoken::tiktoken>(m, "tiktoken_cpp")
    .def(py::init<ankerl::unordered_dense::map<std::string, int>, ankerl::unordered_dense::map<std::string, int>, const std::string &>())
    .def("encode_ordinary", &tiktoken::tiktoken::encode_ordinary)
    .def("encode", &tiktoken::tiktoken::encode)
    .def("encode_single_piece", &tiktoken::tiktoken::encode_single_piece)
    .def("decode", &tiktoken::tiktoken::decode);

  py::class_<MissionTokenizer>(m, "MissionTokenizer")
    .def("encode", &MissionTokenizer::encode)
    .def("decode", &MissionTokenizer::decode)
    .def("encode_history", &MissionTokenizer::encode_history);

  py::class_<Pipeline>(m, "Pipeline")
    .def(py::init<const gpt_params&>())
    .def_property_readonly("tokenizer", [](const Pipeline &self){ return self.tokenizer.get(); })
    .def("write_logfile", &Pipeline::write_logfile)
    .def("generator", &Pipeline::generator)
    .def("tokenize", &Pipeline::tokenize)
    .def("get_params", &Pipeline::get_params);

  py::class_<gpt_params>(m, "gpt_params")
    .def(py::init())
    .def_readwrite("n_threads", &gpt_params::n_threads)
    .def_readwrite("n_ctx", &gpt_params::n_ctx)
    .def_readwrite("n_predict", &gpt_params::n_predict)
    .def_readwrite("n_ctx", &gpt_params::n_ctx)
    .def_readwrite("n_batch", &gpt_params::n_batch)
    .def_readwrite("n_keep", &gpt_params::n_keep)
    .def_readwrite("n_parallel", &gpt_params::n_parallel)
    .def_readwrite("n_gpu_layers", &gpt_params::n_gpu_layers)
    .def_readwrite("main_gpu", &gpt_params::main_gpu)
    .def_readwrite("model", &gpt_params::model)
    .def_readwrite("tiktoken_config", &gpt_params::tiktoken_config)
    .def_readwrite("tiktoken_path", &gpt_params::tiktoken_path)
    .def_readwrite("prompt", &gpt_params::prompt)
    .def_readwrite("chatml", &gpt_params::chatml)
    .def_readwrite("interactive", &gpt_params::interactive)
    .def_readwrite("no_streaming", &gpt_params::no_streaming)
    .def_readwrite("sparams", &gpt_params::sparams);

  py::class_<llama_sampling_params>(m, "llama_sampling_params")
    .def(py::init())
    .def_readwrite("n_prev", &llama_sampling_params::n_prev)
    .def_readwrite("top_k", &llama_sampling_params::top_k)
    .def_readwrite("top_p", &llama_sampling_params::top_p)
    .def_readwrite("temp", &llama_sampling_params::temp)
    .def_readwrite("penalty_repeat", &llama_sampling_params::penalty_repeat);

  m.def("build_tokenizer", &build_tokenizer);
//  m.def("mission_params_parse", &mission_params_parse);

}

}