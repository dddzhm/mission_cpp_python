#include "mission/tiktoken.h"
#include "mission/mission.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mission {

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_C, m) {
  m.doc() = "mission.cpp python binding";

  py::class_<MissionConfig>(m, "MissionConfig")
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

  m.def("build_tokenizer", &build_tokenizer);

}

}