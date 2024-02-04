#pragma once
#include "tiktoken.h"
#include <sstream>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "ggml.h"

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

namespace mission{

#define MISSION_THROW ::mission::LogMessageFatal(__FILE__, __LINE__).stream()
#define MISSION_CHECK(cond) \
if (!(cond)) \
MISSION_THROW << "check failed (" #cond ") "

static const std::string PAT_STR = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

static auto pos_of_char(const unsigned char chr) -> size_t {
  if      (chr >= 'A' && chr <= 'Z') return chr - 'A';
  else if (chr >= 'a' && chr <= 'z') return chr - 'a' + ('Z' - 'A')               + 1;
  else if (chr >= '0' && chr <= '9') return chr - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
  else if (chr == '+' || chr == '-') return 62;
  else if (chr == '/' || chr == '_') return 63;
  else throw std::runtime_error("Input is not valid base64-encoded data.");
}

inline auto decode(std::string_view s) -> std::string {
  if (s.empty()) throw std::runtime_error("empty input");
  size_t length = s.length();
  size_t idx = 0;

  std::string out;
  out.reserve(length / 4 * 3);

  while (idx < length) {
    size_t pos_of_char_1 = pos_of_char(s.at(idx + 1));
    out.push_back(static_cast<std::string::value_type>(((pos_of_char(s.at(idx+0))) << 2 ) + ((pos_of_char_1 & 0x30) >> 4)));
    if ((idx + 2 < length) && s.at(idx + 2) != '=' && s.at(idx + 2) != '.') {
      size_t pos_of_char_2 = pos_of_char(s.at(idx + 2));
      out.push_back(static_cast<std::string::value_type>(((pos_of_char_1 & 0x0f) << 4) + ((pos_of_char_2 & 0x3c) >> 2)));
      if ((idx + 3 < length) && s.at(idx + 3) != '=' && s.at(idx + 3) != '.') {
        out.push_back(static_cast<std::string::value_type>(((pos_of_char_2 & 0x03) << 6) + pos_of_char(s.at(idx+3))));
      }
    }
    idx += 4;
  }
  return out;
}


class LogMessageFatal {
  public:
    LogMessageFatal(const char *file, int line) { oss_ << file << ':' << line << ' '; }
    [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
    auto stream() -> std::ostringstream & { return oss_; }

  private:
    std::ostringstream oss_;
};

struct MissionConfig {
  // common attributes
  int vocab_size;
  int hidden_size;
  int num_attention_heads;
  int num_kv_heads;
  int num_hidden_layers;
  int intermediate_size;
  // for sequence generation
  int max_length;
  // for tokenizer
  int eos_token_id;
  int pad_token_id;
  int im_start_id;
  int im_end_id;
};

class MissionTokenizer {
  public:

    MissionTokenizer(const std::string & tiktoken_path, const MissionConfig &config);

    auto encode(const std::string &text, int max_length) const -> std::vector<int>;

    auto decode(const std::vector<int> &ids) const -> std::string;

    auto encode_history(const std::vector<std::string> &history, int max_length) const -> std::vector<int>;

    auto build_prompt(const std::vector<std::string> &history) const -> std::string;

    auto is_special_id(int id) const -> bool;

    tiktoken::tiktoken tokenizer;
    int eos_token_id;
    int im_start_id;
    int im_end_id;
};

auto build_tokenizer(const std::string &config_path, const std::string &tiktoken_path) -> std::unique_ptr<MissionTokenizer>;
}