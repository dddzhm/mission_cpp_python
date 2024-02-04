#include "mission.h"
#include <json/json.h>

namespace mission {

static std::pair<std::string, int> _parse(const std::string &line) {
    auto pos = line.find(" ");
    if (pos == std::string::npos) {
        throw std::runtime_error("invalid encoder line: " + line);
    }

    auto token = decode({line.data(), pos});
    int rank = 0;
    try {
        rank = std::stoul(line.substr(pos + 1));
    } catch (const std::exception &) {
        throw std::runtime_error("invalid encoder rank: " + line);
    }

    return {std::move(token), rank};
}


MissionTokenizer::MissionTokenizer(const std::string & tiktoken_path, const MissionConfig &config) {
    std::ifstream file(tiktoken_path);
    if (!file) {
        throw std::runtime_error("failed to open encoder file: " + tiktoken_path);
    }

    ankerl::unordered_dense::map<std::string, int> encoder;
    std::string line;
    while (std::getline(file, line)) {
        auto [token, rank] = _parse(line);

        if (!encoder.emplace(std::move(token), rank).second) {
            throw std::runtime_error("duplicate item: " + line);
        }
    }

    std::vector<std::string> special_tokens_s{"<|endoftext|>", "<|im_start|>", "<|im_end|>"};
    char buffer[14];
    for (size_t i = 0; i < 205; i++) {
        snprintf(buffer, 14, "<|extra_%zu|>", i);
        special_tokens_s.push_back(buffer);
    }
    size_t encoder_size = encoder.size();
    ankerl::unordered_dense::map<std::string, int> special_tokens;
    special_tokens.reserve(special_tokens_s.size());
    for (size_t i = 0; i < special_tokens_s.size(); i++) {
        special_tokens[special_tokens_s[i]] = encoder_size + i;
    }

    tokenizer = tiktoken::tiktoken(std::move(encoder), special_tokens, PAT_STR);
    eos_token_id = config.eos_token_id;
    im_start_id = config.im_start_id;
    im_end_id = config.im_end_id;
}

auto MissionTokenizer::build_prompt(const std::vector<std::string> &history) const -> std::string {
    MISSION_CHECK(history.size() % 2 == 1) << "invalid history size " << history.size();

    std::ostringstream oss_prompt;
    oss_prompt << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
    for (size_t i = 0; i < history.size() - 1; i += 2) {
        oss_prompt << "\n<|im_start|>user\n" << history[i] << "<|im_end|>\n<|im_start|>" << history[i + 1] << "<|im_end|>";
    }
    oss_prompt << "\n<|im_start|>user\n" << history.back() <<  "<|im_end|>\n<|im_start|>assistant\n";

    return oss_prompt.str();
}

auto MissionTokenizer::encode(const std::string &text, int max_length) const -> std::vector<int> {
    auto ids = tokenizer.encode(text);
    if ((int)ids.size() > max_length) {
        ids.erase(ids.begin(), ids.end() - max_length);
    }
    return ids;
}

auto MissionTokenizer::decode(const std::vector<int> &ids) const -> std::string {
    std::vector<int> normal_ids(ids);
    normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id) { return is_special_id(id); }), normal_ids.end());
    auto text = tokenizer.decode(normal_ids);
    return text;
}

auto MissionTokenizer::encode_history(
    const std::vector<std::string> &history, int max_length
) const -> std::vector<int> {
    std::string prompt = build_prompt(history);
    std::vector<int> input_ids = encode(prompt, max_length);
    return input_ids;
}

auto MissionTokenizer::is_special_id(int id) const -> bool {
  return id == eos_token_id || id == im_start_id || id == im_end_id;
}

static void readJsonFromStr(const std::string &path, MissionConfig &config)
{
    std::ifstream config_file(path, std::ios::binary);
	Json::Reader reader;
	Json::Value root;

	if (reader.parse(config_file, root))
	{
		config.vocab_size = root["vocab_size"].asInt();
        config.hidden_size = root["hidden_size"].asInt();
        config.num_attention_heads = root["num_attention_heads"].asInt();
        config.num_kv_heads = root["num_kv_heads"].asInt();
        config.num_hidden_layers = root["num_hidden_layers"].asInt();
        config.intermediate_size = root["intermediate_size"].asInt();
        config.max_length = root["seq_length"].asInt();
        config.eos_token_id = root["eos_token_id"].asInt();
        config.pad_token_id = root["pad_token_id"].asInt();
        config.im_start_id = root["im_start_id"].asInt();
        config.im_end_id = root["im_end_id"].asInt();
	}
}

auto build_tokenizer(const std::string &config_path, const std::string &tiktoken_path) -> std::unique_ptr<MissionTokenizer>{
    MissionConfig config;
    readJsonFromStr(config_path, config);

    auto tokenizer = std::make_unique<MissionTokenizer>(tiktoken_path, config);
    return tokenizer;
}

}