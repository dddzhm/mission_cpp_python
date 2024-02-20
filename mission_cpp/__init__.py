import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Union

import mission_cpp._C as _C


def test_tokenizer(
        config_path,
        tiktoken_path
):
    return _C.build_tokenizer(config_path, tiktoken_path)


class Pipeline(_C.Pipeline):
    def __init__(self, n_threads: int, n_ctx: int = 512, n_predict: int = -1, n_batch: int = 512, n_keep: int = 0,
                 n_parallel: int = 1, n_gpu_layers: int = -1, main_gpu: int = 0, model: str = './model/model.gguf',
                 tiktoken_config: str = './model/tiktoken_config.json', tiktoken_path: str = './model/qwen.tiktoken',
                 prompt: str = 'You are a helpful assistant.', chatml: bool = True, interactive: bool = True,
                 no_streaming: bool = False, recoder_history: bool = False, n_prev: int = 64, top_k: int = 40,
                 top_p: float = 0.95, temp: float = 0.8, penalty_repeat: float = 1.1, **kwargs):
        """
        :param n_threads: number of threads
        :param n_ctx: context size
        :param n_predict: new tokens to predict
        :param n_batch: batch size for prompt processing (must be >= 32 to use BLAS)
        :param n_keep: number of tokens to keep from initial prompt
        :param n_parallel: number of parallel sequences to decode
        :param n_gpu_layers: number of layers to store in VRAM (-1 - use defaule)
        :param main_gpu: the GPU that is used for scratch and small tensors
        :param model: model path
        :param tiktoken_config: config file for tiktoken (*.json)
        :param tiktoken_path: tiktoken path (*.tiktoken)
        :param prompt: initial prompt
        :param chatml: chatml mode (used for models trained on chatml syntax)
        :param interactive: interactive mode
        :param no_streaming: chat without streaming results
        :param recoder_history: recoder historical information
        :param n_prev: number of previous tokens to remember
        :param top_k: <= 0 to use vocab size
        :param top_p: 1.0 = disabled
        :param temp: <= 0.0 to sample greedily, 0.0 to not output probabilities
        :param penalty_repeat: 1.0 = disabled
        """
        sparams = _C.llama_sampling_params()
        sparams.n_prev = n_prev
        sparams.top_k = top_k
        sparams.top_p = top_p
        sparams.temp = temp
        sparams.penalty_repeat = penalty_repeat

        params = _C.gpt_params()
        params.n_threads = n_threads
        params.n_ctx = n_ctx
        params.n_predict = n_predict
        params.n_batch = n_batch
        params.n_keep = n_keep
        params.n_parallel = n_parallel
        params.n_gpu_layers = n_gpu_layers
        params.main_gpu = main_gpu
        params.model = model
        params.tiktoken_config = tiktoken_config
        params.tiktoken_path = tiktoken_path
        params.prompt = prompt
        params.chatml = chatml
        params.interactive = interactive
        params.no_streaming = no_streaming
        params.recoder_history = recoder_history
        params.sparams = sparams

        super().__init__(params)

    def gen(self, prompt: str = '') -> str:
        return self.generator(False, prompt)
