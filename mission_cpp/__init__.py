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
    def __init__(self):
        pass
