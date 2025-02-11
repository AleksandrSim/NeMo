# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo.lightning.io.artifact import FileArtifact
from nemo.lightning.io.mixin import track_io

__all__ = []

try:
    from nemo.collections.common.tokenizers import AutoTokenizer

    track_io(
        AutoTokenizer,
        artifacts=[
            FileArtifact("vocab_file", required=False),
            FileArtifact("merges_file", required=False),
        ],
    )
    __all__.append("AutoTokenizer")
except ImportError:
    pass


try:
    from nemo.collections.common.tokenizers import SentencePieceTokenizer

    track_io(SentencePieceTokenizer, artifacts=[FileArtifact("model_path")])
    __all__.append("SentencePieceTokenizer")
except ImportError:
    pass
