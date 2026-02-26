# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Qwen3 Omni model providers and configurations."""

# Core model components
# Bridges for HuggingFace to Megatron conversion
from megatron.bridge.models.qwen_omni.modelling_qwen3_omni.model import Qwen3OmniMoeModel  # noqa: F401
from megatron.bridge.models.qwen_omni.qwen3_omni_bridge import Qwen3OmniMoEBridge

# Dense and MoE model providers
from megatron.bridge.models.qwen_omni.qwen3_omni_provider import Qwen3OmniModelProvider


__all__ = [
    "Qwen3OmniMoeModel",
    "Qwen3OmniMoEBridge",
    "Qwen3OmniModelProvider",
]
