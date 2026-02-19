# Copyright 2026 The LiteRT Torch Authors.
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
# ==============================================================================
"""Exportable modules for extended modules."""

from litert_torch.generative.export_hf.model_ext.gemma3 import vision_exportable as gemma3_vision_exportable
import transformers


def get_vision_exportables(
    model_config: transformers.PretrainedConfig,
):
  """Gets vision exportables."""
  if model_config.model_type == 'gemma3':
    return (
        gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionEncoder,
        gemma3_vision_exportable.LiteRTExportableModuleForGemma3VisionAdapter,
    )
  else:
    raise ValueError(f'Unsupported model type: {model_config.model_type}')
