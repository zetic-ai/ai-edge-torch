# Copyright 2025 The LiteRT Torch Authors.
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
"""Export functions for HuggingFace Transformers models."""

import gc
import os
import shutil
import tempfile

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.core import litert_lm_builder
import torch


def export(
    model: str,
    output_dir: str,
    prefill_lengths=(256,),
    cache_length=4096,
    quantization_recipe: str = 'dynamic_wi8_afp32',
    enable_dynamic_shape: bool = False,
    externalize_embedder: bool = False,
    single_token_embedder: bool = False,
    key_ts_idx: int = 2,
    value_ts_idx: int = 3,
    split_cache: bool = False,
    auto_model_override: str | None = None,
    keep_temporary_files: bool = False,
    # target_accelerator: str | None = None,
    trust_remote_code: bool = False,
    use_jinja_template: bool = False,
    task: str = 'text_generation',
    bundle_litert_lm: bool = True,
    experimental_use_mixed_precision: bool = False,
    litert_lm_model_type_override: str | None = None,
    export_vision_encoder: bool = False,
    # TODO(weiyiw): Update when b/481323182 is fixed.
    vision_encoder_quantization_recipe: str = 'weight_only_wi8_afp32',
):
  """Exports HuggingFace Transformers model to tflite.

  Args:
    model: The name of the HuggingFace Transformers model to export, or the path
      to the safetensors directory.
    output_dir: The directory to export the model to.
    prefill_lengths: The lengths of the prefill input, separated by comma.
    cache_length: The length of the cache.
    quantization_recipe: The quantization recipes to use, separated by comma.
    enable_dynamic_shape: Whether to enable dynamic shape.
    externalize_embedder: Whether to externalize the embedder.
    single_token_embedder: Whether to use a single token embedder.
    key_ts_idx: The index of time step dimension in the key tensor.
    value_ts_idx: The index of time step dimension in the value tensor.
    split_cache: Whether to use split cache attention.
    auto_model_override: Overriding the AutoModel class to use for export.
    keep_temporary_files: Whether to keep the temporary files.
    trust_remote_code: Whether to trust remote code.
    use_jinja_template: Whether to use jinja template.
    task: The task to export the model for. Use 'text_generation' for text only
      LLMs, and 'image_text_to_text' for Vision LLMs.
    bundle_litert_lm: Whether to bundle the model as a LiteRT LM file.
    experimental_use_mixed_precision: Whether to enable mixed precision.
    litert_lm_model_type_override: Overriding the LiteRT LM model type.
    export_vision_encoder: Whether to export the vision encoder.
    vision_encoder_quantization_recipe: The quantization recipe to use for the
      vision encoder.
  """
  os.makedirs(output_dir, exist_ok=True)
  if not keep_temporary_files:
    work_dir = tempfile.mkdtemp(dir=output_dir)
  else:
    work_dir = output_dir
  source_model_artifacts = export_lib.load_model(
      model,
      trust_remote_code=trust_remote_code,
      auto_model_override=auto_model_override,
      task=task,
  )
  pt_model, config, text_model_config, tokenizer, image_processor = (
      source_model_artifacts.model,
      source_model_artifacts.model_config,
      source_model_artifacts.text_model_config,
      source_model_artifacts.tokenizer,
      source_model_artifacts.image_processor,
  )

  if export_vision_encoder:
    assert (
        externalize_embedder
    ), 'Exporting vision encoder requires externalize_embedder to be enabled.'
  if split_cache:
    assert (
        externalize_embedder
    ), 'Split_cache requires externalize_embedder to be enabled.'

  export_config = exportable_module.ExportableModuleConfig(
      batch_size=1,
      prefill_lengths=prefill_lengths,
      cache_length=cache_length,
      prefill_length_dim=torch.export.Dim('prefill_length', min=1, max=1024)
      if enable_dynamic_shape
      else None,
      cache_length_dim=torch.export.Dim('cache_length')
      if enable_dynamic_shape
      else None,
      externalize_embedder=externalize_embedder,
      single_token_embedder=single_token_embedder,
      k_ts_idx=key_ts_idx,
      v_ts_idx=value_ts_idx,
      split_cache=split_cache,
      externalize_rope=split_cache,
      cache_implementation='LiteRTLMSplitCache'
      if split_cache
      else 'LiteRTLMCache',
  )
  export_lib.export_text_prefill_decode_model(
      pt_model,
      text_model_config,
      export_config,
      work_dir,
      quantization_recipe,
      experimental_use_mixed_precision=experimental_use_mixed_precision,
  )
  if export_vision_encoder:
    # TODO(weiyiw): Add support for packaging vision encoder models.
    export_lib.export_vision_encoder_models(
        pt_model,
        image_processor,
        config,
        tokenizer,
        export_config,
        work_dir,
        vision_encoder_quantization_recipe or quantization_recipe,
    )
  gc.collect()
  if externalize_embedder:
    export_lib.export_embedder_model(
        pt_model,
        text_model_config,
        export_config,
        work_dir,
        quantization_recipe,
    )
  gc.collect()
  if split_cache:
    export_lib.export_auxiliary_model(
        pt_model,
        text_model_config,
        export_config,
        work_dir,
        quantization_recipe,
    )
  gc.collect()
  tokenizer_model_path = export_lib.export_tokenizer(tokenizer, work_dir)
  tflite_model_path = os.path.join(
      work_dir,
      'model_quantized.tflite' if quantization_recipe else 'model.tflite',
  )
  if externalize_embedder or split_cache or export_vision_encoder:
    # TODO(weiyiw): Add support for packaging models.
    return
  if bundle_litert_lm:
    litert_lm_builder.package_model(
        pt_model,
        tokenizer,
        tflite_model_path,
        tokenizer_model_path,
        cache_length,
        work_dir,
        output_dir,
        use_jinja_template,
        litert_lm_model_type_override,
    )
  if not keep_temporary_files:
    print(f'Removing temporary files from: {work_dir}')
    shutil.rmtree(work_dir)
