# Copyright 2024 The AI Edge Torch Authors.
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

"""Attention related modules and functions."""

from typing import Optional, Tuple, Union

from ai_edge_torch.generative.layers import builder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_torch.generative.layers import lora as lora_utils
from ai_edge_torch.generative.layers import model_config as cfg
from ai_edge_torch.generative.layers import rotary_position_embedding as rotary_pos_emb
from ai_edge_torch.generative.layers import scaled_dot_product_attention
from ai_edge_torch.generative.layers import sdpa_with_kv_update
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """A CausalSelfAttention layer built from the Edge Generative API layers."""

    def __init__(self, config: cfg.TransformerBlockConfig, mcfg: cfg.ModelConfig):
        super().__init__()
        self.config = config.attn_config
        self.enable_hlfb = mcfg.enable_hlfb

        # Construct model layers.
        self.qkv_projection = nn.Linear(
            mcfg.embedding_dim,
            (self.config.num_heads + 2 * self.config.num_query_groups)
            * self.config.head_dim,
            bias=self.config.qkv_use_bias,
        )

        self.query_norm = (
            nn.Identity()
            if self.config.query_norm_config is None
            else nn.RMSNorm(
                self.config.head_dim, eps=self.config.query_norm_config.epsilon
            )
        )

        self.key_norm = (
            nn.Identity()
            if self.config.key_norm_config is None
            else nn.RMSNorm(
                self.config.head_dim, eps=self.config.key_norm_config.epsilon
            )
        )

        self.output_projection = nn.Linear(
            self.config.num_heads * self.config.head_dim,
            mcfg.embedding_dim,
            bias=self.config.output_proj_use_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[kv_utils.KVCacheEntry] = None,
        lora: Optional[lora_utils.LoRAEntry] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
        """Forward function of the CausalSelfAttention layer, which can support

           MQA, GQA and MHA.

        Args:
          x (torch.Tensor): the input tensor.
          rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
          mask (torch.Tensor): the optional mask tensor.
          input_pos (torch.Tensor): the optional input position tensor.
          kv_cache (KVCacheEntry): the KV cache entry corresponding to this module.
          lora (LoRAEntry): the optional lora entry.

        Returns:
          output activation from this self attention layer, and the updated
            KV Cach Entry (if passed in).
        """
        # Batch size, sequence length, embedding dimensionality.
        B, T, _ = x.size()
        qkv = self.qkv_projection(x)

        # Assemble into a number of query groups to support MHA, MQA and GQA.
        q_per_kv = self.config.num_heads // self.config.num_query_groups
        # Each group has >=1 queries, 1 key, and 1 value.
        if self.config.qkv_transpose_before_split:
            qkv = qkv.view(B, T, -1, self.config.head_dim)
            q, k, v = qkv.split(
                (
                    q_per_kv * self.config.num_query_groups,
                    self.config.num_query_groups,
                    self.config.num_query_groups,
                ),
                dim=-2,
            )
        else:
            qkv = qkv.view(B, T, self.config.num_query_groups, -1)
            q, k, v = qkv.split(
                (
                    q_per_kv * self.config.head_dim,
                    self.config.head_dim,
                    self.config.head_dim,
                ),
                dim=-1,
            )

        q = self.query_norm(q)
        k = self.key_norm(k)

        if rope is not None:
            cos, sin = rope
            q, k = rotary_pos_emb.apply_rope_inline(q, k, cos, sin)

        sdpa_out, kv_cache = sdpa_with_kv_update.sdpa_with_kv_update(
            q, k, v, kv_cache, input_pos, mask, self.config, self.enable_hlfb
        )

        y = self.output_projection(sdpa_out)

        if kv_cache is None:
            return y
        else:
            return y, kv_cache


class CausalSelfAttentionWithLoRA(CausalSelfAttention):
    """A CausalSelfAttention layer with LoRA support."""

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[kv_utils.KVCacheEntry] = None,
        lora: Optional[lora_utils.LoRAEntry] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
        """Forward function of the CausalSelfAttention layer with LoRA support.

        Exactly the same as CausalSelfAttention but we add LoRA on the projection.

        Args:
          x (torch.Tensor): the input tensor.
          rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
          mask (torch.Tensor): the optional mask tensor.
          input_pos (torch.Tensor): the optional input position tensor.
          kv_cache (KVCacheEntry): the KV cache entry corresponding to this module.
          lora (LoRAEntry): the optional lora entry.

        Returns:
          output activation from this self attention layer with LoRA, and the
            updated KV Cach Entry (if passed in).
        """

        # Batch size, sequence length, embedding dimensionality.
        B, T, _ = x.size()

        # qkv = self.qkv_projection(x) + lora.qkv_lora(x)
        qkv = self.qkv_projection(x)
        if lora is not None:
            qkv += lora.attn_qkv_lora(x)

        # Assemble into a number of query groups to support MHA, MQA and GQA.
        q_per_kv = self.config.num_heads // self.config.num_query_groups
        # Each group has >=1 queries, 1 key, and 1 value.
        if self.config.qkv_transpose_before_split:
            qkv = qkv.view(B, T, -1, self.config.head_dim)
            q, k, v = qkv.split(
                (
                    q_per_kv * self.config.num_query_groups,
                    self.config.num_query_groups,
                    self.config.num_query_groups,
                ),
                dim=-2,
            )
        else:
            qkv = qkv.view(B, T, self.config.num_query_groups, -1)
            q, k, v = qkv.split(
                (
                    q_per_kv * self.config.head_dim,
                    self.config.head_dim,
                    self.config.head_dim,
                ),
                dim=-1,
            )

        q = self.query_norm(q)
        k = self.key_norm(k)

        if rope is not None:
            cos, sin = rope
            q, k = rotary_pos_emb.apply_rope_inline(q, k, cos, sin)

        sdpa_out, kv_cache = sdpa_with_kv_update.sdpa_with_kv_update(
            q, k, v, kv_cache, input_pos, mask, self.config, self.enable_hlfb
        )

        # y = self.output_projection(sdpa_out) + lora.o_lora(sdpa_out)
        y = self.output_projection(sdpa_out)
        if lora is not None:
            y += lora.attn_output_lora(sdpa_out)

        if kv_cache is None:
            return y
        else:
            return y, kv_cache


class TransformerBlock(nn.Module):
    """A TransformerBlock layer built from the Edge Generative API layers."""

    def __init__(self, config: cfg.TransformerBlockConfig, mcfg: cfg.ModelConfig):
        super().__init__()
        self.pre_atten_norm = builder.build_norm(
            mcfg.embedding_dim, config.pre_attention_norm_config
        )
        self.atten_func = builder.build_attention_layer(config, mcfg)
        self.post_atten_norm = builder.build_norm(
            mcfg.embedding_dim, config.post_attention_norm_config
        )
        self.ff = builder.build_ff_layer(config, mcfg)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[kv_utils.KVCacheEntry] = None,
        lora: Optional[lora_utils.LoRAEntry] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, kv_utils.KVCacheEntry]]:
        """Forward function of the TransformerBlock layer.

        Args:
          x (torch.Tensor): the input tensor.
          rope (Tuple[torch.Tensor, torch.Tensor]): the input rope tensor.
          mask (torch.Tensor): the optional mask tensor.
          input_pos (torch.Tensor): the optional input position tensor.
          kv_cache (KVCacheEntry): the KV cache entry corresponding to this module.
          lora (LoRAEntry): the optional lora entry.

        Returns:
          output activation from this transformer block, and the updated
            KV Cach Entry (if passed in).
        """
        x_norm = self.pre_atten_norm(x)
        res = self.atten_func(x_norm, rope, mask, input_pos, kv_cache, lora)
        if kv_cache is not None:
            attn_out, kv_cache = res
        else:
            attn_out = res

        x = x + self.post_atten_norm(attn_out)
        x = x + self.ff(x)

        if kv_cache is None:
            return x
        else:
            return x, kv_cache
