import torch
import torch.nn as nn
import math
import ai_edge_torch.generative.layers.rotary_position_embedding as rotary_pos_emb
from ai_edge_torch.generative.layers.kv_cache import KVCacheEntry, KV_LAYOUT_TRANSPOSED
from ai_edge_torch.generative.custom_ops import dynamic_update_slice as dus


# Dynamically create Main and CacheUpdate modules
def get_main_module_class(num_layers):
    kv_k_names = [f"kv_cache_k_{i}" for i in range(num_layers)]
    kv_v_names = [f"kv_cache_v_{i}" for i in range(num_layers)]
    all_arg_names = (
        [
            "embeddings",
            "input_pos",
            "mask_global",
            "mask_local",
            "pos_emb_cos",
            "pos_emb_sin",
            "pos_emb_local_cos",
            "pos_emb_local_sin",
        ]
        + kv_k_names
        + kv_v_names
    )
    arg_list = ", ".join(all_arg_names)

    class_src = f"""
class DynamicGemma3Main(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.num_layers = self.config.num_layers
        self.transformer_blocks = model.transformer_blocks
        self.final_norm = model.final_norm
        self.lm_head = model.lm_head

    def forward(self, {arg_list}):
        # Transparent forward pass without StableHLOCompositeBuilder
        # This allows the quantization pass to see all internal weights.
        
        k_in_list = [{", ".join(kv_k_names)}]
        v_in_list = [{", ".join(kv_v_names)}]
        
        h = embeddings
        p_idx = input_pos
        m_g, m_l = mask_global, mask_local
        c_g, s_g = pos_emb_cos, pos_emb_sin
        c_l, s_l = pos_emb_local_cos, pos_emb_local_sin
        
        res_slices = {{}}
        for i, block in enumerate(self.transformer_blocks):
            l_rope = (c_l, s_l) if (i + 1) % 6 == 0 else (c_g, s_g)
            l_mask = m_l if (i + 1) % 6 == 0 else m_g
            
            x_norm = block.pre_atten_norm(h)
            
            # --- manual attn to replicate layers but return slices ---
            qkv = block.atten_func.qkv_projection(x_norm)
            B, T, _ = qkv.size()
            hp = block.atten_func.config.head_dim
            ng = block.atten_func.config.num_query_groups
            nh = block.atten_func.config.num_heads
            q_per_kv = nh // ng
            
            qkv = qkv.view(B, T, -1, hp)
            q, k, v = qkv.split((q_per_kv * ng, ng, ng), dim=-2)
            
            q = block.atten_func.query_norm(q)
            k = block.atten_func.key_norm(k)
            v = block.atten_func.value_norm(v)
            
            from ai_edge_torch.generative.layers.rotary_position_embedding import apply_rope_inline
            q, k = apply_rope_inline(q, k, l_rope[0], l_rope[1])
            
            # k_new [B, ng, T, hp], v_new [B, ng, hp, T]
            k_new = k.transpose(1, 2)
            v_new = v.transpose(1, 2).transpose(2, 3)
            
            res_slices[f"kv_slice_k_{{i}}"] = k_new
            res_slices[f"kv_slice_v_{{i}}"] = v_new
            
            # Manual Transposed SDPA Logic
            k_cache = k_in_list[i]
            v_cache = v_in_list[i]
            
            from ai_edge_torch.generative.layers.kv_cache import update_transposed as kv_up_t
            dummy_entry = KVCacheEntry(k_cache=k_cache, v_cache=v_cache, kv_layout=KV_LAYOUT_TRANSPOSED)
            updated_entry = kv_up_t(dummy_entry, p_idx, k_new, v_new)
            
            q_attn = q.permute(0, 2, 1, 3)
            k_attn = updated_entry.k_cache
            v_attn = updated_entry.v_cache
            
            if nh > ng:
                k_attn = k_attn.repeat_interleave(nh // ng, dim=1)
                v_attn = v_attn.repeat_interleave(nh // ng, dim=1)
            
            scores = (q_attn @ k_attn.transpose(-2, -1)) * (1.0 / math.sqrt(hp))
            if block.atten_func.config.logit_softcap is not None:
                scores = torch.tanh(scores / block.atten_func.config.logit_softcap) * block.atten_func.config.logit_softcap
            
            scores = scores + l_mask
            probs = torch.softmax(scores.float(), dim=-1).type_as(q)
            
            attn_out = probs @ v_attn.transpose(-2, -1)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
            
            y = block.atten_func.output_projection(attn_out)
            x = h + block.post_atten_norm(y)
            h = x + block.ff(x)
            
        h = self.final_norm(h)
        logits = self.lm_head(h)
        
        final_dict = {{f"kv_slice_k_{{i}}": res_slices[f"kv_slice_k_{{i}}"] for i in range(self.num_layers)}}
        final_dict.update({{f"kv_slice_v_{{i}}": res_slices[f"kv_slice_v_{{i}}"] for i in range(self.num_layers)}})
        final_dict["logits"] = logits
        return final_dict
"""
    local_scope = {
        "torch": torch,
        "math": math,
        "KVCacheEntry": KVCacheEntry,
        "KV_LAYOUT_TRANSPOSED": KV_LAYOUT_TRANSPOSED,
        "dus": dus,
    }
    exec(class_src, globals(), local_scope)
    return local_scope["DynamicGemma3Main"]


def get_cache_update_module_class(num_layers):
    cache_names = [f"kv_cache_k_{i}" for i in range(num_layers)] + [
        f"kv_cache_v_{i}" for i in range(num_layers)
    ]
    slice_names = [f"kv_slice_k_{i}" for i in range(num_layers)] + [
        f"kv_slice_v_{i}" for i in range(num_layers)
    ]
    arg_list = ", ".join(cache_names + slice_names)

    class_src = f"""
class DynamicGemma3CacheUpdate(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
    def forward(self, input_pos, {arg_list}):
        ks_in = [{", ".join(cache_names[:num_layers])}]
        vs_in = [{", ".join(cache_names[num_layers:])}]
        sk_in = [{", ".join(slice_names[:num_layers])}]
        sv_in = [{", ".join(slice_names[num_layers:])}]
        
        updated = {{}}
        p0 = input_pos[0].to(torch.int32)
        z = torch.tensor(0, dtype=torch.int32)
        for i in range(self.num_layers):
            updated[f"kv_cache_k_{{i}}"] = dus.dynamic_update_slice(ks_in[i], sk_in[i], [z, z, p0, z])
            updated[f"kv_cache_v_{{i}}"] = dus.dynamic_update_slice(vs_in[i], sv_in[i], [z, z, z, p0])
        return updated
"""
    local_scope = {"torch": torch, "dus": dus}
    exec(class_src, globals(), local_scope)
    return local_scope["DynamicGemma3CacheUpdate"]


class Gemma3Embedder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.tok_embedding = model.tok_embedding
        self.embedding_scale = getattr(model.config, "embedding_scale", 1.0)

    def forward(self, token_ids):
        embeds = self.tok_embedding(token_ids)
        if self.embedding_scale is not None and self.embedding_scale != 1.0:
            embeds = embeds * self.embedding_scale
        return {"embeddings": embeds}


class Gemma3Rope(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input_pos):
        attn_config = self.config.block_config(0).attn_config
        c10k, s10k = rotary_pos_emb.build_rope(input_pos, attn_config.head_dim, 10000.0)
        c1M, s1M = rotary_pos_emb.build_rope(input_pos, attn_config.head_dim, 1000000.0)
        return {
            "pos_emb_cos": c10k,
            "pos_emb_sin": s10k,
            "pos_emb_local_cos": c1M,
            "pos_emb_local_sin": s1M,
        }


class Gemma3Mask(nn.Module):
    def __init__(self, kv_len=1280):
        super().__init__()
        self.kv_len = kv_len

    def forward(self, time_step, input_tokens):
        batch_size, seq_len = input_tokens.shape
        mask = torch.zeros((batch_size, 1, seq_len, self.kv_len), dtype=torch.float32)
        return {"mask_global": mask, "mask_local": mask}
