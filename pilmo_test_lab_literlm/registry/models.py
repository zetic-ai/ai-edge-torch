from ai_edge_torch.generative.examples.gemma3 import gemma3

# Model specifications registry
# To add a new model, simply add a new key-value pair here.
# 'builder' should be the function that takes (checkpoint_path, mask_cache_size)
# 'litertlm_metadata' contains model-specific token and prompt settings.

MODEL_SPECS = {
    "gemma-3-270m": {
        "repo_id": "google/gemma-3-270m-it",
        "builder": gemma3.build_model_270m,
        "litertlm_metadata": {
            "start_token_id": 2,
            "stop_token_ids": [106, 1],
            "user_prompt_prefix": "<start_of_turn>user\n",
            "user_prompt_suffix": "<end_of_turn>\n",
            "model_prompt_prefix": "<start_of_turn>model\n",
            "model_prompt_suffix": "<end_of_turn>\n",
            "llm_model_type": "gemma3",  # 니증에 아닌거 같으면 지워
        },
    },
    "gemma-3-1b": {
        "repo_id": "google/gemma-3-1b-it",
        "builder": gemma3.build_model_1b,
        "litertlm_metadata": {
            "start_token_id": 2,
            "stop_token_ids": [106, 1],
            "user_prompt_prefix": "<start_of_turn>user\n",
            "user_prompt_suffix": "<end_of_turn>\n",
            "model_prompt_prefix": "<start_of_turn>model\n",
            "model_prompt_suffix": "<end_of_turn>\n",
            "llm_model_type": "gemma3",
        },
    },
}
