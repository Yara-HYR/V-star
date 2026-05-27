"""VED optimized helpers: KV cache + no output_hidden_states."""

import torch
from typing import List, Tuple


def _forward_with_hidden(model, input_ids, attention_mask,
                         past_key_values=None, use_cache=False):
    """Forward via model.model() + lm_head (no output_hidden_states=True)."""
    base_model = model.model
    base_out = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
    )
    hidden = base_out.last_hidden_state
    logits = model.lm_head(hidden)
    pkv = base_out.past_key_values if use_cache else None
    return logits, hidden, pkv


def _clone_past_key_values(past_key_values):
    """Deep-clone past_key_values."""
    if past_key_values is None:
        return None
    if hasattr(past_key_values, 'key_cache'):
        from transformers.cache_utils import DynamicCache
        new_cache = DynamicCache()
        for i in range(len(past_key_values.key_cache)):
            new_cache.update(
                past_key_values.key_cache[i].clone(),
                past_key_values.value_cache[i].clone(),
                i,
            )
        return new_cache
    return tuple((k.clone(), v.clone()) for k, v in past_key_values)
