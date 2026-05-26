"""
Run next-token prediction on a set of prompts across Pythia model sizes.
Returns top-k token distributions for the final position of each prompt.
Results cached as .npz files.
"""

import json
import numpy as np
from pathlib import Path
from . import svd
from .weights import MODELS_DIR, PYTHIA_MODELS


def run_next_token(
    prompts: list,          # list of strings; we predict the next token after each
    model_names: list = None,
    top_k: int = 20,
) -> dict:
    """
    For each model and each prompt, return the top-k next-token probabilities.
    Returns: {model_name: [{token: str, prob: float}, ...] for each prompt}
    Cached as "inference_{model}_{prompt_hash}.npz" per model.
    """
    import hashlib

    if model_names is None:
        model_names = PYTHIA_MODELS

    prompt_hash = hashlib.md5(json.dumps(prompts).encode()).hexdigest()[:8]
    cache_name = f"inference_{prompt_hash}"

    if svd.results_exist(cache_name):
        print(f"  [cached] {cache_name}")
        raw = svd.load_results(cache_name)
        return _unpack(raw, prompts, model_names, top_k)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"  Computing {cache_name} for {len(model_names)} models x {len(prompts)} prompts ...")

    # all Pythia models share the EleutherAI/gpt-neox-20b tokenizer
    TOKENIZER_DIR = MODELS_DIR.parent / "tokenizer"
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    save_data = {"prompts": np.array(prompts), "model_names": np.array(model_names)}

    for model_name in model_names:
        model_dir = MODELS_DIR / model_name
        print(f"    Loading {model_name} ...")
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
        model.eval()

        for pi, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                logits = model(ids).logits[0, -1]   # (vocab,)
            probs = torch.softmax(logits, dim=-1).numpy()
            top_idx = np.argsort(probs)[::-1][:top_k]
            top_probs = probs[top_idx]
            top_tokens = [tok.decode([i]) for i in top_idx]
            save_data[f"{model_name}__p{pi}__probs"] = top_probs.astype(np.float32)
            save_data[f"{model_name}__p{pi}__ids"]   = top_idx.astype(np.int32)
            save_data[f"{model_name}__p{pi}__tokens"] = np.array(top_tokens)
            print(f"      prompt {pi}: top={repr(top_tokens[0])} ({top_probs[0]:.3f})")

        del model

    svd.save_results(cache_name, save_data)
    return _unpack(save_data, prompts, model_names, top_k)


def _unpack(raw, prompts, model_names, top_k):
    results = {}
    for model_name in model_names:
        results[model_name] = []
        for pi in range(len(prompts)):
            key = f"{model_name}__p{pi}__"
            if key + "probs" not in raw:
                continue
            probs  = raw[key + "probs"][:top_k]
            tokens = raw[key + "tokens"][:top_k]
            results[model_name].append([
                {"token": str(t), "prob": float(p)}
                for t, p in zip(tokens, probs)
            ])
    return results
