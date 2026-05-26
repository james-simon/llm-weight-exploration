"""
Run next-token prediction and perplexity across Pythia model sizes.
Results cached as .npz files.
"""

import json
import numpy as np
from pathlib import Path
from . import svd
from .weights import MODELS_DIR, PYTHIA_MODELS

TOKENIZER_DIR = MODELS_DIR.parent / "tokenizer"

# model parameter counts (for perplexity plot x-axis)
MODEL_PARAMS = {
    "pythia-70m":   70e6,
    "pythia-160m":  160e6,
    "pythia-410m":  410e6,
    "pythia-1b":    1e9,
    "pythia-1.4b":  1.4e9,
    "pythia-2.8b":  2.8e9,
    "pythia-6.9b":  6.9e9,
    "pythia-12b":   12e9,
}


def _load_model_and_tok(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODELS_DIR / model_name), torch_dtype=torch.float32)
    model.eval()
    return model, tok


def sample_pile_sentences(n_sentences: int = 6, min_words: int = 8,
                          max_words: int = 30, seed: int = 42) -> list:
    """
    Stream a few documents from The Pile and extract clean sentences.
    Saved to results cache so we reuse the same sentences.
    """
    cache_name = f"pile_sentences_{n_sentences}_s{seed}"
    if svd.results_exist(cache_name):
        data = svd.load_results(cache_name)
        return [str(s) for s in data["sentences"]]

    print(f"  Streaming Pile sentences ...")
    from datasets import load_dataset
    import re

    # monology/pile-uncopyrighted is a HF-hosted subset of The Pile
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True,
                      trust_remote_code=True)
    rng = np.random.default_rng(seed)

    sentences = []
    for doc in ds:
        text = doc["text"]
        # split on sentence-ending punctuation
        candidates = re.split(r'(?<=[.!?])\s+', text)
        for s in candidates:
            s = s.strip()
            words = s.split()
            if min_words <= len(words) <= max_words and s[-1] in ".!?":
                sentences.append(s)
                if len(sentences) >= n_sentences * 10:
                    break
        if len(sentences) >= n_sentences * 10:
            break

    # pick n_sentences at random from candidates
    chosen = [sentences[i] for i in rng.choice(len(sentences), n_sentences, replace=False)]
    svd.save_results(cache_name, {"sentences": np.array(chosen)})
    print(f"  Sampled {n_sentences} sentences from The Pile.")
    return chosen


def run_next_token(
    prompts: list,
    model_names: list = None,
    top_k: int = 20,
) -> dict:
    """
    For each model and each prompt, return the top-k next-token probabilities.
    The prompt is fed in full; we predict the token after the last one.
    """
    import hashlib

    if model_names is None:
        model_names = PYTHIA_MODELS

    prompt_hash = hashlib.md5(json.dumps(prompts).encode()).hexdigest()[:8]
    cache_name = f"inference_{prompt_hash}"

    if svd.results_exist(cache_name):
        print(f"  [cached] {cache_name}")
        raw = svd.load_results(cache_name)
        return _unpack_next_token(raw, prompts, model_names, top_k)

    import torch
    print(f"  Computing {cache_name} for {len(model_names)} models x {len(prompts)} prompts ...")

    save_data = {"prompts": np.array(prompts), "model_names": np.array(model_names)}

    for model_name in model_names:
        print(f"    Loading {model_name} ...")
        model, tok = _load_model_and_tok(model_name)

        for pi, prompt in enumerate(prompts):
            ids = tok(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                logits = model(ids).logits[0, -1]
            probs = torch.softmax(logits, dim=-1).numpy()
            top_idx = np.argsort(probs)[::-1][:top_k]
            top_probs = probs[top_idx]
            top_tokens = [tok.decode([i]) for i in top_idx]
            save_data[f"{model_name}__p{pi}__probs"]  = top_probs.astype(np.float32)
            save_data[f"{model_name}__p{pi}__ids"]    = top_idx.astype(np.int32)
            save_data[f"{model_name}__p{pi}__tokens"] = np.array(top_tokens)
            print(f"      prompt {pi}: top={repr(top_tokens[0])} ({top_probs[0]:.3f})")

        del model

    svd.save_results(cache_name, save_data)
    return _unpack_next_token(save_data, prompts, model_names, top_k)


def run_perplexity(
    texts: list,
    model_names: list = None,
    stride: int = 512,
) -> dict:
    """
    Compute average per-token perplexity of each model on a list of texts.
    Uses a sliding window so long texts aren't truncated.
    Returns {model_name: float}
    """
    import hashlib

    if model_names is None:
        model_names = PYTHIA_MODELS

    text_hash = hashlib.md5(json.dumps(texts).encode()).hexdigest()[:8]
    cache_name = f"perplexity_{text_hash}"

    if svd.results_exist(cache_name):
        print(f"  [cached] {cache_name}")
        raw = svd.load_results(cache_name)
        return {m: float(raw[m]) for m in model_names if m in raw}

    import torch
    print(f"  Computing perplexity for {len(model_names)} models ...")

    save_data = {}
    full_text = "\n\n".join(texts)

    for model_name in model_names:
        print(f"    Loading {model_name} ...")
        model, tok = _load_model_and_tok(model_name)
        max_len = model.config.max_position_embeddings

        ids = tok(full_text, return_tensors="pt").input_ids[0]
        nlls, n_tokens = [], 0
        for begin in range(0, len(ids), stride):
            end = min(begin + max_len, len(ids))
            chunk = ids[begin:end].unsqueeze(0)
            # only score tokens after the first stride (avoid double-counting)
            target_len = end - begin if begin == 0 else end - begin - (max_len - stride)
            with torch.no_grad():
                out = model(chunk, labels=chunk)
            # loss is mean NLL over the chunk; we weight by target_len
            nlls.append(out.loss.item() * target_len)
            n_tokens += target_len
            if end == len(ids):
                break

        ppl = float(np.exp(sum(nlls) / n_tokens))
        print(f"      {model_name}: ppl={ppl:.2f} over {n_tokens} tokens")
        save_data[model_name] = np.array(ppl)
        del model

    svd.save_results(cache_name, save_data)
    return {m: float(save_data[m]) for m in model_names if m in save_data}


def _unpack_next_token(raw, prompts, model_names, top_k):
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
