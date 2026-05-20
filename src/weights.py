"""
Load MLP weight matrices from Pythia safetensors files.

Pythia (GPT-NeoX) MLP tensor names:
  gpt_neox.layers.{L}.mlp.dense_h_to_4h.weight  -> fan_out (up-projection)  shape: (4h, h)
  gpt_neox.layers.{L}.mlp.dense_4h_to_h.weight  -> fan_in  (down-projection) shape: (h, 4h)
"""

import json
from pathlib import Path
import numpy as np
from safetensors import safe_open


MODELS_DIR = Path("/mnt/xdata/llm_weights/models")

PYTHIA_MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]


def get_model_dir(model_name: str) -> Path:
    return MODELS_DIR / model_name


def get_num_layers(model_name: str) -> int:
    config_path = get_model_dir(model_name) / "config.json"
    if config_path.exists():
        return json.load(open(config_path))["num_hidden_layers"]
    # fall back to counting cached per-layer SVD results
    from . import svd
    for prefix in [f"spectra_all_layers_{model_name}_fan_out",
                   f"spectra_all_layers_{model_name}_fan_in"]:
        if svd.results_exist(prefix):
            data = svd.load_results(prefix)
            keys = [k for k in data.keys() if k.startswith("layer_")]
            if keys:
                return max(int(k.split("_")[1]) for k in keys) + 1
    raise FileNotFoundError(f"Cannot determine layer count for {model_name}: no config.json or cached spectra found")


def _open_safetensors(model_dir: Path):
    """Return list of open safetensors handles covering all shards."""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = json.load(open(index_file))
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    handles = [
        safe_open(str(model_dir / f), framework="np")
        for f in shard_files
    ]
    return handles


def _find_tensor(handles, key: str) -> np.ndarray:
    for h in handles:
        if key in h.keys():
            return h.get_tensor(key).astype("float32")
    raise KeyError(f"Tensor '{key}' not found in any shard")


def load_mlp_weights(model_name: str, layer_idx: int) -> dict:
    """
    Returns dict with keys:
      fan_out: np.ndarray (intermediate, hidden)  — up-projection W
      fan_in:  np.ndarray (hidden, intermediate)  — down-projection W
      layer:   int
      model:   str
    """
    model_dir = get_model_dir(model_name)
    handles = _open_safetensors(model_dir)

    fan_out = _find_tensor(handles, f"gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h.weight")
    fan_in  = _find_tensor(handles, f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight")
    bias_up   = _find_tensor(handles, f"gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h.bias")
    bias_down = _find_tensor(handles, f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.bias")

    return {
        "fan_out":   fan_out,
        "fan_in":    fan_in,
        "bias_up":   bias_up,
        "bias_down": bias_down,
        "layer":     layer_idx,
        "model":     model_name,
    }


def load_all_mlp_weights(model_name: str) -> list:
    """Load MLP weights for every layer in a model."""
    model_dir = get_model_dir(model_name)
    handles = _open_safetensors(model_dir)
    n_layers = get_num_layers(model_name)

    results = []
    for layer_idx in range(n_layers):
        fan_out = _find_tensor(handles, f"gpt_neox.layers.{layer_idx}.mlp.dense_h_to_4h.weight")
        fan_in  = _find_tensor(handles, f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h.weight")
        results.append({
            "fan_out": fan_out,
            "fan_in":  fan_in,
            "layer":   layer_idx,
            "model":   model_name,
        })
    return results
