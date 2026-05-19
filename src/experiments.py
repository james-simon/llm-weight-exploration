"""
Experiments. Each function checks for cached results and skips if present.
Results are saved as .npz files via svd.save_results().
"""

import numpy as np
from . import weights, svd, metrics


# ── Experiment A ─────────────────────────────────────────────────────────────

def singular_spectra_all_layers(model_name: str, matrix_type: str = "fan_out") -> dict:
    """
    For one model, compute singular values of every MLP matrix.
    matrix_type: "fan_out" or "fan_in"
    Returns dict: {layer_idx -> singular_values array}
    """
    name = f"spectra_all_layers_{model_name}_{matrix_type}"
    if svd.results_exist(name):
        print(f"  [cached] {name}")
        data = svd.load_results(name)
        layer_keys = [k for k in data.keys() if k.startswith("layer_")]
        return {int(k.split("_")[1]): data[k] for k in layer_keys}

    print(f"  Computing {name} ...")
    all_weights = weights.load_all_mlp_weights(model_name)
    spectra = {}
    for w in all_weights:
        matrix = w[matrix_type]
        S = svd.compute_singular_values(matrix)
        spectra[w["layer"]] = S

    save_data = {f"layer_{i}": S for i, S in spectra.items()}
    save_data["model"] = np.array(model_name)
    save_data["matrix_type"] = np.array(matrix_type)
    # save matrix shape so render can compute random baseline without weights
    sample_w = all_weights[0][matrix_type]
    save_data["matrix_shape"] = np.array(sample_w.shape)
    svd.save_results(name, save_data)
    return spectra


# ── Random baseline (Marchenko-Pastur) ───────────────────────────────────────

def random_matrix_spectrum(m: int, n: int, n_samples: int = 5) -> dict:
    """
    Compute singular spectra of iid Gaussian matrices of shape (m, n),
    averaged over n_samples draws. Also returns the theoretical MP bulk edges.
    Entries ~ N(0, 1/m) to match the typical init scale.
    Results keyed by shape so they cache across experiments.
    """
    name = f"random_spectrum_{m}x{n}"
    if svd.results_exist(name):
        return svd.load_results(name)

    print(f"  Computing random baseline {m}x{n} ...")
    spectra = []
    for _ in range(n_samples):
        W = np.random.randn(m, n) / np.sqrt(m)
        spectra.append(np.linalg.svd(W, compute_uv=False))
    S_mean = np.mean(spectra, axis=0)

    gamma = n / m
    mp_max = (1 + np.sqrt(gamma))   # in units of 1 (matrix scaled by 1/sqrt(m))
    mp_min = (1 - np.sqrt(gamma))

    result = {
        "S_mean": S_mean,
        "mp_max": np.array(mp_max),
        "mp_min": np.array(mp_min),
        "m": np.array(m),
        "n": np.array(n),
    }
    svd.save_results(name, result)
    return result


# ── Experiment B ─────────────────────────────────────────────────────────────

def midlayer_spectra_comparison(
    model_names: list = None,
    matrix_type: str = "fan_out",
) -> dict:
    """
    For each model, extract the middle-layer MLP singular spectrum.
    Returns dict: {model_name -> singular_values array}
    """
    if model_names is None:
        model_names = weights.PYTHIA_MODELS

    name = f"midlayer_comparison_{'_'.join(m.split('-')[1] for m in model_names)}_{matrix_type}"
    if svd.results_exist(name):
        print(f"  [cached] {name}")
        data = svd.load_results(name)
        return {m: data[m] for m in model_names}

    print(f"  Computing {name} ...")
    spectra = {}
    for model_name in model_names:
        n_layers = weights.get_num_layers(model_name)
        mid_layer = n_layers // 2
        w = weights.load_mlp_weights(model_name, mid_layer)
        S = svd.compute_singular_values(w[matrix_type])
        spectra[model_name] = S
        print(f"    {model_name} layer {mid_layer}: shape {w[matrix_type].shape}, rank {len(S)}")

    svd.save_results(name, spectra)
    return spectra


# ── Experiment C ─────────────────────────────────────────────────────────────

def fanin_fanout_alignment(model_name: str, layer_idx: int = None, top_k: int = 32) -> dict:
    """
    Full SVD of fan_in and fan_out at a given layer, compute alignment metrics.
    Defaults to middle layer if layer_idx not specified.
    Returns dict with singular values, principal angles, and alignment scores.
    """
    n_layers = weights.get_num_layers(model_name)
    if layer_idx is None:
        layer_idx = n_layers // 2

    name = f"fanin_fanout_alignment_{model_name}_layer{layer_idx}_k{top_k}"
    if svd.results_exist(name):
        print(f"  [cached] {name}")
        return svd.load_results(name)

    print(f"  Computing {name} ...")
    w = weights.load_mlp_weights(model_name, layer_idx)

    svd_out = svd.compute_svd(w["fan_out"])   # shape (intermediate, hidden)
    svd_in  = svd.compute_svd(w["fan_in"])    # shape (hidden, intermediate)

    # fan_out U lives in intermediate space; fan_in Vh.T lives in intermediate space
    # compare these to find alignment in the intermediate (neuron) space
    U_out = svd_out["U"]          # (intermediate, rank)
    U_in  = svd_in["Vh"].T        # (intermediate, rank)  — right singular vecs of fan_in

    angles = metrics.principal_angles(U_out, U_in, top_k)
    alignment = metrics.subspace_alignment(U_out, U_in, top_k)

    result = {
        "S_fan_out":   svd_out["S"],
        "S_fan_in":    svd_in["S"],
        "principal_angles": angles,
        "subspace_alignment": np.array(alignment),
        "top_k":       np.array(top_k),
        "layer":       np.array(layer_idx),
        "model":       np.array(model_name),
    }
    svd.save_results(name, result)
    return result
