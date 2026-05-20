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


# ── SVD overlap: fan_out RSVs vs fan_in LSVs (neuron space) ──────────────────

def fanout_fanin_overlap(model_name: str, layer_idx: int = None, top_k: int = None) -> dict:
    """
    For a given layer, compute the overlap matrix M[i,k] = <u_i^out, u_k^in>^2
    where u_i^out are the LEFT singular vectors of fan_out (neuron space, cols of U_out)
    and u_k^in are the RIGHT singular vectors of fan_in (neuron space, rows of Vh_in -> cols of Vh_in.T).

    fan_out shape: (intermediate, hidden) -> U_out is (intermediate, rank): neuron-space LSVs
    fan_in  shape: (hidden, intermediate) -> Vh_in is (rank, intermediate): neuron-space RSVs

    top_k: only compute top-k x top-k submatrix (cheaper, still revealing).
    """
    name_prefix = f"fanout_fanin_overlap_{model_name}_layer"

    # resolve layer_idx — try cache first to avoid needing model files locally
    if layer_idx is None:
        suffix = "" if top_k is None else f"_k{top_k}"
        existing = list(svd.RESULTS_DIR.glob(f"{name_prefix}*{suffix}.npz"))
        if existing:
            # use the first cached result that matches
            name = existing[0].stem
            print(f"  [cached] {name}")
            return svd.load_results(name)
        layer_idx = weights.get_num_layers(model_name) // 2

    name = f"{name_prefix}{layer_idx}"
    if top_k is not None:
        name += f"_k{top_k}"

    if svd.results_exist(name):
        print(f"  [cached] {name}")
        return svd.load_results(name)

    print(f"  Computing {name} ...")
    w = weights.load_mlp_weights(model_name, layer_idx)

    # full SVD of both matrices
    U_out, S_out, Vh_out = np.linalg.svd(w["fan_out"], full_matrices=False)  # U_out:(int,rank) Vh_out:(rank,hid)
    U_in,  S_in,  Vh_in  = np.linalg.svd(w["fan_in"],  full_matrices=False)  # U_in:(hid,rank)  Vh_in:(rank,int)

    n_intermediate = w["fan_out"].shape[0]  # ambient neuron-space dimension (e.g. 8192)
    k = top_k if top_k is not None else S_out.shape[0]
    U_out_k = U_out[:, :k]        # (intermediate, k) — W_up LSVs in neuron space
    V_in_k  = Vh_in[:k, :].T      # (intermediate, k) — W_down RSVs in neuron space

    # overlap matrix: M[i,j] = <u_i^out, v_j^in>^2
    M = (U_out_k.T @ V_in_k) ** 2  # (k, k)

    # cos-sim of each singular vector with the all-ones direction (1/sqrt(N))
    ones = np.ones(n_intermediate) / np.sqrt(n_intermediate)
    cossim_out = U_out_k.T @ ones   # (k,) — one value per LSV of fan_out
    cossim_in  = V_in_k.T  @ ones   # (k,) — one value per RSV of fan_in

    # cos-sim of biases with the all-ones direction in their respective spaces
    b_up   = w["bias_up"]    # (n_intermediate,)  — lives in neuron space
    b_down = w["bias_down"]  # (n_hidden,)         — lives in hidden space
    ones_hid = np.ones(b_down.shape[0]) / np.sqrt(b_down.shape[0])
    bias_cossim_up   = float(b_up   / (np.linalg.norm(b_up)   + 1e-12) @ ones)
    bias_cossim_down = float(b_down / (np.linalg.norm(b_down) + 1e-12) @ ones_hid)

    result = {
        "M":              M,
        "S_out":          S_out,
        "S_in":           S_in,
        "cossim_out":       cossim_out,
        "cossim_in":        cossim_in,
        "bias_cossim_up":   np.array(bias_cossim_up),
        "bias_cossim_down": np.array(bias_cossim_down),
        "top_k":            np.array(k),
        "n_intermediate": np.array(n_intermediate),
        "layer":          np.array(layer_idx),
        "model":          np.array(model_name),
        # singular vectors — stored as float32 to keep file size reasonable
        # W_up (fan_out, shape int×hid): LSVs in neuron space, RSVs in hidden space
        # W_down (fan_in, shape hid×int): LSVs in hidden space, RSVs in neuron space
        "U_up":    U_out[:, :k].astype(np.float32),    # (n_intermediate, k)
        "Vh_up":   Vh_out[:k, :].astype(np.float32),   # (k, n_hidden)
        "U_down":  U_in[:, :k].astype(np.float32),     # (n_hidden, k)
        "Vh_down": Vh_in[:k, :].astype(np.float32),    # (k, n_intermediate)
    }
    svd.save_results(name, result)
    return result


def fanout_fanin_overlap_all_layers(model_name: str) -> dict:
    """
    Run fanout_fanin_overlap for every layer. Results cached per-layer via
    fanout_fanin_overlap; this function just collects them and caches a combined
    summary (align_k curves + M corners) for fast rendering.
    Returns dict: {layer_idx -> per-layer result dict}
    """
    name = f"fanout_fanin_overlap_all_layers_{model_name}"
    n_layers = weights.get_num_layers(model_name)

    results = {}
    for layer_idx in range(n_layers):
        results[layer_idx] = fanout_fanin_overlap(model_name, layer_idx)
    return results


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
