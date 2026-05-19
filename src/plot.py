"""
Plotting primitives. All functions return matplotlib Figure objects.
No data loading or computation here — takes pre-computed result dicts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def rcsetup():
    plt.rc("figure", dpi=120, facecolor=(1, 1, 1))
    plt.rc("font", family='stixgeneral', size=15)
    plt.rc("axes", titlesize=19)
    plt.rc("axes", facecolor=(1, 1, 1))
    plt.rc("mathtext", fontset='cm')


def _colors(n):
    return cm.viridis(np.linspace(0.1, 0.9, n))


# ── Single layer spectrum ─────────────────────────────────────────────────────

def plot_single_spectrum(S: np.ndarray, layer_idx: int, model_name: str, matrix_type: str = "fan_out") -> plt.Figure:
    """One panel: singular spectrum for a single layer."""
    rcsetup()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(S, lw=1.5, color="#2266cc")
    ax.set_yscale("log")
    ax.set_xlabel("Index")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_title(f"Layer {layer_idx}")
    fig.tight_layout()
    return fig


# ── Plot A ────────────────────────────────────────────────────────────────────

def plot_spectra_all_layers(spectra: dict, model_name: str, matrix_type: str = "fan_out") -> plt.Figure:
    """
    spectra: {layer_idx -> singular_values array}
    One line per layer, colored by depth.
    """
    rcsetup()
    n_layers = len(spectra)
    colors = _colors(n_layers)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (layer_idx, S) in enumerate(sorted(spectra.items())):
        ax.plot(S / S[0], color=colors[i], alpha=0.8, lw=1.2, label=f"L{layer_idx}" if n_layers <= 12 else None)

    ax.set_xlabel("Singular value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_0$")
    ax.set_title(f"{model_name} — {matrix_type} singular spectra")
    ax.set_yscale("log")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, n_layers - 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Layer")

    fig.tight_layout()
    return fig


# ── Plot B ────────────────────────────────────────────────────────────────────

def plot_midlayer_comparison(spectra: dict, matrix_type: str = "fan_out") -> plt.Figure:
    """
    spectra: {model_name -> singular_values array}
    One line per model size, normalized.
    """
    rcsetup()
    model_names = list(spectra.keys())
    colors = _colors(len(model_names))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i, model_name in enumerate(model_names):
        S = spectra[model_name]
        x = np.arange(len(S)) / (len(S) - 1)  # normalize x to [0,1] for cross-size comparison
        label = model_name.replace("pythia-", "")

        axes[0].plot(S / S[0], color=colors[i], lw=1.5, label=label)
        axes[1].plot(x, S / S[0], color=colors[i], lw=1.5, label=label)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel(r"$\sigma_i / \sigma_0$")
        ax.legend(fontsize=11, ncol=2)

    axes[0].set_xlabel("Singular value index")
    axes[0].set_title(f"Mid-layer {matrix_type} spectra (raw index)")
    axes[1].set_xlabel("Fractional rank")
    axes[1].set_title(f"Mid-layer {matrix_type} spectra (normalized rank)")

    fig.tight_layout()
    return fig


# ── Plot C ────────────────────────────────────────────────────────────────────

def plot_fanin_fanout_alignment(result: dict) -> plt.Figure:
    """
    result: output of experiments.fanin_fanout_alignment()
    Three panels: singular spectra, principal angles, alignment score.
    """
    rcsetup()
    model_name = str(result["model"])
    layer = int(result["layer"])
    top_k = int(result["top_k"])
    S_out = result["S_fan_out"]
    S_in  = result["S_fan_in"]
    angles = result["principal_angles"]
    alignment = float(result["subspace_alignment"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{model_name} layer {layer} — fan-in / fan-out alignment", fontsize=16)

    # Panel 1: singular spectra
    axes[0].plot(S_out / S_out[0], label="fan_out", lw=1.5)
    axes[0].plot(S_in  / S_in[0],  label="fan_in",  lw=1.5, linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel(r"$\sigma_i / \sigma_0$")
    axes[0].set_title("Singular spectra")
    axes[0].legend()

    # Panel 2: principal angles between top-k subspaces
    axes[1].plot(np.degrees(angles), "o-", ms=4, lw=1.5)
    axes[1].axhline(90, color="gray", lw=1, linestyle="--", label="Random (90°)")
    axes[1].set_xlabel("Principal angle index")
    axes[1].set_ylabel("Angle (degrees)")
    axes[1].set_title(f"Principal angles (top-{top_k})")
    axes[1].legend(fontsize=11)

    # Panel 3: alignment score summary bar
    axes[2].bar(["Subspace\nalignment"], [alignment], color="steelblue", width=0.4)
    axes[2].bar(["Random\nbaseline"], [1 / top_k], color="gray", width=0.4)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Score")
    axes[2].set_title(f"Mean sq. cosine sim (k={top_k})")

    fig.tight_layout()
    return fig
