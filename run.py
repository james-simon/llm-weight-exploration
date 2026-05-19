"""
CLI for running experiments and rendering the results webpage.

Usage:
  python run.py expt singular_spectra_all_layers --model pythia-1b
  python run.py expt midlayer_spectra_comparison
  python run.py expt fanin_fanout_alignment --model pythia-1b --layer 8
  python run.py render          # re-render all registered experiment pages
  python run.py serve [--port 8000]
"""

import argparse
import http.server
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
from src import experiments, plot, render
from src.svd import results_exist, load_results
from src.weights import PYTHIA_MODELS

Path("results").mkdir(exist_ok=True)


# ── Experiment runners ────────────────────────────────────────────────────────

def run_singular_spectra_all_layers(model: str, matrix_type: str):
    spectra = experiments.singular_spectra_all_layers(model, matrix_type)

    figs = [plot.plot_single_spectrum(S, layer_idx, model, matrix_type)
            for layer_idx, S in sorted(spectra.items())]
    labels = [f"Layer {i}" for i in sorted(spectra.keys())]

    n = len(figs)
    slug = f"singular_spectra_all_layers_{model}_{matrix_type}"
    title = f"{model} — {matrix_type} singular spectra, all {n} layers"
    desc = (f"Singular value spectrum of the MLP {matrix_type} weight matrix "
            f"for each of the {n} layers in {model}. "
            f"Matrix shape: {list(spectra.values())[0].shape[0]} singular values per layer.")

    render.write_experiment_page(
        slug=slug,
        title=title,
        figures=figs,
        description=desc,
        grid_cols=2,
    )


def run_midlayer_spectra_comparison(models: list, matrix_type: str):
    spectra = experiments.midlayer_spectra_comparison(models, matrix_type)

    fig = plot.plot_midlayer_comparison(spectra, matrix_type)
    slug = f"midlayer_spectra_comparison_{matrix_type}"
    title = f"Mid-layer {matrix_type} spectra — Pythia model comparison"
    desc = ("Singular value spectrum of the middle-layer MLP weight matrix "
            "for each Pythia model size, normalized for comparison.")

    render.write_experiment_page(
        slug=slug,
        title=title,
        figures=[fig],
        description=desc,
        grid_cols=1,
    )


def run_fanin_fanout_alignment(model: str, layer: int, top_k: int):
    result = experiments.fanin_fanout_alignment(model, layer, top_k)
    layer_idx = int(result["layer"])

    fig = plot.plot_fanin_fanout_alignment(result)
    slug = f"fanin_fanout_alignment_{model}_layer{layer_idx}"
    title = f"{model} layer {layer_idx} — fan-in / fan-out alignment"
    desc = (f"SVD alignment analysis between the fan-out and fan-in MLP weight matrices "
            f"at layer {layer_idx} of {model}. "
            f"Principal angles computed between top-{top_k} singular subspaces.")

    render.write_experiment_page(
        slug=slug,
        title=title,
        figures=[fig],
        description=desc,
        grid_cols=1,
    )


# ── Re-render all registered experiments ─────────────────────────────────────

def cmd_render(_args):
    registry = render._load_registry()
    if not registry:
        print("No registered experiments. Run some experiments first.")
        return

    print(f"Re-rendering {len(registry)} experiment(s)...")
    for entry in registry:
        slug = entry["slug"]

        if slug.startswith("singular_spectra_all_layers_"):
            # parse model and matrix_type from slug
            parts = slug[len("singular_spectra_all_layers_"):]
            for mt in ["fan_out", "fan_in"]:
                if parts.endswith(f"_{mt}"):
                    model = parts[: -(len(mt) + 1)]
                    run_singular_spectra_all_layers(model, mt)
                    break

        elif slug.startswith("midlayer_spectra_comparison_"):
            mt = slug[len("midlayer_spectra_comparison_"):]
            run_midlayer_spectra_comparison(PYTHIA_MODELS, mt)

        elif slug.startswith("fanin_fanout_alignment_"):
            # can't easily re-derive args from slug alone; re-run from cached results
            result_name = slug  # slug matches result name
            if results_exist(result_name):
                result = load_results(result_name)
                fig = plot.plot_fanin_fanout_alignment(result)
                render.write_experiment_page(
                    slug=slug,
                    title=entry["title"],
                    figures=[fig],
                    description=entry.get("description", ""),
                    grid_cols=1,
                )


# ── CLI ───────────────────────────────────────────────────────────────────────

def cmd_expt(args):
    name = args.name

    if name == "singular_spectra_all_layers":
        run_singular_spectra_all_layers(args.model or "pythia-1b", args.matrix_type)

    elif name == "midlayer_spectra_comparison":
        models = args.models.split(",") if args.models else PYTHIA_MODELS
        run_midlayer_spectra_comparison(models, args.matrix_type)

    elif name == "fanin_fanout_alignment":
        run_fanin_fanout_alignment(args.model or "pythia-1b", args.layer, args.top_k)

    else:
        print(f"Unknown experiment: {name}")
        print("Available: singular_spectra_all_layers, midlayer_spectra_comparison, fanin_fanout_alignment")
        sys.exit(1)


def cmd_serve(args):
    os.chdir(Path(__file__).parent / "web")
    print(f"Serving at http://localhost:{args.port}  (Ctrl-C to stop)")
    http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=args.port)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_expt = sub.add_parser("expt")
    p_expt.add_argument("name")
    p_expt.add_argument("--model", default=None)
    p_expt.add_argument("--models", default=None)
    p_expt.add_argument("--layer", type=int, default=None)
    p_expt.add_argument("--top-k", type=int, default=32)
    p_expt.add_argument("--matrix-type", default="fan_out", choices=["fan_out", "fan_in"])

    sub.add_parser("render")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    {"expt": cmd_expt, "render": cmd_render, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    main()
