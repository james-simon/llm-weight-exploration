"""
CLI for running experiments and rendering the results page.

Usage:
  python run.py expt singular_spectra_all_layers --model pythia-70m
  python run.py expt midlayer_spectra_comparison
  python run.py expt fanin_fanout_alignment --model pythia-70m --layer 3
  python run.py render
  python run.py serve [--port 8000]
"""

import argparse
import http.server
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src import experiments, plot, render
from src.weights import PYTHIA_MODELS


def cmd_expt(args):
    name = args.name

    if name == "singular_spectra_all_layers":
        model = args.model or "pythia-70m"
        spectra = experiments.singular_spectra_all_layers(model, args.matrix_type)
        fig = plot.plot_spectra_all_layers(spectra, model, args.matrix_type)
        fig.savefig(f"plots/spectra_all_layers_{model}_{args.matrix_type}.png", bbox_inches="tight")
        print(f"Saved plot.")

    elif name == "midlayer_spectra_comparison":
        models = args.models.split(",") if args.models else PYTHIA_MODELS
        spectra = experiments.midlayer_spectra_comparison(models, args.matrix_type)
        fig = plot.plot_midlayer_comparison(spectra, args.matrix_type)
        fig.savefig(f"plots/midlayer_comparison_{args.matrix_type}.png", bbox_inches="tight")
        print(f"Saved plot.")

    elif name == "fanin_fanout_alignment":
        model = args.model or "pythia-70m"
        layer = args.layer  # None means middle layer
        result = experiments.fanin_fanout_alignment(model, layer, args.top_k)
        layer_str = str(result["layer"])
        fig = plot.plot_fanin_fanout_alignment(result)
        fig.savefig(f"plots/alignment_{model}_layer{layer_str}.png", bbox_inches="tight")
        print(f"Saved plot.")

    else:
        print(f"Unknown experiment: {name}")
        print("Available: singular_spectra_all_layers, midlayer_spectra_comparison, fanin_fanout_alignment")
        sys.exit(1)


def cmd_render(args):
    """Regenerate index.html from all cached results."""
    import matplotlib
    matplotlib.use("Agg")

    sections = []

    # Experiment A: all-layer spectra for each model that has cached results
    for model in PYTHIA_MODELS:
        for matrix_type in ["fan_out", "fan_in"]:
            result_name = f"spectra_all_layers_{model}_{matrix_type}"
            from src.svd import results_exist, load_results
            from src.weights import get_num_layers
            if results_exist(result_name):
                data = load_results(result_name)
                n_layers = get_num_layers(model)
                spectra = {i: data[f"layer_{i}"] for i in range(n_layers)}
                fig = plot.plot_spectra_all_layers(spectra, model, matrix_type)
                sections.append({
                    "title": f"[A] {model} — {matrix_type} spectra, all layers",
                    "fig": fig,
                })

    # Experiment B: mid-layer comparison
    for matrix_type in ["fan_out", "fan_in"]:
        available = [m for m in PYTHIA_MODELS
                     if results_exist(f"spectra_all_layers_{m}_{matrix_type}") or
                        any(results_exist(f"midlayer_comparison_{s}_{matrix_type}")
                            for s in ["_".join(m2.split('-')[1] for m2 in PYTHIA_MODELS)])]
        # simpler: just check the dedicated midlayer cache
        from src.svd import results_exist as re
        cache_name = f"midlayer_comparison_{'_'.join(m.split('-')[1] for m in PYTHIA_MODELS)}_{matrix_type}"
        if re(cache_name):
            from src.svd import load_results as lr
            data = lr(cache_name)
            spectra = {m: data[m] for m in PYTHIA_MODELS if m in data}
            if spectra:
                fig = plot.plot_midlayer_comparison(spectra, matrix_type)
                sections.append({
                    "title": f"[B] Mid-layer {matrix_type} spectra — all model sizes",
                    "fig": fig,
                })

    # Experiment C: fan-in / fan-out alignment
    import re as re_module
    results_dir = Path("results")
    if results_dir.exists():
        for npz in sorted(results_dir.glob("fanin_fanout_alignment_*.npz")):
            result_name = npz.stem
            from src.svd import load_results as lr
            result = lr(result_name)
            fig = plot.plot_fanin_fanout_alignment(result)
            model = str(result["model"])
            layer = int(result["layer"])
            sections.append({
                "title": f"[C] {model} layer {layer} — fan-in/fan-out alignment",
                "fig": fig,
            })

    if not sections:
        print("No cached results found. Run some experiments first.")
        return

    render.write_page(sections)
    print(f"Rendered {len(sections)} section(s) to web/index.html")


def cmd_serve(args):
    port = args.port
    os.chdir(Path(__file__).parent / "web")
    print(f"Serving at http://localhost:{port}  (Ctrl-C to stop)")
    http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=port)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_expt = sub.add_parser("expt")
    p_expt.add_argument("name")
    p_expt.add_argument("--model")
    p_expt.add_argument("--models")
    p_expt.add_argument("--layer", type=int, default=None)
    p_expt.add_argument("--top-k", type=int, default=32)
    p_expt.add_argument("--matrix-type", default="fan_out", choices=["fan_out", "fan_in"])

    p_render = sub.add_parser("render")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.cmd == "expt":   cmd_expt(args)
    elif args.cmd == "render": cmd_render(args)
    elif args.cmd == "serve":  cmd_serve(args)


if __name__ == "__main__":
    main()
