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
    import numpy as np
    spectra = experiments.singular_spectra_all_layers(model, matrix_type)

    # read matrix shape from cached result (saved at compute time)
    cached = load_results(f"spectra_all_layers_{model}_{matrix_type}")
    m, n = int(cached["matrix_shape"][0]), int(cached["matrix_shape"][1])

    random_result = experiments.random_matrix_spectrum(m, n)

    n_layers = len(spectra)
    slug = f"singular_spectra_all_layers_{model}_{matrix_type}"
    title = f"{model} — {matrix_type} singular spectra, all {n_layers} layers"
    desc = (f"Singular value spectra of the MLP {matrix_type} weight matrix "
            f"({m}×{n}) for each of the {n_layers} layers in {model}. "
            f"Dashed gray: random iid Gaussian baseline. "
            f"Dashed red/blue lines: Marchenko-Pastur bulk edges.")

    render.write_interactive_spectra_page(
        slug=slug,
        title=title,
        spectra=spectra,
        random_S=random_result["S_mean"],
        mp_min=float(random_result["mp_min"]),
        mp_max=float(random_result["mp_max"]),
        description=desc,
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
            parts = slug[len("singular_spectra_all_layers_"):]
            for mt in ["fan_out", "fan_in"]:
                if parts.endswith(f"_{mt}"):
                    model = parts[: -(len(mt) + 1)]
                    run_singular_spectra_all_layers(model, mt)
                    break

        elif slug.startswith("midlayer_spectra_comparison_"):
            mt = slug[len("midlayer_spectra_comparison_"):]
            run_midlayer_spectra_comparison(PYTHIA_MODELS, mt)

        elif slug.startswith("fanout_fanin_overlap_all_layers_"):
            model = slug[len("fanout_fanin_overlap_all_layers_"):]
            cmd_expt(argparse.Namespace(name="fanout_fanin_overlap_all_layers", model=model,
                                        models=None, layer=None, top_k=32, matrix_type="fan_out"))

        elif slug.startswith("fanout_fanin_overlap_"):
            result_name = slug
            if results_exist(result_name):
                result = load_results(result_name)
                render.write_overlap_page(slug=slug, title=entry["title"], result=result)

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

    elif name == "fanout_fanin_overlap":
        import numpy as np
        model = args.model or "pythia-1b"
        result = experiments.fanout_fanin_overlap(model, args.layer)
        layer = int(result["layer"])
        slug = f"fanout_fanin_overlap_{model}_layer{layer}"

        # load random baselines and n_intermediate from spectra cache
        rand_out, rand_in = None, None
        n_intermediate_inferred = None
        for mt in ["fan_out", "fan_in"]:
            cached_name = f"spectra_all_layers_{model}_{mt}"
            if results_exist(cached_name):
                cached = load_results(cached_name)
                if "matrix_shape" in cached:
                    m, n = int(cached["matrix_shape"][0]), int(cached["matrix_shape"][1])
                    if mt == "fan_out":
                        n_intermediate_inferred = m  # fan_out is (intermediate, hidden)
                    rand_cache = f"random_spectrum_{m}x{n}"
                    if results_exist(rand_cache):
                        r = load_results(rand_cache)
                        if mt == "fan_out": rand_out = r["S_mean"].tolist()
                        else:               rand_in  = r["S_mean"].tolist()

        # patch n_intermediate into result if not already stored (old cache files lack it)
        if "n_intermediate" not in result and n_intermediate_inferred is not None:
            result = dict(result)
            import numpy as np
            result["n_intermediate"] = np.array(n_intermediate_inferred)

        render.write_overlap_page(
            slug=slug,
            title=f"{model} layer {layer} — W_out / W_in overlap",
            result=result,
            random_out=rand_out,
            random_in=rand_in,
        )

    elif name == "fanout_fanin_overlap_all_layers":
        import numpy as np
        model = args.model or "pythia-1b"

        # load random baselines and n_intermediate from spectra cache
        rand_out, rand_in = None, None
        n_intermediate_inferred = None
        for mt in ["fan_out", "fan_in"]:
            cached_name = f"spectra_all_layers_{model}_{mt}"
            if results_exist(cached_name):
                cached = load_results(cached_name)
                if "matrix_shape" in cached:
                    m, n = int(cached["matrix_shape"][0]), int(cached["matrix_shape"][1])
                    if mt == "fan_out":
                        n_intermediate_inferred = m
                    rand_cache = f"random_spectrum_{m}x{n}"
                    if results_exist(rand_cache):
                        r = load_results(rand_cache)
                        if mt == "fan_out": rand_out = r["S_mean"].tolist()
                        else:               rand_in  = r["S_mean"].tolist()

        results = experiments.fanout_fanin_overlap_all_layers(model)

        # patch n_intermediate into any layers that lack it (old cache files)
        if n_intermediate_inferred is not None:
            for layer_idx, result in results.items():
                if "n_intermediate" not in result:
                    results[layer_idx] = dict(result)
                    results[layer_idx]["n_intermediate"] = np.array(n_intermediate_inferred)

        slug = f"fanout_fanin_overlap_all_layers_{model}"
        render.write_overlap_all_layers_page(
            slug=slug,
            title=f"{model} — W_out / W_in overlap, all layers",
            results=results,
            random_out=rand_out,
            random_in=rand_in,
            n_intermediate=n_intermediate_inferred,
        )

    elif name == "next_token":
        from src import inference

        custom_prompts = [
            "The rain in Spain falls mainly on the",
            "I have a lot of frustrated energy from the reading",
            "Y'all made it to Imbue? No murdered",
        ]
        # strip last word from each Pile sentence so model predicts it
        pile_sentences = inference.sample_pile_sentences(n_sentences=4, seed=42)
        pile_prompts = [" ".join(s.split()[:-1]) for s in pile_sentences]

        prompts = custom_prompts + pile_prompts
        pile_indices = list(range(len(custom_prompts), len(prompts)))

        models = args.models.split(",") if args.models else PYTHIA_MODELS
        top_k = args.top_k or 15

        results = inference.run_next_token(prompts, model_names=models, top_k=top_k)
        perplexity = inference.run_perplexity(pile_sentences, model_names=models)

        slug = "next_token_pythia"
        render.write_next_token_page(
            slug=slug,
            title="Pythia — next-token predictions & perplexity",
            prompts=prompts,
            results=results,
            model_names=models,
            perplexity=perplexity,
            pile_indices=pile_indices,
            description=(
                "Top next-token predictions from each Pythia model size. "
                "Prompts marked [Pile] are truncated sentences from the training corpus. "
                "Bars normalized to the top token per panel."
            ),
        )

    else:
        print(f"Unknown experiment: {name}")
        print("Available: singular_spectra_all_layers, midlayer_spectra_comparison, fanin_fanout_alignment, fanout_fanin_overlap, fanout_fanin_overlap_all_layers, next_token")
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
