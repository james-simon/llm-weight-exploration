"""
HTML rendering utilities.

Page structure:
  web/index.html                          — master index, links to all experiment pages
  web/expts/{expt_slug}/index.html        — one self-contained page per experiment
"""

import base64
import io
import json
from pathlib import Path
from string import Template
import matplotlib.pyplot as plt

WEB_DIR = Path(__file__).parent.parent / "web"
EXPTS_DIR = WEB_DIR / "expts"
MASTER_INDEX = WEB_DIR / "index.html"
REGISTRY_FILE = WEB_DIR / "registry.json"  # tracks all rendered experiments
TEMPLATES_DIR = WEB_DIR / "templates"


def _render_template(name: str, **kwargs) -> str:
    src = (TEMPLATES_DIR / name).read_text()
    return Template(src).safe_substitute(**kwargs)

CSS = """
body {
  font-family: Georgia, serif;
  max-width: 1200px;
  margin: 40px auto;
  padding: 0 24px;
  background: #f9f9f9;
  color: #222;
}
h1 { font-size: 2em; border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-bottom: 8px; }
h2 { font-size: 1.3em; margin-top: 40px; color: #333; }
p  { color: #555; font-size: 0.95em; max-width: 860px; line-height: 1.6; }
a  { color: #2266cc; text-decoration: none; }
a:hover { text-decoration: underline; }
.back { font-size: 0.9em; margin-bottom: 24px; display: inline-block; }
.meta { font-size: 0.85em; color: #888; margin: 4px 0 20px; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 16px; }
.grid img { border: 1px solid #ddd; border-radius: 4px; max-width: 100%; }
.grid-2 img { width: calc(50% - 6px); }
.grid-4 img { width: calc(25% - 9px); }
.expt-list { list-style: none; padding: 0; margin-top: 20px; }
.expt-list li { margin: 0; border-bottom: 1px solid #eee; }
.expt-list li a { display: block; padding: 12px 4px; font-size: 1.05em; }
.expt-list li .desc { font-size: 0.88em; color: #777; margin: 2px 0 8px 0; }
"""


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _html_skeleton(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


def write_experiment_page(
    slug: str,
    title: str,
    figures: list,
    description: str = "",
    grid_cols: int = 4,
    fig_labels: list = None,
) -> Path:
    """
    Write a self-contained experiment page with a grid of figures.

    slug:       URL-safe identifier, becomes web/expts/{slug}/index.html
    title:      Human-readable title shown as <h1>
    figures:    List of matplotlib Figures, one per panel
    description: Optional paragraph shown below the title
    grid_cols:  Number of columns in the image grid (1, 2, or 4)
    fig_labels: Optional list of captions below each figure
    """
    out_dir = EXPTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_class = {1: "", 2: "grid-2", 4: "grid-4"}.get(grid_cols, "grid-4")

    imgs_html = []
    for i, fig in enumerate(figures):
        b64 = fig_to_base64(fig)
        label = fig_labels[i] if fig_labels and i < len(fig_labels) else ""
        caption = f"<div style='font-size:0.8em;color:#666;text-align:center'>{label}</div>" if label else ""
        imgs_html.append(f'<div><img src="data:image/png;base64,{b64}" />{caption}</div>')

    desc_html = f"<p>{description}</p>" if description else ""
    body = f"""<a class="back" href="../../index.html">← Back to experiments</a>
<h1>{title}</h1>
{desc_html}
<div class="grid {grid_class}">
{"".join(imgs_html)}
</div>"""

    path = out_dir / "index.html"
    path.write_text(_html_skeleton(title, body))
    print(f"  Wrote {path}")

    _register_experiment(slug, title, description, path)
    return path


def _load_registry() -> list:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return []


def _register_experiment(slug: str, title: str, description: str, path: Path):
    registry = _load_registry()
    entry = {"slug": slug, "title": title, "description": description}
    registry = [e for e in registry if e["slug"] != slug]  # remove stale entry
    registry.append(entry)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
    write_master_index(registry)


def write_master_index(registry: list = None):
    if registry is None:
        registry = _load_registry()

    items = "\n".join(
        f'<li><a href="expts/{e["slug"]}/index.html">{e["title"]}</a></li>'
        for e in registry
    )

    body = f"""<h1>LLM Weight Exploration</h1>
<p>Spectral analysis of weight matrices in Pythia language models.</p>
<ul class="expt-list">
{items}
</ul>"""

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    MASTER_INDEX.write_text(_html_skeleton("LLM Weight Exploration", body))
    print(f"  Updated {MASTER_INDEX}")


def write_interactive_spectra_page(
    slug: str,
    title: str,
    spectra: dict,           # {layer_idx: np.ndarray of singular values}
    random_S: "np.ndarray",  # singular values of a random matrix same shape
    mp_min: float,
    mp_max: float,
    description: str = "",
) -> Path:
    """
    Write a fully interactive spectra viewer page.
    - Checkboxes to toggle layers on/off
    - One big canvas plot with depth-colored lines
    - Random matrix baseline (numerical + MP theory)
    - Settings menu: log x, log y, normalize y
    """
    import numpy as np

    # Serialize all data as JSON for embedding
    layers_data = {str(k): v.tolist() for k, v in sorted(spectra.items())}
    n_layers = len(layers_data)
    random_data = random_S.tolist()

    data_json = json.dumps({
        "layers": layers_data,
        "random": random_data,
        "mp_min": float(mp_min),
        "mp_max": float(mp_max),
        "n_layers": n_layers,
    })

    out_dir = EXPTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    desc_html = f"<p>{description}</p>" if description else ""

    html = _render_template("spectra.html", css=CSS, title=title, description=desc_html, slug=slug, data_json=data_json)


    path = out_dir / "index.html"
    path.write_text(html)
    print(f"  Wrote {path}")
    _register_experiment(slug, title, description, path)
    return path


def write_overlap_page(
    slug: str,
    title: str,
    result: dict,           # output of experiments.fanout_fanin_overlap()
    random_out: list = None,  # singular values of random matrix, same shape as fan_out
    random_in:  list = None,  # singular values of random matrix, same shape as fan_in
) -> Path:
    """
    Interactive page with three panels:
      1. Spectra of fan_out and fan_in on the same plot (checkable, log/linear)
      2. 20x20 heatmap of top-left corner of M
      3. align_k vs k, with random baseline
    """
    import numpy as np

    M      = result["M"]
    S_out  = result["S_out"].tolist()
    S_in   = result["S_in"].tolist()
    n_rank = int(result["top_k"])
    layer  = int(result["layer"])
    model  = str(result["model"])
    # n_intermediate: ambient neuron-space dimension (e.g. 8192 for pythia-1b)
    # random baseline for align_k is k/n_intermediate, not k/n_rank
    n_intermediate = int(result["n_intermediate"]) if "n_intermediate" in result else None

    # align_k curve: cumulative mean of top-left k×k block
    # align_k = (1/k) * sum_{i,j<=k} M[i,j]  (0-indexed, so block is M[:k, :k])
    align_k = []
    for k in range(1, n_rank + 1):
        block_sum = float(M[:k, :k].sum())
        align_k.append(block_sum / k)

    heatmap_n = 20
    M_corner  = M[:heatmap_n, :heatmap_n].tolist()

    data_json = json.dumps({
        "S_out":          S_out,
        "S_in":           S_in,
        "S_rand_out":     random_out if random_out is not None else [],
        "S_rand_in":      random_in  if random_in  is not None else [],
        "align_k":        align_k,
        "M_corner":       M_corner,
        "heatmap_n":      heatmap_n,
        "n_rank":         n_rank,
        "n_intermediate": n_intermediate,
        "layer":          layer,
        "model":          model,
    })

    out_dir = EXPTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    html = _render_template("overlap.html", css=CSS, title=title, description="", slug=slug, data_json=data_json, heatmap_n=heatmap_n, first_n_rank=n_rank, n_intermediate=n_intermediate, n_hidden=n_rank, layer=layer, model=model)

    path = out_dir / "index.html"
    path.write_text(html)
    print(f"  Wrote {path}")
    _register_experiment(slug, title, "", path)
    return path


def write_overlap_all_layers_page(
    slug: str,
    title: str,
    results: dict,           # {layer_idx -> result dict from fanout_fanin_overlap}
    random_out: list = None,
    random_in:  list = None,
    n_intermediate: int = None,
) -> Path:
    """
    All-layers overlap page: same three panels as write_overlap_page, but with
    a layer-selector toggle bar at the top. Clicking a layer updates all plots.
    """
    import numpy as np

    layer_indices = sorted(results.keys())
    n_layers = len(layer_indices)
    heatmap_n = 20

    # write per-layer vector files to web/data/{slug}/ for fetch()-based loading
    vec_dir = WEB_DIR / "data" / slug
    vec_dir.mkdir(parents=True, exist_ok=True)

    # build per-layer data
    layers_data = {}
    for layer_idx in layer_indices:
        r = results[layer_idx]
        M = r["M"]
        n_rank = int(r["top_k"])
        S_out = r["S_out"].tolist()
        S_in  = r["S_in"].tolist()

        align_k = []
        for k in range(1, n_rank + 1):
            align_k.append(float(M[:k, :k].sum()) / k)

        ni = int(r["n_intermediate"]) if "n_intermediate" in r else n_intermediate
        n_hid = r["S_out"].shape[0]  # rank = n_hidden

        layers_data[layer_idx] = {
            "S_out":          S_out,
            "S_in":           S_in,
            "align_k":        align_k,
            "M_corner":       M[:heatmap_n, :heatmap_n].tolist(),
            "n_rank":         n_rank,
            "n_intermediate": ni,
            "n_hidden":       n_hid,
            "cossim_out":     r["cossim_out"].tolist() if "cossim_out" in r else [],
            "cossim_in":      r["cossim_in"].tolist()  if "cossim_in"  in r else [],
            "bias_up":        r["bias_up"].tolist()     if "bias_up"    in r else [],
        }

        # write binary float32 vector files: one file per (matrix, side)
        # shape: (n_vecs, dim) stored row-major as float32
        for key, arr_key, label in [
            ("U_up",    "U_up",    "U_up"),
            ("Vh_up",   "Vh_up",   "Vh_up"),
            ("U_down",  "U_down",  "U_down"),
            ("Vh_down", "Vh_down", "Vh_down"),
        ]:
            if arr_key in r:
                arr = r[arr_key]  # shape: (dim, n_vecs) for U, (n_vecs, dim) for Vh
                # normalise to (n_vecs, dim) so JS can slice by row
                if arr_key.startswith("U_"):
                    mat = arr.T.astype(np.float32)   # (n_vecs, dim)
                else:
                    mat = arr.astype(np.float32)      # (n_vecs, dim) — Vh already (k, dim)
                fpath = vec_dir / f"layer{layer_idx}_{key}.bin"
                mat.tofile(str(fpath))
                layers_data[layer_idx][f"dim_{key}"] = mat.shape[1]  # dim of each vector

    # use first layer's n_intermediate as fallback for random baseline
    ni_fallback = n_intermediate or next(
        (int(r["n_intermediate"]) for r in results.values() if "n_intermediate" in r), None
    )

    bias_cossim_up   = [float(results[li]["bias_cossim_up"])   if "bias_cossim_up"   in results[li] else None for li in layer_indices]
    bias_cossim_down = [float(results[li]["bias_cossim_down"]) if "bias_cossim_down" in results[li] else None for li in layer_indices]

    data_json = json.dumps({
        "layers":          layers_data,
        "layer_indices":   layer_indices,
        "S_rand_out":      random_out if random_out is not None else [],
        "S_rand_in":       random_in  if random_in  is not None else [],
        "heatmap_n":       heatmap_n,
        "n_intermediate":  ni_fallback,
        "bias_cossim_up":  bias_cossim_up,
        "bias_cossim_down": bias_cossim_down,
    })

    out_dir = EXPTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    first_n_rank = layers_data[layer_indices[0]]["n_rank"]
    n_intermediate = ni_fallback or 0
    n_hidden = first_n_rank  # rank of fan_out = min(d_int, d_hid) = d_hid

    html = _render_template("overlap_all_layers.html", css=CSS, title=title, description="", slug=slug, data_json=data_json, n_intermediate=n_intermediate, n_hidden=n_hidden, heatmap_n=heatmap_n, first_n_rank=first_n_rank)

    path = out_dir / "index.html"
    path.write_text(html)
    print(f"  Wrote {path}")
    _register_experiment(slug, title, "", path)
    return path
