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
import matplotlib.pyplot as plt

WEB_DIR = Path(__file__).parent.parent / "web"
EXPTS_DIR = WEB_DIR / "expts"
MASTER_INDEX = WEB_DIR / "index.html"
REGISTRY_FILE = WEB_DIR / "registry.json"  # tracks all rendered experiments

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
    figures: list[plt.Figure],
    description: str = "",
    grid_cols: int = 4,
    fig_labels: list[str] = None,
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
        f"""<li>
  <a href="expts/{e['slug']}/index.html">{e['title']}</a>
  <div class="desc">{e.get('description', '')}</div>
</li>"""
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
