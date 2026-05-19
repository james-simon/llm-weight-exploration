"""
Renders experiment figures into a single index.html page.
"""

import base64
import io
from pathlib import Path
import matplotlib.pyplot as plt

PLOTS_DIR = Path(__file__).parent.parent / "plots"
WEB_DIR   = Path(__file__).parent.parent / "web"


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def render_html(sections: list[dict]) -> str:
    """
    sections: list of {"title": str, "description": str, "fig": Figure}
    Returns full HTML string.
    """
    parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LLM Weight Exploration</title>
<style>
  body { font-family: Georgia, serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #f9f9f9; color: #222; }
  h1 { font-size: 2em; border-bottom: 2px solid #ccc; padding-bottom: 10px; }
  h2 { font-size: 1.4em; margin-top: 50px; color: #333; }
  p  { color: #555; font-size: 0.95em; max-width: 800px; }
  img { display: block; margin: 16px 0; border: 1px solid #ddd; border-radius: 4px; }
  .section { margin-bottom: 60px; }
  .toc a { display: block; margin: 4px 0; color: #2266cc; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
</style>
</head>
<body>
<h1>LLM Weight Exploration</h1>
<div class="toc">
"""]

    for i, s in enumerate(sections):
        anchor = f"section-{i}"
        parts.append(f'  <a href="#{anchor}">{s["title"]}</a>\n')

    parts.append("</div>\n")

    for i, s in enumerate(sections):
        anchor = f"section-{i}"
        b64 = fig_to_base64(s["fig"])
        desc = s.get("description", "")
        parts.append(f"""
<div class="section" id="{anchor}">
  <h2>{s["title"]}</h2>
  {"<p>" + desc + "</p>" if desc else ""}
  <img src="data:image/png;base64,{b64}" />
</div>
""")

    parts.append("</body>\n</html>")
    return "".join(parts)


def write_page(sections: list[dict], path: Path = None):
    if path is None:
        path = WEB_DIR / "index.html"
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    html = render_html(sections)
    path.write_text(html)
    print(f"Wrote {path}")
    return path
