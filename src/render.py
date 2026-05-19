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

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{CSS}
.controls {{ margin: 16px 0 8px; }}
.layer-checkboxes {{
  display: flex; flex-wrap: wrap; gap: 4px 10px;
  margin: 10px 0 12px; font-size: 0.85em;
}}
.layer-checkboxes label {{ display: flex; align-items: center; gap: 4px; cursor: pointer; white-space: nowrap; }}
.layer-checkboxes input {{ cursor: pointer; }}
.cb-row {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }}
.toggle-btn {{
  font-size: 0.8em; padding: 2px 8px; cursor: pointer;
  border: 1px solid #bbb; border-radius: 3px; background: #f0f0f0;
}}
.toggle-btn:hover {{ background: #e0e0e0; }}

/* plot wrapper + gear (mlp-sharpness style) */
.plot-wrap {{
  position: relative;
  width: 100%; max-width: 975px;
  margin: 0 auto;
}}
.plot-wrap canvas {{
  display: block; width: 100%; height: 544px;
  border: 1px solid #ddd; border-radius: 4px; background: #fff;
}}
.plot-gear {{
  position: absolute; top: 6px; right: 6px;
  width: 22px; height: 22px; padding: 0;
  background: rgba(255,255,255,0.85);
  border: 1px solid #ccc; border-radius: 4px;
  cursor: pointer; font-size: 13px; line-height: 22px; text-align: center;
  color: #555; opacity: 0; transition: opacity 0.15s; z-index: 10;
  user-select: none;
}}
.plot-wrap:hover .plot-gear {{ opacity: 1; }}
.plot-gear:hover {{ background: #fff; border-color: #999; color: #222; }}
.gear-menu {{
  position: absolute; top: 30px; right: 6px; z-index: 100;
  background: #fff; border: 1px solid #ccc; border-radius: 5px;
  padding: 6px 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  min-width: 170px; display: none;
}}
.gear-menu.open {{ display: block; }}
.gear-menu label {{
  display: flex; align-items: center; gap: 6px;
  padding: 3px 2px; cursor: pointer; font-size: 0.88em; white-space: nowrap;
}}
.gear-menu label:hover {{ background: #f4f4f4; border-radius: 3px; }}
</style>
</head>
<body>
<a class="back" href="../../index.html">← Back to experiments</a>
<h1>{title}</h1>
{desc_html}

<div class="controls">
  <div class="cb-row">
    <strong style="font-size:0.9em">Layers:</strong>
    <button class="toggle-btn" id="btn-all">all</button>
    <button class="toggle-btn" id="btn-none">none</button>
    <button class="toggle-btn" id="btn-every2">every 2nd</button>
  </div>
  <div class="layer-checkboxes" id="layer-checkboxes"></div>
</div>

<div class="plot-wrap" id="plot-wrap">
  <canvas id="plot"></canvas>
  <div class="plot-gear" id="gear-btn">⚙</div>
  <div class="gear-menu" id="gear-menu">
    <label><input type="checkbox" id="opt-logy" checked> log y</label>
    <label><input type="checkbox" id="opt-logx"> log x</label>
    <label><input type="checkbox" id="opt-normy"> normalize (σ/σ₀)</label>
    <label><input type="checkbox" id="opt-normmean"> normalize (mean σ = 1)</label>
    <label><input type="checkbox" id="opt-random" checked> random baseline</label>
    <label><input type="checkbox" id="opt-mp" checked> M-P theory</label>
  </div>
</div>

<div style="margin-top: 18px;">
  <div style="font-size:0.85em; color:#888; margin-bottom:6px; letter-spacing:0.3px;">Singular value distribution</div>
  <div class="plot-wrap" id="hist-wrap">
    <canvas id="hist"></canvas>
    <div class="plot-gear" id="hist-gear-btn">⚙</div>
    <div class="gear-menu" id="hist-gear-menu">
      <label><input type="checkbox" id="hist-logy"> log y</label>
      <label><input type="checkbox" id="hist-logx"> log x (σ)</label>
      <label><input type="checkbox" id="hist-density" checked> density (area=1)</label>
      <label><input type="checkbox" id="hist-random" checked> random baseline</label>
      <label style="gap:6px;">bins&nbsp;<input type="number" id="hist-bins" value="60" min="10" max="300" style="width:48px;font-size:0.9em;padding:1px 4px;border:1px solid #ccc;border-radius:3px;"></label>
    </div>
  </div>
</div>

<script>
const RAW = {data_json};

// ── Color scale: red=early layers, violet=late ────────────────────────────────
function layerColor(i, n) {{
  const t = i / Math.max(n - 1, 1);
  // roygbiv: hue 0 (red) → 270 (violet)
  const hue = Math.round(t * 270);
  return `hsl(${{hue}}, 85%, 45%)`;
}}

// ── Build checkboxes ──────────────────────────────────────────────────────────
const cbContainer = document.getElementById('layer-checkboxes');
const layerKeys = Object.keys(RAW.layers).map(Number).sort((a,b) => a-b);
const checkboxes = {{}};
layerKeys.forEach(i => {{
  const label = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox'; cb.checked = true;
  cb.style.accentColor = layerColor(i, RAW.n_layers);
  checkboxes[i] = cb;
  const swatch = document.createElement('span');
  swatch.style.cssText = `display:inline-block;width:12px;height:12px;border-radius:2px;background:${{layerColor(i, RAW.n_layers)}}`;
  label.appendChild(cb); label.appendChild(swatch);
  label.append(` L${{i}}`);
  cbContainer.appendChild(label);
}});

// ── Spectrum gear menu ────────────────────────────────────────────────────────
const gearBtn  = document.getElementById('gear-btn');
const gearMenu = document.getElementById('gear-menu');
gearBtn.addEventListener('click', e => {{ gearMenu.classList.toggle('open'); e.stopPropagation(); }});
document.addEventListener('click', () => {{ gearMenu.classList.remove('open'); histGearMenu.classList.remove('open'); }});
gearMenu.addEventListener('click', e => e.stopPropagation());

// ── localStorage persistence ──────────────────────────────────────────────────
const STORAGE_KEY = 'spectra-state-{slug}';
const OPT_IDS = ['opt-logy','opt-logx','opt-normy','opt-normmean','opt-random','opt-mp',
                 'hist-logy','hist-logx','hist-density','hist-random'];
const OPT_DEFAULTS = {{ 'opt-logy': true, 'opt-random': true, 'opt-mp': true, 'hist-density': true, 'hist-random': true }};

function saveState() {{
  const state = {{}};
  OPT_IDS.forEach(id => state[id] = document.getElementById(id).checked);
  state['hist-bins'] = document.getElementById('hist-bins').value;
  layerKeys.forEach(i => state['layer-'+i] = checkboxes[i].checked);
  try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }} catch(e) {{}}
}}

function loadState() {{
  let saved = null;
  try {{ saved = JSON.parse(localStorage.getItem(STORAGE_KEY)); }} catch(e) {{}}
  OPT_IDS.forEach(id => {{
    const el = document.getElementById(id);
    el.checked = saved ? (saved[id] ?? (OPT_DEFAULTS[id] ?? false)) : (OPT_DEFAULTS[id] ?? false);
  }});
  if (saved?.['hist-bins']) document.getElementById('hist-bins').value = saved['hist-bins'];
  layerKeys.forEach(i => {{
    checkboxes[i].checked = saved ? (saved['layer-'+i] ?? true) : true;
  }});
}}

function redraw() {{ draw(); drawHist(); }}

loadState();
OPT_IDS.forEach(id => document.getElementById(id).addEventListener('change', () => {{ saveState(); redraw(); }}));
document.getElementById('hist-bins').addEventListener('input', () => {{ saveState(); drawHist(); }});
layerKeys.forEach(i => checkboxes[i].addEventListener('change', () => {{ saveState(); redraw(); }}));
document.getElementById('btn-all').onclick    = () => {{ layerKeys.forEach(i => checkboxes[i].checked = true);          saveState(); redraw(); }};
document.getElementById('btn-none').onclick   = () => {{ layerKeys.forEach(i => checkboxes[i].checked = false);         saveState(); redraw(); }};
document.getElementById('btn-every2').onclick = () => {{ layerKeys.forEach((k,i) => checkboxes[k].checked=(i%2===0));   saveState(); redraw(); }};

// histogram gear
const histGearBtn  = document.getElementById('hist-gear-btn');
const histGearMenu = document.getElementById('hist-gear-menu');
histGearBtn.addEventListener('click', e => {{ histGearMenu.classList.toggle('open'); e.stopPropagation(); }});
document.addEventListener('click', () => histGearMenu.classList.remove('open'));
histGearMenu.addEventListener('click', e => e.stopPropagation());

function getOpts() {{
  return {{
    logY:     document.getElementById('opt-logy').checked,
    logX:     document.getElementById('opt-logx').checked,
    normY:    document.getElementById('opt-normy').checked,
    normMean: document.getElementById('opt-normmean').checked,
    random:   document.getElementById('opt-random').checked,
    mp:       document.getElementById('opt-mp').checked,
  }};
}}

// ── Canvas drawing ────────────────────────────────────────────────────────────
const canvas = document.getElementById('plot');
const histCanvas = document.getElementById('hist');

function draw() {{
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth;
  const H = canvas.offsetHeight || 544;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const opts = getOpts();
  const PAD = {{ top: 30, right: 30, bottom: 50, left: 70 }};
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top  - PAD.bottom;

  function normalize(S) {{
    if (opts.normY)    return S.map(v => v / S[0]);
    if (opts.normMean) return S.map(v => v / (S.reduce((a,b) => a+b, 0) / S.length));
    return S;
  }}

  // Collect active series
  const series = [];
  layerKeys.forEach(i => {{
    if (!checkboxes[i].checked) return;
    series.push({{ S: normalize(RAW.layers[String(i)]), color: layerColor(i, RAW.n_layers), label: 'L'+i }});
  }});
  if (opts.random) {{
    series.push({{ S: normalize(RAW.random), color: '#888', label: 'random', dash: [4,3] }});
  }}

  if (series.length === 0) {{ ctx.clearRect(0, 0, W, H); return; }}

  // Axis ranges
  const allY = series.flatMap(s => s.S.filter(v => v > 0));
  const allX = series.map(s => s.S.length).reduce((a,b) => Math.max(a,b), 1);
  let xMin = opts.logX ? 1 : 0, xMax = allX;
  let yMin = Math.min(...allY), yMax = Math.max(...allY);
  if (opts.logY) {{ yMin = Math.max(yMin, 1e-10); }}

  // Marchenko-Pastur lines (in same y-units)
  const mpLines = [];
  if (opts.mp) {{
    let mpMax = RAW.mp_max, mpMin = RAW.mp_min;
    if (opts.normY) {{
      const r0 = RAW.random[0];
      mpMax /= r0; mpMin /= r0;
    }} else if (opts.normMean) {{
      const rMean = RAW.random.reduce((a,b) => a+b, 0) / RAW.random.length;
      mpMax /= rMean; mpMin /= rMean;
    }}
    mpLines.push({{ y: mpMax, color: '#d44', label: 'MP max', dash: [6,3] }});
    mpLines.push({{ y: mpMin, color: '#44d', label: 'MP min', dash: [6,3] }});
    yMax = Math.max(yMax, mpMax);
    yMin = Math.min(yMin, mpMin > 0 ? mpMin : yMin);
  }}

  // Add padding to y range
  if (opts.logY) {{
    yMin = Math.pow(10, Math.floor(Math.log10(yMin)));
    yMax = Math.pow(10, Math.ceil(Math.log10(yMax)));
  }} else {{
    const pad = (yMax - yMin) * 0.05;
    yMin = Math.max(0, yMin - pad); yMax = yMax + pad;
  }}

  // dataToX: maps a 1-based index value to canvas x position
  function dataToX(idx1) {{
    const x = opts.logX ? Math.log10(idx1) : idx1;
    const xMinT = opts.logX ? 0 : 1;  // log10(1)=0; linear starts at 1
    const xMaxT = opts.logX ? Math.log10(allX) : allX;
    return PAD.left + (x - xMinT) / (xMaxT - xMinT) * pw;
  }}
  // toX: converts 0-based array index to canvas x (used when drawing series)
  function toX(i) {{ return dataToX(i + 1); }}
  function toY(v) {{
    if (opts.logY) {{
      const ly = Math.log10(Math.max(v, 1e-30));
      const lyMin = Math.log10(yMin), lyMax = Math.log10(yMax);
      return PAD.top + (1 - (ly - lyMin) / (lyMax - lyMin)) * ph;
    }}
    return PAD.top + (1 - (v - yMin) / (yMax - yMin)) * ph;
  }}

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
  drawGrid(ctx, opts, PAD, pw, ph, xMin, xMax, yMin, yMax, allX, dataToX, toY);

  // Axes
  ctx.strokeStyle = '#999'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top); ctx.lineTo(PAD.left, PAD.top + ph);
  ctx.lineTo(PAD.left + pw, PAD.top + ph);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#444'; ctx.font = '13px Georgia, serif'; ctx.textAlign = 'center';
  ctx.fillText('index i', PAD.left + pw/2, H - 8);
  ctx.save(); ctx.translate(14, PAD.top + ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(opts.normY ? 'σᵢ / σ₀' : 'σᵢ', 0, 0);
  ctx.restore();

  // M-P horizontal lines
  mpLines.forEach(line => {{
    const y = toY(line.y);
    if (y < PAD.top || y > PAD.top + ph) return;
    ctx.strokeStyle = line.color; ctx.lineWidth = 1.5;
    ctx.setLineDash(line.dash);
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = line.color; ctx.font = '11px Georgia,serif'; ctx.textAlign = 'left';
    ctx.fillText(line.label, PAD.left + 4, y - 3);
  }});

  // Data series
  series.forEach(s => {{
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.dash ? 1.5 : 1.8;
    ctx.globalAlpha = s.dash ? 0.85 : 0.75;
    if (s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
    ctx.beginPath();
    let started = false;
    s.S.forEach((v, i) => {{
      if (v <= 0) return;
      const x = toX(i), y = toY(v);
      if (!started) {{ ctx.moveTo(x, y); started = true; }}
      else ctx.lineTo(x, y);
    }});
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }});
}}

function drawGrid(ctx, opts, PAD, pw, ph, xMin, xMax, yMin, yMax, allX, dataToX, toY) {{
  ctx.save();
  ctx.strokeStyle = '#eee'; ctx.lineWidth = 1; ctx.fillStyle = '#888';
  ctx.font = '11px Georgia,serif';

  // Y ticks
  const yTicks = opts.logY ? logTicks(yMin, yMax) : linTicks(yMin, yMax, 6);
  yTicks.forEach(v => {{
    const y = toY(v);
    if (y < PAD.top || y > PAD.top + ph + 1) return;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + pw, y); ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(fmtNum(v), PAD.left - 6, y + 4);
  }});

  // X ticks — tick values are 1-based indices, use dataToX
  const xTicks = opts.logX ? logTicks(1, allX) : linTicks(1, allX, 8);
  xTicks.forEach(v => {{
    const x = dataToX(v);
    if (x < PAD.left || x > PAD.left + pw + 1) return;
    ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + ph); ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(Math.round(v), x, PAD.top + ph + 16);
  }});
  ctx.restore();
}}

function linTicks(lo, hi, n) {{
  const step = niceStep((hi - lo) / n);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-10; v += step) ticks.push(v);
  return ticks;
}}
function logTicks(lo, hi) {{
  const ticks = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {{
    [1, 2, 5].forEach(m => {{ const v = m * Math.pow(10, e); if (v >= lo && v <= hi) ticks.push(v); }});
  }}
  return ticks;
}}
function niceStep(rough) {{
  const e = Math.pow(10, Math.floor(Math.log10(rough)));
  const f = rough / e;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * e;
}}
function fmtNum(v) {{
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(1);
  if (Number.isInteger(v) || Math.abs(v) >= 10) return String(Math.round(v));
  return v.toPrecision(2);
}}

window.addEventListener('resize', redraw);
redraw();

// ── Histogram drawing ─────────────────────────────────────────────────────────
function getHistOpts() {{
  return {{
    logY:    document.getElementById('hist-logy').checked,
    logX:    document.getElementById('hist-logx').checked,
    density: document.getElementById('hist-density').checked,
    random:  document.getElementById('hist-random').checked,
    nBins:   Math.max(10, Math.min(300, parseInt(document.getElementById('hist-bins').value) || 60)),
  }};
}}

function makeHistogram(S, edges) {{
  const counts = new Array(edges.length - 1).fill(0);
  S.forEach(v => {{
    let lo = 0, hi = edges.length - 1;
    while (lo < hi - 1) {{ const mid = (lo+hi)>>1; if (edges[mid] <= v) lo = mid; else hi = mid; }}
    if (lo < counts.length) counts[lo]++;
  }});
  return counts;
}}

function drawHist() {{
  const dpr = window.devicePixelRatio || 1;
  const W = histCanvas.offsetWidth;
  const H = histCanvas.offsetHeight || 544;
  histCanvas.width  = W * dpr;
  histCanvas.height = H * dpr;
  const ctx = histCanvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const opts = getHistOpts();
  const specOpts = getOpts();  // read normalization from the spectrum plot
  const PAD = {{ top: 30, right: 30, bottom: 50, left: 70 }};
  const pw = W - PAD.left - PAD.right;
  const ph = H - PAD.top  - PAD.bottom;

  // collect active raw S arrays (apply same normalization as spectrum plot)
  function normalize(S) {{
    if (specOpts.normY)    return S.map(v => v / S[0]);
    if (specOpts.normMean) return S.map(v => v / (S.reduce((a,b)=>a+b,0)/S.length));
    return S;
  }}

  const activeLayers = layerKeys.filter(i => checkboxes[i].checked);
  const allSeries = activeLayers.map(i => ({{
    S: normalize(RAW.layers[String(i)]), color: layerColor(i, RAW.n_layers)
  }}));
  if (opts.random) allSeries.push({{ S: normalize(RAW.random), color: '#999', dash: [4,3] }});

  if (allSeries.length === 0) {{ ctx.clearRect(0, 0, W, H); return; }}

  // global value range across all active series
  const allVals = allSeries.flatMap(s => s.S);
  const vMin = Math.min(...allVals), vMax = Math.max(...allVals);

  // build bin edges (linear or log in σ space)
  const n = opts.nBins;
  const edges = [];
  if (opts.logX) {{
    const lMin = Math.log10(Math.max(vMin, 1e-10)), lMax = Math.log10(vMax);
    for (let k = 0; k <= n; k++) edges.push(Math.pow(10, lMin + (lMax - lMin) * k / n));
  }} else {{
    for (let k = 0; k <= n; k++) edges.push(vMin + (vMax - vMin) * k / n);
  }}
  const binWidths = edges.slice(0,-1).map((e,i) => edges[i+1] - e);
  const binCenters = edges.slice(0,-1).map((e,i) => (e + edges[i+1]) / 2);

  // compute histograms
  const histSeries = allSeries.map(s => {{
    const counts = makeHistogram(s.S, edges);
    const total = s.S.length;
    const vals = opts.density
      ? counts.map((c,i) => c / (total * binWidths[i]))
      : counts.map(c => c);
    return {{ vals, color: s.color, dash: s.dash }};
  }});

  // axis ranges
  const allY = histSeries.flatMap(s => s.vals.filter(v => v > 0));
  if (allY.length === 0) {{ ctx.clearRect(0, 0, W, H); return; }}
  let yMin = 0, yMax = Math.max(...allY);
  let xMin = edges[0], xMax = edges[edges.length-1];

  if (opts.logY) {{
    yMin = Math.pow(10, Math.floor(Math.log10(Math.min(...allY.filter(v=>v>0)))));
    yMax = Math.pow(10, Math.ceil(Math.log10(yMax)));
  }} else {{
    yMax *= 1.05;
  }}

  function toX(v) {{
    if (opts.logX) {{
      const lv = Math.log10(Math.max(v, 1e-30));
      return PAD.left + (lv - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * pw;
    }}
    return PAD.left + (v - xMin) / (xMax - xMin) * pw;
  }}
  function toY(v) {{
    if (opts.logY) {{
      const ly = Math.log10(Math.max(v, 1e-30));
      return PAD.top + (1 - (ly - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * ph;
    }}
    return PAD.top + (1 - (v - yMin) / (yMax - yMin)) * ph;
  }}

  ctx.clearRect(0, 0, W, H);

  // grid
  ctx.save();
  ctx.strokeStyle = '#eee'; ctx.lineWidth = 1; ctx.fillStyle = '#888';
  ctx.font = '11px Georgia,serif';
  const yTicks = opts.logY ? logTicks(yMin, yMax) : linTicks(yMin, yMax, 6);
  yTicks.forEach(v => {{
    const y = toY(v); if (y < PAD.top || y > PAD.top+ph+1) return;
    ctx.beginPath(); ctx.moveTo(PAD.left,y); ctx.lineTo(PAD.left+pw,y); ctx.stroke();
    ctx.textAlign='right'; ctx.fillText(fmtNum(v), PAD.left-6, y+4);
  }});
  const xTicks = opts.logX ? logTicks(xMin, xMax) : linTicks(xMin, xMax, 7);
  xTicks.forEach(v => {{
    const x = toX(v); if (x < PAD.left || x > PAD.left+pw+1) return;
    ctx.beginPath(); ctx.moveTo(x,PAD.top); ctx.lineTo(x,PAD.top+ph); ctx.stroke();
    ctx.textAlign='center'; ctx.fillText(fmtNum(v), x, PAD.top+ph+16);
  }});
  ctx.restore();

  // axes
  ctx.strokeStyle = '#999'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(PAD.left, PAD.top); ctx.lineTo(PAD.left, PAD.top+ph);
  ctx.lineTo(PAD.left+pw, PAD.top+ph); ctx.stroke();

  // axis labels
  ctx.fillStyle = '#444'; ctx.font = '13px Georgia,serif'; ctx.textAlign = 'center';
  ctx.fillText('σ', PAD.left + pw/2, H - 8);
  ctx.save(); ctx.translate(14, PAD.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(opts.density ? 'density' : 'count', 0, 0);
  ctx.restore();

  // draw step-line histograms back-to-front (random last so it's on top)
  [...histSeries].reverse().forEach(s => {{
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.dash ? 1.5 : 1.8;
    ctx.globalAlpha = s.dash ? 0.7 : 0.65;
    if (s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
    ctx.beginPath();
    let started = false;
    s.vals.forEach((v, k) => {{
      if (opts.logY && v <= 0) {{ started = false; return; }}
      const x0 = toX(edges[k]), x1 = toX(edges[k+1]);
      const y  = toY(opts.logY ? Math.max(v, yMin) : v);
      const yb = toY(opts.logY ? yMin : 0);
      if (!started) {{ ctx.moveTo(x0, yb); started = true; }}
      ctx.lineTo(x0, y); ctx.lineTo(x1, y);
    }});
    // close to baseline
    if (started) {{
      const lastX = toX(edges[n]);
      ctx.lineTo(lastX, toY(opts.logY ? yMin : 0));
    }}
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }});
}}
</script>
</body>
</html>"""

    path = out_dir / "index.html"
    path.write_text(html)
    print(f"  Wrote {path}")
    _register_experiment(slug, title, description, path)
    return path


def write_overlap_page(
    slug: str,
    title: str,
    result: dict,   # output of experiments.fanout_fanin_overlap()
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

    # align_k curve: cumulative mean of top-left k×k block
    # align_k = (1/k) * sum_{i,j<=k} M[i,j]  (0-indexed, so block is M[:k, :k])
    max_k = min(n_rank, 512)
    align_k = []
    for k in range(1, max_k + 1):
        block_sum = float(M[:k, :k].sum())
        align_k.append(block_sum / k)

    # random baseline: each M[i,j] ~ 1/n_intermediate, so align_k_rand = k/n_intermediate
    n_intermediate = M.shape[0]  # ambient neuron-space dim used (= top_k)
    # but actual ambient dim is 8192; we need to use that for the true random baseline
    # We can infer it: row sums of M tell us what fraction of variance is captured
    # Simpler: just note align_k_rand = k / n_ambient, annotate on plot

    heatmap_n = 20
    M_corner  = M[:heatmap_n, :heatmap_n].tolist()

    data_json = json.dumps({
        "S_out":    S_out,
        "S_in":     S_in,
        "align_k":  align_k,
        "M_corner": M_corner,
        "heatmap_n": heatmap_n,
        "n_rank":   n_rank,
        "layer":    layer,
        "model":    model,
    })

    out_dir = EXPTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{CSS}
.plot-wrap {{
  position: relative;
  width: 100%; max-width: 975px;
  margin: 16px auto 0;
}}
.plot-wrap canvas {{
  display: block; width: 100%; height: 420px;
  border: 1px solid #ddd; border-radius: 4px; background: #fff;
}}
.plot-wrap.heatmap canvas {{
  height: 480px;
}}
.plot-gear {{
  position: absolute; top: 6px; right: 6px;
  width: 22px; height: 22px; padding: 0;
  background: rgba(255,255,255,0.85);
  border: 1px solid #ccc; border-radius: 4px;
  cursor: pointer; font-size: 13px; line-height: 22px; text-align: center;
  color: #555; opacity: 0; transition: opacity 0.15s; z-index: 10;
  user-select: none;
}}
.plot-wrap:hover .plot-gear {{ opacity: 1; }}
.plot-gear:hover {{ background: #fff; border-color: #999; color: #222; }}
.gear-menu {{
  position: absolute; top: 30px; right: 6px; z-index: 100;
  background: #fff; border: 1px solid #ccc; border-radius: 5px;
  padding: 6px 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  min-width: 160px; display: none;
}}
.gear-menu.open {{ display: block; }}
.gear-menu label {{
  display: flex; align-items: center; gap: 6px;
  padding: 3px 2px; cursor: pointer; font-size: 0.88em; white-space: nowrap;
}}
.gear-menu label:hover {{ background: #f4f4f4; border-radius: 3px; }}
.plot-label {{ font-size:0.85em; color:#888; margin: 24px 0 4px; max-width:975px; margin-left:auto; margin-right:auto; }}
</style>
</head>
<body>
<a class="back" href="../../index.html">← Back to experiments</a>
<h1>{title}</h1>
<p style="color:#555;font-size:0.95em">
  Layer {layer} of {model}.
  M<sub>ik</sub> = ⟨u<sub>i</sub><sup>out</sup>, v<sub>k</sub><sup>in</sup>⟩²
  where u<sup>out</sup> are LSVs of W<sub>out</sub> and v<sup>in</sup> are RSVs of W<sub>in</sub>, both in neuron space.
</p>

<div class="plot-label">Singular spectra</div>
<div class="plot-wrap" id="spectra-wrap">
  <canvas id="spectra-canvas"></canvas>
  <div class="plot-gear" id="spec-gear-btn">⚙</div>
  <div class="gear-menu" id="spec-gear-menu">
    <label><input type="checkbox" id="spec-logy" checked> log y</label>
    <label><input type="checkbox" id="spec-logx"> log x</label>
    <label><input type="checkbox" id="spec-norm"> normalize (σ/σ₀)</label>
  </div>
</div>

<div class="plot-label">Overlap matrix M (top {heatmap_n}×{heatmap_n})</div>
<div class="plot-wrap heatmap" id="heatmap-wrap">
  <canvas id="heatmap-canvas"></canvas>
</div>

<div class="plot-label">align<sub>k</sub> = k⁻¹ Σ<sub>i,j≤k</sub> M<sub>ij</sub></div>
<div class="plot-wrap" id="align-wrap">
  <canvas id="align-canvas"></canvas>
  <div class="plot-gear" id="align-gear-btn">⚙</div>
  <div class="gear-menu" id="align-gear-menu">
    <label><input type="checkbox" id="align-logx"> log x</label>
    <label><input type="checkbox" id="align-logy"> log y</label>
  </div>
</div>

<script>
const RAW = {data_json};

// ── Persistence ───────────────────────────────────────────────────────────────
const STORAGE_KEY = 'overlap-state-{slug}';
const OPT_IDS = ['spec-logy','spec-logx','spec-norm','align-logx','align-logy'];
const OPT_DEFAULTS = {{'spec-logy': true}};

function saveState() {{
  const s = {{}};
  OPT_IDS.forEach(id => s[id] = document.getElementById(id).checked);
  try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); }} catch(e) {{}}
}}
function loadState() {{
  let saved = null;
  try {{ saved = JSON.parse(localStorage.getItem(STORAGE_KEY)); }} catch(e) {{}}
  OPT_IDS.forEach(id => {{
    document.getElementById(id).checked = saved
      ? (saved[id] ?? (OPT_DEFAULTS[id] ?? false))
      : (OPT_DEFAULTS[id] ?? false);
  }});
}}
loadState();
OPT_IDS.forEach(id => document.getElementById(id).addEventListener('change', () => {{ saveState(); redraw(); }}));

// ── Gear menus ────────────────────────────────────────────────────────────────
function setupGear(btnId, menuId) {{
  const btn = document.getElementById(btnId);
  const menu = document.getElementById(menuId);
  btn.addEventListener('click', e => {{ menu.classList.toggle('open'); e.stopPropagation(); }});
  menu.addEventListener('click', e => e.stopPropagation());
}}
setupGear('spec-gear-btn',  'spec-gear-menu');
setupGear('align-gear-btn', 'align-gear-menu');
document.addEventListener('click', () => {{
  document.querySelectorAll('.gear-menu').forEach(m => m.classList.remove('open'));
}});

// ── Shared helpers ────────────────────────────────────────────────────────────
function linTicks(lo, hi, n) {{
  const step = niceStep((hi-lo)/n);
  const ticks = [];
  for (let v = Math.ceil(lo/step)*step; v <= hi+1e-10; v += step) ticks.push(v);
  return ticks;
}}
function logTicks(lo, hi) {{
  const ticks = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    [1,2,5].forEach(m => {{ const v=m*Math.pow(10,e); if(v>=lo&&v<=hi) ticks.push(v); }});
  return ticks;
}}
function niceStep(r) {{
  const e=Math.pow(10,Math.floor(Math.log10(r))), f=r/e;
  return (f<1.5?1:f<3?2:f<7?5:10)*e;
}}
function fmtNum(v) {{
  if(Math.abs(v)>=1000||(Math.abs(v)<0.01&&v!==0)) return v.toExponential(1);
  if(Number.isInteger(v)||Math.abs(v)>=10) return String(Math.round(v));
  return v.toPrecision(2);
}}
function drawAxes(ctx, PAD, pw, ph) {{
  ctx.strokeStyle='#999'; ctx.lineWidth=1.5;
  ctx.beginPath();
  ctx.moveTo(PAD.left,PAD.top); ctx.lineTo(PAD.left,PAD.top+ph);
  ctx.lineTo(PAD.left+pw,PAD.top+ph); ctx.stroke();
}}
function drawGrid(ctx, PAD, pw, ph, xTicks, yTicks, toX, toY) {{
  ctx.save(); ctx.strokeStyle='#eee'; ctx.lineWidth=1; ctx.fillStyle='#888';
  ctx.font='11px Georgia,serif';
  yTicks.forEach(v => {{
    const y=toY(v); if(y<PAD.top||y>PAD.top+ph+1) return;
    ctx.beginPath(); ctx.moveTo(PAD.left,y); ctx.lineTo(PAD.left+pw,y); ctx.stroke();
    ctx.textAlign='right'; ctx.fillText(fmtNum(v),PAD.left-6,y+4);
  }});
  xTicks.forEach(v => {{
    const x=toX(v); if(x<PAD.left||x>PAD.left+pw+1) return;
    ctx.beginPath(); ctx.moveTo(x,PAD.top); ctx.lineTo(x,PAD.top+ph); ctx.stroke();
    ctx.textAlign='center'; ctx.fillText(fmtNum(v),x,PAD.top+ph+16);
  }});
  ctx.restore();
}}

// ── 1. Spectra plot ───────────────────────────────────────────────────────────
const specCanvas = document.getElementById('spectra-canvas');

function drawSpectra() {{
  const dpr=window.devicePixelRatio||1;
  const W=specCanvas.offsetWidth, H=specCanvas.offsetHeight||420;
  specCanvas.width=W*dpr; specCanvas.height=H*dpr;
  const ctx=specCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const logY=document.getElementById('spec-logy').checked;
  const logX=document.getElementById('spec-logx').checked;
  const norm=document.getElementById('spec-norm').checked;
  const PAD={{top:30,right:30,bottom:50,left:70}};
  const pw=W-PAD.left-PAD.right, ph=H-PAD.top-PAD.bottom;

  const series = [
    {{S: RAW.S_out, color:'#2266cc', label:'W_out'}},
    {{S: RAW.S_in,  color:'#cc4422', label:'W_in', dash:[5,3]}},
  ];
  const normalized = series.map(s => ({{
    ...s,
    S: norm ? s.S.map(v=>v/s.S[0]) : s.S,
  }}));

  const allS = normalized.flatMap(s=>s.S.filter(v=>v>0));
  const allN = Math.max(...normalized.map(s=>s.S.length));
  let yMin=Math.min(...allS), yMax=Math.max(...allS);
  if(logY) {{
    yMin=Math.pow(10,Math.floor(Math.log10(yMin)));
    yMax=Math.pow(10,Math.ceil(Math.log10(yMax)));
  }} else {{ const p=(yMax-yMin)*0.05; yMin=Math.max(0,yMin-p); yMax+=p; }}

  function toX(i) {{ return logX
    ? PAD.left + Math.log10(i+1)/Math.log10(allN)*pw
    : PAD.left + i/allN*pw; }}
  function toY(v) {{ return logY
    ? PAD.top+(1-(Math.log10(Math.max(v,1e-30))-Math.log10(yMin))/(Math.log10(yMax)-Math.log10(yMin)))*ph
    : PAD.top+(1-(v-yMin)/(yMax-yMin))*ph; }}

  ctx.clearRect(0,0,W,H);
  const xTicks = logX ? logTicks(1,allN) : linTicks(0,allN,8);
  const yTicks = logY ? logTicks(yMin,yMax) : linTicks(yMin,yMax,6);
  const toXgrid = logX ? (v=>PAD.left+Math.log10(v)/Math.log10(allN)*pw) : (v=>PAD.left+v/allN*pw);
  drawGrid(ctx,PAD,pw,ph,xTicks,yTicks,toXgrid,toY);
  drawAxes(ctx,PAD,pw,ph);

  ctx.fillStyle='#444'; ctx.font='13px Georgia,serif'; ctx.textAlign='center';
  ctx.fillText('index i', PAD.left+pw/2, H-8);
  ctx.save(); ctx.translate(14,PAD.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(norm?'σᵢ/σ₀':'σᵢ',0,0); ctx.restore();

  // legend
  normalized.forEach((s,si) => {{
    ctx.strokeStyle=s.color; ctx.lineWidth=2;
    if(s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
    ctx.beginPath();
    s.S.forEach((v,i) => {{
      if(v<=0) return;
      const x=toX(i), y=toY(v);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }});
    ctx.stroke(); ctx.setLineDash([]);
    // legend label
    const lx=PAD.left+pw-80, ly=PAD.top+18+si*20;
    ctx.strokeStyle=s.color; ctx.lineWidth=2;
    if(s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(lx+22,ly); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=s.color; ctx.font='12px Georgia,serif'; ctx.textAlign='left';
    ctx.fillText(s.label,lx+26,ly+4);
  }});
}}

// ── 2. Heatmap ────────────────────────────────────────────────────────────────
const hmCanvas = document.getElementById('heatmap-canvas');

function drawHeatmap() {{
  const dpr=window.devicePixelRatio||1;
  const W=hmCanvas.offsetWidth, H=hmCanvas.offsetHeight||480;
  hmCanvas.width=W*dpr; hmCanvas.height=H*dpr;
  const ctx=hmCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const PAD={{top:30,right:30,bottom:50,left:70}};
  const pw=W-PAD.left-PAD.right, ph=H-PAD.top-PAD.bottom;
  const n=RAW.heatmap_n;
  const M=RAW.M_corner;

  // color scale: white=0, deep blue=max
  const flat=M.flat();
  const vMax=Math.max(...flat);
  function valToColor(v) {{
    const t=Math.sqrt(v/vMax);  // sqrt for better contrast on small values
    const r=Math.round(255*(1-t*0.85));
    const g=Math.round(255*(1-t*0.72));
    const b=Math.round(255*(1-t*0.1));
    return `rgb(${{r}},${{g}},${{b}})`;
  }}

  ctx.clearRect(0,0,W,H);
  const cellW=pw/n, cellH=ph/n;
  for(let i=0;i<n;i++) for(let j=0;j<n;j++) {{
    ctx.fillStyle=valToColor(M[i][j]);
    ctx.fillRect(PAD.left+j*cellW, PAD.top+i*cellH, cellW, cellH);
  }}

  // cell borders (light)
  ctx.strokeStyle='rgba(0,0,0,0.08)'; ctx.lineWidth=0.5;
  for(let i=0;i<=n;i++) {{
    ctx.beginPath(); ctx.moveTo(PAD.left,PAD.top+i*cellH); ctx.lineTo(PAD.left+pw,PAD.top+i*cellH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD.left+i*cellW,PAD.top); ctx.lineTo(PAD.left+i*cellW,PAD.top+ph); ctx.stroke();
  }}

  // axis tick labels
  ctx.fillStyle='#666'; ctx.font='11px Georgia,serif';
  const step=Math.max(1,Math.floor(n/10));
  for(let i=0;i<n;i+=step) {{
    ctx.textAlign='right';
    ctx.fillText(i, PAD.left-5, PAD.top+i*cellH+cellH/2+4);
    ctx.textAlign='center';
    ctx.fillText(i, PAD.left+i*cellW+cellW/2, PAD.top+ph+16);
  }}

  // axis labels
  ctx.fillStyle='#444'; ctx.font='13px Georgia,serif'; ctx.textAlign='center';
  ctx.fillText('j  (W_in RSV index)', PAD.left+pw/2, H-8);
  ctx.save(); ctx.translate(14,PAD.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('i  (W_out LSV index)',0,0); ctx.restore();

  // outer border
  ctx.strokeStyle='#aaa'; ctx.lineWidth=1;
  ctx.strokeRect(PAD.left,PAD.top,pw,ph);

  // colorbar (simple gradient strip)
  const cbX=PAD.left+pw+8, cbY=PAD.top, cbW=14, cbH=ph;
  const grad=ctx.createLinearGradient(0,cbY,0,cbY+cbH);
  grad.addColorStop(0, valToColor(vMax));
  grad.addColorStop(1, valToColor(0));
  ctx.fillStyle=grad; ctx.fillRect(cbX,cbY,cbW,cbH);
  ctx.strokeStyle='#aaa'; ctx.lineWidth=1; ctx.strokeRect(cbX,cbY,cbW,cbH);
  ctx.fillStyle='#666'; ctx.font='10px Georgia,serif'; ctx.textAlign='left';
  ctx.fillText(fmtNum(vMax), cbX+cbW+3, cbY+10);
  ctx.fillText('0', cbX+cbW+3, cbY+cbH);
}}

// ── 3. align_k plot ───────────────────────────────────────────────────────────
const alignCanvas = document.getElementById('align-canvas');

function drawAlign() {{
  const dpr=window.devicePixelRatio||1;
  const W=alignCanvas.offsetWidth, H=alignCanvas.offsetHeight||420;
  alignCanvas.width=W*dpr; alignCanvas.height=H*dpr;
  const ctx=alignCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const logX=document.getElementById('align-logx').checked;
  const logY=document.getElementById('align-logy').checked;
  const PAD={{top:30,right:30,bottom:50,left:70}};
  const pw=W-PAD.left-PAD.right, ph=H-PAD.top-PAD.bottom;

  const ak=RAW.align_k;
  const maxK=ak.length;
  const n_rank=RAW.n_rank;
  // random baseline: align_k_rand = k / n_rank  (since M[i,j] ~ 1/n_rank on average)
  const akRand=ak.map((_,i)=>(i+1)/n_rank);

  const allY=[...ak,...akRand].filter(v=>v>0);
  let yMin=Math.min(...allY), yMax=Math.max(...allY);
  if(logY) {{
    yMin=Math.pow(10,Math.floor(Math.log10(yMin)));
    yMax=Math.pow(10,Math.ceil(Math.log10(yMax)));
  }} else {{ yMin=0; yMax=yMax*1.05; }}

  function toX(k1) {{ // 1-based k
    return logX
      ? PAD.left+Math.log10(k1)/Math.log10(maxK)*pw
      : PAD.left+(k1-1)/(maxK-1)*pw;
  }}
  function toY(v) {{ return logY
    ? PAD.top+(1-(Math.log10(Math.max(v,1e-30))-Math.log10(yMin))/(Math.log10(yMax)-Math.log10(yMin)))*ph
    : PAD.top+(1-(v-yMin)/(yMax-yMin))*ph; }}

  ctx.clearRect(0,0,W,H);
  const xTicks = logX ? logTicks(1,maxK) : linTicks(1,maxK,8);
  const yTicks = logY ? logTicks(yMin,yMax) : linTicks(yMin,yMax,6);
  drawGrid(ctx,PAD,pw,ph,xTicks,yTicks,toX,toY);
  drawAxes(ctx,PAD,pw,ph);

  ctx.fillStyle='#444'; ctx.font='13px Georgia,serif'; ctx.textAlign='center';
  ctx.fillText('k', PAD.left+pw/2, H-8);
  ctx.save(); ctx.translate(14,PAD.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('alignₖ',0,0); ctx.restore();

  // random baseline
  ctx.strokeStyle='#aaa'; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
  ctx.beginPath();
  akRand.forEach((v,i) => {{ const x=toX(i+1),y=toY(v); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }});
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle='#aaa'; ctx.font='11px Georgia,serif'; ctx.textAlign='left';
  ctx.fillText('random', toX(maxK*0.6), toY(akRand[Math.floor(maxK*0.6)])-5);

  // align_k
  ctx.strokeStyle='#2266cc'; ctx.lineWidth=2;
  ctx.beginPath();
  ak.forEach((v,i) => {{ const x=toX(i+1),y=toY(v); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }});
  ctx.stroke();
}}

function redraw() {{ drawSpectra(); drawHeatmap(); drawAlign(); }}
window.addEventListener('resize', redraw);
redraw();
</script>
</body>
</html>"""

    path = out_dir / "index.html"
    path.write_text(html)
    print(f"  Wrote {path}")
    _register_experiment(slug, title, "", path)
    return path
