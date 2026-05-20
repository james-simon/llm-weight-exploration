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
  M<sub>ik</sub> = ⟨u<sub>i</sub><sup>up</sup>, v<sub>k</sub><sup>down</sup>⟩²
  where u<sup>up</sup> are LSVs of W<sub>up</sub> and v<sup>down</sup> are RSVs of W<sub>down</sub>, both in neuron space.
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
    <label><input type="checkbox" id="align-ratio"> show ratio (alignₖ / random)</label>
    <label style="gap:6px;">max k&nbsp;<input type="number" id="align-maxk" value="{n_rank}" min="1" max="{n_rank}" style="width:56px;font-size:0.9em;padding:1px 4px;border:1px solid #ccc;border-radius:3px;"></label>
  </div>
</div>

<script>
const RAW = {data_json};

// ── Persistence ───────────────────────────────────────────────────────────────
const STORAGE_KEY = 'overlap-state-{slug}';
const OPT_IDS = ['spec-logy','spec-logx','spec-norm','align-logx','align-logy','align-ratio'];
const OPT_DEFAULTS = {{'spec-logy': true}};

function saveState() {{
  const s = {{}};
  OPT_IDS.forEach(id => s[id] = document.getElementById(id).checked);
  s['align-maxk'] = document.getElementById('align-maxk').value;
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
  if (saved?.['align-maxk']) document.getElementById('align-maxk').value = saved['align-maxk'];
}}
loadState();
OPT_IDS.forEach(id => document.getElementById(id).addEventListener('change', () => {{ saveState(); redraw(); }}));
document.getElementById('align-maxk').addEventListener('input', () => {{ saveState(); drawAlign(); }});

// ── Gear menus ────────────────────────────────────────────────────────────────
function setupGear(btnId, menuId) {{
  const btn = document.getElementById(btnId);
  const menu = document.getElementById(menuId);
  if (!btn || !menu) return;
  btn.addEventListener('click', e => {{ menu.classList.toggle('open'); e.stopPropagation(); }});
  menu.addEventListener('click', e => e.stopPropagation());
}}
setupGear('spec-gear-btn',   'spec-gear-menu');
setupGear('align-gear-btn',  'align-gear-menu');
setupGear('cossim-gear-btn', 'cossim-gear-menu');
setupGear('svhist-gear-btn', 'svhist-gear-menu');
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

  function normS(S) {{
    if (!norm) return S;
    const mean = S.reduce((a,b)=>a+b,0)/S.length;
    return S.map(v=>v/mean);
  }}

  const series = [
    {{S: RAW.S_out,      color:'#2266cc', label:'W_up'}},
    {{S: RAW.S_in,       color:'#cc4422', label:'W_down',  dash:[5,3]}},
    ...(RAW.S_rand_out.length ? [{{S: RAW.S_rand_out, color:'#888',   label:'random', dash:[4,2]}}] : []),
  ];
  const normalized = series.map(s => ({{...s, S: normS(s.S)}}));

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
  ctx.fillText(norm?'σᵢ / ⟨σ⟩':'σᵢ',0,0); ctx.restore();

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
  const PAD={{top:30,right:60,bottom:50,left:70}};
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
  ctx.fillText('j  (W_down RSV index)', PAD.left+pw/2, H-8);
  ctx.save(); ctx.translate(14,PAD.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('i  (W_up LSV index)',0,0); ctx.restore();

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
  ctx.fillText(fmtNum(vMax), cbX, cbY-2);
  ctx.fillText('0', cbX, cbY+cbH+10);
}}

// ── Heatmap tooltip ───────────────────────────────────────────────────────────
(function() {{
  const tooltip = document.createElement('div');
  tooltip.style.cssText = 'position:fixed;background:rgba(0,0,0,0.75);color:#fff;font:12px Georgia,serif;padding:4px 8px;border-radius:4px;pointer-events:none;display:none;z-index:999';
  document.body.appendChild(tooltip);

  hmCanvas.addEventListener('mousemove', e => {{
    const rect = hmCanvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (hmCanvas.width / rect.width / (window.devicePixelRatio||1));
    const my = (e.clientY - rect.top)  * (hmCanvas.height / rect.height / (window.devicePixelRatio||1));
    const PAD={{top:30,right:30,bottom:50,left:70}};
    const pw = hmCanvas.width/(window.devicePixelRatio||1) - PAD.left - PAD.right;
    const ph = hmCanvas.height/(window.devicePixelRatio||1) - PAD.top  - PAD.bottom;
    const n = RAW.heatmap_n;
    const ci = Math.floor((mx - PAD.left) / (pw/n));
    const ri = Math.floor((my - PAD.top)  / (ph/n));
    if (ri>=0 && ri<n && ci>=0 && ci<n) {{
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX+14)+'px';
      tooltip.style.top  = (e.clientY-20)+'px';
      tooltip.textContent = `i=${{ri}}, j=${{ci}}: ${{RAW.M_corner[ri][ci].toFixed(4)}}`;
    }} else {{
      tooltip.style.display = 'none';
    }}
  }});
  hmCanvas.addEventListener('mouseleave', () => {{ tooltip.style.display='none'; }});
}})();

// ── 3. align_k plot ───────────────────────────────────────────────────────────
const alignCanvas = document.getElementById('align-canvas');

function drawAlign() {{
  const dpr=window.devicePixelRatio||1;
  const W=alignCanvas.offsetWidth, H=alignCanvas.offsetHeight||420;
  alignCanvas.width=W*dpr; alignCanvas.height=H*dpr;
  const ctx=alignCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const logX  = document.getElementById('align-logx').checked;
  const logY  = document.getElementById('align-logy').checked;
  const ratio = document.getElementById('align-ratio').checked;
  const PAD={{top:30,right:30,bottom:50,left:80}};
  const pw=W-PAD.left-PAD.right, ph=H-PAD.top-PAD.bottom;

  const akFull=RAW.align_k;
  const maxKRaw = Math.max(1, Math.min(akFull.length, parseInt(document.getElementById('align-maxk').value) || akFull.length));
  const ak = akFull.slice(0, maxKRaw);
  const maxK = ak.length;
  const n_ambient = RAW.n_intermediate || RAW.n_rank;
  const akRand = ak.map((_,i) => (i+1)/n_ambient);

  // in ratio mode: plot ak[i]/akRand[i] = ak[i] * n_ambient / (i+1); random baseline = 1
  const plotAk   = ratio ? ak.map((v,i) => v / akRand[i]) : ak;
  const plotRand = ratio ? ak.map(() => 1)                 : akRand;

  const allY = [...plotAk, ...(ratio ? [] : plotRand)].filter(v=>v>0);
  let yMin=Math.min(...allY), yMax=Math.max(...allY);
  if(logY) {{
    yMin=Math.pow(10,Math.floor(Math.log10(yMin)));
    yMax=Math.pow(10,Math.ceil(Math.log10(yMax)));
  }} else {{
    yMin = ratio ? 0 : 0;
    yMax = yMax*1.05;
  }}

  function toX(k1) {{
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
  ctx.fillText(ratio ? 'alignₖ / random' : 'alignₖ', 0, 0);
  ctx.restore();

  if (ratio) {{
    // draw y=1 reference line
    const y1 = toY(1);
    ctx.strokeStyle='#aaa'; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.moveTo(PAD.left,y1); ctx.lineTo(PAD.left+pw,y1); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#aaa'; ctx.font='11px Georgia,serif'; ctx.textAlign='left';
    ctx.fillText('random (= 1)', PAD.left+4, y1-4);
  }} else {{
    // draw random curve
    ctx.strokeStyle='#aaa'; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
    ctx.beginPath();
    plotRand.forEach((v,i) => {{ const x=toX(i+1),y=toY(v); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }});
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle='#aaa'; ctx.font='11px Georgia,serif'; ctx.textAlign='left';
    const labelI = Math.floor(maxK*0.6);
    ctx.fillText('random', toX(labelI+1), toY(plotRand[labelI])-5);
  }}

  // main curve
  ctx.strokeStyle='#2266cc'; ctx.lineWidth=2;
  ctx.beginPath();
  plotAk.forEach((v,i) => {{ const x=toX(i+1),y=toY(v); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }});
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

    data_json = json.dumps({
        "layers":       layers_data,
        "layer_indices": layer_indices,
        "S_rand_out":   random_out if random_out is not None else [],
        "S_rand_in":    random_in  if random_in  is not None else [],
        "heatmap_n":    heatmap_n,
        "n_intermediate": ni_fallback,
    })

    out_dir = EXPTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    first_n_rank = layers_data[layer_indices[0]]["n_rank"]
    n_intermediate = ni_fallback or 0
    n_hidden = first_n_rank  # rank of fan_out = min(d_int, d_hid) = d_hid

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
.plot-wrap.heatmap canvas {{ height: 480px; }}
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

/* floating layer sidebar */
.layer-sidebar {{
  position: fixed; left: 12px; top: 50%;
  transform: translateY(-50%);
  display: flex; flex-direction: column; gap: 4px;
  z-index: 200;
}}
.layer-btn {{
  padding: 4px 8px; font-size: 0.8em; cursor: pointer;
  border: 1px solid #ccc; border-radius: 4px;
  background: #f5f5f5; color: #444;
  transition: background 0.1s, color 0.1s, border-color 0.1s;
  user-select: none; white-space: nowrap;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.layer-btn:hover {{ background: #e8e8e8; }}
.layer-btn.active {{
  background: #2266cc; color: #fff; border-color: #1a55bb;
}}
</style>
<script>
MathJax = {{ tex: {{ inlineMath: [['$','$']] }}, options: {{ skipHtmlTags: ['script','noscript','style','textarea'] }} }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>
</head>
<body>
<a class="back" href="../../index.html">← Back to experiments</a>
<h1>{title}</h1>

<div style="max-width:860px;font-size:0.95em;color:#444;line-height:1.7;margin-bottom:8px;">
<p>
The MLP applies $x \\mapsto W_\\mathrm{{down}}\\,\\phi(W_\\mathrm{{up}}\\,x)$, where
$W_\\mathrm{{up}} \\in \\mathbb{{R}}^{{d_\\mathrm{{int}} \\times d_\\mathrm{{hid}}}}$ (first, expands) and
$W_\\mathrm{{down}} \\in \\mathbb{{R}}^{{d_\\mathrm{{hid}} \\times d_\\mathrm{{int}}}}$ (second, contracts),
with $d_\\mathrm{{int}} = {n_intermediate}$ and $d_\\mathrm{{hid}} = {n_hidden}$.
Both matrices act in <em>neuron space</em> $\\mathbb{{R}}^{{d_\\mathrm{{int}}}}$, where the nonlinearity $\\phi$ lives.
</p>
<p>
SVDs: $W_\\mathrm{{up}} = U_\\mathrm{{up}} \\Sigma_\\mathrm{{up}} V_\\mathrm{{up}}^\\top$ and
$W_\\mathrm{{down}} = U_\\mathrm{{down}} \\Sigma_\\mathrm{{down}} V_\\mathrm{{down}}^\\top$.
The neuron-space bases are $u_i^\\mathrm{{up}}$ (columns of $U_\\mathrm{{up}}$, LSVs of $W_\\mathrm{{up}}$, shape $d_\\mathrm{{int}} \\times d_\\mathrm{{hid}}$)
and $v_k^\\mathrm{{down}}$ (columns of $V_\\mathrm{{down}}$, RSVs of $W_\\mathrm{{down}}$, shape $d_\\mathrm{{hid}} \\times d_\\mathrm{{int}}$).
</p>
<p>
<strong>Overlap matrix.</strong>
$M_{{ik}} = \\langle u_i^\\mathrm{{up}},\\, v_k^\\mathrm{{down}} \\rangle^2 \\in [0,1]$.
Random baseline: $\\mathbb{{E}}[M_{{ik}}] = 1/d_\\mathrm{{int}} = 1/{n_intermediate}$ for independent random unit vectors.
</p>
<p>
<strong>Alignment score.</strong>
$\\mathrm{{align}}_k = \\frac{{1}}{{k}}\\sum_{{i,j \\le k}} M_{{ij}}$
measures overlap of the top-$k$ singular subspaces of $W_\\mathrm{{up}}$ and $W_\\mathrm{{down}}$ in neuron space.
Random baseline: $k / {n_intermediate}$.
</p>
<p>
<strong>All-ones projection.</strong>
$c_i^\\mathrm{{up}} = \\langle u_i^\\mathrm{{up}},\\, \\hat{{1}} \\rangle$,
$c_k^\\mathrm{{down}} = \\langle v_k^\\mathrm{{down}},\\, \\hat{{1}} \\rangle$,
where $\\hat{{1}} = \\mathbf{{1}}/\\sqrt{{d_\\mathrm{{int}}}}$.
A large value means that singular mode uniformly activates all neurons — a "broadcast" direction.
</p>
</div>

<div class="layer-sidebar" id="layer-selector"></div>

<div class="plot-label">Singular spectra of $W_\\mathrm{{up}}$ and $W_\\mathrm{{down}}$</div>
<div class="plot-wrap" id="spectra-wrap">
  <canvas id="spectra-canvas"></canvas>
  <div class="plot-gear" id="spec-gear-btn">⚙</div>
  <div class="gear-menu" id="spec-gear-menu">
    <label><input type="checkbox" id="spec-logy" checked> log y</label>
    <label><input type="checkbox" id="spec-logx"> log x</label>
    <label><input type="checkbox" id="spec-norm"> normalize ($\\sigma / \\langle\\sigma\\rangle$)</label>
  </div>
</div>

<div class="plot-label">Alignment: $\\mathrm{{align}}_k = k^{{-1}}\\sum_{{i,j \\le k}} M_{{ij}}$</div>
<div class="plot-wrap" id="align-wrap">
  <canvas id="align-canvas"></canvas>
  <div class="plot-gear" id="align-gear-btn">⚙</div>
  <div class="gear-menu" id="align-gear-menu">
    <label><input type="checkbox" id="align-logx"> log x</label>
    <label><input type="checkbox" id="align-logy"> log y</label>
    <label><input type="checkbox" id="align-ratio"> show ratio ($\\mathrm{{align}}_k$ / random)</label>
    <label style="gap:6px;">max $k$&nbsp;<input type="number" id="align-maxk" value="{first_n_rank}" min="1" max="{first_n_rank}" style="width:56px;font-size:0.9em;padding:1px 4px;border:1px solid #ccc;border-radius:3px;"></label>
  </div>
</div>

<div class="plot-label">Overlap matrix $M_{{ik}} = \\langle u_i^\\mathrm{{up}},\\, v_k^\\mathrm{{down}} \\rangle^2$ (top {heatmap_n}×{heatmap_n})</div>
<div class="plot-wrap heatmap" id="heatmap-wrap">
  <canvas id="heatmap-canvas"></canvas>
</div>

<div class="plot-label">Projection onto all-ones: $c_i^\\mathrm{{up}} = \\langle u_i^\\mathrm{{up}},\\,\\hat{{1}}\\rangle$ and $c_k^\\mathrm{{down}} = \\langle v_k^\\mathrm{{down}},\\,\\hat{{1}}\\rangle$</div>
<div class="plot-wrap" id="cossim-wrap">
  <canvas id="cossim-canvas"></canvas>
  <div class="plot-gear" id="cossim-gear-btn">⚙</div>
  <div class="gear-menu" id="cossim-gear-menu">
    <label><input type="checkbox" id="cossim-sq"> show cos²</label>
    <label><input type="checkbox" id="cossim-logx"> log x</label>
  </div>
</div>

<div class="plot-label">Singular vector histogram</div>
<div style="max-width:975px;margin:6px auto 6px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:0.88em;color:#444;">
  <label style="display:flex;align-items:center;gap:5px;">
    Matrix
    <select id="svh-matrix" style="font-size:0.95em;padding:2px 4px;border:1px solid #ccc;border-radius:3px;">
      <option value="up">$W_\\mathrm{{up}}$</option>
      <option value="down">$W_\\mathrm{{down}}$</option>
    </select>
  </label>
  <label style="display:flex;align-items:center;gap:5px;">
    Vectors
    <select id="svh-side" style="font-size:0.95em;padding:2px 4px;border:1px solid #ccc;border-radius:3px;">
      <option value="L">Left singular vectors (LSVs)</option>
      <option value="R">Right singular vectors (RSVs)</option>
    </select>
  </label>
  <label style="display:flex;align-items:center;gap:5px;">
    Index
    <input type="number" id="svh-index" value="0" min="0" style="width:56px;font-size:0.95em;padding:2px 4px;border:1px solid #ccc;border-radius:3px;">
  </label>
  <span id="svh-dim-label" style="color:#888;font-style:italic;"></span>
</div>
<div class="plot-wrap" id="svhist-wrap">
  <canvas id="svhist-canvas"></canvas>
  <div class="plot-gear" id="svhist-gear-btn">⚙</div>
  <div class="gear-menu" id="svhist-gear-menu">
    <label><input type="checkbox" id="svhist-logy"> log y</label>
    <label style="gap:6px;">bins&nbsp;<input type="number" id="svhist-bins" value="80" min="10" max="400" style="width:48px;font-size:0.9em;padding:1px 4px;border:1px solid #ccc;border-radius:3px;"></label>
  </div>
</div>

<script>
const RAW = {data_json};

// ── Layer selector ────────────────────────────────────────────────────────────
let activeLayer = RAW.layer_indices[0];

const selectorEl = document.getElementById('layer-selector');
RAW.layer_indices.forEach(li => {{
  const btn = document.createElement('button');
  btn.className = 'layer-btn' + (li === activeLayer ? ' active' : '');
  btn.textContent = 'L' + (li + 1);
  btn.dataset.layer = li;
  btn.addEventListener('click', () => {{
    activeLayer = li;
    document.querySelectorAll('.layer-btn').forEach(b =>
      b.classList.toggle('active', +b.dataset.layer === li));
    saveState();
    redraw();
  }});
  selectorEl.appendChild(btn);
}});

function layerData() {{ return RAW.layers[activeLayer]; }}

// ── Persistence ───────────────────────────────────────────────────────────────
const STORAGE_KEY = 'overlap-all-{slug}';
const OPT_IDS = ['spec-logy','spec-logx','spec-norm','align-logx','align-logy','align-ratio','cossim-sq','cossim-logx','svhist-logy'];
const OPT_DEFAULTS = {{'spec-logy': true}};

function saveState() {{
  const s = {{}};
  OPT_IDS.forEach(id => s[id] = document.getElementById(id).checked);
  s['align-maxk']   = document.getElementById('align-maxk').value;
  s['svhist-bins']  = document.getElementById('svhist-bins').value;
  s['svhist-index'] = document.getElementById('svh-index').value;
  s['svhist-matrix']= document.getElementById('svh-matrix').value;
  s['svhist-side']  = document.getElementById('svh-side').value;
  s['active-layer'] = activeLayer;
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
  if (saved?.['align-maxk'])   document.getElementById('align-maxk').value  = saved['align-maxk'];
  if (saved?.['svhist-bins'])  document.getElementById('svhist-bins').value = saved['svhist-bins'];
  if (saved?.['svhist-index']) document.getElementById('svh-index').value   = saved['svhist-index'];
  if (saved?.['svhist-matrix']) document.getElementById('svh-matrix').value = saved['svhist-matrix'];
  if (saved?.['svhist-side'])  document.getElementById('svh-side').value    = saved['svhist-side'];
  if (saved?.['active-layer'] != null && RAW.layers[saved['active-layer']]) {{
    activeLayer = saved['active-layer'];
    document.querySelectorAll('.layer-btn').forEach(b =>
      b.classList.toggle('active', +b.dataset.layer === activeLayer));
  }}
}}
loadState();
OPT_IDS.forEach(id => document.getElementById(id).addEventListener('change', () => {{ saveState(); redraw(); }}));
document.getElementById('align-maxk').addEventListener('input', () => {{ saveState(); drawAlign(); }});
document.getElementById('svhist-bins').addEventListener('input', () => {{ saveState(); drawSvHist(); }});
document.getElementById('svh-index').addEventListener('input', () => {{ saveState(); fetchAndDrawSvHist(); }});
document.getElementById('svh-matrix').addEventListener('change', () => {{ saveState(); fetchAndDrawSvHist(); }});
document.getElementById('svh-side').addEventListener('change', () => {{ saveState(); fetchAndDrawSvHist(); }});

// ── Gear menus ────────────────────────────────────────────────────────────────
function setupGear(btnId, menuId) {{
  const btn = document.getElementById(btnId);
  const menu = document.getElementById(menuId);
  if (!btn || !menu) return;
  btn.addEventListener('click', e => {{ menu.classList.toggle('open'); e.stopPropagation(); }});
  menu.addEventListener('click', e => e.stopPropagation());
}}
setupGear('spec-gear-btn',   'spec-gear-menu');
setupGear('align-gear-btn',  'align-gear-menu');
setupGear('cossim-gear-btn', 'cossim-gear-menu');
setupGear('svhist-gear-btn', 'svhist-gear-menu');
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

// ── 1. Spectra ────────────────────────────────────────────────────────────────
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
  const ld = layerData();

  function normS(S) {{
    if (!norm) return S;
    const mean = S.reduce((a,b)=>a+b,0)/S.length;
    return S.map(v=>v/mean);
  }}

  const series = [
    {{S: ld.S_out,      color:'#2266cc', label:'W_up'}},
    {{S: ld.S_in,       color:'#cc4422', label:'W_down', dash:[5,3]}},
    ...(RAW.S_rand_out.length ? [{{S: RAW.S_rand_out, color:'#888', label:'random', dash:[4,2]}}] : []),
  ];
  const normalized = series.map(s => ({{...s, S: normS(s.S)}}));

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
  const toXgrid = logX ? (v=>PAD.left+Math.log10(v)/Math.log10(allN)*pw) : (v=>PAD.left+v/allN*pw);
  drawGrid(ctx,PAD,pw,ph,
    logX?logTicks(1,allN):linTicks(0,allN,8),
    logY?logTicks(yMin,yMax):linTicks(yMin,yMax,6),
    toXgrid,toY);
  drawAxes(ctx,PAD,pw,ph);

  ctx.fillStyle='#444'; ctx.font='13px Georgia,serif'; ctx.textAlign='center';
  ctx.fillText('index i', PAD.left+pw/2, H-8);
  ctx.save(); ctx.translate(14,PAD.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(norm?'σᵢ / ⟨σ⟩':'σᵢ',0,0); ctx.restore();

  normalized.forEach((s,si) => {{
    ctx.strokeStyle=s.color; ctx.lineWidth=2;
    if(s.dash) ctx.setLineDash(s.dash); else ctx.setLineDash([]);
    ctx.beginPath();
    s.S.forEach((v,i) => {{ if(v<=0) return; const x=toX(i),y=toY(v); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }});
    ctx.stroke(); ctx.setLineDash([]);
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
  const PAD={{top:30,right:60,bottom:50,left:70}};
  const pw=W-PAD.left-PAD.right, ph=H-PAD.top-PAD.bottom;
  const n=RAW.heatmap_n;
  const M=layerData().M_corner;

  const flat=M.flat(), vMax=Math.max(...flat);
  function valToColor(v) {{
    const t=Math.sqrt(v/vMax);
    return `rgb(${{Math.round(255*(1-t*0.85))}},${{Math.round(255*(1-t*0.72))}},${{Math.round(255*(1-t*0.1))}})`;
  }}

  ctx.clearRect(0,0,W,H);
  const cellW=pw/n, cellH=ph/n;
  for(let i=0;i<n;i++) for(let j=0;j<n;j++) {{
    ctx.fillStyle=valToColor(M[i][j]);
    ctx.fillRect(PAD.left+j*cellW, PAD.top+i*cellH, cellW, cellH);
  }}
  ctx.strokeStyle='rgba(0,0,0,0.08)'; ctx.lineWidth=0.5;
  for(let i=0;i<=n;i++) {{
    ctx.beginPath(); ctx.moveTo(PAD.left,PAD.top+i*cellH); ctx.lineTo(PAD.left+pw,PAD.top+i*cellH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD.left+i*cellW,PAD.top); ctx.lineTo(PAD.left+i*cellW,PAD.top+ph); ctx.stroke();
  }}
  const step=Math.max(1,Math.floor(n/10));
  ctx.fillStyle='#666'; ctx.font='11px Georgia,serif';
  for(let i=0;i<n;i+=step) {{
    ctx.textAlign='right'; ctx.fillText(i,PAD.left-5,PAD.top+i*cellH+cellH/2+4);
    ctx.textAlign='center'; ctx.fillText(i,PAD.left+i*cellW+cellW/2,PAD.top+ph+16);
  }}
  ctx.fillStyle='#444'; ctx.font='13px Georgia,serif'; ctx.textAlign='center';
  ctx.fillText('k  (W_down RSV index)', PAD.left+pw/2, H-8);
  ctx.save(); ctx.translate(14,PAD.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('i  (W_up LSV index)',0,0); ctx.restore();
  ctx.strokeStyle='#aaa'; ctx.lineWidth=1; ctx.strokeRect(PAD.left,PAD.top,pw,ph);

  const cbX=PAD.left+pw+8, cbY=PAD.top, cbW=14, cbH=ph;
  const grad=ctx.createLinearGradient(0,cbY,0,cbY+cbH);
  grad.addColorStop(0,valToColor(vMax)); grad.addColorStop(1,valToColor(0));
  ctx.fillStyle=grad; ctx.fillRect(cbX,cbY,cbW,cbH);
  ctx.strokeStyle='#aaa'; ctx.lineWidth=1; ctx.strokeRect(cbX,cbY,cbW,cbH);
  ctx.fillStyle='#666'; ctx.font='10px Georgia,serif'; ctx.textAlign='left';
  ctx.fillText(fmtNum(vMax),cbX,cbY-2);
  ctx.fillText('0',cbX,cbY+cbH+10);
}}

// ── Heatmap tooltip ───────────────────────────────────────────────────────────
(function() {{
  const tooltip = document.createElement('div');
  tooltip.style.cssText = 'position:fixed;background:rgba(0,0,0,0.75);color:#fff;font:12px Georgia,serif;padding:4px 8px;border-radius:4px;pointer-events:none;display:none;z-index:999';
  document.body.appendChild(tooltip);
  hmCanvas.addEventListener('mousemove', e => {{
    const rect=hmCanvas.getBoundingClientRect();
    const mx=(e.clientX-rect.left)*(hmCanvas.width/rect.width/(window.devicePixelRatio||1));
    const my=(e.clientY-rect.top)*(hmCanvas.height/rect.height/(window.devicePixelRatio||1));
    const PAD={{top:30,right:60,bottom:50,left:70}};
    const pw=hmCanvas.width/(window.devicePixelRatio||1)-PAD.left-PAD.right;
    const ph=hmCanvas.height/(window.devicePixelRatio||1)-PAD.top-PAD.bottom;
    const n=RAW.heatmap_n;
    const ci=Math.floor((mx-PAD.left)/(pw/n));
    const ri=Math.floor((my-PAD.top)/(ph/n));
    if(ri>=0&&ri<n&&ci>=0&&ci<n) {{
      tooltip.style.display='block';
      tooltip.style.left=(e.clientX+14)+'px';
      tooltip.style.top=(e.clientY-20)+'px';
      tooltip.textContent=`i=${{ri}}, j=${{ci}}: ${{layerData().M_corner[ri][ci].toFixed(4)}}`;
    }} else {{ tooltip.style.display='none'; }}
  }});
  hmCanvas.addEventListener('mouseleave', ()=>{{ tooltip.style.display='none'; }});
}})();

// ── 3. align_k ────────────────────────────────────────────────────────────────
const alignCanvas = document.getElementById('align-canvas');

function drawAlign() {{
  const dpr=window.devicePixelRatio||1;
  const W=alignCanvas.offsetWidth, H=alignCanvas.offsetHeight||420;
  alignCanvas.width=W*dpr; alignCanvas.height=H*dpr;
  const ctx=alignCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const logX  =document.getElementById('align-logx').checked;
  const logY  =document.getElementById('align-logy').checked;
  const ratio =document.getElementById('align-ratio').checked;
  const PAD={{top:30,right:30,bottom:50,left:80}};
  const pw=W-PAD.left-PAD.right, ph=H-PAD.top-PAD.bottom;
  const ld=layerData();

  const akFull=ld.align_k;
  const maxKRaw=Math.max(1,Math.min(akFull.length,parseInt(document.getElementById('align-maxk').value)||akFull.length));
  const ak=akFull.slice(0,maxKRaw);
  const maxK=ak.length;
  const n_ambient=ld.n_intermediate || RAW.n_intermediate || ld.n_rank;
  const akRand=ak.map((_,i)=>(i+1)/n_ambient);

  const plotAk  =ratio?ak.map((v,i)=>v/akRand[i]):ak;
  const plotRand=ratio?ak.map(()=>1):akRand;

  const allY=[...plotAk,...(ratio?[]:plotRand)].filter(v=>v>0);
  let yMin=Math.min(...allY), yMax=Math.max(...allY);
  if(logY) {{
    yMin=Math.pow(10,Math.floor(Math.log10(yMin)));
    yMax=Math.pow(10,Math.ceil(Math.log10(yMax)));
  }} else {{ yMin=0; yMax=yMax*1.05; }}

  function toX(k1) {{ return logX
    ?PAD.left+Math.log10(k1)/Math.log10(maxK)*pw
    :PAD.left+(k1-1)/(maxK-1)*pw; }}
  function toY(v) {{ return logY
    ?PAD.top+(1-(Math.log10(Math.max(v,1e-30))-Math.log10(yMin))/(Math.log10(yMax)-Math.log10(yMin)))*ph
    :PAD.top+(1-(v-yMin)/(yMax-yMin))*ph; }}

  ctx.clearRect(0,0,W,H);
  drawGrid(ctx,PAD,pw,ph,
    logX?logTicks(1,maxK):linTicks(1,maxK,8),
    logY?logTicks(yMin,yMax):linTicks(yMin,yMax,6),
    toX,toY);
  drawAxes(ctx,PAD,pw,ph);

  ctx.fillStyle='#444'; ctx.font='13px Georgia,serif'; ctx.textAlign='center';
  ctx.fillText('k',PAD.left+pw/2,H-8);
  ctx.save(); ctx.translate(14,PAD.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(ratio?'alignₖ / (k/N)':'alignₖ',0,0); ctx.restore();

  if(ratio) {{
    const y1=toY(1);
    ctx.strokeStyle='#aaa'; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.moveTo(PAD.left,y1); ctx.lineTo(PAD.left+pw,y1); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#aaa'; ctx.font='11px Georgia,serif'; ctx.textAlign='left';
    ctx.fillText('random  k/N = 1',PAD.left+4,y1-4);
  }} else {{
    ctx.strokeStyle='#aaa'; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
    ctx.beginPath();
    plotRand.forEach((v,i)=>{{ const x=toX(i+1),y=toY(v); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }});
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle='#aaa'; ctx.font='11px Georgia,serif'; ctx.textAlign='left';
    const li=Math.floor(maxK*0.6);
    ctx.fillText('random',toX(li+1),toY(plotRand[li])-5);
  }}

  ctx.strokeStyle='#2266cc'; ctx.lineWidth=2;
  ctx.beginPath();
  plotAk.forEach((v,i)=>{{ const x=toX(i+1),y=toY(v); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }});
  ctx.stroke();
}}

// ── 4. Cos-sim scatter ────────────────────────────────────────────────────────
const cossimCanvas = document.getElementById('cossim-canvas');

function drawCossim() {{
  const dpr=window.devicePixelRatio||1;
  const W=cossimCanvas.offsetWidth, H=cossimCanvas.offsetHeight||420;
  cossimCanvas.width=W*dpr; cossimCanvas.height=H*dpr;
  const ctx=cossimCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const sq   = document.getElementById('cossim-sq').checked;
  const logX = document.getElementById('cossim-logx').checked;
  const PAD={{top:30,right:30,bottom:50,left:70}};
  const pw=W-PAD.left-PAD.right, ph=H-PAD.top-PAD.bottom;
  const ld=layerData();

  const raw_out = ld.cossim_out || [];
  const raw_in  = ld.cossim_in  || [];
  if (!raw_out.length && !raw_in.length) {{
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='#aaa'; ctx.font='14px Georgia,serif'; ctx.textAlign='center';
    ctx.fillText('(no data — recompute experiment)', W/2, H/2);
    return;
  }}

  const vals_out = sq ? raw_out.map(v=>v*v) : raw_out;
  const vals_in  = sq ? raw_in.map(v=>v*v)  : raw_in;
  const n = Math.max(vals_out.length, vals_in.length);

  const allVals = [...vals_out, ...vals_in];
  const yMin_data=Math.min(...allVals), yMax_data=Math.max(...allVals);
  const yPad=(yMax_data-yMin_data)*0.07||0.05;
  const yMin=yMin_data-yPad, yMax=yMax_data+yPad;

  function toX(i) {{
    return logX
      ? PAD.left+(i>0?Math.log10(i+1)/Math.log10(n):0)*pw
      : PAD.left+i/(n-1||1)*pw;
  }}
  function toY(v) {{ return PAD.top+(1-(v-yMin)/(yMax-yMin))*ph; }}

  ctx.clearRect(0,0,W,H);
  const toXgrid=logX?(v=>PAD.left+Math.log10(Math.max(v,1))/Math.log10(n)*pw):(v=>PAD.left+v/(n-1||1)*pw);
  drawGrid(ctx,PAD,pw,ph,
    logX?logTicks(1,n):linTicks(0,n,8),
    linTicks(yMin,yMax,6),
    toXgrid,toY);
  drawAxes(ctx,PAD,pw,ph);

  const y0=toY(0);
  if(y0>=PAD.top&&y0<=PAD.top+ph) {{
    ctx.strokeStyle='#ccc'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(PAD.left,y0); ctx.lineTo(PAD.left+pw,y0); ctx.stroke();
    ctx.setLineDash([]);
  }}

  ctx.fillStyle='#444'; ctx.font='13px Georgia,serif'; ctx.textAlign='center';
  ctx.fillText('singular vector index i', PAD.left+pw/2, H-8);
  ctx.save(); ctx.translate(14,PAD.top+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText(sq?'cos²(u, 1̂)':'cos(u, 1̂)',0,0); ctx.restore();

  const series=[
    {{vals:vals_out, color:'#2266cc', label:'W_up LSVs (uᵢᵘᵖ)'}},
    {{vals:vals_in,  color:'#cc4422', label:'W_down RSVs (vₖᵈᵒʷⁿ)'}},
  ];
  series.forEach((s,si)=>{{
    ctx.fillStyle=s.color; ctx.globalAlpha=0.7;
    s.vals.forEach((v,i)=>{{
      const x=toX(i),y=toY(v);
      ctx.beginPath(); ctx.arc(x,y,2.5,0,2*Math.PI); ctx.fill();
    }});
    ctx.globalAlpha=1;
    const lx=PAD.left+pw-110, ly=PAD.top+16+si*18;
    ctx.fillStyle=s.color; ctx.beginPath(); ctx.arc(lx,ly,4,0,2*Math.PI); ctx.fill();
    ctx.font='12px Georgia,serif'; ctx.textAlign='left';
    ctx.fillText(s.label,lx+8,ly+4);
  }});
}}

function redraw() {{ drawSpectra(); drawAlign(); drawHeatmap(); drawCossim(); drawSvHist(); }}
window.addEventListener('resize', redraw);
redraw();
fetchAndDrawSvHist();

// ── 5. Singular vector histogram ──────────────────────────────────────────────
const svhistCanvas = document.getElementById('svhist-canvas');
let svhistVec = null;  // currently loaded Float32Array

function svhistKey() {{
  const matrix = document.getElementById('svh-matrix').value;  // 'up' or 'down'
  const side   = document.getElementById('svh-side').value;    // 'L' or 'R'
  // map to the stored file key
  // W_up  LSVs = U_up  (neuron space, dim=n_intermediate)
  // W_up  RSVs = Vh_up (hidden space, dim=n_hidden)
  // W_down LSVs = U_down (hidden space, dim=n_hidden)
  // W_down RSVs = Vh_down (neuron space, dim=n_intermediate)
  if (matrix === 'up'   && side === 'L') return 'U_up';
  if (matrix === 'up'   && side === 'R') return 'Vh_up';
  if (matrix === 'down' && side === 'L') return 'U_down';
  if (matrix === 'down' && side === 'R') return 'Vh_down';
}}

function svhistDimLabel() {{
  const ld = layerData();
  const key = svhistKey();
  const dim = ld['dim_' + key];
  const space = (key === 'U_up' || key === 'Vh_down') ? 'neuron space' : 'hidden space';
  return dim ? `dim = ${{dim}}  (${{space}})` : '';
}}

function updateSvhistControls() {{
  const ld = layerData();
  const key = svhistKey();
  const n_vecs = ld.n_rank;
  const idx = Math.max(0, Math.min(n_vecs - 1, parseInt(document.getElementById('svh-index').value) || 0));
  document.getElementById('svh-index').max = n_vecs - 1;
  document.getElementById('svh-index').value = idx;
  document.getElementById('svh-dim-label').textContent = svhistDimLabel();
}}

function fetchAndDrawSvHist() {{
  updateSvhistControls();
  const key = svhistKey();
  const idx = parseInt(document.getElementById('svh-index').value) || 0;
  const ld = layerData();
  const dim = ld['dim_' + key];
  if (!dim) {{ svhistVec = null; drawSvHist(); return; }}

  const url = `../../data/{slug}/layer${{activeLayer}}_${{key}}.bin`;
  fetch(url)
    .then(r => r.arrayBuffer())
    .then(buf => {{
      const all = new Float32Array(buf);  // (n_vecs * dim,) row-major
      svhistVec = all.slice(idx * dim, (idx + 1) * dim);
      drawSvHist();
    }})
    .catch(() => {{ svhistVec = null; drawSvHist(); }});
}}

function drawSvHist() {{
  const dpr = window.devicePixelRatio || 1;
  const W = svhistCanvas.offsetWidth, H = svhistCanvas.offsetHeight || 420;
  svhistCanvas.width = W * dpr; svhistCanvas.height = H * dpr;
  const ctx = svhistCanvas.getContext('2d'); ctx.scale(dpr, dpr);
  const PAD = {{top:30, right:30, bottom:50, left:70}};
  const pw = W - PAD.left - PAD.right, ph = H - PAD.top - PAD.bottom;

  ctx.clearRect(0, 0, W, H);

  if (!svhistVec || svhistVec.length === 0) {{
    ctx.fillStyle = '#aaa'; ctx.font = '14px Georgia,serif'; ctx.textAlign = 'center';
    ctx.fillText('loading…', W/2, H/2);
    return;
  }}

  const logY = document.getElementById('svhist-logy').checked;
  const nBins = Math.max(10, Math.min(400, parseInt(document.getElementById('svhist-bins').value) || 80));
  const vals = Array.from(svhistVec);
  const vMin = Math.min(...vals), vMax = Math.max(...vals);

  // bin edges
  const edges = [];
  for (let b = 0; b <= nBins; b++) edges.push(vMin + (vMax - vMin) * b / nBins);
  const counts = new Array(nBins).fill(0);
  vals.forEach(v => {{
    let bi = Math.floor((v - vMin) / (vMax - vMin) * nBins);
    if (bi >= nBins) bi = nBins - 1;
    counts[bi]++;
  }});

  const yMax_raw = Math.max(...counts);
  let yMin = 0, yMax = yMax_raw * 1.08;
  if (logY) {{
    const nonzero = counts.filter(c => c > 0);
    yMin = Math.pow(10, Math.floor(Math.log10(Math.min(...nonzero))));
    yMax = Math.pow(10, Math.ceil(Math.log10(yMax_raw)));
  }}

  function toX(v) {{ return PAD.left + (v - vMin) / (vMax - vMin) * pw; }}
  function toY(v) {{
    if (logY) {{
      const ly = Math.log10(Math.max(v, 1e-30));
      return PAD.top + (1 - (ly - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * ph;
    }}
    return PAD.top + (1 - (v - yMin) / (yMax - yMin)) * ph;
  }}

  drawGrid(ctx, PAD, pw, ph,
    linTicks(vMin, vMax, 8),
    logY ? logTicks(yMin, yMax) : linTicks(yMin, yMax, 6),
    toX, toY);
  drawAxes(ctx, PAD, pw, ph);

  // draw bars
  const barColor = document.getElementById('svh-matrix').value === 'up' ? '#2266cc' : '#cc4422';
  ctx.fillStyle = barColor; ctx.globalAlpha = 0.7;
  counts.forEach((c, bi) => {{
    if (logY && c === 0) return;
    const x0 = toX(edges[bi]), x1 = toX(edges[bi+1]);
    const y0 = toY(logY ? Math.max(c, yMin) : c), y1 = toY(logY ? yMin : 0);
    ctx.fillRect(x0, y0, Math.max(x1 - x0 - 0.5, 0.5), y1 - y0);
  }});
  ctx.globalAlpha = 1;

  // axis labels
  const key = svhistKey();
  const matrix = document.getElementById('svh-matrix').value;
  const side   = document.getElementById('svh-side').value;
  const idx    = parseInt(document.getElementById('svh-index').value) || 0;
  const sideLabel = side === 'L' ? 'LSV' : 'RSV';
  const label = `W_${{matrix}} ${{sideLabel}} #${{idx}}  (n=${{vals.length}})`;
  ctx.fillStyle = '#444'; ctx.font = '13px Georgia,serif'; ctx.textAlign = 'center';
  ctx.fillText(label, PAD.left + pw/2, H - 8);
  ctx.save(); ctx.translate(14, PAD.top + ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('count', 0, 0); ctx.restore();

  // normal reference curve
  const mu = vals.reduce((a,b)=>a+b,0)/vals.length;
  const sd = Math.sqrt(vals.reduce((a,b)=>a+(b-mu)**2,0)/vals.length);
  if (sd > 0) {{
    ctx.strokeStyle = '#888'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
    ctx.beginPath();
    const binW = (vMax - vMin) / nBins;
    let started = false;
    for (let px = PAD.left; px <= PAD.left + pw; px += 1) {{
      const v = vMin + (px - PAD.left) / pw * (vMax - vMin);
      const density = Math.exp(-0.5*((v-mu)/sd)**2) / (sd * Math.sqrt(2*Math.PI));
      const c = density * vals.length * binW;
      if (logY && c < yMin) {{ started = false; continue; }}
      const y = toY(logY ? Math.max(c, yMin) : c);
      if (!started) {{ ctx.moveTo(px, y); started = true; }} else ctx.lineTo(px, y);
    }}
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#888'; ctx.font = '11px Georgia,serif'; ctx.textAlign = 'left';
    ctx.fillText('Gaussian fit', PAD.left + 4, PAD.top + 14);
  }}
}}
</script>
</body>
</html>"""

    path = out_dir / "index.html"
    path.write_text(html)
    print(f"  Wrote {path}")
    _register_experiment(slug, title, "", path)
    return path
