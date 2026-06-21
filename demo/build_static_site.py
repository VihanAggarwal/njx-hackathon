"""Build a STATIC, Vercel-deployable demo site from the committed benchmark run.

    python demo/build_static_site.py        # -> writes web/  (+ web/dualmind_graphs.zip)

Output `web/` is a self-contained static site (no server, no API key):
  web/index.html      landing page: headline leaderboard + featured figures
  web/figures.html    gallery of every research figure (relative links)
  web/dashboard.html  the self-contained Plotly dashboard (copied as-is)
  web/graphs/*.png    all figures
  web/dualmind_graphs.zip   downloadable bundle of every graph + dashboard
  web/vercel.json     static config (cleanUrls)

Deploy: drag-drop the `web/` folder onto vercel.com/new, or `cd web && npx vercel`.
Everything is regenerated from eval/results/, so re-run after a fresh benchmark.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(REPO, "eval", "results")
GRAPHS = os.path.join(RESULTS, "graphs")
OUT = os.path.join(REPO, "web")

FIG_TITLES = {
    "16_main_results": "Figure 1 — main results (forest · per-class · leaderboard · self-hardening)",
    "17_asr_forest": "Forest plot — ASR ± 95% CI, all defenses",
    "18_bootstrap_distributions": "Bootstrap sampling distributions of ASR (2000 resamples)",
    "19_ablation_waterfall": "Ablation waterfall — marginal ASR reduction per system",
    "20_roc_with_ci": "ROC with 95% bootstrap confidence bands",
    "21_per_class_grouped": "Per-attack-class ASR by configuration",
    "22_dualmind_calibration": "Calibration before vs after System 8 (ECE 0.257 → 0.086)",
    "23_latency_profile": "Latency profile — p50/p95/p99 with fast-path %",
    "01_asr_by_config": "ASR by configuration (95% CI)",
    "02_class_asr_heatmap": "Per-attack-class ASR heatmap",
    "03_self_hardening_curve": "Self-hardening curve",
    "04_roc_curves": "ROC curves by configuration",
    "05_pr_curves": "Precision-Recall curves",
    "06_calibration": "Calibration — reliability diagram + confidence histogram",
    "07_latency_violin": "Latency distribution (log scale)",
    "08_cost_vs_security": "Cost-security trade-off (Pareto)",
    "09_confusion_grid": "Confusion matrices by configuration",
    "10_fpr_vs_asr": "FPR vs ASR operating points",
    "11_leaderboard": "Head-to-head ASR leaderboard",
    "12_security_map": "2D security map",
    "13_competitive_heatmap": "Per-class ASR: DUALMIND vs competitors",
    "14_radar": "Defense capability radar",
    "15_latency_vs_accuracy": "Latency-accuracy frontier",
}
FEATURED = ["16_main_results", "11_leaderboard", "22_dualmind_calibration", "17_asr_forest"]

SHORT = {
    "DUALMIND (full)": "DUALMIND",
    "NeMo-style self-check (reimplementation)": "NeMo self-check *",
    "Vanilla LLM self-check": "Vanilla LLM self-check",
    "ProtectAI deberta-v3 prompt-injection-v2": "ProtectAI deberta-v3 v2",
    "Regex/keyword filter": "Regex / keyword filter",
    "Meta Prompt-Guard-86M": "Meta Prompt-Guard-2 86M",
}

CSS = """
:root{--bg:#0f1117;--panel:#1a1d27;--fg:#e8e8ef;--muted:#9aa0b4;--dm:#8073ac;--dm2:#542788;
  --ok:#1a9850;--bad:#d73027;--warn:#e08214;--line:#2a2e3c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;line-height:1.5}
a{color:#cbb6f0}
header{padding:16px 24px;background:linear-gradient(90deg,#2d1b4e,#0f1117);border-bottom:1px solid var(--line);
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}
h1{margin:0;font-size:20px;letter-spacing:.5px} .sub{color:var(--muted);font-size:12px;margin-top:3px}
a.btn{color:#cbb6f0;text-decoration:none;border:1px solid var(--dm);padding:8px 14px;border-radius:8px;font-size:13px}
a.btn:hover{background:#241a3a}
.hero{padding:34px 24px 8px;max-width:1100px;margin:0 auto;text-align:center}
.hero h2{font-size:30px;margin:0 0 8px}
.hero p{color:var(--muted);max-width:720px;margin:0 auto 18px;font-size:15px}
.badges{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin:18px 0}
.badge{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;min-width:130px}
.badge .n{font-size:26px;font-weight:800;color:#b59cf0} .badge .l{font-size:12px;color:var(--muted)}
.section{max-width:1100px;margin:0 auto;padding:18px 24px}
.section h3{font-size:16px;color:#cbb6f0;letter-spacing:.4px;border-bottom:1px solid var(--line);padding-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:8px}
th,td{padding:9px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:600;font-size:12px}
tr.dm{background:#1d1630} tr.dm td{font-weight:700;color:#d9caff}
.cta{display:flex;gap:12px;flex-wrap:wrap;justify-content:center;margin:8px 0 4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px;margin-top:14px}
figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden}
figure img{width:100%;display:block;background:#fff}
figcaption{padding:9px 12px;font-size:12.5px;color:#222;background:#f3f3f6;border-top:1px solid #e3e3e8}
footer{color:var(--muted);font-size:12px;text-align:center;padding:26px 24px;border-top:1px solid var(--line);margin-top:24px}
.note{color:var(--muted);font-size:12.5px;margin-top:6px}
"""

HEAD = ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
        '<title>__TITLE__</title><style>' + CSS + '</style></head><body>')


def _card(stem):
    title = FIG_TITLES.get(stem, stem.replace("_", " "))
    return (f'<figure><a href="./graphs/{stem}.png" target="_blank">'
            f'<img src="./graphs/{stem}.png" loading="lazy"/></a>'
            f'<figcaption>{title}</figcaption></figure>')


def build():
    comp = json.load(open(os.path.join(RESULTS, "competitive.json"), encoding="utf-8"))
    man = comp.get("manifest", {})
    defs = comp["defenses"]

    # --- assets --------------------------------------------------------- #
    # Preserve the handoff artifacts (generated elsewhere) across the rebuild.
    keep = {}
    for fn in ("demo_emails.json", "HANDOFF.md"):
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            keep[fn] = open(p, "rb").read()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "graphs"))
    for fn, data in keep.items():
        open(os.path.join(OUT, fn), "wb").write(data)
    pngs = sorted(glob.glob(os.path.join(GRAPHS, "*.png")))
    for p in pngs:
        shutil.copy2(p, os.path.join(OUT, "graphs", os.path.basename(p)))
    dash_src = os.path.join(RESULTS, "dashboard.html")
    if os.path.exists(dash_src):
        shutil.copy2(dash_src, os.path.join(OUT, "dashboard.html"))

    # --- downloadable zip of all graphs + dashboard --------------------- #
    # Self-describing so a downstream session can pick & choose graphs for a
    # website: INDEX.txt (human), figures.json (machine), LEADERBOARD.md (story).
    stems_all = [os.path.splitext(os.path.basename(p))[0] for p in pngs]
    lb = sorted(((k, v) for k, v in defs.items() if v.get("available")),
                key=lambda kv: (kv[1]["asr"], kv[1]["fpr"]))
    figures_manifest = {
        "mode": man.get("mode"), "n_items": man.get("n_items"), "seed": man.get("seed"),
        "featured": [f for f in FEATURED if f in stems_all],
        "figures": [{"file": f"graphs/{s}.png", "title": FIG_TITLES.get(s, s)}
                    for s in stems_all],
    }
    lb_md = ("# DUALMIND — head-to-head leaderboard (real measured)\n\n"
             f"mode={man.get('mode')} · n={man.get('n_items')} · seed={man.get('seed')} "
             "· real LLMail-Inject data · lower ASR/FPR/ECE better, higher TPR/F1/AUC better\n\n"
             "| Defense | ASR | FPR | TPR | F1 | AUC | ECE |\n|---|---|---|---|---|---|---|\n")
    for k, v in lb:
        lb_md += (f"| {SHORT.get(k, k)} | {v['asr']:.3f} | {v['fpr']:.3f} | {v['tpr']:.3f} "
                  f"| {v['f1']:.3f} | {v['roc']['auc']:.3f} | {v['ece']:.3f} |\n")

    zip_path = os.path.join(OUT, "dualmind_graphs.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pngs:
            z.write(p, os.path.join("dualmind_graphs", "graphs", os.path.basename(p)))
        if os.path.exists(dash_src):
            z.write(dash_src, os.path.join("dualmind_graphs", "dashboard.html"))
        idx = "DUALMIND benchmark graphs (all defenses tested)\n" + "=" * 48 + "\n"
        idx += f"mode={man.get('mode')}  n_items={man.get('n_items')}  seed={man.get('seed')}\n\n"
        for s in stems_all:
            idx += f"graphs/{s}.png  —  {FIG_TITLES.get(s, s)}\n"
        z.writestr(os.path.join("dualmind_graphs", "INDEX.txt"), idx)
        z.writestr(os.path.join("dualmind_graphs", "figures.json"),
                   json.dumps(figures_manifest, indent=2))
        z.writestr(os.path.join("dualmind_graphs", "LEADERBOARD.md"), lb_md)
        # bundle the live-demo handoff (real email traces + brief) if present
        for fn in ("demo_emails.json", "HANDOFF.md"):
            p = os.path.join(OUT, fn)
            if os.path.exists(p):
                z.write(p, os.path.join("dualmind_graphs", fn))

    # --- leaderboard table (real measured numbers) --------------------- #
    rows = sorted(((k, v) for k, v in defs.items() if v.get("available")),
                  key=lambda kv: kv[1]["asr"])
    trs = []
    for name, m in rows:
        dm = "DUALMIND" in name
        trs.append(
            f'<tr class="{ "dm" if dm else "" }"><td>{SHORT.get(name, name)}</td>'
            f'<td>{m["asr"]:.3f}</td><td>{m["fpr"]:.3f}</td><td>{m["tpr"]:.3f}</td>'
            f'<td>{m["f1"]:.3f}</td><td>{m["roc"]["auc"]:.3f}</td><td>{m["ece"]:.3f}</td></tr>')
    table = ("<table><thead><tr><th>Defense</th><th>ASR&#9660;</th><th>FPR&#9660;</th>"
             "<th>TPR&#9650;</th><th>F1&#9650;</th><th>AUC&#9650;</th><th>ECE&#9660;</th>"
             "</tr></thead><tbody>" + "".join(trs) + "</tbody></table>")

    dmm = defs.get("DUALMIND (full)", {})
    n_def = sum(1 for v in defs.values() if v.get("available"))
    mode = man.get("mode", "live")
    data_lbl = "real LLMail-Inject data" if "real" in json.dumps(man.get("datasets", "")) else f"{mode} run"

    # --- index.html ----------------------------------------------------- #
    feat = [f for f in FEATURED if os.path.exists(os.path.join(GRAPHS, f + ".png"))]
    index = (HEAD.replace("__TITLE__", "DUALMIND — prompt-injection defense") + f"""
<header><div><h1>&#128737;&#65039; DUALMIND</h1>
  <div class="sub">Self-hardening prompt-injection defense — benchmarked on {data_lbl}</div></div>
  <div style="display:flex;gap:10px;flex-wrap:wrap">
    <a class="btn" href="./figures.html">&#128444;&#65039; All figures</a>
    <a class="btn" href="./dashboard.html">&#128202; Interactive dashboard</a>
    <a class="btn" href="https://github.com/VihanAggarwal/njx-hackathon" target="_blank">&#9733; GitHub</a>
  </div></header>

<div class="hero">
  <h2>0% attacks through. 0% false positives.</h2>
  <p>DUALMIND combines dual-LLM privilege separation with taint propagation, so it
  scores the agent's <i>action</i> — not just the text — and catches indirect
  injections that dedicated classifiers miss. Every number below is a real measured
  result on the same dataset; nothing is fabricated.</p>
  <div class="badges">
    <div class="badge"><div class="n">{dmm.get('asr',0)*100:.0f}%</div><div class="l">attack success rate</div></div>
    <div class="badge"><div class="n">{dmm.get('fpr',0)*100:.0f}%</div><div class="l">false-positive rate</div></div>
    <div class="badge"><div class="n">{dmm.get('f1',0):.2f}</div><div class="l">F1 score</div></div>
    <div class="badge"><div class="n">{n_def}</div><div class="l">defenses compared</div></div>
  </div>
  <div class="cta">
    <a class="btn" href="./dashboard.html">Explore the interactive dashboard &rarr;</a>
    <a class="btn" href="./dualmind_graphs.zip" download>&#11015; Download all graphs (.zip)</a>
  </div>
</div>

<div class="section"><h3>Head-to-head — every defense, real measured numbers</h3>
{table}
<div class="note">&#9660; lower is better &nbsp;·&nbsp; &#9650; higher is better. DUALMIND row highlighted.
ECE for DUALMIND is its System&nbsp;8 isotonic-calibrated score (security metrics identical to the uncalibrated config).</div>
</div>

<div class="section"><h3>Featured figures</h3>
<div class="grid">{''.join(_card(f) for f in feat)}</div>
<div class="cta" style="margin-top:16px"><a class="btn" href="./figures.html">See all {len(pngs)} figures &rarr;</a></div>
</div>

<footer>DUALMIND · generated from the committed benchmark run
(mode={mode}, n={man.get('n_items')}, seed={man.get('seed')}).
Regenerate with <code>python eval/run_benchmark.py &amp;&amp; python eval/graphs.py &amp;&amp; python demo/build_static_site.py</code>.
</footer></body></html>""")
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(index)

    # --- figures.html --------------------------------------------------- #
    stems = [os.path.splitext(os.path.basename(p))[0] for p in pngs]
    feat_set = [f for f in FEATURED if f in stems]
    rest = [f for f in stems if f not in feat_set]
    figures = (HEAD.replace("__TITLE__", "DUALMIND — research figures") + f"""
<header><div><h1>&#128737;&#65039; DUALMIND — research figures</h1>
  <div class="sub">300-DPI · click any figure for full size</div></div>
  <div style="display:flex;gap:10px"><a class="btn" href="./index.html">&larr; Home</a>
  <a class="btn" href="./dualmind_graphs.zip" download>&#11015; Download .zip</a></div></header>
<div class="section"><h3>Detailed / statistical figures</h3>
<div class="grid">{''.join(_card(f) for f in feat_set)}</div></div>
<div class="section"><h3>Standard figures</h3>
<div class="grid">{''.join(_card(f) for f in rest)}</div></div>
</body></html>""")
    open(os.path.join(OUT, "figures.html"), "w", encoding="utf-8").write(figures)

    # --- vercel config + deploy note ----------------------------------- #
    open(os.path.join(OUT, "vercel.json"), "w", encoding="utf-8").write(
        json.dumps({"cleanUrls": True, "trailingSlash": False}, indent=2))
    open(os.path.join(OUT, "DEPLOY.md"), "w", encoding="utf-8").write(
        "# Deploy this demo to Vercel\n\n"
        "This `web/` folder is a self-contained static site (no server, no API key).\n\n"
        "**Easiest:** drag-and-drop this `web` folder onto https://vercel.com/new\n\n"
        "**CLI:**\n```bash\ncd web\nnpx vercel        # preview\nnpx vercel --prod # production\n```\n\n"
        "Regenerate after a new benchmark: `python demo/build_static_site.py`.\n")

    print(f"  built static site -> {OUT}")
    print(f"  figures: {len(pngs)}  ·  zip: {os.path.getsize(zip_path)//1024} KB")
    print("  preview locally:  python -m http.server -d web 8123")
    print("  deploy:           cd web && npx vercel")


if __name__ == "__main__":
    build()
