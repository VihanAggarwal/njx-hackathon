"""Eval — generate all 15 static graphs from saved results.

Every graph is regenerable WITHOUT re-running any LLM:

    python eval/graphs.py --from-results eval/results/results.json

Reads results.json, competitive.json, self_hardening.json, and results.csv (for raw
latency distributions). Writes 150-DPI PNGs to eval/results/graphs/.

Standards: titles, labeled axes, legends, 95% CI error bars where applicable, a
colorblind-safe palette, tight layout. DUALMIND configs are blue/purple and
prominent; competitors are gray/orange.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GRAPH_DIR = os.path.join(RESULTS_DIR, "graphs")

CONFIG_ORDER = [
    "1_no_defense", "2_prefilter_only", "3_dual_llm_only",
    "4_dual_llm_taint", "5_full_pre_hardening", "6_full_post_hardening",
]
CONFIG_LABELS = {
    "1_no_defense": "No defense",
    "2_prefilter_only": "Pre-filter only",
    "3_dual_llm_only": "Dual-LLM only",
    "4_dual_llm_taint": "Dual-LLM + taint",
    "5_full_pre_hardening": "Full (pre-harden)",
    "6_full_post_hardening": "Full (post-harden)",
}
ATTACK_CLASSES = ["direct", "indirect", "multi_hop_1", "multi_hop_2",
                  "multi_hop_3", "obfuscated", "encoded", "jailbreak"]

# DUALMIND blue->purple gradient; competitors gray/orange.
_DM_CMAP = plt.get_cmap("BuPu")
DM_COLORS = {c: _DM_CMAP(0.30 + 0.65 * i / (len(CONFIG_ORDER) - 1))
             for i, c in enumerate(CONFIG_ORDER)}
COMPETITOR_COLOR = "#7f7f7f"
COMPETITOR_ORANGE = "#e08214"
DUALMIND_HL = "#542788"  # prominent purple for DUALMIND in competitive plots


# --------------------------------------------------------------------------- #
def _load(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    base = os.path.dirname(os.path.abspath(results_path))
    comp = _maybe(os.path.join(base, "competitive.json"))
    hard = results.get("self_hardening") or _maybe(os.path.join(base, "self_hardening.json"))
    latencies = _load_latencies(os.path.join(base, "results.csv"))
    return results, comp, hard, latencies, base


def _maybe(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_latencies(csv_path):
    out = defaultdict(list)
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out[row["config"]].append(float(row["latency_ms"]))
            except (KeyError, ValueError):
                pass
    return out


def _save(fig, name):
    os.makedirs(GRAPH_DIR, exist_ok=True)
    path = os.path.join(GRAPH_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _ci_err(metric_ci, point):
    lo, hi = metric_ci
    return [[max(0, point - lo)], [max(0, hi - point)]]


# --------------------------------------------------------------------------- #
# 1. ASR by configuration
# --------------------------------------------------------------------------- #
def g1_asr_by_config(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    asr = [configs[k]["asr"] for k in keys]
    errs = np.array([_ci_err(configs[k]["asr_ci"], configs[k]["asr"]) for k in keys]).T.reshape(2, -1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar([CONFIG_LABELS[k] for k in keys], asr, yerr=errs, capsize=4,
                  color=[DM_COLORS[k] for k in keys], edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, asr):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Attack Success Rate (lower is better)")
    ax.set_title("ASR by DUALMIND configuration (95% bootstrap CI)")
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "01_asr_by_config.png")


# --------------------------------------------------------------------------- #
# 2. Per-attack-class ASR heatmap (DUALMIND configs)
# --------------------------------------------------------------------------- #
def _class_matrix(rowsrc, row_keys, label_fn):
    classes = [c for c in ATTACK_CLASSES
               if any(c in rowsrc[k].get("per_class_asr", {}) for k in row_keys)]
    mat = np.full((len(row_keys), len(classes)), np.nan)
    for i, k in enumerate(row_keys):
        pc = rowsrc[k].get("per_class_asr", {})
        for j, c in enumerate(classes):
            if c in pc:
                mat[i, j] = pc[c]
    return mat, classes


def _heatmap(mat, row_labels, col_labels, title, fname):
    fig, ax = plt.subplots(figsize=(1.1 * len(col_labels) + 3, 0.6 * len(row_labels) + 2.5))
    cmap = plt.get_cmap("RdYlGn_r")
    im = ax.imshow(np.nan_to_num(mat, nan=0.0), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if (v < 0.25 or v > 0.75) else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ASR (green=safe, red=fails)")
    ax.set_title(title)
    _save(fig, fname)


def g2_class_heatmap(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    mat, classes = _class_matrix(configs, keys, CONFIG_LABELS)
    _heatmap(mat, [CONFIG_LABELS[k] for k in keys], classes,
             "Per-attack-class ASR by DUALMIND config (taint kills multi-hop)",
             "02_class_asr_heatmap.png")


# --------------------------------------------------------------------------- #
# 3. Self-hardening curve
# --------------------------------------------------------------------------- #
def g3_self_hardening(hard):
    if not hard or not hard.get("asr_per_round"):
        return
    asr = hard["asr_per_round"]
    x = list(range(len(asr)))
    fig, ax = plt.subplots(figsize=(8, 5))
    start = asr[0]
    ax.plot(x, asr, "-o", color=DUALMIND_HL, linewidth=2, markersize=6, label="ASR")
    ax.fill_between(x, asr, start, color=DUALMIND_HL, alpha=0.15,
                    label="improvement vs round 0")
    ax.axhline(start, color="gray", linestyle="--", linewidth=0.8)
    ax.annotate(f"final ASR={asr[-1]:.2f}", xy=(x[-1], asr[-1]),
                xytext=(x[-1] - 0.5, asr[-1] + 0.08),
                arrowprops=dict(arrowstyle="->"), fontsize=10, fontweight="bold")
    ax.set_xlabel("Red-team round")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Self-hardening: ASR decreases as the system is attacked")
    ax.set_ylim(-0.02, max(0.3, start + 0.15))
    ax.legend()
    _save(fig, "03_self_hardening_curve.png")


# --------------------------------------------------------------------------- #
# 4 & 5. ROC and PR overlays
# --------------------------------------------------------------------------- #
def g4_roc(configs):
    keys = [k for k in CONFIG_ORDER if k in configs and configs[k].get("roc")]
    fig, ax = plt.subplots(figsize=(7, 7))
    for k in keys:
        roc = configs[k]["roc"]
        ax.plot(roc["fpr"], roc["tpr"], color=DM_COLORS[k], linewidth=1.8,
                label=f"{CONFIG_LABELS[k]} (AUC={roc['auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves by DUALMIND configuration")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "04_roc_curves.png")


def g5_pr(configs):
    keys = [k for k in CONFIG_ORDER if k in configs and configs[k].get("pr")]
    fig, ax = plt.subplots(figsize=(7, 7))
    for k in keys:
        pr = configs[k]["pr"]
        ax.plot(pr["recall"], pr["precision"], color=DM_COLORS[k], linewidth=1.8,
                label=f"{CONFIG_LABELS[k]} (AP={pr['ap']:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves by DUALMIND configuration")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, "05_pr_curves.png")


# --------------------------------------------------------------------------- #
# 6. Calibration reliability diagram
# --------------------------------------------------------------------------- #
def g6_calibration(configs, comp):
    systems = []
    if "2_prefilter_only" in configs:
        systems.append(("Pre-filter", configs["2_prefilter_only"], DUALMIND_HL))
    if comp:
        oranges = ["#e08214", "#fdb863", "#b35806"]
        i = 0
        for name, m in comp.get("defenses", {}).items():
            if m.get("available") and m.get("reliability") and "DUALMIND" not in name:
                systems.append((name, m, oranges[i % len(oranges)])); i += 1
    if not systems:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect calibration")
    for name, m, color in systems:
        rel = m["reliability"]
        conf, acc, counts = rel["confidence"], rel["accuracy"], rel["counts"]
        pts = [(c, a) for c, a, n in zip(conf, acc, counts) if n > 0]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, markersize=4,
                label=f"{name} (ECE={m.get('ece', 0):.3f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration reliability diagram")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "06_calibration.png")


# --------------------------------------------------------------------------- #
# 7. Latency violin per config
# --------------------------------------------------------------------------- #
def g7_latency(configs, latencies):
    keys = [k for k in CONFIG_ORDER if latencies.get(k)]
    if not keys:
        return
    data = [latencies[k] for k in keys]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    parts = ax.violinplot(data, showmedians=True)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(DM_COLORS[keys[i]]); b.set_alpha(0.7)
    for i, k in enumerate(keys, start=1):
        p95 = configs[k]["latency"]["p95"]; p50 = configs[k]["latency"]["p50"]
        ax.text(i, p95, f"p95={p95:.1f}", fontsize=7, ha="center", va="bottom")
    if "2_prefilter_only" in configs:
        ref = configs["2_prefilter_only"]["latency"]["p95"]
        ax.axhline(ref, color="green", linestyle="--", linewidth=0.8,
                   label=f"pre-filter p95 ({ref:.1f}ms)")
        ax.legend(fontsize=8)
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([CONFIG_LABELS[k] for k in keys], rotation=20)
    ax.set_ylabel("Latency per request (ms)")
    ax.set_title("Latency distribution: fast pre-filter path vs slow dual-LLM path")
    _save(fig, "07_latency_violin.png")


# --------------------------------------------------------------------------- #
# 8. Cost vs security tradeoff (Pareto)
# --------------------------------------------------------------------------- #
def _pareto_frontier(xs, ys):
    """Lower x better, higher y better -> upper-left frontier."""
    pts = sorted(zip(xs, ys), key=lambda p: (p[0], -p[1]))
    frontier, best_y = [], -1
    for x, y in pts:
        if y >= best_y:
            frontier.append((x, y)); best_y = y
    return frontier


def g8_cost_security(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    xs = [configs[k]["cost_per_1000_usd"] for k in keys]
    ys = [1 - configs[k]["asr"] for k in keys]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for k, x, y in zip(keys, xs, ys):
        ax.scatter(x, y, s=110, color=DM_COLORS[k], edgecolor="black", zorder=3)
        ax.annotate(CONFIG_LABELS[k], (x, y), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)
    fr = _pareto_frontier(xs, ys)
    if len(fr) > 1:
        fx, fy = zip(*fr)
        ax.plot(fx, fy, "--", color="purple", linewidth=1, label="Pareto frontier")
        ax.legend()
    ax.set_xlabel("Cost per 1000 requests (USD)")
    ax.set_ylabel("Security  (1 - ASR, higher is better)")
    ax.set_title("Cost vs security tradeoff")
    _save(fig, "08_cost_vs_security.png")


# --------------------------------------------------------------------------- #
# 9. Confusion matrix grid
# --------------------------------------------------------------------------- #
def g9_confusion_grid(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    n = len(keys)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, k in enumerate(keys):
        c = configs[k]["confusion"]
        total = max(1, sum(c.values()))
        mat = np.array([[c["TP"], c["FN"]], [c["FP"], c["TN"]]]) / total * 100
        ax = axes[idx]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
        labels = [["TP", "FN"], ["FP", "TN"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{mat[i, j]:.0f}%", ha="center",
                        va="center", fontsize=8,
                        color="white" if mat[i, j] > 50 else "black")
        ax.set_title(CONFIG_LABELS[k], fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred+", "pred-"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["atk", "ben"], fontsize=7)
    for idx in range(len(keys), len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Confusion matrices by configuration (% of all items)")
    _save(fig, "09_confusion_grid.png")


# --------------------------------------------------------------------------- #
# 10. FPR vs ASR tradeoff (configs)
# --------------------------------------------------------------------------- #
def g10_fpr_asr(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in keys:
        ax.scatter(configs[k]["fpr"], configs[k]["asr"], s=110,
                   color=DM_COLORS[k], edgecolor="black", zorder=3)
        ax.annotate(CONFIG_LABELS[k], (configs[k]["fpr"], configs[k]["asr"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("False Positive Rate (lower is better)")
    ax.set_ylabel("Attack Success Rate (lower is better)")
    ax.set_title("FPR vs ASR — bottom-left is ideal")
    ax.set_xlim(-0.03, max(0.3, max(configs[k]["fpr"] for k in keys) + 0.05))
    ax.set_ylim(-0.03, 1.05)
    _save(fig, "10_fpr_vs_asr.png")


# --------------------------------------------------------------------------- #
# Competitive graphs (11-15)
# --------------------------------------------------------------------------- #
def _competitive_rows(comp):
    rows = []
    for name, m in comp.get("defenses", {}).items():
        if m.get("available"):
            rows.append((name, m))
    return rows


def g11_leaderboard(comp):
    rows = sorted(_competitive_rows(comp), key=lambda r: r[1]["asr"], reverse=True)
    if not rows:
        return
    names = [r[0] for r in rows]
    asr = [r[1]["asr"] for r in rows]
    errs = np.array([_ci_err(r[1]["asr_ci"], r[1]["asr"]) for r in rows]).T.reshape(2, -1)
    colors = [DUALMIND_HL if "DUALMIND" in n else COMPETITOR_COLOR for n in names]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 2))
    bars = ax.barh(names, asr, xerr=errs, capsize=3, color=colors, edgecolor="black")
    for b, v in zip(bars, asr):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Attack Success Rate (lower is better)")
    ax.set_title("Head-to-head ASR leaderboard (DUALMIND vs competitors)")
    ax.set_xlim(0, 1.05)
    _save(fig, "11_leaderboard.png")


def g12_security_map(comp):
    rows = _competitive_rows(comp)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for name, m in rows:
        is_dm = "DUALMIND" in name
        ax.scatter(m["fpr"], m["asr"], s=180 if is_dm else 90,
                   color=DUALMIND_HL if is_dm else COMPETITOR_COLOR,
                   edgecolor="black", marker="*" if is_dm else "o", zorder=3)
        ax.annotate(name if is_dm else name.split("(")[0].strip(),
                    (m["fpr"], m["asr"]), textcoords="offset points",
                    xytext=(7, 5), fontsize=8,
                    fontweight="bold" if is_dm else "normal")
    ax.plot([0, 1], [1, 0], color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("False Positive Rate (lower is better)")
    ax.set_ylabel("Attack Success Rate (lower is better)")
    ax.set_title("The 2D security map — bottom-left corner is ideal")
    ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.03, 1.0)
    _save(fig, "12_security_map.png")


def g13_competitive_heatmap(comp):
    rows = _competitive_rows(comp)
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r[1]["asr"])
    rowdict = {n: m for n, m in rows}
    mat, classes = _class_matrix(rowdict, [n for n, _ in rows], lambda x: x)
    _heatmap(mat, [n.split("(")[0].strip() for n, _ in rows], classes,
             "Per-attack-class ASR: DUALMIND vs competitors (multi-hop columns)",
             "13_competitive_heatmap.png")


def g14_radar(comp):
    rows = _competitive_rows(comp)
    dm = [r for r in rows if "DUALMIND" in r[0]]
    others = sorted([r for r in rows if "DUALMIND" not in r[0]], key=lambda r: r[1]["asr"])[:3]
    chosen = dm + others
    if len(chosen) < 2:
        return
    axes_labels = ["Catch rate\n(1-ASR)", "Low FPR\n(1-FPR)", "Multi-hop\ndefense",
                   "Latency\nefficiency", "Calibration\n(1-ECE)"]
    max_p95 = max((m["latency"]["p95"] for _, m in rows), default=1) or 1

    def vec(m):
        pc = m.get("per_class_asr", {})
        mh = [pc[c] for c in ("multi_hop_1", "multi_hop_2", "multi_hop_3") if c in pc]
        mh_def = 1 - (sum(mh) / len(mh)) if mh else 1.0
        return [1 - m["asr"], 1 - m["fpr"], mh_def,
                1 - m["latency"]["p95"] / max_p95, 1 - m.get("ece", 0)]

    angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    for i, (name, m) in enumerate(chosen):
        is_dm = "DUALMIND" in name
        vals = vec(m); vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.2 if is_dm else 1.3,
                color=DUALMIND_HL if is_dm else None,
                label=name.split("(")[0].strip())
        ax.fill(angles, vals, alpha=0.25 if is_dm else 0.08,
                color=DUALMIND_HL if is_dm else None)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(axes_labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Defense capability radar (outer = better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    _save(fig, "14_radar.png")


def g15_latency_accuracy(comp):
    rows = _competitive_rows(comp)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6))
    xs = [m["latency"]["p95"] for _, m in rows]
    ys = [1 - m["asr"] for _, m in rows]
    for (name, m), x, y in zip(rows, xs, ys):
        is_dm = "DUALMIND" in name
        ax.scatter(x, y, s=160 if is_dm else 80,
                   color=DUALMIND_HL if is_dm else COMPETITOR_COLOR,
                   marker="*" if is_dm else "o", edgecolor="black", zorder=3)
        ax.annotate(name.split("(")[0].strip(), (x, y), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)
    fr = _pareto_frontier(xs, ys)
    if len(fr) > 1:
        fx, fy = zip(*fr)
        ax.plot(fx, fy, "--", color="purple", linewidth=1, label="Pareto frontier")
        ax.legend()
    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("Security (1 - ASR)")
    ax.set_title("Latency vs accuracy frontier")
    _save(fig, "15_latency_vs_accuracy.png")


# --------------------------------------------------------------------------- #
def build_inlined_dashboard(results_path):
    """Emit eval/results/dashboard.html with data inlined (works on double-click)."""
    template = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    if not os.path.exists(template):
        return
    results, comp, _, _, base = _load(results_path)
    with open(template, "r", encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps({"results": results, "competitive": comp}, default=str)
    html = html.replace("/*__DUALMIND_DATA__*/",
                        f"const DUALMIND_DATA = {payload};")
    out = os.path.join(RESULTS_DIR, "dashboard.html")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  wrote dashboard.html (self-contained, {len(html)//1024} KB)")


def generate_all(results_path):
    results, comp, hard, latencies, base = _load(results_path)
    configs = results.get("configs", {})
    print(f"Generating graphs -> {GRAPH_DIR}")
    graphs = [
        ("ASR by config", lambda: g1_asr_by_config(configs)),
        ("class heatmap", lambda: g2_class_heatmap(configs)),
        ("self-hardening", lambda: g3_self_hardening(hard)),
        ("ROC", lambda: g4_roc(configs)),
        ("PR", lambda: g5_pr(configs)),
        ("calibration", lambda: g6_calibration(configs, comp)),
        ("latency", lambda: g7_latency(configs, latencies)),
        ("cost-security", lambda: g8_cost_security(configs)),
        ("confusion", lambda: g9_confusion_grid(configs)),
        ("fpr-asr", lambda: g10_fpr_asr(configs)),
    ]
    if comp:
        graphs += [
            ("leaderboard", lambda: g11_leaderboard(comp)),
            ("security map", lambda: g12_security_map(comp)),
            ("competitive heatmap", lambda: g13_competitive_heatmap(comp)),
            ("radar", lambda: g14_radar(comp)),
            ("latency-accuracy", lambda: g15_latency_accuracy(comp)),
        ]
    for label, fn in graphs:
        try:
            fn()
        except Exception as e:
            print(f"  [warn] graph '{label}' failed: {e}")
    try:
        build_inlined_dashboard(results_path)
    except Exception as e:
        print(f"  [warn] dashboard build failed: {e}")
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate DUALMIND graphs from results")
    ap.add_argument("--from-results", default=os.path.join(RESULTS_DIR, "results.json"))
    args = ap.parse_args(argv)
    generate_all(args.from_results)


if __name__ == "__main__":
    main()
