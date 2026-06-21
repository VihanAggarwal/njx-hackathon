"""Eval — publication-quality figures, regenerable from saved results.

    python eval/graphs.py --from-results eval/results/results.json

Reads results.json, competitive.json, self_hardening.json, and results.csv (raw
per-attack rows, used to recompute bootstrap sampling distributions, ROC CI bands,
etc. without re-running any LLM). Writes 300-DPI PNGs to eval/results/graphs/.

Design language: serif typography, perceptually-uniform colormaps, despined axes,
95% bootstrap CI error bars / bands, panel labels (a)(b)(c), value annotations.
DUALMIND configurations are a viridis ramp (ordered); competitors are muted
gray/orange. The intent is figures that read like a paper, not a slide deck.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np

from eval import metrics as M

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
CLASS_LABELS = {c: c.replace("multi_hop_", "m-hop ").replace("_", " ") for c in ATTACK_CLASSES}
MULTIHOP = {"multi_hop_1", "multi_hop_2", "multi_hop_3"}


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.linewidth": 0.9, "axes.edgecolor": "#2b2b2b",
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#d9d9d9", "grid.linewidth": 0.6,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.color": "#2b2b2b", "ytick.color": "#2b2b2b",
        "legend.fontsize": 8.5, "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc", "legend.fancybox": False,
    })


def _viridis(n):
    cmap = plt.get_cmap("viridis")
    return [cmap(x) for x in np.linspace(0.12, 0.88, n)]


DM_RAMP = _viridis(6)
DM_COLORS = {c: DM_RAMP[i] for i, c in enumerate(CONFIG_ORDER)}
DUALMIND_HL = "#3b1f6b"     # deep purple for DUALMIND in competitive plots
COMPETITOR_GRAY = "#8c8c8c"
COMPETITOR_EDGE = "#5a5a5a"


def _despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)


def _panel(ax, letter, dx=-0.08, dy=1.04):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="right")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    base = os.path.dirname(os.path.abspath(results_path))
    comp = _maybe(os.path.join(base, "competitive.json"))
    hard = results.get("self_hardening") or _maybe(os.path.join(base, "self_hardening.json"))
    rows = _load_rows(os.path.join(base, "results.csv"))
    return results, comp, hard, rows


def _maybe(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_rows(csv_path):
    """Per-config raw result rows -> {config: [ {is_attack,blocked,score,attack_class,latency_ms} ]}."""
    out = defaultdict(list)
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out[r["config"]].append({
                    "is_attack": r["is_attack"].lower() == "true",
                    "blocked": r["blocked"].lower() == "true",
                    "score": float(r["score"]),
                    "attack_class": r["attack_class"],
                    "latency_ms": float(r["latency_ms"]),
                })
            except (KeyError, ValueError):
                pass
    return out


def _save(fig, name):
    os.makedirs(GRAPH_DIR, exist_ok=True)
    fig.savefig(os.path.join(GRAPH_DIR, name))
    plt.close(fig)
    print(f"  wrote {name}")


def _ci_err(ci, point):
    lo, hi = ci
    return [max(0, point - lo)], [max(0, hi - point)]


def _bootstrap_dist(rows, stat=M.attack_success_rate, n=2000, seed=42):
    rows = list(rows)
    if not rows:
        return np.array([0.0])
    rng = random.Random(seed)
    k = len(rows)
    return np.array([stat([rows[rng.randrange(k)] for _ in range(k)]) for _ in range(n)])


# ============================ ABLATION FIGURES ============================= #
def g1_asr_by_config(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    asr = [configs[k]["asr"] for k in keys]
    lo = [a - configs[k]["asr_ci"][0] for a, k in zip(asr, keys)]
    hi = [configs[k]["asr_ci"][1] - a for a, k in zip(asr, keys)]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(keys))
    bars = ax.bar(x, asr, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], capsize=4,
                  color=[DM_COLORS[k] for k in keys], edgecolor="#222", linewidth=0.7,
                  error_kw={"elinewidth": 1.1, "ecolor": "#444"})
    for xi, v in zip(x, asr):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold")
    # cumulative-reduction annotation
    if len(asr) >= 2 and asr[0] > asr[-1]:
        ax.annotate("", xy=(x[-1], asr[-1] + 0.0), xytext=(x[0], asr[0]),
                    arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.3,
                                    connectionstyle="arc3,rad=-0.25"))
        ax.text(np.mean(x), asr[0] * 0.62,
                f"{(asr[0]-asr[-1])*100:.0f} pp ASR reduction",
                color="#b2182b", fontsize=9.5, ha="center", style="italic")
    ax.set_xticks(x); ax.set_xticklabels([CONFIG_LABELS[k] for k in keys], rotation=18, ha="right")
    ax.set_ylabel("Attack Success Rate"); ax.set_ylim(0, 1.12)
    ax.set_title("Cumulative effect of each defense layer on ASR")
    ax.grid(axis="x", visible=False)
    ax.text(0.99, 0.97, "lower is better", transform=ax.transAxes, ha="right",
            va="top", fontsize=8.5, style="italic", color="#666")
    _despine(ax)
    _save(fig, "01_asr_by_config.png")


def _heatmap(ax, mat, row_labels, col_labels, cmap="RdYlGn_r", annot=True,
             cbar_label="ASR", fontsize=7.5):
    im = ax.imshow(np.nan_to_num(mat, nan=0.0), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    if annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    v = mat[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fontsize,
                            color="white" if (v < 0.28 or v > 0.72) else "#222")
    return im


def _class_matrix(rowsrc, row_keys):
    classes = [c for c in ATTACK_CLASSES
               if any(c in rowsrc[k].get("per_class_asr", {}) for k in row_keys)]
    mat = np.full((len(row_keys), len(classes)), np.nan)
    for i, k in enumerate(row_keys):
        pc = rowsrc[k].get("per_class_asr", {})
        for j, c in enumerate(classes):
            if c in pc:
                mat[i, j] = pc[c]
    return mat, classes


def g2_class_heatmap(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    mat, classes = _class_matrix(configs, keys)
    fig, ax = plt.subplots(figsize=(1.05 * len(classes) + 3.2, 0.62 * len(keys) + 2.6))
    im = _heatmap(ax, mat, [CONFIG_LABELS[k] for k in keys],
                  [CLASS_LABELS.get(c, c) for c in classes])
    # bracket the multi-hop columns
    mh_idx = [j for j, c in enumerate(classes) if c in MULTIHOP]
    if mh_idx:
        ax.add_patch(plt.Rectangle((min(mh_idx) - .5, -.5), len(mh_idx), len(keys),
                     fill=False, edgecolor="#08306b", lw=2.2))
        ax.text((min(mh_idx) + max(mh_idx)) / 2, -.85, "multi-hop",
                ha="center", va="bottom", fontsize=9, color="#08306b", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("ASR  (green = defended, red = bypassed)", fontsize=9)
    ax.set_title("Per-attack-class ASR by configuration")
    _save(fig, "02_class_asr_heatmap.png")


def g3_self_hardening(hard):
    if not hard or not hard.get("asr_per_round"):
        return
    asr = hard["asr_per_round"]; x = list(range(len(asr)))
    byp = hard.get("bypasses_per_round", [])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x, asr, asr[0], color="#6a51a3", alpha=0.12)
    ax.plot(x, asr, "-o", color="#3b1f6b", lw=2.4, ms=7, mfc="white", mec="#3b1f6b",
            mew=1.6, label="ASR (held-out red-team pool)", zorder=5)
    ax.axhline(asr[0], color="#999", ls="--", lw=0.9)
    ax.annotate(f"initial {asr[0]:.2f}", xy=(0, asr[0]), xytext=(0.4, asr[0] - 0.06),
                fontsize=9, color="#555")
    ax.annotate(f"final {asr[-1]:.2f}", xy=(x[-1], asr[-1]),
                xytext=(x[-1] - 1.2, asr[-1] + 0.07), fontsize=10, fontweight="bold",
                color="#3b1f6b", arrowprops=dict(arrowstyle="->", color="#3b1f6b"))
    if byp:
        ax2 = ax.twinx()
        ax2.bar([i + 1 for i in range(len(byp))], byp, width=0.35, color="#fdae6b",
                alpha=0.6, label="new bypasses found")
        ax2.set_ylabel("new bypasses / round", color="#a85", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#a85")
        ax2.grid(False)
    ax.set_xlabel("Red-team round"); ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(-0.03, max(0.25, asr[0] + 0.18))
    ax.set_title("Self-hardening: ASR falls as the system is attacked")
    ax.legend(loc="upper right"); _despine(ax)
    _save(fig, "03_self_hardening_curve.png")


def g4_roc(configs, rows):
    keys = [k for k in CONFIG_ORDER if k in configs and configs[k].get("roc")]
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    for i, k in enumerate(keys):
        roc = configs[k]["roc"]
        ax.plot(roc["fpr"], roc["tpr"], color=DM_COLORS[k], lw=2.0,
                label=f"{CONFIG_LABELS[k]}  (AUC={roc['auc']:.2f})")
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1.0, label="chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal"); ax.set_title("ROC curves by configuration")
    ax.legend(loc="lower right"); _despine(ax)
    _save(fig, "04_roc_curves.png")


def g5_pr(configs):
    keys = [k for k in CONFIG_ORDER if k in configs and configs[k].get("pr")]
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    for k in keys:
        pr = configs[k]["pr"]
        ax.plot(pr["recall"], pr["precision"], color=DM_COLORS[k], lw=2.0,
                label=f"{CONFIG_LABELS[k]}  (AP={pr['ap']:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05); ax.set_aspect("equal")
    ax.set_title("Precision-Recall curves by configuration")
    ax.legend(loc="lower left"); _despine(ax)
    _save(fig, "05_pr_curves.png")


def g6_calibration(configs, comp):
    """Guo-style reliability diagram + confidence histogram (two panels)."""
    systems = []
    if configs.get("2_prefilter_only", {}).get("reliability"):
        systems.append(("Pre-filter", configs["2_prefilter_only"], DUALMIND_HL))
    if comp:
        oranges = ["#e6550d", "#fd8d3c", "#a63603"]
        i = 0
        for n, m in comp.get("defenses", {}).items():
            if m.get("available") and m.get("reliability") and "DUALMIND" not in n:
                systems.append((n, m, oranges[i % 3])); i += 1
    if not systems:
        return
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 7.4),
                             gridspec_kw={"height_ratios": [3, 1.1]}, sharex=True)
    top, bot = axes
    top.plot([0, 1], [0, 1], color="#888", ls="--", lw=1.0, label="perfect calibration")
    width = 0.9 / max(1, len(systems))
    for s, (name, m, color) in enumerate(systems):
        rel = m["reliability"]; conf, acc, cnt = rel["confidence"], rel["accuracy"], rel["counts"]
        pts = [(c, a) for c, a, n in zip(conf, acc, cnt) if n > 0]
        if pts:
            xs, ys = zip(*pts)
            top.plot(xs, ys, "-o", color=color, ms=4.5, lw=1.6,
                     label=f"{_short(name)}  (ECE={m.get('ece', 0):.3f})")
        # gap bars (predicted - observed) on the bottom panel via confidence hist
        centers = np.linspace(0.05, 0.95, len(cnt))
        bot.bar(centers + (s - len(systems)/2) * width, cnt, width=width,
                color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
    top.set_ylabel("Observed accuracy"); top.set_ylim(-0.02, 1.02)
    top.set_title("Calibration: reliability diagram + confidence histogram")
    top.legend(loc="upper left"); _despine(top)
    bot.set_xlabel("Predicted probability (confidence)"); bot.set_ylabel("count")
    bot.set_xlim(-0.02, 1.02); _despine(bot)
    fig.align_ylabels(axes)
    _save(fig, "06_calibration.png")


def g7_latency(configs, rows):
    keys = [k for k in CONFIG_ORDER if rows.get(k)]
    if not keys:
        return
    data = [np.maximum([r["latency_ms"] for r in rows[k]], 1e-3) for k in keys]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(DM_COLORS[keys[i]]); b.set_alpha(0.55); b.set_edgecolor("#333")
    bp = ax.boxplot(data, widths=0.18, showfliers=False, patch_artist=True,
                    medianprops=dict(color="white", lw=1.4),
                    boxprops=dict(facecolor="#333", edgecolor="#333"),
                    whiskerprops=dict(color="#333"), capprops=dict(color="#333"))
    for i, k in enumerate(keys, 1):
        p95 = configs[k]["latency"]["p95"]
        ax.annotate(f"p95={p95:.0f}ms" if p95 >= 10 else f"p95={p95:.1f}ms",
                    xy=(i, max(data[i-1])), xytext=(i, max(data[i-1]) * 2.2),
                    ha="center", fontsize=7.5, color="#444")
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels([CONFIG_LABELS[k] for k in keys], rotation=18, ha="right")
    ax.set_ylabel("Latency per request (ms, log scale)")
    ax.set_title("Latency distribution: fast pre-filter path vs slow dual-LLM path")
    ax.grid(axis="x", visible=False); _despine(ax)
    _save(fig, "07_latency_violin.png")


def _pareto(xs, ys):
    pts = sorted(zip(xs, ys), key=lambda p: (p[0], -p[1]))
    fr, best = [], -np.inf
    for x, y in pts:
        if y >= best:
            fr.append((x, y)); best = y
    return fr


def g8_cost_security(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    xs = [max(configs[k]["cost_per_1000_usd"], 1e-4) for k in keys]
    ys = [1 - configs[k]["asr"] for k in keys]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    fr = _pareto(xs, ys)
    if len(fr) > 1:
        fx, fy = zip(*fr)
        ax.plot(fx, fy, "--", color="#3b1f6b", lw=1.3, zorder=1, label="Pareto frontier")
        ax.fill_between(fx, fy, 0, color="#3b1f6b", alpha=0.05)
    for k, x, y in zip(keys, xs, ys):
        ax.scatter(x, y, s=150, color=DM_COLORS[k], edgecolor="#222", lw=0.8, zorder=3)
        ax.annotate(CONFIG_LABELS[k], (x, y), textcoords="offset points",
                    xytext=(7, 6), fontsize=8.5)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Cost per 1000 requests (USD, symlog)")
    ax.set_ylabel("Security  (1 - ASR)")
    ax.set_title("Cost-security trade-off (efficient frontier)")
    ax.legend(loc="lower right"); _despine(ax)
    _save(fig, "08_cost_vs_security.png")


def g9_confusion_grid(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    cols = 3; rows_ = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(3.4 * cols, 3.1 * rows_))
    axes = np.array(axes).reshape(-1)
    for idx, k in enumerate(keys):
        c = configs[k]["confusion"]; tot = max(1, sum(c.values()))
        mat = np.array([[c["TP"], c["FN"]], [c["FP"], c["TN"]]]) / tot * 100
        ax = axes[idx]
        im = ax.imshow(mat, cmap="Purples", vmin=0, vmax=100)
        lab = [["TP", "FN"], ["FP", "TN"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{lab[i][j]}\n{mat[i,j]:.0f}%", ha="center", va="center",
                        fontsize=9, color="white" if mat[i, j] > 55 else "#222")
        ax.set_title(CONFIG_LABELS[k], fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred attack", "pred benign"], fontsize=7.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["attack", "benign"], fontsize=7.5)
        ax.tick_params(length=0)
    for idx in range(len(keys), len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Confusion matrices by configuration (% of all items)",
                 fontsize=13, fontweight="bold")
    _save(fig, "09_confusion_grid.png")


def g10_fpr_asr(configs):
    keys = [k for k in CONFIG_ORDER if k in configs]
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.axhspan(0, 0.1, xmin=0, xmax=0.12, color="#1a9850", alpha=0.06)
    ax.text(0.005, 0.02, "ideal", color="#1a9850", fontsize=9, style="italic")
    for k in keys:
        ax.scatter(configs[k]["fpr"], configs[k]["asr"], s=150, color=DM_COLORS[k],
                   edgecolor="#222", lw=0.8, zorder=3)
        ax.annotate(CONFIG_LABELS[k], (configs[k]["fpr"], configs[k]["asr"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8.5)
    ax.set_xlabel("False Positive Rate  (lower better)")
    ax.set_ylabel("Attack Success Rate  (lower better)")
    ax.set_title("FPR-ASR operating points (bottom-left is ideal)")
    ax.set_xlim(-0.03, max(0.3, max(configs[k]["fpr"] for k in keys) + 0.05))
    ax.set_ylim(-0.03, 1.05); _despine(ax)
    _save(fig, "10_fpr_vs_asr.png")


# ========================= COMPETITIVE FIGURES ============================= #
def _comp_rows(comp):
    return [(n, m) for n, m in comp.get("defenses", {}).items() if m.get("available")]


def _short(n):
    return (n.replace(" (reimplementation)", "*").replace("ProtectAI ", "")
             .replace("Meta ", "").replace("DUALMIND (full, post-hardening)", "DUALMIND"))


def g11_leaderboard(comp):
    rows = sorted(_comp_rows(comp), key=lambda r: r[1]["asr"], reverse=True)
    if not rows:
        return
    names = [_short(n) for n, _ in rows]
    asr = [m["asr"] for _, m in rows]
    lo = [a - m["asr_ci"][0] for a, (_, m) in zip(asr, rows)]
    hi = [m["asr_ci"][1] - a for a, (_, m) in zip(asr, rows)]
    colors = [DUALMIND_HL if "DUALMIND" in n else COMPETITOR_GRAY for n, _ in rows]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.8))
    ax.barh(names, asr, xerr=[np.maximum(lo, 0), np.maximum(hi, 0)], capsize=3.5,
            color=colors, edgecolor="#222", linewidth=0.7,
            error_kw={"elinewidth": 1.1, "ecolor": "#444"})
    for i, v in enumerate(asr):
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Attack Success Rate  (95% CI, lower better)")
    ax.set_xlim(0, 1.08); ax.set_title("Head-to-head ASR leaderboard")
    ax.legend(handles=[Patch(color=DUALMIND_HL, label="DUALMIND"),
                       Patch(color=COMPETITOR_GRAY, label="competitor")],
              loc="lower right")
    ax.grid(axis="y", visible=False); _despine(ax)
    _save(fig, "11_leaderboard.png")


def g12_security_map(comp):
    rows = _comp_rows(comp)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.2, 7))
    ax.axhspan(0, 0.15, color="#1a9850", alpha=0.05)
    ax.axvspan(0, 0.15, color="#1a9850", alpha=0.05)
    ax.text(0.01, 0.02, "ideal region", color="#1a9850", fontsize=9, style="italic")
    for name, m in rows:
        dm = "DUALMIND" in name
        ax.scatter(m["fpr"], m["asr"], s=320 if dm else 110,
                   color=DUALMIND_HL if dm else COMPETITOR_GRAY,
                   marker="*" if dm else "o", edgecolor="#222", lw=0.9, zorder=3)
        ax.annotate(_short(name), (m["fpr"], m["asr"]), textcoords="offset points",
                    xytext=(9, 6), fontsize=9, fontweight="bold" if dm else "normal")
    ax.set_xlabel("False Positive Rate  (lower better)")
    ax.set_ylabel("Attack Success Rate  (lower better)")
    ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.03, 1.0)
    ax.set_title("Security map: the bottom-left corner is best")
    _despine(ax)
    _save(fig, "12_security_map.png")


def g13_competitive_heatmap(comp):
    rows = sorted(_comp_rows(comp), key=lambda r: r[1]["asr"])
    if not rows:
        return
    rowdict = {n: m for n, m in rows}
    mat, classes = _class_matrix(rowdict, [n for n, _ in rows])
    fig, ax = plt.subplots(figsize=(1.05 * len(classes) + 3.4, 0.6 * len(rows) + 2.4))
    im = _heatmap(ax, mat, [_short(n) for n, _ in rows],
                  [CLASS_LABELS.get(c, c) for c in classes])
    mh_idx = [j for j, c in enumerate(classes) if c in MULTIHOP]
    if mh_idx:
        ax.add_patch(plt.Rectangle((min(mh_idx) - .5, -.5), len(mh_idx), len(rows),
                     fill=False, edgecolor="#08306b", lw=2.2))
        ax.text((min(mh_idx) + max(mh_idx)) / 2, -.8, "multi-hop",
                ha="center", fontsize=9, color="#08306b", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("ASR", fontsize=9)
    ax.set_title("Per-attack-class ASR: DUALMIND vs competitors")
    _save(fig, "13_competitive_heatmap.png")


def g14_radar(comp):
    rows = _comp_rows(comp)
    dm = [r for r in rows if "DUALMIND" in r[0]]
    others = sorted([r for r in rows if "DUALMIND" not in r[0]], key=lambda r: r[1]["asr"])[:3]
    chosen = dm + others
    if len(chosen) < 2:
        return
    labels = ["Attack catch\n(1-ASR)", "Low FPR\n(1-FPR)", "Multi-hop\ndefense",
              "Latency\nefficiency", "Calibration\n(1-ECE)"]
    maxp95 = max((m["latency"]["p95"] for _, m in rows), default=1) or 1

    def vec(m):
        pc = m.get("per_class_asr", {})
        mh = [pc[c] for c in MULTIHOP if c in pc]
        mhd = 1 - (sum(mh) / len(mh)) if mh else 1.0
        return [1 - m["asr"], 1 - m["fpr"], mhd,
                1 - m["latency"]["p95"] / maxp95, 1 - m.get("ece", 0)]

    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7.6, 7.6), subplot_kw=dict(polar=True))
    for name, m in chosen:
        d = "DUALMIND" in name; v = vec(m); v += v[:1]
        ax.plot(ang, v, lw=2.4 if d else 1.4, color=DUALMIND_HL if d else None,
                label=_short(name))
        ax.fill(ang, v, alpha=0.22 if d else 0.07, color=DUALMIND_HL if d else None)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1); ax.set_rlabel_position(18)
    ax.set_title("Defense capability radar (outer = better)", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12))
    _save(fig, "14_radar.png")


def g15_latency_accuracy(comp):
    rows = _comp_rows(comp)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    xs = [max(m["latency"]["p95"], 1e-2) for _, m in rows]
    ys = [1 - m["asr"] for _, m in rows]
    fr = _pareto(xs, ys)
    if len(fr) > 1:
        fx, fy = zip(*fr)
        ax.plot(fx, fy, "--", color="#3b1f6b", lw=1.3, label="Pareto frontier")
    for (name, m), x, y in zip(rows, xs, ys):
        d = "DUALMIND" in name
        ax.scatter(x, y, s=300 if d else 90, color=DUALMIND_HL if d else COMPETITOR_GRAY,
                   marker="*" if d else "o", edgecolor="#222", lw=0.9, zorder=3)
        ax.annotate(_short(name), (x, y), textcoords="offset points", xytext=(7, 6),
                    fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("p95 latency (ms, log)"); ax.set_ylabel("Security (1 - ASR)")
    ax.set_title("Latency-accuracy frontier"); ax.legend(loc="lower right"); _despine(ax)
    _save(fig, "15_latency_vs_accuracy.png")


# ===================== NEW, MORE DETAILED FIGURES ========================== #
def g16_main_results(configs, comp, hard):
    """Multi-panel 'Figure 1': ASR forest + per-class heatmap + leaderboard + hardening."""
    keys = [k for k in CONFIG_ORDER if k in configs]
    fig = plt.figure(figsize=(13.5, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28,
                           height_ratios=[1, 1])

    # (a) ASR forest (configs)
    axa = fig.add_subplot(gs[0, 0])
    yk = list(reversed(keys))
    for i, k in enumerate(yk):
        a = configs[k]["asr"]; lo, hi = configs[k]["asr_ci"]
        axa.errorbar(a, i, xerr=[[max(0, a - lo)], [max(0, hi - a)]], fmt="o",
                     color=DM_COLORS[k], ecolor="#555", elinewidth=1.4, capsize=4, ms=8,
                     mec="#222")
        axa.text(min(1.02, hi + 0.03), i, f"{a:.2f} [{lo:.2f},{hi:.2f}]", va="center",
                 fontsize=8)
    axa.set_yticks(range(len(yk))); axa.set_yticklabels([CONFIG_LABELS[k] for k in yk])
    axa.set_xlim(-0.03, 1.25); axa.set_xlabel("ASR (95% CI)")
    axa.set_title("Ablation: ASR with confidence intervals")
    axa.grid(axis="y", visible=False); _despine(axa); _panel(axa, "a", dx=-0.32)

    # (b) per-class heatmap (configs)
    axb = fig.add_subplot(gs[0, 1])
    mat, classes = _class_matrix(configs, keys)
    im = _heatmap(axb, mat, [CONFIG_LABELS[k] for k in keys],
                  [CLASS_LABELS.get(c, c) for c in classes], fontsize=6.8)
    mh = [j for j, c in enumerate(classes) if c in MULTIHOP]
    if mh:
        axb.add_patch(plt.Rectangle((min(mh) - .5, -.5), len(mh), len(keys),
                      fill=False, edgecolor="#08306b", lw=2))
    fig.colorbar(im, ax=axb, fraction=0.03, pad=0.02).set_label("ASR", fontsize=8)
    axb.set_title("Per-attack-class ASR"); _panel(axb, "b", dx=-0.18)

    # (c) competitive leaderboard
    axc = fig.add_subplot(gs[1, 0])
    if comp:
        rows = sorted(_comp_rows(comp), key=lambda r: r[1]["asr"], reverse=True)
        names = [_short(n) for n, _ in rows]; asr = [m["asr"] for _, m in rows]
        lo = [a - m["asr_ci"][0] for a, (_, m) in zip(asr, rows)]
        hi = [m["asr_ci"][1] - a for a, (_, m) in zip(asr, rows)]
        cols = [DUALMIND_HL if "DUALMIND" in n else COMPETITOR_GRAY for n, _ in rows]
        axc.barh(names, asr, xerr=[np.maximum(lo, 0), np.maximum(hi, 0)], capsize=3,
                 color=cols, edgecolor="#222", lw=0.6, error_kw={"elinewidth": 1, "ecolor": "#555"})
        for i, v in enumerate(asr):
            axc.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)
        axc.set_xlim(0, 1.1); axc.set_xlabel("ASR (95% CI)")
        axc.set_title("DUALMIND vs competitors")
        axc.grid(axis="y", visible=False); _despine(axc)
    _panel(axc, "c", dx=-0.30)

    # (d) self-hardening
    axd = fig.add_subplot(gs[1, 1])
    if hard and hard.get("asr_per_round"):
        asr = hard["asr_per_round"]; x = list(range(len(asr)))
        axd.fill_between(x, asr, asr[0], color="#6a51a3", alpha=0.12)
        axd.plot(x, asr, "-o", color="#3b1f6b", lw=2.2, ms=6, mfc="white", mec="#3b1f6b", mew=1.5)
        axd.annotate(f"{asr[0]:.2f} -> {asr[-1]:.2f}", xy=(x[-1], asr[-1]),
                     xytext=(0.3, asr[0] - 0.06), fontsize=10, color="#3b1f6b", fontweight="bold")
        axd.set_xlabel("Red-team round"); axd.set_ylabel("ASR")
        axd.set_ylim(-0.03, max(0.25, asr[0] + 0.15))
        axd.set_title("Self-hardening curve"); _despine(axd)
    _panel(axd, "d", dx=-0.18)

    fig.suptitle("DUALMIND — main results", fontsize=16, fontweight="bold", y=0.99)
    _save(fig, "16_main_results.png")


def g17_forest(configs, comp):
    """Forest plot of ASR across ALL configs + competitors with 95% CIs."""
    items = [(CONFIG_LABELS[k], configs[k], DM_COLORS[k], "ablation")
             for k in CONFIG_ORDER if k in configs]
    if comp:
        for n, m in _comp_rows(comp):
            if "DUALMIND" in n:
                continue
            items.append((_short(n), m, COMPETITOR_GRAY, "baseline"))
    items = [it for it in items if "asr_ci" in it[1]]
    items.sort(key=lambda it: it[1]["asr"])
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(items) + 2))
    for i, (name, m, color, kind) in enumerate(items):
        a = m["asr"]; lo, hi = m["asr_ci"]
        ax.errorbar(a, i, xerr=[[max(0, a - lo)], [max(0, hi - a)]], fmt="s" if kind == "baseline" else "o",
                    color=color, ecolor="#666", elinewidth=1.3, capsize=4, ms=8, mec="#222")
        ax.text(1.02, i, f"{a:.2f}  [{lo:.2f}, {hi:.2f}]", va="center", fontsize=8.5,
                family="monospace")
    ax.axvline(0, color="#1a9850", ls=":", lw=1)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, *_ in items])
    ax.set_xlim(-0.03, 1.35); ax.set_xlabel("Attack Success Rate (95% bootstrap CI)")
    ax.set_title("Forest plot: ASR with confidence intervals (all defenses)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", color="w", mfc="#555", label="DUALMIND ablation"),
                       plt.Line2D([], [], marker="s", color="w", mfc=COMPETITOR_GRAY, label="competitor")],
              loc="lower right")
    ax.grid(axis="y", visible=False); _despine(ax)
    _save(fig, "17_asr_forest.png")


def g18_bootstrap_dist(rows, comp):
    """Bootstrap ASR sampling distributions (the inference behind the CIs)."""
    series = []
    for k in CONFIG_ORDER:
        if rows.get(k):
            atk = [r for r in rows[k] if r["is_attack"]]
            if atk:
                series.append((CONFIG_LABELS[k], _bootstrap_dist(atk), DM_COLORS[k]))
    for cfg, lst in rows.items():
        if cfg.startswith("baseline:"):
            atk = [r for r in lst if r["is_attack"]]
            if atk:
                series.append((_short(cfg.split(":", 1)[1]), _bootstrap_dist(atk), COMPETITOR_GRAY))
    if not series:
        return
    series.sort(key=lambda s: np.mean(s[1]))
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(series) + 2))
    for i, (name, dist, color) in enumerate(series):
        # ridgeline-style KDE via histogram
        hist, edges = np.histogram(dist, bins=24, range=(0, 1), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        h = hist / (hist.max() or 1) * 0.8
        ax.fill_between(centers, i, i + h, color=color, alpha=0.7, lw=0.8, edgecolor="#222")
        m = np.mean(dist)
        ax.plot([m, m], [i, i + 0.85], color="#b2182b", lw=1.2)
    ax.set_yticks([i + 0.1 for i in range(len(series))])
    ax.set_yticklabels([n for n, *_ in series])
    ax.set_xlim(-0.02, 1.02); ax.set_xlabel("Bootstrap ASR (2000 resamples)")
    ax.set_title("Bootstrap sampling distributions of ASR (red line = mean)")
    ax.grid(axis="y", visible=False); _despine(ax)
    _save(fig, "18_bootstrap_distributions.png")


def g19_ablation_waterfall(configs):
    """Marginal ASR reduction contributed by each added system."""
    keys = [k for k in CONFIG_ORDER if k in configs]
    if len(keys) < 2:
        return
    asr = [configs[k]["asr"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    running = asr[0]
    ax.bar(0, asr[0], color=DM_COLORS[keys[0]], edgecolor="#222", width=0.62)
    ax.text(0, asr[0] + 0.02, f"{asr[0]:.2f}", ha="center", fontsize=9, fontweight="bold")
    for i in range(1, len(keys)):
        drop = asr[i - 1] - asr[i]
        bottom = asr[i]
        ax.bar(i, drop, bottom=bottom, color="#b2182b", alpha=0.55, edgecolor="#222",
               width=0.62, hatch="//")
        ax.bar(i, asr[i], color=DM_COLORS[keys[i]], edgecolor="#222", width=0.62)
        ax.text(i, asr[i - 1] + 0.02, f"-{drop:.2f}" if drop > 0 else "0",
                ha="center", fontsize=8.5, color="#b2182b")
        ax.plot([i - 1, i], [asr[i - 1], asr[i - 1]], color="#888", ls=":", lw=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([CONFIG_LABELS[k] for k in keys], rotation=18, ha="right")
    ax.set_ylabel("Attack Success Rate"); ax.set_ylim(0, 1.1)
    ax.set_title("Ablation waterfall: marginal ASR reduction per added system")
    ax.legend(handles=[Patch(facecolor="#b2182b", alpha=0.55, hatch="//", label="ASR removed by this layer")],
              loc="upper right")
    ax.grid(axis="x", visible=False); _despine(ax)
    _save(fig, "19_ablation_waterfall.png")


def g20_roc_ci(rows, configs):
    """ROC curves with bootstrap confidence bands for key configs."""
    try:
        from sklearn.metrics import roc_curve
    except Exception:
        return
    grid = np.linspace(0, 1, 50)
    pick = [k for k in ("2_prefilter_only", "4_dual_llm_taint", "6_full_post_hardening")
            if rows.get(k)]
    if not pick:
        return
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    rng = random.Random(7)
    for k in pick:
        data = rows[k]
        labels = [1 if r["is_attack"] else 0 for r in data]
        if len(set(labels)) < 2:
            continue
        curves = []
        for _ in range(300):
            idx = [rng.randrange(len(data)) for _ in range(len(data))]
            yl = [labels[i] for i in idx]; ys = [data[i]["score"] for i in idx]
            if len(set(yl)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yl, ys)
            curves.append(np.interp(grid, fpr, tpr))
        if not curves:
            continue
        curves = np.array(curves)
        mean = curves.mean(0); lo = np.percentile(curves, 2.5, 0); hi = np.percentile(curves, 97.5, 0)
        color = DM_COLORS[k]
        ax.plot(grid, mean, color=color, lw=2.0,
                label=f"{CONFIG_LABELS[k]} (AUC={configs[k]['roc']['auc']:.2f})")
        ax.fill_between(grid, lo, hi, color=color, alpha=0.18)
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_title("ROC with 95% bootstrap confidence bands")
    ax.legend(loc="lower right"); _despine(ax)
    _save(fig, "20_roc_with_ci.png")


def g21_per_class_grouped(configs):
    """Grouped bars: per-attack-class ASR across configs, multi-hop region shaded."""
    keys = [k for k in CONFIG_ORDER if k in configs]
    mat, classes = _class_matrix(configs, keys)
    if not classes:
        return
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    x = np.arange(len(classes)); w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        ax.bar(x + (i - len(keys)/2) * w + w/2, mat[i], width=w, color=DM_COLORS[k],
               edgecolor="#222", linewidth=0.4, label=CONFIG_LABELS[k])
    for j, c in enumerate(classes):
        if c in MULTIHOP:
            ax.axvspan(j - 0.5, j + 0.5, color="#08306b", alpha=0.06)
    ax.set_xticks(x); ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in classes],
                                         rotation=25, ha="right")
    ax.set_ylabel("Attack Success Rate"); ax.set_ylim(0, 1.08)
    ax.set_title("Per-attack-class ASR by configuration (multi-hop region shaded)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="x", visible=False); _despine(ax)
    _save(fig, "21_per_class_grouped.png")


def g22_dualmind_calibration(configs):
    """DUALMIND reliability before (dashed) vs after (solid) the System 8 layer."""
    pre, post = configs.get("6_full_post_hardening"), configs.get("7_full_calibrated")
    if not (pre and post and pre.get("reliability") and post.get("reliability")):
        return
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1.0, label="perfect calibration")
    for m, style, color, lbl in [(pre, "--", "#b2182b", "uncalibrated"),
                                 (post, "-", "#1a9850", "calibrated (System 8)")]:
        rel = m["reliability"]
        pts = [(c, a) for c, a, n in zip(rel["confidence"], rel["accuracy"], rel["counts"]) if n > 0]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, style, marker="o", color=color, lw=2.2, ms=6,
                    label=f"{lbl}  (ECE={m['ece']:.3f})")
    ax.set_xlabel("Predicted probability (confidence)")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_title("DUALMIND calibration: before vs after System 8")
    ax.legend(loc="upper left"); _despine(ax)
    _save(fig, "22_dualmind_calibration.png")


# --------------------------------------------------------------------------- #
def build_inlined_dashboard(results_path):
    template = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    if not os.path.exists(template):
        return
    results, comp, _, _ = _load(results_path)
    with open(template, "r", encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps({"results": results, "competitive": comp}, default=str)
    html = html.replace("/*__DUALMIND_DATA__*/", f"const DUALMIND_DATA = {payload};")
    with open(os.path.join(RESULTS_DIR, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  wrote dashboard.html (self-contained, {len(html)//1024} KB)")


def generate_all(results_path):
    _apply_style()
    results, comp, hard, rows = _load(results_path)
    configs = results.get("configs", {})
    print(f"Generating figures -> {GRAPH_DIR}")
    graphs = [
        ("16 main results", lambda: g16_main_results(configs, comp, hard)),
        ("01 asr bars", lambda: g1_asr_by_config(configs)),
        ("02 class heatmap", lambda: g2_class_heatmap(configs)),
        ("03 self-hardening", lambda: g3_self_hardening(hard)),
        ("04 roc", lambda: g4_roc(configs, rows)),
        ("05 pr", lambda: g5_pr(configs)),
        ("06 calibration", lambda: g6_calibration(configs, comp)),
        ("07 latency", lambda: g7_latency(configs, rows)),
        ("08 cost-security", lambda: g8_cost_security(configs)),
        ("09 confusion", lambda: g9_confusion_grid(configs)),
        ("10 fpr-asr", lambda: g10_fpr_asr(configs)),
        ("17 forest", lambda: g17_forest(configs, comp)),
        ("18 bootstrap dist", lambda: g18_bootstrap_dist(rows, comp)),
        ("19 ablation waterfall", lambda: g19_ablation_waterfall(configs)),
        ("20 roc ci", lambda: g20_roc_ci(rows, configs)),
        ("21 per-class grouped", lambda: g21_per_class_grouped(configs)),
        ("22 dualmind calibration", lambda: g22_dualmind_calibration(configs)),
    ]
    if comp:
        graphs += [
            ("11 leaderboard", lambda: g11_leaderboard(comp)),
            ("12 security map", lambda: g12_security_map(comp)),
            ("13 competitive heatmap", lambda: g13_competitive_heatmap(comp)),
            ("14 radar", lambda: g14_radar(comp)),
            ("15 latency-accuracy", lambda: g15_latency_accuracy(comp)),
        ]
    for label, fn in graphs:
        try:
            fn()
        except Exception as e:
            print(f"  [warn] figure '{label}' failed: {e}")
    try:
        build_inlined_dashboard(results_path)
    except Exception as e:
        print(f"  [warn] dashboard build failed: {e}")
    print("Done.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate DUALMIND figures from results")
    ap.add_argument("--from-results", default=os.path.join(RESULTS_DIR, "results.json"))
    args = ap.parse_args(argv)
    generate_all(args.from_results)


if __name__ == "__main__":
    main()
