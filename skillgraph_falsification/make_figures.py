"""Figures for REPORT.md (static PNG, light theme, validated categorical palette).
python make_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titlecolor": INK, "axes.titleweight": "bold", "lines.linewidth": 2,
    "legend.frameon": False, "legend.labelcolor": INK2,
})
MODEL_ORDER = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
               "EleutherAI/pythia-1b", "allenai/OLMo-1B-hf"]
NICE = {m: m.split("/")[1].replace("pythia-", "Pythia-").replace("-hf", "") for m in MODEL_ORDER}
COLOR = {m: SERIES[i] for i, m in enumerate(MODEL_ORDER)}


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)


def fig_formal():
    d = json.loads((R / "formal_zipf_check.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    for ax, s in zip(axes, (1.0, 1.5)):
        rows = [r for r in d["zipf_exact"] if r["zipf_s"] == s and r["k"] == 4 and r["theta"] == 0.1]
        n = [r["n"] for r in rows]
        ax.plot(n, [r["count_frac_zero_competence"] for r in rows], color=SERIES[0], marker="o", ms=5,
                label="share of skills, by count")
        ax.plot(n, [r["measure_zero_competence"] for r in rows], color=SERIES[1], marker="o", ms=5,
                label="share of skill measure µ1")
        ax.plot(n, [r["thm14_tightest_alpha"] for r in rows], color=MUTED, ls="--", lw=1.5,
                label="max measure Theorem 14 allows")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("number of skills n")
        ax.set_title(f"Zipf exponent s = {s:g}  (k = 4, P(Y) = 0.1)")
    axes[0].set_ylabel("skills with zero competence")
    axes[0].legend(loc="center left", fontsize=8)
    save(fig, "formal_zipf.png")


def fig_exp2():
    p = R / "exp2_summary.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    models = [m for m in MODEL_ORDER if m in d]
    # (a) Gini of per-skill error vs L; (b) share of skills flat/declining per window
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for m in models:
        ck = d[m]["checkpoints"]
        L = [c["L"] for c in ck]
        axes[0].plot(L, [c["gini_err"] for c in ck], color=COLOR[m], marker="o", ms=3.5, label=NICE[m])
        W = d[m]["windows"]
        if W:
            mid = [(w["L_from"] + w["L_to"]) / 2 for w in W]
            axes[1].plot(mid, [w["F_pi0"] for w in W], color=COLOR[m], marker="o", ms=4, label=NICE[m])
            axes[1].plot(mid, [w["F_cons"] for w in W], color=COLOR[m], marker="o", ms=3.5, ls=":", lw=1.5)
    for ax in axes:
        ax.invert_xaxis()
        ax.set_xlabel("Pile-val loss L (nats/token)  →  lower")
    axes[0].set_ylabel("Gini of per-skill error rate")
    axes[0].set_title("Error concentration across 85 skills")
    axes[1].axhline(0.2, color=MUTED, lw=1, ls="--")
    axes[1].set_ylabel("share of skills flat or declining")
    axes[1].set_title("Per 0.1-nat window: π̂0 (solid), conservative (dotted)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, "exp2_concentration_windows.png")

    # (c) per-skill curves for the largest Pythia in the sweep
    for m in ("EleutherAI/pythia-1b", "EleutherAI/pythia-410m", "allenai/OLMo-1B-hf"):
        if m not in d:
            continue
        ck = d[m]["checkpoints"]
        L = [c["L"] for c in ck]
        skills = sorted(ck[-1]["acc"])
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharex=True)
        for ax, pre, chance, title in ((axes[0], "blimp:", 0.5, "BLiMP paradigms (67)"),
                                       (axes[1], "lex:", 0.25, "Lexical skills (18)")):
            ss = [s for s in skills if s.startswith(pre)]
            for s in ss:
                ax.plot(L, [c["acc"][s] for c in ck], color=SERIES[0], alpha=0.22, lw=1)
            ax.plot(L, [np.mean([c["acc"][s] for s in ss]) for c in ck], color=INK, lw=2.2)
            ax.axhline(chance, color=MUTED, lw=1, ls="--")
            ax.set_title(f"{NICE[m]}: {title}")
            ax.set_ylim(0, 1.02)
            ax.invert_xaxis() if ax is axes[0] else None
            ax.set_xlabel("Pile-val loss L  →  lower")
        axes[0].set_ylabel("accuracy per skill (mean in black)")
        save(fig, f"exp2_skill_curves_{NICE[m]}.png")


def fig_exp4():
    p = R / "exp4_summary.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    sizes = list(d["per_size"])
    if not sizes:
        return
    big = d["per_size"][sizes[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    x = [math.log10(r["f12"] + 1) for r in big["rows"]]
    y = [r["acc"] for r in big["rows"]]
    axes[0].scatter(x, y, s=26, color=SERIES[0], edgecolor=SURF, linewidth=1.2, zorder=3)
    axes[0].axhline(0.25, color=MUTED, lw=1, ls="--")
    axes[0].set_xlabel("log10(1 + Pile co-occurrence of the two skills' cues)")
    axes[0].set_ylabel("pair accuracy (4-way cloze)")
    axes[0].set_title(f"Pythia-{sizes[-1]}: 35 skill pairs")
    axes[0].set_ylim(0, 1.02)
    b = d["beta1_by_size"]
    xs = np.arange(len(b))
    axes[1].errorbar(xs, [r["beta1"] for r in b], yerr=[[r["beta1"] - r["ci"][0] for r in b], [r["ci"][1] - r["beta1"] for r in b]],
                     color=SERIES[0], marker="o", ms=5, lw=0, elinewidth=1.5, capsize=3)
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].set_xticks(xs, [f"Pythia-{r['size']}" for r in b], rotation=30, ha="right")
    axes[1].set_ylabel("β1: log-odds per 10x co-occurrence")
    axes[1].set_title("Co-occurrence effect after controls (95% CI)")
    save(fig, "exp4_cooccurrence.png")


def fig_exp3():
    d = json.loads((R / "exp3_skillmix.json").read_text())
    rows = [r for r in d["rates"] if r["metric"] == "full_marks"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    xs = np.array([0, 1, 2, 3.6, 4.6, 5.6])
    ax.bar(xs, [r["ratio"] for r in rows], width=0.6, color=SERIES[0], edgecolor=SURF, linewidth=2)
    for xx, r in zip(xs, rows):
        ax.annotate(f"{r['ratio']:.2f}", (xx, r["ratio"]), xytext=(0, -13), textcoords="offset points",
                    ha="center", fontsize=8.5, color="#ffffff", fontweight="bold")
    pred = d["chinchilla"]["predicted_kstar_ratio_loss_indexed"]
    ax.axhline(2.0, color=SERIES[1], lw=1.5, ls="--")
    ax.axhline(pred, color=SERIES[2], lw=1.5, ls="--")
    ax.annotate("doc's reading: x2 per 10x compute", (-0.4, 2.05), ha="left", fontsize=8, color=INK2)
    ax.annotate(f"paper, indexed by excess loss: x{pred:.2f}", (-0.4, pred + 0.08), ha="left", fontsize=8, color=INK2)
    ax.set_xticks(xs, [f"τ = {r['tau']}" for r in rows], fontsize=8.5)
    ax.text(1, -0.14, "GPT-4 grader", ha="center", fontsize=9, color=INK2, transform=ax.get_xaxis_transform())
    ax.text(4.6, -0.14, "LLaMA-2-70B grader", ha="center", fontsize=9, color=INK2, transform=ax.get_xaxis_transform())
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("k*(70B) / k*(7B)")
    ax.set_ylim(0, 2.3)
    ax.set_title("Skill-Mix, LLaMA-2-Chat 7B -> 70B (published scores, Ratio of Full Marks)")
    save(fig, "exp3_skillmix_ratio.png")


if __name__ == "__main__":
    fig_formal()
    fig_exp3()
    fig_exp2()
    fig_exp4()
