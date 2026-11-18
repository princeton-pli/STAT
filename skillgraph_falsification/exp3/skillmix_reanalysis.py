"""Experiment 3 (training-free, generation-free): re-analysis of the published Skill-Mix
results (Yu et al., 2023, arXiv:2310.17567, Tables 1 and 2) for the LLaMA-2-Chat family.

Pre-registered: k* = largest k with p(k) >= tau, linearly interpolated; rate =
log2(k*(70B)/k*(7B)) per decade of compute; prediction >= 1 (doubling), refuted if < 0.5.

Notes on missing cells. "-" in the paper means the cell was skipped because the metric had
already fallen below a cut-off (0.05 for the GPT-4 grader, 0.3 for the LLaMA-2 grader) at a
smaller k; we then bound p(k) in [0, p(k-1)). k=1 was not run: when p(2) < tau we interpolate
from an assumed p(1)=1, which gives an UPPER bound on k*.

Secondary (not in the pre-registration's detail): LLaMA-2 7B and 70B were both trained on
2T tokens, so 10x compute was not Chinchilla-optimal scaling. Using the Chinchilla fit
(Hoffmann et al. 2022: E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28) we compute the
excess-loss ratio this 10x actually buys, and the k* ratio the paper predicts from it
(Corollary 13: excess loss / r => k' x r at the same competence).
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "exp3_skillmix.json"

# (Ratio of Full Marks, Ratio of All Skills) by k = 2..6; None = skipped ("-")
T1_GPT4 = {  # Table 1, graded by GPT-4
    "LLaMA-2-7B-Chat": {2: (.05, .11), 3: (.00, .01), 4: (.00, .00)},
    "LLaMA-2-13B-Chat": {2: (.22, .35), 3: (.05, .05), 4: (.02, .02)},
    "LLaMA-2-70B-Chat": {2: (.28, .28), 3: (.02, .05), 4: (.00, .01)},
    "GPT-3.5-turbo": {2: (.53, .60), 3: (.20, .25), 4: (.08, .13), 5: (.04, .16)},
    "GPT-4": {2: (.94, .97), 3: (.68, .70), 4: (.52, .55), 5: (.36, .38), 6: (.29, .29)},
}
T2_LLAMA = {  # Table 2, graded by LLaMA-2-70B-Chat
    "LLaMA-2-7B-Chat": {2: (.53, .63), 3: (.33, .50), 4: (.13, .13)},
    "LLaMA-2-13B-Chat": {2: (.63, .87), 3: (.33, .63), 4: (.33, .43), 5: (.13, .17)},
    "LLaMA-2-70B-Chat": {2: (.90, .97), 3: (.47, .50), 4: (.40, .47), 5: (.13, .23)},
    "GPT-3.5-turbo": {2: (.77, .80), 3: (.53, .53), 4: (.33, .33), 5: (.20, .40)},
    "GPT-4": {2: (.90, .93), 3: (.87, .87), 4: (.57, .60), 5: (.63, .67), 6: (.27, .33)},
}
PARAMS = {"LLaMA-2-7B-Chat": 7e9, "LLaMA-2-13B-Chat": 13e9, "LLaMA-2-70B-Chat": 70e9}
TOKENS = 2e12


def kstar(p, tau):
    """Returns (estimate, lower, upper, note)."""
    ks = sorted(p)
    if p[ks[0]] < tau:  # below threshold already at k=2: interpolate from assumed p(1)=1
        est = 1 + (1 - tau) / (1 - p[ks[0]])
        return est, 1.0, est, "p(2)<tau: assumed p(1)=1 (upper bound)"
    for k in ks:
        nxt = k + 1
        if p[k] >= tau and (nxt not in p or p[nxt] < tau):
            if nxt in p:
                est = k + (p[k] - tau) / (p[k] - p[nxt])
                return est, est, est, ""
            # next cell skipped (or beyond k=6): p(k+1) in [0, p(k)) -> k* in [k + (p-tau)/p, k+1)
            lo = k + (p[k] - tau) / p[k]
            return lo, lo, k + 1.0, f"p({nxt}) not reported: bounded"
    return float("nan"), float("nan"), float("nan"), "no crossing"


def chinchilla_excess(N, D, E=1.69, A=406.4, B=410.7, a=0.34, b=0.28):
    return A / N ** a + B / D ** b


def main():
    res = {"rows": [], "rates": []}
    for grader, table in (("GPT-4", T1_GPT4), ("LLaMA-2-70B", T2_LLAMA)):
        for mi, metric in enumerate(("full_marks", "all_skills")):
            for tau in (0.5, 0.25, 0.1):
                ks = {}
                for model, cells in table.items():
                    p = {k: v[mi] for k, v in cells.items()}
                    est, lo, hi, note = kstar(p, tau)
                    ks[model] = (est, lo, hi)
                    res["rows"].append({"grader": grader, "metric": metric, "tau": tau, "model": model,
                                        "kstar": est, "lo": lo, "hi": hi, "note": note})
                a, b = ks["LLaMA-2-7B-Chat"], ks["LLaMA-2-70B-Chat"]
                rate = math.log2(b[0] / a[0])
                rate_lo = math.log2(b[1] / a[2]) if a[2] > 0 else float("nan")
                rate_hi = math.log2(b[2] / a[1]) if a[1] > 0 else float("nan")
                res["rates"].append({"grader": grader, "metric": metric, "tau": tau,
                                     "k7": a[0], "k70": b[0], "ratio": b[0] / a[0],
                                     "rate_per_decade": rate, "rate_lo": rate_lo, "rate_hi": rate_hi,
                                     "refutes_doubling(<0.5)": rate_hi < 0.5,
                                     "point_below_0.5": rate < 0.5})
    d7, d70 = chinchilla_excess(7e9, TOKENS), chinchilla_excess(70e9, TOKENS)
    res["chinchilla"] = {"excess_7B": d7, "excess_70B": d70, "excess_ratio": d7 / d70,
                         "predicted_kstar_ratio_loss_indexed": d7 / d70,
                         "predicted_rate_loss_indexed": math.log2(d7 / d70),
                         "compute_optimal_10x_excess_ratio":
                             chinchilla_excess(7e9, 140e9) / chinchilla_excess(7e9 * 10 ** 0.5, 140e9 * 10 ** 0.5)}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"{'grader':12s} {'metric':10s} {'tau':>4s} {'k*7B':>6s} {'k*70B':>6s} {'ratio':>6s} {'rate':>6s}  [lo, hi]")
    for r in res["rates"]:
        print(f"{r['grader']:12s} {r['metric']:10s} {r['tau']:>4} {r['k7']:6.2f} {r['k70']:6.2f} {r['ratio']:6.2f} "
              f"{r['rate_per_decade']:6.2f}  [{r['rate_lo']:.2f}, {r['rate_hi']:.2f}]")
    print(json.dumps(res["chinchilla"], indent=1))


if __name__ == "__main__":
    main()
