"""Experiment 4 (pair competence vs Pile co-occurrence) and Experiment 3b (loss-indexed
scaling of pair vs single competence), as fixed in PREREGISTRATION.md sections 3-4.

python exp4/analyze.py   -> results/exp4_summary.json
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "exp4_summary.json"
COUNTS = json.loads((ROOT / "results" / "exp4_corpus_counts.json").read_text())
SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]
RNG = np.random.default_rng(20261118)


def load(size):
    p = RAW / f"EleutherAI__pythia-{size}__step143000.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if "pairs" in d and "loss" in d else None


def pair_table(d, count_key="pair_mcf500k"):
    by = defaultdict(lambda: defaultdict(list))
    for r in d["pairs"]:
        kind, name = r["task"].split(":", 1)
        lp = np.array(r["lps"])
        p = np.exp(lp - lp.max())
        by[name][kind].append((r["x"], float(int(np.argmax(lp)) == r["label"]), p[r["label"]] / p.sum(),
                               r["tags"][int(np.argmax(lp))]))
    rows = []
    for name, kinds in by.items():
        outer, inner = name.split("|")
        pair = kinds["pair"]
        inner_acc = {x: a for x, a, _, _ in kinds["inner"]}
        outer_acc = {x: a for x, a, _, _ in kinds["outer"]}
        rows.append({
            "pair": name, "outer": outer, "inner": inner, "n": len(pair),
            "acc": float(np.mean([a for _, a, _, _ in pair])),
            "cont": float(np.mean([c for _, _, c, _ in pair])),
            "c_inner": float(np.mean(list(inner_acc.values()))),
            "c_outer": float(np.mean(list(outer_acc.values()))),
            "items": [{"x": x, "pass": a, "inner_ok": inner_acc.get(x), "outer_ok": outer_acc.get(x),
                       "chosen": tag} for x, a, _, tag in pair],
            "f12": COUNTS[count_key][name]["count"],
            "f1": COUNTS["single"][outer]["count"], "f2": COUNTS["single"][inner]["count"],
        })
    rows.sort(key=lambda r: r["pair"])
    return rows


def design(rows, with_f12=True):
    X = []
    for r in rows:
        x = [math.log10(r["f1"]), math.log10(r["f2"]), r["c_inner"], r["c_outer"]]
        if with_f12:
            x = [math.log10(r["f12"] + 1)] + x
        X.append(x)
    return sm.add_constant(np.array(X), has_constant="add")


def glm_pair(rows):
    y = np.array([[r["acc"] * r["n"], (1 - r["acc"]) * r["n"]] for r in rows])
    X = design(rows)
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    names = ["const", "log_f12", "log_f_outer", "log_f_inner", "c_inner", "c_outer"]
    return {"params": dict(zip(names, map(float, m.params))), "bse": dict(zip(names, map(float, m.bse))),
            "pvalues": dict(zip(names, map(float, m.pvalues))),
            "beta1": float(m.params[1]), "beta1_ci": [float(v) for v in m.conf_int()[1]],
            "beta1_p": float(m.pvalues[1]), "beta1_p_one_sided_pos": float(m.pvalues[1] / 2 if m.params[1] > 0 else 1 - m.pvalues[1] / 2)}


def logit_items(rows):
    y, X, groups = [], [], []
    for gi, r in enumerate(rows):
        for it in r["items"]:
            y.append(it["pass"])
            X.append([math.log10(r["f12"] + 1), math.log10(r["f1"]), math.log10(r["f2"]),
                      float(it["inner_ok"] or 0), float(it["outer_ok"] or 0)])
            groups.append(gi)
    X = sm.add_constant(np.array(X), has_constant="add")
    try:
        m = sm.Logit(np.array(y), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": np.array(groups)})
        return {"beta1": float(m.params[1]), "beta1_ci": [float(v) for v in m.conf_int()[1]],
                "beta1_p": float(m.pvalues[1])}
    except Exception as e:  # perfect separation etc.
        return {"error": str(e)}


def tercile_test(rows, n_boot=5000):
    y = np.array([[r["acc"] * r["n"], (1 - r["acc"]) * r["n"]] for r in rows])
    m = sm.GLM(y, design(rows, with_f12=False), family=sm.families.Binomial()).fit()
    resid = np.array([r["acc"] for r in rows]) - m.fittedvalues
    f = np.array([r["f12"] for r in rows])
    lo_cut, hi_cut = np.quantile(f, [1 / 3, 2 / 3])
    lo, hi = resid[f <= lo_cut], resid[f >= hi_cut]
    diffs = [RNG.choice(lo, len(lo)).mean() - RNG.choice(hi, len(hi)).mean() for _ in range(n_boot)]
    return {"low_minus_high": float(lo.mean() - hi.mean()), "ci95": [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))],
            "n_low": int(len(lo)), "n_high": int(len(hi)), "cuts": [float(lo_cut), float(hi_cut)]}


def error_attribution(rows):
    c = defaultdict(int)
    for r in rows:
        for it in r["items"]:
            if not it["pass"]:
                c[it["chosen"]] += 1
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()} if tot else {}


def exp3b(models):
    N = np.array([m["n_params"] for m in models], float)
    L = np.array([m["L"] for m in models])

    def f(n, E, A, a):
        return E + A * n ** (-a)
    try:
        (E, A, a), _ = optimize.curve_fit(f, N, L, p0=[1.8, 30.0, 0.15], maxfev=20000)
    except Exception:
        E, A, a = float("nan"), float("nan"), float("nan")
    out = {"fit": {"E": float(E), "A": float(A), "a": float(a)}, "tests": {}}
    for label, e in (("fit", E), ("E=1.6", 1.6), ("E=1.8", 1.8)):
        rows = []
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                d1, d2 = m1["L"] - e, m2["L"] - e
                if d2 <= 0 or d1 <= 0 or not (d2 <= d1 / 2):
                    continue
                rows.append({"M1": m1["size"], "M2": m2["size"], "delta1": d1, "delta2": d2,
                             "cc_single_M1": m1["cc_single"], "cc_pair_M2": m2["cc_pair"],
                             "prediction_holds": m2["cc_pair"] >= m1["cc_single"]})
        out["tests"][label] = {"pairs": rows, "n": len(rows),
                               "frac_holds": (float(np.mean([r["prediction_holds"] for r in rows])) if rows else None),
                               "refuted(majority fail)": (np.mean([r["prediction_holds"] for r in rows]) < 0.5) if rows else None}
    return out


def main():
    res = {"per_size": {}, "beta1_by_size": []}
    models = []
    for size in SIZES:
        d = load(size)
        if d is None:
            continue
        rows = pair_table(d)
        rows_def = pair_table(d, "pair")
        g = glm_pair(rows)
        cc = lambda a: (a - 0.25) / 0.75  # noqa: E731
        summary = {
            "n_params": d["n_params"], "L": d["loss"]["pile_val_nll"],
            "mean_pair_acc": float(np.mean([r["acc"] for r in rows])),
            "mean_inner_acc": float(np.mean([r["c_inner"] for r in rows])),
            "mean_outer_acc": float(np.mean([r["c_outer"] for r in rows])),
            "glm": g, "glm_default_counts": glm_pair(rows_def), "item_logit": logit_items(rows),
            "tercile": tercile_test(rows), "error_attribution": error_attribution(rows),
            "spearman_acc_logf12": float(stats.spearmanr([r["acc"] for r in rows], [r["f12"] for r in rows]).statistic),
            "rows": [{k: v for k, v in r.items() if k != "items"} for r in rows],
        }
        res["per_size"][size] = summary
        res["beta1_by_size"].append({"size": size, "n_params": d["n_params"], "beta1": g["beta1"],
                                     "ci": g["beta1_ci"], "p": g["beta1_p"]})
        models.append({"size": size, "n_params": d["n_params"], "L": summary["L"],
                       "cc_pair": cc(summary["mean_pair_acc"]),
                       "cc_single": cc((summary["mean_inner_acc"] + summary["mean_outer_acc"]) / 2)})
        print(f"{size:>5s} L={summary['L']:.3f} pair={summary['mean_pair_acc']:.3f} inner={summary['mean_inner_acc']:.3f} "
              f"outer={summary['mean_outer_acc']:.3f} beta1={g['beta1']:+.3f} [{g['beta1_ci'][0]:+.3f},{g['beta1_ci'][1]:+.3f}] "
              f"p={g['beta1_p']:.3f} rho(acc,f12)={summary['spearman_acc_logf12']:+.2f} "
              f"tercile low-high={summary['tercile']['low_minus_high']:+.3f} {summary['tercile']['ci95']}")
    if len(res["beta1_by_size"]) >= 3:
        x = np.log10([b["n_params"] for b in res["beta1_by_size"]])
        y = [b["beta1"] for b in res["beta1_by_size"]]
        sl = stats.linregress(x, y)
        res["beta1_slope_vs_log10_params"] = {"slope": float(sl.slope), "p": float(sl.pvalue)}
    largest = res["beta1_by_size"][-1] if res["beta1_by_size"] else None
    if largest:
        big = res["per_size"][largest["size"]]
        res["verdict"] = {
            "largest_size": largest["size"],
            "R4a_beta1_pos_sig": bool(largest["beta1"] > 0 and big["glm"]["beta1_p"] < 0.05),
            "R4b_beta1_not_shrinking": bool(res.get("beta1_slope_vs_log10_params", {}).get("slope", -1) >= 0),
            "R4c_low_below_high": bool(big["tercile"]["ci95"][1] < 0),
        }
    if len(models) >= 4:
        res["exp3b"] = exp3b(models)
        print(json.dumps(res["exp3b"], indent=1, default=float))
    print("verdict", res.get("verdict"), res.get("beta1_slope_vs_log10_params"))
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))


if __name__ == "__main__":
    main()
