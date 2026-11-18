"""Exploratory robustness checks for the Exp 4 co-occurrence effect (beta1). Added after the
first Exp 4 numbers were seen; NOT pre-registered. The pre-registered statistics are in
exp4/analyze.py.

1. Quasi-binomial GLM: the pair-level binomial GLM assumes binomial item noise only, so
   between-pair variation inflates its significance. Scale by Pearson chi2 / df.
2. Freedman-Lane permutation test: residualise log10(f12+1) on the controls, permute the
   residuals across pairs, refit; two-sided p for beta1.
3. Leave-one-skill-out: drop every pair that uses skill s (as inner or outer) and refit.
4. Unconditional association: Spearman rho(pair accuracy, f12) and rho(f12, f1*f2).

python exp4/robustness.py  -> results/exp4_robustness.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp4"))
from analyze import SIZES, design, load, pair_table  # noqa: E402

OUT = ROOT / "results" / "exp4_robustness.json"
RNG = np.random.default_rng(20261118)


def endog(rows):
    return np.array([[r["acc"] * r["n"], (1 - r["acc"]) * r["n"]] for r in rows])


def fit(rows, X=None):
    X = design(rows) if X is None else X
    return sm.GLM(endog(rows), X, family=sm.families.Binomial()).fit()


def quasi(m):
    """Quasi-binomial inference for beta1. Scale = Pearson chi2 / df from the binomial fit
    (statsmodels' fit(scale="X2") mis-scales two-column binomial endog, so do it by hand)."""
    phi = float(m.pearson_chi2 / m.df_resid)
    se = float(m.bse[1]) * math.sqrt(phi)
    b = float(m.params[1])
    return {"beta1": b, "se": se, "ci": [b - 1.96 * se, b + 1.96 * se],
            "p": float(2 * stats.norm.sf(abs(b) / se)), "scale": phi}


def freedman_lane(rows, n_perm=5000):
    X = design(rows)
    obs = fit(rows, X).params[1]
    Z = np.delete(X, 1, axis=1)  # const + controls
    x = X[:, 1]
    beta = np.linalg.lstsq(Z, x, rcond=None)[0]
    xhat, res = Z @ beta, x - Z @ beta
    null = []
    for _ in range(n_perm):
        Xp = X.copy()
        Xp[:, 1] = xhat + RNG.permutation(res)
        try:
            null.append(fit(rows, Xp).params[1])
        except Exception:
            continue
    null = np.array(null)
    return {"beta1": float(obs), "p_two_sided": float((np.abs(null) >= abs(obs)).mean()),
            "n_perm": int(len(null))}


def leave_one_skill_out(rows):
    skills = sorted({r["outer"] for r in rows} | {r["inner"] for r in rows})
    out = {}
    for s in skills:
        keep = [r for r in rows if s not in (r["outer"], r["inner"])]
        if len(keep) < 12:
            continue
        q = quasi(fit(keep))
        out[s] = {"n_pairs": len(keep), "beta1": q["beta1"], "p_quasi": q["p"]}
    b = [v["beta1"] for v in out.values()]
    return {"by_skill": out, "beta1_min": float(min(b)), "beta1_max": float(max(b)),
            "n_dropped_sign_flip": int(sum(v < 0 for v in b)),
            "n_dropped_quasi_p_ge_0.05": int(sum(v["p_quasi"] >= 0.05 for v in out.values()))}


def main():
    res = {}
    for size in SIZES:
        d = load(size)
        if d is None:
            continue
        rows = pair_table(d)
        q = quasi(fit(rows))
        lf12 = [math.log10(r["f12"] + 1) for r in rows]
        res[size] = {
            "quasi_binomial": q,
            "freedman_lane": freedman_lane(rows),
            "leave_one_skill_out": leave_one_skill_out(rows),
            "spearman_acc_f12": float(stats.spearmanr([r["acc"] for r in rows], lf12).statistic),
            "spearman_f12_f1f2": float(stats.spearmanr(lf12, [math.log10(r["f1"] * r["f2"]) for r in rows]).statistic),
        }
        r = res[size]
        print(f"{size:>5s} quasi beta1={r['quasi_binomial']['beta1']:+.3f} {['%+.3f' % v for v in r['quasi_binomial']['ci']]} "
              f"p={r['quasi_binomial']['p']:.3f} scale={r['quasi_binomial']['scale']:.2f} | perm p={r['freedman_lane']['p_two_sided']:.4f} "
              f"| LOSO beta1 [{r['leave_one_skill_out']['beta1_min']:+.2f}, {r['leave_one_skill_out']['beta1_max']:+.2f}] "
              f"n.s. after dropping {r['leave_one_skill_out']['n_dropped_quasi_p_ge_0.05']} of {len(r['leave_one_skill_out']['by_skill'])} skills "
              f"| rho(acc,f12)={r['spearman_acc_f12']:+.2f} rho(f12,f1f2)={r['spearman_f12_f1f2']:+.2f}")
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
