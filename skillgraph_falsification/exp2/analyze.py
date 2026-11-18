"""Experiment 2 analysis: per-skill competence vs Pile-val loss across checkpoints.
Implements exactly the statistics fixed in PREREGISTRATION.md section 2.

python exp2/analyze.py            -> results/exp2_summary.json (+ per-window tables)
"""
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "exp2_summary.json"
RNG = np.random.default_rng(20261118)
DELTA_L = 0.10
FLAT = 0.02


def step_of(rev):
    m = re.match(r"step(\d+)", rev)
    return int(m.group(1)) if m else 10 ** 9


def load_model(repo):
    ck = []
    for f in glob.glob(str(RAW / f"{repo.replace('/', '__')}__*.json")):
        d = json.loads(Path(f).read_text())
        if not all(k in d for k in ("loss", "blimp", "lex")):
            continue
        ck.append(d)
    ck.sort(key=lambda d: step_of(d["revision"]))
    return ck


def skill_items(d):
    """Return {skill: (acc_vector, cont_vector, chance)} with items in a fixed order."""
    out = {}
    by = defaultdict(list)
    for r in d["blimp"]:
        by[("blimp:" + r["UID"])].append(r)
    for s, rows in by.items():
        rows.sort(key=lambda r: r["pair_id"])
        diff = np.array([r["lp_good"] - r["lp_bad"] for r in rows])
        out[s] = ((diff > 0).astype(float), 1 / (1 + np.exp(-diff)), 0.5)
    by = defaultdict(list)
    for r in d["lex"]:
        by["lex:" + r["task"].split(":", 1)[1]].append(r)
    for s, rows in by.items():
        acc, cont = [], []
        for r in rows:  # file order is fixed
            lp = np.array(r["lps"])
            acc.append(float(int(np.argmax(lp)) == r["label"]))
            p = np.exp(lp - lp.max())
            cont.append(p[r["label"]] / p.sum())
        out[s] = (np.array(acc), np.array(cont), 0.25)
    return out


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def concentration(err):
    err = np.sort(np.asarray(err))[::-1]
    k = max(1, int(round(0.1 * len(err))))
    return float(err[:k].sum() / err.sum()) if err.sum() > 0 else 0.0


def windows(L):
    w, i = [], 0
    while i < len(L) - 1:
        j = next((j for j in range(i + 1, len(L)) if L[i] - L[j] >= DELTA_L), None)
        if j is None:
            break
        w.append((i, j))
        i = j
    return w


def storey_pi0(p, lam=0.5):
    p = np.asarray(p)
    return float(min(1.0, (p > lam).sum() / ((1 - lam) * len(p))))


def window_stats(A, B, n_boot=2000):
    rows = []
    for s in A:
        if s not in B:
            continue
        a_acc, a_c, _ = A[s]
        b_acc, b_c, _ = B[s]
        d_acc = b_acc - a_acc
        d_c = b_c - a_c
        if np.allclose(d_c, 0):
            p = 1.0
        else:
            t = stats.ttest_1samp(d_c, 0.0, alternative="greater")
            p = float(t.pvalue)
        n = len(d_acc)
        boots = d_acc[RNG.integers(0, n, size=(n_boot, n))].mean(1)
        rows.append({"skill": s, "n": n, "d_acc": float(d_acc.mean()), "d_cont": float(d_c.mean()),
                     "p_improve": p, "d_acc_ci_hi": float(np.quantile(boots, 0.975)),
                     "d_acc_ci_lo": float(np.quantile(boots, 0.025))})
    F_raw = np.mean([r["d_acc"] <= FLAT for r in rows])
    F_pi0 = storey_pi0([r["p_improve"] for r in rows])
    F_cons = np.mean([r["d_acc_ci_hi"] < FLAT for r in rows])
    return rows, float(F_raw), F_pi0, float(F_cons)


def analyze(repo):
    ck = load_model(repo)
    if len(ck) < 2:
        return None
    L = [d["loss"]["pile_val_nll"] for d in ck]
    S = [skill_items(d) for d in ck]
    skills = sorted(set.intersection(*[set(s) for s in S]))
    per_ck = []
    for d, s, l in zip(ck, S, L):
        acc = {k: float(s[k][0].mean()) for k in skills}
        err = np.array([1 - acc[k] for k in skills])
        chance = np.array([s[k][2] for k in skills])
        err_norm = np.clip(err / (1 - chance), 0, None)
        bl = np.array([1 - acc[k] for k in skills if k.startswith("blimp:")])
        n_items = sum(len(s[k][0]) for k in skills)
        theta = sum((1 - s[k][0]).sum() for k in skills) / n_items
        per_ck.append({"revision": d["revision"], "step": step_of(d["revision"]), "L": l,
                       "mean_acc_blimp": float(np.mean([acc[k] for k in skills if k.startswith("blimp:")])),
                       "mean_acc_lex": float(np.mean([acc[k] for k in skills if k.startswith("lex:")])),
                       "item_error_rate_theta": float(theta),
                       "gini_err": gini(err), "top10_share_err": concentration(err),
                       "gini_err_chance_norm": gini(err_norm), "top10_share_chance_norm": concentration(err_norm),
                       "gini_err_blimp_only": gini(bl), "top10_share_blimp_only": concentration(bl),
                       "acc": acc})
    W = []
    for i, j in windows(L):
        rows, F_raw, F_pi0, F_cons = window_stats({k: S[i][k] for k in skills}, {k: S[j][k] for k in skills})
        W.append({"from": ck[i]["revision"], "to": ck[j]["revision"], "L_from": L[i], "L_to": L[j],
                  "dL": L[i] - L[j], "F_raw": F_raw, "F_pi0": F_pi0, "F_cons": F_cons,
                  "n_declining_sig": int(sum(r["d_acc_ci_hi"] < 0 for r in rows)),
                  "skills": rows})
    late = [c for c in per_ck if c["L"] < 4.0]
    rho = {}
    for key in ("gini_err", "top10_share_err", "gini_err_chance_norm", "gini_err_blimp_only"):
        if len(late) >= 4:
            r = stats.spearmanr([c["L"] for c in late], [c[key] for c in late])
            rho[key] = float(r.statistic)
    return {
        "repo": repo, "n_checkpoints": len(ck), "n_skills": len(skills), "checkpoints": per_ck,
        "windows": W,
        "R2a_frac_windows_Fpi0_gt_0.2": (float(np.mean([w["F_pi0"] > 0.2 for w in W])) if W else None),
        "R2a_holds": (np.mean([w["F_pi0"] > 0.2 for w in W]) >= 0.5) if W else None,
        "spearman_L_vs_concentration(L<4)": rho,
        "R2b_holds": (rho.get("gini_err", 0) < -0.5) if rho else None,
    }


def takeaway3(a):
    """Paper's Takeaway 3 (a prediction, not a refutation test): at a given L, skills that are
    more frequent in the corpus should be more competent. Lexical skills only (frequency from
    infini-gram cue counts). OLS acc ~ L + logf + L:logf with skill-clustered SEs."""
    import statsmodels.api as sm
    counts = json.loads((ROOT / "results" / "exp4_corpus_counts.json").read_text())["single"]
    rows = []
    for c in a["checkpoints"]:
        for s, v in c["acc"].items():
            if s.startswith("lex:"):
                k = s[4:]
                rows.append((c["L"], math.log10(counts[k]["count"]), v, k))
    if not rows:
        return None
    L = np.array([r[0] for r in rows]); lf = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
    g = np.array([sorted(set(r[3] for r in rows)).index(r[3]) for r in rows])
    X = sm.add_constant(np.column_stack([L, lf, L * lf]))
    m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
    Lmed = float(np.median(L))
    eff = m.params[2] + m.params[3] * Lmed
    cov = m.cov_params()
    se = math.sqrt(cov[2, 2] + Lmed ** 2 * cov[3, 3] + 2 * Lmed * cov[2, 3])
    last = a["checkpoints"][-1]
    fin = [(math.log10(counts[s[4:]]["count"]), v) for s, v in last["acc"].items() if s.startswith("lex:")]
    rho = stats.spearmanr([x for x, _ in fin], [v for _, v in fin])
    return {"coef_logf_at_median_L": float(eff), "se": se, "median_L": Lmed,
            "params": [float(v) for v in m.params], "pvalues": [float(v) for v in m.pvalues],
            "spearman_final_logf_vs_acc": float(rho.statistic), "spearman_p": float(rho.pvalue)}


MODELS = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
          "EleutherAI/pythia-1b", "allenai/OLMo-1B-hf"]


def main():
    res = {}
    for m in MODELS:
        a = analyze(m)
        if a is None:
            continue
        a["takeaway3"] = takeaway3(a)
        res[m] = a
        print(f"\n== {m}: {a['n_checkpoints']} ckpts, {a['n_skills']} skills")
        for c in a["checkpoints"]:
            print(f"  {c['revision']:>24s} L={c['L']:.3f} blimp={c['mean_acc_blimp']:.3f} lex={c['mean_acc_lex']:.3f} "
                  f"theta={c['item_error_rate_theta']:.3f} gini={c['gini_err']:.3f} top10={c['top10_share_err']:.3f}")
        for w in a["windows"]:
            print(f"  window {w['from']}->{w['to']} dL={w['dL']:.3f} F_raw={w['F_raw']:.2f} "
                  f"F_pi0={w['F_pi0']:.2f} F_cons={w['F_cons']:.2f} sig_decl={w['n_declining_sig']}")
        print("  takeaway3:", a["takeaway3"])
        print("  R2a frac windows F_pi0>0.2:", a["R2a_frac_windows_Fpi0_gt_0.2"], "| rho:", a["spearman_L_vs_concentration(L<4)"])
    # family-level verdicts
    fam = {"pythia": [m for m in res if "pythia" in m], "olmo": [m for m in res if "OLMo" in m]}
    verdict = {}
    for f, ms in fam.items():
        if ms:
            verdict[f] = {"R2a_any_model": any(bool(res[m]["R2a_holds"]) for m in ms),
                          "R2a_all_models": all(bool(res[m]["R2a_holds"]) for m in ms),
                          "R2b_any_model": any(bool(res[m]["R2b_holds"]) for m in ms),
                          "R2b_all_models": all(bool(res[m]["R2b_holds"]) for m in ms)}
    res["_verdict"] = verdict
    OUT.write_text(json.dumps(res, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    print("\nverdict", json.dumps(verdict))


if __name__ == "__main__":
    main()
