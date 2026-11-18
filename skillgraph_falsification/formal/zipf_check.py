"""Numerical check of the doc's "Formal counterexample: Zipf-distributed skills"
against the exact statements in Arora & Goyal (2023), arXiv:2307.15936.

What it does
1. Implements the theorem's condition. Theorem 8 / Theorem 14 condition (eq. 8/10):
       H(theta) + k*theta*( H(ba) - ba*log(1/a) - (1-ba)*log(1/(1-a)) ) < 0
   The bracket equals -KL(Bern(ba) || Bern(a)), so the condition is
       k * theta * KL(beta*alpha || alpha) > H(theta)            (nats)
   Conclusion (Thm 14, general measure): the mu1-MEASURE of skills with at most
   beta*theta of their edges in Y is >= 1 - alpha.
2. Builds the doc's Zipf failure set Y exactly (no union bound) and measures
   (a) the COUNT fraction of skills with zero competence and
   (b) the mu1-MEASURE of those skills.
3. Checks (b) against the tightest alpha the theorem allows: no contradiction
   means the counterexample only refutes a count-weighted reading of the theorem.
4. Monte-Carlo check on a sampled skill graph.
5. Reports when the theorem is vacuous (no alpha<=0.5 with beta*theta<1).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "formal_zipf_check.json"


def H(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log(x) - (1 - x) * math.log(1 - x)


def kl_bern(p, q):
    out = 0.0
    if p > 0:
        out += p * math.log(p / q)
    if p < 1:
        out += (1 - p) * math.log((1 - p) / (1 - q))
    return out


def valid(alpha, beta, k, theta):
    if not (beta > 1 and alpha * beta < 1 and 0 < theta < 1):
        return False
    return k * theta * kl_bern(beta * alpha, alpha) > H(theta)


def min_beta(alpha, k, theta, grid=4000):
    """Smallest beta in (1, 1/alpha) satisfying the condition (KL is increasing in beta)."""
    lo, hi = 1.0, 1.0 / alpha
    if not valid(alpha, hi * (1 - 1e-12), k, theta):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if valid(alpha, mid, k, theta):
            hi = mid
        else:
            lo = mid
    return hi


def performance_curve(k, theta, alphas=None):
    """For each alpha: guaranteed competence 1 - beta*theta on >= (1-alpha) measure of skills."""
    if alphas is None:
        alphas = np.linspace(0.005, 0.6, 120)
    rows = []
    for a in alphas:
        b = min_beta(a, k, theta)
        if b is None:
            continue
        rows.append({"alpha": float(a), "beta": float(b), "frac_skills": float(1 - a),
                     "competence_lb": float(1 - b * theta)})
    return rows


def _theta_grid(theta):
    # Y of measure theta is contained in some Y' of measure theta' >= theta, and a skill with
    # <= beta*theta' of its edges in Y' has <= beta*theta' in Y; so the usable bound at theta
    # is the best one over theta' >= theta (monotone envelope).
    return [theta] + [float(t) for t in np.linspace(theta, 0.99, 60)[1:]]


def competence_lb(alpha, k, theta):
    """Best guaranteed competence on >= (1-alpha) of the skill measure. A guarantee for a
    larger set (smaller alpha') also holds here, so take the max over alpha' <= alpha."""
    vals = [_competence_lb_at(a, k, theta) for a in np.linspace(0.02, alpha, 8)]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _competence_lb_at(alpha, k, theta):
    best = None
    for t in _theta_grid(theta):
        b = min_beta(alpha, k, t)
        if b is not None:
            v = max(0.0, 1 - b * t)
            best = v if best is None else max(best, v)
    return best


def tightest_alpha_nonvacuous(k, theta):
    return min((a for a in (_tightest_alpha_at(k, t) for t in _theta_grid(theta)[:25]) if a is not None),
               default=None)


def _tightest_alpha_at(k, theta):
    """Smallest alpha for which some valid beta has beta*theta < 1 (i.e. a non-trivial
    competence bound on 1-alpha of the measure). Skills with ZERO competence must lie in
    the excluded alpha-measure set for every such pair, so their measure <= this alpha."""
    best = None
    for a in np.geomspace(1e-7, 0.999, 600):
        b = min_beta(a, k, theta)
        if b is not None and b * theta < 1:
            best = float(a)
            break
    return best


def zipf_probs(n, s):
    r = np.arange(1, n + 1, dtype=np.float64)
    p = r ** (-s)
    return p / p.sum()


def zipf_construction(n, s, k, theta):
    """Y = all pieces containing >=1 skill of rank > m, with m the smallest rank cut such that
    P(Y) = 1-(1-T_m)^k <= theta (exact for iid draws with replacement, Definition 5)."""
    p = zipf_probs(n, s)
    tail = np.concatenate([np.cumsum(p[::-1])[::-1][1:], [0.0]])  # tail[m-1] = sum_{r>m} p_r
    # P(Y) for cut m:
    py = 1 - (1 - tail) ** k
    ok = np.nonzero(py <= theta)[0]
    m = int(ok[0]) + 1  # smallest m with P(Y)<=theta
    T = float(tail[m - 1])
    return {
        "n": n, "zipf_s": s, "k": k, "theta": theta, "m": m,
        "P_Y": float(py[m - 1]),
        "count_frac_zero_competence": (n - m) / n,
        "measure_zero_competence": T,
        "doc_union_bound_m": (n ** (1 - theta / k)) if s == 1 else None,
    }


def monte_carlo(n, s, k, theta, n_pieces=400_000, seed=0):
    rng = np.random.default_rng(seed)
    p = zipf_probs(n, s)
    c = zipf_construction(n, s, k, theta)
    m = c["m"]
    pieces = rng.choice(n, size=(n_pieces, k), p=p)  # 0-indexed ranks
    in_Y = (pieces >= m).any(axis=1)
    # per-skill competence: fraction of incident pieces not in Y
    flat = pieces.ravel()
    fail = np.repeat(in_Y, k)
    tot = np.bincount(flat, minlength=n)
    bad = np.bincount(flat, weights=fail, minlength=n)
    seen = tot > 0
    comp = np.where(seen, 1 - bad / np.maximum(tot, 1), np.nan)
    zero = seen & (comp == 0)
    return {
        "n": n, "zipf_s": s, "k": k, "theta": theta, "n_pieces": n_pieces,
        "empirical_P_Y": float(in_Y.mean()),
        "skills_seen": int(seen.sum()),
        "count_frac_zero_among_seen": float(zero.sum() / seen.sum()),
        "count_frac_zero_competence_all_skills_assuming_unseen_are_tail": float(((~seen) | zero).sum() / n),
        "measure_zero_competence": float(p[zero].sum() + p[~seen].sum()),
        "mean_competence_count_weighted": float(np.nanmean(comp)),
        "mean_competence_measure_weighted": float(np.nansum(comp * p) / p[seen].sum()),
    }


def main():
    res = {"definitions": {
        "condition": "k*theta*KL(beta*alpha||alpha) > H(theta)  [eq. 8/10 rewritten]",
        "thm14_conclusion": "mu1-measure of skills with <= beta*theta of edges in Y is >= 1-alpha",
    }}

    # 1. Performance curves (reproduce the paper's Figure 2 setting) and vacuity map
    curves = {}
    for k in (2, 4, 8, 16):
        for theta in (0.05, 0.1, 0.2, 0.4):
            curves[f"k={k},theta={theta}"] = performance_curve(k, theta)
    res["performance_curves"] = curves

    vac = []
    for k in (1, 2, 4, 8, 16, 32):
        for theta in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            a = tightest_alpha_nonvacuous(k, theta)
            # best competence lower bound for 80% of measure
            vac.append({"k": k, "theta": theta, "tightest_alpha": a,
                        "competence_lb_on_80pct": competence_lb(0.2, k, theta),
                        "competence_lb_on_50pct": max(competence_lb(0.5, k, theta) or 0.0,
                                                      competence_lb(0.2, k, theta) or 0.0)})
    res["vacuity_map"] = vac

    # 2-3. Exact Zipf construction vs the theorem
    rows = []
    for s in (1.0, 1.5):
        for n in (10**3, 10**4, 10**5, 10**6):
            for k in (2, 4, 8):
                for theta in (0.05, 0.1, 0.2):
                    c = zipf_construction(n, s, k, theta)
                    a = tightest_alpha_nonvacuous(k, c["P_Y"]) if c["P_Y"] > 0 else None
                    c["thm14_tightest_alpha"] = a
                    c["consistent_with_thm14"] = (a is None) or (c["measure_zero_competence"] <= a + 1e-12)
                    rows.append(c)
    res["zipf_exact"] = rows

    # 4. Monte Carlo on sampled graphs
    mc = []
    for s in (1.0, 1.5):
        for k in (4, 8):
            mc.append(monte_carlo(10**5, s, k, 0.1))
    res["monte_carlo"] = mc

    # Uniform control: same theta, uniform mu1 — the worst Y the doc's construction can do
    uni = []
    for k in (2, 4, 8):
        for theta in (0.05, 0.1, 0.2):
            T = 1 - (1 - theta) ** (1 / k)  # fraction of skills one can zero out
            uni.append({"k": k, "theta": theta, "count_frac_zero_competence": T,
                        "thm14_tightest_alpha": tightest_alpha_nonvacuous(k, theta)})
    res["uniform_control"] = uni

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))

    print("Zipf construction (exact): count-fraction vs measure of zero-competence skills")
    print(f"{'s':>4} {'n':>8} {'k':>2} {'theta':>5} {'m':>8} {'P(Y)':>6} {'count0':>7} {'meas0':>7} {'alpha*':>7} ok")
    for c in rows:
        if c["n"] in (10**3, 10**6):
            print(f"{c['zipf_s']:>4} {c['n']:>8} {c['k']:>2} {c['theta']:>5} {c['m']:>8} {c['P_Y']:.3f} "
                  f"{c['count_frac_zero_competence']:.4f} {c['measure_zero_competence']:.4f} "
                  f"{(c['thm14_tightest_alpha'] or float('nan')):.4f} {c['consistent_with_thm14']}")
    print("\nMonte Carlo (n=1e5, theta=0.1):")
    for r in mc:
        print(json.dumps(r))
    print("\nVacuity: tightest alpha with a non-trivial competence bound")
    for r in vac:
        if r["theta"] in (0.05, 0.1, 0.3, 0.5, 0.9):
            print(r)
    print("all consistent:", all(c["consistent_with_thm14"] for c in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
