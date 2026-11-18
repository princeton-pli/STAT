# Training-free falsification tests of the Arora–Goyal skill-graph theory: results

Design doc: "Falsifying the Arora–Goyal Skill-Graph Theory: Experimental Designs" (2026-11-17).
Paper: Arora & Goyal, "A Theory for Emergence of Complex Skills and Language Models",
arXiv:2307.15936. Constraint: no model training. Everything ran on a 4-core CPU.
Criteria and thresholds were fixed in `PREREGISTRATION.md` (commit `f8fb404`) before any
experiment ran. Anything added after seeing data is marked **exploratory**.

<!-- SUMMARY -->

## 0. What the theory claims, and what can refute it

Reading the paper closely changed three things in the doc's plan. They were fixed before
any data was seen (`PREREGISTRATION.md` §0).

1. **The general theorem is about skill measure, not skill count (C1).** Theorem 14 (§6)
   says the µ1-*measure* of skills with at most βθ of their edges in the failure set Y is at
   least 1 − α. Only for uniform µ1 (§5, Theorem 8) does measure equal count. Takeaway 3
   says frequent skills become competent first. So "rare skills stall" is the paper's own
   prediction, not a refutation of it.
2. **The scaling statements are lower bounds (C2).** Corollary 13 and Takeaway 1 bound the
   guaranteed competence; the paper says actual gains could be higher. So only growth
   *slower* than predicted can count against them, and even then only if the bound is close
   to tight (see §3).
3. **The theorem is often vacuous (C3).** Its condition is k·θ·KL(βα‖α) > H(θ), where θ is
   the share of text pieces the model gets wrong. For k = 1 it never gives a bound. For
   k ≤ 8 and θ ≥ 0.5 it guarantees nothing on 80% of the skill measure.

As a result most of the tests below test a **strong, descriptive reading**: lower loss
improves every skill, and 10x scale doubles the number of composable skills. The paper's
text invites this reading, but its theorems do not assert it. Each section says which
reading it tests.

## 1. Formal check: the doc's Zipf counterexample

`formal/zipf_check.py` builds the doc's failure set Y exactly, with no union bound: all text
pieces that contain at least one skill ranked beyond a cut m, with m chosen so that P(Y) ≤ θ.
It checks 72 configurations: Zipf exponent s ∈ {1, 1.5}, n ∈ {10³, …, 10⁶} skills,
k ∈ {2, 4, 8}, θ ∈ {0.05, 0.1, 0.2}.

![Zipf construction](figures/formal_zipf.png)

| s | n | k | θ | skills with zero competence, by count | …by measure µ1 | max measure Theorem 14 allows |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 10³ | 4 | 0.1 | 17.6% | 2.6% | 4.9% |
| 1.0 | 10⁶ | 4 | 0.1 | 31.2% | 2.6% | 5.0% |
| 1.5 | 10³ | 4 | 0.1 | 76.1% | 2.6% | 4.9% |
| 1.5 | 10⁶ | 4 | 0.1 | 99.9% | 2.6% | 5.0% |

- The doc's arithmetic is right: by count, the share of skills with zero competence grows
  with n, and reaches 99.9% for s = 1.5.
- Those skills carry only about 2.6% of the measure, below the largest share Theorem 14
  allows, in **all 72 configurations**. A Monte Carlo run on sampled graphs agrees: with
  s = 1.5, k = 4 and θ = 0.1, mean competence is 0.04 by count and 0.905 by measure.
- **Verdict: not a refutation of Theorem 14.** It refutes only a count-weighted reading,
  which the paper makes only for uniform µ1. It does mark a real limit: the theorem says
  nothing about how many *distinct* skills a model has. Under a heavy-tailed µ1 almost all
  skills can be absent while the theorem holds.

Vacuity, i.e. the competence the theorem guarantees on 80% / 50% of the skill measure
(from `results/formal_zipf_check.json`):

| k | θ = 0.1 | θ = 0.3 | θ = 0.5 |
| --- | --- | --- | --- |
| 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | 0.44 / 0.44 | 0 / 0.14 | 0 / 0 |
| 4 | 0.60 / 0.77 | 0.00 / 0.43 | 0 / 0.11 |
| 8 | 0.69 / 0.82 | 0.22 / 0.50 | 0 / 0.21 |

<!-- EXP2 -->

## 3. Experiment 3: Skill-Mix scaling, from the published tables

There was no new generation, since there is no grader API and no GPU. The data are Yu et
al. (2023), Tables 1 (GPT-4 grader) and 2 (LLaMA-2-70B grader), for LLaMA-2-{7, 70}B-Chat.
Both models were trained on 2T tokens, so 7B → 70B is 10x compute. k\* is the largest k
whose score is at least τ, linearly interpolated; cells the paper skipped are bounded.

![Skill-Mix ratio](figures/exp3_skillmix_ratio.png)

| grader | metric | τ | k\*(7B) | k\*(70B) | ratio | doublings per decade | bounds from skipped cells |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4 | full marks | 0.5 | 1.53 | 1.69 | 1.11 | 0.15 | [−0.61, 0.76] |
| GPT-4 | full marks | 0.25 | 1.79 | 2.12 | 1.18 | 0.24 | [0.24, 1.08] |
| GPT-4 | full marks | 0.1 | 1.95 | 2.69 | 1.38 | 0.47 | [0.47, 1.43] |
| GPT-4 | all skills | 0.5 | 1.56 | 1.69 | 1.08 | 0.12 | [−0.64, 0.76] |
| GPT-4 | all skills | 0.25 | 1.84 | 2.13 | 1.16 | 0.21 | [0.21, 1.09] |
| GPT-4 | all skills | 0.1 | 2.10 | 2.78 | 1.33 | 0.41 | [0.41, 0.41] |
| LLaMA-2-70B | full marks | 0.5 | 2.15 | 2.93 | 1.36 | 0.45 | [0.45, 0.45] |
| LLaMA-2-70B | full marks | 0.25 | 3.40 | 4.56 | 1.34 | 0.42 | [0.42, 0.42] |
| LLaMA-2-70B | full marks | 0.1 | 4.23 | 5.23 | 1.24 | 0.31 | [0.07, 0.50] |
| LLaMA-2-70B | all skills | 0.5 | 3.00 | 3.00 | 1.00 | 0.00 | [0.00, 0.00] |
| LLaMA-2-70B | all skills | 0.25 | 3.68 | 4.92 | 1.34 | 0.42 | [0.42, 0.42] |
| LLaMA-2-70B | all skills | 0.1 | 4.23 | 5.57 | 1.32 | 0.40 | [0.15, 0.50] |

- **The doc's reading (k\* doubles per 10x compute) fails.** All 12 point estimates are
  below the pre-registered refutation threshold of 0.5 doublings per decade. In 5 of 12
  cells the whole bound interval is below it too.
- **That reading overstates the paper.** §2.1 of the paper says excess loss halves when N
  *and* D both grow 10x, which is 100x compute. LLaMA-2 7B → 70B kept D fixed. With the
  Chinchilla fit (E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28), that step cuts excess
  loss by only 1.43x. Indexed by excess loss, Corollary 13 then predicts a k\* ratio of
  1.43, or 0.52 doublings. A compute-optimal 10x would also give only 1.42x. *Disclosure:
  I noticed this after computing the k\* ratios. The pre-registration fixed the doubling
  rule.*
- Against 1.43, the observed ratios (1.00–1.38) still fall short in all 12 cells, but by
  0.05–0.43. In 5 of 12 cells the bounds from skipped cells reach the loss-indexed rate.
- Corollary 13 bounds the *guaranteed* competence curve. Actual k\* can grow more slowly
  than the guarantee whenever the smaller model already beats its guarantee. So even a
  clear shortfall would contradict only the descriptive reading. The Skill-Mix paper itself
  uses that reading.
- **Strength: weak.** One family, chat-tuned models, small per-cell samples, and results
  that depend on the grader (the LLaMA-2 grader gives 1.4–2.2x the k\* of GPT-4).

<!-- EXP3B -->

<!-- EXP4 -->

<!-- CAVEATS -->
