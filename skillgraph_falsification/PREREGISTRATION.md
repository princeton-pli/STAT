# Pre-registration: training-free falsification tests of Arora & Goyal (2023)

Written 2026-11-18, before any Experiment 2/3/4 result was computed, and committed to git
before the runs. One pilot run (Pythia-160m, final checkpoint) was made only to check the
pipeline and timing. Its numbers were seen: Pile-val NLL 2.64, BLiMP 77.7%, lexical singles
61.5%, pairs 27.9%, pair controls 64%/62%. Nothing below was changed after seeing them,
except the compute-driven cut of the Pile-val sample from 400 to 200 documents.

Design source: "Falsifying the Arora–Goyal Skill-Graph Theory: Experimental Designs"
(H. Wang, 2026-11-17). Paper: arXiv:2307.15936 (v2). Constraint from the requester: no
model training. Hardware: 4 CPU cores (AMX bf16), 15 GB RAM, no GPU.

## 0. Corrections to the design doc after reading the paper

These come from the paper's exact statements and are fixed before any data was seen.

- **C1. The general-measure theorem is measure-weighted.** Theorem 14 (§6) says the
  *µ1-measure* of skills with at most βθ of their edges in Y is ≥ 1 − α. Takeaway 3 (§7)
  says more frequent skills reach competence first. So:
  - the doc's Zipf proposition refutes only a count-weighted reading, which the paper
    makes only for uniform µ1 (§5);
  - every count-weighted statistic below tests that uniform reading, and is reported next to
    a frequency-weighted version where frequencies exist;
  - the doc's Exp 2 criterion "rare skills stall while common skills improve at the same
    loss" is the paper's *own prediction* (Takeaway 3), not a refutation. It is
    reclassified as a prediction check.
- **C2. The scaling statements are lower bounds.** Corollary 13 and Takeaway 1 are bounds:
  "actual gains could be higher". So only k\* growing *slower* than the predicted rate can
  refute. The doc's "above double" branch is dropped. The rate is conditioned on 10x scale
  halving *excess* loss (Chinchilla); where loss is available, the loss-indexed version is
  primary.
- **C3. Vacuity.** The theorem's condition is k·θ·KL(βα‖α) > H(θ). For k = 1 it never gives a
  non-trivial bound. For k ≤ 8 and θ ≥ 0.5 the bound on 80% of the measure is 0
  (`formal/zipf_check.py`). Single-skill probe items are proxies for the paper's statistical
  task τs, as in the doc. The theorem itself can only be contradicted where θ is small.

## 1. Formal check (no data)

`formal/zipf_check.py`: exact (not union-bound) Zipf construction for s ∈ {1, 1.5},
n ∈ {10³…10⁶}, k ∈ {2, 4, 8}, θ ∈ {0.05, 0.1, 0.2}, plus Monte Carlo and a vacuity map.
Pass/fail is reported as "consistent with Theorem 14 (measure version)": yes or no.

## 2. Experiment 2: loss vs per-skill competence across checkpoints

**Models and checkpoints**
- Pythia 70m, 160m, 410m: steps 512, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000,
  24000, 33000, 48000, 66000, 100000, 143000.
- Pythia 1b: steps 512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 100000, 143000.
- OLMo-1B-hf: step1000-tokens4B, step2000-tokens8B, step4000-tokens16B,
  step10000-tokens41B, step30000-tokens125B, step117850-tokens494B,
  step330000-tokens1383B, step738020-tokens3094B.
- Pythia 1.4b and 2.8b final checkpoints, as extra cross-size points.

**Skills (85, fixed)**
- The 67 BLiMP paradigms (Warstadt et al., 2020), with 200 minimal pairs each (fixed random
  subset, seed 20261118). The paradigm list was fixed by its authors, so it cannot be relabelled.
- The 18 lexical skills in `exp4/skills.py` (`single:<skill>` items, 4-option 3-shot cloze;
  7–62 items each).

**Metrics**
- L = mean next-token NLL (nats) on the first 200 Pile-validation documents in file order
  (mit-han-lab/pile-val-backup), each truncated to 1024 tokens with a BOS token prepended.
  The same documents are used for OLMo, as a shared held-out corpus.
- BLiMP competence c(s) = accuracy (lp_good > lp_bad, summed log-probs). The continuous
  version is the mean of sigmoid(lp_good − lp_bad).
- Lexical competence c(s) = accuracy (argmax summed log-prob over the 4 options). The
  continuous version is the mean normalised probability of the correct option.

**Windows (defined from L only)**
For each model, start at the first checkpoint. A window ends at the first later checkpoint
whose L is at least 0.10 nats lower; the next window starts there.

**Statistics per window**
- F_raw = fraction of skills with Δaccuracy ≤ +0.02.
- F_π0 (primary) = Storey's π̂0 (λ = 0.5) from one-sided paired t-tests (H0: mean Δ ≤ 0,
  continuous metric, items paired across the two checkpoints). It estimates the share of
  skills that are truly flat or declining, corrected for item noise.
- F_cons = fraction whose 95% bootstrap upper bound of Δaccuracy is < +0.02 (conservative).

**Statistics per checkpoint**
- Error concentration: share of all errors on the worst 10% of skills, and the Gini of
  per-skill error rates. Both use accuracy and are count-weighted.

**Refutation criteria (doc's "lower loss improves every skill" reading)**
- R2a: F_π0 > 0.20 in at least half of a model's windows, in both families.
- R2b: Gini rises as L falls. Spearman ρ(L, Gini) < −0.5 over checkpoints after L < 4.0,
  in both families.
- Strength: per C1/C3 these test the count-weighted, every-skill reading. They do not test
  Theorem 14 itself.

**Prediction check (paper's Takeaway 3, not a refutation)**
For the 18 lexical skills, with frequency f(s) from infini-gram cue counts: across Pythia
checkpoints, regress c(s) on L × log f(s). The paper predicts that frequent skills reach a
given competence at higher L.

## 3. Experiment 3: Skill-Mix scaling rate (re-analysis of published numbers only)

No new generations: there is no grader API, and CPU-only generation is infeasible.
- Data: Yu et al. 2023 (arXiv:2310.17567) Table 1 (GPT-4 grader) and Table 2 (LLaMA-2-70B
  grader), LLaMA-2-{7,13,70}B-Chat (same 2T-token recipe, so 7B→70B is 10x compute).
- k\* = largest k with p(k) ≥ τ, linear interpolation between integer k, for τ = 0.5 (doc)
  and also τ ∈ {0.25, 0.1}. Metrics: Ratio of Full Marks and Ratio of All Skills.
- Rate = log2(k\*(70B)/k\*(7B)) per decade of compute. Prediction: ≥ 1. Refutes (C2, doc's
  lower branch): < 0.5. Single family and instruct models only, so at most weak evidence.

**Exp 3b (cloze version, theory-indexed by loss; our own data)**
- Fit L(N) = E + A·N^(−a) on Pythia final checkpoints (70m…2.8b), with δ = L − E.
- For every model pair (M1, M2) with δ(M2) ≤ δ(M1)/2, Corollary 13 / Takeaway 1 predicts
  chance-corrected pair competence of M2 ≥ chance-corrected single-skill competence of M1.
  Single-skill competence here is the mean of the pair's inner and outer control accuracy.
- Refutes if the prediction fails for the majority of such model pairs. Sensitivity is
  reported for E ∈ {fit, 1.6, 1.8}.

## 4. Experiment 4: pair competence vs co-occurrence frequency

- Skills, pairs (35 compositions outer∘inner) and cue phrases: `exp4/skills.py`. Items:
  `exp4/items.py` (seed 20261118). Per pair there are three item sets:
  - pair outer(inner(x));
  - inner(x), the control c(s_inner);
  - outer(y) with y = inner(x) given, the control c(s_outer).
- Frequencies come from infini-gram index `v4_piletrain_llama` (Pile train):
  - f(s) = count of the OR of the skill's cue phrases;
  - f(s1, s2) = count of the CNF (cues of s1) AND (cues of s2), co-occurring within the
    engine's default window.
  This replaces the doc's hand-labelled tagger. It is a crude proxy for explicit,
  metalinguistic co-use.
- Models: Pythia 70m, 160m, 410m, 1b, 1.4b, 2.8b, final checkpoints (trained on the Pile).
- Primary regression per model: pair-level binomial GLM (logit link) on
  log10(f12 + 1), log10 f1, log10 f2, c_inner, c_outer, weighted by item count.
  Secondary: item-level logistic with pair-clustered SEs.
- Refutation criteria (doc):
  - R4a: β1 > 0 with p < 0.05 at the largest size.
  - R4b: β1 does not shrink with size (OLS slope of β1 on log10 params ≥ 0).
  - R4c: at the largest size, the lowest-tercile co-occurrence pairs are below the
    highest-tercile pairs after residualising on the controls (difference of mean residuals,
    95% bootstrap CI excludes 0).
  - Strength: one corpus only (the doc asks for two), 35 pairs (the doc asks for ~1000), and
    cue phrases instead of a tagger. A positive result is suggestive; a null is weak.

## 5. Not run

- Experiment 1 needs training from scratch, which the requester excluded.
- The OLMo/Dolma arm of Experiment 4 is not run (no Dolma index of comparable coverage was
  set up), so it uses one corpus only.
