# Training-free falsification tests of the Arora–Goyal skill-graph theory

Code and results for running the design doc "Falsifying the Arora–Goyal Skill-Graph Theory:
Experimental Designs" without training any model, on a 4-core CPU.

- `PREREGISTRATION.md`: skills, metrics, thresholds and refutation criteria, committed
  before any experiment ran (commit `f8fb404`).
- `REPORT.md`: results and interpretation.

## Layout

| Path | What it does |
| --- | --- |
| `formal/zipf_check.py` | Exact Zipf construction vs Theorem 14 (measure version); vacuity map of the bound |
| `exp2/prepare_data.py` | Freezes the BLiMP sample (67 x 200 pairs) and 200 Pile-validation documents |
| `exp4/skills.py`, `exp4/items.py` | 18 checkable lexical skills, 35 two-skill compositions, 4-option 3-shot cloze items |
| `exp4/corpus_counts.py` | Pile-train skill and pair frequencies from the infini-gram API |
| `common/lm.py`, `common/evaluate.py` | CPU scoring: bf16 body, fp32 head; Pile loss, BLiMP, cloze items |
| `run_sweep.py` | The pre-registered checkpoint sweep (Pythia 70m–2.8b, OLMo-1B), with prefetching |
| `exp2/analyze.py` | Windows, π̂0, error concentration, Takeaway-3 check |
| `exp4/analyze.py` | Co-occurrence GLMs (β1), tercile test, loss-indexed pair-vs-single test (Exp 3b) |
| `exp3/skillmix_reanalysis.py` | k\* scaling from the published Skill-Mix tables |
| `make_figures.py` | Figures in `figures/` |

## Reproduce

```bash
python -m venv venv && venv/bin/pip install torch transformers safetensors huggingface_hub \
    numpy scipy pandas statsmodels matplotlib zstandard pyarrow requests
venv/bin/python exp2/prepare_data.py          # data/ (BLiMP + Pile val)
venv/bin/python exp4/items.py                 # data/lexical_*.jsonl
venv/bin/python exp4/corpus_counts.py         # results/exp4_corpus_counts.json
venv/bin/python run_sweep.py                  # results/raw/*.json (about 7 h on 4 CPU cores)
venv/bin/python exp2/analyze.py && venv/bin/python exp4/analyze.py
python formal/zipf_check.py && python exp3/skillmix_reanalysis.py
venv/bin/python make_figures.py
```

Network: huggingface.co plus its file CDN, and api.infini-gram.io. `results/raw/` (per-item
scores, about 2 MB per checkpoint) is not committed; the summaries in `results/` are.
