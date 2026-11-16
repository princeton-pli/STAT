# STAT Reproduction Report

Reproduction of the experiments in [princeton-pli/STAT](https://github.com/princeton-pli/STAT)
("Skill-Targeted Adaptive Training", arXiv:2510.10023) on a single machine with
2x RTX 6000 Ada 48GB + 1x L4 24GB.

## 1. Environment

- Python 3.12 (pinned: `uv sync --python 3.12`; torch 2.5.1 has no cp313 wheels)
- torch 2.5.1+cu124, transformers 4.50.3, trl 0.18.2, peft 0.14.0,
  bitsandbytes 0.45.5, vllm 0.6.6.post1 (versions match `uv.lock`)
- `pip install word2number` (missing from requirements, needed by `eval/parser.py`)
- Models: `unsloth/Llama-3.2-3B-Instruct`, `unsloth/Llama-3.2-1B-Instruct`
  (byte-identical public mirrors; the HF token had no access to the gated
  `Meta-Llama/*` repos), `Qwen/Qwen2.5-3B`

## 2. Bugs found in the released repo (all fixed here)

1. **`scripts/eval_sft.sh` calls `sft/eval_sft.py`, which does not exist.**
   Fixed to select `sft/eval_instruct_model.py` / `sft/eval_base_model.py` by model type.
2. **Broken imports**: `sft/eval_*.py` do `from evaluation import parser, grader`
   but the package directory is `eval/`; `eval/parser.py` also uses flat imports
   (`from utils import *`). Fixed via an `evaluation -> eval` symlink plus
   `sys.path` patches in the eval scripts.
3. **`sft/finetune_base_model.py` produced garbage training data.**
   `format_dataset_to_messages` returns *plain-text strings* under a `"messages"`
   key, then the script applies the chat template to them. With the Qwen
   tokenizer, jinja2 silently rendered **every example as only**
   `<|im_start|>system\nYou are a helpful assistant.<|im_end|>` — i.e. the
   "Qwen2.5-3B + SFT" runs train on nothing. With the Llama tokenizer it crashes
   (`'str object' has no attribute 'role'`). Fixed to emit a proper raw-text field.
4. **Wrong instruction**: the finetune scripts use
   `"...final answer within \\boxed{{}}"` (a Python f-string artifact -> literal
   double braces) while eval uses `\\boxed{}`. Fixed.
5. **Inconsistent train/eval prompt formats**: `run_sft_llama_instruct.sh`
   trains instruct models with raw text, but `eval_instruct_model.py` evaluates
   with the chat template (and a different question wrapper). The consistent
   pairing is: instruct models -> chat-template train+eval; base models -> raw
   train+eval.
6. **Hyperparameters contradict the paper**: the scripts hardcode lr=5e-6, but
   the paper (App. C.1) sweeps peak LR over {5e-4, 2e-4, 1e-4, 8e-5, 2e-5} and
   uses weight decay 0.1 (code default 0.0). Fixed weight decay; LR swept.

## 3. Data release issues (important)

- **`STAT_data/STAT-Sel/*_9.5k.json` is broken relative to the paper.**
  - All 9,592 "solutions" are byte-identical to the **original Hendrycks MATH
    training solutions** (`eval/data/math/train.jsonl`), not the GPT-4-rewritten
    MATH-Augment (TIGER-Lab/MATH-plus) solutions described in Table 2.
  - Massive duplication: only 1,461 unique (question, solution) pairs for
    Llama-3.2-3B (1,529 for Qwen, 3,080 for Llama-1B) — the same pair repeated
    up to 54 times. Paper Table 2 says 4k unique questions / 9.5k pairs.
  - SFT on this data **degrades** Llama-3.2-3B from 43.4% to 21-30% on math500
    at every LR/format/loss-masking combination we tried (14 runs), with
    arithmetic-corrupted outputs and repetition loops.
- `STAT_data/STAT-Syn/*` is fine: all pairs unique, solutions in the detailed
  teacher style described in the paper.
- Part 3 (training-data creation code, incl. the Missing-Skill-Profile
  selection) is not released, so the exact STAT-Sel set cannot be rebuilt.

## 4. What reproduces

Eval protocol: MATH-500 (first 500 rows of `sft/eval_data/math500.jsonl`),
0-shot, temperature 0.1, top_p 0.9, seed 0 — matches the paper's baselines.

| Model | math500 base (ours) | math500 base (paper) |
|---|---|---|
| Llama-3.2-3B-Instruct | 41.6 (chat) / 43.4 (raw) | 44.0 |
| Llama-3.2-1B-Instruct | 26.0 (chat) | 26.0 |
| Qwen2.5-3B            | 56.0 (raw) | 55.8 |

**STAT-Syn training** (prompt-completion, loss on response only — llama-cookbook
style; lr 8e-5, wd 0.1, 3 epochs; QLoRA or bf16 LoRA both work):

| Model | base | +STAT-Syn (ours) | +STAT-Syn (paper) |
|---|---|---|---|
| Llama-3.2-3B-Instruct | 43.4 | **48.8** | 50.2 |
| Llama-3.2-1B-Instruct | 26.0 | **31.2** | 34.5 |
| Qwen2.5-3B            | 56.0 | **56.6** | 59.4 |

Out-of-distribution gains (pass@1, same recipe) reproduce and even exceed the paper:

| Benchmark | Model | base (ours) | +STAT-Syn (ours) | paper (base -> STAT-Syn) |
|---|---|---|---|---|
| GSM8K (500) | Llama-3.2-3B | 73.8 | **82.2** | 73.0 -> 79.2 |
| GSM8K (500) | Llama-3.2-1B | 40.4 | **48.6** | 40.7 -> 45.6 |
| AMC23 (40)  | Llama-3.2-3B | 20.0 | **30.0** | 21.7 -> 23.9 (avg@64) |

The paper's headline claim — that skill-targeted *synthesized* training data
squeezes extra performance out of a model "saturated" on MATH and generalizes
OOD, where naive SFT does not — **qualitatively reproduces**.

## 5. What does not reproduce

**STAT-Sel**: with the released (broken) data, training hurts all models
regardless of LR, prompt format, loss masking, duplication handling, or
quantization. Rebuilding the set by matching the released question selection to
MATH-plus solutions (970/1461 questions matched for the 3B) still degrades
(34-37% vs base 43.4%, paper 51.5%). Since the Missing-Skill-Profile-driven
selection/weighting code (paper Part 3) is unreleased, faithful STAT-Sel
reproduction is not possible from this repository.

## 6. Artifacts

- `logs/` — all training logs
- `output/finetuned_adapters/` — LoRA adapters (many diagnostic runs)
- `output/merged_models/` — adapter-merged models used for evaluation
- `output/evaluation_results/` — graded jsonl outputs per model/benchmark
- `sft/eval_vllm.py` — fast vLLM evaluation mirroring the repo's eval logic
- `sft/finetune_base_model_pc.py` — fixed raw-text training (response-only loss)
- `sft/manual_sft.py` — TRL-free reference training loop
- `sft/test_prompt_formats.py`, `sft/test_merge_vs_4bit.py` — diagnosis tools
