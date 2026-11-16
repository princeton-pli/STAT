"""Test different prompt formats for a merged SFT model (diagnosis)."""
import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval")))
from evaluation import grader, parser
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}"
model_path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 500

ds = load_dataset("json", data_files="sft/eval_data/math500.jsonl", split="train").select(range(n))
tok = AutoTokenizer.from_pretrained(model_path)

def make_prompt(fmt, q):
    if fmt == "chat_train":       # chat template, training-time content
        msgs = [{"role": "user", "content": f"{INSTRUCTION}\n{q}".strip()}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if fmt == "chat_eval":        # chat template, committed eval content
        msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\nQuestion: {q}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if fmt == "raw":              # raw text, no chat template
        return f"{INSTRUCTION}\n{q}"
    raise ValueError(fmt)

llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.5,
          max_model_len=4096, seed=0)
sp = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=1024, n=1, seed=0)

for fmt in ["chat_train", "chat_eval", "raw"]:
    prompts = [make_prompt(fmt, ex["question"] if "question" in ex else ex["problem"]) for ex in ds]
    outs = llm.generate(prompts, sp)
    correct = 0
    for ex, out in zip(ds, outs):
        text = out.outputs[0].text.strip()
        pred = parser.extract_answer(text, "math")
        gt = str(ex["answer"])
        if "boxed" in gt:
            gt = str(parser.extract_answer(gt, "math"))
        if grader.math_equal(pred, gt):
            correct += 1
    print(f"FORMAT={fmt}: accuracy {correct/len(ds)*100:.2f}% ({correct}/{len(ds)})", flush=True)
