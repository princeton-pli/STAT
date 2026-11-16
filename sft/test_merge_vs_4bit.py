"""Quick comparison: merged model vs 4-bit base + adapter (no merge) on N samples."""
import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval")))
from evaluation import grader, parser
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import torch

mode = sys.argv[1]          # "merged" or "4bit_adapter"
model_id = sys.argv[2]      # base model id or merged path
adapter_path = sys.argv[3] if len(sys.argv) > 3 else None
n = int(sys.argv[4]) if len(sys.argv) > 4 else 50

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}"
ds = load_dataset("json", data_files="sft/eval_data/math500.jsonl", split="train").select(range(n))

tok = AutoTokenizer.from_pretrained(model_id)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

if mode == "merged":
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
else:
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)

model.eval()
gen_cfg = GenerationConfig(max_new_tokens=1024, temperature=0.1, top_p=0.9, do_sample=True,
                           pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                           num_return_sequences=1)
torch.manual_seed(0)

correct = 0
for ex in ds:
    q = ex["question"] if "question" in ex else ex["problem"]
    prompt = f"{INSTRUCTION}\n{q}"
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    pred = parser.extract_answer(text, "math")
    gt = str(ex["answer"])
    if "boxed" in gt:
        gt = str(parser.extract_answer(gt, "math"))
    ok = grader.math_equal(pred, gt)
    correct += ok
    if not ok:
        print("WRONG | pred:", repr(pred), "| gt:", gt, "| out tail:", repr(text[-90:]))
print(f"MODE={mode} n={n}: accuracy {correct/len(ds)*100:.2f}% ({correct}/{len(ds)})")
