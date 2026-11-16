"""vLLM-accelerated evaluation mirroring sft/eval_instruct_model.py and
sft/eval_base_model.py logic:

- same prompt format (chat template for instruct models, raw text for base)
- same sampling params (temperature=0.1, top_p=0.9)
- same answer extraction (evaluation.parser) and grading (evaluation.grader)
- same output jsonl format
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval")))
from evaluation import grader, parser  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}"


def get_args():
    p = argparse.ArgumentParser(description="Fast vLLM math evaluation.")
    p.add_argument("--model_path", type=str, required=True,
                   help="Model to evaluate (base model or adapter-merged model).")
    p.add_argument("--model_type", type=str, default="instruct", choices=["instruct", "base"],
                   help="instruct: chat-template prompts; base: raw-text prompts.")
    p.add_argument("--test_data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--output_filename", type=str, required=True)
    p.add_argument("--num_test_samples", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Defaults to 1024 for instruct, 2048 for base (matches repo eval scripts).")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    return p.parse_args()


def format_prompt(tokenizer, model_type, question):
    if model_type == "instruct":
        messages = [{"role": "user", "content": f"{INSTRUCTION}\n\nQuestion: {question}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{INSTRUCTION}\n{question}"


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)

    if os.path.exists(output_path):
        print(f"Results already exist at {output_path}, skipping generation. "
              f"Delete the file to re-run.")
        return

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 1024 if args.model_type == "instruct" else 2048

    test_dataset = load_dataset("json", data_files=args.test_data_path, split="train")
    if args.num_test_samples != -1:
        test_dataset = test_dataset.select(range(min(len(test_dataset), args.num_test_samples)))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts, questions, ground_truths = [], [], []
    for example in test_dataset:
        question = example["question"] if "question" in example else example["problem"]
        if args.test_data_path.endswith("gsm8k.jsonl") or "/gsm8k/" in args.test_data_path:
            ground_truth = str(parser.extract_answer(example["answer"], "gsm8k"))
        elif "gt" in example:
            ground_truth = str(example["gt"])
        elif "answer" in example:
            ground_truth = str(example["answer"])
        else:
            raise KeyError(f"No ground truth field in example keys: {list(example.keys())}")
        if "boxed" in ground_truth:
            ground_truth = str(parser.extract_answer(ground_truth, "math"))
        prompts.append(format_prompt(tokenizer, args.model_type, question))
        questions.append(question)
        ground_truths.append(ground_truth)

    print(f"Loading model {args.model_path} with vLLM ...")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        seed=0,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=max_new_tokens,
        n=1,
        seed=0,
    )
    print(f"Generating {len(prompts)} responses ...")
    outputs = llm.generate(prompts, sampling)

    results, correct = [], 0
    for question, ground_truth, prompt, out in zip(questions, ground_truths, prompts, outputs):
        model_output = out.outputs[0].text.strip()
        key = "gsm8k" if ("gsm8k" in args.test_data_path) else "math"
        extracted = parser.extract_answer(model_output, key)
        is_correct = grader.math_equal(extracted, ground_truth)
        if is_correct:
            correct += 1
        results.append({
            "question": question,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "full_model_output": model_output,
            "extracted_answer": extracted,
            "is_correct": is_correct,
        })

    accuracy = correct / len(results) * 100
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Evaluation finished. Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
