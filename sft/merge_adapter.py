"""Merge a LoRA adapter into its base model and save the merged model.

Used to prepare models for fast vLLM-based evaluation.
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base_model_id", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading base model {args.base_model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"Loading adapter {args.adapter_path} ...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    print("Merging weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
