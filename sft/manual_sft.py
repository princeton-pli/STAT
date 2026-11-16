"""Minimal manual SFT loop (no TRL) for diagnosis.

- base bf16 + LoRA (peft), loss on response tokens only
- text: {instruction}\n{question}\n{solution}<eos>
- AdamW fused, cosine, warmup 3%, wd 0.1, clip 0.3, bs 2 x accum 4
"""
import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=1024)
    return p.parse_args()


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.data = json.load(open(path))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        prompt = f"{INSTRUCTION}\n{x['question']}".strip()
        completion = "\n" + x['solution'].strip()
        p_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
        c_ids = self.tok(completion, add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id]
        input_ids = (p_ids + c_ids)[: self.max_len]
        labels = ([-100] * len(p_ids) + c_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(0); torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    lcfg = LoraConfig(lora_alpha=32, lora_dropout=0.05, r=16, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.cuda()

    ds = SFTDataset(args.dataset_path, tok, args.max_seq_length)
    dl = DataLoader(ds, batch_size=args.per_device_train_batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    steps_per_epoch = math.ceil(len(dl) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.learning_rate, weight_decay=0.1, fused=True)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)

    step = 0
    model.train()
    for epoch in range(args.num_train_epochs):
        for i, batch in enumerate(dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.gradient_accumulation_steps
            loss.backward()
            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.3)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item()*args.gradient_accumulation_steps:.4f}", flush=True)

    final = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final)
    print(f"Adapter saved to {final}")


if __name__ == "__main__":
    main()
