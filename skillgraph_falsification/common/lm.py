"""CPU scoring utilities shared by Experiments 2 and 4 (no training anywhere).

Transformer body runs in bf16 (AMX on this CPU); the LM head and log-softmax run in
fp32 so log-probabilities are not quantised. Every scoring function returns
log-probabilities in nats.
"""
import gc
import math
import os
import shutil
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTPS through the allowed CDN host
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(int(os.environ.get("NTHREADS", os.cpu_count() or 4)))
CACHE = os.environ.get("HF_HOME_MODELS", "/home/user/hf_models")


class LM:
    def __init__(self, repo, revision="main", dtype=torch.bfloat16):
        t0 = time.time()
        self.repo, self.revision = repo, revision
        self.path = snapshot_download(repo, revision=revision, cache_dir=CACHE,
                                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                                      max_workers=4)
        if not any(f.endswith(".safetensors") for f in os.listdir(self.path)):
            # older revisions only ship pytorch_model.bin
            self.path = snapshot_download(repo, revision=revision, cache_dir=CACHE,
                                          allow_patterns=["*.json", "*.bin", "*.txt", "*.model"],
                                          max_workers=4)
        self.download_s = time.time() - t0
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=dtype)
        self.model.eval()
        self.body = self.model.base_model
        head = self.model.get_output_embeddings()
        self.W = head.weight.detach().float()
        self.b = None if getattr(head, "bias", None) is None else head.bias.detach().float()
        bos = self.tok.bos_token_id
        self.bos = bos if bos is not None else self.tok.eos_token_id
        self.n_params = sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------ core
    def _logits(self, h):
        out = h.float() @ self.W.T
        return out if self.b is None else out + self.b

    @torch.inference_mode()
    def seq_logprobs(self, batch_ids, max_len=None):
        """batch_ids: list of token lists (already include BOS). Returns list of per-token
        log p(token_t | <t) for t>=1 (len = len(ids)-1). Right padding, no position issues."""
        L = max(len(x) for x in batch_ids)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.bos
        ids = torch.full((len(batch_ids), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch_ids), L), dtype=torch.long)
        for i, x in enumerate(batch_ids):
            ids[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
        h = self.body(input_ids=ids, attention_mask=att).last_hidden_state
        # gather all real (position -> next token) pairs, then one fp32 head pass in chunks;
        # log p = logit[target] - logsumexp(logits)  (identical to log_softmax + gather)
        sel = att[:, 1:].bool()
        hs = h[:, :-1][sel]
        tg = ids[:, 1:][sel]
        lps = []
        for j in range(0, hs.shape[0], 2048):
            lg = self._logits(hs[j:j + 2048])
            lps.append(lg.gather(1, tg[j:j + 2048, None])[:, 0] - torch.logsumexp(lg, dim=-1))
        flat = torch.cat(lps).tolist()
        out, k = [], 0
        for x in batch_ids:
            out.append(flat[k:k + len(x) - 1])
            k += len(x) - 1
        return out

    @torch.inference_mode()
    def continuation_logprobs(self, prompt, options):
        """Sum log p(option | prompt) for each option string (options carry their leading space).
        Prompt is run once; multi-token options reuse its KV cache."""
        p_ids = [self.bos] + self.tok(prompt, add_special_tokens=False)["input_ids"]
        o_ids = [self.tok(o, add_special_tokens=False)["input_ids"] for o in options]
        x = torch.tensor([p_ids])
        out = self.body(input_ids=x, use_cache=True)
        past = out.past_key_values
        last = F.log_softmax(self._logits(out.last_hidden_state[0, -1]), dim=-1)
        res, ntok = [], []
        for ids in o_ids:
            lp = float(last[ids[0]])
            if len(ids) > 1:
                o = self.body(input_ids=torch.tensor([ids[:-1]]), past_key_values=past, use_cache=True)
                lps = F.log_softmax(self._logits(o.last_hidden_state[0]), dim=-1)
                lp += float(lps.gather(1, torch.tensor(ids[1:])[:, None]).sum())
                past.crop(-(len(ids) - 1))  # drop the option tokens again
            res.append(lp)
            ntok.append(len(ids))
        return res, ntok

    def close(self, delete_files=True):
        path = self.path
        del self.model, self.body, self.W
        gc.collect()
        if delete_files:
            delete_snapshot(path)


def delete_snapshot(path):
    """Remove one revision's snapshot and the blobs it links to (other prefetched revisions of
    the same repo stay intact), so disk does not fill up across checkpoints."""
    if not path or not os.path.isdir(path):
        return
    for f in os.listdir(path):
        p = os.path.join(path, f)
        if os.path.islink(p):
            blob = os.path.realpath(p)
            if os.path.exists(blob):
                os.remove(blob)
    shutil.rmtree(path, ignore_errors=True)


def mean_nll_windows(lm, docs_ids, ctx=1024, batch=4):
    """Mean next-token NLL (nats) over fixed windows: each doc truncated to ctx tokens
    (BOS prepended). Returns (mean_nll, per_doc_mean_nll)."""
    per_doc, tot, n = [], 0.0, 0
    for i in range(0, len(docs_ids), batch):
        chunk = [[lm.bos] + d[:ctx - 1] for d in docs_ids[i:i + batch]]
        for lps in lm.seq_logprobs(chunk):
            s = -sum(lps)
            per_doc.append(s / len(lps))
            tot += s
            n += len(lps)
    return tot / n, per_doc
