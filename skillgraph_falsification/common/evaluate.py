"""Evaluate one (model, revision): Pile-val loss, BLiMP per-paradigm competence,
lexical-skill cloze items. Writes one JSON per checkpoint under results/raw/.

Usage: python common/evaluate.py <repo> <revision> [--parts loss,blimp,lex,pairs] [--keep]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
from lm import LM, mean_nll_windows  # noqa: E402

DATA = ROOT / "data"
RAW = ROOT / "results" / "raw"


def eval_loss(lm, ctx=1024):
    docs = [json.loads(l) for l in open(DATA / "pile_val_sample.jsonl")]
    ids = [lm.tok(d["text"], add_special_tokens=False)["input_ids"] for d in docs]
    keep = [(d, x) for d, x in zip(docs, ids) if len(x) >= 2]
    nll, per_doc = mean_nll_windows(lm, [x for _, x in keep], ctx=ctx)
    ntok = [min(len(x), ctx - 1) for _, x in keep]
    return {"pile_val_nll": nll, "per_doc_nll": per_doc, "per_doc_ntok": ntok,
            "per_doc_set": [d["set"] for d, _ in keep]}


def eval_blimp(lm, batch=32):
    rows = [json.loads(l) for l in open(DATA / "blimp_sample.jsonl")]
    sents = []
    for r in rows:
        sents += [r["sentence_good"], r["sentence_bad"]]
    ids = [[lm.bos] + lm.tok(s, add_special_tokens=False)["input_ids"] for s in sents]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    lp = [None] * len(ids)
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        for j, v in zip(idx, lm.seq_logprobs([ids[j] for j in idx])):
            lp[j] = (sum(v), len(v))
    out = []
    for k, r in enumerate(rows):
        g, b = lp[2 * k], lp[2 * k + 1]
        out.append({"UID": r["UID"], "pair_id": r["pair_id"], "lp_good": g[0], "lp_bad": b[0],
                    "n_good": g[1], "n_bad": b[1]})
    return out


def eval_cloze(lm, path):
    items = [json.loads(l) for l in open(path)]
    out = []
    cache = {}
    for it in items:
        key = (it["prompt"], tuple(it["options"]))
        if key not in cache:
            cache[key] = lm.continuation_logprobs(it["prompt"], it["options"])
        lps, ntok = cache[key]
        out.append({"task": it["task"], "x": it["x"], "label": it["label"], "lps": lps, "ntok": ntok,
                    "tags": it["tags"], "pair": it.get("pair")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("--parts", default="loss,blimp,lex")
    ap.add_argument("--keep", action="store_true", help="keep downloaded weights")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    name = f"{a.repo.replace('/', '__')}__{a.revision}{a.tag}.json"
    out_path = RAW / name
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    parts = [p for p in a.parts.split(",") if p and p not in res]
    if not parts:
        print("done already", out_path)
        return
    t0 = time.time()
    lm = LM(a.repo, a.revision)
    res.update({"repo": a.repo, "revision": a.revision, "n_params": lm.n_params})
    timing = res.get("timing", {})
    timing["download_s"] = lm.download_s
    for p in parts:
        t = time.time()
        if p == "loss":
            res["loss"] = eval_loss(lm)
        elif p == "blimp":
            res["blimp"] = eval_blimp(lm)
        elif p == "lex":
            res["lex"] = eval_cloze(lm, DATA / "lexical_singles.jsonl")
        elif p == "pairs":
            res["pairs"] = eval_cloze(lm, DATA / "lexical_pairs.jsonl")
        timing[p] = time.time() - t
        print(f"{a.repo}@{a.revision} {p}: {timing[p]:.0f}s", flush=True)
        res["timing"] = timing
        out_path.write_text(json.dumps(res))
    lm.close(delete_files=not a.keep)
    summ = {}
    if "loss" in res:
        summ["L"] = round(res["loss"]["pile_val_nll"], 4)
    if "blimp" in res:
        summ["blimp_acc"] = round(sum(r["lp_good"] > r["lp_bad"] for r in res["blimp"]) / len(res["blimp"]), 4)
    for k in ("lex", "pairs"):
        if k in res:
            acc = defaultdict(list)
            for r in res[k]:
                acc[r["task"].split(":")[0]].append(int(max(range(4), key=lambda i: r["lps"][i]) == r["label"]))
            summ[k] = {kk: round(sum(v) / len(v), 3) for kk, v in acc.items()}
    print(json.dumps({"model": f"{a.repo}@{a.revision}", **summ, "total_s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
