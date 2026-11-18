"""Download and freeze the evaluation data for Experiment 2.

- BLiMP (Warstadt et al., 2020): 67 paradigms x 1000 minimal pairs. Each paradigm is one
  pre-registered "skill" (the paradigm list was fixed by its authors, so it cannot be
  relabelled after seeing results). We keep a fixed random subset of N_PER pairs per paradigm.
- Pile validation (mit-han-lab/pile-val-backup, the Pile's val split): the first N_DOCS
  documents in file order, as the held-out set for global loss L.
"""
import io
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import pandas as pd
import zstandard
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
N_PER = int(os.environ.get("BLIMP_N_PER", 200))
N_DOCS = int(os.environ.get("PILE_N_DOCS", 200))
SEED = 20261118


def blimp():
    api = HfApi()
    files = [f for f in api.list_repo_files("nyu-mll/blimp", repo_type="dataset") if f.endswith(".parquet")]
    assert len(files) == 67, len(files)
    rows = []
    rng = random.Random(SEED)
    for f in sorted(files):
        p = hf_hub_download("nyu-mll/blimp", f, repo_type="dataset", cache_dir=str(DATA / "hf"))
        df = pd.read_parquet(p)
        idx = sorted(rng.sample(range(len(df)), min(N_PER, len(df))))
        for i in idx:
            r = df.iloc[i]
            rows.append({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()})
    out = DATA / "blimp_sample.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("blimp", len(rows), "pairs ->", out)


def pile_val():
    p = hf_hub_download("mit-han-lab/pile-val-backup", "val.jsonl.zst", repo_type="dataset",
                        cache_dir=str(DATA / "hf"))
    docs = []
    with open(p, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            d = json.loads(line)
            docs.append({"text": d["text"], "set": d.get("meta", {}).get("pile_set_name")})
            if len(docs) >= N_DOCS:
                break
    out = DATA / "pile_val_sample.jsonl"
    with out.open("w") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    print("pile val", len(docs), "docs ->", out)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    what = sys.argv[1:] or ["blimp", "pile"]
    if "blimp" in what:
        blimp()
    if "pile" in what:
        pile_val()
