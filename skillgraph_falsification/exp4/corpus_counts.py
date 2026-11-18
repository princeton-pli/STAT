"""Pile-train frequency estimates for skills and skill pairs via the infini-gram API.

f(s)       = count of (cue_1 OR cue_2 ...) for skill s
f(s1, s2)  = count of (cues of s1) AND (cues of s2): infini-gram CNF count, which counts
             co-occurrences of the clauses within a window of max_diff_tokens tokens.
Index: v4_piletrain_llama (the Pile train split, Llama-2 tokenizer).
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skills import PAIRS, SKILLS  # noqa: E402

API = "https://api.infini-gram.io/"
INDEX = "v4_piletrain_llama"
OUT = Path(__file__).resolve().parents[1] / "results" / "exp4_corpus_counts.json"
MAX_DIFF_TOKENS = 100   # infini-gram default window for CNF co-occurrence
MAX_CLAUSE_FREQ = 50000  # default; above this, CNF counts are estimated by sampling (approx=True)


def q(query):
    for attempt in range(6):
        try:
            payload = {"index": INDEX, "query_type": "count", "query": query}
            if " AND " in query or " OR " in query:
                payload |= {"max_diff_tokens": MAX_DIFF_TOKENS, "max_clause_freq": MAX_CLAUSE_FREQ}
            r = requests.post(API, json=payload, timeout=60)
            d = r.json()
            if "error" in d:
                raise RuntimeError(d["error"])
            return {"count": d["count"], "approx": d.get("approx", False)}
        except Exception as e:  # rate limits / transient errors
            time.sleep(2 ** attempt)
            last = e
    raise RuntimeError(f"failed: {query}: {last}")


def clause(skill):
    return " OR ".join(SKILLS[skill]["cues"])


def main():
    res = {"index": INDEX, "max_diff_tokens": MAX_DIFF_TOKENS, "max_clause_freq": MAX_CLAUSE_FREQ,
           "single": {}, "pair": {}, "single_cue": {}}
    for s in SKILLS:
        res["single"][s] = q(clause(s)) | {"query": clause(s)}
        for c in SKILLS[s]["cues"]:
            res["single_cue"][c] = q(c)
        print(s, res["single"][s]["count"], flush=True)
    for outer, inner, _ in PAIRS:
        query = f"{clause(outer)} AND {clause(inner)}"
        res["pair"][f"{outer}|{inner}"] = q(query) | {"query": query}
        print(outer, inner, res["pair"][f"{outer}|{inner}"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
