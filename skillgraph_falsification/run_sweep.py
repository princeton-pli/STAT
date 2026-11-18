"""Run the pre-registered checkpoint sweep (Experiments 2 and 4) sequentially on CPU,
prefetching the next checkpoint's weights while the current one is evaluated.

python run_sweep.py            # everything in PLAN order, skipping finished parts
"""
import os
import subprocess
import sys
import threading
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "common"))
from huggingface_hub import snapshot_download  # noqa: E402

CACHE = os.environ.get("HF_HOME_MODELS", "/home/user/hf_models")
PY = sys.executable

P_SMALL = [512, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000, 33000, 48000, 66000, 100000, 143000]
P_1B = [512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 100000, 143000]
OLMO = ["step1000-tokens4B", "step2000-tokens8B", "step4000-tokens16B", "step10000-tokens41B",
        "step30000-tokens125B", "step117850-tokens494B", "step330000-tokens1383B", "step738020-tokens3094B"]

PLAN = []
for size in ("70m", "160m", "410m"):
    for s in P_SMALL:
        parts = "loss,blimp,lex" + (",pairs" if s == 143000 else "")
        PLAN.append((f"EleutherAI/pythia-{size}", f"step{s}", parts))
for size in ("1.4b", "2.8b"):
    PLAN.append((f"EleutherAI/pythia-{size}", "step143000", "loss,blimp,lex,pairs"))
for s in P_1B[::-1]:  # final first so Exp 4 has all sizes early
    PLAN.append(("EleutherAI/pythia-1b", f"step{s}", "loss,blimp,lex" + (",pairs" if s == 143000 else "")))
for r in OLMO:
    PLAN.append(("allenai/OLMo-1B-hf", r, "loss,blimp,lex"))


def prefetch(repo, rev):
    try:
        snapshot_download(repo, revision=rev, cache_dir=CACHE,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"], max_workers=4)
    except Exception as e:  # evaluation will retry the download itself
        print("prefetch failed", repo, rev, e, flush=True)


def done(repo, rev, parts):
    import json
    p = ROOT / "results" / "raw" / f"{repo.replace('/', '__')}__{rev}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    return all(k in d for k in parts.split(","))


def main():
    todo = [x for x in PLAN if not done(*x)]
    print(f"{len(todo)} of {len(PLAN)} runs to do", flush=True)
    for i, (repo, rev, parts) in enumerate(todo):
        th = None
        if i + 1 < len(todo):
            th = threading.Thread(target=prefetch, args=todo[i + 1][:2], daemon=True)
            th.start()
        r = subprocess.run([PY, str(ROOT / "common" / "evaluate.py"), repo, rev, "--parts", parts],
                           capture_output=True, text=True)
        out = [l for l in (r.stdout + r.stderr).splitlines() if "Warning" not in l and "warn" not in l]
        print("\n".join(out[-6:]), flush=True)
        if r.returncode != 0:
            print(f"FAILED {repo}@{rev} rc={r.returncode}", flush=True)
        if th is not None:
            th.join()


if __name__ == "__main__":
    main()
