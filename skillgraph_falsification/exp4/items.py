"""Build cloze items (4 options, 3-shot Q/A prompts) for single skills and skill pairs.

Item kinds
- single:<skill>            the skill on its own domain (Experiment 2 skill inventory)
- pair:<outer>|<inner>      outer(inner(x))  (Experiment 4)
- inner:<outer>|<inner>     inner(x) for the same x          (per-pair control c(s_inner))
- outer:<outer>|<inner>     outer(y) with y = inner(x) given  (per-pair control c(s_outer))

Distractors for pairs are attributable where the types allow it:
  skip_inner = outer(x)  (answering as if the inner skill was not applied)
  skip_outer = inner(x)  (answering as if the outer skill was not applied)
and the rest are outer(inner(x')) for other x'.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skills import ADJ, COLOR_FR, DAYS, DOMAIN, NOUN_PLURAL, PAIRS, SKILLS, VERB  # noqa: E402

SEED = 20261118
QUOTED = {"adj", "verb", "noun", "adj_cmp", "verb_past", "noun_pl", "num_word", "word_fr", "word_en"}
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def type_of(skill, x_type):
    out = SKILLS[skill]["out"]
    return x_type if out == "same" else out


def render(v, t):
    return f'"{v}"' if t in QUOTED else str(v)


def apply(skill, x):
    try:
        y = SKILLS[skill]["f"](x)
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    return y


def accepts(skill, t):
    inp = SKILLS[skill]["inp"]
    return t in inp


def answer_ok(y, t):
    if y is None:
        return False
    if t == "letter":
        return y in LETTERS
    return True


def norm(s):
    return str(s).strip().lower()


def question(np_text):
    return f"What is {np_text}?"


def make_prompt(shots, q):
    lines = [f"Q: {sq}\nA: {sa}" for sq, sa in shots]
    lines.append(f"Q: {q}\nA:")
    return "\n\n".join(lines)


def build_task(kind, triples, rng, n_shots=3):
    """triples: list of (x, question_text, answer, answer_type, extra_distractor_candidates).
    Shots come from other x in the same task. Returns item dicts."""
    items = []
    answers_pool = [t[2] for t in triples]
    for i, (x, q, a, at, special) in enumerate(triples):
        others = [t for j, t in enumerate(triples) if j != i and norm(t[2]) != norm(a)]
        if len(others) < n_shots:
            others_any = [t for j, t in enumerate(triples) if j != i]
        else:
            others_any = others
        shots_src = rng.sample(others_any, min(n_shots, len(others_any)))
        shots = [(t[1], str(t[2])) for t in shots_src]
        opts, tags = [str(a)], ["correct"]
        for tag, d in special:
            if d is not None and answer_ok(d, at) and norm(d) not in {norm(o) for o in opts}:
                opts.append(str(d))
                tags.append(tag)
        pool = [p for p in dict.fromkeys(answers_pool) if norm(p) not in {norm(o) for o in opts}]
        rng.shuffle(pool)
        if at == "letter":
            pool += [L for L in rng.sample(sorted(LETTERS), 26) if norm(L) not in {norm(o) for o in opts} | {norm(p) for p in pool}]
        while len(opts) < 4 and pool:
            opts.append(str(pool.pop()))
            tags.append("other")
        if len(opts) < 4:
            continue
        opts, tags = opts[:4], tags[:4]
        order = list(range(4))
        rng.shuffle(order)
        items.append({
            "task": kind, "x": x, "prompt": make_prompt(shots, q),
            "options": [" " + opts[k] for k in order], "tags": [tags[k] for k in order],
            "label": order.index(0),
        })
    return items


def single_domain(skill):
    if skill == "opposite":
        return [(a, "adj") for a in ADJ] + [(v, "verb") for v in VERB]
    if skill == "to_french":
        return ([(a, "adj") for a in ADJ if ADJ[a][2] != ADJ[a][0]] + [(c, "color") for c in COLOR_FR]
                + [(d, "day") for d in DAYS] + [(n, "num") for n in range(1, 13)])
    if skill in ("first_letter", "last_letter"):
        return ([(a, "adj") for a in list(ADJ)[:10]] + [(v, "verb") for v in list(VERB)[:10]]
                + [(n, "noun") for n in list(NOUN_PLURAL)[:10]])
    if skill == "plural":
        return [(n, "noun") for n in list(NOUN_PLURAL)[:15]] + [(n, "noun") for n in list(NOUN_PLURAL)[33:]]
    t = sorted(SKILLS[skill]["inp"])[0]
    return [(x, t) for x in DOMAIN[t]]


def build_singles(rng):
    items = []
    for s in SKILLS:
        triples = []
        for x, t in single_domain(s):
            y = apply(s, x)
            at = type_of(s, t)
            if not answer_ok(y, at):
                continue
            # first_letter / last_letter / french on a quoted word
            triples.append((x, question(SKILLS[s]["np"].format(x=render(x, t))), y, at, []))
        items += build_task(f"single:{s}", triples, rng)
    return items


def build_pairs(rng):
    items = []
    for outer, inner, bt in PAIRS:
        name = f"{outer}|{inner}"
        pair_t, inner_t, outer_t = [], [], []
        for x in DOMAIN[bt]:
            y = apply(inner, x)
            yt = type_of(inner, bt)
            if y is None or not accepts(outer, yt):
                continue
            z = apply(outer, y)
            zt = type_of(outer, yt)
            if not answer_ok(z, zt):
                continue
            special = []
            if accepts(outer, bt):
                special.append(("skip_inner", apply(outer, x)))
            yt_kind = "num" if yt == "num" else ("letter" if yt == "letter" else "str")
            zt_kind = "num" if zt == "num" else ("letter" if zt == "letter" else "str")
            if yt_kind == zt_kind:
                special.append(("skip_outer", y))
            q_pair = question(SKILLS[outer]["np"].format(x=SKILLS[inner]["np"].format(x=render(x, bt))))
            pair_t.append((x, q_pair, z, zt, special))
            inner_t.append((x, question(SKILLS[inner]["np"].format(x=render(x, bt))), y, yt, []))
            outer_t.append((x, question(SKILLS[outer]["np"].format(x=render(y, yt))), z, zt, []))
        # identical shots/x ordering across the three tasks (same rng state per pair)
        seed = rng.randrange(1 << 30)
        for kind, tr in (("pair", pair_t), ("inner", inner_t), ("outer", outer_t)):
            r = random.Random(seed)
            for it in build_task(f"{kind}:{name}", tr, r):
                it["pair"] = name
                items.append(it)
    return items


def main(out_dir):
    rng = random.Random(SEED)
    singles = build_singles(rng)
    pairs = build_pairs(rng)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lexical_singles.jsonl").write_text("".join(json.dumps(i) + "\n" for i in singles))
    (out_dir / "lexical_pairs.jsonl").write_text("".join(json.dumps(i) + "\n" for i in pairs))
    from collections import Counter
    c = Counter(i["task"] for i in singles + pairs)
    for k, v in sorted(c.items()):
        print(f"{v:4d} {k}")
    print("singles", len(singles), "pair-related", len(pairs))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data")
