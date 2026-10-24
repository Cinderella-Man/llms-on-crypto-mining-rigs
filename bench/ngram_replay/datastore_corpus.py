#!/usr/bin/env python3
"""Token streams for the continuation datastore (registry row Q161) from generated answers, with Q51's test set kept out.

Reads answers in the format of bench/staircase_drafter/generate.py (one JSON object per line: id, category, prompt_tokens,
gen_tokens, ...), keeps the first answer per id, and writes the generated tokens of every answer, answers in a shuffled
order (fixed seed) and each followed by -1, as int32 streams. A store of N tokens is the first N tokens of a stream (cut at
an answer boundary), so smaller stores are subsets of larger ones.

Held-out check: an answer is left out when its prompt shares any span of SHINGLE words (lowercased, split on whitespace)
with the user or tool messages of a Q51 prompt (bench/expert_overlap/prompts.py: the chat and code requests, the edit
instructions and the files they edit, the agent tasks and the files the agents read). System messages and tool
definitions are not compared: they carry no test content (the agent prompts of both sets use the same four tools).

Streams written to OUT_DIR:
  clean.i32     answers that pass the check
  all.i32       every answer
  noagent.i32   clean answers without the agent and tool-call categories (how much of an agent result comes from
                answers to the same four tools)
  index.json    per stream, the answers in order: id, category, source file, tokens, and why answers were left out

usage: python datastore_corpus.py OUT_DIR PROMPTS.jsonl Q51_PROMPTS.jsonl GEN.jsonl [GEN.jsonl ...] [--seed N]
       [--shingle 12]
  PROMPTS.jsonl: the prompts the answers were generated from (id, messages), to compare with Q51's
"""
import argparse
import json
import random
import struct
import sys
from pathlib import Path


def texts(messages):
    """The user and tool message contents of a prompt."""
    out = []
    for m in messages:
        if m.get("role") in ("user", "tool") and isinstance(m.get("content"), str):
            out.append(m["content"])
    return out


def shingles(text, n):
    w = text.lower().split()
    return {hash(tuple(w[i:i + n])) for i in range(max(0, len(w) - n + 1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("prompts")
    ap.add_argument("q51_prompts")
    ap.add_argument("gen", nargs="+")
    ap.add_argument("--seed", type=int, default=161)
    ap.add_argument("--shingle", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    test = set()
    for line in open(args.q51_prompts):
        for t in texts(json.loads(line)["messages"]):
            test |= shingles(t, args.shingle)

    prompts = {}
    for line in open(args.prompts):
        r = json.loads(line)
        prompts[r["id"]] = r["messages"]

    answers, seen, dup = [], set(), 0
    for path in args.gen:
        for line in open(path):
            r = json.loads(line)
            if r["id"] in seen:
                dup += 1
                continue
            seen.add(r["id"])
            if not r["gen_tokens"]:
                continue
            answers.append({"id": r["id"], "category": r["category"], "source": r.get("source", ""),
                            "file": Path(path).name, "gen": r["gen_tokens"]})

    why = {"no prompt": 0, "shares text with Q51": 0}
    for a in answers:
        msgs = prompts.get(a["id"])
        if msgs is None:
            a["clean"] = False
            why["no prompt"] += 1
            continue
        hit = any(shingles(t, args.shingle) & test for t in texts(msgs))
        a["clean"] = not hit
        why["shares text with Q51"] += hit

    random.Random(args.seed).shuffle(answers)
    streams = {
        "all": answers,
        "clean": [a for a in answers if a["clean"]],
        "noagent": [a for a in answers if a["clean"] and a["category"] not in ("agent", "tool")],
    }
    index = {"seed": args.seed, "shingle": args.shingle, "duplicates_skipped": dup, "left_out": why,
             "left_out_by_category": {}, "streams": {}}
    for a in answers:
        if not a["clean"]:
            c = index["left_out_by_category"]
            c[a["category"]] = c.get(a["category"], 0) + 1
    for name, rows in streams.items():
        n_tok = 0
        with open(out / f"{name}.i32", "wb") as f:
            for a in rows:
                f.write(struct.pack(f"<{len(a['gen']) + 1}i", *a["gen"], -1))
                n_tok += len(a["gen"]) + 1
        index["streams"][name] = {"answers": len(rows), "tokens": n_tok,
                                  "rows": [[a["id"], a["category"], a["file"], len(a["gen"])] for a in rows]}
        print(f"{name}: {len(rows)} answers, {n_tok} tokens (separators included)")
    print(f"left out: {why}; by category {index['left_out_by_category']}; duplicate ids skipped: {dup}")
    with open(out / "index.json", "w") as f:
        json.dump(index, f)


if __name__ == "__main__":
    sys.exit(main())
