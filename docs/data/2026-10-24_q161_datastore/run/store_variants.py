#!/usr/bin/env python3
"""Q161 (2026-10-25): licence-filtered variants of the answer store, as token streams for `datastore build`.

Same rules as the current store (flint-answers-clean-13.8M.pstore, datastore_corpus.py): first answer per id in file order,
Q51 left out by 12-word shingles of user/tool messages, answers shuffled with seed 161, each followed by -1, cut at an
answer boundary at CAP tokens (separators included; the current store has 13,798,889).

Variants:
  noshare  every clean answer except those to Aeala/ShareGPT_Vicuna_unfiltered prompts
  strict   only hermes-function-calling-v1, NuminaMath-CoT, OpenR1-Math-220k, ToolACE, glaive-function-calling-v2, and the
           edit and agent prompts built from files of the Python standard library and the llama.cpp tree (its vendor/
           directory included); not third-party Python packages (site-packages/) or build outputs (build*/)
Pools:
  same     the answer files of the current store
  all      those plus the answer files written since (F2, FV3, V1)

usage: python3 -I store_variants.py OUT_DIR
Inputs come from the environment (ANSWERS, LLAMA_SRC, Q51_PROMPTS, NGRAM_REPLAY), as for store_build.py.
"""
import array
import collections
import json
import os
import random
import struct
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, os.environ.get("NGRAM_REPLAY", os.path.join(REPO, "bench/ngram_replay")))
from datastore_corpus import shingles, texts  # noqa: E402

W = os.environ["ANSWERS"]
SAME = [f"{W}/gen-{n}.jsonl" for n in "A.final B B2 B3 C C2 C2s E1 E2 F1".split()]
NEW = [f"{W}/gen-{n}.jsonl" for n in "F2 FV3 V1".split()]
PROMPTS = [f"{W}/prompts-all123.jsonl", f"{W}/prompts-tools.jsonl"]
Q51 = os.environ.get("Q51_PROMPTS", os.path.join(REPO, "release/flint-v2/reproduce/data/q51/prompts.jsonl"))
CAP = 13_798_889
SHARE = "Aeala/ShareGPT_Vicuna_unfiltered"
STRICT_SETS = {"NousResearch/hermes-function-calling-v1", "AI-MO/NuminaMath-CoT", "open-r1/OpenR1-Math-220k",
               "Team-ACE/ToolACE", "glaiveai/glaive-function-calling-v2"}
LLAMA_TOP = set(os.listdir(os.environ["LLAMA_SRC"]))
# Debian ships these standard-library packages separately, so they are not in /usr/lib/python3.12 there
STDLIB_TOP = set(os.listdir("/usr/lib/python3.12")) | {"idlelib", "lib2to3", "tkinter", "turtledemo"}


def file_class(path):
    top = path.split("/")[0]
    if top in ("site-packages", "dist-packages"):
        return "third-party package"
    if top.startswith("build"):
        return "build output"
    if top == "vendor":
        return "llama.cpp vendor"
    if top in LLAMA_TOP:
        return "llama.cpp"
    if top in STDLIB_TOP:
        return "python stdlib"
    return "unknown"


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    test = set()
    for line in open(Q51):
        for t in texts(json.loads(line)["messages"]):
            test |= shingles(t, 12)
    prompts = {}
    for p in PROMPTS:
        for line in open(p):
            r = json.loads(line)
            prompts[r["id"]] = r
    pools = {}
    answers, seen = [], set()
    for pool, files in (("same", SAME), ("all", NEW)):
        for path in files:
            for line in open(path):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                if not r.get("gen_tokens"):
                    continue
                pr = prompts.get(r["id"])
                a = {"id": r["id"], "category": r["category"], "file": os.path.basename(path),
                     "gen": array.array("i", r["gen_tokens"]),
                     "source": pr.get("source", "") if pr else "", "clean": False, "why": "no prompt"}
                if pr:
                    a["clean"] = not any(shingles(t, 12) & test for t in texts(pr["messages"]))
                    a["why"] = "" if a["clean"] else "shares text with Q51"
                a["class"] = file_class(a["source"]) if a["category"] in ("edit", "agent") else a["source"]
                answers.append(a)
        pools[pool] = list(answers)
    for pool in pools:
        random.Random(161).shuffle(pools[pool])
    report = {}
    for pool, rows in pools.items():
        report[pool] = {"answers": len(rows), "left_out": dict(collections.Counter(a["why"] for a in rows if a["why"]))}
        cls = collections.Counter()
        for a in rows:
            if a["clean"]:
                cls[a["class"] if a["category"] in ("edit", "agent") else "data set " + a["source"]] += len(a["gen"]) + 1
        report[pool]["clean_tokens_by_source_class"] = dict(cls.most_common())
        unknown = sorted({a["source"] for a in rows if a["clean"] and a["class"] == "unknown"})
        report[pool]["unknown_file_sources"] = unknown[:50]
        for variant in ("clean", "noshare", "strict"):
            if variant == "clean":
                # the current store's rule: must reproduce corpus/clean.i32 for the same pool
                keep = [a for a in rows if a["clean"]]
            elif variant == "noshare":
                keep = [a for a in rows if a["clean"] and a["source"] != SHARE]
            else:
                keep = [a for a in rows if a["clean"] and (a["source"] in STRICT_SETS or a["class"] in (
                    "python stdlib", "llama.cpp", "llama.cpp vendor"))]
            name = f"{variant}-{pool}"
            n_tok, n_ans, cats, srcs = 0, 0, collections.Counter(), collections.Counter()
            with open(f"{out}/{name}.i32", "wb") as f:
                for a in keep:
                    k = len(a["gen"]) + 1
                    if n_tok + k > CAP:
                        break
                    a["gen"].tofile(f)
                    f.write(struct.pack("<i", -1))
                    n_tok += k
                    n_ans += 1
                    cats[a["category"]] += k
                    srcs[a["class"] if a["category"] in ("edit", "agent") else a["source"]] += k
            report[name] = {"answers": n_ans, "tokens": n_tok, "available_tokens": sum(len(a["gen"]) + 1 for a in keep),
                            "share_by_category": {c: round(t / n_tok, 4) for c, t in cats.most_common()},
                            "tokens_by_source": dict(srcs.most_common())}
            print(f"{name}: {n_ans} answers, {n_tok} tokens (of {report[name]['available_tokens']} available); "
                  + ", ".join(f"{c} {t / n_tok:.1%}" for c, t in cats.most_common()))
    json.dump(report, open(f"{out}/variants.json", "w"), indent=1)
    for pool in pools:
        print(pool, "clean tokens by source class:", report[pool]["clean_tokens_by_source_class"])
        print(pool, "left out:", report[pool]["left_out"], "unknown file sources:", report[pool]["unknown_file_sources"][:10])


if __name__ == "__main__":
    main()
