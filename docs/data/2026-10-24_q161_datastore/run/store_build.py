#!/usr/bin/env python3
"""Q161 (2026-10-25): the answer store's token stream from an allow-list of prompt sources.

Same rules as the first store (datastore_corpus.py): first answer per id in file order, Q51 left out by 12-word shingles
of user and tool messages, answers shuffled with seed 161, each followed by -1, cut at an answer boundary at CAP tokens
(separators included). Then `datastore build OUT.i32 99999999 OUT.pstore` makes the PIPESTO1 file.
With --q51-rule both (default) a prompt is also left out when a 12-word span of all its messages' text joined (system
included, spans across message boundaries) occurs in Q51's, as q51_overlap.py (docs/data/2026-10-24_q20_drafter/run/)
tests the drafter data; --q51-rule store keeps the first store's rule alone (it reproduces the first store's stream).

The allow-list file has one entry per line ('#' starts a comment):
  a data-set name, as the prompt rows' "source" field has it (e.g. AI-MO/NuminaMath-CoT)
  files:<tree> for the edit and agent prompts, by the tree their file comes from:
    files:python-stdlib     the Python 3.12 standard library
    files:llama.cpp         the llama.cpp tree, without vendor/ and build outputs
    files:llama.cpp-vendor  llama.cpp's vendor/ (third-party code it ships)
    files:site-packages     third-party Python packages
    files:build-output      generated files in build directories
  path:<prefix> for the edit and agent prompts on files whose path (relative to its tree, e.g.
    site-packages/pyarrow/) starts with the prefix, whatever their tree; !path:<prefix> leaves such files out even
    when another entry allows them (e.g. a vendored package with another licence)
Answers whose source is not allowed are left out; every source seen is listed in OUT.json with its tokens and whether it
was allowed, so a shorter list after an audit is a one-line change.
--ids FILE (one answer id per line) keeps only the listed answers, e.g. the per-package licence ruling as an answer
list has it; with --allow as well, an answer must pass both. The Q51 hold-out applies either way, and OUT.json
reports the listed answers by source as this script classifies them.

Writes OUT_PREFIX.i32 (the stream), .json (composition), .manifest.tsv (every answer in the stream, in order: id,
writer, category, prompt source as data-set name or file path, licence as the allow-list entry's comment states it,
generated tokens) and .allowed.txt (the ids of every allowed answer, before the cap).

usage: python3 -I store_build.py OUT_PREFIX (--allow LIST | --ids FILE | both) [--pool same|all] [--cap 13798889]
  --pool same: the first store's answer files; all (default): those plus the ones written since (F2, FV3, V1)
Inputs come from the environment:
  ANSWERS       directory with the drafter's answer files (gen-<label>.jsonl) and prompt files (prompts-all123.jsonl,
                prompts-tools.jsonl), as bench/staircase_drafter wrote them
  LLAMA_SRC     the llama.cpp tree the edit and agent prompts were cut from (its top-level names classify the files)
  Q51_PROMPTS   Q51's prompts (default: release/flint-v2/reproduce/data/q51/prompts.jsonl of this repository)
  NGRAM_REPLAY  the directory of datastore_corpus.py (default: this repository's bench/ngram_replay)
"""
import argparse
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
WRITER = {f"gen-{n}.jsonl": "Flint Q8_0, llama.cpp" for n in "A.final B B2 B3 C C2 C2s".split()}
WRITER.update({f"gen-{n}.jsonl": "Flint G3, llama.cpp" for n in "E1 E2 F1 F2".split()})
WRITER.update({"gen-FV3.jsonl": "Flint UD-Q4_K_XL, llama.cpp", "gen-V1.jsonl": "Flint NVFP4, vLLM"})
LLAMA_TOP = set(os.listdir(os.environ["LLAMA_SRC"]))
# Debian ships these standard-library packages separately, so they are not in /usr/lib/python3.12 there
STDLIB_TOP = set(os.listdir("/usr/lib/python3.12")) | {"idlelib", "lib2to3", "tkinter", "turtledemo"}


def joined_words(messages):
    """q51_overlap.py's Q51 rule: every message's text joined, so a span may cross message boundaries."""
    return " ".join((m.get("content") or "") for m in messages).lower().split()


def file_tree(path):
    top = path.split("/")[0]
    if top in ("site-packages", "dist-packages"):
        return "files:site-packages"
    if top.startswith("build"):
        return "files:build-output"
    if top == "vendor":
        return "files:llama.cpp-vendor"
    if top in LLAMA_TOP:
        return "files:llama.cpp"
    if top in STDLIB_TOP:
        return "files:python-stdlib"
    return "files:unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--allow")
    ap.add_argument("--ids")
    ap.add_argument("--pool", choices=("same", "all"), default="all")
    ap.add_argument("--cap", type=int, default=13_798_889)
    ap.add_argument("--q51-rule", choices=("both", "store"), default="both")
    args = ap.parse_args()
    if not args.allow and not args.ids:
        ap.error("give --allow, --ids or both")
    # an entry's comment is its licence, as the manifest reports it
    licence = {}
    for ln in open(args.allow) if args.allow else ():
        e, _, c = ln.partition("#")
        if e.strip():
            licence[e.strip()] = c.strip()
    allow = set(licence) if args.allow else None
    ids = {ln.strip() for ln in open(args.ids)} - {""} if args.ids else None
    allow_paths = [e[len("path:"):] for e in allow or () if e.startswith("path:")]
    deny_paths = [e[len("!path:"):] for e in allow or () if e.startswith("!path:")]

    def licence_of(src, path):
        hits = [p for p in allow_paths if path and path.startswith(p)]
        if hits:
            return licence["path:" + max(hits, key=len)]
        return licence.get(src, "")

    def allowed(a):
        _, _, src, _, _, path, _ = a
        if allow is None:
            return True
        if path and any(path.startswith(p) for p in deny_paths):
            return False
        return src in allow or bool(path) and any(path.startswith(p) for p in allow_paths)

    test, joined = set(), set()
    for line in open(Q51):
        msgs = json.loads(line)["messages"]
        for t in texts(msgs):
            test |= shingles(t, 12)
        w = joined_words(msgs)
        joined |= {" ".join(w[i:i + 12]) for i in range(len(w) - 11)}

    def q51_clean(msgs):
        if any(shingles(t, 12) & test for t in texts(msgs)):
            return False
        if args.q51_rule == "both":
            w = joined_words(msgs)
            return not any(" ".join(w[i:i + 12]) in joined for i in range(len(w) - 11))
        return True
    prompts = {}
    for p in PROMPTS:
        for line in open(p):
            r = json.loads(line)
            prompts[r["id"]] = r

    answers, seen = [], set()
    for gen_file in SAME + (NEW if args.pool == "all" else []):
        for line in open(gen_file):
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
            src = pr.get("source", "") if pr else "<no prompt>"
            path = ""
            if r["category"] in ("edit", "agent") and pr:
                path, src = src, file_tree(src)
            clean = bool(pr) and q51_clean(pr["messages"])
            answers.append((r["id"], r["category"], src, clean, array.array("i", r["gen_tokens"]), path,
                            os.path.basename(gen_file)))
    random.Random(161).shuffle(answers)

    seen_src = collections.Counter()
    for _, _, src, clean, gen, _, _ in answers:
        if clean:
            seen_src[src] += len(gen) + 1
    keep = [a for a in answers if a[3] and allowed(a) and (ids is None or a[0] in ids)]
    with open(args.out + ".allowed.txt", "w") as f:
        f.write("".join(a[0] + "\n" for a in keep))
    n_tok, n_ans, cats, srcs = 0, 0, collections.Counter(), collections.Counter()
    by_path = collections.Counter()
    man = open(args.out + ".manifest.tsv", "w")
    man.write("n\tid\twriter\tcategory\tsource\tlicence\ttokens\n")
    with open(args.out + ".i32", "wb") as f:
        for rid, cat, src, _, gen, path, gfile in keep:
            k = len(gen) + 1
            if n_tok + k > args.cap:
                break
            gen.tofile(f)
            f.write(struct.pack("<i", -1))
            n_tok += k
            n_ans += 1
            man.write(f"{n_ans}\t{rid}\t{WRITER.get(gfile, gfile)}\t{cat}\t{path or src}\t{licence_of(src, path)}\t"
                      f"{len(gen)}\n")
            cats[cat] += k
            srcs[src] += k
            for p in allow_paths:
                if path.startswith(p):
                    by_path[p] += k
    report = {"pool": args.pool, "cap": args.cap, "allow": sorted(allow) if allow else None, "ids": args.ids,
              "answers": n_ans, "tokens": n_tok,
              "available_tokens": sum(len(a[4]) + 1 for a in keep),
              "left_out_q51": sum(1 for a in answers if not a[3] and a[2] != "<no prompt>"),
              "no_prompt": sum(1 for a in answers if a[2] == "<no prompt>"),
              "share_by_category": {c: round(t / n_tok, 4) for c, t in cats.most_common()},
              "tokens_by_source": dict(srcs.most_common()),
              "tokens_by_path_rule": dict(by_path.most_common()),
              "clean_sources_seen": {s: {"tokens": t, "allowed": allow is None or s in allow or (
                                         "by path rules" if allow_paths and s.startswith("files:") else False)}
                                     for s, t in seen_src.most_common()}}
    if ids is not None:
        listed = [a for a in answers if a[0] in ids]
        by_src = collections.Counter()
        for a in listed:
            by_src[a[2]] += len(a[4]) + 1
        report["listed"] = {"ids": len(ids), "found": len(listed), "not_in_answer_files": len(ids - {a[0] for a in answers}),
                            "held_out_q51": sum(1 for a in listed if not a[3]),
                            "tokens_by_source": dict(by_src.most_common())}
        print(f"listed: {len(ids)} ids, {len(listed)} found in the answer files, {report['listed']['held_out_q51']} "
              f"held out for Q51; by source: {dict(by_src.most_common())}")
    man.close()
    json.dump(report, open(args.out + ".json", "w"), indent=1)
    print(f"{args.out}: {n_ans} answers, {n_tok} tokens (of {report['available_tokens']} allowed); "
          + ", ".join(f"{c} {t / n_tok:.1%}" for c, t in cats.most_common()))
    print("by source:", {s: f"{t / n_tok:.1%}" for s, t in srcs.most_common()})
    missing = sorted(e for e in allow - set(seen_src) if ":" not in e or e.startswith("files:")) if allow else []
    missing += sorted(p for p in allow_paths if p not in by_path)
    if missing:
        print("allowed but not present:", missing)
    if "files:unknown" in seen_src:
        print("WARNING: edit/agent files of unknown origin:", seen_src["files:unknown"], "tokens (left out)")


if __name__ == "__main__":
    main()
