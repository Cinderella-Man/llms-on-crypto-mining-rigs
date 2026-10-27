#!/usr/bin/env python3
"""Q161 (2026-10-25): check a continuation store against the release rules, independently of the builder (store_build.py).

Rules (licence rulings of 2026-10-25): every answer the store was built from answers
  - an edit or agent prompt on a file of the Python standard library, of llama.cpp outside vendor/ and build*/, or of a
    package (or vendored sub-path) marked "yes" in the per-package ruling (strict-sitepkgs.md, read here), or
  - a prompt from OpenR1-Math (default config) or OpenAssistant oasst2;
and no prompt shares a 12-word span with a Q51 prompt, by either test: user and tool messages one at a time (the store's
rule) or all messages' text joined (q51_overlap.py's rule).

Checks, per manifest row: the id is unique; the writer's answer file holds it with the manifest's category and token
count; its prompt's source is the manifest's and passes the rules; the Q51 tests pass. Then the manifest rebuilds the
store: the rows' answers in order, each followed by -1, must equal the stream file byte for byte, and the PIPESTO1 file
must hold that stream reversed (its suffix array is not checked here).

usage: python3 -I verify_store.py MANIFEST.tsv STREAM.i32 STORE.pstore SITEPKGS.md
Inputs come from the environment:
  ANSWERS       directory with the drafter's answer files (gen-<label>.jsonl) and prompt files (prompts-all123.jsonl,
                prompts-tools.jsonl), as bench/staircase_drafter wrote them
  LLAMA_SRC     the llama.cpp tree the edit and agent prompts were cut from (its top-level names classify the files)
  Q51_PROMPTS   Q51's prompts (default: release/flint-v2/reproduce/data/q51/prompts.jsonl of this repository)
Prints totals per source and per licence, every failure, and PASS or FAIL.
"""
import array
import collections
import hashlib
import json
import os
import re
import struct
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
W = os.environ["ANSWERS"]
PROMPTS = [f"{W}/prompts-all123.jsonl", f"{W}/prompts-tools.jsonl"]
Q51 = os.environ.get("Q51_PROMPTS", os.path.join(REPO, "release/flint-v2/reproduce/data/q51/prompts.jsonl"))
WRITER_FILES = {"Flint Q8_0, llama.cpp": "A.final B B2 B3 C C2 C2s", "Flint G3, llama.cpp": "E1 E2 F1 F2",
                "Flint UD-Q4_K_XL, llama.cpp": "FV3", "Flint NVFP4, vLLM": "V1"}
DATASETS = {"open-r1/OpenR1-Math-220k": "Apache-2.0", "OpenAssistant/oasst2": "Apache-2.0"}
LLAMA_TOP = set(os.listdir(os.environ["LLAMA_SRC"]))
STDLIB_TOP = set(os.listdir("/usr/lib/python3.12")) | {"idlelib", "lib2to3", "tkinter", "turtledemo"}
ALIAS = {"PyYAML": ["yaml", "_yaml"]}


def sitepkgs_rules(md):
    """(prefix under site-packages/, included, licence) from the ruling's table rows."""
    rules = []
    for ln in open(md):
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(yes|no)\s*\|", ln)
        if not m:
            continue
        name, lic, inc = m.group(1).split(" (")[0].split()[0], m.group(2), m.group(3) == "yes"
        for p in (ALIAS.get(name, [name]) if "/" not in name else [name]):
            rules.append((p, inc, lic))
    return rules


def sitepkg_verdict(path, rules):
    rel = path[len("site-packages/"):]
    hits = [r for r in rules if rel == r[0] or rel.startswith(r[0] + "/")]
    if not hits:
        return False, "no ruling"
    p, inc, lic = max(hits, key=lambda r: len(r[0]))
    return inc, lic


def tree_verdict(path, rules):
    top = path.split("/")[0]
    if top == "site-packages":
        ok, lic = sitepkg_verdict(path, rules)
        return ok, lic, "site-packages/" + path.split("/")[1]
    if top in LLAMA_TOP and top != "vendor" and not top.startswith("build"):
        return True, "MIT", "llama.cpp"
    if top in STDLIB_TOP and top not in LLAMA_TOP:
        return True, "PSF-2.0", "python stdlib"
    return False, "not an allowed tree", "other: " + top


def shingles(text):
    w = text.lower().split()
    return {tuple(w[i:i + 12]) for i in range(len(w) - 11)}


def joined(msgs):
    w = " ".join((m.get("content") or "") for m in msgs).lower().split()
    return {" ".join(w[i:i + 12]) for i in range(len(w) - 11)}


def per_message(msgs):
    out = set()
    for m in msgs:
        if m.get("role") in ("user", "tool") and isinstance(m.get("content"), str):
            out |= shingles(m["content"])
    return out


def main():
    man_path, stream_path, store_path, md = sys.argv[1:5]
    rules = sitepkgs_rules(md)
    rows = [ln.rstrip("\n").split("\t") for ln in open(man_path)][1:]
    fails = []

    q51_msg, q51_join = set(), set()
    for ln in open(Q51):
        msgs = json.loads(ln)["messages"]
        q51_msg |= per_message(msgs)
        q51_join |= joined(msgs)
    prompts = {}
    for p in PROMPTS:
        for ln in open(p):
            r = json.loads(ln)
            prompts[r["id"]] = r
    want = {r[1]: r for r in rows}
    if len(want) != len(rows):
        fails.append(f"duplicate ids in the manifest: {len(rows) - len(want)}")
    gen = {}
    for writer, names in WRITER_FILES.items():
        for nm in names.split():
            for ln in open(f"{W}/gen-{nm}.jsonl"):
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                m = want.get(r["id"])
                if m and m[2] == writer and r["id"] not in gen:
                    gen[r["id"]] = (r["category"], array.array("i", r["gen_tokens"]))

    by_src, by_lic = collections.Counter(), collections.Counter()
    stream = array.array("i")
    for n, rid, writer, cat, source, lic, tok in rows:
        if rid not in gen:
            fails.append(f"{rid}: not in {writer}'s answer files")
            continue
        gcat, toks = gen[rid]
        if gcat != cat or len(toks) != int(tok):
            fails.append(f"{rid}: category/tokens {gcat}/{len(toks)} in the answer file, {cat}/{tok} in the manifest")
        pr = prompts.get(rid)
        if pr is None:
            fails.append(f"{rid}: no prompt")
            continue
        if pr.get("source") != source:
            fails.append(f"{rid}: prompt source {pr.get('source')!r}, manifest {source!r}")
        if cat in ("edit", "agent"):
            ok, rlic, group = tree_verdict(pr["source"], rules)
        else:
            ok, rlic, group = pr.get("source") in DATASETS, DATASETS.get(pr.get("source"), "not allowed"), pr.get("source")
        if not ok:
            fails.append(f"{rid}: source {pr.get('source')!r} not allowed ({rlic})")
        if per_message(pr["messages"]) & q51_msg or joined(pr["messages"]) & q51_join:
            fails.append(f"{rid}: shares a 12-word span with Q51")
        by_src[group] += len(toks) + 1
        by_lic[rlic] += len(toks) + 1
        stream.extend(toks)
        stream.append(-1)

    data = stream.tobytes()
    disk = open(stream_path, "rb").read()
    if data != disk:
        fails.append(f"the manifest does not rebuild {stream_path} ({len(data)} vs {len(disk)} bytes)")
    with open(store_path, "rb") as f:
        magic, n = f.read(8), struct.unpack("<q", f.read(8))[0]
        rev = array.array("i")
        rev.frombytes(f.read(4 * n))
    if magic != b"PIPESTO1" or n != len(stream) or list(reversed(rev)) != list(stream):
        fails.append(f"{store_path} does not hold the manifest's stream (magic {magic!r}, {n} tokens)")

    total = sum(by_src.values())
    print(f"{os.path.basename(store_path)}: {len(rows)} answers, {total} tokens (separators included); stream md5 "
          f"{hashlib.md5(disk).hexdigest()}, store md5 {hashlib.md5(open(store_path, 'rb').read()).hexdigest()}")
    print("per source:")
    for s, t in by_src.most_common():
        print(f"  {s:45} {t:9}  {t / total:6.1%}")
    print("per licence:", ", ".join(f"{k} {v / total:.1%}" for k, v in by_lic.most_common()))
    for x in fails[:30]:
        print("FAIL:", x)
    print("PASS" if not fails else f"FAIL ({len(fails)} problems)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
