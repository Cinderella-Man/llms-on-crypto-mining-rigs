#!/usr/bin/env python3
"""C951 (Q1687): the tables of run/run.sh's output. Nothing timed is read or written.

Usage: python3 -I c951_tables.py <run.sh's WORK/RUN folder> <output folder>
Writes:
  loads.tsv    one row per load: model, arm, build, draft head, whether it answered, the server's exit, the
               "creating indexer KV cache" lines (the target's and, for a sparse-attention draft, the draft's), the
               draft's k-pool table and pool-mask lines, the assertion that stopped it (file:line and condition only),
               MiB per card after the load and after the requests
  replies.tsv  one row per reply: model, arm, round, request, prompt tokens, generated tokens, md5 of the token ids
               and of the text, drafted tokens, accepted drafts, stop reason
  agree.tsv    one row per model, request and round: the branch's preset-kind head against the base's (the same token
               ids, the same drafted and accepted counts), the distilled head against the preset-kind head, each head
               against plain decoding (the same token ids, or the first token that differs, counted from 0), and each
               load's two rounds alike
"""
import csv
import json
import os
import re
import sys

RUN, OUT = sys.argv[1], sys.argv[2]
BUILD = {"new": "branch c9c3134d4", "ref": "base 602691153"}
HEAD = {"own": "Swift-1.5 MTP head (block ratio 4)", "dist": "distilled head (block ratio 0, shared embd,head)",
        "plain": "none (plain decoding)"}
GREEDY = ("code", "prose", "deep")


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def mem(path):
    rows = [r.split(",") for r in read(path).strip().splitlines() if r.strip()]
    return " ".join(r[1].strip() for r in rows) if rows else ""


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


loads, replies, by = [], [], {}
for d in sorted(os.listdir(RUN)):
    p = os.path.join(RUN, d)
    if not os.path.isdir(p) or d.count("-") < 2:
        continue
    model, head, build = d.split("-", 2)
    log = read(os.path.join(p, "server.log"))
    up = read(os.path.join(p, "load.txt")).strip().replace("up=", "")
    rc = read(os.path.join(p, "server.rc")).strip().replace("rc=", "")
    asserts = sorted({re.sub(r"^.*/", "", m.group(1)).strip()
                      for m in re.finditer(r"(\S*:\d+: GGML_ASSERT\([^)]*\) failed)", log)})
    loads.append([model, head, BUILD[build], HEAD[head], "yes" if up == "1" else "no", rc,
                  log.count("creating indexer KV cache"),
                  len(re.findall(r"kpool_dev_init: .*\(MTP draft\)", log)),
                  len(re.findall(r"pool_mask_dev_init: .*\(MTP draft\)", log)),
                  "; ".join(asserts), mem(os.path.join(p, "mem-load.csv")), mem(os.path.join(p, "mem.csv"))])
    for line in read(os.path.join(p, "session.tsv")).splitlines():
        f = line.split("\t")
        if len(f) < 9:
            continue
        rnd, req, pn, ntok, ids, tmd5, dn, da, stop = f[:9]
        replies.append([model, head, BUILD[build], rnd, req, pn, ntok, ids, tmd5, dn, da, stop])
        try:
            with open(os.path.join(p, f"{rnd}-{req}.json"), encoding="utf-8") as fj:
                toks = json.load(fj).get("tokens") or []
        except (OSError, ValueError):
            toks = []
        by[(model, f"{head}-{build}", rnd, req)] = (ids, dn, da, toks)

write("loads.tsv", ["model", "draft", "build", "draft head", "answered", "server exit", "indexer cache lines",
                    "draft k-pool table lines", "draft pool mask lines", "assertion", "MiB per card after load",
                    "MiB per card after requests"], loads)
write("replies.tsv", ["model", "draft", "build", "round", "request", "prompt tokens", "tokens", "token_ids_md5",
                      "text_md5", "drafted", "accepted", "stop"], replies)


def same(a, b, i=0):
    if a is None or b is None or not a[0] or not b[0]:
        return "n/a"
    return "yes" if a[i] == b[i] else "NO"


def first_diff(a, b):  # "yes", or "no, from token N" (the first token that differs, counted from 0)
    if a is None or b is None or not a[3] or not b[3]:
        return "n/a"
    for i, (x, y) in enumerate(zip(a[3], b[3])):
        if x != y:
            return f"no, from token {i}"
    return "yes" if len(a[3]) == len(b[3]) else f"no, from token {min(len(a[3]), len(b[3]))}"


agree = []
for model in sorted({k[0] for k in by}):
    for rnd in sorted({k[2] for k in by if k[0] == model}):
        for req in ("code", "prose", "deep", "deep-s", "code-s"):
            g = lambda arm: by.get((model, arm, rnd, req))
            on, orf, dn, pn = g("own-new"), g("own-ref"), g("dist-new"), g("plain-new")
            drafts = "n/a" if not on or not orf else ("yes" if on[1:3] == orf[1:3] else "NO")
            r2 = [same(by.get((model, arm, "r1", req)), by.get((model, arm, "r2", req)))
                  for arm in ("own-new", "own-ref", "dist-new", "plain-new")] if rnd == "r1" else [""] * 4
            agree.append([model, rnd, req, "greedy" if req in GREEDY else "sampled, seed 7", first_diff(on, orf), drafts,
                          f"{on[1]}/{on[2]}" if on else "", f"{orf[1]}/{orf[2]}" if orf else "", first_diff(dn, on),
                          f"{dn[1]}/{dn[2]}" if dn else "", first_diff(on, pn), first_diff(dn, pn)] + r2)
write("agree.tsv", ["model", "round", "request", "sampling", "own: branch = base ids", "own: branch = base drafts",
                    "own branch drafted/accepted", "own base drafted/accepted", "dist = own ids",
                    "dist drafted/accepted", "own = plain ids", "dist = plain ids", "r1 = r2 own-new",
                    "r1 = r2 own-ref", "r1 = r2 dist-new", "r1 = r2 plain-new"], agree)
for r in loads:
    print("\t".join(str(x) for x in r))
for r in agree:
    print("\t".join(str(x) for x in r))
