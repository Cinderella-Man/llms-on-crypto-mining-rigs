#!/usr/bin/env python3
"""C964 (Q1694): the tables of this folder from the run's output (c964.sh's WORK, copied back).

Usage: python3 -I c964_tables.py <WORK copy> <this folder> [C951's replies.tsv] [C951's out/ folder]
  <WORK copy>     holds mtp/ and plain/ as c964.sh wrote them
  replies.tsv     docs/data/2026-11-21_c951_flash_head_gpu/replies.tsv: the 3090 replies' md5s (file "maker", round r1)
  C951's out/     C951's raw output (maker-own-new/, maker-plain-new/ with r1-<request>.json): first differing tokens
                  against the 3090's ids; without it only the md5s are compared
Writes replies.tsv (one line per arm, pass and request), tokens.tsv (the generated token ids of pass p1),
compare.tsv (per request: the draft arm against plain decoding, the first differing token and both arms' top two
candidates there, p1 = p2) and vs3090.tsv (per request and arm: the same token ids as C951's 3090 arm, or the first
token that differs).
"""
import hashlib
import json
import math
import os
import sys

WORK, OUT = sys.argv[1], sys.argv[2]
C951_TSV = sys.argv[3] if len(sys.argv) > 3 else ""
C951_OUT = sys.argv[4] if len(sys.argv) > 4 else ""
ARMS = ("mtp", "plain")
REQS = ("code", "prose", "deep")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def tsv(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def first_diff(a, b):  # the first index (from 0) where two id lists differ; None when the same
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def top2(reply, i):  # the two most likely tokens at generated position i: "id piece p", from n_probs 2
    cp = (reply or {}).get("completion_probabilities") or []
    if i >= len(cp):
        return ["n/a", "n/a"]
    ent = cp[i]
    cands = ent.get("top_logprobs") or ent.get("top_probs") or ent.get("probs") or []
    out = []
    for c in cands[:2]:
        if "logprob" in c:
            p = math.exp(c["logprob"])
        else:
            p = c.get("prob", float("nan"))
        out.append(f"{c.get('id', '?')} {json.dumps(c.get('token', ''))} {p:.4f}")
    while len(out) < 2:
        out.append("n/a")
    return out


R = {(a, p, q): load(os.path.join(WORK, a, f"{p}-{q}.json")) for a in ARMS for p in ("p1", "p2") for q in REQS}
ids = {k: (v or {}).get("tokens") or [] for k, v in R.items()}

rows = [["arm", "pass", "request", "prompt tokens", "tokens", "token_ids_md5", "text_md5", "drafted", "accepted",
         "stop"]]
for a in ARMS:
    path = os.path.join(WORK, a, "session.tsv")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                rows.append([a] + c)
tsv("replies.tsv", rows)

tsv("tokens.tsv", [["arm", "request", "token ids (pass p1)"]]
    + [[a, q, ",".join(str(x) for x in ids[(a, "p1", q)])] for a in ARMS for q in REQS])

rows = [["request", "mtp = plain (p1)", "first differing token (from 0)", "mtp's token there", "plain's token there",
         "mtp: top two there (p2)", "plain: top two there (p2)", "mtp p1 = p2", "plain p1 = p2"]]
for q in REQS:
    m, p = ids[("mtp", "p1", q)], ids[("plain", "p1", q)]
    same12 = ["yes" if ids[(a, "p1", q)] == ids[(a, "p2", q)] and ids[(a, "p1", q)] else "NO" for a in ARMS]
    if not m or not p:
        rows.append([q, "n/a", "", "", "", "", ""] + same12)
        continue
    d = first_diff(m, p)
    if d is None:
        rows.append([q, "yes", "", "", "", "", ""] + same12)
        continue
    mt = R[("mtp", "p2", q)] if same12[0] == "yes" else None
    pt = R[("plain", "p2", q)] if same12[1] == "yes" else None
    rows.append([q, "no", d, m[d] if d < len(m) else "end", p[d] if d < len(p) else "end",
                 " | ".join(top2(mt, d)), " | ".join(top2(pt, d))] + same12)
tsv("compare.tsv", rows)

c951 = {}
if C951_TSV:
    with open(C951_TSV) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > 7 and c[0] == "maker" and c[3] == "r1" and c[2].startswith("branch"):
                c951[(c[1], c[4])] = c[7]
rows = [["request", "arm (P106)", "C951 arm (3090)", "token ids md5 equal", "first differing token (from 0)"]]
for q in REQS:
    for a, b in (("mtp", "own"), ("plain", "plain")):
        md5 = hashlib.md5(",".join(str(x) for x in ids[(a, "p1", q)]).encode()).hexdigest()
        eq = "n/a" if (b, q) not in c951 else ("yes" if c951[(b, q)] == md5 else "no")
        fd = ""
        if C951_OUT:
            other = load(os.path.join(C951_OUT, f"maker-{b}-new", f"r1-{q}.json"))
            if other and other.get("tokens") and ids[(a, "p1", q)]:
                d = first_diff(ids[(a, "p1", q)], other["tokens"])
                fd = "same" if d is None else str(d)
        rows.append([q, a, b + " (branch c9c3134d4)", eq, fd])
tsv("vs3090.tsv", rows)


def mib(path):  # "MiB per card" from nvidia-smi's index, used, total lines
    try:
        with open(path) as f:
            return " ".join(line.split(",")[1].strip() for line in f if line.strip())
    except OSError:
        return "n/a"


rows = [["arm", "load", "server exit", "MiB per card after the load (cards 0-8)", "MiB per card after the requests"]]
for a in ARMS:
    def read(name):
        try:
            with open(os.path.join(WORK, a, name)) as f:
                return f.read().strip()
        except OSError:
            return "n/a"
    rows.append([a, read("load.txt"), read("server.rc"), mib(os.path.join(WORK, a, "mem-load.csv")),
                 mib(os.path.join(WORK, a, "mem.csv"))])
tsv("loads.tsv", rows)
for name in ("compare.tsv", "vs3090.tsv", "loads.tsv"):
    with open(os.path.join(OUT, name)) as f:
        print(f.read())
