#!/usr/bin/env python3
"""C1128 (Q1861): C964's three requests as verify-drift's inputs: each prompt as a text file (the exact string C964's
run/session.py sent to /completion) and plain decoding's reply as comma-separated token ids (C964's tokens.tsv, arm
plain, pass p1).

Usage: python3 -I make_inputs.py <C964 data folder> <docs/switches.md at commit 6a9cf9d9> <output folder>
Writes <out>/<request>.prompt.txt and <out>/<request>.ids for code, prose and deep, and prints verify-drift's KLD_REQS
for the folder. The deep prompt is the first 170 lines of the text plus C964's question; its md5 is checked against
C964's (dea4956d2c52140e5ebfedf0dd46c770).
"""
import hashlib
import os
import sys

C964, TEXT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
P_CODE = ("Write a Python function that parses an ISO 8601 date string into a datetime object, "
          "then three pytest tests for it.")
P_PROSE = "Explain in one paragraph how a hash map handles collisions, then give one example in Python."
QUESTION = ("\n\nQuestion: list the switches above that change how the MTP draft runs, "
            "one line each with what it does.\nAnswer:\n")
DEEP_LINES, DEEP_MD5 = 170, "dea4956d2c52140e5ebfedf0dd46c770"

with open(TEXT, encoding="utf-8") as f:
    deep = "".join(f.read().splitlines(keepends=True)[:DEEP_LINES]) + QUESTION
if hashlib.md5(deep.encode("utf-8")).hexdigest() != DEEP_MD5:
    sys.exit("the deep prompt's md5 is not C964's: use docs/switches.md at commit 6a9cf9d9")
prompts = {"code": P_CODE, "prose": P_PROSE, "deep": deep}

ids = {}
with open(os.path.join(C964, "tokens.tsv"), encoding="utf-8") as f:
    next(f)
    for line in f:
        arm, req, toks = line.rstrip("\n").split("\t")
        if arm == "plain":
            ids[req] = toks
os.makedirs(OUT, exist_ok=True)
reqs = []
for name in ("code", "prose", "deep"):
    if len(ids[name].split(",")) != 256:
        sys.exit(f"plain decoding's {name} reply does not hold 256 token ids")
    pf, idf = os.path.join(OUT, f"{name}.prompt.txt"), os.path.join(OUT, f"{name}.ids")
    with open(pf, "w", encoding="utf-8", newline="") as f:
        f.write(prompts[name])
    with open(idf, "w", encoding="utf-8") as f:
        f.write(ids[name])
    reqs.append(f"{name}={pf}={idf}")
print(";".join(reqs))
