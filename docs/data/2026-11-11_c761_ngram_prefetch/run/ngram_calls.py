#!/usr/bin/env python3
"""C761 (Q1339): one server load with the n-gram row prefetch off against one with it on, call by call.

Usage: python3 -I ngram_calls.py <off server.log> <on server.log> <out.tsv>

Both logs come from llama-server with LLAMA_UBATCH_TRACE=1 on the same requests, the second with LLAMA_NGRAM_PREFETCH=1.
Each `ngram-prefetch: call` line is one decode call of the main model whose n-gram input ran: its tokens, micro-batches,
rows issued (by the helper and by set_input), rows whose helper copy differed from set_input's, the distinct pages of
the rows, the bytes passed to madvise, an order-free digest of the rows, the most bytes the helper had issued ahead
and the cap. Writes one row per call with both loads' fields and prints the verdict: the same calls, each with the same
rows, pages and digest; the on load's helper issued rows, none differed, and the window stayed within the cap. Exit
status 1 when any of that fails.
"""
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"ngram-prefetch: call n_tokens = (\d+) ubatches = (\d+) (\w+) rows = (\d+) helper = (\d+) inline = (\d+) "
                  r"mismatch = (\d+) pages = (\d+) bytes = (\d+) madv_bytes = (\d+) digest = ([0-9a-f]+) window_max = (\d+) "
                  r"cap = (\d+) helper_us = (-?\d+) inline_us = (-?\d+)")
FIELDS = ["n_tokens", "ubatches", "mode", "rows", "helper", "inline", "mismatch", "pages", "bytes", "madv_bytes", "digest",
          "window_max", "cap", "helper_us", "inline_us"]


def calls(path):
    out = []
    for raw in open(path, encoding="utf-8", errors="replace"):
        m = LINE.search(ANSI.sub("", raw))
        if m:
            out.append(dict(zip(FIELDS, m.groups())))
    return out


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    off, on = calls(sys.argv[1]), calls(sys.argv[2])
    ok = True
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        f.write("\t".join(["call"] + [f"off_{k}" for k in FIELDS] + [f"on_{k}" for k in FIELDS] + ["same"]) + "\n")
        for i in range(max(len(off), len(on))):
            a = off[i] if i < len(off) else {k: "" for k in FIELDS}
            b = on[i] if i < len(on) else {k: "" for k in FIELDS}
            same = all(a[k] == b[k] for k in ["n_tokens", "ubatches", "rows", "pages", "bytes", "digest"])
            f.write("\t".join([str(i)] + [a[k] for k in FIELDS] + [b[k] for k in FIELDS] + [str(int(same))]) + "\n")
    n_same = sum(1 for i in range(min(len(off), len(on)))
                 if all(off[i][k] == on[i][k] for k in ["n_tokens", "ubatches", "rows", "pages", "bytes", "digest"]))
    if not off or len(off) != len(on) or n_same != len(off):
        print(f"ngram calls: {len(off)} off, {len(on)} on, {n_same} the same (rows, pages, digest)")
        ok = False
    else:
        print(f"ngram calls: {len(off)} calls, each with the same tokens, micro-batches, rows, pages and digest off and on")
    rows_off = sum(int(c["rows"]) for c in off)
    rows_on = sum(int(c["rows"]) for c in on)
    helper = sum(int(c["helper"]) for c in on)
    inline = sum(int(c["inline"]) for c in on)
    mismatch = sum(int(c["mismatch"]) for c in on)
    planned = sum(1 for c in on if c["mode"] == "planned")
    pages_off = sum(int(c["pages"]) for c in off)
    pages_on = sum(int(c["pages"]) for c in on)
    wmax = max((int(c["window_max"]) for c in on), default=0)
    cap = max((int(c["cap"]) for c in on), default=0)
    over = sum(1 for c in on if int(c["window_max"]) > int(c["cap"]))
    off_helper = sum(int(c["helper"]) for c in off)
    print(f"rows: off {rows_off} (helper {off_helper}), on {rows_on} = helper {helper} + inline {inline}; "
          f"{mismatch} rows differed; {planned} calls planned; pages off {pages_off}, on {pages_on}; "
          f"window max {wmax} of cap {cap} bytes, {over} calls over the cap")
    if off_helper != 0 or helper == 0 or mismatch != 0 or over != 0 or rows_on != helper + inline or rows_off != rows_on:
        ok = False
    print("ngram calls: PASS" if ok else "ngram calls: FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
