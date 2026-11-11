#!/usr/bin/env python3
# R104 (Q1349): C762's switch off and on, side by side, from R81's tables of each load (r81_read.py's
# read07/prompt_sums.tsv and read10/prompt_sums.tsv) and the loads' own files (R43's load-A/: memory_loaded.txt,
# requests.tsv, server.log). Prints Markdown: the draft's catch-up passes per card (wait w, copies c, launch l, total),
# the requests (prompt ms and the reply's md5), the memory per card after loading, and the load's buffer lines.
# Usage: python3 -I r104_compare.py <rigrun dir with out-c762off/ and out-c762on/> <tables dir with off/ and on/>
import csv
import os
import re
import sys

LOADS = ("off", "on")


def load_dir(rigrun, l):
    return os.path.join(rigrun, f"out-c762{l}", "warm4096", "load-A")


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sums(tables, l, req):
    rows = read_tsv(os.path.join(tables, l, f"read{req}", "prompt_sums.tsv"))
    return {r["part"]: r for r in rows if r["part"].startswith("draft call of")}


def main():
    rigrun, tables = sys.argv[1], sys.argv[2]
    print("# R104: C762's LLAMA_DRAFT_EMBD_DEV, off and on\n")
    for req in ("07", "10"):
        s = {l: sums(tables, l, req) for l in LOADS}
        parts = []
        for l in LOADS:
            for p in s[l]:
                if p not in parts:
                    parts.append(p)
        print(f"## The draft's calls in request {int(req)} (µs; n, sum, median, min, max)\n")
        print("| Part | off n | off sum | off median | off min | off max | on n | on sum | on median | on min | on max |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in parts:
            cells = []
            for l in LOADS:
                r = s[l].get(p, {})
                cells += [r.get(k, "") or "" for k in ("n", "sum_us", "median_us", "min_us", "max_us")]
            print(f"| {p} | " + " | ".join(cells) + " |")
        print()

    print("## Requests (R43's requests.tsv): prompt ms and the reply's md5\n")
    req = {l: read_tsv(os.path.join(load_dir(rigrun, l), "requests.tsv")) for l in LOADS}
    print("| # | kind | task | arm | prompt_n | off prompt_ms | on prompt_ms | off gen_md5 | on gen_md5 | same reply |")
    print("|---:|---|---|---|---:|---:|---:|---|---|---|")
    n_same = 0
    for i, (a, b) in enumerate(zip(req["off"], req["on"])):
        same = a.get("gen_md5") == b.get("gen_md5")
        n_same += same
        print(f"| {i} | {a.get('kind', '')} | {a.get('task', '')} | {a.get('arm', '')} | {a.get('prompt_n', '')} | "
              f"{a.get('prompt_ms', '')} | {b.get('prompt_ms', '')} | {a.get('gen_md5', '')} | {b.get('gen_md5', '')} | "
              f"{'yes' if same else 'NO'} |")
    print(f"\n{n_same} of {min(len(req['off']), len(req['on']))} replies the same\n")

    print("## Memory per card after loading (MiB, nvidia-smi)\n")
    mem = {}
    for l in LOADS:
        mem[l] = {}
        path = os.path.join(load_dir(rigrun, l), "memory_loaded.txt")
        if os.path.exists(path):
            for line in open(path):
                m = re.match(r"\s*(\d+),\s*(\d+) MiB,\s*(\d+) MiB", line)
                if m:
                    mem[l][int(m.group(1))] = int(m.group(2))
    print("| Card | off | on | on − off |")
    print("|---:|---:|---:|---:|")
    for c in sorted(set(mem["off"]) | set(mem["on"])):
        a, b = mem["off"].get(c), mem["on"].get(c)
        d = f"{b - a:+d}" if a is not None and b is not None else ""
        print(f"| {c} | {a if a is not None else ''} | {b if b is not None else ''} | {d} |")
    print()

    print("## The load's buffer lines\n")
    pat = re.compile(r"(model buffer size|compute buffer size|KV buffer size|h_nextn buffer size|LLAMA_DRAFT_EMBD_DEV)")
    for l in LOADS:
        print(f"{l}:\n```")
        path = os.path.join(load_dir(rigrun, l), "server.log")
        if os.path.exists(path):
            for line in open(path, errors="replace"):
                if pat.search(line):
                    print(re.sub(r"^[0-9.]+ [A-Z] ", "", line.rstrip()))
        print("```\n")


if __name__ == "__main__":
    main()
