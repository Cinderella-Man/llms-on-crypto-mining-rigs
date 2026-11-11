#!/usr/bin/env python3
"""Which build each GPU function of a merged build comes from.

Usage:
  sass_attrib.py dump sass|ptx < <cuobjdump -sass|-ptx listing>   > <funcs.tsv>
  sass_attrib.py attrib <merged funcs.tsv> <base funcs.tsv> <label>=<branch funcs.tsv>[@<parent label>] ...

dump cuts a listing into functions as C103's funcs.py does (docs/data/2026-11-11_c103_union_gather/run/funcs.py: "Function
: <name>" in SASS, ".entry"/".func" in PTX, PTX's per-module label and stack numbers dropped, runs of spaces as one) and
writes one line per function: its name and the sorted md5s of its copies.

attrib compares the merged build with the base and with each branch built on that base (or on another listed branch, its
parent), function by function. A branch changed a function when its code differs from its parent's (else the base's) or
that lacks it. Each function of the merged build must be:
  base      the base's code, and no branch changed it;
  <label>   the code of the branch that changed it last (the branch's code, which its own checks examined): one branch,
            or a branch and the parents it was built on;
and every function of the base and of each branch must be in the merged build, but for a function of the base that a
branch removed (removed). Anything else is listed: a function two branches changed (MERGED: its code is neither's), one whose code is no build's (NEW), one the merged build lacks
(MISSING), one a branch changed that the merged build has as the base's (LOST). Prints the counts per class, the lists,
and "verdict: OK" when nothing is listed, else "verdict: CHECK".
"""
import hashlib
import re
import sys
from collections import defaultdict

LABEL = re.compile(r"\$L__BB\d+_")
DEPOT = re.compile(r"__local_depot\d+")


def split(lines, kind):
    head = re.compile(r"^\s*Function : (\S+)") if kind == "sass" else \
        re.compile(r"^\s*(?:\.visible\s+|\.weak\s+|\.extern\s+)*\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)")
    funcs = defaultdict(list)
    name, body = None, []

    def flush():
        if name is not None:
            text = "\n".join(" ".join(x.split()) for x in body)
            if kind == "ptx":
                text = DEPOT.sub("__local_depot", LABEL.sub("$L__BB_", text))
            funcs[name].append(hashlib.md5(text.encode()).hexdigest())

    for line in lines:
        line = line.rstrip("\n")
        m = head.match(line)
        if m and not (kind == "ptx" and (line.rstrip().endswith(";") or ".extern" in line)):
            flush()
            name, body = m.group(1), [line]
        elif name is not None:
            if kind == "sass" and (line.startswith("Fatbin ") or re.match(r"^\s*\.{5,}", line)):
                flush()
                name, body = None, []
                continue
            if kind == "ptx":
                if re.match(r"^\s*\.version\s", line):
                    flush()
                    name, body = None, []
                    continue
                if not line.strip() or line.lstrip().startswith("//"):
                    continue
            body.append(line)
    flush()
    return {k: ",".join(sorted(v)) for k, v in funcs.items()}


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            out[k] = v
    return out


def main():
    if sys.argv[1] == "dump":
        for k, v in sorted(split(sys.stdin, sys.argv[2]).items()):
            print(f"{k}\t{v}")
        return
    merged, base = load(sys.argv[2]), load(sys.argv[3])
    branches, parent = [], {}
    for arg in sys.argv[4:]:
        label, path = arg.split("=", 1)
        if "@" in path:
            path, parent[label] = path.split("@", 1)
        branches.append((label, load(path)))
    table = dict(branches)

    def ancestors(label):
        out = []
        while label in parent:
            label = parent[label]
            out.append(label)
        return out
    # the functions each branch changed, and those of the base it no longer has
    changed, removed = defaultdict(list), defaultdict(list)
    for label, b in branches:
        p = table[parent[label]] if label in parent else base
        for k, v in b.items():
            if p.get(k) != v:
                changed[k].append(label)
        for k in p:
            if k not in b:
                removed[k].append(label)
    counts = defaultdict(int)
    listed = []
    for k, v in sorted(merged.items()):
        by = changed.get(k, [])
        if not by:
            if base.get(k) == v:
                counts["base"] += 1
            else:
                listed.append(("NEW", k, "no branch changed it; not the base's code"))
            continue
        same = [lbl for lbl in by if table[lbl].get(k) == v]
        last = max(same, key=lambda lbl: len(ancestors(lbl))) if same else None
        if last and all(x == last or x in ancestors(last) for x in by):
            counts[last] += 1
        elif same:
            listed.append(("MERGED", k, f"changed by {' '.join(by)}; the code of {' '.join(same)}"))
        elif base.get(k) == v:
            listed.append(("LOST", k, f"changed by {' '.join(by)}; the merged build has the base's code"))
        else:
            listed.append(("MERGED", k, f"changed by {' '.join(by)}; the code of none of them"))
    for k in sorted(set(base) | set(changed)):
        if k not in merged and k in removed and k not in changed:
            counts["removed"] += 1
        elif k not in merged:
            listed.append(("MISSING", k, f"in {'the base' if k in base else ''} {' '.join(changed.get(k, []))}".strip()))
    print(f"merged {len(merged)} functions: " + ", ".join(f"{n} {c}" for n, c in sorted(counts.items())) +
          f"; listed {len(listed)}")
    for label, b in branches:
        n = sum(1 for k in changed if label in changed[k])
        print(f"  {label}: changed {n} functions, {counts.get(label, 0)} of them in the merged build with its code")
    for cls, k, why in listed:
        print(f"  {cls}: {k}: {why}")
    print("verdict: " + ("OK" if not listed else "CHECK"))


if __name__ == "__main__":
    main()
