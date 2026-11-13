#!/usr/bin/env python3
"""The GPU functions of two builds compared by name, for an upstream merge.

Usage: sass_diff.py <new funcs.tsv> <base funcs.tsv> [<out.tsv>]

Both files are sass_attrib.py dump's output (docs/data/2026-11-11_v4_day_merge/run/sass_attrib.py: one line per function,
its name and the sorted md5s of its copies, anonymous-namespace hashes masked). Every function is one of:
  same     in both builds, the same code
  changed  in both builds, different code
  new      only in the new build
  gone     only in the base build
Prints the counts, then per kernel family (the demangled name up to its template or argument list, with c++filt when
it is on the PATH) the counts of each class that is not "same", most changed first. With <out.tsv> also writes one line
per function that is not "same": class, family, mangled name.
"""
import shutil
import subprocess
import sys
from collections import Counter, defaultdict


def load(path):
    funcs = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, _, md5s = line.partition("\t")
            funcs[name] = md5s
    return funcs


def demangle(names):
    if not names or shutil.which("c++filt") is None:
        return {n: n for n in names}
    out = subprocess.run(["c++filt"], input="\n".join(names) + "\n", capture_output=True, text=True).stdout.split("\n")
    return {n: (out[i] if i < len(out) and out[i] else n) for i, n in enumerate(names)}


def family(d):
    # "void mul_mat_q<(ggml_type)12, 64, false>(...)" -> "mul_mat_q"; strip a leading return type
    cut = len(d)
    for ch in "<(":
        i = d.find(ch)
        if i >= 0:
            cut = min(cut, i)
    head = d[:cut].strip()
    return head.split(" ")[-1] if head else d


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    new, base = load(sys.argv[1]), load(sys.argv[2])
    cls = {}
    for n in set(new) | set(base):
        if n in new and n in base:
            cls[n] = "same" if new[n] == base[n] else "changed"
        elif n in new:
            cls[n] = "new"
        else:
            cls[n] = "gone"
    counts = Counter(cls.values())
    print(f"functions: new build {len(new)}, base {len(base)}; same {counts['same']}, changed {counts['changed']}, "
          f"new {counts['new']}, gone {counts['gone']}")
    diff = sorted(n for n, c in cls.items() if c != "same")
    dm = demangle(diff)
    fam = defaultdict(Counter)
    for n in diff:
        fam[family(dm[n])][cls[n]] += 1
    for f, c in sorted(fam.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
        print(f"  {f}: " + ", ".join(f"{k} {c[k]}" for k in ("changed", "new", "gone") if c[k]))
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            f.write("class\tfamily\tname\n")
            for n in diff:
                f.write(f"{cls[n]}\t{family(dm[n])}\t{n}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
