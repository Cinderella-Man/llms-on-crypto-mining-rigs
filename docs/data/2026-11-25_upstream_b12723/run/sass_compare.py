#!/usr/bin/env python3
"""upstream-b12723 (from land-v5-batch3): two cuobjdump -sass listings compared kernel by kernel (standard library only).

Usage: python3 -I sass_compare.py <base.sass> <head.sass> <out.tsv>
A kernel is the text from its "Function : <name>" line to the next such line; kernels are matched by name and the
order of copies with that name. Per kernel: the same bytes, or differing in how many lines; per family (the name up to
its first template argument or digit run) the counts go to out.tsv.
"""
import collections
import hashlib
import re
import sys


def kernels(path):
    out, name, buf, seen = {}, None, [], collections.Counter()
    def flush():
        if name is not None:
            seen[name] += 1
            out[(name, seen[name])] = buf[:]
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*Function : (\S+)", line)
            if m:
                flush()
                name, buf = m.group(1), []
            elif name is not None:
                buf.append(line.rstrip("\n"))
    flush()
    return out


def base_name(name):
    """the kernel's unqualified function name from its mangled name: _Z<len><id>... or _ZN<len><id>...E..."""
    if not name.startswith("_Z"):
        return name
    i, parts = 2, []
    nested = name[i:i + 1] == "N"
    if nested:
        i += 1
    while i < len(name):
        if name[i] == "L":  # internal linkage
            i += 1
            continue
        m = re.match(r"(\d+)", name[i:])
        if not m:
            break
        n = int(m.group(1))
        i += len(m.group(1))
        parts.append(name[i:i + n])
        i += n
        if not nested:
            break
    return "::".join(parts) or name


def family(name):
    return base_name(name)[:80]


b, h = kernels(sys.argv[1]), kernels(sys.argv[2])
rows = collections.Counter()
diff_lines = collections.defaultdict(list)
differing = []
for k in sorted(set(b) | set(h)):
    fam = family(k[0])
    if k not in h:
        rows[(fam, "only in base")] += 1
    elif k not in b:
        rows[(fam, "only in head")] += 1
    elif hashlib.md5("\n".join(b[k]).encode()).digest() == hashlib.md5("\n".join(h[k]).encode()).digest():
        rows[(fam, "same bytes")] += 1
    else:
        n = sum(1 for x, y in zip(b[k], h[k]) if x != y) + abs(len(b[k]) - len(h[k]))
        rows[(fam, "differ")] += 1
        diff_lines[fam].append(n)
        differing.append((k[0], k[1], n))
with open(sys.argv[3], "w", encoding="utf-8") as f:
    f.write("kernel family\tverdict\tkernels\tlines differing (min, max)\n")
    for (fam, v), n in sorted(rows.items()):
        d = diff_lines.get(fam) if v == "differ" else None
        f.write(f"{fam}\t{v}\t{n}\t{f'{min(d)}, {max(d)}' if d else '-'}\n")
with open(sys.argv[3].replace(".tsv", "_differing.tsv"), "w", encoding="utf-8") as f:
    f.write("kernel (mangled)\tcopy\tlines differing\n")
    for name, copy, n in sorted(differing):
        f.write(f"{name}\t{copy}\t{n}\n")
tot = collections.Counter()
for (fam, v), n in rows.items():
    tot[v] += n
print(f"kernels: base {len(b)}, head {len(h)}; " + ", ".join(f"{v} {n}" for v, n in sorted(tot.items())))
for (fam, v), n in sorted(rows.items()):
    if v != "same bytes":
        d = diff_lines.get(fam) if v == "differ" else None
        print(f"  {fam}: {v} {n}" + (f" (lines differing {min(d)} to {max(d)})" if d else ""))
