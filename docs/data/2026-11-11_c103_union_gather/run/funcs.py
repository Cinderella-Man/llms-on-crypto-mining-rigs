#!/usr/bin/env python3
"""Compares two cuobjdump listings (-sass or -ptx) of the same binary from two builds, kernel by kernel.

Usage: funcs.py sass|ptx <base listing> <branch listing>

Each listing is cut into functions ("Function : <name>" in SASS, ".entry"/".func" in PTX; ".extern" declarations are
not functions); a name that appears in several translation units (static kernels of a shared header) keeps all its
copies, compared as a sorted list. PTX numbers its branch labels and local stack arrays by the function's place in the
module ($L__BB<function>_<block>, __local_depot<function>), so a function added to a module renumbers those of the
ones after it: the function number is dropped before comparing; a PTX function ends at the next module's ".version", and comment and blank lines
are left out. Runs of spaces count as one (cuobjdump pads its SASS columns to a width that differs between builds). Prints, per listing, how many of the base's functions are the same, which differ or
are missing, the functions only the branch has, and a verdict line: "verdict: SAME" when every function of the base is
the same in the branch, "verdict: DIFFERS" otherwise.
"""
import hashlib
import re
import sys
from collections import defaultdict

LABEL = re.compile(r"\$L__BB\d+_")
DEPOT = re.compile(r"__local_depot\d+")


def split(path, kind):
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

    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = head.match(line)
            # PTX: a declaration (".extern .func name(...);", possibly over several lines) is not a body
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
    return {k: sorted(v) for k, v in funcs.items()}


def main():
    kind, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
    base, branch = split(a, kind), split(b, kind)
    same, differ, missing = [], [], []
    for k, v in base.items():
        if k not in branch:
            missing.append(k)
        elif branch[k] == v:
            same.append(k)
        else:
            differ.append(k)
    new = sorted(k for k in branch if k not in base)
    nb = sum(len(v) for v in base.values())
    print(f"{kind}: base {len(base)} functions ({nb} copies), branch {len(branch)}; same {len(same)}, "
          f"differ {len(differ)}, missing {len(missing)}, new {len(new)}")
    sparse = [k for k in same if "flash_attn_ext_sparse_pascal" in k or "flash_attn_mask_to_sparse_indices" in k]
    for k in sorted(sparse):
        print(f"  same (sparse path): {k}")
    for k in sorted(differ):
        print(f"  DIFFERS: {k}")
    for k in sorted(missing):
        print(f"  MISSING: {k}")
    for k in new:
        print(f"  new: {k}")
    print("verdict: " + ("SAME" if not differ and not missing else "DIFFERS"))


if __name__ == "__main__":
    main()
