#!/usr/bin/env python3
"""upstream-b12723: kernels of one family compared by their machine code alone, whatever their names (standard library).

A template argument added upstream renames every instance of a kernel (gated_delta_net_cuda gained `replay`,
rms_norm_mul_rope_f32 gained `supports_mrope`), so sass_compare.py, which matches kernels by name, sees the base's
instances as gone and the head's as new. This script hashes each instance's listing without its "Function :" line and
says, per family, how many of the base's listings the head holds byte for byte, and lists the rest.

Usage: python3 -I sass_bodies.py <base.sass> <head.sass> <family regex> [<family regex> ...]
"""
import collections
import hashlib
import re
import sys


def bodies(path, fam):
    out, name, buf = [], None, []
    def flush():
        if name is not None and fam.search(name):
            out.append((name, hashlib.md5("\n".join(buf).encode()).hexdigest()))
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


for pat in sys.argv[3:]:
    fam = re.compile(pat)
    b, h = bodies(sys.argv[1], fam), bodies(sys.argv[2], fam)
    hc = collections.Counter(x for _, x in h)
    left = collections.Counter(hc)
    missing = []
    for name, x in b:
        if left[x] > 0:
            left[x] -= 1
        else:
            missing.append(name)
    print(f"{pat}: base {len(b)} instances, head {len(h)}; base listings found byte for byte in the head: "
          f"{len(b) - len(missing)} of {len(b)}; head listings with no base twin: {sum(left.values())}")
    for name in missing:
        print(f"  base only: {name}")
    for name, x in h:
        if left[x] > 0:
            left[x] -= 1
            print(f"  head only: {name}")
