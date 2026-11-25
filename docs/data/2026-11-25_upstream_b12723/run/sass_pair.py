#!/usr/bin/env python3
"""upstream-b12723: one kernel of the base against one of the head (named in full), instruction by instruction.

Usage: python3 -I sass_pair.py <base.sass> <head.sass> <base kernel name> <head kernel name> [-q]
Prints the instruction counts, how many instructions differ once each line's address comment and encoding are left
out, how many still differ when the kernel parameters' constant-bank offsets (c[0x0][0x...]) are left out too (an
argument added before them moves those offsets), and with no -q the differing instructions side by side (up to 40).
"""
import re
import sys


def listing(path, want):
    out, on = [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*Function : (\S+)", line)
            if m:
                if on:
                    break
                on = m.group(1) == want
                continue
            if on:
                # "/*0040*/  IMAD ... ;  /* 0x... */": the instruction text alone
                t = re.sub(r"/\*[0-9a-f]{4,}\*/", "", line)
                t = re.sub(r"/\* 0x[0-9a-f]+ \*/", "", t).strip()
                if t and not t.startswith("/*") and not t.startswith(".") and ";" in t:
                    out.append(re.sub(r"\s+", " ", t))
    return out


b, h = listing(sys.argv[1], sys.argv[3]), listing(sys.argv[2], sys.argv[4])
diff = [(i, x, y) for i, (x, y) in enumerate(zip(b, h)) if x != y]
par = re.compile(r"c\[0x0\] ?\[0x[0-9a-f]+\]")
diff_np = [(i, x, y) for i, x, y in diff if par.sub("c[0x0][P]", x) != par.sub("c[0x0][P]", y)]
print(f"{sys.argv[4]}\tinstructions base {len(b)}, head {len(h)}\tdiffering {len(diff)}\t"
      f"differing past parameter offsets {len(diff_np)}")
if "-q" not in sys.argv[5:]:
    for i, x, y in diff[:40]:
        print(f"  {i:5d}  {x}\n         {y}")
