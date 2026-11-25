#!/usr/bin/env python3
"""Static instruction counts of the inner loops of one kernel's sm_61 SASS (cuobjdump -sass output, one function).

A loop is a backward branch (BRA to a lower address) and the instructions between its target and the branch. The
script prints every loop that holds DP4A instructions (IDP.4A) and has no such loop inside it, with its instructions
grouped by kind: the instructions of one pass along one path, from the loop's first instruction to its backward branch,
taking each unconditional branch and falling through each conditional one (a switch inside the loop, such as the fast
kernels' choice of words by alignment, then counts one of its cases; with --all, every instruction between the two
addresses). With --per N the counts are divided by N (e.g. the 32-weight blocks one pass of the loop handles), so
the table reads "instructions per 32 weights". Static counts: what the compiler emitted, not what a GPU spent.

Usage: sass_loops.py <kernel.sass> [--units U]   (U = 32-weight blocks of one matrix that one pass of the loop handles:
                                                  16 for an IQ super-block of gate and up, 8 for Q2_0's 2 blocks of both)
       sass_loops.py <kernel.sass> --list        (all backward branches with their DP4A counts)
       --tsv NAME: one line per loop and kind (NAME, loop, DP4A, kind, count, count per unit) instead of the text table
"""
import bisect
import re
import sys
from collections import Counter, OrderedDict

KINDS = OrderedDict([
    ("dp4a", lambda o: o.startswith("IDP")),
    ("prmt", lambda o: o == "PRMT"),
    ("xmad", lambda o: o.startswith("XMAD")),
    ("popc", lambda o: o == "POPC"),
    ("logic", lambda o: o.startswith("LOP")),
    ("shift/bitfield", lambda o: o.split(".")[0] in ("SHL", "SHR", "SHF", "BFE", "BFI", "FLO", "BREV")),
    ("int add/compare/select", lambda o: o.split(".")[0] in ("IADD", "IADD3", "IADD32I", "ISCADD", "ISCADD32I", "ISETP",
                                                              "ICMP", "IMNMX", "SEL", "ISET", "LEA", "IMAD", "IMUL")),
    ("lds.32", lambda o: o.startswith("LDS") and not any(w in o for w in (".64", ".128"))),
    ("lds.64", lambda o: o.startswith("LDS") and ".64" in o),
    ("lds.128", lambda o: o.startswith("LDS") and ".128" in o),
    ("sts", lambda o: o.startswith("STS")),
    ("ldg", lambda o: o.startswith("LDG") or o.startswith("LD.") or o == "LD"),
    ("stg", lambda o: o.startswith("STG") or o.startswith("ST.") or o == "ST"),
    ("float", lambda o: o.split(".")[0] in ("FFMA", "FADD", "FMUL", "FMNMX", "FSETP", "FSET", "FCHK", "MUFU", "FFMA32I",
                                             "FADD32I", "FMUL32I", "RRO", "FSWZADD")),
    ("conversion", lambda o: o.split(".")[0] in ("I2F", "F2I", "F2F", "I2I", "HADD2", "HMUL2", "HFMA2")),
    ("shuffle/vote", lambda o: o.split(".")[0] in ("SHFL", "VOTE")),
    ("move", lambda o: o.split(".")[0] in ("MOV", "MOV32I", "S2R", "CS2R", "P2R", "R2P", "PSETP", "PLOP3", "CSET", "CSETP")),
    ("control", lambda o: o.split(".")[0] in ("BRA", "SSY", "SYNC", "PBK", "BRK", "BAR", "EXIT", "NOP", "CAL", "RET",
                                               "MEMBAR", "DEPBAR", "JCAL", "BRX", "PCNT", "CONT")),
])

LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")


preds = []  # per instruction: whether it carries a predicate guard (@P0, @!P1 ...)


def parse(path):
    ins = []
    for line in open(path):
        m = LINE.search(line)
        if m:
            ins.append((int(m.group(1), 16), m.group(3), m.group(4)))
            preds.append(bool(m.group(2)) and "PT" not in m.group(2))
    return ins


def kind(op):
    for k, f in KINDS.items():
        if f(op):
            return k
    return "other:" + op


def main():
    path = sys.argv[1]
    ins = parse(path)
    addrs = [a for a, _, _ in ins]
    loops = []
    for i, (a, op, rest) in enumerate(ins):
        if op.startswith("BRA"):
            m = re.search(r"0x([0-9a-f]+)", rest)
            if m:
                t = int(m.group(1), 16)
                # a branch target is a 32-byte bundle (its control word first): the first instruction at or after it
                j = bisect.bisect_left(addrs, t)
                if t <= a and j < len(ins):
                    body = ins[j:i + 1]
                    loops.append((j, i, sum(1 for x in body if x[1].startswith("IDP"))))
    if "--list" in sys.argv:
        for j, i, nd in loops:
            print(f"loop {ins[j][0]:#x}-{ins[i][0]:#x}: {i - j + 1} instructions, {nd} DP4A")
        return
    units = float(sys.argv[sys.argv.index("--units") + 1]) if "--units" in sys.argv else 1.0
    tsv = sys.argv[sys.argv.index("--tsv") + 1] if "--tsv" in sys.argv else None
    inner = [L for L in loops if L[2] > 0 and not any(o is not L and o[2] > 0 and L[0] <= o[0] and o[1] <= L[1]
                                                      for o in loops)]
    for j, i, nd in inner:
        if "--all" in sys.argv:
            body = ins[j:i + 1]
        else:
            body, k, seen = [], j, set()
            while k <= i and k not in seen:
                seen.add(k)
                a, op, rest = ins[k]
                body.append(ins[k])
                if k == i:
                    break
                m = re.search(r"0x([0-9a-f]+)", rest) if op.startswith("BRA") else None
                if op.startswith("BRA") and m and not preds[k]:
                    k = bisect.bisect_left(addrs, int(m.group(1), 16))
                else:
                    k += 1
        nd = sum(1 for x in body if x[1].startswith("IDP"))
        c = Counter(kind(x[1]) for x in body)
        tot = len(body)
        loop = f"{ins[j][0]:#x}-{ins[i][0]:#x}"
        if tsv is None:
            print(f"loop {loop}: {tot} instructions, {nd} DP4A, per pass / {units:g}")
        for k in list(KINDS) + sorted(x for x in c if x.startswith("other:")) + ["total"]:
            n = tot if k == "total" else c.get(k, 0)
            if n:
                if tsv is None:
                    print(f"  {k:24s} {n:6d}  {n / units:8.2f}")
                else:
                    print(f"{tsv}\t{loop}\t{nd}\t{k}\t{n}\t{n / units:.2f}")


if __name__ == "__main__":
    main()
