#!/usr/bin/env python3
# C854 step two: classify every kernel whose SASS differs between two builds (sass_compare.py's extracts), one row per
# kernel: family, instructions in base and new, the kind of difference, and the opcodes whose counts change.
#   sass_classify.py h-B.tsv h-N.tsv ext-B.txt ext-N.txt > classes.tsv ; a summary per family and kind on stderr
# kind: const = only MOV32I constants differ (a printed source line or a string address);
#       reorder = the same instructions as a multiset, in another order or with other registers/operands;
#       code = the instruction multiset differs.
import sys, re, collections

MOV32I = re.compile(r"^MOV32I (R\d+), (0x[0-9a-f]+) ;$")
OP = re.compile(r"^(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?")


def load_h(p):
    out = {}
    with open(p) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for ln in f:
            r = dict(zip(hdr, ln.rstrip("\n").split("\t")))
            if r.get("kq_fp32") != "true":
                out.setdefault(r["kernel"], r)
    return out


def read_extract(path):
    out = collections.defaultdict(list)
    cur = None
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith("### "):
            cur = []
            out[ln[4:]].append(cur)
        else:
            cur.append(ln)
    return out


def ops(lines):
    c = collections.Counter()
    for x in lines:
        m = OP.match(x)
        if m:
            c[m.group(1) + (m.group(2) or "")] += 1
    return c


def family(k):
    k = re.sub(r"^void ", "", k)
    m = re.match(r"^([A-Za-z_0-9:]+)", k.replace("(anonymous namespace)::", "").replace("<unnamed>::", ""))
    return m.group(1) if m else k[:40]


hB, hN = load_h(sys.argv[1]), load_h(sys.argv[2])
eB, eN = read_extract(sys.argv[3]), read_extract(sys.argv[4])
summ = collections.Counter()
print("kernel\tfamily\tinstr_base\tinstr_new\tkind\topcode_changes")
for k in sorted(eB):
    a, b = eB[k], eN.get(k, [])
    nb, nn = hB.get(k, {}).get("instructions", "?"), hN.get(k, {}).get("instructions", "?")
    la = [x for c in a for x in c]
    lb = [x for c in b for x in c]
    if len(la) == len(lb) and all(x == y or (MOV32I.match(x) and MOV32I.match(y) and MOV32I.match(x).group(1) == MOV32I.match(y).group(1)) for x, y in zip(la, lb)):
        kind, chg = "const", ""
    else:
        oa, ob = ops(la), ops(lb)
        if oa == ob:
            kind, chg = "reorder", ""
        else:
            kind = "code"
            d = []
            for o in sorted(set(oa) | set(ob)):
                if oa[o] != ob[o]:
                    d.append(f"{o}{ob[o] - oa[o]:+d}")
            chg = " ".join(d)
    fam = family(k)
    summ[(fam, kind)] += 1
    print(f"{k}\t{fam}\t{nb}\t{nn}\t{kind}\t{chg}")
for (fam, kind), n in sorted(summ.items()):
    print(f"{fam}\t{kind}\t{n}", file=sys.stderr)
