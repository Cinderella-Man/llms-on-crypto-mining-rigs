#!/usr/bin/env python3
"""C992 (Q1725): the tables of run.sh's outputs.

Usage: python3 -I c992_tables.py <work dir> <out dir>
Reads <work dir>/check (the logs and replay logs of run.sh's check step), <work dir>/rec/<base|head>/cases and
<work dir>/kept/cases (the records), and writes into <out dir>:
  kept.tsv      C968's kept case replayed: binary, setting, verdict, each compared node's error (w_cut, ids_cut, out),
                the live check on the restored inputs
  replay.tsv    each record replayed by each binary in each setting: cases, OK, FAIL, the failing cases
  verdicts.tsv  per record and setting, the cases whose verdict differs between the base and the head
  live.tsv      the live runs (live: the check step's, soak: the soak step's): runs, cases, OK, FAIL, and each
                failing case with its count
  failures.tsv  each failing live case: which outputs failed, and whether a cut slot's id or the product differed
  draws.tsv     the routers of every recorded TOPK_MOE_TAIL case: per record the tokens, the tokens whose
                n_expert_used + 1 best probabilities have a neighbour pair closer than a relative 2^-16 (in double), or
                within 4 float32 ulps (in float32, as the processor rounds them), and the smallest gap in ulps
"""
import glob
import math
import os
import re
import struct
import sys

W, OUT = sys.argv[1], sys.argv[2]
C = os.path.join(W, "check")
os.makedirs(OUT, exist_ok=True)
CASE = re.compile(r"\b(TOPK_MOE_TAIL|MOE_TAIL)\(([^()]*)\): .*?(OK|FAIL)\s*$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
BINS, SETS = ("base", "head"), ("unset", "p106")


def lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [ANSI.sub("", x.rstrip("\n")) for x in f]


def cases(path):
    """{op(params): verdict} of a test-backend-ops log"""
    out = {}
    for x in lines(path):
        m = CASE.search(x)
        if m:
            out[f"{m.group(1)}({m.group(2)})"] = m.group(3)
    return out


def rlog(path):
    """{case name: (verdict, errors)} of a --replay-log file"""
    out = {}
    if os.path.exists(path):
        for x in lines(path):
            f = x.split("\t")
            if len(f) >= 3:
                out[f[1]] = (f[2], f[3] if len(f) > 3 else "")
    return out


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


# kept
rows = []
for b in BINS:
    for s in SETS:
        t = f"kept-{b}-{s}"
        r = rlog(os.path.join(C, t + ".tsv"))
        cpu = [x.strip() for x in lines(os.path.join(C, t + ".log")) if "cpu-check:" in x] if os.path.exists(os.path.join(C, t + ".log")) else []
        for name, (v, e) in r.items():
            rows.append((b, s, v, e, cpu[0] if cpu else ""))
        if not r:
            rows.append((b, s, "no replay log", "", ""))
write("kept.tsv", ("binary", "setting", "verdict", "errors (w_cut,ids_cut,out)", "live check on the restored inputs"), rows)

# replay and verdicts
rows, vrows = [], []
for rec in BINS:
    for s in SETS:
        got = {b: rlog(os.path.join(C, f"replay-{rec}-{b}-{s}.tsv")) for b in BINS}
        for b in BINS:
            r = got[b]
            fails = sorted(n for n, (v, _) in r.items() if v != "OK")
            rows.append((rec, b, s, len(r), sum(v == "OK" for v, _ in r.values()), len(fails), "; ".join(fails)))
        names = sorted(set(got["base"]) | set(got["head"]))
        differ = [n for n in names if got["base"].get(n, ("missing",))[0] != got["head"].get(n, ("missing",))[0]]
        vrows.append((rec, s, len(names), len(names) - len(differ), len(differ),
                      " | ".join(f"{n}: base {got['base'].get(n, ('missing', ''))} head {got['head'].get(n, ('missing', ''))}"
                                 for n in differ)))
write("replay.tsv", ("record", "binary", "setting", "cases", "OK", "FAIL", "failing cases"), rows)
write("verdicts.tsv", ("record", "setting", "cases", "same verdict", "verdict differs", "the differing cases (verdict, errors)"), vrows)

# live
rows = []
for d in [os.path.join(C, f"live-{b}-{s}") for b in BINS for s in SETS] + sorted(glob.glob(os.path.join(C, "soak-*"))):
    if not os.path.isdir(d):
        continue
    kind, b, s = os.path.basename(d).split("-", 2)
    logs = sorted(glob.glob(os.path.join(d, "*.log")), key=lambda p: int(os.path.basename(p)[:-4]))
    n_ok = n_fail = 0
    per_run, failed = [], {}
    for p in logs:
        c = cases(p)
        per_run.append(len(c))
        for name, v in c.items():
            if v == "OK":
                n_ok += 1
            else:
                n_fail += 1
                failed[name] = failed.get(name, 0) + 1
    rc = lines(os.path.join(d, "rc.txt")) if os.path.exists(os.path.join(d, "rc.txt")) else []
    cpr = sorted(set(per_run))
    rows.append((kind, b, s, len(logs), sum(not x.endswith("rc=0") for x in rc), ", ".join(map(str, cpr)) or "0",
                 n_ok, n_fail, "; ".join(f"{k} x{v}" for k, v in sorted(failed.items()))))
write("live.tsv", ("runs of", "binary", "setting", "runs", "runs with rc != 0", "cases per run", "OK", "FAIL", "failing cases (times)"), rows)

# failures: each failing live case, with the outputs that failed (MOE_TAIL: the cut ids, as the cut weights only
# differ by rounding; MUL_MAT_ID: the expert product) and the slots' kind (a batch of up to n_skip tokens marks a cut
# slot -1, a larger one keeps its own expert at weight 0)
ERR = re.compile(r"\[([A-Z_]+)\] ERR = ([0-9.e+-]+) > ")
rows = []
for d in [os.path.join(C, f"live-{b}-{s}") for b in BINS for s in SETS] + sorted(glob.glob(os.path.join(C, "soak-*"))):
    for p in sorted(glob.glob(os.path.join(d, "*.log")), key=lambda q: int(os.path.basename(q)[:-4])):
        for x in lines(p):
            m = CASE.search(x)
            if not m or m.group(3) != "FAIL":
                continue
            v = dict(kv.split("=") for kv in m.group(2).split(","))
            errs = ERR.findall(x)
            ops = {op for op, _ in errs}
            nt, ns = int(v.get("n_tokens", 0)), int(v.get("n_skip", 0))
            kind = ("a cut slot's id" if ops == {"MOE_TAIL"} else
                    "the product (a kept slot's expert)" if "MUL_MAT_ID" in ops else "other")
            rows.append((os.path.basename(d), os.path.basename(p)[:-4], f"{m.group(1)}({m.group(2)})",
                         "skip (-1)" if nt <= ns else "own expert at 0", " ".join(f"{op} {e}" for op, e in errs), kind))
write("failures.tsv", ("runs of", "run", "case", "cut slots", "failed outputs (error)", "what differed"), rows)


# draws
def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<i", struct.pack("<f", x))[0]


def meta(path):
    m = {"tensors": []}
    for x in lines(path):
        f = x.split("\t")
        if f[0] == "tensor":
            m["tensors"].append((f[2], f[3], int(f[5])))
        elif len(f) >= 2:
            m[f[0]] = f[1]
    return m


def draws(folder):
    n_case = n_tok = n_rel = n_ulp = 0
    min_ulp = None
    for mp in sorted(glob.glob(os.path.join(folder, "cases", "*.meta"))):
        m = meta(mp)
        if "test_topk_moe_tail" not in m.get("class", ""):
            continue
        p = dict(kv.split("=") for kv in m["name"][m["name"].index("(") + 1:-1].split(","))
        ne, nt, k, g = int(p["n_expert"]), int(p["n_tokens"]), int(p["n_expert_used"]), int(p["gating_func"])
        off = 0
        for name, _, size in m["tensors"]:
            if name == "logits":
                break
            off += size
        with open(mp[:-5] + ".in", "rb") as f:
            b = f.read()
        x = struct.unpack_from("<%df" % (ne * nt), b, off)
        n_case += 1
        k1 = min(k + 1, ne)
        for t in range(nt):
            row = x[t * ne:(t + 1) * ne]
            mx = max(row)
            if g == 1:  # sigmoid
                pd = [1.0 / (1.0 + math.exp(-v)) for v in row]
                pf = [f32(1.0 / (1.0 + f32(math.exp(-v)))) for v in row]
            else:  # softmax, up to its common factor (the float32 ulps then compare exp(x - max))
                pd = [math.exp(v - mx) for v in row]
                pf = [f32(math.exp(f32(v - mx))) for v in row]
            top = sorted(pd, reverse=True)[:k1]
            topf = sorted(pf, reverse=True)[:k1]
            n_tok += 1
            if any(top[i - 1] - top[i] < 2.0 ** -16 * top[i - 1] for i in range(1, k1)):
                n_rel += 1
            u = min(abs(bits(topf[i - 1]) - bits(topf[i])) for i in range(1, k1))
            if u <= 4:
                n_ulp += 1
            min_ulp = u if min_ulp is None else min(min_ulp, u)
    return n_case, n_tok, n_rel, n_ulp, min_ulp


rows = []
for name, folder in (("C968's kept case", os.path.join(W, "kept")), ("drawn by the base", os.path.join(W, "rec", "base")),
                     ("drawn by the head", os.path.join(W, "rec", "head"))):
    rows.append((name,) + draws(folder))
write("draws.tsv", ("record", "TOPK_MOE_TAIL cases", "tokens", "tokens with a pair under a relative 2^-16",
                    "tokens with a pair within 4 float32 ulps", "smallest gap (float32 ulps)"), rows)

for n in ("kept.tsv", "replay.tsv", "verdicts.tsv", "live.tsv", "failures.tsv", "draws.tsv"):
    print(f"== {n}")
    with open(os.path.join(OUT, n), encoding="utf-8") as f:
        for x in f:
            print(x.rstrip("\n")[:400])
