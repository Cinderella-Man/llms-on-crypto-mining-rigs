"""Checks of llama-imatrix's span matrices (--span-mask, Q897) against its usual matrix; each prints one verdict line
(OK or FAIL and the numbers) and exits non-zero on FAIL.
  span_check.py same A.gguf B.gguf
      every .counts, .in_sum2 and .in_sum tensor of A equals B's bit for bit (and both have the same tensors)
  span_check.py zero A.gguf
      every such tensor of A is zero
  span_check.py sum FULL.gguf PART.gguf...
      the parts' counts add up to FULL's exactly, their sums within --rtol (.in_sum2 relative to FULL's, .in_sum
      relative to sqrt(count x in_sum2), the scale of a sum of that many terms of that mean square); every part's
      dense counts equal the tokens of its class it says it processed (imatrix.span.tokens), and its routed-expert
      counts per layer are the experts used per token times those tokens
  span_check.py diff BIG.gguf SMALL.gguf PART.gguf
      PART's counts are BIG's minus SMALL's exactly (BIG over one more chunk than SMALL, PART that chunk's class)
  span_check.py variance A.gguf [--layer 0] [--max 1e-4]
      per dense tensor of the layer, the variance of its input columns over the rows counted, relative to their mean
      square: zero when every counted row was the same vector (a class of one token id, read where the input depends
      on the token alone); OK when the smallest is at most --max, with --expect-above the smallest must exceed it
Standard library only; gguf_reshard.py from the repository's scripts/.
"""
import argparse
import math
import os
import re
import sys
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "scripts")))
from gguf_reshard import Header  # noqa: E402

PART = re.compile(r"^(.*)\.(counts|in_sum2|in_sum)$")
EXPERTS = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight$")


def load(path):
    """{tensor: {kind: (dims, array)}} and the header."""
    h = Header(path)
    out = {}
    with open(path, "rb") as f:
        for name, dims, typ, off in h.tensors:
            m = PART.match(name)
            if not m:
                continue
            assert typ == 0, f"{path}: {name} has type {typ}, not F32"
            f.seek(h.data0 + off)
            a = array("f")
            a.frombytes(f.read(4 * math.prod(dims)))
            if sys.byteorder != "little":
                a.byteswap()
            out.setdefault(m[1], {})[m[2]] = (dims, a)
    return out, h


def verdict(ok, text):
    print(("OK " if ok else "FAIL ") + text)
    sys.exit(0 if ok else 1)


def cmd_same(a):
    x, _ = load(a.a)
    y, _ = load(a.b)
    if set(x) != set(y):
        verdict(False, f"tensors differ: {len(set(x) ^ set(y))} in one file only")
    bad = [f"{t}.{k}" for t in x for k in set(x[t]) | set(y[t])
           if k not in x[t] or k not in y[t] or x[t][k][1].tobytes() != y[t][k][1].tobytes()]
    verdict(not bad, f"{len(x)} tensors, {len(bad)} differ{': ' + ', '.join(bad[:5]) if bad else ''}")


def cmd_zero(a):
    x, _ = load(a.a)
    bad = [f"{t}.{k}" for t in x for k, (_d, v) in x[t].items() if any(v)]
    verdict(not bad, f"{len(x)} tensors, {len(bad)} not zero{': ' + ', '.join(bad[:5]) if bad else ''}")


def dense_counts(x):
    return {int(k["counts"][1][0]) for t, k in x.items() if not EXPERTS.match(t) and len(k["counts"][1]) == 1}


def cmd_sum(a):
    full, _ = load(a.full)
    parts = [load(p) for p in a.parts]
    worst, bad, left_out = {"in_sum2": 0.0, "in_sum": 0.0}, [], []
    for t, kinds in full.items():
        have = [t in p for p, _h in parts]
        if not any(have):
            left_out.append(t)  # rows that are not the batch's tokens: the collector leaves the tensor out
            continue
        counts, s2 = kinds["counts"][1], kinds["in_sum2"][1]
        ne0 = len(s2) // len(counts)
        for k, (_dims, v) in kinds.items():
            if not all(have) or any(k not in p[t] for p, _h in parts):
                bad.append(f"{t}.{k} missing in a part")
                continue
            for i, fv in enumerate(v):
                s = [p[t][k][1][i] for p, _h in parts]
                if k == "counts":
                    if sum(s) != fv:
                        bad.append(f"{t}.counts[{i}] {sum(s)} != {fv}")
                    continue
                d = abs(math.fsum(s) - fv)
                scale = abs(fv) if k == "in_sum2" else math.sqrt(counts[i // ne0] * s2[i])
                worst[k] = max(worst[k], d / scale if scale > 1e-30 else d)
    for k, w in worst.items():
        if w > a.rtol:
            bad.append(f"{k} off by {w:.3g}")
    full_tokens = dense_counts(full)
    full_tokens = full_tokens.pop() if len(full_tokens) == 1 else None
    for (p, h), path in zip(parts, a.parts):
        tokens = h.value("imatrix.span.tokens")
        if tokens is None or dense_counts(p) - {tokens} or not dense_counts(p):
            bad.append(f"{os.path.basename(path)}: dense counts {sorted(dense_counts(p))[:4]}, span tokens {tokens}")
            continue
        for t, kinds in p.items():
            if EXPERTS.match(t) and full_tokens:
                per_token = sum(full[t]["counts"][1]) / full_tokens
                if sum(kinds["counts"][1]) != per_token * tokens:
                    bad.append(f"{os.path.basename(path)}: {t} {sum(kinds['counts'][1]):.0f} routings for {tokens} tokens")
                    break
    verdict(not bad, f"{len(full)} tensors ({len(left_out)} left out), {len(parts)} parts: counts add up exactly, "
                     f"in_sum2 within {worst['in_sum2']:.3g}, in_sum within {worst['in_sum']:.3g}; "
                     f"{'; '.join(bad[:5]) if bad else 'every part counts its own tokens'}")


def cmd_diff(a):
    big, _ = load(a.big)
    small, _ = load(a.small)
    part, _ = load(a.part)
    bad = []
    for t, kinds in big.items():
        c = kinds["counts"][1]
        s = small[t]["counts"][1] if t in small else [0.0] * len(c)
        p = part[t]["counts"][1] if t in part else None
        if p is None or any(pc != bc - sc for pc, bc, sc in zip(p, c, s)):
            bad.append(t)
    verdict(not bad, f"{len(big)} tensors: the part's counts are the difference in "
                     f"{len(big) - len(bad)}{', not in ' + ', '.join(bad[:5]) if bad else ''}")


def cmd_variance(a):
    x, _ = load(a.a)
    rows = []
    for t, kinds in x.items():
        if not t.startswith(f"blk.{a.layer}.") or EXPERTS.match(t) or "in_sum" not in kinds:
            continue
        (_d, cnt), (dims, s2), (_d1, s1) = kinds["counts"], kinds["in_sum2"], kinds["in_sum"]
        if len(cnt) != 1 or cnt[0] == 0:
            continue
        n = cnt[0]
        ms = math.fsum(s2) / n
        var = math.fsum(max(0.0, q / n - (m / n) ** 2) for q, m in zip(s2, s1))
        rows.append((var / ms if ms else 0.0, t, int(n)))
    if not rows:
        verdict(False, f"no dense tensor of layer {a.layer} with counted rows")
    rows.sort()
    text = ", ".join(f"{t} {v:.3g} ({n} rows)" for v, t, n in rows[:4])
    if a.expect_above is not None:
        verdict(rows[0][0] > a.expect_above, f"smallest relative variance above {a.expect_above:g}: {text}")
    verdict(rows[0][0] <= a.max, f"smallest relative variance at most {a.max:g}: {text}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("same"); s.add_argument("a"); s.add_argument("b")
    z = sub.add_parser("zero"); z.add_argument("a")
    u = sub.add_parser("sum"); u.add_argument("full"); u.add_argument("parts", nargs="+")
    u.add_argument("--rtol", type=float, default=1e-4)
    d = sub.add_parser("diff"); d.add_argument("big"); d.add_argument("small"); d.add_argument("part")
    v = sub.add_parser("variance"); v.add_argument("a"); v.add_argument("--layer", type=int, default=0)
    v.add_argument("--max", type=float, default=1e-4); v.add_argument("--expect-above", type=float)
    a = ap.parse_args()
    {"same": cmd_same, "zero": cmd_zero, "sum": cmd_sum, "diff": cmd_diff, "variance": cmd_variance}[a.cmd](a)


if __name__ == "__main__":
    main()
