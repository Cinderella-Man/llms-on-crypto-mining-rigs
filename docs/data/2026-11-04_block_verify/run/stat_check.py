#!/usr/bin/env python3
"""Q797 / Q802: do sampled outputs with block verification and forced tokens follow the target's distribution?

Reads the TSVs of run/stat.sh (seed, output, drafted, accepted) from a directory and tests every arm against plain
decoding (nospec) with two-sample chi-square tests of homogeneity:
  plain-*  the first token (drawn before any draft: a check of the test), the second and the third, the first two and
           the first three jointly
  gram-*   the tool call's outcome (tool and digit, 30 cells); every reply must also be valid JSON
Cells are pooled until each has an expected count of at least 5 in both samples. A test fails at p < ALPHA (1e-4,
small enough for the ~20 tests run at once). The control arm (plain decoding at temperature 0.8) must fail its joint
test of the first two tokens at p < 1e-6: the power check. The base build's token-by-token rejection (arms rej,
rej2) is tested and reported, outside the verdict. Also prints the total variation distance between the empirical
distributions and the drafted and accepted tokens per arm; writes check.json, check.tsv and acceptance.tsv.

Usage: python3 -I stat_check.py <dir>   (exit 0 when every exact arm passes and the control is told apart)
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

ALPHA = 1e-4
POWER = 1e-6


def gammaincc(a, x):
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes: series below a + 1, continued fraction above)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(10000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    b = x + 1 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_two_sample(ca, cb):
    """Chi-square test of homogeneity of two count vectors (dicts); cells pooled to expected counts >= 5."""
    na, nb = sum(ca.values()), sum(cb.values())
    # by count, then by the cell itself: the pooling (and so the statistic) does not depend on the hash seed
    keys = sorted(set(ca) | set(cb), key=lambda k: (-(ca.get(k, 0) + cb.get(k, 0)), str(k)))
    cells, cur_a, cur_b = [], 0, 0
    for k in keys:
        cur_a += ca.get(k, 0)
        cur_b += cb.get(k, 0)
        tot = cur_a + cur_b
        if tot * na / (na + nb) >= 5 and tot * nb / (na + nb) >= 5:
            cells.append((cur_a, cur_b))
            cur_a, cur_b = 0, 0
    if cur_a or cur_b:
        if cells:
            a, b = cells.pop()
            cells.append((a + cur_a, b + cur_b))
        else:
            cells.append((cur_a, cur_b))
    if len(cells) < 2:
        return 0.0, 0, 1.0
    stat = 0.0
    for a, b in cells:
        tot = a + b
        ea, eb = tot * na / (na + nb), tot * nb / (na + nb)
        stat += (a - ea) ** 2 / ea + (b - eb) ** 2 / eb
    dof = len(cells) - 1
    return stat, dof, gammaincc(dof / 2.0, stat / 2.0)


def tv(ca, cb):
    na, nb = sum(ca.values()), sum(cb.values())
    return 0.5 * sum(abs(ca.get(k, 0) / na - cb.get(k, 0) / nb) for k in set(ca) | set(cb))


def read(path):
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            rows.append((parts[1], int(float(parts[2] or 0)), int(float(parts[3] or 0))))
    return rows


def kind_of(arm):
    """control: plain decoding at another temperature; base: the base build's token-by-token rejection (rej, rej2),
    reported but not part of the verdict; exact: every other arm (block verification, forced tokens, the fix)."""
    if arm.startswith("t0"):
        return "control"
    if arm.rstrip("0123456789") == "rej":
        return "base"
    return "exact"


def main():
    d = Path(sys.argv[1])
    ok = True
    results = []

    def test(name, arm, ca, cb):
        nonlocal ok
        kind = kind_of(arm)
        stat, dof, p = chi2_two_sample(ca, cb)
        dist = tv(ca, cb)
        passed = p < POWER if kind == "control" else p >= ALPHA
        results.append({"test": name, "arm": arm, "kind": kind, "n": sum(ca.values()), "n_ref": sum(cb.values()),
                        "chi2": round(stat, 2), "dof": dof, "p": p, "tv": round(dist, 4), "pass": passed})
        note = {"control": " (control: must differ)", "base": " (the base build's path)", "exact": ""}[kind]
        print(f"{name:40s} chi2 {stat:9.2f} dof {dof:4d} p {p:.3g} TV {dist:.4f} "
              f"{'ok' if passed else ('BIASED' if kind == 'base' else 'FAIL')}{note}")
        if not passed and kind != "base":
            ok = False

    # plain text: five tokens, the first from the prompt pass, the rest from verification
    plain = {p.stem[len("plain-"):]: read(p) for p in sorted(d.glob("plain-*.tsv"))}
    if "nospec" in plain:
        ref = [r[0].split(",") for r in plain["nospec"]]
        for arm, rows in plain.items():
            if arm == "nospec":
                continue
            toks = [r[0].split(",") for r in rows]
            if kind_of(arm) == "control":
                test(f"plain {arm} tokens 1-2", arm, Counter(tuple(t[:2]) for t in toks),
                     Counter(tuple(t[:2]) for t in ref))
                continue
            for k in range(3):
                test(f"plain {arm} token {k + 1}", arm, Counter(t[k] for t in toks if len(t) > k),
                     Counter(t[k] for t in ref if len(t) > k))
            for k in (2, 3):
                test(f"plain {arm} tokens 1-{k}", arm, Counter(tuple(t[:k]) for t in toks),
                     Counter(tuple(t[:k]) for t in ref))

    # the tool call under a grammar
    gram = {p.stem[len("gram-"):]: read(p) for p in sorted(d.glob("gram-*.tsv"))}

    def outcome(text):
        try:
            j = json.loads(text)
            return f"{j['tool']}:{j['n']}"
        except Exception:
            return None

    if "nospec" in gram:
        ref = Counter(outcome(r[0]) for r in gram["nospec"])
        for arm, rows in gram.items():
            outs = [outcome(r[0]) for r in rows]
            n_bad = sum(o is None for o in outs)
            print(f"gram {arm}: {len(outs)} replies, {n_bad} not valid JSON")
            if n_bad:
                ok = False
            if arm != "nospec":
                test(f"gram {arm} outcome", arm, Counter(outs), ref)

    # drafted and accepted tokens per arm
    print("acceptance (drafted, accepted, accepted / drafted):")
    acc = []
    for kind, arms in (("plain", plain), ("gram", gram)):
        for arm, rows in arms.items():
            dr = sum(r[1] for r in rows)
            ac = sum(r[2] for r in rows)
            if dr:
                print(f"  {kind}-{arm:9s} {dr:7d} {ac:7d} {ac / dr:.4f}")
                acc.append((f"{kind}-{arm}", len(rows), dr, ac))

    n_exact = sum(r["kind"] == "exact" for r in results)
    n_pass = sum(r["kind"] == "exact" and r["pass"] for r in results)
    base = [r for r in results if r["kind"] == "base"]
    ctrl = [r for r in results if r["kind"] == "control"]
    print(f"verdict: {n_pass}/{n_exact} exactness tests pass at p >= {ALPHA:g}")
    print(f"base: {sum(r['pass'] for r in base)}/{len(base)} tests of the base build's token-by-token rejection pass"
          + "".join(f"; {r['test']} p = {r['p']:.3g}" for r in base if not r["pass"]))
    for r in ctrl:
        print(f"power: the control (temperature 0.8) {'told apart' if r['pass'] else 'NOT told apart'} at p = {r['p']:.3g}")
    (d / "check.json").write_text(json.dumps(results, indent=1))
    with open(d / "check.tsv", "w") as f:
        f.write("test\tkind\tn\tn_ref\tchi2\tdof\tp\ttv\tpass\n")
        for r in results:
            f.write(f"{r['test']}\t{r['kind']}\t{r['n']}\t{r['n_ref']}\t{r['chi2']}\t{r['dof']}\t{r['p']:.3g}\t"
                    f"{r['tv']}\t{'yes' if r['pass'] else 'no'}\n")
    with open(d / "acceptance.tsv", "w") as f:
        f.write("arm\trequests\tdrafted\taccepted\taccepted_per_drafted\n")
        for name, n, dr, ac in acc:
            f.write(f"{name}\t{n}\t{dr}\t{ac}\t{ac / dr:.4f}\n")
    return 0 if ok and ctrl and all(r["pass"] for r in ctrl) else 1


if __name__ == "__main__":
    sys.exit(main())
