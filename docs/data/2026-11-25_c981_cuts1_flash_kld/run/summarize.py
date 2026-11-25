#!/usr/bin/env python3
"""C981 (Q1714): the tables from kld-cuts' output (run/kld_cuts.cpp), one pass at a time.

Usage: python3 -I summarize.py <pass folder with kld.pos.tsv[.gz] and kld.calls.tsv> <C905's plans.tsv> <out prefix>
Writes <out prefix>.summary.tsv (per pair and row set), <out prefix>.by_size.tsv (per pair and call size) and
<out prefix>.checks.txt, and prints the summary as Markdown.

Rows. A pair (ref:test) "touches" a call from the first micro-batch at which the two arms' micro-batches differ (from
the calls' logged plans; an arm that logged none ran the plain split: each stretch between the call's cuts, at
N - 516 and N - 4, in micro-batches of 512 from its start). Row sets per pair:
  changed   the calls whose plans differ: their touched prompt rows and the decoded rows after them (the rule's set)
  prompt    the same calls' touched prompt rows only
  last      the same calls' last prompt row (the distribution the server's first reply token is drawn from)
  decode    the same calls' decoded rows
  half      the same calls' prompt rows from N/2 on (llama-perplexity's scored half, as in FN95), touched or not
  untouched the prompt rows before the touched ones, in every call (expected: the same bits)
  same      the calls whose plans are the same for the pair: every row (expected: the same bits)
Per set: rows, calls, mean KLD with its standard error over rows and over calls, the 99th percentile and the maximum,
the share of rows with the same top token, the rows that are the same bits, and the perplexity of the text's next
tokens under each arm with the test's change against the reference's (paired over rows, with its standard error).
"""
import csv
import gzip
import math
import os
import sys
from collections import defaultdict

N_UBATCH = 512
YARDSTICK = 0.285104  # Swift GSQ-RCO Q2_0 against Swift's Q8_0 on this agent text (FN2)
LIMIT = YARDSTICK / 10


def plan_sizes(line, n):
    """Micro-batch sizes of a call from its 'ubatch taper:' line, or the plain split with the server's cuts."""
    if line and line != "-":
        sizes = []
        for tok in line.split(":")[-1].split():
            if "x" in tok:
                s, c = tok.split("x")
                sizes += [int(s)] * int(c)
            else:
                sizes.append(int(tok))
        if sum(sizes) != n:
            raise ValueError(f"plan {line!r} does not add up to {n}")
        return sizes
    cuts = [c for c in (n - 4 - N_UBATCH, n - 4) if 0 < c < n]
    sizes, a = [], 0
    for b in cuts + [n]:
        while a < b:
            sizes.append(min(N_UBATCH, b - a))
            a += sizes[-1]
    return sizes


def first_diff(pa, pb):
    """Start of the first micro-batch that is not the same in both plans (None when the plans are the same)."""
    if pa == pb:
        return None
    a = 0
    for x, y in zip(pa, pb):
        if x != y:
            return a
        a += x
    return a


def stats(rows, arms_ref, arms_test):
    n = len(rows)
    if n == 0:
        return None
    kl = [r["kld"] for r in rows]
    mean = sum(kl) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in kl) / (n - 1)) if n > 1 else 0.0
    by_call = defaultdict(list)
    for r in rows:
        by_call[r["chunk"]].append(r["kld"])
    cm = [sum(v) / len(v) for v in by_call.values()]
    c = len(cm)
    csd = math.sqrt(sum((x - sum(cm) / c) ** 2 for x in cm) / (c - 1)) if c > 1 else 0.0
    s = sorted(kl)
    p99 = s[min(n - 1, int(math.ceil(0.99 * n)) - 1)]
    d = [r["lp_t"] - r["lp_r"] for r in rows]
    dm = sum(d) / n
    dsd = math.sqrt(sum((x - dm) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    lr = sum(r["lp_r"] for r in rows) / n
    lt = sum(r["lp_t"] for r in rows) / n
    return {
        "rows": n, "calls": c, "mean_kld": mean, "se_rows": sd / math.sqrt(n), "se_calls": csd / math.sqrt(c),
        "p99_kld": p99, "max_kld": s[-1], "same_top_pct": 100.0 * sum(r["sametop"] for r in rows) / n,
        "same_bits": sum(r["samebits"] for r in rows), "ppl_ref": math.exp(-lr), "ppl_test": math.exp(-lt),
        # perplexity ratio test/ref - 1, in percent, and its standard error from the paired row differences
        "ppl_change_pct": 100.0 * (math.exp(-dm) - 1.0), "ppl_change_se_pct": 100.0 * dsd / math.sqrt(n),
    }


def main():
    folder, plans_tsv, out = sys.argv[1], sys.argv[2], sys.argv[3]
    calls = list(csv.DictReader(open(os.path.join(folder, "kld.calls.tsv")), delimiter="\t"))
    plans = {}  # (chunk, arm) -> sizes
    size_of = {}
    for c in calls:
        ch, n = int(c["chunk"]), int(c["size"])
        plans[(ch, c["arm"])] = plan_sizes(c["call_plan"], n)
        size_of[ch] = n
    pos_path = os.path.join(folder, "kld.pos.tsv")
    f = gzip.open(pos_path + ".gz", "rt") if os.path.exists(pos_path + ".gz") else open(pos_path)
    rd = csv.reader(f, delimiter="\t")
    head = next(rd)
    col = {h: i for i, h in enumerate(head)}
    pairs = [h[4:].split("_", 1) for h in head if h.startswith("kld_")]
    acc = {tuple(p): defaultdict(list) for p in pairs}
    acc_size = {tuple(p): defaultdict(list) for p in pairs}
    complete = {int(c["chunk"]) for c in calls}
    for row in rd:
        ch = int(row[col["chunk"]])
        if ch not in complete:
            continue
        n = int(row[col["size"]])
        r = int(row[col["row"]])
        for a, b in pairs:
            s = first_diff(plans[(ch, a)], plans[(ch, b)])
            rec = {
                "chunk": ch, "kld": float(row[col[f"kld_{a}_{b}"]]), "sametop": int(row[col[f"sametop_{a}_{b}"]]),
                "samebits": int(row[col[f"samebits_{a}_{b}"]]), "lp_r": float(row[col[f"lp_{a}"]]),
                "lp_t": float(row[col[f"lp_{b}"]]),
            }
            sets = acc[(a, b)]
            if s is None:
                sets["same"].append(rec)
                continue
            if r < n and r < s:
                sets["untouched"].append(rec)
            else:
                sets["changed"].append(rec)
                acc_size[(a, b)][n].append(rec)
                sets["prompt" if r < n else "decode"].append(rec)
            if r == n - 1:
                sets["last"].append(rec)
            if n // 2 <= r < n:
                sets["half"].append(rec)
    f.close()

    keys = ["rows", "calls", "mean_kld", "se_rows", "se_calls", "p99_kld", "max_kld", "same_top_pct", "same_bits",
            "ppl_ref", "ppl_test", "ppl_change_pct", "ppl_change_se_pct"]
    order = ["changed", "prompt", "last", "decode", "half", "untouched", "same"]
    with open(out + ".summary.tsv", "w") as fo:
        fo.write("pair\tset\t" + "\t".join(keys) + "\n")
        for a, b in pairs:
            for name in order:
                st = stats(acc[(a, b)].get(name, []), a, b)
                if st:
                    fo.write(f"{a}:{b}\t{name}\t" + "\t".join(f"{st[k]:.6g}" for k in keys) + "\n")
    with open(out + ".by_size.tsv", "w") as fo:
        fo.write("pair\tsize\t" + "\t".join(keys) + "\n")
        for a, b in pairs:
            for n in sorted(acc_size[(a, b)]):
                st = stats(acc_size[(a, b)][n], a, b)
                fo.write(f"{a}:{b}\t{n}\t" + "\t".join(f"{st[k]:.6g}" for k in keys) + "\n")

    # checks: the logged plans against C905's (the rig's before, the CPU plan after), the untouched and same rows
    ref = {}
    for p in csv.DictReader(open(plans_tsv), delimiter="\t"):
        if p["setting"] == "short=1400:64":
            ref[int(p["call_tokens"])] = (p["before"], p["after"])

    def rle(sizes):
        outl, i = [], 0
        while i < len(sizes):
            j = i
            while j < len(sizes) and sizes[j] == sizes[i]:
                j += 1
            outl.append(f"{sizes[i]}x{j - i}" if j - i > 1 else str(sizes[i]))
            i = j
        return " ".join(outl)

    lines = []
    seen = {}
    for c in calls:
        n, arm = int(c["size"]), c["arm"]
        seen.setdefault((n, arm), set()).add(rle(plans[(int(c["chunk"]), arm)]))
    for n in sorted({k[0] for k in seen}):
        got = {arm: " / ".join(sorted(v)) for (m, arm), v in seen.items() if m == n}
        want = ref.get(n)
        verdict = ""
        if want:
            ok_v5 = got.get("v5") == want[0] and got.get("rep", want[0]) == want[0]
            ok_v5c = got.get("v5c") == want[1]
            verdict = "as C905 (rig before, CPU plan after)" if ok_v5 and ok_v5c else f"DIFFERS from C905 {want}"
        lines.append(f"size {n}: off {got.get('off')} | v5 {got.get('v5')} | v5c {got.get('v5c')} | "
                     f"rep {got.get('rep')}: {verdict}")
    for a, b in pairs:
        for name in ("untouched", "same"):
            rs = acc[(a, b)].get(name, [])
            if rs:
                lines.append(f"{a}:{b} {name}: {sum(r['samebits'] for r in rs)} of {len(rs)} rows the same bits")
    pre = [c["prefix_samebits_first_arm"] for c in calls if c.get("prefix_samebits_first_arm", "-1") != "-1"]
    if pre:
        lines.append(f"prefix call's last logits the same as the first arm's: {pre.count('1')} of {len(pre)} arm calls")
    fills = [int(c["fills"]) for c in calls]
    lines.append(f"in-call checkpoints filled and read per arm call: min {min(fills)}, max {max(fills)}")
    lines.append(f"calls: {len(complete)}")
    open(out + ".checks.txt", "w").write("\n".join(lines) + "\n")

    print("| pair | set | rows | calls | mean KLD | ± rows | ± calls | 99th pct | max | same top | same bits | "
          "ppl change |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a, b in pairs:
        for name in order:
            st = stats(acc[(a, b)].get(name, []), a, b)
            if st:
                print(f"| {a}:{b} | {name} | {st['rows']} | {st['calls']} | {st['mean_kld']:.6f} | "
                      f"{st['se_rows']:.6f} | {st['se_calls']:.6f} | {st['p99_kld']:.6f} | {st['max_kld']:.6f} | "
                      f"{st['same_top_pct']:.3f} % | {st['same_bits']} | {st['ppl_change_pct']:+.3f} ± "
                      f"{st['ppl_change_se_pct']:.3f} % |")
    print()
    print("\n".join(lines))
    st = stats(acc[("v5", "v5c")].get("changed", []), "v5", "v5c") if ("v5", "v5c") in acc else None
    if st:
        print(f"\nrule: v5:v5c changed mean KLD {st['mean_kld']:.6f} against the limit {LIMIT:.4f}: "
              f"{'met' if st['mean_kld'] <= LIMIT else 'NOT met'}")


if __name__ == "__main__":
    main()
