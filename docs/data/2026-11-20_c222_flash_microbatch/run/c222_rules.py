#!/usr/bin/env python3
"""c222_rules.py: what other taper settings would read the replay in (est.), composed turn by turn from C222's
measured passes (c222_tables.py's turns.tsv).

    python3 -I c222_rules.py tables/turns.tsv > rules.tsv

Why a composition works: a turn is one llama-server call (-b 4096; the replay's calls hold at most 3,313 tokens), and
the taper picks a call's micro-batches from its length alone: the tokens the call reads (turns.tsv's call_tokens, the
server's prompt_n), which is the turn's new tokens plus one for 50 turns and plus 516 for the one turn that resumes
from a checkpoint. Under short=N:64 a call of 601 to N tokens is split
exactly as in the short=2048:64 passes and every other call exactly as in the control's (short=600:64); with two size
rules (64 up to 600 tokens, 128 from 601 to N, which needs code: the taper holds one short rule) a call of 601 to N
tokens is split as in the short=2048:128 passes. So each setting's seconds a pass are the sum, over the 62 turns, of
the mean of the passes whose split that turn would get. Est.: it assumes a turn's time does not depend on the splits
of the turns before it. The control's passes are the warm ones (all but the load's first; --all-control keeps it).

Prints a TSV: setting, est. seconds a pass, the difference from the control, the turns the setting changes. Also the
seconds of the turns under 600 new tokens by finer size, per arm, and every turn of 601 to 2,048 call tokens against
the warm controls. Standard library only.
"""
import argparse, csv, statistics

ap = argparse.ArgumentParser()
ap.add_argument("turns")
ap.add_argument("--all-control", action="store_true", help="keep the load's first control pass")
a = ap.parse_args()

rows = list(csv.DictReader(open(a.turns), delimiter="\t"))
cols = [c for c in rows[0] if c.startswith("p")]
ctrl = [c for c in cols if c.endswith(":short=600:64")]
if not a.all_control:
    ctrl = ctrl[1:]
a128 = [c for c in cols if c.endswith(":short=2048:128")]
a64 = [c for c in cols if c.endswith(":short=2048:64")]


def secs(r, cs):
    return statistics.mean(float(r[c]) for c in cs) / 1000


def est(rule):
    return sum(secs(r, rule(int(r["call_tokens"]))) for r in rows)


base = est(lambda n: ctrl)
print("setting\test_s_a_pass\tdelta_s\tdelta_pct\tturns_changed")
print(f"short=600:64 (control, passes {', '.join(c.split(':')[0] for c in ctrl)})\t{base:.2f}\t+0.00\t+0.00\t0")
cases = [(f"short={n}:64", (lambda n: lambda k: a64 if 600 < k <= n else ctrl)(n), lambda k, n=n: 600 < k <= n)
         for n in (1023, 1200, 1300, 1400, 1600, 2048)]
cases += [(f"short={n}:128", (lambda n: lambda k: a128 if k <= n else ctrl)(n), lambda k, n=n: k <= n)
          for n in (1400, 2048)]
cases += [(f"two rules: 64 to 600, 128 to {n} (code)", (lambda n: lambda k: a128 if 600 < k <= n else ctrl)(n),
           lambda k, n=n: 600 < k <= n) for n in (1023, 1400, 1600, 2048)]
for name, rule, changed in cases:
    v = est(rule)
    print(f"{name}\t{v:.2f}\t{v - base:+.2f}\t{(v - base) / base * 100:+.2f}\t"
          f"{sum(1 for r in rows if changed(int(r['call_tokens'])))}")
print()
print("new_tokens\tturns\tcontrol_s\tshort=2048:128_s\tshort=2048:64_s")
for lo, hi in ((0, 100), (100, 300), (300, 500), (500, 600)):
    rs = [r for r in rows if lo <= int(r["new_tokens"]) < hi]
    print(f"{lo}-{hi - 1}\t{len(rs)}\t{sum(secs(r, ctrl) for r in rs):.2f}\t{sum(secs(r, a128) for r in rs):.2f}\t"
          f"{sum(secs(r, a64) for r in rs):.2f}")
print()
print("task\tturn\tcall_tokens\tcontrol_s\tshort=2048:128_delta_s\tshort=2048:64_delta_s")
for r in sorted(rows, key=lambda r: int(r["call_tokens"])):
    n = int(r["call_tokens"])
    if 600 < n <= 2048:
        c = secs(r, ctrl)
        print(f"{r['task']}\t{r['turn']}\t{n}\t{c:.2f}\t{secs(r, a128) - c:+.2f}\t{secs(r, a64) - c:+.2f}")
