#!/usr/bin/env python3
"""r225_calls.py: R225's (Q1637) measured settings against what C222's passes composed for them (est.), and every
call of 601 to 2,048 tokens by setting.

    python3 -I r225_calls.py --r225 tables/turns.tsv --c222 ../2026-11-20_c222_flash_microbatch/turns.tsv > calls.tsv

Both files are c222_tables.py's turns.tsv: R225's with its warm-up pass left out (--drop-pass 0), C222's as published
(its pass 0, the load's first, is left out here as in c222_rules.py). C222's composition: under short=N:64 a call of
601 to N tokens reads as in C222's short=2048:64 passes and every other call as in its warm controls.

Prints two TSVs: per setting, the measured seconds a pass (each pass by value and the mean), the control's mean, the
difference, and C222's est. for it; then per call of 601 to 2,048 tokens, the control's mean in each run and each
setting's difference from its own run's control. Standard library only.
"""
import argparse, csv, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--r225", required=True)
ap.add_argument("--c222", required=True)
ap.add_argument("--control", default="short=600:64")
a = ap.parse_args()


def load(path):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    cols = [c for c in rows[0] if c.startswith("p") and ":" in c]
    return {(r["task"], r["turn"]): r for r in rows}, cols


r225, c25 = load(a.r225)
c222, c22 = load(a.c222)
keys = list(r225)
ctl25 = [c for c in c25 if c.split(":", 1)[1] == a.control]
ctl22 = [c for c in c22 if c.split(":", 1)[1] == a.control and c.split(":")[0] != "p0"]
a64 = [c for c in c22 if c.split(":", 1)[1] == "short=2048:64"]
settings = sorted({c.split(":", 1)[1] for c in c25} - {a.control}, key=lambda s: int(s.split("=")[1].split(":")[0]))


def ms(r, cs):
    return statistics.mean(float(r[c]) for c in cs) / 1000


def limit(s):
    return int(s.split("=")[1].split(":")[0])


base22 = sum(ms(c222[k], ctl22) for k in keys)
print("setting\tpasses\ts_by_pass\tmean_s\tcontrol_mean_s\tdelta_s\tdelta_pct\tc222_est_delta_s\tturns_changed")
cm = statistics.mean(sum(float(r225[k][c]) for k in keys) / 1000 for c in ctl25)
print(f"{a.control}\t{', '.join(c.split(':')[0] for c in ctl25)}\t"
      f"{', '.join(f'{sum(float(r225[k][c]) for k in keys) / 1000:.2f}' for c in ctl25)}\t{cm:.2f}\t{cm:.2f}\t+0.00\t+0.00\t+0.00\t0")
for s in settings:
    cs = [c for c in c25 if c.split(":", 1)[1] == s]
    v = [sum(float(r225[k][c]) for k in keys) / 1000 for c in cs]
    n = limit(s)
    est = sum(ms(c222[k], a64 if 600 < int(c222[k]["call_tokens"]) <= n else ctl22) for k in keys) - base22
    ch = sum(1 for k in keys if 600 < int(r225[k]["call_tokens"]) <= n)
    m = statistics.mean(v)
    print(f"{s}\t{', '.join(c.split(':')[0] for c in cs)}\t{', '.join(f'{x:.2f}' for x in v)}\t{m:.2f}\t{cm:.2f}\t"
          f"{m - cm:+.2f}\t{(m - cm) / cm * 100:+.2f}\t{est:+.2f}\t{ch}")
print()
print("task\tturn\tcall_tokens\tcontrol_s\t" + "\t".join(f"{s}_delta_s" for s in settings)
      + "\tc222_control_s\tc222_short=2048:64_delta_s")
for k in sorted(keys, key=lambda k: int(r225[k]["call_tokens"])):
    n = int(r225[k]["call_tokens"])
    if not 600 < n <= 2048:
        continue
    c = ms(r225[k], ctl25)
    d = [ms(r225[k], [x for x in c25 if x.split(":", 1)[1] == s]) - c for s in settings]
    c2 = ms(c222[k], ctl22)
    print(f"{k[0]}\t{k[1]}\t{n}\t{c:.2f}\t" + "\t".join(f"{x:+.2f}" for x in d)
          + f"\t{c2:.2f}\t{ms(c222[k], a64) - c2:+.2f}")
