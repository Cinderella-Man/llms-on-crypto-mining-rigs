#!/usr/bin/env python3
"""r60_buckets.py: R60's (Q1235) tables by the size of a turn's new prompt, the class of every taper setting in every
size bucket against the taper off, and the pick by R60's rule.

    python3 -I r60_buckets.py --load a=r60a/load-A --load ro8=r60ro8/load-A --base a --out tables
    python3 -I ../../2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py tables/turns.csv <arm>... > prompt_buckets.tsv

Input: the load folders of rig_taper.sh (requests.tsv of taper_replay.py). Only the replayed turns count (kind
"turn"); the unmeasured prefix requests and the fresh prompts are left out. A turn is binned by its planned new tokens
(n_new) in R90's buckets (prompt_buckets.py: under 512, 512-1,023, 1,024-1,999, 2,000 and more), so the same turns
pair across passes and loads.

Writes, in --out:
  turns.csv      every replayed turn in the agent check's turns.csv form for prompt_buckets.py: arm is
                 <load>:p<pass>:<setting>, prompt_n the turn's new tokens, prompt_ms the server's prompt time, ttft_s
                 the request's wall time (its prompt plus 8 greedy tokens; the replay has no streamed first token)
  buckets.tsv    per load, setting and bucket: the turns, new tokens, seconds of each pass by value, their mean, the
                 base's taper-off passes by value and mean, the difference, the spread and the class
  replies.tsv    per load and setting: turns whose greedy reply equals the reference's in every pass
  decision.md    the classes in words and the pick by R60's rule
Class: gain when the difference < -spread, loss when > +spread, negligible otherwise; the spread is the larger of the
base's two taper-off passes' difference in that bucket and the setting's own (when it has two passes). Standard
library only.
"""
import argparse, csv, os, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--load", action="append", required=True, help="name=folder of one load")
ap.add_argument("--base", required=True, help="the load whose taper-off passes are the reference")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

BUCKETS = ((0, 512), (512, 1024), (1024, 2000), (2000, 10 ** 9), (0, 10 ** 9))
NAMES = {"off": "off", "on": "on", "tail=512:128,short=600:64": "tail+short", "short=600:64": "short"}


def bname(lo, hi):
    return "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo}+" if hi == 10 ** 9 else f"{lo}-{hi - 1}")


def setting(arm):
    return NAMES.get(arm, arm)


loads = {}
for spec in a.load:
    name, path = spec.split("=", 1)
    rows = [r for r in csv.DictReader(open(os.path.join(path, "requests.tsv")), delimiter="\t") if r["kind"] == "turn"]
    for r in rows:
        r["pass"], r["n_new"] = int(r["pass"]), int(r["n_new"])
        r["prompt_ms"], r["wall_s"] = float(r["prompt_ms"]), float(r["wall_s"])
        r["setting"] = setting(r["arm"])
    loads[name] = rows
assert a.base in loads


def passes(load, s):
    """{pass: {(task, turn): row}} of one setting in one load"""
    out = {}
    for r in loads[load]:
        if r["setting"] == s:
            out.setdefault(r["pass"], {})[(r["task"], r["turn"])] = r
    return dict(sorted(out.items()))


def settings(load):
    order = ["off", "on", "tail+short", "short"]
    have = {r["setting"] for r in loads[load]}
    return [s for s in order if s in have] + sorted(have - set(order))


# the turns every pass of every load holds
keys = None
for name in loads:
    for s in settings(name):
        for ts in passes(name, s).values():
            keys = set(ts) if keys is None else keys & set(ts)
keys = sorted(keys)
ref = passes(a.base, "off")
n_new = {k: next(iter(ref.values()))[k]["n_new"] for k in keys}


def sums(load, s, lo, hi):
    return [sum(ts[k]["prompt_ms"] for k in keys if lo <= n_new[k] < hi) / 1000 for ts in passes(load, s).values()]


# turns.csv for prompt_buckets.py
with open(f"{a.out}/turns.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "task_id", "turn", "prompt_n", "prompt_ms", "ttft_s"])
    for name in loads:
        for s in settings(name):
            for p, ts in passes(name, s).items():
                for k in keys:
                    r = ts[k]
                    w.writerow([f"{name}:p{p}:{s}", k[0], k[1], r["n_new"], r["prompt_ms"], r["wall_s"]])

# buckets.tsv and the classes
cls = {}  # (load, setting, bucket) -> class
lines = []
with open(f"{a.out}/buckets.tsv", "w") as f:
    f.write("load\tsetting\tnew_tokens\tturns\ttokens\tprompt_s_by_pass\tmean_s\toff_s_by_pass\toff_mean_s\t"
            "delta_s\tdelta_pct\tspread_s\tclass\n")
    for name in loads:
        for s in settings(name):
            for lo, hi in BUCKETS:
                kb = [k for k in keys if lo <= n_new[k] < hi]
                if not kb:
                    continue
                v, o = sums(name, s, lo, hi), sums(a.base, "off", lo, hi)
                m, om = statistics.mean(v), statistics.mean(o)
                spread = max(max(o) - min(o), max(v) - min(v))
                d = m - om
                c = "base" if (name == a.base and s == "off") else (
                    "gain" if d < -spread else "loss" if d > spread else "negligible")
                cls[(name, s, bname(lo, hi))] = (c, v, m, o, om, d, spread)
                f.write(f"{name}\t{s}\t{bname(lo, hi)}\t{len(kb)}\t{sum(n_new[k] for k in kb)}\t"
                        f"{', '.join(f'{x:.2f}' for x in v)}\t{m:.2f}\t{', '.join(f'{x:.2f}' for x in o)}\t{om:.2f}\t"
                        f"{d:+.2f}\t{d / om * 100:+.2f}\t{spread:.2f}\t{c}\n")


# replies: against the base's off (every setting) and, for another load, against the base's same setting
def same(load, s, ref_load, ref_s):
    rp = passes(ref_load, ref_s)
    if not rp:
        return None
    want = {k: {ts[k]["gen_md5"] for ts in rp.values()} for k in keys}
    return sum(1 for k in keys if all(ts[k]["gen_md5"] in want[k] for ts in passes(load, s).values()))


with open(f"{a.out}/replies.tsv", "w") as f:
    f.write("load\tsetting\treference\tturns\tsame_reply\n")
    for name in loads:
        for s in settings(name):
            refs = [(a.base, "off")] + ([(a.base, s)] if name != a.base and s != "off" else [])
            for rl, rs in refs:
                if (name, s) == (rl, rs):
                    continue
                n = same(name, s, rl, rs)
                if n is not None:
                    f.write(f"{name}\t{s}\t{rl}:{rs}\t{len(keys)}\t{n}\n")

# decision: R60's rule over the base load's taper settings
bk = [bname(lo, hi) for lo, hi in BUCKETS if (lo, hi) != (0, 10 ** 9)]
with open(f"{a.out}/decision.md", "w") as f:
    f.write(f"{len(keys)} replayed turns in every pass; the reference is load {a.base}'s taper off.\n\n")
    for name in loads:
        for s in settings(name):
            if (name, s) == (a.base, "off"):
                continue
            c = cls[(name, s, "all")]
            per = ", ".join(f"{b} {cls[(name, s, b)][0]} ({cls[(name, s, b)][5]:+.2f} s)" for b in bk if (name, s, b) in cls)
            f.write(f"- {name} {s}: {', '.join(f'{x:.2f}' for x in c[1])} s a pass against "
                    f"{', '.join(f'{x:.2f}' for x in c[3])} ({c[5]:+.2f} s, {c[5] / c[4] * 100:+.2f} %, spread "
                    f"{c[6]:.2f}) → {c[0]}; by new tokens: {per}\n")
    ok = [(cls[(a.base, s, "all")][2], s) for s in settings(a.base) if s != "off"
          and not any(cls[(a.base, s, b)][0] == "loss" for b in bk if (a.base, s, b) in cls)]
    lose_1k = [s for s in settings(a.base) if s != "off" and cls.get((a.base, s, "1024-1999"), ("",))[0] == "loss"]
    f.write("\n")
    if ok:
        m, s = min(ok)
        f.write(f"Pick: {s} ({m:.2f} s a pass), the lowest of the settings that lose in no bucket "
                f"({', '.join(x for _, x in sorted(ok))}).\n")
    elif len(lose_1k) == len([s for s in settings(a.base) if s != "off"]):
        f.write("Every taper setting loses at 1,024-1,999 new tokens: the taper leaves the candidate.\n")
    else:
        f.write("No taper setting is free of a loss in every bucket, and not every one loses at 1,024-1,999.\n")
print(open(f"{a.out}/decision.md").read())
