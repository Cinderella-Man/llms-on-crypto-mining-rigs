#!/usr/bin/env python3
"""r123a_ladder.py: R123a's (Q1401) tables, the prompt switches as one ladder on R43's replay of an agent session's 62
prompts, one load a rung with two passes each, every rung on top of the one before.

    python3 -I r123a_ladder.py --load base=r123a_base/load-A --load graphs=r123a_graphs/load-A ... \
        --load base2=r123a_base2/load-A --base base --base2 base2 --out tables

The loads are given in ladder order: the first is the base, the last the base again (--base2); each load between is
classed against the load before it. Input per load folder (rig_taper.sh's load-A): requests.tsv (taper_replay.py),
memavail.txt (the watchdog's MemAvailable every 2 s). Only the replayed turns count (kind "turn"); the unmeasured
prefix requests are left out. A turn is binned by its planned new tokens (n_new) in R90's buckets (under 512,
512-1,023, 1,024-1,999, 2,000 and more), so the same turns pair across passes and loads.

Writes, in --out:
  turns.csv      every replayed turn in the agent check's turns.csv form for prompt_buckets.py (arm <load>:p<pass>,
                 prompt_n the turn's new tokens, prompt_ms the server's prompt time, ttft_s the request's wall time:
                 its prompt plus 8 greedy tokens, as the replay streams no first token)
  ladder.tsv     per load and bucket: turns, tokens, the server's prompt seconds of each pass by value, their mean,
                 tok/s over the mean, the load before's mean, the difference, the spread and the class; the last
                 rung also against the base's mean as "stack"
  first.tsv      per load and pass: the median and p90 of the request's wall time and of the server's prompt time over
                 the 62 turns
  replies.tsv    per load: turns whose greedy reply (8 tokens) is in the base's set in both passes, the same against
                 the base again, and turns whose two passes agree
  memory.tsv     per load: the lowest MemAvailable and when, relative to the load's first sample
  decision.md    the classes in words and what goes into the stacked line by the rule
Class: a load's value is the mean of its two passes. Gain when it is below the load before's by more than the spread,
loss when above by more, negligible otherwise. The spread is the difference between the two base loads' means in that
bucket (the same line loaded twice, first and last). p90 is statistics.quantiles(n=10, method="inclusive")[8].
Standard library only.
"""
import argparse, csv, os, statistics

ap = argparse.ArgumentParser()
ap.add_argument("--load", action="append", required=True, help="name=folder of one load, in ladder order")
ap.add_argument("--base", required=True)
ap.add_argument("--base2", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

BUCKETS = ((0, 512), (512, 1024), (1024, 2000), (2000, 10 ** 9), (0, 10 ** 9))


def bname(lo, hi):
    return "all" if (lo, hi) == (0, 10 ** 9) else (f"{lo}+" if hi == 10 ** 9 else f"{lo}-{hi - 1}")


order, folder, loads = [], {}, {}
for spec in a.load:
    name, path = spec.split("=", 1)
    order.append(name)
    folder[name] = path
    rows = [r for r in csv.DictReader(open(os.path.join(path, "requests.tsv")), delimiter="\t") if r["kind"] == "turn"]
    per = {}
    for r in rows:
        r["n_new"], r["prompt_ms"], r["wall_s"] = int(r["n_new"]), float(r["prompt_ms"]), float(r["wall_s"])
        per.setdefault(int(r["pass"]), {})[(r["task"], r["turn"])] = r
    loads[name] = dict(sorted(per.items()))
assert a.base in loads and a.base2 in loads

# the turns every pass of every load holds
keys = None
for name in order:
    for ts in loads[name].values():
        keys = set(ts) if keys is None else keys & set(ts)
keys = sorted(keys)
first_pass = next(iter(loads[a.base].values()))
n_new = {k: first_pass[k]["n_new"] for k in keys}
nk = len(keys)


def secs(name, lo, hi):
    return [sum(ts[k]["prompt_ms"] for k in keys if lo <= n_new[k] < hi) / 1000 for ts in loads[name].values()]


def mean(name, lo, hi):
    return statistics.mean(secs(name, lo, hi))


def spread(lo, hi):
    return abs(mean(a.base, lo, hi) - mean(a.base2, lo, hi))


def fmt(v):
    return ", ".join(f"{x:.2f}" for x in v)


# turns.csv for prompt_buckets.py
with open(f"{a.out}/turns.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "task_id", "turn", "prompt_n", "prompt_ms", "ttft_s"])
    for name in order:
        for p, ts in loads[name].items():
            for k in keys:
                r = ts[k]
                w.writerow([f"{name}:p{p}", k[0], k[1], r["n_new"], r["prompt_ms"], r["wall_s"]])

# ladder.tsv and the classes
cls = {}  # (load, vs, bucket) -> (class, delta, pct, spread, mean, ref_mean)
with open(f"{a.out}/ladder.tsv", "w") as f:
    f.write("load\tagainst\tnew_tokens\tturns\ttokens\tprompt_s_by_pass\tmean_s\ttok_s\tagainst_mean_s\tdelta_s\t"
            "delta_pct\tspread_s\tclass\n")
    for i, name in enumerate(order):
        pairs = []
        if name == a.base:
            pairs = [(None, "reference")]
        elif name == a.base2:
            pairs = [(a.base, "reference")]
        else:
            pairs = [(order[i - 1], None)]
            if i == len(order) - 2:
                pairs.append((a.base, "stack"))
        for ref, tag in pairs:
            for lo, hi in BUCKETS:
                kb = [k for k in keys if lo <= n_new[k] < hi]
                if not kb:
                    continue
                v, m, tok = secs(name, lo, hi), mean(name, lo, hi), sum(n_new[k] for k in kb)
                sp = spread(lo, hi)
                if ref is None:
                    f.write(f"{name}\t-\t{bname(lo, hi)}\t{len(kb)}\t{tok}\t{fmt(v)}\t{m:.2f}\t{tok / m:.1f}\t-\t-\t-\t"
                            f"{sp:.2f}\treference\n")
                    continue
                rm = mean(ref, lo, hi)
                d = m - rm
                c = tag if tag == "reference" else ("gain" if d < -sp else "loss" if d > sp else "negligible")
                cls[(name, tag or "before", bname(lo, hi))] = (c, d, d / rm * 100, sp, m, rm, v, secs(ref, lo, hi))
                f.write(f"{name}\t{ref}\t{bname(lo, hi)}\t{len(kb)}\t{tok}\t{fmt(v)}\t{m:.2f}\t{tok / m:.1f}\t{rm:.2f}\t"
                        f"{d:+.2f}\t{d / rm * 100:+.2f}\t{sp:.2f}\t{c}\n")


# first.tsv
def p90(v):
    return statistics.quantiles(v, n=10, method="inclusive")[8]


with open(f"{a.out}/first.tsv", "w") as f:
    f.write("load\tpass\tturns\twall_median_s\twall_p90_s\tprompt_median_s\tprompt_p90_s\n")
    for name in order:
        for p, ts in loads[name].items():
            w = [ts[k]["wall_s"] for k in keys]
            pm = [ts[k]["prompt_ms"] / 1000 for k in keys]
            f.write(f"{name}\t{p}\t{nk}\t{statistics.median(w):.2f}\t{p90(w):.2f}\t{statistics.median(pm):.2f}\t"
                    f"{p90(pm):.2f}\n")


# replies.tsv
def same(name, ref):
    want = {k: {ts[k]["gen_md5"] for ts in loads[ref].values()} for k in keys}
    return sum(1 for k in keys if all(ts[k]["gen_md5"] in want[k] for ts in loads[name].values()))


def differ(name, ref):
    want = {k: {ts[k]["gen_md5"] for ts in loads[ref].values()} for k in keys}
    return [f"{k[0]}:{k[1]}" for k in keys if not all(ts[k]["gen_md5"] in want[k] for ts in loads[name].values())]


rep = {}
with open(f"{a.out}/replies.tsv", "w") as f:
    f.write("load\tturns\tsame_as_base\tsame_as_base2\tpasses_agree\tturns_not_as_base\n")
    for name in order:
        agree = sum(1 for k in keys if len({ts[k]["gen_md5"] for ts in loads[name].values()}) == 1)
        sb, sb2 = same(name, a.base), same(name, a.base2)
        rep[name] = sb
        f.write(f"{name}\t{nk}\t{sb}\t{sb2}\t{agree}\t{' '.join(differ(name, a.base)) or '-'}\n")

# memory.tsv
mem = {}
with open(f"{a.out}/memory.tsv", "w") as f:
    f.write("load\tsamples\tlowest_memavailable_mib\tat_s\tfirst_mib\tlast_mib\n")
    for name in order:
        s = [tuple(map(int, line.split()[:2])) for line in open(os.path.join(folder[name], "memavail.txt"))
             if len(line.split()) >= 2]
        lo = min(s, key=lambda x: x[1])
        mem[name] = lo[1]
        f.write(f"{name}\t{len(s)}\t{lo[1]}\t{lo[0] - s[0][0]}\t{s[0][1]}\t{s[-1][1]}\n")

# decision.md
bk = [bname(lo, hi) for lo, hi in BUCKETS if (lo, hi) != (0, 10 ** 9)]
with open(f"{a.out}/decision.md", "w") as f:
    b, b2 = cls[(a.base2, "reference", "all")][7], cls[(a.base2, "reference", "all")][6]
    f.write(f"{nk} replayed turns in every pass of every load. The reference: {a.base} {fmt(b)} s a pass, "
            f"{a.base2} {fmt(b2)}; the spread {spread(0, 10 ** 9):.2f} s ({a.base} {mean(a.base, 0, 10 ** 9):.2f}, "
            f"{a.base2} {mean(a.base2, 0, 10 ** 9):.2f}). Replies: {a.base2} {rep[a.base2]} of {nk} as {a.base}.\n\n")
    for i, name in enumerate(order):
        if name in (a.base, a.base2):
            continue
        for tag in ("before", "stack"):
            if (name, tag, "all") not in cls:
                continue
            c, d, pct, sp, m, rm, v, rv = cls[(name, tag, "all")]
            ref = order[i - 1] if tag == "before" else a.base
            per = ", ".join(f"{x} {cls[(name, tag, x)][0]} ({cls[(name, tag, x)][1]:+.2f} s, spread "
                            f"{cls[(name, tag, x)][3]:.2f})" for x in bk if (name, tag, x) in cls)
            lost = [x for x in bk if (name, tag, x) in cls and cls[(name, tag, x)][0] == "loss"]
            if tag == "before":
                into = (c == "gain" or (c == "negligible" and rep[name] == nk))
                verdict = ("into the stacked line" if into else "to its own row") + (
                    f"; loses at {', '.join(lost)}" if lost else "")
            else:
                verdict = "the stack against the base"
            f.write(f"- {name} against {ref}: {fmt(v)} s a pass against {fmt(rv)} (means {m:.2f} and {rm:.2f}, "
                    f"{d:+.2f} s, {pct:+.2f} %, spread {sp:.2f}) → {c}; replies {rep[name]} of {nk} as {a.base}; "
                    f"lowest MemAvailable {mem[name]} MiB → {verdict}. By new tokens: {per}\n")
print(open(f"{a.out}/decision.md").read())
