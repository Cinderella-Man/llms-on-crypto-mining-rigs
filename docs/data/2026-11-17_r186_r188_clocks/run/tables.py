#!/usr/bin/env python3
"""tables.py (R186 Q1551, R188 Q1553): the tables of run.sh's output.

  python3 -I tables.py DIR      (DIR holds run.sh's OUT: requests.tsv, irq.txt, snap.txt, run.log, smi-*.csv)
Writes DIR/tables/: requests.tsv (every request), arms.tsv (token time by arm, each request listed), clocks.tsv (each
card's SM clock in each decode window, by arm), r186a_timeline.tsv (the 500 ms log, one row a sample, card by card),
r186a_after.tsv (each card's first readings after the decode ended), requests_clocks.tsv (the
cards' clocks before and in each request of the 100 ms logs), interrupts.tsv (NVIDIA interrupts by window),
memory.tsv and modules.tsv (the snapshots). Times in the logs: requests in the client's epoch, nvidia-smi in the rig's
local time; run.log's first line gives the offset between the two clocks.
"""
import datetime as dt
import os
import re
import statistics as st
import sys

D = sys.argv[1]
T = os.path.join(D, "tables")
os.makedirs(T, exist_ok=True)


def w(name, header, rows):
    with open(os.path.join(T, name), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


# clock offset rig - client
m = re.search(r"clock offset: local ([\d.]+), rig ([\d.]+), local ([\d.]+)", open(os.path.join(D, "run.log")).read())
OFF = float(m.group(2)) - (float(m.group(1)) + float(m.group(3))) / 2

# requests
REQ = []
for line in open(os.path.join(D, "requests.tsv")):
    tag, t0, t1, pn, pms, n, ms, mspt = line.rstrip("\n").split("\t")
    REQ.append(dict(tag=tag, t0=float(t0) + OFF, t1=float(t1) + OFF, pn=int(pn), pms=float(pms), n=int(n),
                    ms=float(ms), mspt=float(mspt)))
w("requests.tsv", ["tag", "rig_start", "rig_end", "prompt_tokens", "prompt_ms", "tokens", "decode_ms", "ms_a_token"],
  [(r["tag"], f"{r['t0']:.3f}", f"{r['t1']:.3f}", r["pn"], r["pms"], r["n"], r["ms"], r["mspt"]) for r in REQ])
arms = {}
for r in REQ:
    arms.setdefault(r["tag"], []).append(r["mspt"])
w("arms.tsv", ["arm", "requests", "ms_a_token_each", "mean"],
  [(a, len(v), ", ".join(f"{x:.3f}" for x in v), f"{st.mean(v):.3f}") for a, v in arms.items()])


def window(r):  # the decode part of a request, in rig time
    return r["t0"] + r["pms"] / 1000 + 0.2, r["t1"] - 0.1


def smi(tag):
    p = os.path.join(D, f"smi-{tag}.csv")
    rows = []
    if not os.path.exists(p):
        return rows
    for line in open(p):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 8 or not f[1].isdigit():
            continue
        t = dt.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        rows.append(dict(t=t, card=int(f[1]), ps=f[2], sm=int(f[3]), mem=int(f[4]), w=f[5], reasons=f[6], util=f[7]))
    return rows


# clocks in each decode window, by arm
out = []
for log, tags in (("modesA", ("A_mode0", "A_mode1")), ("keepaliveB", ("B_keepalive",)), ("r186a", ("r186a",))):
    S = smi(log)
    for tag in tags:
        for card in range(9):
            v, ps, rs = [], set(), set()
            for r in REQ:
                if r["tag"] != tag:
                    continue
                a, b = window(r)
                for s in S:
                    if s["card"] == card and a <= s["t"] <= b:
                        v.append(s["sm"]); ps.add(s["ps"]); rs.add(s["reasons"])
            if v:
                out.append((tag, card, len(v), min(v), st.median(v), max(v),
                            f"{100 * sum(x >= 1800 for x in v) / len(v):.1f}", "/".join(sorted(ps)),
                            "/".join(sorted(rs))))
w("clocks.tsv", ["arm", "card", "samples", "sm_min", "sm_median", "sm_max", "pct_at_1800_or_more", "pstates",
                 "reasons"], out)

# R186 (a): the 500 ms log across the end of a decode
S = smi("r186a")
ra = [r for r in REQ if r["tag"] == "r186a"]
if S and ra:
    r = ra[0]
    a, b = window(r)
    rounds = []  # one nvidia-smi poll = one row per card, in card order
    for s in S:
        if s["card"] == 0 or not rounds:
            rounds.append({})
        rounds[-1][s["card"]] = s
    rows = []
    for c in rounds:
        t = c[min(c)]["t"]
        phase = "decode" if a <= t <= b else ("before" if t < r["t0"] else ("prompt/edge" if t < b + 0.2 else "after"))
        rows.append([f"{t - r['t1']:.1f}", phase] + [f"{c[k]['ps']} {c[k]['sm']} {c[k]['reasons']}" if k in c else ""
                                                    for k in range(9)])
    w("r186a_timeline.tsv", ["s_after_decode_end", "phase"] + [f"card{k}" for k in range(9)], rows)
    after = []
    for card in range(9):
        busy = [s["sm"] for s in S if s["card"] == card and a <= s["t"] <= b]
        med = st.median(busy) if busy else None
        below = [s for s in S if s["card"] == card and a <= s["t"] <= b and med and s["sm"] < med]
        seq, last = [], None
        for s in S:
            if s["card"] == card and s["t"] > r["t1"]:
                k = (s["ps"], s["sm"], s["reasons"])
                if k != last:
                    seq.append(f"+{s['t'] - r['t1']:.1f}s {s['ps']} {s['sm']} MHz {s['reasons']}")
                    last = k
        after.append((card, med, len(busy), len(below), min(busy) if busy else "", "; ".join(seq[:6])))
    w("r186a_after.tsv", ["card", "busy_median_mhz", "decode_samples", "decode_samples_below_median", "decode_min_mhz",
                          "after_the_decode_each_change"], after)

# each request of the two 100 ms logs: the cards' clocks just before it and in its decode
rows = []
for log, pre in (("modesA", "A_"), ("keepaliveB", "B_")):
    S = smi(log)
    for i, r in enumerate(q for q in REQ if q["tag"].startswith(pre)):
        a, b = window(r)
        before = {}
        for s in S:
            if s["t"] < r["t0"]:
                before[s["card"]] = s
        inside = [s for s in S if a <= s["t"] <= b]
        first = [s["sm"] for s in inside if s["t"] <= a + 1.0]
        rows.append((r["tag"], i + 1, r["mspt"], min(x["sm"] for x in before.values()) if before else "",
                     "/".join(sorted({x["ps"] for x in before.values()})), min(first) if first else "",
                     min(s["sm"] for s in inside), f"{100 * sum(s['sm'] >= 1800 for s in inside) / len(inside):.1f}",
                     len(inside)))
w("requests_clocks.tsv", ["arm", "order", "ms_a_token", "lowest_card_clock_before", "pstates_before",
                          "lowest_clock_first_1s", "lowest_clock_in_decode", "pct_samples_1800_or_more",
                          "samples_all_cards"], rows)

# interrupts
blocks, cur = {}, None
for line in open(os.path.join(D, "irq.txt")):
    if line.startswith("## irq"):
        _, _, tag, t = line.split()
        cur = tag
        blocks[tag] = dict(t=float(t), lines={})
    elif cur:
        f = line.split()
        irq = f[0].rstrip(":")
        n = 0
        for x in f[1:]:
            if not x.isdigit():
                break
            n += int(x)
        blocks[cur]["lines"][irq] = n
irqs = sorted(blocks["i0"]["lines"], key=lambda x: int(x) if x.isdigit() else 0)


def delta(x, y):
    return {i: blocks[y]["lines"][i] - blocks[x]["lines"][i] for i in irqs}


rows = []
idle_rates = []
for x, y in (("i0", "i1"), ("i2", "i3")):
    d, s = delta(x, y), blocks[y]["t"] - blocks[x]["t"]
    idle_rates.append(sum(d.values()) / s)
    rows.append((f"idle {x}-{y}", f"{s:.2f}", "", sum(d.values()), f"{sum(d.values()) / s:.2f}", "",
                 " ".join(f"{i}:{d[i]}" for i in irqs)))
idle = st.mean(idle_rates)
for (x, y), tag in ((("i1", "i2"), "irq1"), (("i3", "i4"), "irq2")):
    d, s = delta(x, y), blocks[y]["t"] - blocks[x]["t"]
    r = [q for q in REQ if q["tag"] == tag][0]
    tot = sum(d.values())
    rows.append((f"request {tag} {x}-{y}", f"{s:.2f}", r["n"] + r["pn"], tot, f"{tot / s:.2f}",
                 f"{(tot - idle * s) / (r['n'] + r['pn']):.3f}", " ".join(f"{i}:{d[i]}" for i in irqs)))
w("interrupts.tsv", ["window", "seconds", "tokens_prompt_plus_reply", "nvidia_interrupts", "per_second",
                     "per_token_above_idle_rate", "by_irq"], rows)

# memory and modules
snaps, cur = {}, None
for line in open(os.path.join(D, "snap.txt")):
    if line.startswith("## snap"):
        cur = line.split()[2]
        snaps[cur] = []
    elif cur:
        snaps[cur].append(line.rstrip("\n"))
mem_rows = []
keys = ["MemTotal", "MemFree", "MemAvailable", "Cached", "AnonPages", "Shmem", "Unevictable", "Mlocked", "Slab",
        "SUnreclaim", "KernelStack", "PageTables", "VmallocUsed", "CmaFree", "SwapTotal", "SwapFree"]
for tag, lines in snaps.items():
    kv = {}
    for l in lines:
        mm = re.match(r"(\w+):\s+(\d+) kB", l)
        if mm:
            kv[mm.group(1)] = int(mm.group(2)) // 1024
    top = [l.split() for l in lines if re.match(r"\s*\d+\s+\d+\s+\S+$", l)]
    srv = [int(c[1]) // 1024 for c in top if c[2] == "llama-server"]
    other = [int(c[1]) // 1024 for c in top if c[2] != "llama-server"]
    mem_rows.append([tag] + [kv.get(k, "") for k in keys] + [srv[0] if srv else 0, other[0] if other else ""])
w("memory.tsv", ["snapshot"] + [k + "_MiB" for k in keys] + ["llama_server_rss_MiB", "largest_other_rss_MiB"],
  mem_rows)
mods = [l.split() for l in snaps.get("empty", []) if l.startswith("nvidia")]
w("modules.tsv", ["module", "size_bytes", "size_MiB", "users"],
  [(x[0], x[1], f"{int(x[1]) / 2**20:.2f}", x[2]) for x in mods])
print("tables in", T)
