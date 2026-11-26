#!/usr/bin/env python3
"""R244 (Q1924): the tables of run/r244.sh's six loads (two of each arm: p512 the v5 preset, u256 the same with
-ub 256, l131 the same with -c 131072 -ub 256).

    python3 -I tables.py OUT TABLES_DIR [USABLE_MIB]

OUT holds one folder per load (p512a, u256a, l131a, p512b, u256b, l131b) as r244.sh writes it: replay.jsonl,
replay_window.txt and rig/ (cards.csv, meminfo.log). Writes to TABLES_DIR:
  requests.tsv  every replayed request: load, arm, group (first, erased, small, mid), task, new tokens read
                (prompt_n), tokens from the cache (cache_n), prompt ms, ms a prompt token, tokens generated, and
                R243's own prompt_n and cache_n for the same request (its first run, the recorded log), so a request
                that read other tokens than R243's shows
  loads.tsv     per load: prompt ms of the first request, of the erased one, the first's extra over the erased one
                (paid once a load), of the four small first turns together (and their first, the first short call
                after the long reads, apart from the other three), of the two mid ones together and of all eight;
                card 8's MiB after loading and at its highest over the replay; the
                host's lowest MemAvailable and highest swap use over the replay
  classes.tsv   each measure (those of loads.tsv, the seven requests after the first, card 8) for each pair of arms (u256 against p512: the micro-batch alone; l131 against u256: the
                context alone; l131 against p512: R243's comparison): both loads' values, the means, each arm's spread
                (the larger value less the smaller), the difference and its class by the release check's rule, as
                R243's run/same_sizes.py applies it: a gain or a loss only beyond the larger of the two arms'
                spreads, else negligible; prompt ms and MiB are better lower
  cards.tsv     every card's MiB after loading and at its highest, per load, and what the highest leaves of USABLE_MIB
Card memory is on nvidia-smi's scale: each card's first reading in the card log (taken before the model loads) is
subtracted, as R243's run/card_memory.py does; USABLE_MIB (default 6,068) is what a P106-090 offers a process under
driver 580 (6,144 less its 76 MiB reserve). Standard library only.
"""
import csv
import datetime
import json
import sys
from pathlib import Path

out, tdir = Path(sys.argv[1]), Path(sys.argv[2])
usable = int(sys.argv[3]) if len(sys.argv) > 3 else 6068
tdir.mkdir(parents=True, exist_ok=True)
LOADS = ["p512a", "u256a", "l131a", "p512b", "u256b", "l131b"]
GROUPS = ["first", "erased", "small", "mid"]


req_lines, loads = [], {}
for load in LOADS:
    f = out / load / "replay.jsonl"
    if not f.exists():
        continue
    rows = [json.loads(x) for x in open(f) if x.strip()]
    per = {g: [] for g in GROUPS}
    for r in rows:
        if r.get("error"):
            sys.exit(f"{load}: request {r['trace']} #{r['idx']} failed: {r['error']}")
        g = r["label"].split("-", 1)[1]
        per[g].append(r)
        rec = r.get("rec") or {}
        task = r["trace"].split("/")[1]
        pn, pm = r.get("prompt_n"), r.get("prompt_ms")
        req_lines.append([load, load[:-1], g, task, pn, r.get("cache_n"), f"{pm:.1f}", f"{pm / pn:.3f}" if pn else "",
                          r.get("predicted_n"), rec.get("prompt_n"), rec.get("cache_n")])
    win = (out / load / "replay_window.txt").read_text().split("\n")
    first_t, last_t = win[0].strip(), win[1].strip()
    cards = list(csv.DictReader(ln for ln in open(out / load / "rig" / "cards.csv") if not ln.startswith("#")))
    before = [c for c in cards if c["time"][:19] < first_t]
    during = [c for c in cards if first_t <= c["time"][:19] <= last_t]
    card = {}
    for i in range(9):
        col = f"g{i}_used_mib"
        idle = float(cards[0][col])
        after = float(before[-1][col]) - idle
        peak = max(float(c[col]) for c in during) - idle
        card[i] = (int(after), int(peak))
    mem = [ln.split() for ln in open(out / load / "rig" / "meminfo.log") if ln.strip()]
    # meminfo.log rows are "<epoch> <MemAvailable kB> <swap used kB>"; the replay window in epoch seconds
    e0 = datetime.datetime.strptime(first_t, "%Y-%m-%d %H:%M:%S").timestamp()
    e1 = datetime.datetime.strptime(last_t, "%Y-%m-%d %H:%M:%S").timestamp()
    mw = [m for m in mem if len(m) == 3 and e0 <= int(m[0]) <= e1 + 1]
    loads[load] = dict(
        first=sum(r["prompt_ms"] for r in per["first"]),
        erased=sum(r["prompt_ms"] for r in per["erased"]),
        small=sum(r["prompt_ms"] for r in per["small"]),
        mid=sum(r["prompt_ms"] for r in per["mid"]),
        small_n=sum(r["prompt_n"] for r in per["small"]),
        mid_n=sum(r["prompt_n"] for r in per["mid"]),
        erased_cache=[r.get("cache_n") for r in per["erased"]],
        tri=sum(r["prompt_ms"] for r in per["small"][:1]),
        small_rest=sum(r["prompt_ms"] for r in per["small"][1:]),
        card8_after=card[8][0], card8_peak=card[8][1], cards=card,
        avail_min=min(int(m[1]) for m in mw) // 1024 if mw else -1,
        swap_max=max(int(m[2]) for m in mw) // 1024 if mw else -1,
        n=len(rows))
    d = loads[load]
    d["once"] = d["first"] - d["erased"]
    d["all8"] = d["first"] + d["erased"] + d["small"] + d["mid"]
    d["after_first"] = d["all8"] - d["first"]

with open(tdir / "requests.tsv", "w") as w:
    w.write("load\tarm\tgroup\ttask\tprompt_n\tcache_n\tprompt_ms\tms_per_token\tpredicted_n\tr243_prompt_n\tr243_cache_n\n")
    for ln in req_lines:
        w.write("\t".join("" if x is None else str(x) for x in ln) + "\n")

with open(tdir / "loads.tsv", "w") as w:
    w.write("load\tarm\trequests\tfirst_ms\terased_ms\terased_cache_n\tonce_ms\tsmall_ms\tsmall_tokens\tfirst_short_ms\t"
            "other_short_ms\tmid_ms\tmid_tokens\tall8_ms\t"
            "card8_after_load_mib\tcard8_peak_mib\tcard8_left_mib\thost_min_avail_mib\thost_max_swap_mib\n")
    for load, d in loads.items():
        w.write(f"{load}\t{load[:-1]}\t{d['n']}\t{d['first']:.1f}\t{d['erased']:.1f}\t{','.join(map(str, d['erased_cache']))}\t"
                f"{d['once']:.1f}\t{d['small']:.1f}\t{d['small_n']}\t{d['tri']:.1f}\t{d['small_rest']:.1f}\t"
                f"{d['mid']:.1f}\t{d['mid_n']}\t{d['all8']:.1f}\t{d['card8_after']}\t{d['card8_peak']}\t"
                f"{usable - d['card8_peak']}\t{d['avail_min']}\t{d['swap_max']}\n")

with open(tdir / "cards.tsv", "w") as w:
    w.write("load\tcard\tafter_load_mib\tpeak_mib\tleft_at_peak_mib\n")
    for load, d in loads.items():
        for i, (a, p) in d["cards"].items():
            w.write(f"{load}\t{i}\t{a}\t{p}\t{usable - p}\n")

MEASURES = [("first request, empty cache on a fresh server (prompt ms)", "first"),
            ("the same request after erasing the slot (prompt ms)", "erased"),
            ("the first request's extra over the erased one, paid once a load (prompt ms)", "once"),
            ("4 first turns under 600 new tokens (prompt ms together)", "small"),
            ("of them the first, the first short call after the long reads (javascript triangle; prompt ms)", "tri"),
            ("of them the other three (prompt ms together)", "small_rest"),
            ("2 first turns of 600 to 1,400 new tokens (prompt ms together)", "mid"),
            ("all eight requests (prompt ms together)", "all8"),
            ("the seven after the first (prompt ms together)", "after_first"),
            ("card 8 after loading (MiB)", "card8_after"),
            ("card 8 at its highest (MiB)", "card8_peak")]
PAIRS = [("u256", "p512", "the micro-batch alone"), ("l131", "u256", "the context alone"),
         ("l131", "p512", "both (R243's comparison)")]
with open(tdir / "classes.tsv", "w") as w:
    w.write("measure\tpair\twhat\tarm_loads\tbase_loads\tarm_mean\tbase_mean\tarm_spread\tbase_spread\tdiff\tdiff_pct\tclass\n")
    for label, key in MEASURES:
        for arm, base, what in PAIRS:
            av = [loads[arm + s][key] for s in "ab" if arm + s in loads]
            bv = [loads[base + s][key] for s in "ab" if base + s in loads]
            if len(av) != 2 or len(bv) != 2:
                continue
            am, bm = sum(av) / 2, sum(bv) / 2
            asp, bsp, d = max(av) - min(av), max(bv) - min(bv), am - bm
            cls = "negligible" if abs(d) <= max(asp, bsp) else ("gain" if d < 0 else "loss")
            fmt = (lambda x: f"{x:.1f}") if isinstance(av[0], float) else (lambda x: f"{x:g}")
            w.write(f"{label}\t{arm} vs {base}\t{what}\t{fmt(av[0])} {fmt(av[1])}\t{fmt(bv[0])} {fmt(bv[1])}\t{am:.1f}\t"
                    f"{bm:.1f}\t{asp:.1f}\t{bsp:.1f}\t{d:+.1f}\t{d / bm * 100:+.1f} %\t{cls}\n")
print(f"{len(loads)} loads, {len(req_lines)} requests -> {tdir}")
