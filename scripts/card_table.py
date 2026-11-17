#!/usr/bin/env python3
"""card_table.py: per-card summary of a card log (scripts/card_log.py's cards.csv, or scripts/nvml_watchdog.py's log).

For each card over the log, or over a window of it:
  - seconds at each P-state;
  - busy samples, the SM clock's median and lowest while busy, and the busy samples below --low-mhz (1,800);
  - the memory clock's median while busy;
  - peak power, peak temperature, the most memory used;
  - the throttle (clock event) reasons by sample count, over all samples and over busy samples.
Then every stretch of at least --stretch-s seconds (2) in which a card was busy below --low-mhz in every sample: when,
how long, its lowest clock and the reasons seen. A reply's ramp from a resting clock lasts under a second on a P106
(R186), so a stretch of 2 s or more is a card that ran slow, not one waking up.

Busy is, by default, "a kernel ran": utilization above --util-min (0) %, where the log has utilization (card_log.py).
A log without it (nvml_watchdog.py) counts as busy a sample that is not in P8 and has no idle reason; --busy picks
either. Each sample stands for the time to the next one, up to --max-gap seconds (2); time beyond that is reported
as unlogged.

  python3 -I card_table.py cards.csv[.gz] [more logs] [--since T] [--until T] [--out cards.md] [--tsv cards.tsv]
          [--low-mhz 1800] [--stretch-s 2] [--busy auto|util|reason] [--util-min 0] [--max-gap 2] [--max-stretches 50]

T is Unix seconds or a local or ISO time ("2026-11-17 15:20:00", "2026-11-17T15:20:00+01:00"), e.g. a load's
`started` from the deep lane's loads.tsv. Prints Markdown (to --out, else stdout).
"""
import argparse
import csv
import gzip
import io
import re
import sys
import time
from datetime import datetime

REASONS = [(0x1, "idle"), (0x2, "applications clocks"), (0x4, "power cap"), (0x8, "hardware slowdown"),
           (0x10, "sync boost"), (0x20, "software thermal"), (0x40, "hardware thermal"), (0x80, "power brake"),
           (0x100, "display clock")]


def when(s):
    try:
        return float(s)
    except ValueError:
        return datetime.fromisoformat(s.strip()).timestamp()


def stamp(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def reason_names(mask):
    if mask == 0:
        return ["none"]
    names = [n for b, n in REASONS if mask & b]
    rest = mask & ~sum(b for b, _ in REASONS)
    return names + ([f"0x{rest:x}"] if rest else [])


def read_logs(paths):
    """(header, rows, events): rows as dicts with 't' added, sorted by time; events are the '#' lines."""
    header, rows, events = None, [], []
    for p in paths:
        raw = gzip.open(p, "rt") if p.endswith(".gz") else open(p)
        with raw as f:
            lines = []
            for line in f:
                if line.startswith("#"):
                    events.append(line[1:].strip())
                elif line.strip():
                    lines.append(line)
        if not lines:
            continue
        h = next(csv.reader([lines[0]]))
        if header is None:
            header = h
        elif h != header:
            sys.exit(f"{p}: its columns differ from the first log's")
        for r in csv.reader(io.StringIO("".join(lines[1:]))):
            if r == h or len(r) != len(h):
                continue
            d = dict(zip(h, r))
            if "epoch" in d:
                t = num(d["epoch"])
            else:  # nvml_watchdog.py: local "YYYY-MM-DD HH:MM:SS.mmm"
                try:
                    t = time.mktime(time.strptime(d["time"][:19], "%Y-%m-%d %H:%M:%S")) + int(d["time"][20:23] or 0) / 1000
                except (KeyError, ValueError):
                    t = None
            if t is not None:
                d["t"] = t
                rows.append(d)
    rows.sort(key=lambda d: d["t"])
    return header or [], rows, events


def median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def fmt(x, nd=0):
    if x is None:
        return "–"
    return f"{x:,.{nd}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--low-mhz", type=float, default=1800)
    p.add_argument("--stretch-s", type=float, default=2.0)
    p.add_argument("--busy", choices=("auto", "util", "reason"), default="auto")
    p.add_argument("--util-min", type=float, default=0)
    p.add_argument("--max-gap", type=float, default=2.0)
    p.add_argument("--max-stretches", type=int, default=50)
    p.add_argument("--out")
    p.add_argument("--tsv")
    a = p.parse_args()

    header, rows, events = read_logs(a.logs)
    since = when(a.since) if a.since else None
    until = when(a.until) if a.until else None
    rows = [d for d in rows if (since is None or d["t"] >= since) and (until is None or d["t"] <= until)]
    cards = sorted(int(m.group(1)) for c in header for m in [re.fullmatch(r"g(\d+)_sm_mhz", c)] if m)
    if not rows or not cards:
        sys.exit("no samples with an SM clock in the window")
    has_util = all(f"g{g}_util" in header for g in cards)
    busy_by = ("util" if has_util else "reason") if a.busy == "auto" else a.busy
    if busy_by == "util" and not has_util:
        sys.exit("--busy util: this log has no utilization column")
    busy_text = (f"utilization above {fmt(a.util_min)} % (a kernel ran in NVML's last sample period)" if busy_by == "util"
                 else "not in P8 and no idle reason (the log has no utilization, so a card held open at its base "
                      "clock with no work counts as busy)")

    # each sample stands for the time to the next one, capped
    ts = [d["t"] for d in rows]
    gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    step = median(gaps) or 0.0
    dts = [min(max(g, 0.0), a.max_gap) for g in gaps] + [min(step, a.max_gap)]
    long_gaps = [g for g in gaps if g > a.max_gap]
    unlogged = sum(g - a.max_gap for g in long_gaps)

    per, stretches = [], []
    pstates_seen = set()
    for g in cards:
        col = lambda k: f"g{g}_{k}"  # noqa: E731
        st = {"card": g, "samples": 0, "pstate_s": {}, "busy": 0, "busy_s": 0.0, "sm_busy": [], "mem_busy": [],
              "low": 0, "w": None, "c": None, "mib": None, "reasons": {}, "reasons_busy": {}}
        run = []

        def close_run():
            if run:
                dur = run[-1][0] - run[0][0] + run[-1][1]
                if dur >= a.stretch_s:
                    stretches.append({"card": g, "from": run[0][0], "to": run[-1][0] + run[-1][1], "s": dur,
                                      "samples": len(run), "lowest": min(r[2] for r in run),
                                      "median": median([r[2] for r in run]),
                                      "pstates": sorted(set(r[3] for r in run if r[3])),
                                      "reasons": sorted(set(n for r in run for n in r[4]))})
                run.clear()

        for d, dt in zip(rows, dts):
            sm = num(d.get(col("sm_mhz")))
            if sm is None:
                close_run()
                continue
            st["samples"] += 1
            ps = d.get(col("pstate"), "")
            if ps:
                pstates_seen.add(ps)
                st["pstate_s"][ps] = st["pstate_s"].get(ps, 0.0) + dt
            rs = d.get(col("reasons"), "")
            mask = int(rs, 16) if rs else None
            names = reason_names(mask) if mask is not None else []
            for n in names:
                st["reasons"][n] = st["reasons"].get(n, 0) + 1
            if busy_by == "util":
                u = num(d.get(col("util")))
                busy = u is not None and u > a.util_min
            else:
                busy = ps != "P8" and not (mask is not None and mask & 0x1)
            for k, key in (("w", "w"), ("c", "c"), ("used_mib", "mib")):
                v = num(d.get(col(k)))
                if v is not None and (st[key] is None or v > st[key]):
                    st[key] = v
            if busy:
                st["busy"] += 1
                st["busy_s"] += dt
                st["sm_busy"].append(sm)
                m = num(d.get(col("mem_mhz")))
                if m is not None:
                    st["mem_busy"].append(m)
                for n in names:
                    st["reasons_busy"][n] = st["reasons_busy"].get(n, 0) + 1
                if sm < a.low_mhz:
                    st["low"] += 1
                    run.append((d["t"], dt, sm, ps, names))
                    continue
            close_run()
        close_run()
        per.append(st)

    pss = sorted(pstates_seen, key=lambda s: int(s[1:]) if s[1:].isdigit() else 99)
    low = fmt(a.low_mhz)
    out = []
    w = out.append
    w(f"# Cards: {', '.join(a.logs)}")
    w("")
    w(f"{len(rows):,} samples from {stamp(ts[0])} to {stamp(ts[-1])} ({fmt(ts[-1] - ts[0], 1)} s); "
      f"{len(cards)} cards; a sample every {fmt(step, 3)} s (median), longest gap {fmt(max(gaps) if gaps else 0, 2)} s, "
      f"{len(long_gaps)} gaps over {fmt(a.max_gap, 1)} s ({fmt(unlogged, 1)} s unlogged).")
    w(f"Busy: {busy_text}. Low: an SM clock below {low} MHz.")
    if since or until:
        w(f"Window: {stamp(since) if since else 'start'} to {stamp(until) if until else 'end'}.")
    w("")
    w("| card | samples | " + " | ".join(f"{s} s" for s in pss) + f" | busy s | SM MHz busy, median | lowest | "
      f"busy below {low} MHz | memory MHz busy, median | peak W | peak °C | most MiB | reasons, all samples | "
      "reasons, busy samples |")
    w("|" + "---:|" * (6 + len(pss)) + "---:|---:|---:|---:|---|---|")
    for st in per:
        share = f"{st['low']:,} of {st['busy']:,} ({100 * st['low'] / st['busy']:.2f} %)" if st["busy"] else "–"
        rs = " · ".join(f"{n} {c:,}" for n, c in sorted(st["reasons"].items(), key=lambda x: -x[1])) or "–"
        rb = " · ".join(f"{n} {c:,}" for n, c in sorted(st["reasons_busy"].items(), key=lambda x: -x[1])) or "–"
        w(f"| {st['card']} | {st['samples']:,} | " + " | ".join(fmt(st["pstate_s"].get(s, 0.0), 1) for s in pss)
          + f" | {fmt(st['busy_s'], 1)} | {fmt(median(st['sm_busy']))} | {fmt(min(st['sm_busy']) if st['sm_busy'] else None)}"
          f" | {share} | {fmt(median(st['mem_busy']))} | {fmt(st['w'], 2)} | {fmt(st['c'])} | {fmt(st['mib'])} | {rs} | {rb} |")
    w("")
    w(f"## Busy stretches below {low} MHz, {fmt(a.stretch_s, 1)} s or longer")
    w("")
    if not stretches:
        w(f"None: no card was busy below {low} MHz for {fmt(a.stretch_s, 1)} s in a row.")
    else:
        stretches.sort(key=lambda s: s["from"])
        w(f"{len(stretches)} stretches.")
        w("")
        w("| card | from | to | s | samples | lowest MHz | median MHz | P-states | reasons |")
        w("|---:|---|---|---:|---:|---:|---:|---|---|")
        for s in stretches[:a.max_stretches]:
            w(f"| {s['card']} | {stamp(s['from'])} | {stamp(s['to'])} | {fmt(s['s'], 1)} | {s['samples']:,} | "
              f"{fmt(s['lowest'])} | {fmt(s['median'])} | {', '.join(s['pstates']) or '–'} | {', '.join(s['reasons']) or '–'} |")
        if len(stretches) > a.max_stretches:
            w(f"\nand {len(stretches) - a.max_stretches} more (--max-stretches).")
    if events:
        w("")
        w("## Log events")
        w("")
        for e in events:
            w(f"- {e}")
    text = "\n".join(out) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)

    if a.tsv:
        names = [n for _, n in REASONS] + ["none"]
        cols = (["card", "samples"] + [f"{s}_s" for s in pss] + ["busy", "busy_s", "sm_busy_median", "sm_busy_min",
                f"busy_below_{int(a.low_mhz)}", "mem_busy_median", "peak_w", "peak_c", "most_mib"]
                + [f"reason_{n.replace(' ', '_')}" for n in names] + [f"busy_reason_{n.replace(' ', '_')}" for n in names])
        with open(a.tsv, "w") as f:
            f.write("\t".join(cols) + "\n")
            for st in per:
                v = ([st["card"], st["samples"]] + [f"{st['pstate_s'].get(s, 0.0):.1f}" for s in pss]
                     + [st["busy"], f"{st['busy_s']:.1f}", median(st["sm_busy"]), min(st["sm_busy"]) if st["sm_busy"] else "",
                        st["low"], median(st["mem_busy"]), st["w"], st["c"], st["mib"]]
                     + [st["reasons"].get(n, 0) for n in names] + [st["reasons_busy"].get(n, 0) for n in names])
                f.write("\t".join("" if x is None else (f"{x:g}" if isinstance(x, float) else str(x)) for x in v) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
