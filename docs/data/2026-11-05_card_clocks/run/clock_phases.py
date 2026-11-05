#!/usr/bin/env python3
"""The cards' clocks per phase of clocks_run.sh (C532, Q1021).

    python3 -I clock_phases.py OUT

Reads OUT's smi.csv (nvidia-smi, csv,noheader,nounits: timestamp, index, pstate, clocks.sm, clocks.mem, power.draw,
clocks_event_reasons.active), marks.tsv, bench_out.tsv and bench_err.tsv (llama-bench's jsonl and --progress lines,
each stamped with $EPOCHREALTIME), reply.json (the served request's timings) and writes into OUT:
  phases.tsv       every phase's windows (epoch seconds)
  bench.tsv        llama-bench's results per instance
  served.tsv       the served request's timings
  clocks.tsv       per phase and card (and "all"): samples, the share at base (SM < BASE_MHZ, 1,400) and at boost
                   (SM >= BOOST_MHZ, 1,800), SM and memory clock quartiles, P-states, the clock-event reasons, power
  clocks.md        the same as Markdown tables
  timeline.tsv     the served request second by second (from 3 s before the send to 3 s after the reply): each card's
                   median SM clock in that second, and the phase
Phases: idle (nothing loaded), bench_k4 (the timed repetitions of the 4-token passes), bench_k1 (tg128's timed
repetitions), served_idle (the server loaded, no request), served_prompt and served_decode (the request, split by the
reply's prompt_ms and predicted_ms: the prompt from the send, the decode up to the reply). Standard library only.
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

BASE_MHZ, BOOST_MHZ = 1400, 1800
REASONS = [(0x1, "idle"), (0x4, "sw_power_cap"), (0x8, "hw_slowdown"), (0x20, "sw_thermal"), (0x40, "hw_thermal"),
           (0x80, "power_brake"), (0x2, "app_clocks"), (0x10, "sync_boost"), (0x100, "display")]
ORDER = ["idle", "bench_k4", "bench_k1", "served_idle", "served_prompt", "served_decode"]


def smi_epoch(ts):
    day, frac = ts.split(".") if "." in ts else (ts, "0")
    return time.mktime(time.strptime(day, "%Y/%m/%d %H:%M:%S")) + float("0." + frac)


def stamped(path):
    for line in path.read_text().splitlines():
        t, _, rest = line.partition("\t")
        yield float(t), rest


def phases(out):
    marks = {name: t for t, name in stamped(out / "marks.tsv")}
    win = {p: [] for p in ORDER}
    win["idle"].append((marks["idle_start"], marks["idle_end"]))
    starts = {}
    for t, l in stamped(out / "bench_err.tsv"):
        m = re.search(r"benchmark (\d+)/\d+: (prompt|generation) run 1/", l)
        if m:
            starts[int(m.group(1))] = t
    bench = []
    results = [(t, json.loads(l)) for t, l in stamped(out / "bench_out.tsv") if l.startswith("{")]
    for i, (t, r) in enumerate(results, 1):
        phase = "bench_k4" if r.get("n_prompt", 0) > 0 else "bench_k1"
        if i in starts:
            win[phase].append((starts[i], t))
        bench.append((i, phase, r))
    served = {}
    reply = out / "reply.json"
    if "request_sent" in marks and reply.exists() and reply.stat().st_size:
        tm = json.loads(reply.read_text()).get("timings", {})
        sent, got = marks["request_sent"], marks["reply_received"]
        win["served_idle"].append((marks["server_ready"], sent))
        win["served_prompt"].append((sent, sent + tm.get("prompt_ms", 0) / 1000))
        win["served_decode"].append((got - tm.get("predicted_ms", 0) / 1000, got))
        served = dict(tm, wall_s=got - sent)
    return win, bench, served


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def stats(rows):
    sm = [r[3] for r in rows]
    mem = [r[4] for r in rows]
    pw = [r[5] for r in rows]
    n = len(rows)
    d = {"samples": n,
         "base_pct": 100 * sum(s < BASE_MHZ for s in sm) / n,
         "mid_pct": 100 * sum(BASE_MHZ <= s < BOOST_MHZ for s in sm) / n,
         "boost_pct": 100 * sum(s >= BOOST_MHZ for s in sm) / n,
         "sm_p10": q(sm, .1), "sm_med": q(sm, .5), "sm_p90": q(sm, .9),
         "mem_p10": q(mem, .1), "mem_med": q(mem, .5),
         "power_mean_w": statistics.fmean(pw), "power_max_w": max(pw)}
    for ps in ("P0", "P2", "P5", "P8"):
        d[ps + "_pct"] = 100 * sum(r[2] == ps for r in rows) / n
    d["no_reason_pct"] = 100 * sum(r[6] == 0 for r in rows) / n
    for bit, name in REASONS:
        d[name + "_pct"] = 100 * sum(bool(r[6] & bit) for r in rows) / n
    return d


def main():
    out = Path(sys.argv[1])
    win, bench, served = phases(out)
    rows = []
    for line in (out / "smi.csv").read_text().splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            rows.append((smi_epoch(f[0]), int(f[1]), f[2], float(f[3]), float(f[4]), float(f[5]), int(f[6], 16)))
        except (ValueError, IndexError):
            continue
    cards = sorted({r[1] for r in rows})
    with open(out / "phases.tsv", "w") as fh:
        fh.write("phase\tstart\tend\tseconds\n")
        for p in ORDER:
            for a, b in win[p]:
                fh.write(f"{p}\t{a:.3f}\t{b:.3f}\t{b - a:.3f}\n")
    with open(out / "bench.tsv", "w") as fh:
        fh.write("instance\tphase\tn_prompt\tn_gen\treps\tavg_tok_s\tsd_tok_s\tavg_ms_a_pass\n")
        for i, p, r in bench:
            ns = r.get("samples_ns", [])
            per_pass = statistics.fmean(ns) / 1e6 / (r["n_gen"] if r["n_gen"] else 1) if ns else float("nan")
            fh.write(f"{i}\t{p}\t{r['n_prompt']}\t{r['n_gen']}\t{len(ns)}\t{r['avg_ts']:.2f}\t{r['stddev_ts']:.2f}"
                     f"\t{per_pass:.2f}\n")
    with open(out / "served.tsv", "w") as fh:
        keys = ["prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second",
                "draft_n", "draft_n_accepted", "wall_s"]
        fh.write("\t".join(keys) + "\n" + "\t".join(str(served.get(k, "")) for k in keys) + "\n")
    cols = None
    table = []
    for p in ORDER:
        sel = [r for r in rows if any(a <= r[0] <= b for a, b in win[p])]
        if not sel:
            continue
        for c in cards + ["all"]:
            part = sel if c == "all" else [r for r in sel if r[1] == c]
            if part:
                d = stats(part)
                cols = cols or list(d)
                table.append((p, c, d))
    with open(out / "clocks.tsv", "w") as fh:
        fh.write("phase\tcard\t" + "\t".join(cols) + "\n")
        for p, c, d in table:
            fh.write(f"{p}\t{c}\t" + "\t".join(f"{d[k]:.1f}" if isinstance(d[k], float) else str(d[k]) for k in cols)
                     + "\n")
    interval = [b[0] - a[0] for a, b in zip([r for r in rows if r[1] == 0], [r for r in rows if r[1] == 0][1:])]
    md = [f"Sample interval on card 0: median {1000 * q(interval, .5):.0f} ms, p90 {1000 * q(interval, .9):.0f} ms "
          f"({len(rows)} rows, cards {cards[0]}-{cards[-1]}).", ""]
    for p in ORDER:
        part = [(c, d) for pp, c, d in table if pp == p]
        if not part:
            continue
        secs = sum(b - a for a, b in win[p])
        md += [f"### {p} ({secs:.1f} s in {len(win[p])} window(s))", "",
               "| card | samples | base % | mid % | boost % | SM MHz p10 / med / p90 | mem MHz med | P0/P2/P5/P8 % "
               "| no reason % | idle % | power cap % | thermal % | power W mean / max |",
               "|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|"]
        for c, d in part:
            md.append(f"| {c} | {d['samples']} | {d['base_pct']:.0f} | {d['mid_pct']:.0f} | {d['boost_pct']:.0f} | "
                      f"{d['sm_p10']:.0f} / {d['sm_med']:.0f} / {d['sm_p90']:.0f} | {d['mem_med']:.0f} | "
                      f"{d['P0_pct']:.0f}/{d['P2_pct']:.0f}/{d['P5_pct']:.0f}/{d['P8_pct']:.0f} | "
                      f"{d['no_reason_pct']:.0f} | {d['idle_pct']:.0f} | {d['sw_power_cap_pct']:.0f} | "
                      f"{d['sw_thermal_pct'] + d['hw_thermal_pct']:.0f} | {d['power_mean_w']:.1f} / {d['power_max_w']:.1f} |")
        md.append("")
    (out / "clocks.md").write_text("\n".join(md))
    if win["served_prompt"]:
        sent, prompt_end = win["served_prompt"][0]
        decode_start, got = win["served_decode"][0]
        near = [r for r in rows if sent - 3 <= r[0] < got + 4]
        with open(out / "timeline.tsv", "w") as fh:
            fh.write("s_from_send\tphase\t" + "\t".join(f"card{c}_sm_mhz" for c in cards) + "\n")
            for s in range(-3, int(got - sent) + 4):
                a = sent + s
                phase = ("served_idle" if a < sent else "served_prompt" if a < prompt_end else
                         "served_decode" if decode_start <= a < got else "after" if a >= got else "gap")
                med = [q([r[3] for r in near if r[1] == c and a <= r[0] < a + 1], .5) for c in cards]
                fh.write(f"{s}\t{phase}\t" + "\t".join(f"{m:.0f}" for m in med) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
