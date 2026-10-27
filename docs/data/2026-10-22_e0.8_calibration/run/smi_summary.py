#!/usr/bin/env python3
"""Per-card power, clock and power-cap summary of llama-bench runs from a 500 ms nvidia-smi trace.

usage: smi_summary.py RUN_DIR... > smi_summary.csv

Each RUN_DIR holds smi.csv (nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,
temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active --format=csv -lms 500) and one llama-bench
JSONL per GPU set (gpus_0.jsonl, gpus_0,1.jsonl, ...), as scripts/bench.sh writes them. For every test in the JSONL
(pp512 / tg128, flash attention off / on) the window runs from its test_time over the warm-up (a whole prompt for pp,
one token for tg) and the repetitions, plus 0.5 s, and ends early where the next test starts. Rows: one per run, GPU
set, test and card in the set, over the samples of that card in the window. The nvidia-smi timestamps are local time
and test_time is UTC: the offset is the whole hour (-3..+3) that puts the most utilisation inside the windows.
"""
import csv
import datetime as dt
import glob
import json
import os
import sys

CAP, HW_SLOW, SW_THERM, HW_THERM, HW_BRAKE = 0x4, 0x8, 0x20, 0x40, 0x80


def num(s):
    return float(s.strip().split()[0])


def load_smi(path):
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if len(x) < 10:
                continue
            try:
                t = dt.datetime.strptime(x[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                rows.append(dict(t=t, gpu=int(x[1]), pstate=x[2].strip(), sm=num(x[3]), mem=num(x[4]), w=num(x[5]),
                                 temp=num(x[6]), util=num(x[7]), reasons=int(x[9].strip(), 16)))
            except ValueError:
                continue
    return rows


def tests(run_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        # gpus_0,1.jsonl (scripts/bench.sh today) or gpu0,1.jsonl (its 2026-10-22 version)
        gpus = [int(g) for g in os.path.basename(p)[:-6].lstrip("gpus_").split(",")]
        for line in open(p):
            if not line.strip():
                continue
            j = json.loads(line)
            start = dt.datetime.strptime(j["test_time"], "%Y-%m-%dT%H:%M:%SZ")
            reps = len(j["samples_ns"])
            # warm-up: a whole prompt for pp tests, one token for tg tests
            dur = ((reps + 1) * j["avg_ns"] if j["n_prompt"] else reps * j["avg_ns"]) / 1e9 + 0.5
            name = f"pp{j['n_prompt']}" if j["n_prompt"] else f"tg{j['n_gen']}"
            out.append(dict(file=os.path.basename(p), gpus=gpus, test=name, fa=j["flash_attn"], start=start,
                            end=start + dt.timedelta(seconds=dur), tok_s=j["avg_ts"], sd=j["stddev_ts"]))
    out.sort(key=lambda t: t["start"])
    for a, b in zip(out, out[1:]):
        a["end"] = min(a["end"], b["start"])
    return out


def window(smi, t, off, gpu):
    lo, hi = t["start"] + off, t["end"] + off
    return [s for s in smi if s["gpu"] == gpu and lo <= s["t"] < hi]


def main():
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["run", "gpu_set", "test", "flash_attn", "tok_s", "tok_s_sd", "card", "samples", "util_mean_pct",
                "power_mean_w", "power_max_w", "sm_clock_mean_mhz", "sm_clock_min_mhz", "sm_clock_max_mhz",
                "mem_clock_mhz", "pstates", "power_cap_share_pct", "hw_slowdown_share_pct", "thermal_share_pct",
                "temp_max_c"])
    for run_dir in sys.argv[1:]:
        smi = load_smi(os.path.join(run_dir, "smi.csv"))
        ts = tests(run_dir)
        best = None
        for h in range(-3, 4):
            off = dt.timedelta(hours=h)
            u = [s["util"] for t in ts for g in t["gpus"] for s in window(smi, t, off, g)]
            score = sum(u) / len(u) if u else -1
            if best is None or score > best[0]:
                best = (score, off)
        off = best[1]
        print(f"{run_dir}: local time = UTC {off}", file=sys.stderr)
        for t in ts:
            for g in t["gpus"]:
                s = window(smi, t, off, g)
                if not s:
                    continue
                n = len(s)
                share = lambda mask: round(100 * sum(1 for x in s if x["reasons"] & mask) / n, 1)
                mems = sorted(set(int(x["mem"]) for x in s))
                w.writerow([os.path.basename(run_dir.rstrip("/")), ",".join(map(str, t["gpus"])), t["test"], t["fa"],
                            round(t["tok_s"], 2), round(t["sd"], 2), g, n,
                            round(sum(x["util"] for x in s) / n, 1), round(sum(x["w"] for x in s) / n, 1),
                            round(max(x["w"] for x in s), 1), round(sum(x["sm"] for x in s) / n),
                            int(min(x["sm"] for x in s)), int(max(x["sm"] for x in s)),
                            "/".join(map(str, mems)), "/".join(sorted(set(x["pstate"] for x in s))),
                            share(CAP), share(HW_SLOW | HW_BRAKE), share(SW_THERM | HW_THERM),
                            int(max(x["temp"] for x in s))])


if __name__ == "__main__":
    main()
