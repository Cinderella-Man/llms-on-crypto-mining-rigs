#!/usr/bin/env python3
"""waits.py: the long host waits inside steps, and the host's memory per request (S37, Q961).

  python3 -I waits.py RES OUT --loads warm,d1off,d1cache,d1carry,d1on,d1off-2 [--requests long,greedy,smoke]
      [--over-ms 50] --waits-tsv waits.tsv --mem-tsv host_mem.tsv

RES/<load>/steps.tsv is step_split.py's --tsv; OUT is step_loads.sh's output folder (status.txt with the epoch time
around every request, <load>/ram_watchdog-<load>.tsv with one row a second: epoch, MemAvailable, CmaFree, usable,
swap used, swap-out in the last 10 s (all MiB), memory pressure some and full avg10 in percent).
waits.tsv, per load and request: the generation steps, those with a turn over --over-ms (the host's time after a
draft pass's GPU work, see step_split.py), which pass's turn it was, the sum of those turns and the first and last
such step. host_mem.tsv, per load and request: MemAvailable at the request's start and its minimum, swap used at its
start and end, the largest pressure readings.
"""
import argparse
import csv
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res")
    ap.add_argument("out")
    ap.add_argument("--loads", required=True)
    ap.add_argument("--requests", default="long,greedy,smoke")
    ap.add_argument("--over-ms", type=float, default=50.0)
    ap.add_argument("--waits-tsv", required=True)
    ap.add_argument("--mem-tsv", required=True)
    a = ap.parse_args()
    loads, reqs = a.loads.split(","), a.requests.split(",")

    with open(a.waits_tsv, "w") as f:
        f.write("load\trequest\tgeneration_steps\tsteps_with_a_long_turn\tafter_pass0\tafter_pass1\tafter_pass2\t"
                "after_the_last_pass\tsum_long_turns_ms\tmedian_long_turn_ms\tfirst_step\tlast_step\n")
        for ld in loads:
            p = os.path.join(a.res, ld, "steps.tsv")
            if not os.path.exists(p):
                continue
            with open(p) as g:
                rows = list(csv.DictReader(g, delimiter="\t"))
            for r in reqs:
                st = [x for x in rows if x["request"] == r]
                by, turns, idx = [0, 0, 0, 0], [], []
                for x in st:
                    n, hit = int(x["n_dp"]), False
                    for k in range(n):
                        t = float(x[f"turn{k}_ms"])
                        if t > a.over_ms:
                            by[3 if k == n - 1 else min(k, 2)] += 1
                            turns.append(t)
                            hit = True
                    if hit:
                        idx.append(int(x["i"]))
                med = sorted(turns)[len(turns) // 2] if turns else 0.0
                f.write(f"{ld}\t{r}\t{len(st)}\t{len(idx)}\t{by[0]}\t{by[1]}\t{by[2]}\t{by[3]}\t{sum(turns):.1f}\t{med:.1f}\t"
                        f"{idx[0] if idx else ''}\t{idx[-1] if idx else ''}\n")

    win = {}
    pat = re.compile(r"^\S+ (\S+) (\S+): request (start|exit \d+ end) ([0-9.]+)")
    with open(os.path.join(a.out, "status.txt")) as f:
        for line in f:
            m = pat.match(line)
            if m:
                win.setdefault((m.group(1), m.group(2)), {})["start" if m.group(3) == "start" else "end"] = float(m.group(4))
    with open(a.mem_tsv, "w") as f:
        f.write("load\trequest\tseconds\tmem_available_start_mib\tmem_available_min_mib\tswap_used_start_mib\t"
                "swap_used_end_mib\tswapped_out_mib\tpressure_some_max_pct\tpressure_full_max_pct\n")
        for ld in loads:
            p = os.path.join(a.out, ld, f"ram_watchdog-{ld}.tsv")
            if not os.path.exists(p):
                continue
            samples = []
            with open(p) as g:
                for line in g:
                    if line.startswith("#"):
                        continue
                    c = line.split("\t")
                    samples.append((int(c[0]), int(c[1]), int(c[4]), float(c[6]), float(c[7])))
            for r in reqs:
                w = win.get((ld, r))
                if not w or "start" not in w or "end" not in w:
                    continue
                xs = [s for s in samples if w["start"] - 1 <= s[0] <= w["end"] + 1]
                if not xs:
                    continue
                f.write(f"{ld}\t{r}\t{w['end'] - w['start']:.1f}\t{xs[0][1]}\t{min(s[1] for s in xs)}\t{xs[0][2]}\t{xs[-1][2]}\t"
                        f"{xs[-1][2] - xs[0][2]}\t{max(s[3] for s in xs):.2f}\t{max(s[4] for s in xs):.2f}\n")


if __name__ == "__main__":
    main()
