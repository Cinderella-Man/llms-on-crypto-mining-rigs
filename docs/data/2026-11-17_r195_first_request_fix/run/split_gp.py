#!/usr/bin/env python3
"""split_gp.py: cut a deep-lane load's GGML_CUDA_PROFILE file (graph mode) into one file per depth, holding only that
depth's first request and its repeat, so R192's graph_passes.py, which compares the first two generation requests it
finds, runs unchanged on each depth (R195, Q1562).

  python3 -I split_gp.py GP.csv --depths 4096,6144,8192 --prefix OUT/gp-k1 [--gap-ms 3000] [--min-main 45]
      [--main-device 0]

Pieces and generation requests are found as graph_passes.py finds them: rows sorted by host_ms, a gap of more than
--gap-ms cuts a piece, a piece with at least --min-main evaluations on --main-device is a generation request. In a load
with the depths in order, the generation requests are the first request and its repeat at each depth, two by two (the
warm-up request and the primes are shorter). Writes PREFIX-d<depth>.csv (the original columns and lines, in time
order) and prints one line per piece. Standard library only.
"""
import argparse
import csv
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gp")
    ap.add_argument("--depths", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--gap-ms", type=float, default=3000.0)
    ap.add_argument("--min-main", type=int, default=45)
    ap.add_argument("--main-device", type=int, default=0)
    a = ap.parse_args()
    depths = [int(d) for d in a.depths.split(",")]
    with open(a.gp, newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        rows = []
        for r in rd:
            try:
                t = float(r["host_ms"])
                int(r["device"])
            except (KeyError, TypeError, ValueError):
                continue  # a line cut at the server's exit
            rows.append((t, r))
    rows.sort(key=lambda x: x[0])
    pieces, cur, tmax = [], [], None
    for t, r in rows:
        if cur and t > tmax + a.gap_ms:
            pieces.append(cur)
            cur, tmax = [], None
        cur.append(r)
        tmax = t if tmax is None else max(tmax, t)
    if cur:
        pieces.append(cur)
    gen = []
    for i, pc in enumerate(pieces):
        n0 = sum(1 for r in pc if int(r["device"]) == a.main_device)
        is_gen = n0 >= a.min_main
        if is_gen:
            gen.append(pc)
        print(f"piece {i}: {len(pc)} rows, {n0} on card {a.main_device}, host_ms {float(pc[0]['host_ms']):.1f} to "
              f"{float(pc[-1]['host_ms']):.1f}{', generation request ' + str(len(gen)) if is_gen else ''}")
    if len(gen) != 2 * len(depths):
        print(f"{len(gen)} generation requests, expected {2 * len(depths)}", file=sys.stderr)
        sys.exit(1)
    for k, d in enumerate(depths):
        path = f"{a.prefix}-d{d}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in gen[2 * k] + gen[2 * k + 1]:
                w.writerow(r)
        print(f"depth {d}: {path}")


if __name__ == "__main__":
    main()
