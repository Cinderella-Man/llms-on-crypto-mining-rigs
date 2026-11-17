#!/usr/bin/env python3
"""split_trace.py: one profiler file of a deep-lane load cut into one file per generation request, streaming, so that
the readers (verify_parts.py, step_split.py) hold one request in memory at a time (R121, Q1393).

  python3 -I split_trace.py PROFILE.csv --out DIR [--names d0p1,d0p2,d0s,d4096p1,...] [--gap-ms 3000]
      [--min-main 45] [--main-device 0]

PROFILE.csv is a GGML_CUDA_PROFILE file (group mode: device, eval, host_ms, ...) or a GGML_CUDA_PROFILE_HOST file
(device, eval, host_enter_ms, ...); the time column is host_ms or host_enter_ms. The deep lane pauses before every
request (PAUSE), so a gap of more than --gap-ms after the latest time seen cuts the file into pieces. A piece with at
least --min-main evaluations on --main-device (the main model's passes: the MTP draft runs on the last card only) is a
generation request (256 tokens took 59 steps or more in R121); the server's load-time evaluation, the client's
32-token warm-up request (35 passes or fewer) and the primes (20 or fewer) have fewer. The generation pieces are written in order as DIR/<name>.csv, named by --names
(default: the deep lane's order for depths 0, 4,096 and 8,192). Prints one line per piece: its rows, its main passes,
its time span and the file it went to. Exits 1 when the generation pieces are not as many as the names.
Standard library only.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", default="d0p1,d0p2,d0s,d4096p1,d4096p2,d4096s,d8192p1,d8192p2,d8192s")
    ap.add_argument("--gap-ms", type=float, default=3000.0)
    ap.add_argument("--min-main", type=int, default=45)
    ap.add_argument("--main-device", type=int, default=0)
    a = ap.parse_args()
    names = a.names.split(",")
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, ".piece.tmp")
    pieces = []  # (rows, main, t0, t1)

    def close(f, rows, main, t0, t1):
        f.close()
        k = len(pieces)
        if main >= a.min_main:
            g = sum(1 for p in pieces if p[4])
            name = names[g] if g < len(names) else f"extra{g}"
            os.replace(tmp, os.path.join(a.out, name + ".csv"))
        else:
            name = ""
            os.remove(tmp)
        pieces.append((rows, main, t0, t1, name))
        print(f"piece {k}: {rows} rows, {main} main passes, {t0:.1f} to {t1:.1f} ms -> {name or '(not a generation request)'}")

    with open(a.profile) as src:
        header = src.readline()
        cols = header.strip().split(",")
        tcol = cols.index("host_ms") if "host_ms" in cols else cols.index("host_enter_ms")
        dcol = cols.index("device")
        f, rows, main, t0, tmax, last_eval = None, 0, 0, None, None, None
        for line in src:
            parts = line.split(",", tcol + 2)
            try:
                t = float(parts[tcol])
                d = int(parts[dcol])
                ev = parts[cols.index("eval")] if "eval" in cols else None
            except (ValueError, IndexError):
                continue  # a line cut at the server's exit
            if f is not None and t > tmax + a.gap_ms:
                close(f, rows, main, t0, tmax)
                f = None
            if f is None:
                f = open(tmp, "w")
                f.write(header)
                rows, main, t0, tmax, last_eval = 0, 0, t, t, None
            f.write(line)
            rows += 1
            tmax = max(tmax, t)
            if d == a.main_device and ev != last_eval:  # a group-mode evaluation spans several rows
                main += 1
                last_eval = ev
        if f is not None:
            close(f, rows, main, t0, tmax)
    got = sum(1 for p in pieces if p[4])
    print(f"{got} generation requests of {len(names)} names")
    sys.exit(0 if got == len(names) else 1)


if __name__ == "__main__":
    main()
