#!/usr/bin/env python3
"""slot_gap.py LOG...: per llama-server log, the time from choosing the slot to processing each request.

Text reuse (LLAMA_SERVER_TEXT_REUSE=1) compares the new prompt's text with the cache when the server chooses the slot,
before the request is processed, so its own cost sits in this gap and in no prompt time. For each log, pairs every
`slot       select:` line with the next `processing task` line and prints the number of requests, the median, the
largest and the sum of those gaps in ms. The logs' time stamps are minutes, seconds, milliseconds and microseconds
since the server started (`MM.SS.mmm.uuu`). Standard library only:
  python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/slot_gap.py <TABLES>/rig/server-f5c1.log ...
"""
import re
import statistics
import sys

STAMP = re.compile(r"(\d+)\.(\d+)\.(\d+)\.(\d+)")


def stamp_s(line):
    m = STAMP.match(line)
    return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 1e3 + int(m[4]) / 1e6 if m else None


print("log\trequests\tmedian_ms\tmax_ms\tsum_ms")
for path in sys.argv[1:]:
    gaps, chosen = [], None
    for line in open(path, errors="replace"):
        if "slot       select:" in line:
            chosen = stamp_s(line)
        elif "processing task" in line and chosen is not None:
            t = stamp_s(line)
            if t is not None:
                gaps.append((t - chosen) * 1e3)
            chosen = None
    name = path.rsplit("/", 1)[-1]
    if gaps:
        print(f"{name}\t{len(gaps)}\t{statistics.median(gaps):.2f}\t{max(gaps):.2f}\t{sum(gaps):.1f}")
    else:
        print(f"{name}\t0\t\t\t")
