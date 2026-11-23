#!/usr/bin/env python3
"""C954 (Q1687 point 8): a process's resident memory from /proc/<pid>/status every 5 ms until the process ends, for
seeing a short-lived host buffer while a model loads (an output head's copy through host memory lives for a fraction
of a second). RssAnon is the process's own memory (malloc and friends); RssFile counts the model file's mapped pages,
which a load with mmap maps and unmaps by the gigabyte, so a head-sized host buffer shows in RssAnon only.

Usage: python3 -I rss_sampler.py <pid> <out.tsv>
Writes a header and a line whenever RssAnon moved 1 MiB or more since the last line, and at least once a second:
t_ms (epoch milliseconds), rss_anon_kib, rss_file_kib, vm_rss_kib, vm_hwm_kib. The last line, starting with '#',
holds the largest RssAnon seen and when, and the number of samples read.
"""
import sys
import time

PID, OUT = sys.argv[1], sys.argv[2]
KEYS = ("RssAnon", "RssFile", "VmRSS", "VmHWM")


def read():
    with open(f"/proc/{PID}/status") as f:
        text = f.read()
    d = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in KEYS:
            d[k] = int(v.split()[0])
    return d


n = 0
peak, peak_t = -1, 0
with open(OUT, "w") as f:
    f.write("t_ms\trss_anon_kib\trss_file_kib\tvm_rss_kib\tvm_hwm_kib\n")
    last_anon, last_t = -(1 << 30), 0
    while True:
        try:
            d = read()
        except (OSError, ValueError, IndexError):
            break
        now = int(time.time() * 1000)
        n += 1
        a = d.get("RssAnon", 0)
        if a > peak:
            peak, peak_t = a, now
        if abs(a - last_anon) >= 1024 or now - last_t >= 1000:
            f.write(f"{now}\t{a}\t{d.get('RssFile', 0)}\t{d.get('VmRSS', 0)}\t{d.get('VmHWM', 0)}\n")
            f.flush()
            last_anon, last_t = a, now
        time.sleep(0.005)
    f.write(f"# peak_rss_anon_kib {peak} at {peak_t}; samples {n}\n")
