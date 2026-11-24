#!/usr/bin/env python3
"""C877 (Q1583): runs C950's session (docs/data/2026-11-21_c950_flint_slot_plan_gpu/run/session.py) unchanged and
writes, for each of its /completion requests, the steady clock's reading in ms when the request was sent and when its
reply had arrived, to <out>/marks.tsv (round, request index, start, end). The steady clock is CLOCK_MONOTONIC, the
clock whose reading GGML_CUDA_PROFILE_HOST logs as its file's 0 ms, so every line of a profile can be given to the
request it belongs to. The marks only place the lines; no time is reported from them.

Usage: python3 -I session_marks.py <C950's session.py> <server URL> <output folder> <round tag>
"""
import io
import os
import runpy
import sys
import time
import urllib.request

SESSION, URL, OUT, ROUND = sys.argv[1:5]
_urlopen = urllib.request.urlopen
_n = [0]


def _ms():
    return time.clock_gettime(time.CLOCK_MONOTONIC) * 1000.0


def urlopen(req, *args, **kwargs):
    url = req.full_url if hasattr(req, "full_url") else str(req)
    if not url.endswith("/completion"):
        return _urlopen(req, *args, **kwargs)
    t0 = _ms()
    try:
        with _urlopen(req, *args, **kwargs) as r:
            body = r.read()
    finally:
        t1 = _ms()
        _n[0] += 1
        with open(os.path.join(OUT, "marks.tsv"), "a") as f:
            f.write(f"{ROUND}\t{_n[0]}\t{t0:.3f}\t{t1:.3f}\n")
    return io.BytesIO(body)


os.makedirs(OUT, exist_ok=True)
urllib.request.urlopen = urlopen
sys.argv = [SESSION, URL, OUT, ROUND]
runpy.run_path(SESSION, run_name="__main__")
