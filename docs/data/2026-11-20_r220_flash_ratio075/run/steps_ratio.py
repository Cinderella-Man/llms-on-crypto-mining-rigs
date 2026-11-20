#!/usr/bin/env python3
"""R220 (Q1635): ms a decode step by context, in the buckets where the sparse attention ratio acts.

    python3 -I steps_ratio.py ARM=SERVER_LOG=TURNS_CSV ...

R203's flash_steps.py (docs/data/2026-11-18_r203_ornith_offsets/run/flash_steps.py: each request in the bucket of its
middle context, decode steps from the draft's accepted tokens and mean length) with its buckets cut where a four-token
verify pass changes kernel: from 6,153 cells at ratio 0.75 and from 8,204 at 1.0 (ratio x 2,051 kept cells x 4 query
tokens, R69). A request is placed by its middle context, so one that crosses a cut counts in one bucket. Run from the
repository root. Standard library only.
"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("flash_steps", "docs/data/2026-11-18_r203_ornith_offsets/run/flash_steps.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
fs.BUCKETS = [(0, 6153), (6153, 8204), (8204, 16000), (16000, 10 ** 9)]
fs.main(sys.argv[1:])
