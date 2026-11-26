#!/usr/bin/env python3
"""C1175 (Q1908): ms a decode step by context, in the buckets where the sparse attention switches act.

    python3 -I steps_cells.py ARM=SERVER_LOG=TURNS_CSV ...

R203's flash_steps.py (docs/data/2026-11-18_r203_ornith_offsets/run/flash_steps.py: each request in the bucket of its
middle context, decode steps from the draft's accepted tokens and mean length) with its buckets cut where a four-token
verify pass changes kernel: from 4,102 cells at _MIN_RATIO=0.5 and from 6,153 at 0.75 (ratio x 2,051 kept cells x 4
query tokens), as R220's run/steps_ratio.py does for 0.75 and 1.0. Under 4,102 cells every arm runs the dense kernels;
from 4,102 to 6,153 only _MIN_RATIO=0.5 runs the per-token sparse kernel; from 6,153 every arm runs it, the preset in 3
splits, _NSM=10 in 5. A request is placed by its middle context, so one that crosses a cut counts in one bucket. Run
from the repository root. Standard library only.
"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("flash_steps", "docs/data/2026-11-18_r203_ornith_offsets/run/flash_steps.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
fs.BUCKETS = [(0, 4102), (4102, 6153), (6153, 8204), (8204, 16000), (16000, 10 ** 9)]
fs.main(sys.argv[1:])
