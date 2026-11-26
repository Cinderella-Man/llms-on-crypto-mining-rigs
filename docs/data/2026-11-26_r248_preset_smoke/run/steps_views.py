#!/usr/bin/env python3
"""R248 (Q1938): ms a decode step by context, cut where a four-token verify changes kernel at _MIN_RATIO=0.45 and 0.5.

    python3 -I steps_views.py ARM=SERVER_LOG=TURNS_CSV ...

C1175's run/steps_cells.py (docs/data/2026-11-26_c1175_nsm10_rig2/) with its buckets cut by R245's view rule
(docs/data/2026-11-26_r245_flash_ratio2/README.md, "The switches"): a four-token verify goes to the per-token sparse
kernel when its cache view, which grows in steps of 256 cells, holds at least _MIN_RATIO x 2,051 x 4 cells. At 0.45
that is from a view of 3,840, so from 3,585 cells of context; at 0.5 from a view of 4,352, so from 4,097 cells. From
3,585 to 4,096 cells only 0.45 runs the sparse kernel; under 3,585 both run the dense kernels, from 4,097 both the
sparse one. A request is placed by its middle context, so one that crosses a cut counts in one bucket. Run from the
repository root. Standard library only.
"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("flash_steps", "docs/data/2026-11-18_r203_ornith_offsets/run/flash_steps.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
fs.BUCKETS = [(0, 3585), (3585, 4097), (4097, 6153), (6153, 8204), (8204, 16000), (16000, 10 ** 9)]
fs.main(sys.argv[1:])
