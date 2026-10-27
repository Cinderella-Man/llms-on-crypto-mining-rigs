"""Q20/D1 (2026-10-24): compare the GPU-top-k dump (gt-b) with the default path (gt-a) on Q51's transcripts; last line
ADOPT or KEEP. The rates are the dump tool's own throughput lines, read from its two logs.

usage: python3 -I gtcheck_eval.py TEST_DIR [STAIRCASE_DIR]
  TEST_DIR holds gt-a.bin, gt-b.bin (the two dumps) and gt-a.log, gt-b.log; STAIRCASE_DIR (for feats.py) defaults to
  this repository's bench/staircase_drafter.
"""
import os
import re
import sys

import numpy as np

T = sys.argv[1]
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../../bench/staircase_drafter"))
import feats  # noqa: E402


def rate(log):
    m = re.search(r"\((\d+) tokens/s", open(log).read())
    return float(m.group(1)) if m else 0.0


a, b = feats.FeatFile(f"{T}/gt-a.bin"), feats.FeatFile(f"{T}/gt-b.bin")
rb = {b[i].seq_id: b[i] for i in range(len(b))}
top1, ov, dl = [], [], []
for i in range(len(a)):
    x, y = a[i], rb.get(a[i].seq_id)
    if y is None:
        continue
    top1.append((x.top_ids[:, 0] == y.top_ids[:, 0]).mean())
    ov.append(np.mean([len(set(p) & set(q)) / len(p) for p, q in zip(x.top_ids, y.top_ids)]))
    dl.append(np.median(np.abs(x.top_lp[:, 0] - y.top_lp[:, 0])))
ra, rb_ = rate(f"{T}/gt-a.log"), rate(f"{T}/gt-b.log")
ok = len(top1) == len(a) > 30 and min(top1) >= 0.97 and min(ov) >= 0.95 and max(dl) < 0.01
print(f"{len(top1)} sequences; top-1 same min {min(top1, default=0):.3f}, top-20 overlap min {min(ov, default=0):.3f}, "
      f"median |dlogp| max {max(dl, default=9):.4f}; default {ra:.0f} tok/s, gpu top-k {rb_:.0f} tok/s")
print(("ADOPT" if ok and rb_ > 1.1 * ra else "KEEP") + f" x{rb_ / max(ra, 1):.2f} match {ok}")
