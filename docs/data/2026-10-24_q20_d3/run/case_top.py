"""Top logits of each step's expected output in a check_export.py case.bin (to tell a near-tie from a real mismatch when
llama-pipe-drafter-check reports a different top id). usage: python case_top.py case.bin [k]"""
import struct
import sys

import numpy as np

path, k = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
with open(path, "rb") as f:
    magic, n_table, d, n_prompt, n_steps, nv1, n_tap, fuse = struct.unpack("<8i", f.read(32))
    f.seek(4*(n_table*d + n_prompt + 2*n_prompt*d), 1)
    for s in range(n_steps):
        n_commit, n_window = struct.unpack("<2i", f.read(8))
        f.seek((n_commit + n_window)*(8 + 4*n_tap*d), 1)
        lg = np.frombuffer(f.read(4*nv1), np.float32)
        top = np.argsort(-lg)[:k]
        print(f"step {s}: " + ", ".join(f"{i} {lg[i]:.5f}" for i in top) + f"  (gap {lg[top[0]] - lg[top[1]]:.5f})")
