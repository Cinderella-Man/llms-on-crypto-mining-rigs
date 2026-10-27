# Q74 (2026-10-23), the check of Flint's INT4 files: see ../README.md for the order of the steps. Env: QAT_DATA
# (fetched slices, default ../data), FLINT_INT4_GGUF (the makers' Q4_1 GGUF, read in place).
"""Download slices (along axis 0) of remote safetensors tensors with HTTP range requests.

usage: python -I fetch.py <base_url> <header.json> <out_dir> <tensor_name> <i0> <i1> [<tensor_name> <i0> <i1> ...]
Writes <out_dir>/<tensor_name>@<i0>-<i1>.bin (raw bytes) and appends a line to <out_dir>/slices.json.
i1 = -1 means "to the end of axis 0".
"""
import json
import math
import os
import subprocess
import sys

DT = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "U32": 4, "I8": 1}

base, hdr_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
args = sys.argv[4:]
hdr = json.load(open(hdr_path))
ds = hdr["data_start"]
os.makedirs(out_dir, exist_ok=True)
log_path = os.path.join(out_dir, "slices.json")
log = json.load(open(log_path)) if os.path.exists(log_path) else {}
total = 0
for k in range(0, len(args), 3):
    name, i0, i1 = args[k], int(args[k + 1]), int(args[k + 2])
    t = hdr["tensors"][name]
    shape = t["shape"]
    if i1 < 0:
        i1 = shape[0]
    row = math.prod(shape[1:]) * DT[t["dtype"]]
    a, b = t["data_offsets"]
    assert b - a == row * shape[0], (name, b - a, row, shape)
    start = ds + a + i0 * row
    end = ds + a + i1 * row - 1
    fn = os.path.join(out_dir, f"{name}@{i0}-{i1}.bin")
    if not (os.path.exists(fn) and os.path.getsize(fn) == end - start + 1):
        subprocess.run(["curl", "-sSL", "--fail", "-r", f"{start}-{end}", "-o", fn, base], check=True)
    assert os.path.getsize(fn) == end - start + 1, fn
    total += end - start + 1
    log[f"{name}@{i0}-{i1}"] = {"file": fn, "dtype": t["dtype"], "shape": [i1 - i0] + shape[1:], "url": base,
                                "range": [start, end]}
    print(f"{name}[{i0}:{i1}] {t['dtype']} {[i1 - i0] + shape[1:]} {(end - start + 1) / 1e6:.1f} MB")
json.dump(log, open(log_path, "w"), indent=1)
print(f"total {total / 1e6:.1f} MB")
