# Q74 (2026-10-23), the check of Flint's INT4 files: see ../README.md for the order of the steps. Env: QAT_DATA
# (fetched slices, default ../data), FLINT_INT4_GGUF (the makers' Q4_1 GGUF, read in place).
"""Fetch a remote safetensors header via HTTP range requests and save it as JSON.

usage: python -I st_header.py <url> <out.json>
The output JSON holds {"header_len": n, "data_start": 8+n, "tensors": {...}}.
"""
import json
import struct
import subprocess
import sys

url, out = sys.argv[1], sys.argv[2]


def rng(a, b):
    return subprocess.run(["curl", "-sSL", "-r", f"{a}-{b}", url], check=True, capture_output=True).stdout


n = struct.unpack("<Q", rng(0, 7))[0]
assert n < 50_000_000, n
h = json.loads(rng(8, 8 + n - 1))
json.dump({"header_len": n, "data_start": 8 + n, "tensors": h}, open(out, "w"), indent=0)
print(url.rsplit("/", 1)[-1], "header", n, "entries", len(h))
