#!/usr/bin/env python3
# census.py: the unquantized tensors (F32, F16, BF16) of a split GGUF, one TSV line each: shard, tensor, type, dims
# (ggml order: row width first), bytes, absolute file offset of the data. Standard library only (scripts/gguf_reshard.py
# parses the header).
#   python3 -I census.py SHARD... > census.tsv
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../scripts"))
from gguf_reshard import Header  # noqa: E402

TYPES = {0: ("F32", 4), 1: ("F16", 2), 30: ("BF16", 2)}

print("shard\ttensor\ttype\tdims\tbytes\toffset")
for path in sys.argv[1:]:
    h = Header(path)
    for name, dims, typ, off in h.tensors:
        if typ not in TYPES:
            continue
        tname, size = TYPES[typ]
        n = 1
        for d in dims:
            n *= d
        print(f"{os.path.basename(path)}\t{name}\t{tname}\t{'x'.join(map(str, dims))}\t{n * size}\t{h.data0 + off}")
