# Q74 (2026-10-23), the check of Flint's INT4 files: see ../README.md for the order of the steps. Env: QAT_DATA
# (fetched slices, default ../data), FLINT_INT4_GGUF (the makers' Q4_1 GGUF, read in place).
"""Parse a GGUF header (pure Python) and dump metadata + tensor table to JSON.

usage: python -I gguf_index.py <file.gguf> <out.json>
"""
import json
import struct
import sys

path, out = sys.argv[1], sys.argv[2]
f = open(path, "rb")


def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def st():
    return f.read(rd("Q")).decode("utf-8", "replace")


SZ = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def val(t):
    if t == 8:
        return st()
    if t == 9:
        et, n = rd("I"), rd("Q")
        if et in (8, 9):
            v = [val(et) for _ in range(n)]
        else:
            v = [rd(SZ[et]) for _ in range(n)]
        return v if n <= 64 else f"<array type {et} len {n}>"
    return rd(SZ[t])


assert f.read(4) == b"GGUF"
version = rd("I")
ntens = rd("Q")
nkv = rd("Q")
meta = {}
for _ in range(nkv):
    k = st()
    meta[k] = val(rd("I"))
tensors = []
for _ in range(ntens):
    name = st()
    nd = rd("I")
    ne = [rd("Q") for _ in range(nd)]
    typ = rd("I")
    off = rd("Q")
    tensors.append({"name": name, "ne": ne, "type": typ, "offset": off})
align = meta.get("general.alignment", 32)
pos = f.tell()
data_start = (pos + align - 1) // align * align
json.dump({"version": version, "meta": meta, "data_start": data_start, "tensors": tensors}, open(out, "w"), indent=1)
print("tensors", ntens, "kv", nkv, "data_start", data_start)
