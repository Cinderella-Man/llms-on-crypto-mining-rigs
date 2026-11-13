#!/usr/bin/env python3
# Lists every tensor of one or more GGUF files (shards) as: name, ggml type, shape (ne0 x ne1 x ...).
# Reads only the header, with the standard library (no numpy, no gguf-py), so it runs on any CPU host.
# Usage: python3 gguf_types.py <file.gguf> [...] > types.tsv
import struct, sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
         11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
         19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
         27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 39: "MXFP4", 41: "Q1_0", 42: "Q2_0", 500: "N2P", 501: "N3P",
         502: "N2P_128", 503: "N3P_128", 504: "Q4_1_64"}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

def rd(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))[0]

def rstr(f):
    return f.read(rd(f, "<Q")).decode("utf-8", "replace")

def skip_val(f, t):
    if t in SCALAR:
        f.read(struct.calcsize(SCALAR[t]))
    elif t == 8:
        rstr(f)
    elif t == 9:
        at = rd(f, "<I"); n = rd(f, "<Q")
        for _ in range(n):
            skip_val(f, at)
    else:
        raise ValueError("unknown kv type %d" % t)

print("file\ttensor\ttype\tshape")
for path in sys.argv[1:]:
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", path
        rd(f, "<I")
        nt = rd(f, "<Q"); nkv = rd(f, "<Q")
        for _ in range(nkv):
            rstr(f); skip_val(f, rd(f, "<I"))
        for _ in range(nt):
            name = rstr(f); nd = rd(f, "<I")
            ne = [rd(f, "<Q") for _ in range(nd)]
            t = rd(f, "<I"); rd(f, "<Q")
            print("%s\t%s\t%s\t%s" % (path.rsplit("/", 1)[-1], name, TYPES.get(t, str(t)), "x".join(map(str, ne))))
