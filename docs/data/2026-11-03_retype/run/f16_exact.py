#!/usr/bin/env python3
# f16_exact.py: how many weights of the BF16 tensors that match a regex F16 holds exactly. A BF16 value has 8
# significant bits, so F16 holds it exactly when its exponent lies in F16's normal range (2^-14 up to below 2^16), or
# lower down to 2^-17, where F16's subnormals (steps of 2^-24) still hold all 8 bits; zeros are exact, smaller values
# round, and from 2^16 up they overflow. Counts per tensor name, layers summed. Standard library and
# scripts/gguf_reshard.py.
#   SCRIPTS=<repo>/scripts python3 -I f16_exact.py '<regex>' SHARD... > f16_exact.tsv
import collections, math, os, re, struct, sys

sys.path.insert(0, os.environ["SCRIPTS"])
import gguf_reshard as g  # noqa: E402

BF16 = 30
pat = re.compile(sys.argv[1])
rows = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # weights, zero, exact, rounded, overflow, largest |bits|
for p in sys.argv[2:]:
    h = g.Header(p)
    with open(p, "rb") as f:
        for name, dims, typ, off in h.tensors:
            if typ != BF16 or not pat.search(name):
                continue
            n = math.prod(dims)
            r = rows[re.sub(r"^blk\.\d+\.", "blk.*.", name)]
            r[0] += n
            for v, c in collections.Counter(memoryview(os.pread(f.fileno(), 2 * n, h.data0 + off)).cast("H")).items():
                e = (v >> 7) & 0xFF
                k = 1 if v & 0x7FFF == 0 else 4 if e >= 143 else 2 if e >= 110 else 3
                r[k] += c
                r[5] = max(r[5], v & 0x7FFF)
print("\t".join(["tensor", "weights", "zero", "exact", "rounded", "overflow", "largest_abs"]))
for name, r in sorted(rows.items()):
    largest = struct.unpack("<f", struct.pack("<I", r[5] << 16))[0]
    print("\t".join([name, *map(str, r[:5]), f"{largest:.6g}"]))
