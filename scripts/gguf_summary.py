#!/usr/bin/env python3
# gguf_summary.py FILE.gguf: a GGUF's metadata (arrays shown by length and their first items) and its tensor bytes per
# role and type, per block, the non-block tensors and the last block's tensors. Standard library only (no numpy, no
# gguf-py), so it runs on any host Python 3. Used for the Ornith-1.5-122B files (docs/data/2026-10-27_ornith/).
import collections, re, struct, sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

f = open(sys.argv[1], "rb")
def rd(fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def rstr():
    return f.read(rd("<Q")).decode("utf-8", "replace")
def rval(t):
    if t in SCALAR:
        return rd(SCALAR[t])
    if t == 8:
        return rstr()
    if t == 9:
        et, n = rd("<I"), rd("<Q")
        items = [rval(et) for _ in range(n)]
        return ("array", et, n, items[:8] if et != 8 else items[:3])
    raise ValueError(f"type {t}")

assert f.read(4) == b"GGUF"
ver, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
print(f"version {ver}, {nt} tensors, {nkv} keys")
align = 32
for _ in range(nkv):
    k, t = rstr(), rd("<I")
    v = rval(t)
    if k == "general.alignment":
        align = v
    if isinstance(v, tuple):
        print(f"  {k} = array[{v[2]}] {v[3] if v[2] <= 64 else '…'}")
    else:
        print(f"  {k} = {v}")
infos = []
for _ in range(nt):
    name = rstr()
    nd = rd("<I")
    dims = [rd("<Q") for _ in range(nd)]
    infos.append([name, dims, rd("<I"), rd("<Q")])
data0 = (f.tell() + align - 1) // align * align
f.seek(0, 2)
end = f.seek(0, 1) - data0
infos.sort(key=lambda x: x[3])
for i, x in enumerate(infos):
    x.append((infos[i + 1][3] if i + 1 < len(infos) else end) - x[3])

def role(name):
    m = re.match(r"blk\.(\d+)\.(.*)", name)
    return re.sub(r"\.weight$|\.bias$", "", m.group(2)) if m else name
by = collections.defaultdict(lambda: [0, 0])
blocks = collections.defaultdict(int)
for name, dims, t, off, size in infos:
    r = (role(name), TYPES.get(t, t))
    by[r][0] += 1
    by[r][1] += size
    m = re.match(r"blk\.(\d+)\.", name)
    if m:
        blocks[int(m.group(1))] += size
print(f"\ntensor data {end / 1e9:.2f} GB; per role and type (count, GB):")
for (r, t), (n, s) in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print(f"  {r:32s} {t:8s} {n:4d} {s / 1e9:8.3f}")
print("\nper block (GB):", " ".join(f"{b}:{s / 1e9:.2f}" for b, s in sorted(blocks.items())))
odd = [x for x in infos if not re.match(r"blk\.\d+\.", x[0])]
print("non-block tensors:", ", ".join(f"{x[0]} {TYPES.get(x[2])} {x[1]}" for x in odd))
last = max(blocks) if blocks else -1
print(f"last block {last}:", ", ".join(f"{x[0]} {TYPES.get(x[2])} {x[1]}" for x in infos if x[0].startswith(f"blk.{last}.")))
