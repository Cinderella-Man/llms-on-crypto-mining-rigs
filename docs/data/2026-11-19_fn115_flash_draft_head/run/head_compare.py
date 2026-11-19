#!/usr/bin/env python3
"""head_compare.py A.gguf B.gguf: two GGUF files' headers side by side, standard library only (no numpy, no gguf-py).

Reads only the header (metadata and tensor infos, never tensor data) and prints Markdown:
- the files: version, tensor count, key count, header bytes, the sum of the tensors' bytes;
- every metadata key in either file: its value in A and in B (an array by its element type, its length and the md5 of
  its items, so two vocabularies are equal exactly when their md5s are), and whether they are equal;
- every tensor in either file: type and shape in A and in B, its bytes, and whether they are equal;
- the bytes per tensor type in each file.
FN115 (Q1545) used it on Flash-Next's MTP draft heads: the preset's head against a published distilled head.
    python3 -I head_compare.py preset.gguf other.gguf > head_compare.md
"""
import hashlib
import struct
import sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
         43: "Q4_1_64", 500: "N2P", 501: "N3P"}
# (block size in elements, bytes per block) of the types a head can hold; others print their bytes as "?"
BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 20: (32, 18), 23: (256, 136),
         24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8)}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
VTYPE = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32", 6: "f32", 7: "bool", 8: "str", 9: "arr",
         10: "u64", 11: "i64", 12: "f64"}


def read_header(path):
    f = open(path, "rb")

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
            h = hashlib.md5()
            first = []
            small = [] if et != 8 and n <= 4096 else None  # a short numeric array is kept whole, to list its differences
            for i in range(n):
                v = rval(et)
                h.update(repr(v).encode())
                if i < 3:
                    first.append(v)
                if small is not None:
                    small.append(v)
            return ("array", VTYPE.get(et, et), n, h.hexdigest(), first, small)
        raise ValueError(f"metadata type {t}")

    if f.read(4) != b"GGUF":
        sys.exit(f"{path}: not a GGUF file")
    ver, nt, nkv = rd("<I"), rd("<Q"), rd("<Q")
    kv = {}
    for _ in range(nkv):
        k, t = rstr(), rd("<I")
        kv[k] = rval(t)
    tensors = {}
    for _ in range(nt):
        name = rstr()
        nd = rd("<I")
        shape = [rd("<Q") for _ in range(nd)]
        t = rd("<I")
        rd("<Q")  # offset
        n = 1
        for d in shape:
            n *= d
        b = BLOCK.get(t)
        nbytes = n // b[0] * b[1] if b else None
        tensors[name] = (TYPES.get(t, str(t)), shape, nbytes)
    header = f.tell()
    f.close()
    return {"version": ver, "nt": nt, "nkv": nkv, "kv": kv, "tensors": tensors, "header": header}


def show(v):
    if isinstance(v, tuple):
        _, et, n, md5, first, _ = v
        return f"{et}[{n}] md5 {md5[:12]}… first {first!r}"[:160]
    if isinstance(v, float):
        return f"{v:.9g}"
    return repr(v)[:160]


def cell(s):
    return str(s).replace("|", "\\|")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pa, pb = sys.argv[1], sys.argv[2]
    A, B = read_header(pa), read_header(pb)
    print(f"# Headers: A `{pa.rsplit('/', 1)[-1]}`, B `{pb.rsplit('/', 1)[-1]}`\n")
    print("| | A | B |\n|---|---|---|")
    for k, lab in (("version", "GGUF version"), ("nt", "tensors"), ("nkv", "metadata keys"), ("header", "header bytes")):
        print(f"| {lab} | {A[k]:,} | {B[k]:,} |")
    sa = sum(x[2] or 0 for x in A["tensors"].values())
    sb = sum(x[2] or 0 for x in B["tensors"].values())
    print(f"| tensor bytes | {sa:,} | {sb:,} |\n")

    keys = list(dict.fromkeys(list(A["kv"]) + list(B["kv"])))
    same_k = [k for k in keys if k in A["kv"] and k in B["kv"] and A["kv"][k] == B["kv"][k]]
    print(f"## Metadata: {len(same_k)} of {len(keys)} keys equal\n")
    print("| key | A | B | equal |\n|---|---|---|---|")
    for k in keys:
        va, vb = A["kv"].get(k, "(absent)"), B["kv"].get(k, "(absent)")
        eq = "yes" if k in same_k else "**no**"
        print(f"| `{k}` | {cell(show(va))} | {cell(show(vb))} | {eq} |")
    for k in keys:  # numeric arrays of one length that differ: each index that differs
        va, vb = A["kv"].get(k), B["kv"].get(k)
        if isinstance(va, tuple) and isinstance(vb, tuple) and va != vb and va[5] is not None and vb[5] is not None \
                and len(va[5]) == len(vb[5]):
            diff = [f"{i}: {x} against {y}" for i, (x, y) in enumerate(zip(va[5], vb[5])) if x != y]
            print(f"\n`{k}` differs at {len(diff)} of {len(va[5])} indices (A against B): {'; '.join(diff)}")
            print(f"A: {va[5]}\nB: {vb[5]}")

    names = list(dict.fromkeys(list(A["tensors"]) + list(B["tensors"])))
    same_t = [n for n in names if A["tensors"].get(n) == B["tensors"].get(n)]
    print(f"\n## Tensors: {len(same_t)} of {len(names)} with the same type and shape in both\n")
    print("| tensor | A type | A shape | A bytes | B type | B shape | B bytes | equal |\n|---|---|---|---:|---|---|---:|---|")
    for n in names:
        ta, tb = A["tensors"].get(n), B["tensors"].get(n)
        ca = (ta[0], "×".join(map(str, ta[1])), f"{ta[2]:,}" if ta[2] is not None else "?") if ta else ("(absent)", "", "")
        cb = (tb[0], "×".join(map(str, tb[1])), f"{tb[2]:,}" if tb[2] is not None else "?") if tb else ("(absent)", "", "")
        eq = "yes" if n in same_t else "**no**"
        print(f"| `{n}` | {ca[0]} | {ca[1]} | {ca[2]} | {cb[0]} | {cb[1]} | {cb[2]} | {eq} |")

    print("\n## Bytes per type\n")
    print("| type | A tensors | A bytes | B tensors | B bytes |\n|---|---:|---:|---:|---:|")
    tys = sorted({x[0] for x in A["tensors"].values()} | {x[0] for x in B["tensors"].values()})
    for ty in tys:
        xa = [x for x in A["tensors"].values() if x[0] == ty]
        xb = [x for x in B["tensors"].values() if x[0] == ty]
        print(f"| {ty} | {len(xa)} | {sum(x[2] or 0 for x in xa):,} | {len(xb)} | {sum(x[2] or 0 for x in xb):,} |")


main()
