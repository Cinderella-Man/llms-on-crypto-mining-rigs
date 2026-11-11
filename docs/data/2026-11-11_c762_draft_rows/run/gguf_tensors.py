#!/usr/bin/env python3
# C762 (Q1340): the tensors of a GGUF file from its header alone (no numpy, no weights read): name, type, shape and
# bytes, for the tensors whose names match a regular expression. Bytes come from the type's block size where the type
# is a standard ggml one; for another type (a fork's own) the gap to the next tensor's offset is printed instead,
# marked "<=".
# Usage: python3 -I gguf_tensors.py <file.gguf> [name regex, default: all] > tensors.tsv
import re
import struct
import sys

# ggml type id: (name, block size, bytes per block)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def main():
    path = sys.argv[1]
    pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else "")
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize(fmt)
            return struct.unpack(fmt, f.read(n))[0]

        def rstr():
            return f.read(rd("<Q")).decode("utf-8", "replace")

        def skip_value(t):
            if t == 8:
                f.seek(rd("<Q"), 1)
            elif t == 9:
                at, n = rd("<I"), rd("<Q")
                if at in SCALAR:
                    f.seek(struct.calcsize(SCALAR[at]) * n, 1)
                else:
                    for _ in range(n):
                        skip_value(at)
            else:
                f.seek(struct.calcsize(SCALAR[t]), 1)

        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF file")
        version, n_tensors, n_kv = rd("<I"), rd("<Q"), rd("<Q")
        alignment = 32
        for _ in range(n_kv):
            key, t = rstr(), rd("<I")
            if key == "general.alignment" and t == 4:
                alignment = rd("<I")
            else:
                skip_value(t)
        infos = []
        for _ in range(n_tensors):
            name = rstr()
            dims = [rd("<Q") for _ in range(rd("<I"))]
            infos.append((name, dims, rd("<I"), rd("<Q")))
        offsets = sorted(i[3] for i in infos)
    print("file\tname\ttype\tshape\tbytes\tMiB")
    for name, dims, t, off in infos:
        if not pat.search(name):
            continue
        n = 1
        for d in dims:
            n *= d
        if t in TYPES:
            tname, blk, size = TYPES[t]
            nbytes, mark = n // blk * size, ""
        else:
            nxt = [o for o in offsets if o > off]
            tname, nbytes, mark = f"type{t}", (nxt[0] - off) if nxt else 0, "<="
        print(f"{path.rsplit('/', 1)[-1]}\t{name}\t{tname}\t{'x'.join(map(str, dims))}\t{mark}{nbytes}\t{mark}{nbytes / 1048576:.2f}")


if __name__ == "__main__":
    main()
