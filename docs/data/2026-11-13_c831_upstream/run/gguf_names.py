#!/usr/bin/env python3
"""The tensors of two GGUF files compared by name, shape and type, from their headers (no dependencies).

Usage: gguf_names.py <a.gguf> <b.gguf>

Prints "tensors: a <n>, b <n>; only tip <n>, only base <n>, differ <n>" (a is the tip's file, b the base's), then one
line per tensor that is in one file only or whose shape or type (ggml type id) differs.
"""
import struct
import sys

# GGUF metadata value types: fixed sizes; 8 = string, 9 = array
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_header(path):
    with open(path, "rb") as f:
        def u32():
            return struct.unpack("<I", f.read(4))[0]

        def u64():
            return struct.unpack("<Q", f.read(8))[0]

        def string():
            return f.read(u64()).decode("utf-8", "replace")

        def skip(vtype):
            if vtype == 8:
                f.seek(u64(), 1)
            elif vtype == 9:
                itype, n = u32(), u64()
                if itype in SIZES:
                    f.seek(SIZES[itype] * n, 1)
                else:
                    for _ in range(n):
                        skip(itype)
            else:
                f.seek(SIZES[vtype], 1)

        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        u32()  # version
        n_tensors, n_kv = u64(), u64()
        for _ in range(n_kv):
            string()
            skip(u32())
        tensors = {}
        for _ in range(n_tensors):
            name = string()
            dims = tuple(u64() for _ in range(u32()))
            ttype = u32()
            u64()  # offset
            tensors[name] = (dims, ttype)
        return tensors


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = read_header(sys.argv[1]), read_header(sys.argv[2])
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    differ = sorted(n for n in set(a) & set(b) if a[n] != b[n])
    print(f"tensors: a {len(a)}, b {len(b)}; only tip {len(only_a)}, only base {len(only_b)}, differ {len(differ)}")
    for n in only_a:
        print(f"only tip: {n} {a[n]}")
    for n in only_b:
        print(f"only base: {n} {b[n]}")
    for n in differ:
        print(f"differ: {n} tip {a[n]} base {b[n]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
