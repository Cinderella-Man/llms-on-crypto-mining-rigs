#!/usr/bin/env python3
# C957 (2026-11-23): the tensor types of a GGUF file, counted, from its header only (standard library, no gguf-py):
# which of the fork's Pascal kernels the file's weights can reach (the matrix-vector kernels take Q4_0, Q4_1, Q5_0,
# Q8_0 and the other quantized types, the F16 twin BF16 only). Prints "<file> <arch> <type>=<count> ..." per file.
# With -v first, also one line per tensor: name, type, shape.
# Usage: python3 -I model_types.py [-v] <file.gguf> ...
import struct
import sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 20: "IQ4_NL", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
         28: "F64", 30: "BF16"}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def string(f):
    return f.read(read(f, "Q")).decode("utf-8", "replace")


def value(f, t):
    if t in SCALAR:
        return read(f, SCALAR[t])
    if t == 8:
        return string(f)
    if t == 9:
        et, n = read(f, "I"), read(f, "Q")
        return [value(f, et) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {t}")


def main() -> int:
    args = sys.argv[1:]
    verbose = bool(args) and args[0] == "-v"
    if verbose:
        args = args[1:]
    for path in args:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                print(f"{path} not a GGUF file")
                continue
            read(f, "I")
            n_t, n_kv = read(f, "Q"), read(f, "Q")
            arch = "?"
            for _ in range(n_kv):
                k = string(f)
                v = value(f, read(f, "I"))
                if k == "general.architecture":
                    arch = v
            counts = {}
            for _ in range(n_t):
                name = string(f)
                nd = read(f, "I")
                shape = [read(f, "Q") for _ in range(nd)]
                t = TYPES.get(read(f, "I"), "?")
                read(f, "Q")
                counts[t] = counts.get(t, 0) + 1
                if verbose:
                    print(f"  {name}\t{t}\t{'x'.join(map(str, shape))}")
        print(path, arch, " ".join(f"{t}={n}" for t, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
