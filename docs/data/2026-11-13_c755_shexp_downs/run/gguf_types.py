#!/usr/bin/env python3
"""List every tensor of one or more GGUF files with its ggml type and shape (C755): name, type, ne0, ne1, ne2, file.

Reads only the files' headers. The type names come from the given ggml.h (its enum ggml_type), so the fork's own types
(Q2_0, N2P ...) are named as the build that loads the file names them. No dependencies beyond the standard library.

Usage: gguf_types.py <ggml.h> <file.gguf> [<file.gguf> ...] > types.tsv
"""
import os
import re
import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_fmt(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def read_str(f):
    return f.read(read_fmt(f, "<Q")).decode("utf-8", "replace")


def skip_value(f, vtype):
    if vtype in SCALAR:
        f.seek(struct.calcsize(SCALAR[vtype]), 1)
    elif vtype == 8:
        f.seek(read_fmt(f, "<Q"), 1)
    elif vtype == 9:
        etype, n = read_fmt(f, "<I"), read_fmt(f, "<Q")
        if etype in SCALAR:
            f.seek(n * struct.calcsize(SCALAR[etype]), 1)
        else:
            for _ in range(n):
                skip_value(f, etype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def main():
    names = {}
    with open(sys.argv[1], encoding="utf-8") as h:
        for m in re.finditer(r"GGML_TYPE_(\w+)\s*=\s*(\d+)", h.read()):
            names.setdefault(int(m.group(2)), m.group(1))
    print("name\ttype\tne0\tne1\tne2\tfile")
    for path in sys.argv[2:]:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"{path}: not a GGUF file")
            read_fmt(f, "<I")
            n_tensors, n_kv = read_fmt(f, "<Q"), read_fmt(f, "<Q")
            for _ in range(n_kv):
                read_str(f)
                skip_value(f, read_fmt(f, "<I"))
            for _ in range(n_tensors):
                name = read_str(f)
                ne = [read_fmt(f, "<Q") for _ in range(read_fmt(f, "<I"))]
                ttype = read_fmt(f, "<I")
                read_fmt(f, "<Q")
                ne += [1] * (3 - len(ne))
                print(f"{name}\t{names.get(ttype, ttype)}\t{ne[0]}\t{ne[1]}\t{ne[2]}\t{os.path.basename(path)}")


if __name__ == "__main__":
    main()
