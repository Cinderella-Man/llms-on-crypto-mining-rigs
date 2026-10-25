#!/usr/bin/env python3
# The tensors of a GGUF as a manifest, to check a rebuilt model file against the shipped one tensor by tensor
# (release/flint-v2/reproduce/model.md). The header is left out on purpose: metadata differs between builds (names,
# licence fields, tool versions), the tensors must not.
#
#   tensor_manifest.py hash FILE.gguf > FILE.tensors.tsv     name, type, shape, bytes and SHA-256 of every tensor
#   tensor_manifest.py types FILE.gguf > FILE.types          name and type only: the tensor-type map
#   tensor_manifest.py compare A.tsv B.tsv [--ignore REGEX]  same names, types, shapes and bytes? exit 0 if so
#   tensor_manifest.py quantize-args TYPES [--skip REGEX]    llama-quantize options that give every listed tensor its
#                                                            type (one --tensor-type '^name$=type' each; tensors that
#                                                            stay in F32/F16/BF16 and those matching --skip left out)
# Needs numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py) for hash and types.
import argparse
import hashlib
import re
import sys

FLOAT_TYPES = {"F32", "F16", "BF16"}


def read_tensors(path):
    import gguf
    return gguf.GGUFReader(path).tensors


def cmd_hash(path):
    print("name\ttype\tshape\tbytes\tsha256")
    for t in read_tensors(path):
        h = hashlib.sha256(memoryview(t.data).cast("B")).hexdigest()
        print(f"{t.name}\t{t.tensor_type.name}\t{'x'.join(str(int(d)) for d in t.shape)}\t{int(t.n_bytes)}\t{h}")


def cmd_types(path):
    for t in read_tensors(path):
        print(f"{t.name}\t{t.tensor_type.name}")


def load_tsv(path):
    rows = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if not f or f[0] == "name":
            continue
        rows[f[0]] = f[1:]
    return rows


def cmd_compare(a, b, ignore):
    ra, rb = load_tsv(a), load_tsv(b)
    pat = re.compile(ignore) if ignore else None
    bad = 0
    for name in sorted(set(ra) | set(rb)):
        if pat and pat.search(name):
            continue
        if name not in ra or name not in rb:
            print(f"only in {'B' if name not in ra else 'A'}: {name}")
            bad += 1
        elif ra[name] != rb[name]:
            print(f"differs: {name}: {ra[name]} vs {rb[name]}")
            bad += 1
    checked = len([n for n in set(ra) & set(rb) if not (pat and pat.search(n))])
    print(f"{checked} tensors compared, {bad} differences" + (f" (ignored: {ignore})" if ignore else ""))
    return 1 if bad else 0


def cmd_quantize_args(types, skip):
    pat = re.compile(skip) if skip else None
    for line in open(types):
        f = line.split()
        if len(f) < 2 or f[1] in FLOAT_TYPES or (pat and pat.search(f[0])):
            continue
        print(f"--tensor-type ^{re.escape(f[0])}$={f[1].lower()}")


def main():
    ap = argparse.ArgumentParser(description="Tensor manifests of GGUF files (see the header).")
    ap.add_argument("cmd", choices=["hash", "types", "compare", "quantize-args"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ignore")
    ap.add_argument("--skip")
    a = ap.parse_args()
    if a.cmd == "hash":
        cmd_hash(a.files[0])
    elif a.cmd == "types":
        cmd_types(a.files[0])
    elif a.cmd == "compare":
        return cmd_compare(a.files[0], a.files[1], a.ignore)
    else:
        cmd_quantize_args(a.files[0], a.skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
