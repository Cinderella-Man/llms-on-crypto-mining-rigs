#!/usr/bin/env python3
"""The tensors of two GGUF files compared by name, shape and type.

Usage: gguf_names.py <gguf-py dir> <a.gguf> <b.gguf>

Prints "tensors: a <n>, b <n>; only tip <n>, only base <n>, differ <n>" (a is the tip's file, b the base's), then one
line per tensor that is in one file only or whose shape or type differs.
"""
import sys

sys.path.insert(0, sys.argv[1])
from gguf import GGUFReader  # noqa: E402


def tensors(path):
    return {t.name: (tuple(int(x) for x in t.shape), t.tensor_type.name) for t in GGUFReader(path).tensors}


def main():
    a, b = tensors(sys.argv[2]), tensors(sys.argv[3])
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
