#!/usr/bin/env python3
"""Replaces one tensor of a split GGUF with the tensor of the same name and shape from another GGUF (its type may
differ), when the tensor is alone in its shard: that shard is written again with its own key/value pairs (split.no,
split.count, split.tensors.count) and the other file's tensor info and bytes, so the file keeps its shard count and
tensor count. Made to give a requantized Flash-Next file the n-gram table (`per_layer_token_embd`, IQ4_NL) that every
published build ships.
  python3 -I swap_tensor.py --tensor per_layer_token_embd.weight --source <GGUF holding it> --out <new shard> SHARD...
Prints the shard it replaces; the caller moves --out into its place. Standard library only; gguf_reshard.py from the
repository's scripts/ (next to this file or in ../../../../scripts).
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "..", "..", "..", "scripts")):
    sys.path.insert(0, os.path.abspath(d))
from gguf_reshard import Header, copy_range, write_header  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    src = Header(a.source)
    found = [t for t in src.tensors if t[0] == a.tensor]
    assert len(found) == 1, f"{a.source}: no {a.tensor}"
    _n, sdims, stype, soff = found[0]
    target = next((h for h in map(Header, a.shards) if any(t[0] == a.tensor for t in h.tensors)), None)
    assert target, f"no shard holds {a.tensor}"
    assert len(target.tensors) == 1, f"{target.path}: {a.tensor} shares the shard with {len(target.tensors) - 1} tensors"
    _n, tdims, ttype, _off = target.tensors[0]
    assert tdims == sdims, f"shapes differ: {tdims} against {sdims}"
    slot = src.slot[a.tensor]
    with open(a.out, "wb") as dst:
        write_header(dst, target.version, target.kv, [[a.tensor, sdims, stype, 0, slot]], target.align)
        with open(src.path, "rb") as f:
            copy_range(f, dst, src.data0 + soff, slot, target.align)
    print(f"{target.path}: {a.tensor} type {ttype} -> type {stype} from {a.source}; written {a.out} "
          f"({os.path.getsize(a.out):,} bytes)")


if __name__ == "__main__":
    main()
