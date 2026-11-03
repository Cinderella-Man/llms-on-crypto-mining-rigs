#!/usr/bin/env python3
"""The layout of a (split) GGUF: key counts, tensors and bytes per type and per tensor kind, which shard holds the
n-gram table (`per_layer_token_embd`) and alone or not, and the bytes the GPUs hold: everything but the table and the
token embeddings, which llama.cpp keeps on the host.
  python3 -I layout.py SHARD...
Standard library only; gguf_reshard.py from the repository's scripts/ (next to this file or in ../../../../scripts).
"""
import collections, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "..", "..", "..", "scripts")):
    sys.path.insert(0, os.path.abspath(d))
from gguf_reshard import Header  # noqa: E402

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 6: "Q5_0", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
         14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
         29: "IQ1_M", 30: "BF16"}
HOST = ("per_layer_token_embd.weight", "token_embd.weight")


def main():
    by_type, by_kind = collections.Counter(), collections.defaultdict(collections.Counter)
    total = host = n = 0
    for i, p in enumerate(sys.argv[1:], 1):
        h = Header(p)
        names = [t[0] for t in h.tensors]
        if "per_layer_token_embd.weight" in names:
            print(f"n-gram table in shard {i} ({os.path.basename(p)}), {'alone' if len(names) == 1 else f'with {len(names) - 1} other tensors'}")
        if i == 1:
            arch = h.value("general.architecture")
            arch = arch[8:].decode() if isinstance(arch, bytes) else arch
            print(f"architecture {arch}, block_count {h.value(f'{arch}.block_count')}, "
                  f"nextn_predict_layers {h.value(f'{arch}.nextn_predict_layers', 0)}, "
                  f"split.tensors.count {h.value('split.tensors.count')}, file_type {h.value('general.file_type')}")
        for name, _dims, typ, _off in h.tensors:
            size = h.slot[name]
            t = TYPES.get(typ, str(typ))
            by_type[t] += size
            by_kind[re.sub(r"blk\.\d+\.", "blk.N.", name)][t] += 1
            total += size
            host += size if name in HOST else 0
            n += 1
    print(f"{n} tensors, {total / 1e9:.2f} GB; on the GPUs {(total - host) / 1e9:.2f} GB, on the host {host / 1e9:.2f} GB")
    for t, size in by_type.most_common():
        print(f"  {t:8s} {size / 1e9:8.2f} GB")
    for kind in sorted(by_kind):
        print(f"  {kind}: " + ", ".join(f"{c} {t}" for t, c in sorted(by_kind[kind].items())))


if __name__ == "__main__":
    main()
