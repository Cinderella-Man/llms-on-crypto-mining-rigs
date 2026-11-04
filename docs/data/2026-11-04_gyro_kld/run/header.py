#!/usr/bin/env python3
"""What a Gyro GGUF holds, from its header alone: the model keys (architecture, layers, experts), every tensor type
with its bytes, the tensors per kind and type, the n-gram table (`per_layer_token_embd`) and the bytes the GPUs hold
(everything but the table and the token embeddings, which llama.cpp keeps on the host). Type ids are those of
agentionai's llama.cpp fork (ggml.h), which adds the trellis types 143-149.
  python3 -I header.py FILE...
Standard library only; gguf_reshard.py from the repository's scripts/ (next to this file or in ../../../../scripts).
"""
import collections, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "..", "..", "..", "scripts")):
    sys.path.insert(0, os.path.abspath(d))
from gguf_reshard import Header  # noqa: E402

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
         21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
         41: "Q1_0", 42: "Q2_0", 143: "PTQ1_0", 144: "TQ2_T", 145: "TQK6", 146: "TQK7", 147: "TQP5", 148: "TQP6",
         149: "TQP7"}
HOST = ("per_layer_token_embd.weight", "token_embd.weight")
GiB = 1024 ** 3


def text(v):
    return v[8:].decode() if isinstance(v, bytes) else v


def main():
    for p in sys.argv[1:]:
        h = Header(p)
        arch = text(h.value("general.architecture"))
        print(f"== {os.path.basename(p)}")
        for k in ("general.name", "general.file_type", "general.quantization_version", f"{arch}.block_count",
                  f"{arch}.expert_count", f"{arch}.expert_used_count", f"{arch}.expert_shared_count",
                  f"{arch}.expert_feed_forward_length", f"{arch}.nextn_predict_layers", f"{arch}.context_length",
                  f"{arch}.vocab_size", "split.count"):
            v = h.value(k)
            if v is not None:
                print(f"  {k} = {text(v)}")
        by_type, by_kind = collections.Counter(), collections.defaultdict(collections.Counter)
        total = host = 0
        dims = {}
        for name, d, typ, _off in h.tensors:
            size = h.slot[name]
            t = TYPES.get(typ, f"type{typ}")
            by_type[t] += size
            kind = re.sub(r"blk\.\d+\.", "blk.N.", name)
            by_kind[kind][t] += 1
            dims.setdefault(kind, d)
            total += size
            host += size if name in HOST else 0
        table = h.slot.get("per_layer_token_embd.weight", 0)
        print(f"  {len(h.tensors)} tensors, {total / GiB:.2f} GiB; on the GPUs {(total - host) / GiB:.2f} GiB, "
              f"n-gram table {table / GiB:.2f} GiB, token embeddings {(host - table) / GiB:.2f} GiB")
        for t, size in by_type.most_common():
            print(f"    {t:8s} {size / GiB:8.2f} GiB")
        for kind in sorted(by_kind):
            print(f"    {kind} {list(dims[kind])}: " + ", ".join(f"{c} {t}" for t, c in sorted(by_kind[kind].items())))


if __name__ == "__main__":
    main()
