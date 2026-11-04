#!/usr/bin/env python3
"""A (split) GGUF's tensor index as JSON, from its headers only: never a byte of tensor data is read.
  python3 -I dump.py NAME SHARD... > NAME.json
Writes the model's shape keys (block count, experts, attention layout, file type), each shard's basename, size and
tensor count, and every tensor as [name, type, dims, bytes, shard]. Bytes are the tensor's slot in its shard (its data
plus the alignment padding up to the next tensor, at most 31 bytes). model.py and fit.sh read these files.
Standard library only; gguf_reshard.py from the repository's scripts/.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "scripts")))
from gguf_reshard import Header  # noqa: E402

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16", 42: "Q2_0"}
KEYS = ("block_count", "embedding_length", "expert_count", "expert_used_count", "expert_feed_forward_length",
        "expert_shared_feed_forward_length", "full_attention_interval", "context_length")


def text(v):
    return v[8:].decode() if isinstance(v, bytes) else v


def main():
    name, shards = sys.argv[1], sys.argv[2:]
    heads = [Header(p) for p in shards]
    arch = text(heads[0].value("general.architecture"))
    out = {"name": name, "architecture": arch, "file_type": heads[0].value("general.file_type"),
           "keys": {k: heads[0].value(f"{arch}.{k}") for k in KEYS},
           "shards": [], "tensors": []}
    for i, h in enumerate(heads, 1):
        out["shards"].append({"file": os.path.basename(h.path), "bytes": h.size, "tensors": len(h.tensors),
                              "header_bytes": h.data0})
        for tname, dims, typ, _off in h.tensors:
            out["tensors"].append([tname, TYPES.get(typ, str(typ)), dims, h.slot[tname], i])
    json.dump(out, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
