#!/usr/bin/env python3
"""What a decoded token reads, by weight type, and which of those types have a Pascal kernel: from the GGUF headers
of a (split) file only, never its tensor data.
  python3 -I iq_layout.py [--pascal TYPES] SHARD...
Prints the model's expert counts, every tensor kind with its shape and types, then per weight type: the bytes on the
GPUs, the bytes one token reads there (dense tensors whole; routed experts, ffn_*_exps, times experts used / experts),
and whether the fork's Pascal one-token kernels take that type (--pascal, a comma-separated list; the default is the
list of Miner Llama v3's mmvq-pascal, mmvq-pascal-kcol (Q3_K: plain products) and mmvf-pascal kernels, with this
folder's IQ2_XXS, IQ2_XS and IQ3_S). The n-gram table (per_layer_token_embd) and the token embeddings stay in host
memory, so they are listed apart. F32 tensors are the routers (ffn_gate_inp, read whole every token) and the norms.
Standard library only; gguf_reshard.py from the repository's scripts/ (next to this file or in ../../../../scripts).
"""
import argparse, collections, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "..", "..", "..", "scripts")):
    sys.path.insert(0, os.path.abspath(d))
from gguf_reshard import Header  # noqa: E402

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
         21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16", 42: "Q2_0"}
HOST = re.compile(r"^(per_layer_token_embd|token_embd)\.")
EXPS = re.compile(r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$")
PASCAL = "Q4_0,Q4_1,Q5_0,Q8_0,Q3_K,Q4_K,Q5_K,Q6_K,IQ4_XS,Q2_0,IQ2_S,IQ3_XXS,BF16,IQ2_XXS,IQ2_XS,IQ3_S"


def scalar(h, key, default=None):
    return h.value(key, default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--pascal", default=PASCAL, help="weight types with a Pascal one-token kernel")
    a = ap.parse_args()
    pascal = set(a.pascal.split(","))
    heads = [Header(p) for p in a.shards]
    h0 = heads[0]
    arch = h0.value("general.architecture")
    arch = arch[8:].decode() if isinstance(arch, bytes) else arch
    n_exp = scalar(h0, f"{arch}.expert_count")
    n_used = scalar(h0, f"{arch}.expert_used_count")
    print(f"architecture {arch}, block_count {scalar(h0, f'{arch}.block_count')}, embedding_length "
          f"{scalar(h0, f'{arch}.embedding_length')}, expert_count {n_exp}, expert_used_count {n_used}, "
          f"expert_feed_forward_length {scalar(h0, f'{arch}.expert_feed_forward_length')}, "
          f"expert_shared_feed_forward_length {scalar(h0, f'{arch}.expert_shared_feed_forward_length')}, "
          f"file_type {scalar(h0, 'general.file_type')}")
    kinds = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, None]))  # kind -> type -> n, B, dims
    gpu, tok, host = collections.Counter(), collections.Counter(), collections.Counter()
    for i, h in enumerate(heads, 1):
        for name, dims, typ, _off in h.tensors:
            t = TYPES.get(typ, str(typ))
            size = h.slot[name]
            k = kinds[re.sub(r"blk\.\d+\.", "blk.N.", name)][t]
            k[0] += 1
            k[1] += size
            k[2] = k[2] or "x".join(map(str, dims))
            if HOST.match(name):
                host[t] += size
                print(f"host tensor {name} {t} {'x'.join(map(str, dims))} {size / 1e9:.2f} GB in shard {i}"
                      f" ({len(h.tensors)} tensors in that shard)")
                continue
            gpu[t] += size
            tok[t] += size * n_used / n_exp if EXPS.match(name) else size
    print("\nkind\ttype\ttensors\tshape (ne0 x ne1 x ne2)\tGB")
    for kind in sorted(kinds):
        for t, (n, size, dims) in sorted(kinds[kind].items()):
            print(f"{kind}\t{t}\t{n}\t{dims}\t{size / 1e9:.3f}")
    G, T = sum(gpu.values()), sum(tok.values())
    print(f"\non the GPUs {G / 1e9:.2f} GB ({G / 2**20:.0f} MiB); host {sum(host.values()) / 1e9:.2f} GB; "
          f"one token reads {T / 1e9:.3f} GB on the GPUs")
    print("type\tGPU GB\tper token MB\tshare of a token\tPascal kernel")
    for t, size in sorted(tok.items(), key=lambda x: -x[1]):
        print(f"{t}\t{gpu[t] / 1e9:.2f}\t{size / 1e6:.1f}\t{100 * size / T:.1f} %\t"
              f"{'yes' if t in pascal else 'no'}")
    cov = sum(s for t, s in tok.items() if t in pascal)
    print(f"covered by Pascal kernels: {100 * cov / T:.1f} % of a token's bytes")


if __name__ == "__main__":
    main()
