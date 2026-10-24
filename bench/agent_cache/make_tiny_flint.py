"""A tiny Flint with random weights, for exactness checks of the server's cache paths on a CPU (Q111).

    python make_tiny_flint.py FLINT.gguf OUT.gguf [--layers 6] [--seed 1]

It keeps everything that decides what the server caches and reuses: Flint's architecture, its tokenizer and chat
template, the 513-token sliding window and its layer pattern (two dense full-attention layers, then three
sliding-window layers per full-attention one), and shrinks the rest (256-wide, 4 heads, 8 experts), so a CPU runs
prompts of thousands of tokens in seconds. Its output is noise; the checks compare logits bit for bit between a cold
server and the cache paths. Needs numpy and gguf-py (llama.cpp's gguf-py on sys.path or installed).
"""
import argparse
import os
import sys

import numpy as np

for p in (os.environ.get("GGUF_PY", ""), "gguf-py"):
    if p and os.path.isdir(p):
        sys.path.insert(0, p)
import gguf  # noqa: E402

DIMS = {"embedding_length": 256, "feed_forward_length": 512, "attention.head_count": 4, "attention.head_count_kv": 2,
        "attention.key_length": 64, "attention.value_length": 64, "expert_count": 8, "expert_used_count": 2,
        "expert_feed_forward_length": 64, "expert_shared_feed_forward_length": 128}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flint")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    r = gguf.GGUFReader(args.flint)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(args.out, arch)
    n_layer = args.layers
    pattern = r.fields[f"{arch}.attention.sliding_window_pattern"].contents()[:n_layer]
    rope = r.fields[f"{arch}.rope_pattern" if f"{arch}.rope_pattern" in r.fields else f"{arch}.attention.rope_pattern"].contents()[:n_layer]
    n_dense = r.fields[f"{arch}.leading_dense_block_count"].contents()

    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture" or key.startswith("quantize."):
            continue
        short = key[len(arch) + 1:] if key.startswith(arch + ".") else None
        vtype = field.types[0]
        sub = field.types[1] if vtype == gguf.GGUFValueType.ARRAY and len(field.types) > 1 else None
        if short == "block_count":
            val = n_layer
        elif short in DIMS:
            val = DIMS[short]
        elif short == "attention.sliding_window_pattern":
            val = pattern
        elif short in ("rope_pattern", "attention.rope_pattern"):
            val = rope
        elif key == "general.file_type":
            val = 1  # F16
        else:
            val = field.contents()
        w.add_key_value(key, val, vtype, sub_type=sub)

    rng = np.random.default_rng(args.seed)
    d = DIMS
    n_embd, n_head, n_kv, hd = d["embedding_length"], d["attention.head_count"], d["attention.head_count_kv"], d["attention.key_length"]
    n_vocab = len(r.fields["tokenizer.ggml.tokens"].data)
    n_exp, ff_exp, ff_sh, ff = d["expert_count"], d["expert_feed_forward_length"], d["expert_shared_feed_forward_length"], d["feed_forward_length"]

    def mat(*shape, scale=0.05):
        # numpy shape is the reverse of the GGUF (ggml) shape
        return (rng.standard_normal(shape[::-1]) * scale).astype(np.float16)

    def ones(n):
        return np.ones(n, dtype=np.float32)

    w.add_tensor("token_embd.weight", mat(n_embd, n_vocab, scale=1.0))
    w.add_tensor("output_norm.weight", ones(n_embd))
    w.add_tensor("output.weight", mat(n_embd, n_vocab))
    for i in range(n_layer):
        b = f"blk.{i}."
        w.add_tensor(b + "attn_norm.weight", ones(n_embd))
        w.add_tensor(b + "attn_q.weight", mat(n_embd, n_head * hd))
        w.add_tensor(b + "attn_k.weight", mat(n_embd, n_kv * hd))
        w.add_tensor(b + "attn_v.weight", mat(n_embd, n_kv * hd))
        w.add_tensor(b + "attn_q_norm.weight", ones(hd))
        w.add_tensor(b + "attn_k_norm.weight", ones(hd))
        w.add_tensor(b + "attn_output.weight", mat(n_head * hd, n_embd))
        w.add_tensor(b + "attn_sinks.weight", (rng.standard_normal(n_head) * 0.5).astype(np.float32))
        w.add_tensor(b + "ffn_norm.weight", ones(n_embd))
        if i < n_dense:
            w.add_tensor(b + "ffn_gate.weight", mat(n_embd, ff))
            w.add_tensor(b + "ffn_up.weight", mat(n_embd, ff))
            w.add_tensor(b + "ffn_down.weight", mat(ff, n_embd))
        else:
            w.add_tensor(b + "ffn_gate_inp.weight", mat(n_embd, n_exp, scale=0.5))
            w.add_tensor(b + "ffn_gate_exps.weight", mat(n_embd, ff_exp, n_exp))
            w.add_tensor(b + "ffn_up_exps.weight", mat(n_embd, ff_exp, n_exp))
            w.add_tensor(b + "ffn_down_exps.weight", mat(ff_exp, n_embd, n_exp))
            w.add_tensor(b + "ffn_gate_shexp.weight", mat(n_embd, ff_sh))
            w.add_tensor(b + "ffn_up_shexp.weight", mat(n_embd, ff_sh))
            w.add_tensor(b + "ffn_down_shexp.weight", mat(ff_sh, n_embd))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{args.out}: {n_layer} layers, sliding-window pattern {pattern}, vocab {n_vocab}")


if __name__ == "__main__":
    main()
