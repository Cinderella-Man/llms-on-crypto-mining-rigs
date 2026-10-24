"""Export a trained staircase-guesser head (Q20/D1) to the GGUF file the fork's pipelined executor loads
(llama_pipe_draft_init, llama-pipespec / llama-server --pipe-draft).

usage: python export_gguf.py --head heads/A --vocab vocab.npy --out drafter-A.gguf [--weights head.pt] [--type q8_0]
                             [--alpha 3] [--stair L9,L17,L33,L41]
needs numpy, torch and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py). --head is a train.py output folder (config.json and
head.pt, or --weights for another checkpoint: head-last.pt, ckpt.pt).

The file holds the head as drafter.py defines it, with every RMS norm's weight folded into the projection after it:
  sd.emb.weight         [d, d]        w_emb * emb_norm
  sd.anc.weight         [d, d, 6]     w_anc[k] * anc_norm[k], anchors L9, L17, L25, L33, L41, final state
  sd.anchor_emb.weight  [d, 7]        anchor embeddings (emb, L9 .. L41, final)
  sd.fuse.weight        [d, d, 5]     w_fuse[k] * fuse_norm[k] (fused heads only)
  sd.blk.<i>.wq/wk/wv   [d, n]        times attn_norm
  sd.blk.<i>.wo         [n, d]
  sd.blk.<i>.w1/w3      [d, ffn]      times mlp_norm
  sd.blk.<i>.w2         [ffn, d]
  sd.head.weight        [d, V + 1]    head * the final norm (V draft ids and the "other" class)
  sd.head.bias          [V + 1]
  sd.vocab              [V] int32     draft id -> Flint token id
Metadata (staircase.*): sizes, rms_eps, rope_theta (interleaved pairs: ggml rope mode 0), anchor_layers (the layer
inputs the anchors read; the final state after Flint's output norm is the last anchor), fuse, committed (the anchor of
committed rows), stair (anchors of the tokens in flight, newest first, from the head's profile), alpha (the guess is
drawn from q^alpha), profile.
2-D and 3-D matrices are stored as --type (q8_0, f16 or f32), everything else as f32.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANCHOR_LAYERS = [9, 17, 25, 33, 41]       # drafter.ANCHOR_IDS: L9 .. L41, then "full"


def profile_stair(profile):
    """The anchors of the tokens in flight for the rig: the head's first window that is not the after-miss one."""
    wins = profile["windows"]
    for name, anchors in wins.items():
        if name != "miss":
            return list(anchors)
    return list(wins["miss"])


def export(sd, head, n_layer, fuse, vocab, out, qtype="q8_0", alpha=3.0, stair=None, head_dim=128, rms_eps=1e-6,
           rope_theta=1e5, committed=None):
    """sd: the head's state dict (name -> numpy float32), head: profile name (drafter.PROFILES)."""
    import gguf
    import drafter

    profile = drafter.PROFILES[head]
    stair = stair or profile_stair(profile)
    committed = committed or profile["committed"]
    d = sd["w_emb.weight"].shape[1]
    n_q = sd["blocks.0.wq.weight"].shape[0]
    n_kv = sd["blocks.0.wk.weight"].shape[0]
    n_ff = sd["blocks.0.w1.weight"].shape[0]
    n_v1 = sd["head.weight"].shape[0]
    assert n_v1 == len(vocab) + 1, (n_v1, len(vocab))
    assert n_q % head_dim == 0 and n_kv % head_dim == 0

    def fold(w, norm):
        return w * norm[None, :]

    mats = {
        "sd.emb.weight": fold(sd["w_emb.weight"], sd["emb_norm.weight"]),
        "sd.anc.weight": np.stack([fold(sd[f"w_anc.{k}.weight"], sd[f"anc_norm.{k}.weight"]) for k in range(6)]),
        "sd.head.weight": fold(sd["head.weight"], sd["norm.weight"]),
    }
    if fuse:
        mats["sd.fuse.weight"] = np.stack([fold(sd[f"w_fuse.{k}.weight"], sd[f"fuse_norm.{k}.weight"]) for k in range(5)])
    for i in range(n_layer):
        b = f"blocks.{i}."
        an, mn = sd[b + "attn_norm.weight"], sd[b + "mlp_norm.weight"]
        mats[f"sd.blk.{i}.wq"] = fold(sd[b + "wq.weight"], an)
        mats[f"sd.blk.{i}.wk"] = fold(sd[b + "wk.weight"], an)
        mats[f"sd.blk.{i}.wv"] = fold(sd[b + "wv.weight"], an)
        mats[f"sd.blk.{i}.wo"] = sd[b + "wo.weight"]
        mats[f"sd.blk.{i}.w1"] = fold(sd[b + "w1.weight"], mn)
        mats[f"sd.blk.{i}.w3"] = fold(sd[b + "w3.weight"], mn)
        mats[f"sd.blk.{i}.w2"] = sd[b + "w2.weight"]

    w = gguf.GGUFWriter(out, "staircase-drafter")
    w.add_uint32("staircase.n_embd", d)
    w.add_uint32("staircase.n_layer", n_layer)
    w.add_uint32("staircase.n_head", n_q // head_dim)
    w.add_uint32("staircase.n_head_kv", n_kv // head_dim)
    w.add_uint32("staircase.head_dim", head_dim)
    w.add_uint32("staircase.n_ff", n_ff)
    w.add_uint32("staircase.n_vocab", len(vocab))
    w.add_float32("staircase.rms_eps", rms_eps)
    w.add_float32("staircase.rope_theta", rope_theta)
    w.add_array("staircase.anchor_layers", ANCHOR_LAYERS)
    w.add_bool("staircase.fuse", bool(fuse))
    w.add_string("staircase.committed", committed)
    w.add_string("staircase.stair", ",".join(stair))
    w.add_float32("staircase.alpha", float(alpha))
    w.add_string("staircase.profile", head)

    q8 = gguf.GGMLQuantizationType.Q8_0
    for name, m in mats.items():
        m = np.ascontiguousarray(m, dtype=np.float32)
        if qtype == "q8_0":
            w.add_tensor(name, gguf.quants.quantize(m, q8), raw_dtype=q8)
        elif qtype == "f16":
            w.add_tensor(name, m.astype(np.float16))
        else:
            w.add_tensor(name, m)
    w.add_tensor("sd.anchor_emb.weight", np.ascontiguousarray(sd["anchor_emb.weight"], dtype=np.float32))
    w.add_tensor("sd.head.bias", np.ascontiguousarray(sd["head.bias"], dtype=np.float32))
    w.add_tensor("sd.vocab", np.ascontiguousarray(vocab, dtype=np.int32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"d": d, "n_layer": n_layer, "n_vocab": len(vocab), "fuse": bool(fuse), "stair": stair, "committed": committed,
            "alpha": alpha, "type": qtype}


def state_dict_numpy(path):
    import torch
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model", sd)
    return {k: v.float().numpy() for k, v in sd.items() if hasattr(v, "float")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="train.py output folder")
    ap.add_argument("--weights", help="checkpoint (default: <head>/head.pt)")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="q8_0", choices=("q8_0", "f16", "f32"))
    ap.add_argument("--alpha", type=float, default=3.0, help="q^alpha before the draw (score.py --alpha)")
    ap.add_argument("--stair", help="anchors of the tokens in flight, newest first (default: from the head's profile)")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.head, "config.json")))
    sd = state_dict_numpy(args.weights or os.path.join(args.head, "head.pt"))
    vocab = np.load(args.vocab)
    info = export(sd, cfg["head"], cfg["args"].get("layers", 2), cfg["args"].get("fuse", False), vocab, args.out,
                  qtype=args.type, alpha=args.alpha, stair=args.stair.split(",") if args.stair else None)
    info["bytes"] = os.path.getsize(args.out)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
