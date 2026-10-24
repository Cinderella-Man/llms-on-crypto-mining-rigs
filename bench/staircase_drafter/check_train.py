"""Checks of the training options before a long run (CPU is enough):
- key slicing in the window attention gives the same outputs as attending to every committed key;
- a fused model (--fuse) loaded from an unfused checkpoint reproduces that checkpoint exactly (zero-initialized fusion);
- the LK loss has finite gradients; sampled windows are valid staircases.

usage: python check_train.py --feats FEAT.bin --vocab vocab.npy --flint FLINT_DIR --head HEAD.pt [--device cpu]
"""
import argparse
import random

import numpy as np
import torch

import drafter
import feats
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--flint", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--record", type=int, default=1)
    args = ap.parse_args()
    dev, dtype = args.device, torch.float32
    vocab = np.load(args.vocab)
    emb, head_rows, norm_w = train.load_flint(args.flint, vocab, dev)
    emb = emb.float()
    t2d = train.t2d_table(vocab, emb.shape[0])
    rec = feats.FeatFile(args.feats)[args.record]
    prof = drafter.PROFILES["A"]
    sd = torch.load(args.head, map_location=dev)
    sd = sd.get("model", sd)
    base = drafter.Drafter(emb, head_rows, norm_w).to(dev)
    base.load_state_dict(sd, strict=False)
    base.eval()
    b = train.Batch(rec, prof, t2d, min_target=0, all_anchors=True)
    d = train.to_device(b, dev, dtype)
    committed, ca, win, _ = train.build_inputs(d, prof, prof["windows"], rec.n_embd, dtype)
    with torch.no_grad():
        drafter.SLICE_KEYS = True
        o1 = base(d["tokens"], committed, ca, win)
        drafter.SLICE_KEYS = False
        o2 = base(d["tokens"], committed, ca, win)
        drafter.SLICE_KEYS = True
    for name in o1:
        print(f"key slicing, window {name}: max |diff| {float((o1[name] - o2[name]).abs().max()):.2e}")
    fused = drafter.Drafter(emb, head_rows, norm_w, fuse=True).to(dev)
    missing, _ = fused.load_state_dict(sd, strict=False)
    fused.eval()
    committed, ca, win, cextra = train.build_inputs(d, prof, prof["windows"], rec.n_embd, dtype, fuse=True)
    with torch.no_grad():
        o3 = fused(d["tokens"], committed, ca, win, committed_extra=cextra)
    print(f"fused model, new parameters: {len(missing)}")
    for name in o1:
        print(f"fused warm start, window {name}: max |diff| {float((o1[name] - o3[name]).abs().max()):.2e}")
    fused.train()
    for w in fused.w_fuse:
        torch.nn.init.normal_(w.weight, std=1e-3)
    out = fused(d["tokens"], committed, ca, win, committed_extra=cextra)
    p = feats.sampler_dist(d["top_lp"])
    loss, kl, ce = drafter.lk_loss(out["stair"], (d["top_draft"], p), d["sampled"])
    loss.backward()
    g = [x.grad for x in fused.parameters() if x.grad is not None]
    print(f"LK loss {loss.item():.3f} (KL {kl.item():.3f}); gradients finite: "
          f"{all(torch.isfinite(x).all().item() for x in g)}; fusion weights got gradients: "
          f"{all(w.weight.grad is not None and w.weight.grad.abs().sum() > 0 for w in fused.w_fuse[:4])}")
    rng = random.Random(0)
    for _ in range(5):
        w = train.sample_window(prof, rng, (0.2, 0.2, 0.6))["sample"]
        ids = [drafter.ANCHOR_IDS[a] for a in w]
        assert ids == sorted(ids) and 6 not in ids, w
        print("sampled window", w)


if __name__ == "__main__":
    main()
