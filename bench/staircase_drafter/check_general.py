"""Checks of the depth-general head (train.py --general) before a long run (CPU is enough):
- sample_general draws valid windows (recorded layers only, non-decreasing depths, at most 12 rows) and the card-count
  mix of CARD_WEIGHTS; the card-1 staircase per card count is printed;
- a general head loaded from a six-card head (init_general) gives the same outputs as that head without fused anchors
  on six-card windows.

usage: python check_general.py --feats FEAT.bin --vocab vocab.npy --flint FLINT_DIR --head HEAD.pt [--record 1]
"""
import argparse
import collections
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
    ap.add_argument("--record", type=int, default=1)
    args = ap.parse_args()
    ff = feats.FeatFile(args.feats)
    rec = ff[args.record]
    print("recorded layers:", rec.layers)
    for n in (3, 4, 5, 6, 7, 8, 9, 10, 12):
        split = drafter.even_split(n)
        print(f"{n} cards {split}: stair {drafter.split_depths(split)} -> "
              f"{[drafter.depth_name(x, rec.layers) for x in drafter.split_depths(split)]}")
    rng = random.Random(0)
    lens, newest = collections.Counter(), collections.Counter()
    for _ in range(5000):
        w = train.sample_general(rec.layers, rng)["sample"]
        d = [drafter.anchor_index(a) for a in w]
        assert d == sorted(d) and len(w) <= 12 and all(a == "emb" or int(a[1:]) in rec.layers for a in w), w
        lens[len(w)] += 1
        newest[w[0]] += 1
    print("window lengths:", sorted(lens.items()))
    print("newest anchor:", newest.most_common(8))
    for cards in ([6], [9], [6, 9]):   # specialists (--cards): only the standard splits' windows
        seen = collections.Counter(tuple(train.sample_general(rec.layers, rng, cards=cards)["sample"])
                                   for _ in range(2000))
        print(f"cards {cards}: {len(seen)} distinct windows:", ", ".join(f"{list(w)} {c}" for w, c in seen.most_common()))

    vocab = np.load(args.vocab)
    emb, head_rows, norm_w = train.load_flint(args.flint, vocab, "cpu")
    emb = emb.float()
    t2d = train.t2d_table(vocab, emb.shape[0])
    sd = torch.load(args.head, map_location="cpu")
    sd = sd.get("model", sd)
    six = drafter.Drafter(emb, head_rows, norm_w)
    six.load_state_dict(sd, strict=False)
    gen = drafter.Drafter(emb, head_rows, norm_w, general=True)
    gen.load_state_dict(sd, strict=False)
    gen.init_general(sd)
    six.eval()
    gen.eval()
    prof = drafter.PROFILES["A"]
    b = train.Batch(rec, prof, t2d, min_target=0)
    d = train.to_device(b, "cpu", torch.float32)
    committed, ca, win, _ = train.build_inputs(d, prof, prof["windows"], rec.n_embd, torch.float32)
    with torch.no_grad():
        o1 = six(d["tokens"], committed, ca, win)
        o2 = gen(d["tokens"], committed, ca, win)
    for name in o1:
        print(f"general vs six-card head, window {name}: max |diff| {float((o1[name] - o2[name]).abs().max()):.2e}")


if __name__ == "__main__":
    main()
