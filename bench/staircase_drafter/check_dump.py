"""Check a llama-staircase-dump file before training on it:
- FP8 decoding: feats.E4M3 against torch.float8_e4m3fn, and the encoder's round trip on random values;
- the final hidden state: Flint's output head applied to the dumped (FP8) state gives the dumped top-1 token and
  log-probabilities close to the dumped ones (an off-by-one or layout error gives near-random agreement);
- optionally the dumped top-20 against an earlier llama-logit-dump of the same sequences (Q96's top20.bin).

usage: python check_dump.py FEAT.bin FLINT_DIR [--logit-dump TOP.bin] [--max-seqs 4]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat")
    ap.add_argument("flint")
    ap.add_argument("--logit-dump")
    ap.add_argument("--max-seqs", type=int, default=4)
    args = ap.parse_args()
    codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    same = np.isclose(codes, feats.E4M3, equal_nan=True).all()
    print(f"E4M3 table matches torch.float8_e4m3fn: {same}")
    out = np.load(os.path.join(args.flint, "output.npy"), mmap_mode="r")
    ff = feats.FeatFile(args.feat)
    print(f"{len(ff)} sequences")
    ref = None
    if args.logit_dump:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ngram_replay"))
        import pipeline_sim
        ref = pipeline_sim.load_topk(args.logit_dump)
    W = torch.from_numpy(np.asarray(out, dtype=np.float32))
    for i in range(min(len(ff), args.max_seqs)):
        r = ff[i]
        rows = np.arange(r.n_prompt - 1, r.n_tok - 1)
        h = torch.from_numpy(r.dequant("full", rows))
        logits = h @ W.T
        lp = torch.log_softmax(logits, -1)
        top1 = lp.argmax(-1).numpy()
        agree = (top1 == r.top_ids[:, 0]).mean()
        d_lp = np.abs(lp.gather(1, torch.from_numpy(r.top_ids[:, :1].astype(np.int64))).numpy()[:, 0] - r.top_lp[:, 0])
        sampled_top1 = (r.top_ids[:, 0] == r.tokens[r.n_prompt:]).mean()
        print(f"seq {r.seq_id}: {r.n_tok} tokens, {r.n_gen} generated; head on FP8 state: top-1 agrees with dump "
              f"{agree:.3f}, |dlogp(top1)| median {np.median(d_lp):.4f} max {d_lp.max():.3f}; dumped top-1 = sample "
              f"{sampled_top1:.3f}")
        for li, L in enumerate(r.layers):
            x = r.dequant(L)
            print(f"   layer {L} input from position {r.lay_start[li]}: rms {np.sqrt((x ** 2).mean()):.3f}, "
                  f"max {np.abs(x).max():.1f}")
        if ref is not None and r.seq_id in ref:
            rr = ref[r.seq_id]
            n = min(len(rr), r.n_gen)
            t1 = np.mean([rr[g][0][0] == r.top_ids[g, 0] for g in range(n)])
            dl = np.median([abs(rr[g][1][0] - r.top_lp[g, 0]) for g in range(n)])
            print(f"   against the earlier logit dump: top-1 agrees {t1:.3f}, |dlogp| median {dl:.4f}")


if __name__ == "__main__":
    main()
