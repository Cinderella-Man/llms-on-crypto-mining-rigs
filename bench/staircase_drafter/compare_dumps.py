"""Compare two llama-staircase-dump files record by record (same seq_ids): the top-20 tokens and log-probabilities, and
whether the stored hidden states agree. Used to check a change to the dump tool (e.g. DUMP_GPU_TOPK) against its
default path on the same sequences.

usage: python compare_dumps.py A.bin B.bin
"""
import sys

import numpy as np

import feats


def main():
    fa, fb = feats.FeatFile(sys.argv[1]), feats.FeatFile(sys.argv[2])
    rb = {fb[i].seq_id: fb[i] for i in range(len(fb))}
    for i in range(len(fa)):
        a = fa[i]
        b = rb.get(a.seq_id)
        if b is None:
            continue
        top1 = (a.top_ids[:, 0] == b.top_ids[:, 0]).mean()
        same_set = np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a.top_ids, b.top_ids)])
        d_lp = np.abs(a.top_lp[:, 0] - b.top_lp[:, 0])
        full = np.abs(a.dequant("full") - b.dequant("full")).max()
        lay = max(float(np.abs(a.dequant(L) - b.dequant(L)).max()) for L in a.layers if L in b.layers)
        print(f"seq {a.seq_id}: {a.n_gen} positions; top-1 same {top1:.3f}, top-20 overlap {same_set:.3f}, "
              f"|dlogp top-1| median {np.median(d_lp):.2e} max {d_lp.max():.2e}; max |d| final state {full:.2e}, "
              f"layers {lay:.2e}")


if __name__ == "__main__":
    main()
