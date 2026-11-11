#!/usr/bin/env python3
"""shapes.py: the launch shape Miner Llama's Pascal BF16 kernel picks for Flash-Next's BF16 products, per token count.

A copy of mmvfp_pick_shape (ggml/src/ggml-cuda/mmvf-pascal.cu, at the defaults: GGML_CUDA_MMVF_PASCAL_RULE=2, no
GGML_CUDA_MMVF_PASCAL_SHAPE) for 128-thread blocks. Prints, per shape, SM count and token count, G (lanes per row
group), R (rows per group), the grid's blocks, and whether G is the one-token shape's: a product's sums depend on G
only, so the same G gives each token the one-token kernel's sums bit for bit.

Usage: python3 shapes.py [nsm ...]   (default: 5, the P106-090)
"""
import sys

NT = 128
SHAPES = [  # name, K (row width), N (rows)
    ("hc_up", 320, 10240),
    ("hc_down", 10240, 320),
    ("hc_inject", 10240, 4),
    ("indexer_q", 2560, 512),
    ("ssm_ab", 2560, 48),
]


def blocks(nrows, G, R):
    rows_per_block = (NT // G) * R
    return (nrows + rows_per_block - 1) // rows_per_block


def pick(nchunks, nrows, ncols, nsm, rule=2):
    rmax = (4 if ncols <= 2 else 2) if rule == 0 else (4 if ncols <= 4 else 8)
    target = 4 * nsm
    mid = rule >= 2 and ncols in (3, 4)
    G = 8
    while G < (32 if mid else 128) and nchunks > 16 * G:
        G *= 4
    R = rmax
    if mid:
        while G < 128 and blocks(nrows, G, R) < target and nchunks >= 8 * G:
            G *= 4
    while R > 1 and blocks(nrows, G, R) < target:
        R //= 2
    while G < 128 and blocks(nrows, G, R) < target and nchunks >= 8 * G:
        G *= 4
    return G, R


def main():
    nsms = [int(a) for a in sys.argv[1:]] or [5]
    print("shape\tK\tN\tSMs\ttokens\tG\tR\tblocks\tsame G as one token")
    for name, K, N in SHAPES:
        for nsm in nsms:
            g1, _ = pick(K // 8, N, 1, nsm)
            for t in range(1, 9):
                G, R = pick(K // 8, N, t, nsm)
                print(f"{name}\t{K}\t{N}\t{nsm}\t{t}\t{G}\t{R}\t{blocks(N, G, R)}\t{'yes' if G == g1 else 'no'}")


if __name__ == "__main__":
    main()
