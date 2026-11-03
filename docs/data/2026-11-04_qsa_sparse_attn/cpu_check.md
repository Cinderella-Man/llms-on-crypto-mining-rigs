# The sparse decode attention's CPU check (bench/fattn_sparse, 2026-11-04)

`bench/fattn_sparse/check.sh` on Miner Llama `pascal-p106-v3-b12076-qsa` `bdb902d95`, on a build machine's CPU in
`nvidia/cuda:12.9.2-devel-ubuntu24.04` (no GPU). The host model of the kernel (the header's own decoding, split,
merge and sink functions, the warp's shuffles stepped lane by lane) against an FP64 softmax over each query's listed
cells and against ggml's CPU `FLASH_ATTN_EXT` over the whole masked cache. Split counts as on a 5-SM P106-090.

| Case | Cache | Splits | Listed cells (row 0) | Max error / max output vs FP64 | NMSE vs ggml CPU | |
|---|---|---:|---:|---:|---:|---|
| Flash-Next decode, 8,192 cells | F16 | 10 | 2,051 | 3.57e-07 | 1.21e-05 | pass |
| Flash-Next verify, 5 tokens | F16 | 2 | 2,051 | 5.83e-08 | 1.32e-07 | pass |
| Flash-Next decode, 32,768 cells | F16 | 10 | 2,051 | 3.85e-07 | 1.11e-05 | pass |
| Flash-Next, 1 split | F16 | 1 | 2,051 | 9.52e-07 | 1.06e-05 | pass |
| Flash-Next, 2 tokens, 32 splits | F16 | 32 | 2,051 | 1.57e-08 | 3.55e-08 | pass |
| two sequences, 3 tokens (one list empty) | F16 | 2 | 2,051 | 6.29e-08 | 1.35e-07 | pass |
| head 128, 4 × 8 heads, sinks | F16 | 5 | 1,027 | 3.15e-07 | 5.66e-06 | pass |
| head 128, 2 × 4 heads, 7 tokens | F16 | 2 | 333 | 7.84e-08 | 2.15e-07 | pass |
| 1 × 16 heads, list shorter than a tile | F16 | 1 | 31 | 2.60e-07 | 2.40e-07 | pass |
| one tile, 2 tokens | F16 | 1 | 32 | 9.91e-08 | 3.43e-08 | pass |
| Flash-Next decode, 8,192 cells | Q8_0 | 10 | 2,051 | 4.16e-07 | 6.69e-06 | pass |
| Flash-Next verify, 5 tokens | Q8_0 | 2 | 2,051 | 6.51e-08 | 1.90e-08 | pass |
| Flash-Next decode, 32,768 cells | Q8_0 | 10 | 2,051 | 3.67e-07 | 6.74e-06 | pass |
| Flash-Next, 1 split | Q8_0 | 1 | 2,051 | 9.50e-07 | 6.14e-06 | pass |
| Flash-Next, 2 tokens, 32 splits | Q8_0 | 32 | 2,051 | 1.89e-08 | 4.91e-09 | pass |
| two sequences, 3 tokens (one list empty) | Q8_0 | 2 | 2,051 | 5.47e-08 | 1.99e-08 | pass |
| head 128, 4 × 8 heads, sinks | Q8_0 | 5 | 1,027 | 3.04e-07 | 6.19e-06 | pass |
| head 128, 2 × 4 heads, 7 tokens | Q8_0 | 2 | 333 | 5.35e-08 | 1.94e-07 | pass |
| 1 × 16 heads, list shorter than a tile | Q8_0 | 1 | 31 | 2.63e-07 | 5.37e-06 | pass |
| one tile, 2 tokens | Q8_0 | 1 | 32 | 1.20e-07 | 2.22e-07 | pass |

Pass: within 2e-5 of FP64 and NMSE below 5e-4 (test-backend-ops' FLASH_ATTN_EXT tolerance) against ggml. The NMSE
against ggml is larger at one token because ggml's CPU kernel accumulates V in F16 for an F16 cache and rounds the
query to the cache's type for the dot products; against FP64 every case is within 1e-6.

## The same check made to fail

| Run | Failing cases (of 20) |
|---|---|
| fault 1: output × 1.01 | 20 |
| fault 2: mask values ignored | 20 |
| fault 3: every split's last entry dropped | 20 |
| mutation: Q8_0 rows read without their 2-byte scale | 10 (exactly the Q8_0 cases) |
| mutation: the merge adds the splits' sums unscaled | 14 (all but the single-split cases) |
| mutation: every split but the first skips its first entry | 14 (all but the single-split cases) |
| mutation: scores mapped to the wrong key of the tile | 20 |

The mutation that leaves the splits' sums unscaled stays within ggml's tolerance on some cases (NMSE 7.8e-05 on the
5-token verify) and fails them on the FP64 comparison (2.1e-02), which is why the check keeps both references.
