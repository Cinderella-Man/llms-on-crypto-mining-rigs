# Q546: a Pascal BF16 matrix-vector kernel for Qwen3.8-Flash-Next (2026-11-03)

Qwen3.8-Flash-Next keeps its hyper-connection projections, its indexer's query projection and its recurrent layers'
alpha and beta in BF16. In R13's profile of Swift Q2_0 on nine P106-090 cards
([Q215](../2026-11-03_flash_next_baseline/README.md), `decode_profile_v3.txt`) they take 43 of the 136 ms of GPU time
per decoded token, 39 ms of it in four products per layer:

| Tensor | K → N | MB per call | Calls per token | ms per token (R13) | per call | GB/s |
|---|---|---|---|---|---|---|
| `hc_attn_up`, `hc_ffn_up` | 320 → 10,240 | 6.55 | 96 | 32.1 | 0.33 ms | ~20 |
| `hc_attn_down`, `hc_ffn_down` | 10,240 → 320 | 6.55 | 96 | 7.0 | 73 µs | ~90 |
| `hc_attn_inject`, `hc_ffn_inject` | 10,240 → 4 | 0.08 | 96 | 1.3 | 14 µs | ~6 |
| `indexer.q_proj` | 2,560 → 512 | 2.62 | 12 | 0.6 | 53 µs | ~50 |
| `ssm_alpha`, `ssm_beta` | 2,560 → 48 | 0.25 | 72 | 1.2 | 16 µs | ~15 |

This folder holds the correctness checks of the kernel that replaces llama.cpp's `mul_mat_vec_f` for them on compute
6.1. The registry row is Q546 in [quants/experiments.md](../../../quants/experiments.md); the timing on a P106 is
[bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md#flash-nexts-bf16-products-q546) `--bf16`.

## Why the stock kernel is slow

llama.cpp picks `mul_mat_vec_f` for BF16 weights at 1 to 8 tokens on compute 6.1 (Pascal has no BF16 or FP16 tensor
cores, so `ggml_cuda_should_use_mmvf` says yes and cuBLAS is never reached). It launches one block per row and sizes the
block to the row: for K = 320 that is 160 threads, each reading one 4-byte pair of weights once, then a reduction
over its 5 warps through shared memory. 10,240 such blocks of 640 bytes each, at most 12 resident per SM
(thread-limited), keep ~38 KB in flight across the card's 5 SMs: by Little's law ~20 GB/s at ~2 µs per block, what
the profile shows. The 10,240-wide rows of the down projections fare better (256 threads, 20 loads each) but still
read 4 bytes per load.

## The kernel

`ggml/src/ggml-cuda/mmvf-pascal.cu` on Miner Llama's branch `pascal-p106-v3-bf16mv` (off `pascal-p106-v3`):

- A row group of G lanes (8, 32, or 128 spanning the block's 4 warps) owns R rows (1, 2 or 4). Lane l reads chunks
  l, l + G, … of 8 weights with one coalesced 128-bit load each, and the 8 matching FP32 activations with two 128-bit
  loads that all R rows share. The loop is unrolled twice, so a lane has 2R weight loads in flight.
- Each weight is converted exactly by moving its 16 bits into the top half of an FP32 (`w << 16`, `w & 0xFFFF0000`)
  and added with one FMA in column order; lanes are reduced with shuffles, and for G = 128 the four warps through shared
  memory, always in the same order. The result equals an FP32 product up to summation order.
- The launch shape comes from the row width and the SM count: G is the smallest of 8, 32, 128 that leaves a lane at
  most 16 chunks; R starts at 4 (2 for 3 and 4 tokens) and halves while the grid has fewer than 4 blocks per SM; if one
  row per group still leaves too few blocks, G grows. On a P106-090: `hc_*_up` G = 8, R = 4 (160 blocks);
  `hc_*_down` G = 128, R = 4 (80); `hc_*_inject` G = 128, R = 1 (4); `indexer.q_proj` G = 32, R = 4 (32); alpha/beta
  G = 128, R = 1 (48).
- 1 to 4 tokens (a speculative-decoding verify step reads each weight once for all tokens), a fused bias for one token;
  fused gates, expert ids, batched weights and rows that are no multiple of 8 weights stay with `mul_mat_vec_f`.
- `GGML_CUDA_MMVF_PASCAL`: 0 off, 1 compute 6.1 only (default), 2 any NVIDIA GPU (testing).
  `GGML_CUDA_MMVF_PASCAL_NSM=n` picks the launch shapes of an n-SM card; `GGML_CUDA_MMVF_PASCAL_SHAPE=G,R` forces one.

## Result: correct, not yet timed

All checks ran on an RTX 3090 with the build's compute_61 PTX (JIT-compiled by the driver), the same device code the
P106's sm_61 machine code is compiled from; timings there say nothing about Pascal, so none were taken. The commit
is `a4acc80a3`, the binaries those of the rig tarball.

| Check | Result |
|---|---|
| test-backend-ops, the 265 BF16 cases (`MUL_MAT`, `MUL_MAT_VEC_FUSION`), kernel off | 265/265 |
| the same at the launch shapes a P106-090 picks (`GGML_CUDA_MMVF_PASCAL_NSM=5`) | 265/265 |
| the same at each of the nine forced launch shapes (G = 8, 32, 128 × R = 1, 2, 4) | 265/265 each |
| every `MUL_MAT` case of every type, kernel forced on with the 3090's own launch shapes | 2,741/2,741 |
| a mutation build that drops one weight of every 8: kernel off | 265/265 |
| the mutation, at the P106's launch shapes and at each forced shape | 207/265 each: the 58 cases the kernel takes fail (Flash-Next's shapes at 1–4 tokens, the 16-byte-aligned row views, the fused bias, and the suite's other 1–4-token BF16 products); the unaligned views, K = 100, gates and batched products pass on `mul_mat_vec_f` |
| `bench/mmvq_pascal --bf16 --check-only --tokens 1,2,3,4` against double precision (tolerance 1e-5 of the sum of absolute terms) | 190/190, worst 0.0044 of the tolerance ([bench_check.csv](bench_check.csv)) |

The BF16 cases cover Flash-Next's five shapes at 1 to 4 tokens, an odd row count (10,241), rows of 125 and 9 chunks,
a row width the kernel leaves to `mul_mat_vec_f` (K = 100), rows that are views of wider rows (16-byte aligned and
not), and one token with a fused bias, gate, or both.

**Expected on the rig:** the tuned quantized kernels reach 105–117 GB/s at 4–13 MB per call on a P106, which streams
at 179 GB/s ([2026-10-23_rig_step1](../2026-10-23_rig_step1/README.md)); BF16 needs fewer instructions per byte. At
100–130 GB/s the two 6.55 MB shapes take 50–66 µs a call (0.33 ms and 73 µs today), and with the small shapes at an
est. 4–6 µs (inject, alpha/beta) and 24–33 µs (indexer) the BF16 tensors cost est. 11–14 ms per token instead of 43:
−29 to −32 ms, so R13's 136 ms of GPU time per token becomes ~104–107 (7.4 → ~9.5 tok/s if the cards ran back to
back). The quantized kernels ran 13–33 µs slower per call in the model than in the benchmark
([Q99](../2026-10-28_q99_tlb/README.md)); a gap like that would add up to ~6 ms. FN17's
Q8_0 retype reads half the bytes and should be faster still, but changes the file; this kernel serves the files as
published.

**Running Ampere on the sm_61 PTX:** reliable for every case above. The full `MUL_MAT_VEC_FUSION` suite aborts with
"CUDA error: invalid argument" at `MUL_MAT_VEC_FUSION(type=q8_0, m=8, n=32, k=256, use_id=1, n_mats=16, n_used=8,
with_bias=1, with_gate=0)`, after 273 passing cases, in the unmodified v3 base (`37e717ff6`) as well: an 8-token
quantized expert product this kernel does not touch (Q556).

## Files

| File | What |
|---|---|
| `bench_check.csv` | `bench/mmvq_pascal --bf16 --check-only --tokens 1,2,3,4` on the RTX 3090: per shape, token count and variant the launch shape, registers (of the JIT-compiled code) and the worst error over the tolerance; timing columns are 0 |
| `run/check.sh` | the test-backend-ops runs above, one container each, on one GPU |

## How it was made

On 2026-11-03, on a 7-GPU machine (RTX 3090 used), in `nvidia/cuda:12.9.2-devel-ubuntu24.04`. From the repository
root, with `WORK` a work folder on the build machine:

```bash
# the build (sm_61 and compute_61 PTX), with test-backend-ops
WORK=$WORK TESTS=1 JOBS=8 scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-bf16mv
# the correctness runs on GPU n
BIN=$WORK/llama.cpp-20261103-a4acc80a3-pascal-p106-v3-bf16mv WORK=$WORK GPU=n docs/data/2026-11-03_bf16_kernel/run/check.sh
# the mutation: drop the last FMA of mmvfp_dot8 in ggml/src/ggml-cuda/mmvf-pascal-impl.cuh, save the diff as
# $WORK/patches/mut.diff, build with PATCHES=patches/mut.diff and run every run but "on" on that build
BIN=$WORK/llama.cpp-20261103-a4acc80a3-pascal-p106-v3-bf16mv-mut WORK=$WORK GPU=n TAG=mut- \
  RUNS="off nsm5 g8_r1 g8_r2 g8_r4 g32_r1 g32_r2 g32_r4 g128_r1 g128_r2 g128_r4" docs/data/2026-11-03_bf16_kernel/run/check.sh
# the benchmark's own check
bench/mmvq_pascal/build.sh
bench/mmvq_pascal/build/mmvq_pascal --bf16 --check-only --tokens 1,2,3,4 --csv bench_check.csv
```

The timing on a P106 (one card, minutes) is the benchmark's `--bf16` run without `--check-only`, as in
[its README](../../../bench/mmvq_pascal/README.md#flash-nexts-bf16-products-q546), and `test-backend-ops perf -o MUL_MAT
-p type_a=bf16` with `GGML_CUDA_MMVF_PASCAL=0` and `=1` from the rig tarball. The weights are random; no model file
or text is involved.
