# Q223: a Pascal Q2_0 matrix-vector kernel for Qwen3.8-Flash-Next's experts (2026-11-03)

The Swift Q2_0 file of Qwen3.8-Flash-Next stores its experts in Q2_0: 18 bytes per 64 weights, one FP16 scale and
2-bit codes for −1, 0, 1, 2. Each of the 48 layers runs 10 of 512 routed experts and a shared expert, all 640 wide, on a
hidden width of 2,560. In R13's profile of the file on nine P106-090 cards
([Q215](../2026-11-03_flash_next_baseline/README.md), `decode_profile_v3.txt`) the Q2_0 products take ~28 of the 136
ms of GPU time per decoded token, one call of each launch per layer:

| Launch | K → N | MB per call | Calls per token | ms per token (R13) | per call | GB/s |
|---|---|---|---|---|---|---|
| routed gate + up + SwiGLU (`MUL_MAT_ID` × 2, fused; in some layers with the shared expert's gate/up in the launch) | 2,560 → 640, 10 experts | 9.22 (10.1 with the shared expert) | 48 | 14.9 (12.9 + 2.0) | 0.31 ms | ~30 |
| routed down (`MUL_MAT_ID`) | 640 → 2,560, 10 experts | 4.61 | 48 | 10.9 | 0.23 ms | ~20 |
| shared expert's down | 640 → 2,560 | 0.46 | 48 | 0.6 | 12 µs | ~39 |

That is ~11 ps per routed weight (2.36 G per token), where the card's ~173 GB/s would allow ~1.6. This folder holds the correctness checks of the
kernel that replaces llama.cpp's `mul_mat_vec_q` for Q2_0 on compute 6.1. The registry row is Q223 in
[quants/experiments.md](../../../quants/experiments.md); the timing on a P106 is
[bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md#flash-nexts-q2_0-experts-q223) `--q2_0` and
`test-backend-ops perf` (below).

## Why the stock kernel is slow

On compute 6.1 llama.cpp runs Q2_0 with `mul_mat_vec_q`. A Q2_0 row of 2,560 or 640 weights has fewer blocks (40, 10)
than one pass of a 4-warp block covers (64), so it takes the small-K shape: 4 warps and 4 rows per block, each thread one
32-weight half block. On the gate/up rows 80 of a block's 128 threads hold work, on the down rows 20; each loads its
8 bytes of codes as four 16-bit loads, and the four warps' partial sums meet in shared memory behind a
`__syncthreads`. The down projection launches 6,400 such blocks per call, each reading 720 bytes of weights.

## The kernel

Q2_0 joins the Pascal matrix-vector kernels of `ggml/src/ggml-cuda/mmvq-pascal.cu` (the Q4_0 to Q6_K family) on Miner
Llama's branch `pascal-p106-v3-q2_0mv` (off `pascal-p106-v3`), so it shares their work split, MUL_MAT_ID handling, fused
gate/up with every GLU, the routed + shared expert launch and the 2–4 token kernel:

- A warp is cut into lane groups of 2^l lanes sized to the row; a lane group owns R rows (twice that with a fused gate)
  that share every activation load. Each lane takes two blocks (36 bytes, 128 weights) per iteration, so its chunks
  stay 4-byte aligned although a block is 18 bytes.
- The codes are looked up with `__byte_perm` in a 4-byte table, `0x020100FF` (codes 0..3 → −1..2), with the code word
  itself as the selector: 4 lookups and 3 shifts per 16 weights, four `dp4a` against the activations, no masks, no
  offsets and no activation sums. The lookups return weights (0, 2, 4, 6) and (1, 3, 5, 7) of each run of 8, so the
  int8 activations are split into even and odd bytes once per load (8 byte permutes per 32 activations) and shared by
  all rows of the lane group. The block's sum is `d2 * (d8_0 * isum_0 + d8_1 * isum_1)` in FP32.
- Gate/up, K = 2,560: 4 lanes per row, R = 2, 5 iterations; each lane group stages its rows' 144-byte segments through
  shared memory with coalesced 128-bit loads before every iteration.
- Down, K = 640: 180-byte rows are only 4-byte aligned, so segment staging cannot use 128-bit loads, and direct loads
  would have each lane read its own row 4 bytes at a time. Instead each warp copies its 32 rows (contiguous: 5,760
  bytes) into dynamic shared memory once with coalesced 128-bit loads (tile staging), then every lane, one per row,
  reads its 36-byte chunks from there (a 45-word stride, free of bank conflicts). 4 warps per block use 23 KB.
- `GGML_CUDA_MMVQ_PASCAL=0` turns all Pascal kernels off, `2` runs them on any GPU (tests).
  `GGML_CUDA_MMVQ_PASCAL_Q2_0=G,P,T` picks Q2_0's rows per lane group with a fused gate (G) and without (P), 1 or 2, and
  its tile staging T: 0 off (direct loads), 1 where segments can't be staged (default), 2 wherever it fits.

Not measured on a P106 yet; the launch shapes come from reasoning and the bench's alternatives settle them.

## Correctness

On an RTX 3090 running the build's compute 6.1 PTX through the driver's JIT ([checks.tsv](checks.tsv)); no timings.
test-backend-ops compares each product with the CPU backend (normalized mean squared error below 5e-4).

| Run | Cases | Result |
|---|---|---|
| Q2_0 cases, Pascal kernels off (`GGML_CUDA_MMVQ_PASCAL=0`) | 331 | all pass (the 8 large-activation cases need the Pascal kernels' FP32 sums and report "not supported" without them) |
| Q2_0 cases, Pascal kernels on, the fork's launch shapes | 339 | all pass |
| the same at each of the 12 shapes of `GGML_CUDA_MMVQ_PASCAL_Q2_0` (R 1 or 2 with and without a gate, tile staging off, default, everywhere) and with staged activations | 339 each | all pass |
| every type's MUL_MAT cases, Pascal kernels on | 2,805 | all pass |
| every type's MUL_MAT_ID, MUL_MAT_ID_SHARED and MUL_MAT_VEC_FUSION cases of 1–4 tokens, Pascal kernels on | 500, 36, 1,077 | all pass |
| mutation build (one of the four lookups per 16 weights dropped), on, at the default and two forced shapes | 339 | 209 fail, the cases the kernel takes: every Q2_0 product of 1–4 tokens whose rows hold an even number of blocks; the 130 that pass are batches of 5 or more tokens and rows of 1, 3 or 5 blocks, left to llama.cpp's kernels |
| mutation build, off | 331 | all pass |
| `bench/mmvq_pascal --q2_0 --check-only`: `llama`, `pascal` and every `rR_*` variant at the four shapes, against double precision | 24 | all pass, worst 0.003 of the tolerance ([bench_check.txt](bench_check.txt)) |

The Q2_0 cases are Flash-Next's shapes at 1–4 tokens (routed gate/up fused, down, both with 32 and 512 experts; the
shared expert plain, fused and with a bias), activations whose blocks sum past FP16's range, the routed + shared expert
launch, tokens that share experts, odd and partial row counts, rows of 1,280 and 768 weights, row views, and
llama.cpp's own Q2_0 cases. With more than 4 tokens, test-backend-ops aborts on the 3090 with "CUDA error: invalid
argument" at a launch of llama.cpp's kernels, with the Pascal kernels off as well (`big0`): the host picks the launch for
compute 8.6 while the code is compiled for 6.1 (registry Q556); those batches never reach the Pascal kernels.

Expected on a P106: gate/up and down at 3–6 ps per weight would take the routed experts' 25.7 ms per token (R13's
12.9 + 2.0 + 10.9) to ~7–14 ms, at 47–94 GB/s.

## Files

| File | What |
|---|---|
| [run/check.sh](run/check.sh) | the test-backend-ops runs: the Q2_0 cases off, on, with staged activations and at each of the 12 launch shapes, every type's product cases, the larger batches with the Pascal kernels off, the mutation build |
| [run/q2_0-mut.diff](run/q2_0-mut.diff) | the mutation (drops the last of the four code lookups per 16 weights) built with `PATCHES=` to prove the cases reach the kernel |
| [checks.tsv](checks.tsv) | one line per run: build, run, cases passed, failed, exit status |
| [bench_check.txt](bench_check.txt) | `bench/mmvq_pascal --q2_0 --check-only` through the compute 6.1 PTX: every variant at every shape against the double-precision reference |

## Commands

Build (any x86-64 Linux PC with Docker; the tarball runs on the rig):

```bash
WORK=$WORK TESTS=1 JOBS=4 scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-q2_0mv
# the mutation build, for the checks only: the patch copied to $WORK/patches/ first
WORK=$WORK TESTS=1 JOBS=4 PATCHES=patches/q2_0-mut.diff scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-q2_0mv
bench/mmvq_pascal/build.sh
```

Correctness on a newer NVIDIA GPU through the compute 6.1 PTX:

```bash
BIN=$WORK/llama.cpp-20261103-7a9fb7cff-pascal-p106-v3-q2_0mv WORK=$WORK GPU=0 docs/data/2026-11-03_q2_0_kernel/run/check.sh
BIN=$WORK/llama.cpp-20261103-7a9fb7cff-pascal-p106-v3-q2_0mv-q2_0-mut WORK=$WORK GPU=0 TAG=mut- RUNS="off on sa s2_1_0 s2_2_2" docs/data/2026-11-03_q2_0_kernel/run/check.sh
bench/mmvq_pascal/build/mmvq_pascal --q2_0 --check-only --verbose
```

On a P106-090 (one card, the rig otherwise idle), `BIN` the unpacked tarball:

```bash
CUDA_VISIBLE_DEVICES=0 $BIN/bin/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED -p 'type_a=q2_0|type=q2_0'
CUDA_VISIBLE_DEVICES=0 /tmp/mmvq_pascal --q2_0 --csv /tmp/q2_0.csv
for m in 0 1; do CUDA_VISIBLE_DEVICES=0 GGML_CUDA_MMVQ_PASCAL=$m $BIN/bin/test-backend-ops perf -b CUDA0 -o MUL_MAT,MUL_MAT_ID -p type_a=q2_0; done
```

## Provenance and licence

Random weights and activations generated by the tools themselves; no model data. The kernel and the tests are part of
Miner Llama's llama.cpp fork (MIT, the ggml authors and contributors); the scripts and tables here are this
repository's licence.
