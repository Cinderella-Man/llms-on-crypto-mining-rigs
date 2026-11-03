# R7 / Q591: Flash-Next's routed experts in a k-token pass, each distinct expert read once (2026-11-04)

A speculative verify, a DFlash2 block or a pipelined guess runs k = 2 to 8 tokens through one pass of
Qwen3.8-Flash-Next, and each token picks 10 of the 512 routed experts of every layer. On llama.cpp's kernels every
(token, slot) entry reads its expert's weights again, so on nine P106-090 cards the routed experts of a pass cost
27.2 ms of GPU time at k = 1 and 70.7 ms at k = 5 ([Q583](../2026-11-03_verify_cost/README.md)). Miner Llama's
one-token Q2_0 kernel (Q223) brought one token down to 6.96 ms ([R16](../2026-11-03_kernel_timing/README.md)), but at
2–4 tokens inside ggml its gate/up product was slower than llama.cpp's (Q610), and 5–8 tokens ran on llama.cpp's
kernel. This is the experts' half of Q591 (the dense products are C70's): a kernel whose cost follows the number of
distinct experts a pass touches, not k × 10. The registry row is Q591 in [quants/experiments.md](../../../quants/experiments.md).

**Status: written, built and statically checked; not yet run on any GPU** (all seven GPUs of the build machine were
busy and the rigs off). The commands for the correctness run and the P106 timing are below.

## The kernel

`mul_mat_vec_q2_0_pascal_mtok` in `ggml/src/ggml-cuda/mmvq-pascal-mtok.cu` of the fork, branch
`pascal-p106-v3-b12076-mtok` (commit `c70687382`, one commit on `pascal-p106-v3-b12076-ladder` `98bbd6a2f`, which holds
the Q223 Q2_0 kernel and the Q546 BF16 kernel). It takes every Q2_0 `MUL_MAT_ID` of 2 to 8 tokens on compute 6.1,
ahead of the earlier multi-token Pascal kernel (2–4 tokens) and llama.cpp's `mul_mat_vec_q_moe` (5–8):

- **Groups.** Block y belongs to entry y of the pass's (token, slot) entries. Among the entries that picked one expert,
  in entry order, every gcap-th leads a group (gcap = k by default, so every distinct expert is one group): its block
  serves it and the next gcap − 1 entries of that expert, and every other block exits at once. Each warp finds its
  block's group itself with 8 ballots over the ids (one load per lane and 32 entries), with no shared memory and no
  barrier. The earlier kernel walked the ids entry by entry in every thread.
- **Activations once per block.** The block stages its members' Q8_1 activations in shared memory: each block's int8
  values regrouped into even and odd bytes as the one-token kernel regroups them, in 32-byte runs with 16 bytes of
  padding per lane chunk (the four lanes of a gate/up row read different chunks and hit different banks), the FP32
  scales beside them. A member's 32 weights then cost two 128-bit shared loads, shared by the gate and up rows.
- **Weights once per group.** Each Q2_0 code word is looked up once (4 `__byte_perm` and 3 shifts per 16 weights) and
  multiplied by every member (4 `dp4a` per 16 weights per member); the one-token kernel repeats the lookups per token.
  Lane groups, two blocks per lane and the weight staging are the one-token kernel's (gate/up: 4 lanes per row, row
  segments staged with 128-bit loads; down: one lane per 180-byte row, each warp staging its 32 rows at once), with one
  row per lane group: the members' accumulators take the registers of the one-token kernel's second row.
- **Row blocks per block.** A block loops over 4 consecutive row blocks of its expert (32 rows each for gate/up, 128 for
  down), so the group search and the staging serve 128 or 512 rows.
- **Fusions as before:** gate/up fused with SwiGLU (and the other GLU operations), biases, and the routed + shared
  expert launch (one more block row serves all tokens with the shared expert). The weighted sum over the slots stays
  the separate `moe_weighted_reduction` launch, as in the one-token path.
- **Bit-identical to the one-token kernel:** each token's integer sums, FP32 operations and the reduction over a row's
  lanes are those of `mul_mat_vec_q_pascal`, so the k-token pass computes exactly what k one-token passes compute.

`GGML_CUDA_MMVQ_PASCAL_MTOK=E,CAP,RB,G` (missing fields and 0 keep the defaults): `E` 0 turns it off (the earlier
kernels, for an A/B on one binary); `CAP` entries per group (1: no sharing, the kernel's own cost per entry); `RB` row
blocks per block; `G` 1 loads the gate/up weights directly instead of staging them. `GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0`
sends Q2_0 to llama.cpp's kernels at every token count, as before.

## Routing: how much a pass can share

No routing trace of Flash-Next exists yet, so the bench uses route profiles. With uniform independent routing, k tokens
touch 512 × (1 − (1 − 10/512)^k) distinct experts: 19.8, 38.8, 47.7 and 74.7 at k = 2, 4, 5 and 8, so sharing would save
1–7 %. Flint's measured routing (Q51) shared more, 74 % of the slots distinct at 8 consecutive tokens: the bench's
`local` profile assumes the same for Flash-Next (popularity skewed so that a quarter of the experts take 44 % of the
picks, each expert of the token before kept with probability 0.16). `copy` keeps 0.6 (repetitive text, a bound), and
`--route-trace` replays a real trace once one is dumped (Q625).

## Expected

Per token at k = 1 the one-token kernel reads the routed experts in 6.06 ms (R16: 2.13 and 3.45 ps per weight for
gate/up and down, 48 layers; 6.96 ms with the shared expert). If the multi-token kernel keeps that speed per distinct
expert, a k-token pass costs ≈ d / 10 × 6.06 ms for d distinct experts, plus 4 `dp4a` per 16 weights for each further
member (about a third of a one-token product's instructions, partly hidden under the memory time of gate/up):

| k | d (uniform) | est. ms (uniform) | d (`local`, est.) | est. ms (`local`) | Q583, llama.cpp's kernel |
|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 6.1 | 10 | 6.1 | 27.2 |
| 2 | 19.8 | ~12 | ~19 | ~12 | 29.9 |
| 3 | 29.4 | ~18 | ~27 | ~17 | 44.3 |
| 4 | 38.8 | ~24 | ~34 | ~21 | 58.3 |
| 5 | 47.7 | ~29 | ~41 | ~26 | 70.7 |
| 8 | 74.7 | ~46 | ~59 | ~38 | — |

So the kernel can take the routed experts of a 5-token pass from 70.7 ms to an estimated ~26–29 ms, but with 512
fine-grained experts they stay roughly 4–5× a token's, because the tokens of a pass share few experts; on top of a dense
part that C70 flattens, that is est. +20–23 ms per 5-token pass. The bench's `pertok` and `mtok_c1` rows separate the
two effects: the per-token speed (one-token kernel launched per token) and the sharing.

## Files

| File | What it holds |
|---|---|
| [registers.tsv](registers.tsv) | registers, spills and static shared memory per instantiation on sm_61: the fork's (from the test-backend-ops build) and the bench's copy (ptxas) |
| [run/p106_checks.sh](run/p106_checks.sh) | the run on one P106: test-backend-ops at every `GGML_CUDA_MMVQ_PASCAL_MTOK` setting and off, `--q2_0-mtok`, and the perf A/B |

The bench: [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md#flash-nexts-experts-in-a-k-token-pass-r7-q591)
`--q2_0-mtok` (`src/q2_0_mtok.cuh`).

## Checks so far

- Builds: the fork's rig build (`TESTS=1`, sm_61 + compute_61 PTX, CUDA 12.9 in Docker) and the bench (sm_61,
  compute_61 PTX, sm_120) compile without warnings in the new files. Register use per instantiation is in
  `registers.tsv`: 76–168 registers, no spills in the fork; the bench's copy of the fused gate/up for 8 members spills
  4 bytes at the 168-register cap, and 44 bytes when compiled for 4 blocks per SM (`mtok_mb4`, at most 128 registers;
  every other instantiation fits without spills).
- Not run yet: everything on a GPU.
- test-backend-ops cases added (eval): 5 to 8 tokens of the existing Q2_0 expert cases (routed gate/up and down, fused
  gate/up + SwiGLU, large activations, the routed + shared launch, repeated ids within a token), and at 2 to 8 tokens
  routing where consecutive tokens share experts (`init_mul_mat_id_ids_overlap`: one expert picked by every token, each
  other expert kept from the token before with probability 0.5 or 1.0, so groups of 1 to k entries): fused gate/up
  with and without a bias, one gate or up projection, the down projection, partial row blocks staged by segment (641
  rows) and by tile (2,564), two-lane tiles (K = 768), a fused gate with direct loads (K = 1,280), and all 512 experts
  at 3, 5 and 8 tokens. Perf: the routed products at 5–8 tokens and with overlapping routing (keep 0.1 and 0.5) at
  2–8 tokens.

## Commands

The rig build and the bench, on any x86-64 machine with Docker:

```bash
WORK=<work dir> TESTS=1 JOBS=6 scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-b12076-mtok
bench/mmvq_pascal/build.sh
```

Correctness on a newer card through the compute_61 PTX (no timings):

```bash
GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=1 <build>/bin/test-backend-ops -b CUDA0 \
  -o MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED -p 'type_a=q2_0|type=q2_0'
CUDA_FORCE_PTX_JIT=1 bench/mmvq_pascal/build/mmvq_pascal --q2_0-mtok --check-only
```

On one P106 (the rig otherwise idle; ~30 min, most of it the correctness runs):

```bash
ssh $RIG "BIN=<build>/bin BENCH=<dir>/mmvq_pascal OUT=<work dir>/out setsid nohup bash <work dir>/run/p106_checks.sh \
  > <work dir>/p106.out 2>&1 < /dev/null &"
```

In the model, the k-token passes of Q583 on this build with the kernel on and off (its `run/verify_cost.sh`, the
profile and wall steps, k = 1 to 8):

```bash
ssh $RIG "BIN=<build>/bin MODEL=<Swift Q2_0 first shard> DRAFT=<draft> OUT=<work dir>/on/out STEPS='prof wall' KS=1,2,3,4,5,6,7,8 \
  setsid nohup bash <work dir>/run/verify_cost.sh > <work dir>/on/verify_cost.out 2>&1 < /dev/null &"
ssh $RIG "GGML_CUDA_MMVQ_PASCAL_MTOK=0 BIN=<build>/bin MODEL=<…> DRAFT=<…> OUT=<work dir>/off/out STEPS='prof wall' KS=1,2,3,4,5,6,7,8 \
  setsid nohup bash <work dir>/run/verify_cost.sh > <work dir>/off/verify_cost.out 2>&1 < /dev/null &"
python3 run/verify_split.py <work dir>/on/out; python3 run/verify_split.py <work dir>/off/out   # on another machine
```

## Provenance and licence

Written and built on 2026-11-03/04 on the build machine (CPU only). The kernel is part of Miner Llama's fork of
llama.cpp (MIT). No model data; the tables hold compiler output.
