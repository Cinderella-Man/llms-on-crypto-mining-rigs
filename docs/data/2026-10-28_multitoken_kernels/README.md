# Expert kernels that read and decode each weight once for 2–4 tokens (2026-10-28)

Registry row Q211. Ornith-1.5's built-in MTP layer is right 0.85–0.91 of the time on its first draft and 0.74 / 0.67 on
the next two ([2026-10-28_ornith_mtp_accept](../2026-10-28_ornith_mtp_accept/README.md), Q341, Q366), so llama.cpp's
draft-and-verify (`llama-server --spec-type draft-mtp`) should pay on the rig: two drafts, then one 3-token step of the
model. It did not on v2.0 (8.8 → 7.9 tok/s, [the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md)),
because the Pascal kernels ran 2–4-token products slower than one token per weight read
([2026-10-28_lowbit_rig](../2026-10-28_lowbit_rig/README.md), [2026-10-27_iq_kernels](../2026-10-27_iq_kernels/README.md)):

- **Expert products cost 2–3× more per distinct expert at 2–4 tokens than at one, in every format, Q8_0 included,**
  which needs no decoding: an N2P expert down product at 2 tokens took 660 µs, against ~220 µs for its ~15 distinct
  experts at the one-token rate. The earlier multi-token kernel scanned the step's expert ids entry by entry in every
  block, so its overhead grew with the square of the entries.
- **Dense products repeated the decoding per token:** 1.86× one token at 2 tokens for N2P, 1.72× for IQ2_S, against
  1.19× for Q8_0, whose weights need no decoding.

Fork branch `pascal-p106-v3-mtpk` (off `pascal-p106-v3`, with the IQ2_S/IQ3_XXS kernels of `pascal-p106-v3-iqmmvq` and
the N2P/N3P types of `pascal-p106-v3-lowbit` merged) adds `mul_mat_vec_q_pascal_mt` for the types of Ornith's files:
IQ2_S, IQ3_XXS, N2P, N3P, Q8_0 and Q5_0 (the design is at the top of `ggml/src/ggml-cuda/mmvq-pascal.cu`):

- **One block per group of tokens that picked the same expert,** found by each warp from the step's expert ids with
  shuffles and ballots (one load of ids per lane); blocks past the last group exit at once.
- **The block's tokens' activations staged once in shared memory,** padded against bank conflicts, in tiles where a
  row needs more than 16 KB; where a row fits, a block handles several row blocks, so the staging and the IQ grid
  tables serve more rows.
- **Each 32-weight sub-block decoded once** into signed int8 words and multiplied by every token's activations; the
  token count (1–4) is a compile-time constant of the inner loop.
- **Every token's result is bit-identical to the one-token kernel's:** the same integer sums and the same FP32
  operations in the same order. `GGML_CUDA_MMVQ_PASCAL_MT=0` restores the earlier kernel.

## Results so far

- **Correct** natively (V100), through the compute 6.1 PTX the P106 runs, and on a P106 (its small cases: MUL_MAT
  513/513, MUL_MAT_ID 204/204), every 2–4-token product bit-identical to the single-token products of its columns:
  [correctness.md](correctness.md).
- **Kernel time on one P106** (rig card 0, `run/rig_kernels.sh`, [kernels.tsv](kernels.tsv)): the new kernel (MT), the
  earlier one (MC) and llama.cpp's kernel at 1–4 tokens, all three running the same one-token Pascal kernel at 1 token.
  Expert products read 8 of 64 random experts per token, so 3 tokens touch ~21 distinct experts: **2.6× one token is
  the floor** for a kernel that reads each distinct expert once at the one-token rate. Cost at 3 tokens relative to 1:

  | Type | Product | MT | MC | llama.cpp |
  |---|---|---|---|---|
  | N2P | fused gate/up 3072 → 1024 | **2.60** | 3.64 | 2.64 |
  | N2P | expert down 1024 → 3072 | **3.67** | 9.76 | 3.71 |
  | N2P | dense 3072 → 8192 | **1.57** | 2.72 | 2.42 |
  | N3P | fused gate/up | 2.57 | 3.46 | **2.47** |
  | N3P | expert down | **3.83** | 10.85 | 3.96 |
  | N3P | dense 3072 → 8192 | **1.61** | 2.35 | 2.53 |
  | IQ2_S | fused gate/up | 5.32 | 4.65 | **3.46** |
  | IQ2_S | expert down | **2.81** | 5.23 | 2.83 |
  | IQ2_S | dense 3072 → 8192 | **1.34** | 2.00 | 2.01 |
  | IQ3_XXS | fused gate/up | 4.15 | 5.18 | **3.73** |
  | IQ3_XXS | expert down | 4.41 | 6.59 | **4.37** |
  | IQ3_XXS | dense 3072 → 8192 | **1.67** | 2.23 | 2.40 |
  | Q5_0 | dense 3072 → 8192 | **1.85** | 2.03 | 2.29 |
  | Q5_0 | dense 8192 → 3072 | 2.56 | 2.17 | 2.25 |
  | Q5_0 | output layer 3072 → 248,320 | **1.96** | 2.19 | 2.48 |
  | Q5_0 | expert down | 5.37 | 10.07 | **4.08** |
  | Q8_0 | dense 3072 → 8192 | **1.33** | 1.38 | 1.84 |
  | Q8_0 | dense 8192 → 3072 | 1.79 | **1.47** | 1.96 |
  | Q8_0 | expert down | 6.33 | 6.23 | **3.65** |

  - **The types of the N2P/N3P file run best on the new kernel, or level with llama.cpp's:** the fused gate/up sits at
    the 2.6× floor, the down product ~1.4× above it, dense products at 1.6× of one token (the earlier kernel 2.4–2.7×).
    That file's dense Q5_0 tensors and its Q5_0 output layer are fastest on it too.
  - **Worse than llama.cpp's kernel:** the Q8_0 and Q5_0 expert products (5.4–6.3× against 3.7–4.1×) and IQ2_S's gate
    products. These shapes give a block few rows (Q8_0's down product: 16 rows of K = 1024), so the per-block work of
    finding the groups and staging the activations dominates. The K = 8192 dense products, staged in tiles, trail the
    earlier kernel.
  - **More row blocks per block close most of the remaining gap** (`MT RB=…` rows; the default chose 1 for the gate
    product and 2 for the down product). With 4, at 3 tokens: N2P gate 353 → 275 µs (2.62×, the floor), N2P down
    431 → 342 µs (2.89×), N3P gate 2.79×, N3P down 3.08×, IQ3_XXS down 4.41 → 3.90×; dense products move by ≤ 5 %.
    A smaller activation budget (8 KB) costs 3–10 %, a larger one (32 KB) changes nothing.
  - At 2 tokens and 4 the order of the kernels is the same ([kernels.tsv](kernels.tsv), `vs_1_token`).
- The agent check with `--spec-type draft-mtp`: next (below).

## Files

| File | What |
|---|---|
| [correctness.md](correctness.md) | test-backend-ops cases passed per type and setting, bit-identity and memcheck results |
| [kernels.tsv](kernels.tsv) | kernel time on one P106: kernel (MT, MC, llama.cpp, MT variants), type, product, tokens, µs per call, cost relative to 1 token, ps per weight at 1 token |
| `run/build.sh` | the CUDA build on the quality machine (compute 6.1 + PTX, sm_70), and the rig tarball |
| `run/correctness.sh` | the correctness runs: test-backend-ops (native, PTX; three kernel settings), the Flint types, `multitoken_bitid`, compute-sanitizer |
| `run/rig_kernels.sh`, `run/table.sh` | the kernel timing on one P106 (new, earlier and llama.cpp's kernel; 1–4 tokens; row-block and tile settings) and its table |

## Made with

```bash
D=docs/data/2026-10-28_multitoken_kernels/run
QM=<quality machine> WORK=<its work folder> FORK=<fork checkout> COMMIT=f0f2be53b bash $D/build.sh
# multitoken_bitid against that build's static libraries (bench/mmvq_pascal/verify/build_multitoken_bitid.sh's link line)
BIN=<WORK>/out/build/bin BITID=<multitoken_bitid> GPUS=<a V100> OUT=correct_native.log JITS=0 bash $D/correctness.sh
BIN=<WORK>/out/build/bin BITID=<multitoken_bitid> GPUS=<a newer card> OUT=correct_ptx.log JITS=1 MTS="1|1 1 3" \
  MEMCHECK=0 bash $D/correctness.sh
# the rig: a tarball with test-backend-ops (here test-backend-ops alone, 50 MB), then correctness and timing on card 0
RIG=rig QM=<quality machine> WORK=<its work folder> NAME=<build name> MD5=<tarball md5> bash $D/rig_kernels.sh > rig_kernels.txt
bash $D/table.sh < rig_kernels.txt > kernels.tsv
```

- **Build:** fork branch `pascal-p106-v3-mtpk` at `f0f2be53b`: static, CUDA 12.9, compute 6.1 + compute_61 PTX and sm_70.
- **Machines:** the quality machine's V100 32 GB and RTX 3080 (correctness only; no timings there); the rig's P106-090
  card 0 with nothing else running on the rig (2026-10-28, 20:06–21:11: correctness 42 min, timing 23 min). Weights in
  the timing are random bytes (`GGML_TEST_RAW_QUANT=1`), the experts of each call routed afresh per repetition.

## Where this stopped (2026-10-28)

- **Done:** the kernel (fork `pascal-p106-v3-mtpk` `f0f2be53b`, pushed); correctness natively, through the compute 6.1
  PTX and on a P106; kernel time on one P106 at Ornith's shapes ([kernels.tsv](kernels.tsv), summary above): best or
  level with llama.cpp's on everything the N2P/N3P file runs; worse on Q8_0/Q5_0 experts, IQ2_S gates, K = 8192.
- **Half-done:**
  - Q8_0's MUL_MAT_ID, fused and shared-expert cases through the PTX: the run aborted at 5–8 tokens (llama.cpp's
    own MoE kernel, a test-setup limit); rerun them at 1–4 tokens only.
  - Q411's tool is ready but has not run: `bench/expert_overlap` with a `ROUTE_TEXT` mode and windows of 1–8 tokens,
    scripts in `docs/data/2026-10-28_ornith_expert_overlap/run/`.
- **Next, in order:**
  0. Defaults from the timing: row blocks per block ≥ 4 (and try 8) for the expert products, including the fused
     gate/up (not timed with the `RB` settings yet); the kernel per type at 2–4 tokens: MT for N2P, N3P, the dense IQ
     and Q5_0/Q8_0 products at K ≤ 3072 and the output layer; llama.cpp's kernel (`GGML_CUDA_MMVQ_PASCAL_MC=0`'s path)
     for the Q8_0/Q5_0 expert products and IQ2_S's gate products until MT beats it; the earlier kernel for K = 8192 until
     the tiles are faster. Rebuild, re-time with `run/rig_kernels.sh` (~65 min on card 0), check correctness again.
  1. The Q8_0 PTX rerun: `env GGML_CUDA_MMVQ_PASCAL=2 CUDA_FORCE_PTX_JIT=1 test-backend-ops -b CUDA0 -o MUL_MAT_ID -p
     "=q8_0,.*n=[1-4],"`, and the same for `MUL_MAT_ID_SHARED`, and `-o MUL_MAT_VEC_FUSION -p "type=q8_0,.*m=[1-4],"`.
  2. Q411: `run_overlap.sh` on the 35B Q8_0 and the 122B UD-Q2_K_XL over the agent (12 × 8,192 tokens) and code
     (80 × 512) texts of [2026-10-28_ornith_mtp_accept](../2026-10-28_ornith_mtp_accept/README.md); then the measured
     distinct experts per window replace the assumed 15 / 22 / 28.5 in
     [cheap_mtp_draft.md](../../research/notes/cheap_mtp_draft.md) §3.
  3. A rig build with llama-server (`run/build.sh`), then the agent check of valid1 (Q212's file) on that one binary,
     without and with `--spec-type draft-mtp --spec-draft-n-max 2` (candidates `valid1` and `valid1-mtp` of
     `docs/data/2026-10-27_ornith/run/agent_ornith.sh`), after a fit probe of the MTP line (the MTP block goes to the
     last card with the output layer). Baseline: valid1 at 14.5 tok/s (Q189).
  4. A profile of the 3-token verify step (`GGML_CUDA_PROFILE`): attention and Gated DeltaNet at 3 tokens run
     llama.cpp's kernels, outside this work; then Q409 (a confidence-gated chain length).
- **Kill criterion:** if the agent check with draft-mtp is not faster per task than without it on the same binary,
  profile the verify step before any more kernel work. With 4 row blocks per block the N2P/N3P expert products at
  3 tokens cost 1.0–1.2× the floor this design aims at, the distinct experts' one-token cost (summary above).
- **Open:** registers per block (`MMVQP_MT_MIN_BLOCKS` 3 against 2, not measured); N2P's per-8-weight scale (Q421).

## Provenance and licence

- The weights and activations of the checks are random (test-backend-ops; `multitoken_bitid` draws its rows from 256
  quantized random rows per shape).
- This folder: tables and scripts under this repository's licence.
