# The other dense products of 2–4 tokens, read once and exact (C722, Q1277, 2026-11-11)

What this is: a design note and the checks behind a new kernel, not yet a measurement on a P106. It carries C690's
method ([2026-11-11_c690_head_four_rows](../2026-11-11_c690_head_four_rows/README.md): the output head's Q5_K product
of four rows at about its one-row cost, each row's sums the one-row kernel's) to the other dense products of
Qwen3.8-Flash-Next's verify pass. In a four-token pass on Miner Llama's Flash-Next preset the attention and recurrent
projections take 34.13 ms of GPU time against 15.81 ms at one token
([2026-11-10_release_profile](../2026-11-10_release_profile/README.md), R41). This note counts, from the code, where the
k-column kernel spends a four-token product for each type of those products, then describes one kernel for IQ4_XS and
Q4_K, the two types whose products on the k-column kernel give other sums than one-token decoding. The rig run that
checks and times it natively is the last section.

The code is on the fork's branch `pascal-p106-v4-b12246-c722` (on C690's `pascal-p106-v4-b12246-c690`), behind its own
switch, off by default: `GGML_CUDA_MMVQ_PASCAL_DMT` ([docs/switches.md](../../switches.md)). Registry row
[Q1277](../../../quants/experiments.md).

## The products and the kernels that take them today

R41's four-token pass, GPU time per pass summed over the nine cards, ms (its `summaries/release-k1.txt` and
`release-k4.txt`; the one-token column is blank where that summary lists no time):

| Product | Type | Rows × width | 1 token | 4 tokens | Kernel at 2–8 tokens under the preset | Kernel at 1 token |
|---|---|---|---:|---:|---|---|
| attn_qkv | Q3_K | 10,240 × 2,560 | 3.23 | 5.42 | k-column | k-column (no one-token Pascal kernel) |
| attn_gate | Q3_K | 6,144 × 2,560 | 2.87 | 4.95 | k-column | k-column |
| ssm_out | IQ4_XS | 2,560 × 6,144 | 1.29 | 3.88 | k-column | one-token Pascal kernel |
| attn_qkv | IQ4_XS | 10,240 × 2,560 | 1.51 | 3.82 | k-column | one-token |
| ssm_out | Q4_K | 2,560 × 6,144 | 0.73 | 2.54 | k-column | one-token |
| attn_q | Q3_K | 12,288 × 2,560 | 1.28 | 2.14 | k-column | k-column |
| attn_q | IQ4_XS | 12,288 × 2,560 | 0.52 | 1.37 | k-column | one-token |
| attn_qkv | Q4_K | 10,240 × 2,560 | 0.41 | 1.15 | k-column | one-token |
| attn_output | Q6_K | 2,560 × 6,144 | 0.40 | 1.06 | k-column | one-token |
| attn_output | Q5_K | 2,560 × 6,144 | 0.34 | 0.96 | k-column (C690's kernel with its switch) | one-token |
| attn_output | Q4_K | 2,560 × 6,144 | 0.27 | 0.93 | k-column | one-token |
| ssm_out | Q3_K | 2,560 × 6,144 | 0.50 | 0.83 | k-column | k-column |
| attn_gate | IQ4_XS | 6,144 × 2,560 | 0.31 | 0.77 | k-column | one-token |
| ssm_out | Q5_K | 2,560 × 6,144 | 0.25 | 0.72 | k-column (C690's kernel with its switch) | one-token |
| attn_gate | Q4_K | 6,144 × 2,560 | | 0.56 | k-column | one-token |
| ssm_out | Q6_K | 2,560 × 6,144 | | 0.53 | k-column | one-token |

- **The path:** `ggml_cuda_mul_mat_vec_q` asks the k-column hook first (`mmvq-pascal-kcol.cu`). The preset sets
  `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768` and `_KCOL_TABLE=0`, so the k-column kernel takes every one of these products
  at 2 to 8 tokens, with `pick_shape`'s rule from before Q823 and a 32 KB budget.
- **At one token** the k-column kernel takes only Q3_K, IQ4_NL and short Q8_0 rows. The other types go to the one-token
  Pascal kernel (`mul_mat_vec_q_pascal`: one super-block per lane, L lanes per row, staged).
- **So Q3_K is already exact at 2–8 tokens:** its one-token kernel is the k-column kernel, whose sums depend on the row
  width only (each lane adds its units in order, then the lanes are added with the same shuffles). With the same
  32 KB budget the k-column kernel even picks the same shape for Q3_K at one and at four columns (one row per lane
  group, the whole K in one activation tile). A faster Q3_K kernel has to keep that order; it is a separate row
  (below).
- **IQ4_XS, Q4_K, Q5_K and Q6_K are not exact:** at 2–8 tokens the k-column kernel sums a row as
  `fmaf((float) isum * scale, d8, acc)` per 32 weights, the one-token kernel per super-block through `sumd` and `summ`.
  Q5_K has C690's kernel. This note's kernel takes IQ4_XS and Q4_K.
- **The hyper-connections** (BF16 in the release file, 10.19 ms at four tokens against 8.23 at one) are not quantized
  products: the BF16 kernel takes them, not these. **The shared expert's** gate and up projections (Q2_0) run fused with
  the routed experts in one `MUL_MAT_ID` launch. Both are separate rows (below).

### Where a four-token product's time goes on the k-column kernel

The counts are read from the code, per lane and per 256-weight super-block of one row, for four tokens. The clocks are
estimates for a P106-090 as in C690's note: 5 SMs at est. 1.7 GHz, one warp-wide load or store issued per clock per SM,
128 bytes a clock into registers, so a 128-bit warp load costs 4 clocks, a 64-bit one 2, a 32-bit one 1. A warp
iteration is 32 such lane-units.

| Per lane and super-block, 4 tokens | IQ4_XS | Q4_K | Q3_K | Q5_K (C690) |
|---|---|---|---|---|
| k-column shape under the preset, K = 2,560 | R = 1, 5 iterations a tile, 30,848 B | R = 1, 5 a tile, 31,872 B | R = 1, 5 a tile, warp staging (2-byte-aligned blocks), 32,384 B | R = 1, 3 a tile |
| activation loads per sub-block and token | 2 × 128-bit + 64-bit (d, S) | the same | 2 × 128-bit + 2 × 64-bit (d, S and the half sums) | 2 × 128-bit + 64-bit |
| activation clocks | est. 320 | est. 320 | est. 384 | est. 320 |
| weight staging and reads | 8.5 + 8.5 128-bit, 17 64-bit: est. 102 | 9 + 9 + 9 128-bit: est. 108 | warp staging est. 60, 50 32-bit reads: est. 110 | 11 + 11 + 11: est. 132 |
| load-store clocks per warp iteration | **est. 422** | **est. 428** | **est. 494** | est. 452 |
| the one-token kernel at 1 token (18 128-bit loads of the `block_q8_1` run, then the weights) | est. 174 | est. 180 | est. 206 (k-column, 1 column) | est. 204 |

- **Every row reads every token's activation record per 32 weights**, so the activation loads are three quarters of the
  load-store work: est. 320 of 422 clocks for IQ4_XS, 320 of 428 for Q4_K, 384 of 494 for Q3_K.
- **At K = 6,144 (ssm_out, attn_output)** the k-column kernel holds one iteration of 8 lanes' records per tile
  (10,752 B at 4 tokens, R = 1): three tiles a row block, each staged by the block behind `__syncthreads`, est. 42 more
  clocks a warp iteration.
- **Against the weights:** a warp iteration's 32 super-blocks are 4,352 B (IQ4_XS) or 4,608 B (Q4_K), est. 241 and
  255 clocks of one SM's share of the measured 153 GB/s. The one-token kernel is under that (memory-bound); the
  k-column kernel at four tokens is est. 1.75 and 1.68 times over it before any latency. R41's four-token times are
  2.53 (attn_qkv IQ4_XS) and 3.01 (ssm_out IQ4_XS) times the one-token ones; the rest is latency (est.).

## The design

`mul_mat_vec_q_pascal_dmt<type, NT, R, PF>` in `ggml/src/ggml-cuda/mmvq-pascal-dmt.cu`, for plain IQ4_XS and Q4_K
products of 2 to 4 tokens (`MUL_MAT`, no fusion), on compute capability 6.1. It is C690's Q5_K kernel with each type's
own super-block step.

- **The one-token kernel's lanes and order:** L lanes per row (the largest power of two, at most 32, that divides the
  row's super-blocks), lane `sl` takes super-blocks `sl`, `sl + L`, … in that order, the lanes' sums added with the same
  xor shuffles.
- **Each 32-weight sub-block of a row decoded once** into eight words of int8 weights, then multiplied with dp4a by
  every token's activations: IQ4_XS's 4-bit indices looked up in `kvalues_iq4nl` with `__byte_perm`, as
  `mmvqp_dot_iq4_xs` does; Q4_K's nibbles, the high ones shifted down (the one-token kernel multiplies them in place
  and shifts the sum back by 4, the same integer).
- **Rows per lane group chosen per product.** R = 2 shares every activation load between two rows where its warps hold
  at least as many rows as R = 1's; else R = 1. `GGML_CUDA_MMVQ_PASCAL_DMT_R` forces either.
- **The activations once per block.** Every token's int8 values (IQ4_XS's regrouped as the one-token kernel regroups
  them) and per sub-block d8 (Q4_K also S = d8 · the sum of the int8 values, as the one-token kernel computes it) are
  staged once into shared memory; the grid is persistent (about two blocks per SM), each block looping over row
  blocks.
- **Padding against bank conflicts.** The activation records sit an odd number of bank groups apart. The lane groups'
  weight segments get the smallest padding (a multiple of 16 bytes) under which the lanes' reads of their super-blocks
  hit distinct banks: 128-bit reads per quarter warp for Q4_K, 64-bit reads per half warp for IQ4_XS, whose 136-byte
  super-blocks are only 8-byte aligned (`dmt_gstride` finds it on the host).
- **A register prefetch** of each warp's next segments (`GGML_CUDA_MMVQ_PASCAL_DMT=2` runs without it).
- **Where it takes a product:** IQ4_XS rows of an even number of super-blocks (so that a segment is whole 16-byte
  units), every Q4_K row, 16-byte-aligned rows, and only where the activations and at least one warp's staging fit
  48 KB.

The shapes at Flash-Next's widths, four tokens (the warps per block are the most that fit 48 KB, at most 6 at R = 1 and
3 at R = 2):

| Type, K | activations | R | weight staging a warp | warps a block | blocks a SM |
|---|---:|---:|---:|---:|---:|
| IQ4_XS, 2,560 | 11,728 B | 2 | 8,960 B (16 B of padding a lane group) | 3 | 2 |
| IQ4_XS, 6,144 | 28,128 B | 2 | 8,960 B (64 B a lane group) | 2 | 2 |
| Q4_K, 2,560 | 13,040 B | 2 | 9,728 B (32 B a lane group) | 3 | 2 |
| Q4_K, 6,144 | 31,296 B | 1 | 4,608 B | 3 | 2 |

The load-store count of the new kernel, per lane and super-block, four tokens:

| | IQ4_XS, R = 2 | IQ4_XS, R = 1 | Q4_K, R = 2 | Q4_K, R = 1 |
|---|---|---|---|---|
| activation loads per sub-block and token (shared by the R rows) | 2 × 128-bit + 32-bit (d8): est. 288 | est. 288 | 2 × 128-bit + 64-bit: est. 320 | est. 320 |
| weights per row | est. 102 | est. 102 | est. 108 | est. 108 |
| per 32 lane-units (one row each) | **est. 246** | est. 390 | **est. 268** | est. 428 |
| against the k-column kernel's | 422 | 422 | 428 | 428 (and no tile restaging at K = 6,144) |

- **At R = 2 the four-token product costs about the one-token kernel's load-store work** (est. 246 against 174 for
  IQ4_XS, 268 against 180 for Q4_K), est. at or near the weights' read time (241 and 255).
- **Q4_K at K = 6,144 stays at R = 1:** four tokens' activations (31,296 B) leave room for one warp at R = 2 against
  three at R = 1. It still drops the tile restaging and the latency of 64-row blocks; the rig's `perf` step compares
  R = 1 and R = 2 there.
- **Estimate for a four-token pass:** each product at the larger of its one-token time and that time scaled by the new
  kernel's load-store clocks over the weights' read clocks: est. 2.39 ms for IQ4_XS's three 2,560-wide products against
  5.96, est. 1.32 for its ssm_out against 3.88, est. 0.43 for Q4_K's attn_qkv against 1.15, est. 1.23 and 0.45 for its
  ssm_out and attn_output against 2.54 and 0.93. In all est. 8.6 ms of GPU time a pass, if latency stays hidden at 4 to
  6 warps a SM (est.; the rig's `perf` step measures it).

### Why each token's result is the one-token kernel's, bit for bit

The one-token kernel adds a super-block to a lane's sum, per sub-block s:

- IQ4_XS (`mmvqp_dot_iq4_xs`): `sumf = fmaf(d8, (float) (isum * (ls - 32)), sumf)`; then `fmaf(d, sumf, acc)`;
- Q4_K (`mmvqp_dot_q4_K`): `sumd = fmaf(d8, (float) (isum * sc), sumd)` and `summ = fmaf((float) m, S, summ)`; then
  `acc + d * sumd - dmin * summ`.

The new kernel runs the same FP32 operations in the same order, on the same operands:

- **The integer sums** are exact whatever the order.
- **The conversions without I2F and the integer multiply**, as in C690's kernel: the dp4a chain starts at
  `0x4B400000`, so its result read as a float is 12,582,912 + isum, exact while |isum| < 2²²:
  - IQ4_XS: |isum| ≤ 32 · 127 · 127 = 516,128; Q4_K: |isum| ≤ 32 · 15 · 127 = 60,960.
  - `fmaf(12582912 + isum, f, -12582912 · f)` is then isum · f rounded once, which is exact: |isum · (ls − 32)| ≤
    516,128 · 32 = 16,516,096 < 2²⁴, |isum · sc| ≤ 60,960 · 63 < 2²⁴; and 12,582,912 · f has at most 8 significant bits.
  - ls − 32, sc and m (6 bits each) come out exactly as floats from the same trick: `0x4B400000 | ls` read as a float,
    less 12,582,944; sc and m with C690's `__byte_perm`.
- So `x` equals the one-token kernel's converted product, and `sumf`, `sumd`, `summ` and the super-block's last
  expression see the same values in the same order. The last expression is written as in the one-token kernel, so the
  compiler contracts it the same way (checked in the SASS below for Q4_K; IQ4_XS's is an explicit `fmaf` in both).

### With the switch off

- `GGML_CUDA_MMVQ_PASCAL_DMT=0` (the default) changes nothing: the k-column kernel's new test asks
  `ggml_cuda_mmvq_pascal_dmt_takes`, which is false with the switch off, and the new hook in `ggml_cuda_mul_mat_vec_q`
  launches nothing.
- The kernel follows `GGML_CUDA_MMVQ_PASCAL` (0 off, 2 on any NVIDIA GPU) and `GGML_CUDA_MMVQ_PASCAL_SKIP`, as the
  one-token kernel does; `GGML_CUDA_MMVQ_PASCAL_DMT_SKIP=q4_K` (or `iq4_xs`) leaves a type to the k-column kernel.
- It declines a product whose activations and one warp's staging pass 48 KB. At four tokens that means rows of more than
  34 super-blocks (Q4_K: 8,704 weights) or 38 (IQ4_XS: 9,728); Flash-Next's are 10 and 24. Such a product goes to the
  k-column kernel as before. Five to eight tokens go there too.

## What was checked without a GPU

All on a build machine's CPU, in containers without a GPU; nothing here ran a kernel.

- **The rig tarball builds** (`scripts/build_llamacpp_docker.sh`, `TESTS=1`, compute 6.1 only, CPU code without AVX):
  `dd72554b6`, the branch's tip, with `test-backend-ops`; the new file compiles without a warning.
- **Registers and spills** ([sass_counts.tsv](sass_counts.tsv), from `run/sass_check.sh` on `cuobjdump`'s output):
  - at four tokens and R = 2, 204 registers for Q4_K and 184 for IQ4_XS with the prefetch, 134 and 114 without; at
    R = 1, 136 and 127 with it;
  - no stack and no local memory in any of the 24 instantiations, so the launch bounds' 2 blocks per SM hold.
- **The instruction mix, four tokens, R = 2** (one super-block's loop body with the staging around it):
  - dp4a: 520 for Q4_K (8 · 8 · 4 · 2, and 8 for the activations' block sums), 512 for IQ4_XS;
  - shared loads: 114 for Q4_K (96 of activations shared by the 2 rows, 18 of weights), 130 for IQ4_XS (96 and 34: its
    super-blocks read 64 bits at a time);
  - FFMA: 208 for Q4_K (3 per sub-block, row and token, 2 per super-block's last expression), 136 for IQ4_XS (2, and
    its last `fmaf`);
  - PRMT: 392 for IQ4_XS (6 per code word for the lookup, and the activations' regrouping);
  - no I2F in the loop (the one or two in the code are the staging's: the block sums' conversion and its index
    division).
- **The same rounding as the one-token kernel for Q4_K:** every instantiation ends a super-block with
  `t = FFMA(·, ·, acc)` and then `acc = FFMA(summ, -dmin, t)`, as `mul_mat_vec_q_pascal<Q4_K>` does (the last column of
  the table). IQ4_XS ends it with an explicit `fmaf(d, sumf, acc)` in both kernels.
- **The replay set `dmt`** (`scripts/record_checks.sh`): 315 cases, every IQ4_XS and Q4_K `MUL_MAT` case of
  `test-backend-ops` at 1–5 tokens (the new ones at Flash-Next's shapes and lane-group sizes, the k-column cases,
  upstream's own), each case's inputs and the CPU backend's references recorded by the rig build's `test-backend-ops`
  on the build machine's CPU, 1.01 GB ([replay_cases.tsv](replay_cases.tsv)). Its twelve runs
  ([replay_runs.tsv](replay_runs.tsv)) are the kernel's settings under the preset's k-column switches, the defaults
  alone, the switch off, each type left out, and `SKIP` for both.
- **The bit-identity harness** links against the same tree (`multitoken_bitid dmt`).
- **Not checked:** any result on a GPU. That the sums are the one-token kernel's, bit for bit, rests on the reading
  above until the rig's `bitid` step. The kernel's speed is an estimate until its `perf` step.

## The rig run (R79)

Run on 2026-11-11 on the v4 night merge: [2026-11-11_r79_c722_p106](../2026-11-11_r79_c722_p106/README.md). Exact at every setting, and at R = 1 a gain at every Flash-Next shape at 2–4 tokens.

`run/rig_c722.sh` on one P106 of an idle rig, detached. Its steps:

- **replay:** the recorded set through `scripts/replay_checks.sh`. Passing: every run's line ends `ok`.
- **bitid:** `multitoken_bitid dmt` at the default, without the prefetch, at R = 1 and at R = 2. Passing:
  `0 with differences` each. The control under the preset's k-column switches must differ.
- **perf:** `test-backend-ops perf` at Flash-Next's shapes of both types, 1 to 5 tokens. Arms `off` (the preset: the
  k-column kernel at 2 to 8 tokens), `on`, `nopf`, `r1`, `r2`, `r1nopf`, `nw2`, two passes alternated;
  `perf/us_run.tsv`. The 1-token and 5-token cases run the same kernels in every arm, so they give the spread.
- **served:** R49's profile of a served verify pass on the preset with the MTP draft, the switch off and on in the order
  off, on, on, off. `on` takes the perf step's fastest arm at 4 tokens.

What it decides:

- **Exact or not:** `bitid` with 0 differences at every setting, and the replay passing, say that the verify pass's
  IQ4_XS and Q4_K projections give each token the one-token kernel's results. Any difference is a fault to find before
  any timing counts.
- **Gain, negligible or loss:** each shape at 2–4 tokens, `on` against `off`, in the two passes, each listed by value;
  the target is the 1-token time.
- **In the served step:** the projections' time per verify pass (R49's parts), and the step's time from the warm loads.

Variables: `RIG` the rig's ssh alias, `BUILD` the build machine's, `W` the build folder there, `R` the run folder
on the rig. The model, draft, preset, server script and request are R49's.

```bash
# on the build machine (no GPU): the rig tarball with test-backend-ops, the set, the bit-identity harness, SASS
TESTS=1 JOBS=8 WORK=$W DOCKER_ARGS="--cpu-shares 256" nice -n 19 scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c722
OUT=$W/sets SUITE=dmt BIN=$W/<tarball>/bin JOBS=8 bash scripts/record_checks.sh
bash bench/mmvq_pascal/verify/build_multitoken_bitid.sh $W/src $W/src
O=$W/src/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
cuobjdump -res-usage $O/mmvq-pascal-dmt.cu.o > $W/sass/dmt_res.txt
cuobjdump -sass $O/mmvq-pascal-dmt.cu.o > $W/sass/dmt.sass
cuobjdump -sass $O/mmvq-pascal.cu.o | gzip > $W/sass/mmvqp.sass.gz
SASS=$W/sass bash run/sass_check.sh > sass_counts.tsv
# relay the tarball, $W/sets/dmt.tar, bench/mmvq_pascal/build/multitoken_bitid, scripts/replay_checks.sh,
# run/rig_c722.sh and R49's verify_prof.sh to the rig; then on the rig:
BIN=$R/<tarball>/bin OUT=$R/out SET=$R/dmt REPLAY=$R/replay_checks.sh BITID=$R/multitoken_bitid \
  MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<release/flash-next-v3/settings.env> \
  RIG_SERVER=<bench/agent_check/rig_server.sh> REQ=<2026-11-04_draft_one_card/run/greedy_request.json> \
  VERIFY_PROF=$R/verify_prof.sh setsid -f nohup bash $R/rig_c722.sh > $R/console.txt 2>&1 < /dev/null
# the served tables on the build machine, as R49's: run/verify_parts.py of 2026-11-10_served_verify_parts on each
# load's prof/prof-verify.csv
```

## What is left for later rows

- **Q3_K at 2–8 tokens** (Q1285; 5.42, 4.95, 2.14 and 0.83 ms at four tokens): already exact; a kernel in the
  k-column kernel's order with two rows per lane group and the activations staged once (est. 302 load-store clocks a
  warp iteration against 494).
- **Q6_K** (Q1286; attn_output and ssm_out, 6,144 wide): this kernel with 2-byte-aligned super-block reads.
- **The shape at K = 6,144** (Q1287): whole-K activation staging leaves 2 to 3 warps a block (and C690's Q5_K kernel,
  at its default R = 2, one); K tiles or another split, after the rig's `perf` step.
- **Five to eight tokens** (Q1288; the draft line's five-token prompt passes): this kernel and C690's take 2 to 4.
- **The BF16 hyper-connections** (Q1289) and **the shared expert** (Q1290) at 2–4 tokens: other kernels (the BF16
  product, the fused `MUL_MAT_ID` launch).

## Files

| File | What |
|---|---|
| `run/rig_c722.sh` | the rig run: replay, bit identity, kernel time at 1–5 tokens, the served verify pass |
| `../../../scripts/record_checks.sh` (suite `dmt`) | the replay set: IQ4_XS's and Q4_K's `MUL_MAT` cases at 1–5 tokens, twelve switch settings |
| `../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp` (mode `dmt`) | every 2–4-token product against its tokens' one-token products, bit for bit |
| `run/sass_check.sh` | the static check: registers, spills, the instruction mix, Q4_K's last two FFMA per instantiation |
| [sass_counts.tsv](sass_counts.tsv) | its output for the branch's tip |
| [replay_cases.tsv](replay_cases.tsv) | the replay set's cases (id, case string) |
| [replay_runs.tsv](replay_runs.tsv) | the replay set's runs (name, switches, ops, filter, expected verdict) |

Provenance: the code is the fork's (MIT, as llama.cpp). The test inputs are test-backend-ops' random tensors; the
times in the first table are R41's. No model data is in this folder.
