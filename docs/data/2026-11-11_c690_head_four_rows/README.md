# The output head's four-row product at its one-row cost, exact (C690, Q1222, 2026-11-11)

What this is: a design note and the checks behind a new kernel, not yet a measurement on a P106. In a served verify
pass of Miner Llama's Flash-Next preset, the output head's product (`output`, Q5_K, 2,560 × 248,320, 437 MB on the
last card) takes 8.05 and 8.06 ms of GPU time for its four rows, against 2.86 ms for one row: 2.82 times as long
([2026-11-10_served_verify_parts](../2026-11-10_served_verify_parts/README.md), R49;
[2026-11-10_release_profile](../2026-11-10_release_profile/README.md), R41). One row's time is the time it takes to
read the weights (437 MB in 2.86 ms is 153 GB/s; a P106-090 reads about 173). This note says which kernel takes the
four-row product today and why it costs that much, then describes a kernel that reads and decodes the head once for
all four rows and gives each row exactly the one-row kernel's sums. The rig run that checks and times it natively is
the last section.

The code is on the fork's branch `pascal-p106-v4-b12246-c690` (`ef91bff04`, six commits on `pascal-p106-v4`
`84f7bd3d2`), behind its own switch, off by default: `GGML_CUDA_MMVQ_PASCAL_Q5K_MT`
([docs/switches.md](../../switches.md)). Registry row [Q1222](../../../quants/experiments.md).

## Which kernel takes the head today

- **The path:** `ggml_cuda_mul_mat` → `ggml_cuda_mul_mat_vec_q` (up to 8 rows) → the k-column hook
  (`mmvq-pascal-kcol.cu`) first.
- **Four rows: the k-column kernel** (`mul_mat_kcol<KQ5_K, 4, R>`). The preset sets `KCOL_TABLE=0`, so no table
  sends the product elsewhere. With the table on, it would go there too: for Q5_K rows of up to 4,096 weights the
  table leaves only 2 and 3 rows to the other kernels.
- **One row: the one-token Pascal kernel** (`mul_mat_vec_q_pascal` with Q5_K's shape: one row per lane group, one
  super-block per lane, staged). The k-column kernel takes one row only for Q3_K, IQ4_NL and short Q8_0 rows.
- **The launch shape under the preset.** `KCOL_SMEM=32768` selects `pick_shape`'s rule from before Q823, with a
  32 KB budget:
  - a row is 10 super-blocks, so 2 lanes per row and 16 rows per warp;
  - two rows per lane group would need 45 KB of weight staging, so it runs one;
  - the activation tile holds 3 of a row's 5 iterations (2,688 bytes each) beside 22.5 KB of staging: 30,592 bytes
    per block, 3 blocks and 12 warps per SM;
  - 3,880 blocks of 64 rows. Each block stages its activation records twice (two tiles, 13.4 KB in all), behind
    `__syncthreads`.

## Why four rows cost 2.82 one-row passes

The counts below are read from the code, per lane and per super-block (176 bytes, 256 weights of one row). The
clocks are estimates for a P106-090: 5 SMs at est. 1.7 GHz under load ([mining_rig.md](../../mining_rig.md)), one
warp-wide load or store issued per clock per SM, and 128 bytes a clock into registers, so a 128-bit warp load takes
4 clocks. A product of the head is 15,520 warp iterations of 16 rows per SM.

| Per lane and super-block | one-token kernel, 1 row | k-column kernel, 4 rows | new kernel, 4 rows, R = 1 | new kernel, 4 rows, R = 2 (default) |
|---|---|---|---|---|
| weight staging: 128-bit global loads, shared stores, shared loads | 11, 11, 11 | 11, 11, 11 | 11, 11, 11 | 11, 11, 11 per row |
| decoding the 64 code words | once | once | once | once per row |
| activation loads | 18 × 128-bit global (288 bytes of `block_q8_1`), and 64 dp4a for the block sums | 96 shared (2 × 128 + 64 bits per sub-block and row) | 96 shared | 96 shared for 2 rows (48 a row) |
| per sub-block and row: integer sum and scaling | 8 dp4a, an integer multiply, 2 I2F, 2 FFMA | 8 dp4a, I2F, FMUL, 2 FFMA | 8 dp4a, 3 FFMA (no I2F) | 8 dp4a, 3 FFMA |
| load-store clocks per warp iteration of 16 rows | est. 204 | est. 452 | est. 452 | est. 292 |
| the head's product from those clocks alone | est. 1.9 ms (2.86 measured: memory-bound) | est. 4.1 ms (8.05 measured) | est. 4.1 ms | est. 2.7 ms (so memory-bound: est. 2.9) |

- **The k-column kernel is bound by its shared loads, then by latency.** With one row per lane group, every row reads
  every column's activation record for every 32 weights. That is est. 320 of the 452 load-store clocks of a warp's
  iteration, est. 4.1 ms for the product before any stall. Its arithmetic, est. 700 instructions per lane and
  super-block, takes est. 175 clocks of the four schedulers.
- **The other half of its 8.05 ms is latency (est.).** There are 12 warps per SM, no prefetch of the next weight
  segment, and a block of 64 rows passes two `__syncthreads` and stages 13.4 KB of activation records twice. The old
  shape (48 KB, two rows, one block of 4 warps per SM, one iteration per tile) was slower still: 11,563.8 µs in Q591.
- **The one-token kernel at one row is memory-bound:** est. 204 load-store clocks a warp iteration, est. 1.9 ms, under
  the 2.86 ms of the weights' read.

## The design

`mul_mat_vec_q5_K_pascal_mt<NT, R, PF>` in `ggml/src/ggml-cuda/mmvq-pascal-q5k.cu`, for plain Q5_K products of 2 to
4 rows (`MUL_MAT`, no fusion), on compute capability 6.1.

- **The one-token kernel's lanes and order.**
  - L lanes per row: the largest power of two, at most 32, that divides the row's super-blocks (2 for the head).
  - Lane `sl` of a row takes super-blocks `sl`, `sl + L`, … in that order.
  - Its 176-byte segment is staged by the lane group with 128-bit loads, as in the one-token kernel.
  - The lanes' sums are added with the same xor shuffles.
- **Each 32-weight sub-block of a row is decoded once** into eight words of 5-bit codes, then multiplied with dp4a by
  every row's activations. The earlier multi-token kernel (`mul_mat_vec_q_pascal_mc`) decodes it once per row.
- **Two rows per lane group by default (R = 2), sharing every activation load.** This halves the shared loads per
  row. `GGML_CUDA_MMVQ_PASCAL_Q5K_MT_R=1` keeps the one-token kernel's single row. R changes no row's sum.
- **The activations once per block.**
  - Every row's int8 values and its pair (d8, S) per 32 values are staged once into shared memory, with
    S = d8 · (the sum of the int8 values), as the one-token kernel computes it.
  - Padding sends the L lanes of a group to distinct banks. Every lane group of a warp reads the same L records.
  - The grid is persistent: about two blocks per SM, each looping over row blocks.
  - At four rows and K = 2,560 the activations take 13 KB, beside 34.5 KB of weight staging at R = 2 (3 warps, each
    lane group's two segments padded by 32 bytes so that a quarter warp hits distinct banks) and 33 KB at R = 1
    (6 warps).
- **A register prefetch.** Each warp loads its next segments (11 128-bit loads per lane and row) before it computes
  the current ones. `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=2` runs without it, to measure it.

### Why each row's result is the one-token kernel's, bit for bit

The one-token kernel adds a super-block to a lane's sum as `mmvqp_dot_q5_K` does, per sub-block s:

- `sumd = fmaf(d8, (float) (isum * sc), sumd)`
- `summ = fmaf((float) m, S, summ)`

and then `acc + d * sumd - dmin * summ`. The new kernel runs the same FP32 operations in the same order, on the same
operands:

- **The integer sums** are exact whatever the order.
- **Two conversions are made without the quarter-rate I2F and the integer multiply.** The dp4a chain starts at
  `0x4B400000`, so its result read as a float is 12,582,912 + isum: \|isum\| ≤ 32 · 31 · 127 = 125,984 < 2²².
  `fmaf(12582912 + isum, sc, -12582912 · sc)` is then isum · sc rounded once. That is exact: \|isum · sc\| <
  7.94 · 10⁶ < 2²⁴, and 12,582,912 · sc has 8 significant bits.
- **sc and m (6 bits each)** are placed as the low mantissa bits of the same float with one `__byte_perm`, and one
  subtraction gives them exactly.
- So `x` equals `(float) (isum * sc)` and `mf` equals `(float) m`, and `sumd`, `summ` and the super-block's
  expression see the same values in the same order. The final expression is written as in `mmvqp_dot_q5_K`, so the
  compiler contracts it the same way (checked in the SASS below).

### With the switch off

- `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=0` (the default) changes nothing. The k-column kernel's new test asks
  `ggml_cuda_mmvq_pascal_q5k_mt_takes`, which is false with the switch off. The hook in `ggml_cuda_mul_mat_vec_q`
  launches nothing.
- The new kernel follows `GGML_CUDA_MMVQ_PASCAL` (0 off, 2 on any NVIDIA GPU) and `GGML_CUDA_MMVQ_PASCAL_SKIP=q5_K`,
  as the one-token kernel does.
- It declines a product whose activations and staging pass 48 KB. At four rows that means rows wider than 7,168
  weights at R = 2 and wider than 8,192 at R = 1; the head's rows are 2,560. Such a product goes to the k-column kernel
  as before.

## What was checked without a GPU

All on a build machine's CPU, in containers without a GPU; nothing here ran a kernel.

- **The rig tarball builds** (`scripts/build_llamacpp_docker.sh`, `TESTS=1`, compute 6.1 only, CPU code without AVX):
  `ef91bff04`, the branch's tip, with `test-backend-ops`.
- **Registers and spills** ([sass_counts.tsv](sass_counts.tsv), from `run/sass_check.sh` on `cuobjdump`'s output):
  - at four rows, 245 registers at R = 2 with the prefetch, 193 without, 160 and 139 at R = 1;
  - no stack and no local memory in any of the 12 instantiations;
  - 2 blocks per SM fit everywhere: at the default, 6 warps per SM, 48,368 bytes of shared memory a block.
- **The instruction mix of one super-block, four rows, R = 2:**
  - 520 dp4a: 8 · 8 · 4 · 2, plus the 8 of the activations' block sums;
  - 118 shared loads: 22 for the two rows' weights, 96 for the activations, which the two rows share;
  - 208 FFMA: 3 per sub-block, row and column, and 2 per super-block's last expression;
  - no I2F in the loop (the 2 in the code are the block sums' conversion and the division of the staging loop);
  - all shared accesses `LDS`/`STS`, none generic.
- **The same rounding as the one-token kernel.**
  - Every instantiation ends a super-block with `t = FFMA(·, ·, acc)` and then `acc = FFMA(summ, -dmin, t)`, as
    `mul_mat_vec_q_pascal<Q5_K>` does (the last column of the table).
  - `sumd`, `summ` and `isum · sc` are one FFMA each, on the operands the design names.
- **The replay set `q5kmt`** (`scripts/record_checks.sh`):
  - 112 cases: Q5_K `MUL_MAT` at 1–4 rows, the output layer's four among them;
  - each case's inputs and the CPU backend's references, 2.07 GB, recorded by the rig build's `test-backend-ops` on
    the build machine's CPU;
  - [replay_cases.tsv](replay_cases.tsv), and its ten runs in [replay_runs.tsv](replay_runs.tsv): the kernel's
    settings under the preset's k-column switches, the defaults alone, the switch off, `SKIP=q5_K`;
  - recorded with `1a3eea354`'s binary. The later commits change only `mmvq-pascal-q5k.cu`, so the references are the
    tip's as well.
- **The bit-identity harness** links against the same tree (`bench/mmvq_pascal/verify/build_multitoken_bitid.sh`).
- **Not checked:** any result on a GPU. That the sums are the one-token kernel's, bit for bit, rests on the reading
  above until the rig's `bitid` step. The kernel's speed is an estimate until its `perf` step.

## The rig run (R75)

**Run on 2026-11-11 through the v4 night merge ([2026-11-11_r75_c690_p106](../2026-11-11_r75_c690_p106/README.md)):**
exact at every setting; R = 1 without the prefetch is the fastest at 2–4 rows (four rows 3,340.78 and 3,346.73 µs
against the k-column kernel's 8,103.16 and 8,100.11), and R = 2 is slower than R = 1 (Q1347); in a served verify pass
the head takes 3.149, 3.155, 3.149 and 3.151 ms against 8.105, 8.125, 8.115 and 8.120.

`run/rig_c690.sh` on one P106 of an idle rig, detached. Its steps:

- **replay:** the recorded set through `scripts/replay_checks.sh`. Passing: every run's line ends `ok`.
- **bitid:** `multitoken_bitid q5k` at the default, without the prefetch and at R = 1. Passing: `0 with differences`
  each. The control under the preset's k-column switches must differ.
- **perf:** `test-backend-ops perf` at the head's shape, 1 to 4 rows. Arms `off` (the preset: the k-column kernel at
  2 to 4 rows), `on`, `nopf`, `r1`, `r1nopf`, `nw2`, two passes alternated; `perf/us_run.tsv`.
- **served:** R49's profile of a served verify pass on the preset with the MTP draft, the switch off and on in the
  order off, on, on, off. `on` takes the perf step's fastest arm at 4 rows.

What it decides:

- **Exact or not:** `bitid` with 0 differences at every setting, and the replay passing, say that the verify pass's head
  gives each row the one-token kernel's logits. Any difference is a fault to find before any timing counts.
- **Gain, negligible or loss:** the head's product at 4 rows, `on` against `off`, in the two passes, each listed by
  value. The 1-row case is the same kernel in every arm, so it is the spread. The target is the 1-row time.
- **In the served step:** the head's product per verify pass (R49: 8.05 and 8.06 ms), and the step's time from the
  warm loads.

Variables: `RIG` the rig's ssh alias, `BUILD` the build machine's, `W` the build folder there, `R` the run folder
on the rig. The model, draft, preset, server script and request are R49's.

```bash
# on the build machine (no GPU): the rig tarball with test-backend-ops, the set, the bit-identity harness
TESTS=1 JOBS=8 WORK=$W DOCKER_ARGS="--cpu-shares 256" nice -n 19 scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c690
OUT=$W/sets SUITE=q5kmt BIN=$W/<tarball>/bin JOBS=8 bash scripts/record_checks.sh
bash bench/mmvq_pascal/verify/build_multitoken_bitid.sh $W/src $W/src
# relay the tarball, $W/sets/q5kmt.tar, bench/mmvq_pascal/build/multitoken_bitid, scripts/replay_checks.sh,
# run/rig_c690.sh and R49's verify_prof.sh to the rig; then on the rig:
BIN=$R/<tarball>/bin OUT=$R/out SET=$R/q5kmt REPLAY=$R/replay_checks.sh BITID=$R/multitoken_bitid \
  MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<release/flash-next-v3/settings.env> \
  RIG_SERVER=<bench/agent_check/rig_server.sh> REQ=<2026-11-04_draft_one_card/run/greedy_request.json> \
  VERIFY_PROF=$R/verify_prof.sh setsid -f nohup bash $R/rig_c690.sh > $R/console.txt 2>&1 < /dev/null
# the served tables on the build machine, as R49's: run/verify_parts.py of 2026-11-10_served_verify_parts on each
# load's prof/prof-verify.csv
```

## Files

| File | What |
|---|---|
| `run/rig_c690.sh` | the rig run: replay, bit identity, kernel time at 1–4 rows, the served verify pass |
| `../../../scripts/record_checks.sh` (suite `q5kmt`) | the replay set: Q5_K's `MUL_MAT` cases at 1–4 rows, ten switch settings |
| `../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp` (mode `q5k`) | every 2–4-row product against its rows' one-row products, bit for bit |
| `run/sass_check.sh` | the static check: registers, spills, the instruction mix and the last two FFMA per instantiation |
| [sass_counts.tsv](sass_counts.tsv) | its output for `ef91bff04` |
| [replay_cases.tsv](replay_cases.tsv) | the replay set's 112 cases (id, case string) |
| [replay_runs.tsv](replay_runs.tsv) | the replay set's ten runs (name, switches, ops, filter, expected verdict) |


Provenance: the code is the fork's (MIT, as llama.cpp). The test inputs are test-backend-ops' random tensors. No model
data is in this folder.
