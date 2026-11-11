# Q3_K's products of 2–8 tokens, read once and in the one-token order (C726, Q1285, 2026-11-11)

What this is: a design note and the checks behind a new kernel, not yet a measurement on a P106. It carries C722's
kernel ([2026-11-11_c722_short_batch_products](../2026-11-11_c722_short_batch_products/README.md): IQ4_XS's and Q4_K's
products of 2–4 tokens with every weight read and decoded once, each token's sums the one-token kernel's) to Q3_K, the
type of most of Qwen3.8-Flash-Next's attention projections, whose one-token kernel is a different one: the k-column
kernel itself. The rig run that checks and times it natively is the last section.

The code is on the fork's branch `pascal-p106-v4-b12246-c726` (on `pascal-p106-v4-b12246-c748`, `5b329df16`), behind
its own switch, off by default: `GGML_CUDA_MMVQ_PASCAL_Q3K_MT` ([docs/switches.md](../../switches.md)). Registry row
[Q1285](../../../quants/experiments.md).

## The products

GPU time per pass summed over the nine cards, ms, at one and four tokens
([2026-11-10_release_profile](../2026-11-10_release_profile/README.md), R41; the shared expert's from
[2026-11-11_c731_shared_expert](../2026-11-11_c731_shared_expert/README.md)):

| Product | Rows × width | 1 token | 4 tokens |
|---|---|---:|---:|
| attn_qkv | 10,240 × 2,560 | 3.23 | 5.42 |
| attn_gate | 6,144 × 2,560 | 2.87 | 4.95 |
| attn_q | 12,288 × 2,560 | 1.28 | 2.14 |
| ssm_out | 2,560 × 6,144 | 0.50 | 0.83 |
| the shared expert's gate (12) and up (21) | 640 × 2,560 | 0.810 | 1.195 |
| all Q3_K products | | 8.69 | 14.535 |

- **Q3_K has no one-token Pascal kernel:** under the Flash-Next preset (`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`,
  `_KCOL_TABLE=0`) the k-column kernel (`mmvq-pascal-kcol.cu`) takes Q3_K at one token and at 2–8 tokens alike. Its
  sums depend on the row width only (each lane adds its super-blocks in order, the lanes are added with the same
  shuffles), so these products already give every token the one-token sums. A faster kernel has to keep that order,
  not the one-token Pascal kernel's as C722's does. (The shared expert's 6 gate/up pairs fused at one token go through
  llama.cpp's own kernel there and are not exact at 2–4 tokens either way: C756.)
- **Where a four-token product's time goes on the k-column kernel** (C722's count, per lane and super-block): every row
  reads every token's activation record per 32 weights, two 128-bit and two 64-bit loads, est. 384 of 494 load-store
  clocks a warp iteration; the block stages its activation records again for every 64 rows.

## The design

`mul_mat_vec_q_pascal_q3kmt<NT, R, PF>` in `ggml/src/ggml-cuda/mmvq-pascal-q3kmt.cu` (new file; one test in
`mmvq-pascal-kcol.cu` beside C722's, one hook in `mmvq.cu` after C722's), for plain Q3_K products of 2 to 8 tokens
(`MUL_MAT`, no fusion), on compute capability 6.1.

- **The k-column kernel's lanes and order:** L lanes per row (the largest power of two, at most 32, that divides the
  row's super-blocks), lane `sl` takes super-blocks `sl`, `sl + L`, … in that order, a super-block's eight 32-weight
  blocks in order, each added as `acc = fmaf(fmaf((float) i1, d·(sc1 − 32), (float) i0 · (d·(sc0 − 32))), d8 / f, acc)`
  (`kcol::unit_q3_K`: i0 and i1 the two 16-weight halves' integer sums, started at −4 f times the halves' activation
  sums, f = 1, 4, 16, 16 for the block's place in its group of four, the codes read in place), and the lanes' sums added
  with the same xor shuffles.
- **Each 32-weight block of a row decoded once** into eight words (the codes masked in place, the high bit moved next
  to them, as `kcol::unit_q3_K`) and multiplied with dp4a by every token's activations.
- **The activation terms from the Q8_1 blocks** that `ggml_cuda_mul_mat_vec_q` quantizes, computed as the k-column
  kernel's own quantizer (`kcol::quantize_records<KQ3_K>`) computes them: d8 / f with the same expression and −4 f
  times the sums of each half's int8 values. The two quantizers give the same int8 values and FP16 scale: in their
  sm_61 code both take d = amax · (1/127) with one FMUL, then MUFU.RCP of d, FMUL by x, and the same rounding, and
  convert d to FP16 and back.
- **The conversions exact without I2F:** each half's dp4a chain starts at 0x4B400000 plus its offset (staged so), so
  its result read as a float is 12,582,912 + i (|i| ≤ 16 · 4 · 16 · 127 = 130,048 < 2²²) and less 12,582,912 it is
  (float) i, exactly; the 6-bit scales come out of a `__byte_perm` into the same float, less 12,582,944 they are
  (float) (sc − 32). Every FP32 operation then sees the k-column kernel's operands in its order: the FMUL d · (sc − 32),
  the FMUL (float) i0 · dsc0 and the two FFMA.
- **The activations once per block:** every token's int8 words, d8 / f and the two offsets are staged once into shared
  memory (per super-block position, block and token; the positions an odd number of bank groups apart, so that the L
  lanes of a group read distinct banks, and every lane group of a warp reads the same L records, a broadcast); the grid
  is persistent (about two blocks per SM), each block looping over row blocks.
- **The weights:** Q3_K's 110-byte super-blocks are 2-byte aligned, so each warp stages each of its rows' segments (L
  super-blocks a row) from the 16-byte boundary below it into a row slot of 8 L 16-byte units with 128-bit loads, as the
  k-column kernel's warp staging does, and each lane reads its super-block's words from there with a `__byte_perm` a
  word where it lies 2 bytes off. Every row slot is padded by the 16-byte units (0 to 7) under which a 32-bit read of
  the same word of every lane's super-block meets the fewest distinct words in one bank (without it every row's slot
  starts in the same bank).
- **One row per lane group by default** (R79: C722's kernel was faster at R = 1 than at R = 2 at every Flash-Next
  shape); `_R=2` shares every activation load between two rows.
- **A register prefetch** of each warp's next segments (`GGML_CUDA_MMVQ_PASCAL_Q3K_MT=2` runs without it).
- **Where it takes a product:** rows of an even number of super-blocks (so that a row slot of 8 L units holds the
  segment from the 16-byte boundary below it and the 4 bytes past a super-block that the read of its last word takes),
  and only where the activations leave room for at least two warps' staging in 48 KB. A block of one warp leaves 2 warps
  an SM, and C722's kernel lost 46–47 % to the k-column kernel where it ran so (Q4_K at K = 6,144, R = 2; R79); `_NW=1`
  allows it.

The shapes at Flash-Next's widths ([shapes.tsv](shapes.tsv), from `run/shapes.sh`, the host code replayed; the row
slot holds 8 L 16-byte units and one unit of padding at both widths; "conflicts" is the most distinct words a bank
serves in one 32-bit read of the weights, with the padding and without it):

| K (L) | Tokens | R | Activations | Conflicts | Warps a block | Shared memory | Blocks a SM |
|---|---:|---:|---:|---|---:|---:|---:|
| 2,560 (2) | 2 | 1 | 7,328 B | 2 (4) | 6 | 33,440 B | 2 |
| 2,560 (2) | 4 | 1 | 14,368 B | 2 (4) | 6 | 40,480 B | 2 |
| 2,560 (2) | 4 | 2 | 14,368 B | 4 (8) | 3 | 40,480 B | 2 |
| 2,560 (2) | 5 | 1 | 17,888 B | 2 (4) | 6 | 44,000 B | 2 |
| 2,560 (2) | 8 | 1 | 28,448 B | 2 (4) | 4 | 45,856 B | 2 |
| 6,144 (8) | 2 | 1 | 17,568 B | 2 (8) | 6 | 42,528 B | 2 |
| 6,144 (8) | 3 | 1 | 26,016 B | 2 (8) | 5 | 46,816 B | 2 |
| 6,144 (8) | 4 | 1 | 34,464 B | 2 (8) | 3 | 46,944 B | 2 |
| 6,144 (8) | 4 | 2 | 34,464 B | 2 (8) | 1: left to the k-column kernel | | |
| 6,144 (8) | 5 | 1 | 42,912 B | 2 (8) | 1: left to the k-column kernel | | |
| 6,144 (8) | 6–8 | 1 | 51,360–68,256 B | | 0: left to the k-column kernel | | |

- **At K = 2,560** (attn_qkv, attn_gate, attn_q and the shared expert's products) every token count of 2–8 runs it, at
  R = 1 with 4 to 6 warps a block, 8 to 12 warps a SM.
- **At K = 6,144** (ssm_out, 0.83 ms of a four-token pass) it runs 2–4 tokens at R = 1 with 3 to 6 warps a block; from 5
  tokens on (and at 4 tokens with R = 2) the activations leave room for one warp or none, so the k-column kernel keeps
  those. Tiling K there is Q1287's question, asked for C722's kernel at the same width.

### The load-store count

Per 32 lane-units (a warp iteration at R = 1: one super-block of one row per lane), four tokens, est. as in C722's note
(a 128-bit warp load 4 clocks, 64-bit 2, 32-bit 1):

| | k-column kernel (C722's count) | this kernel, R = 1 | this kernel, R = 2 (per row) |
|---|---:|---:|---:|
| activations per block and token | 2 × 128-bit + 2 × 64-bit | 2 × 128-bit + 32-bit + 64-bit | the same, shared by 2 rows |
| activation clocks | est. 384 | est. 352 | est. 176 |
| weight staging (128-bit stores, 8 a row) | est. 60 | est. 32 | est. 32 |
| weight reads (32 32-bit loads a row, from the SASS) × the bank conflict degree | est. 50 | est. 64 (degree 2) | est. 64–128 (degree 2–4) |
| **load-store clocks a warp iteration** | **est. 494** | **est. 448** | **est. 272–336** |
| activation records restaged per 64 rows | yes | no | no |

- At R = 1 the count drops by est. a tenth; the k-column kernel's real four-token time is 1.68 times its one-token time
  on attn_qkv (R41), which these counts do not explain, as they did not for C722's kernel: C722's IQ4_XS kernel at R = 1
  (est. 390 against 422) ran at half the k-column kernel's time on a P106 (R79). Its gain came with the same structure
  this kernel keeps (the activations staged once, the persistent grid, the prefetch).
- At R = 2 the activation loads are halved and the count falls to est. 272–336, the row's est. 302. R79 found R = 2
  slower for C722's kernel (fewer warps an SM); the rig run times both.

## What was checked without a GPU

All on a build machine's CPU, in containers without a GPU; nothing here ran a kernel.
[summary.tsv](summary.tsv) holds every step's verdict.

- **The rig tarball builds** (`scripts/build_llamacpp_docker.sh`, `TESTS=1`, compute 6.1 only, CPU code without AVX):
  `77343063d`, the branch's tip, `llama.cpp-20261111-77343063d-pascal-p106-v4-b12246-c726.tar.gz`, 662,639,042 B, md5
  `e2cbfd77f53014112d929b94fff90f88`; its `test-backend-ops` md5 `d1af76cc8cf5628f79d77f7404671fb3`.
- **Every other kernel's code is the base's** ([sass_objects.tsv](sass_objects.tsv)): `cuobjdump -sass` of each of the
  194 objects of the CUDA backend against `5b329df16`'s rig build, the anonymous namespaces' hashes masked: 193 the
  same, `mmvq-pascal-q3kmt.cu.o` new. The two hooks (`mmvq.cu`, `mmvq-pascal-kcol.cu`) change host code only.
- **Registers and spills** ([sass_counts.tsv](sass_counts.tsv), from `run/sass_check.sh`): 136–167 registers at R = 1,
  212–253 at R = 2; no stack and no local memory in any of the 28 instantiations (2–8 tokens, R = 1 and 2, with and
  without the prefetch).
- **The loop, four tokens, R = 1, with the prefetch** (one step: one super-block of the lane's row for four tokens):
  256 dp4a, 160 shared loads (128 of activations: two 128-bit, one 32-bit and one 64-bit a block and token; 32 of the
  weights' words), 8 128-bit shared stores and 8 128-bit global loads of staging, 64 FFMA, 48 FMUL, 84 FADD (64 of them
  the conversions, 20 the reduction), no I2F; 1,194 instructions. Each block and token in it reads FADD (i0), FADD (i1),
  FMUL (f0 · dsc0), FFMA (f1 · dsc1 + it), FFMA (· d8 / f + acc), as the k-column kernel's I2F, I2F, FMUL, FFMA, FFMA.
- **The staging's d8 / f and the k-column quantizer's** are the same three instructions (I2F.F32.S32 of f, MUFU.RCP,
  FMUL.FTZ) in every instantiation; and `quantize_q8_1` and `kcol::quantize_records<KQ3_K>` compute the int8 values
  and d alike ([quantizers_sass.txt](quantizers_sass.txt): the same FMUL by 1/127, MUFU.RCP, FMUL, the same rounding
  by FADD.RZ of ±0.5 and truncation, the same F2F to FP16).
- **The replay set `q3kmt`** (`scripts/record_checks.sh`): 170 cases, every Q3_K `MUL_MAT` case of `test-backend-ops`
  at 1–8 tokens (the new ones, the k-column ones, the large activations, upstream's), each case's inputs and the CPU
  backend's references recorded by the tip's `test-backend-ops` on the build machine's CPU, 12 runs
  ([replay_cases.tsv](replay_cases.tsv), [replay_runs.tsv](replay_runs.tsv)): the kernel's settings under the preset's
  k-column switches, the defaults alone, the switch off, the k-column kernel off or without Q3_K, and the fault, which
  must fail. `q3kmt.tar` 408,965,120 B, md5 `bb5c27c123efe68eae489a0e1a3a5a8c`.
- **The bit-identity harness** (`multitoken_bitid q3k`) links against the tip's tree: md5
  `0a03968ce43f4e1357b401a6c95e9484`, built from `multitoken_bitid.cpp` as committed (with the `bf16` and `f32` modes
  that C750 and C754 added the same day; the first build, before that merge, was `91d6e938…`).
- **With every switch off the CPU results are the base's:** a static native CPU build of the tip, `test-llama-archs -b
  CPU -s 1` against `5b329df16`'s build: 146 architectures' logits compared, 0 differ, with no switch and with
  `GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1`; ctest 70 of 70 (C565's exclusions).
- **Not checked:** any result on a GPU. That the sums are the k-column kernel's at one token, bit for bit, rests on the
  reading above until the rig's `bitid` step; the speed is an estimate until its `perf` step.

## The rig run (R103)

`run/rig_c726.sh` on one P106 of an idle rig, detached (C722's `rig_c722.sh` with Q3_K's arms). Its steps:

- **replay:** the recorded set `q3kmt` through `scripts/replay_checks.sh`. Passing: every run's line ends `ok`, the
  fault run's with its 2–8-token cases failing.
- **bitid:** `multitoken_bitid q3k` at `Q3K_MT=1`, `=2`, `_R=2`, `_PAD=0`, and with the switch off (the k-column kernel
  at every token count, also 0 differences: its Q3_K sums are the one-token ones). Passing: `0 with differences` each.
  Then `_FAULT=1`, the harness's own check: it must report differences.
- **perf:** `test-backend-ops perf` at the five Flash-Next Q3_K shapes, 1–5 tokens. Arms `off` (the preset: the
  k-column kernel at every token count), `on`, `nopf`, `r2`, `r2nopf`, `nw3`, `pad0`, two passes alternated;
  `perf/us_run.tsv`. The 1-token cases run the k-column kernel in every arm, so they give the spread.
- **served:** R49's profile of a served verify pass on the preset with the MTP draft, the switch off and on in the order
  off, on, on, off. `on` takes the perf step's fastest arm at 4 tokens. The tables: R49's `verify_parts.py` on each
  profiled load, then R79's `run/served_table.py --types q3_K`
  ([2026-11-11_r79_c722_p106](../2026-11-11_r79_c722_p106/README.md)).

What it decides:

- **Exact or not:** `bitid` with 0 differences at every setting, and the replay passing, say that the verify pass's
  Q3_K products give each token the k-column kernel's one-token result. Any difference is a fault to find before any
  timing counts.
- **Gain, negligible or loss:** each shape at 2–5 tokens, `on` against `off`, in the two passes, each listed by value;
  R = 1 against R = 2; the target is the 1-token time.
- **In the served step:** the Q3_K products' time per verify pass, and the step's time from the warm loads.

Variables: `RIG` the rig's ssh alias, `BUILD` the build machine's, `W` the build folder there, `R` the run folder
on the rig. The model, draft, preset, server script and request are R49's.

```bash
# on the build machine (no GPU): the rig tarball, SASS, the set, the bit-identity harness, the CPU checks
WORK=$W REPO=<this repository> BRANCH=pascal-p106-v4-b12246-c726 BASE_BUILD=<5b329df16's rig build tree>/build \
  BASE_CPU=<5b329df16's static native CPU build>/bin LOCK=<a lock file> JOBS=8 bash run/checks.sh
bash run/shapes.sh 2560 6144 > shapes.tsv
# relay the tarball, $W/sets/q3kmt.tar, bench/mmvq_pascal/build/multitoken_bitid, scripts/replay_checks.sh,
# run/rig_c726.sh and R49's verify_prof.sh to the rig; then on the rig:
BIN=$R/<tarball>/bin OUT=$R/out SET=$R/q3kmt REPLAY=$R/replay_checks.sh BITID=$R/multitoken_bitid \
  MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<release/flash-next-v3/settings.env> \
  RIG_SERVER=<bench/agent_check/rig_server.sh> REQ=<2026-11-04_draft_one_card/run/greedy_request.json> \
  VERIFY_PROF=$R/verify_prof.sh setsid -f nohup bash $R/rig_c726.sh > $R/console.txt 2>&1 < /dev/null
# the served tables on the build machine: verify_parts.py of 2026-11-10_served_verify_parts on each load's
# prof/prof-verify.csv, then served_table.py of 2026-11-11_r79_c722_p106 with --types q3_K
```

## Files

| File | What |
|---|---|
| `run/checks.sh` | the build machine's steps: the rig tarball, SASS of every object against the base, the replay set, the harness, the CPU build, `test-llama-archs` against the base, ctest |
| `run/sass_check.sh` | the static check: registers, spills, the instruction mix of the kernel and of its loop, the staging's d8 / f beside the k-column quantizer's |
| `run/shapes.sh` | the launch shapes from the host code: padding, bank conflicts, warps, shared memory, blocks per SM |
| `run/rig_c726.sh` | the rig run: replay, bit identity, kernel time at 1–5 tokens, the served verify pass |
| `../../../scripts/record_checks.sh` (suite `q3kmt`) | the replay set: Q3_K's `MUL_MAT` cases at 1–8 tokens, twelve switch settings with the fault |
| `../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp` (mode `q3k`) | every 2–8-token Q3_K product against its tokens' one-token products, bit for bit |
| [summary.tsv](summary.tsv) | every step's verdict, as `run/checks.sh` wrote it |
| [sass_objects.tsv](sass_objects.tsv) | per object of the CUDA backend: the base's and the branch's SASS md5 (anonymous-namespace hashes masked), the same or not |
| [sass_counts.tsv](sass_counts.tsv) | `run/sass_check.sh`'s output for the branch's tip |
| [shapes.tsv](shapes.tsv) | `run/shapes.sh 2560 6144` |
| [replay_cases.tsv](replay_cases.tsv) | the replay set's cases (id, case string) |
| [replay_runs.tsv](replay_runs.tsv) | the replay set's runs (name, switches, ops, filter, expected verdict) |

Provenance: the code is the fork's (MIT, as llama.cpp). The test inputs are test-backend-ops' and the harness's random
tensors; the times in the first table are R41's and C731's. No model data is in this folder.
