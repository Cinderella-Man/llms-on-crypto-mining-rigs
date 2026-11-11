# Q6_K's products of 2–4 tokens, read once and exact (C727, Q1286, 2026-11-11)

What this is: a design note and the checks behind a new kernel path, not yet a measurement on a P106. It adds Q6_K to
C722's read-once kernel ([2026-11-11_c722_short_batch_products](../2026-11-11_c722_short_batch_products/README.md):
IQ4_XS's and Q4_K's products of 2–4 tokens with every weight read and decoded once, each token's sums the one-token
kernel's). Q6_K is the type of some of Qwen3.8-Flash-Next's attention output and `ssm_out` projections, whose 2–4-token
products the k-column kernel takes today, with sums other than one-token decoding's. The rig run that checks and times
it natively is the last section.

The code is on the fork's branch `pascal-p106-v4-b12246-c727` (on the v4 day merge `pascal-p106-v4-b12246-day`,
`18d0fa309`), behind its own switch, off by default: `GGML_CUDA_MMVQ_PASCAL_Q6K_MT`
([docs/switches.md](../../switches.md)). Registry row [Q1286](../../../quants/experiments.md).

## The products

- **Flash-Next's Q6_K products in a verify pass:** `attn_output` and `ssm_out`, 2,560 rows of 6,144 weights, take 1.06
  and 0.53 ms of a four-token pass's GPU time summed over the nine cards (`attn_output` 0.40 at one token;
  [2026-11-10_release_profile](../2026-11-10_release_profile/README.md), R41). Under the Flash-Next preset
  (`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`, `_KCOL_TABLE=0`) the k-column kernel takes them at 2–8 tokens, and the
  one-token Pascal kernel (`mul_mat_vec_q_pascal`) at one token, so a token's sums in a verify are not the ones it gets
  in plain decoding.
- **In a served verify pass** (R79's `off` load, [2026-11-11_r79_c722_p106](../2026-11-11_r79_c722_p106/README.md)):
  `attn_output` 1.054 ms (4 layers) and `ssm_out` 0.525 ms (2 layers) a pass.
- **Not taken here:** Q6_K `attn_k` and `attn_v` (2,560 → 512; 0.046 and 0.131 ms of that pass), the shared expert's two
  Q6_K layers (2,560 → 640; 0.061 and 0.062 ms) and the MTP draft's Q6_K output layer (2,560 → 248,320) have 2,560-wide
  rows: two lanes per row, whose 420-byte segments are not whole 16-byte units, and rows that start 4 bytes off a
  16-byte boundary. They stay on the k-column kernel (a new row, C782/Q1371).

## The design

`mul_mat_vec_q_pascal_dmt<GGML_TYPE_Q6_K, NT, R, PF>` in `ggml/src/ggml-cuda/mmvq-pascal-dmt.cu`: C722's kernel (its
lanes, order, persistent grid, activation staging, register prefetch) with Q6_K's super-block step. No new hook: the
two tests that send IQ4_XS's and Q4_K's products there (`mmvq-pascal-kcol.cu`, `mmvq.cu`) ask the same function, which
now takes Q6_K when `GGML_CUDA_MMVQ_PASCAL_Q6K_MT` is set.

- **The one-token kernel's order:** L lanes per row (the largest power of two, at most 32, that divides the row's
  super-blocks: 8 at K = 6,144), lane `sl` takes super-blocks `sl`, `sl + L`, … in that order, the lanes' sums added
  with the same xor shuffles. Per super-block, `mmvqp_dot_q6_K`'s step, for sub-block s (32 weights) and token:
  `sumf = fmaf(d8, (float) (sc0 · (isum0 − t0) + sc1 · (isum1 − t1)), sumf)`, then `acc = fmaf(d, sumf, acc)`; isum0
  and isum1 are the dp4a sums of the 6-bit codes (0–63) and the int8 activations of the sub-block's two 16-weight
  halves, t0 and t1 32 times the halves' int8 sums (`mmvqp_load_act`), sc0 and sc1 the halves' int8 scales.
- **Each 32-weight sub-block of a row decoded once** into eight words of codes (the nibbles of `ql` OR-ed with a 2-bit
  field of `qh`, as the one-token kernel decodes them) and multiplied with dp4a by every token's activations.
- **The integer expression made exact in FP32, without I2F or integer multiplies:** each half's dp4a chain starts at
  0x4B400000 − t (staged per token beside d8), so its result read as a float is 12,582,912 + (isum − t), with
  |isum − t| ≤ 16 · 32 · 128 = 2¹⁶ < 2²². Then `fmaf(12582912 + (isum − t), sc, −12582912 · sc)` is sc · (isum − t)
  rounded once, which is exact (|·| ≤ 2²³), and the two halves' sum is the integer expression exactly (|·| ≤ 2²⁴). The
  int8 scales come out of a `__byte_perm` of (scale XOR 0x80) into 12,582,912's low mantissa bits, less 12,583,040:
  (float) scale exactly. So every FP32 operation of the one-token kernel (the FFMA with d8, the FFMA with d, the
  shuffles' FADD) sees the same operands, and every token's result is the one-token kernel's bit for bit.
- **The activations once per block:** every token's int8 words (natural order), d8 and the two dp4a starts are staged
  once into shared memory per super-block position; the positions an odd number of bank groups apart, so that the L
  lanes of a group read distinct banks (two 128-bit reads of the words, one 32-bit of d8, one 64-bit of the starts a
  sub-block and token); every lane group of a warp reads the same L records, a broadcast.
- **The weights, 2-byte aligned:** Q6_K's super-blocks are 210 bytes. A row segment (L super-blocks) is staged in
  16-byte units into the warp's buffer, so rows must be 16-byte aligned and L a multiple of 8 (rows of a multiple of 8
  super-blocks: K = 2,048, 4,096, 6,144, …). Each lane reads its super-block from the staging with 32-bit loads of the
  aligned words around it, shifted by 2 bytes with a `__byte_perm` where it starts at an odd half-word, as the one-token
  kernel reads its staging (`mmvqp_lds_h`); one half of the super-block at a time (its `ql`, `qh` and scale words), to
  keep the registers down. The lane groups' segments are padded (a host search, `dmt_gstride`, now also for 32-bit
  reads) so that the 32 lanes' reads hit distinct banks: at L = 8 a 16-byte pad does it; at L = 16 and 32 no pad does,
  and the reads meet 2 words a bank.
- **One row per lane group by default** (`_DMT_R=2` shares the activation loads between two rows), the register
  prefetch of each warp's next segments at R = 1 (`GGML_CUDA_MMVQ_PASCAL_Q6K_MT=2` runs without it; at R = 2 the two
  rows' prefetched segments took 255 registers and 24 bytes of stack at 4 tokens, so R = 2 never prefetches), and a
  product whose activations leave room for one warp's staging only is left to the k-column kernel (C722's kernel lost
  46–47 % to it at one warp a block, [R79](../2026-11-11_r79_c722_p106/README.md)), unless `_DMT_NW=1`.
- **`_Q6K_MT_FAULT=1`** (checks only): every token stages the first token's activations (a column stride of 0, set on
  the host); the checks must fail.
- **IQ4_XS and Q4_K unchanged:** the type branches are compile-time, the kernel's arguments and staging code are
  C722's, and the host rules for those types are as they were; the SASS check below compares their code.

The shapes at Flash-Next's width (the host code's rules; R = 1 unless noted):

| K (L) | Tokens | R | Activations | Weight staging a warp | Warps a block | Shared memory | Blocks a SM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6,144 (8) | 2 | 1 | 17,472 B | 6,784 B | 4 | 44,624 B | 2 |
| 6,144 (8) | 3 | 1 | 25,920 B | 6,784 B | 3 | 46,288 B | 2 |
| 6,144 (8) | 4 | 1 | 34,368 B | 6,784 B | 2 | 47,952 B | 2 |
| 6,144 (8) | 2 | 2 | 17,472 B | 13,440 B | 2 | 44,368 B | 2 |
| 6,144 (8) | 3–4 | 2 | 25,920–34,368 B | 13,440 B | 1: left to the k-column kernel | | |

At four tokens a SM holds 4 warps (2 blocks of 2), the K = 6,144 case of Q1287 (C728), which tiles K for this kernel
family.

### The code

From `run/sass_check.sh` on the tip's `mmvq-pascal-dmt.cu.o` ([sass_counts.tsv](sass_counts.tsv), every Q6_K
instantiation; "a step" is the main loop: a warp's staging, its next segments' prefetch, one super-block of the lane
group's rows for every token, the reduction and the stores):

| Tokens, R, prefetch | Registers | Stack, local | dp4a | FFMA | FMUL | FADD (+ FADD32I) | I2F | shared loads | instructions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2, 1, no | 90 | 0, 0 | 128 | 50 | 16 | 26 (+ 16) | 0 | 110 | 817 |
| 2, 1, yes | 148 | 0, 0 | 128 | 50 | 16 | 26 (+ 16) | 0 | 110 | 821 |
| 3, 1, yes | 161 | 0, 0 | 192 | 75 | 16 | 39 (+ 16) | 0 | 138 | 962 |
| 4, 1, no | 116 | 0, 0 | 256 | 100 | 16 | 52 (+ 16) | 0 | 167 | 1,102 |
| 4, 1, yes | 168 | 0, 0 | 256 | 100 | 16 | 52 (+ 16) | 0 | 167 | 1,110 |
| 4, 2, no | 149 | 0, 0 | 512 | 200 | 32 | 104 (+ 32) | 0 | 220 | 2,140 |

- **The counts are the design's:** 64 dp4a a token per row (8 sub-blocks of two 4-dp4a chains), 25 FFMA a token per
  row (per sub-block the two exact products and the sumf FFMA with d8; per super-block the FFMA with d), 8 FADD a token
  per row (the halves' sums) plus the shuffle reduction's, and per row 16 scales as `PRMT …, 0x765x, 0x4b400000`,
  `FADD32I −12583040` and 16 `FMUL −12582912`. No I2F in any step (the kernel's one I2F is the staging loop's integer
  division, outside it); the XMAD in the steps are addresses.
- **No spills:** 89 to 168 registers, no stack and no local memory in the 9 instantiations (2–4 tokens; R = 1 with and
  without the prefetch, R = 2 without it).

## What was checked without a GPU

All on a build machine's CPU, in containers without a GPU, under the build machine's lock, at low priority; nothing
here ran a kernel. [summary.tsv](summary.tsv) holds every step's verdict (`run/checks.sh`).

- **The rig tarball builds** (`scripts/build_llamacpp_docker.sh`, `TESTS=1`, compute 6.1 only, CPU code without AVX):
  `0e113d1f3`, the branch's tip, `llama.cpp-20261111-0e113d1f3-pascal-p106-v4-b12246-c727.tar.gz`, 672,374,961 B, md5
  `95ad9981f03eb7bbb9ff086e6682c314`, 0 AVX instructions, glibc 2.38 at most; its `test-backend-ops` md5
  `64de5bd83a1fd9b97ef655ee9785082c`.
- **Every other kernel's code is the base's** ([sass_binary.tsv](sass_binary.tsv), [sass_dmt.tsv](sass_dmt.tsv),
  [sass_differ.txt](sass_differ.txt); `run/compare_sass.py`): `cuobjdump -sass` of both tarballs' `bin/llama-server`,
  which links every kernel, function by function with the anonymous namespaces' hashes masked: the base's 8,902
  functions all the same in the tip (IQ4_XS's and Q4_K's 24 instantiations of the same kernel among them), and 9 new,
  the Q6_K instantiations.
- **The emulation** ([q6k_emul.tsv](q6k_emul.tsv), [q6k_exact.tsv](q6k_exact.tsv); `run/q6k_emul.c`): the one-token
  kernel's and the new kernel's arithmetic from the same super-block bytes, read as each kernel reads them (32-bit words
  around a 2-byte-aligned super-block in a 16-byte-aligned staging, shifted at odd half-words), with the same lanes,
  order and shuffles. At Flash-Next's 2,560 × 6,144 and at 8, 16 and 32 lanes per row (K = 2,048, 4,096, 8,192,
  12,288), random and extreme weights (every code 0 or 63, scales −128 or 127) times normal, large (uniform × 250,000)
  and extreme (±127) activations: 30 products, 61,440 outputs, **0 differ**; 8,847,360 sub-blocks, 0 where x0 + x1 or a
  scale is not the one-token kernel's (float) integer expression or (float) scale. Both controls differ in all 30
  products (the one-token arithmetic at half the lanes: 295 to 1,464 outputs a product; the halves folded inexactly,
  `fmaf(F1, sc1, x0 − 12582912 · sc1)`: 482 to 4,094). And exhaustively: the dp4a start's float for every
  e in [−65,536, 65,536] (131,073 cases), (float) scale for all 256 bytes, `fmaf(12582912 + e, sc, −12582912 · sc)` =
  (float) (sc · e) for every scale and every such e (33,554,688), and x0 + x1 = (float) (sc0 · e0 + sc1 · e1) at 100
  million random and extreme operands: 0 differ.
- **The replay set `q6kmt`** (`scripts/record_checks.sh`): 139 cases, every Q6_K `MUL_MAT` case of `test-backend-ops`
  at 1–5 tokens (the new ones, the k-column ones, the large activations, upstream's), each case's inputs and the CPU
  backend's references recorded by the tip's `test-backend-ops` on the build machine's CPU, 13 runs
  ([replay_cases.tsv](replay_cases.tsv), [replay_runs.tsv](replay_runs.tsv)): the kernel's settings under the preset's
  k-column switches (`=1`, `=2`, `_DMT_R=2` with and without the prefetch switch, `_DMT_NW=1` and `=2`, beside
  `_DMT=1`), the defaults alone, the switch off, Q6_K left out (`_DMT_SKIP`, `GGML_CUDA_MMVQ_PASCAL_SKIP`), the fault
  without the switch (must pass) and the fault (must fail). `q6kmt.tar` 493,434,880 B, md5
  `60b8a228ccd75fa323c3d2655aac90fd`.
- **The bit-identity harness** (`multitoken_bitid q6k`) links against the tip's tree: md5
  `0d40552d75cdd6e8ed9eb24e2cee72ae`.
- **With every switch off the CPU results are the base's:** a static native CPU build of the tip, `test-llama-archs -b
  CPU -s 1` against the logits the base's own static CPU build wrote: 146 architectures compared, 0 differ, with no
  switch, with `GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1` and with its fault too; ctest 70 of 70 (C565's exclusions).
- **The first tip, `6a5686dd7`, failed two of these** and was superseded the same hour: the fault was made in the
  staging code (each token took the next token's activations), and with it `mul_mat_vec_q_pascal_dmt<Q4_K, 4, 1,
  false>` compiled to other code than the base's; and the Q6_K instantiation at four tokens, R = 2 with the prefetch
  took 255 registers and 24 bytes of stack. `0e113d1f3` makes the fault on the host and runs R = 2 without the
  prefetch; every check above is the second tip's.
- **Not checked:** any result on a GPU. That the sums are the one-token kernel's on a P106 rests on the emulation and
  the reading above until the rig's `bitid` step; the speed is unknown until its `perf` step.

## The rig run (R117)

`run/rig_c727.sh` on one P106 of an idle rig, detached (C726's `rig_c726.sh` with Q6_K's arms). Its steps:

- **replay:** the recorded set `q6kmt` through `scripts/replay_checks.sh`. Passing: every run's line ends `ok`, the
  fault run's with its 2–4-token cases failing.
- **bitid:** `multitoken_bitid q6k` at `Q6K_MT=1`, `=2`, `_DMT_R=2`, `_DMT_NW=1`: `0 with differences` each. Then the
  controls, which must report differences: the switch off (the k-column kernel at 2–4 tokens) and `_FAULT=1`.
- **perf:** `test-backend-ops perf` at Flash-Next's Q6_K shape (2,560 × 6,144), 1–5 tokens, arms `off` (the preset),
  `on`, `nopf`, `r2`, `nw1`, `nw1nopf`, two passes alternated; `perf/us_run.tsv`. The 1-token cases run the one-token
  kernel and the 5-token ones the k-column kernel in every arm, so they give the spread.
- **served:** R49's profile of a served verify pass on the preset with the MTP draft, the switch off and on in the order
  off, on, on, off; `on` takes the perf step's fastest arm at 4 tokens. The tables: R49's `verify_parts.py` on each
  profiled load, then R79's `run/served_table.py --types q6_K`
  ([2026-11-11_r79_c722_p106](../2026-11-11_r79_c722_p106/README.md)).

Decides: exact or not (any difference is a fault to find first); gain, negligible or loss at 2–4 tokens against `off`,
each pass by value; the Q6_K products' time per verify pass.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict and detail (`run/checks.sh`) |
| [sass_binary.tsv](sass_binary.tsv) | `bin/llama-server`'s GPU functions, base against tip: the same, different, only in one |
| [sass_dmt.tsv](sass_dmt.tsv) | every instantiation of `mul_mat_vec_q_pascal_dmt`: code hash in each build, registers, stack, local memory |
| [sass_differ.txt](sass_differ.txt) | every function only in one build or different, and whether expected (the 9 new ones) |
| [sass_counts.tsv](sass_counts.tsv) | the Q6_K instantiations' registers and instruction counts, whole kernel and main loop (`run/sass_check.sh`) |
| [q6k_emul.tsv](q6k_emul.tsv) | the emulation: per product and sample, outputs that differ, sub-blocks whose terms differ, the two controls |
| [q6k_exact.tsv](q6k_exact.tsv) | the exhaustive checks of the exact conversions |
| [replay_cases.tsv](replay_cases.tsv) | the replay set's 139 cases (record id, case) |
| [replay_runs.tsv](replay_runs.tsv) | its 13 runs: name, switches, ops, case filter, pass or fail expected |
| `run/checks.sh` | the check chain on a build machine (build, sass, cpu, off, ctest, record, bitid, emul) |
| `run/compare_sass.py` | the function-by-function comparison of the two builds' SASS |
| `run/sass_check.sh` | the Q6_K instantiations' registers and instruction counts |
| `run/q6k_emul.c` | the CPU emulation of both kernels' Q6_K arithmetic |
| `run/rig_c727.sh` | the rig run (R117): replay, bitid, perf, served |

## Commands

On a build machine with Docker (no GPU needed), from this repository, with a llama.cpp clone of the fork in
`$WORK/rig/src` (remote `fork` with an https URL; a clone of an earlier build's tree with `git clone --local` works, its
`origin` set back to `https://github.com/ggml-org/llama.cpp`) and, to save time, an earlier build's ccache copied to
`$WORK/rig/ccache`:

```bash
WORK=<work dir> REPO=$PWD BRANCH=pascal-p106-v4-b12246-c727 \
  BASE_TARBALL=<the day merge's rig tarball, llama.cpp-20261111-18d0fa309-pascal-p106-v4-b12246-day.tar.gz> \
  BASE_LOGITS=<the logits-*.f32 that the day merge's static CPU build's test-llama-archs wrote> \
  CPU_CCACHE_SEED=<an earlier CPU build's ccache> LOCK=<flock file> JOBS=8 \
  bash docs/data/2026-11-11_c727_q6k_readonce/run/checks.sh
```

The emulation alone, on any x86-64 machine with GCC 12 or later:

```bash
gcc -O2 -ffp-contract=off -o q6k_emul docs/data/2026-11-11_c727_q6k_readonce/run/q6k_emul.c -lm
./q6k_emul rows > q6k_emul.tsv && ./q6k_emul exact > q6k_exact.tsv
```

On a P106 rig, the rig run (its header lists the inputs; the tarball, the set and the harness come from the build
machine):

```bash
BIN=<tarball>/bin OUT=<out> SET=<q6kmt folder> REPLAY=scripts/replay_checks.sh BITID=<multitoken_bitid> MODEL=<model> \
  DRAFT=<MTP draft> PRESET=release/flash-next-v3/settings.env RIG_SERVER=bench/agent_check/rig_server.sh \
  REQ=docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
  VERIFY_PROF=docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh \
  setsid -f nohup bash docs/data/2026-11-11_c727_q6k_readonce/run/rig_c727.sh > console.txt 2>&1 < /dev/null
```

## Provenance and licence

Everything here was produced by the scripts in `run/` from the fork's code (MIT, as llama.cpp) and random weights and
activations made by `test-backend-ops`, `multitoken_bitid` and `run/q6k_emul.c` from fixed seeds; no model file was
read. The tables are this repository's (MIT).
