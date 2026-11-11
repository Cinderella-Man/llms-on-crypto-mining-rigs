# Q5_K's one-token products through C690's kernel, exact (C767, Q1348, 2026-11-11)

What this is: a code change and its checks on a build machine without a GPU, and the rig run that will time it. At the
shape of Qwen3.8-Flash-Next's output layer (`output`, Q5_K, 2,560 × 248,320, 437 MB on the last card), a P106-090 took
3,023.54 µs for one token on the one-token Pascal kernel (the median of 12 runs), and 2,575.94 and 2,574.12 µs for two
tokens on C690's multi-token kernel at one row per lane group without the prefetch
([R75](../2026-11-11_r75_c690_p106/README.md)): 437 MB read at about 170 GB/s against 145 (a P106-090 reads about
173). C690's kernel gives every token the one-token kernel's sums bit for bit, so running one-token products on it
changes no result. One-token products of the output layer run in every plain decode step and in the MTP draft's passes
that read the whole head, and in every model with a Q5_K head.

The code is on the fork's branch `pascal-p106-v4-b12246-c767` (`b58a98253`, one commit on `pascal-p106-v4-b12246-c766`
`9cb1b4a14`), behind its own switch, off by default: `GGML_CUDA_MMVQ_PASCAL_Q5K_1R` ([docs/switches.md](../../switches.md)).
Registry row [Q1348](../../../quants/experiments.md).

## Result

**Every check that runs without a GPU passes: the new path gives every one-token result the one-token kernel's bits in
a CPU emulation of both kernels' arithmetic (0 of 555,264 outputs differ, while both controls differ), every kernel of
the build keeps its code (8,952 of 8,952 functions), and with the switch off the build computes what `9cb1b4a14` does.**
The native bit identity and the timing are the rig run's ([R111](#the-rig-run-r111)).

- **Exact in the emulation** ([order_emul.tsv](order_emul.tsv), [run/order_emul.c](run/order_emul.c)): both kernels'
  FP32 arithmetic on random Q5_K rows, at Flash-Next's four Q5_K shapes and at every lane count (1 to 32 lanes per row),
  with normal and large activations: 0 of 555,264 outputs differ, and in none of the 45,879,296 sub-blocks is the new
  path's `x`, `sc` or `m` other than the one-token kernel's `(float) (isum * sc)`, `(float) sc`, `(float) m`. Two
  controls show that the emulation sees a change of order: another lane count changes outputs in 20 of the 22 products
  (the two with one super-block a row have nothing to reorder), and the super-block's last expression left uncontracted
  changes outputs in all 22 (at the output layer's shape 179,308 and 115,714 of 248,320 outputs with normal activations).
- **Every other kernel unchanged** ([sass_objects.tsv](sass_objects.tsv), [sass_binary.tsv](sass_binary.tsv),
  [sass_differ.txt](sass_differ.txt)): CUDA 12.9's `cuobjdump` of this rig build and `9cb1b4a14`'s (C766's), sm_61.
  8,952 of 8,952 functions in 193 objects the same SASS; 192 of 193 objects the same PTX; `bin/llama-server` 8,847 of
  8,847 kernels the same. What is new is only the four one-token instantiations, in `mmvq-pascal-q5k.cu`'s object
  (whose PTX holds them) and in `llama-server`.
- **The new instantiations** ([sass_q5k.tsv](sass_q5k.tsv), [sass_counts.tsv](sass_counts.tsv), C690's
  [sass_check.sh](../2026-11-11_c690_head_four_rows/run/sass_check.sh)): no stack and no local memory in any;
  the 12 earlier ones keep their code and registers.

  | `mul_mat_vec_q5_K_pascal_mt<1, R, PF>` | registers | dp4a | shared loads | FFMA | FADD | I2F | the super-block's last two FFMA |
  |---|---:|---:|---:|---:|---:|---:|---|
  | R = 1, no prefetch (the default) | 105 | 72 | 35 | 26 | 21 | 2 | `FFMA.FTZ R27, R27, R29, R68; FFMA.FTZ R68, R53, -R26, R27` |
  | R = 1, prefetch | 139 | 72 | 35 | 26 | 21 | 2 | `FFMA.FTZ R91, R52, R58, R91; FFMA.FTZ R91, R50, -R48, R91` |
  | R = 2, no prefetch | 139 | 136 | 46 | 52 | 42 | 2 | `FFMA.FTZ R9, R9, R11, R44; FFMA.FTZ R44, R6, -R12, R9` |
  | R = 2, prefetch | 242 | 136 | 46 | 52 | 42 | 2 | `FFMA.FTZ R35, R161, R38, R35; FFMA.FTZ R35, R130, -R37, R35` |
  | the one-token kernel `mul_mat_vec_q_pascal<Q5_K, 1, 1, staged>` | | | | | | | `FFMA.FTZ R2, R12, R2, R93; FFMA.FTZ R93, R5, -R3, R2` |

  In the code at R = 1: 64 dp4a for a super-block's 8 sub-blocks and 8 in the activations' staging loop (their block
  sums); 3 FFMA per sub-block (`x`, `sumd`, `summ`) and the 2 of the super-block's expression, ending as the one-token
  kernel's does (`t = FFMA(sumd, d, acc)`, then `acc = FFMA(summ, -dmin, t)`); 16 FADD for the scales' and mins'
  conversions and 5 for the lane shuffles; the two I2F are outside the loop (the block sums' conversion and the staging
  loop's division).
- **Off = the base:** `test-llama-archs -b CPU -s 1` gives `9cb1b4a14`'s logits on 146 of 146 random models (151 tests
  pass); ctest 70 of 70 (the static CPU build; the tests that need a download, the network or torch left out, as in
  C565) ([summary.tsv](summary.tsv)).
- **The replay set `q5k1r`** ([replay_cases.tsv](replay_cases.tsv), [replay_runs.tsv](replay_runs.tsv)): 29 cases (every
  Q5_K `MUL_MAT` case of one token, the output layer's and Flash-Next's other Q5_K shapes among them), recorded with
  this rig build's `test-backend-ops` on the build machine's CPU, 526,018,560 B, md5
  `1971666b3feaaf237df9cd146e044610`; 11 runs: the switch at its settings, with the 2-4-token switch, without the
  preset's k-column switches, off, Q5_K left out, and the fault, which must fail cases.
- **The bit-identity harness** `multitoken_bitid` (new mode `q5k1`) links against this build (md5
  `7734d7ac17165463244e090ea6c4f140`).
- **The rig tarball:** `llama.cpp-20261111-b58a98253-pascal-p106-v4-b12246-c767.tar.gz`, 656,692,645 B, md5
  `54f87e7ec5f4fbc1678ba359887c6fb4` (`TESTS=1`, sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38,
  `test-llama-archs` in the tree's `bin/`).

**What it should win, est.:** at the output layer's shape C690's kernel read 437 MB in 2,575.94 and 2,574.12 µs for two
tokens (R75); one token does half its dp4a and FFMA, so est. 2.58 ms or less against the one-token kernel's 3,023.54 µs,
est. 0.45 ms or more per product in `test-backend-ops` (memory-bound at 173 GB/s: est. 2.53 ms). In a served one-token
pass the output layer's product takes 2.86 ms (R41), so est. about 0.3 ms a plain decode step there. For
`attn_output` and `ssm_out` (6,144 → 2,560, 8 lanes per row) and the 2,560 → 512 products nothing is measured yet; the
rig run times them, and `_Q5K_1R_ROWS=100000` keeps the switch to the output layer if they lose.

## The change

- **The kernel at one token.** `mul_mat_vec_q5_K_pascal_mt<NT, R, PF>` ([C690](../2026-11-11_c690_head_four_rows/README.md))
  gets four more instantiations, NT = 1 at R = 1 and 2, with and without the prefetch. The code is the same template
  with one column: no line of the kernel changed.
- **The switch.** `ggml_cuda_mmvq_pascal_q5k_mt_takes` (in `mmvq-pascal-q5k.cu`) also takes one-token products when
  `GGML_CUDA_MMVQ_PASCAL_Q5K_1R=1`, and then the hooks that already send 2–4-token products to the kernel send these:
  `ggml_cuda_mul_mat_vec_q` (`mmvq.cu`) runs it after quantizing the activations, before the one-token kernel; the
  k-column kernels leave the products it takes to it (`mmvq-pascal-kcol.cu`, a test that was there for 2–4 tokens).
  `mmvq.cu` and `mmvq-pascal-kcol.cu` change only in a comment, with the same line count.
- **Its settings** are the 2–4-token kernel's: `_Q5K_MT_R` (rows per lane group, default 1), `_Q5K_MT_PF` (the register
  prefetch, default off), `_Q5K_MT_NW` (warps per block, default the most), whatever `_Q5K_MT` itself is.
  `_Q5K_1R_ROWS=n` limits it to products of at least n rows (`100000`: the output layer alone).
- **The fault** `_Q5K_1R_FAULT=1`: the host gives the kernel one iteration fewer per row (each row's last L super-blocks
  left out), for the one-token products this switch takes. It changes no kernel; the replay and the bit-identity run
  must catch it, which shows that the switch sends those products to the kernel.
- **What it takes:** plain `MUL_MAT` of a Q5_K matrix by one F32 column (no fusion, no `ids`, one channel and sample),
  rows 16-byte aligned, on compute capability 6.1 (or any NVIDIA GPU with `GGML_CUDA_MMVQ_PASCAL=2`), Q5_K not in
  `GGML_CUDA_MMVQ_PASCAL_SKIP`, and the activations and one warp's staging within 48 KB of shared memory: K up to 32,256
  weights at R = 1. In Flash-Next that is the output layer (2,560 → 248,320), `attn_output` and `ssm_out`
  (6,144 → 2,560), and `attn_k` and `attn_v` (2,560 → 512) and the shared expert's Q5_K gate or up (2,560 → 640) where
  they run as plain products. It does not take what reaches other launches first: the Q/K/V launch of
  `GGML_CUDA_ATTN_BLOCK` (on by default on a P106: the attention projections of a one-token pass where the graph has its
  pattern) and a gate and up of one type fused with their GLU.
- **Off, nothing changes:** with the switch at 0 the test returns false for one token as before, and no kernel's code
  changes (below).
- **`test-backend-ops`:** Flash-Next's other Q5_K shapes (6,144 → 2,560, 2,560 → 512 and 640) join the Q5_K cases at
  1–4 tokens, and 6,144 → 2,560 and 2,560 → 512 the perf cases at 1–8 tokens.

### Why each result is the one-token kernel's, bit for bit

The one-token kernel (`mul_mat_vec_q_pascal<Q5_K, R = 1, C = 1, staged>`) and the new path give a row the same lanes
and order:

- **Lanes:** L, the largest power of two up to 32 that divides the row's super-blocks, the same rule in both; lane sl
  takes super-blocks sl, sl + L, … in order, and the lanes' sums are added with the same xor shuffles (offsets below L).
- **One super-block:** per 32-weight sub-block s, `sumd = fmaf(d8, (float) (isum * sc), sumd)` and
  `summ = fmaf((float) m, S, summ)` (S = d8 · the sum of the int8 activations, in FP32 in both); then
  `acc = fmaf(-dmin, summ, fmaf(d, sumd, acc))`. The new path forms `(float) (isum * sc)`, `(float) sc` and `(float) m`
  without I2F, exactly (C690's design: the dp4a chain starts at `0x4B400000`, so its result read as a float is
  12,582,912 + isum, and `fmaf(12582912 + isum, sc, -12582912 · sc)` is isum · sc, which needs at most 23 bits).
- So every FP32 operation sees the same operands in the same order. On a P106 C690's kernel gave 0 differences in 21
  comparisons at 2–4 tokens at three settings (R75), the same template this change instantiates at one token.

## The rig run (R111)

`run/rig_c767.sh` on card 0 of an idle rig, detached, with the branch's tarball, the `q5k1r` set and the harness. Its
steps:

- **replay:** the recorded set through `scripts/replay_checks.sh`. Passing: every run's line ends `ok` (`q5k1r-fault`'s
  `ok` means it failed cases, as it must).
- **bitid:** `multitoken_bitid q5k1`: first the one-token kernel writes every product's values (`_Q5K_1R=0`), then each
  setting compares with them (`on`, `_R=2`, `_PF=1`, both, `_NW=1`, `_ROWS=100000`, without the preset's k-column
  switches): `0 with differences` each; `_FAULT=1` must differ (exit 1). Then `multitoken_bitid q5k` with
  `_Q5K_MT=1 _Q5K_1R=1`: every 2–4-token product against its columns' one-token products, now on the same kernel.
- **perf:** `test-backend-ops perf` at one token at 248,320 × 2,560, 2,560 × 6,144 and 512 × 2,560, arms `off on r2 pf
  nw4`, three passes alternated (`perf/us_run.tsv`); `off` is the one-token kernel.
- **served:** a plain decode step under the profiler (`run/decode_prof.sh`: the v3 preset without the MTP draft, so every
  generated token is a one-token pass), loads `off on on off`, each `warm prof`; `on` takes the perf step's fastest arm
  at the output layer.

What it decides:

- **Exact or not:** `bitid` with 0 differences at every setting (and the fault caught), and the replay passing, say that
  the switch gives every one-token Q5_K product the one-token kernel's bits on a P106. In the served step the greedy
  replies must carry the same text off and on. Any difference is a fault to find before any timing counts.
- **Gain, negligible or loss** per shape: `on` against `off` in each pass, by value; the spread is the passes' own.
  At the output layer the target is two tokens' 2,575.94 and 2,574.12 µs or less.
- **In the served step:** the output layer's product per pass (R41: 2.86 ms) from R49's `verify_parts.py` on each
  trace (it reads a trace without draft passes as steps of one pass each), every Q5_K product by name with
  R75's `served_table.py`, and the step from the warm loads' server timings.

Variables: `RIG` the rig's ssh alias, `BUILD` the build machine's, `W` the build folder there, `R` the run folder on the
rig. The model, preset, server script and request are R49's and R75's.

```bash
# on the rig
BIN=$R/<tarball>/bin OUT=$R/out SET=$R/set/q5k1r REPLAY=$R/scripts/replay_checks.sh BITID=$R/multitoken_bitid \
  MODEL=<Swift Q2_0 first shard> PRESET=<release/flash-next-v3/settings.env> RIG_SERVER=<bench/agent_check/rig_server.sh> \
  REQ=<2026-11-04_draft_one_card/run/greedy_request.json> DECODE_PROF=$R/decode_prof.sh \
  setsid -f nohup bash $R/rig_c767.sh > $R/console.txt 2>&1 < /dev/null
# then on the build machine, per profiled load (k-arm = 1-off, 2-on, 3-on, 4-off), as R75's tables
python3 -I docs/data/2026-11-10_served_verify_parts/run/verify_parts.py --kpass-parts docs/data/2026-11-04_kpass_profile/run/kpass_parts.py \
  --prof-summary scripts/prof_summary.py --event-us 7.6 --out $W/parts/<k-arm> $W/traces/<k-arm>/prof/prof-verify.csv > $W/parts/<k-arm>.md
python3 -I docs/data/2026-11-11_r75_c690_p106/run/served_table.py $W/served --parts $W/parts --out $W/tables > served.md
```

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict: the rig tarball, the SASS comparison, the CPU build, the logits identity, ctest, the replay set, the harness, the emulation |
| [sass_objects.tsv](sass_objects.tsv) | per CUDA object of the build tree: functions in each build, the same, different, in one only; the PTX the same or not |
| [sass_binary.tsv](sass_binary.tsv) | the same for the rig tarball's `bin/llama-server` |
| [sass_differ.txt](sass_differ.txt) | every function in one build only, and whether this change expects it |
| [sass_q5k.tsv](sass_q5k.tsv) | the 16 instantiations of `mul_mat_vec_q5_K_pascal_mt<NT, R, PF>`: a hash of each build's code, the same or new, registers |
| [sass_counts.tsv](sass_counts.tsv) | C690's `sass_check.sh` on this build's Q5_K object: registers, spills, instruction counts of one super-block, the last two FFMA, beside the one-token kernel's |
| [order_emul.tsv](order_emul.tsv) | the CPU emulation: per product and activation sample, the outputs, how many differ, the sub-blocks whose conversions differ, the two controls |
| [replay_cases.tsv](replay_cases.tsv) | the replay set's 29 cases (id, case string) |
| [replay_runs.tsv](replay_runs.tsv) | its 11 runs (name, switches, ops, filter, expected verdict) |
| [run/checks.sh](run/checks.sh) | the build machine's steps: the rig tarball, the SASS dumps and comparison, the CPU build, the logits identity, ctest, the replay set, the harness, the emulation |
| [run/compare_sass.py](run/compare_sass.py) | compares two builds' `cuobjdump` output function by function, with this change's expectations |
| [run/order_emul.c](run/order_emul.c) | the emulation |
| [run/rig_c767.sh](run/rig_c767.sh) | the rig run: replay, bit identity, kernel time at one token, the served plain decode step |
| [run/decode_prof.sh](run/decode_prof.sh) | the served plain decode step under the profiler (R49's `verify_prof.sh` without the draft) |
| `../../../scripts/record_checks.sh` (suite `q5k1r`) | the replay set |
| `../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp` (mode `q5k1`) | every one-token product against the one-token kernel's, through a reference file, bit for bit |

## Commands

On the build machine, CPU only (Docker with `--cpu-shares=256`, `nice -n 19`, `-j 8`, no GPU, `cuobjdump` from the
`nvidia/cuda:12.9.2-devel-ubuntu24.04` image), the work folder `WORK` holding in `rig/src` a clone of llama.cpp with
the fork as remote `fork`. The base's builds are those of [C766](../2026-11-11_c766_q5k_default/README.md)
(`9cb1b4a14`: its rig build tree and tarball, and its static CPU build):

```bash
WORK=$WORK REPO=<this repository> BRANCH=pascal-p106-v4-b12246-c767 \
  BASE_CPU=<bin/ with 9cb1b4a14's test-llama-archs> \
  BASE_OBJ=<9cb1b4a14's rig build tree>/build/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir \
  BASE_BIN=<9cb1b4a14's rig tarball tree>/bin \
  LOCK=<a flock file> JOBS=8 DOCKER_ARGS='--label pascal-quants=1' bash docs/data/2026-11-11_c767_q5k_one_row/run/checks.sh
```

It ran 2026-11-11 13:57:29–14:11:35.

## Provenance and licence

Everything here comes from building the fork's code (MIT, as llama.cpp) and from `test-backend-ops`' and the
emulation's random inputs (a fixed seed); no model file is read beyond `test-llama-archs`' own random models. The
speeds quoted are R75's and R41's, measured on a P106-090. The tables and scripts here are this repository's licence.
