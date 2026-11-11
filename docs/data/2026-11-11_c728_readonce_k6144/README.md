# The read-once kernels at K = 6,144: activations in K tiles (C728, Q1287, 2026-11-11)

What this is: a design note and the checks behind a change to C722's kernel
([2026-11-11_c722_short_batch_products](../2026-11-11_c722_short_batch_products/README.md): IQ4_XS's and Q4_K's
products of 2–4 tokens with every weight read and decoded once, each token's sums the one-token kernel's), not yet a
measurement on a P106. The rig run that checks and times it natively is the last section.

The code is on the fork's branch `pascal-p106-v4-b12246-c728` (`aa8879e50`, on the v4 day merge
`pascal-p106-v4-b12246-day`, `18d0fa309`, so it carries C758's R = 1 default; the checks below ran on `894ed8d1c`, and the
last two commits change only a host condition for merges with other branches: the tiles decline token counts other than
2–4, which this branch never passes them), behind its own switch, off by default:
`GGML_CUDA_MMVQ_PASCAL_DMT_KT` ([docs/switches.md](../../switches.md)). Registry row
[Q1287](../../../quants/experiments.md).

## Why K = 6,144 falls behind

Qwen3.8-Flash-Next's `ssm_out` (IQ4_XS and Q4_K layers) and `attn_output` (Q4_K) are 2,560 × 6,144 products. On a
P106, at C722's kernel with one row per lane group and its register prefetch (R79's `r1` arm,
[2026-11-11_r79_c722_p106](../2026-11-11_r79_c722_p106/README.md), µs per product, two passes; one token is the
one-token kernel):

| Type | Rows × K | 1 token | 2 tokens | 3 tokens | 4 tokens | 4 tokens / 1 token |
|---|---|---:|---:|---:|---:|---:|
| IQ4_XS | 2,560 × 6,144 | 75.83 | 80.35, 80.70 | 88.96, 88.93 | 113.15, 113.08 | 1.49 |
| Q4_K | 2,560 × 6,144 | 66.00 | 73.53, 74.16 | 98.79, 99.31 | 141.15, 142.40 | 2.14 |
| IQ4_XS | 6,144 × 2,560 | 74.88 | 94.00, 95.42 | 85.79, 85.44 | 92.70, 93.51 | 1.24 |
| Q4_K | 6,144 × 2,560 | 67.43 | 64.04, 64.13 | 64.51, 64.29 | 76.70, 77.40 | 1.14 |

Both shapes hold the same 15.7 million weights. The difference is how many warps a SM runs. C722's kernel stages all
tokens' activations of a row once per block, so their size grows with K, and what is left of a block's 48 KB holds
the warps' weight staging (4,352 B a warp for IQ4_XS, 4,608 B for Q4_K, at R = 1):

| Type | Tokens | Activations at K = 2,560 | Warps a block | Activations at K = 6,144 | Warps a block | Warps a SM (2 blocks) | R79, × 1 token |
|---|---:|---:|---:|---:|---:|---:|---:|
| IQ4_XS | 2 | 5,968 B | 6 | 14,304 B | 6 | 12 | 1.06 |
| IQ4_XS | 3 | 8,848 B | 6 | 21,216 B | 6 | 12 | 1.17 |
| IQ4_XS | 4 | 11,728 B | 6 | 28,128 B | 4 | 8 | 1.49 |
| Q4_K | 2 | 6,640 B | 6 | 15,936 B | 6 | 12 | 1.11 |
| Q4_K | 3 | 9,840 B | 6 | 23,616 B | 5 | 10 | 1.50 |
| Q4_K | 4 | 13,040 B | 6 | 31,296 B | 3 | 6 | 2.14 |

The slow cases are exactly those with fewer than 12 warps a SM. At 12 warps the kernel runs at about its load-store
count: C722's note puts a four-token IQ4_XS warp iteration at est. 390 load-store clocks (R = 1); 384 warp iterations a
SM at est. 1.7 GHz is est. 88 µs, and the 6,144 × 2,560 product took 92.70. Below 12 warps it waits on latency.

## The design

`mul_mat_vec_q_pascal_dmt_kt<type, NT, M, PF>` in `ggml/src/ggml-cuda/mmvq-pascal-dmt.cu`, beside C722's kernel (the
new code follows C722's; no device line of the existing kernels moved):

- **A K tile:** the activations of one iteration of the lanes, L super-block positions of every token (L = 8 lanes per
  row at K = 6,144, so 3 tiles a row). The block's warps step through the tiles together.
- **Rounds:** warp w of the block takes M warp row blocks for the block's round c. For each tile t = 0 … niter − 1 the
  block stages tile t, then each warp adds iteration t of its M row blocks. After the last tile the warps add their
  lanes' sums and write their rows, and the block's next round starts at tile 0 again. The grid is persistent: as many
  blocks as the SMs hold by registers and shared memory (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` beside the
  48 KB rule), each looping over rounds.
- **The same sums:** each lane adds its super-blocks t·L + sl in the order t = 0, 1, … with C722's super-block steps
  (`dmt_superblock_iq4_xs`, `dmt_superblock_q4_K`), the activations in the layout and with the values C722's staging
  gives them (the same int8 words, IQ4_XS's regrouped for its lookup, d8 and Q4_K's S = d8 · the sum of the int8
  values), and the lanes' sums are added with the same xor shuffles. So each token's sums are the one-token kernel's,
  bit for bit, as C722's are. With M = 2 the two row blocks' sums stay in registers and rotate after each step, so the
  step's code is there once.
- **Staging a tile:** coalesced 128-bit loads of each token's `block_q8_1` run (2,304 B a token at L = 8) into a raw
  buffer, then a conversion into the tile, position fastest, so that the 128-bit stores of the converted words hit
  distinct banks (C722's whole staging writes sub-block fastest: 8 lanes on the same banks). The next tile's loads go
  into registers (at most 4 16-byte units a thread) before the current tile's products, so their latency hides behind
  them. Two barriers a tile: after the raw stores, and after the conversion.
- **Which products:** only IQ4_XS and Q4_K products that C722's kernel takes, at R = 1 (the tiles decline any other type the file's host function passes, such as C727's Q6_K). With `GGML_CUDA_MMVQ_PASCAL_DMT_KT=1` the tiles take a
  product where they give a block more warps than the whole staging does (at Flash-Next's widths: K = 6,144 at 4 tokens,
  both types, and Q4_K's at 3 tokens); `=2` wherever they fit (the checks run every width that way). The k-column
  kernel's and C757's GLU kernel's decisions call `ggml_cuda_mmvq_pascal_dmt_takes`, whose answer does not change.
- **Switches:** `_KT_NW` warps per block (1–6, default 6; a thread loads at most 4 16-byte units of the next tile, so the tiles take 4 tokens at K = 6,144 only at 5 or 6 warps, and decline the product below that); `_KT_M=2` two row blocks a round; `_KT_FAULT=1` each
  iteration takes the next tile's activations (a fault the checks must catch). `GGML_CUDA_MMVQ_PASCAL_DMT=2` runs the
  tiles without the register prefetch of the weights, as it does C722's kernel.

Shared memory a block at K = 6,144 and 6 warps (the tile, the raw buffer, the warps' weight staging):

| Type | Tokens | Tile | Raw | Weight staging | Block | Blocks a SM | Warps a SM | Whole staging: warps a SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IQ4_XS | 3 | 7,072 B | 6,912 B | 26,112 B | 40,096 B | 2 | 12 | 12 |
| IQ4_XS | 4 | 9,376 B | 9,216 B | 26,112 B | 44,704 B | 2 | 12 | 8 |
| Q4_K | 3 | 7,872 B | 6,912 B | 27,648 B | 42,432 B | 2 | 12 | 10 |
| Q4_K | 4 | 10,432 B | 9,216 B | 27,648 B | 47,296 B | 2 | 12 | 6 |

The cost is the tiles' staging, once a round instead of once a block. Per tile at 4 tokens (the load-store clocks of
C722's note): 18 128-bit loads and 18 stores of the raw run (est. 144 clocks), and 8 warp rows of conversion, 9 32-bit
loads (2-way bank conflicts) and two 128-bit stores each (est. 224): est. 368 clocks, against est. 7,020 (IQ4_XS) and
7,704 (Q4_K) for a round's products at 6 warps, M = 1: est. +16 % and +14 %, half that at M = 2.

**Estimate** (est.; the rig run measures it): at 12 warps a SM, about the 6,144 × 2,560 shape's time plus the staging.
Q4_K at 4 tokens est. 87 µs (M = 1) and 82 (M = 2) against 141.15, at 3 tokens est. 74 µs (M = 1) against 98.79;
IQ4_XS at 4 tokens est. 108 µs (M = 1) and 100 (M = 2) against 113.15. In R79's served verify pass these products took
1.905 ms (IQ4_XS `ssm_out`), 1.543 and 0.567 ms (Q4_K `ssm_out`, `attn_output`): est. 0.90 (M = 1) and 1.10 ms
(M = 2) less a pass.

## What was checked without a GPU

On the build machine's CPU, in containers without GPUs at low priority, one step at a time under a shared lock
(`run/checks.sh`; every verdict in [summary.tsv](summary.tsv)). The base is the day merge `18d0fa309`.

| Check | Result |
|---|---|
| Rig tarball | `llama.cpp-20261111-894ed8d1c-pascal-p106-v4-b12246-c728.tar.gz`, 673,971,772 B, md5 `e58e04b0cd177939e1b10715cd9e3312`, `TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38 at most; its `test-backend-ops` md5 `d82742863341457065a25e5744bab6c3` |
| Every other function | llama-server's SASS and PTX, function by function against the base's (the per-file hashes of anonymous namespaces masked): 8,507 functions, 8,483 the base's code, the 24 `mul_mat_vec_q_pascal_dmt_kt` the only new ones, none gone or changed ([sass_functions.tsv](sass_functions.tsv)) |
| The new kernels | 24 instantiations (2 types × 2–4 tokens × M = 1, 2 × prefetch on and off), 99 to 167 registers, no stack or local memory. The super-block step's instruction counts are those of C722's kernel of the same type and tokens at R = 1 (four tokens: 256 dp4a, 68 FFMA, 200 PRMT for IQ4_XS; 264, 104, 16 for Q4_K). Q4_K ends each super-block with `t = FFMA(·, ·, acc); acc = FFMA(·, −dmin, t)`, as the one-token kernel does. I2F: Q4_K's `(float) sum` of the staging only, none in IQ4_XS (C722's kernels also hold one `I2F.F32.S32.RP` each, of the integer division in their staging, which the tiles do without). 2 barriers ([sass_counts.tsv](sass_counts.tsv)) |
| Replay set `dmtkt` | 345 cases (IQ4_XS's and Q4_K's `MUL_MAT` at 1–5 tokens: 315 hardlinked from C722's `dmt` set, the 30 new K = 6,144 cases recorded), 12 runs, one of them the fault ([replay_cases.tsv](replay_cases.tsv), [replay_runs.tsv](replay_runs.tsv)); `dmtkt.tar` 1,173,278,720 B, md5 `3f10371bc253b2cf2e1994a62505bf33`, recorded with the `test-backend-ops` of the branch's commit before last (`22dbaccc8`: the same test sources and CPU code; its CPU backend alone computes the references) |
| Bit-identity harness | `multitoken_bitid` with mode `dmtkt`, linked against the tip: md5 `ae2d12e7ab19fd9ea51af182d33d24ac`, 90,383,496 B |
| Off = base | `test-llama-archs -b CPU -s 1`: the base's static CPU build writes 146 architectures' logits, the tip's gives the same 146 with no switch and with `GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_DMT_KT=1` (CUDA switches, so the CPU path must not move) |
| ctest | 70 of 70 in the tip's CPU build (the seven tests that need a download, the network or torch not run) |

The 24 new kernels have 99 to 167 registers: at most 170 lets 2 blocks of 6 warps share a SM's 65,536 registers, so
the 12 warps a SM of the shared-memory table above hold.

Nothing here ran on a GPU: the tiles' sums, the fault's failure and the speed are the rig run's.

## The rig run (R118)

`run/rig_c728.sh` on one P106 of an idle rig, detached (C722's `rig_c722.sh` with the tiles' arms). Its steps:

- **replay:** the recorded set `dmtkt` through `scripts/replay_checks.sh`. Passing: every run's line ends `ok`, and the
  fault run fails cases (its 3–4-token cases at K = 6,144).
- **bitid:** `multitoken_bitid dmtkt` with C722's kernel alone (`DMT=1`), the tiles where they help (`_KT=1`), wherever
  they fit (`_KT=2`), with two row blocks a round, without the prefetch, and at 4 warps. Passing: `0 with differences`
  each. Then `_KT_FAULT=1`, the harness's own check: it must report differences at K = 6,144.
- **perf:** `test-backend-ops perf` at 2,560 × 6,144, both types, 1–5 tokens. Arms `off` (the preset: the k-column
  kernel), `dmt` (C722's kernel, whole staging), `kt1`, `kt1m2` (`_KT_M=2`), `kt1nw5` (`_KT_NW=5`), `kt1nopf`
  (`DMT=2`), `kt2`, two passes alternated; `perf/us_run.tsv`. The 1-token cases run the one-token kernel and the 5-token
  cases the k-column kernel in every arm, so they give the spread.
- **served:** R49's profile of a served verify pass on the preset with the MTP draft and C722's kernel (`DMT=1`), the
  tiles off and on in the order off, on, on, off. `on` takes the perf step's fastest `kt` arm at 4 tokens. The tables:
  R49's `verify_parts.py` on each profiled load, then R79's `run/served_table.py`
  ([2026-11-11_r79_c722_p106](../2026-11-11_r79_c722_p106/README.md)).

What it decides:

- **Exact or not:** `bitid` with 0 differences at every setting, and the replay passing, say that the tiles give each
  token the one-token kernel's result. Any difference is a fault to find before any timing counts.
- **Gain, negligible or loss:** each shape at 3 and 4 tokens, each `kt` arm against `dmt`, in the two passes, each
  listed by value; M = 1 against M = 2; the target is the 1-token time.
- **In the served step:** the IQ4_XS and Q4_K products' time per verify pass (`ssm_out`, `attn_output`), and the step's
  time from the warm loads.

Variables: `RIG` the rig's ssh alias, `BUILD` the build machine's, `W` the build folder there, `R` the run folder
on the rig. The model, draft, preset, server script and request are R49's.

```bash
# on the build machine (no GPU): the rig tarball, SASS, the set, the bit-identity harness, the CPU checks
WORK=$W REPO=<this repository> BRANCH=pascal-p106-v4-b12246-c728 BASE_COMMIT=18d0fa309 \
  BASE_TARBALL=<18d0fa309's rig tarball> LINK_FROM=<C722's recorded dmt set> LOCK=<a lock file> JOBS=8 \
  bash run/checks.sh
# relay the tarball, $W/sets/dmtkt.tar, bench/mmvq_pascal/build/multitoken_bitid, scripts/replay_checks.sh,
# run/rig_c728.sh and R49's verify_prof.sh to the rig; then on the rig:
BIN=$R/<tarball>/bin OUT=$R/out SET=$R/dmtkt REPLAY=$R/replay_checks.sh BITID=$R/multitoken_bitid \
  MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<release/flash-next-v3/settings.env> \
  RIG_SERVER=<bench/agent_check/rig_server.sh> REQ=<2026-11-04_draft_one_card/run/greedy_request.json> \
  VERIFY_PROF=$R/verify_prof.sh setsid -f nohup bash $R/rig_c728.sh > $R/console.txt 2>&1 < /dev/null
# the served tables on the build machine: verify_parts.py of 2026-11-10_served_verify_parts on each load's
# prof/prof-verify.csv, then served_table.py of 2026-11-11_r79_c722_p106
```

## Files

| File | What |
|---|---|
| `run/checks.sh` | the build machine's steps: the rig tarball, SASS and PTX of every function against the base's, the replay set, the harness, the base's and the tip's CPU builds, `test-llama-archs` against the base, ctest |
| `run/sass_check.sh` | the static check: registers, spills, the instruction mix of C722's kernels and of the tiles', Q4_K's last two FFMA beside the one-token kernel's |
| `run/rig_c728.sh` | the rig run: replay, bit identity, kernel time at 1–5 tokens, the served verify pass |
| `../../../scripts/record_checks.sh` (suite `dmtkt`) | the replay set: IQ4_XS's and Q4_K's `MUL_MAT` cases at 1–5 tokens, twelve switch settings with the fault |
| `../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp` (mode `dmtkt`) | every 2–4-token IQ4_XS and Q4_K product against its tokens' one-token products, bit for bit |
| [summary.tsv](summary.tsv) | every step's verdict, as `run/checks.sh` wrote it |
| [sass_functions.tsv](sass_functions.tsv) | the functions of llama-server's SASS and PTX that are not the base's code (new, gone or different), and the counts |
| [sass_counts.tsv](sass_counts.tsv) | `run/sass_check.sh`'s output for the branch's tip |
| [replay_cases.tsv](replay_cases.tsv) | the replay set's cases (id, case string) |
| [replay_runs.tsv](replay_runs.tsv) | the replay set's runs (name, switches, ops, filter, expected verdict) |

Provenance: the code is the fork's (MIT, as llama.cpp). The test inputs are test-backend-ops' and the harness's random
tensors; the times in the first tables are R79's. No model data is in this folder.
