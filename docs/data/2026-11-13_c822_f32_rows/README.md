# A row limit that follows the token count: `GGML_CUDA_MMVF_F32_MT_ROWS=auto` (C822, Q1436, 2026-11-13)

What this is: a new value for the row limit of C754's F32 switch, with its checks on a build machine's CPU, with no
GPU, and its timing on a P106-090 ([R144](#on-a-p106-r144), Q1448), which places the numbers.

`GGML_CUDA_MMVF_F32_MT=1` sends an F32 weight matrix times 4 to 8 tokens to llama.cpp's `mul_mat_vec_f` instead of
cuBLAS, for a matrix of at most `GGML_CUDA_MMVF_F32_MT_ROWS` rows (32 by default, [C807](../2026-11-12_c807_f32_router/README.md)).
On a P106-090 ([R132](../2026-11-12_c807_f32_router/README.md#perf-where-mul_mat_vec_f-stops-gaining)) `mul_mat_vec_f` beat
cuBLAS on F32 rows of 2,560:
- up to 33 rows at every token count from 4 to 8;
- at 64 rows at 5-8 tokens, but not at 4 (13.05 / 13.17 µs on cuBLAS against 13.48 / 13.58 on `mul_mat_vec_f`);
- at 128 rows only at 5 tokens.

cuBLAS is about three times faster at 4 tokens than at 5-8 (12.11 / 12.21 µs at 32 rows and 4 tokens, 34.53 / 34.49 at
5), so `mul_mat_vec_f` stops gaining at fewer rows at 4 tokens. One limit for every token count cannot follow that.

Registry row [Q1436](../../../quants/experiments.md) (from [Q1413](../../../quants/experiments.md)); fork branch
`pascal-p106-v4-b12246-c822` (`cc97bb6b6`), merged into `pascal-p106-v4` as `2144cf760`.

**The answer:**
- **`GGML_CUDA_MMVF_F32_MT_ROWS=auto` sets a limit per token count: 48 rows at 4 tokens, 64 at 5-8**, and a P106
  holds both at widths 2,560 and 2,048 ([R144](#on-a-p106-r144)). A list such as `48,64` sets one limit per token
  count from 4 tokens on, its last entry holding after it. A single number keeps C807's meaning at every token count, and **the default stays 32**.
- **Nothing changes for Qwen3.8-Flash-Next.** Its F32 matrices are 1 row (the shared expert's gate) and 512 rows (the
  router). Unset, at 32 and at `auto` the gate goes to `mul_mat_vec_f` and the router to cuBLAS at 4-8 tokens: the
  same products, so the same kernels and the same sums. The GPU code is the base's, function for function.
- **For other models:** an F32 matrix of 33-64 rows times 4-8 tokens, such as the router of a model with 32 to 64
  experts (llama.cpp keeps `ffn_gate_inp` in F32 when it quantizes), goes to `mul_mat_vec_f` at `auto`.
- **Host code only.** Every CPU check passed (below).

## The change

`ggml/src/ggml-cuda/mmvf-f32-rows.h` (new: the value's parsing and the decision, no GPU needed), `mmvf.cu` (the switch
asks it), `mmvf.cuh` (its line), `tests/test-cuda-mmvf-f32-rows.cpp` (new) and `tests/CMakeLists.txt`,
`tests/test-backend-ops.cpp` (cases).

| `GGML_CUDA_MMVF_F32_MT_ROWS` | Limit at 4 tokens | at 5 | at 6 | at 7 | at 8 |
|---|---:|---:|---:|---:|---:|
| unset | 32 | 32 | 32 | 32 | 32 |
| `N` (one number) | N | N | N | N | N |
| `0` or below | any | any | any | any | any |
| `auto` | 48 | 64 | 64 | 64 | 64 |
| `48,64` | 48 | 64 | 64 | 64 | 64 |
| `64,48` | 64 | 48 | 48 | 48 | 48 |
| `33,49,65,31,7` | 33 | 49 | 65 | 31 | 7 |

- Each number, alone or in a list, is read as `atoi` reads it, so a single number means what it meant in C807, and an
  empty or non-numeric entry is 0 (any row count). Only `auto` itself is a name.
- **A build before C822 reads the whole value with `atoi`:** `auto` is 0 there, any row count, which puts
  Flash-Next's router on `mul_mat_vec_f` (+1.668 ms a verify pass in R120b); a list reads as its first entry. Set
  `auto` only on a build from C822 on.
- The decision is `ggml_cuda_mmvf_f32_rows_take(limits, rows, tokens)`: false outside 4-8 tokens, else whether the
  token count's limit is 0 or below or at least the rows. The switch asks it where C807 compared the rows with its one
  limit; all the switch's other tests are unchanged, and with the switch off it returns before reading the value.
- **Tests:**
  - `test-cuda-mmvf-f32-rows` (in ctest): every single number from -3 to 600 and 12 other values (empty,
    non-numeric, signs, a leading zero or space, `AUTO`, `Auto`, `auto ` and ` auto`) against C807's rule, unset too,
    over every row count from 1 to 1,100 and every token count from 0 to 12; `auto` and ten lists against their
    tables (trailing and leading commas, entries past 8 tokens, 0 and negative entries), `auto` and three lists over
    the same rows and tokens; Flash-Next's two shapes at 1-8 tokens unset, at 32, at `auto` and at `48,64`; the
    replay set's shapes under `auto` and `64,48`.
  - `test-backend-ops`: F32 cases at `auto`'s edges (48 and 49, 64 and 65 rows of 2,560; 49 rows of 64, 64 rows of 100
    and 65 rows of 300, mul_mat_vec_f's other block sizes; 64 rows of 2,048), 1-8 tokens; perf cases at 40, 48 and 56
    rows of 2,560 and at 32, 33, 40, 48, 56 and 64 rows of 2,048.
  - `scripts/record_checks.sh`'s suite `f32x`: new runs at `_ROWS=auto` (`f32x-on-auto`, `f32x-fault-auto` must fail,
    `f32x-fault-auto-above` must pass, and the `=2` twins) and at the list `64,48`, the limits the other way round
    (`f32x-fault-list` must fail, `f32x-fault-list-above` must pass). They are listed only when the set holds C822's
    cases, which a build records from C822 on, so a set recorded on an older build keeps C807's 11 runs.

## Checks on the build machine

[run/chain.sh](run/chain.sh), CPU only, every step in a container without GPUs at a low CPU share, 2026-11-13
08:34-08:59; the merged tree's step 09:00-09:01. Results in [summary.tsv](summary.tsv).

| Check | Result |
|---|---|
| rig tarball | `llama.cpp-20261113-cc97bb6b6-pascal-p106-v4-b12246-c822.tar.gz`, 678,645,408 bytes, md5 `83a53352b160d2d21f47831bb3317489`; `TESTS=1 EXTRA_TARGETS=test-cuda-mmvf-f32-rows`, sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most; `llama-server` md5 `2f65512306e49218469e6e088c32b10f`, `test-backend-ops` md5 `e01a8db17d7f696f7349c97e00bb7c6b`, `test-cuda-mmvf-f32-rows` md5 `430f40e8b5703411cdb86a3b5858181f` |
| every GPU function | `cuobjdump -sass` and `-ptx` of `llama-server` and `test-backend-ops` against the `pascal-p106-v4-b12246-c817` build (`5f8dd491b`, whose `ggml/` tree is `pascal-p106-v4` `65ac9a47d`'s, the base), function by function with C103's [funcs.py](../2026-11-11_c103_union_gather/run/funcs.py): 8,540 of 8,540 the same in each of the four listings, none new, none missing |
| the rule | the rig build's `test-cuda-mmvf-f32-rows`: 9,152,269 checks, 0 failed |
| CPU, static | the base `65ac9a47d`'s `test-llama-archs` writes 146 architectures' logits; the branch builds every target, ctest 72 of 72 pass (`test-cuda-mmvf-f32-rows` among them; the 7 tests that need a download, the network or torch not run), `test-llama-archs` 151 of 151 cases, and its 146 logits files are the base's, 0 differ |
| CPU, shared libraries | every target, ctest 72 of 72, `test-llama-archs` 151 of 151, its 146 files written and compared |
| replay set `f32x` | 227 cases (F32's 2-D `MUL_MAT` at 1-8 tokens: this fork's 184, C822's 64 among them, and 43 of llama.cpp's own), 18 runs ([replay_runs.tsv](replay_runs.tsv), [replay_cases.tsv](replay_cases.tsv)), recorded on the CPU with the rig build's `test-backend-ops`; tar 472,432,640 bytes, md5 `8e1eed113ca7d1d35a8aa9224ffb70fa` |
| what each run selects | [select.tsv](select.tsv), by the run's own limit read independently of the engine (unset 32, a number, `auto`, a list): every fault run that must fail selects only cases its limit lets the switch take, and every `-above` run only cases it leaves to cuBLAS (table below) |
| CPU replay | the set's 227 cases replayed on the CPU with the rig build: 227 of 227 pass (the set's files are whole) |
| every switch off, merged | the `--no-ff` merge into `pascal-p106-v4` (`2144cf760` on `65ac9a47d`, the line's tip; its tree is the branch's, so it adds exactly the branch's 6 files): its `test-llama-archs` against `65ac9a47d`'s 146 logits files, static, 151 of 151 cases, 0 differ; its `test-cuda-mmvf-f32-rows` 9,152,269 checks, 0 failed |

The fault runs, by the cases they select (4-8 tokens; "taken": the run's limit lets the switch take the product):

| Run | `_ROWS` | Must | Cases | Taken | Left to cuBLAS |
|---|---|---|---:|---:|---:|
| `f32x-fault`, `f32x-fault2` | unset (32) | fail every case | 37 | 37 | 0 |
| `f32x-fault-above`, `f32x-fault2-above` | unset (32) | pass | 72 | 0 | 72 |
| `f32x-fault-rows0`, `f32x-fault2-rows0` | `0` | fail every case | 72 | 72 | 0 |
| `f32x-fault-auto`, `f32x-fault2-auto` | `auto` | fail every case | 79 | 79 | 0 |
| `f32x-fault-auto-above`, `f32x-fault2-auto-above` | `auto` | pass | 30 | 0 | 30 |
| `f32x-fault-list` | `64,48` | fail every case | 64 | 64 | 0 |
| `f32x-fault-list-above` | `64,48` | pass | 45 | 0 | 45 |
| `f32x-fault-off` | the switch off | pass | 109 | 0 | 109 |

`auto` and `64,48` split the same 109 cases differently: 49 and 64 rows at 4 tokens are `auto`'s "above" and the
list's "taken", and at 5-8 tokens the other way round, so a mix-up of the token counts fails a run. Two of the 37
cases at the default are llama.cpp's own (1 row of 2,048 at 7 and 8 tokens), which C807's filter did not list.

**What shows that Flash-Next's path is the same, byte for byte:**
- with the switch off nothing reads the value, and with it on the decision for every single number, the value unset
  among them, is C807's for every row count from 1 to 1,100 and every token count (`test-cuda-mmvf-f32-rows`);
- at `auto` Flash-Next's two F32 shapes are decided as at 32: the gate (1 row) taken and the router (512 rows) left to
  cuBLAS at 4-8 tokens, nothing taken at 1-3 (the same test);
- every GPU function of `llama-server` and `test-backend-ops` is the base's (8,540 of 8,540): the routing is host
  code, so the kernels a product reaches, and their sums, are the base's;
- with every switch off the 146 architectures' logits are the base's, bit for bit, static and merged.

## What the rig run must show (R144)

[run/rig_r144.sh](run/rig_r144.sh) on a P106-090 rig, the rig idle, with the branch's rig tarball (md5
`83a53352b160d2d21f47831bb3317489`) and the set `f32x` (md5 `8e1eed113ca7d1d35a8aa9224ffb70fa`); est. 20 min:

1. **replay:** the set `f32x` through `scripts/replay_checks.sh`; every one of the 18 runs ends `ok`. The fault runs
   that must fail fail every one of their cases (the table above), the `-above` runs pass all of theirs. That is the
   switch's routing at `auto` and at a list on the card.
2. **perf:** `test-backend-ops perf` at 32, 33, 40, 48, 56 and 64 rows of 2,560 and of 2,048, 1-8 tokens, arms `off`
   (cuBLAS), `on` (the default, 32 rows), `rows0` (`mul_mat_vec_f` at every row count) and `auto`, alternated over two
   passes. [run/perf_r144.py](run/perf_r144.py) classes each case against its shape's spread at 1-3 tokens (R132's
   rule) and gives, per width and token count, the largest row count up to which `mul_mat_vec_f` gains over cuBLAS in
   both passes. That places `auto`'s two numbers and the default; a change to either is a new code row.

## On a P106 (R144)

Step 2, the perf step, on one P106-090 (card 0 of a nine-card rig, nothing else running), 2026-11-13 20:38:56-20:52:41:
`test-backend-ops perf` of `v4-rc2` (`pascal-p106-v4` `eb82916d3`, which holds C822 as `2144cf760`; the rig tree built
from `eb312b7bc`, `test-backend-ops` md5 `e38809f0b16869abc065483f07c4798a`), arms `off`, `on`, `rows0`, `auto` in
that order, two passes, 96 cases an arm and pass (768 in all). Step 1, the replay, is not run here: the set `f32x` was
recorded with the b12246 test program, and upstream b12360's `test-backend-ops` names its `MUL_MAT` cases differently,
so the set does not replay on `v4-rc2`. It needs the set recorded again with `v4-rc2`'s test program.

**Where `mul_mat_vec_f` beats cuBLAS** (`rows0` against `off`, each case a gain only when faster by more than its
shape's spread in both passes; [rig/place.tsv](rig/place.tsv)):

| width | tokens | gains up to (rows) | the next row count |
|---:|---:|---:|---|
| 2,560 | 4 | 48 | 56 negligible (12.91 / 12.76 and 13.02 / 12.78 µs, cuBLAS / `mul_mat_vec_f`, passes 1 and 2; spread 3.1 %), 64 a loss (13.65 / 14.24 and 13.62 / 14.28) |
| 2,560 | 5, 6, 7, 8 | 64 | none: a gain at every row count measured |
| 2,048 | 4 | 48 | 56 a loss (10.79 / 11.55 and 10.88 / 11.59, +7.0 % and +6.5 %; spread 6.0 %), 64 a loss (10.81 / 12.31 and 10.90 / 12.34) |
| 2,048 | 5, 6, 7 | 64 | none: a gain at every row count measured |
| 2,048 | 8 | 56 | 64 negligible (27.85 / 25.78 and 28.08 / 25.83, -7.4 % and -8.0 %; spread 8.5 %) |

- **`auto`'s 48 rows at 4 tokens holds at both widths:** 48 rows gain (2,560: 12.87 / 10.70 and 12.97 / 10.72 µs;
  2,048: 10.76 / 9.66 and 10.84 / 9.67), 56 is no gain at either width and a loss at 2,048, 64 a loss at both.
- **`auto`'s 64 rows at 5-8 tokens holds:** at 2,560 every row count to 64 gains at every token count from 5 to 8; at
  2,048 the one case that is not a gain, 64 rows at 8 tokens, is negligible (faster in both passes, within the
  spread), not a loss. `auto` itself against `off` there is a gain: 25.43 and 25.43 µs against 27.85 and 28.08
  (-8.7 % and -9.4 %, spread 8.5 %).
- **`auto` against `off` is a gain at every case where they differ** (4-8 tokens at 32-48 rows and 5-8 tokens at 56 and
  64 rows, both widths), and **against `on` (the default, 32 rows) at every case where they differ** (33-48 rows at 4
  tokens, 33-64 at 5-8; 46 cases). Where `auto` leaves a product to cuBLAS (56 and 64 rows at 4 tokens), against
  `rows0` (`mul_mat_vec_f`): 56 rows negligible at both widths (2,560: 12.76 / 13.15 and 12.78 / 13.17 µs, `rows0` /
  `auto`; 2,048: 11.55 / 10.99 and 11.59 / 10.98), 64 rows a gain (2,560: 14.24 / 13.75 and 14.28 / 13.78; 2,048:
  12.31 / 11.01 and 12.34 / 10.92).
- **The default 32** keeps products of 33-64 rows on cuBLAS, which is slower at every one of them up to 48 rows at 4
  tokens and up to 64 at 5-8 (the `auto` against `on` cases).
- The spread of a shape is the largest over its 1-3-token cases, where every arm runs `mul_mat_vec_f`
  ([rig/spread.tsv](rig/spread.tsv)). Two shapes at 2,048 have wide ones, both from 2 tokens: 33 rows 10.5 % and 64
  rows 8.5 %.

Every case's µs per arm and pass is in [rig/wide.tsv](rig/wide.tsv) (1-8 tokens) and, with each comparison's class, in
[rig/perf.md](rig/perf.md).

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | the build machine's checks, one line each |
| [select.tsv](select.tsv) | each run of the set `f32x`: its switches, the cases its filter selects, how many of them its limit lets the switch take and how many it leaves (4-8 tokens), and the cases by rows × tokens. In the runs that are not fault runs, "taken" counts by the limit alone (an odd width or stride stays on cuBLAS whatever the limit) |
| [replay_runs.tsv](replay_runs.tsv) | the set's runs: switches, filter, expected verdict |
| [replay_cases.tsv](replay_cases.tsv) | the set's cases: id and case string |
| [run/chain.sh](run/chain.sh) | the build machine's checks |
| [run/rig_r144.sh](run/rig_r144.sh) | the rig run (R144) |
| [run/perf_r144.py](run/perf_r144.py) | its perf tables, classes and placement (on R132's data it gives R132's classes) |
| [rig/us_run.tsv](rig/us_run.tsv) | R144's perf on a P106: one line per arm, pass and case (arm, pass, type, rows, width, tokens, µs/run) |
| [rig/wide.tsv](rig/wide.tsv) | the same, one line per case: µs/run of every arm and pass |
| [rig/place.tsv](rig/place.tsv) | per width and token count: each row count's class (`rows0` against `off`), the largest row count up to which all gain, the first that does not |
| [rig/verdict.tsv](rig/verdict.tsv) | every comparison at 4-8 tokens (`rows0`, `auto`, `on` against `off`; `auto` against `on` and `rows0`): µs per pass, change, spread, class |
| [rig/spread.tsv](rig/spread.tsv) | each shape's spread at 1-3 tokens |
| [rig/perf.md](rig/perf.md) | `perf_r144.py`'s report: the tables above as Markdown |

Also changed for this row: `scripts/record_checks.sh` (suite `f32x`), [docs/switches.md](../../switches.md).

## Commands

On an x86-64 build machine with Docker and the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image, no GPU (`BASE` is a rig
tarball whose GPU code is the base's, here `pascal-p106-v4-b12246-c817`'s; `SRC` a llama.cpp clone with the fork as
remote `fork`; `LOCK` a lock file shared with other builds on the machine):

```bash
WORK=$WORK PUB=$PWD BASE=$BASE BASE_REF=65ac9a47d BRANCH=pascal-p106-v4-b12246-c822 LOCK=$LOCK JOBS=8 \
    DOCKER_ARGS='--label pascal-quants=1' SEED_SRC=$SRC nice -n 19 bash docs/data/2026-11-13_c822_f32_rows/run/chain.sh
# the merged tree, every switch off (the merge commit fetched into $WORK/w/src first):
WORK=$WORK PUB=$PWD BRANCH=pascal-p106-v4-b12246-c822 MERGED=2144cf760 MERGE_BASE=65ac9a47d LOCK=$LOCK JOBS=8 \
    STEPS='merged tables' nice -n 19 bash docs/data/2026-11-13_c822_f32_rows/run/chain.sh
```

On the rig (the branch's tarball unpacked, the set and the replay harness copied):

```bash
BIN=<tarball>/bin OUT=<out> SET=<set>/f32x REPLAY=scripts/replay_checks.sh \
    setsid nohup bash docs/data/2026-11-13_c822_f32_rows/run/rig_r144.sh > console.txt 2>&1 < /dev/null &
```

R144 ran the perf step alone, on `v4-rc2`'s rig tree (`BIN` its `bin/`):

```bash
STEPS=perf BIN=<v4-rc2 tree>/bin OUT=<out> \
    setsid nohup bash docs/data/2026-11-13_c822_f32_rows/run/rig_r144.sh > console.txt 2>&1 < /dev/null &
```

Its tables, on a build machine's CPU (the rig's `out/` copied to `$W/out`), from this repository's root:

```bash
mkdir -p $W/tables/perf
nice -n 19 python3 -I docs/data/2026-11-13_c822_f32_rows/run/perf_r144.py $W/out/perf/us_run.tsv --out $W/tables/perf \
    > $W/tables/perf/perf.md
```

## Provenance and licence

Code: Miner Llama (`github.com/Cinderella-Man/miner-llama`, MIT, llama.cpp's licence), branch
`pascal-p106-v4-b12246-c822` at `cc97bb6b6` (one commit on `pascal-p106-v4` `65ac9a47d`), merged into
`pascal-p106-v4` as `2144cf760`. The tests' inputs are generated (fixed seeds), not model weights. Everything here is
this repository's licence.
