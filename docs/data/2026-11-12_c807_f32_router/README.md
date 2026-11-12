# The router back on cuBLAS: a row limit for `GGML_CUDA_MMVF_F32_MT` (C807, Q1413, 2026-11-12)

What this is: a row limit for C754's F32 switch, its checks on a build machine's CPU, with no GPU, and its check on a
P106-090 rig ([R132](#the-rig-run-r132)). On a P106-090
([R120b](../2026-11-12_r120b_kern_speed/README.md)) `GGML_CUDA_MMVF_F32_MT=1` made one of its two Qwen3.8-Flash-Next
products of a four-token verify pass faster and the other slower:

| Product (47 calls a verify pass) | cuBLAS, the switch off (ms a pass) | `mul_mat_vec_f`, the switch on (ms a pass) | on − off |
|---|---|---|---:|
| the shared expert's gate `ffn_gate_inp_shexp` (2,560 → 1) | 0.610, 0.613, 0.607, 0.618 | 0.297, 0.291, 0.278, 0.276 | −0.327 |
| the router `ffn_gate_inp` (2,560 → 512) | 2.412, 2.405, 2.404, 2.410 | 4.072, 4.072, 4.073, 4.085 | +1.668 |

Registry row [Q1413](../../../quants/experiments.md) (from [Q1330](../2026-11-11_c754_f32_mmvf/README.md)); fork
branch `pascal-p106-v4-b12246-c807` (`11f8d329a`), merged into `pascal-p106-v4` as `993902ae9`.

**The answer:**
- **`GGML_CUDA_MMVF_F32_MT=1` now takes an F32 weight matrix of at most `GGML_CUDA_MMVF_F32_MT_ROWS` rows, 32 by
  default.** In Flash-Next's verify pass the shared expert's gate (1 row) stays on `mul_mat_vec_f` and the router
  (512 rows) goes back to cuBLAS, as with the switch off. `_ROWS=0` keeps C754's rule (any row count), for timing.
- **On a P106-090 the switch is now a gain: −0.420 ms a served verify pass against the switch off** (100.421,
  100.359, 100.440 and 100.377 ms a pass on, 100.824, 100.800, 100.825 and 100.826 off). The router takes cuBLAS's
  time again (2.405, 2.395, 2.402 and 2.401 ms a pass on, 2.403, 2.399, 2.398 and 2.400 off). The shared expert's gate
  takes 0.265, 0.265, 0.258 and 0.256 ms on, against 0.609, 0.619, 0.634 and 0.622 off (−0.360).
- **The limit is exact on the card.** Every run of the replay set comes out as expected (11 of 11), and the bit
  identity holds at the shapes the switch takes (0 of 42 at `=1` and `=2`). At the default, the products of more than
  32 rows differ from the one-token sums as cuBLAS's do, and no product of 32 rows or fewer does.
- **`GGML_CUDA_MMVF_F32_MT=1` goes into the v4 candidate line (`v4c2`).**
- **Host code only:** with the switch off nothing changes, and every GPU function is the base build's.
- **The default of 32 rows is low:** `mul_mat_vec_f` also beats cuBLAS at 33 rows at 4–8 tokens and at 64 rows at
  5–8 tokens. Flash-Next's F32 matrices are 1 and 512 rows, so this changes nothing for it; raising the default is
  [Q1436](../../../quants/experiments.md).

## Why a row limit

`mul_mat_vec_f` runs one block a row, and each block reads its row's weights once and every token's activations
again. At 4 tokens of a 2,560-wide row a 256-thread block takes 48 registers a thread (C754's
[machine code table](../2026-11-11_c754_f32_mmvf/README.md#the-machine-code); 56-64 at 5, 6 and 8 tokens), so a P106-090's
5 SMs hold 25 such blocks at a time (20 at 5, 6 and 8 tokens). From R120b's means over the 47 calls of a pass:

- **the gate, 1 row:** one block, 6.07 µs a call on `mul_mat_vec_f` against 13.02 on cuBLAS;
- **the router, 512 rows:** about 21 rounds of 25 blocks, each reading the four tokens' 40 KiB of activations again
  (20 MiB a call beside the matrix's 5 MiB), 86.71 µs a call against cuBLAS's 51.23, which reads the activations once
  per tile.

A limit of 32 rows is at most two rounds at 4-8 tokens of a 2,560-wide row (est. about 10 µs a call, under cuBLAS's
13.02 at one row; est.). Every F32 matrix of Flash-Next is either 1 row or 512, so any limit from 1 to 511 picks the
same products there. R132's perf step times 1, 16, 32, 33, 64, 128, 256 and 512 rows with `_ROWS=0` against cuBLAS to
place it.

## The change

`ggml/src/ggml-cuda/mmvf.cu` (`ggml_cuda_mul_mat_vec_f_f32_mt`: the limit, after the switch's own test), `mmvf.cuh`
(its line), `tests/test-backend-ops.cpp` (cases).

- **Where it applies.** With the switch on, a product goes to `mul_mat_vec_f` as in C754 only when its matrix has at
  most `_ROWS` rows (`ne01`); a product of more rows goes on to cuBLAS. `_ROWS=0` (or below) applies no limit.
  `_FAULT=1` reaches only the products the switch takes, so above the limit it changes nothing.
- **Tests:** `test-backend-ops` gets F32 products of 1-8 tokens at the limit's edge (32 and 33 rows of 2,560) and at
  32, 31 and 17 rows at the other block sizes (row widths 64, 100 and 300), and a 32-row view of 336-float rows; its
  perf list gets 16, 32, 33, 64, 128 and 256 rows of 2,560 beside the gate's and the router's shapes.
- **The checks' tools:** `scripts/record_checks.sh`'s suite `f32x` runs the fault on the cases the switch takes at the
  default limit (`f32x-fault`, must fail), on those above it (`f32x-fault-above`, must pass: cuBLAS takes them) and on
  those with `_ROWS=0` (`f32x-fault-rows0`, must fail: the switch takes them again), and `f32x-on-rows0`;
  `multitoken_bitid` gets the mode `f32r`, the bit identity at shapes of at most 32 rows (its mode `f32` holds the
  router and gives 0 differences only with `_ROWS=0`).
- With the switch at 0 the dispatch is the base's line for line.

## Checks on the build machine

[run/chain.sh](run/chain.sh), CPU only, every step in a container without GPUs at a low CPU share, 2026-11-12
17:59-18:43, the merged tree's step 18:44-18:48; results in [summary.tsv](summary.tsv):

| Check | Result |
|---|---|
| rig tarball | `llama.cpp-20261112-11f8d329a-pascal-p106-v4-b12246-c807.tar.gz`, 678,376,401 bytes, md5 `24c812619b84c958ee59083463604049`; `TESTS=1`, sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most; `llama-server` md5 `d11366bb0f57c010d1c0ae6af8609017`, `test-backend-ops` md5 `9e3fb3cd1091edc91ef0ab2d76e6b775` |
| every GPU function | `cuobjdump -sass` and `-ptx` of `llama-server` and `test-backend-ops` against the `pascal-p106-v4-b12246-c806` build (`74881101b`, whose GPU code is `pascal-p106-v4` `d21820340`'s), function by function: 8,540 of 8,540 the same in each of the four listings, none new, none missing (the change is host code) |
| CPU, static | the base `d21820340`'s `test-llama-archs` writes 146 architectures' logits; the branch builds every target, ctest 70 of 70 pass (the 7 tests that need a download, the network or torch not run), `test-llama-archs` 151 of 151 cases, and its 146 logits files are the base's, 0 differ |
| CPU, shared libraries | every target, ctest 70 of 70, `test-llama-archs` 151 of 151, its 146 files written and compared |
| `multitoken_bitid` | modes `f32r` and `f32` among them, linked against the rig build's tree: md5 `7bed9a3c801a52c4b2c0276c61bc9a49` (it runs on the rig) |
| replay set `f32x` | 163 cases (F32's 2-D `MUL_MAT` at 1-8 tokens: C754's and this row's 120 and 43 of llama.cpp's own), 11 runs ([replay_runs.tsv](replay_runs.tsv), [replay_cases.tsv](replay_cases.tsv)), recorded on the CPU with the rig build's `test-backend-ops`; tar 392,519,680 bytes, md5 `3e895c13f4f9d82ffd761de69ecbd70b` |
| what each run selects | [select.tsv](select.tsv): the fault runs that must fail select the 35 cases of at most 32 rows (1, 7, 17, 31 and 32 rows at 4-8 tokens) and nothing above; `f32x-fault-above` and the two `-rows0` fault runs the 32 cases above (33, 48, 129, 256 and 512 rows, two of them llama.cpp's own 48 × 2,560) and nothing at or below 32 |
| CPU replay | the set's 163 cases replayed on the CPU with the rig build: 163 of 163 pass (the set's files are whole) |
| every switch off, merged | the `--no-ff` merge into `pascal-p106-v4` (`993902ae9` on `e27947287`, the line's tip after C811's merge; it adds exactly the branch's diff): its `test-llama-archs` against `e27947287`'s 146 logits files, static, 151 of 151 cases, 0 differ |

The branch changes only host code and `test-backend-ops`' list of cases, so the GPU code is the base's, and the CPU
builds compile the base's code: ctest and `test-llama-archs` show the CPU side untouched. With the switch off the new
test is never reached: `ggml_cuda_mul_mat_vec_f_f32_mt` returns at its first test, as in C754. The kernel itself is
C754's, unchanged, so C754's emulation and machine-code checks of its sums hold as they were.

## What the rig run must show (R132)

[run/rig_c807.sh](run/rig_c807.sh) on a P106-090 rig, the rig idle, with the branch's tarball, the set and the
harness:

1. **replay:** the set `f32x` through `scripts/replay_checks.sh`; every run ends `ok`: `f32x-fault` and `f32x-fault2`
   fail every one of their 35 cases, `f32x-fault-above` and `f32x-fault2-above` pass all 32 (above 32 rows the switch
   leaves the product to cuBLAS on the card too), and `f32x-fault-rows0` and `f32x-fault2-rows0` fail all 32.
2. **bitid:** `multitoken_bitid f32r` with the switch at 1 and at 2 gives `0 with differences`, off (the control)
   differs at 4-8 tokens; `multitoken_bitid f32` with `_ROWS=0` gives `0 with differences` (C754's rule, as R120a
   found it), and at the default differs only at the shapes of more than 32 rows at 4-8 tokens (the limit).
3. **perf:** `test-backend-ops perf` at 1, 16, 32, 33, 64, 128, 256 and 512 rows of 2,560, 1-8 tokens, arms `off`,
   `on` and `rows0`, two passes. At 4-8 tokens `on` must match `rows0` up to 32 rows and `off` above; `rows0` against
   `off`, each pass by value, places the limit: gain, negligible or loss per row count. A row count above 32 that
   gains, or one at or below 32 that loses, moves the default (a new row).
4. **served:** R49's profile of a served verify pass
   ([verify_prof.sh](../2026-11-10_served_verify_parts/run/verify_prof.sh)) on the Flash-Next preset with the MTP
   draft, `off on on off`, `on` = `GGML_CUDA_MMVF_F32_MT=1`: the shared expert's gate and the router per verify pass,
   each load by value, against R120b's; the router's time must be `off`'s and the gate's est. 0.327 ms a pass less.

## The rig run (R132)

One detached run of [run/rig_c807.sh](run/rig_c807.sh) on a nine-card P106-090 rig with nothing else running, on
2026-11-12 from 20:18:41 to 21:15:25 ([rig/status.txt](rig/status.txt)). Load average was 0.89 at the start, and
MemAvailable at least 3,100,564 kB at every check of the run. It used the branch's rig tarball (md5
`24c812619b84c958ee59083463604049`), the set `f32x` (md5 `3e895c13f4f9d82ffd761de69ecbd70b`) and `multitoken_bitid`
(md5 `7bed9a3c801a52c4b2c0276c61bc9a49`), each checked on the rig after the copy.
- Replay 20:18:41-20:19:33, bit identity 20:19:33-20:19:36, perf 20:19:36-20:26:51 (card 0).
- Served loads 20:26:51-21:15:25: 12 min 22 s, 12 min 6 s, 12 min 7 s and 11 min 59 s.

### Replay and bit identity: the limit is exact on the card

**Replay** ([rig/replay.tsv](rig/replay.tsv)): 11 runs as expected, 0 wrong.

| Run | Switches | Must | Cases passed |
|---|---|---|---:|
| `f32x-off`, `f32x-on`, `f32x-on2`, `f32x-on-rows0` | none; `=1`; `=2`; `=1 _ROWS=0` | pass | 163/163 each |
| `f32x-fault-off` | `_FAULT=1` with the switch off | pass | 67/67 |
| `f32x-fault`, `f32x-fault2` | `=1` or `=2`, `_FAULT=1`: the cases of at most 32 rows | fail every case | 0/35 each |
| `f32x-fault-above`, `f32x-fault2-above` | the same, the cases above 32 rows (cuBLAS takes them) | pass | 32/32 each |
| `f32x-fault-rows0`, `f32x-fault2-rows0` | `_ROWS=0`, `_FAULT=1`: the cases above 32 rows | fail every case | 0/32 each |

**Bit identity** ([rig/bitid.tsv](rig/bitid.tsv), every differing product in
[rig/bitid_diffs.tsv](rig/bitid_diffs.tsv)): each product of 4-8 tokens against its one-token columns, 42 comparisons
an arm.

| Mode | Arm | Switches | With differences | Of them at most 32 rows | Must |
|---|---|---|---:|---:|---|
| `f32r` | on | `=1` | 0 | 0 | 0 |
| `f32r` | on2 | `=2` | 0 | 0 | 0 |
| `f32r` | control | off | 30 | 30 | differ |
| `f32` | rows0 | `=1 _ROWS=0` | 0 | 0 | 0 |
| `f32` | limit | `=1` | 20 | 0 | differ above 32 rows only |

The limit arm's 20 are the four shapes above 32 rows (33 × 100, 48 × 64, 129 × 300 and the router's 512 × 2,560) at
4, 5, 6, 7 and 8 tokens: the products the limit leaves to cuBLAS.

### perf: where `mul_mat_vec_f` stops gaining

[rig/perf/perf.md](rig/perf/perf.md) (from [run/perf_rows.py](run/perf_rows.py)): every case's µs per product, each
pass by value, and its class, with the same rule as R120b. A shape's spread is the largest (max − min) / median over
every arm and pass at 1-3 tokens, where every arm runs `mul_mat_vec_f`. A case gains or loses when it is faster or
slower by more than that spread in both passes.

`rows0` (`mul_mat_vec_f` at every row count) against `off` (cuBLAS), F32 rows of 2,560, µs a product, pass 1 / pass 2:

| Rows | Spread | 4 tokens: `off` → `rows0` | Class | 5-8 tokens |
|---:|---:|---|---|---|
| 1 | 1.1 % | 10.61 / 10.61 → 4.74 / 4.73 | gain | gain at each |
| 16 | 15.1 % | 12.09 / 12.18 → 6.17 / 6.20 | gain | gain at each |
| 32 | 5.3 % | 12.11 / 12.21 → 9.16 / 9.56 | gain | gain at each |
| 33 | 1.0 % | 12.37 / 12.78 → 9.42 / 9.21 | gain | gain at each (5 tokens: 34.53 / 34.55 → 10.29 / 10.82) |
| 64 | 0.8 % | 13.05 / 13.17 → 13.48 / 13.58 | loss | gain at each (5 tokens: 34.55 / 34.52 → 16.41 / 16.52; 8: 34.28 / 34.53 → 27.61 / 27.63) |
| 128 | 2.3 % | 15.34 / 15.43 → 21.99 / 21.98 | loss | gain at 5 (34.65 / 34.67 → 28.80 / 27.90), loss at 6, 7 and 8 |
| 256 | 4.8 % | 25.43 / 25.53 → 45.13 / 45.32 | loss | loss at each |
| 512 | 1.7 % | 48.61 / 48.77 → 85.22 / 85.48 | loss | loss at each (8 tokens: 95.16 / 95.77 → 184.76 / 184.23) |

- **`mul_mat_vec_f` gains up to 33 rows at every token count from 4 to 8, and at 64 rows at 5 to 8 tokens.** So the
  default of 32 is lower than it needs to be. Flash-Next's F32 matrices are 1 and 512 rows, so its products are the
  same at any limit from 1 to 63; raising the default is Q1436.
- **At 4 tokens cuBLAS is about three times faster than at 5-8 tokens** (12.11 / 12.21 µs at 32 rows against 34.53 /
  34.49 at 5 tokens), so `mul_mat_vec_f` stops gaining at fewer rows there.
- **`on` (the default limit) runs as it should:** against `rows0` it is negligible at every case of 1-32 rows but one
  (1 × 2,560 at 5 tokens, −1.2 % / −4.3 %, the same kernel). Against `off` above 32 rows, 21 of its 25 cases are
  negligible and 4 mixed, none a gain or a loss.

### served: the gate gains, the router takes cuBLAS's time again

R49's profile of a served verify pass ([verify_prof.sh](../2026-11-10_served_verify_parts/run/verify_prof.sh)): the
Miner Llama v3 preset exactly (UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine cards, the MTP draft at
three tokens), four loads `off`, `on`, `on`, `off` (`on` = the preset plus `GGML_CUDA_MMVF_F32_MT=1`). Each is a `warm`
load (no profile) and a `prof` load (the fork's profiler in group mode), each answering the greedy request twice. The
verify pass is split by R120b's [served_stack.py](../2026-11-12_r120b_kern_speed/run/served_stack.py)
([rig/served/stack.tsv](rig/served/stack.tsv), [rig/served/stack.md](rig/served/stack.md)): all nine cards, the
profiler's 7.6 µs subtracted per launch group, requests 1 and 2 of each profiled load, 48 launch groups a pass for each
F32 product.

| Unit | `off`: 1 r1, 1 r2, 4 r1, 4 r2 (ms a pass) | `on`: 2 r1, 2 r2, 3 r1, 3 r2 | on − off | Spread | Class |
|---|---|---|---:|---:|---|
| the shared expert's gate `ffn_gate_inp_shexp` (2,560 → 1) | 0.609, 0.619, 0.634, 0.622 | 0.265, 0.265, 0.258, 0.256 | −0.360 | 0.025 | gain |
| the router `ffn_gate_inp` (2,560 → 512) | 2.403, 2.399, 2.398, 2.400 | 2.405, 2.395, 2.402, 2.401 | +0.001 | 0.010 | negligible |
| everything else (R120b's other units) | each negligible | | | | |
| **the pass (sum of the parts)** | 100.824, 100.800, 100.825, 100.826 | 100.421, 100.359, 100.440, 100.377 | **−0.420** | 0.081 | **gain** |

The estimate was −0.327 ms a pass (R120b's gate difference); the gate alone gives −0.360 here. In R120b's stack the switch
cost +1.341 ms a pass, and the router took 4.072, 4.072, 4.073 and 4.085 ms on.

**The step, from the warm loads** (the server's own timings, [rig/served/replies.tsv](rig/served/replies.tsv); ms a step
= the reply's generation time over its steps):

| Load | Request 1: ms a step | tok/s | Request 2: ms a step | tok/s | Text | Request 2: accepted drafts, steps |
|---|---:|---:|---:|---:|---|---|
| 1 `off` | 135.77 | 21.84 | 122.07 | 24.87 | `e460d038a1f6` | 172 of 247, 84 |
| 2 `on` | 142.28 | 19.48 | 123.06 | 23.02 | `b851d07caac9` | 166 of 266, 90 |
| 3 `on` | 139.27 | 19.90 | 121.63 | 23.29 | `b851d07caac9` | 166 of 266, 90 |
| 4 `off` | 129.32 | 22.93 | 123.14 | 24.65 | `e460d038a1f6` | 172 of 247, 84 |

- **The step does not move beyond its loads' spread:** in request 2, 123.06 and 121.63 ms on against 122.07 and 123.14
  off. The pass's −0.420 ms is 0.3 % of a step.
- **The reply's text changes with the switch.** The gate's four-token sums become the one-token sums, so the greedy
  path rounds differently. All eight `on` replies carry one text and all eight `off` replies another. The `on` text
  accepts fewer drafts in request 2 (166 of 266 in 90 steps, 2.84 tokens a step, against 172 of 247 in 84, 3.05), so
  its tok/s is lower (23.02 and 23.29 against 24.87 and 24.65). That is this one reply's acceptance, not the pass's
  speed.
- **Request 1 is not the comparison:** it is the load's first and holds the first-call waits (C811), in both arms.
- These are single-request diagnostics. The switch's agentic speed is measured in the stacked agent check (R72).

### What goes into the v4 candidate line

`GGML_CUDA_MMVF_F32_MT=1`, with its default `_ROWS=32`, goes into `v4c2`. It is exact on a P106, and the served verify
pass gains 0.420 ms with the router back on cuBLAS.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | the build machine's checks, one line each |
| [select.tsv](select.tsv) | each run of the set `f32x`: the cases its filter selects, at most 32 rows and above |
| [replay_runs.tsv](replay_runs.tsv) | the set's runs: switches, filter, expected verdict (`runs-ptx.tsv`'s, for a newer card, run `=2` instead of `=1`) |
| [replay_cases.tsv](replay_cases.tsv) | the set's cases: id and case string |
| [run/chain.sh](run/chain.sh) | the build machine's checks |
| [run/rig_c807.sh](run/rig_c807.sh) | the rig run (R132) |
| [run/r132_tables.sh](run/r132_tables.sh) | the rig run's replay and bit-identity tables |
| [run/perf_rows.py](run/perf_rows.py) | the rig run's perf tables and classes |
| [rig/status.txt](rig/status.txt) | the rig run's steps, times and host checks |
| [rig/replay.tsv](rig/replay.tsv) | the replay on the card: each run's switches, cases passed, verdict |
| [rig/bitid.tsv](rig/bitid.tsv), [rig/bitid_diffs.tsv](rig/bitid_diffs.tsv) | the bit identity per arm, and every product that differs |
| [rig/perf/us_run.tsv](rig/perf/us_run.tsv) | every perf case's µs per product: arm, pass, type, rows, K, tokens (no header) |
| [rig/perf/wide.tsv](rig/perf/wide.tsv), [rig/perf/spread.tsv](rig/perf/spread.tsv), [rig/perf/verdict.tsv](rig/perf/verdict.tsv), [rig/perf/perf.md](rig/perf/perf.md) | per shape and token count by arm and pass; each shape's spread; every comparison's change and class; the report |
| [rig/served/stack.tsv](rig/served/stack.tsv), [rig/served/stack.md](rig/served/stack.md) | the verify pass by unit per load and request, the change and the class |
| [rig/served/products.tsv](rig/served/products.tsv), [rig/served/by_name.tsv](rig/served/by_name.tsv), [rig/served/parts.tsv](rig/served/parts.tsv) | the one-op weight products by type and by name, and the pass by R41's parts, per load and request |
| [rig/served/replies.tsv](rig/served/replies.tsv), [rig/served/served.md](rig/served/served.md) | every reply (text checksum, drafts, accepted, steps, ms a step, tok/s) and R79's report |
| [rig/served/loads/](rig/served/loads/) | per load: `verify_parts.py`'s report, its cut per request, every main pass, per card and per launch-group label |

Also changed for this row: `scripts/record_checks.sh` (suite `f32x`), `bench/mmvq_pascal/verify/multitoken_bitid.cpp`
(mode `f32r`), [docs/switches.md](../../switches.md).

## Commands

On an x86-64 build machine with Docker and the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image, no GPU (`BASE` is the
base build's rig tarball, here `pascal-p106-v4-b12246-c806`'s; `SRC` a llama.cpp clone with the fork as remote
`fork`; `LOCK` a lock file shared with other builds on the machine):

```bash
WORK=$WORK PUB=$PWD BASE=$BASE BASE_REF=d21820340 BRANCH=pascal-p106-v4-b12246-c807 LOCK=$LOCK JOBS=8 \
    DOCKER_ARGS='--label pascal-quants=1' SEED_SRC=$SRC nice -n 19 bash docs/data/2026-11-12_c807_f32_router/run/chain.sh
# the merged tree, every switch off (the merge commit fetched into $WORK/w/src first):
WORK=$WORK PUB=$PWD BRANCH=pascal-p106-v4-b12246-c807 MERGED=993902ae9 MERGE_BASE=e27947287 LOCK=$LOCK JOBS=8 \
    STEPS='merged tables' nice -n 19 bash docs/data/2026-11-12_c807_f32_router/run/chain.sh
```

On the rig (the branch's tarball unpacked, the set and the harness copied; `rig_c807.sh`'s header names every
variable):

```bash
BIN=<tarball>/bin OUT=<out> SET=<set>/f32x REPLAY=scripts/replay_checks.sh BITID=<multitoken_bitid> \
    MODEL=<Flash-Next GGUF> DRAFT=<MTP draft GGUF> PRESET=release/flash-next-v3/settings.env \
    RIG_SERVER=bench/agent_check/rig_server.sh REQ=docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
    VERIFY_PROF=docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh \
    setsid nohup bash docs/data/2026-11-12_c807_f32_router/run/rig_c807.sh > console.txt 2>&1 < /dev/null &
```

The rig run's tables, on a build machine's CPU (the rig's `out/` copied to `$W/out`, each profiled load's
`prof/prof-verify.csv` to `$W/traces/<k>-<arm>/prof/`), from this repository's root:

```bash
D=docs/data/2026-11-12_c807_f32_router
OUT=$W/out DEST=$W/tables bash $D/run/r132_tables.sh
nice -n 19 python3 -I $D/run/perf_rows.py $W/out/perf/us_run.tsv --out $W/tables/perf > $W/tables/perf/perf.md
for d in 1-off 2-on 3-on 4-off; do
  mkdir -p $W/tables/served/$d
  nice -n 19 python3 -I docs/data/2026-11-10_served_verify_parts/run/verify_parts.py \
    --kpass-parts docs/data/2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary scripts/prof_summary.py \
    --event-us 7.6 --out $W/tables/served/$d $W/traces/$d/prof/prof-verify.csv > $W/tables/served/$d/report.md
done
nice -n 19 python3 -I docs/data/2026-11-11_r79_c722_p106/run/served_table.py $W/out/served --parts $W/tables/served \
  --out $W/tables/served --types iq4_xs,q4_K,q6_K,q3_K,f32 > $W/tables/served/served.md
nice -n 19 python3 -I docs/data/2026-11-12_r120b_kern_speed/run/served_stack.py $W/tables/served --out $W/tables/served > $W/tables/served/stack.md
```

## Provenance and licence

Code: Miner Llama (`github.com/Cinderella-Man/miner-llama`, MIT, llama.cpp's licence), branch
`pascal-p106-v4-b12246-c807` at `11f8d329a` (one commit on `pascal-p106-v4` `d21820340`), merged into
`pascal-p106-v4` as `993902ae9`. The tests' inputs are
generated (fixed seeds), not model weights. The rig run's served model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next (its maker's licence) with its MTP draft layer, and its request is S37's
[greedy request](../2026-11-04_draft_one_card/run/greedy_request.json). Everything here is this repository's licence.
