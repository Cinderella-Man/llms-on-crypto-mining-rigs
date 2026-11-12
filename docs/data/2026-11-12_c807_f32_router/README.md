# The router back on cuBLAS: a row limit for `GGML_CUDA_MMVF_F32_MT` (C807, Q1413, 2026-11-12)

What this is: a row limit for C754's F32 switch, and its checks on a build machine's CPU, with no GPU. On a P106-090
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
- **Est. −0.327 ms a verify pass against the switch off** (est.: R120b's gate difference, with the router's time
  cuBLAS's again; R132 measures it).
- **Host code only:** with the switch off nothing changes, and every GPU function is the base build's.
- **Not yet on a P106:** the replay, the bit identity at the new limit, where `mul_mat_vec_f` stops gaining as the
  rows grow, and the served pair are R132's.

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

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | the build machine's checks, one line each |
| [select.tsv](select.tsv) | each run of the set `f32x`: the cases its filter selects, at most 32 rows and above |
| [replay_runs.tsv](replay_runs.tsv) | the set's runs: switches, filter, expected verdict (`runs-ptx.tsv`'s, for a newer card, run `=2` instead of `=1`) |
| [replay_cases.tsv](replay_cases.tsv) | the set's cases: id and case string |
| [run/chain.sh](run/chain.sh) | the build machine's checks |
| [run/rig_c807.sh](run/rig_c807.sh) | the rig run (R132) |

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

## Provenance and licence

Code: Miner Llama (`github.com/Cinderella-Man/miner-llama`, MIT, llama.cpp's licence), branch
`pascal-p106-v4-b12246-c807` at `11f8d329a` (one commit on `pascal-p106-v4` `d21820340`), merged into
`pascal-p106-v4` as `993902ae9`. The tests' inputs are
generated (fixed seeds), not model weights. Everything here is this repository's licence.
