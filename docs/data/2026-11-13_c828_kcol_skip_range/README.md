# C828: C755's skip only at 1–4 tokens (Q1458)

**Answer.** `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP` now leaves a listed type's products to the other kernels only at 1–4
tokens, the range of the Pascal kernels in `mmvq-pascal.cu`. At 5–8 tokens the k-column kernel keeps them, as without
the switch. With the Flash-Next preset and `KCOL_SKIP=q4_0,q5_0`, the shared expert's 23 Q4_0 and Q5_0 down
projections run `mul_mat_vec_q_pascal_mc` at 2–4 tokens (R150's gain: 0.213 ms less a four-token verify pass on a P106)
and the k-column kernel at 5–8 tokens. Before this change they ran llama.cpp's `mul_mat_vec_q` there, est. 0.253, 0.343,
0.449 and 0.244 ms slower a pass at 5, 6, 7 and 8 tokens (R150, [C755's data](../2026-11-13_c755_shexp_downs/README.md)).
With the switch unset nothing changes. The change is host code only, and every CPU check passed (below). **On a P106
the switch loses nowhere** (R155, [below](#on-a-p106-r155)): on `v4-rc2`'s build the products at 2–4 tokens are bit for
bit the one-token products and run `mul_mat_vec_q_pascal_mc` at R150's times, and at 5–8 tokens they run the k-column
kernel at its own times.

- Fork branch `pascal-p106-v4-b12360-c828` (`eb312b7bc`; the change is `94a68155e`, then a comment rewrap), on
  `v4-rc1` (`da2920d45`).
- Merged `--no-ff` into `pascal-p106-v4` as `eb82916d3`; its tree is the branch's. The tag for `v4-rc2` is the owner's
  decision.
- Files: `ggml/src/ggml-cuda/mmvq-pascal-kcol.cu` and `.cuh` (27 lines added, 3 removed).
- Rig tarball `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828.tar.gz`, 834,148,416 B, md5
  `ac5f32d66ec286eeef4cf28f5d484816` (`TESTS=1 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs"`, as `v4-rc1`'s;
  sm_61, 0 AVX instructions, GLIBC_2.38). `bin/llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`,
  `test-backend-ops` `e38809f0b16869abc065483f07c4798a`, `test-cuda-keepalive` `e5f2e0f90cddd2d0b459f5c351455e27`,
  `test-llama-archs` `7a1ebe27b514f684badf2e91447a58a1`.
- `multitoken_bitid` linked against this tarball's build (for R155): 94,451,944 B, md5
  `284b3baa0dcbb8e1649819eea2c37551`.

## The change

`ggml_cuda_mmvq_pascal_kcol` (the k-column hook that `ggml_cuda_mul_mat_vec_q` asks first for a plain product) used to
return at once for any type in the switch's list. It now asks a new predicate once it knows the column count:

```cpp
bool ggml_cuda_mmvq_pascal_kcol_skips(const ggml_type type, const int64_t ncols) {
    return ncols >= 1 && ncols <= kcol_skip_max_cols && kcol_skip_type(type);   // kcol_skip_max_cols = 4
}
```

`kcol_skip_type` reads the list as before. Unset, the list is empty, so the predicate is false at every column count,
as `kcol_skip_type` was: every product takes the same path as before.

**Why a bound and not a token range in the switch's value** (for example `q4_0:2-4`): the bound is the smaller change,
and it is not a tuning choice. The skip exists to hand a type to the Pascal kernels, which take 1 to 4 tokens
(`MMVQP_MAX_COLS`); past 4 tokens no Pascal kernel exists to hand it to, only llama.cpp's generic kernel, which R150
timed slower than the k-column kernel. A range needs a new syntax in two parsers (C757's gate/up launch in
`mmvq-pascal-glu.cu` reads the same switch with its own copy of the parser), and the preset would have to change its
value. With the bound, the preset keeps `q4_0,q5_0` and `mmvq-pascal-glu.cu` is untouched.

**One token** routes as before: in the default mode 1 the k-column hook takes no one-token Q4_0 or Q5_0 product anyway,
and in mode 2 (`GGML_CUDA_MMVQ_PASCAL_KCOL=2`, testing) the skip still sends it to the one-token Pascal kernel.

**The bound applies to every listed type.** Two readers keep asking the type-level question
(`ggml_cuda_mmvq_pascal_kcol_on`, and C757's own copy of the list), so for a listed type they step aside at every
token count:
- `GGML_CUDA_MMVQ_PASCAL_Q3K_MT` takes no Q3_K product when `q3_K` is listed: at 5–8 tokens Q3_K now stays on the
  k-column kernel (whose sums the Q3_K multi-token kernel keeps bit for bit), where before it went to llama.cpp's.
- `GGML_CUDA_MMVQ_PASCAL_GLU_MT` fuses no Q2_0 or Q3_K gate/up pair whose type is listed: three launches, each exact.
- `docs/data/2026-11-04_iq_kcol/run/gpu_checks.sh`'s `off` arm lists the six IQ types to get "the kernels before" the
  k-column kernels; on a build with C828 its 5–8-token products stay on the k-column kernel. That check ran on builds
  before C828.

## How the switch routes a product now

Flash-Next's preset (`GGML_CUDA_MMVQ_PASCAL_KCOL=1`, `KCOL_TABLE=0`), Q4_0 and Q5_0 in rows of whole 64-weight pairs of
blocks (the shared expert's downs):

| Tokens | Switch unset | `KCOL_SKIP=q4_0,q5_0` before C828 | `KCOL_SKIP=q4_0,q5_0` with C828 |
|---|---|---|---|
| 1 | the one-token Pascal kernel | the one-token Pascal kernel | the one-token Pascal kernel |
| 2–4 | k-column | `mul_mat_vec_q_pascal_mc` | `mul_mat_vec_q_pascal_mc` |
| 5–8 | k-column | llama.cpp's `mul_mat_vec_q` | k-column |

The routing check ([route.tsv](route.tsv), [run/kcol_skip_route.cpp](run/kcol_skip_route.cpp)) runs no GPU code. It
links the rig build's static libraries and asks `ggml_cuda_mmvq_pascal_kcol_skips`, the predicate the hook calls, for
seven types at 1–8 tokens, in three runs:

| Switch | Skipped (type at tokens) | Q4_0 and Q5_0, by tokens |
|---|---|---|
| unset | none of the 56 | 1: one-token Pascal kernel; 2–8: k-column |
| `q4_0,q5_0` (the preset's value) | Q4_0 and Q5_0 at 1–4; nothing at 5–8 | 1: one-token Pascal kernel; 2–4: `mul_mat_vec_q_pascal_mc`; 5–8: k-column |
| `q4_0,q5_0,q8_0,q2_0,q3_K,q4_K,iq4_xs` | all seven at 1–4; nothing at 5–8 | as with `q4_0,q5_0` |

Every row is as designed (56 of 56 in each run). The kernel column follows from the predicate and the dispatch
described in `kcol_skip_route.cpp`'s header.

## The checks

All on a build machine's CPU, in containers without GPUs, at a low CPU share ([summary.tsv](summary.tsv)):

| Check | Result |
|---|---|
| Rig tarball | built (34 warning lines, as `v4-rc1`'s); md5s above |
| Routing at 1–8 tokens, switch unset, at `q4_0,q5_0` and with seven types | as designed: skipped only where listed and at 1–4 tokens; the predicate is defined in `libggml-cuda.a` |
| C755's `multitoken_bitid shdown` on ggml's CPU backend (`BITID_CPU=1`) | 72 comparisons, 0 differ (2 types × 4 shapes × 2–4 tokens × 3 inputs) |
| `llama-server`'s GPU code against `v4-rc1`'s tarball | SASS and PTX: 8,797 functions in each, every one the base's; the masked dumps are the same files (SASS md5 `715388aa…`, PTX `7f13f752…`) |
| `test-llama-archs -b CPU -s 1`, 146 random models, against `da2920d45`'s logits | 146 of 146 the same with the switch unset and at `q4_0,q5_0` (a CUDA switch: the CPU backend never reads it) |
| ctest, static and shared builds | 73 of 75 in each; the two that fail are `v4-rc1`'s own two in C831's run (`test-jinja-py` needs `torch`, `test-kv-cache-scales` a source path); the network test, archs (146 files written and compared) and the hand tests pass |

## What the rig run must show (R155)

R155 (Q1470) runs R150's `bitid` and `perf` steps
([`rig_c755.sh`](../2026-11-13_c755_shexp_downs/run/rig_c755.sh)) on this branch's rig tarball, the rig idle, est.
25 min. The `multitoken_bitid` it needs is this branch's (md5 in the checks table), since the harness links ggml
statically.

- **bitid:** as R150: `on` and `table` exit 0 with 0 of 72 products different; the fault arms `off` (44 of 72 in R150)
  and `mc0` (72 of 72) exit 1.
- **perf:** `test-backend-ops perf` at Q4_0's and Q5_0's 640 → 2,560, 1–8 tokens, arms `kcol` and `mc`, two passes.
  At 2–4 tokens `mc` should run `mul_mat_vec_q_pascal_mc` at R150's times (at 4 tokens, µs a call, passes 1 and 2:
  Q4_0 19.34 and 19.34, Q5_0 28.87 and 28.82); at 5–8 tokens `mc` should run the k-column kernel, at `kcol`'s times
  within the spread of the 1-token control.

```bash
# on the rig, from a copy of this repository's docs/data folder; BIN = the C828 tarball's bin/
BIN=<tree>/bin BITID=<dir>/multitoken_bitid OUT=<out> STEPS="bitid perf" \
  setsid -f nohup bash docs/data/2026-11-13_c755_shexp_downs/run/rig_c755.sh > <out>/console.txt 2>&1 < /dev/null
```

## On a P106 (R155)

**The switch loses nowhere.** With `KCOL_SKIP=q4_0,q5_0` on `v4-rc2`, the shared expert's Q4_0 and Q5_0 downs at 2–4
tokens are bit for bit the one-token products and run `mul_mat_vec_q_pascal_mc` at R150's times (est. −0.234 and
−0.233 ms a four-token pass on the 23 products, in two passes); at 5–8 tokens they stay on the k-column kernel, at its
own times. The switch can go into the release settings with `v4-rc2`.

R155 (Q1470) ran the `bitid` and `perf` steps on a P106 rig on 2026-11-13, 19:44:53–19:46:23, the rig idle, card 0.
The build was `v4-rc2`'s, which is this tarball's tree: `bin/llama-server` md5 `e7afd26db01f4c671549ef12c504ada3` and
`test-backend-ops` `e38809f0b16869abc065483f07c4798a`, checked on the rig. The harness was this branch's
`multitoken_bitid` (md5 `284b3baa0dcbb8e1649819eea2c37551`, checked at both ends of the copy). Tables:
[rig_check/](rig_check/), made by [run/r155_tables.py](run/r155_tables.py).

- **bitid** ([bitid.tsv](rig_check/bitid.tsv)):
  - `on` (the preset's k-column settings and the skip): exit 0, 0 of 72 comparisons differ;
  - `table` (no switch): exit 0, 0 of 72;
  - the fault arms are caught: `off` exit 1, 44 of 72 (all 24 "big", Q5_0 normal 8 of 12, Q5_0 uniform 12 of 12), and
    `mc0` exit 1, 72 of 72.

  Every count, down to the number of values that differ, is R150's.
- **perf at 2–4 tokens** ([perf.tsv](rig_check/perf.tsv), [perf_pass.tsv](rig_check/perf_pass.tsv)): `mc` runs
  `mul_mat_vec_q_pascal_mc` at R150's times. At 4 tokens, µs a call, passes 1 and 2:

  | Type | k-column (`kcol`) | `mul_mat_vec_q_pascal_mc` (`mc`) | R150's `mc` (the other rig) |
  |---|---:|---:|---:|
  | Q4_0 | 31.35, 31.28 | 19.44, 19.43 | 19.34, 19.34 |
  | Q5_0 | 35.07, 34.97 | 28.84, 28.79 | 28.87, 28.82 |

  For the 16 Q4_0 and 7 Q5_0 downs of a pass, est. ms at 2, 3 and 4 tokens: −0.285, −0.251, −0.234 in pass 1 and
  −0.284, −0.244, −0.233 in pass 2 (R150: −0.284, −0.241, −0.229 and −0.289, −0.246, −0.235).
- **perf at 5–8 tokens:** `mc` runs the k-column kernel at `kcol`'s times. Both types together, est. ms a pass at 5, 6,
  7 and 8 tokens: +0.000, −0.001, −0.049, +0.003 in pass 1 and +0.000, +0.000, −0.010, +0.004 in pass 2. Before C828
  the skip cost +0.253, +0.343, +0.449, +0.244 and +0.251, +0.342, +0.438, +0.238 there (R150).
  - The two points where `mc` is slower are Q4_0 at 8 tokens, +0.35 µs a call in both passes (56.75 against 56.40,
    56.62 against 56.27; est. +0.006 ms a pass), and Q5_0 at 6 tokens, +0.10 and +0.08 (est. +0.001 ms).
  - Both are smaller than the 1-token control, where the two arms run the same kernel and differ by −0.53 and −0.03 µs
    (Q4_0) and −0.15 and +0.02 (Q5_0). R150's two-pass rule calls them a loss only because each arm's two passes lie
    0.13 and 0.09 µs apart there.
  - The largest difference is faster: Q4_0 at 7 tokens, 35.38 against 38.43 in pass 1 and 37.66 against 38.27 in
    pass 2.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check: row, check, verdict, detail |
| [route.tsv](route.tsv) | the routing check: switch value, type, tokens, skipped, kernel (Q4_0 and Q5_0) |
| [run/checks.sh](run/checks.sh) | the check chain (build machine, CPU only) |
| [run/kcol_skip_route.cpp](run/kcol_skip_route.cpp) | the routing check's program |
| [rig_check/perf.tsv](rig_check/perf.tsv) | R155's perf: per type and token count, each arm's kernel and µs a call in each pass, `mc` − `kcol`, est. ms a pass, the class by R150's rule, R150's values |
| [rig_check/perf_pass.tsv](rig_check/perf_pass.tsv) | both types together, est. ms a pass by token count, R155 and R150 |
| [rig_check/bitid.tsv](rig_check/bitid.tsv) | R155's bitid: arm, exit, the k-column hook's answer, input, type, comparisons, differing, values differing, values |
| [rig_check/r155.md](rig_check/r155.md) | the three tables as printed by the script |
| [rig_check/status.txt](rig_check/status.txt) | the rig driver's log: each step's start, exit and a quiet check |
| [run/r155_tables.py](run/r155_tables.py) | makes the `rig_check/` tables |

## Commands

The chain ran on the build machine from a copy of this repository, detached, under a lock shared with other build jobs:

```bash
M=<work> REPO=<this repo> LOCK=<lock file> BASE_COMMIT=da2920d45cbb3630b6b6cceba215fcf49e0717b1 \
  BASE_TREE=<v4-rc1's unpacked rig tarball> BASE_CPU=<bin/ of da2920d45's static CPU build> \
  SEED_SRC=<a llama.cpp clone with the fork as remote "fork"> JOBS=8 THREADS=8 \
  bash docs/data/2026-11-13_c828_kcol_skip_range/run/checks.sh
```

This run took the steps `rig route sass cpu off c565 trim` (13:53:38–14:18:23), then `STEPS=bitid`
(14:19:03–14:19:07), which was added to the script after the first run started; `summary.tsv` holds both runs' rows.

R155 on the rig, with the harness and C755's driver copied to a temporary folder (removed afterwards), and the tables
on any machine:

```bash
BIN=<v4-rc2 tree>/bin BITID=<dir>/multitoken_bitid OUT=<out> STEPS="bitid perf" \
  setsid -f nohup bash rig_c755.sh > console.txt 2>&1 < /dev/null
python3 -I docs/data/2026-11-13_c828_kcol_skip_range/run/r155_tables.py --run <out> \
  --r150 docs/data/2026-11-13_c755_shexp_downs/rig_check/perf.tsv --out <tables> > <tables>/r155.md
```

Provenance: the fork's code is MIT-licensed (llama.cpp's licence); the tables are this repository's. No model file is
read: the CPU checks use `test-llama-archs`'s random models.
