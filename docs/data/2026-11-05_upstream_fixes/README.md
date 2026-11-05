# Upstream fixes: two llama.cpp bugs reproduced on upstream `master`, before and after the patches (2026-11-05)

Two bugs found while testing Miner Llama v3 are in upstream llama.cpp's own code, and upstream `master` at
[`45b6604b2`](https://github.com/ggml-org/llama.cpp/commit/45b6604b20f17b7d7638042306c0cd1b6af5999c) still has them.
This folder rebuilds that `master` with and without my patches and reproduces both: the CUDA `mul_mat_vec_q` row guard
on one RTX 3090, the biased draws after a speculative checkpoint restore on a CPU. It also checks what remains of a third
bug that upstream fixed the same day. The write-up, with the code and the reasoning, is
[docs/upstream_fixes.md](../../upstream_fixes.md); the patches are pull requests inside my fork,
[Cinderella-Man/llama.cpp#1](https://github.com/Cinderella-Man/llama.cpp/pull/1) and
[Cinderella-Man/llama.cpp#2](https://github.com/Cinderella-Man/llama.cpp/pull/2). The registry row is Q707 in
[quants/experiments.md](../../../quants/experiments.md).

**Result:**
- **Row guard:** the two new `test-backend-ops` cases fail on `master` in 30 / 30 and 29 / 30 runs on the sm_86 code and
  10 / 10 each through the compute 6.1 PTX; with the fix 0 of 80.
- **Draws after a restore:** under a grammar, token-by-token rejection on `master` differs from plain decoding at
  p = 6.05e-9 (total variation 0.095, 1,500 requests an arm); with the fix p = 0.23.
- **Batch limit through older PTX:** `MUL_MAT_ID_SHARED` with 8 tokens still aborts under `CUDA_FORCE_PTX_JIT=1` on
  `master` ("invalid argument"); upstream's fix (#31679) covers builds without that setting, and I made no patch for it.

## The builds

Upstream `master` `45b6604b2` and the fork branch `upstream-fixes` on top of it, copied to the build machine as a git
repository (`git archive 45b6604b2`, then `git format-patch 45b6604b2..upstream-fixes | git am`); each stage's tree
hash equals the branch commit's. [builds.tsv](builds.tsv) lists them with the build machine's commit names.

| Stage | Branch commit | Tree | What it adds | Built |
|---|---|---|---|---|
| master | `45b6604b2` | `177f3db3` | upstream `master` | `llama-server`, CPU |
| test | [`e52e82346`](https://github.com/Cinderella-Man/llama.cpp/commit/e52e8234622b4615eaa2a63a27a1d691983b6f76) | `d19a3da7` | the two `MUL_MAT_ID` cases | `test-backend-ops`, CUDA |
| guard fix | [`44956c778`](https://github.com/Cinderella-Man/llama.cpp/commit/44956c778bf973de8beba846d31ce2f2f8e869c8) | `823a7eae` | the row-guard fix | `test-backend-ops`, CUDA |
| draws fix | `4bf81afc0` | `0347acbd` | the restore keeps the draws | `llama-server`, CPU |

The draws fix was measured as `4bf81afc0`. The commit in the branches,
[`dce733aa6`](https://github.com/Cinderella-Man/llama.cpp/commit/dce733aa650e327026bb1e7df728fb8b0e93bd07) on
`upstream-fixes` and [`b50c521d7`](https://github.com/Cinderella-Man/llama.cpp/commit/b50c521d7feb74dcfeaa8fbc428a484c0eb58805)
on `upstream-fix/spec-replay-draws` (alone on `master`), differs from it in one comment line and the commit message;
the server code the stat test ran is the same. The CUDA build is static, for `61-virtual;86-real` (sm_86 machine
code for the RTX 3090, and compute 6.1 PTX that `CUDA_FORCE_PTX_JIT=1` makes the driver compile instead), with CUDA
12.9.2 (nvcc 12.9.86) and GCC 13.3 in `nvidia/cuda:12.9.2-devel-ubuntu24.04`. The CPU build is static with
`GGML_NATIVE=ON`, in Ubuntu 24.04. The GPU: one RTX 3090, driver 595.99.02, no Xid before or after.

## The row guard

[run/guard_reps.sh](run/guard_reps.sh): `test-backend-ops -b CUDA0 -o MUL_MAT_ID -p 'n_mats=32,n_used=10,b=0,m=129,n=1,'`
(the two new cases) 30 times per build on the sm_86 code, then 10 times per build through the compute 6.1 PTX. The
builds alternate within each repetition, and each run draws new random inputs. The limit is NMSE 5e-4.

| Case | Code | `master` + test: failed | NMSE of the failures, median (range) | With the fix: failed |
|---|---|---:|---:|---:|
| Q4_0, K = 640 (4 rows a block) | sm_86 | 30 / 30 | 0.018 (0.0029–1.0) | 0 / 30 |
| Q2_K, K = 4096 (2 rows a block) | sm_86 | 29 / 30 | 0.0026 (0.00063–0.016) | 0 / 30 |
| Q4_0, K = 640 | compute 6.1 PTX | 10 / 10 | 0.012 (0.0011–0.032) | 0 / 10 |
| Q2_K, K = 4096 | compute 6.1 PTX | 10 / 10 | 0.0049 (0.00071–1.0) | 0 / 10 |

## The draws after a checkpoint restore

[run/stat.sh](run/stat.sh): the grammar arm of [2026-11-04_block_verify](../2026-11-04_block_verify/README.md)'s
statistical test (its `serve_lib.sh`, `stat_tool.gbnf` and `stat_check.py`, and its random-weight `qwen4exp` target
and drafter, GPT-2 vocabulary), on `master` and on the draws fix. 1,500 requests an arm with their own seeds,
temperature 1, top-k 8, a JSON tool call with 30 possible outcomes; the draft arms run draft-simple with
`--spec-draft-n-max 4 --spec-draft-sampling probabilistic --spec-draft-p-min 0` and `LLAMA_TRACE=1`, which logs each
verified draft and marks the ones that restored the checkpoint.

| Arm | Build | Seconds | Drafted | Accepted | Restores | p (χ², dof) | Total variation |
|---|---|---:|---:|---:|---:|---:|---:|
| plain decoding | `master` | 472 | — | — | 0 | the reference | — |
| draft-simple, rejection sampling | `master` | 950 | 48,436 | 0 | 12,109 | **6.05e-9** (78.9, 20) | 0.095 |
| draft-simple, rejection sampling | draws fix | 951 | 48,436 | 0 | 12,109 | **0.23** (24.2, 20) | 0.057 |

No drafted token is valid under the grammar here, so every verification restores. The `master` draft arm's 1,500
replies are identical to the same arm's on my fork's base (2026-11-04); plain decoding differs from it in 1 reply and
the fixed arm in 4.

## The batch limit through older PTX

The same GPU run, then `test-backend-ops -o MUL_MAT_ID_SHARED` once per CUDA build under `CUDA_FORCE_PTX_JIT=1`: the
q4_K cases with 1 to 4 tokens pass, and the 8-token case aborts with "CUDA error: invalid argument" (exit 134) on both
builds, since neither patch touches it ([jit_shared.tsv](jit_shared.tsv)).

## Files

| File | What |
|---|---|
| [guard_sass.tsv](guard_sass.tsv), [guard_jit.tsv](guard_jit.tsv) | every run of the row-guard cases: build, run, case, verdict, error (NMSE, printed for failures) |
| [jit_shared.tsv](jit_shared.tsv) | `MUL_MAT_ID_SHARED` under `CUDA_FORCE_PTX_JIT=1`, per case, and the run's exit status |
| [stat.tsv](stat.tsv), [acceptance.tsv](acceptance.tsv), [stat_arms.txt](stat_arms.txt) | the chi-square tests, drafted and accepted tokens per arm, each arm's time, verified drafts and restores |
| [builds.tsv](builds.tsv) | each build: kind, stage, the build machine's commit, tree hash, exit status |
| [patches/](patches/) | the patches as `git format-patch` files: `mmvq-row-guard/` (2) and `spec-replay-draws/` (1), on `45b6604b2` |
| [run/all.sh](run/all.sh) | every step in order, as one detached job: builds, the stat test, the GPU runs, the summary |
| [run/build.sh](run/build.sh) | the builds in Docker, one build directory per kind |
| [run/guard_reps.sh](run/guard_reps.sh) | the repeated `test-backend-ops` runs and their table |
| [run/stat.sh](run/stat.sh) | the three arms of the statistical test |

## Commands

```bash
# on the build machine: SRC = a git tree with the four stages, MODELS = 2026-11-04_block_verify's stat models
SRC=<tree> WORK=<dir> REPO=<this repository> MODELS=<dir with st/ and sd/> \
    C_MASTER=<45b6604b2> C_TEST=<e52e82346> C_GUARD=<44956c778> C_DRAWS=<4bf81afc0> \
    GPU_ARGS='--gpus device=<one card>' JOBS=8 LOCK=<build lock> CPU_IMAGE=<ubuntu 24.04 with cmake> \
    setsid nohup bash docs/data/2026-11-05_upstream_fixes/run/all.sh > all.log 2>&1 < /dev/null &
# the GPU step alone, again: STEPS=guard (the builds stay in WORK)
```

The GPU step ran three times. The first run's table parser expected another output format, and its batch-limit step
lacked a variable; the second did not record a run that crashed after printing some cases. Both runs' logs show the
same verdicts as the third, and the tables here are from the third run, with the scripts as published.

## Provenance and licence

The code is upstream llama.cpp (MIT) and my patches on it (MIT). The stat test's models have random weights and
GPT-2's vocabulary file from llama.cpp's `models/`; no model output is published beyond the outcome counts in the
tables. The tables are this project's measurements (MIT, as the repository).
