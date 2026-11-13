# C793: one gate/up/SWIGLU launch at 5–8 tokens for IQ4_XS, Q4_K and Q5_K pairs

Registry: [Q1388](../../../quants/experiments.md) (the rig check is Q1454, R148). Built on C757's launch
([2026-11-11_c757_shexp_fused](../2026-11-11_c757_shexp_fused/README.md)) and C729's read-once kernels at 5–8 tokens
([2026-11-11_c729_readonce_5to8](../2026-11-11_c729_readonce_5to8/README.md)).

## What changed

`GGML_CUDA_MMVQ_PASCAL_GLU_MT` runs a gate product, an up product and their SWIGLU as one launch (C757). It took 2–8
tokens where both weight types are Q2_0 or Q3_K, and only 2–4 where one of them is IQ4_XS, Q4_K or Q5_K, because the
separate products of those types ran read once (C722's and C690's kernels) at 2–4 tokens only. C729's
`GGML_CUDA_MMVQ_PASCAL_RO8` runs them read once at 5–8 tokens as well, in the one-token kernel's summation order, so the
one launch can take those pairs at 5–8 tokens too.

- **Two new values of the switch, off by default.** `3` is `1` plus the pairs with an IQ4_XS, Q4_K or Q5_K product at
  5–8 tokens; `4` is the same without the register prefetch (as `2` is to `1`). `0`, `1` and `2` behave as before.
- **A pair is taken only where its separate products would run read once:** the build's own
  `ggml_cuda_mmvq_pascal_dmt_takes` and `ggml_cuda_mmvq_pascal_q5k_mt_takes` decide, at that token count, so `RO8` must be
  on (`=1` or `=2`). Where they decline (wide rows whose staging passes 48 KB, one-warp blocks), the three launches run
  as before.
- **A kernel of its own,** `mul_mat_vec_q_pascal_glu_ro8<NT, PF>` (NT 5–8, with and without the prefetch: 8
  instantiations), with every type's warp code. C757's `mul_mat_vec_q_pascal_glu<NT, PF>` keeps its code: at 5–8 tokens
  it still carries only the Q2_0 and Q3_K warps, so its registers do not grow.
- **The sums:** each warp sums its rows in its type's one-token order (C757's design), which is what `RO8`'s kernels
  give at R = 1, so each gate and up sum is the separate product's bit for bit and the SWIGLU is the separate op's
  expression on the same two values.
- **Flash-Next (Swift GSQ-RCO Q2_0):** the 41 shared-expert layers whose gate and up are not fused with the routed
  experts form 16 type pairs. 11 pairs (22 layers) have an IQ4_XS, Q4_K or Q5_K product; every one fits 48 KB of shared
  memory at 5–8 tokens with 1–4 warp pairs a block ([shapes.tsv](shapes.tsv)). With `=3` and `RO8=1` the launch takes
  39 of the 41 layers at 5–8 tokens, as `=1` does at 2–4; only the two Q6_K layers stay three launches. These are the
  verify passes of 5 or more tokens that longer and adaptive drafts produce.

## Result (build machine, CPU only)

Fork branch `pascal-p106-v4-b12246-c793` `ad97b59ca` (one commit on `pascal-p106-v4` `2144cf760`), merged `--no-ff`
into `pascal-p106-v4` as `eb204a388` (on C826's `36ef48889`). Every check passed ([checks.tsv](checks.tsv)):

| Check | Result |
|---|---|
| Rig tarball | `llama.cpp-20261113-ad97b59ca-pascal-p106-v4-b12246-c793.tar.gz`, 750,874,423 B, md5 `e01f9f34b135fb8d3f94f9037790134c`; `TESTS=1`, `test-llama-archs` in `bin/`, 0 AVX instructions, GLIBC_2.38; `test-backend-ops` md5 `2c713e7615dbdd15bb739daddcfa8c35`, `llama-server` md5 `90fa3dd65803cf9646e4e0cc68b7d422` |
| Harness | `multitoken_bitid` built against the same tree, md5 `94ff9ee795118cec9fe7a13ac0599efe`; both takes functions defined in it |
| GPU code of everything else | `llama-server`: 8,548 functions in SASS and in PTX, 8,540 of them the same as the base's (the `pascal-p106-v4` `65ac9a47d` rig tarball; every merge since is host code); the 8 new are `mul_mat_vec_q_pascal_glu_ro8` ([sass_new_functions.txt](sass_new_functions.txt)) |
| New kernels | registers without / with the prefetch: 5 tokens 195 / 245, 6 tokens 201 / 243, 7 tokens 205 / 243, 8 tokens 212 / 245; no stack, no local memory ([sass_counts.tsv](sass_counts.tsv)) |
| SWIGLU | 602 of 602 copies in the 22 instantiations are the separate GLU kernel's op sequence ([swiglu.txt](swiglu.txt)) |
| Replay set `glumt` | 636 cases (C757's, linked), 17 runs: C757's 9 and C793's 8 (`glumt8-*`: on, without the prefetch, one warp pair a block, without the 2–4-token read-once kernels, without `RO8`, the value 1, off, and the fault over the new pairs at 5–8 tokens, which must fail); `glumt.tar` 353,658,880 B, md5 `b6d4472926e40005dcd39e3fccd8af92` ([replay_runs.tsv](replay_runs.tsv)) |
| Off | `test-llama-archs -b CPU -s 1`: 146 of 146 architectures the same as the base with no switch, with `GLU_MT=3 RO8=1` and with `GLU_MT=4 RO8=2` |
| ctest | 72 of 72, static and shared, plus the network test, the archs files and the hand tests |
| Merged `eb204a388` | a static native CPU build of the merged tip (C818's and C826's merges included); `test-llama-archs` 146 of 146 the same as the base with no switch and with `GLU_MT=3 RO8=1` |

### What waited for the rig, and why

Two checks need a GPU: `multitoken_bitid glu` (the launch against the separate products, bit for bit, through the
compute-6.1 PTX on a newer card or natively on a P106) and the fault runs (the harness's `_FAULT=1` arm and the replay
set's `glumt8-fault`, which must fail). The build machine's GPUs served another workload that day, and none of this
project's GPU jobs ran there. Both checks are R148's first two steps, natively on a P106, which covers what the PTX run
would have shown.

## The rig check (R148, not run)

[run/rig_r148.sh](run/rig_r148.sh) on the rig, detached, with the tarball's `bin/`, the set and the harness:

1. **replay:** the set's eight `glumt8-*` runs; every run ends "ok", the fault run by failing its cases;
2. **bitid:** `multitoken_bitid glu` at `GLU_MT=3`, `=4`, `=3` with `_NP=1`, `=1` and off, then the fault. Passing: every
   line at 640 × 2,560 (Flash-Next's shapes) is bit-identical, and no line differs at `=3` that does not differ off (the
   separate products at the harness's 33 × 8,192 width differ from the one-token ones, C805); the fault differs on every
   new pair at 5–8 tokens;
3. **perf:** `test-backend-ops perf -o MUL_MAT_GLU` at 2,560 → 640, every pair of Q2_0, Q3_K, Q4_K, Q5_K and IQ4_XS,
   5–8 tokens, `GLU_MT=1` (off for these pairs) against `=3`, alternated over two passes; every arm has the preset's
   k-column switches and `DMT=1 DMT_R=1 Q5K_MT=1 RO8=1`;
4. **table:** `perf/table.tsv` and `perf/est_ms.txt`: per pass and token count, the est. ms a verify pass saves, the sum
   over the file's 22 new-pair layers of layers × (off − on); the 17 Q2_0/Q3_K layers run the same launch in both arms
   and give the spread.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | every check of the chain: row, check, verdict, detail |
| [shapes.tsv](shapes.tsv) | the launch shape of the 11 new pairs at 5–8 tokens (lanes per row, slots, rows per row block, shared memory, warp pairs a block), from [run/shapes.py](run/shapes.py) |
| [sass_counts.tsv](sass_counts.tsv) | per GLU kernel instantiation: registers, stack, local memory, instruction counts, SWIGLU copies the same |
| [sass_new_functions.txt](sass_new_functions.txt) | the `llama-server` functions that are not the base's (SASS, then PTX) |
| [swiglu.txt](swiglu.txt) | the separate GLU kernel's SWIGLU sequence and each instantiation's first copy |
| [replay_runs.tsv](replay_runs.tsv) | the replay set's runs: name, switches, ops, filter, expectation |
| [run/checks.sh](run/checks.sh) | the check chain (steps rig, sass, bitid, record, cpu, off, c565, trim; ptx and merged on their own) |
| [run/sass_check.sh](run/sass_check.sh) | C757's static SASS check, extended to the new kernel |
| [run/shapes.py](run/shapes.py) | the host code's shape choice, written out again |
| [run/rig_r148.sh](run/rig_r148.sh) | R148's driver |

Also changed: `bench/mmvq_pascal/verify/multitoken_bitid.cpp` (mode `glu` at 2–8 tokens for every pair; a pair whose
IQ4_XS, Q4_K or Q5_K product the read-once kernels decline at 5–8 tokens has its one-token comparison printed as
declined and not counted, unless `BITID_ALL=1`), `scripts/record_checks.sh` (suite `glumt`: the `glumt8-*` runs) and
[docs/switches.md](../../switches.md).

## Commands

```bash
# the chain (build machine; paths in the environment)
M=<work> REPO=<this repository> LOCK=<flock file> BASE_COMMIT=2144cf760 BASE_TREE=<the base's rig tree> \
  BASE_CPU=<bin/ of the base's static CPU build> SEED_SRC=<a llama.cpp clone with the fork as remote "fork"> \
  GLUMT_SET=<C757's sets/glumt> bash run/checks.sh
# after the merge
STEPS=merged M=... (the same) bash run/checks.sh
# shapes
python3 -I run/shapes.py > shapes.tsv
# R148, on the rig
BIN=<tarball>/bin OUT=<out> SET=<sets/glumt> REPLAY=scripts/replay_checks.sh BITID=<multitoken_bitid> \
  setsid -f nohup bash run/rig_r148.sh > console.txt 2>&1 < /dev/null
```

## Provenance and licence

Everything here was produced by the scripts above from the fork's source (MIT, as llama.cpp) and synthetic test weights;
no model file was read. The tables are this repository's licence.
