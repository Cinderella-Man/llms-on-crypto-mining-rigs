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

## The rig check (R148)

**`GGML_CUDA_MMVQ_PASCAL_GLU_MT=3` gives the same values as `=1` on a P106, and it is slower: a loss.** At 5–8 tokens the
one launch takes longer than the two read-once products and the SWIGLU it replaces. Over Flash-Next's 22 layers with an
IQ4_XS, Q4_K or Q5_K product, a verify pass loses an est. 0.207 ms at 5 tokens, 0.249 at 6, 0.774 at 7 and 1.352 at 8
(pass 1; pass 2: 0.207, 0.245, 0.775, 1.320). `=3` and `=4` stay off; `=1` is the value to use.

One P106-090 on an otherwise idle rig, 2026-11-13, with the `-c793` tarball, the set `glumt` and `multitoken_bitid`
(md5s above). Every arm has the Flash-Next preset's k-column switches (`_KCOL_SMEM=32768 _KCOL_TABLE=0`) and the
read-once kernels (`_DMT=1 _DMT_R=1 _Q5K_MT=1 _RO8=1`). The drivers are [run/rig_r148.sh](run/rig_r148.sh) (replay,
bitid) and [run/rig_r148_prof.sh](run/rig_r148_prof.sh) (speed). [run/r148_tables.py](run/r148_tables.py) makes the
tables in [rig_check/](rig_check/).

| Step | Result |
|---|---|
| Replay | the 7 `glumt8-*` runs that must pass: 636 of 636 cases each; `glumt8-fault`: 0 of 44 cases pass (46 FAIL lines, exit 1), as it must ([replay_results.tsv](rig_check/replay_results.tsv)) |
| Bit identity, 640 × 2,560 | 210 of 210 lines bit-identical (15 pairs, 2–8 tokens, both comparisons) at `=3`, `=4`, `=3` with `_NP=1`, `=1` and off ([bitid_checks.txt](rig_check/bitid_checks.txt), [bitid_lines.tsv](rig_check/bitid_lines.tsv)) |
| `=3`, `=4` and `_NP=1` against `=1` | the same 126 lines differ in each, all at the harness's 33 × 8,192 width; the harness counts 42 of them and lists 84 as one-token comparisons the read-once kernels decline |
| Against off | 14 lines differ at `=3` and not off, and 14 others differ off and not at `=3`. All 28 are at 33 × 8,192 and 2 or 3 tokens: "GLU/apart" at `=3`, "GLU/1-tok" off. At that width the separate products run on the k-column kernel, whose sums are not the one-token ones (C805), while the launch sums in the one-token order, so the differing comparison changes sides. `=1` gives the same 14 lines as `=3` |
| Fault | 1,126 of 1,260 lines differ, 210 of 210 at 640 × 2,560. Each of the 21 pairs with an IQ4_XS, Q4_K or Q5_K product has differing lines at 5–8 tokens (508 of 592). The 84 that do not differ are "GLU/apart" at 33 × 8,192, where the read-once kernels decline those products and the launch does not run; at `=3` the same 84 are bit-identical |
| Speed | below |

### Speed

`test-backend-ops perf` cannot time this launch. Its perf mode repeats only the graph's last node, so a `MUL_MAT_GLU`
case times one gate, up and SWIGLU (or one fused launch) followed by about 8,190 copies of the SWIGLU alone. The perf
step of `rig_r148.sh` therefore timed the SWIGLU in both arms, and its est. ms are 0.000 at every token count
([perf_swiglu_only.tsv](rig_check/perf_swiglu_only.tsv), [perf_swiglu_only_est_ms.txt](rig_check/perf_swiglu_only_est_ms.txt)).

[run/rig_r148_prof.sh](run/rig_r148_prof.sh) runs each case's perf loop under the fork's profiler (`GGML_CUDA_PROFILE`,
mode `group`). The profiler records CUDA events between the launch groups inside the captured CUDA graph. From every
replayed evaluation the driver takes the GPU time of the first 10 launch groups:
- **off (`=1`):** gate, up, SWIGLU and 7 more SWIGLUs;
- **on (`=3`):** the fused launch and 9 SWIGLUs.

Both arms then hold 10 events at the same places in the graph, so the profiler's own cost cancels:
- off − on = gate + up − fused − one SWIGLU;
- the time a served graph saves = off − on + 2 × one SWIGLU's time without events. Mode `graph` measures that SWIGLU
  time as an evaluation's GPU time over its launch groups: 2.369 µs at 5 tokens, 2.371 at 6, 2.426 at 7 and 2.435 at 8
  ([prof_sw.tsv](rig_check/prof_sw.tsv)).

The three Q2_0/Q3_K pairs run the same fused launch in both arms, so their off − on is the method's spread.
- **Cases:** the file's 14 pairs at 5–8 tokens.
- **Repeats:** 3 runs of a case in a row, 7 replayed evaluations each, so 21 per arm and pass.
- **Order:** off and on alternated over two passes; the tables use medians.
- **Fused or not:** group 0's label shows the fused launch in every on case of a new pair and in no off case
  ([prof_cases.tsv](rig_check/prof_cases.tsv), [prof_evals.tsv.gz](rig_check/prof_evals.tsv.gz)).

| Tokens | Est. ms a verify pass saves, 22 new-pair layers, pass 1 | pass 2 | 17 Q2_0/Q3_K layers (the spread), pass 1 | pass 2 |
|---:|---:|---:|---:|---:|
| 5 | −0.207 | −0.207 | 0.003 | −0.008 |
| 6 | −0.249 | −0.245 | −0.006 | −0.002 |
| 7 | −0.774 | −0.775 | 0.009 | 0.007 |
| 8 | −1.352 | −1.320 | 0.017 | −0.001 |

([prof_est_ms.txt](rig_check/prof_est_ms.txt)). The loss is many times the spread at every token count, and it grows
with the tokens:
- **7 and 8 tokens:** the pairs with a Q3_K or Q5_K partner lose most. IQ4_XS + Q3_K loses 78.08 and 76.04 µs a layer
  at 7 tokens (passes 1, 2) and 122.12 and 119.03 at 8.
- **Where `=3` wins:** a few pairs and token counts, all small:
  - Q3_K + Q5_K at 5 tokens: 15.99 and 13.96 µs a layer;
  - Q3_K + IQ4_XS at 5 tokens: 11.89 and 8.83;
  - Q2_0 + Q4_K at 7 tokens: 8.95 and 6.90.

  Taking only those is parked as C833 (Q1466).

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
| [run/rig_r148.sh](run/rig_r148.sh) | R148's driver: replay, bitid, and a perf step that times only the SWIGLU (above) |
| [run/rig_r148_prof.sh](run/rig_r148_prof.sh) | R148's speed driver: the perf loop under the profiler, the first 10 launch groups of every replayed evaluation; the SWIGLU's time in graph mode |
| [run/r148_tables.py](run/r148_tables.py) | R148's tables from both drivers' output |
| [rig_check/replay_results.tsv](rig_check/replay_results.tsv) | the replay's 8 runs: switches, expectation, cases passed, FAIL lines, exit status, verdict |
| [rig_check/bitid_lines.tsv](rig_check/bitid_lines.tsv) | every `multitoken_bitid glu` line of the 6 arms: arm, comparison, pair, tokens, shape, verdict, values that differ |
| [rig_check/bitid_checks.txt](rig_check/bitid_checks.txt) | the bit-identity checks by value |
| [rig_check/perf_swiglu_only.tsv](rig_check/perf_swiglu_only.tsv), [perf_swiglu_only_est_ms.txt](rig_check/perf_swiglu_only_est_ms.txt) | `rig_r148.sh`'s perf step (the SWIGLU alone in both arms) |
| [rig_check/prof_sw.tsv](rig_check/prof_sw.tsv) | graph mode: each replayed evaluation's GPU time and launch groups, per token count and run |
| [rig_check/prof_evals.tsv.gz](rig_check/prof_evals.tsv.gz) | group mode: per arm, pass, pair, token count, run and replayed evaluation, the first 10 groups' time, groups 0–2, the mean of the rest, group 0's label |
| [rig_check/prof_cases.tsv](rig_check/prof_cases.tsv) | per pass, pair and token count: the medians, off − on, the SWIGLU's time, the time saved a layer |
| [rig_check/prof_est_ms.txt](rig_check/prof_est_ms.txt) | the est. ms a verify pass saves, per pass and token count |

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
# R148, on the rig: replay, bitid (and the SWIGLU-only perf step)
BIN=<tarball>/bin OUT=<out> SET=<sets/glumt> REPLAY=scripts/replay_checks.sh BITID=<multitoken_bitid> \
  setsid -f nohup bash run/rig_r148.sh > console.txt 2>&1 < /dev/null
# R148, on the rig: speed under the profiler (sw, then prof; 2 passes x 2 arms x 14 pairs x 4 token counts x 3 runs)
BIN=<tarball>/bin OUT=<prof> setsid -f nohup bash run/rig_r148_prof.sh > prof-console.txt 2>&1 < /dev/null
# the tables, on any machine with Python 3
python3 -I run/r148_tables.py --bitid <out>/bitid --prof <prof> --out rig_check
```

## Provenance and licence

Everything here was produced by the scripts above from the fork's source (MIT, as llama.cpp) and synthetic test weights;
no model file was read. The tables are this repository's licence.
