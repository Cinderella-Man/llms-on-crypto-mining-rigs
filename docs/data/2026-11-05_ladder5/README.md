# C537 (Q1026): `-ladder5`, every Miner Llama feature in one branch, on upstream b12128 (2026-11-05)

Miner Llama's features were spread over a dozen fork branches cut from the cumulative branch
`pascal-p106-v3-b12076-ladder4` (`e3b30303d`): the switch for every fork behaviour (C536), the ports of the old v3 base's
branches (C80) and of the v2-era branches (C524), the keep-alive (C533) and the fix of `-ladder4`'s failing CPU tests
(C565). This folder merges all of them into one branch, `pascal-p106-v3-b12076-ladder5`, then merges the newest upstream
release of the day into a copy of it, `pascal-p106-v3-b12128-ladder5`, and runs every merged branch's own published
checks on both trees, CPU only. Every feature sits behind its own switch at its own default (the list:
[docs/switches.md](../../switches.md)), so with no switch set both trees compute what `-ladder4` computes. The registry
row is Q1026 in [quants/experiments.md](../../../quants/experiments.md).

**Result:** both trees build for the rig and pass every merged branch's CPU checks. With no switch set, `-ladder5`
computes exactly what `-ladder4` does: `test-llama-archs`' logits are the same bit for bit on all 143 files (the 136 the
unfixed base can write, and the 7 MoE models against `-ladder4fix`), for the rig build and the CPU build, and with each
C536 switch and every port's CUDA switch flipped on a CPU. The sm_61 code of every kernel the two share is the same
except the 12 `k_moe_layer` instances the F16 router changes by design. **`-ladder5` is the v3 release build**, at
`1551f5fe1` since T0c fixed the one PTX failure (below); the
upstream line passes the same checks but one: with the card-built mask (`LLAMA_GPU_KQ_MASK=1`, off by default) upstream's
new gemma4 graph crashes the reserve (C588's first item before the release base moves). A `llama-perplexity` flake on the
loaded machine was found on the way (C589).

## The branches

`pascal-p106-v3-b12076-ladder5` at **`1f7215948`**: one merge commit per branch on `-ladder4` `e3b30303d`, in this order
(every branch was cut from `e3b30303d`).

| Commit | Merge of | Brings | Conflicts and their resolution |
|---|---|---|---|
| `5012146fa` | `-ladder4fix` `fdd2e565d` (C565) | `test-llama-archs`' logits file per model, `ggml_graph_view` exported for shared builds | none |
| `7ae1907a1` | `-flags` `b4d6d427b` (C536) | 20 new switches and 2 extended, over every default-path fork behaviour | none |
| `49c8e446f` | `-textreuse` `5b8a616df` (C80) | `LLAMA_SERVER_TEXT_REUSE` | none |
| `47dd900d5` | `-decodemask` `33f88566a` (C80) | `LLAMA_GPU_KQ_MASK_MAX` | none |
| `5c04dd1d4` | `-promptpath` `1cb98250c` (C80) | `LLAMA_SERVER_TAIL_CKPT_MIN` (the branch holds the two above; merged after them, it adds its own commit) | none |
| `2cd34b4e5` | `-cheapguess` `046beb835` (C80) | the staircase drafter's options, `GGML_CUDA_MMVQ_MAX_COLS` | none (`mmvq.cu` and `server-context.cpp` beside `-flags`' lines) |
| `4575f3cd9` | `-mtpk` `7655db25b` (C80) | `GGML_CUDA_MMVQ_PASCAL_MT` | none |
| `1da928303` | `-f16router` `9a0e4c8cc` (C524) | `GGML_CUDA_MMVQ_PASCAL_F16` | none |
| `05fbbee99` | `-prefill-split` `3be460d5a` (C524) | `GGML_CUDA_MMQ_MOE_LEGACY=2` | `mmq.cu`, 1 hunk: `-flags` kept RDNA3/4's per-expert column estimate under `GGML_CUDA_MMQ_MOE_LEGACY=1`, the port passed the token count to the legacy test; both kept. `-flags`' MMF legacy switch reads the same variable, so it now takes `2` the same way (the old layout up to 8 tokens); `0` and `1` are unchanged |
| `cbac06eb5` | `-toolsample-screen` `21c36929c` (C524) | `LLAMA_SCREEN_HEAD_GRAMMAR` | none |
| `81616e2a7` | `-q41g64` `59ac8ce59` (C524) | the Q4_1_64 type (id 504) | `mmvq.cu` (`get_vec_dot_q_cuda`) and `vecdotq.cuh` with `-flags`' FP32/FP16 activation-sum templates: both kept. Q4_1_64's dot product is the port's, with block_q8_1's FP16 sum as in v2.0, for either value of `GGML_CUDA_MMVQ_FP32_SUMS` |
| `1f7215948` | `-keepalive` `f6c1c6a4a` (C533) | `GGML_CUDA_KEEPALIVE` | none. No other branch adds a CUDA free, pinned allocation or device-wide synchronize, so no keep-alive stop was missing |

Every branch's whole diff against `e3b30303d` is in the merged tree (`git apply -R --check`), except the lines of the two
resolutions above and `-textreuse`'s, whose block `-promptpath` splits with its own function.

`pascal-p106-v3-b12128-ladder5` at **`832fb5aa4`**: `-ladder5` plus one merge of upstream tag `b12128` (`a01dff908`),
the newest release on 2026-11-05: b12127 plus #31426 (a test change). The conflicts are the three files of
`pascal-p106-v3-b12127-ladder4` (`a042afb15`), whose inputs are unchanged, so they are resolved the same way:
`ggml-cuda/mmq.cuh` (10 hunks, #31296's NVFP4 scales through the fork's PR #31047 signatures) and `src/llama-context.cpp`
(an include) are `a042afb15`'s files; `tests/test-backend-ops.cpp` keeps both sides (the fork's Pascal MoE cases, then
upstream's lightning-indexer cases). Every other file of the merge equals `a042afb15`'s with `-ladder5`'s twelve merges
applied (`git merge-file`, 56 files). Upstream's changes since b12076 that touch the fork's files or behaviour are
`a042afb15`'s (its README: [2026-11-04_ladder4_merge](../2026-11-04_ladder4_merge/README.md)): a quantized KV cache's
device state copies copy every element (#30541), views bounded by the bytes their strides reach (#31584), MTP verify rows
for a draft sharing the target's memory (#31356, not Flash-Next's), the NVFP4-only gate/up fusion (#31296), and on random
models gpt2's optional output bias (#30993) and qwen2's FFN scales (#31657). b12128 adds no default change.

## The checks

Each check is the merged branch's own published script, run on the merged tree (`run/checks.sh` drives them, one work
folder per branch, Docker, `--cpu-shares=256`, no GPU). Base: `-ladder4` `e3b30303d` for `-ladder5`, `-b12127-ladder4`
`a042afb15` for the upstream line (b12128 differs from b12127 only in a test).

| Check (script) | `-ladder5` | `-b12128-ladder5` |
|---|---|---|
| rig tarball and 11 extra tests; CPU builds | OK | OK |
| union of the merged branches' lines (`2026-11-04_ladder4_merge`) | REVIEW: only the `-prefill-split` resolution's lines in `mmq.cu`, `mmf.cu` | the same, and one blank line of the upstream merge |
| identity with the base, rig and CPU builds | OK, 136 compared + 7 written by the fixed test only | OK |
| C536: 8 graph/cache switches and the 9 CUDA switches flipped; test-llama-context with 7 cache flips; 7 server arms | all SAME / OK; server arms ACTS as designed | the same |
| the ports' CUDA switches together on a CPU (`GGML_CUDA_MMVQ_PASCAL_MT=1 _F16=1 GGML_CUDA_MMQ_MOE_LEGACY=2 GGML_CUDA_MMVQ_MAX_COLS=4 GGML_CUDA_KEEPALIVE=decode`) | SAME on 143 | SAME on 143 |
| `-ladder4`'s branches: moeskip, coldtier, combo, adraft, units | OK (coldtier after its retry fix) | OK |
| C533: the keep-alive switch on without a device; `test-cuda-keepalive` ×3 | OK; 86/86 ×3 | OK |
| C80: text reuse, + tail checkpoint spacing, the decode mask bound, the staircase drafter in 9 option sets and against `-ladder4`'s | OK | OK, but the mask check's two card-mask arms crash (segfault) on gemma4 in the reserve |
| C524: SASS of every kernel against the base; `test-quantize-fns` (Q4_1_64); the screened head with a grammar (Flint, 16 BFCL requests) | REVIEW: 16 differ (12 `k_moe_layer`; 4 allreduce kernels, the one compared differing only in a source line number, from `-keepalive`'s one-line include), 369 renamed with the same code (`-flags`' template flag); OK; 16/16 identical | the same |
| C565's whole CPU set, static and shared: ctest, `test-arg-parser` with network, archs plain/write/compare, identity, the hand tests | ctest 66/66 + 1/1; archs 148/148 ×3; identity with `-ladder4fix` 143/143 | the same; identity with `-b12127-ladder4`+fix 143/143; against `-ladder5`: gpt2, qwen2, gemma4 differ (upstream #30993, #31657, rounding) |
| C566: upstream's download tests (stories15M, torch) | 71/72: `test-conversion-fp8` needs `transformers` too | the same |
| T0: ctest with `transformers==4.57.6` too, static, all but the five download tests and the net test ([summary-t0-fp8.tsv](summary-t0-fp8.tsv)) | 67/67, `test-conversion-fp8` passes (with C566: 72/72) | the same |
| T0: one RTX 3090 through the compute_61 PTX, the fork's kernels forced on with a P106's launch shapes (`2026-11-04_ladder4_merge/run/gpu_checks.sh`, `MODE=ptx`, every part; [ptx-l5.txt](ptx-l5.txt)) | **FAIL**: `MUL_MAT_ID(type_a=n3p_128,type_b=f32,n_mats=32,n_used=10,b=0,m=2561,n=1,k=640)` NMSE 9.13e-4 > 5e-4 (`-ladder4`: OK); every other run OK (the other 4758 expert cases, skipped slots, tail op, archs 136 compared with no difference, combo, moe-ids, dense 3423/3423; `archs-off-base` exits 1 on `-ladder4`'s own binary, C565's shared-logits test bug, as in `-ladder4`'s run) | not run |
| T0b: the failing case 30 times per build, each `-ladder5` draw replayed on `-ladder4`; one draw 10 times per build and under compute-sanitizer; its N2P-128 and N3P-128 neighbours ([run/case_reps.sh](run/case_reps.sh), [run/case_diag.sh](run/case_diag.sh); [t0b-case.tsv](t0b-case.tsv), [t0b-replays.tsv](t0b-replays.tsv), [t0b-neighbours.tsv](t0b-neighbours.tsv)) | not a `-ladder5` regression: over 5e-4 in 23/30 runs, `-ladder4` 19/30 (20/30 on `-ladder5`'s draws); the same inputs give 7.4e-6 to 1.2e-3 from run to run on either build; a row guard in llama.cpp's `mul_mat_vec_q` (below) | not run |
| T0c: the fix, `-ladder5` `1551f5fe1` (the rows bounded by the row count): rig tarball and extras; identity with `1f7215948`, rig and CPU builds; C565's static set with ident against `1f7215948` ([summary-t0c.tsv](summary-t0c.tsv), [summary-t0c-cpu.tsv](summary-t0c-cpu.tsv)) | OK; 143/143 each; ctest 66/66, archs 148/148 ×3, ident 143/143 | the fix cherry-picked (`3b552d9b4`), not built |
| T0c: the PTX check on one RTX 3090, every part ([ptx-t0c.txt](ptx-t0c.txt)); the failing case 30 times, each draw replayed on `1f7215948`; its neighbours twice ([t0c-case.tsv](t0c-case.tsv), [t0c-neighbours.tsv](t0c-neighbours.tsv)) | **OK**: every part, the expert cases 4759/4759; the case 30/30 (NMSE ≤ 9.2e-15; `1f7215948` in the same run: 22/30 over the limit, 20/30 on the fixed build's draws); the neighbours 24/24 (`1f7215948` fails only the case itself) | |

**The flake (C589):** on the quant machine at a load average of ~50, `llama-perplexity` sometimes exited 0 with its last
chunk's row and the summary missing from its log, also when the log was written straight to a file inside the
container: 4 of 20 KL runs with `-coldtier`'s own Nov 4 CPU build, 1 and 2 of 20 with the two new builds; the KL files
were complete and the same bytes each time. `2026-11-04_cold_tier/run/cpu_checks.sh` now runs such a run again (up to
three times). Three more check scripts are fixed: `-ladder4`'s `coldtier` call (its prefix assignments broke on
`${CPUS:+...}`), and the switch flips and the keep-alive check, which compared the base's test summary for equality and
so failed every branch carrying C565's test fix.

**T0b, the failing case.** Over the 5e-4 limit: `-ladder5` 23 of 30 runs, `-ladder4` 19 of 30, `-ladder4` replaying
`-ladder5`'s draws 20 of 30; the NMSE reached 1.0 in 1, 4 and 2 of them (test-backend-ops divides by the GPU output's
own energy, so one huge wrong value gives 1). One recorded draw replayed 10 times: 7.4e-6 to 1.2e-3 on `-ladder5` (7
over the limit) and 8.5e-5 to 1.3e-3 on `-ladder4` (5), and the same values recur on both builds; under
compute-sanitizer's memcheck, initcheck and racecheck 0 errors, and the case still fails. Every neighbour (N2P-128 and
N3P-128, m 2560 and 2561, 1 and 2 tokens, b 0 and 1; 6 checks each, from 2 runs: the run was stopped there) gives about 6e-15. The cause: with 2561 rows of
270 bytes the Pascal kernel declines the product (its tile staging needs rows × row bytes to be a multiple of 16), so it
goes to llama.cpp's `mul_mat_vec_q`, whose small-K shape at K = 640 puts 4 rows in a block. That kernel bounds the rows
with `row0 + i < stride_col_dst`, and for MUL_MAT_ID with one token the column stride is m × n_used (25610), not m. The
last block of each expert slot therefore writes rows 2561 to 2563, computed from the weights after its expert's
matrix, over the first three rows of the next slot, racing with that slot's own block; the fork's skipped-slot zeros in
the same kernel use the same bound. It takes a one-token MUL_MAT_ID whose row count is not a multiple of the block's
rows and that the Pascal kernel declines; no model in the release has such a matrix. The guard is upstream's
(llama.cpp #25445, still in its master).

**T0c, the fix.** `-ladder5` `1551f5fe1` passes the matrix's row count to `mul_mat_vec_q` and bounds all three
row guards by it: the write, the bias prefetch and the skipped-slot zeros. On every launch the row count is at most the
column stride, and rows at or past it exist only when it is not a multiple of the block's rows, so every other shape,
every release model's included, computes what it did bit for bit. Its CPU builds give `1f7215948`'s logits on all 143
files. Through the compute_61 PTX on one RTX 3090 it passes every part of the check, the expert cases 4,759 of 4,759,
and T0b's case in all 30 runs with an NMSE of at most 9.2e-15, while `1f7215948` in the same run fails the case in 22
of 30 runs and in 20 of 30 on the fixed build's inputs. On a rig, the greedy identity check
([run/rig_greedy.sh](run/rig_greedy.sh)) decodes one prompt token by token with each build at every default and
compares the generated text byte for byte.

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the checks of one merged branch: the merged branches' own scripts with this tree, in one work folder |
| [run/rig_ladder5.sh](run/rig_ladder5.sh) | the rig steps: identity with `-ladder4` on Flash-Next's files (`llama-perplexity` distributions, byte for byte) and T(k) on S35's line |
| `summary-*.tsv` | every verdict line of the runs |
| `ptx-l5.txt` | T0: the PTX check's console (one line per run) and the failing case |
| [run/case_reps.sh](run/case_reps.sh) | T0b: cases run many times per build on one card, each draw of the first build replayed on the others |
| [run/case_diag.sh](run/case_diag.sh) | T0b: one recorded draw replayed many times per build, then under compute-sanitizer |
| `t0b-*.tsv` | T0b: build, run, draw, case, verdict and NMSE per run (`case`: the failing case; `replays`: one draw; `neighbours`) |
| `summary-t0c*.tsv`, `ptx-t0c.txt` | T0c: the verdict lines of the fixed build's CPU checks; its PTX check's console |
| `t0c-*.tsv` | T0c: as `t0b-*.tsv`, the fixed build (`t0c`) and `1f7215948` (`l5`) |
| [run/rig_greedy.sh](run/rig_greedy.sh) | T0c: greedy tokens of one prompt with each build on a rig, compared byte for byte |

## Commands

```bash
# on the build machine, one work folder per branch (WORK has the fork's clone in src/ and a ccache)
WORK=<dir> REPO=<this repository> BRANCH=pascal-p106-v3-b12076-ladder5 BASE=e3b30303d \
  BASE_BUILD=<-ladder4's unpacked rig tarball, extras in bin/> TEXT=<text> TINY=<tiny Flint gguf> CASE=<drafter case> \
  MODEL=<Flint G3-screen gguf> TASKS=<BFCL tasks.jsonl> MERGE_OWN=' ' \
  HEADS='ladder4fix=fdd2e565d flags=b4d6d427b textreuse=5b8a616df decodemask=33f88566a promptpath=1cb98250c cheapguess=046beb835 mtpk=7655db25b f16router=9a0e4c8cc prefill-split=3be460d5a toolsample-screen=21c36929c q41g64=59ac8ce59 keepalive=f6c1c6a4a' \
  JOBS=8 THREADS=8 LOCK=<flock file> SERVER_LOCK=<flock file> bash docs/data/2026-11-05_ladder5/run/checks.sh
# the upstream line: BRANCH=pascal-p106-v3-b12128-ladder5 BASE=a042afb15 MERGE_OWN=832fb5aa4, BASE_BUILD=-b12127-ladder4's tree
# C565's set on both, static and shared:
TREES='l4fix=fdd2e565d:test-llama-archs u4fix=a042afb15+fd8276bbc+fdd2e565d:test-llama-archs l5=1f7215948@l4fix u5=832fb5aa4@u4fix l5u=832fb5aa4:test-llama-archs@l5' \
  SRC=<clone> WORK=<dir> SHARED=OFF bash docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh   # then SHARED=ON
# on a rig (from a machine that only runs ssh): identity on Flash-Next's files, then T(k)
RIG=<alias> MODEL=<first shard> BUILDS='<-ladder4 folder> <-ladder5 folder>' TEXT=<text on the rig> DIR=<folder> OUT=<dir> \
  STEPS='identity tk' bash docs/data/2026-11-05_ladder5/run/rig_ladder5.sh
# T0b, in a container of the build image that sees one card (the PTX settings are in the scripts)
BINS='l5=<-ladder5 tree>/bin l4=<-ladder4 tree>/bin' OUT=<dir> REPS=30 bash docs/data/2026-11-05_ladder5/run/case_reps.sh
BINS='l5=... l4=...' OUT=<dir> REPS=30 KEEP_REC=none \
  CASES='type_a=n[23]p_128,type_b=f32,n_mats=32,n_used=10,b=[01],m=256[01],n=[12],k=640,m_v=0,ids_offset=0,skip=0' bash docs/data/2026-11-05_ladder5/run/case_reps.sh
BINS='l5=... l4=...' REC=<OUT>/rec/l5-1 OUT=<dir> REPS=10 bash docs/data/2026-11-05_ladder5/run/case_diag.sh
# T0c on the build machine: the fixed tip's tarball, identity with 1f7215948's (BASE_BUILD its tree, BASE_CPU its CPU build)
WORK=<dir> REPO=<this repository> BRANCH=pascal-p106-v3-b12076-ladder5 BASE=1f7215948 BASE_BUILD=<1f7215948's tree> \
  BASE_CPU=<1f7215948's CPU bin/> CPU_TARGETS=test-llama-archs STEPS='build cpubuild identity' \
  EXTRA='test-llama-archs test-llama-context test-draft-mem test-hop-cast test-bytes-switches test-moe-ids test-spec-adapt test-hop-mailbox test-route-once test-cuda-keepalive test-quantize-fns' \
  JOBS=8 LOCK=<flock file> bash docs/data/2026-11-04_ladder4_merge/run/checks.sh
TREES='l5=1f7215948:test-llama-archs t0c=1551f5fe1@l5' SRC=<clone> WORK=<dir> SHARED=OFF STEPS='tree build ctest archs ident' \
  bash docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh
# T0c on one card (a container of the build image): the PTX check, then the case and its neighbours as T0b's
MODE=ptx BIN=<fixed tree>/bin BASE_BIN=<-ladder4 tree>/bin OUT=<dir> CHECKS=<-ladder5's checks.sh WORK/checks> TEXT=<text> \
  bash docs/data/2026-11-04_ladder4_merge/run/gpu_checks.sh
BINS='t0c=<fixed tree>/bin l5=<1f7215948 tree>/bin' OUT=<dir> REPS=30 bash docs/data/2026-11-05_ladder5/run/case_reps.sh
# T0c on a rig: one model, the old tip's folder first
RIG=<alias> MODEL=<first shard> TAG=<model> ARGS='<the model split>' DIR=<folder> OUT=<dir> \
  BUILDS='<1f7215948 folder> <1551f5fe1 folder>' bash docs/data/2026-11-05_ladder5/run/rig_greedy.sh
```
