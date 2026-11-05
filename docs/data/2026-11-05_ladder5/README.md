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
except the 12 `k_moe_layer` instances the F16 router changes by design. **`-ladder5` is the v3 release build**; the
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

**The flake (C589):** on the quant machine at a load average of ~50, `llama-perplexity` sometimes exited 0 with its last
chunk's row and the summary missing from its log, also when the log was written straight to a file inside the
container: 4 of 20 KL runs with `-coldtier`'s own Nov 4 CPU build, 1 and 2 of 20 with the two new builds; the KL files
were complete and the same bytes each time. `2026-11-04_cold_tier/run/cpu_checks.sh` now runs such a run again (up to
three times). Three more check scripts are fixed: `-ladder4`'s `coldtier` call (its prefix assignments broke on
`${CPUS:+...}`), and the switch flips and the keep-alive check, which compared the base's test summary for equality and
so failed every branch carrying C565's test fix.

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the checks of one merged branch: the merged branches' own scripts with this tree, in one work folder |
| [run/rig_ladder5.sh](run/rig_ladder5.sh) | the rig steps: identity with `-ladder4` on Flash-Next's files (`llama-perplexity` distributions, byte for byte) and T(k) on S35's line |
| `summary-*.tsv` | every verdict line of the runs |

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
```
