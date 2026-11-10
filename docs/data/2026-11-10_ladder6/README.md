# C662 (Q1150): `-ladder6`, the v3.0 release tree on upstream llama.cpp b12246 (2026-11-10)

Miner Llama v3.0 is built from `pascal-p106-v3-b12076-ladder5` (`1551f5fe1`), which sits on upstream llama.cpp 16
commits past its b12076 release. This folder moves that tree onto upstream's newest release of 2026-11-10, b12246
(`78ffbaa18`, 154 upstream commits later), as the first engine step of the v4 line: one merge commit on the new branch
`pascal-p106-v4-b12246-ladder6`, the two recurrent-state fixes of Q1158 that upstream has not merged yet, and one
guard that the card-built attention mask needs on upstream's new gemma4 graph. Every feature and switch of the release
is carried at its default ([docs/switches.md](../../switches.md)); nothing of Miner Llama's own changes a default. The
registry rows are Q1150 and Q1158 in [quants/experiments.md](../../../quants/experiments.md).

**Result:** the branch builds for the rig and passes every merged feature's checks, and `pascal-p106-v4` is
fast-forwarded to it. With no switch set its CPU build gives stock b12246's logits bit for bit on 145 of the 146 random
models `test-llama-archs` builds; the 146th, qwen4exp, differs in the last bits of 95 of 32,768 logits through the
fork's fused GDN conv, as the release differs from its own unfused path. Against the release 141 of 143 models differ,
all of it upstream's: with #31854 (the CPU backend's F32 activations) reverted the branch gives the release's bits on
113. The sm_61 code of every fork kernel is unchanged. Through the compute 6.1 PTX on one card, which is the code a P106
runs, the whole operator suite passes with the fork's paths off and forced on (24,152 of 24,154 and 24,668 of 24,670
cases), as do the release tests' one-card set (the expert cases with a P106's launch shapes 4,782 of 4,782) and
`test-llama-archs` on the card (153 of 153 three ways). Three failures have a name, and the release build has each of
them: llama.cpp's Meta device with the fused GDN ops (Q877), the copies of the hop shape just over their error bound
(Q1229), and a repeated expert id with the fork's kernels off (Q1231). One whole CPU run gave deepseek4 different logits
at rounding level and was not seen again (Q1230). Not yet run on a P106 (Q1224).

## The branch

`pascal-p106-v4-b12246-ladder6` at **`84f7bd3d2`**, four commits on `pascal-p106-v4` (`1551f5fe1`, the v3.0 tag):

| Commit | What |
|---|---|
| `80307d042` | the merge of upstream tag `b12246` (`78ffbaa18`): 154 upstream commits over 731 files, against the fork's 219 changed files. 22 files conflicted, in 71 hunks; a merge, so no pushed history is rewritten |
| `2b2da8a12` | the recurrent state of a new sequence is cleared with a fill instead of a multiply by zero, so a NaN cannot stay in a cell: upstream's open pull request [#31913](https://github.com/ggml-org/llama.cpp/pull/31913) as it is (the fix, its deepseek4 companion, its test case `nan_clear`) |
| `5385cfd25` | a sequence that the recurrent memory's slot finder moves to another cell takes its rollback depth along: commit `824170e1f` of upstream's open pull request [#31669](https://github.com/ggml-org/llama.cpp/pull/31669), the fix of issue [#31656](https://github.com/ggml-org/llama.cpp/issues/31656), written for the snapshot mode this tree has (the moved sequence keeps its final state only, so its depth becomes 0). With it: a count of the cells the slot finder moves (`llama_memory_recurrent_n_swaps`) and the test case `one_slot` |
| `84f7bd3d2` | gemma4 keeps every row through its KV-shared layers when the attention masks are built on the cards (`LLAMA_GPU_KQ_MASK`, off by default). Upstream's new gemma4 graph ([#31356](https://github.com/ggml-org/llama.cpp/pull/31356)) prunes the non-output rows there and gathers the same rows of the host-filled masks, which are empty in that mode: the graph reserve read a null tensor. This was the one failing check of the last upstream line (`pascal-p106-v3-b12128-ladder5`, Q1073) |

## The conflicts

22 files, 71 hunks. The earlier upstream line, `pascal-p106-v3-b12128-ladder5` (`3b552d9b4`: the release plus
upstream b12128), had already resolved 11 of them (in `mmq.cuh`, `llama-context.cpp` and `test-backend-ops.cpp`), and
those resolutions are reused: the merged tree is the tree that merging b12246 into that line gives, and every file
without a conflict is the same on both routes.

| File | Hunks | Upstream's side | The fork's side | Resolution |
|---|---:|---|---|---|
| `ggml/src/ggml-cuda/mmq.cuh`, `mmq.cu`, `mmf.cu`, `mmid.cu`, `mmid.cuh` | 11, 3, 2, 4, 1 | [#31047](https://github.com/ggml-org/llama.cpp/pull/31047), the linear MoE layout for MMQ and MMF, merged as `69ee2ecc0` | the same pull request's head as the fork has carried it since v3 (`a4020ce3a`), with `GGML_CUDA_MMQ_MOE_LEGACY`, the skipped expert slots and the N2P/N3P types on top | upstream's merged files equal the head the fork carried, byte for byte in `mmf.cu`, `mmf.cuh`, `mmid.cu` and `mmid.cuh`: the fork's files stay as they are. `mmq.cuh` differs from the fork's only in the wrapping of two kernel launches, taken from upstream; against upstream's file the merged one only adds lines (40) |
| `ggml/src/ggml-cuda/mmvq.cu` | 1 | [#31679](https://github.com/ggml-org/llama.cpp/pull/31679): MUL_MAT_ID's matrix-vector batch limit from the highest architecture the build holds | the fork's own fix of the same bug, which also bounds it by the architecture of the code the device loaded (a newer GPU running the compute 6.1 PTX) | the fork's line: it contains upstream's bound |
| `ggml/src/ggml-cuda/rope.cu` | 1 | [#30206](https://github.com/ggml-org/llama.cpp/pull/30206): the YaRN correction takes a truncate flag | the fused norm + RoPE of the two-launch attention block (`GGML_CUDA_ATTN_BLOCK`) calls the same function | the fork's call with the new argument |
| `ggml/src/ggml-backend.cpp` | 4 | [#31419](https://github.com/ggml-org/llama.cpp/pull/31419): no upload for the inputs of nodes that are not computed; [#30966](https://github.com/ggml-org/llama.cpp/pull/30966): a CPU split's inputs fetched before the previous split runs | the packed input uploads (`LLAMA_INPUT_UPLOAD`, `LLAMA_PIPE_UPLOAD`), the skipped inputs of pipelined decoding, the scheduler context per graph | both: a packed input is left to its pack, every other input is copied only when it is read |
| `ggml/include/ggml-rpc.h` | 1 | [#29413](https://github.com/ggml-org/llama.cpp/pull/29413): protocol major version 9 | the fork's own major version (stock + 100: its op numbers differ from stock's) | 109, patch 3 as before |
| `src/llama-graph.cpp` | 10 | #30966: `build_moe_ffn`'s expert chain becomes a lambda that runs twice, for the cached experts and the misses computed on the CPU; [#31733](https://github.com/ggml-org/llama.cpp/pull/31733): the architecture is read from the model; [#31927](https://github.com/ggml-org/llama.cpp/pull/31927): the draft's hidden rows from a device buffer | the same chain as a lambda over a set of expert tensors (the cold tier, Q238), route once (Q860), the MoE tail (Q618); the draft's hidden rows from its carry (Q961) | `build_moe_ffn` merged by hand: the fork's lambda takes upstream's two arguments (the cache slots, cached or not), the down projection's biases apply once after upstream's sum of the two parts, a tiered layer stays out of the cache. The draft's input reads the carry, else upstream's device rows, else the host rows |
| `src/llama-graph.h` | 2 | #31927, #31733, #30966: new fields of the graph parameters, the cache-slots struct | the fork's graph parameters, the cold-tier struct | both |
| `src/models/delta-net-base.cpp`, `qwen4exp.cpp` | 1, 2 | [#31660](https://github.com/ggml-org/llama.cpp/pull/31660): the conv history of every rollback snapshot stored in one write (`build_conv_state_store`); #31927 | without rollback snapshots the history goes to the state group of the pipe slot (pipelined decoding on hybrid models); the draft's carry | upstream's function, with the state group as a defaulted last argument that only these two callers pass |
| `src/llama-context.cpp`, `src/llama-ext.h` | 2, 1 | #31927: no host copy of the hidden rows while the device buffer is on; its API | the scheduler of the graph's own size (`LLAMA_SCHED_PER_SIZE`); the fork's API | both |
| `common/speculative.cpp` | 3 | #31927: the MTP driver's device rows; [#31185](https://github.com/ggml-org/llama.cpp/pull/31185): a model loads from a list of paths | the head shortlist, the embedding row cache, the carry, `LLAMA_DRAFT_SHARE` | both |
| `common/fit.cpp` | 9 | [#31569](https://github.com/ggml-org/llama.cpp/pull/31569): the fit counts skipped NextN layers; #31185 | the fork's port of #31569 behind `LLAMA_FIT_NEXTN_SKIP`, and `LLAMA_FIT_DRAFT_CTX` | upstream's calls; both switches kept |
| `common/common.h`, `tools/server/server-context.cpp`, `server-task.h` | 2, 7, 1 | [#30137](https://github.com/ggml-org/llama.cpp/pull/30137): checkpoints are shared between a slot and its copies, their buffers not zero-filled | the fork's port of #30137 behind `--ctx-checkpoints-share` (off) and `LLAMA_STATE_NOINIT` (on) | the fork's port: the same types, and the flag still chooses between sharing and copying |
| `tests/test-backend-ops.cpp`, `tests/test-llama-archs.cpp` | 2, 1 | new cases; a minimum layer count for the random models | the fork's cases; a real tokenizer for the random models (`LLAMA_ARCHS_VOCAB`) | both |

The `union` check lists the lines of the merged tree that are neither upstream's nor a commit's of the branch
([union.tsv](union.tsv)): they are the resolutions above and nothing else.

## What upstream changed by default

With no switch set, the merged tree differs from the release in what these upstream changes do. The first group acts
on the models Miner Llama serves on the rig; the rig rows that follow this one (identity, speed and fit against
`1551f5fe1`) measure them.

| Upstream change | What changes with no switch set | Reaches |
|---|---|---|
| [#31927](https://github.com/ggml-org/llama.cpp/pull/31927) | An MTP draft with one head and its own memory reads the target's hidden rows from a buffer on the target's output card; the host keeps one row per sequence. The buffer holds `n_batch + n_seq_max` rows of the model's hidden width in F32: for Flash-Next at `-b 2048`, 2,049 rows of 10,240 values, 80.0 MiB (est. from the code) on the last card. If it cannot be allocated, the rows go through the host as before, with a warning. A CPU build has no output device, so the CPU checks here run the host path | Flash-Next's preset |
| [#31660](https://github.com/ggml-org/llama.cpp/pull/31660) | The conv history of all rollback snapshots is stored by one `set_rows`; snapshots deeper than the micro-batch are no longer written (the memory never reads them). qwen4exp's default path is the fork's fused `GDN_CONV`, which already stored them with one copy; this is its unfused path (`LLAMA_GDN_FUSE=0`) and every other recurrent model | Flash-Next with `LLAMA_GDN_FUSE=0`; the other recurrent models when a draft is loaded |
| [#31419](https://github.com/ggml-org/llama.cpp/pull/31419) | Inputs that only uncomputed nodes read are not uploaded | every model |
| [#30966](https://github.com/ggml-org/llama.cpp/pull/30966), [#31888](https://github.com/ggml-org/llama.cpp/pull/31888) | The scheduler copies a CPU split's inputs from the previous card before that card's split computes. `build_moe_ffn` adds the expert input to the graph before the cache lookup. With the MoE cache on, part of its misses run on the CPU | every layer split with a CPU split; MoE models |
| [#31084](https://github.com/ggml-org/llama.cpp/pull/31084) | CUDA's radix top-k picks tied candidates in a fixed order | device sampling (`-bs` with top-k) |
| [#31803](https://github.com/ggml-org/llama.cpp/pull/31803) | The lightning indexer's shared-memory layout (bank conflicts) | Flash-Next's sparse attention layers |
| [#31005](https://github.com/ggml-org/llama.cpp/pull/31005) | The fused MoE router replaces a NaN selection score by the lowest float | MoE models |
| [#31851](https://github.com/ggml-org/llama.cpp/pull/31851) | The KQ product of attention without flash attention asks for BF16 accumulators instead of F32, and CUDA leaves a MUL_MAT_ID that asks for BF16 or F32 accumulators to the CPU | `-fa off` |
| [#31854](https://github.com/ggml-org/llama.cpp/pull/31854) | The CPU backend multiplies F16 and BF16 matrices with F32 activations instead of converting the activations to F16 | CPU layers; the CPU reference of the tests |
| [#31683](https://github.com/ggml-org/llama.cpp/pull/31683) | Context shift moves the KV cells correctly; each model builds its own RoPE | a context that shifts |
| [#31569](https://github.com/ggml-org/llama.cpp/pull/31569) | `--fit` counts NextN layers that are not loaded: the fork's default already (`LLAMA_FIT_NEXTN_SKIP=1`) | — |
| [#31047](https://github.com/ggml-org/llama.cpp/pull/31047), [#31679](https://github.com/ggml-org/llama.cpp/pull/31679) | The MoE layout of MMQ and MMF, MUL_MAT_ID's batch limit: the fork's defaults already | — |
| [#31956](https://github.com/ggml-org/llama.cpp/pull/31956), [#28697](https://github.com/ggml-org/llama.cpp/pull/28697), [#26145](https://github.com/ggml-org/llama.cpp/pull/26145) | llama-server: tools with a `response_format` get a full grammar, not a lazy one; a `json_schema` outside its wrapper is accepted; the Responses API converts `text.format.json_schema` | requests that set `response_format` |
| [#30779](https://github.com/ggml-org/llama.cpp/pull/30779), [#31714](https://github.com/ggml-org/llama.cpp/pull/31714) | llama-server visits parent slots before child slots; a session update that is not an object is rejected | several slots |
| [#31356](https://github.com/ggml-org/llama.cpp/pull/31356), [#31635](https://github.com/ggml-org/llama.cpp/pull/31635), [#30993](https://github.com/ggml-org/llama.cpp/pull/30993), [#31657](https://github.com/ggml-org/llama.cpp/pull/31657), [#31449](https://github.com/ggml-org/llama.cpp/pull/31449), [#30541](https://github.com/ggml-org/llama.cpp/pull/30541), [#31584](https://github.com/ggml-org/llama.cpp/pull/31584) | gemma4's graph and assistant, gpt2's output bias, the NVFP4 scales of qwen2 and Granite, the on-device state copies of a quantized KV cache, the bounds of views | other models |
| [#29413](https://github.com/ggml-org/llama.cpp/pull/29413) | RPC protocol 9 (the fork: 109): a v3 and a v4 build refuse each other | two rigs over RPC |

One upstream default is **not** taken: since [#30137](https://github.com/ggml-org/llama.cpp/pull/30137) upstream always
shares a slot's checkpoints with its prompt-cache entries and child slots. The fork has that change behind
`--ctx-checkpoints-share`, off by default, and the merge keeps the flag at its default, so without it the copies are
made as in the release. The presets run one slot without a prompt cache, where the two behave the same. Following
upstream's default is Q1227.

New options that change nothing unless used: `--ple` ([#31185](https://github.com/ggml-org/llama.cpp/pull/31185), the
n-gram table of a model from its own file), the MoE cache's CPU part (#30966), `truncate=false` for YaRN (#30206).

## The checks

The CPU checks run in Docker at a low CPU share with no GPU ([run/checks.sh](run/checks.sh)): the merged features' own
published scripts, with this tree. They use two bases. Against **the release's builds** they show what the merge
changes: the logits of every architecture, the machine code of every kernel. Against **the branch's own build at its
defaults** they show what each switch changes, as the switch checks of the release did against `-ladder4`. A third
base, **stock upstream b12246**, shows what the fork changes on a CPU with no switch set
([run/stock_identity.sh](run/stock_identity.sh)): the fork's version of `test-llama-archs` uses no API that stock lacks,
so it builds against the unmodified upstream tree, whose build writes each random model's logits for the branch's
build to compare.

| Check (script) | Result |
|---|---|
| the rig tarball (sm_61 code and compute_61 PTX, 0 AVX, GLIBC_2.38), 11 extra tests, the CPU build | OK |
| `union` ([2026-11-04_ladder4_merge](../2026-11-04_ladder4_merge/README.md)): the lines that are neither upstream's nor a later commit's | REVIEW, as meant: 729 files change against the release; 10 hold such lines (45 added, 37 removed), all of them the resolutions above ([union.tsv](union.tsv)) |
| **against stock upstream b12246**, a CPU, every switch at its default ([logits-by-model.tsv](logits-by-model.tsv), [stock-and-revert.txt](stock-and-revert.txt)) | 146 models compared: **145 give stock's logits bit for bit**. qwen4exp differs in the last bit of 95 of 32,768 logits (largest difference 7.451e-09, nmse 1.503e-17): the fused GDN conv, on by default. With `LLAMA_GDN_FUSE=0` its logits are stock's, and the release differs from its own unfused graph the same way (143 logits, 7.451e-09) |
| against the release, rig builds and CPU builds | REVIEW: 143 models compared, 141 differ in both (126 with an nmse below 1e-7, 15 above; mamba and mamba2 are the same bits), 3 are new (kimi-k31, ricursive, xing4_0). The largest: mpt 8.137e-01, granite_swa 6.296e-01, granite and granitemoe 6.287e-01, granitehybrid 5.022e-01, gpt2 2.138e-01, minicpm 1.341e-01, qwen2 8.021e-02. All of it is upstream's: the row above |
| the same with upstream's [#31854](https://github.com/ggml-org/llama.cpp/pull/31854) reverted in a test tree (`REVERT=12ab59798`) | 113 of the 143 models give the release's bits: the CPU backend's F32 activations account for 111 of the 141. The other 30 differ through upstream's other changes (gpt2's output bias, the NVFP4 scales of qwen2 and Granite, gemma4's graph, the per-model RoPE) |
| against the last upstream line (`832fb5aa4`, b12128), CPU builds | REVIEW: 141 of 143 differ, 13 with an nmse above 1e-7 (the 15 without gpt2 and qwen2, which that line already had) |
| C536's switches flipped one by one against the branch's defaults (8 graph and cache switches; the 9 CUDA switches together) | SAME on 146 models, but one run: with `LLAMA_KV_HEAD_TAIL=0` deepseek4's logits differed (nmse 2.236e-08). Not seen again: 34 more whole runs with that setting, 64 at the defaults and 400 runs of deepseek4 alone on the branch's build give the same bits each time, as do 30 whole runs of stock's build and 30 of the release's ([repeats.txt](repeats.txt)). The cause is not found (Q1230) |
| the ports' CUDA switches together on a CPU (`GGML_CUDA_MMVQ_PASCAL_MT=1 _F16=1 GGML_CUDA_MMQ_MOE_LEGACY=2 GGML_CUDA_MMVQ_MAX_COLS=4 GGML_CUDA_KEEPALIVE=decode`) | SAME on 146 |
| `test-llama-context` on six random models (22 cases each), then with 7 cache flips ([context-cases.tsv](context-cases.tsv)) | 96 OK, 36 SKIP, 0 FAIL; every flip the same verdicts |
| Q1158's two cases among them | `nan_clear` passes on mamba, the one model it applies to, and fails there when the fill is turned back into the multiply by zero ([run/nan_mutation.sh](run/nan_mutation.sh)). `one_slot` passes on qwen35 and qwen4exp: one sequence with rollback snapshots, six verify batches with rollbacks, a checkpoint of the prefix restored, six more, and the slot finder moves no cell |
| llama-server's checkpoint arms (7 servers) | the default, the base and `LLAMA_STATE_NOINIT=0` give the same tokens, restore points and cached tokens; the four checkpoint arms ACTS as designed |
| `-ladder4`'s branches: moeskip, coldtier, combo, adraft, units | OK (the tail op against the graph-level ops on 146 models: no difference; `test-spec-adapt` 13,507 of 13,507; `test-draft-mem` 10,885 checks) |
| C533: the keep-alive on without a device; `test-cuda-keepalive` three times | OK; 86 of 86 three times |
| C80: text reuse, the tail checkpoint spacing, the card-built mask's three arms, the staircase drafter in 9 option sets | OK. The mask arms pass on all 151 tests, gemma4 included (1,578 card masks equal the host's) |
| C80: the staircase drafter's step lines against the release's, rig builds | not the same lines, and both pass: the drafter's largest error against its reference falls from 0.00259 to 0.00216. With #31854 reverted the branch gives the release's lines ([run/drafter_revert.sh](run/drafter_revert.sh)): the drafter's F16 weights now meet F32 activations |
| C524: the sm_61 code of every kernel against the release's rig build ([sass-by-kernel.tsv](sass-by-kernel.tsv)) | REVIEW: of the release's 8,733 kernels 7,726 are identical, every kernel the fork adds among them (the Pascal matrix-vector, k-column, multi-token, persistent MoE and attention kernels). 16 differ: `topk_moe_cuda` 13 (upstream's kernel, which also carries the fork's fused tail; upstream's #31005 changed it), two of the radix top-k and one of CUB; 991 are gone and 1,023 new (`mul_mat_q` 976 and 1,008, the lightning indexer's 12 under a new name, two of the radix top-k, one SSM scan): upstream's changes to those kernels (#31084, #31296, #31803, #31818 among them) |
| C524: `test-quantize-fns` (Q4_1_64); the screened head with a grammar (Flint, 16 BFCL requests) | OK; 16 of 16 identical |
| C565's whole CPU set, static and shared: ctest, the test with network, the archs runs, the hand tests | OK both ways: ctest 67 of 67 and `test-arg-parser` 1 of 1 with network; archs 151 of 151 three times, 146 logits files written and compared with no difference; the nine hand tests ([summary-c565-static.tsv](summary-c565-static.tsv), [summary-c565-shared.tsv](summary-c565-shared.tsv)) |

**The PTX checks** ran on one RTX 3080 through the build's compute 6.1 PTX, which is the code a P106 runs; nothing is
timed. Two sets: the operator checks of the upstream rebase ([2026-11-03_upstream_rebase](../2026-11-03_upstream_rebase/README.md),
`run/check.sh`), with the whole `test-backend-ops` in five shards by op name ([run/tbo_shards.sh](run/tbo_shards.sh):
their 2,609 to 7,829 cases add up to the 27,343 of a run without a filter), and the release tests' one-card set
([2026-11-04_ladder4_merge](../2026-11-04_ladder4_merge/README.md), `run/gpu_checks.sh`, every part). Consoles:
[ptx-ops.txt](ptx-ops.txt), [ptx.txt](ptx.txt).

| Check | Result |
|---|---|
| `test-llama-archs` on the card, plain, the fork's paths forced on, and with the card-built mask checked (`-b CUDA0`) | 153 of 153 tests pass three times; 1,579 card masks equal the host's |
| the same on every device of the test (the card, the CPU, llama.cpp's Meta device) | stops at qwen4exp's Meta row, as the release build does: the fused GDN ops have no split rule on the Meta device (Q877). With the two GDN fusions off: 418 of 418 three times, 2,861 masks equal |
| the whole `test-backend-ops`, the fork's paths forced on; the persistent MoE kernel's self-check and its planted fault; the fused norm | 24,668 of 24,670 supported cases pass over the five shards; the 2 failures are copies of the hop shape (below). The persistent MoE kernel's self-check: 17 of 17 layers, 0 differing from the per-op path; its planted fault caught 17 of 17. The fused norm: 32 of 32 |
| the whole `test-backend-ops`, the fork's paths off | 24,152 of 24,154 pass over the five shards, without the repeated-id cases (below); the 2 failures are copies of the hop shape |
| the release tests' set: skipped expert slots, the tail op, the expert cases, the tiered model, the dense products; `test-llama-archs` on the card against the release build's logits | every part passes: skipped slots 100 of 100 past 8 tokens, 604 of 604 with the defaults and with `GGML_CUDA_MMVQ_PASCAL=0`, 41 of 41 of Q2_0 with `GGML_CUDA_MMVQ_PASCAL_MTOK=0`; the tail op and the fused router chain 533 of 533; the expert cases 4,782 of 4,782; the tiered model with the cut at `-ub` 128 and 1 (the op = the graph-level cut bit for bit; KLD 0.000000 against the untiered model, 0.397838 and 0.397970 for the cut itself) and `test-moe-ids` 13 of 13; the dense products at the rig's k-column setting 3,423 of 3,423; `test-llama-archs` on the card with the tail op against the graph-level ops on 146 models, none differing. Against the release build's logits on the card 141 of 143 models differ, as on the CPU |

Three failures have a name, and the release build `1551f5fe1` has each of them:

- **The Meta device** (Q877), above.
- **The quantizing copies of a 2,560 × 4 × 512 tensor** (the hop cast's shape). `test-backend-ops -o CPY` fails one
  to four of its 333 cases in every run: the copies from F32 to Q5_1, Q5_0, Q4_1, Q4_0 or Q8_0 land just above their
  bound (Q5_1: 0.000000004 to 0.000000014 against 0.000000003; the test's bound falls with the tensor's element
  count, here 5,242,880).
  Ten runs of the release build fail one to three cases each, ten of the branch two to four
  ([run/cpy_reps.sh](run/cpy_reps.sh)); the copy kernels and the CPU quantizers are the same files in both. Q1229.
- **A repeated expert id with the fork's kernels off.** `MUL_MAT_ID(type_a=iq2_xxs,n_mats=28,n_used=10,m=256,n=6,k=2560,dup=1)`
  ends in an illegal memory access when `GGML_CUDA_MMVQ_PASCAL=0` sends it to llama.cpp's own expert kernel; with the
  fork's kernels on, as the rig runs, the eight repeated-id cases at 5 to 8 tokens pass. Both builds behave the same
  ([run/tbo_case.sh](run/tbo_case.sh)). A router never repeats an expert for a token. Q1231.


## Not checked here

- **The device rows of #31927 with the fork's draft switches** (the head shortlist, the embedding row cache, the
  carry): a CPU build takes the host path, and one card does not hold Flash-Next. A whole-model run on three cards
  (Q1225) and the rig's identity check (Q1224) cover it.
- **Speed and memory on a P106:** nothing here is timed. The rig row (Q1224) compares the three presets with
  `1551f5fe1`: identity, the agent check, and whether Flash-Next's last card still has room for the 80.0 MiB buffer.
- **The MoE cache's CPU part with the fork's expert skip** (`LLAMA_MOE_TAIL`'s -1 ids) and with a cold tier: both are
  off by default and no check combines them (Q1228).

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the CPU checks of the branch: the merged features' own published scripts with this tree, against the release's builds and against the branch's own defaults |
| [run/stock_identity.sh](run/stock_identity.sh) | stock upstream with the fork's test file against the branch's CPU build; with `REVERT`, the branch with upstream commits reverted against an older build |
| [run/stock_flips.sh](run/stock_flips.sh), [run/repeat_arch.sh](run/repeat_arch.sh), [run/repeat_full.sh](run/repeat_full.sh) | single settings against stock's logits; one model, or the whole run, repeated to see whether a result changes between runs |
| [run/drafter_revert.sh](run/drafter_revert.sh) | the staircase drafter's step lines on the release, the branch, and the branch with upstream commits reverted |
| [run/nan_mutation.sh](run/nan_mutation.sh) | the case `nan_clear` on a build with the fill turned back into the multiply by zero |
| [run/tbo_shards.sh](run/tbo_shards.sh) | the whole `test-backend-ops` on one card in five shards by op name, the fork's paths off and forced on; `MODE=support` checks that the shards cover every case once |
| [run/tbo_case.sh](run/tbo_case.sh), [run/cpy_reps.sh](run/cpy_reps.sh) | a filtered operator run per build; the copy cases several times per build |
| [run/card_holders.sh](run/card_holders.sh) | who else holds one card of a shared machine, by its PCI bus id, before a check starts on it |
| [run/tables.sh](run/tables.sh) | makes the tables below from the work folders, with paths replaced by variable names |
| [summary.tsv](summary.tsv), `summary-c80.tsv`, `summary-c524.tsv`, `summary-c565-*.tsv` | every verdict line of the CPU checks |
| [union.tsv](union.tsv) | per changed file: the lines of the branch that upstream or a later commit explains, and those of neither (`added_extra`, `removed_extra`) |
| [logits-by-model.tsv](logits-by-model.tsv) | per random model: the nmse of the branch's logits against the release (rig and CPU builds), the last upstream line, the release with #31854 reverted, and stock b12246 |
| [stock-and-revert.txt](stock-and-revert.txt) | the verdict lines of the stock comparison, the single settings, the revert and the drafter |
| [sass-by-kernel.tsv](sass-by-kernel.tsv) | the kernels whose sm_61 code differs from the release's, is new or is gone, by name |
| [context-cases.tsv](context-cases.tsv) | `test-llama-context`: model, case, verdict |
| [repeats.txt](repeats.txt) | the repeated runs: whole runs per build, deepseek4 alone, the fused against the unfused GDN path per build |
| [ptx-ops.txt](ptx-ops.txt), [ptx.txt](ptx.txt) | the consoles of the PTX checks: one line per run, the failing cases |

In the tables `l6` is the branch, `rel` the release `1551f5fe1`, `u5` the last upstream line `832fb5aa4`, `stock`
upstream b12246 and `norev` the branch with #31854 reverted.

## Commands

```bash
# the merge, in a clone with upstream as origin and the fork as a second remote
git worktree add <dir> -b pascal-p106-v4-b12246-ladder6 pascal-p106-v4 && cd <dir>
git merge --no-commit --no-ff b12246          # 22 files conflict
# the same merge on the earlier upstream line gives the tree with that line's resolutions reused:
git merge-tree --write-tree 3b552d9b4 b12246  # its conflicted files replace the working copies; every other file is equal

# on the build machine: the CPU checks (WORK has the fork's clone in src/ and a ccache)
WORK=<dir> REPO=<this repository> BRANCH=pascal-p106-v4-b12246-ladder6 BASE=1551f5fe1 UPSTREAM=78ffbaa18 \
  BASE_BUILD=<the release's unpacked rig tarball, extras in bin/> BASE_CPU=<bin/ of the release's CPU build> \
  PREV_CPU=<bin/ of 832fb5aa4's CPU build> OWN='2b2da8a12 5385cfd25 84f7bd3d2' TEXT=<text> TINY=<tiny Flint gguf> \
  CASE=<drafter case> MODEL=<Flint G3-screen gguf> TASKS=<BFCL tasks.jsonl> JOBS=8 THREADS=8 LOCK=<flock file> \
  SERVER_LOCK=<flock file> bash docs/data/2026-11-10_ladder6/run/checks.sh
# stock upstream against the branch, then the branch with #31854 reverted against the release, then the drafter
WORK=<dir> SRC=<clone> UPSTREAM=b12246 BRANCH_REF=84f7bd3d2 BRANCH_BIN=<WORK>/cpu/bin bash docs/data/2026-11-10_ladder6/run/stock_identity.sh
WORK=<dir2> SRC=<clone> UPSTREAM=b12246 BRANCH_REF=84f7bd3d2 BRANCH_BIN=<WORK>/cpu/bin REVERT=12ab59798 \
  BASE_BIN=<bin/ of the release's CPU build> bash docs/data/2026-11-10_ladder6/run/stock_identity.sh
WORK=<dir3> SRC=<clone> BASE=1551f5fe1 BRANCH_REF=84f7bd3d2 REVERT=12ab59798 CASE=<drafter case> bash docs/data/2026-11-10_ladder6/run/drafter_revert.sh
STOCK=<dir>/stock/bin BRANCH=<WORK>/cpu/bin OUT=<dir> bash docs/data/2026-11-10_ladder6/run/stock_flips.sh
BIN=<WORK>/cpu/bin OUT=<dir> FLIP=LLAMA_KV_HEAD_TAIL=0 N=30 bash docs/data/2026-11-10_ladder6/run/repeat_full.sh
BASE=<stock bin/> BINS='l6=<WORK>/cpu/bin' ARCH=deepseek4 OUT=<dir> N=25 FLIP=LLAMA_KV_HEAD_TAIL=0 bash docs/data/2026-11-10_ladder6/run/repeat_arch.sh
WORK=<dir> SRC=<clone> REF=84f7bd3d2 MODEL=<WORK>/checks/models/mamba/mamba-dense.gguf GOOD_BIN=<WORK>/cpu/bin bash docs/data/2026-11-10_ladder6/run/nan_mutation.sh

# on one card (GPU=<its uuid>), each in a container of the build image; first: bash run/card_holders.sh <PCI bus id>
BIN=<tree> ARCHS=<tree>/bin/test-llama-archs WORK=<dir> GPU=<uuid> RUNS='moe-check moe-fault norm-q8 archs archs-on archs-mask' \
  bash docs/data/2026-11-03_upstream_rebase/run/check.sh    # archs*: also with DOCKER_ARGS='-e LLAMA_GDN_FUSE=0 -e GGML_CUDA_GDN_FUSE=0', and with -b CUDA0
GPU=<uuid> TREE=<tree> OUT=<dir> MODE=support SHARDS='mulmat mulmatx fattn conv rest all' ARMS=on bash docs/data/2026-11-10_ladder6/run/tbo_shards.sh
GPU=<uuid> TREE=<tree> OUT=<dir> bash docs/data/2026-11-10_ladder6/run/tbo_shards.sh
MODE=ptx BIN=<tree>/bin BASE_BIN=<the release's tree>/bin OUT=<dir> CHECKS=<WORK>/checks TEXT=<text> \
  bash docs/data/2026-11-04_ladder4_merge/run/gpu_checks.sh
GPU=<uuid> BINS='rel=<the release's tree> l6=<tree>' OUT=<dir> N=5 bash docs/data/2026-11-10_ladder6/run/cpy_reps.sh
GPU=<uuid> BINS='l6=<tree> rel=<the release's tree>' OUT=<dir> ARGS="-o MUL_MAT_ID -p 'type_a=iq2_xxs,.*n=[5-8],.*dup=1'" bash docs/data/2026-11-10_ladder6/run/tbo_case.sh

# the tables of this folder
WORK=<WORK> STOCK=<dir> NOREV=<dir2> DRAFTER=<dir3> PTX_OPS=<dir> EXTRA=<dir> BASE_BUILD=... BASE_CPU=... PREV_CPU=... OUT=<dir> bash docs/data/2026-11-10_ladder6/run/tables.sh
```

Provenance: the fork is [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama) (MIT, as
llama.cpp); the random models are written by `test-llama-archs`; the text file, the tiny Flint model, the drafter case
and the BFCL task ids are those of the folders whose scripts use them.
