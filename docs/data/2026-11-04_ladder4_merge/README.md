# C407 (Q962): every off-by-default v3 engine branch in one tree, `-ladder4` (2026-11-04/05)

Between 2026-11-03 and 2026-11-05 the Flash-Next engine work for the nine-P106 rig grew into some forty fork branches
on the v3 base `pascal-p106-v3-b12076`, each a feature behind a switch that is off by default, each built and checked
on its own. `pascal-p106-v3-b12076-ladder4` is the one tree that holds them all on top of `-kshape` (`d27845b67`), so
every rig arm (the draft length per step, the expert skip, the mailbox hops, the draft on one card, block verify and the
rest) runs on one binary with its own switches set, and with every switch off it gives `-kshape`'s tokens. This folder
holds the merge record and the scripts that check it: [run/checks.sh](run/checks.sh) (without a GPU) and
[run/gpu_checks.sh](run/gpu_checks.sh) (one card, through the compute 6.1 PTX). The registry row is Q962 in
[quants/experiments.md](../../../quants/experiments.md).

**Status:** merged, pushed (`e3b30303d`) and built for the rig (see "The build" below), and the same with upstream
b12127 merged in (`a042afb15`, below). The one-card PTX check ran in part (below: every run that finished passed);
the CPU checks have not run.

## What is merged

Fork branch `pascal-p106-v3-b12076-ladder4` at `e3b30303d`: merge commits on `-kshape` `d27845b67` (which already holds
`-ladder3`'s thirteen branches and `-adraft`), one per branch tip as it stood at the merge, plus two commits of the
merge's own (rows 3 and 8) and one change of its own inside a merge (row 9). Rows 1–3 were merged on 2026-11-04, rows
4–15 on 2026-11-05.

| Order | Commit | Branch, tip (cut from) | What it adds | Its folder | Switches (defaults unchanged) |
|---|---|---|---|---|---|
| 1 | `30bb223c2` | `-moeskip` `f3072cbe8` (`-kshape`) | `GGML_OP_MOE_TAIL`: `LLAMA_MOE_TAIL`'s cut as one op; a dropped slot's id is −1 and the expert kernels skip its reads (C120, Q660, Q661) | [moe_skip](../2026-11-04_moe_skip/) | `LLAMA_MOE_TAIL` (off); `graph=`, `skip=`, `repeat=`, `hist=` |
| 2 | `ef34f7d71` | `-coldtier` `f5e600bd2` (`-kshape` `39c3ab5f1`) | a MoE layer's routed experts in a hot and a cold tensor set, each in its own type; `test-moe-ids` (Q238) | [cold_tier](../2026-11-04_cold_tier/) | only files that hold `ffn_*_exps_cold` |
| 3 | `6e57e208b` | own | a tiered layer takes the tail's graph-level cut (its tier lookup reads a row per routed id, so no id may be −1) | — | only with a cold tier and `LLAMA_MOE_TAIL` |
| 4 | `0b25e37bf` | `-blockver` `b8f42a3ff` (`-ladder2` `a330672c1`) | sampled drafts verified as a block; the text a grammar forces appended to the draft; token-by-token rejection keeps its draws across a replay (Q797, Q802, Q808) | [block_verify](../2026-11-04_block_verify/) | `LLAMA_SPEC_BLOCK_VERIFY`, `LLAMA_SPEC_FORCE` (`_CHARS`, `_MAX`), `LLAMA_SPEC_DRAFT_KEEP`, `LLAMA_SPEC_KEEP_DRAWS` (off) |
| 5 | `450f22c57` | `-specdump2` `3e22fd7c1` (`-specdump`, in `-kshape`) | `LLAMA_SPEC_DUMP_FULL_BLOCK` as a second DFlash2 block decoded for the record only (Q804) | [full_block_dump](../2026-11-04_full_block_dump/) | only with `LLAMA_SPEC_DUMP` |
| 6 | `9485352e8` | `-spancount` `f6c157897` (`-kshape` `39c3ab5f1`) | `llama-imatrix --span-mask`, `--span-dump`: one importance matrix per token class (Q897) | [toolcall_experts](../2026-11-04_toolcall_experts/) | the tool's options only |
| 7 | `3327a3720` | `-routeonce` `b3825c215` (`-ladder3` `7bc1552f4`) | route once per verify block: a block's later tokens pick among its first token's experts (Q860) | [route_once](../2026-11-04_route_once/) | `LLAMA_MOE_ROUTE_ONCE` (off) |
| 8 | `6fd71e172` | own | route once's histogram reads the routed experts before the tail's cut | — | only with both switches and `hist=1` |
| 9 | `6a9021389` | `-rpc` `ba068ae6e` (`-ladder3`) | ggml-rpc on the fork's op table (Q743), here with `GGML_OP_COUNT` 105 and protocol patch 3 | [two_rigs](../2026-11-04_two_rigs/) | built only with `RPC=1` |
| 10 | `b861d7b58` | `-replay` `a89d6b281` (`-kshape`) | `test-backend-ops --record DIR` / `--replay DIR` (Q931) | [replay_checks](../2026-11-04_replay_checks/) | tests only |
| 11 | `ffc33bf63` | `-mailbox` `5aebbddd9` (`-kshape`) | card hops through a mailbox in mapped host memory, a kernel on each card (C143, Q682) | [mailbox_hops](../2026-11-04_mailbox_hops/) | `GGML_CUDA_HOP_MAILBOX` (off) and its `_CHUNKS`, `_SLOTS`, `_SLOT_KB`, `_MAX_KB`, `_STREAM`, `_THREADS`, `_POLL_US`, `_TIMEOUT_MS` |
| 12 | `78e22dfcd` | `-hopcast4` `3dbf1105a` (`-kshape`) | `LLAMA_HOP_CAST` below 8 bits: `q6_k`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl` (Q963) | [hopcast_lowbit](../2026-11-04_hopcast_lowbit/) | `LLAMA_HOP_CAST` (off) |
| 13 | `02a8e196f` | `-draft1card` `648c720c9` (`-kshape`) | the MTP draft's token rows and hidden rows kept on its own card (Q961) | [draft_one_card](../2026-11-04_draft_one_card/) | `LLAMA_DRAFT_EMBD_CACHE`, `LLAMA_DRAFT_H_CARRY` (off) |
| 14 | `d29c67616` | `-n2p` `1a0832375` (`-kshape`) | the N2P, N3P, N2P-128 and N3P-128 types (ids 500–503) with their Pascal one-token, multi-token and MMQ kernels (Q928, Q220) | [n2p_flash](../2026-11-04_n2p_flash/) | used only by files that hold these types |
| 15 | `e3b30303d` | `-hybrid` `8c25952e4` (`pascal-p106-v3-b12076` `b05ecc2d1`) | pipelined decoding for the delta-net hybrids, their recurrent states in a state group per pipe slot; the hybrid memory's attention mask built on the cards (Q550) | [flash_next_pipelined](../2026-11-03_flash_next_pipelined/), [upstream_rebase](../2026-11-03_upstream_rebase/) | `--pipe-depth` (0), `LLAMA_GPU_KQ_MASK` (off) |

Merged earlier and so already in `-kshape`: `-ladder`, `-ladder2`, `-ladder3` and every branch in them (`-iqmmvq`,
`-kcol`, `-mtok`, `-gdnfuse`, `-ckpt`, `-draftmem`, `-bytesw`, `-iq2xs`, `-iqkcol`, `-ccflags`, `-fap`, `-qsa`,
`-lightprof`, `-hops`, `-vgraphs`, `-ckptrc`, `-specdump`, `-adraft`, `-taper`); see
[2026-11-04_ladder3_merge](../2026-11-04_ladder3_merge/README.md).

**Left out:** `-bf16mv` `abd60e7c0` and `-q2_0mv` `d29b0cc56`, whose kernels are already in the tree through `-ladder`
(`git cherry` marks `-bf16mv`'s two commits as applied; `-q2_0mv`'s tree equals `-ladder`'s without the BF16 kernel).
Every other `pascal-p106-v3-b12076-*` branch on the fork is in `-ladder4`.

### Where branches meet

- **`-specdump2` × `-adraft`** (`common/speculative.{cpp,h}`): both sides' declarations kept; in the DFlash selector
  loop, `-specdump2` drops the old full-block branch and `-adraft`'s `dp.result_p` stays in the probability test.
- **`-routeonce` × `-moeskip`**: five places where both add code (the C API, the graph-reuse test, the graph functions,
  `tests/CMakeLists.txt`); both kept. With route once's histogram and the tail op both on, the histogram would read the
  tail's −1 ids and stop on its own check; commit 8 has it read the routed experts before the cut.
- **`-rpc` × `-moeskip`**: `GGML_OP_MOE_TAIL` is appended to the op enum, so `ggml-rpc.h`'s count is 105 and the
  protocol's patch version 3. Ops below it keep `-rpc`'s numbers, so an `-rpc` server still serves this client and
  refuses only `GGML_OP_MOE_TAIL`.
- **`-n2p` × `-moeskip`**: the N types' multi-token expert kernel finds its experts through the same grouping function
  as the IQ types' (`mtok_iq::group`), which `-moeskip` taught to write zeros for a −1 slot without reading an expert;
  MMQ's skipped ids are handled before any type's kernel. The combination has not run.
- **`-n2p` × the existing types**: the Pascal launcher's tile staging became a function that the N types share; for
  Q2_0 and the half-word types (Q6_K, the grid types) the choice of kernel is unchanged.
- **`-hybrid` × the rest**: with pipelining off (one state group) the recurrent states sit where they did, the
  attention mask is built on the host unless `LLAMA_GPU_KQ_MASK` is set, and `qwen4exp`'s conv rows go to group 0.

## On upstream b12127: `pascal-p106-v3-b12127-ladder4`

A second branch, `pascal-p106-v3-b12127-ladder4` at `a042afb15`, is `-ladder4` with upstream llama.cpp's release tag
`b12127` (`44f91a02b`, 2026-11-05) merged in as one commit: the 35 upstream commits since the v3 base's `20a51f24a`
(b12076 and 16). `-ladder4` itself stays on b12076, so the rig's comparisons against `-kshape` keep one variable. The
registry row is Q1022.

Three files conflicted:

| File | Hunks | What met | Resolution |
|---|---:|---|---|
| `ggml/src/ggml-cuda/mmq.cuh` | 10 | upstream's fused NVFP4 gate/up ([#31296](https://github.com/ggml-org/llama.cpp/pull/31296)) adds two scale pointers to the MMQ write-back and tile functions, whose signatures the fork carries in PR [#31047](https://github.com/ggml-org/llama.cpp/pull/31047)'s form | the fork's signatures and names plus the two pointers; in the fused write-back, upstream's NVFP4 scaling and the fork's NVFP4 activation-scale branch (the same condition). Every line upstream owns equals b12127's file |
| `src/llama-context.cpp` | 1 | an include next to one upstream dropped | the fork's `<cctype>` kept |
| `tests/test-backend-ops.cpp` | 1 | both sides add performance cases at the same place | both kept |

Upstream changes in the files the fork also patches, read for what they change on the rig:
- [#31356](https://github.com/ggml-org/llama.cpp/pull/31356): the MTP draft keeps only the rows it needs when it shares
  the target's memory (Gemma4's assistant). Flash-Next's MTP draft does not share it (the context keeps the target as
  its `ctx_other` only for Gemma4's assistant, EAGLE3 and DFlash), so its drafts are unchanged.
- [#30541](https://github.com/ggml-org/llama.cpp/pull/30541): the on-device state copies of a quantized KV cache count
  every weight of a block; before, they counted a block as one element, so a Q8_0 cache's copy held a 32nd of its
  bytes. This fixes device state copies with a quantized KV cache, and so changes what they hold.
- [#31584](https://github.com/ggml-org/llama.cpp/pull/31584): a view must stay inside its source over the bytes its
  strides reach, and a view with explicit strides needs a source contiguous along dim 0. Every tensor constructor call
  in the fork's ggml code takes the new strides argument.
- [#31296](https://github.com/ggml-org/llama.cpp/pull/31296): the new scaled gate/up fusion matches NVFP4 weights only,
  and the Pascal matrix-vector kernel declines a fusion that carries scales.
- [#31444](https://github.com/ggml-org/llama.cpp/pull/31444): `common_get_path_from_env()` is gone from `common/`;
  nothing in the fork used it.
- No op or type was added upstream, so `GGML_OP_COUNT` stays 105 and the RPC protocol's patch version 3.

Its build, the same way in its own work folder:

| File | Bytes | md5 |
|---|---:|---|
| `llama.cpp-20261105-a042afb15-pascal-p106-v3-b12127-ladder4.tar.gz` (sm_61 code and compute_61 PTX, no AVX, highest glibc symbol 2.38) | 605,617,906 | `355292fb5ebd1b9b219e339c593cf6c5` |
| `extra-a042afb15.tar.gz` (the same nine extra tests) | 466,178,389 | `ef61d3151b02cb7f5962a21308571f3e` |

Its checks are `-ladder4`'s with `BRANCH=pascal-p106-v3-b12127-ladder4` and `a042afb15` added to `MERGE_OWN` (so the
upstream merge's lines count as the merge's own in `union`); `identity` against `-kshape` may differ only on the
architectures upstream changed (Gemma4).

## The build

`scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-b12076-ladder4` with `TESTS=1` (sm_61 code and compute_61 PTX,
no AVX, `test-backend-ops` included), through `run/checks.sh`'s `build` step, which also builds the extra tests from
the same tree as a second tarball (`test-llama-archs`, `test-llama-context`, `test-draft-mem`, `test-hop-cast`,
`test-bytes-switches`, `test-moe-ids`, `test-spec-adapt`, `test-hop-mailbox`, `test-route-once`):

```bash
WORK=<work dir> REPO=<this repository> STEPS=build JOBS=8 bash docs/data/2026-11-04_ladder4_merge/run/checks.sh
```

| File | Bytes | md5 |
|---|---:|---|
| `llama.cpp-20261105-e3b30303d-pascal-p106-v3-b12076-ladder4.tar.gz` (sm_61 code and compute_61 PTX, no AVX, highest glibc symbol 2.38) | 605,120,820 | `5607273282362c674d8c9e9187e9ac7c` |
| `extra-e3b30303d.tar.gz` (the nine extra tests above) | 465,933,069 | `7f94f62b85b95d102725a0798df3c937` |

For the upstream branch the same command with `BRANCH=pascal-p106-v3-b12127-ladder4` in its own `WORK`.

## The checks still to run

Without a GPU, every step of [run/checks.sh](run/checks.sh) after the build: `union` (every changed line against
`-kshape` is a merged branch's or one of the three own commits'), `cpubuild`, `identity` (`test-llama-archs` on every
architecture, the rig build and a CPU build against `-kshape`'s, bit for bit), `moeskip`, `coldtier`, `combo`, `adraft`
and `units` (block verify, spec dump, draft memory, the N types' quantize functions, the hop casts and route once):

```bash
WORK=<work dir> REPO=<this repository> BASE_BUILD=<-kshape's unpacked rig tarball> \
  BASE_CPU=<bin/ of a CPU build with -kshape's CPU code> TEXT=<a text file for llama-perplexity> \
  STEPS='union cpubuild identity moeskip coldtier combo adraft units' bash docs/data/2026-11-04_ladder4_merge/run/checks.sh
```

`union` lists `tests/test-backend-ops.cpp` for review: `-hybrid` was cut before `-kshape` and carries the
`HC_STREAM_MEAN` test case, which `-kshape` already has.

On one card, through the compute 6.1 PTX ([run/gpu_checks.sh](run/gpu_checks.sh); `PARTS` defaults to
`moeskip combo moeids dense`), in a container that sees only that card:

```bash
MODE=ptx BIN=<the tarball tree>/bin BASE_BIN=<-kshape's tree>/bin OUT=<out dir> DEV=0 CHECKS=<WORK>/checks \
  TEXT=<the text file> bash docs/data/2026-11-04_ladder4_merge/run/gpu_checks.sh
```

The mailbox hops need two cards and the other branches' GPU checks are their own folders' (`run/` in each), now with
this build.

## The PTX check on one card (2026-11-05, in part)

`PARTS='moeskip dense'` on an RTX 3090 through the compute 6.1 PTX (the other two parts need the CPU checks' random
models), with `-kshape`'s tree as `BASE_BIN`. It was stopped during `archs-tail-op`, because the next cumulative build
gets a check of its own; `archs-tail-op`, `-op0`, `-rep` and `dense` did not run. No Xid during the run.

| Run | Result |
|---|---|
| `skip-mmq`, `skip`, `skip-mmvq`, `skip-mtok0` (the skipped slots) | 100/100, 568/568, 568/568, 41/41 passed, 0 FAIL lines |
| `tail` (the tail op, the fused router chain) | 525/525 passed |
| `experts` (every expert case, nothing skipped; with the N types) | 4,612/4,612 passed |
| `archs-off` (the switch off against `-kshape`) | 143 compared, 7 with a difference: the 7 MoE models whose logits file has its Dense sibling's name, which differ within one build's own run too (a test bug, [2026-11-05_c565_ladder4fix](../2026-11-05_c565_ladder4fix/README.md)); the other 136 equal |
| `archs-tail-graph` | logits written |

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the build and every CPU check of the merged tree; `summary.tsv` one line per check |
| [run/gpu_checks.sh](run/gpu_checks.sh) | `-moeskip`'s and the combination's checks on one card through the compute 6.1 PTX |

Provenance: the fork `github.com/Cinderella-Man/llama.cpp`, branch `pascal-p106-v3-b12076-ladder4`; llama.cpp's MIT
licence.
