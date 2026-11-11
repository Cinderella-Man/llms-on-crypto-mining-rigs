# The v4 night merge: the night's 17 engine branches in one branch, `pascal-p106-v4-b12246-night` (2026-11-11)

On the night of 2026-11-10/11, seventeen fork branches each added one speed-up to the v4 line (`pascal-p106-v4`,
`84f7bd3d2`: the v3.0 release tree on upstream llama.cpp b12246), each behind its own switch, off by default, each built
and checked on a CPU by its own session. This folder merges all of them into one branch, `pascal-p106-v4-b12246-night`,
builds it once for the rig (compute 6.1) and runs every merged branch's own published check on the merged tree, CPU only,
plus the test-llama-archs identity of the whole tree with every switch off against `pascal-p106-v4`. `pascal-p106-v4`
itself is not moved. The registry row is Q1319 in [quants/experiments.md](../../../quants/experiments.md); every switch
is in [docs/switches.md](../../switches.md).

**Result:** the merged tree builds for the rig and every merged branch's own check passes on it. With every switch off
it computes what `pascal-p106-v4` computes: `test-llama-archs`' logits are the same bit for bit on all 146 models, and
the whole CPU test set passes (ctest 70 of 70, static and shared). One fault of a merged branch showed up and is fixed on
the merged branch: C719's new op broke every build with `GGML_RPC=ON` (`c408dabcb`). The four QSA switches that meet in
one function give the same logits on together as off. One replay set holds the CUDA cases of the night's seven check
suites (967 cases) for the rigs, and it replays on a CPU. Nothing ran on a GPU; the speed of each switch is its rig row's
(R89 orders them).

## The branch

`pascal-p106-v4-b12246-night` at **`c408dabcb`**: one merge commit per branch on `pascal-p106-v4` `84f7bd3d2`
(`git merge --no-ff`, no rebase), each branch after the branch it was cut from, then two commits.

| Commit | Merge of | Cut from | Brings (off by default) | Conflicts and their resolution |
|---|---|---|---|---|
| `a0d47e260` | `-c675` `b2763cce7` ([C675](../2026-11-11_c675_end_of_turn_stop/README.md)) | `pascal-p106-v4` | `LLAMA_SPEC_REJECT_STOP=1`: token-by-token rejection stops after an accepted end of generation | none |
| `7b35ebbae` | `-c711` `ca3c0b4fd` ([C711](../2026-11-11_c711_block_verify_stop/README.md)) | `-c675` | the same switch cuts a verified block (`LLAMA_SPEC_BLOCK_VERIFY=1`) at its first drafted end of generation | none |
| `9fc695bf0` | `-c705` `baac08803` ([C705](../2026-11-11_c705_sparse_prompt_rule/README.md)) | `pascal-p106-v4` | `GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1`: prompt micro-batches of 9 to 128 tokens through the sparse attention by a rule of their own | none |
| `7cb42c356` | `-c690` `ef91bff04` ([C690](../2026-11-11_c690_head_four_rows/README.md)) | `pascal-p106-v4` | `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1`: Q5_K's 2–4-token products (the output head in a verify) with the one-token kernel's sums | none |
| `89caeb7c5` | `-c620` `ca3055285` ([C620](../2026-11-11_c620_draft_kv_catchup/README.md)) | `pascal-p106-v4` | `LLAMA_MTP_KV_ONLY=1`: the MTP draft's catch-up passes write its caches and nothing else | none |
| `6571ba57d` | `-c100` `ce611b9db` ([C100](../2026-11-11_c100_kpool_on_cards/README.md)) | `pascal-p106-v4` | `LLAMA_QSA_KPOOL_DEV=1`: the QSA k-pool tables kept on the cards, the changed rows sent | `src/models/models.h`: C620's `build_layer_attn_kv` and `build_qsa_kv` beside C100's `build_kpool_dev`, both kept |
| `513f00710` | `-c720` `27e3d69fa` ([C720](../2026-11-11_c720_kpool_small_inputs/README.md)) | `-c100` | `LLAMA_QSA_KPOOL_STAGE=1`: the five small k-pool inputs inside C100's update | none |
| `53dd878fb` | `-c704` `763fc55ca` ([C704](../2026-11-11_c704_indexer_gather/README.md)) | `pascal-p106-v4` | `LLAMA_QSA_INDEXER_ROWS=1`: the indexer reads the pooled keys in place, no gather | `src/models/qwen4exp.cpp`, `build_qsa_sel`'s scatter and gather: C704's in-place block, with C720's staged `new_pool_rep` and C100's local `pool_cells` (its device copy with C100's switch), which the in-place call takes too, as C704's notes give it (a contiguous I32 view of the pool cells, as the op requires) |
| `d1cdbadb9` | `-c103` `4a3423134` ([C103](../2026-11-11_c103_union_gather/README.md)) | `-c705` | `GGML_CUDA_FATTN_SPARSE_PASCAL_UNION=1`: verify passes and prompt micro-batches through a union kernel that reads the kept cells of 4 query tokens once | none |
| `3029d66ec` | `-c722` `dd72554b6` ([C722](../2026-11-11_c722_short_batch_products/README.md)) | `-c690` | `GGML_CUDA_MMVQ_PASCAL_DMT=1`: C690's method for IQ4_XS and Q4_K | none |
| `5d929bf0a` | `-c703` `db253c8a7` ([C703](../2026-11-11_c703_catchup_overlap/README.md)) | `-c620` | `LLAMA_MTP_CATCHUP_OVERLAP=1`: the draft's prompt catch-up runs once the next call is queued | none |
| `258ec6b19` | `-c702` `2ec3a0927` ([C702](../2026-11-11_c702_prompt_graphs/README.md)) | `pascal-p106-v4` | `GGML_CUDA_GRAPHS_FORCE=1`: CUDA graphs for prompt micro-batches, captured at every change | none |
| `02c29f7dc` | `-c691` `d18c71761` ([C691](../2026-11-11_c691_sampling_launches/README.md)) | `pascal-p106-v4` | `LLAMA_SAMPLING_TOP_K_ROWS=1`: one top-k over a verify pass's output rows | none |
| `b57f15b9b` | `-c719` `b59cfb001` ([C719](../2026-11-11_c719_pool_mask_on_cards/README.md)) | `-c100` | `LLAMA_QSA_POOL_MASK_DEV=1`: the pool mask built on each card (new op `ggml_range_mask`) | four files against C720 (also one commit on `-c100`), each kept both: `llama-memory-hybrid-idx.cpp` (C720's `kpool_stage_on` beside C719's mask functions), `models.h` (`build_kpool_small` beside `build_pool_mask`), `qwen4exp.cpp` (`set_input` runs both, `can_reuse` tests both, both members, both builders; `build_qsa_sel` keeps C720's staged inputs and C719's local mask, and the indexer is C704's choice of call with C719's local mask in either), `tests/CMakeLists.txt` (both tests) |
| `29fc4e9bc` | `-c217` `68ea07e60` ([C217](../2026-11-11_c217_incall_checkpoints/README.md)) | `pascal-p106-v4` | `--ctx-checkpoints-incall`: a prompt's checkpoints taken inside the decode call | `common/speculative.cpp` and `.h` against C703: C703's `pending`/`flush` and C217's `incall_ok`/`get_state_at` on the drafter's base class and in the API, both kept |
| `2c0d9babb` | `-c701` `769d4c274` ([C701](../2026-11-11_c701_microbatch_trace/README.md)) | `pascal-p106-v4` | `LLAMA_UBATCH_TRACE=1`: each micro-batch's graph reuse, host parts and splits in the log | none |
| `755a4f600` | `-c405` `ca8725c62` ([C405](../2026-11-11_c405_server_gap/README.md)) | `pascal-p106-v4` | `LLAMA_STEP_TRACE=<file>`: host timestamps along every served step | four files, each kept both: `common/speculative.cpp` (C703 moved the draft's catch-up into `catch_up()`, run at once or at the next call; C405's first mark, `mtp.cu`, opens `catch_up()`, so it marks the catch-up where it runs, and its other marks merged into the body as written), `src/llama-context.cpp` (C701's trace clocks beside C405's two marks), `src/llama-ext.h` (C100's and C719's statistics beside the step-trace API), `tests/test-llama-archs.cpp` (C620's and C703's options beside C405's `--mtp`) |
| `24ae0aa83` | | | | C217's `common_speculative_get_state_at` runs a pending catch-up first, as C703 makes every other public drafter call do; with C703's switch off nothing is pending |
| `c408dabcb` | | | | `ggml-rpc.h`: C719's `GGML_OP_RANGE_MASK` is the 106th op, so the RPC protocol's static assert says 106 and its patch version goes from 3 to 4 (a server of patch 3 rejects that op, as one of patch 2 rejects `GGML_OP_MOE_TAIL`). Only builds with `GGML_RPC=ON` compile this header: C701's check, the one that builds with it, failed on `24ae0aa83` |

Every switch keeps its own default, off. Where two switches meet, on together:
- C100's tables on the cards with C620's KV-only catch-up: the KV-only graph leaves the cards' copies out, and C100
  rewrites every copy a graph left out in full at the next pass, so the result is exact; the rewrite costs a full table
  upload after each KV-only pass (C748: the KV-only pass updates the copies and reads C720's staged inputs).
- C217's in-call checkpoints with C703's kept catch-up: the kept catch-up would run without C217's cuts (C735, from
  C217's notes).
- C719's mask ranges are not yet among C720's staged inputs (C733).
- C100, C720, C719 and C704 meet in `build_qsa_sel`; all four are exact, and the `combo` check below runs them on
  together.

## The checks

`run/checks.sh` ran every step on a build machine's CPU, in Docker containers at `--cpu-shares=256` and `nice 19`, one
build or test at a time, no GPU: the whole chain on `24ae0aa83`, then the steps the last commit could change, on
`c408dabcb` (the rig build, the identity with every switch off, C620's check, the replay, C701's check) and the new
`combo` step. Every other build of `c408dabcb` compiles the same code as `24ae0aa83`'s (only builds with `GGML_RPC=ON`
compile `ggml-rpc.h`). Every verdict line is in [summary.tsv](summary.tsv) (its first column is the commit).

| Branch | Its check, run on the merged tree | Result |
|---|---|---|
| every switch off | C620's `off`: `test-llama-archs -b CPU -s 1` on every architecture, the merged CPU build against `pascal-p106-v4`'s logits | 146 of 146 models the same bit for bit (151 of 151 tests pass in both builds), on both commits |
| every switch off | [C565's CPU test set](../2026-11-05_c565_ladder4fix/README.md), static and shared libraries | ctest 70 of 70 (the six that need a download or Python left out, as before), `test-arg-parser` with network, `test-llama-archs` written and compared (146 files), the nine hand tests |
| C675 | its `cpu_check.sh` against `84f7bd3d2` | ctest 5 of 5; the rejection test prints the same 128 lines in 20,000 trials on the base, the merged tree and the merged tree with the switch on |
| C711 | its `cpu_check.sh` against `-c675` | ctest 6 of 6; the switch-off block results' digest over 4,000 random drafts is `-c675`'s (`1404fbd5c5a6a29a`); `test-spec-block` prints `-c675`'s 11 configuration lines alike, and 2 more for the cut |
| C705, C103 | `bench/fattn_sparse/check.sh` on the merged tree (C103's version, which holds C705's rule cases) | 18 ok, 0 WRONG |
| C690, C722 | their `sass_check.sh` on the merged rig build's objects | the same tables as their own builds (13 and 26 kernels) |
| C620 | `--mtp-kv-only` with the rig build and the CPU build, seeds 1 to 3 | 6 of 6 |
| C100 | its `cpu_check.sh` against `pascal-p106-v4`'s CPU build | `test-kpool-dev` 16 of 16 arms, 18,032 outputs bit for bit; `test-llama-archs` off 146 of 146 the same, qwen4exp with the switch, with its check and with `=2` the same |
| C720 | its `cpu_check.sh` | `test-kpool-stage` 16 of 16 arms, 18,032 outputs; off 146 of 146; qwen4exp on, with C100's check and with `=2` the same |
| C719 | its `cpu_check.sh` | `test-pool-mask-dev` 16 of 16 arms, 18,032 outputs, every mask read back equal to the host mask; off 146 of 146; qwen4exp on, with the check, with `=2` and with C100's switch the same |
| C704 | its chain (`cpu tbo archs graph`) | `LIGHTNING_INDEXER` 323 of 323 cases; off and on 146 of 146 the same; at top-k 8 on = off for qwen4exp and glm5-next, and the control (top-k 8 against every cell) differs, as it must; with the switch qwen4exp's graph has 37 `GET_ROWS` fewer and none of its 37 indexers reads a gathered copy |
| C703 | `--mtp-catchup-overlap`, both builds, seeds 1 to 3 | 6 of 6 |
| C702 | `test-cuda-graph-force` and `test-cuda-profile-light` from the rig build's tree | 2,000 random runs of 400 evaluations: with the switch no stale replay, the control stale in 711; 6,070 and 49,267 checks, 0 failed |
| C691 | its chain (`cpu tbo sampler graph`) | `TOP_K`, `TOP_K_ROWS` 682 of 682; the samplers' 85 outputs the same off and on; `test-backend-sampler` 21 of 22 tests off and on, 20 of 21 on the base (`test_logit_bias` fails on every build with the random model, C691's note); the switch-off graph is the base's node for node; on: 9 `TOP_K` fewer, no `PAD` |
| C217 | its `checks.sh` (`cpubuild context control server`) | 71 case runs on 3 models, 0 FAIL, the in-call stores in 6; the control as designed; llama-server off and in-call give the base's tokens, probabilities, cache counts and restored checkpoints on all five requests |
| C405 | its `cpu_check.sh` (`cpubuild smoke`) | 15 of 15 replies the base's (off, on, an unwritable trace file); the traces' drafted and accepted counts the server's (597 and 0 with the MTP draft, 180 and 122 with the draft model); off writes nothing |
| C701 | its `cpu_check.sh` against `84f7bd3d2` | on `24ae0aa83` the build failed (the RPC assert above); on `c408dabcb` 16 of 16: the text the same bytes in the base, off and on, on the CPU and through an RPC server; trace lines only when on |
| C100, C720, C719, C704 together | `combo`: qwen4exp with all four switches and both device checks on against the merged tree's logits with them off | the same, at the model's top-k and at 8 |
| the replay set | `record_checks.sh SUITE=v4night`, then each of its 8 filters replayed on the CPU | 967 cases (9,669,294,080 B); 937 replay and pass, 0 FAIL; the other 30 are large-activation products (`big=`) the CPU backend does not take, which only a card runs |

The rig tarball `llama.cpp-20261111-c408dabcb-pascal-p106-v4-b12246-night.tar.gz` (656,347,720 B, md5
`736363d6ca12e6c81ff5f5b684ef31b3`) is built with `TESTS=1` for compute 6.1: CUDA 12.9, image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`, no AVX, highest glibc symbol 2.38, with `test-backend-ops` and
`test-llama-archs`. The set's tar is `v4night.tar`, md5 `95a1ac635a761619e1f4fe6790424b85`; its run lists and cases are
below. Neither is on a rig yet.

## Files

| File | What it is |
|---|---|
| [run/checks.sh](run/checks.sh) | the driver: every step above, in order, on a build machine |
| [summary.tsv](summary.tsv) | every verdict line: the commit, the row, the check, the verdict, the detail (work paths as `$M`); `LOG` lines are the two chains' log lines, judged in the table above |
| [set_runs.tsv](set_runs.tsv), [set_runs_ptx.tsv](set_runs_ptx.tsv) | the replay set's 52 runs for a P106 and 43 for a newer card through the compute 6.1 PTX: name, switches, `-o`, `-p`, expected (`pass`, or `fail` for a fault run), wrapper |
| [set_manifest.tsv](set_manifest.tsv) | the set's 967 cases: id, case string, input bytes and md5, reference bytes and md5 |

## Commands

On the build machine, with this repository at `$REPO` and a clone of llama.cpp with the fork as remote `fork` (https) at
`$M/w/src` (the work folder of `scripts/build_llamacpp_docker.sh`):

```bash
M=<work root> REPO=<this repository> LOCK=<flock file> BASE_CPU=<bin/ of pascal-p106-v4's static native CPU build> \
  TINY=<a small GGUF> ORIG=<the night sessions' work root, optional> C565_SEED=<optional> JOBS=8 THREADS=8 \
  DOCKER_ARGS='--label <yours>' bash docs/data/2026-11-11_v4_night_merge/run/checks.sh
```

`BASE_CPU` is `pascal-p106-v4`'s static native CPU build (`GGML_CUDA=OFF GGML_NATIVE=ON BUILD_SHARED_LIBS=OFF
LLAMA_BUILD_TESTS=ON`, with `test-llama-archs` and `llama-server`); `TINY` a small GGUF for `test-backend-sampler` (C691
used a random-weight Flint). One step: `STEPS=combo`. Nothing here needs a GPU.

The replay set for the rigs is `scripts/record_checks.sh` with `SUITE=v4night` (the night's seven suites: `qsa-prompt`,
`qsa-union`, `q5kmt`, `dmt`, `indexer-rows`, `range-mask`, `top-k-rows`) and `BIN` the merged rig build's `bin/`; on a
rig, `scripts/replay_checks.sh` with `SET` its folder.

Provenance: the fork is [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama) (MIT, as llama.cpp);
the models the checks run are random-weight models the tests write themselves.
