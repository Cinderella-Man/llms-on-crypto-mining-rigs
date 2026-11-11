# The v4 day merge: the day's 17 engine branches in one branch, `pascal-p106-v4-b12246-day` (2026-11-11)

On 2026-11-11, seventeen fork branches each added one change to the v4 line, each behind its own switch, off by default,
each built and checked on a CPU by its own session: sixteen on `pascal-p106-v4-b12246-c748` (`5b329df16`: the night
merge [`-night`](../2026-11-11_v4_night_merge/README.md) with C748's fix) and one, C749, on `-night`. This folder merges all
of them into one branch, `pascal-p106-v4-b12246-day`, builds it once for the rig (compute 6.1) and runs every merged
branch's own published check on the merged tree, CPU only, plus the `test-llama-archs` identity of the whole tree with
every switch off against `-c748`, the switches of both merges on together, and the GPU code of the merged build function
by function against the branches' own builds. `pascal-p106-v4`, `-night` and `-c748` are not moved. The registry row is
Q1359 in [quants/experiments.md](../../../quants/experiments.md); every switch is in [docs/switches.md](../../switches.md).

**Result:** the merged tree builds for the rig and every merged branch's own check passes on it. With every switch off
it computes what `-c748` computes: `test-llama-archs`' logits are the same bit for bit on all 146 models, and the whole
CPU test set passes (ctest 70 of 70, static and shared). With every switch of both merges on together (25 switches; the
CUDA ones change nothing on a CPU) the logits are still `-c748`'s on 146 of 146 models, and llama-server's MTP-drafted
replies with the stacked run's (R90's) host switches, and with every exact host switch together, are the base's: the
same tokens, probabilities, draft counts and 406 verify steps. Every GPU function of the merged rig build is `-c748`'s
or that of the one branch build that changed it (8,483 of 8,483 in the SASS). One replay set holds the CUDA cases of the
day's seven kernel suites (1,512 cases, 71 runs) for the rigs, and it replays on a CPU. No fault of a merged branch
showed up. Nothing ran on a GPU; the speed of each switch is its rig row's.

## The branch

`pascal-p106-v4-b12246-day` at **`18d0fa309`**: one merge commit per branch on `pascal-p106-v4-b12246-c748` `5b329df16`
(`git merge --no-ff`, no rebase). The order: the host-code branches first, a branch that rewrites a function before the
branches that change the same function (C738 splits `process_ubatch`, which C619, C740, C761 and C615 also change), C761
before C615 (C615's own check merged it with C761), then the kernel branches, each after the branch it was cut from (C766
before C767), C749 (cut from `-night`) and C767 last, as it finished last (14:25).

| Commit | Merge of | Cut from | Brings (off by default) | Conflicts and their resolution |
|---|---|---|---|---|
| `81da58bd3` | `-c735` `9e4bbe897` ([C735](../2026-11-11_c735_c217_c703/README.md)) | `-c748` | C217's in-call checkpoints with C703's kept catch-up give the same bytes as separate calls (the night merge's open combination) | none |
| `16bc17a52` | `-c693` `8d329fa0e` ([C693](../2026-11-11_c693_upstream_switches/README.md)) | `-c748` | `LLAMA_MTP_H_DEV`, `GGML_SCHED_PREFETCH`: switches for two defaults upstream b12246 brought (default: upstream's) | `tests/test-llama-archs.cpp`: C735's `--mtp-incall-overlap` and C693's `--mtp-h-dev` add a mode at the same five places (usage, test function, flag, option, dispatch): both kept, each option under its own `else if` |
| `dd92615bd` | `-c738` `b45b4afd1` ([C738](../2026-11-11_c738_verify_prepare/README.md)) | `-c748` | `LLAMA_VERIFY_PREPARE=1\|2`: a step's verify pass prepared during draft-mtp's last drafting pass | `ggml/src/ggml-backend.cpp`: C693's `prefetch` flag and C738's pre-upload fields at the end of `ggml_backend_sched`, both kept; `tests/test-llama-archs.cpp`: `--verify-prepare` beside the other modes |
| `b159d6f6c` | `-c619` `916d1cf71` ([C619](../2026-11-11_c619_draft_graph_reuse/README.md)) | `-c748` | `LLAMA_MTP_GRAPH_KEEP=1`: the MTP draft's graphs kept per batch shape | `tests/test-llama-archs.cpp`: `--mtp-graph-keep` beside the other modes. `src/llama-context.cpp` merged without a conflict, C619's per-size scheduler lines inside C738's `process_ubatch_graph` (C738's merge note) |
| `d55ebcac9` | `-c740` `f023e94bf` ([C740](../2026-11-11_c740_seq_rm_bound/README.md)) | `-c748` | `LLAMA_KV_SEQ_RM_BOUND=1`: `seq_rm` bounded by the cells it can change | `tests/test-llama-archs.cpp`: `--kv-seq-rm` and `--kv-seq-rm-time` beside the other modes |
| `0d33df02f` | `-c762` `950844caf` ([C762](../2026-11-11_c762_draft_rows/README.md)) | `-c748` | `LLAMA_DRAFT_EMBD_DEV=layer`: the MTP draft's own token table on its layer's card | none |
| `d0bfa4dc8` | `-c741` `be9db2460` ([C741](../2026-11-11_c741_server_yields/README.md)) | `-c748` | `LLAMA_SERVER_YIELD=1\|2`: one yield to the server's task queue a step instead of three | none |
| `95437293e` | `-c761` `cbc253131` ([C761](../2026-11-11_c761_ngram_prefetch/README.md)) | `-c748` | `LLAMA_NGRAM_PREFETCH=1`: the n-gram rows of a prompt call's later micro-batches read ahead by a helper thread | five files against C738 (and C693, C740, C762), each kept both: `src/llama-context.cpp` (C761's changes to the inputs-and-compute block of `process_ubatch` go into C738's `process_ubatch`, the half C738 split it into that sets the inputs: the trace's reset of the n-gram counts, `set_inputs` with the input times, the trace line's `inputs_by`; a call C738 prepares sets its inputs in two parts, not timed by type. In `decode`, C738's refusal of a call it cannot prepare first, then C761's plan of the call's micro-batches, only when the call runs: a prepared call is planned when it is submitted, with its token ids. The constructor's C693 and C761 blocks, C740's and C761's API functions: both kept), `src/llama-context.h` and `src/llama-ext.h` (both sides' members and declarations), `src/llama-graph.h` (C761's `input_time` and `set_inputs(times)` beside C738's `set_inputs_part`), `tests/test-llama-archs.cpp` (`--ngram-prefetch` beside the other modes) |
| `df94b9ad6` | `-c615` `08c2d98c6` ([C615](../2026-11-11_c615_draft_prefetch/README.md)) | `-c748` | `LLAMA_DRAFT_PREFETCH=1`: each drafted token's n-gram rows read ahead as it is drafted | none |
| `cdef63e28` | `-c750` `69dfccc81` ([C750](../2026-11-11_c750_bf16_exact/README.md)) | `-c748` | `GGML_CUDA_MMVF_PASCAL_EXACT=1`: Flash-Next's BF16 `hc_*_down` at 3–4 tokens with the one-token sums | none |
| `3218ce060` | `-c754` `bca4cc4c6` ([C754](../2026-11-11_c754_f32_mmvf/README.md)) | `-c748` | `GGML_CUDA_MMVF_F32_MT=1`: F32 products of 4–8 tokens (the shared expert's gate, the router) on `mul_mat_vec_f` | none |
| `a95ee0dfb` | `-c726` `77343063d` ([C726](../2026-11-11_c726_q3k_multitoken/README.md)) | `-c748` | `GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1`: Q3_K's 2–8-token products read once, with the k-column kernel's sums | none |
| `5319258fb` | `-c757` `1a8c38ea0` ([C757](../2026-11-11_c757_shexp_fused/README.md)) | `-c748` | `GGML_CUDA_MMVQ_PASCAL_GLU_MT=1`: the shared expert's gate, up and SWIGLU of 2–8 tokens in one launch | `tests/test-backend-ops.cpp`: C726's Q3_K cases and C757's `MUL_MAT_GLU` cases at the same place, both kept, C726's first |
| `92e77f05a` | `-c758` `59a147df5` ([C758](../2026-11-11_c758_dmt_default/README.md)) | `-c748` | `GGML_CUDA_MMVQ_PASCAL_DMT=1` now runs R = 1, R79's fastest setting (`_R=2` the old) | none |
| `d5949a158` | `-c766` `9cb1b4a14` ([C766](../2026-11-11_c766_q5k_default/README.md)) | `-c748` | `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1` now runs R = 1 without the prefetch, R75's fastest setting | none |
| `2351765ec` | `-c749` `be5d25629` ([C749](../2026-11-11_c749_lid_rows_fault/README.md)) | `-night` | `GGML_CUDA_LID_ROWS_FAULT=1` (a check's fault) takes a list longer than the cache | none |
| `18d0fa309` | `-c767` `b58a98253` ([C767](../2026-11-11_c767_q5k_one_row/README.md)) | `-c766` | `GGML_CUDA_MMVQ_PASCAL_Q5K_1R=1`: Q5_K's one-token products (the output head's among them) on C690's kernel | `tests/test-backend-ops.cpp`: the comment over Flash-Next's dense products names C726's Q3_K shapes and C767's Q5_K shapes (both sides' shapes are in the list) |

Where two switches meet, on together:
- C738's prepared verify pass with C761's prefetch: decode runs twice for a prepared call (prepared with a stand-in
  token id, then submitted with the real one), so C761's plan of the call is made at the submission only; a prepared
  call is one micro-batch, so the helper has no later micro-batch to read ahead for, and the n-gram input itself, which
  needs the token ids, is set at the submission.
- C738 with C619: C619's per-size schedulers live in C738's `process_ubatch_graph`; C738 prepares the target's verify
  pass and C619 keeps the MTP draft's graphs, so each acts on its own context.
- C615 with C761 and C738: as C615's merge notes give them (its issue of the last token's rows comes after C738's
  `last_pass` hook, before the verify pass is submitted).
- C757's one launch takes the shared expert's products where C758's `DMT=1` takes them otherwise (`ggml_cuda_mmvq_pascal_dmt_takes`, whose answer C758 does not change), and keeps its own prefetch (C758's notes).
- C767 runs one-token Q5_K products on C690's kernel; C766's `Q5K_MT=1` runs its 2–4-token products on it; both are in.

## The checks

`run/checks.sh` ran every step on a build machine's CPU, in Docker containers at `--cpu-shares=256` and `nice 19`, one
build or test at a time, no GPU, on `18d0fa309` (2026-11-11 14:29–15:50). The builds the branches' scripts share were made
once in `main` and hardlinked into each check's folder; each branch's script then ran its own steps there, with the merged
branch as its branch and `-c748`'s static CPU build as its base. Every verdict line is in [summary.tsv](summary.tsv) (its
first column is the row; the first try of main's server step and rig build, which failed on the driver's setup and not
on the code, is kept there as `main-try1`: the scripts lay outside the folder their containers mount, and the build's clone
named a host path as its origin; the rig build's two lines appear twice, copied again by the second try).

| Branch | Its check, run on the merged tree | Result |
|---|---|---|
| builds | C762's `cpu` and `build`: a static native CPU build of every target, a `GGML_RPC=ON` build of `llama-server` and `ggml-rpc-server`, the rig tarball (CUDA 12.9, sm_61, no AVX) with `test-llama-archs` and `test-backend-ops` | all three built, 0 error lines; the tarball has 0 AVX instructions and needs GLIBC 2.38 at most |
| every switch off | C762's `off`: `test-llama-archs -b CPU -s 1` on every architecture against `-c748`'s logits | 146 of 146 models the same bit for bit (151 of 151 tests pass in both builds); the same with `LLAMA_DRAFT_EMBD_DEV=layer` and with `LLAMA_TENSOR_HASH` |
| every switch off | [C565's CPU test set](../2026-11-05_c565_ladder4fix/README.md), static and shared libraries | ctest 70 of 70 (the tests that need a download or Python left out, as before), `test-arg-parser` with network, `test-llama-archs` written and compared (146 files), the nine hand tests |
| every switch on | `combo`: `test-llama-archs -b CPU -s 1` with the 25 switches of both merges on together (the host ones of C620, C703, C100, C720, C719, C704, C691, C619, C738, C740, C741, C761, C615, C762, C693's two flipped from their defaults, and the CUDA ones of C722/C758, C690/C766, C726, C757, C750, C754, C705, C103, C702) | 146 of 146 models the same as `-c748` |
| the stacked run on llama-server | `combo`: [run/server_combo.sh](run/server_combo.sh), a random qwen4exp model with an MTP block and a mapped n-gram table, the MTP draft at 3 tokens, four requests (two greedy, two sampled), `-c748`'s llama-server as the reference | R90's host switches (`LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1`, with the backend sampler) and every exact host switch of both merges with `--ctx-checkpoints-incall`, with and without the backend sampler: each the base's tokens, top probabilities, draft counts and 406 verify steps. The switches that log say they acted: C738 prepared 398 of 398 decodes, C615 issued 6,420 rows, C761 planned 6 calls (3,808 rows from the helper, 0 differing), C703 ran 2 catch-ups behind the next call |
| C762 | its `server` (the RPC build's arms: two RPC devices on this machine, the draft's layer on the second and its table on the first) | every arm gives its reference's tokens, probabilities, draft counts and 406 verify steps |
| C748 | its `kvonly`, `overlap`, `kvrig` (seeds 1 to 3) | `--mtp-kv-only` 21 of 21 runs with the CPU build and 21 of 21 with the rig build, over 7 switch sets; the control (C748's test without the fix) fails as designed on the tables, 18 of 21 runs, and passes plain; `--mtp-catchup-overlap` 9 of 9 |
| C735 | its `incall`, `overlap`, `kvonly`, `context`, `server`, `incallrig` | `--mtp-incall-overlap` 9 of 9 with the CPU build and 9 of 9 with the rig build; `--mtp-catchup-overlap` 9 of 9, `--mtp-kv-only` 6 of 6; C217's context cases 72 runs, 0 FAIL; llama-server: every arm (the switches alone and together, in-call checkpoints at vram 2, 4 and 8, spread 200) the base's tokens, probabilities, cache counts, checkpoints and 70 verify steps |
| C693 | its `hdev`, `overlap`, `off`, `server`, `hdevrig` | `--mtp-h-dev` 9 of 9 and 3 of 3 with the rig build; `--mtp-catchup-overlap` 3 of 3; off with `LLAMA_MTP_H_DEV=0`, `=2`, `GGML_SCHED_PREFETCH=0` and both: 146 of 146 the same each; the server 9 of 9 arms the base's tokens and draft counts, the h rows where each arm puts them |
| C619 | its `keep`, `regress`, `server`, `keeprig` | `--mtp-graph-keep` 9 of 9 and 9 of 9 with the rig build; C620's and C703's checks with the switch 6 of 6; llama-server 36 of 36 replies and dumps identical, the draft's graphs of up to 8 tokens rebuilt 6, 7 and 6 times instead of 411 |
| C738 | its `vprep`, `fault`, `server`, `off`, `vrig` | `--verify-prepare` 6 of 6 and 6 of 6 with the rig build; the fault fails 26 of 196 prepared rows, as designed; off with `=1` and `=2` 146 of 146 the same; llama-server, the switch unset, `=1` and `=2`, with and without the backend sampler, and the RPC build's arms: the base's tokens, probabilities, draft counts and 406 verify steps |
| C740 | its `bound`, `control`, `time`, `server`, `trace`, `boundrig` | `--kv-seq-rm` every arm OK with the CPU and the rig builds; five tests (plamo2 and minimax-01) stop on a ggml assert with the switch off in both contexts too, as on `-c740`, and are left out of their arms; the control finds its difference; the timing runs; llama-server 36 of 36 identical, at 65,536 tokens 16 of 16 |
| C761 | its `npf`, `server`, `off`, `vrig` | `--ngram-prefetch` 3 of 3 with each build; llama-server: every arm (on, a 64 KiB cap, the trace off and on) the base's tokens, probabilities, draft counts and 406 verify steps; off with the switch and with its counts 146 of 146 the same |
| C741 | its `unit`, `server`, `off` | `test-server-component server_queue_yield` 5 of 5 runs, 22 assertions each; llama-server `=1`, `=2`, with and without the backend sampler, and two slots: the base's replies; off with `=2` 146 of 146 the same |
| C615 | its `server`, `off`, `rsrv`, `trace` | llama-server: every arm the base's; with the switch each of 406 evaluations after a draft found all its rows issued ahead (6,448 of 6,448), also with the rig build's server; the cap of 0 issues nothing; off with the switch and with its counts 146 of 146 the same |
| C615 with C761 | `combo`: C761's `--ngram-prefetch` with `LLAMA_DRAFT_PREFETCH=1` (C615's merge test) | 3 of 3 seeds pass |
| C749 | its chain, `emu replay tables`: a CPU stand-in of the fault built from the merged tree, the night set's `lid-rows-fault` run (113 in-place indexer cases) | the rig build with the switch passes 113 of 113 (the CPU ignores it); the stand-in passes 113 of 113 without it and fails 113 of 113 with it, ending with its summary |
| C750, C754, C726, C757, C758, C766, C767, C749 | `sass`: [run/sass_attrib.py](run/sass_attrib.py) on `cuobjdump -sass` and `-ptx` of `llama-server` from the merged rig build, `-c748`'s and each kernel branch's own (C767's against C766's, which it was cut from) | **SASS:** 8,483 of 8,483 functions are `-c748`'s (8,428) or the code of the one branch that changed them: C726 28, C757 14, C750 8, C767 4, C749 1; C754, C758 and C766 change no kernel (they choose among existing ones). So each kernel branch's own SASS checks (instruction counts, registers, the launch picks) hold on the merged build as they stand. **PTX:** the same, but for 10 `flash_attn_tile` kernels whose PTX in C726's and C757's own builds differs from `-c748`'s only in the name hash of a file-local shared variable (`_INTERNAL_<hash>_..._cu_<hash>`), which comes from how those builds were run, not from their code (neither touches flash attention; their SASS is `-c748`'s); the merged build's PTX of them is `-c748`'s |
| C750, C754, C726, C757, C758, C767 | `bitid`: `bench/mmvq_pascal/verify/multitoken_bitid` linked against the merged rig build's static libraries | built (modes `dmt`, `q5k`, `q5k1`, `q3k`, `bf16`, `f32`, `glu`): the rig rows' bit-identity tool for this build |
| the replay set | `record`, `replay`: the set `v4day` recorded with the rig build's `test-backend-ops`, then every distinct filter of its runs replayed on the CPU | 1,512 cases (1,503 hardlinked from the branches' own sets, 9 recorded here); the 10 filters replay 1,578 case runs, 0 FAIL, no case of a filter without its record |

Left out, as on the night merge: each script's own CPU build, ctest and rig build (main's); the `off` steps that only
repeat main's (C748's, C735's, C619's, C740's); C735's `nohalf` control and C615's `merge` test, which build a tree of
their own (the merged tree holds C761, so C615's merge test runs as the `combo` line above); C726's, C757's and C758's own
`record` and `cpu`/`off` with a CUDA switch (the CPU ignores CUDA switches: the `combo` line covers them); C750's and C754's
order emulations (they test the kernels' arithmetic in C, independent of the tree).

## The replay set for the rigs

`scripts/record_checks.sh` with `SUITE=v4day` (added here, with `LINK_FROM` now taking several sets) records the CUDA
cases of the day's seven kernel suites as one set: `dmt` (C722, at C758's default), `q5kmt` (C690, at C766's), `q3kmt`
(C726), `bf16x` (C750), `f32x` (C754), `glumt` (C757) and `q5k1r` (C767). Of its 1,512 cases, 1,503 are the very files the
branches' own sets hold (the same case string gives the same id; hardlinked from the night set, C726's, C757's, C767's,
C750's and C754's), and 9 were recorded with the merged build. Set `v4day`: 71 runs (dmt 14, q3kmt 12, q5k1r 11, q5kmt 10, bf16x 9, glumt 9, f32x 6), tar 5,473,720,320 bytes, md5
`f23c05d9365e55b70df90401eb064ca4`. On a rig:
`scripts/replay_checks.sh` with `SET` the unpacked set and `BIN` the merged tarball's `bin/`, every run of its
`runs.tsv`; it stands for the replay steps of the kernel rows (C750, C754, C726, C757, C767) and for the `dmt` and `q5kmt`
runs at the new defaults.

## The rig tarball

`llama.cpp-20261111-18d0fa309-pascal-p106-v4-b12246-day.tar.gz`, 671,157,298 bytes, md5
`ad766d3f8198be0c77f9f41ebf9c69d5` (`TESTS=1`, CUDA 12.9 image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, sm_61, 0 AVX
instructions, highest glibc symbol GLIBC_2.38, `test-llama-archs` and `test-backend-ops` in `bin/`). Every switch is off
by default, so with nothing set it runs `-c748`'s code; the stacked run (R90) sets the switches its rig rows proved:
`GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1
LLAMA_QSA_KPOOL_DEV=1` (on this build `DMT=1` and `Q5K_MT=1` alone are R79's and R75's fastest settings).

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the driver: every step above, each branch's script run on the merged tree |
| [run/server_combo.sh](run/server_combo.sh) | llama-server with the switches of both merges on together against the base |
| [run/sass_attrib.py](run/sass_attrib.py) | which build each GPU function of the merged build comes from |
| [summary.tsv](summary.tsv) | every verdict line (row, check, verdict, detail) |
| [sass_attrib.txt](sass_attrib.txt) | `sass_attrib.py attrib`'s output for the SASS and the PTX |
| [set_runs.tsv](set_runs.tsv), [set_runs_ptx.tsv](set_runs_ptx.tsv), [set_manifest.tsv](set_manifest.tsv) | the set's runs (on a P106-090, and on a newer card through the PTX) and its cases (id, case, input and reference bytes and md5s) |
| [server_combo.tsv](server_combo.tsv) | the combination's server arms: per request the prompt and generated tokens and the draft counts |

## Commands

On the build machine, with this repository at `$REPO`, a llama.cpp clone with the fork as remote `fork` at `SEED_SRC`, the
base branch's static CPU build at `BASE` and its rig tarball tree at `BASE_RIG`:

```bash
M=<work folder> REPO=$REPO LOCK=<flock file> BASE=<-c748's CPU build/bin> BASE_RIG=<-c748's rig tarball tree> \
SEED_SRC=<clone> C748_BEFORE=<C748's control build/bin> NIGHT_SET=<the v4night set> \
SASS_BUILDS='c750=<tree> c754=<tree> c726=<tree> c757=<tree> c758=<tree> c766=<tree> c767=<tree>@c766 c749=<tree>' \
LINK_FROM='<v4night set> <q3kmt set> <glumt set> <q5k1r set> <bf16x set> <f32x set>' \
bash docs/data/2026-11-11_v4_day_merge/run/checks.sh
```

It ran in three parts on a disk with little room (`STEPS` names the steps): `main` (`MAIN_STEPS='server build'` the
second time), `prep` to `sass`, then `bitid c749 c565`, then `record replay`, with the build trees the later steps no
longer read deleted in between.

## What it decides

- The day's branches can be run together from one build: R90 (the gains together) runs on this tarball, and every open
  rig row of the day's and the night's switches can take it in place of its branch's, with only its own switch set (the
  others off compute what `-c748` computes).
- `pascal-p106-v4` is not moved; whether it fast-forwards to `-day` once R90 has run is the owner's call.

Provenance and licence: the cases are random tensors drawn by `test-backend-ops`; the models are `test-llama-archs`'
random-weight models; everything here is this repository's MIT licence.
