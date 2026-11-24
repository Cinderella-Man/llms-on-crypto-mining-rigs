# v5 on upstream b12674: the merge and its checks on the build machine (C912)

Registry: [Q1649](../../../quants/experiments.md). Date: 2026-11-24.

**On the build machine, v5 on upstream b12674 checks out.** With every switch at its default the three release files
give the same greedy replies, token for token, and hold the same memory on every card as `pascal-p106-v5` does. Every
test that ran on both builds gives the same verdict, and the card suite passes with the fork's Pascal kernels forced, after
one fork fix that upstream's new tests exposed. Every GPU kernel and every random-model logit that changed is traced to a
named upstream change. Before `pascal-p106-v5` moves to it, a P106 rig checks it, and one coding-agent run each on Ornith
and Flash-Next shows no loss.

## What was merged

Miner Llama's cumulative engine branch `pascal-p106-v5` stood at `af00bb13b`: upstream llama.cpp b12483 plus every
fork feature landed so far, the last 14 branches that morning ([land-v5-batch2](../2026-11-24_land_v5_batch2/README.md)).
The project merges upstream's newest tag after each batch of fork work, so v5 stays on upstream's current code. The
newest tag was **b12674** (`fd24ee26a`, 2026-11-23), 191 commits past b12483. It was merged with `git merge --no-ff`
(pushed history is never rebased) on the candidate branch `pascal-p106-v5-cand-b12674` of
[the fork](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v5-cand-b12674):

| Commit | What |
|---|---|
| `57dcd92f3` | the merge of `b12674` (first parent `af00bb13b`) |
| `2003d7573` | the device-built KQ mask's reuse checks read the model's hparams (upstream #32596 removed them from the graph parameters) |
| `d7d4ef4a7` | `server_prompt`'s `lora` field once (upstream merged #32434, which the fork carried, and both sides had added the field) |

`pascal-p106-v5` itself does not move: as with b12483, the candidate lands after a check on a P106 rig and one coding
agent run each on Ornith and Flash-Next.

## The conflicts

26 files conflicted in 50 hunks. Each was resolved keeping both sides' intent; [conflicts.tsv](conflicts.tsv) lists
every file with upstream's change, the fork's and the resolution. The ones that needed more than putting both sides'
lines together:

- **The op table.** Upstream added two ops after `GGML_OP_GLU`, where the fork keeps its own three. Upstream's go after
  the fork's, so the fork's op numbers (which its RPC protocol depends on) stay; `GGML_OP_COUNT` is 105 and the fork's
  RPC patch version 6.
- **Weight scales.** Upstream now applies a weight's `<name>.scale` tensor by itself (#32669) and dropped the explicit
  scale arguments. The fork's two-tier expert layout (a "cold" set of experts in tensors of their own) passed its own
  scales; its scale tensors carry their tensors' names, so the automatic lookup finds them, and the explicit path went.
  Every caller of the two changed functions was checked for arguments now landing in the wrong place: none.
- **QSA k-pools** (`qwen4exp`, Flash-Next's architecture). Upstream's indexer now scores only the pools of the
  sequences in the micro-batch (#32519). The fork can keep the pool tables on the cards and update only changed rows
  (`LLAMA_QSA_KPOOL_DEV`, on in Flash-Next's preset); those tables hold every sequence's pools, so with them the list
  keeps every sequence's pools as before, and without them it is upstream's.
- **The MoE down projection's rescale** (Flint's architecture is one of four that rescale it): upstream's new rule
  (#32677) and the fork's rule that skips it for one-token passes on a P106 both apply.
- **The attention mask built on the cards** moved into upstream's split attention function (#32429), at its old place
  in the graph.
- **Quantization**: N2P and N3P's fallbacks went into upstream's rewritten fallback table (#15550).

## Carried upstream fixes

| Fork commit | Upstream | In b12674 | Now |
|---|---|---|---|
| `602691153` "slot_sched: fix lora and recurrent", carried since b12483 | #32434 | yes, merged 2026-11-18 (`52975342f`) | upstream's version taken; its code is the carried one's, its test runs one token a micro-batch and compares logprobs exactly. The fork keeps only its own additions there (`--ctx-checkpoints-share`'s copy, the sliding-window trim's two arguments) |
| `2b2da8a12` "clear a recurrent state with a fill" (Q1158) | #31913 | yes, merged 2026-11-19 (`0a7675746`) | the same two lines; upstream also tests a NaN-filled state, the fork's own `nan_clear` case stays beside it |
| `5385cfd25` `rs_max` moves with the sequence | #31656 | no | still carried |
| `d86fc66c2` per-size graph arenas and schedulers for small batches (`LLAMA_SCHED_PER_SIZE`, off by default) | a port of #31156 | no | still carried |

## One behaviour that is the owner's call: #32239

Upstream merged #32239 (2026-11-17): no quantized matrix-vector fusion on any Pascal card. Git merged it without a
conflict. The fork already has that rule as a switch, `GGML_CUDA_MMID_GLU_UNFUSE=2` (C865), measured per file on the
P106: a gain on Ornith (whose preset sets it), a loss on Flash-Next (a steady step +2.3 % at depth 0; its preset leaves it
unset), a small gain on Flint. As merged, the switch would have done nothing and every file would run unfused, Flash-Next
at its measured loss. **The candidate keeps the fork's default** (the fusion on compute 6.1 unless the switch is `2`), so
every preset runs as measured. Whether the default should follow upstream, with a way to keep Flash-Next's fusion, is
C893 (Q1613), which waited for exactly this merge.

## What upstream changed that a P106 runs

[watch.md](watch.md) goes through the 191 commits for compute 6.1, speculation, KV removal, the batch API and the
release files' architectures. The two that change the P106's own prompt kernels are #31352 (64-column passes take
128-row tiles on Pascal: Flint's and Flash-Next's 64-token micro-batches) and #32374 (64-bit output offsets in
`mul_mat_q`). Both are speed questions for the rig, not correctness ones; C1024 is the row for the first.

## The build

One rig build of the candidate (`scripts/build_llamacpp_docker.sh` with `TESTS=1`, the web page of upstream b12674,
CUDA 12.9, sm_61), the base's tarball (`af00bb13b`, md5 `b51ad2740d5822e7e14af656c413ec22`) beside it:

| | Candidate `eee615cdb` |
|---|---|
| tarball | `llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674.tar.gz`, 1,403,235,928 bytes, md5 `575ef3c702f69f14d7d3022c1287c38f` |
| `bin/llama-server` | `2727faf28b5b34dfe42399a9d431da67` |
| `bin/test-backend-ops` | `7e6731c4d42a98cfd6a90cf4bdeb0cc5` |
| AVX instructions in `llama-bench`, highest glibc symbol | 0, GLIBC_2.38 |
| warnings | 32 lines on both builds, none new |

[binaries.tsv](binaries.tsv) has every binary's md5 on both builds.

## The checks

Correctness only: nothing was timed. [checks.tsv](checks.tsv) has every check with both builds' results.

| Check | Base `af00bb13b` | Candidate | Verdict |
|---|---|---|---|
| CPU build, every target (`-DGGML_RPC=ON`) | OK | OK | the same; one CMake warning moved lines (the web page's, offline) |
| ctest, test by test ([ctest.tsv](ctest.tsv)) | 79 of 80 pass | 82 of 83 pass | the same: `test-kv-cache-scales` fails on both, as before; upstream's new `test-backend-sched`, `test-quantize-target` and `test-sleep` pass |
| hand-run tests, verdicts | 9 of 9 OK | 9 of 9 OK | the same |
| CPU pipe step: 5 architectures × identity, sampled, race, rollback ([pipe.tsv](pipe.tsv)) | 20 of 20 OK | 20 of 20 OK | the same |
| card pipe step: one card and split over three, F32 on llama.cpp's kernels and Q8_0 on the fork's forced ([gpu_pipe.tsv](gpu_pipe.tsv)) | 60 of 60 OK | 60 of 60 OK | the same; plain decoding's hashes the same for `flint` and `qwen35`, other for `qwen4exp`, `qwen35moe` and `qwen3next`, whose random models gained a scale tensor (below) |
| greedy replies, three release files × two requests ([replies.tsv](replies.tsv)) | | | the same 256 token ids in all six, the same text, Flash-Next's drafts the same (239 drafted, 175 accepted; 286 and 159) |
| memory per card after each load and after its requests | | | the same MiB on every card |
| `test-backend-ops MUL_MAT`, the fork's Pascal kernels forced | 5,125 of 5,125 | 5,161 of 5,161 | passes (36 new upstream cases) |
| `MUL_MAT` with `GGML_CUDA_MMVQ_PASCAL_MT=2` | 5,125 of 5,125 | 5,161 of 5,161 | passes |
| `MUL_MAT_ID`, forced | 2,370 of 2,370 | 2,389 of 2,389 | passes (19 new upstream cases, the 65,536-token one among them, after the fix below) |
| `FLASH_ATTN_EXT` unset / with the Pascal tile forced / with every Pascal attention switch | 4,679 of 4,679 each | 4,679 of 4,679 each | passes, the same cases |
| random-model logits, `test-llama-archs -b CPU -s 1` (rig builds and CPU builds) | 148 files | 148 compared | 120 differ, all from upstream's automatic weight scales (below) |
| every sm_61 kernel of `llama-server` ([sass_families.tsv](sass_families.tsv)) | 8,842 | | 4,572 byte for byte the same; every difference explained (below) |

The release presets ran on three RTX 3090s as on a P106: the layer split changed to the three cards (`-ts 1,1,1`), and
the fork's Pascal matrix-vector kernels forced with a P106's launch shapes, since on these cards they stay off unless
forced. Flash-Next used Swift's own MTP head as its draft (the preset's draft file is on the rigs only).

### The random-model logits: upstream applies weights' scales by itself

`test-llama-archs` builds a random model of every architecture without a file, and the loader then creates every
optional tensor, `.scale` ones included, filled with random values. Upstream's #32669 now applies each weight's
`<name>.scale` automatically, where before only the scales a model's graph code passed were applied. So 120 of the 148
architectures give other logits than on the base, among them `qwen35moe` and `qwen4exp`; `flint`, `qwen3`, `qwen3moe` and
`qwen35` are the same, since their graphs already passed every scale they have.

The trace: the candidate built once more with the automatic lookup switched off (a one-line local commit, never on the
fork) and compared against the base's logits files. **91 of the 120 then give the base's logits bit for bit.** The 29 that
differ in that build are exactly the architectures whose model files #32669 stopped passing scales explicitly (`flint`,
`qwen3moe`, `qwen35moe`, `qwen4exp` among them), which that build therefore leaves unscaled. The release files carry no
`.scale` tensor, so this changes nothing for them, which the replies above confirm. The hand-run tests' logs differ for
the same reason: the random `qwen4exp` model gains `blk.N.ffn_gate_up_exps.scale` (upstream #32657 loads it) and one
metadata key, and every hand-run test passes on both builds. So do the card pipe step's models, which each build's own
`test-llama-archs` writes: the `qwen4exp`, `qwen35moe` and `qwen3next` files gain `blk.N.ffn_gate_up_exps.scale`, so
their plain runs differ between the builds (`flint` and `qwen35` gain only the key and decode the same), while within
each build every pipelined run gives plain's tokens and logits.

### The GPU kernels: two upstream changes reach the P106's prompt kernels

`cuobjdump -sass` of both builds' `llama-server`, one kernel at a time ([sass_families.tsv](sass_families.tsv)). Of the
base's 8,842 kernels, 4,572 are byte for byte the same, among them every one of the fork's Pascal kernels (matrix-vector,
attention, MoE router, Gated DeltaNet). The rest:

| Kernels | What changed | Cause |
|---|---|---|
| 308 `mul_mat_q` with real code | 12 to 36 more instructions of 64-bit address arithmetic (`IADD.X`, `LEA.HI.X`, `SHF.L.U64`) | #32374, 64-bit output offsets |
| of those, the 64-column (J=64) kernels of Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ2_XXS, IQ3_XXS, IQ4_NL, IQ4_XS | 648 to 1,230 more instructions: twice as many outputs per thread | #31352, 128-row tiles for 64-column passes on Pascal |
| 896 `mul_mat_q_stream_k_fixup` | 18 to 36 more instructions | #32374 |
| 700 `mul_mat_q`, 768 `mul_mat_f_ids` | stubs a compute 6.1 card never runs (an error path); only a printed source line moved | upstream's edits above them |
| 1,536 `mul_mat_f` missing, 768 new | the tensor-core kernel split into a dense and an id kernel; stubs on compute 6.1 | upstream |
| 1,008 `mul_mat_q` new | the `force_tiling` instances, launched on AMD GCN only | upstream |
| 8 `mm_ids_helper`, `k_skip_ids_zero_rows`, `batch_copy` renamed | new parameters, a renamed file | #32505, this candidate's fix, #32489 |
| `set_rows`, `rope`, `norm`, `conv2d`, `pool1d`, `ssm_conv`, `sleep_ns` new instances | BF16 rows and ropes, fused norms, new ops | #32387, #32454, #31855, #27258 and others |

So on a P106, with the switches unset, the candidate runs the base's kernels except in prompt processing (`mul_mat_q`):
every such kernel computes its output offsets in 64 bits, and the 64-token micro-batches of the types listed take
128-row tiles. That is where the rig check looks first (C1024).

## A fork fault the new tests found

Upstream's #32505 added a `MUL_MAT_ID` case of 65,536 tokens. The fork's zero-rows kernel, which every `MUL_MAT_ID`
runs by default (it clears the outputs of slots whose expert id is -1, for the MoE tail), launched one grid row per token,
and a grid has at most 65,535 rows, so the case stopped with "invalid configuration argument" (the first pass of this
check, on `d7d4ef4a7`). Commit `eee615cdb` caps the grid and loops over the rest; up to 65,535 tokens every launch is as
before. No release preset comes near: their micro-batches are at most 512 tokens. The whole card part ran again on
`eee615cdb`.

## Next: on a P106 rig

The rig check is C854b's, as C854 step two ran it: the new build against `pascal-p106-v5`'s, every switch unset, the suite,
then the identity check and launches a token on the four release files, then one coding-agent run each on Ornith and
Flash-Next on their v5 presets. The archs part compares the candidate against itself, since the random models differ from
the base's (above). If they pass, `pascal-p106-v5` fast-forwards to `eee615cdb`.

## Files

| File | What |
|---|---|
| [conflicts.tsv](conflicts.tsv) | every conflicted file: upstream's change, the fork's, the resolution |
| [watch.md](watch.md) | upstream b12483..b12674 read for what a P106 runs |
| [checks.tsv](checks.tsv) | every check, both builds, the verdict |
| [ctest.tsv](ctest.tsv) | ctest test by test on both CPU trees |
| [pipe.tsv](pipe.tsv), [gpu_pipe.tsv](gpu_pipe.tsv) | the CPU and card pipe steps' verdicts |
| [replies.tsv](replies.tsv) | every greedy reply: tokens, characters, drafts, memory per card, the binary's md5 |
| [bops.tsv](bops.tsv) | the card suite per arm |
| [sass_families.tsv](sass_families.tsv) | the sm_61 kernels per family: identical, different, missing, new |
| [binaries.tsv](binaries.tsv) | both builds' binaries' md5s |
| [run/run.sh](run/run.sh) | the driver (land-v5-batch2's with the SASS step, Flash-Next and the second request) |
| [run/sass_check.sh](run/sass_check.sh), [run/sass_compare.py](run/sass_compare.py), [run/sass_classify.py](run/sass_classify.py) | the SASS comparison (C854b's, C861's tool reading `mul_mat_q`'s seventh template argument) |
| [run/tables.py](run/tables.py) | these tables from a copy of the work folder |

## Commands

On the build machine, the driver detached, the work folder `W` on its fast disk, the base's tarball and the release
files where the variables say (the machine's own paths stay in the private launcher):

```bash
W=<work dir> SEED=<llama.cpp clone holding af00bb13b> REPO=<copy of run/ and the scripts it calls> \
  HEAD_SHA=eee615cdb9d26c917de01e3c49f794c41de31684 \
  BASE_TGZ=<llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2.tar.gz> BASE_MD5=b51ad2740d5822e7e14af656c413ec22 \
  BASE_RIG_LOG=<its build log> QUANT=<llama-quantize> PREFLIGHT=<cuda_preflight.sh> \
  FN=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf> DRAFT=<Swift's MTP head> \
  FLINT_Q8=<flint-pascal-Q8.gguf> ORNITH=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> \
  MOUNTS='<folders the containers mount>' GPUS=0,1,2 LOCK=<build lock> JOBS=8 \
  setsid nohup bash run.sh > run.out 2>&1 < /dev/null &
```

The first pass ran every step on `d7d4ef4a7` (`HEAD_SHA=d7d4ef4a7422426c1bc273ee40aa13211f7d8dab`); the second,
`STEPS="src rig sass"` with every card step, on `eee615cdb` after the zero-rows fix (the CPU steps do not compile CUDA
code, so their results stand). The trace ran `cpu_tests.sh` (`docs/data/2026-11-05_c565_ladder4fix/run/`) on the local
commit with `TREES='noscale=<commit>:test-llama-archs@base' STEPS='tree build ident'`, the base's logits files beside it.
Tables: `python3 -I run/tables.py <copy of W> .`

Provenance: upstream llama.cpp (MIT) and Miner Llama's fork (MIT), the random models `test-llama-archs` generates, and
the project's own release files (Flint Pascal Q8 GGUF, Ornith-1.5-122B Pascal N2P GGUF, Swift Qwen3.8-Flash-Next Pascal Q2
GGUF with Swift's MTP head, as published with their releases).

