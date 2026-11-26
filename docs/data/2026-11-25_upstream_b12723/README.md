# v5 on upstream b12723: the merge and its checks on the build machine (C1118)

Registry: [Q1851](../../../quants/experiments.md). Date: 2026-11-25.

**v5 on upstream b12723 checks out on the build machine.** With every switch at its default, the three release files
give the base's greedy replies token for token and hold the same memory on every card, and every test that ran on both
builds gives the same verdict, the whole card suite included. Three upstream changes would have changed how a release
file runs, so the candidate keeps the measured behaviour and puts upstream's behind a switch each, to be measured on a
P106. The [final build](#the-final-build), the candidate's tip `caf27fdae`, differs from the first build in one function
of Flash-Next's graph that only upstream's replay switch reaches: the same logits, the same machine code for the cards
and Flash-Next's same replies, and with upstream's three switches Flash-Next now loads and answers. Before
`pascal-p106-v5` moves to the candidate, a P106 rig checks the final build, and one coding-agent run each on Ornith and
Flash-Next shows no loss.

**The P106 check passed on 2026-11-26** ([2026-11-26_c1118_b12723_p106](../2026-11-26_c1118_b12723_p106/README.md)):
on the v5 presets every release file gives the same replies, the same MiB per card and the same launch groups on the
final build as on `eee615cdb` (`2d839da1d` without its seven fork merges, which change none of these), the kernel record
sets give the last check's results case by case, and the random models differ in exactly the 8 files below, by the
same amounts. The agent runs remain: Flash-Next's, and Ornith's as a compatibility check.

## What was merged

Miner Llama's cumulative engine branch `pascal-p106-v5` stood at `2d839da1d`: upstream llama.cpp b12674 plus every fork
feature landed so far, the last eight branches that afternoon ([land-v5-batch3](../2026-11-25_land_v5_batch3/README.md)).
The project merges upstream's newest tag after each batch of fork work. The newest tag was **b12723** (`abca69200`,
2026-11-25), 49 commits past b12674. It was merged with `git merge --no-ff` (pushed history is never rebased) on the
candidate branch `pascal-p106-v5-cand-b12723` of
[the fork](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v5-cand-b12723):

| Commit | What |
|---|---|
| `0a6a04e22` | the merge of `b12723` (first parent `2d839da1d`), with the three switches below |
| `caf27fdae` | Flash-Next's architecture (`qwen4exp`) builds its conv window without the fork's `GDN_CONV` op when `LLAMA_RS_REPLAY=1`: that op reads the snapshot layout, and upstream's replay keeps another one in the same row (with the switch on, the load asserted). With the switch unset the graph is the same |

`pascal-p106-v5` itself does not move: the candidate lands after a check on a P106 rig and one coding-agent run each on
Ornith and Flash-Next.

## The conflicts

7 files conflicted in 20 hunks, all around two upstream changes: the recurrent state's new rollback (#31669) and the
fused norm and RoPE kernel (#32500). Each was resolved keeping both sides; [conflicts.tsv](conflicts.tsv) lists every
file with upstream's change, the fork's and the resolution.

- **The recurrent state's memory** (`llama-memory-recurrent`): the fork keeps state groups there for pipelined
  decoding; upstream added the replay's rows (a wider conv history and the pending tokens' inputs). Both are in: the
  replay's rows exist only with the replay on, and without it the rollback snapshots stay groups of the state tensors,
  as before.
- **The gated delta net kernel**: the fork's in-place state read (each sequence's state read straight from its cache
  row) and upstream's replay arguments both pass through every launch.
- **The fused norm and RoPE kernel** (`rope.cu`): the fork had moved its arguments into a structure that its one-token
  attention block (`GGML_CUDA_ATTN_BLOCK`) reuses; upstream's M-RoPE sections and modes went into that structure.

## Upstream defaults kept behind switches

Three upstream changes would change how a release file runs on a P106 with no switch set, and none was measured on a
P106. As with #32239 in the b12674 merge, the candidate keeps the measured behaviour, and a switch gives upstream's:

| Switch | Upstream's change | Unset (the candidate's default) | Registry row |
|---|---|---|---|
| `LLAMA_RS_REPLAY=1` | #31669: the delta-net models roll back by replaying the kept tokens' state updates in the next micro-batch instead of restoring a snapshot; upstream removed the snapshots from those models' graphs | a rollback restores one of `n_rs_seq + 1` snapshots (Flash-Next's MTP draft writes four a verify, through the fork's GDN fusions) | [Q1852](../../../quants/experiments.md) |
| `GGML_CUDA_NORM_ROPE_MROPE=1` | #32500: the fused RMS_NORM + MUL + ROPE takes M-RoPE and IMROPE rows, so Ornith's and Flash-Next's attention layers run their Q and K norms and RoPE in one launch | three launches, as before (Flint, NEOX RoPE, fuses already) | [Q1853](../../../quants/experiments.md) |
| `GGML_CUDA_NORM_ROPE_BS=1` | #32500: that kernel's block size by row length and token count (32 to 256 threads) | 256 threads a row below 1,024 columns, so a row's sum of squares is added in one order for one token and for several | [Q1854](../../../quants/experiments.md) |

The one-token attention block now refuses M-RoPE rows (its epilogue has no sections), and the fork's in-place state
read is off for the replay form. [watch.md](watch.md) lists every upstream change that reaches a P106 and what to
watch on the rig.

## Carried upstream fixes

| Fork commit | Upstream | In b12723 | Now |
|---|---|---|---|
| `5385cfd25` `rs_max` moves with the sequence when `find_slot` swaps cells | issue #31656, fixed inside #31669 | yes, merged 2026-11-23 | upstream's code (it also swaps the replay's pending count) |
| `d86fc66c2` per-size graph arenas and schedulers for small batches (`LLAMA_SCHED_PER_SIZE`, off by default) | a port of #31156 | no, still open | still carried |

## The checks

On a build machine with three RTX 3090 cards (correctness only, nothing timed), running the rig build's compute 6.1 code
through the driver's JIT, the candidate (head) against `pascal-p106-v5` `2d839da1d`'s `TESTS=1` tarball (base, md5
`7d774f05…`). The driver is [run/run.sh](run/run.sh): land-v5-batch3's, with Flash-Next and a second request back in the
model loads, a load of each model with the three switches on, a card-suite arm for the operators the merge changed and
a machine-code step. The results, the head `0a6a04e22` against the base (the card suite's last arms ran on two of the
three cards, the first card taking the third card's arms after its own):

| Check | Head against base |
|---|---|
| `TESTS=1` rig build | 32 warnings on each, none new; no AVX instruction, newest glibc symbol `GLIBC_2.38` on both |
| `test-llama-archs -b CPU -s 1` of the rig builds | 157 of 157 tests pass on both; 8 of the 152 logits files differ (see below) |
| CPU trees: ctest | 82 of 83 on both, the same verdict test by test (`test-kv-cache-scales` fails on both) |
| CPU trees: the hand-run tests | every test passes on both; the logs differ only in the new `rs_replay = false` line, paths and reserve times |
| CPU trees: the logits (ident) | the rig builds' 8 files plus `inkling2` and `inkling2_loop` (see below) |
| CPU pipe step | every architecture and arm OK on both trees |
| card pipe step | every arm, layout, architecture and check OK on both builds |
| machine code (`cuobjdump -sass` of `test-backend-ops`) | base 10,075 kernels, head 10,099; 9,581 the same bytes; different bytes: `mul_mat_vec_f` 440 (#32965), `soft_max_f32_parallelize_cols` 7 (#32897), `dequantize_block_nvfp4` 3 (#32875), `rope_multi` 20 (one instruction each); renamed by #31669's template argument: the gated delta net's 8 non-KDA instances the same instructions, its 8 KDA instances different (KDA models only); `rms_norm_mul_rope`'s 8 instances the same instructions apart from parameter offsets |
| greedy replies, the three release files, two requests each, switches unset | the same token ids on both builds; Flash-Next's MTP draft 236 drafted and 176 accepted on the coding request, 286 and 159 on the prose one, on both |
| memory per card after the load and after the requests | the same MiB on every card (table below) |
| `test-backend-ops` live, every case, on the head | `MUL_MAT` with the P106 kernels forced, with `GGML_CUDA_MMVQ_PASCAL_MT=2` added and with `GGML_CUDA_MMQ_J64_I64=2` added: 5,458 of 5,458 tests pass in each; `FLASH_ATTN_EXT` unset, with `GGML_CUDA_FATTN_TILE_PASCAL=2` and with every Pascal attention switch: 4,679 of 4,679 in each ([bops.tsv](bops.tsv)) |
| the operators the merge changed (`OPS_RE`: 20 operators, the P106 kernels forced) | head 2,412 of 2,412, base 2,233 of 2,233, head with the three switches 2,412 of 2,412. The 2,288 cases both builds name give the same verdict on both, and every case gives the same verdict on the head with and without the three switches. Only the head names upstream's rewritten gated delta net cases (their names now carry the replay's pending tokens) and its new ones: 156 of the fused norm and RoPE with M-RoPE modes, 3 of `FILL`, one each of `SOFT_MAX` and `SOFTPLUS` ([ops_cases.tsv](ops_cases.tsv)) |
| `MUL_MAT_ID` and `MUL_MAT` replayed from the kept record (C1023's, the P106 kernels forced) | 2,389 and 5,458 cases on each build, every case the same verdict and error on both; with `--cpu-check 25` the processor's references are the record's (84 and 216 checked, all identical); the head names no case the record lacks ([replays.tsv](replays.tsv), [replay_cases.tsv](replay_cases.tsv)) |

Memory per card (MiB, cards 0 / 1 / 2), the same on head and base:

| Model | After the load | After the two requests |
|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (with its MTP draft) | 13,917 / 13,191 / 16,009 | 13,995 / 13,265 / 16,095 |
| Flint Pascal Q8 GGUF | 15,953 / 17,585 / 13,851 | 16,005 / 17,637 / 13,907 |
| Ornith-1.5-122B Pascal N2P GGUF | 15,247 / 16,463 / 15,425 | 15,333 / 16,549 / 15,499 |

With the three switches on (upstream's behaviour, the head's load with `LLAMA_RS_REPLAY=1 GGML_CUDA_NORM_ROPE_MROPE=1
GGML_CUDA_NORM_ROPE_BS=1`):

- **Flash-Next** stopped at an assertion in its conv-window step during the load on `0a6a04e22`. `caf27fdae` fixes it:
  on the final build it loads with the switches, and its replies differ from the final build's own from token 124
  (coding) and token 56 (prose), with coherent text ([the final build](#the-final-build)).
- **Flint** replies differ from the head's from token 11 (coding) and token 0 (prose), with the same memory on every
  card and coherent text. Only `GGML_CUDA_NORM_ROPE_BS` reaches Flint: its block sizes add a row's sum of squares in
  another order.
- **Ornith** replies differ from token 29 (coding) and token 50 (prose), with coherent text; after the requests card 2
  holds 15,497 MiB against 15,499, the other cards the same.

Each switch is measured on its own on a P106 (Q1852 to Q1854).

## The random models' logits

The rig builds' `test-llama-archs -b CPU -s 1` logits differ from the base's in 8 of 152 files: `mamba`, `mamba2`,
`jamba`, `falcon-h1`, `nemotron_h`, `nemotron_h_moe`, `metis16` and `granitehybrid`. The CPU trees differ in the same 8
plus `inkling2` and `inkling2_loop`. A trace build, the head with upstream's #32655 (the CPU softplus through `log1pf`)
and its three Inkling 2 graph changes (#32948, #32855, #32940) reverted, gives the base's logits on all 152 files. The
softplus feeds the CPU selective scan of the 8 Mamba-family models, and the Inkling 2 changes reach only `inkling2`. None
of them is a release file's architecture, and the CUDA softplus is unchanged.

## The final build

The rigs get the candidate's tip `caf27fdae`, built `TESTS=1` with upstream's web page pinned to b12722
(`HF_UI_VERSION=b12722`; upstream keeps no archive of the page for b12723, so the first build had taken its newest):
`llama.cpp-20261125-caf27fdae-pascal-p106-v5-cand-b12723.tar.gz`, 1,630,376,772 bytes, md5
`779a6b2473045c77a643a9871ffbc96f`. It was checked against the first build `0a6a04e22` on the same build machine, the
card steps on two of its three cards:

| Check | Final `caf27fdae` against the first build `0a6a04e22` |
|---|---|
| `TESTS=1` rig build | 32 warnings, the first build's; the same 24 programs |
| programs, byte by byte and function by function ([bin_compare.tsv](bin_compare.tsv), [run/bin_compare.sh](run/bin_compare.sh)) | every md5 differs, since each program carries its commit id and an ELF build id. The 5 programs without libllama (`test-backend-ops`, `llama-gguf`, `test-cuda-keepalive`, `test-hop-mailbox`, `test-top-k-sort`) differ in 29 bytes: those two ids. The 19 that link libllama (statically) differ in the size of one function, `llama_model_qwen4exp::graph::build_layer_attn_linear` (from 2,888 to 2,913 bytes), the commit's change; the code after it moves, so millions of bytes differ at the same file size (`test-hop-cast`: one page larger) |
| `test-llama-archs -b CPU -s 1` | 157 of 157 tests pass; all 152 logits files the same bits as the first build's |
| machine code for the cards (`cuobjdump -sass`), kernel by kernel | `test-backend-ops`: 10,099 kernels, every one the same bytes; `llama-server`: 10,099 kernels, every one the same bytes |
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF, its v5 preset with the MTP draft on two cards (`-ts 1,1`), every switch unset, two greedy requests | the same token ids on both builds; the MTP draft 236 drafted and 176 accepted on the coding request, 273 and 163 on the prose one, on both; the same MiB on both cards |
| the same on the final build with the three switches (`UP_ENV`) | it loads (on `0a6a04e22` it had asserted); coherent replies that differ from the final build's own from token 124 (coding) and token 56 (prose); 239 drafted and 175 accepted, 283 and 160 |

Memory per card (MiB, cards 0 / 1):

| Build | After the load | After the two requests |
|---|---|---|
| final and first build, every switch unset (the same on both) | 20,181 / 22,371 | 20,287 / 22,485 |
| final with the three switches | 20,007 / 22,217 | 20,121 / 22,337 |

With the switches Flash-Next holds 174 and 154 MiB less after the load; of the three, only the replay changes what the
recurrent memory holds (one state per sequence and the pending tokens' inputs, where the snapshots kept
`n_rs_seq + 1` states). On two cards the layer split, and so the card boundaries the residual crosses in Q8_0
(`LLAMA_HOP_CAST=q8_0` in the preset), differ from the three-card loads above, so the prose reply differs from the
three-card one; each comparison pairs two builds on the same cards.

## Files

| File | What it holds |
|---|---|
| [checks.tsv](checks.tsv) | every check with its verdict: the head against the base, and the final build against the first |
| [conflicts.tsv](conflicts.tsv) | every conflicted file: upstream's change, the fork's, the resolution |
| [ctest.tsv](ctest.tsv) | ctest test by test on both CPU trees |
| [hand.tsv](hand.tsv) | the hand-run tests' logs compared line by line |
| [pipe.tsv](pipe.tsv) | the CPU pipe step's verdicts by architecture and arm, both trees |
| [gpu_pipe.tsv](gpu_pipe.tsv) | the card pipe step's verdicts and plain decoding's hashes, both builds |
| [replies.tsv](replies.tsv) | the release files' greedy replies: tokens, draft counts, memory per card, the `llama-server` md5 |
| [bops.tsv](bops.tsv) | the card suite's arms |
| [ops_cases.tsv](ops_cases.tsv) | the operators the merge changed: cases and verdicts by operator, head against base and with the three switches |
| [replays.tsv](replays.tsv), [replay_cases.tsv](replay_cases.tsv) | the kept record's replays and the processor check; each case head against base |
| [logits.tsv](logits.tsv) | the random models' logits files that differ, and the trace |
| [sass_compare.tsv](sass_compare.tsv), [sass_pairs.tsv](sass_pairs.tsv) | the machine code by kernel family, and the renamed kernels instance by instance |
| [bin_compare.tsv](bin_compare.tsv) | the final build's programs against the first build's: bytes and functions |
| [binaries.tsv](binaries.tsv) | the tarballs' and programs' md5s: base, head, final |
| [watch.md](watch.md) | upstream's changes that reach a P106, and what to watch on the rig |
| [run/run.sh](run/run.sh) | the driver: build, CPU steps and GPU steps |
| [run/tables.py](run/tables.py) | the tables from the driver's work folder (checks, ctest, hand-run logs, pipe steps, replies, card suite, replays, logits, machine code, binaries) |
| [run/sass_compare.sh](run/sass_compare.sh), [run/sass_compare.py](run/sass_compare.py) | two builds' machine code compared kernel by kernel |
| [run/sass_bodies.py](run/sass_bodies.py), [run/sass_pair.py](run/sass_pair.py), [run/sass_pairs.sh](run/sass_pairs.sh) | renamed kernels compared by their instructions |
| [run/bin_compare.sh](run/bin_compare.sh) | two builds' programs compared byte by byte and by the size of each function |

## Commands

The driver ran on the build machine, detached, with the base's tarball and the release files on its disks:

```bash
W=<work dir> CLONE=<llama.cpp clone holding 2d839da1d> REPO=<the scripts, as run.sh's usage lists> \
HEAD_SHA=0a6a04e22784a72f9b39cf8b2933f11a2fcb2dd0 \
BASE_TGZ=<pascal-p106-v5 2d839da1d's TESTS=1 tarball> BASE_MD5=7d774f05c29460a6993b19d38796047c \
BASE_RIG_LOG=<its build log> RIG_CCACHE=<compiler cache> CPU_CCACHE=<compiler cache> JIT_SEED=<PTX JIT cache> \
QUANT=<llama-quantize> REC=<C1023's kept record> PREFLIGHT=<cuda_preflight.sh> \
FN=<Swift Qwen3.8-Flash-Next Pascal Q2 GGUF, first shard> DRAFT=<its MTP draft> \
FLINT_Q8=<Flint Pascal Q8 GGUF> ORNITH=<Ornith-1.5-122B Pascal N2P GGUF> \
MOUNTS='<the folders above>' GPUS=0,1,2 LOCK=<build lock> JOBS=8 bash run/run.sh
```

The `sass` step ran by hand, as `MOUNTS="$W" W="$W/sass" BASE=<base bin> HEAD=<head bin> bash run/sass_compare.sh` in the order
`run.sh` now uses (the first copy set `W` before `MOUNTS` and mounted the subfolder twice); the renamed kernels with
`run/sass_pairs.sh`. The trace build is the head with `git revert` of #32655, #32948, #32855 and #32940, run through the
same `cpu` step.

The rest of the card suite: `STEPS=none GPU_STEPS=bops GPUS=<two cards> RUN_TAG=bops` with the same settings (with two
cards the first takes the third card's arms after its own). The final build ran in its own work folder in two parts,
both with `REF` unchanged, `CLONE=<the first run's clone> HEAD_SHA=caf27fdaed35a10307420d5b823bef518a97f6b4
BASE_SHA=0a6a04e22784a72f9b39cf8b2933f11a2fcb2dd0 BASE_TGZ=<the first build's tarball>
BASE_MD5=1e6424f42700d187b9f9642d36388c6a BASE_RIG_LOG=<its build log> RIG_CCACHE=<the first run's compiler cache>
HF_UI_VERSION=b12722 GPUS=<two cards> RUN_TAG=final` and the other settings as above: the processor steps
(`STEPS="src rig base sass archs" GPU_STEPS=none`) beside the card suite, then the load
(`STEPS=none GPU_STEPS=load MODELS_RUN=flashnext`). `RUN_TAG` labels each run's containers, so that each run stops only
its own at its end. Its `sass` step ran by hand, as
`MOUNTS="$W" BASE=<the first build's bin> HEAD=<the final build's bin> W="$W/sass" bash run/sass_compare.sh`, and again
with `PROG=llama-server` and `W="$W/sass-server"`: the copy that ran set `W` before `BASE` and `HEAD` in the same prefix
and found no build, and `run.sh` now sets `W` last. The programs:
`A=<the first build's bin> B=<the final build's bin> OUT=bin_compare.tsv bash run/bin_compare.sh`. The tables:
`python3 -I run/tables.py <a copy of the work folder, with the final build's in final/> <this folder>`.

Provenance and licence: llama.cpp and Miner Llama under MIT; the release files as their model pages state.
