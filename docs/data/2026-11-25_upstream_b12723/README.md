# v5 on upstream b12723: the merge and its checks on the build machine (C1118)

Registry: [Q1851](../../../quants/experiments.md). Date: 2026-11-25.

**So far, v5 on upstream b12723 checks out on the build machine.** With every switch at its default, the three
release files give the base's greedy replies token for token and hold the same memory on every card, and every test
that ran on both builds gives the same verdict. Three upstream changes would have changed how a release file runs, so the
candidate keeps the measured behaviour and puts upstream's behind a switch each, to be measured on a P106. The card suite
stopped partway, when the build machine was needed for other work, and the final build is not made yet: both are under
[Still to run](#still-to-run). Before `pascal-p106-v5` moves to the candidate, a P106 rig checks it, and one
coding-agent run each on Ornith and Flash-Next shows no loss.

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
a machine-code step. The results so far, the head `0a6a04e22` against the base:

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
| `test-backend-ops` live, every case | `FLASH_ATTN_EXT` unset, `FLASH_ATTN_EXT` with `GGML_CUDA_FATTN_TILE_PASCAL=2` and `MUL_MAT` with the P106 kernels forced: every case passes on the head |

Memory per card (MiB, cards 0 / 1 / 2), the same on head and base:

| Model | After the load | After the two requests |
|---|---|---|
| Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (with its MTP draft) | 13,917 / 13,191 / 16,009 | 13,995 / 13,265 / 16,095 |
| Flint Pascal Q8 GGUF | 15,953 / 17,585 / 13,851 | 16,005 / 17,637 / 13,907 |
| Ornith-1.5-122B Pascal N2P GGUF | 15,247 / 16,463 / 15,425 | 15,333 / 16,549 / 15,499 |

With the three switches on (upstream's behaviour, the head's load with `LLAMA_RS_REPLAY=1 GGML_CUDA_NORM_ROPE_MROPE=1
GGML_CUDA_NORM_ROPE_BS=1`):

- **Flash-Next** stopped at an assertion in its conv-window step during the load. `caf27fdae` fixes it; its load with
  the switches is part of the final build's check.
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

## Still to run

- **The card suite's other arms**, on `0a6a04e22`: `MUL_MAT` with `GGML_CUDA_MMVQ_PASCAL_MT=2` and with
  `GGML_CUDA_MMQ_J64_I64=2`; `MUL_MAT_ID` and `MUL_MAT` replayed from the kept record on both builds, then with the
  processor check, and any new cases recorded; `FLASH_ATTN_EXT` with every Pascal attention switch; the operators the
  merge changed (`OPS_RE`) on the head, the base and the head with the three switches.
- **The final build**: `caf27fdae` with upstream's web page pinned to b12722 (`HF_UI_VERSION=b12722`). Upstream has no
  archive of the page for b12723, so the first build took its newest. Checked against the first build: every program
  but `llama-server` the same bytes, the same logits and machine code, Flash-Next's replies, and Flash-Next's load with
  the three switches.
- **The tables** from the work folder (`run/tables.py`).

## Files

| File | What it holds |
|---|---|
| [conflicts.tsv](conflicts.tsv) | every conflicted file: upstream's change, the fork's, the resolution |
| [watch.md](watch.md) | upstream's changes that reach a P106, and what to watch on the rig |
| [run/run.sh](run/run.sh) | the driver: build, CPU steps and GPU steps |
| [run/tables.py](run/tables.py) | the tables from the driver's work folder (checks, ctest, hand-run logs, pipe steps, replies, card suite, replays, logits, machine code, binaries) |
| [run/sass_compare.sh](run/sass_compare.sh), [run/sass_compare.py](run/sass_compare.py) | two builds' machine code compared kernel by kernel |
| [run/sass_bodies.py](run/sass_bodies.py), [run/sass_pair.py](run/sass_pair.py), [run/sass_pairs.sh](run/sass_pairs.sh) | renamed kernels compared by their instructions |

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

The rest of the card suite: `STEPS=none GPU_STEPS=bops` with the same settings. The final build: `REF` unchanged,
`HEAD_SHA=<caf27fdae's full id> BASE_SHA=0a6a04e22784a72f9b39cf8b2933f11a2fcb2dd0 BASE_TGZ=<the first build's tarball>
HF_UI_VERSION=b12722 STEPS="src rig base sass archs" GPU_STEPS=load MODELS_RUN=flashnext`.

Provenance and licence: llama.cpp and Miner Llama under MIT; the release files as their model pages state.
