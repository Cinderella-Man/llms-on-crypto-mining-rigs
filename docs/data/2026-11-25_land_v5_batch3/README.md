# Eight branches landed in pascal-p106-v5 (2026-11-25)

Miner Llama's v5 engine collects in one fork branch, `pascal-p106-v5`, and every finished change lands there behind its
own switch, off by default. On 2026-11-25 the eight branches finished on 2026-11-24 were merged into it at once, on top
of upstream llama.cpp b12674 (merged the evening before): two log fixes, a fixed test seed, a profiling column, a test
fix, a test-fault factor, the old 64-row prompt tile behind a switch, and a skipped NVML probe.

This folder records the merge and the checks of the combined build against the branch before it (`eee615cdb`).
Correctness only: nothing here is timed. Each change's speed is measured on a P106 rig, on this combined build, with
its switch set against the same build with it unset (the rows are in each branch's registry entry).

**Result: all eight landed.** Every merge was clean. With no switch set, the combined build computes what the base
computes everywhere the checks look: the same logits on every random model, the same tests passing, the same pipelined
decoding on the processor and on the cards, the same card-suite verdicts and errors case by case, and the same greedy
replies and memory per card on Flint Pascal Q8 GGUF (`p065`) and Ornith-1.5-122B Pascal N2P GGUF (`valid1`). The card
code is the base's but for the two branches that change it on purpose (below).

One check failed once and never again: in the head's first `MUL_MAT` replay with `--cpu-check 25`, one BF16
matrix-vector case gave an error of 0.4995. It passed in the head's five later runs of that replay, in the base's five,
and 700 times on its own on the two builds. The kernels it runs are byte for byte the same in both builds, so it
is not a difference between them; it is followed up as its own registry row (Q1844).

`pascal-p106-v5` was then fast-forwarded to the landing merge `2d839da1d`.

## What was merged

`pascal-p106-v5` at `eee615cdb`, plus one `--no-ff` merge per branch, on the branch `pascal-p106-v5-land-batch3`:
[merges.tsv](merges.tsv) (branch, its head, the merge commit, code row, registry row, the switch it brings and its
default). `pascal-p106-v5-log-flush-all` carries `pascal-p106-v5-evalcb-flush`, so eight branches came in seven merges.
No merge had a conflict. Two files are changed by two branches each, in other functions:
`ggml/src/ggml-cuda/ggml-cuda.cu` (the profile's anchors in the host profile's functions, the NVML probe in
`ggml_cuda_init`) and `tests/test-backend-ops.cpp` (the `TOPK_MOE_TAIL` check in `test_case` and `test_topk_moe_tail`,
the 297 new 64-column cases in the case list). In each, the merged change is the two branches' lines exactly.

| Branch | What it brings | Default |
|---|---|---|
| `-log-flush-all` (with `-evalcb-flush`) | the common logger flushed when a program returns or exits, in `llama-eval-callback`, `test-thread-safety` and 38 other programs | no switch: the programs print only what they already logged |
| `-ctest-seed` | ctest's random models written with seed 1 | no switch: test only |
| `-profile-anchors` | `GGML_CUDA_PROFILE_HOST_ANCHORS`: each evaluation's start and end on the card in the host profile | read only when the host profile (`GGML_CUDA_PROFILE_HOST`) is on; then on with `GGML_CUDA_PROFILE_HOST_NOWAIT=1`, itself off by default |
| `-topk-tail-tie` | `test-backend-ops`' `TOPK_MOE_TAIL` check no longer fails on router ties | no switch: test only |
| `-fault-scale` | `GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE`, the factor of the Pascal attention kernels' test fault | 1.05, read only with the test fault `GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1`, off by default |
| `-mmq-j64-tile` | `GGML_CUDA_MMQ_J64_I64`, the 64-row tile back for 64-column prompt products | 0, off |
| `-nvml-probe` | `GGML_CUDA_NVML_PROBE`: upstream's NVML probe at CUDA start only where a card reads its answer | unset: skipped on cards without fast BF16, so never on a P106 rig; on cards with fast BF16 (these RTX 3090s) it runs, as upstream's does |

Each switch has its row in [docs/switches.md](../../switches.md).

## The checks

Run on a build machine with three RTX 3090s, 13:00–13:54: one detached driver ([run/run.sh](run/run.sh)), the processor
steps beside the card steps. Then, 13:44–14:12, repeats of the one failure ([run/repeat_case.sh](run/repeat_case.sh) and
C1023's replay driver) and a comparison of the two builds' machine code ([run/sass_compare.sh](run/sass_compare.sh)).
Every check in [checks.tsv](checks.tsv):

| Check (head `2d839da1d` against base `eee615cdb`) | Result |
|---|---|
| Rig build (`TESTS=1`, every check tool, web page b12674), 13:00–13:04 | `llama.cpp-20261125-2d839da1d-pascal-p106-v5-land-batch3.tar.gz`, 1,626,909,591 bytes, **md5 `7d774f05c29460a6993b19d38796047c`** |
| Warnings | rig build 32 lines on both, CPU build 2 on both; none new |
| Logits, `test-llama-archs -b CPU -s 1`, the rig builds | 152 of 152 files the base's bit for bit; 157 of 157 cases pass on both |
| CPU trees (`cpu_tests.sh`, `-DGGML_RPC=ON`): ident | 152 of 152 files the base's |
| ctest, test by test | 82 of 83 pass on both; `test-kv-cache-scales` fails on both, as before; every test the same |
| Hand-run tests | 9 of 9 pass on both; logs line by line 24 of 24 the same |
| CPU pipe step: 5 architectures × identity, sampled, race, recurrent, streams | 25 of 25 OK on both |
| Card pipe step, one card and split over three, F32 on llama.cpp's kernels, Q8_0 with the fork's forced, the Flash-Next mix | 125 of 125 OK on both; plain decoding's tokens and logits hashes the same in all 25 pairs |
| Flint Pascal Q8 GGUF, v5 preset on three cards (`-ts 1,1,1`), the fork's compute-6.1 paths forced, one greedy reply of 256 tokens | the same 256 token ids; MiB per card after the load 15,953 / 17,585 / 13,851 and after the request 15,997 / 17,627 / 13,899 on both builds |
| Ornith-1.5-122B Pascal N2P GGUF, the same | the same 256 token ids; 15,247 / 16,463 / 15,425 and 15,319 / 16,535 / 15,485 MiB on both builds |
| `test-backend-ops MUL_MAT` live, the fork's matrix-vector kernels forced at a P106's shapes | 5,458 of 5,458 (the base's 5,161 plus the 297 new 64-column cases); the same with `GGML_CUDA_MMVQ_PASCAL_MT=2` and with `GGML_CUDA_MMQ_J64_I64=2` (base, in the b12674 check: 5,161 of 5,161) |
| `MUL_MAT_ID` and `MUL_MAT` replayed from the kept record, both builds | every case with a record passes on both: `MUL_MAT_ID` 2,370 of 2,370, `MUL_MAT` 5,125 of 5,125; the processor's references still the record's (`--cpu-check 25`: 84 and 204 identical) |
| The cases the record lacked, recorded into it on the head (live check of each) | `MUL_MAT_ID` 19 of 19 (upstream b12674's new cases), `MUL_MAT` 333 of 333 (36 of b12674's, 297 of the new tile) |
| Both builds replayed again, every case recorded | head `MUL_MAT_ID` 2,389 of 2,389 and `MUL_MAT` 5,458 of 5,458, base 2,389 and 5,161; **each case's verdict and error the same on both builds** (2,389 and 5,161 cases) |
| `test-backend-ops FLASH_ATTN_EXT` unset, with `GGML_CUDA_FATTN_TILE_PASCAL=2`, and with every Pascal attention switch | 4,679 of 4,679 in each (base, in the b12674 check: the same) |
| `MUL_MAT` replay with `--cpu-check 25`, every run | head 5 of 6 runs pass: the first ran 5,124 of 5,125, the one BF16 case below; base 5 of 5 |
| The failing case alone, 100 or 150 runs a build and way (replayed, live, after its F16 sibling's live check as in the failing run) | 350 of 350 OK on the head, 350 of 350 on the base |
| `test-backend-ops`' machine code, kernel by kernel (`cuobjdump -sass`) | 8,334 of the base's 10,009 kernels the same bytes; 1,664 `mul_mat_q` kernels differ in one line each (a printed source line number, as in C1024's check); 11 Pascal attention kernels differ in the test fault's multiplies (C993's change); 66 new kernels in the head (the 64-row tile's copies); every matrix-vector kernel, `mul_mat_vec_bf16_pascal` among them, the same bytes |

No CUDA error, illegal access or timeout anywhere. The base's live card-suite numbers are those of its own check
([docs/data/2026-11-24_c912_upstream_b12674/](../2026-11-24_c912_upstream_b12674/README.md)), run with the same
`test-backend-ops` binary (md5 `7e6731c4…`).

**The one failure.** `MUL_MAT(type_a=bf16,type_b=f32,m=1057,n=1,k=128,bs=[8,1],nr=[4,1],per=[0,2,1,3])` gave a
normalized error of 0.4995 against the recorded reference (limit 0.0005) in the head's first replay with
`--cpu-check 25`, right after its F16 sibling was checked live on the processor. It ran on the fork's BF16 Pascal
matrix-vector kernel (the forced settings send it there), whose machine code
is the same in both builds, and the comparison code it went through is unchanged too (the `TOPK_MOE_TAIL` branch adds
a per-node hook whose default is the old error function). It did not come back in the head's 11 later `MUL_MAT`
replays (5 with the processor check), in any of the base's 12 (5 with it), or in 700 runs of the case alone (350 a
build), so it is rare and belongs to both builds' code. It is tracked as Q1844.

## The kept record

The landing replayed `MUL_MAT_ID` and `MUL_MAT` from C1023's record
([docs/data/2026-11-24_c1023_suite_record/](../2026-11-24_c1023_suite_record/README.md)) instead of computing every
processor reference again, and added to it the 352 cases upstream b12674 and the new tile brought (`STEPS=record
RESUME=1` keeps every case already there). The record now covers both builds' case lists: [record.tsv](record.tsv)
(cases, files, bytes) and the md5 of every file, [record_manifest_mul_mat_id.md5.gz](record_manifest_mul_mat_id.md5.gz)
and [record_manifest_mul_mat.md5.gz](record_manifest_mul_mat.md5.gz). The run's first replays took 280 s
(`MUL_MAT_ID`) and 195 s (`MUL_MAT`), where the landing before computed `MUL_MAT_ID` live in 38 minutes; later `MUL_MAT`
replays, from the page cache, under a minute each ([cpucheck_runs.tsv](cpucheck_runs.tsv) lists every one).

## Files

| File | What it holds |
|---|---|
| [merges.tsv](merges.tsv) | the seven merges |
| [checks.tsv](checks.tsv) | every check, head against base, with its verdict |
| [ctest.tsv](ctest.tsv) | ctest per test on both CPU builds |
| [pipe.tsv](pipe.tsv) | the CPU pipe step's verdicts per architecture and arm, both trees |
| [gpu_pipe.tsv](gpu_pipe.tsv) | the card pipe step's verdicts per build, arm, layout, architecture and check |
| [replies.tsv](replies.tsv) | the two release files' greedy replies and memory per card, head and base |
| [bops.tsv](bops.tsv) | each live `test-backend-ops` arm: settings, cases passed, failures |
| [replays.tsv](replays.tsv) | each replay and record step of the kept record: cases passed, without a record, the processor check |
| [replay_cases.tsv](replay_cases.tsv) | per share, each case's verdict and error, head against base |
| [cpucheck_runs.tsv](cpucheck_runs.tsv) | every `MUL_MAT` replay of both builds, the one failure among them |
| [repeats.tsv](repeats.tsv) | the failing case alone: runs and verdicts per build and way |
| [sass_compare.tsv](sass_compare.tsv) | `test-backend-ops`' kernels per family: the same bytes, differing, only in one build |
| [record.tsv](record.tsv), `record_manifest_*.md5.gz` | the kept record after this landing |
| [binaries.tsv](binaries.tsv) | the rig tarballs and their programs' md5s |
| [run/run.sh](run/run.sh) | the driver: build, CPU steps and GPU steps |
| [run/tables.py](run/tables.py) | this folder's tables from the driver's work folder |
| [run/repeat_case.sh](run/repeat_case.sh) | one `test-backend-ops` case run again and again per build |
| [run/sass_compare.sh](run/sass_compare.sh), [run/sass_compare.py](run/sass_compare.py) | two builds' machine code compared kernel by kernel |

## Commands

On a build machine with Docker, three NVIDIA cards and this repository's scripts copied into one folder
(`scripts/build_llamacpp_docker.sh`, `scripts/pascal_forced.sh`, `docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh`
and `compare_hand.py`, `docs/data/2026-11-24_c1023_suite_record/run/run.sh` as `suite_record.sh`, `bench/pipespec/` as
`pipespec/`, `release/flint-v5/settings.env` and `release/ornith-v5/settings.env` under `release/`):

```bash
W=<work dir> CLONE=<llama.cpp clone holding eee615cdb> REPO=<that folder> \
BASE_TGZ=<the base's TESTS=1 tarball> BASE_MD5=575ef3c702f69f14d7d3022c1287c38f BASE_RIG_LOG=<its build log> \
FLINT_Q8=<flint-pascal-Q8.gguf> ORNITH=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> \
REC=<C1023's record folder> QUANT=<a llama-quantize, as the base tarball has none> \
PREFLIGHT=<a CUDA check, run as $PREFLIGHT -g 0,1,2> MOUNTS='<folders holding the builds, models and record>' \
GPUS=0,1,2 LOCK=<build lock> JOBS=8 setsid nohup bash run.sh > run.out 2>&1 < /dev/null &
# the failing case alone, 150 times a build, after its F16 sibling's live check (as in the failing run)
W=<out> BUILDS='head=<head bin> base=<base bin>' REC=<record>/mul_mat GPU=0 N=150 MODES=cpucheck \
CASE='^type_a=[^,]+,type_b=f32,m=1057,n=1,k=128,bs=\[8,1\],nr=\[4,1\],per=\[0,2,1,3\],k_v=0,o=1,src_overlap=0,m_v=0,pad=0,offset_a=0$' \
MOUNTS='<folders>' bash repeat_case.sh
# the MUL_MAT replay with the processor check again, one build a call (a plain replay, then --cpu-check 25)
WORK=<out> NEW=<bin> BUILDS='head=<bin>' REC=<record> GPU=1 STEPS=replay SHARES=mul_mat ALLOW_MISSING=1 CPU_CHECK=25 \
MOUNTS='<folders>' bash suite_record.sh
# the two builds' machine code
W=<out> BASE=<base bin> HEAD=<head bin> MOUNTS='<folders>' bash sass_compare.sh
```

The base tarball is the b12674 candidate's, `scripts/build_llamacpp_docker.sh` of `eee615cdb` with `TESTS=1` and
`HF_UI_VERSION=b12674` (1,403,235,928 bytes, md5 `575ef3c702f69f14d7d3022c1287c38f`); it was built before the script
put `llama-quantize` in every `TESTS=1` tarball, hence `QUANT` for the card pipe step. The head's is the same command on
`fork/pascal-p106-v5-land-batch3`. Fork: <https://github.com/Cinderella-Man/miner-llama>, branches `pascal-p106-v5` and
`pascal-p106-v5-land-batch3`.

## Provenance and licence

The random models are written by the build's own `test-llama-archs` with fixed seeds. The release files are this
project's own (their cards on Hugging Face give the licences). The tables are this project's, under the repository's
licence.
