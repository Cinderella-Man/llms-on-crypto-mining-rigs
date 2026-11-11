# Q1246: prompt micro-batches through the sparse attention, by a rule of their own (2026-11-11)

Qwen3.8-Flash-Next's 12 QSA layers attend to about 2,051 kept cells per query token, and Miner Llama's sparse
attention below Turing ([Q230](../2026-11-04_qsa_sparse_attn/README.md)) reads only those. Its switch took a call of
at most 8 query tokens, and in mode 1 only once the cache held 1.5 × 2,051 cells per query token. That rule fits decode
and verify passes and keeps every prompt micro-batch on the dense kernels: 393,792 cells for 128 query tokens. The deep
lane's sixth run ([Q1214](../2026-11-10_deep_lane/README.md#the-sixth-run-prompt-micro-batches-through-the-sparse-kernel-q1214))
sent micro-batches of 128 through the kernel with the switch at 2 and `_NQ=128`: prompts read 46.7 % faster over a
load's three chunks, and 131.3 and 131.9 tok/s on 32,572 cached tokens against the preset's 70.2 and 70.0. The same
setting slowed the decode step at depth 0 to 282.35 and 282.53 ms against 113.12 and 113.14. This folder holds the rule
that gives prompt micro-batches their own threshold, its checks without a GPU, the check set for a P106 and the rig
commands that time it.

**Status: written, built and checked on a CPU; the check set for a P106 is recorded; through the compute_61 PTX on a
newer card its runs `qp-stock` and `qp-m2` pass 72 of 72 cases, the sparse kernel at 129, 256 and 512 query tokens
included ([FN63](../2026-11-11_fn63_sparse_256_512/README.md)); not yet run on a P106.** Miner
Llama branch `pascal-p106-v4-b12246-c705` `baac08803` (one commit on `pascal-p106-v4` `84f7bd3d2`), behind
`GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT` (off by default; then the switch takes exactly the calls it took on
`pascal-p106-v4`, and the GPU code of every kernel is the same).

## The rule

| Call | Mode 1 (GPUs without Turing tensor cores) | Mode 2 (any NVIDIA GPU, for tests) |
|---|---|---|
| Decode or verify: at most `_NQ` (default 8) query tokens per sequence | from `_MIN_RATIO` (default 1.5) × `n_kv_max` cells per query token, as before | at any cache size, as before |
| Prompt micro-batch, with `_PROMPT=1`: more than `_NQ` and at most `_PROMPT_NQ` (default 128) query tokens | from `_PROMPT_MIN_RATIO` (default 1.0) × `n_kv_max` cells, whatever the micro-batch's size | at any cache size |
| Any other call of more than `_NQ` query tokens | the dense kernels | the dense kernels |

The kernel's own limits apply as before: at most 1,024 columns of query token × KV head (512 query tokens at
Flash-Next's 2 KV heads), F16 or Q8_0 K and V, head size 128 or 256. The rule is `fasp_rule_takes` in
`ggml/src/ggml-cuda/fattn-sparse-pascal-impl.cuh`, host code free of ggml, so the public check builds it on a CPU.

| Variable | Values |
|---|---|
| `GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT` | 0 off (default); 1 prompt micro-batches by the rule above |
| `_PROMPT_MIN_RATIO` | cells per `n_kv_max` from which mode 1 takes a prompt micro-batch, default 1.0 (2,051 cells at Flash-Next's selection) |
| `_PROMPT_NQ` | most query tokens per sequence of a prompt micro-batch, default 128 (the size the sixth run read); 512 lets in what the kernel's columns hold |

**Why a prompt's threshold does not grow with its query tokens.** A decode or verify call is a read of the cache: the
dense kernels read each cell once for all the call's query tokens, the kernel one list of 2,051 cells per query token,
so k query tokens break even at about k × 2,051 cells and the threshold is `_MIN_RATIO` × `n_kv_max` × k. A prompt
micro-batch is arithmetic: the dense kernels score every query token against every cell, n_q × n_kv scores, and the
kernel n_q × 2,051, so the kernel does less work as soon as the cache holds more than the kept cells, whatever n_q.
Below `n_kv_max` cells the mask keeps every cell and the kernel reads as many as the dense kernels, hence the default
of 1.0. The sixth run is in line with it: on the chunk that grew the cache from 0 to 16,320 cells, micro-batches of 128
through the kernel read 128.2 and 128.3 tok/s against 118.2 and 118.6 on the dense kernels at the same micro-batch
(+8.4 %), and the 64-token tail on an empty cache read 32.6 and 32.1 against 32.4 and 32.7. Where prompt calls cross
over on a P106 is the rig row's question (below).

**The decode steps.** With the default `_NQ` of 8 mode 2 costs nothing in decode at depth 0 (the eighth run: 123.71 and
124.24 ms a step against mode 1's 123.56 and 123.62), so the sixth run's 282.35 and 282.53 ms come with `_NQ=128` (or
with mode 2 and `-ub 128` together, which no run had without `_NQ=128`). If they are calls of more than 8 query tokens
on a cache of fewer than 2,051 cells, the rule keeps every one of them on the dense kernels; the server does not log
which calls they were. The rig row times the step at depth 0 and 8,192 with the rule on.

**Where the rule switches** at Flash-Next's 2,051 kept cells, from `fasp_check --rule` ([rule.txt](rule.txt)): the
smallest cache, in cells, from which mode 1 sends a call of n_q query tokens to the kernel.

| Setting | 1 | 2 | 4 | 8 | 9 | 16 | 64 | 128 | 129 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `=1` (`pascal-p106-v4`, and this branch with `_PROMPT` off) | 3,077 | 6,153 | 12,306 | 24,612 | never | never | never | never | never | never | never |
| `=1 _PROMPT=1` | 3,077 | 6,153 | 12,306 | 24,612 | 2,051 | 2,051 | 2,051 | 2,051 | never | never | never |
| `=1 _MIN_RATIO=1.0 _PROMPT=1` (the verify ratio of [Q1212](../2026-11-10_deep_lane/README.md#the-eighth-run-the-sparse-kernels-threshold-for-verify-passes-q1212)) | 2,051 | 4,102 | 8,204 | 16,408 | 2,051 | 2,051 | 2,051 | 2,051 | never | never | never |
| `=1 _PROMPT=1 _PROMPT_MIN_RATIO=2.5` | 3,077 | 6,153 | 12,306 | 24,612 | 5,128 | 5,128 | 5,128 | 5,128 | never | never | never |
| `=1 _PROMPT=1 _PROMPT_NQ=512` | 3,077 | 6,153 | 12,306 | 24,612 | 2,051 | 2,051 | 2,051 | 2,051 | 2,051 | 2,051 | 2,051 |

With `-ub 128` and `_PROMPT=1`, every prompt micro-batch of Flash-Next's main model and of its MTP draft
(`--spec-draft-n-ubatch 128`) goes to the kernel once the cache it attends to holds 2,051 cells; the preset's
`-ub 512` micro-batches stay on the dense kernels unless `_PROMPT_NQ=512`.

## Checks without a GPU

On a build machine's CPU, no GPU, with [run/night.sh](run/night.sh) (each step under a lock shared with the other
builds of the night, at `--cpu-shares=256`):

| Check | Result |
|---|---|
| The rig build (`TESTS=1`, sm_61 + compute_61 PTX) | builds; tarball 647,364,343 B, md5 `6a5239a02ffa53d327b955f0ea240db9`, 0 AVX instructions, GLIBC_2.38 at most |
| `fasp_check --rule`: the rule over 35,964,000 settings × calls (mode 1 and 2; `_NQ` 1, 4, 8, 16, 128; `_MIN_RATIO` 0.25, 1.0, 1.5; `_PROMPT` 0 and 1; `_PROMPT_NQ` 16, 128, 512; `_PROMPT_MIN_RATIO` 0.5, 1.0, 2.5; `n_kv_max` 31, 333, 2,051; 1 to 600 query tokens; 37 cache sizes around every threshold) | **pass:** with `_PROMPT` off 0 calls differ from the rule of `pascal-p106-v4`; with it on 0 decode or verify calls differ from that rule and 0 prompt calls from the prompt rule's definition (5,240,274 prompt calls taken) |
| The same with three mutations of the rule (prompt calls by cells per query token, prompt calls taken with `_PROMPT` off, a call of exactly `_NQ` tokens by the prompt rule) | each fails |
| `fasp_check`: the host model of the kernel against FP64 and ggml's CPU `FLASH_ATTN_EXT`, 13 shapes × F16/Q8_0, now with prompt micro-batches of 16 and 128 query tokens at Flash-Next's shape and 512 query tokens over 2 KV heads (the kernel's 1,024 columns) | **26 / 26 pass**: the largest error over the largest output against FP64 at most 9.52·10⁻⁷ (one split, F16); the three new shapes 1.19, 1.63 and 1.72·10⁻⁷ (F16) and 1.23, 1.51 and 1.73·10⁻⁷ (Q8_0) ([check.txt](check.txt)) |
| The same with the kernel's three fault codes | each fails all 26 |
| The same with the four source mutations of the kernel | 13, 14, 14 and 26 cases fail |
| The GPU code of `llama-server` and `test-backend-ops` against `pascal-p106-v4`'s build (`-ladder6`, md5 `327332d51af131b6babfdba27d933129`): `cuobjdump -sass` and `-ptx` | **the same, byte for byte**, in both binaries: the sm_61 code (11,242,279 lines, md5 `b948516ed6106a0c1abd71882c2ccd21`) and the compute_61 PTX (9,615,682 lines, md5 `eb01ac1f02984d7c45fc4815be177875`) ([code.tsv](code.tsv)) |

The branch leaves the kernel as it is: a prompt micro-batch's arithmetic is the decode kernel's with more query
rows, one column per query token × KV head, as the 512-token case's 1,024 columns check.

## The check set for a P106

`scripts/record_checks.sh`'s suite `qsa-prompt`: every `FLASH_ATTN_EXT` case with the `n_kv_max` hint, recorded with the
branch's own `test-backend-ops` (md5 `77bf0ba5e598d58226805ff21799de6f`) in a container without a GPU, so a P106 replays
them against the stored CPU references ([scripts/replay_checks.sh](../../../scripts/replay_checks.sh)).

| | |
|---|---|
| Cases | 85 ([set.tsv](set.tsv)): the 24 new prompt cases (12 shapes × F16/Q8_0) and the 61 sparse-hinted cases the tree had (decode and verify at Flash-Next's shape on 4,096 to 32,768 cells, the other head shapes, the gather cases) |
| Size | 2,124,927,866 B of cases; the tar 2,125,137,920 B, md5 `9a498be37790ab86555e0050084af11d`, equal to the folder (`tar -d`) |
| Recorded | 14 s, 8 workers, at nice 19 |
| Runs on a P106 ([runs.tsv](runs.tsv)) | `qp-stock` (`=0`), `qp-sparse1` (`=1`, the rule off), `qp-prompt` (`=1 _PROMPT=1`), `qp-r100` (and `_MIN_RATIO=1.0`), `qp-pratio` (`_PROMPT_MIN_RATIO=2.5`), `qp-pnq512` (`_PROMPT_NQ=512`), `qp-m2` (`=2 _PROMPT=1 _PROMPT_NQ=512`), `qp-m2-off` (`=2`), `qp-fault1` (passes: within the test's limit), `qp-fault2` and `qp-fault3` (must fail cases) |
| Through the compute_61 PTX ([runs-ptx.tsv](runs-ptx.tsv)) | `qp-stock` and the five mode-2 runs |

Which kernel each prompt case takes in each run, from the rule (`S` the sparse kernel, `d` the dense ones;
[rule.txt](rule.txt)); `qp-stock` takes `d` everywhere:

| Query tokens, cells, `n_kv_max`, sequences | `qp-sparse1` | `qp-prompt` | `qp-r100` | `qp-pratio` | `qp-pnq512` | `qp-m2` | `qp-m2-off` |
|---|---|---|---|---|---|---|---|
| 9, 8,192, 2,051, 1 | d | S | S | S | S | S | d |
| 16, 8,192, 2,051, 1 | d | S | S | S | S | S | d |
| 64, 8,192, 2,051, 1 | d | S | S | S | S | S | d |
| 128, 8,192, 2,051, 1 | d | S | S | S | S | S | d |
| 9, 3,072, 2,051, 1 | d | S | S | d | S | S | d |
| 128, 3,072, 2,051, 1 | d | S | S | d | S | S | d |
| 128, 32,768, 2,051, 1 | d | S | S | S | S | S | d |
| 64, 4,096, 2,051, 2 | d | S | S | d | S | S | d |
| 32, 4,096, 1,027, 1 (head 128, sinks) | d | S | S | S | S | S | d |
| 129, 4,096, 2,051, 1 | d | d | d | d | S | S | d |
| 256, 4,096, 2,051, 1 | d | d | d | d | S | S | d |
| 512, 4,096, 2,051, 1 | d | d | d | d | S | S | d |

## On a P106

Two parts on one rig, nothing else running on it (registry row Q1272).

**1. The native check, by replay** (est. 15 min): the set on card 0 with the branch's own `test-backend-ops`. It passes
when every console line ends `ok` (`qp-fault2` and `qp-fault3` must fail cases, every other run passes every case),
and when each prompt case's error in a run equals its error in `qp-stock` where the table above says `d` and differs
where it says `S` (the per-case errors are in `OUT/<run>.tsv`). The decode and verify cases follow the rule of
`pascal-p106-v4` in every run.

**2. The timing on the deep lane** (est. 2 h 45 min): [bench/deep_lane](../../../bench/deep_lane/README.md) on the
branch's build, the release preset's line, R52's three arms (the sixth run's) and two more, two loads an arm after a
warm-up load, 256 tokens a decode request at 0, 8,192, 16,384, 32,768 and 49,152 tokens:

| Arm | Switch | `-ub` | What it shows |
|---|---|---:|---|
| `ub512` | `=1` | 512 | the preset (control) |
| `ub128` | `=1` | 128 | the micro-batch alone |
| `sp128` | `=2 _NQ=128` | 128 | the sixth run's line: prompts through the kernel at any size, the decode loss at depth 0 |
| `r128` | `=1 _MIN_RATIO=1.0` | 128 | the decode control for `pr128`: verify passes at Q1212's ratio |
| `pr128` | `=1 _MIN_RATIO=1.0 _PROMPT=1` | 128 | the rule |

The rule does what it is for when `pr128` reads prompts at `sp128`'s speed from 16,384 tokens on and its repeated step
at depth 0 and 8,192 is `r128`'s. Prompts are read with `run/prompt_depth.py` of the deep-lane folder (control
`ub512`), the step with `deep_table.py`.

## Files

| File | What |
|---|---|
| [run/night.sh](run/night.sh) | the build, the CPU check, the GPU code against the base build and the recording, on a build machine, CPU only, each step under a lock |
| [rule.txt](rule.txt) | `fasp_check --rule`: the grid's verdict, the switch points, the kernel each prompt case takes per replay run |
| [check.txt](check.txt) | `bench/fattn_sparse/check.sh`'s verdicts and every run's table |
| [code.tsv](code.tsv) | the GPU code of `llama-server` and `test-backend-ops`, branch against base: md5 and lines of each `cuobjdump` listing |
| [set.tsv](set.tsv) | the check set: one line per recorded case (id, case string, input and reference bytes) |
| [runs.tsv](runs.tsv), [runs-ptx.tsv](runs-ptx.tsv) | the set's runs on a P106 and through the compute_61 PTX on a newer card |
| [bench/fattn_sparse](../../../bench/fattn_sparse/README.md) | the CPU check (`--rule` is new here) |
| [scripts/record_checks.sh](../../../scripts/record_checks.sh), [scripts/replay_checks.sh](../../../scripts/replay_checks.sh) | the recording (suite `qsa-prompt` is new here) and the replay |

Test cases added to `tests/test-backend-ops.cpp`: Flash-Next's QSA call (head 256, 2 × 12 heads, 2,051 cells) at 9,
16, 64 and 128 query tokens on 8,192 cells, 9 and 128 on 3,072, 128 on 32,768, 64 in two sequences on 4,096, head 128
with sinks at 32 tokens, and 129, 256 and 512 query tokens on 4,096 cells, F16 and Q8_0.

## Commands

The night's steps on a build machine with Docker, CPU only ([run/night.sh](run/night.sh); `WORK/src` a clone of
llama.cpp with the fork as remote `fork`, `BASE` the base build's rig tarball, `LOCK` a lock file other jobs share):

```bash
WORK=<work folder> REPO=<this repository> BRANCH=pascal-p106-v4-b12246-c705 LOCK=<lock file> \
  BASE=<work>/llama.cpp-20261110-84f7bd3d2-pascal-p106-v4-b12246-ladder6.tar.gz DOCKER_ARGS="--label <job>=1" \
  setsid -f nohup bash docs/data/2026-11-11_c705_sparse_prompt_rule/run/night.sh > night.out 2>&1 < /dev/null
```

Each step alone: `STEPS=build`, `STEPS=check`, `STEPS=code`, `STEPS=record` (the record step is
`OUT=<work>/sets SUITE=qsa-prompt BIN=<tree>/bin bash scripts/record_checks.sh`). The CPU check by itself:

```bash
SRC=<llama.cpp tree of the branch> DOCKER_ARGS="--cpu-shares=256" bench/fattn_sparse/check.sh
bench/fattn_sparse/build/fasp_check --rule      # the rule alone, seconds
```

**On the rig.** The build and the set, relayed from the build machine (`CLIENT`) through the operator's machine, with
the md5s compared on arrival:

```bash
T=llama.cpp-20261110-baac08803-pascal-p106-v4-b12246-c705
ssh $CLIENT "cat $WORK/$T.tar.gz" | ssh $RIG "tee >(md5sum > /tmp/$T.md5) | tar xzf - -C /opt/llama-builds; cat /tmp/$T.md5"
ssh $CLIENT "cat $WORK/sets/qsa-prompt.tar" | ssh $RIG "mkdir -p $DIR && tee >(md5sum > $DIR/qsa-prompt.md5) | tar xf - -C $DIR"
ssh $RIG "cat > $DIR/replay_checks.sh" < scripts/replay_checks.sh
```

The replay, detached:

```bash
ssh $RIG "SET=$DIR/qsa-prompt BIN=/opt/llama-builds/$T/bin OUT=$DIR/out DEV=0 CPU_CHECK=200 \
  setsid nohup bash $DIR/replay_checks.sh > $DIR/replay.log 2>&1 < /dev/null &"
```

Through the compute_61 PTX on a newer card (mode 1 leaves such a card on its own kernels, so the list holds the mode-2
runs): `RUNS_FILE=$DIR/qsa-prompt/runs-ptx.tsv` with the same command in a container with one card.

The deep lane, with the deep lane's `wait` and `collect` commands
([2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md#commands); `DRY=1` first: its five server lines differ only
in `-ub` and the switch's variables):

```bash
RIG=rig BIN=/opt/llama-builds/llama.cpp-20261110-baac08803-pascal-p106-v4-b12246-c705/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
DEPTHS="0 8192 16384 32768 49152" N_PREDICT=256 ARMS="ub512 ub128 sp128 r128 pr128" \
ARM_ENV_ub512=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
ARM_ENV_ub128=GGML_CUDA_FATTN_SPARSE_PASCAL=1 ARM_UB_ub128=128 \
ARM_ENV_sp128="GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NQ=128" ARM_UB_sp128=128 \
ARM_ENV_r128="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0" ARM_UB_r128=128 \
ARM_ENV_pr128="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1" ARM_UB_pr128=128 \
LOADS="warm:ub512 ub512_1:ub512 ub128_1:ub128 sp128_1:sp128 r128_1:r128 pr128_1:pr128 ub512_2:ub512 ub128_2:ub128 sp128_2:sp128 r128_2:r128 pr128_2:pr128" \
bash bench/deep_lane/remote.sh start
```

Its tables:

```bash
python3 -I bench/deep_lane/deep_table.py <results folder> --out table.md --tsv classes.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <results folder> --control ub512 --out prompt_depth.md --tsv prompt_depth
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py <results folder> > host_memory.tsv
```

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), branch `pascal-p106-v4-b12246-c705`. The check's inputs are random tensors
drawn by test-backend-ops and `fasp_check`; no model or text is in the set. The rig row's model: Swift-1.5 GSQ-RCO
Qwen3.8-Flash-Next Q2_0 ([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open
License 1.0 and qwen-community-1.0), as retyped by [Q218](../2026-11-03_retype/README.md), and the deep lane's recorded
agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)).
