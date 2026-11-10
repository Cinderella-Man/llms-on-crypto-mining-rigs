# The small QSA k-pool inputs in the tables' update (C720, Q1275, 2026-11-11)

What this is: a new switch on the v4 line, `LLAMA_QSA_KPOOL_STAGE=1` on the fork branch `pascal-p106-v4-b12246-c720`
(`27e3d69fa`, one commit on C100's `pascal-p106-v4-b12246-c100` `ce611b9db`), off by default, which works together
with C100's `LLAMA_QSA_KPOOL_DEV=1` ([2026-11-11_c100_kpool_on_cards](../2026-11-11_c100_kpool_on_cards/README.md)).
With C100 on, each card that runs one of Qwen3.8-Flash-Next's QSA layers receives, per micro-batch, the update of the
k-pool tables as one copy, plus the pool mask, plus five small inputs, each a copy of its own over the card's PCIe 1.0 x1
link: `k_idxs`, `tail_idxs`, `new_pool_idxs`, `new_pool_rep` and `new_pool_pos`. A copy has a fixed cost on these links
whatever its size ([2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md#outside-the-kernels-measured-r64)).
The switch puts the five into the update's copy, so a card gets two copies a micro-batch (the update and the mask)
instead of seven. This page is the design, the CPU check, and the rig run that is still to do
([Q1275](../../../quants/experiments.md)).

## Result

**On a CPU the switch is exact: at every micro-batch each op that reads one of the five inputs reads the same values
from the casts of the staged copy as from the host's tensors today, and every logit is the same with the switch and
without it.** The checks ([summary.tsv](summary.tsv), [stage.tsv](stage.tsv)):

- **`test-kpool-stage`, 16 arms, all pass.** Four contexts (without a switch; C100's tables with the stage and C100's
  check; with the tables and the stage; C100's tables alone) took the same 237 or 239 decodes per arm: 18,032 outputs
  compared, none differing in any bit. An eval callback read, after each op that reads one of the five, what it read:
  21,180 reading ops (4,236 of each input) and 197,732 values, equal between the context without the switch and the
  one with it in every op; the first read only the host's tensors and the second only the casts. Per arm 28 or 39
  rollbacks of a verify step, 13 or 20 two-sequence batches, 5 or 7 partial checkpoints and 3 or 4 whole sequence states
  restored, 3 cleared memories; C100's check compared 4,236 of 4,236 micro-batches (61,208 test checks, 0 failed).
- **`test-llama-archs` against `pascal-p106-v4`'s build (`84f7bd3d2`):** with the switches off all 146 random models
  give its logits bit for bit (one batch, then steps of 1 to 8 tokens); qwen4exp gives them with
  `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1`, with C100's check, with the check at full verbosity (the stage
  switch's line in each of the three contexts, the check's line, no difference) and with `LLAMA_QSA_KPOOL_DEV=2`.
- **The rig tarball builds** (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38): `27e3d69fa`, 647,700,954 B, md5
  `453ca4d04984ce4adee76ff597ba29cd`.

What a CPU cannot show is the speed: the rig row R80 ([Q1293](../../../quants/experiments.md)) below.

## What goes up, and how

Per micro-batch the host builds, for each QSA card (four-cell pools; n_new is the number of pools the micro-batch
completes, padded to a stable bound: 1 for a one-token pass of one sequence, 2 for a four-token verify pass):

| Input | What it holds | Type today | Bytes today, 1 token / 4 tokens | In the update with the switch |
|---|---|---|---:|---|
| `k_idxs` | the cache cell each token's indexer key is written to | I64, a token | 8 / 32 | F32, 4 / 16 B |
| `tail_idxs` | each token's own incomplete pool's other cells (or the sentinel) | I32, 3 a token | 12 / 48 | F32, the same bytes |
| `new_pool_idxs` | the member cells of each pool to pool now | I32, 4 a new pool | 16 / 32 | F32, the same bytes |
| `new_pool_rep` | the cell each new pooled key is written to | I64, a new pool | 8 / 16 | F32, 4 / 8 B |
| `new_pool_pos` | the M-RoPE position of each new pool's first member | I32, 4 a new pool | 16 / 32 | F32, the same bytes |
| **the five** | | **five copies** | **60 / 160** | **in the update's copy, 52 / 136 B** |

Copies a QSA card receives per micro-batch:

| Line | Copies | Which |
|---|---:|---|
| `pascal-p106-v4` | 8 | `pool_cells`, `pool_idxs`, `pool_mask` and the five |
| + `LLAMA_QSA_KPOOL_DEV=1` (C100) | 7 | the tables' update, `pool_mask` and the five |
| + `LLAMA_QSA_KPOOL_STAGE=1` (this) | 2 | the update with the five in it, and `pool_mask` |

The pool mask is the other part left: Q1274 builds it on the cards from 8 bytes a token, sent as a copy of its own, and
Q1294 would put those bytes in the same update, one copy a card and micro-batch.

- **The layout.** The update's F32 input, which the first QSA layer on each card already reads whole (named
  `kpool_stage`, placed on that layer's card, so the scheduler uploads it once a card), grows by the five, in that
  order, after the tables' rows: `[rows | cells | members | k_idxs | tail_idxs | new_pool_idxs | new_pool_rep |
  new_pool_pos]`. Its size is a function of the graph's shapes only (the update's padded rows, the tokens, n_new), so a
  decode step's graph is still reused step after step.
- **On the card,** five casts (F32 to I32) of five views of the staged copy, once per graph and card, named
  `kpool_k_idxs-<layer>` and so on, and placed with the stage; every QSA layer on that card reads them where it read the
  host's tensors: the indexer key's `SET_ROWS` into the cache, the new pools' `GET_ROWS`, their rotation (`ROPE`), the
  pooled keys' `SET_ROWS`, and the top-k mask (or the selection's `CONCAT` without the fused mask). Both
  `SET_ROWS` take I32 row numbers where they took I64; ggml's CPU and CUDA kernels take either, and no kernel is new
  (`test-backend-ops` has cases for every one: `CPY` F32 to I32, `SET_ROWS` and `GET_ROWS` with I32 rows).
- **Exact.** Every value is a cell number, the n_kv sentinel or a position, all whole numbers far below 2^24, which
  F32 holds exactly; the host asserts each one is inside ±2^24 before it writes it. The host still fills the five
  tensors as it does without the switch (C100's `LLAMA_QSA_KPOOL_DEV_CHECK=1` compares them with its own build every
  micro-batch) and converts them into the update; no op reads them, so nothing uploads them.
- **Off by default, and off means C100's graph.** Without `LLAMA_QSA_KPOOL_STAGE` (or without `LLAMA_QSA_KPOOL_DEV`,
  which it needs: it has no effect where the tables are uploaded whole, and pipelined decoding turns both off) the graph
  is C100's: the only change is that the five host inputs carry names. `LLAMA_QSA_KPOOL_DEV=2` (the MTP draft's
  context only) with the stage switch stages the draft's five only.

## What the rig run must show (R80)

The CPU check covers the values; on the cards it covers nothing but the arithmetic is the same kernels' (`CPY` F32
to I32 runs in C100's update already). R80 ([Q1293](../../../quants/experiments.md)) is R64's host profile with the
switch on, after R74 (C100's own run, which measures each copy's fixed cost per card): one deep-lane run
([bench/deep_lane](../../../bench/deep_lane/README.md)) with the c720 tarball at 0, 16,384 and 49,152 tokens:

| Load | Arm | What it shows |
|---|---|---|
| `warm` | the preset with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` | the control, and the warm-up |
| `dev` | + `LLAMA_QSA_KPOOL_DEV=1` | C100 alone: the step and the depth-0 reply hash |
| `stage` | + `LLAMA_QSA_KPOOL_STAGE=1` | the step at 16,384 and 49,152 against `dev`; the depth-0 reply hash equal to `dev`'s and `warm`'s |
| `check` | `stage` + `LLAMA_QSA_KPOOL_DEV_CHECK=1`, `-lv 4` | no abort on any micro-batch; the stage switch's log line |
| `hostdev` | R64's host profile with `LLAMA_QSA_KPOOL_DEV=1`, `LLAMA_QSA_INPUT_STATS=1`, `-lv 4` | each card's host gap before its part of the verify pass, the draft's four crossings to card 8 |
| `hoststage` | the same with `LLAMA_QSA_KPOOL_STAGE=1` | the same parts with two copies a card instead of seven: each gap shrinks by five copies' fixed cost |

```bash
# from a machine that reaches the rig over ssh (RIG: its ssh alias; BIN: the c720 tarball's bin/ on the rig; MODEL,
# MODEL_DRAFT: paths on the rig; WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests)
RIG=$RIG WORK=$WORK BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="sparse dev stage check hostdev hoststage" \
  ARM_ENV_dev="LLAMA_QSA_KPOOL_DEV=1" \
  ARM_ENV_stage="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1" \
  ARM_ENV_check="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_KPOOL_DEV_CHECK=1" ARM_ARGS_check="-lv 4" \
  ARM_ENV_hostdev="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hostdev.csv" ARM_ARGS_hostdev="-lv 4" \
  ARM_ENV_hoststage="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hoststage.csv" ARM_ARGS_hoststage="-lv 4" \
  LOADS="warm:sparse dev:dev stage:stage check:check hostdev:hostdev hoststage:hoststage" DEPTHS="0 16384 49152" \
  WARM_DEPTHS="16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect

# on a build machine, CPU only: each profiled load cut as R64's, --log for the k-pool lines
R=docs/data
H="nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/host_depth.py \
  --step-split $R/2026-11-10_served_step_split/run/step_split.py"
for l in hostdev hoststage; do
  $H --host $OUT/$l.csv --requests $OUT/requests.csv --load $l --ref-load warm --grow 16384:49152 \
    --log $OUT/server-$l.log --out $RES/$l > $RES/$l/host_depth.md
done
grep -a 'go up in the k-pool tables' $OUT/server-check.log | head -n 2   # the switch's line, once per memory
grep -a 'kpool_dev_check' $OUT/server-check.log | tail -n 3            # the check's counts; no "differs" line
```

## The CPU check

On a build machine's CPU (no GPU), in containers without network, `run/cpu_check.sh`:

- **build:** a static CPU build of the branch's `test-llama-archs` and the new `tests/test-kpool-stage`.
- **stage:** `test-kpool-stage` on a random qwen4exp from `test-llama-archs` (QSA layers with four-cell pools), C100's
  `test-kpool-dev` steps with four contexts: without a switch; with the tables on the devices, the stage and C100's
  check; with the tables and the stage; with the tables alone (C100). Every logit of every step must be equal bit for
  bit in all four. In the first two an eval callback stops after every op that reads one of the five inputs and reads
  what it read (in the first the host's tensors, in the second the casts from the stage); per step and input the two
  lists, one entry per reading op in graph order, must be equal value for value, the first must read only the host's
  tensors and the second only the casts. Sixteen arms: per-sequence and unified KV, the model's top-k (every pool
  kept) and 32 (the selection drops pools), micro-batches of 16 and 512 tokens, two seeds.
- **archs:** `test-llama-archs`' logits files: `pascal-p106-v4`'s CPU build (`84f7bd3d2`, the line C100 was cut from)
  writes every random model's logits (one batch, then steps of 1 to 8 tokens), the branch compares them with the
  switches off on every architecture, then qwen4exp with `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1`, with C100's
  check (once more at `-v 5`, which prints the library's lines), and with `LLAMA_QSA_KPOOL_DEV=2` (the draft only).

```bash
# on a build machine with Docker, CPU only. WORK: a work folder; SRC: a llama.cpp clone holding the branch; BASE_BIN:
# bin/ of pascal-p106-v4's static CPU build (test-llama-archs); LOCK: a flock file shared with other builds
WORK=$WORK SRC=$SRC TIP=27e3d69fa BASE_BIN=$BASE_BIN LOCK=$LOCK JOBS=8 THREADS=4 DOCKER_ARGS="--label c720=1" \
  nice -n 19 bash docs/data/2026-11-11_c720_kpool_small_inputs/run/cpu_check.sh
# the rig tarball (TESTS=1 adds test-backend-ops)
WORK=$WORK TESTS=1 JOBS=8 DOCKER_ARGS="--label pascal-quants=1 --cpu-shares 256" \
  nice -n 19 bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c720
```

## Files

| File | What it holds |
|---|---|
| [summary.tsv](summary.tsv) | every check of the CPU run: its arm, verdict and detail |
| [stage.tsv](stage.tsv) | per `test-kpool-stage` arm: decodes, outputs compared, reading ops and values compared per input, rollbacks, checkpoints and states restored, micro-batches checked |
| [run/cpu_check.sh](run/cpu_check.sh) | the CPU check |

The branch's own files: `src/models/qwen4exp.cpp` (the layout, `set_input_small`, `build_kpool_small`),
`src/models/models.h`, `src/llama-memory-hybrid-idx.{h,cpp}` (the switch), `tests/test-kpool-stage.cpp`.

## Limits

- **Nothing here ran on a card;** the speed and the casts on a P106 are R80's.
- **What a copy saves is not measured yet.** R64 measured the inputs' bytes at the link's speed (~189 MB/s); the fixed
  cost of each copy per card is R74's host profile, and R80's `hoststage` against `hostdev` is the saving.
- **The host still builds the five tensors** and then converts them (a few hundred values a decode step); the bytes
  per card fall a little (I64 to F32 for two of them), the host's work does not.
- **The random qwen4exp has one QSA layer** and takes the fused top-k mask (one read of each input per micro-batch),
  so the CPU check covers one layer reading the casts and not the unfused selection's `CONCAT` and cast; on the rig
  every QSA layer of a card reads the same casts, as every layer reads C100's tables there.
- **A random model's selection** keeps every pool at its own top-k; the top-k 32 arms make it drop pools.
