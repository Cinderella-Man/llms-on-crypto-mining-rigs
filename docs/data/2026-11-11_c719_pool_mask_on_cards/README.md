# The QSA pool mask built on the cards (C719, Q1274, 2026-11-11)

What this is: a new switch on the v4 line, `LLAMA_QSA_POOL_MASK_DEV=1` on the fork branch
`pascal-p106-v4-b12246-c719` (`b59cfb001`, one commit on C100's `pascal-p106-v4-b12246-c100` `ce611b9db`), off by
default. With it on, Qwen3.8-Flash-Next's QSA layers no longer receive their pool mask from the host: each card that
runs a QSA layer gets each token's range of visible pools (8 bytes a token) and builds the same mask itself with a new
ggml operation, `ggml_range_mask`. With C100's `LLAMA_QSA_KPOOL_DEV=1` the mask was the largest k-pool input still
sent whole ([2026-11-11_c100_kpool_on_cards](../2026-11-11_c100_kpool_on_cards/README.md),
[Q1274](../../../quants/experiments.md)). This page is the design, the CPU check, the CUDA cases recorded for a P106,
and the rig run that is still to do.

## Result

**On a CPU the switch is exact: the mask each device builds from the ranges is byte for byte the mask the host fills
today, at every micro-batch, and every logit is the same with the switch and without it.** The checks
([summary.tsv](summary.tsv), [mask.tsv](mask.tsv)):

- **`test-pool-mask-dev`, 16 arms, all pass.** Four contexts (without the switch; with it and
  `LLAMA_QSA_POOL_MASK_DEV_CHECK=1`; with it alone; with it, C100's `LLAMA_QSA_KPOOL_DEV=1` and both checks) took the
  same 237 or 239 decodes per arm: 18,032 outputs compared, none differing in any bit. Per arm 28 or 39 rollbacks of a
  verify step, 13 or 20 two-sequence batches, 5 or 7 partial checkpoints and 3 or 4 whole sequence states restored,
  3 cleared memories. The check read 8,472 masks back after their micro-batches, 4,236 in each of the two checked
  contexts (one a micro-batch: the CPU is one device), every one equal to the host mask byte for byte; in the fourth
  context C100's own check passed all 4,236 micro-batches as well (18,816 test checks, 0 failed).
- **`test-llama-archs` against the base's build:** with the switch off all 146 random models give `ce611b9db`'s
  (`pascal-p106-v4-b12246-c100`) logits bit for bit (one batch, then steps of 1 to 8 tokens); qwen4exp gives them with
  `LLAMA_QSA_POOL_MASK_DEV=1`, with the check (at `-v 5` its three contexts logged 2, 9 and 32 micro-batches, 43 masks
  read back, all equal), with `=2` (which leaves this test's contexts off: none of them is an MTP draft) and with
  C100's switch and both checks.
- **The rig tarball builds** (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38): `b59cfb001`, 647,941,112 B, md5
  `0bc2cecac62b9a47a90ccbe6bc1e7d40`.
- **The CUDA cases are recorded for a P106:** every `RANGE_MASK` case of that tarball's `test-backend-ops` with the
  CPU's results (22 cases, [record.tsv](record.tsv); the set `range-mask.tar`, 121,733,120 B, md5
  `81d491308ebd67cb91d3d7512b41784c`), and the set reads back on the CPU: 22 of 22 cases pass against their record.

What a CPU cannot show is the CUDA kernel on a P106 and the speed: the rig row R78 ([Q1284](../../../quants/experiments.md))
below.

## What is sent, and what is now

A QSA layer scores whole pools of four cells. Which pools a token may see is a mask of 2 bytes a pool and token (F16:
0 visible, -inf not), filled on the host every micro-batch and uploaded to every card that runs a QSA layer. Each
token's row is one range: from its sequence's first pool, every pool that ends at or before the token (in sequence
order, as qwen4exp numbers its pools). So two numbers a token give the whole row.

Bytes a card receives for the mask in one micro-batch (pools padded to a multiple of 64, from R64's shapes):

| Context depth | Pools | Today, 1 token / 4 tokens | With the switch, 1 token / 4 tokens |
|---|---:|---:|---:|
| 16,384 | 4,160 | 8,320 / 33,280 B | 8 / 32 B |
| 49,152 | 12,352 | 24,704 / 98,816 B | 8 / 32 B |

With C100's switch on too, the k-pool inputs of a decode step no longer grow with depth at all: the tables' update is
128 rows (3,072 B) and the ranges 8 bytes a token, whatever the context.

**Est. saving** (from R64's host profile, which put every upload interval at the link's speed, 186.5 and 187.7 MB/s,
and est. 9.76 ms of the step's growth from 16,384 to 49,152 tokens on the k-pool inputs' bytes; scaled here by the
mask's bytes): at 49,152 tokens est. 0.53 ms for the draft's four waits on card 8's upload (4 × 24,704 B), 0.53 ms for
card 0's upload in the server's interval (98,816 B) and est. 1.90 ms of the verify pass's crossings (R64's +4.41 ms
for 229,376 B more a card, scaled to 98,816 B): **est. 2.96 ms a step (1.6 % of 183.58)**; at 16,384 tokens est.
1.00 ms (0.18 + 0.18 + 0.64). On top of C100's est. 12.43 and 2.70 ms. Each card runs one more kernel a micro-batch
(est. a few µs on a P106) and the host no longer writes the mask (n_pool × n_tokens × 2 bytes a context).

## How it works

- **The ranges.** `set_input_kpool` already worked out each token's first visible pool `p0` and the count `nv` to fill
  the mask row. With the switch the input is I32 `[2, n_tokens]` and it writes `p0` and `p0 + nv` instead; the host
  mask is filled only for the check. The same code computes both, so the mask a card builds is the mask the host
  would have sent.
- **`ggml_range_mask(ranges, n, type)`**, a new ggml operation (CPU and CUDA): from I32 `[2, ne1, ne2, ne3]` it
  builds an `[n, ne1, ne2, ne3]` mask of F16 or F32, 0 where `start <= j < end` and -inf elsewhere. On CUDA one launch
  a card: a block row per token, a thread per pool. `GGML_CUDA_RANGE_MASK_FAULT=1` makes one more pool visible after
  each range, so a replay can show that a wrong mask fails.
- **One mask per card.** The graph builds the mask once per graph and card, at the first QSA layer on that card
  (the node `kpool_mask`, which the scheduler callback places on the layer's card, as it does the card-built
  attention masks' `kq_mask`), and every QSA layer there reads it. The ranges are one small input read whole by that
  node, so each card receives one copy, as it received the mask.
- **Per memory, like C100's switch:** `1` every qwen4exp context (the main model's and the MTP draft's), `2` the MTP
  draft's only; off with tensor split, where a layer has no single card. Pipelined decoding (`llama_pipe_init`) does
  not turn it off, as the mask holds no state between micro-batches; the check runs only outside it, and neither was
  tested there.
- **`LLAMA_QSA_POOL_MASK_DEV_CHECK=1`** keeps each card's mask as a graph output; after every micro-batch the context
  synchronizes, reads every card's mask back and compares it byte for byte with the mask the host fills without the
  switch, and aborts on any difference. `llama_pool_mask_dev_stats` counts the micro-batches, the checks and the
  masks compared; each memory logs the same counts when it is freed. On the rig this checks the CUDA kernel's own
  output in the real graph, every micro-batch and every card.
- **Off by default, and off means the old path:** without the switch the graph uploads the F16 mask as before and
  `set_input_kpool` fills it from the same values; `test-llama-archs` gives the base's logits bit for bit with the
  switch off.

## The rig row (R78, Q1284)

On a P106 rig, with the c719 tarball (which holds C100's switch too):

1. **The replay** of the recorded `RANGE_MASK` cases on one card (`scripts/replay_checks.sh`, set `range-mask`): run
   `range-mask` (every case) must pass, `range-mask-fault` must fail cases. It stops there otherwise.
2. **One deep-lane run** ([bench/deep_lane](../../../bench/deep_lane/README.md)), R64's line at depths 0, 16,384 and
   49,152, five loads:

| Load | Arm | What it shows |
|---|---|---|
| `warm` | the preset with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` | the control, and the warm-up |
| `kpool` | + `LLAMA_QSA_KPOOL_DEV=1` | C100 alone (R74's `dev`) |
| `mask` | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1` | the step at 16,384 and 49,152 against `kpool` (est. -1.00 and -2.96 ms) and `warm`; the depth-0 reply hash equal to `warm`'s |
| `check` | + both checks, `-lv 4` | every micro-batch's tables and masks on all nine cards identical to the host's (no abort; the checks' log lines) |
| `hostmask` | R64's host profile with both switches, `LLAMA_QSA_INPUT_STATS=1`, `-lv 4` | each card's host gap in the verify pass and the draft's card-8 crossings no longer growing with depth |

A greedy text is not an identity check between two loads of this line from 8,192 tokens on
([2026-11-10_draft_line_exactness](../2026-11-10_draft_line_exactness/README.md)), so the depth-0 reply hashes are the
text check and the `check` load is the exactness check at depth.

```bash
# from a machine that reaches the rig and the build machine over ssh, at the repository's root (RIG, BUILD: their ssh
# aliases; SETS: the recorded set's folder on the build machine; BIN: the c719 tarball's bin/ on the rig; MODEL,
# MODEL_DRAFT: paths on the rig; WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests; OUT: a
# results folder here)
# 1. the recorded set on one card: run range-mask must pass every case, run range-mask-fault must fail cases
ssh $BUILD "cat $SETS/range-mask.tar" | ssh $RIG "mkdir -p $WORK/sets && tar xf - -C $WORK/sets"
ssh $RIG "cat > $WORK/replay_checks.sh" < scripts/replay_checks.sh
ssh $RIG "SET=$WORK/sets/range-mask BIN=$BIN OUT=$WORK/replay DEV=0 bash $WORK/replay_checks.sh; cat $WORK/replay/console.txt"
ssh $RIG "cat $WORK/replay/results.tsv" > $OUT/replay_results.tsv && ssh $RIG "rm -rf $WORK/sets"

# 2. the deep lane

RIG=$RIG WORK=$WORK BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="sparse kpool mask check hostmask" \
  ARM_ENV_kpool="LLAMA_QSA_KPOOL_DEV=1" \
  ARM_ENV_mask="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1" \
  ARM_ENV_check="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1 LLAMA_QSA_POOL_MASK_DEV_CHECK=1" \
  ARM_ARGS_check="-lv 4" \
  ARM_ENV_hostmask="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hostmask.csv" \
  ARM_ARGS_hostmask="-lv 4" \
  LOADS="warm:sparse kpool:kpool mask:mask check:check hostmask:hostmask" DEPTHS="0 16384 49152" \
  WARM_DEPTHS="16384 49152" PAUSE=4 bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect

# on a build machine, CPU only: the profiled load cut as R64's, --log for the k-pool lines
R=docs/data
nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/host_depth.py \
  --step-split $R/2026-11-10_served_step_split/run/step_split.py \
  --host $OUT/hostmask.csv --requests $OUT/requests.csv --load hostmask --ref-load warm --grow 16384:49152 \
  --log $OUT/server-hostmask.log --out $RES/hostmask > $RES/hostmask/host_depth.md
grep -a 'pool_mask_dev_check\|kpool_dev_check' $OUT/server-check.log | tail -n 4   # the checks' counts; no "differs" line
```

## The CPU check

On a build machine's CPU (no GPU), in containers without network, `run/cpu_check.sh`:

- **build:** a static CPU build of the branch's `test-llama-archs` and the new `tests/test-pool-mask-dev`.
- **mask:** `test-pool-mask-dev` on a random qwen4exp from `test-llama-archs`. Four contexts take
  `test-kpool-dev`'s steps and every logit of every step must be equal bit for bit: without the switch; with it and
  the check; with it alone; with it, C100's `LLAMA_QSA_KPOOL_DEV=1` and both checks. The steps: prompts in
  micro-batches, one-token steps, four-token steps of which a rollback keeps one to three, two sequences alone and in
  one batch, a partial checkpoint (the recurrent state) restored and the cells after it removed as the server's
  speculative steps do, a whole sequence's state saved and restored, a sequence removed and prompted again, a cleared
  memory. Sixteen arms: per-sequence and unified KV, the model's top-k (every pool kept) and 32 (the selection drops
  pools, so the logits depend on which pools the mask lets through), micro-batches of 16 and 512 tokens, two seeds.
- **archs:** `test-llama-archs`' logits files: C100's CPU build (`ce611b9db`, this branch's base) writes every random
  model's logits (one batch, then steps of 1 to 8 tokens), the branch compares them with the switch off on every
  architecture, then qwen4exp with `LLAMA_QSA_POOL_MASK_DEV=1`, with the check (once more at `-v 5`, whose library
  lines give the check's counts), with `=2` and with C100's switch and both checks.
- **The CUDA cases for a P106:** `scripts/record_checks.sh`, suite `range-mask`, records every `RANGE_MASK` case of
  the rig tarball's `test-backend-ops` with the CPU's results, for `scripts/replay_checks.sh` on the rig.

```bash
# on a build machine with Docker, CPU only. WORK: a work folder; SRC: a llama.cpp clone holding the branch; BASE_BIN:
# bin/ of pascal-p106-v4-b12246-c100's static CPU build (test-llama-archs); IMAGE: an image with git, cmake, a compiler
# and ccache
WORK=$WORK SRC=$SRC TIP=b59cfb001 BASE_BIN=$BASE_BIN JOBS=8 THREADS=4 \
  nice -n 19 bash docs/data/2026-11-11_c719_pool_mask_on_cards/run/cpu_check.sh
# the rig tarball (TESTS=1 adds test-backend-ops)
WORK=$WORK TESTS=1 JOBS=8 DOCKER_ARGS="--cpu-shares 256" nice -n 19 bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c719
# the RANGE_MASK cases with the CPU's results, from that tarball (BIN: its bin/, unpacked)
OUT=$WORK/sets SUITE=range-mask BIN=$BIN JOBS=4 nice -n 19 bash scripts/record_checks.sh
# the record read back on the CPU before a rig sees it, in the CUDA image without GPUs and with the toolkit's libcuda
# stub on the library path, as record_checks.sh runs it
docker run --rm --network none -v $BIN:$BIN:ro -v $WORK/sets:$WORK/sets:ro nvidia/cuda:12.9.2-devel-ubuntu24.04 sh -c \
  "mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 &&
   LD_LIBRARY_PATH=/tmp/stub $BIN/test-backend-ops -b CPU -o RANGE_MASK --replay $WORK/sets/range-mask"
```

## Files

| File | What it holds |
|---|---|
| [summary.tsv](summary.tsv) | every check of the CPU run: its arm, verdict and detail |
| [mask.tsv](mask.tsv) | per `test-pool-mask-dev` arm: decodes, outputs compared, rollbacks, checkpoints and states restored, micro-batches, checks and masks compared |
| [record.tsv](record.tsv) | the recorded `RANGE_MASK` cases: id, case, input and reference bytes and md5 |
| [run/cpu_check.sh](run/cpu_check.sh) | the CPU check |

The branch's own files: `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ops.cpp` and
`ggml/src/ggml-cuda/range-mask.cu` (the operation), `src/llama-memory-hybrid-idx.{h,cpp}` (the switch, the ranges,
the check), `src/models/qwen4exp.cpp` (the input and the per-card mask), `src/llama-graph.{h,cpp}` and
`src/llama-context.cpp` (the check after the graph has run, the node's placement), `tests/test-pool-mask-dev.cpp`,
22 cases in `tests/test-backend-ops.cpp`.

## Limits

- **Nothing here ran on a card.** The CUDA kernel compiled in the rig tarball and its cases are recorded; the replay,
  the check on nine cards and the speed are R78's.
- **The saving is an estimate** from R64's bytes-to-time line; the mask was the smaller part of the k-pool inputs
  (9 % of a drafting pass's bytes, 29 % of a verify pass's, C100).
- **The ranges are one more small input a card**, where C720 (Q1275) folds the other small k-pool inputs into C100's
  one stage: merged together, the ranges can go into that stage as well.
- **A random model's selection** keeps every pool at its own top-k; the top-k 32 arms make it drop pools.
