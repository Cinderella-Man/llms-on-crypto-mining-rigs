# The QSA k-pool tables kept on the cards (C100, Q640, 2026-11-11)

What this is: a new switch on the v4 line, `LLAMA_QSA_KPOOL_DEV=1` on the fork branch `pascal-p106-v4-b12246-c100`
(`ce611b9db`, three commits on `pascal-p106-v4` `84f7bd3d2`), off by default. With it on, the two tables that tell
Qwen3.8-Flash-Next's QSA layers which cells form each pool stay on the cards between micro-batches, and each
micro-batch sends only the rows that changed. Today every micro-batch builds both tables on the host and uploads them
whole to every card that runs a QSA layer, and the R64 host profile put those uploads on the token's path
([2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md#outside-the-kernels-measured-r64),
[Q640](../../../quants/experiments.md)). This page is the design, the CPU check, and the rig run that is still to do.

## Result

**On a CPU the switch is exact: the tables the cards would hold are byte for byte the ones uploaded today, at every
micro-batch, and every logit is the same with the switch and without it.** The checks
([summary.tsv](summary.tsv), [kpool.tsv](kpool.tsv)):

- **`test-kpool-dev`, 16 arms, all pass.** Three contexts (without the switch, with it and
  `LLAMA_QSA_KPOOL_DEV_CHECK=1`, with it alone) took the same 237 or 239 decodes per arm: 18,032 outputs compared, none
  differing in any bit. Per arm 28 or 39 rollbacks of a verify step, 13 or 20 two-sequence batches, 5 or 7 partial
  checkpoints and 3 or 4 whole sequence states restored, 3 cleared memories. The check compared 4,236 of 4,236
  micro-batches (14,224 test checks, 0 failed): every card copy as the steps before left it, every update, and every
  other k-pool input against the host path's own.
- **`test-llama-archs` against the base's build:** with the switch off all 146 random models give `84f7bd3d2`'s
  logits bit for bit (one batch, then steps of 1 to 8 tokens); qwen4exp gives them with `LLAMA_QSA_KPOOL_DEV=1`, with
  the check (0 of 32,768 logits differ; 43 micro-batches checked over three contexts at full verbosity) and with `=2`.
- **Each update stays small:** one full update per context (its first micro-batch), and at most 51.2 rows a
  micro-batch on average in any arm (each arm's figure in [kpool.tsv](kpool.tsv)). These are short contexts (at most
  320 table rows), where the padded rows (up to 64, which move with every
  micro-batch) are most of a table; the update does not grow with the context, so at 49,152 tokens a step sends its
  128 rows where the tables have 12,352.
- **The rig tarball builds** (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38): `ce611b9db`, 647,468,347 B, md5
  `1ac99860e87ad6eaaedec22bc9271e72`.

What a CPU cannot show is the speed and CUDA's own `SET_ROWS` kernel writing the tables: the rig row R74
([Q1273](../../../quants/experiments.md)) below.

## What was uploaded, and what is now

A QSA layer reads its cache through pools of four cells. Per micro-batch the host builds, for every pool:

| Input | Bytes a pool | Uploaded today | With `LLAMA_QSA_KPOOL_DEV=1` |
|---|---:|---|---|
| `pool_cells` (the cell holding the pool's pooled key) | 4 | whole, to each card with a QSA layer | kept on each card, the changed rows only |
| `pool_idxs` (the pool's four member cells) | 16 | whole, to each card | kept on each card, the changed rows only |
| `pool_mask` (which pools each token may see) | 2 per token | whole | whole, as before |

Bytes a card receives for the two tables in one micro-batch (`pool_cells` and `pool_idxs`; the pools are padded to a
multiple of 64, from R64's shapes):

| Context depth | Pools | Today | With the switch, a decode step | Today's mask, 1 token / 4 tokens (unchanged) |
|---|---:|---:|---:|---:|
| 16,384 | 4,160 | 83,200 B | 3,072 B | 8,320 / 33,280 B |
| 49,152 | 12,352 | 247,040 B | 3,072 B | 24,704 / 98,816 B |

So the tables' upload stops growing with depth: a decode step's update is 128 rows of 24 bytes (each row's number, its
cell and its four members) whatever the context, and R64's est. saving was 2.70 ms a step at 16,384 tokens and
12.43 at 49,152 (the draft's four uploads to its card est. 2.08 and 6.18 of it). The mask is the part left, 9 % of a
drafting pass's bytes and 29 % of a four-token verify pass's (new row Q1274: build it on the cards too).

## How it works

- **One copy per card.** The memory of each qwen4exp context (the main model's and the MTP draft's) allocates, for
  every card that holds a QSA layer's indexer cache, a `pool_cells` table (I32, 1 × rows) and a `pool_idxs` table
  (I32, 4 × rows) in a buffer of its own. Rows: the padded count of every pool the cache can hold, plus 256 spare rows
  at the end. At the preset's 65,536-token context that is 16,448 + 256 rows, 334,080 B a card; the host keeps the same
  tables once per context (328,960 B), which is what every copy holds.
- **The rows that changed, picked before the graph is built.** When a micro-batch is applied to the memory, the host
  builds that micro-batch's rows the way the host path builds them (each pool's rep cell and members, the sentinel for
  a sequence absent from a multi-stream micro-batch, then the padded pools on a cell of the micro-batch) straight into
  its copy, and lists the rows that differ: the padded pools (their cell moves with every micro-batch), the pools the
  micro-batch completes, and after a rollback or a restored checkpoint the pools from the edit on. The update's size
  is a graph shape, so a decode step's list is padded to a stable 128 rows (one graph, reused step after step); a longer
  one (a prompt, the first micro-batch) is padded to the next multiple of 64, and the padded entries write the spare rows
  past the end, which no graph reads.
- **One upload a card, written on the card.** The rows, their cells and their members go up as one F32 input (whole
  numbers below 2^24, exact), which the first QSA layer on each card reads whole (as the card-built masks' `kq_stage`),
  so the scheduler uploads it once per card. There the rows are cast to I32 and two `SET_ROWS` write the update into
  the copy; every QSA layer on that card reads its first n_pool rows. CUDA's `SET_ROWS` gained an I32 destination for
  this (F32 to I32); `test-backend-ops` has three new cases for it.
- **The host's copy is committed only after the micro-batch succeeded** (the memory context's `next()`). An update no
  graph ran, or a card a graph left out, makes the next micro-batch rewrite every row. Pipelined decoding
  (`llama_pipe_init`), whose slots' graphs are in flight together, turns the switch off.
- **`LLAMA_QSA_KPOOL_DEV=2`:** the MTP draft's context only, the row's cheapest first part (one card, four passes a
  step).
- **`LLAMA_QSA_KPOOL_DEV_CHECK=1`** synchronizes before the inputs and, every micro-batch, (1) builds all QSA inputs
  with the host path into tensors of its own and compares them byte for byte with what the switch sends: the mask, the
  tail, the new pools' lists and the host's copy of the tables; (2) reads each card's copy, compares it with what the
  host held before this micro-batch (when the copies were in step), applies this update to it and compares the result
  with the host path's tables. Any difference aborts. `llama_kpool_dev_stats` counts the micro-batches, full updates,
  rows sent and checks; each memory logs the same counts when it is freed.
- **Off by default, and off means the old path:** without the switch the memory allocates nothing and the graph
  uploads `pool_cells` and `pool_idxs` as `pascal-p106-v4` does; the host path's code is unchanged but for the branch
  that skips its two tables when they are on the cards.

## What the rig run must show (R74)

The CPU check covers the tables' contents, the update and its commit. On the cards, one deep-lane run
([bench/deep_lane](../../../bench/deep_lane/README.md)) with the c100 tarball, R64's command with five loads:

| Load | Arm | What it shows |
|---|---|---|
| `warm` | the preset with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` | the control, and the warm-up |
| `dev` | + `LLAMA_QSA_KPOOL_DEV=1` | the step at 16,384 and 49,152 against `warm`; the reply hash at depth 0 equal to `warm`'s |
| `check` | + `LLAMA_QSA_KPOOL_DEV_CHECK=1`, `-lv 4` | every micro-batch's copies on all nine cards identical to the host path's tables (no abort; the check's log lines); slower by the synchronize and the read-back |
| `host` | R64's host profile with `LLAMA_QSA_INPUT_STATS=1`, `-lv 4` | R64's parts with the k-pool lines in the log (R71's load) |
| `hostdev` | the same with `LLAMA_QSA_KPOOL_DEV=1` | the parts that should shrink: each card's host gap in the verify pass, the draft's four crossings to card 8, the server's interval |

A greedy text is not an identity check between two loads of this line from 8,192 tokens on: two loads of one arm part
there ([2026-11-10_draft_line_exactness](../2026-11-10_draft_line_exactness/README.md)). So the depth-0 reply hashes
are the text check, and the `check` load is the exactness check at depth.

```bash
# from a machine that reaches the rig over ssh (RIG: its ssh alias; BIN: the c100 tarball's bin/ on the rig; MODEL,
# MODEL_DRAFT: paths on the rig; WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests)
RIG=$RIG WORK=$WORK BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="sparse dev check host hostdev" \
  ARM_ENV_dev="LLAMA_QSA_KPOOL_DEV=1" \
  ARM_ENV_check="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1" ARM_ARGS_check="-lv 4" \
  ARM_ENV_host="LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/host.csv" ARM_ARGS_host="-lv 4" \
  ARM_ENV_hostdev="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hostdev.csv" ARM_ARGS_hostdev="-lv 4" \
  LOADS="warm:sparse dev:dev check:check host:host hostdev:hostdev" DEPTHS="0 16384 49152" WARM_DEPTHS="16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect

# on a build machine, CPU only: each profiled load cut as R64's, --log for the k-pool lines
R=docs/data
H="nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/host_depth.py \
  --step-split $R/2026-11-10_served_step_split/run/step_split.py"
for l in host hostdev; do
  $H --host $OUT/$l.csv --requests $OUT/requests.csv --load $l --ref-load warm --grow 16384:49152 \
    --log $OUT/server-$l.log --out $RES/$l > $RES/$l/host_depth.md
done
grep -a 'kpool_dev_check' $OUT/server-check.log | tail -n 3   # the check's counts; no "differs" line
```

Also on a card, once: `test-backend-ops -o SET_ROWS` from the tarball (the three I32 cases among them).

## The CPU check

On a build machine's CPU (no GPU), in containers without network, `run/cpu_check.sh`:

- **build:** a static CPU build of the branch's `test-llama-archs` and the new `tests/test-kpool-dev`.
- **kpool:** `test-kpool-dev` on a random qwen4exp from `test-llama-archs` (QSA layers with four-cell pools). Three
  contexts take the same steps and every logit of every step must be equal bit for bit: without the switch, with it
  and the check, and with it alone. The steps: prompts in micro-batches, one-token steps, four-token steps of which a
  rollback keeps one to three (`n_rs_seq` 4, as the server's speculative decoding), two sequences alone and in one
  batch, a partial checkpoint (the recurrent state) restored and the cells after it removed as the server's
  speculative steps do, a whole sequence's state saved and restored, a sequence removed and prompted again, a cleared
  memory. Sixteen arms: per-sequence and unified KV, the model's top-k (every pool kept) and 32 (the selection drops
  pools, so the logits depend on which pools the tables name), micro-batches of 16 and 512 tokens, two seeds.
- **archs:** `test-llama-archs`' logits files: the base's CPU build (`84f7bd3d2`) writes every random model's logits
  (one batch, then steps of 1 to 8 tokens), the branch compares them with the switch off on every architecture, then
  qwen4exp with `LLAMA_QSA_KPOOL_DEV=1`, with the check, and with `=2`.

```bash
# on a build machine with Docker, CPU only. WORK: a work folder; SRC: a llama.cpp clone holding the branch; BASE_BIN:
# bin/ of pascal-p106-v4's static CPU build (test-llama-archs); IMAGE: an image with git, cmake, a compiler and ccache
WORK=$WORK SRC=$SRC TIP=ce611b9db BASE_BIN=$BASE_BIN JOBS=8 THREADS=4 \
  nice -n 19 bash docs/data/2026-11-11_c100_kpool_on_cards/run/cpu_check.sh
# the check arm once more at full verbosity, which prints the switch's own lines (BIN: bin/ of the branch's CPU build)
cd $(mktemp -d) && LLAMA_ARCHS_LOGITS=$PWD/logits- $BASE_BIN/test-llama-archs -b CPU -s 1 -a qwen4exp && \
  LLAMA_ARCHS_LOGITS=$PWD/logits- LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1 $BIN/test-llama-archs -b CPU -s 1 -a qwen4exp -v 5
# the rig tarball (TESTS=1 adds test-backend-ops)
WORK=$WORK TESTS=1 JOBS=8 DOCKER_ARGS="--cpu-shares 256" nice -n 19 bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c100
```

## Files

| File | What it holds |
|---|---|
| [summary.tsv](summary.tsv) | every check of the CPU run: its arm, verdict and detail (the last row the by-hand run at full verbosity) |
| [kpool.tsv](kpool.tsv) | per `test-kpool-dev` arm: decodes, outputs compared, rollbacks, checkpoints and states restored, micro-batches, full updates, rows sent and checks |
| [run/cpu_check.sh](run/cpu_check.sh) | the CPU check |

The branch's own files: `src/llama-memory-hybrid-idx.{h,cpp}` (the copies, the update, the commit, the check),
`src/models/qwen4exp.cpp` (the stage and the two `SET_ROWS`), `ggml/src/ggml-cuda/set-rows.cu` (I32 destination),
`tests/test-kpool-dev.cpp`, three cases in `tests/test-backend-ops.cpp`.

## Limits

- **Nothing here ran on a card.** The CUDA `SET_ROWS` with an I32 destination compiled in the rig tarball and was not
  run; the speed is R74's.
- **The mask still goes up whole** (2 bytes a pool and token): Q1274.
- **Host and card memory grow a little:** the host's copy (20 bytes a row per context: 328,960 B for the main model at
  65,536 tokens, the same for the draft), the card's copy (334,080 B a card), and the update's input reserved for a
  full rewrite (24 bytes a row) where the two tables' inputs were reserved (20 bytes a row). The rig's lowest
  MemAvailable is R74's to read.
- **The first micro-batch, and one after a failed one, rewrites every row**, and a prompt micro-batch sends a
  hundred-odd rows; the saving is in decode steps.
- **A random model's selection** keeps every pool at its own top-k; the top-k 32 arms make it drop pools.
