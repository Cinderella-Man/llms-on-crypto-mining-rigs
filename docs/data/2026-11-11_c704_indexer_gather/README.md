# C704 (Q1244): the k-pool indexer reads the pooled keys where they are (2026-11-11, the night's code rows)

Qwen3.8-Flash-Next's QSA layers pick, for every token, the blocks of four cache cells it attends to. Each block is
scored by one pooled indexer key, and the cache keeps that key in the row of the block's last cell, beside the cell's
raw indexer key. Before scoring, every QSA layer gathered all of the context's pooled keys by cell into a new F32
tensor (a `GET_ROWS` of `pool_cells`), and `LIGHTNING_INDEXER` read that copy. On nine P106-090 cards that gather took
4.97, 9.84 and 14.69 ms of a served step at 16,384, 32,768 and 49,152 tokens: +9.71 ms of the step's 30.44 ms of
growth over that range, the largest kernel part that grows with depth
([2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md), R51).

This folder holds the change that removes the gather, behind a new switch on Miner Llama's v4 line, and its CPU
checks. The registry row is Q1244 in [quants/experiments.md](../../../quants/experiments.md); the switch is listed in
[docs/switches.md](../../switches.md). The check on a P106 and the time it saves are the rig row Q1282.

**Result:** with `LLAMA_QSA_INDEXER_ROWS=1` the indexer reads each pool's key where the cache keeps it, through a new
form of the op, `ggml_lightning_indexer_rows`, and the gather is gone from the graph. On a build machine's CPU the new
op gives the gathered copy's scores bit for bit in all 110 cases that compute both, and every operator case passes
(323 of 323). The whole-model check gives `pascal-p106-v4`'s logits bit for bit on all 146 random models
`test-llama-archs` builds, with the switch off and with it on. Where the selection follows the scores (a top-k of 8 of
qwen4exp's and glm5-next's 128 cells), on and off give the same 32,768 logits. Without the gather, qwen4exp's graphs
hold 37 fewer `GET_ROWS` nodes, one per indexer call. The rig tarball builds. The CUDA cases are recorded for a P106
(269 cases, 1.6 GB). Not run on a card yet: whether the step gets faster is rig row Q1282.

## Two ways, and the one taken

The row named two ways to the same scores:

1. **An indexer that takes `pool_cells` and reads the cache's pooled rows.** The op gets the list of cells as a fifth
   source and reads key `i` from cell `pool_cells[i]`.
2. **Pooled keys kept in pool order, so that the gather becomes a view.** The cache would hold the pooled keys in a
   buffer of their own, in the order the indexer scores them.

**The first is the one written,** for three reasons:

- **It leaves the cache as it is.** A pooled key sits in its block's last row today, so every cache operation already
  moves it with its cells: a sequence copy, a removal and the speculative rollback after a rejected draft, a state save
  and restore, a context checkpoint. Keys in pool order would be a second structure that each of those would have to
  keep in step. Pool numbers are also not stable: they are laid out sequence after sequence, so a pool that one
  sequence gains moves every later sequence's pools, and an edit regroups the pools from the edited position on.
- **A view cannot follow the cells.** A ggml view has a fixed offset and stride when the graph is built. The pools of
  a sequence are rows at a stride of four cells only while its cells are contiguous and in order, and the graph is
  reused across steps. A list of cells read inside the kernel follows any layout.
- **The read is one index more in a kernel that loads each key anyway.** The vec and tile kernels load a key row from
  `K + i*nb2`; with the list they load it from `K + rows[i]*nb2`. Nothing else in them changes.

It also halves what the indexer reads: the cache holds F16 keys (the preset's cache type), the gathered copy was F32.

## The change

Branch `pascal-p106-v4-b12246-c704` at **`763fc55ca`**, three commits on `pascal-p106-v4` (`84f7bd3d2`, the
`-ladder6` tree): `f8f339506` (the change), `173745df1` (two test details) and `763fc55ca` (a name clash that the first
build found).

| File | What |
|---|---|
| `ggml/include/ggml.h`, `ggml/src/ggml.c` | `ggml_lightning_indexer_rows(q, k, weights, mask, rows)`: the scores of `ggml_lightning_indexer(q, ggml_get_rows(k, rows), weights, mask)`. `rows` (I32, one row of `n_kv`) is the op's `src[4]`; `k` is any `[n_embd, 1, n_rows]` with row strides, here every cell's pooled slot |
| `ggml/src/ggml-cpu/ops.cpp` | score `i` reads row `rows[i]` (bounds checked) |
| `ggml/src/ggml-cuda/lightning-indexer.cu` | the vec and tile kernels take a `ROWS` template flag and the list; a list never takes the mma kernel (below); `GGML_CUDA_LID_ROWS_FAULT=1` scores rows 0 to n−1 instead of the listed ones, for the checks; the support check accepts a list (I32, contiguous) with 16-byte aligned rows |
| Metal, Vulkan, SYCL, OpenCL, Hexagon | decline an indexer with a list (their kernels read every row of `k`), so the scheduler runs it on the CPU there; the tensor-split meta backend asserts the list mirrored |
| `src/llama-memory-hybrid-idx.{h,cpp}` | `kpool_access::pooled_cells(scattered)`: every cell's pooled slot as a strided view of the scatter's result, so that the indexer depends on the scatter that writes this ubatch's new pooled keys in every graph order; `kpool_access::indexer_rows()` reads the switch |
| `src/models/qwen4exp.cpp` | `build_qsa_sel`: with the switch on, no gather; the indexer reads `pool_cells` from the cache |
| `src/models/glm5-next.cpp` | the same in its fused-indexer path (glm5-next caches key, gate and pooled key per cell) |
| `tests/test-backend-ops.cpp` | `test_lightning_indexer_rows` (below) and two perf cases at Flash-Next's 49,152-token shape |
| `tests/test-llama-archs.cpp` | `LLAMA_ARCHS_INDEXER_TOP_K=<n>`: a top-k smaller than the context, so that a model's logits depend on the indexer's scores |

**The switch:** `LLAMA_QSA_INDEXER_ROWS=1` (read once); unset or `0` is the default. With it off the graph is
`pascal-p106-v4`'s node for node: the scatter, the gather and `ggml_lightning_indexer` as before, and the CPU and CUDA
kernels take their old path (the list is null, the `ROWS=false` kernels index row `i` as before).

**Why the scores are the gather's bit for bit.** The gather turned each cache element into an F32 (`GET_ROWS` widens F16
and dequantizes quantized types exactly), and the indexer then read F32 keys. With the list, the kernels read the cache
type and widen each element themselves:

- the CPU converts a row with the type's `to_float`, which gives the same F32 values as the gather's conversion, and
  then runs the same dot products in the same order;
- the CUDA vec kernel (one to seven tokens, and every batch at 32 or 64 heads) makes the same float of each element
  (`__half2float`, or the same dequantization as `GET_ROWS`); the tile kernel (four heads, eight tokens or more) stages
  keys as halves, and an F16 element read directly is the half the F32 copy rounded back to;
- the mma kernel, which rounds the queries to half, never took the F32 copy (it serves F16 and quantized keys on Turing
  and newer). A list stays on the vec and tile kernels on every card, so the result does not depend on the card. The
  P106 has no mma kernel either way.

The padded pools (the list's entries past the real pools) read the same dummy cell as the gather did, and their mask
is −∞ in both.

**What it saves (est.):** the gather, 4.97 / 9.84 / 14.69 ms a step at 16,384 / 32,768 / 49,152 tokens (R51,
3.3 % / 5.9 % / 8.1 % of the unprofiled 151.42 / 165.67 / 181.86 ms), and part of the indexer's scores (1.98 / 3.69 /
5.39 ms), which now read 256 bytes a key instead of 512. The compute buffer no longer holds the copy: est. 8 MiB a layer
at a 65,536-token context (16,384 pools × 128 F32).

## The checks

All on the build machine's CPU, with no GPU, in containers at the lowest CPU priority on 2026-11-10 from 22:16 to
22:57 ([checks.tsv](checks.tsv)):

| Check | Result |
|---|---|
| the rig build (`TESTS=1`) | builds: `llama.cpp-20261110-763fc55ca-pascal-p106-v4-b12246-c704.tar.gz`, 648,108,603 B, md5 `0f7095f4c040f2ddef2811dc97ec6efb`, 0 AVX instructions, GLIBC_2.38 at most |
| `test-backend-ops -b CPU -o LIGHTNING_INDEXER`, the rig build's CPU code against itself | 323 of 323 OK ([cpu_cases.tsv](cpu_cases.tsv)) |
| the recorded set, replayed on the CPU | 269 of 269 OK, none missing ([cpu_replay_cases.tsv](cpu_replay_cases.tsv)) |
| `test-llama-archs` against `pascal-p106-v4`'s CPU build, the switch off | 146 models, 0 differ in any logit |
| the same, the switch on | 146 models, 0 differ in any logit |
| qwen4exp and glm5-next at top-k 8, on against off | 0 of 32,768 logits differ in each |
| the control: top-k 8 against every cell | both differ, so at top-k 8 the selection decides the logits |
| qwen4exp's graphs, off and on (`GGML_SCHED_DEBUG=2`) | 555 and 518 `GET_ROWS` nodes; 37 indexer nodes each, 37 and 0 of them reading a gathered copy |

**`test_lightning_indexer_rows`** builds a cache of `n_cells` rows of `2 × 128` elements (raw key, pooled key), takes the
pooled half as a strided view, and lists `kv` of its cells in a random order with every seventh entry a repeat. With
`same=1` (110 cases) the output is the in-place scores and, beside them, `ggml_lightning_indexer` over the list's
`get_rows`; the case fails if the two differ in any bit, on the tested backend or on the reference, and otherwise
compares the backends by the finite scores' normalized error (limit 1e-6). The cases: the eight key types the CUDA
kernels take × 4, 32 and 64 heads × 1, 4, 8 and 512 tokens (1,000 cells, 250 listed); lists of 1, 7, 63, 65 and 1,100
entries at 4 and 9 tokens (the vec and the tile kernel, the tail of a block, a list longer than the cache);
Flash-Next's shapes, 4,096 and 12,288 pools of a 65,536-cell F16 cache (16,384 and 49,152 tokens) at 1 and 4 tokens;
and three cases of two streams with one list (`same=0`).

On the CPU the backend and the reference are the same code, so `-b CPU` checks the identity of the two paths and
little else; the CUDA kernels are checked by the recorded set on a card (Q1282).

**The recorded set** (`scripts/record_checks.sh SUITE=indexer-rows`, made with the rig build's own `test-backend-ops`,
md5 `363e512b16b347666e6f22c6d40d406d`): 269 cases, the 113 in-place ones and the 156 others that select the same
kernels as Flash-Next (every four-head case) or are small (one and 32 tokens at 32 and 64 heads; their 512-token cases
are 7.6 GB and take the same kernel). 1,623,428,321 B of files; the tar is 1,624,135,680 B, md5
`ef22b0d2b3215d3ef8fc899fc7a40101`. Two runs ([set_runs.tsv](set_runs.tsv)): `lid`, every case, which must pass; and
`lid-rows-fault` with `GGML_CUDA_LID_ROWS_FAULT=1` over the in-place cases, which must fail cases. The cases, their
sizes and md5s: [set_manifest.tsv](set_manifest.tsv).

## What is not checked yet

- **On a P106:** the replay of the recorded set on a card, the greedy replies at depth with the switch off and on, and
  the step's time: rig row Q1282 below.
- **glm5-next** has no model file here; its path is covered by `test-llama-archs`' random model only.
- **The existing indexer cases cannot fail where their mask has −∞ blocks.** `test_lightning_indexer` compares with the
  default normalized squared error. Where the mask holds −∞ (cases of 40,960 scores or more: every 512-token case and
  the Flash-Next prefill case), both backends' scores hold −∞ in the same places, the error of those pairs is NaN, and
  `NaN > limit` is false, so such a case passes whatever the finite scores are (read from the code, Q1283). The new
  `test_lightning_indexer_rows` compares the finite scores only.

## The rig row (Q1282)

On one rig, est. 60 minutes in all (the set's relay and replay est. 5, the deep lane est. 55: a warm-up and five
loads at four depths, one of them profiled), with 1.7 GB free on a rig disk for the set, after Q1224 (R57, the
`-ladder6` line's first rig check) has run there. Paths are variables: `RIG` (the rig's ssh alias), `BUILD` (the build machine's ssh alias), `SETS` (the recorded set's folder on
the build machine), `WORK` (a folder on the rig), `BIN` (the `bin/` of this branch's rig tarball on the rig), `MODELS`
(the model folder on the rig), `RECORDED` (the release check's recording here), `OUT` (a results folder here). From a
machine that only runs ssh, at the repository's root:

```bash
# 1. the recorded set on one card (CPU references from the build machine; every case must pass in run lid, and run
#    lid-rows-fault must fail cases)
ssh $BUILD "cat $SETS/indexer-rows.tar" | ssh $RIG "mkdir -p $WORK/sets && tar xf - -C $WORK/sets"
ssh $RIG "cat > $WORK/replay_checks.sh" < scripts/replay_checks.sh
ssh $RIG "SET=$WORK/sets/indexer-rows BIN=$BIN OUT=$WORK/replay DEV=0 bash $WORK/replay_checks.sh; cat $WORK/replay/console.txt"
ssh $RIG "cat $WORK/replay/results.tsv" > $OUT/replay_results.tsv && ssh $RIG "rm -rf $WORK/sets"

# 2. the deep lane: the v4 candidate (R70's five switches) with the switch off and on, two loads each, then one
#    profiled load with it on (R51's method); greedy replies and the step at 0 to 49,152 tokens
V4C_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
V4C_ARGS="--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512"
RIG=$RIG WORK=$WORK/lane BIN=$BIN/llama-server \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=$RECORDED/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 PAUSE=4 \
  ARMS="v4c rows prof" \
  ARM_ENV_v4c="$V4C_ENV" ARM_ARGS_v4c="$V4C_ARGS" \
  ARM_ENV_rows="$V4C_ENV LLAMA_QSA_INDEXER_ROWS=1" ARM_ARGS_rows="$V4C_ARGS" \
  ARM_ENV_prof="$V4C_ENV LLAMA_QSA_INDEXER_ROWS=1 GGML_CUDA_PROFILE=$WORK/lane/out/prof-rows.csv GGML_CUDA_PROFILE_MODE=group" \
  ARM_ARGS_prof="$V4C_ARGS" \
  LOADS="warm:v4c v4c1:v4c rows1:rows v4c2:v4c rows2:rows prof:prof" WARM_DEPTHS="16384" DEPTHS="0 16384 32768 49152" \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK/lane bash bench/deep_lane/remote.sh wait
ssh $RIG "gzip -k -9 $WORK/lane/out/prof-rows.csv"
RIG=$RIG WORK=$WORK/lane OUT=$OUT bash bench/deep_lane/remote.sh collect

# 3. on a build machine, CPU only: the profiled load cut as R51's, beside R51's buckets in
#    docs/data/2026-11-10_step_parts_depth/buckets.tsv (TRACE: the collected prof-rows.csv; SHALLOW: R49's
#    prof-verify.csv, md5 92a891f032d1aa9c6278f0e59acf34ce; RES: a folder for the tables)
R=docs/data
V="nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/depth_parts.py \
  --verify-parts $R/2026-11-10_served_verify_parts/run/verify_parts.py --kpass-parts $R/2026-11-04_kpass_profile/run/kpass_parts.py"
$V --event-us 7.6 --shallow $SHALLOW --shallow-label R49 --shallow-depth "79 to 335" \
  --deep $TRACE --requests $OUT/requests.csv --load prof --ref-load rows1 --out $RES > $RES/depth_parts.md
```

What it decides: the replay passes or the change stops there. `deep_table.py`'s check (written by the driver) must show
the same greedy reply for both arms at depth 0; at depth, where the control's own two loads already differ, the line
counts replies per arm (as in R70). The step on pass 2 per depth classes the switch as a gain, negligible or a loss
against the spread of the control's two loads; the profiled load names what the indexer's part of the step became
(no `GET_ROWS` before it, the scores' time against R51's 5.39 ms at 49,152). The server logs' compute buffer sizes
per card, both arms, give the memory. A gain goes into the v4 preset candidate and the agent check (Q1261).

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | every check: command, build, setting, result |
| [cpu_cases.tsv](cpu_cases.tsv) | `test-backend-ops -b CPU -o LIGHTNING_INDEXER`: the 323 cases and their verdicts |
| [cpu_replay_cases.tsv](cpu_replay_cases.tsv) | the recorded set replayed on the CPU: 269 cases and their verdicts |
| [set_manifest.tsv](set_manifest.tsv) | the recorded set: per case its id, case string, input and reference bytes and md5s |
| [set_runs.tsv](set_runs.tsv) | the set's two runs for `scripts/replay_checks.sh` |
| [run/chain.sh](run/chain.sh) | the build machine's chain: build, CPU build, operator cases, record, replay, whole models, graph |

## Commands

On the build machine (CPU only; no GPU), from a checkout of this repository, with the branch on the fork remote of
`$WORK/w/src` (a llama.cpp clone with an https fetch URL for the fork):

```bash
WORK=<a folder of its own> PUB=<this repository> BASE_CPU=<bin/ of pascal-p106-v4's static native CPU build> \
  LOCK=<a flock file shared with the machine's other jobs> DOCKER_ARGS="--label <your label>" \
  nice -n 19 bash docs/data/2026-11-11_c704_indexer_gather/run/chain.sh
```

[run/chain.sh](run/chain.sh) builds the rig tarball (`TESTS=1 scripts/build_llamacpp_docker.sh
fork/pascal-p106-v4-b12246-c704`, `-j 8`, CPU shares 256, `nice -n 19`), a static native CPU build of
`test-llama-archs` and `test-backend-ops` from the same checkout (`-DGGML_NATIVE=ON -DGGML_CUDA=OFF`, the flags of the
base build), runs `test-backend-ops -b CPU -o LIGHTNING_INDEXER`, records the set with
`OUT=$WORK/sets SUITE=indexer-rows BIN=<tarball>/bin bash scripts/record_checks.sh`, replays it on the CPU with the
suite's filter, runs the `test-llama-archs` comparisons with their control, and counts qwen4exp's graph nodes with the
switch off and on. `BASE_CPU` was the `-ladder6` session's CPU build of `84f7bd3d2` (C662), made with the same image
and flags. `STEPS` picks steps (default all, in order).

## Provenance

Code: the fork's branch `pascal-p106-v4-b12246-c704` (MIT, as llama.cpp). Every number in this folder comes from
random tensors and random tiny models generated by the tests; no model file or user data is involved. The R51
numbers quoted above are from [2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md).
