# C702 (Q1240): CUDA graphs for prompt micro-batches (2026-11-11, the night's code rows)

On the release line no prompt micro-batch of the main model runs as a CUDA graph: in
[R42's 4,097-token load](../2026-11-10_prompt_timeline/README.md) every one of them was launched kernel by kernel
("eager"; the load's 36 captures and 9 replays were all in generation). A 512-token split is then 109 to 327 launch
groups queued one by one, and the host stops inside a call's third queued micro-batch on cards 1 to 7 until each card
is part way through its first: 8,851.0 of the prompt's 29,983.0 ms (29.5 %), 15,483.3 of 29,807.6 ms (51.9 %) with
the micro-batch taper, the pattern of a full launch queue. Captured, a split reaches its card as one launch per
graph chunk.

This folder holds the change, behind a new switch on Miner Llama's v4 line, and its CPU check. The registry row is
Q1240 in [quants/experiments.md](../../../quants/experiments.md); the switch is listed in
[docs/switches.md](../../switches.md). The idea is the multi-GPU fork's `c8832a5f0`
([note](../../research/notes/multigpu_fork_read.md), rank 1 of its port list), which measured prompts going from 859
to 2,500 tok/s on five RTX 3090s with it and three other fixes.

**Result:** the switch `GGML_CUDA_GRAPHS_FORCE=1` captures a prompt micro-batch's graph at every change and replays a
graph only from executables captured from its current properties. The port's decision could replay another graph's
executables as soon as a batched graph runs eagerly. The fixed decision, run on a CPU in a model of the backend's whole
capture logic, never replayed stale executables in 2,000 random runs of 400 evaluations (200,993 captures, 465,116
replays); the port's decision did in 711 of the 2,000. With the switch off the backend runs exactly as before. The
rig tarball builds (0 AVX instructions). Nothing ran on a GPU: the identity on the cards and the prompt time are the
rig row, Q1297.

## Why prompt graphs are never captured

ggml's CUDA backend keeps one CUDA graph per key (the graph's first node) and records every node's properties: the
node itself (operation, parameters, shape, strides, data pointer, view, source pointers) and each source's data
pointer, shape and strides. It captures a graph only after a warm-up of two calls whose properties did not change,
replays it while they stay the same, and goes back to eager launches and a new warm-up when they change. Under
pipeline parallelism a prompt micro-batch's graph changes every time: the KV views grow with the cache, the k-pool
tables grow with it, and the scheduler rotates each split's inputs through four copies, so the inputs' pointers move.
The warm-up never completes.

## The change

Branch `pascal-p106-v4-b12246-c702` at **`2ec3a0927`**, four commits on `pascal-p106-v4` (`84f7bd3d2`, the
`-ladder6` tree): `692bbbc9a` the port (every changed graph captured at once), `67d5f6835` the rules below that keep
a capture exact, `096b8e50d` the decision moved to a header and its CPU test, `2ec3a0927` the test's random runs
starting on a decode graph (so that its control meets the case it must fail).

| File | What |
|---|---|
| `ggml/src/ggml-cuda/graph-force.h` | new, host code only: which graphs the switch takes (`ggml_cuda_graph_force_eligible`) and the decision for one evaluation of such a graph (`ggml_cuda_graph_force_next`) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `GGML_CUDA_GRAPHS_FORCE`, read once (an info line when it is on); the device and stream conditions; the forced branch before the warm-up's branches in `ggml_backend_cuda_graph_compute` |
| `ggml/src/ggml-cuda/common.cuh` | one flag per backend context: a batched graph has run eagerly on it |
| `tests/test-cuda-graph-force.cpp`, `tests/CMakeLists.txt` | the CPU check below |

**Which graphs:** a graph that works on more than 8 tokens (`MMVQ_MAX_BATCH_SIZE`, the vector kernels' limit): a
prompt micro-batch or a draft's catch-up pass. The tokens are read where they are unambiguous: `MUL_MAT_ID`'s
activations (their third dimension), `ROPE`'s input (its third) and `FLASH_ATTN_EXT`'s queries (their second). A
plain `MUL_MAT`'s rows never count, because they can be heads (MLA's absorbed queries of one token are a 2-D
`[d, n_head]` tensor). Every Flash-Next layer has a `MUL_MAT_ID`. Decode and verify graphs (up to 8 tokens) keep the
warm-up: it already ends in replays for them, and it makes their lazily made state (the persistent MoE kernel's
mapped host words, the Pascal attention kernels' counters) outside a capture.

**The decision for such a graph:**

1. The backend context's first batched graph runs eagerly, so that the stream's cuBLAS handle and its workspace,
   made at their first use, are made outside a capture.
2. After that, a graph whose properties changed, or whose key last ran eagerly, is captured at once (a new capture
   generation, as a completed warm-up).
3. It is replayed only when its properties are the ones its executables were captured from. The key's executables
   are shared with the warm-up's graphs, and an eager evaluation records its properties but leaves the older
   executables in place, so the warm-up flag says whether the executables belong to the recorded properties. The
   port replayed whenever the properties had not changed since the last evaluation; once some batched graph runs
   eagerly (rule 1), that could launch the executables of other properties. The CPU check's control catches it.

**Where the switch does nothing:**

- a device without virtual memory management (VMM). Its legacy memory pool, on a failed allocation, synchronizes the
  device, frees its cached buffers and drops every captured graph, which no capture in progress survives. The P106-090
  reports VMM: yes, and the VMM pool only grows and never unmaps, so every pool address inside a captured graph stays
  valid;
- concurrent streams (`GGML_CUDA_GRAPH_OPT=1`), whose pools and cuBLAS handles are made per stream;
- a graph with an `ARGSORT` over more than 1,024 columns and several rows. On that path CUB sorts with
  `DeviceSegmentedSort` in an eager launch but with `DeviceSegmentedRadixSort` while the stream captures
  (`argsort.cu`), so equal keys could come out in another order. One row (`DeviceRadixSort`) and up to 1,024 columns
  (the bitonic kernel) are the same either way, and so is every `TOP_K` path in this tree (CUDA 12.9's CCCL has no
  `DeviceTopK`).

**With the switch off** the backend runs the warm-up's branches exactly as before: the only base line the branch
changes is `if (!graph->warmup_complete) {`, which becomes `} else if (!graph->warmup_complete) {` after the forced
branch, and the forced branch is not entered. The flag in the context is never read.

## Can a captured graph replay stale inputs?

A replay launches the same kernels with the same arguments as the capture. It is exact when nothing a kernel depends
on has changed since without changing a recorded property. The review of this tree, by what changes between
micro-batches:

| What changes | How | Effect on a captured graph |
|---|---|---|
| Shapes: `n_kv` and the KV views, the micro-batch's tokens, the k-pool tables' lengths, the taper's sizes | node and source shapes | recorded: a new capture |
| Pointers: the scheduler's four input copies, the KV views' offsets | node and source data pointers | recorded: a new capture |
| A graph the scheduler hands over again unchanged (llama's graph reuse) | the split keeps its id, so its properties are not compared again | nothing moved: the scheduler did not re-split or re-allocate, and its input copy is the same |
| Input contents: tokens, positions, masks, the k-pool tables (`pool_cells`, `pool_idxs`, `pool_mask`, `tail_idxs`) | copied into the split's device input copy before its graph, on the card's stream, outside the capture | read when the kernels run, as in an eager launch. No CUDA kernel reads a host buffer (every source is in device memory), and no kernel argument is taken from a tensor's contents |
| Checkpoint copies (`--ctx-checkpoints-vram`'s device stores, the state save and restore) | tensor copies between graph evaluations, not graph nodes | queued on the card's stream before or after the graph launch, in the order they had around eager launches |
| Pool scratch | VMM pool: grows only; a capture's addresses stay mapped, and later users of the same range come after it in stream order | none |
| Host state an operation reads at launch | the norm's Q8_1 row (cleared per evaluation, a function of the graph), fusions (functions of the nodes), sample mode's phase sets (the capture generation), the persistent MoE kernel's check mode (graphs off), `ARGSORT`'s capture branch (excluded above), `MEAN`'s threshold (with graphs enabled on the card, the same eager and captured) | none |
| An executable updated while its previous launch runs (the next micro-batch is captured while the card still works on the last) | `cudaGraphExecUpdate` applies to later launches only; an executable destroyed while it runs is freed when it ends | none |
| The pipeline's ordering | the scheduler's event per card and input copy is recorded after the graph launch, as after the eager launches | none |

So a replay happens only where every recorded property equals the capture's, and the inputs that change without
changing a property are written before the graph runs, in stream order. Captured and eager launches then run the same
kernels on the same data in the same order and give the same bits; kernels whose result depends on timing (atomics)
are as nondeterministic either way.

## The CPU check

`run/checks.sh` on a build machine, CPU only. It builds the branch's rig tarball (`TESTS=1`, sm_61 code, no AVX),
then two tests from the same build tree, run in a container without a GPU against CUDA's stub driver:

- **`test-cuda-graph-force`:** the switch's choice of graphs on layers of 1, 4, 8, 9, 64, 128 and 512 tokens, each
  signal alone, a plain product, MLA's one-token queries, a one-token candidates' `MUL_MAT_ID`, and `ARGSORT` shapes
  on both sides of the CUB rule. Then the decision, in a model of `ggml_backend_cuda_graph_compute`'s whole warm-up and
  capture logic (the warm-up's branches as they are in `ggml-cuda.cu`, the forced branch through the header): a
  prompt's changing micro-batches, a reused graph, the context's first batched graph on either of two keys, decode
  graphs with the switch on against off, and 2,000 random runs of 400 evaluations each over one to three keys of one
  context, graphs the switch takes and leaves sharing keys, one evaluation in 17 that CUDA cannot capture, and sample
  mode's phase sets in a quarter of the runs. An executable is the properties it was captured from, and every replay
  must launch executables of the current properties. The control runs the port's decision (`692bbbc9a`) behind
  rule 1 in the same model and must be caught replaying stale executables.
- **`test-cuda-profile-light`:** the existing model of the warm-up's logic with sample mode's phase sets, unchanged
  by this branch.

### Results

Run on `2ec3a0927`, 2026-11-10 23:42 to 2026-11-11 01:09 (most of it waiting for the shared lock), CPU only:

| Step | Verdict | Detail |
|---|---|---|
| rig tarball | OK | `llama.cpp-20261110-2ec3a0927-pascal-p106-v4-b12246-c702.tar.gz`, 647,583,908 bytes, md5 `2bb77d3a2324f1842fcd5a6a0240f784`; sm_61, `TESTS=1`, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most |
| tests, built | OK | `test-cuda-graph-force` and `test-cuda-profile-light` from the rig build's tree |
| `test-cuda-graph-force` | OK | 6,070 checks, 0 failed |
| `test-cuda-profile-light` | OK | 49,267 checks, 0 failed |

`test-cuda-graph-force` in numbers:

| What | Result |
|---|---|
| choice of graphs | not taken at 1, 4 and 8 tokens, taken at 9, 64, 128 and 512, by each of the three signals alone; a plain product (a prompt's, or MLA's one-token queries) and a one-token candidates' `MUL_MAT_ID` never count; `ARGSORT` of 151,936 × 2 and of 2,048 × 1 × 3 keep a 512-token graph out, of 151,936 × 1, 1,024 × 512 and 8 × 512 do not |
| a prompt's eight changing micro-batches | switch off: eager, eager, ... (eight); on: eager, then seven captures |
| the port's failing case (two decode steps captured, then the context's first batched graph twice on the same key) | switch: eager, then a capture; the port's decision: eager, then a replay of the decode graph's executables (caught) |
| decode graphs, switch on against off | the same decision at every evaluation |
| 2,000 random runs × 400 evaluations, switch on | 200,993 captures, 465,116 replays, none of stale executables; at most one eager batched graph a run |
| the same runs, switch off | none of stale executables |
| the same runs, the port's decision (control) | stale executables replayed in 711 of 2,000 runs |

[summary.tsv](summary.tsv) holds the verdicts, [test-cuda-graph-force.log](test-cuda-graph-force.log) the test's
output.

## The rig row (Q1297)

On rig 1, where R42's driver and R43's scripts are, after Q1224 (R57, the `-ladder6` line's first rig check).
Est. 70 minutes (pad 90): the identity est. 20, R42's driver est. 48 (three loads an arm). Paths are variables:
`RIG` (the rig's ssh alias), `WORK` (a folder on the rig), `BUILDS` (the builds folder on the rig), `C702` (this
branch's tarball folder name in it), `BIN` (`$BUILDS/$C702/bin`), `MODELS` (the model folder on the rig), `R42`
(the rig folder with `r42_run.sh`), `R43` (R43's folder: `r43_run.sh`, `rig_taper.sh`, `taper_replay.py`,
`turns.csv`, `settings.env`), `RECORDED` (the release check's recording here), `OUT` (a results folder here). From a
machine that only runs ssh, at the repository's root:

```bash
# 1. identity, untimed: the v4 candidate (R70's five switches) eagerly and with the switch, greedy, 16 tokens a
#    request at depths 0 and 32,768 (the 32,768-token prime runs its micro-batches through captured graphs), every
#    verify step's drafted tokens and probabilities dumped; the control runs twice for its own spread
V4C_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5"
V4C_ARGS="--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ubatch-taper on --spec-draft-n-ubatch 512"
RIG=$RIG WORK=$WORK/lane BIN=$BIN/llama-server \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=$RECORDED/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 N_PREDICT=16 DEPTHS="0 32768" \
  ARMS="eager graphs eagerb" \
  ARM_ENV_eager="$V4C_ENV LLAMA_SPEC_DUMP=$WORK/lane/out/dump-eager.jsonl" ARM_ARGS_eager="$V4C_ARGS" \
  ARM_ENV_graphs="$V4C_ENV GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SPEC_DUMP=$WORK/lane/out/dump-graphs.jsonl" ARM_ARGS_graphs="$V4C_ARGS" \
  ARM_ENV_eagerb="$V4C_ENV LLAMA_SPEC_DUMP=$WORK/lane/out/dump-eagerb.jsonl" ARM_ARGS_eagerb="$V4C_ARGS" \
  LOADS="eager1:eager graphs1:graphs eager2:eagerb" \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK/lane bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK/lane OUT=$OUT/lane bash bench/deep_lane/remote.sh collect

# 2. R42's driver on this build, eagerly and with the switch: the unprofiled load (times), the graph profile (eager,
#    capture or replay per evaluation; windows) and the host profile (GPU time per micro-batch; blocked splits)
for arm in off on; do
  e=; [ $arm = on ] && e="GGML_CUDA_GRAPHS_FORCE=1"
  timeout 60 ssh $RIG "cd $R42 && env $e R43=$R43 BUILDS=$BUILDS BUILD=$C702 \
    MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
    DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=$WORK/r42-$arm R42_LOADS='warm4096 graph4096 host4096' \
    setsid nohup bash $R42/r42_run.sh > $WORK/r42-$arm.out 2>&1 < /dev/null &"
  ssh $RIG "while [ ! -e $WORK/r42-$arm/DONE ]; do sleep 30; done"
done
ssh $RIG "tar czf - -C $WORK r42-off r42-on" > $OUT/r42.tar.gz

# 3. on a build machine, CPU only: R42's tables for each arm
for arm in off on; do
  python3 -I docs/data/2026-11-10_prompt_timeline/run/timeline.py --load warm4096=r42-$arm/warm4096 \
    --load graph4096=r42-$arm/graph4096 --load host4096=r42-$arm/host4096 --pair graph4096=host4096 --out tables-$arm
done
```

What it decides:

- **Identity (step 1):** `deep_table.py`'s check must show the same greedy reply in all three loads at depth 0, and
  the dumps must give the same drafted tokens and probabilities in `graphs1` as in `eager1` wherever `eager1` and
  `eager2` agree. A difference stops the change there.
- **The graphs (step 2, `counts.tsv`):** with the switch on, every main-model prompt evaluation after each card's
  first is a capture or a replay, and generation's evaluations have the same eager, capture and replay counts as with
  it off (decode keeps the warm-up). The server log has the switch's info line.
- **The time (steps 2 and 3, `waits.tsv`, `busy.tsv`):** the 4,097-token prompt's window, the blocked splits' time
  (R42: 8,851.0 ms in 15 splits) and the cards' busy shares, off against on in the same build, each against R42's
  release-line loads of the same names. The window classes the switch as a gain, negligible or a loss against the
  spread of the two fresh prompts in each load. The host's lowest MemAvailable per load is in `status.txt`.

A gain goes into the v4 preset candidate (R70) and its agent check (R72); Q1298 checks the identity through the compute
6.1 PTX on a build machine's card first, if a card is free.

## Not checked

- Nothing ran on a GPU. Whether every kernel on the prompt path captures cleanly on a P106 (the first capture of a
  path also sets its kernels' shared-memory attributes, which they set at their first launch, inside the capture),
  and what a capture and an executable update per split per micro-batch cost the rig's 2-core host against the
  launches they replace, are the rig row's.
- A capture that fails stops the process with the CUDA error (as in the warm-up's captures); it does not fall back to
  an eager launch.
- The model of the decision is the test's: it takes `ggml-cuda.cu`'s warm-up branches as written, and the forced
  branch from the header the backend calls.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | the check's verdicts |
| [test-cuda-graph-force.log](test-cuda-graph-force.log) | the test's output |
| [run/checks.sh](run/checks.sh) | the check, on a build machine |

## How to run it

On a build machine with Docker, from a checkout of this repository, with the fork as remote `fork` of the work
folder's llama.cpp clone (`scripts/build_llamacpp_docker.sh`, its header):

```bash
WORK=<work dir> BUILD=scripts/build_llamacpp_docker.sh BRANCH=pascal-p106-v4-b12246-c702 JOBS=8 LOCK=<flock file> \
  setsid -f nohup bash docs/data/2026-11-11_c702_prompt_graphs/run/checks.sh > <work dir>/logs/driver.out 2>&1 < /dev/null
```

Verdicts land in `$WORK/summary.tsv`, the tests' output in `$WORK/tests/`, and `$WORK/C702_DONE` marks the end. The
test by hand, from any build of the branch with tests on:

```bash
<build>/bin/test-cuda-graph-force
```

## Provenance and licence

Code and tests are the fork's (MIT, as llama.cpp). The check reads no model and no data. Tables and scripts are this
repository's licence.
