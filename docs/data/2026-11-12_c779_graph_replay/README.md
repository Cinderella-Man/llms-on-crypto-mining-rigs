# Forced prompt graphs patched and replayed, not captured anew (C779, Q1368, 2026-11-12)

Code, CPU checks and the rig tarball of `GGML_CUDA_GRAPHS_PATCH`, on the fork branch `pascal-p106-v4-b12246-c779`
(`771cafa97`, five commits on `pascal-p106-v4` `e27947287`), merged into `pascal-p106-v4` as `b9a1c1fa7`. Off by
default; with it unset, nothing changes. Its P106 check is R136: in part 1 (2026-11-13) the patch load gave the force
load's replies and verify steps, but it patched none of its captures ([the rig check](#the-rig-check-r136)). Registry
row: [Q1368](../../../quants/experiments.md).

## The problem

`GGML_CUDA_GRAPHS_FORCE=1` (C702) captures every graph of more than 8 tokens as a CUDA graph whenever its properties
change. Under pipeline parallelism a prompt micro-batch's split changes every micro-batch: the KV views, the attention
mask and the k-pool tables grow with the cache, and the scheduler rotates each split's inputs through its four copies.
So every split is captured again and its executable updated, never replayed: in a fresh 4,097-token prompt R82 counted
621 captures and 0 replays, and the splits' own host time grew by 195.3 ms (376.1 to 571.4 ms, est. the captures and
the updates) ([R82](../2026-11-11_r82_c702_p106/README.md)). Yet almost every node of a split keeps its properties from
one micro-batch to the next, and so do its kernels' parameters.

## The change

The row offered two ways: keep each split's properties the same from micro-batch to micro-batch (inputs at fixed
addresses, the cache length a kernel argument), or patch only the changed kernel parameters. The first needs the
attention, mask and k-pool kernels to read the cache length from memory (new GPU code) and an extra copy per split
input, or a cache length fixed for a whole call, which makes the QSA indexer score every padded position. The second
changes host code only, in one place, and falls back to today's full capture whenever anything does not match. It is
the one built here.

| Switch | What it does |
|---|---|
| `GGML_CUDA_GRAPHS_PATCH` unset (default) | nothing changes |
| `GGML_CUDA_GRAPHS_PATCH=1` with `GGML_CUDA_GRAPHS_FORCE=1` | a forced graph that would be captured again is patched when it can be, and replays |
| `GGML_CUDA_GRAPHS_PATCH_WINDOW` (32) | a group is dispatched again when a changed node lies within this many nodes after its first node |
| `GGML_CUDA_GRAPHS_PATCH_MAX` (100) | the most nodes, in percent of the graph's, dispatched again; above it the graph is captured in full. At 100 that never happens: in patch mode a full capture instantiates its executables anew, which can cost more than a patch of most nodes |

How it works, in `ggml_backend_cuda_graph_compute` and `ggml/src/ggml-cuda/graph-patch.h`:

1. **A full capture records.** While the switch is on, each forced capture notes every dispatch group of the
   evaluation loop (one node, or several the loop fuses into one launch): its nodes, the stream pool's use before it,
   and the capture chain's last node after it (`cudaStreamGetCaptureInfo`). When a chunk's capture ends, its graph is
   instantiated anew and kept, and its chain of CUDA nodes is cut into the groups that made it. A capture that is not
   one chain, a group that starts with pool memory held by another, or a persistent MoE layer leaves nothing recorded.
2. **The next evaluation compares.** The property check that decides a capture also notes which nodes changed, and
   whether only their parameters did: the node count, and every node's op, type, flags and sources, must be the same.
3. **The groups near a change are dispatched again.** Those holding a changed node, and those starting up to
   `_WINDOW` nodes before one (a fusion decision reads the nodes after its first, so a change there could fuse the
   graph differently), are dispatched in order into one scratch capture. Each must start from the pool use it had,
   fuse exactly the nodes it fused and make CUDA nodes of the same kinds and number; the loop's skip decisions (views,
   no-ops, the gated delta net's state gathers) must stay as recorded.
4. **Their parameters are set in place.** Kernel nodes through the driver API (`cuGraphExecKernelNodeSetParams`,
   which also takes the kernels cuBLAS launches), memsets and copies through the runtime's
   `cudaGraphExec*NodeSetParams`. The executables then replay; a launch already queued is not affected by the update.
5. **Anything else captures in full,** as before, and records again: no recording yet, a changed structure, more than
   `_MAX` percent of the nodes to dispatch, a changed skip decision, another pool use, another fusion, other CUDA
   nodes, or an update CUDA refuses.

With a profile mode other than `graph` (which only times whole evaluations) nothing is patched; the graph profile
labels a patched evaluation `patch`. Every 64 decisions the process logs, as a warning so that llama-server prints it:
`GGML_CUDA_GRAPHS_PATCH: <n> of <m> captures patched (<g> groups, <k> ggml nodes, <c> CUDA nodes set); captured in
full: not_ready <n> structure <n> share <n> skip <n> pool <n> fuse <n> nodes <n> api <n>`.

## Results on a CPU

All on a build machine, CPU only, 2026-11-12: the branch tip `771cafa97` 19:31–20:01, the merged tree 20:02–20:04
([checks.md](checks.md)); nothing ran on a GPU.

| Check | Result |
|---|---|
| `ggml-cuda.cu` alone, compiled as the rig build compiles it (sm_61) | compiles, no warning |
| The rig tarball (`TESTS=1`; sm_61, no AVX) | built: `llama.cpp-20261112-771cafa97-pascal-p106-v4-b12246-c779.tar.gz`, 678,532,453 B, md5 `ceb3d5db5b748910e06e00f26211d2bd`; 0 AVX instructions, highest symbol GLIBC_2.38; the base build's 34 warning lines, none in the changed files |
| The switch list (C770) | 865 names; `GGML_CUDA_GRAPHS_PATCH`, `_WINDOW` and `_MAX` among them, the first in `bin/llama-server` |
| `llama-server`'s SASS against `cfc657de5`'s (C811's checked tarball; `e27947287` adds C803, host code only) | the same file: 8,540 functions |
| `llama-server`'s PTX against the same | 8,540 functions; 10 `flash_attn_tile` instances differ in 70 lines, every one the name of a function-local `__shared__` array: nvcc numbers internal names by the entities declared before them, and `common.cuh` now declares more ([run/ptx_names.sh](run/ptx_names.sh)). With that number masked as well, the two whole PTX dumps are the same file (md5 `80551b469e1173101aab20a1fb7ebee5`) |
| `test-cuda-graph-patch` | 21,013 checks, 0 failed, the same output in two runs. In the model, 2,000 random runs of 300 micro-batches: 70,068 evaluations patched (41.2 groups dispatched again each) and every executable after every evaluation the same as a fresh capture's; the full captures by reason: not_ready 6,669, structure 6,492, share 222,925 (the runs at a 60 % limit), pool 239,435, fuse 49,254, nodes 415. The control, a window of 2 nodes against fusions that read 5 ahead, made other executables than a fresh capture in 30,018 evaluations of 277 runs |
| `test-cuda-graph-force` (C702's, unchanged) | 6,070 checks, 0 failed |
| `test-llama-archs` against `cfc657de5`'s logits | 146 of 146 the same with no switch, with `GGML_CUDA_GRAPHS_FORCE=1`, with both switches, and with both at window 0 and a 100 % limit (a CPU build: the CUDA switches must change nothing there) |
| C565's set, static and shared | ctest 71 of 71 (C811's 70 and `test-cuda-graph-patch`), the network test 1 of 1, archs 146 written and compared, the nine hand tests pass |
| The merged `pascal-p106-v4` (`b9a1c1fa7`: `-c779` on C808's `a1347d9dd`, after C807, C812 and C809; it adds exactly `-c779`'s diff) | a static CPU build of every target; `test-cuda-graph-patch` 21,013 of 21,013; `test-llama-archs` 146 of 146 the same as `cfc657de5` with every switch off and with the force and patch switches |

The model test (`tests/test-cuda-graph-patch.cpp`) checks the three pure parts one by one (which groups a change
marks, the chain's cut into groups and its failures, the decision before a patch) and then runs a model of the
capture and the patch: a graph of nodes with a fixed topology whose properties change as a prompt's do (input readers
rotating through four copies, cache readers growing, now and then a changed op, another node count or an eager
evaluation), a dispatch loop that fuses patterns of up to six nodes on conditions that read the pattern's later nodes,
groups that make one to three CUDA nodes (one kind's count depends on a property), and a rare group that holds pool
memory into the next. It shows patches replaying without a GPU, and that the window must cover the longest fusion.

## The rig check (R136)

On nine P106-090 cards, with [run/r136_run.sh](run/r136_run.sh), against `GGML_CUDA_GRAPHS_FORCE=1` alone (in `v4c2`'s
line since R123a). Three parts, each one run:

1. **Identity, untimed** (R82's lane): the v4 candidate with the force switch, then with the patch, 16 tokens at
   depths 0 and 32,768 (the prime is 64 forced micro-batches), every verify step dumped. Passes when the patch load
   gives the force load's replies and the same dumps at every step (the patch is built to give exactly a fresh
   capture's executables).
2. **R42's driver** with the force switch, then with the patch: the fresh 4,097-token prompt's window and the splits'
   host time (R82: 571.4 ms with the force switch, 376.1 ms eagerly), classed against each load's two fresh prompts;
   R82's counts (`run/r82_counts.py`), in which a patched evaluation is `patch`; and the server log's
   `GGML_CUDA_GRAPHS_PATCH` lines, which tell how many groups each patch dispatched again and why any capture was full.
3. **R43's replay** of an agent session's 62 prompts on `v4c2`'s line (R123b's kept line), with the force switch and
   then with the patch: seconds a pass by size bucket, the same 8 greedy tokens on every turn.

Est. up to ~0.2 s a fresh 4,097-token prompt (the captures' and updates' host time), more if a patch dispatches few
groups. If part 2 shows most groups dispatched again at the 32-node window, `ARM=patchw` (window `W`, 8 by default)
runs parts 1 and 2 at a shorter window; it must keep part 1's dumps.

### Part 1 on a P106 (2026-11-13): the same replies and steps, but no capture patched

The `-c779` build (`771cafa97`) on nine P106-090 cards: the load `force1` 08:18:27 to 08:27:46, then `patch1` 08:27:46
to 08:37:01, each started on a quiet rig ([r136/lane/table.md](r136/lane/table.md)).

- **Identity holds.** The greedy reply is the same on every pass and load, at depth 0 and at 32,768. The patch load
  gives the force load's 40 verify steps with every field the same: the drafted tokens, their candidates and
  probabilities, the main model's targets and probabilities, the tokens kept
  ([r136/lane/dump_compare.tsv](r136/lane/dump_compare.tsv)). The two dumps differ only in `run`, the field that names
  the process. Neither load reallocated the target's graph.
- **But the patch load patched no capture, so part 1 did not test a patched graph.** Its last counts line reads 0 of
  768 captures patched, all 768 in the 32,768-token prime
  ([r136/lane/patch_counts.tsv](r136/lane/patch_counts.tsv)). Every capture was full, for one of three reasons:
  - **`structure`, 705 of 768.** The check compares a node's sources as tensor objects (`src[j]` and `view_src`), and
    they changed from one micro-batch to the next. The likely cause (est.) is the scheduler's four rotating input
    copies: each copy is a separate tensor. C779's CPU model treated the rotation as a change of properties.
  - **`api`, 62 of the 63 captures whose structure held.** CUDA refused a parameter update, after 1,302 CUDA nodes in
    all had been set. The counts line does not say which call it refused: kernel, memset or memcpy.
  - **`nodes`, 1:** the scratch capture made other CUDA nodes than the recording.
- **What it shows:** patch mode's full captures are exact on a P106, nothing more. Parts 2 and 3 would time patch mode
  with every capture full. [C826](../2026-11-13_c826_graph_patch_fix/README.md) (Q1449) compares sources by kind,
  leaves unchanged memsets and memcpys alone and names every refused update; R145 (Q1450) runs part 1 again on its
  tarball, and parts 2 and 3 follow on it.

## Commands

```bash
# the checks, on a build machine (CPU only; M a work folder, REPO a checkout of this repository, LOCK a flock file,
# BASE_TREE an unpacked rig tarball of pascal-p106-v4-b12246-c811 cfc657de5, BASE_CPU the bin/ of its static CPU
# build, SEED_SRC a llama.cpp clone with the fork as remote "fork")
M=$M REPO=$REPO LOCK=$LOCK BASE_TREE=$BASE_TREE BASE_CPU=$BASE_CPU SEED_SRC=$SEED_SRC JOBS=8 \
    bash $REPO/docs/data/2026-11-12_c779_graph_replay/run/checks.sh
# the PTX with internal names' numbers masked
NEW=<the new tarball>/bin/llama-server BASE=$BASE_TREE/bin/llama-server OUT=$M/sass/ptxnames LOCK=$LOCK \
    bash $REPO/docs/data/2026-11-12_c779_graph_replay/run/ptx_names.sh
# after the merge into pascal-p106-v4: the merged tree's "every switch off" check
M=$M ... STEPS=merged bash $REPO/docs/data/2026-11-12_c779_graph_replay/run/checks.sh

# the rig check, from a machine that only runs ssh, at this repository's root (RIG, WORK, BUILDS, BUILD, MODELS, R42,
# R43 and RECORDED as in the script's header; BUILD=llama.cpp-20261112-771cafa97-pascal-p106-v4-b12246-c779)
STEP=lane bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh
STEP=lanewait bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh
STEP=lanecollect OUT=<here>/lane bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh
STEP=r42 ARM=force bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh    # ends with $WORK/r42-force/DONE
STEP=r42 ARM=patch bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh    # after r42-force/DONE
STEP=agent ARM=force bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh  # ends with $R43/r136_force/DONE
STEP=agent ARM=patch bash docs/data/2026-11-12_c779_graph_replay/run/r136_run.sh

# part 1's reading, on a build machine's CPU, in a folder that holds the collected lane/
python3 -I $REPO/docs/data/2026-11-11_r82_c702_p106/run/dump_compare.py --control lane/dump-force.jsonl \
  --control2 lane/dump-force.jsonl --arm lane/dump-patch.jsonl --out lane/dump_compare.tsv
python3 -I $REPO/docs/data/2026-11-12_c779_graph_replay/run/patch_counts.py lane/server-patch1.log > lane/patch_counts.tsv
```

The lane's greedy replies and its other checks are `deep_table.py`'s, in `table.md`; `requests.csv` holds each reply's
md5.

Part 1's dumps compare with R82's [dump_compare.py](../2026-11-11_r82_c702_p106/run/dump_compare.py) (the force load
as both controls), part 2's tables with R42's
[timeline.py](../2026-11-10_prompt_timeline/run/timeline.py) and R82's
[r82_counts.py](../2026-11-11_r82_c702_p106/run/r82_counts.py), part 3's with R123a's
[r123a_ladder.py](../2026-11-12_r123a_prompt_ladder/run/r123a_ladder.py), all on a build machine's CPU.

## Files

| File | What it holds |
|---|---|
| [checks.md](checks.md) | every check's line from the build machine (`summary.tsv`) |
| [run/checks.sh](run/checks.sh) | the checks: `ggml-cuda.cu` alone, the rig tarball, the switch list, `llama-server`'s GPU code against the base's, a static CPU build, `test-cuda-graph-patch`, `test-llama-archs` against the base's logits, C565's ctest set static and shared, the merged tree |
| [run/ptx_names.sh](run/ptx_names.sh) | the PTX comparison with internal names' numbers masked |
| [run/r136_run.sh](run/r136_run.sh) | the rig check R136 (from a machine that runs ssh) |
| [r136/lane/table.md](r136/lane/table.md), [classes.tsv](r136/lane/classes.tsv), [requests.csv](r136/lane/requests.csv), [loads.tsv](r136/lane/loads.tsv) | R136 part 1, the lane: `deep_table.py`'s tables and checks, every request, each load's line |
| [r136/lane/dump_compare.tsv](r136/lane/dump_compare.tsv) | the 40 verify steps: the patch load against the force load (as both controls), and which fields differ |
| [r136/lane/patch_counts.tsv](r136/lane/patch_counts.tsv) | the patch load's counts lines: decisions, captures patched, CUDA nodes set, full captures by reason |
| [run/patch_counts.py](run/patch_counts.py) | the counts lines of a server log as a TSV |

## Limits

- **Not yet patched on a P106.** The CPU checks show that nothing changes with the switch unset or on a CPU, that the
  device code is the base's, and that the patch logic gives a fresh capture's executables in a model. R136's part 1
  shows patch mode exact on a P106 with every capture full: none of 768 was patched (`structure` 705, `api` 62,
  `nodes` 1). Whether CUDA's driver accepts the updates, how many groups a real split dispatches again, and what it
  saves wait for C826 and R136's parts 2 and 3.
- **The window is a bound, not a measurement.** 32 nodes covers the fusions the loop tries on a prompt graph; a fusion
  that reads further ahead than the window, conditioned on a property that changes between micro-batches, would keep
  the old partition while a fresh capture fuses differently: the same nodes computed by other kernels. R136's part 1
  compares every verify step's dump with the force load's, which tests the window only once captures are patched.
- **A full capture costs more in patch mode:** it instantiates each chunk anew (the patch needs the graph its
  executable was instantiated from), where the force switch alone updates the executable. It happens once a graph,
  and after any fallback.
- **The counts line is a warning** every 64 decisions, so that llama-server prints it at its default verbosity.

## Provenance and licence

Code and checks of 2026-11-12, on the Miner Llama fork of llama.cpp (MIT). The CPU checks ran on a build machine with no
GPU in use. R136 ran on a P106 rig on 2026-11-13. The recorded agent text that its lane cuts prompts from, the dumps
(which hold token ids of that text) and the server logs are not published; its tables hold counts, times, ids and
md5s only. No model output is in this folder. Licence: the repository's.
