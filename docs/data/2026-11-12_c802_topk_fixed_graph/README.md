# C691's top-k over the output rows with one graph for every micro-batch (C802, Q1406, 2026-11-12)

[FN84](../2026-11-12_fn84_graph_realloc/README.md) found why v4's `gains` line plans the main model's compute buffers
again on every request. The cause is `LLAMA_SAMPLING_TOP_K_ROWS=1` (C691: one top-k over all the output rows). With
it, the logits' padding row was built only in micro-batches without outputs, so the graph had one node more in a
prompt's middle (6,944 on Flash-Next) than at its end, in a decode step or in a verify pass (6,943). Each change
between the two made ggml's allocator plan the buffers again. Before each new plan the scheduler waits for all nine
cards, and the new plan fits no larger micro-batch. One 3,313-token turn took 42.49 s with the switch and 21.14 s
without it.

This folder holds the fix, why the first idea for it (a view of the logits in the pad's place) does not work, and the
checks on a build machine's CPU. The registry row is [Q1406](../../../quants/experiments.md); the switch is in
[docs/switches.md](../../switches.md). On nine P106 cards (the rig check R127, below) the switch with the fix
reallocates the main model's graph 0 times in 8 requests of v4's `gains` line, with R60's replies and checkpoints, and
the 3,313-token turn takes 21.14 s (without the switch 21.14 s, with it before the fix 42.49 s; one request each).

**Result:** with the fix, the switch keeps one graph in every micro-batch, and the allocator never plans again. A test
model with Flint's 150,144-token vocabulary ran 13 decodes: prompt middles, prompt ends, decode steps and verify
passes of up to four outputs.

| Build, switch on | Nodes a graph | Re-plans in 13 decodes | Compute buffer |
|---|---|---:|---:|
| before the fix (`d097d6d88` with the test) | 481 without outputs, 480 with | 8 | 2.35 MiB |
| a view of the logits in the pad's place (the first idea) | 481 in all | 8 | 2.35 MiB |
| **the fix** (`8a55934f3`) | **482 in all** | **0** | **2.92 MiB** |
| the switch off (control, every build) | 490 in all | 0 | 5.21 MiB |

The sampled tokens and candidates are the same with the switch off and on (17 tokens, 340 candidates). Per verify pass
the fix computes nothing more than C691 did: the pad is in the graph but never computed when the micro-batch has
outputs. The compute buffer holds at most one row of the vocabulary more than before the fix: 0.57 MiB here, and est.
0.95 MiB for Flash-Next's 248,320 columns, on the card that holds the output layer. The switch off pads all the output
rows plus one.

The rest of the CPU set passes as well:

- every switch off, and the switch on, gives `d097d6d88`'s logits on all 146 of `test-llama-archs`' models;
- ctest passes 70 of 70, with static and with shared libraries;
- C691's switch test and its recorded set pass (`tk` 39 of 39, `tk-rows` 75 of 75);
- with the switch off the graphs are `d097d6d88`'s, node for node.

The branch is merged into `pascal-p106-v4` (`c96561876`), off by default.

## The change

Branch `pascal-p106-v4-b12246-c802` of the fork, on `pascal-p106-v4` at `d097d6d88`:

| Commit | What |
|---|---|
| `884163683` | `test_top_k_rows_fixed_graph` in `tests/test-backend-sampler.cpp` (below); on this commit it fails with the switch on |
| `1444b0237` | the first form of the fix: the whole logits padded in every micro-batch, the pad not computed with outputs |
| `2b76bb706` | `ggml-alloc.c`: the allocator's re-plan trace (`LLAMA_UBATCH_TRACE=1`, C701) prints a place the plan gave no memory instead of crashing |
| `8a55934f3` | the fix as it stands: the pad is one row of padding of a view of none of the logits' rows |

Once every check below had passed, the branch was merged into `pascal-p106-v4` with `git merge --no-ff`, as
`c96561876`. It merged on top of C805's merge `d569b8bc2`, which had landed meanwhile and has no file in common with
this branch, so the merge adds exactly the branch's diff.

**How the allocator decides.** When a context is made, ggml's allocator plans the compute buffers from a graph with
outputs: where each node's memory goes, and how much. Each later graph is checked against that plan, node by node, in
order. The plan holds if the graph has the same number of nodes, and each node's tensor fits the memory planned for its
place. A view (a node that only reshapes or slices another tensor) takes no memory of its own, so it always fits. A node
that needs memory does not fit a place where the plan put a view. Otherwise the allocator plans again from the graph in
hand. On several cards the scheduler first waits for every card. The new plan is sized for that graph only, so the next
larger micro-batch does not fit it either.

**What C691 did, and the fix.** Without the switch, every sampling chain reads a padded copy of the logits: the padding
row gives a chain without an output row of its own a valid input. C691 dropped the copy where the micro-batch has outputs
and kept it where it has none, so the graphs of a prompt's middle had one node more. The fix builds the pad in every
micro-batch, before the chains, from a view of none of the logits' rows, so it is always one row:

- without outputs it is the chains' input, as before;
- with outputs the chains read the logits themselves, as C691 does, and nothing reads the pad;
- it is added without the compute flag (`ggml_build_forward_order`), so it is computed in neither case: no chain is
  selected in a micro-batch without outputs, and nothing reads it in one with outputs.

So every micro-batch has the same nodes in the same order, and the one-row pad has memory in every plan, the one made
when the context is made among them. The switch off builds the graph as before.

**Why a view in the pad's place is not enough.** The first idea (Q1406) was a view of all the logits' rows where the
pad was, when the micro-batch has outputs. That keeps the node count, but the context's plan is made with outputs, so
it holds a view at that place. The first micro-batch without outputs brings the pad there, which needs memory. The
allocator's trace names it: `node 341 'node_341' (PAD) ne = [150144, 1, 1, 1] needs 600576 bytes, planned none`. Its
new plan is sized for that micro-batch, and the next larger one does not fit it: 8 re-plans in the 13 decodes, as many
as before the fix. [run/view.patch](run/view.patch) is that idea on the fix's tree, and the checks run it as the `view`
arm.

**The pad's size.** The first form of the fix (`1444b0237`) padded the whole logits. The plan then held the output rows
plus one, which is the switch off's compute buffer (5.21 MiB on the test model against 2.35 before the fix). C691 had
saved that memory. The pad of a view of none of the rows is one row in every micro-batch, the size of the pad of a
micro-batch without outputs, so the plan holds that one row: 2.92 MiB.

**The trace's crash.** With `LLAMA_UBATCH_TRACE=1` (C701), the allocator's `galloc-trigger` line read the plan's
buffer for the tensor that does not fit. At a place where the plan gave no memory, that buffer's index is -1, and the
process crashed instead of printing (the first run of the `view` arm, exit 139). Such a place now prints the bytes the
tensor needs and `planned none`. Without the trace nothing changes.

## The test

`test_top_k_rows_fixed_graph` (in `tests/test-backend-sampler.cpp`) makes a context per setting, with the switch off
and on. Each context has one sequence, the server's sampling chain at the presets' settings (a top-k first), micro-batches
of 8 tokens and up to 4 outputs. The test then runs 13 decodes:

| Decode | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tokens | 8 | 8 | 5 | 1 | 4 | 3 | 8 | 2 | 6 | 1 | 8 | 8 | 30 |
| outputs | 0 | 0 | 1 | 1 | 4 | 3 | 0 | 0 | 2 | 1 | 4 | 0 | 1 |

The last decode is split into four micro-batches (8, 8, 8 and 6 tokens). The test counts:

- **each graph's nodes**, with the scheduler's eval callback (asked once for every node): one count in all 12 decodes
  of one micro-batch, and four times that in the last;
- **the allocator's re-plans**, from its trace lines (`galloc-trigger`, `LLAMA_UBATCH_TRACE=1`): none. The allocator
  reads the trace setting once a process. The test sets it, so it counts re-plans when it runs first in its process
  (`--test test_top_k_rows_fixed_graph`); otherwise it says so and checks the node counts only;
- **the sampled tokens and candidates of every output**, which must be the same with the switch off and on.

The switch off is the control and must pass the same checks. The model is a tiny Flint with random weights: 6 layers
of 256, Flint's 150,144-token vocabulary, 167 MB, made for Q111's tests. On a CPU, the allocator plans again without
the scheduler (one buffer), so the scheduler's own count (`graph reallocs`) and `GGML_SCHED_DEBUG_REALLOC` see nothing
there. The allocator's trace is the CPU's count.

## The checks

[run/checks.sh](run/checks.sh) ran every step on a build machine's CPU on 2026-11-12 from 15:06 to 15:34. Each step ran in a
Docker container without GPUs at `--cpu-shares=256` and `nice 19`, one build or test at a time behind the machine's
shared lock. The arms are static CPU builds:

- `test`: `884163683`, the code before the fix with the test;
- `view`: the tip with [run/view.patch](run/view.patch);
- `tip`: `8a55934f3`.

Every verdict line is in [summary.tsv](summary.tsv).

| Check | What ran | Result |
|---|---|---|
| the test, before the fix | `test_top_k_rows_fixed_graph` alone, `LLAMA_UBATCH_TRACE=1`, arm `test` | fails as expected: the switch on has 481 nodes without outputs and 480 with, 8 re-plans; the switch off 490 in all, 0 |
| the test, the view | the same, arm `view` | fails as expected: 481 nodes in all, 8 re-plans; the first is the pad where the plan holds the view |
| the test, the fix | the same, arm `tip` | passes: 482 nodes in all, 0 re-plans; the switch off 490, 0; the same 17 tokens and 340 candidates off and on |
| the rig build | `scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c802`, `TESTS=1` | builds (below); 0 AVX instructions, GLIBC_2.38 at most; its 34 warning lines are the `-kern` build's 34 |
| `test-backend-sampler` | every test in a process of its own on the tiny Flint ([sampler_tests.tsv](sampler_tests.tsv)): the tip with the switch unset, the tip with it on for the whole process, and arm `test` (`d097d6d88`'s code) | 22 of 23 pass on the tip with the switch unset and on, 21 of 23 on `test`. `test_logit_bias` fails in all three on the same token: it expects a +10 bias to win, which it does not on random weights (as in C691). On `test` the fixed-graph test fails too, as it must |
| C691's switch test | `test_top_k_rows_switch`, in the step above | passes with the switch unset and on: 5 pairs of chains, 17 outputs each, the same bytes off and on |
| the graphs | `test_multi_output_cpu_suffix` with `GGML_SCHED_DEBUG=2`: arm `test` and the tip with the switch off, the tip with it on ([graph_nodes.tsv](graph_nodes.tsv); the scheduler does not print views) | off: the same 4,491 node lines on both builds, 18 `PAD`, all computed. On: 4,473 lines, 18 `TOP_K` against 27, 171 `GET_ROWS` against 180, 18 `PAD` and none computed (C691: 4,455 lines and no `PAD`) |
| every switch off | `test-llama-archs -b CPU -s 1`: `d097d6d88`'s static CPU build writes every architecture's logits, the tip compares | 146 of 146 models the same bit for bit, with no switch and with `LLAMA_SAMPLING_TOP_K_ROWS=1` (151 of 151 tests pass) |
| C691's recorded set | `top-k-rows` replayed on the CPU with the rig build's `test-backend-ops` ([cpu_replay_cases.tsv](cpu_replay_cases.tsv)) | `tk`: 39 of 39 `TOP_K` cases at 248,320 columns; `tk-rows`: 75 of 75 `TOP_K_ROWS` cases; none missing |
| every switch off | [C565's CPU test set](../2026-11-05_c565_ladder4fix/README.md) on the tip, static and then shared libraries | both build with 0 error lines; ctest 70 of 70, `test-arg-parser` with network, `test-llama-archs` written and compared (146 files, none differ), the nine hand tests, in each |

The first run of the chain checked the first form of the fix (`1444b0237`): 481 nodes in all and 0 re-plans, but
its graph (the same in the second run) planned a 5.21 MiB compute buffer. The second run also had the trace fix (`2b76bb706`), and its `view` arm printed the 8
re-plans. Each verdict line of those two runs is in [summary_earlier.tsv](summary_earlier.tsv).

The two-card run with `GGML_SCHED_DEBUG_REALLOC=1` that Q1406 lists did not run: the build machine's cards were in use.
R127's load on nine P106 cards counts the scheduler's reallocations directly, so it stands in for that run.

## The rig check (R127)

FN84's load ([run/fn84_loads.sh](../2026-11-12_fn84_graph_realloc/run/fn84_loads.sh)) with nothing dropped
(`DROP=`), on this branch's rig tarball. That is R60's whole `gains` line, `LLAMA_SAMPLING_TOP_K_ROWS=1` included: one
task of R43's replay with the taper off, then one fresh 2,049-token prompt with the taper off and then on. The server
runs at `-lv 5`.

The rig tarball is `llama.cpp-20261112-8a55934f3-pascal-p106-v4-b12246-c802.tar.gz`: 678,285,106 bytes, md5
`ee37cb87c2211a1b90c3e19b70b9fbe4`. It was built with `TESTS=1` in the CUDA 12.9 image
`nvidia/cuda:12.9.2-devel-ubuntu24.04` for sm_61. It has 0 AVX instructions and needs GLIBC_2.38 at most, and its
`test-backend-ops` has md5 `3d4b837dd154e8754ad4a9f7824bd6b7`. It is the branch alone, on `d097d6d88`, so the load
compares with FN84's and R60's on the same code. `pascal-p106-v4` also holds C805's merge, which changes only CUDA
kernels behind its own switch.

**Result: on nine P106-090 cards the main model's graph reallocates 0 times in the 8 requests, and the 6 replies and
the 6 requests' context checkpoints are R60's.** R60's load (the same text, the switch before the fix) had 26
reallocations in those 6 requests. The server's `switches:` line lists 14 set, `LLAMA_SAMPLING_TOP_K_ROWS=1` among them
(FN84's load, without the switch: 13). So the switch keeps its decoding gain (R83: −0.72 ms a step, −0.6 %) without
the prompt cost FN84 found.

| Request | New tokens | Reallocations (same / changed): R60, switch before the fix | FN84, switch off | This load, switch with the fix | Reply and checkpoints as R60's | Prompt ms: R60 / FN84 / this load |
|---|---:|---:|---:|---:|---|---:|
| prefix (unmeasured) | 1,549 | 3 / 1 | 0 / 0 | **0 / 0** | yes, yes | 23,418.2 / 22,239.0 / 22,489.5 |
| turn 1 | 896 | 2 / 2 | 0 / 0 | **0 / 0** | yes, yes | 11,809.7 / 11,528.5 / 11,508.7 |
| turn 2 | 3,313 | 7 / 2 | 0 / 0 | **0 / 0** | yes, yes | 42,492.2 / 21,139.9 / 21,135.7 |
| turn 3 | 302 | 1 / 2 | 0 / 0 | **0 / 0** | yes, yes | 4,883.1 / 4,776.7 / 4,785.0 |
| turn 4 | 30 | 1 / 2 | 0 / 0 | **0 / 0** | yes, yes | 2,360.1 / 2,253.5 / 2,265.3 |
| turn 5 | 92 | 1 / 2 | 0 / 0 | **0 / 0** | yes, yes | 3,400.7 / 3,308.0 / 3,311.4 |
| fresh prompt, taper off | 2,049 | — | 0 / 0 | **0 / 0** | FN84's reply and checkpoints | — / 11,718.3 / 11,652.7 |
| fresh prompt, taper on | 2,049 | — | 0 / 0 | **0 / 0** | FN84's reply and checkpoints | — / 11,416.4 / 11,487.7 |

- **Prompt times, one request each** (a diagnostic; this load was not run for timing): the 3,313-token turn took
  21,135.7 ms, against 42,492.2 ms with the switch before the fix (R60) and 21,139.9 ms without the switch (FN84), the
  same text on the same rig. The fresh 2,049-token prompt took 11,652.7 ms with the taper off and 11,487.7 ms with it
  on. The timing on the line is R43's replay.
- **The MTP draft's own graph** (213 nodes) reallocates 25 times in the 8 requests (3, 2, 7, 1, 1, 1, 5, 5), the same
  per request as in FN84's load. That is older and separate (Q1407).
- **Memory:** after loading, each of the nine cards holds the same MiB as in FN84's load (`nvidia-smi`, MiB for MiB).
  The lowest host `MemAvailable` over the load was 431 MiB (FN84: 449 MiB).
- **What this load does not measure:** the switch's time in a verify pass. That is R83's; with the fix a verify pass
  computes the same nodes as before it (the pad is in the graph but not computed, the graph check above).

The load ran detached from 16:45:10 to 16:52:11 (up after 314 s), and nothing else ran on the rig meanwhile. The rig
scripts are R60's and FN84's, unchanged (`rig_taper.sh` `e82a80981c9e88e6a235cc476b993de8`, `taper_replay.py`
`329df66616395a0162065530a0f6108e`, `turns.csv` `a3c7dc91660a507cd415694d47bf7ced`); this build's `llama-server` has
md5 `61fb0827bdc0715fc0396153ea0e4501`. The server line and replay are FN84's ("Commands" in its folder), with
`LLAMA_SAMPLING_TOP_K_ROWS=1` in the environment.

```bash
BUILD=llama.cpp-20261112-8a55934f3-pascal-p106-v4-b12246-c802
RIG=$RIG WORK=$WORK BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT NAME=c802 DROP= \
  bash docs/data/2026-11-12_fn84_graph_realloc/run/fn84_loads.sh
```

The tables, on a build machine's CPU, from the server logs and request files of R60's load `a`, FN84's load and this
one (from this folder):

```bash
F=../2026-11-12_fn84_graph_realloc/run
python3 -I $F/fn84_requests.py r60a=<R60 load a>/load-A/server.log:6 topk=<FN84's load>/load-A/server.log \
  c802=<this load>/load-A/server.log > r127_requests_reallocs.tsv
python3 -I ../2026-11-12_r60_taper_v4/run/realloc_counts.py c802=<this load>/load-A/server.log > r127_reallocs.tsv
cp <this load>/load-A/requests.tsv r127_requests.tsv
python3 -I run/r127_compare.py r127_requests_reallocs.tsv r60a=<R60 load a>/load-A/requests.tsv:6 \
  topk=<FN84's load>/load-A/requests.tsv c802=<this load>/load-A/requests.tsv > r127_compare.tsv
```

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the build machine's chain: the three CPU arms, the fixed-graph test, the rig tarball, the sampler set, the graphs, `test-llama-archs`, the replay of C691's set, C565's CPU test set |
| [run/view.patch](run/view.patch) | the first idea (a view of all the logits' rows in the pad's place), on the fix's `src/llama-graph.cpp` |
| [summary.tsv](summary.tsv) | every verdict line of the run that checked `8a55934f3` |
| [summary_earlier.tsv](summary_earlier.tsv) | the verdict lines of the two earlier runs (`1444b0237`, `2b76bb706`) |
| [fixed_decodes.tsv](fixed_decodes.tsv) | the fixed-graph test's 13 decodes per arm and setting: tokens, outputs, nodes, the allocator's re-plans so far |
| [fixed_triggers.txt](fixed_triggers.txt) | the allocator's re-plan lines (`galloc-trigger`) of arms `test` and `view` |
| [sampler_tests.tsv](sampler_tests.tsv) | `test-backend-sampler`, each test's exit code: the tip with the switch unset (`off`) and on, and arm `test` (`base`) |
| [graph_nodes.tsv](graph_nodes.tsv) | the node lines of `test_multi_output_cpu_suffix`'s graphs: `TOP_K`, `PAD` (and computed `PAD`), `GET_ROWS` |
| [cpu_replay_cases.tsv](cpu_replay_cases.tsv) | C691's set replayed on the CPU: 114 cases and their verdicts |
| [r127_compare.tsv](r127_compare.tsv) | the rig check per request: prompt ms, the main model's reallocations (same / changed) and their node counts, the draft's, the reply's and the checkpoints' md5, in R60's load `a`, FN84's load and this load; whether each reply and set of checkpoints is R60's |
| [r127_requests_reallocs.tsv](r127_requests_reallocs.tsv) | FN84's `fn84_requests.py` output for the three loads |
| [r127_reallocs.tsv](r127_reallocs.tsv) | this load's reallocations per prompt-size bucket (R60's `realloc_counts.py`) |
| [r127_requests.tsv](r127_requests.tsv) | every request of this load as `taper_replay.py` wrote it |
| [run/r127_compare.py](run/r127_compare.py) | joins `fn84_requests.py`'s output with each load's request file, request by request (FN84's `compare.tsv` for any number of loads) |

## Commands

On the build machine, with this repository at `$REPO`, a llama.cpp clone with the fork as remote `fork` at `SEED_SRC`,
the static CPU build of `d097d6d88` (with `test-llama-archs`) at `BASE_CPU`, and C691's recorded set `top-k-rows` at
`SET` ([2026-11-11_c691_sampling_launches](../2026-11-11_c691_sampling_launches/README.md)):

```bash
M=<work folder> REPO=$REPO LOCK=<flock file> MODEL=<tiny Flint GGUF> SET=<top-k-rows folder> BASE_CPU=<d097d6d88 CPU bin/> \
SEED_SRC=<clone> JOBS=8 THREADS=8 setsid nohup bash docs/data/2026-11-12_c802_topk_fixed_graph/run/checks.sh > chain.out 2>&1 &
```

The fixed-graph test alone, on any CPU build of the branch:

```bash
LLAMA_UBATCH_TRACE=1 build/bin/test-backend-sampler --device cpu --test test_top_k_rows_fixed_graph <model.gguf>
```

## Provenance and licence

Code: the fork's branch `pascal-p106-v4-b12246-c802` (MIT, as llama.cpp). `test-backend-sampler` ran on a tiny Flint
with random weights (6 layers, 256-dimensional, Flint's 150,144-token vocabulary; made for Q111's tests). The operator
checks use `test-backend-ops`' random tensors, and `test-llama-archs` its random-weight models. No trained weights or
user data are involved. FN84's and R60's numbers quoted above are from their folders.

The rig check was measured on one Manli M-P106L9-N6G mining rig (nine P106-090 cards, 4 GB host) on 2026-11-12, and
its tables were made on a build machine's CPU. Its model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with
the MTP draft, as the v3 release lists them. The prompts' text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's
([turns.csv](../2026-11-03_retype_agent/agent/turns.csv)). The reference columns are R60's load `a` and FN84's load.
