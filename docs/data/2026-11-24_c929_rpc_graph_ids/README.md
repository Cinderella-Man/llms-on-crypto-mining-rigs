# ggml's RPC client forgets the graphs its server dropped (C929, Q1667; 2026-11-24)

What this is: a fix to ggml's RPC client in Miner Llama, on the fork branch `pascal-p106-v5-rpc-graph-ids`, and its
checks on a build machine's processor. Some of the fork's correctness checks run two "cards" on one processor through
ggml's RPC devices (two `ggml-rpc-server` processes on the same host), to test multi-card paths without GPUs. C901's
check ([2026-11-21_c901_pipe_slot_plan](../2026-11-21_c901_pipe_slot_plan/README.md)) crashed there as soon as a cached
graph was reused, and had to turn graph reuse off. Registry row Q1667 in
[quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **The fork's CPU checks can now run on RPC devices with graph reuse on.** C901's server check, with reuse on, ran to
  the end on the fixed builds and gave the same tokens as with reuse off and as on one CPU, in every request. That held
  on `pascal-p106-v5` and on C901's own branch, with its switch on and off. Without the fix the same arm crashed in its
  first request, as C901's did.
- **Nothing changes without RPC.** ctest gives the same result for every test before and after the fix, except the new
  RPC test, which fails before and passes after. `test-llama-archs` writes the same 148 logits files, bit for bit.
- **Upstream has not fixed it.** The client code is the same on upstream's `master`, and no pull request touches it.

## The bug

ggml's RPC client sends a graph to its server whole once (`GRAPH_COMPUTE`, with the graph's id) and afterwards, when
llama.cpp reuses the same graph, only asks the server to run it again by its id (`GRAPH_RECOMPUTE`). Both sides keep
the ids of up to 1,024 graphs per device and clear their lists at the same point.

Since upstream's security fix [#24292](https://github.com/ggml-org/llama.cpp/pull/24292) (June 2026) the server
drops every graph it caches, on all of its devices, whenever a buffer is freed, because a cached graph may point into
the freed memory. It keeps the entries and rejects a `GRAPH_RECOMPUTE` for any of them ("graph with uid N not found"),
then closes the connection. That fix expected the client to fall back to sending the graph whole. But the client never
learned of the drop: it still held the ids, so the next reuse sent `GRAPH_RECOMPUTE`, the server closed the connection,
and the client aborted at its next command ("Remote RPC server crashed or returned malformed response").

A scheduler that grows a compute buffer frees the old one. So the crash comes whenever one scheduler re-plans on a
server while another scheduler's graphs are cached there and reused. Pipelined speculation (`--pipe-depth`) gives each
pipe slot its own scheduler, and on two devices the slots re-plan at their first change of device and as the context
grows: C901's check hit it at the first reused pipe-slot graph.

Upstream's `master` (2026-11-23, after `03813c56e`) still has the client as it was, and no open or merged pull
request changes it (searched 2026-11-24: the RPC pull requests and issues on graph reuse, recompute and buffer frees).

## The fix

`ggml/src/ggml-rpc/ggml-rpc.cpp`, client side only (fork commit `60c8ad788`):

- the client keeps each graph id with a flag, true while the server holds that graph;
- freeing a buffer (`rpc_dispatcher::free_buffer`) marks every id of that server dropped, as the server does, and
  queues the `FREE_BUFFER` while it holds the graph lock, so no `GRAPH_RECOMPUTE` decided before the marking can be
  queued after the free;
- a dropped graph is sent whole again, and the server stores it under its id again;
- dropped ids stay in the list, as they stay on the server, so both sides still clear their lists at the same point.
  A client that simply forgot the ids would count fewer than the server, and after 1,024 more graphs it would ask the
  server for a graph the server had cleared.

The server needs no change, and a server built before #24292 (which drops nothing) works with it too: the client then
only sends a few graphs whole that the server still had. Allocating a buffer needs nothing: the server drops nothing
then, a growing buffer is always freed before the new one is allocated, and a graph gets new ids whenever the scheduler
splits it again, which it does before every new allocation.

A new test, `test-rpc-graph-cache` (fork commit `3c8f51216`, the fix's parent; built and run by ctest when RPC is on),
caches a graph, frees another buffer on the same server and reuses the graph; then it fills both lists to 1,000
entries, drops them with a free, sends graphs until the server clears its list at 1,024, and reuses a graph sent before
that clear. Every step checks the result.

The fix has no switch: it only changes what the RPC client sends, and nothing without RPC. The rigs' builds leave RPC
off unless `scripts/build_llamacpp_docker.sh` is run with `RPC=1` (llama-server using other rigs' cards over the
network), and with `RPC=1` this crash could hit too, since graph reuse is on by default.

## The checks

On a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU, each build under the
machine's build lock with `-j 8` at a low CPU share, from a compiler cache copied from C901's folder. Three trees:

| Tree | Commit | What |
|---|---|---|
| base | `3c8f51216` | `pascal-p106-v5` at `8a9629942` plus the new test, the client as it was |
| head | `60c8ad788` | base plus the fix |
| c901 | `19048bb6d` + both commits' patches | C901's branch (its pipe-slot re-plan count and `LLAMA_PIPE_SLOT_PLAN`), the tree C901's crash came from, fixed |

1. [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) on base and head with `CMAKE_ARGS=-DGGML_RPC=ON`
   (new option), static: a CPU build of every target, ctest without network (with the three RPC tests, which a build
   without RPC does not have), `test-llama-archs -b CPU -s 1` (plain, write, compare) and head's 148 logits files
   against base's.
2. llama-server, `ggml-rpc-server` and `test-llama-archs` of each tree, with the same options.
3. [run/c929_server.sh](run/c929_server.sh), C901's server check with arms that keep graph reuse on: llama-server
   `--pipe-depth 4` on a random-weight Flint model, Flint's CPU-side preset switches and the trace, the same four
   requests in each arm (Q1 700 plain tokens with the server's sampling, Q2 the same with temperature 0.7 and top-k 8,
   Q3 a tool call under a JSON schema, Q4 Q1 again), on one CPU or on two RPC devices. Each RPC arm starts two fresh
   servers with `GGML_RPC_DEBUG=2`, whose logs list every graph sent whole, every graph run by its id, every buffer
   allocated and freed, and every id rejected. All three builds run on the same model (head's `test-llama-archs`,
   seed 1).

The whole run took 6 min 32 s (07:26:50 to 07:33:22): the CPU test set 4 min 8 s, the three server builds 37 s, the
server checks 1 min 45 s.

### Builds, ctest and logits ([checks.tsv](checks.tsv), [ctest.tsv](ctest.tsv))

| Check | base | head |
|---|---|---|
| CPU build, every target, RPC on, static | exit 0, no errors | exit 0, no errors, the same warning lines (none in the changed files) |
| ctest, 78 tests (the 75 of a build without RPC and the three RPC tests) | 75 pass | 76 pass |
| `test-rpc-graph-cache` | **fails** | **passes** |
| `test-kv-cache`, `test-kv-cache-scales` | fail | fail (as on every tree in C933's runs of `8a9629942`) |
| every other test | pass | pass, test by test the same |
| `test-llama-archs -b CPU -s 1`: plain, write, compare | 153 of 153 pass each time; 148 logits files | the same |
| head's logits against base's | — | **148 compared, none differ** |

On the base the new test's client stopped before its buffered output was written, so ctest shows none for it; the
server check below shows the same crash with its logs.

### The server check ([server_tokens.tsv](server_tokens.tsv), [server_rpc.tsv](server_rpc.tsv))

| Build | Arm | All four replies | Same tokens as one CPU | Graphs sent whole / run by id (server 1, server 2) | Sent whole again after a drop | Ids rejected |
|---|---|---|---|---|---|---|
| head | cpu_off | yes | (reference) | — | — | — |
| head | rpc_noreuse_off | yes | yes, all four | 2,372 / 0, 2,372 / 0 | 0 | 0 |
| head | **rpc_reuse_off** | **yes** | **yes, all four** | 149 / 2,223, 158 / 2,214 | 0, **9** | **0** |
| base | rpc_noreuse_off | yes | yes, all four | 2,372 / 0, 2,372 / 0 | 0 | 0 |
| base | **rpc_reuse_off** | **no: llama-server aborted in Q1** | no replies | 19 / 2, 19 / 2 | 0 | **1** |
| c901 | cpu_off, cpu_on | yes | yes, all four | — | — | — |
| c901 | rpc_noreuse_off, rpc_noreuse_on | yes | yes, all four | 2,372 / 0, 2,372 / 0 | 0 | 0 |
| c901 | **rpc_reuse_off, rpc_reuse_on** | **yes** | **yes, all four** | 149 / 2,223, 158 / 2,214 | 0, **9** | **0** |

The tokens are also C901's own ([its server_tokens.tsv](../2026-11-21_c901_pipe_slot_plan/server_tokens.tsv)):
Q1 `13046c93f53f`, Q2 `42e3f829214e`, Q3 `42bf0f14000f`, Q4 `7e0c62ba61d5`.

- **The crash, on the base:** the ten pipe slots re-planned at their first change of device (ten `sched-replan`
  lines), the second server freed 11 buffers on the way, and the next reused graph was rejected
  (`graph with uid 116 not found`); llama-server stopped 454 ms into Q1 with "Remote RPC server crashed or returned
  malformed response", C901's message.
- **With the fix:** the servers ran 2,223 and 2,214 of their 2,372 graphs each by their id, and sent 9 graphs
  whole again after a drop, each a graph the base would have crashed on. The two servers freed 15 and 23 buffers per
  arm, as with reuse off. With reuse off every graph goes whole (2,372 per server).
- **C901's switch:** on C901's branch `LLAMA_PIPE_SLOT_PLAN=1` cut the trace's re-plans from 120 to 30 with reuse on,
  as C901 measured with reuse off, and the RPC servers' counts were the same either way.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | `cpu_tests.sh`'s verdicts for base and head (tree, build, ctest, `test-llama-archs`, head's logits against base's) and the three server builds |
| [ctest.tsv](ctest.tsv) | ctest test by test, base and head |
| [server_tokens.tsv](server_tokens.tsv) | per build, arm and request: tokens, prompt tokens, the md5 of the token list, whether it is head's one-CPU reply |
| [server_rpc.tsv](server_rpc.tsv) | per build and arm: llama-server alive after the requests, its aborts, the trace's re-plans and buffer growths, and per RPC server the graphs sent whole (of them sent whole again under a known id), run by their id, buffers allocated and freed, ids rejected |
| [run/run.sh](run/run.sh) | the driver: sources, the CPU test set, the three server builds, the server checks, ctest's table |
| [run/c929_server.sh](run/c929_server.sh) | the server check (C901's, with arms that keep graph reuse on and the RPC servers' command counts) |
| [run/c929_tables.py](run/c929_tables.py) | the four tables from the driver's work folder |

## Commands

```bash
# on the build machine: W holds everything, REPO (inside W) the three scripts
W=$WORK SEED=<a llama.cpp clone> REPO=$WORK/repo CPU_CCACHE=<a compiler cache> LOCK=<the build lock> \
  bash $WORK/repo/run.sh
# the tables, from a copy of $WORK without sources, builds and models
python3 -I docs/data/2026-11-24_c929_rpc_graph_ids/run/c929_tables.py docs/data/2026-11-24_c929_rpc_graph_ids $COPY
```

The builds (each tree's `git archive`, Docker image Ubuntu 24.04 with gcc 13.3.0 and cmake 3.28.3):

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DGGML_RPC=ON
cmake --build build -j 8 -- -k                    # cpu_tests.sh: every target
cmake --build build -j 8 --target llama-server ggml-rpc-server test-llama-archs   # the server builds
```

The server check on one build, in a container without network (its image also has `curl` and `jq`):

```bash
BIN=build/bin VOCAB=src/models/ggml-vocab-gpt-2.gguf OUT=$OUT ARMS="cpu_off rpc_noreuse_off rpc_reuse_off" \
  bash docs/data/2026-11-24_c929_rpc_graph_ids/run/c929_server.sh
```

Provenance and licence: the model is random weights from the fork's `test-llama-archs` (seed 1) with llama.cpp's
GPT-2 test vocabulary; the prompts are the check's own. Everything here is MIT, as llama.cpp.
