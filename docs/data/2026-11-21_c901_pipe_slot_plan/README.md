# Flint's pipelined slots planned again over the whole memory, and counted (C901, Q1629, 2026-11-21)

What this is: an engine change, an agent-check change and their checks on a CPU. With pipelined speculation
(`--pipe-depth 4` in Flint's preset) each token is computed in one of 10 pipe slots, and each slot has a scheduler of
its own. In one traced agent run of Flint Pascal Q8 GGUF ([R181](../2026-11-19_r181_flint_realloc/README.md), 18.0
task-minutes) those schedulers re-planned their memory 390 times: 10 at the first pipelined tokens of the load, all
after a change of card against the plan made at load, and 380 after that, each time the context outgrew the plan of
the moment. Every re-plan first waits for all nine cards. Together they waited 7.12 s and planned 1.07 s (est. at most
8.2 s of the run), and the agent check counted none of them: the server's `graph reallocs` count covers the context's
own scheduler only. Registry row [Q1629](../../../quants/experiments.md).

The change (fork branch `pascal-p106-v5-pipe-slot-plan`, one commit `19048bb6d` on `pascal-p106-v5` `3dfad876f`;
`src/llama-context.cpp`, `src/llama-context.h`, `include/llama.h`, `tools/server/server-context.cpp`):

- **Counted:** each pipe slot's scheduler now reports its re-plans in submitted steps, after no change of device and
  after one. `llama_pipe_replans()` returns them, and the server logs one line per request after `graph reallocs`
  when the context has pipe slots:
  `pipe reallocs = S / D (same devices) / (changed devices), F plans over the whole memory` (all since the load).
- **Planned again:** with the new switch [`LLAMA_PIPE_SLOT_PLAN=1`](../../switches.md#sampling-drafts-and-speculation),
  a slot that re-plans in a step is planned once more over the whole memory, with that step's own samplers (the graph
  `pipe_init` reserves at load, a token over the whole cache), and the step's graph is then allocated on that plan.
  A later step then fits the plan however long the context grows, so the 380 later re-plans should not happen. If the
  step's graph still does not fit the new plan (it sits on other devices than the whole-memory graph, or needs more
  room), the switch is given up for that slot with a warning, and the slot re-plans as before. Each whole-memory plan
  logs one warning line.
- Unset, empty or `0`: the slots plan as before. The one difference is the new log line, which appears only with pipe
  slots.

The task first proposed another fix, planning the slots at load with the sampler's random input where a live step puts
it; why the switch plans from the live step instead is under the result.

## Result (CPU only)

**In plain words:** the change builds, the test suite fails only where its base fails, and every request gives the
same tokens with the switch off and on. One CPU cannot show the problem: with a single device the plan made at load
holds every pipelined token, so nothing re-plans either way. To see it on a CPU, the check also ran the model split
over two RPC devices (ggml's network backend, here two local server processes, each a CPU with its own buffers). There
the slots behave as on the rig: at the first pipelined tokens all 10 re-planned after a change of device, and then
again each time the context grew by a step. Without the switch that made 120 re-plans in four requests. With it, 30:
one per slot at the first pipelined tokens and one per slot in each later request whose sampling settings differed
from the request before, each followed by a plan over the whole memory that held. The agent check's count now reads all of them from the server log. Whether this wins back
R181's est. 8.2 s on nine P106 cards is for the rig run below (est., at most: the 380 later re-plans' drains and
plans, 8.08 s, less the new plans).

| Check | Result |
|---|---|
| CPU build, every target, tests on ([builds.tsv](builds.tsv)) | rc 0, 106 s (from the compiler cache) |
| Compiler warnings | 25 distinct, the same set as `3dfad876f`'s |
| ctest | 71 of 82 pass; test by test the base's results (the 75 tests the base's check ran: the same 71 pass and the same 4 fail; the other 7 download, need the network or need what they download) |
| CPU build with RPC on (`-DGGML_RPC=ON`: `llama-server`, `ggml-rpc-server`, `test-llama-archs`) | rc 0 |
| Rig build (`TESTS=1`, `HF_UI_VERSION=b12483`) | `llama.cpp-20261121-19048bb6d-pascal-p106-v5-pipe-slot-plan.tar.gz`, 707,076,642 bytes, md5 `8d2239e8a0d5921a013171b651777a34`; its `llama-server` md5 `7207345912c9ae9e13b3d916b2698403`; no AVX instruction in its binaries |

The 11 failures are the base's own: three tests that download (and the three that need their downloads, Not Run),
`test-conversion-fp8` and `test-kv-cache-scales` (Python packages the image lacks), two `test-jinja` cases that read
files through a path the compiler cache made relative, and `test-kv-cache` (QUEUES code row C881).

**llama-server with pipelined speculation** ([server_tokens.tsv](server_tokens.tsv),
[server_replans.tsv](server_replans.tsv), [server_trace.tsv](server_trace.tsv)): a random model of Flint's
architecture, Flint's sampling (temperature 1.0, top-p 0.95, top-k 20), `--pipe-depth 4` (10 pipe slots), Flint's
preset switches that apply on a CPU and the trace (`LLAMA_UBATCH_TRACE=1`). Four requests: Q1 700 tokens of plain text
with the server's own sampling, Q2 the same with other settings (temperature 0.7, top-k 8), Q3 a tool call under a JSON
schema (the pipe stays off under a grammar), Q4 Q1 again. The plain replies cross the cache's 256-cell steps three
times.

| Arm | Devices | `LLAMA_PIPE_SLOT_PLAN` | Same tokens as `cpu_off` | Pipe-slot re-plans, Q1 / Q2 / Q3 / Q4 | after a change of device | Plans over the whole memory |
|---|---|---|---|---|---:|---:|
| cpu_off | one CPU | unset | (reference) | 0 / 0 / 0 / 0 | 0 | 0 |
| cpu_on | one CPU | `1` | yes, all four | 0 / 0 / 0 / 0 | 0 | 0 |
| rpc_off | two RPC devices | unset | yes, all four | 40 / 40 / 0 / 40 | 10 | 0 |
| rpc_on | two RPC devices | `1` | yes, all four | 10 / 10 / 0 / 10 | 10 | 30, all held |

In `rpc_off` each slot re-planned at its first token after a change of device (the trace: `graph_size 168 -> 216`, and the
first node it names, by position, a sampler input, `logit_bias`), then once at each of the three cache steps the reply
crossed (the attention mask outgrew the plan of the moment), and again in Q2 and Q4. In `rpc_on` each slot re-planned
once in Q1 (the change of device) and once in Q2 and in Q4 (their sampling settings differ from the request before,
so the sampler part of the graph differs), each time followed by a plan over the whole memory for that request's
samplers. All 30 plans held their step's graph (the warning says "the step's graph fits it"; 7.548 ms for the 30
together), and no slot re-planned as the context grew. The RPC arms ran with graph reuse off
(`LLAMA_GRAPH_REUSE_DISABLE=1`): ggml's RPC server drops the graphs it caches when one of its buffers is freed, which a
re-plan that grows a buffer does, and its client still asks it to run a cached graph again by its id, so the first run
with reuse on stopped at the first reused pipe-slot graph, with and without the switch (QUEUES code row C929, registry
Q1667). With reuse off a graph is sent whole at every step; the slots still re-plan at the same points, since a graph
is rebuilt anyway whenever the context grows by a step.

The agent check's own count of these logs ([agent_check_count.txt](agent_check_count.txt),
[agent_check_requests.tsv](agent_check_requests.tsv)): `rpc_off` "pipe slots' re-plans 120 in 3 of 4 requests (110
same devices, 10 changed devices; 0 plans over the whole memory)", `rpc_on` "30 in 3 of 4 requests (20 same devices,
10 changed devices; 30 plans over the whole memory)". The server's own `graph reallocs` count stayed 0 in every arm.

**Why not plan at load, as the task first proposed:** the load plans each slot with the sampler chain the server holds
then, and each request brings its own chain. A live step's graph did not fit the load's plan: on two RPC devices it was
48 entries larger (`graph_size 168 -> 216`), on R181's nine cards 40 larger (`3352 -> 3392`). R181 named the sampler's
random input because the trace names the first node whose device differs by its position in the graph, so after the
first difference every later node is compared with another one. A plan made from the step's own samplers held every
time here, whatever the request's settings, and a request with other settings costs one more plan per slot.

**What a CPU cannot check:** the cards' drains and the time a re-plan costs on nine P106 cards, the memory of a
whole-memory plan on 6 GB cards, and the real agent run's chains. The rig run (registry Q1666, QUEUES rig 2 table row
2c, on rig 1) is R181's traced run with the switch:

```bash
D=docs/data/2026-11-21_c901_pipe_slot_plan
RIG=rig CLIENT=local MODEL=$MODEL BIN=$BIN RIG_LOGDIR=$WORK/agent-logs \
  RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 CLIENT_DIR=$CLIENT_DIR CLIENT_OUT=$CLIENT_OUT OUT=$OUT \
  PREFIX=acr-c928 ARM=f5spt bash $D/run/c901_rig.sh
python3 -I docs/data/2026-11-19_r181_flint_realloc/run/r181_replans.py $D $CLIENT_OUT/rig/server-f5spt.log
```

and, optionally, its control on the same build (`ARM=f5s0t EXTRA=LLAMA_PIPE_SLOT_PLAN=0`). R181's table of causes then
shows the pipelined slots' rows, and the agent check's report their count.

## Files

| File | What |
|---|---|
| [builds.tsv](builds.tsv) | the builds: commit, flags, exit code, seconds, warnings, ctest |
| [server_tokens.tsv](server_tokens.tsv) | per arm and request: tokens generated, prompt tokens, the md5 of the token list |
| [server_replans.tsv](server_replans.tsv) | per arm and request: the pipe's guesses, the pipe slots' re-plans (same devices, changed devices), the plans over the whole memory, the context's own graph reallocs |
| [server_trace.tsv](server_trace.tsv) | per arm: the trace's re-plan lines (all, after a change of device), their drain and plan time, the whole-memory plans by outcome and their time |
| [agent_check_count.txt](agent_check_count.txt), [agent_check_requests.tsv](agent_check_requests.tsv) | the agent check's count of the four server logs (`scripts/realloc_counts.py --line` and `--requests`, and `report.py`'s phrase) |
| [run/c901_tables.py](run/c901_tables.py) | the three server tables from the check's output |
| [run/c901_server.sh](run/c901_server.sh) | the server check: a random Flint model, the switch unset and `=1`, on one CPU and on two RPC devices |
| [run/c901_rig.sh](run/c901_rig.sh) | the P106 run for the rig check (R181's traced run with `LLAMA_PIPE_SLOT_PLAN=1`, or `=0` for the control) |

## Commands

The fork's tree (`git archive` of `19048bb6d`) was built in Docker on a build machine's CPU (image: Ubuntu 24.04,
gcc 13.3.0, cmake 3.28.3), every target, static, tests on, and a second time with RPC on for the server check:

```bash
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build -j 8 -- -k
cd build && ctest -j 8 --timeout 3600 --output-on-failure
cmake -S src -B build-rpc -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DGGML_RPC=ON \
  -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build-rpc -j 8 --target llama-server ggml-rpc-server test-llama-archs
```

Then, in a container with no network (its image also has `curl` and `jq`), and the tables:

```bash
BIN=$WORK/build/bin RPCBIN=$WORK/build-rpc/bin VOCAB=$WORK/src/models/ggml-vocab-gpt-2.gguf OUT=$WORK/server \
  bash docs/data/2026-11-21_c901_pipe_slot_plan/run/c901_server.sh
D=docs/data/2026-11-21_c901_pipe_slot_plan S=$WORK/server/out
python3 -I $D/run/c901_tables.py $D $S
python3 -I scripts/realloc_counts.py --line cpu_off=$S/cpu_off.server.log cpu_on=$S/cpu_on.server.log \
  rpc_off=$S/rpc_off.server.log rpc_on=$S/rpc_on.server.log > $D/agent_check_count.txt
python3 -I scripts/realloc_counts.py --requests cpu_off=$S/cpu_off.server.log cpu_on=$S/cpu_on.server.log \
  rpc_off=$S/rpc_off.server.log rpc_on=$S/rpc_on.server.log > $D/agent_check_requests.tsv
```

(`agent_check_count.txt` also holds three lines of `bench/agent_check/report.py`'s phrase for the same logs, as its
`gonogo` and `compare` reports print it.)

The rig's build: `TESTS=1 HF_UI_VERSION=b12483 scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-pipe-slot-plan`.
The change is host code only: it changes no CUDA source.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `pascal-p106-v5-pipe-slot-plan` `19048bb6d` on `pascal-p106-v5` `3dfad876f`.
The model is the fork's `test-llama-archs` random-weight model of Flint's architecture (seed 1), with GPT-2's
vocabulary file from the llama.cpp tree; no released weights were used. The tables hold counts, times and hashes, no
text. Built and run on a build machine's CPU on 2026-11-21, no GPU, in Docker with no network.
