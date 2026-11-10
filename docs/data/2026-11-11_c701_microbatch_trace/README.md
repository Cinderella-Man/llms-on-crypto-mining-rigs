# The per-micro-batch trace (C701, Q1238): `LLAMA_UBATCH_TRACE`

A diagnostic switch on the v4 line: with `LLAMA_UBATCH_TRACE=1` the host logs, for every micro-batch of a prompt,
whether its graph was rebuilt or reused and why, the host's time before the micro-batch's first split reaches a card,
each split's wait, copy and launch time, and every re-plan of the compute buffers with its trigger. It exists to explain
[R42's prompt timeline](../2026-11-10_prompt_timeline/README.md) (Q1156): 1.42 of the nine cards busy on average in a
fresh 4,097-token prompt, the host blocked inside each call's third queued micro-batch on cards 1 to 7, and a median
243.7 ms of host time before card 0 for a 512-token micro-batch. Registry row: [Q1238](../../../quants/experiments.md);
the rig row is Q1295.

- Fork branch `pascal-p106-v4-b12246-c701` at `769d4c274` (two commits on `pascal-p106-v4` `84f7bd3d2`: `7e642dfda`
  the trace, `769d4c274` the allocator's helpers moved before their first use), off by default. With the
  variable unset the engine logs nothing more and computes the same; the only change on that path is one `if` on a
  flag around each host wait of the scheduler and per split.
- After four commits of the [multi-GPU fork](https://github.com/lukolszewski/llama.cpp-multigpu) read in
  [the C663 note](../../research/notes/multigpu_fork_read.md): `1567041ef` and `c89c7fc89` (the trace), `c8832a5f0`
  (the allocator's trigger log) and `fa532f399` (the server's phase timing), written again for this tree. Its
  `reserved` field needs the fork's re-reserve, which this tree does not have; the reasons for a rebuild, the host's
  waits per split, the re-plans after a change of backends and the per-call line are new here.

**Result (2026-11-11, on a CPU):** the switch works and changes nothing while unset. On a random-weight model of
Flash-Next's architecture (qwen4exp), through llama-completion on the CPU, through llama-completion with the layers
on a second backend (an RPC server on the same machine's loopback) and through llama-server, the generated text is the
base branch's byte for byte with the switch unset and with it set; the unset runs log no trace line; the set runs log
a line for every decode call, micro-batch and scheduler evaluation, and llama-server its phase averages. The rig's
tarball builds. Two kinds of line did not occur in any CPU run, because nothing there plans a graph again: the graph
has the same nodes at every micro-batch size and the first reservation covers the largest, so `sched-replan` and
`galloc-trigger` are first seen on the rig, where R42 counted 34 re-plans of the MTP draft in one prompt. What the
lines say about R42's prompt is [the rig row](#the-rig-row)'s job.

## The lines

All lines are warnings of the engine's log (llama-server's log, or stderr of the other tools). Times are the host's
clock in microseconds. For one micro-batch they come in this order: the hybrid memory's reuse line if its input refused
the old graph (during the reuse check), the re-plan lines if the graph needs a new plan (during the allocation), then
`sched-trace` (while it is submitted), then `ubatch-trace`. A `decode-trace` line comes before the first micro-batch of
each decode call.

**`decode-trace: <dec|mtp> n_tokens = <n> n_outputs = <n> prep = <us> init = <us> wait = <us> us`**, one per decode
call (`mtp` for the MTP draft's context):

| Field | The host's time between |
|---|---|
| `prep` | the call's start and the memory's slot search: the batch's checks, the wait for pipelined decode steps still in flight, the micro-batch plan (the taper's, if on) |
| `init` | the memory's slot search for every micro-batch of the call (the hybrid memory's recurrent and attention parts) |
| `wait` | the output buffer, and the waits for the other context's rows: the MTP draft waits on the host for the target's hidden rows and layer input |

**`ubatch-trace: <dec|mtp|enc> n_tokens = <n> n_seqs = <n> n_outputs = <n> graph = <reused|rebuilt> [why = <reason>] pre build alloc inputs submit card0 total`**,
one per micro-batch:

| Field | Meaning |
|---|---|
| `graph` | `reused`: the previous graph with new inputs; `rebuilt`: built and allocated again |
| `why` | `in_flight`: pipeline parallelism and an earlier micro-batch of this context still computing (Miner Llama then rebuilds, so the new graph takes the scheduler's next input copy instead of waiting for the cards; `LLAMA_GRAPH_REUSE_WAIT=1` waits instead); `not_allocated`: another graph was allocated on this scheduler since; `disabled`: graph reuse is off (`LLAMA_GRAPH_REUSE_DISABLE`); `params`: the graph parameters differ (tokens, sequences, outputs, graph type); otherwise the type of the first graph input that refused the new micro-batch, for example `llm_graph_input_mem_hybrid` (then a line `ubatch-trace: mem_hybrid rejects reuse: ...` gives each of its checks as graph / new micro-batch: key indices, mask on the device, mask, state copies, main copies, head, `rs_z`, group) |
| `pre` | memory apply, the choice of scheduler (per-size schedulers for batches of up to 8 tokens), the draft's row cache, the graph parameters |
| `build` | the reuse check, then the graph's construction; for a reused graph, the wait for the previous evaluation |
| `alloc` | the scheduler's split of the graph and the allocation, a re-plan included (rebuilt graphs only) |
| `inputs` | the host writing the input tensors (tokens, positions, masks, state indices) |
| `submit` | the scheduler's evaluation, `sched-trace`'s `total`: every split's waits, copies and launches; CPU splits are computed here |
| `card0` | from the micro-batch's start until its first split off the CPU was launched: **the host's part before card 0**; `-1` when every split ran on the CPU |
| `total` | the whole micro-batch on the host |

**`sched-trace: copy = <c> splits = <n> wait = <us> card0 = <us> total = <us> us | <split> <split> ...`**, one per
evaluation of a scheduler, prompt and decode steps alike. `copy` is the input copy in use (pipeline parallelism
rotates four), `wait` the sum of the splits' `w`, `card0` the time from the evaluation's start to its first launch off
the CPU. Each split is `<backend>:i<inputs>:n<nodes>:w<us>:c<us>:l<us>`, a CPU split followed by `:<its first node>`
(for Flash-Next the token embedding's and the n-gram tables' rows):

| Part | The host's time before or in this split |
|---|---|
| `w` | blocked in the scheduler's waits: for the previous split when this one has no inputs, for the evaluation that last used this input copy (`GGML_SCHED_MAX_COPIES` evaluations earlier), for the source backend before a synchronous copy, for a prefetch |
| `c` | the rest before the launch: input copies (asynchronous ones queued, synchronous ones done), packed uploads, the copy callback for experts in host memory, the next split's prefetch |
| `l` | the launch: on a card, queueing its kernels, which blocks when the card's launch queue is full; on the CPU, the computation itself |

So a split that holds the host for a long time says where: in `w` it waits for an event or a card, in `l` for room in
the card's launch queue.

**`sched-replan: #<n> backends_changed = <0|1> [(node <i> '<name>' <from> -> <to>)] size_changed = <0|1> graph_size <a> -> <b> nodes = <n> leafs = <n> splits = <n> drain = <us> plan = <us> us`**:
every re-plan of a scheduler's compute buffers, numbered per scheduler. Each one first waits for every backend
(`drain`) and then plans again (`plan`). The release log (`sched_graph_realloc: graph reallocation`) leaves out the
re-plans after a change of backend assignments; this line has them all, and names the first node index whose backend
differs from the previous graph's.

**`galloc-trigger: ...`**: the allocator's reason for a new plan, printed before the re-plan it causes: `nodes <a> -> <b>`
or `leafs <a> -> <b>` (another graph shape), or `node|src|leaf <i> '<name>' [of '<reader>'] (<op>) ne = [...] needs <bytes> in <buffer type>, planned <bytes>`
(a tensor larger than its place in the plan). With a single buffer type (a CPU-only run) the allocator plans again by
itself and no `sched-replan` line follows.

**`galloc-grow: <buffer type> buffer <MiB> -> <MiB> MiB`**: a compute buffer that grows (`(size only)` when the plan
only measures).

**llama-server's phase timing:** with the switch at `1` the server prints, every 5 s, its phase averages since the last
print (`n_pre_decode`, `avg t_pre_decode`, `avg t_decode`, `avg t_post_decode`, `avg t_sampl`, `n_decode n_sampl`),
as the build-time `DEBUG_TIMINGS` did. At `2` it also waits for the cards after each decode so that `t_decode` holds
their time; that changes the schedule, so `2` is not for timed runs.

The cost with the switch on is the clock reads and a log line per micro-batch and per evaluation (a token step logs
the main model's and the draft's lines); it is a diagnostic for one load, not for speed numbers.

## The CPU check

Run on a 12-core build machine's CPU in containers without network, with
[run/cpu_check.sh](run/cpu_check.sh): static CPU builds (with the RPC backend) of the branch (`769d4c274`) and of the
commit it was cut from (`84f7bd3d2`); the model is a random-weight qwen4exp MoE model (492,130,240 bytes) written by
the branch's `test-llama-archs` with the tokenizer of the tree's `models/ggml-vocab-qwen35.gguf`. One prompt of a
sentence said 30 times (541 tokens), read with `-b 256 -ub 64` in three decode calls (256, 256 and 29 tokens: nine
micro-batches), then 16 greedy tokens (15 one-token decode calls).

| Arm | Text: base, unset, set (md5 of the output) | Trace lines unset | Trace lines set |
|---|---|---|---|
| llama-completion, CPU | `075f2419` in all three (95 bytes) | 0 (base 0) | 18 `decode-trace`, 24 `ubatch-trace` and 1 hybrid-memory line, 24 `sched-trace`, 1 `galloc-grow` |
| llama-completion, layers on an RPC server (`-ngl 99 --rpc`) | `075f2419` in all three | 0 (base 0) | the same counts, 2 `galloc-grow`; every evaluation has two splits (the CPU's token rows, then the RPC backend), every micro-batch a `card0` |
| llama-server, one `/completion` request (TIP only) | `0cf04996` unset and set | 0, no phase line | 20 `decode-trace`, 26 micro-batches and 2 hybrid-memory lines, 26 `sched-trace`, one block of phase averages |

What the set runs show, in the RPC arm's lines ([summary.tsv](summary.tsv), [counts.tsv](counts.tsv); the per-call and
per-micro-batch tables of [run/trace_table.py](run/trace_table.py) are in [table_rpc/](table_rpc/)):

- The first micro-batch is rebuilt (`why = not_allocated`: no graph yet). The second is rebuilt because the hybrid
  memory's input refuses the old graph, and its line says why: `rs_z 0/-1`, the first micro-batch zeroed a fresh
  recurrent state and the second has none to zero. Then the graph is reused (no pipeline parallelism here; on the rig
  the main model's prompt micro-batches are expected as `rebuilt why = in_flight`). Of the eight 64-token
  micro-batches 3 are reused and 5 rebuilt (4 `not_allocated`, 1 by the hybrid memory); the 29-token one is rebuilt
  for its new shape (`params`); of the 15 one-token steps 14 are reused.
- `card0` is the host's time before the RPC backend gets the micro-batch: a median 1,181.5 µs for the 64-token
  micro-batches (1,625 µs for the first, 703 of them in `alloc`) and 731 µs for the one-token steps. The CPU split
  computes the token rows in at most 25 µs (`l`).
- The second call's first micro-batch waits 12,556 µs in the RPC split's `w`: with one input copy (no pipeline
  parallelism) the host waits for the backend to finish the first call before it copies the new inputs. Over the run
  the RPC splits hold the host 19,213 µs in `w`, 10,850 µs in `c` and 969 µs in `l`.
- Each call's micro-batch with the output row is a graph of its own (`rebuilt why = not_allocated`); its first `alloc`
  takes 19,665 µs on the RPC backend, whose tensors are set up over the socket.

Not covered on a CPU: `sched-replan` and `galloc-trigger` (no run plans a graph again), a launch that blocks on a full
CUDA queue, and the cost of the lines on a 2-core host. `test-llama-archs` was not run against the base: the change
adds no operation and alters no graph.

## The rig row

Registry row Q1295: R42's driver ([run/r42_run.sh](../2026-11-10_prompt_timeline/run/r42_run.sh), R43's wrapper and
the release preset's line) with this branch's tarball, the trace on and one load, `warm4096` (two fresh 4,097-token
prompts with the micro-batch taper off and on, then the cut replay of one task). The rig's tarball is
`llama.cpp-20261110-769d4c274-pascal-p106-v4-b12246-c701.tar.gz` (647,620,810 bytes, md5
`17c75d11f585c762c78c4bd14aff6a27`). `$R42` is the folder that holds the driver, `$R43` R43's files (as in R42's
commands), `$BUILDS` the rig's build folder:

```bash
ssh $BUILDER "cat $WORK/llama.cpp-20261110-769d4c274-pascal-p106-v4-b12246-c701.tar.gz" \
  | ssh $RIG "cat > $BUILDS/c701.tar.gz && tar -xzf $BUILDS/c701.tar.gz -C $BUILDS && rm $BUILDS/c701.tar.gz"
ssh $RIG "cd $R42 && LLAMA_UBATCH_TRACE=1 R42_LOADS=warm4096 R43=$R43 BUILDS=$BUILDS \
  BUILD=llama.cpp-20261110-769d4c274-pascal-p106-v4-b12246-c701 \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=$R42/out-c701 \
  setsid nohup bash $R42/r42_run.sh > $R42/r42_c701.out 2>&1 < /dev/null &"
# when $R42/out-c701/DONE exists: the log off the rig, the tables on any machine with Python 3
python3 -I run/trace_table.py out-c701/warm4096/load-A/server.log tables
```

What to read, for the first 4,097-token prompt with the taper off (R42's request 7: calls of 2,048, 1,533, 512 and 4
tokens, then the draft's catch-up after each):
- each main-model micro-batch's `graph` and `why`, and its `card0` (R42: 231.8, 243.7, 237.0, 383.2 and 283.3 ms for
  the 512-token micro-batches inside a call) split into `pre`, `build`, `alloc`, `inputs` and the CPU split's `l`;
- the third micro-batch of the 2,048- and 1,533-token calls on cards 1 to 7, where R42 found the host blocked
  (8,851.0 ms of 29,983.0): time in `w` is a wait for an event or a card, time in `l` a full launch queue (the case for
  forced CUDA graphs, C702, and a larger queue, R62);
- the MTP draft's `sched-replan` lines and the `galloc-trigger` before each (R42 counted 34 re-plans a prompt): the
  trigger for C712; and any main-model re-plan, which would reopen C717.

The trace is a diagnostic: its numbers explain the window, the prompt's speed is not compared with R42's (another
engine line, b12246, and a log line per evaluation).

## Files

| File | What it is |
|---|---|
| [run/cpu_check.sh](run/cpu_check.sh) | the CPU check: two CPU builds, the model, the llama-completion, RPC and llama-server runs, the verdicts |
| [run/trace_table.py](run/trace_table.py) | tables from a log's trace lines: per decode call, per micro-batch, per split, per re-plan, and a summary |
| [summary.tsv](summary.tsv) | the CPU check's verdicts |
| [counts.tsv](counts.tsv) | the trace lines of each kind in each run |
| [table_rpc/](table_rpc/) | `trace_table.py` on the traced RPC run: `calls.tsv`, `ubatch.tsv`, `splits.tsv`, `replan.tsv` (empty), `summary.txt` |

## Commands

On a build machine with Docker, the fork as the remote `fork` of `$WORK/src`. The rig's tarball (CUDA 12.9, sm_61,
with `test-backend-ops`):

```bash
WORK=$WORK TESTS=1 JOBS=8 DOCKER_ARGS="--cpu-shares 256" nice -n 19 \
  scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c701
```

The CPU check (CPU only, every run in a container without network; minutes with a warm ccache):

```bash
WORK=$WORK SRC=$WORK/src TIP=769d4c274 BASE=84f7bd3d2 JOBS=8 THREADS=4 \
  nice -n 19 bash docs/data/2026-11-11_c701_microbatch_trace/run/cpu_check.sh
```

The trace in any run: `LLAMA_UBATCH_TRACE=1 llama-server ...`, then `grep -E '(decode|ubatch|sched)-trace|sched-replan|galloc-' <log>`.

Provenance: the model of the CPU check is random weights written by the tree's own `test-llama-archs` with the
tokenizer of its `models/ggml-vocab-qwen35.gguf`; nothing was downloaded. The four commits it follows are in the
multi-GPU fork, MIT-licensed like llama.cpp.
