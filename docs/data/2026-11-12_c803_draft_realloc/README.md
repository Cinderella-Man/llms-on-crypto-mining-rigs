# The MTP draft's graph reallocations (C803, Q1407, 2026-11-12)

[FN84](../2026-11-12_fn84_graph_realloc/README.md) found the main model's graph planning its compute buffers again on
every request, and [C802](../2026-11-12_c802_topk_fixed_graph/README.md) fixed that. FN84 also saw a second, older
pattern: the MTP draft's own graph (213 nodes, 3 splits, on card 8 with its token table on card 4) is planned again 2
to 5 times a request on every line, R85's v4 night merge too. Before each new plan the scheduler waits for the
draft's cards. This folder finds where in a request those re-plans fall, what causes them, and holds the fix and its
checks on a build machine's CPU. The registry row is [Q1407](../../../quants/experiments.md); the switch is in
[docs/switches.md](../../switches.md). The check on a P106 is the rig row R130 below.

**Result:**
- **Where:** in the prompt only. Every one of the 2,842 counted re-plans in five rig loads falls in a request's
  prompt phase, before its first token; none falls in generation. Every request has at least one, a 30-token turn too,
  and longer prompts have more.
- **Why:** the draft head's shortlist (`LLAMA_DRAFT_SHORTLIST`, Q592), which every rig line sets. A drafting pass
  with one output builds the shortlist head, ten nodes more than the full head of the catch-up and the prompt passes.
  That change of node count makes the context drop the plan it made at its start over the whole memory. From then on
  every plan is sized to the graph that made it, and the next larger graph needs a new one: in the prompt, the QSA
  layer's mask over the cells, which grows with the context; in generation, the catch-up's token rows and the drafting
  pass's hidden row. On a CPU, the shortlist alone gives the draft 104 re-plans in FN84's eight requests, and the
  k-pool switch, the indexer rows switch and a 512-token draft micro-batch give 0 each.
- **With `LLAMA_MTP_GRAPH_KEEP=1`** (C619, in `v4c2`'s line), each draft shape of up to 8 tokens has a scheduler of
  its own, so the shapes stop replacing each other's plan. But each such scheduler is planned from its first graph and
  re-plans, with a larger compute buffer, every time the context grows past it: 26 re-plans and 19 buffer growths in
  the same eight requests.
- **The fix, `LLAMA_SCHED_PER_SIZE_RESERVE=1`** (off by default; branch `pascal-p106-v4-b12246-c803`): a per-size
  scheduler is planned once, when it is made, from its batch's graph over the whole memory, with the shortlist's
  whole room and the k-pool's lists at the batch's own bounds, the way the shared scheduler is planned at the start.
  Nothing computed changes.

On a build machine's CPU, with both switches the draft re-plans 0 times in FN84's requests on R60's and R85's
lines (104 and 128 with neither switch, 26 and 26 with `LLAMA_MTP_GRAPH_KEEP=1` alone), with the same replies; the new
`test-llama-archs --mtp-replan` gives 0 re-plans and the same drafts and draft states over six turns of a growing
context; every switch off, the 146 test models give `c96561876`'s logits, and ctest passes 70 of 70 static and
shared. The cost is memory planned up front, not computation. Merged into `pascal-p106-v4` as `d21820340`. What it
saves on a P106 is est. small: R81 timed about 0.18 ms of planning a re-plan there (6.2 ms a fresh 4,097-token prompt),
plus the drains once the draft's passes queue ahead.

## Where the re-plans fall: five rig loads

`run/draft_reallocs.py` reads llama-server's `-lv 5` log. Each re-plan whose card assignment stayed the same logs
`graph reallocation: nodes = N (changed = C)`, for the main model's context and the draft's alike; the draft's graph
has a few hundred nodes and the main model's thousands. A request's prompt phase ends at its first sampled token.

| Load | Line | Draft micro-batch | Requests | Draft re-plans | In the prompt | In generation | Re-plans a request (count of requests) |
|---|---|---:|---:|---:|---:|---:|---|
| FN84 | R60's `gains` line without `LLAMA_SAMPLING_TOP_K_ROWS` | 512 | 8 | 25 | 25 | 0 | 1: 3, 2: 1, 3: 1, 5 or more: 3 |
| R60 `a` | R90's `gains` line | 512 | 373 | 617 | 617 | 0 | 1: 254, 2: 55, 3: 31, 4: 15, 5 or more: 18 |
| R60 `ro8` | the same with `RO8` | 512 | 154 | 272 | 272 | 0 | 1: 101, 2: 22, 3: 13, 4: 6, 5 or more: 12 |
| R85 `c217v` | the night merge with checkpoints on the cards | 128 | 300 | 964 | 964 | 0 | 1: 116, 2: 44, 3: 71, 4: 12, 5 or more: 57 |
| R85 `c217i` | the same and checkpoints inside the call | 128 | 300 | 964 | 964 | 0 | the same as `c217v` |

Every draft re-plan is at an unchanged graph size (`changed = 0`): 213 nodes on the `-kern` build, 207 on the night
merge. By prompt size (R60 `a`): under 512 new tokens 190 re-plans in 190 requests, 512–1,023 144 in 104, 1,024–1,999
163 in 56, 2,000 and more 120 in 23. R85's draft micro-batch of 128 tokens gives more a request (512–1,023: 245 in 83).
Per request: [rig_logs/requests.tsv](rig_logs/requests.tsv); per load and size: [rig_logs/summary.tsv](rig_logs/summary.tsv).

The scheduler's log line leaves out every re-plan after which a node's card changed (`llama_context::sched_graph_realloc`
returns early), so a re-plan that changes the graph's node count on the cards may not show here. On the CPU below, the
shortlist also makes the draft re-plan in generation, at each change between a catch-up and a drafting pass, which
changes the node count; whether those happen on the cards is what the rig row's trace counts.

## Why: the allocator's trace on a CPU

`run/trace.sh` runs llama-server on a CPU with test-llama-archs's tiny Flash-Next model (3 layers, one MTP block, a
128-token vocabulary, a 65,536-token context) and the MTP block as a separate draft file, as Flash-Next ships it. It
sends FN84's requests (`run/draft_trace.py`: one conversation growing by 1,549, 896, 3,313, 302, 30 and 92 new
tokens, then two fresh 2,049-token prompts, 8 greedy tokens each) with `LLAMA_UBATCH_TRACE=1`. The allocator then logs
the first tensor of each graph that does not fit its plan. A CPU has one compute buffer, so the allocator plans again
by itself where the scheduler on several cards would wait for the cards first; the check that triggers it is the same.

| Arm (the `-kern` build, `d097d6d88`) | Draft micro-batches | Re-plans in the prompt | In generation | Buffer growths after load |
|---|---:|---:|---:|---:|
| R60's switches: `LLAMA_QSA_INDEXER_ROWS=1`, `LLAMA_QSA_KPOOL_DEV=1`, `LLAMA_DRAFT_SHORTLIST=100,8,2`, draft micro-batch 512 | 208 | 20 | 84 | 0 |
| no switch, draft micro-batch 128 | 267 | 0 | 0 | 0 |
| no switch, draft micro-batch 512 | 208 | 0 | 0 | 0 |
| `LLAMA_QSA_KPOOL_DEV=1` alone | 208 | 0 | 0 | 0 |
| `LLAMA_QSA_INDEXER_ROWS=1` alone | 208 | 0 | 0 | 0 |
| `LLAMA_DRAFT_SHORTLIST=100,8,2` alone | 208 | 20 | 84 | 0 |
| the shortlist and `LLAMA_MTP_GRAPH_KEEP=1` | 208 | 5 | 20 (and 1 before the first request) | 19 |
| R60's switches and `LLAMA_MTP_GRAPH_KEEP=1` | 208 | 5 | 20 (and 1 before the first request) | 19 |

Every arm gives the same replies (8 requests, 57 tokens). What each re-plan found (R60's switches, 104):

| Trigger | Count | What it is |
|---|---:|---|
| `nodes 176 -> 186` and `186 -> 176` | 22 and 22 | a drafting pass with the shortlist head after a catch-up with the full head, and back |
| `mtp_tok_embd` (`GET_ROWS`) larger than planned | 24 | a catch-up of 2 to 4 tokens after a plan made by a one-token drafting pass |
| `h_nextn` (`GET_ROWS`) larger than planned | 20 | a drafting pass with one output after a plan made by a catch-up without outputs (planned 0 bytes) |
| a `FILL` larger than planned | 11 | the QSA layer's row over the cells (`keep`, `[n_kv]`), which grows by 256 cells as the context does |
| `kpool_stage` (`CONT`) larger than planned | 5 | the k-pool tables' update, with `LLAMA_QSA_KPOOL_DEV=1` |

Without the shortlist the node count never changes, so the plan the context makes at its start, from a graph over the
whole memory and a full micro-batch, holds every later graph, and nothing re-plans. With `LLAMA_MTP_GRAPH_KEEP=1` the
graphs of up to 8 tokens go to per-size schedulers keyed by their tokens, outputs and head, so the shapes no longer
alternate in one plan; but a per-size scheduler is planned from its first graph, sized for the context of that moment.
Its 26 triggers are the 7 per-size schedulers' first graphs (`nodes 0 -> 168`, `0 -> 178`) and 19 times the `FILL`
over the cells, each with a larger buffer. Rows per draft evaluation: [cpu_trace/lines.tsv](cpu_trace/lines.tsv) (R60's
switches and no switch), [cpu_trace/iso.tsv](cpu_trace/iso.tsv) (one switch at a time),
[cpu_trace/keep.tsv](cpu_trace/keep.tsv) (with `LLAMA_MTP_GRAPH_KEEP=1`); the summaries beside them.

## What the re-plans cost

[R81](../2026-11-11_r81_ubatch_trace/README.md#the-drafts-re-plans) timed the same re-plans on a P106 with the
scheduler's trace (the night merge, the v3 preset): 34 in a fresh 4,097-token prompt, each planned in about 0.18 ms
(6.2 ms a prompt). Before each one the scheduler drains the draft's cards: 1,491.8 ms a prompt there, but the
draft's scheduler then had one input copy, so a pass without a re-plan waited as long. So the re-plans cost est. 6 ms
of planning a long prompt, and their drains count once the draft's passes queue ahead of each other, as with
`LLAMA_MTP_CATCHUP_OVERLAP=1` (C703, in `v4c2`'s line since R123a). On the CPU, a re-plan with `LLAMA_MTP_GRAPH_KEEP=1`
also allocates a larger compute buffer (19 times in FN84's eight requests); on the cards that is a device allocation
each time. Est. small, a fraction of a percent of prompt time; R130 times it on the line that carries the switch.

## The change

Branch `pascal-p106-v4-b12246-c803` on `pascal-p106-v4` `c96561876`, three commits: `568a979c3` the switch and the
test, `611a00bb3` the test's count, `bd632176f` the k-pool bounds.

- **`LLAMA_SCHED_PER_SIZE_RESERVE=1`** acts in a context with per-size schedulers: `LLAMA_MTP_GRAPH_KEEP=1` gives the
  MTP draft one for each batch shape of up to 8 tokens (its tokens, its outputs, the head's shortlist, the hidden-row
  carry), and `LLAMA_SCHED_PER_SIZE=1` any context. Without them it logs a warning and does nothing.
- **When a per-size scheduler is made** (`llama_context::process_ubatch_graph`), `sched_small_plan` builds the
  batch's graph over the whole memory (`init_full`: every cell of the cache, so the QSA layer's mask and row over the
  cells have their largest size), with the shortlist head's part for further ids built for the list's whole room
  (`head_short_n_plan`: 8,192 ids on Flash-Next's line), reserves the scheduler with it and drops it, never computed.
  The batch's own graph is built next, as without the switch, and fits that plan, as does every later graph of the
  same shape.
- **The k-pool's lists at a batch's bounds** (`kpool_plan_for`): a whole-cache context marks every pool as new and
  updates every row of the tables, which the shared scheduler's reserve needs (the first micro-batch after a sequence
  edit can re-pool them all) but a batch of up to 8 tokens never does: it pads its new-pool list to
  `n_tokens/4 + n_seqs` and its tables' update to a 64-row bound. With every pool the plan was 41 MiB a scheduler
  on a 65,536-cell context of the test model; with the bounds 0.48 to 1.31 MiB with R60's k-pool switches and 8.58 to
  9.15 MiB without them (the host tables gather every pool's key). A list past its bound (a long re-pool after an
  edit) is padded further and plans again, as without the switch.
- **Again only** when the shortlist head's part for further ids first appears (the list of further ids goes from
  empty to one id), which adds nodes. On Flash-Next the list holds the 33 special ids from the start.
- **What it costs:** the per-size schedulers' compute buffers have their whole-context size from the start instead
  of growing to it as the context does. On the CPU at 65,536 cells, the seven per-size schedulers of FN84's requests
  plan 5.63 MiB in all with R60's switches and 61.49 MiB with R85's; without the switch, by the eighth request (about
  6,200 tokens of context) each had grown to between 0.09 and 0.33 MiB, and to between 0.90 and 1.14 MiB, and they
  keep growing with the context. The planning takes 0.12
  to 0.26 ms a scheduler on the build machine's CPU, once a load.
- Nothing computed changes: the same graphs run, in the same order, with the same values.

Why per-size schedulers and not one graph shape for every draft pass, as C802 did for the main model: the draft's
passes differ in their outputs (the catch-up has none, a drafting pass one, read by the shortlist head), and its head
is a different graph with and without the shortlist. C619's per-size schedulers already keep each such shape in a
plan of its own; what they lacked was a plan that covers the whole context. On a line without
`LLAMA_MTP_GRAPH_KEEP=1` the switch does nothing; R60's and R85's lines would need both switches.

## The checks

On a build machine's CPU (`run/checks.sh`, a static native CPU build of the branch's tip `bd632176f`; every row in
[checks/summary.tsv](checks/summary.tsv)).

**`test-llama-archs --mtp-replan`** (new, [checks/replan.txt](checks/replan.txt)): the draft-mtp driver with the main
model, as llama-server runs them, over six turns of 300, 250, 420, 120, 30 and 260 new tokens (calls of up to 128,
draft micro-batches of 48), each followed by 10 speculative steps, on a 2,048-cell context; three arms of each run.

| Config | Re-plans, prompt/steps: shared scheduler alone | `LLAMA_MTP_GRAPH_KEEP=1` | both switches | Buffer growths: keep / both | Compute KiB at the end: keep / both |
|---|---|---|---|---|---|
| F16 cache, shortlist `32,8,2` | 4/68 | 1/26 | **0/0** | 27 / 0 | 6,188 / 6,593 |
| the same with the k-pool tables on the device and the indexer rows | 4/68 | 1/26 | **0/0** | 27 / 0 | 6,132 / 6,522 |
| F16 cache, no shortlist | 0/0 | 1/24 | **0/0** | 25 / 0 | 5,873 / 6,211 |
| Q8_0 cache, shortlist | 4/68 | 1/26 | **0/0** | 27 / 0 | 6,188 / 6,593 |

In every config the three arms give the same drafts, the same probabilities and the same draft and driver states; the
main model's graph never re-plans. Without the shortlist the shared scheduler alone does not re-plan either, and the
per-size schedulers of `LLAMA_MTP_GRAPH_KEEP=1` alone do, as the context outgrows their first plans.

**C619's `test-llama-archs --mtp-graph-keep`** without the switch and with `LLAMA_SCHED_PER_SIZE_RESERVE=1` for the
whole process: 8 of 8 configs pass both times, with the same graph builds and drafts; the draft's compute buffers on
its 256-cell model are 1,785 to 2,559 KiB with the switch against 1,713 to 2,358 without.

**llama-server** (`run/trace.sh` with this build, FN84's requests, [checks/server_summary.txt](checks/server_summary.txt)):

| Arm | Line | Draft re-plans (prompt, generation) | Per-size plans | Buffer allocations | Replies |
|---|---|---:|---:|---:|---|
| `r60` | R60's switches, the shortlist, draft micro-batch 512 | 104 (20, 84) | 0 | 7 | — |
| `r60keep` | and `LLAMA_MTP_GRAPH_KEEP=1` | 26 (5, 20; 1 before the first request) | 0 | 33 | as `r60` |
| `r60plan` | and `LLAMA_SCHED_PER_SIZE_RESERVE=1` | **0** | 7 | 14 | as `r60` |
| `r85` | R85's line: the shortlist, draft micro-batch 128 | 128 (44, 84) | 0 | 7 | — |
| `r85keep` | and `LLAMA_MTP_GRAPH_KEEP=1` | 26 (5, 20; 1 before the first request) | 0 | 33 | as `r85` |
| `r85plan` | and `LLAMA_SCHED_PER_SIZE_RESERVE=1` | **0** | 7 | 14 | as `r85` |

The buffer allocations count each new and each larger compute buffer: 7 for the contexts' own, then one per per-size
scheduler (its plan with the switch; its first graph and 19 growths without it).

**Every switch off** (`test-llama-archs -b CPU -s 1`): 146 of 146 models write `c96561876`'s logits with no switch,
and with `LLAMA_SCHED_PER_SIZE_RESERVE=1` (it does nothing without per-size schedulers). The base build is `8a55934f3`'s
(`-c802`): `c96561876` merges it with C805, which changes CUDA files only.

**C565's set** (`2026-11-05_c565_ladder4fix/run/cpu_tests.sh`), static and then shared libraries: every target
builds; ctest 70 of 70 (without the tests that need a download, the network or torch) and the network test 1 of 1;
test-llama-archs writes and compares 146 files, the same; the nine hand tests pass (hop-cast 54 checks, route-once
223 twice, bytes-switches 535 and 536, moe-ids 13, thread safety, the context model, the eval callback).

**The rig tarball** (`scripts/build_llamacpp_docker.sh` with `TESTS=1`, sm_61, image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`): `llama.cpp-20261112-bd632176f-pascal-p106-v4-b12246-c803.tar.gz`, 678,340,600
bytes, md5 `dca813b0206e1e58e3947db62a4c1f59`; no AVX instruction in `llama-bench`, highest glibc symbol 2.38,
`llama-server` md5 `39fba761d806618af555690eb4429050`, `test-backend-ops` md5 `a50d9c02f4e78d8a50d229dcebc151cf`; the
same 34 warning lines as the `-kern` and `-c802` builds.

The branch is merged into `pascal-p106-v4` with `--no-ff` as `d21820340`, on C806's merge `3ab5c876b` (test code only);
the merge adds exactly the branch's diff against `c96561876`. The checks ran on the branch's tip.

## The rig check (R130)

A CPU has one compute buffer per context, so it cannot show the re-plans that follow a change of cards (R81's 4 a
prompt at the draft's hidden-row input, two a step in generation) or the device allocations. R130 runs FN84's load
on rig 1 with the `-c803` rig tarball, R60's whole `gains` line (C691's switch on: `-c803` holds C802) and
`LLAMA_UBATCH_TRACE=1`, three times (`run/r130_loads.sh`, one load each, nothing else on the rig):

| Load | `ADD_ENV` |
|---|---|
| `base` | `LLAMA_UBATCH_TRACE=1` |
| `keep` | `LLAMA_UBATCH_TRACE=1 LLAMA_MTP_GRAPH_KEEP=1` |
| `plan` | `LLAMA_UBATCH_TRACE=1 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_SCHED_PER_SIZE_RESERVE=1` |

Tables on a build machine's CPU: `run/draft_trace.py parse` (the draft's re-plans per request and phase, those after a
change of cards, their drains and plans in µs, the buffer growths) and FN84's `run/fn84_requests.py` (replies and
checkpoints, R60's load `a` as the reference). **Done when:** `plan` re-plans the draft 0 times after its per-size
plans, in prompts and in generation; its 8 replies and checkpoints are `base`'s and `keep`'s (and R60's for the 6
replay requests); its `switches:` line names `LLAMA_SCHED_PER_SIZE_RESERVE=1`; and each load's lowest MemAvailable,
the draft's compute buffers per card at exit, and `base`'s and `keep`'s draft re-plans with their drains and plans are
given by value. The prompt times per request are one request each, not a timing class.

## Files

| File | What |
|---|---|
| `run/draft_reallocs.py` | where the draft's re-plans fall in each request, from llama-server `-lv 5` logs |
| `run/draft_trace.py` | the CPU trace's driver (FN84's requests to llama-server), its parser and the replies' comparison |
| `run/trace.sh` | the CPU trace: the tiny model and its draft file, then llama-server in each arm |
| `run/checks.sh` | the checks of the fix on a build machine's CPU, and the rig tarball |
| `run/r130_loads.sh` | R130's load on the rig: FN84's with switches added (`ADD_ENV`) |
| `rig_logs/requests.tsv`, `rig_logs/summary.tsv` | the five rig loads, per request and per prompt size |
| `cpu_trace/lines.tsv`, `iso.tsv`, `keep.tsv` and their `*_summary.txt` | the CPU trace on `d097d6d88`, per draft evaluation and per arm |
| `checks/summary.tsv` | every check's verdict and detail on the tip |
| `checks/replan.txt` | `test-llama-archs --mtp-replan`'s table |
| `checks/server.tsv`, `checks/server_summary.txt` | the six llama-server arms on the tip, per draft evaluation and per arm |

## Commands

Each on a build machine (CPU only; the containers at `--cpu-shares=256`, the scripts at `nice -n 19`), from this
folder unless said otherwise. The rig logs are FN84's, R60's and R85's own (`$LOGS/<load>/load-A/server.log`):

```bash
python3 -I run/draft_reallocs.py fn84=$LOGS/fn84_topk/load-A/server.log r60a=$LOGS/r60a/load-A/server.log \
    r60ro8=$LOGS/r60ro8/load-A/server.log r85v=$LOGS/c217v/load-A/server.log r85i=$LOGS/c217i/load-A/server.log \
    > rig_logs/requests.tsv 2> rig_logs/summary.tsv
```

The CPU trace, with `BIN` a static CPU build of `pascal-p106-v4` `d097d6d88` (`llama-server`, `test-llama-archs`):

```bash
WORK=$W BIN=$BIN ARMS='r60line@LLAMA_QSA_INDEXER_ROWS=1,LLAMA_QSA_KPOOL_DEV=1,LLAMA_DRAFT_SHORTLIST=100,8,2@--spec-draft-n-ubatch~512 r85line@@--spec-draft-n-ubatch~128' bash run/trace.sh
```

```bash
WORK=$W BIN=$BIN ARMS='n512@@--spec-draft-n-ubatch~512 kpool@LLAMA_QSA_KPOOL_DEV=1@--spec-draft-n-ubatch~512 rows@LLAMA_QSA_INDEXER_ROWS=1@--spec-draft-n-ubatch~512 short@LLAMA_DRAFT_SHORTLIST=100,8,2@--spec-draft-n-ubatch~512' bash run/trace.sh
```

```bash
WORK=$W BIN=$BIN ARMS='shortkeep@LLAMA_DRAFT_SHORTLIST=100,8,2,LLAMA_MTP_GRAPH_KEEP=1@--spec-draft-n-ubatch~512 r60keep@LLAMA_QSA_INDEXER_ROWS=1,LLAMA_QSA_KPOOL_DEV=1,LLAMA_DRAFT_SHORTLIST=100,8,2,LLAMA_MTP_GRAPH_KEEP=1@--spec-draft-n-ubatch~512' bash run/trace.sh
```

The tables of those runs, from `$W` (`r85line` is R85's line without its shortlist):

```bash
python3 -I run/draft_trace.py parse r60line=$W/r60line.server.log r85line=$W/r85line.server.log > cpu_trace/lines.tsv 2> cpu_trace/lines_summary.txt
```

(`iso` and `keep` the same with their arms.) The checks, from this repository's root, with `BASE_CPU` the `bin/` of a
static CPU build of `c96561876` (or of `8a55934f3`, whose CPU code is the same: `c96561876` adds C805's CUDA kernel
only) and `SEED_SRC` a llama.cpp clone with the fork as remote `fork`:

```bash
M=$WORK REPO=$PWD LOCK=$LOCK BASE_CPU=$BASE_CPU SEED_SRC=$SRC JOBS=8 THREADS=8 bash docs/data/2026-11-12_c803_draft_realloc/run/checks.sh
```

R130, from a machine that reaches the rig, once per load (`NAME` and `ADD_ENV` from the table above):

```bash
RIG=rig WORK=$R43 BUILDS=/opt/llama-builds BUILD=llama.cpp-20261112-bd632176f-pascal-p106-v4-b12246-c803 MODEL=$MODEL DRAFT=$DRAFT NAME=c803-plan ADD_ENV='LLAMA_UBATCH_TRACE=1 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_SCHED_PER_SIZE_RESERVE=1' bash run/r130_loads.sh
```

## Provenance and licence

The rig logs are this project's own runs: FN84, R60 and R85 (their folders link the commands and the model). The CPU
runs use test-llama-archs's model with random weights, made by the build itself, and prompts of token ids from a
Park-Miller generator: no outside data. Everything here is this repository's licence.
