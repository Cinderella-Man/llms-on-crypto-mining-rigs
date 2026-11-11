# The MTP draft's graphs kept per batch shape (C619, Q1104, 2026-11-11)

What this is: a change on the v4 line that stops the MTP draft context from building its graph twice in every
speculative step. [R86](../2026-11-11_r86_step_trace/README.md) traced the served step on a P106 with the night merge
(`c408dabcb`, the v3 preset). There the draft context built its graph again for the catch-up pass (the draft layer
over the verify pass's tokens) and again for the first drafting pass. The catch-up's build took 0.689 ms (median) on the
token's path, between the verify pass's wait and the catch-up's launch. The first drafting pass's build took 1.890 ms
after the catch-up was queued, and the host reached that pass 0.28 ms after the catch-up's work on the card had ended.
Drafting passes 2 and 3 reused their graph (0.015 and 0.013 ms).

The change is the switch `LLAMA_MTP_GRAPH_KEEP=1`, off by default, on the fork branch `pascal-p106-v4-b12246-c619`
(`916d1cf71`, three commits on `pascal-p106-v4-b12246-c748` `5b329df16`). This page holds the cause, the change, the
CPU checks and the rig run still to do ([Q1104](../../../quants/experiments.md), rig run Q1334).

## Result

**The draft context rebuilt its graphs because both of them shared one scheduler, which holds one allocated graph. With
the switch each graph keeps a scheduler of its own. In llama-server's MTP path on a CPU the draft's graph builds per
step drop from 2.035 to 0.030, and replies, draft counts and every drafted token's candidates and probabilities stay
the same.** The checks, on a CPU ([summary.tsv](summary.tsv)):

- **llama-server, draft-mtp at three tokens with backend sampling, three variants, three requests each**
  ([server.tsv](server.tsv), [server_builds.md](server_builds.md)). The base build (`5b329df16`) and the branch with the
  switch off and on give the same replies, draft counts and `LLAMA_SPEC_DUMP` lines: 36 of 36 checks. The draft's graph
  builds of up to 8 tokens:
  - plain: 411 off, 6 on, over 202 verify steps (2.035 and 0.030 a step);
  - with the head's shortlist (`LLAMA_DRAFT_SHORTLIST=100,8,2`, 8 flips to the full head): 411 off, 7 on (2.035 and
    0.035);
  - with a draft cut (`--spec-draft-p-min 0.102`, drafts of 0 to 3 tokens, 186 verify steps): 411 off, 6 on (2.210
    and 0.032).
- **`test-llama-archs --mtp-graph-keep`** (new): the draft-mtp driver with the target over a prompt and 16 steps whose
  catch-ups have every width from 1 to 4, in 8 configurations: F16 and Q8_0 caches, flash attention on and off, backend
  and host sampling, `LLAMA_MTP_KV_ONLY`, the head's shortlist, the hidden-row carry. It ran in 3 switch sets × 3 seeds:
  72 of 72 pass with the branch's CPU build, and 72 of 72 with the rig build's binary on the CPU
  ([keep.tsv](keep.tsv)). The draft's state after the prompt and at the end, every draft and its probabilities are the
  same bytes off and on. The 64 draft micro-batches of the steps build 32 graphs off and 4 or 5 on; with the carry, 48
  off and 5 or 7 on. The test without the switch in the build (`2f8715a6e`) fails all 72 for building no fewer graphs,
  with every result the same.
- **C620's and C703's checks with the switch on in both of their runs:** `--mtp-kv-only` 3 of 3 runs (15 of 15
  configurations), `--mtp-catchup-overlap` 3 of 3 (15 of 15).
- **Every switch off = `5b329df16`:** `test-llama-archs -b CPU -s 1` gives the base's logits for 146 of 146
  architectures; ctest 70 of 70.
- **The rig tarball** `llama.cpp-20261111-916d1cf71-pascal-p106-v4-b12246-c619.tar.gz` builds (656,348,735 bytes, md5
  `a9ed11e1c4600c0afbc586b57b69d7a8`, no AVX, GLIBC_2.38).
- **The server's step trace on a CPU** (`LLAMA_STEP_TRACE`, [cpu_step_trace_builds.md](cpu_step_trace_builds.md)):
  2.000 draft graph builds a step off, 0.015 on, over 202 steps that drafted, with the same replies and dumps.

## Why the draft rebuilt

A speculative step runs the draft context twice in a row with different shapes. The catch-up takes the verify pass's
tokens (1 to 4) with no output row and reads its hidden rows from the target's device buffer by index. Each drafting
pass takes one token with one output row and reads a host row. Since upstream's #31156 (Q701) these have separate graph
results, keyed by outputs and by size. The context still had one shared scheduler, though, and a scheduler holds the
allocation of one graph. A graph that the scheduler last allocated for the other shape is not ready to run. So the
catch-up's graph was rebuilt after the drafting passes, and the first drafting pass's graph after the catch-up. In the
CPU server run, `LLAMA_UBATCH_TRACE` gives the reason for all 411 rebuilds of up to 8 tokens as `not_allocated`
([server_builds.md](server_builds.md)). Passes 2 and 3 reused their graph because nothing ran in between.

The V100 fork's `LLAMA_MTP_PAD=4` with `LLAMA_REUSE_OUT_ORDER=1` ([note](../../research/notes/qwen_v100_llamacpp.md),
§6.5) does not fit this cause:
- **Padding** would give both passes one token count, but the two still differ in their outputs, in where their hidden
  rows come from (`llm_graph_params::allow_reuse` compares both) and, with the shortlist, in the head. Every drafting
  pass would also run three extra tokens through the draft layer's MoE, three passes a step, on the draft's card.
- **Reordered outputs** matter when a graph's output rows move between steps. With one sequence the drafting pass's one
  output is always row 0 and the catch-up has none.

## The change

| File | What changes |
|---|---|
| `src/llama-context.cpp`, `src/llama-context.h` | `LLAMA_MTP_GRAPH_KEEP=1`, read when an MTP context is made, turns on the per-size schedulers of `LLAMA_SCHED_PER_SIZE` (Q701) for that context alone. A batch of up to 8 tokens then runs on a scheduler of its own and keeps its graph allocated. The key of those graphs also holds the head's shortlist and the hidden-row carry, which the drafting passes flip between steps, so that each variant keeps its graph. A per-size scheduler waits for a graph in flight only when that graph is its own: the first drafting pass sets its inputs while the catch-up still computes, as it did on the shared scheduler. Off, the key, the waits and the schedulers are as on `5b329df16` |
| `tests/test-llama-archs.cpp` | `--mtp-graph-keep`: the check above. It counts the draft's graph builds from `LLAMA_UBATCH_TRACE`'s lines, which the mode turns on |

What stays: graphs of more than 8 tokens (the catch-up over a prompt chunk) use the shared scheduler and are rebuilt as
before. A shape's first use builds its graph: with the switch, the plain CPU server run builds 6 graphs of up to 8
tokens over its three requests, 4 without an output row and 2 with one. A memory update or reserve resets every graph, as it does
without the switch. The target context is not touched: the switch acts only in a context of type MTP.

What it costs: each per-size scheduler allocates its own compute buffers on the cards its graph uses, at the graph's
first run. On the CPU with the tiny test model the draft context's compute buffers grow from 1,411 to 2,278 KiB
(F16 cache, flash attention; [keep.tsv](keep.tsv)). The rig run reads what the switch adds on each card.

## What it changes on the rig (est., from R86's sizes)

With both graphs reused, est. 0.015 ms each instead of R86's 0.689 and 1.890 ms of building:
- **the catch-up:** est. 0.67 ms a step off the token's path (0.689 − 0.015);
- **the first drafting pass:** its build left the token's path for all but 0.28 ms, so est. up to 0.28 ms a step,
  and 1.875 ms a step of host time freed on the Celeron's two cores while the catch-up computes.

Est. 0.67 + up to 0.28 = up to 0.95 ms a step in all; R86's estimate of 0.97 ms took the reuse at no cost.

## What the rig run must show (R95, Q1334)

On rig 1, untimed, R86's driver ([rig_trace.sh](../2026-11-11_c405_server_gap/run/rig_trace.sh), which now takes
`LOAD_ENV`) with the c619 tarball and the v3 preset. Six loads in one run, each answering S37's greedy request and then
the agent's first-turn request: `trace` and `tracekeep` (the step trace with the switch off and on), then `prof`,
`keep`, `prof-2` and `keep-2` (the host profile alone: off, on, off, on). Est. 40 min of loads (R86's two loads took
13.4 min), plus the relay. The run is right when:
- the six loads give R86's greedy reply text and draft counts, the switch off and on;
- [run/count_builds.py](run/count_builds.py) counts about 2 draft graph builds a step in `trace` and few in
  `tracekeep`; `step_gap.py`'s waterfall then shows `d.ub.apply → d.ub.reuse` where R86 shows `→ d.ub.build`;
- `nvidia-smi` in each load's monitor shows what the switch adds on the draft's cards, and nothing runs out of memory.

The verdict is `keep` and `keep-2` against `prof` and `prof-2` on the step (S37's `arms.py`, its class against the
spread of the two reference loads), and the gap before the verify pass in `tracekeep` against `trace`.

## Files

| File | What it holds |
|---|---|
| [run/checks.sh](run/checks.sh) | the whole check: the CPU builds, the new test, C620's and C703's tests with the switch, the server runs, the identity against the base, ctest, the rig tarball (through C620's `checks.sh`) and the rig binary's runs of the new test |
| [run/server_smoke.py](run/server_smoke.py) | one server arm on the CPU (three requests, the trace and the dump), and the comparison of arms |
| [run/count_builds.py](run/count_builds.py) | the draft's graph builds and reuses per step in `LLAMA_STEP_TRACE` files, for the rig run |
| [summary.tsv](summary.tsv) | one verdict a check (44 lines; the work folder written as `$WORK`, the base's binaries as `$BASE_BIN`) |
| [keep.tsv](keep.tsv) | `--mtp-graph-keep`, one row per build (`tip`, `before`, `rig`), switch set, seed and configuration: identity, shortlist, graphs built in the prompt and in the steps (off/on of the steps' micro-batches), compute buffers, status |
| [server.tsv](server.tsv) | the server runs: each arm's replies and dump against the variant's base arm |
| [server_builds.md](server_builds.md) | the server runs: the draft's micro-batches, rebuilds and reuses per arm, why each was rebuilt |
| [cpu_step_trace_builds.md](cpu_step_trace_builds.md) | `count_builds.py` on two CPU server runs with `LLAMA_STEP_TRACE`, the switch off and on |

## Commands

On a build machine with Docker, CPU only, every build and test under one lock (shared with other sessions' checks) at
`nice -n 19` and `--cpu-shares=256`:

```bash
# WORK: a work folder; SRC: a llama.cpp clone in $WORK/rig/src with the fork as remote "fork" (https fetch URL), which
# scripts/build_llamacpp_docker.sh checks out and cleans; BASE_BIN: bin/ of 5b329df16's static native CPU build with
# test-llama-archs and llama-server; REPO: this repository
WORK=$WORK REPO=$REPO SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c619 BEFORE=2f8715a6e BASE_BIN=$BASE_BIN \
  LOCK=$LOCK JOBS=8 DOCKER_ARGS='--label pascal-quants=1' \
  bash docs/data/2026-11-11_c619_draft_graph_reuse/run/checks.sh
```

The step-trace runs, in the same image with the branch's CPU build in `$B`, the server step's model in `$M` and an
output folder `$S`:

```bash
A="-ngl 0 -t 4 -c 4096 -np 1 --no-webui -fa on -bs --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 128"
R=docs/data/2026-11-11_c619_draft_graph_reuse/run
python3 -I $R/server_smoke.py arm off $S 18620 $B/llama-server $M LLAMA_STEP_TRACE=$S/off.trace.jsonl -- $A
python3 -I $R/server_smoke.py arm on $S 18620 $B/llama-server $M LLAMA_STEP_TRACE=$S/on.trace.jsonl LLAMA_MTP_GRAPH_KEEP=1 -- $A
python3 -I $R/server_smoke.py compare $S plain=off,on
python3 -I $R/count_builds.py off=$S/off.trace.jsonl on=$S/on.trace.jsonl > cpu_step_trace_builds.md
```

Provenance and licence: everything here was generated by the commands above from the fork's code (MIT, as llama.cpp)
on random models that `test-llama-archs` builds; no model file, dataset or recorded text is involved.
