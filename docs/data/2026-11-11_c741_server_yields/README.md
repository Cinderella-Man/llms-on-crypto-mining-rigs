# Fewer yields to the task queue (C741, Q1310, 2026-11-11)

What this is: a switch in Miner Llama's llama-server, `LLAMA_SERVER_YIELD` (off by default), that cuts how often a
speculative step hands the task queue to its worker thread. llama-server runs its heavy calls inside
`yield_to_queue`, so that a worker thread can answer `/metrics` and `/slots` while the main thread computes. Each yield
wakes the worker, and at its end the main thread waits for the worker to stop. On the rig's two-core Celeron each of
those hand-overs costs about a tenth of a millisecond, and a step has three of them: around the draft, the verify
pass and the catch-up. [R86](../2026-11-11_r86_step_trace/README.md) timed them on a P106 rig (below). The registry
row is Q1310 in [quants/experiments.md](../../../quants/experiments.md).

Fork branch `pascal-p106-v4-b12246-c741` at `be9db2460`, two commits on `pascal-p106-v4-b12246-c748` (`5b329df16`):
`ad503b78c` the switch, its counts in the step trace and the queue's unit test, `be9db2460` the lazy wake looks only
at tasks a yield can take (see [The check](#the-check)). Not merged into another branch. Everything here ran on a
build machine's CPU; nothing ran on a card.

## Result

**One yield a step instead of three, and with `=2` no hand-over to the worker at all while no read is waiting, with
the same replies, drafts and probabilities.** On a build machine's CPU (llama-server, the tiny random qwen4exp model
with one MTP block, the draft at three tokens), each of the 406 generation steps of four requests makes 3 yields and
wakes the worker 3 times with the switch off, 1 yield and 1 wake with `=1`, and 1 yield and 0 wakes with `=2`. The six
host stages the yields make add up to 0.0520 ms a step off (median), 0.0160 with `=1` and 0.0010 with `=2`
([yields.md](yields.md)). On that CPU a hand-over is cheap; on the rig R86 measured the same six stages at 0.371 ms a
step in all, so the switch saves **est. 0.23 ms a step with `=1` and est. 0.37 ms with `=2`**, all of it on the
token's path ([below](#what-it-changes-on-the-rig-est-from-r86s-sizes)).

**Other requests are still served within one step.** With two slots, a second request posted while the first is being
answered starts its prompt in the step right after the one it came in, with the switch off and on, and both replies are
the same bytes as each request answered alone. Reads of `/slots` and `/metrics` sent every 2 ms all along are all
answered, either in the step's remaining yield or when the step ends ([two_slots.md](two_slots.md)).

## The change

llama-server's task loop (`server_queue::start_loop`) takes every task in the queue, runs one step (`update_slots`),
and repeats. During a step, the main thread runs its llama calls inside `yield_to_queue`: the worker thread wakes,
takes the tasks that only read the server's state (metrics, slot reads) and declines every other (completions, cancels,
anything that changes a slot), which go back into the queue for the loop. At the yield's end the main thread sets the
yield off, wakes the worker and waits until it has stopped. The step's three yields wrap:
- the draft (`common_speculative_draft`: the MTP draft's three passes);
- the verify pass (`llama_process` and the wait for its outputs, ~100 ms on the rig);
- the catch-up (`common_speculative_process`: the draft context fed the verified tokens).

`LLAMA_SERVER_YIELD`, read once at server start:
- **`1`:** a generation step's draft and catch-up run without a yield. A catch-up whose call holds prompt tokens keeps
  its yield (it is long), as do every decode, the image encodes and the prompt's micro-batches. The verify pass keeps
  its yield, which spans its ~100 ms wait for the cards: a read that comes during the draft or the catch-up is answered
  at that yield or when the step ends.
- **`2`:** as `1`, and a yield wakes the worker only for a task it can take: a read in the queue when the yield starts,
  or one posted during it (`server_queue::set_lazy_wake`, `on_yield_takes`). A yield that no read meets ends without a
  hand-over. Upstream's worker wakes at every yield, finds the loop's own `NEXT_RESPONSE` task (which `update_slots`
  posts at the start of every busy step to keep the loop turning), declines it and puts it back.

Completions and every task that changes a slot are never taken during a yield, off or on: they wait for the step's end
either way, and the loop takes them in the same order. The work inside a yield runs on the server's own thread with or
without the yield, so every output is the same by construction; the check below confirms it.

With `LLAMA_STEP_TRACE` ([C405](../2026-11-11_c405_server_gap/README.md)) each step's line now also counts its yields
(`"yld"`) and the worker's wakes (`"wake"`), and marks a completion or a read posted during the step (`q.cmpl`,
`q.read`) and a read the worker answered in a yield (`q.take`). Off, the trace's stages are unchanged; with the switch
the skipped yields' marks stay, back to back.

## The check

Every build and test ran in a container on a build machine's CPU, at low priority, one at a time under a lock
([run/checks.sh](run/checks.sh)); the results are in [summary.tsv](summary.tsv).

| Check | Verdict |
|---|---|
| static native CPU build of `be9db2460`, every target | OK, 3 warning lines (the base's, in `common/sampling.cpp`) |
| `test-server-component server_queue_yield`, five runs: a wake every yield and the lazy wake (yields and wakes counted; a read posted during a yield and one waiting at its start answered in it; the loop's own task and a completion posted during a yield left for the loop, wake no worker in lazy mode, and are taken in their order after the step) | OK, 22 assertions each |
| llama-server, one slot, nine arms ([run/server_check.sh](run/server_check.sh)) | PASS (below) |
| llama-server, two slots, four arms ([run/two_slots.py](run/two_slots.py)) | PASS (below) |
| `test-llama-archs -b CPU -s 1`, the switch unset and `=2`, against `5b329df16`'s logits | 146 of 146 the same, both |
| ctest (without the tests that need a download, the network or torch) | 70 of 70 |
| rig tarball, `TESTS=1` | OK (below) |

**One slot.** `base` is `5b329df16`'s llama-server, the other arms this branch's: `off` (the switch unset), `y1`, `y2`,
`y2nt` (`=2` without the step trace, which every other arm writes), and the same with the backend sampler (`-bs`, as the
release preset): `bbase`, `boff`, `by1`, `by2`. Each answered four requests: 200 and 600 prompt tokens, 96 or 128
tokens greedy and 96 at temperature 1.0 (top-k 20, top-p 0.95, two seeds), with the top 3 probabilities of every
token. Every arm gives its reference's tokens, top probabilities, draft counts and all 406 verify steps of
`LLAMA_SPEC_DUMP` (the draft, the drafter's candidates and probabilities, the target's top tokens)
([server_verdict.txt](server_verdict.txt), [server.tsv](server.tsv)). The yields of the generation steps
([yields.md](yields.md); ms medians on the CPU, the same 406 steps in each arm):

| arm | yields a step (steps) | wakes a step (steps) | the draft's yield, start and end | the verify pass's yield, start and end | the catch-up's yield, start and end | the six, median of sums |
|---|---|---|---|---|---|---:|
| base (`5b329df16`) | — | — | 0.0010, 0.0140 | 0.0010, 0.0130 | 0.0010, 0.0130 | 0.0420 |
| off | 3: 406 | 3: 406 | 0.0020, 0.0160 | 0.0020, 0.0150 | 0.0020, 0.0150 | 0.0520 |
| `=1` | 1: 406 | 1: 406 | 0.0000, 0.0000 | 0.0020, 0.0140 | 0.0000, 0.0000 | 0.0160 |
| `=2` | 1: 406 | 0: 406 | 0.0000, 0.0000 | 0.0000, 0.0000 | 0.0000, 0.0000 | 0.0010 |
| boff | 3: 406 | 3: 406 | 0.0020, 0.0140 | 0.0010, 0.0130 | 0.0010, 0.0130 | 0.0435 |
| by1 | 1: 406 | 1: 406 | 0.0000, 0.0000 | 0.0020, 0.0140 | 0.0000, 0.0000 | 0.0160 |
| by2 | 1: 406 | 0: 406 | 0.0000, 0.0000 | 0.0000, 0.0000 | 0.0000, 0.0000 | 0.0000 |

The base build counts neither yields nor wakes (its trace predates them). The model's MTP layer is random, so almost
no draft is accepted (2 of 1,206 drafted tokens); the step's path through the server is the same.

**Two slots** (`-np 2`, `--metrics`, the step trace; [two_slots.md](two_slots.md)). Request A (600 prompt tokens,
128 greedy) is posted, and request B (200 prompt tokens, 96 greedy) once `/slots` shows A with 16 tokens; a client
thread reads `/slots` and `/metrics` in turn every 2 ms until both are answered. In all four arms (`p2base`, `p2off`,
`p2y1`, `p2y2`) A and B are the same tokens, top probabilities and draft counts as the one-slot base's answers to the
same requests, and no read failed.

| arm | B came in step | its prompt in step | reads posted in a step | answered in a yield of that step | left to the step's end | client reads | read ms, median | read ms, max | step ms, median | step ms, max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p2base | — | — | — | — | — | 228 | 0.3655 | 25.7990 | 3.3720 | 75.4780 |
| p2off | 19 | 20 | 229 | 225 | 4 | 234 | 0.3280 | 20.0500 | 3.4415 | 83.4620 |
| p2y1 | 19 | 20 | 181 | 168 | 13 | 185 | 0.8370 | 28.8570 | 3.2115 | 75.4070 |
| p2y2 | 19 | 20 | 184 | 174 | 10 | 191 | 0.5920 | 28.4770 | 3.3870 | 80.8460 |

The step columns come from the server's own marks (the base build has none); a read left to the step's end is taken by
the loop when the step ends, as every task left in the queue is. The read times are the client's: with the switch a
read that comes during the draft waits for the verify pass's yield, so the median read takes longer (0.8370 and 0.5920
ms against 0.3280) and stays within the step. The longest steps (75 to 84 ms) are the prompts'.

**What changed during the check.** The first build woke the worker at every yield with `=2` too (1 wake a step): the
queue always holds the loop's own `NEXT_RESPONSE` task during a busy step, and the lazy wake looked at any task.
`be9db2460` gives the queue the task types a yield takes (`on_yield_takes`, the rule `process_single_task` applies), and
the lazy wake looks only at those. The first full run then failed two criteria of the check itself: the unit test's
filter matched no subtest (the filter is matched against the whole name), and the two-slot criterion counted, on the
client's clock, how many steps ended between a read's sending and its answer. One read in `p2y1` saw two: the client
sent it 0.009 ms before a step ended, the server marked it posted 0.089 ms into the next step and answered it in that
step's verify yield 1.145 ms later, and the client received the answer 3.362 ms after that, when the step had ended
too. The check now counts the reads as the server saw them; the unit and server steps ran again with the same build.

**The rig tarball:** `llama.cpp-20261111-be9db2460-pascal-p106-v4-b12246-c741.tar.gz`, 656,351,924 B, md5
`850351deaf0a866e00c3aaf69395eba0`, `TESTS=1`, sm_61, image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, no AVX instruction
in `llama-bench`, highest glibc symbol GLIBC_2.38, `test-llama-archs` in `bin/`.

## What it changes on the rig (est., from R86's sizes)

R86 ([gap_trace.md](../2026-11-11_r86_step_trace/gap_trace.md)) measured the six stages on a P106 rig with the night
merge (medians, ms):

| yield | start | end | where in the step |
|---|---:|---:|---|
| the draft's | 0.004 (`s.drf → s.drf.in`) | 0.100 (`s.drf.out → s.drf.end`) | before the first drafting pass; between the last drafting pass and the verify pass |
| the verify pass's | 0.004 (`s.dec.y → s.dec.in`) | 0.136 (`s.dec.out → s.dec.end`) | before the verify pass's decode; between its outputs and the catch-up's launch |
| the catch-up's | 0.003 (`s.mpd → s.cu.in`) | 0.124 (`s.cu.out → s.cu.end`) | before the catch-up's decode; after the catch-up is queued |

`s.dec.out → s.dec.end` holds nothing but the verify pass's yield end (`s.dec.out` is marked after the wait for the
outputs, inside the yield). Every stage is on the token's path: the catch-up's end runs while
card 8 computes the catch-up, but the host reaches the next drafting pass 0.28 ms after the catch-up's work ends (R86),
so host time there delays the next step. Est.:
- `=1`: the draft's and the catch-up's yields gone, 0.004 + 0.100 + 0.003 + 0.124 = **0.231 ms a step**;
- `=2`: also the verify pass's hand-over, 0.371 ms in all, **est. 0.37 ms a step** (on the CPU `=2` leaves 0.0010 of
  0.0520 ms).
Both also take the worker's turns off the second core during the step. Against the v4 line's greedy step of 128.92 ms
(R86, `prof`) that is est. 0.29 %.

## What the rig run must show (R105, Q1350)

On rig 1, untimed, R86's driver ([rig_trace.sh](../2026-11-11_c405_server_gap/run/rig_trace.sh) with `LOAD_ENV`) with
the c741 tarball and the v3 preset. Seven loads in one run, each answering S37's greedy request and then the agent's
first-turn request: `trace`, `tracey1` and `tracey2` (the step trace with the switch off, `=1` and `=2`), then `prof`,
`y2`, `prof-2` and `y2-2` (the host profile alone: off, `=2`, off, `=2`). Est. 47 min of loads (R86's two loads took
13.4 min), plus the relay. The run is right when the seven loads give R86's greedy reply text and draft counts. The
verdict:
- [run/yields.py](run/yields.py) on the three trace loads: 3, 1 and 1 yields a step, 3, 1 and 0 wakes, and the six
  stages, est. 0.371 → 0.140 → near 0 ms a step;
- `y2` and `y2-2` against `prof` and `prof-2` on the step (S37's `arms.py`, its class against the spread of the two
  reference loads), and `step_gap.py`'s gap before the verify pass in `tracey2` against `trace`.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict, as `run/checks.sh` wrote it (the unit and server rows from the second run; the server row is cut at 2,000 characters: the whole verdict is `server_verdict.txt`) |
| [server_verdict.txt](server_verdict.txt) | `run/server_check.sh`'s verdict, line by line |
| [server.tsv](server.tsv) | every arm's requests: prompt and predicted tokens, drafted and accepted tokens, the tokens |
| [yields.md](yields.md), [yields.tsv](yields.tsv) | `run/yields.py` over the one-slot arms' step traces: yields and wakes a step, the six yield stages |
| [two_slots.md](two_slots.md), [two_slots.tsv](two_slots.tsv) | `run/yields.py --two-slots` over the two-slot arms: B's steps, the reads as the server and the client saw them |
| [run/checks.sh](run/checks.sh) | the driver of every check (the CPU build, the unit test, the servers, the logits, ctest, the rig tarball) |
| [run/server_check.sh](run/server_check.sh) | llama-server's arms, one slot and two, and the verdict |
| [run/two_slots.py](run/two_slots.py) | the two-slot client: A, B mid-reply, the reads all along |
| [run/yields.py](run/yields.py) | the yields, wakes and yield stages of `LLAMA_STEP_TRACE` files; the two-slot table |

The rig build uses C620's [run/checks.sh](../2026-11-11_c620_draft_kv_catchup/run/checks.sh) (its `build` step).

## Commands

On a build machine, CPU only (each build and test in a container at low CPU priority, one at a time under a lock):

```bash
# WORK: a work folder; SRC: a llama.cpp clone in $WORK/rig/src with the fork as remote "fork" (https fetch URL), which
# scripts/build_llamacpp_docker.sh checks out and cleans; BASE_CPU: bin/ of 5b329df16's static native CPU build with
# test-llama-archs; BASE_SRV: the folder of its llama-server; REPO: this repository
WORK=$WORK REPO=$REPO SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c741 BASE_CPU=$BASE_CPU BASE_SRV=$BASE_SRV \
  LOCK=$WORK/lock JOBS=8 bash docs/data/2026-11-11_c741_server_yields/run/checks.sh
```

By hand, with a CPU build of the branch and a model with an MTP block:

```bash
test-server-component 'server_queue_yield.*'
LLAMA_SERVER_YIELD=2 LLAMA_STEP_TRACE=trace.jsonl llama-server -m model.gguf --spec-type draft-mtp --spec-draft-n-max 3
python3 -I docs/data/2026-11-11_c741_server_yields/run/yields.py y2=trace.jsonl
```

## Limits

- Every number here is from a build machine's CPU (a desktop Ryzen with idle cores, where a hand-over is cheap), a
  diagnostic; R105 measures the stages on the rig's two-core Celeron.
- With the switch, a `/metrics` or `/slots` read that comes during the draft waits for the verify pass's yield: on the
  rig the draft's three passes take about 13 ms (R86's stage medians, 13.278 ms summed), so such a read can take that
  much longer. It is still answered within the step. A tool that polls the server during generation sees it; the agent
  check reads `/metrics` only before and after a task.
- The two-slot check posts B at one point of A's reply; the server takes a completion only between steps, off or on,
  so where B lands changes nothing in the rule.

## Provenance and licence

The models are `test-llama-archs`'s random fixtures (this fork's test code). The engine is Miner Llama (MIT, as
llama.cpp). The tables are this project's measurements.
