# Flint's prompt scheduler kept on its load-time plan with per-size schedulers (R216, Q1627, 2026-11-19)

What this measures: whether `LLAMA_SCHED_PER_SIZE=1`, a switch already in the engine, stops Flint Pascal Q8 GGUF from
re-planning its compute buffers in the middle of prompts in agent use. [R181](../2026-11-19_r181_flint_realloc/README.md)
found the cause: the scheduler that reads prompts also runs the one-token passes Flint writes under the tool-call
grammar, and each of those replaces its plan over the whole context with one sized for the moment. After that, every
256 tokens a prompt crosses re-plans while the cards are busy (96 times in R181's run, est. 29.6 s of its 162.1 s of
prompt time). With the switch, every pass of up to 8 tokens runs on a scheduler of its own size, and the preset's
`LLAMA_SCHED_PER_SIZE_RESERVE=2`, which the server ignores without per-size schedulers, plans each of them once over
the whole memory. Registry row [Q1627](../../../quants/experiments.md).

R216a, the first run: one agent check (12 bug-fix tasks) of Flint's v5 preset
([release/flint-v5/settings.env](../../../release/flint-v5/settings.env)) on the v5 engine (`pascal-p106-v5`
`602691153`), nine P106-090 cards, every card at stock, with two variables added: `LLAMA_SCHED_PER_SIZE=1` and the trace
switch `LLAMA_UBATCH_TRACE=1`, which R181 also ran with. It is a trace run, not a timing run, and is compared with R181's
traced run of the same preset without the switch. The two runs are different sessions (temperature 1.0), so their
totals are context; the like-for-like comparisons are the prompts of the same size and the first request, whose prompt
is the same in every run.

## Result (R216a)

**In plain words:** the prompt scheduler no longer re-plans at all. In 72 requests the server counted 0 re-plans, where
R181 counted 200, and the trace finds none on the prompt scheduler and none on the new small schedulers. The prompts
that re-planned in R181 now read much faster: the 19 prompts of the same size in both runs that re-planned there took
**37.1 s here against 70.1 s**, 33.0 s less, about 660 ms for each of the 50 re-plans they had there. That is twice
R181's estimate per re-plan (308.0 ms). The longest of them, 2,872 new tokens, took 4,993.7 ms against 12,387.26 ms.
Every card still fits: 18 or 20 MiB more at the peak, the two fullest cards at 6,072 of 6,144 MiB. 12 of 12 tasks
passed, with coherent replies.

**The cost:** a small scheduler runs without the pipeline's overlap between cards. At each change of card the host
waits for the previous card to finish, so each token written under the grammar takes longer: in the first request,
whose prompt is the same in both runs, 26.92 ms from one such token to the next (median) against 24.77 ms in R181,
+2.15 ms a token (est. 5.8 s over this run's 2,720 one-token passes). The prompt's 4-token output pass also waits for
the prompt's earlier pieces to finish before it starts. The one prompt that never re-planned in R181 (the first request,
2,440 tokens) took 8,254.5 ms here against 8,056.94 ms (+197.56 ms, one request).

So this run passes the row's test (0 re-plans of the prompt scheduler, every card fits, 12/12). Whether the switch is a
gain for a whole run, prompts faster and grammar tokens slower, is for the two untraced runs against Flint's v5 preset
alone (R215's control runs).

### The run against R181

Both runs: Flint's v5 preset on `602691153`, nine P106-090 cards at stock, `LLAMA_UBATCH_TRACE=1`, one run each.

| | R216a, with `LLAMA_SCHED_PER_SIZE=1` | R181, without |
|---|---:|---:|
| Tasks passed | 12 / 12 | 12 / 12 |
| Requests | 72 | 76 |
| Re-plans in the server's count (`graph reallocs`) | 0 in 72 requests | 200 in 75 of 76 |
| Re-plans of the prompt scheduler in the trace | **0** | 328 (200 counted, 128 changes of card) |
| of them with the cards busy | **0** | 96 (drains 75,789.2 ms) |
| Re-plans of the per-size schedulers | 0 | — |
| Re-plans of the pipelined slots (not counted by the server) | 810 | 390 |
| Deepest context a pipelined token saw (the slots' attention mask, cells) | 24,832 | 13,824 |
| Prompt tokens read (a task) | 35,454 (2,954) | 34,502 (2,875) |
| Prompt time | 101.8 s (348 tok/s) | 162.1 s (213 tok/s) |
| Generated tokens, generation | 51,031, 49.2 tok/s | 43,014, 48.6 tok/s |
| First token per turn, median and p90 | 1.30 s, 2.74 s | 1.64 s, 5.39 s |
| Minutes for all tasks | 19.5 | 18.0 |
| Tool calls / failed / malformed | 66 / 2 / 0 | 71 / 2 / 0 |
| Checkpoint restores | 11 | 11 |
| Lowest MemAvailable; swap during the load | 1,030 MB; 28 → 299 MB | 1,061 MB; 28 → 299 MB |

(The agent check's memory line says NO-GO in both runs: the swap grows during the load, as in every release run.) The
pipelined slots have schedulers of their own, which the switch does not touch. They re-plan each time a pipelined token
sees a context 256 cells longer than any before (R181's row in its table), 10 slots at a time, so their count follows
the session's deepest context: 80 steps × 10 = 800 here against 38 × 10 = 380 in R181, plus 10 at their first use in
both ([causes.tsv](causes.tsv)). Their fix is Q1629.

### Prompts of the same size in both runs

20 prompts of 256 new tokens or more have a count of new tokens that occurs once in each run ([prompt_pairs.tsv](prompt_pairs.tsv);
in the check's fixed tasks that is most often the same turn of the same task, but the text was not compared).

| Pairs | R216a s | R181 s | R181's busy re-plans |
|---|---:|---:|---:|
| the 19 that re-planned in R181 | 37.1 | 70.1 | 50 |
| the one that did not (the first request, 2,440 tokens) | 8.25 | 8.06 | 0 |

19 of the 20 are faster here: the least by 253.11 ms (397 tokens), the most by 7,393.56 ms (2,872 tokens). By size of prompt over all requests
([prompts_by_size.tsv](prompts_by_size.tsv)), ms a new token:

| New tokens | R216a: requests, ms a token | R181: requests, ms a token, busy re-plans |
|---|---:|---:|
| 1–63 | 22, 22.530 | 27, 19.007, 0 |
| 64–511 | 25, 4.106 | 25, 4.866, 20 |
| 512–1,999 | 23, **2.219** | 22, **4.383**, 65 |
| 2,000 and more | 2, 2.494 | 2, 3.849, 11 |

Least squares over the 72 requests ([fit.tsv](fit.tsv)): prompt ms = 475.1 (standard error 69.9) + 1.906 (0.091) ×
tokens, against R181's 381.0 (31.2) + 3.003 (0.084) × tokens + 308.0 (25.1) × busy re-plans. R181's fit charged part
of each re-plan's cost to its tokens (a busy re-plan comes every 256 tokens), which is why the pairs show twice its
estimate. The intercept, 94.1 ms higher here (1.2 standard errors of the difference, est.), fits the 4-token
output pass now starting only after the prompt's last pieces have finished.

### The tokens written under the grammar

[grammar.tsv](grammar.tsv): the time from one one-token pass to the next, in runs of such passes (a gap of 1 s or more
is left out), and the medians of the trace's parts.

| | R216a | R181 |
|---|---:|---:|
| request 0 (the same prompt): passes, median ms, mean ms | 27, **26.92**, 28.04 | 28, **24.77**, 24.65 |
| request 1: passes, median, mean | 25, 28.28, 28.93 | 31, 25.78, 26.04 |
| request 2: passes, median, mean | 64, 28.63, 32.03 | 29, 25.94, 26.13 |
| all requests: passes, median, mean (deeper contexts here) | 2,660, 28.29, 30.62 | 3,618, 26.51, 28.28 |
| a one-token pass, reused graph: submit, total (median us) | 23,917, 24,127 | 2,864, 3,069 |
| the 4-token output pass: submit, total (median us) | 109,392, 114,867 | 16,194, 24,988 |

Why: the per-size schedulers are made without pipeline parallelism (one copy of their inputs, `src/llama-context.cpp`).
A scheduler without it has no device events, so at each change of card (`ggml_backend_sched_compute_splits`) the host
waits for the previous card to finish before it hands the next card its inputs; Flint's graphs run as 10 splits,
so a one-token pass waits on the host at each change between them. The submit then lasts as long as the whole pass (23.9 ms), where the prompt scheduler hands all nine cards
their work in 2.9 ms and the waits happen on the cards. The 4-token pass also waits, before it starts, for every piece
of the prompt still on the cards (a per-size scheduler's inputs have one copy, so the engine synchronizes first).
Making the per-size schedulers of a context with pipeline parallelism with events of their own is Q1632 (C903): est.
2.15 ms a grammar token back, and the 4-token pass's wait.

### Memory

[cards.tsv](cards.tsv): each card's MiB when the client first reached the server (loaded, before any request), its
peak over the run and what was left at the peak (6,144 MiB a card).

| Card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R216a loaded | 5,458 | 5,998 | 5,156 | 5,666 | 5,998 | 5,156 | 5,620 | 5,368 | 4,870 |
| R181 loaded | 5,450 | 5,990 | 5,148 | 5,658 | 5,990 | 5,148 | 5,612 | 5,360 | 4,862 |
| R216a peak | 5,526 | 6,072 | 5,220 | 5,730 | 6,072 | 5,220 | 5,684 | 5,442 | 4,948 |
| R181 peak | 5,506 | 6,052 | 5,202 | 5,712 | 6,052 | 5,202 | 5,666 | 5,422 | 4,928 |
| R216a left at the peak | 618 | **72** | 924 | 414 | **72** | 924 | 460 | 702 | 1,196 |

The per-size schedulers add 8 MiB a card after loading (the server's 2-token warm-up pass already runs on one) and 18 or 20 MiB at the
peak. Cards 1 and 4 keep 72 MiB, against 92 MiB without the switch. The host's lowest MemAvailable was 1,030 MB against
1,061 MB.

The server's load log no longer says that `LLAMA_SCHED_PER_SIZE_RESERVE=2` is ignored ([load.tsv](load.tsv)). The
per-size schedulers' own lines are libllama info lines, which the server does not print at its default verbosity, so
[persize.tsv](persize.tsv) is empty here.

## What follows

- **R216b and R216c, two untraced runs** with the switch (`TRACE=0`, arms `f5psb` and `f5psc`), alternated with R215's
  two control runs of the v5 preset alone on the same engine, classed on the agent check's tables. The commands are below.
- **Q1632 (C903), code:** the per-size schedulers of a context with pipeline parallelism made with it (device events,
  their own input copies), so that a one-token pass hands its cards their work as the prompt scheduler does, and the
  4-token pass waits only for its own copy. Est. 2.15 ms a grammar token and the 4-token pass's wait.
- **Q1628 (C900)**, planned as the fallback if R216 failed, is also a way to keep the prompt scheduler's plan while
  every pass stays on the pipelined scheduler: it would avoid the grammar tokens' cost without new schedulers.

## Files

| File | What |
|---|---|
| [causes.tsv](causes.tsv) | the re-plans grouped by scheduler, pass, cause and whether the cards were busy (only the pipelined slots' here) |
| [replans.tsv](replans.tsv) | every re-plan: time, request, micro-batch, node that changed card, graph sizes, drain and plan us, trigger lines |
| [requests.tsv](requests.tsv) | every request: prompt and generated tokens and ms, micro-batches, re-plans, the server's count, drain and plan ms |
| [fit.tsv](fit.tsv) | least squares of prompt time on tokens (no busy re-plan to fit) |
| [load.tsv](load.tsv), [persize.tsv](persize.tsv) | the switches the server read, its lines on `LLAMA_SCHED_PER_SIZE_RESERVE`; the per-size schedulers it logged (none at this verbosity) |
| [prompts_by_size.tsv](prompts_by_size.tsv), [prompt_pairs.tsv](prompt_pairs.tsv) | prompt time by size in both runs; the prompts of the same size in both runs |
| [grammar.tsv](grammar.tsv) | the gaps between one-token passes and the trace's parts of those passes and of the 4-token output pass, both runs |
| [cards.tsv](cards.tsv) | each card's MiB after loading and at the peak, both runs |
| [f5pst_compare.md](f5pst_compare.md), [f5pst_report_table.md](f5pst_report_table.md), [f5pst_gonogo.md](f5pst_gonogo.md) | the agent check's own report of this run |
| [agent/f5pst/](agent/f5pst/) | the run's `attempts.csv` and `turns.csv` |
| [run/r216.sh](run/r216.sh) | starts a run: the release harness's custom arm with Flint's v5 preset, `LLAMA_SCHED_PER_SIZE=1` and, with `TRACE=1`, the trace |
| [run/r216_replans.py](run/r216_replans.py) | R181's `r181_replans.py` with the per-size schedulers told apart (it gives R181's four tables byte for byte on R181's log) |
| [run/r216_prompts.py](run/r216_prompts.py), [run/r216_grammar.py](run/r216_grammar.py), [run/r216_cards.py](run/r216_cards.py) | the prompt, grammar-token and card tables |

## Commands

From the repository's root, on the machine that drives the rig and is the client too (Docker, the sandbox image
`agent-check:0.87.1`). `BIN` is the v5 engine's `llama-server` (`pascal-p106-v5` `602691153`, md5
`86607661d5a19d1a90d19ee4db10c90b`), `MODEL` Flint Pascal Q8 GGUF (`flint-near-q8-p065.gguf`, 40,431,236,544 bytes).
`PREFIX` gives the client's containers names of their own, since another rig's check shared the client.

```bash
RIG=rig CLIENT=local MODEL=$MODEL BIN=$BIN RIG_LOGDIR=$WORK/agent-logs \
  RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 CLIENT_DIR=$CLIENT_DIR CLIENT_OUT=$CLIENT_OUT OUT=$OUT \
  PREFIX=acr-r216a TRACE=1 ARM=f5pst bash docs/data/2026-11-19_r216_flint_persize/run/r216.sh
```

The started command (the harness's line): `LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SCHED_PER_SIZE=1
LLAMA_UBATCH_TRACE=1 llama-server -ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja
--reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --alias flint -b 512
-ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64`, R181's with
`LLAMA_SCHED_PER_SIZE=1` added. The model was loaded after 588 s (R181: 587 s) and the 12 tasks ran from 19:22:53 to
19:42:37. Nothing else ran on the rig meanwhile.

The two untraced runs, R216b and R216c (not run yet), are the same with `TRACE=0` and their own arm, prefix and folders:

```bash
RIG=rig CLIENT=local MODEL=$MODEL BIN=$BIN RIG_LOGDIR=$WORK/agent-logs \
  RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 CLIENT_DIR=$CLIENT_DIR CLIENT_OUT=$CLIENT_OUT OUT=$OUT \
  PREFIX=acr-r216b TRACE=0 ARM=f5psb bash docs/data/2026-11-19_r216_flint_persize/run/r216.sh
RIG=rig CLIENT=local MODEL=$MODEL BIN=$BIN RIG_LOGDIR=$WORK/agent-logs \
  RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 CLIENT_DIR=$CLIENT_DIR CLIENT_OUT=$CLIENT_OUT OUT=$OUT \
  PREFIX=acr-r216c TRACE=0 ARM=f5psc bash docs/data/2026-11-19_r216_flint_persize/run/r216.sh
```

The tables, from the server logs the harness copies to `$CLIENT_OUT/rig/` (this run's `server-f5pst.log`, 53,684,256
bytes, and R181's `server-f5ut.log`; they stay private, since they hold the session's text) and each run's
`rig/cards.csv`:

```bash
D=docs/data/2026-11-19_r216_flint_persize R181=docs/data/2026-11-19_r181_flint_realloc
python3 -I $D/run/r216_replans.py $D $CLIENT_OUT/rig/server-f5pst.log
python3 -I $D/run/r216_prompts.py R216a=$D/requests.tsv R181=$R181/requests.tsv > $D/prompts_by_size.tsv
python3 -I $D/run/r216_prompts.py --pairs R216a=$D/requests.tsv R181=$R181/requests.tsv > $D/prompt_pairs.tsv
python3 -I $D/run/r216_grammar.py R216a=$CLIENT_OUT/rig/server-f5pst.log R181=$R181_CLIENT_OUT/rig/server-f5ut.log > $D/grammar.tsv
python3 -I $D/run/r216_cards.py r181=$R181_CLIENT_OUT/rig/cards.csv@18:34:13 r216a=$CLIENT_OUT/rig/cards.csv@19:22:53 > $D/cards.tsv
```

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), the v5 engine `pascal-p106-v5` `602691153`. Model:
[Flint](https://huggingface.co/RecursiveSI/flint) (Apache-2.0) as my Flint Pascal Q8 GGUF
([models/flint](../../../models/flint/README.md)). The tasks are the agent check's bug-fix set
([bench/agent_check](../../../bench/agent_check/README.md)); the tables hold times, counts, tensor names and task
ids, no text. Measured on a Manli M-P106L9-N6G (nine P106-090, 4 GB host) on 2026-11-19, every card at stock (75 W,
no clock offset); the tables were made with the scripts above.
