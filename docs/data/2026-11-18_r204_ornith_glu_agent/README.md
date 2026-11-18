# R204: Ornith's agent check with the fused matrix-vector products off (Q1593, 2026-11-18)

**In plain words.** On the deep lane, turning off the fused matrix-vector launches with `GGML_CUDA_MMID_GLU_UNFUSE=2`
made Ornith-1.5-122B Pascal N2P GGUF decode 7.0 % faster over depths 0 and 8,192, with every reply the same bytes
([C865b](../2026-11-18_c865b_ornith_glu_rig1/README.md), Q1559). Release settings are chosen on the coding-agent
check, so this runs that check with the switch added to Ornith's released v4 preset. Two runs with the switch are
compared with two runs without it, on the same engine and rig. This page has run 1 so far. The verdict comes with run 2
(R204b).

**Run 1 (R204a): all 12 tasks passed, and a decode step at the same context took 6.5 % and 6.7 % less time than in the
two runs without the switch.** Measured on the same mix of contexts, generation ran at 17.48 tok/s against 16.35 and
16.32. Over the whole session it was 17.66 against 16.30 and 16.19 tok/s. A task at equal work took 2.04 minutes
against 2.17 and 2.21. Prompt reading did not change: the requests read on the same prompt in all three runs took
194.86 s with the switch against 195.22 and 194.05 s without it, the same as the two runs without it differ from each
other. The switch acts only on batches of at most 8 tokens, so prompt reading should not change.

## The method

- **Rig:** nine P106-090 6 GB cards on PCIe 1.0 x1 (the rig in [docs/mining_rig.md](../../mining_rig.md)), every card at
  stock before and after (75 W, 1,911 / 4,004 MHz maximum, no Xid; the card log reads offset 0 on every card).
- **Engine:** the candidate `pascal-p106-v5-cand-b12483` `602691153` (upstream b12483 on `pascal-p106-v5`), the tree
  the v5 line is about to move to ([its P106 check](../2026-11-18_c854b_p106_check/README.md)); `llama-server` md5
  `86607661d5a19d1a90d19ee4db10c90b`, checked on the rig before the run. The switch's code is unchanged from `1f1110c67`,
  the tree C865b measured.
- **Model and preset:** Ornith-1.5-122B Pascal N2P GGUF (`valid1`, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`) with
  its released v4 preset (`release/ornith-v4/settings.env`), 48k context, the agent check's 12 bug-fix tasks once.
- **The one change:** [run/ornith_agent_glu.sh](run/ornith_agent_glu.sh) is the launcher of the control runs
  ([run/ornith_agent.sh](../2026-11-18_c854b_p106_check/run/ornith_agent.sh)) with `GGML_CUDA_MMID_GLU_UNFUSE=2` added to
  the server's environment. The started command is the control's byte for byte with that one setting added, and the
  server's `switches:` line names the control's six settings plus `GGML_CUDA_MMID_GLU_UNFUSE=2` (7 set; the build reads
  889 names, as in the controls).
- **Controls:** the same launcher, tree and rig with the switch unset, run earlier the same day: `o4c2` (14:11,
  [part (5)](../2026-11-18_c854b_p106_check/README.md#part-5-the-agent-check-on-ornith-2026-11-18)) and `o4c3` (16:57,
  [part (8)](../2026-11-18_c854b_p106_check/README.md#part-8-the-candidates-second-run-on-ornith-2026-11-18)). Their
  difference from each other is this tree's own spread on this rig. `pascal-p106-v5`'s own build `1f1110c67` also ran
  twice on the rig that day (`o4v5a` 16:06, `o4v5b` 17:50); those two appear only in the five-run tables.
- **The class** (after run 2): the switch's two runs against the two controls, a gain or a loss only beyond the spread
  of both pairs of runs.

## Run 1 (R204a, arm `o4g2a`, 2026-11-18)

The day's sixth agent run on the rig, started 20 min after the fifth ended. The server started at 18:39:43 and was up
after 683 s (the controls: 683 and 681 s); the tasks ran 18:51:07–19:13:52.

| | **`o4g2a`, the switch (this run)** | `o4c2`, unset | `o4c3`, unset | this run against each control | the controls' own spread (`o4c3` against `o4c2`) |
|---|---|---|---|---|---|
| passed | **12 / 12** | 12 / 12 | 12 / 12 | | |
| ms a decode step at the same mix of contexts | **57.19** | 61.18 | 61.28 | −6.5 %, −6.7 % | +0.2 % |
| generation tok/s at the same mix of contexts | **17.48** | 16.35 | 16.32 | +6.9 %, +7.1 % | −0.2 % |
| generation tok/s, the whole session | **17.66** | 16.30 | 16.19 | +8.3 %, +9.1 % | −0.7 % |
| minutes a task at equal work (`ka`'s work) | **2.04** | 2.17 | 2.21 | −6.0 %, −7.7 % | +1.8 % |
| the 19 requests read on the same prompt in all three runs, prompt s | **194.86** | 195.22 | 194.05 | 0.9982, 1.0042 | 0.9940 |
| of them, the 12 first turns | **118.71** | 119.20 | 118.02 | 0.9959, 1.0058 | 0.9901 |
| of them, the 18 with a cached prefix | **166.14** | 167.40 | 166.18 | 0.9925, 0.9997 | 0.9927 |
| the 22 matched with the cached tokens within one, prompt s | **224.03** | 224.50 | 223.32 | 0.9979, 1.0032 | 0.9948 |
| first token on a task's first turn, median s | **8.53** | 8.58 | 8.55 | | |
| first token median s; p90 | **6.99; 11.18** | 7.56; 11.11 | 5.03; 10.34 | | |
| prompt tok/s, the whole session | **98.5** | 96.2 | 92.0 | | |
| prompt tokens read a task | **3,235** | 3,259 | 3,146 | | |
| tokens generated a task; of them past 8,000 tokens of context | **1,376; 388** | 1,689; 4,320 | 1,801; 5,042 | | |
| reply tokens re-read (turns; their prompt s of all prompt s) | **3,406 (3; 31.2 of 394.3)** | 2,143 (1; 13.2 of 406.6) | 1,371 (2; 17.4 of 410.2) | | |
| minutes, all 12 tasks; turns | **22.6; 62** | 28.0; 63 | 29.5; 72 | | different work |
| tool calls / failed / malformed | **62 / 1 / 0** | 63 / 0 / 0 | 72 / 1 / 0 | | |
| checkpoints restored / created / erased; full re-processing | **14 / 0 / 0; 0** | 12 / 0 / 0; 0 | 13 / 0 / 0; 0 | | |
| graph reallocations | **0 in 62 requests** | 0 in 63 | 0 in 72 | | |
| busy SM clock, mean of the nine cards' medians, MHz | **1,879.8** | 1,876.9 | 1,878.3 | | |
| lowest MemAvailable MiB; swap highest MiB | **1,223; 343** | 1,199; 341 | 1,224; 342 | | |

Each value is one run ([r204a_agent.tsv](r204a_agent.tsv), [r204a_decode.md](r204a_decode.md), [r204a_pairs.md](r204a_pairs.md),
[r204a_pairs_tol1.md](r204a_pairs_tol1.md), [r204a_work.md](r204a_work.md), [r204a_clocks.tsv](r204a_clocks.tsv),
[r204a_host.tsv](r204a_host.tsv), [r204a_reallocs.tsv](r204a_reallocs.tsv), [o4g2a_gonogo.md](o4g2a_gonogo.md); the
controls' rows as in parts (5) and (8)). A ratio is RUN / REF on the same requests; the percentages are this run against
`o4c2` and against `o4c3`.

- **Decode at the same context** ([r204a_decode.md](r204a_decode.md), from the published `turns.csv`, with
  [decode_by_context.py](../2026-11-18_c854b_p106_check/run/decode_by_context.py)). Each request is put in the bucket of
  its middle context (its start plus half the tokens it generated). In every bucket a decode step took less time with
  the switch:

  | middle context, tokens | `o4g2a` ms a token (requests) | `o4c2` | `o4c3` | `o4g2a` against `o4c2`, `o4c3` |
  |---|---:|---:|---:|---|
  | under 3,000 | 53.06 (12) | 56.77 (12) | 56.88 (12) | −6.5 %, −6.7 % |
  | 3,000 to 4,500 | 54.92 (17) | 58.94 (14) | 58.87 (15) | −6.8 %, −6.7 % |
  | 4,500 to 6,000 | 56.76 (20) | 60.51 (23) | 60.52 (22) | −6.2 %, −6.2 % |
  | 6,000 to 8,000 | 58.74 (12) | 61.88 (10) | 62.39 (19) | −5.1 %, −5.9 % |
  | 8,000 and over | 60.64 (1) | 66.08 (4) | 66.18 (4) | |

  The deepest bucket holds one request of this run (Python `scale-generator` turn 6, 388 tokens from 7,934 tokens of
  context), so its 60.64 ms sits lower in the bucket than the controls' requests and is not compared. On the deep lane
  (C865b, the second greedy pass of the first load) a step took 50.01 against 53.64 ms at depth 0 (−6.8 %) and 60.16
  against 64.01 at 8,192 (−6.0 %).
  Weighted by the tokens all three runs generated in each bucket, the runs decode at 57.19, 61.18 and 61.28 ms a token.
  With the day's two `1f1110c67` runs added, 57.07 against 60.94 and 61.05 (the controls), and 61.08 and 60.90
  ([r204a_decode5.md](r204a_decode5.md)).
- **Generation over the whole session** (+8.3 % and +9.1 %) is a little above the same-mix figure because this run's
  replies sat shallower: 388 tokens generated past 8,000 tokens of context, against 4,320 and 5,042, where every run
  decodes slowest.
- **Minutes a task at equal work** price `ka`'s work at each session's prompt and generation tok/s. Both are session
  averages; with the prompt speed unchanged, the −6.0 % and −7.7 % come from generation.
- **Prompt reading** ([r204a_pairs.md](r204a_pairs.md), each pair's rows in `r204a_paired_*.md`): on the 19 requests
  read on the same prompt in all three runs, this run took 194.86 s, between the controls' 195.22 and 194.05 s (9 of 19
  slower than `o4c2`, 7 of 19 slower than `o4c3`). On the 18 with a cached prefix 166.14 s against 167.40 and 166.18.
  With the cached tokens matched within one ([r204a_pairs_tol1.md](r204a_pairs_tol1.md)), 22 requests: 224.03 against
  224.50 and 223.32 s. The session's cold first request (Python `affine-cipher` turn 1, 2,742 tokens, nothing cached)
  took 28.72 s, against 27.82 and 27.87 s; on the same rig that day `1f1110c67`'s first runs took 29.19 and 27.83 s, so
  that request varies by more than this run's difference. Prompt tok/s over the session (98.5 against 96.2 and 92.0)
  follows each session's mix of prompt sizes.
- **The work:** 3,235 prompt tokens read a task. Three replies were read again (Python `pig-latin` turns 4 and 5, 565
  and 1,859 tokens; Python `scale-generator` turn 4, 982 tokens), 31.2 s of the 394.3 s of prompt reading. The server
  restored 14 checkpoints, created or erased none and re-processed nothing in full; the report's cache hit is 85.2 %
  ([o4g2a_report_table.md](o4g2a_report_table.md)). One tool call in `affine-cipher` returned an error to the agent; none
  were malformed. Graph reallocations: 0 in 62 requests ([r204a_reallocs.tsv](r204a_reallocs.tsv)). No NaN or inf, no
  watchdog kill ([o4g2a_gonogo.md](o4g2a_gonogo.md)).
- **Clocks** ([r204a_clocks.tsv](r204a_clocks.tsv)): busy medians 1,847 1,847 1,898 1,898 1,860 1,911 1,873 1,911 1,873
  MHz on cards 0–8; `o4c3`'s card for card but card 2 (1,885), `o4c2`'s but cards 2 and 3 (1,885 and 1,885). The share
  of busy samples below 1,800 MHz is higher on every card but card 7 (card 1: 8.53 % against 6.49 and 5.79 %); not
  traced.
- **Host** ([r204a_host.tsv](r204a_host.tsv)): lowest MemAvailable 1,223 MiB at 19:11:57, during the last task but
  one, and 2,872 MiB at the end of the log; swap 30 MiB at the start and 343 MiB at its highest at 18:50:26, during the
  load. The RAM watchdog stopped nothing. The report's memory gate ([o4g2a_gonogo.md](o4g2a_gonogo.md)) reads the
  client's log shared with earlier runs: NO-GO on its 718 MB low and its 343 MB swap, as for every run in that folder.
- **Card memory:** the peaks in the NVML card log are 5,808 5,828 5,754 5,828 5,436 5,544 5,706 5,782 5,188 MiB on cards
  0–8. That is 22 18 22 18 18 18 18 14 14 MiB below both controls, which peaked at the same MiB as each other. It is
  `1f1110c67`'s first run (part (7)) to the MiB but card 5 (2 MiB lower). So with the fused launches off, the
  candidate's extra peak memory against `1f1110c67` (parts (7) to (9), "Not checked here") is gone (Q1601).
- **The client machine** also ran another project's benchmark containers for the whole run. The prompt and generation
  times are the rig server's own and are not affected; the minutes and the first-token times are measured on the
  client.

**The verdict waits for run 2 (R204b):** the same launcher, arm `o4g2b`, on the same tree and rig.

## Files

| File | What |
|---|---|
| [run/ornith_agent_glu.sh](run/ornith_agent_glu.sh) | the launcher: part (5)'s `run/ornith_agent.sh` with `GGML_CUDA_MMID_GLU_UNFUSE=2` added to the server's environment |
| [run/r204a_tables.sh](run/r204a_tables.sh) | every table of run 1 from the published per-request rows and the run's logs |
| [agent/o4g2a/turns.csv](agent/o4g2a/turns.csv), [agent/o4g2a/attempts.csv](agent/o4g2a/attempts.csv) | the run's per-request rows (server times, prompt and cached tokens, generated tokens) and per-attempt rows (result, turns, tool calls), as the agent check's report writes them |
| [o4g2a_report_table.md](o4g2a_report_table.md), [o4g2a_compare.md](o4g2a_compare.md), [o4g2a_gonogo.md](o4g2a_gonogo.md) | the agent check's own report after this run (over every run in the client's results folder) |
| [r204a_agent.tsv](r204a_agent.tsv) | every Ornith arm of the client's results folder side by side, against `v3ka`: passed, turns, prompt and generation tok/s, first-token median and p90, minutes, minutes a task at equal work |
| [r204a_decode.md](r204a_decode.md), [r204a_decode5.md](r204a_decode5.md) | decode speed by middle context and at one shared mix, the three runs on this tree, and with the day's two `1f1110c67` runs |
| [r204a_same_prompts.md](r204a_same_prompts.md), [r204a_same_prompts_tol1.md](r204a_same_prompts_tol1.md) | the requests read on the same prompt in the three runs, side by side, exactly and with the cached tokens within one |
| `r204a_paired_*.md`, `r204a_paired_*_tol1.md` | request by request: this run against `o4c2` (`c2_g2a`) and `o4c3` (`c3_g2a`), and `o4c3` against `o4c2` (`c2_c3`), on the requests matched in all three |
| [r204a_pairs.md](r204a_pairs.md), [r204a_pairs_tol1.md](r204a_pairs_tol1.md) | the three pairs' totals in one table |
| [r204a_work.md](r204a_work.md), [r204a_work.tsv](r204a_work.tsv) | each run's work: prompt tokens read, tokens generated, replies re-read, the longest first-token wait |
| [r204a_clocks.tsv](r204a_clocks.tsv) | each card's busy SM clock median in each run's task window |
| [r204a_host.tsv](r204a_host.tsv) | lowest MemAvailable, swap and each card's peak memory in this run |
| [r204a_reallocs.tsv](r204a_reallocs.tsv) | graph reallocations by prompt size in the three runs |

## Commands

From the repository root on the machine that drives the rig (here also the agent client), with the rig clean (no
llama process, 0 MiB on every card):

```bash
NEW=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483
# the run (~11 min load, 12 tasks), then its report; CLIENT_OUT is the client folder the control runs used
RIG=$RIG CLIENT=local ARM=o4g2a MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf BIN=/opt/llama-builds/$NEW/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r160 OUT=<out>/agent-o4g2a \
  bash docs/data/2026-11-18_r204_ornith_glu_agent/run/ornith_agent_glu.sh
RIG=$RIG CLIENT=local ARM=o4g2a MODEL=<the same> BIN=<the same> RIG_LOGDIR=<the same> CLIENT_DIR=<the same> \
  CLIENT_OUT=<the same> OUT=<out>/agent STEPS="report collect" bash docs/data/2026-11-18_r204_ornith_glu_agent/run/ornith_agent_glu.sh

# the tables, on a folder holding r207b_tables.sh's inputs plus o4g2a's rows, server log, card log and host log
# (run/r204a_tables.sh lists them); the window is o4g2a's tasks, from the client's first request
O4G2A_FROM="2026-11-18 18:51:07" O4G2A_TO="2026-11-18 19:13:52" \
  bash docs/data/2026-11-18_r204_ornith_glu_agent/run/r204a_tables.sh <tables folder>
```

`agent/o4g2a/` is the client's `res/turns.csv` and `res/attempts.csv` rows of the arm. `r204a_agent.tsv` is the script's `agent.tsv`. The card log is the rig's
`cards.csv` from this run's own `card_log start` line on; the host log is the rig's `meminfo.log` from the server's
start.

## Provenance and licence

The tasks are the agent check's 12 bug fixes, derived from Exercism's Python and JavaScript exercises (MIT; source,
licences and commits in [bench/agent_check](../../../bench/agent_check/README.md)); this folder holds only their IDs
(in [agent/o4g2a/attempts.csv](agent/o4g2a/attempts.csv)). The model file is Ornith-1.5-122B Pascal N2P GGUF
(`valid1`), as in [C865b](../2026-11-18_c865b_ornith_glu_rig1/README.md). The tables summarize this project's own runs
and are published under the repository's licence.
