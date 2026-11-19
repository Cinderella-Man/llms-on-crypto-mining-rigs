# R208 and R210: two switches for Flash-Next's v5 preset, on the agent check (2026-11-19)

Registry rows [Q1602](../../../quants/experiments.md) (R208, the K-quant prompt kernels' scale multiply in FP32) and
[Q1604](../../../quants/experiments.md) (R210, text reuse). Both switches are already in the v5 engine; this folder
decides, on the coding-agent check, whether either goes into Flash-Next's v5 preset.

**Runs 1 and 2 of 6 done.** The control ran as the same preset did the day before: on the v4.0 preset and the v5
engine, 11 of 12 tasks passed (JavaScript `complex-numbers` ran to the 12-minute cap again), a task at equal work took
2.42 minutes against 2.40 the day before on the same rig, and the model generated at 26.98 tok/s against 27.00. With the
FP32 scale (run 2) all 12 passed and a task at equal work took 2.38 minutes; the 13 requests that read the same new
tokens on the same cached prefix in every run took 98.16 s against the control's 98.49 s, less than the difference on
one of them. The two switches are judged after all six runs, against this day's two controls.

## What runs

Six runs of Flash-Next's agent check ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 40-minute budget) on one rig and one boot, alternated so
that both switches share the two control runs and each switch has one run in each half of the day:

| # | Run | Arm | What is added to the v4.0 preset | Status |
|---|---|---|---|---|
| 1 | R208a | `f5c1` | nothing (the control) | done 10:44–11:24 |
| 2 | R208b | `f5s1` | `GGML_CUDA_MMQ_KQ_FP32SCALE=1 GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all` | done 11:33–11:59 |
| 3 | R210a | `f5t1` | `LLAMA_SERVER_TEXT_REUSE=1` | prepared |
| 4 | R208c | `f5c2` | nothing (the control) | planned |
| 5 | R208d | `f5s2` | `GGML_CUDA_MMQ_KQ_FP32SCALE=1 GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all` | planned |
| 6 | R210b | `f5t2` | `LLAMA_SERVER_TEXT_REUSE=1` | planned |

- **The FP32 scale** ([C861](../2026-11-17_c861a_fp32_scale/README.md), Q1549 (a)): the K-quant prompt kernels
  multiply each block's scale in one FP32 instruction instead of three integer ones; `TYPES=all` applies it to Q3_K
  too. On a P106 it gives the same greedy reply, and with `all` `llama-bench` read prompts at micro-batches of 512
  0.28, 0.42 and 0.84 % faster (64, 128 and 512 tokens). The preset reads prompts longer than 600 tokens in
  micro-batches of 512.
- **Text reuse** (`LLAMA_SERVER_TEXT_REUSE=1`, Q260): when the next prompt spells the same text as the cached tokens
  but splits part of the previous reply into other tokens than the ones the model generated, the server keeps the
  cached tokens instead of reading that reply again. On Flint it removed every reply re-read in two runs
  ([R17a, R17b](../2026-11-17_r17_flint_textreuse/README.md)). Flash-Next re-read a reply on 7 turns (51.6 of 413.9
  prompt seconds) in [C854](../2026-11-18_c854b_p106_check/README.md)'s run, and on 2, 1, 5, 7, 4, 2, 2, 8 and 2
  turns in nine v4-era runs (C854's `flash_rereads_runs.tsv`).

Every run: the maker's Swift GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, the n-gram
table in the third shard) with the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`) at 65,536 tokens on nine P106-090
cards, the released v4.0 preset ([release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env))
through R156's driver ([r156.sh](../2026-11-13_r156_agent_v4r/run/r156.sh), the harness's arm `v4r`), on the v5 tree
`pascal-p106-v5` `602691153` (`llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`). The rig's cards run at stock
(75 W, 1,911 / 4,004 MHz maximum, no offset), and the machine that drives the rig is the agent's client.
[run/flash_switch.sh](run/flash_switch.sh) adds the arm's switch to the driver's `V4R_ENV`; the control leaves it at
the driver's default, so its started command is C854's byte for byte.

**The context run** is C854's `v4rc2` (2026-11-18, [part (6)](../2026-11-18_c854b_p106_check/README.md#part-6-the-agent-check-on-flash-next-2026-11-18)):
the same rig, tree, preset and client on the day before. It is shown beside each run for context; the verdicts
compare the switches with this day's two controls only, after all six runs.

## Run 1, `f5c1` (R208a): the control, 10:44–11:24

**The v4.0 preset on the v5 engine ran as it did the day before.** 11 of 12 tasks passed; JavaScript `complex-numbers`
ran to the 12-minute cap again, after 38 turns. A task at equal work took 2.42 minutes against C854's 2.40, and the
model generated at 26.98 tok/s against 27.00. The 12 first turns, the same requests in both runs, took 95.97 s
against 94.62 s; one of them, JavaScript `variable-length-quantity`'s, took 4.82 s against 3.66 s, and the other 11
took 91.15 s against 90.95 s. The server read a previous reply again on 4 turns (1,806 tokens, 23.6 of the run's
427.2 prompt seconds), against 7 turns (4,854 tokens, 51.6 of 413.9 s) in C854's run.

Its started command (1,689 bytes, md5 `4e11746b52bcc136ef186ffe9c161aa0`) is C854's byte for byte, and the server's
`switches:` line names the same 39 switches with the same values. The server started at 10:44:40 and was up after
312 s (C854: 311 s); the tasks ran 10:49:55–11:23:41 and the report ended at 11:24:03. Every card was at stock before
and after (75 W, 1,911 / 4,004 MHz maximum, no offset), with no Xid line in the kernel log.

| | C854 `v4rc2` (2026-11-18, context) | **`f5c1` (R208a)** |
|---|---|---|
| passed | 11 / 12 | **11 / 12** (`complex-numbers` at the cap, 38 turns) |
| minutes a task at equal work (R90 `v3`'s work) | 2.40 | **2.42** |
| the same without `complex-numbers` | 2.44 | **2.44** |
| the 12 first turns, the same requests in both runs, prompt s | 94.62 | **95.97** |
| the same without JavaScript `variable-length-quantity`'s first turn (3.66 and 4.82 s) | 90.95 | **91.15** |
| all 16 requests that read the same prompt on the same cached prefix, prompt s | 127.85 | **128.91** |
| prompt tok/s, the whole session | 139.2 | **133.7** |
| the same without `complex-numbers` | 146.1 | **141.7** |
| generation tok/s | 27.00 | **26.98** |
| the same without `complex-numbers` | 27.99 | **28.07** |
| first token median s; p90 s | 2.52; 9.54 | **2.41; 9.04** |
| first token on a task's first turn, median s | 6.69 | **6.57** |
| the longest first-token wait s | 28.82 | **27.97** |
| prompt tokens read a task | 4,802 | **4,760** |
| the same without `complex-numbers` | 3,516 | **3,499** |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 7; 4,854; 51.6 of 413.9 | **4; 1,806; 23.6 of 427.2** |
| draft acceptance (accepted of drafted tokens) | 0.740 | **0.732** |
| minutes, all 12 tasks; turns | 32.6; 92 | **33.6; 104** |
| graph reallocations | 0 in 91 requests | **0 in 103** |
| busy SM clock, mean of the nine cards' medians, MHz | 1,878.3 | **1,885.3** |
| lowest MemAvailable MiB; swap highest MiB | 387; 464 | **285; 19** |

Each value is one run ([agent.tsv](agent.tsv), [agent_nocx.tsv](agent_nocx.tsv), [same_prompts.md](same_prompts.md),
[work.md](work.md), [work_nocx.tsv](work_nocx.tsv), [first_token.tsv](first_token.tsv),
[rereads_runs.tsv](rereads_runs.tsv), [clocks.tsv](clocks.tsv), [host.tsv](host.tsv), [reallocs.tsv](reallocs.tsv)).
C854's figures are part (6)'s published ones; this folder's scripts give the same agent, matched-prompt and work
figures from its published per-request tables. The 16 matched requests and the 12 first turns are those two runs'
match; with run 2 added, [same_prompts.md](same_prompts.md) matches the requests that read the same tokens in all
three runs (13, run 2's section). No class is given here: the switches are judged against this run and the second
control.

- **The task at the cap.** `complex-numbers` ran 38 turns and 37 tool calls (720 s) with its tests failing, as in
  C854's run (34 turns). The other 11 tasks passed. Without it the two runs did nearly the same work: 3,499 against
  3,516 prompt tokens a task.
- **The longest first-token wait** is the run's first request in both runs, `affine-cipher`'s first turn, which reads
  its whole 2,780-token prompt with nothing cached: 27.97 s against 28.82 s. `pig-latin`'s first turn again read its
  whole 2,324-token prompt (12.66 s against 12.34 s), because the long `complex-numbers` task before it pushed the
  shared system prefix's checkpoint out of the four kept on the cards (the server log: "no checkpoint below 1553").
- **The reply re-reads** ([rereads_turns.tsv](rereads_turns.tsv)): `complex-numbers` turn 34 (a reply of 200 tokens;
  the prompt stopped matching 139 tokens in, inside the tool call) and `pig-latin` turns 3, 7 and 8 (1,291, 128 and
  187 tokens; 266, 85 and 136 tokens in, before the tool call). In all 4 the client sent the reply back as the server
  gave it. `pig-latin` turn 3 re-read in C854's and both R156 runs too.
- **First token** follows the turn mix: the turns after a task's first had a median of 2.27 s against 2.36 s and a
  p90 of 9.03 s against 9.54 s.
- **Clocks** ([clocks.tsv](clocks.tsv)): busy medians 1,860 1,847 1,898 1,898 1,873 1,911 1,885 1,911 1,885 MHz (cards
  0–8), against 1,847 1,847 1,885 1,898 1,860 1,911 1,873 1,911 1,873 in C854's run.
- **Host** ([host.tsv](host.tsv)): the lowest MemAvailable was 285 MiB at 11:23:51, 10 s after the last task ended
  and before the harness stopped the server; during the tasks it was 293 MiB at 11:20:16. Swap stayed at 19 MiB from the
  start (C854: 30 MiB at the start, 464 at its highest). The RAM watchdog stopped nothing. Card peaks 5,190 5,234
  5,092 5,250 4,348 5,108 4,502 5,120 5,698 MiB, within 2 MiB of C854's on every card.

## Run 2, `f5s1` (R208b): the FP32 scale, 11:33–11:59

**The requests both runs read alike took nearly the same time; the run did other work than the control.** 12 of 12
tasks passed: JavaScript `complex-numbers`, at the cap after 38 turns in run 1, passed in 5 turns (64 s), and Python
`pig-latin` took 24 turns (393 s) against 8 (142 s). The 13 requests that read the same new tokens on the same cached
prefix in all three runs took 98.16 s against the control's 98.49 s; 7 of them were faster and 6 slower, and the whole
difference is smaller than one request's: `scale-generator`'s first turn took 7.45 s against 7.92 s (7.20 s in C854's
run), and without it the other 12 took 90.71 s against 90.56 s. The longest of them, `affine-cipher`'s first turn,
which reads its whole 2,780-token prompt in micro-batches of 512 with nothing cached, took 27.95 s against 27.93 s. A
task at equal work took 2.38 minutes against 2.42, and 2.47 against 2.44 without `complex-numbers`.

Its started command (1,755 bytes, md5 `9b73d0302aa9ca72b5ac7f9f637f3b80`) equals [run/dry_line.sh](run/dry_line.sh)'s
line for `f5s1` byte for byte: the control's with `GGML_CUDA_MMQ_KQ_FP32SCALE=1 GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all`
after the driver's last switch. The server's `switches:` line names 41: the control's 39 with the same values and these
two. The server started at 11:33:13 and was up after 312 s; the tasks ran 11:38:28–11:59:37 and the report ended at
11:59:49. Every card was at stock before and after (75 W, 1,911 / 4,004 MHz maximum, no offset), with no Xid line.

| | C854 `v4rc2` (context) | `f5c1` (run 1) | **`f5s1` (R208b)** |
|---|---|---|---|
| passed | 11 / 12 | 11 / 12 (`complex-numbers` at the cap, 38 turns) | **12 / 12** (`complex-numbers` in 5 turns) |
| minutes a task at equal work (R90 `v3`'s work) | 2.40 | 2.42 | **2.38** |
| the same without `complex-numbers` | 2.44 | 2.44 | **2.47** |
| the 13 requests that read the same new tokens on the same cached prefix in all three runs, prompt s | 97.67 | 98.49 | **98.16** |
| of them, the 11 first turns (all but `pig-latin`'s) | 82.28 | 83.30 | **83.04** |
| of them, `affine-cipher`'s first turn (2,780 tokens, nothing cached), prompt s | 28.57 | 27.93 | **27.95** |
| `pig-latin`'s first turn: tokens read; prompt s | 2,324; 12.34 | 2,324; 12.66 | **775; 6.41** (the shared system prefix still cached) |
| prompt tok/s, the whole session | 139.2 | 133.7 | **137.1** |
| the same without `complex-numbers` | 146.1 | 141.7 | **132.8** |
| generation tok/s | 27.00 | 26.98 | **27.85** |
| the same without `complex-numbers` | 27.99 | 28.07 | **27.90** |
| first token median s; p90 s | 2.52; 9.54 | 2.41; 9.04 | **2.13; 8.26** |
| first token on a task's first turn, median s | 6.69 | 6.57 | **6.55** |
| the longest first-token wait s | 28.82 | 27.97 | **27.99** (`affine-cipher` turn 1) |
| prompt tokens read a task | 4,802 | 4,760 | **3,541** |
| the same without `complex-numbers` | 3,516 | 3,499 | **3,428** |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 7; 4,854; 51.6 of 413.9 | 4; 1,806; 23.6 of 427.2 | **3; 2,071; 19.8 of 309.9** |
| draft acceptance (accepted of drafted tokens) | 0.740 | 0.732 | **0.752** |
| minutes, all 12 tasks; turns | 32.6; 92 | 33.6; 104 | **21.0; 82** |
| graph reallocations | 0 in 91 requests | 0 in 103 | **0 in 82** |
| busy SM clock, mean of the nine cards' medians, MHz | 1,878.3 | 1,885.3 | **1,885.3** |
| lowest MemAvailable MiB; swap highest MiB | 387; 464 | 285; 19 | **295; 19** |

Each value is one run, from the same files as run 1's table; the first-turn rows are from `agent/<arm>/turns.csv`. No
class is given here.

- **Where the switch can act.** It changes only the K-quant prompt kernels of Flash-Next's dense layers, and on a
  P106 `llama-bench` measured it faster only at micro-batches of 512, which the preset uses for prompts longer than 600
  tokens. The matched requests and `affine-cipher`'s first turn are where it shows; the session's prompt tok/s also depends on which prompts a run reads and at what
  depth, and this run read other ones (59 prompts of 600 tokens or fewer against 72, 23 longer against 31).
- **The work.** Without `complex-numbers` this run read 3,428 prompt tokens a task against 3,499 and generated 2,248
  against 2,552, over 77 requests against 66 ([work_nocx.tsv](work_nocx.tsv)); `pig-latin`'s 24 turns are most of the
  difference.
- **`pig-latin`'s first turn** read 775 tokens on the 1,549-token shared system prefix: the short `complex-numbers`
  task before it left that prefix's checkpoint among the four kept on the cards (run 1's server log had "no checkpoint
  below 1553" there; this run's has none).
- **The reply re-reads** ([rereads_turns.tsv](rereads_turns.tsv)): `pig-latin` turns 4 and 21 (replies of 552 and
  1,450 tokens; the prompt stopped matching 352 and 906 tokens in, before the tool call) and JavaScript
  `variable-length-quantity` turn 2 (69 tokens; 23 in, inside the tool call). The client sent two replies back as the
  server gave them and the 1,450-token one with its tool call's arguments re-serialized, after the point where the
  prompt stopped matching. `pig-latin` turn 3, which re-read in run 1, C854's run and both R156 runs, did not.
- **Clocks** ([clocks.tsv](clocks.tsv)): busy medians 1,860 1,847 1,898 1,898 1,873 1,911 1,885 1,911 1,885 MHz (cards
  0–8), the same as run 1's on every card.
- **Host** ([host.tsv](host.tsv)): the lowest MemAvailable was 295 MiB at 11:56:21, during `pig-latin`; swap stayed at
  19 MiB. The RAM watchdog stopped nothing. Card peaks 5,190 5,234 5,092 5,250 4,348 5,108 4,502 5,120 5,698 MiB, the
  same as run 1's on every card.

## Run 3, `f5t1` (R210a): text reuse

Not run yet. Its launcher's server line is the control's with `LLAMA_SERVER_TEXT_REUSE=1` after the driver's last
switch (1,715 bytes).

## Runs 4 to 6, `f5c2`, `f5s2`, `f5t2`

Not run yet.

## The verdicts

After the six runs: each switch against the two controls, by R156's rule (a gain or a loss when the difference from
the controls' mean is larger than their spread, negligible otherwise), on a task at equal work, the matched requests'
prompt time and generation; for text reuse also the reply re-reads, their prompt seconds and the longest first-token
wait.

## Files

| File | What |
|---|---|
| `agent/<arm>/turns.csv`, `agent/<arm>/attempts.csv` | each run's per-request and per-task tables (the client's `res/` files, the harness's arm `v4r` renamed by [run/rename_arm.py](run/rename_arm.py)) |
| [agent.tsv](agent.tsv) | per run: tasks passed, turns, prompt and generation tok/s, first token median and p90 and on a task's first turn, the draft's acceptance, minutes, a task at equal work (R90 `v3`'s work); R90 `v3` and C854 `v4rc2` as context rows |
| [same_prompts.md](same_prompts.md) | the requests that read the same new tokens on the same cached prefix in every run, each run's prompt ms, and each run's decode speed |
| [work.md](work.md), [work.tsv](work.tsv) | per run: prompt tokens read and generated a task, and every turn's cache path (reply re-reads, older checkpoints) |
| [first_token.tsv](first_token.tsv) | per run: the longest first-token wait (task, turn, tokens read and cached, prompt ms) and the later turns' median and p90 |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv) | C854's [reply_rereads.py](../2026-11-18_c854b_p106_check/run/reply_rereads.py): per run the turns that read the previous reply again, their tokens and prompt seconds, and per such turn where inside the reply the prompt stopped matching |
| [clocks.tsv](clocks.tsv) | each card's busy SM clock median per run, over the tasks' window |
| [host.tsv](host.tsv) | lowest MemAvailable, swap at the start and its highest, MemAvailable at the end, each card's peak memory |
| [reallocs.tsv](reallocs.tsv) | the target graph's reallocations per prompt-size bucket ([scripts/realloc_counts.py](../../../scripts/realloc_counts.py)) |
| [run/flash_switch.sh](run/flash_switch.sh) | the run: the arm's `V4R_ENV`, then C854's [flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh) |
| [run/dry_line.sh](run/dry_line.sh) | the server line a run would start, built by the unchanged driver chain without a load |
| [run/tables.sh](run/tables.sh) | every table above from the runs' logs |
| [run/rename_arm.py](run/rename_arm.py) | the client's tables with the arm renamed |

## Commands

From the repository root on the machine that drives the rig, with the rig clean; `ARM` is the run's arm, and every
run gets its own `RIGDIR` and `CLIENT_OUT` (the driver skips attempts it finds finished in its client folder):

```bash
ARM=f5c1 RIG=$RIG CLIENT=local \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483 REF=llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828 \
  RIGDIR=<rig folder> OUT=<out> RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/<run> \
  bash docs/data/2026-11-19_r208_r210_flash_switches/run/flash_switch.sh
```

`REF` is required by `r156.sh` and not run. The server line a prepared run would start, without a load:

```bash
ARM=f5s1 WORK=<an empty scratch folder> NEW=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483 \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf bash docs/data/2026-11-19_r208_r210_flash_switches/run/dry_line.sh
```

The tables, standard library only, once a run's client folder and rig logs are back (`<client folder>/<run>/` holds
`res/`, `rig/` with `server-v4r.log`, `meminfo.log` and `cards.csv`, `proxy.jsonl` and `proxy.jsonl.bodies.gz`):

```bash
python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <client folder>/<run>/res \
  docs/data/2026-11-19_r208_r210_flash_switches/agent/<arm> v4r <arm>
# TABLES: rig/server-<arm>.log, rig_<arm>/{meminfo.log,cards.csv}, proxy_<arm>/{proxy.jsonl,proxy.jsonl.bodies.gz},
# window_<arm> (the first task's start and the last task's end), and C854's and R90's server logs as rig/server-v4rc2.log
# and rig/server-r90v3.log
bash docs/data/2026-11-19_r208_r210_flash_switches/run/tables.sh <TABLES> f5c1 [f5s1 f5t1 ...]
```

Given C854's own logs and tables as a stand-in run, `tables.sh` gives part (6)'s published figures for it exactly.

## Provenance and licence

Measured on the project's own rig on 2026-11-19 with the repository's own scripts; the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/<arm>/attempts.csv`). No
model file or task text is in this folder. The tables and scripts here are under the repository's licence.
