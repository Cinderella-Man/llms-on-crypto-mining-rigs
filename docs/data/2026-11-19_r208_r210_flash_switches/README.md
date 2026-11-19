# R208 and R210: two switches for Flash-Next's v5 preset, on the agent check (2026-11-19)

Registry rows [Q1602](../../../quants/experiments.md) (R208, the K-quant prompt kernels' scale multiply in FP32) and
[Q1604](../../../quants/experiments.md) (R210, text reuse). Both switches are already in the v5 engine; this folder
decides, on the coding-agent check, whether either goes into Flash-Next's v5 preset.

**Runs 1 to 5 of 6 done; the FP32 scale is judged: negligible.** Its two runs read the 12 requests that every run read
alike in 91.27 and 91.01 s, against the two controls' 91.56 and 92.43 s: 0.85 s (0.93 %) faster on the means, while the
controls differ from each other by 0.87 s (0.95 %). It was slower on no measure, so it stays in the engine behind its
switch, with the other negligible gains, and does not join the preset by itself ([the verdicts](#the-verdicts)). The control ran as the same preset did the day before: on the v4.0 preset and the v5
engine, 11 of 12 tasks passed (JavaScript `complex-numbers` ran to the 12-minute cap again), a task at equal work took
2.42 minutes against 2.40 the day before on the same rig, and the model generated at 26.98 tok/s against 27.00. With the
FP32 scale (run 2) all 12 passed and a task at equal work took 2.38 minutes; the 13 requests that read the same new
tokens on the same cached prefix in the first three runs took 98.16 s against the control's 98.49 s, less than the
difference on one of them. With text reuse (run 3) no turn read an earlier reply again, against 4 turns in the control
and 3 in run 2; it kept the cached tokens on 32 requests in two tasks, and a task at equal work took 2.43 minutes. The
second control (run 4) passed all 12 tasks; a task at equal work took 2.37 minutes, and the 12 requests that read the
same new tokens on the same cached prefix in every run took 92.43 s against the first control's 91.56 s. How far the two
controls differ is the spread each switch is judged against: 2.07 % on a task at equal work, 0.95 % on the matched
requests and 1.59 % in generation (run 4's last column). The FP32 scale's second run (run 5) passed all 12 tasks; a task
at equal work took 2.35 minutes. Text reuse is judged after its second run (run 6), against the same two controls.

## What runs

Six runs of Flash-Next's agent check ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 40-minute budget) on one rig and one boot, alternated so
that both switches share the two control runs and each switch has one run in each half of the day:

| # | Run | Arm | What is added to the v4.0 preset | Status |
|---|---|---|---|---|
| 1 | R208a | `f5c1` | nothing (the control) | done 10:44–11:24 |
| 2 | R208b | `f5s1` | `GGML_CUDA_MMQ_KQ_FP32SCALE=1 GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all` | done 11:33–11:59 |
| 3 | R210a | `f5t1` | `LLAMA_SERVER_TEXT_REUSE=1` | done 12:44–13:18 |
| 4 | R208c | `f5c2` | nothing (the control) | done 13:59–14:33 |
| 5 | R208d | `f5s2` | `GGML_CUDA_MMQ_KQ_FP32SCALE=1 GGML_CUDA_MMQ_KQ_FP32SCALE_TYPES=all` | done 14:43–15:11 |
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

## Run 3, `f5t1` (R210a): text reuse, 12:44–13:18

**No turn read an earlier reply again.** The server read no previous reply a second time (0 turns, 0 of the run's 404.0
prompt seconds), against 4 turns (1,806 tokens, 23.6 of 427.2 s) in the control and 3 (2,071 tokens, 19.8 of
309.9 s) in run 2. The switch kept the cached tokens on 32 requests in two tasks. In Python `scale-generator` the third
prompt split the second reply (5,058 tokens) into other tokens 3,647 tokens in; in JavaScript `complex-numbers` the
thirteenth prompt did so 152 tokens into the twelfth reply (697 tokens). Both points lie in the reply's text before its
tool call, and the client sent both replies back as the server gave them. 11 of 12 tasks passed: `complex-numbers` failed
again, stopped at the 40-turn cap after 41 turns (537 s) where run 1 ran to the 12-minute cap after 38. A task at equal
work took 2.43 minutes against the control's 2.42, and the model generated at 27.07 tok/s against 26.98. The 12
requests that read the same new tokens on the same cached prefix in all four runs took 92.23 s against 91.56 s
(control) and 91.27 s (run 2); text reuse did not act on any of them.

Its started command (1,715 bytes, md5 `5f0cbde4375a316aafaf49ba4660efcc`) equals [run/dry_line.sh](run/dry_line.sh)'s
line for `f5t1` byte for byte: the control's with `LLAMA_SERVER_TEXT_REUSE=1` after the driver's last switch. The
server's `switches:` line names 40: the control's 39 with the same values and this one. The server started at 12:45:00
and was up after 311 s; the tasks ran 12:50:14–13:17:51 and the report ended at 13:18:05. Every card was at stock before
and after (75 W, 1,911 / 4,004 MHz maximum, no offset), with no Xid line. A first try at 12:07 lost its client at
12:14: an agent check for the other rig started on the same client machine with the same harness arm, and its client,
when it starts, removes the proxy container named for that arm, which was this run's proxy. That try was stopped at
12:43 and the run started again with containers named for the run (Q1619); nothing of the first try is in these tables.

| | C854 `v4rc2` (context) | `f5c1` (run 1) | `f5s1` (run 2) | **`f5t1` (R210a)** |
|---|---|---|---|---|
| passed | 11 / 12 | 11 / 12 (`complex-numbers` at the time cap, 38 turns) | 12 / 12 (`complex-numbers` in 5 turns) | **11 / 12** (`complex-numbers` at the turn cap, 41 turns) |
| minutes a task at equal work (R90 `v3`'s work) | 2.40 | 2.42 | 2.38 | **2.43** |
| the same without `complex-numbers` | 2.44 | 2.44 | 2.47 | **2.45** |
| the 12 requests that read the same new tokens on the same cached prefix in all four runs, prompt s | 90.73 | 91.56 | 91.27 | **92.23** |
| of them, the 11 first turns (all but `pig-latin`'s) | 82.28 | 83.30 | 83.04 | **83.98** |
| of them, `affine-cipher`'s first turn (2,780 tokens, nothing cached), prompt s | 28.57 | 27.93 | 27.95 | **28.13** |
| `pig-latin`'s first turn: tokens read; prompt s | 2,324; 12.34 | 2,324; 12.66 | 775; 6.41 | **2,324; 12.06** |
| prompt tok/s, the whole session | 139.2 | 133.7 | 137.1 | **129.3** |
| the same without `complex-numbers` | 146.1 | 141.7 | 132.8 | **141.8** |
| generation tok/s | 27.00 | 26.98 | 27.85 | **27.07** |
| the same without `complex-numbers` | 27.99 | 28.07 | 27.90 | **28.00** |
| first token median s; p90 s | 2.52; 9.54 | 2.41; 9.04 | 2.13; 8.26 | **2.27; 8.89** |
| first token on a task's first turn, median s | 6.69 | 6.57 | 6.55 | **6.69** |
| the longest first-token wait s | 28.82 | 27.97 | 27.99 | **28.19** (`affine-cipher` turn 1) |
| prompt tokens read a task | 4,802 | 4,760 | 3,541 | **4,354** |
| the same without `complex-numbers` | 3,516 | 3,499 | 3,428 | **3,309** |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 7; 4,854; 51.6 of 413.9 | 4; 1,806; 23.6 of 427.2 | 3; 2,071; 19.8 of 309.9 | **0; 0; 0.0 of 404.0** |
| requests on which text reuse kept the cached tokens (tasks; points) | — | — | — | **32 (2; 2)** |
| draft acceptance (accepted of drafted tokens) | 0.740 | 0.732 | 0.752 | **0.735** |
| minutes, all 12 tasks; turns | 32.6; 92 | 33.6; 104 | 21.0; 82 | **27.5; 103** |
| graph reallocations | 0 in 91 requests | 0 in 103 | 0 in 82 | **0 in 102** |
| busy SM clock, mean of the nine cards' medians, MHz | 1,878.3 | 1,885.3 | 1,885.3 | **1,881.2** |
| lowest MemAvailable MiB; swap highest MiB | 387; 464 | 285; 19 | 295; 19 | **279; 19** |

Each value is one run, from the same files as run 1's table and [text_reuse_points.tsv](text_reuse_points.tsv). No class
is given here: R210's verdict comes after its second run, against both controls.

- **Where the switch acted** ([text_reuse_points.tsv](text_reuse_points.tsv), from the server's own line, for example
  `prompt ids differ from the cache at 7954, same text: 1410 more cached tokens kept (9945 -> 9946 prompt tokens)`).
  `scale-generator` turn 3: the ids differed at token 7,954, 3,647 tokens into turn 2's reply, whose tool call starts
  at its token 5,011; the switch kept the reply's other 1,410 cached tokens, so the server read 582 new tokens. In
  every re-read of runs 1 and 2 the server went back to the previous prompt's end and read the whole reply again; here
  that would have been the 5,058-token reply (est.). `complex-numbers` turn 13: the ids differed at token 12,323, 152
  tokens into turn 12's 697-token reply (its tool call from token 559); 545 cached tokens kept, 226 read.
- **The same point on every later turn.** The cache keeps the ids the model generated, the client sends the whole
  history as text each turn, and the tokenizer splits that reply its own way every time, so each later prompt of the
  task differs from the cache at the same token: 2 more requests in `scale-generator` and 28 more in
  `complex-numbers`, each kept by the switch. Each line's `kept` counts the cached tokens from that point to the end of
  the cache, so they add up turn after turn.
- **The work.** Without `complex-numbers` this run read 3,309 prompt tokens a task against 3,499 and 3,428 and
  generated 2,107 against 2,552 and 2,248, over 62 requests against 66 and 77 ([work_nocx.tsv](work_nocx.tsv)).
  `scale-generator` took 230 s (its 5,058-token reply), against 178 and 103 s.
- **`pig-latin`'s first turn** read its whole 2,324-token prompt, as in run 1: the 41 turns of `complex-numbers` before
  it pushed the shared system prefix's checkpoint out of the four kept on the cards (the server log: "no checkpoint
  below 1553"). Its second turn read 1,047 tokens on 2,376 cached, where runs 1 and 2 read 1,051 on 2,406, so
  [same_prompts.md](same_prompts.md) matches 12 requests in all four runs; run 2's figures above were from the
  three-run match of 13.
- **`affine-cipher`'s first turn**, the run's first request (2,780 tokens, nothing cached, where the switch has nothing
  to compare), took 28.13 s against 27.93 and 27.95 s; the first try's load read it in 28.13 s too.
- **Clocks** ([clocks.tsv](clocks.tsv)): busy medians 1,847 1,847 1,898 1,898 1,873 1,911 1,873 1,911 1,873 MHz (cards
  0–8), one 13 MHz step below runs 1 and 2 on cards 0, 6 and 8.
- **Host** ([host.tsv](host.tsv)): the lowest MemAvailable was 279 MiB at 13:17:58, 7 s after the last task ended and
  before the harness stopped the server; during the tasks it was 290 MiB at 13:17:22. Swap stayed at 19 MiB. The RAM
  watchdog stopped nothing. Card peaks 5,190 5,234 5,092 5,250 4,348 5,108 4,502 5,120 5,698 MiB, the same as runs 1
  and 2 on every card.

## Run 4, `f5c2` (R208c): the control again, 13:59–14:33

**The second control passed all 12 tasks, and on the requests both controls read alike it took 0.95 % longer than the
first.** JavaScript `complex-numbers`, at the 12-minute cap after 38 turns in run 1, passed in 5 turns (47 s). A task
at equal work took 2.37 minutes against run 1's 2.42, and 2.46 against 2.44 without `complex-numbers`; the model
generated at 27.41 tok/s against 26.98. The 12 requests that read the same new tokens on the same cached prefix in all
five runs took 92.43 s against 91.56 s, and `affine-cipher`'s first turn, the run's first request, 28.13 s against
27.93 s. The server read a previous reply again on 2 turns (3,533 tokens, 19.3 of the run's 300.0 prompt seconds),
against 4 (1,806 tokens, 23.6 of 427.2 s) in run 1. Python `scale-generator` wrote a 10,871-token second reply
(413.2 s of generation), against 3,215 tokens in run 1, and took 477 s.

Its started command (1,689 bytes, md5 `4e11746b52bcc136ef186ffe9c161aa0`) equals run 1's and
[run/dry_line.sh](run/dry_line.sh)'s line for `f5c2` byte for byte, and the server's `switches:` line is run 1's
byte for byte (39 set). The server started at 13:59:37 and was up after 311 s; the tasks ran 14:04:51–14:33:30 and the
report ended at 14:33:43. Every card was at stock before and after (75 W, 1,911 / 4,004 MHz maximum, no offset), with
no Xid line.

The last column is how far the two control runs differ, control b less control a, with the percent of control a's
value: the spread that R156's rule compares each switch with. Percents of the rounded tok/s and minutes rows are of
the values shown.

| | `f5c1` (run 1, control a) | `f5s1` (run 2) | `f5t1` (run 3) | **`f5c2` (R208c, control b)** | control b against control a |
|---|---|---|---|---|---|
| passed | 11 / 12 (`complex-numbers` at the time cap, 38 turns) | 12 / 12 | 11 / 12 (`complex-numbers` at the turn cap, 41 turns) | **12 / 12** (`complex-numbers` in 5 turns) | one task more |
| minutes a task at equal work (R90 `v3`'s work) | 2.42 | 2.38 | 2.43 | **2.37** | −0.05 (−2.07 %) |
| the same without `complex-numbers` | 2.44 | 2.47 | 2.45 | **2.46** | +0.02 (+0.82 %) |
| the 12 requests that read the same new tokens on the same cached prefix in all five runs, prompt s | 91.56 | 91.27 | 92.23 | **92.43** | +0.87 (+0.95 %) |
| of them, the 11 first turns (all but `pig-latin`'s) | 83.30 | 83.04 | 83.98 | **84.14** | +0.84 (+1.00 %) |
| of them, `affine-cipher`'s first turn (2,780 tokens, nothing cached), prompt s | 27.93 | 27.95 | 28.13 | **28.13** | +0.19 (+0.70 %) |
| `pig-latin`'s first turn: tokens read; prompt s | 2,324; 12.66 | 775; 6.41 | 2,324; 12.06 | **775; 6.50** | the system prefix cached in b, not in a |
| prompt tok/s, the whole session | 133.7 | 137.1 | 129.3 | **148.4** | +14.7 (+10.99 %) |
| the same without `complex-numbers` | 141.7 | 132.8 | 141.8 | **145.0** | +3.3 (+2.33 %) |
| generation tok/s | 26.98 | 27.85 | 27.07 | **27.41** | +0.43 (+1.59 %) |
| the same without `complex-numbers` | 28.07 | 27.90 | 28.00 | **27.40** | −0.67 (−2.39 %) |
| first token median s; p90 s | 2.41; 9.04 | 2.13; 8.26 | 2.27; 8.89 | **2.63; 8.45** | +0.22 (+9.13 %); −0.59 (−6.53 %) |
| first token on a task's first turn, median s | 6.57 | 6.55 | 6.69 | **6.66** | +0.09 (+1.37 %) |
| the longest first-token wait s | 27.97 | 27.99 | 28.19 | **28.16** (`affine-cipher` turn 1) | +0.19 (+0.68 %) |
| prompt tokens read a task | 4,760 | 3,541 | 4,354 | **3,710** | −1,050 (−22.06 %) |
| the same without `complex-numbers` | 3,499 | 3,428 | 3,309 | **3,610** | +111 (+3.17 %) |
| tokens generated a task without `complex-numbers` | 2,552 | 2,248 | 2,107 | **3,398** | +846 (+33.15 %) |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 4; 1,806; 23.6 of 427.2 | 3; 2,071; 19.8 of 309.9 | 0; 0; 0.0 of 404.0 | **2; 3,533; 19.3 of 300.0** | −2 turns; +1,727 tokens; −4.3 s |
| draft acceptance (accepted of drafted tokens) | 0.732 | 0.752 | 0.735 | **0.743** | +0.011 |
| minutes, all 12 tasks; turns | 33.6; 104 | 21.0; 82 | 27.5; 103 | **28.5; 69** | −5.1; −35 |
| graph reallocations | 0 in 103 requests | 0 in 82 | 0 in 102 | **0 in 69** | none in either |
| busy SM clock, mean of the nine cards' medians, MHz | 1,885.3 | 1,885.3 | 1,881.2 | **1,881.2** | −4.1 (−0.22 %) |
| lowest MemAvailable MiB; swap highest MiB | 285; 19 | 295; 19 | 279; 19 | **316; 19** | +31; 0 |

Each value is one run, from the same files as run 1's table; C854's context figures are in run 1's. No class is given
here: both switches are judged after the sixth run.

- **The spread for the verdicts.** On a task at equal work the two controls differ by 0.05 minutes (2.07 %), and by
  0.02 (0.82 %) without `complex-numbers`; on the 12 matched requests by 0.87 s (0.95 %); in generation by 0.43 tok/s
  (1.59 %), and by 0.67 (2.39 %) without `complex-numbers`.
- **The work.** Without `complex-numbers` this run read 3,610 prompt tokens a task and generated 3,398, over 64
  requests, against run 1's 3,499 and 2,552 over 66 ([work_nocx.tsv](work_nocx.tsv)). Most of the difference is
  `scale-generator`'s second reply, 10,871 tokens in 413.2 s (run 1: 3,215 in 121.7 s; runs 2 and 3: 1,618 and 5,058),
  and JavaScript `variable-length-quantity`'s 8 turns (322 s; 5 turns and 222 s in run 1).
- **The session's prompt tok/s** follows which prompts a run reads: this run read 43 prompts of 511 tokens or fewer
  (77.1 s) of its 69, run 1 66 (118.3 s) of 103 ([reallocs.tsv](reallocs.tsv)). The matched requests are the
  comparison.
- **`pig-latin`'s first turn** read 775 tokens on the 1,549-token shared system prefix, as in run 2: the short
  `complex-numbers` before it left that prefix's checkpoint among the four kept on the cards. So
  [same_prompts.md](same_prompts.md) matches the same 12 requests in all five runs as in four.
- **The reply re-reads** ([rereads_turns.tsv](rereads_turns.tsv)): `affine-cipher` turn 8 (a reply of 152 tokens; the
  prompt stopped matching 88 tokens in, inside the tool call; 3.87 s) and `pig-latin` turn 3 (3,381 tokens; 2,266 in,
  before the tool call; 15.42 s). The client sent both replies back as the server gave them. `pig-latin` turn 3 re-read
  in run 1, C854's run and both R156 runs too, not in run 2.
- **Clocks** ([clocks.tsv](clocks.tsv)): busy medians 1,847 1,847 1,898 1,898 1,873 1,911 1,873 1,911 1,873 MHz (cards
  0–8), the same as run 3's on every card and one 13 MHz step below runs 1 and 2 on cards 0, 6 and 8.
- **Host** ([host.tsv](host.tsv)): the lowest MemAvailable was 316 MiB at 14:33:36, 6 s after the last task ended and
  before the harness stopped the server; during the tasks it was 321 MiB at 14:28:33. Swap stayed at 19 MiB. The RAM
  watchdog stopped nothing. Card peaks 5,190 5,234 5,092 5,250 4,348 5,108 4,502 5,120 5,698 MiB, the same as runs 1
  to 3 on every card.

## Run 5, `f5s2` (R208d): the FP32 scale again, 14:43–15:11

**The FP32 scale's second run read the shared prompts faster than either control, as its first run did.** 12 of 12
tasks passed; JavaScript `complex-numbers` passed in 5 turns (46 s). The 12 requests that read the same new tokens on the
same cached prefix in all six runs (today's five and C854's) took 91.01 s, against 91.27 s in run 2 and 91.56 and
92.43 s in the two controls. `affine-cipher`'s first turn, the run's first request, took 28.04 s against 27.95, 27.93
and 28.13 s. A task at equal work took 2.35 minutes, and 2.44 without `complex-numbers`; the model generated at 27.84
tok/s. It was the day's shortest run in turns: 62, every task in 7 or fewer.

Its started command (1,755 bytes, md5 `9b73d0302aa9ca72b5ac7f9f637f3b80`) equals run 2's and
[run/dry_line.sh](run/dry_line.sh)'s line for `f5s2` byte for byte, and the server's `switches:` line is run 2's byte
for byte (41 set). The server started at 14:43:40 and was up after 313 s; the tasks ran 14:48:56–15:11:29 and the report
ended at 15:11:46. Every card was at stock before and after (75 W, 1,911 / 4,004 MHz maximum, no offset), with no Xid
line.

| | `f5c1` (run 1, control a) | `f5s1` (run 2, FP32 scale a) | `f5t1` (run 3) | `f5c2` (run 4, control b) | **`f5s2` (R208d, FP32 scale b)** |
|---|---|---|---|---|---|
| passed | 11 / 12 (`complex-numbers` at the time cap, 38 turns) | 12 / 12 | 11 / 12 (`complex-numbers` at the turn cap, 41 turns) | 12 / 12 | **12 / 12** (`complex-numbers` in 5 turns) |
| minutes a task at equal work (R90 `v3`'s work) | 2.42 | 2.38 | 2.43 | 2.37 | **2.35** |
| the same without `complex-numbers` | 2.44 | 2.47 | 2.45 | 2.46 | **2.44** |
| the 12 requests that read the same new tokens on the same cached prefix in all six runs, prompt s | 91.56 | 91.27 | 92.23 | 92.43 | **91.01** |
| of them, the 11 first turns (all but `pig-latin`'s) | 83.30 | 83.04 | 83.98 | 84.14 | **82.79** |
| of them, `affine-cipher`'s first turn (2,780 tokens, nothing cached), prompt s | 27.93 | 27.95 | 28.13 | 28.13 | **28.04** |
| `pig-latin`'s first turn: tokens read; prompt s | 2,324; 12.66 | 775; 6.41 | 2,324; 12.06 | 775; 6.50 | **775; 6.42** |
| prompt tok/s, the whole session | 133.7 | 137.1 | 129.3 | 148.4 | **150.8** |
| the same without `complex-numbers` | 141.7 | 132.8 | 141.8 | 145.0 | **147.3** |
| generation tok/s | 26.98 | 27.85 | 27.07 | 27.41 | **27.84** |
| the same without `complex-numbers` | 28.07 | 27.90 | 28.00 | 27.40 | **27.82** |
| first token median s; p90 s | 2.41; 9.04 | 2.13; 8.26 | 2.27; 8.89 | 2.63; 8.45 | **2.48; 8.72** |
| first token on a task's first turn, median s | 6.57 | 6.55 | 6.69 | 6.66 | **6.53** |
| the longest first-token wait s | 27.97 | 27.99 | 28.19 | 28.16 | **28.08** (`affine-cipher` turn 1) |
| prompt tokens read a task | 4,760 | 3,541 | 4,354 | 3,710 | **3,411** |
| the same without `complex-numbers` | 3,499 | 3,428 | 3,309 | 3,610 | **3,286** |
| tokens generated a task without `complex-numbers` | 2,552 | 2,248 | 2,107 | 3,398 | **2,592** |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 4; 1,806; 23.6 of 427.2 | 3; 2,071; 19.8 of 309.9 | 0; 0; 0.0 of 404.0 | 2; 3,533; 19.3 of 300.0 | **2; 1,870; 15.1 of 271.5** |
| draft acceptance (accepted of drafted tokens) | 0.732 | 0.752 | 0.735 | 0.743 | **0.756** |
| minutes, all 12 tasks; turns | 33.6; 104 | 21.0; 82 | 27.5; 103 | 28.5; 69 | **22.4; 62** |
| graph reallocations | 0 in 103 requests | 0 in 82 | 0 in 102 | 0 in 69 | **0 in 62** |
| busy SM clock, mean of the nine cards' medians, MHz | 1,885.3 | 1,885.3 | 1,881.2 | 1,881.2 | **1,879.8** |
| lowest MemAvailable MiB; swap highest MiB | 285; 19 | 295; 19 | 279; 19 | 316; 19 | **300; 18** |

Each value is one run, from the same files as run 1's table; C854's context figures are in run 1's. The FP32 scale's
class is in [the verdicts](#the-verdicts).

- **The work.** Without `complex-numbers` this run read 3,286 prompt tokens a task and generated 2,592, over 57
  requests, against 3,499 and 2,552 over 66 in control a and 3,610 and 3,398 over 64 in control b
  ([work_nocx.tsv](work_nocx.tsv)). The longest replies were Python `scale-generator`'s (6,833 tokens over 5 turns,
  289 s) and JavaScript `variable-length-quantity`'s (6,133 over 5 turns, 236 s).
- **The session's prompt tok/s** follows which prompts a run reads: this run read 38 prompts of 511 tokens or fewer
  (68.2 s) of its 62, control b 43 (77.1 s) of 69 and control a 66 (118.3 s) of 103 ([reallocs.tsv](reallocs.tsv)).
  The matched requests are the comparison.
- **`pig-latin`'s first turn** read 775 tokens on the 1,549-token shared system prefix, as in runs 2 and 4, so
  [same_prompts.md](same_prompts.md) matches the same 12 requests in all six runs as in five.
- **The reply re-reads** ([rereads_turns.tsv](rereads_turns.tsv)): `affine-cipher` turn 3 (a reply of 733 tokens; the
  prompt stopped matching 533 tokens in, before the tool call; 7.06 s) and `list-ops` turn 3 (1,137 tokens; 880 in,
  before the tool call; 8.02 s). The client sent both replies back as the server gave them.
- **Clocks** ([clocks.tsv](clocks.tsv)): busy medians 1,847 1,847 1,898 1,898 1,860 1,911 1,873 1,911 1,873 MHz (cards
  0–8), the same as runs 3 and 4 except card 4, one 13 MHz step lower (1,860 against 1,873).
- **Host** ([host.tsv](host.tsv)): the lowest MemAvailable was 300 MiB at 15:11:42, 13 s after the last task ended and
  before the harness stopped the server; during the tasks it was 316 MiB at 15:07:49. Swap stayed at 18 MiB. The RAM
  watchdog stopped nothing. Card peaks 5,190 5,234 5,092 5,250 4,348 5,108 4,502 5,120 5,698 MiB, the same as runs 1
  to 4 on every card.

## Run 6, `f5t2`

Not run yet.

## The verdicts

Each switch against the two controls, by R156's rule (a gain or a loss when the difference from the controls' mean is
larger than their spread, negligible otherwise), on a task at equal work, the matched requests' prompt time and
generation; for text reuse also the reply re-reads, their prompt seconds and the longest first-token wait. The
controls' spread on each measure is run 4's last column. Text reuse is judged after run 6.

### R208, the FP32 scale (`f5s1`, `f5s2`): negligible

**Negligible, and slower on no measure.** On the 12 requests that every run read alike, where a prompt switch shows,
the FP32 scale's two runs took 91.27 and 91.01 s against the controls' 91.56 and 92.43 s. Both of its runs were faster
than both controls, but on the means the difference is 0.85 s (0.93 %) and the two controls differ by 0.87 s (0.95 %),
so it stays inside their spread. A task at equal work and generation are inside the spread too, once the work is the
same.

The difference is the FP32 scale's mean less the controls' mean, with its percent of the controls' mean; the spread is
control b less control a, with its percent of control a. The request rows are from [same_prompts.md](same_prompts.md)'s
per-request milliseconds.

| | control a `f5c1` | control b `f5c2` | FP32 scale a `f5s1` | FP32 scale b `f5s2` | FP32 scale against the controls | the controls' spread | class |
|---|---|---|---|---|---|---|---|
| the 12 matched requests, prompt s | 91.56 | 92.43 | 91.27 | 91.01 | −0.85 (−0.93 %) | 0.87 (0.95 %) | **negligible** |
| of them, the 11 first turns | 83.30 | 84.14 | 83.04 | 82.79 | −0.81 (−0.96 %) | 0.84 (1.00 %) | negligible |
| of them, the 8 that read more than 600 new tokens (micro-batches of 512) | 77.60 | 78.21 | 77.16 | 76.86 | −0.89 (−1.15 %) | 0.62 (0.79 %) | beyond the spread, a split made after the runs (below) |
| of them, the 4 that read 600 or fewer (micro-batches of 64) | 13.96 | 14.22 | 14.11 | 14.15 | +0.04 (+0.31 %) | 0.25 (1.82 %) | negligible |
| of them, `affine-cipher`'s first turn (2,780 tokens, nothing cached) | 27.93 | 28.13 | 27.95 | 28.04 | −0.03 (−0.12 %) | 0.19 (0.70 %) | negligible |
| minutes a task at equal work | 2.42 | 2.37 | 2.38 | 2.35 | −0.03 (−1.25 %) | 0.05 (2.07 %) | **negligible** |
| the same without `complex-numbers` | 2.44 | 2.46 | 2.47 | 2.44 | +0.005 (+0.20 %) | 0.02 (0.82 %) | negligible |
| generation tok/s | 26.98 | 27.41 | 27.85 | 27.84 | +0.65 (+2.39 %) | 0.43 (1.59 %) | beyond the spread, from the work (below) |
| the same without `complex-numbers` | 28.07 | 27.40 | 27.90 | 27.82 | +0.13 (+0.45 %) | 0.67 (2.39 %) | **negligible** |
| first token on a task's first turn, median s | 6.57 | 6.66 | 6.55 | 6.53 | −0.08 (−1.13 %) | 0.09 (1.37 %) | negligible |
| the longest first-token wait s | 27.97 | 28.16 | 27.99 | 28.08 | −0.03 (−0.11 %) | 0.19 (0.68 %) | negligible |
| passed | 11 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | | | |

- **Where it acts.** On a P106 `llama-bench` measured the switch faster only at micro-batches of 512
  ([C861](../2026-11-17_c861a_fp32_scale/README.md#result-rig-1-2026-11-18)), and the preset reads a prompt of more than
  600 new tokens in micro-batches of 512 (`--ubatch-taper short=600:64`). Split that way after the runs, the 8 longer
  matched requests were 0.89 s (1.15 %) faster against a 0.62 s (0.79 %) spread, 7 of the 8 faster on the means, and
  the 4 shorter ones +0.04 s (+0.31 %), as `llama-bench` at micro-batches of 64 showed nothing. The split was not part
  of the rule set before the runs, and 0.48 s of the 0.89 s come from two requests on which one control alone was slow
  (`space-age`'s first turn 7.29 s in control b against 6.62 s in a; `scale-generator`'s 7.92 s in control a against
  7.02 s in b); without those two the other six were 0.41 s (0.65 %) faster against a 0.85 s (1.35 %) spread. So it
  points to a small gain on long prompts that these four runs cannot separate from the spread.
- **Generation.** The switch changes no kernel of a generation step: a verify pass of 4 tokens runs the 1-to-8-token
  kernels, which keep their own scale multiply (Q1558). Over all 12 tasks the FP32 scale's runs generated 0.65 tok/s
  faster on the means, beyond the 0.43 spread, but that measure carries control a's `complex-numbers` at the cap
  (38 turns, the day's deepest contexts); without that task the difference is +0.13 against 0.67. The draft's
  acceptance (0.752 and 0.756 against 0.732 and 0.743) follows what the model wrote.
- **What follows.** A negligible switch that is not measurably slower stays in the engine behind its switch
  (`pascal-p106-v5`, off by default) and joins the negligible gains tested together on the cumulative build; it does
  not join Flash-Next's v5 preset by itself. The same greedy reply with and without it on a P106 (C861) means it
  changes no answer.

## Files

| File | What |
|---|---|
| `agent/<arm>/turns.csv`, `agent/<arm>/attempts.csv` | each run's per-request and per-task tables (the client's `res/` files, the harness's arm `v4r` renamed by [run/rename_arm.py](run/rename_arm.py)) |
| [agent.tsv](agent.tsv) | per run: tasks passed, turns, prompt and generation tok/s, first token median and p90 and on a task's first turn, the draft's acceptance, minutes, a task at equal work (R90 `v3`'s work); R90 `v3` and C854 `v4rc2` as context rows |
| [same_prompts.md](same_prompts.md) | the requests that read the same new tokens on the same cached prefix in every run, each run's prompt ms, and each run's decode speed |
| [work.md](work.md), [work.tsv](work.tsv) | per run: prompt tokens read and generated a task, and every turn's cache path (reply re-reads, older checkpoints) |
| [first_token.tsv](first_token.tsv) | per run: the longest first-token wait (task, turn, tokens read and cached, prompt ms) and the later turns' median and p90 |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv) | C854's [reply_rereads.py](../2026-11-18_c854b_p106_check/run/reply_rereads.py): per run the turns that read the previous reply again, their tokens and prompt seconds, and per such turn where inside the reply the prompt stopped matching |
| [text_reuse_points.tsv](text_reuse_points.tsv) | [run/text_reuse_points.py](run/text_reuse_points.py): every request on which text reuse kept the cached tokens (the server's `same text` line), the token where the ids differed, the tokens kept, where that lies in the previous reply (before or inside its tool call) and whether the client sent the reply back unchanged |
| [clocks.tsv](clocks.tsv) | each card's busy SM clock median per run, over the tasks' window |
| [host.tsv](host.tsv) | lowest MemAvailable, swap at the start and its highest, MemAvailable at the end, each card's peak memory |
| [reallocs.tsv](reallocs.tsv) | the target graph's reallocations per prompt-size bucket ([scripts/realloc_counts.py](../../../scripts/realloc_counts.py)) |
| [run/flash_switch.sh](run/flash_switch.sh) | the run: the arm's `V4R_ENV`, then C854's [flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh) |
| [run/dry_line.sh](run/dry_line.sh) | the server line a run would start, built by the unchanged driver chain without a load |
| [run/tables.sh](run/tables.sh) | every table above from the runs' logs |
| [run/rename_arm.py](run/rename_arm.py) | the client's tables with the arm renamed |
| [run/text_reuse_points.py](run/text_reuse_points.py) | the `same text` lines of a run's server log placed on its turns (`tables.sh` runs it) |

## Commands

From the repository root on the machine that drives the rig, with the rig clean; `ARM` is the run's arm, and every
run gets its own `RIGDIR` and `CLIENT_OUT` (the driver skips attempts it finds finished in its client folder). When
another agent check may use the same client machine, give each run its own `PREFIX` as well (the client's container
names; with the default, two runs of the same harness arm remove each other's containers, Q1619):

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
