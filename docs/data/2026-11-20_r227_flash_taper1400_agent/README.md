# R227: Flash-Next's prompt calls of up to 1,400 tokens in micro-batches of 64, on the agent check (2026-11-20)

Registry row [Q1641](../../../quants/experiments.md), from R225 ([Q1637](../2026-11-20_r225_flash_taper1400/README.md)).
Flash-Next's v5 preset reads a prompt call of at most 600 tokens in micro-batches of 64 and every longer call in
micro-batches of 512 (`--ubatch-taper short=600:64`). R225 replayed the agent check's 62 recorded prompts with calls of
up to 1,400 tokens in 64s (`short=1400:64`): 208.30 and 207.38 s a pass against the preset's 220.24, 220.41 and 220.55
(−12.56 s, −5.70 %), no size of prompt slower and the same replies. Release settings are chosen on the coding-agent
check, so this folder decides there whether `short=1400:64` goes into Flash-Next's v5 preset.

**Run 1 of 4 done: the control, Flash-Next's v5 preset as it is, passed 12 of 12 tasks.** A task took 2.26 minutes at
equal work, the model generated at 31.05 tok/s over the session, and the prompts took 20.01 s a task (240.07 s in all,
35,504 tokens read). No task ran to its 12-minute cap, so the figures without capped tasks are the same. By the new
tokens of a turn, the prompts took 73.36 s on the 40 turns under 600 tokens, 45.46 s on the 7 turns of 600 to 1,023,
59.97 s on the 8 turns of 1,024 to 1,999 and 61.28 s on the 5 turns of 2,000 or more. The 12 turns of 601 to 1,400 new
tokens, where `short=1400:64` changes the micro-batches, took 81.85 s. For reference only: R220's two runs with ratio
0.75 earlier on the same boot started this exact command (before it became the preset) and took 2.26 and 2.27 minutes a
task at equal work, 30.60 and 30.39 tok/s, and 21.50 and 21.64 s of prompts a task. The started command matches
`settings.env`; every card held the profile, the watchdog never tripped and the boot logged no GPU error.

**Run 2 of 4 done: the same preset with `short=1400:64` passed 12 of 12 tasks.** A task took 2.25 minutes at equal work
(the control's run: 2.26), the model generated at 30.71 tok/s (31.05), and the prompts took 20.37 s a task (20.01),
244.47 s in all for 38,598 tokens read (240.07 s for 35,504). No task ran to its cap. The agent did more work in this
run (66 turns against 60, 31,360 tokens generated against 22,271), so the plain sums compare different work. The 12
turns of 601 to 1,400 new tokens took 66.90 s for 11,285 tokens (the control's 12: 81.85 s for 11,421), and the 12
first turns, the same prompt on the same cached prefix in both runs, took 76.01 s (86.07). By the new tokens of a
turn: 79.93 s on 45 turns under 600, 32.84 s on 7 turns of 600 to 1,023, 65.20 s on 9 turns of 1,024 to 1,999 and
66.51 s on 5 turns of 2,000 or more. The server read `short=1400:64` at its load and its started command differs from
the control's in that value only. No class is given yet: the verdict compares both arms after all four runs.

**Run 3 of 4 done: the second control, the v5 preset as it is, passed 12 of 12 tasks.** A task took 2.24 minutes at
equal work (the first control's run: 2.26; `short=1400:64`'s: 2.25), the model generated at 31.46 tok/s (31.05;
30.71), and the prompts took 21.41 s a task (20.01; 20.37), 256.87 s in all for 39,629 tokens read, with the most work
of the three runs (68 turns, 29,745 tokens generated). No task ran to its cap. The two controls set the spread the
setting is judged against: 2.26 and 2.24 minutes a task at equal work, 31.05 and 31.46 tok/s, and 86.07 and 85.71 s on
the 12 first turns, the same prompt on the same cached prefix in every run, where `short=1400:64` took 76.01 s. On
the 7 of those first turns of 601 to 1,400 new tokens the controls took 45.78 and 45.61 s, `short=1400:64` 36.80. The
second control read JavaScript `variable-length-quantity`'s 577-token first turn in 4.41 s (the first: 4.37), so
`short=1400:64`'s 3.39 s there stays an oddity of that run. The started command is the first control's byte for byte
and the server read `short=600:64`. No class is given yet: the verdict waits for the fourth run.

## What runs

Four runs of Flash-Next's agent check ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 40-minute budget) on one rig, one engine and one boot,
alternated so that each arm has one run in each half:

| # | Run | Arm | `--ubatch-taper` | Status |
|---|---|---|---|---|
| 1 | R227a | `c600a` | `short=600:64`, the v5 preset as it is (the control) | done 16:25–16:46 |
| 2 | R227b | `t1400a` | `short=1400:64` | done 17:15–17:41 |
| 3 | R227c | `c600b` | `short=600:64` (the control) | done 17:49–18:14 |
| 4 | R227d | `t1400b` | `short=1400:64` | planned |

Every run: the maker's Swift GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, the n-gram
table in the third shard) with the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`) at 65,536 tokens on nine P106-090
cards, Flash-Next's v5 preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env), with
`GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75` since R220) on the v5 tree `pascal-p106-v5` `602691153` (`llama-server`
md5 `86607661d5a19d1a90d19ee4db10c90b`). The rig is the second one, with its own soaked clock profile set on this boot
([R219](../2026-11-19_r209_oc_v5/r219_apply.md): memory +1,300 +1,000 +1,000 +1,100 +1,100 +1,200 +900 +900 +1,300 MT/s
and core +256 +256 +269 +231 +206 +206 +282 +256 +231 MHz on cards 0 to 8, 75 W, the kit's watchdog running), as for
R220 and R225. The machine that drives the rig is the agent's client.

**How the preset is started.** [run/flash_taper.sh](run/flash_taper.sh) runs C854's launcher
([flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh): R156's driver
[r156.sh](../2026-11-13_r156_agent_v4r/run/r156.sh), the harness's arm `v4r`, then the report), as R220's
[run/flash_ratio.sh](../2026-11-20_r220_flash_ratio075/run/flash_ratio.sh) does, with `LLAMA_SERVER_TEXT_REUSE=1` added
to the driver's `V4R_ENV` and the driver's `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0` replaced by `=0.75`. That line
is the v5 preset: the same 40 switches in the same order as `settings.env`'s `PRESET_ENV`, and its `PRESET_ARGS`. The
arm sets the driver's `V4C2_TAPER`: `short=600:64` (the control, the driver's own default) or `short=1400:64`, so the
two arms' started commands differ in that one value. [run/preset_check.sh](run/preset_check.sh) checks a started
command against `settings.env` (the environment byte for byte, each flag with its value, the taper's value given), and
[run/dry_line.sh](run/dry_line.sh) prints the command an arm would start without loading anything.

**The reference runs** are R220's two runs with the sparse ratio at 0.75, `r075a` and `r075b`
([R220](../2026-11-20_r220_flash_ratio075/README.md), 11:49–12:10 and 14:00–14:25 on the same rig, engine and boot):
their started command is this control's byte for byte (1,716 bytes, md5 `f588628a98483a96e8ceb8f2969d27dc`), the
preset that R220's verdict then made the v5 preset. R220's two controls (ratio 1.0, 2.28 and 2.32 minutes a task at
equal work, 30.28 and 29.46 tok/s) ran another preset. The reference rows are shown beside each run for reference only;
the verdict compares the two arms' runs of R227.

## Run 1, `c600a` (R227a): the control, 16:25–16:46

**The v5 preset ran 12 of 12 tasks in 16.7 minutes.** Its started command (1,716 bytes, md5
`f588628a98483a96e8ceb8f2969d27dc`) equals [run/dry_line.sh](run/dry_line.sh)'s line for `c600a` and R220's
`r075a`/`r075b` started command byte for byte, and passes [run/preset_check.sh](run/preset_check.sh) against
`settings.env` (40 switches, `PRESET_ARGS` with `--ubatch-taper short=600:64`, plus the harness's `--metrics
--no-webui` and the draft). The server's `switches:` line is R220's `r075b`'s byte for byte (40 set), and its load
logged `micro-batch taper of prompt calls: short=600:64 (n_ubatch 512)` ([taper_lines.tsv](taper_lines.tsv)). The
server started at 16:25:32 and was up after 226 s; the tasks ran 16:29:22–16:46:16 and the report ended at 16:46:28.
Before and after, every card was at the profile's offsets (75 W), the kit's watchdog was running, and the kernel log
held no Xid line.

| | R220 `r075a` (reference) | R220 `r075b` (reference) | **`c600a` (R227a)** |
|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (R90 `v3`'s work) | 2.26 | 2.27 | **2.26** |
| tasks that ran to the 12-minute cap | none | none | **none** |
| generation tok/s | 30.60 | 30.39 | **31.05** |
| prompt tok/s, the whole session | 154.6 | 150.6 | **147.9** |
| prompt s a task (all prompt s) | 21.50 (257.98) | 21.64 (259.62) | **20.01 (240.07)** |
| prompt tokens read a task | 3,324 | 3,259 | **2,959** |
| tokens generated a task | 1,699 | 2,294 | **1,856** |
| prompt s by the turn's new tokens: under 600 (turns) | 74.79 (41) | 74.85 (42) | **73.36 (40)** |
| 600 to 1,023 | 50.30 (8) | 51.67 (8) | **45.46 (7)** |
| 1,024 to 1,999 | 59.01 (8) | 67.42 (9) | **59.97 (8)** |
| 2,000 and more | 73.88 (6) | 65.68 (5) | **61.28 (5)** |
| the turns of 601 to 1,400 new tokens, where `short=1400:64` acts: turns; tokens; prompt s | 12; 11,139; 78.64 | 12; 11,152; 80.14 | **12; 11,421; 81.85** |
| the 12 first turns, the same prompt on the same cached prefix in all three runs, prompt s | 84.97 | 86.46 | **86.07** |
| first token median s; p90 s | 2.28; 7.91 | 2.64; 7.94 | **2.44; 7.94** |
| first token on a task's first turn, median s | 6.22 | 6.35 | **6.38** |
| the longest first-token wait s (`affine-cipher` turn 1, 2,780 tokens, nothing cached) | 27.48 | 27.50 | **27.46** |
| draft acceptance; tokens a decode step | 0.755; 3.27 | 0.743; 3.23 | **0.767; 3.30** |
| ms a decode step, all requests | 106.97 | 106.45 | **106.56** |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 0; 0; 0.0 | 0; 0; 0.0 | **1; 195; 1.9 of 240.1** |
| requests on which text reuse kept the cached tokens | 8 | 12 | **2 (1 task)** |
| minutes, all 12 tasks; turns | 15.9; 63 | 19.9; 64 | **16.7; 60** |
| tool calls / failed / malformed | 58 / 0 / 0 | 60 / 2 / 0 | **54 / 0 / 0** |
| graph reallocations | 0 | 0 | **0 in 60 requests** |
| busy SM clock medians, cards 0 to 8, MHz (mean) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | the same (2,127.0) | **2113 2151 2151 2100 2113 2100 2151 2151 2126 (2,128.4)** |
| lowest MemAvailable MiB; swap highest MiB | 386; 448 | 297; 50 | **318; 50** |

Each value is one run ([agent.tsv](agent.tsv), [work.md](work.md), [prompt_tasks.tsv](prompt_tasks.tsv),
[prompt_buckets.tsv](prompt_buckets.tsv), [same_prompts.md](same_prompts.md), [first_token.tsv](first_token.tsv),
[steps.md](steps.md), [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv),
[text_reuse_points.tsv](text_reuse_points.tsv), [clocks.tsv](clocks.tsv), [host.tsv](host.tsv),
[reallocs.tsv](reallocs.tsv); the agent check's own report: [c600a_report_table.md](c600a_report_table.md),
[c600a_compare.md](c600a_compare.md), [c600a_gonogo.md](c600a_gonogo.md)). The reference columns are this folder's
scripts on R220's published tables and server logs, and R220's own clock and host tables.

- **Prompt seconds a task** ([prompt_tasks.tsv](prompt_tasks.tsv)): from 13.62 s (JavaScript `triangle`, 4 turns) to
  41.64 s (Python `affine-cipher`, whose first turn reads 2,780 tokens with nothing cached); every other task 15.69 to
  21.44 s.
- **The turns `short=1400:64` will change** ([prompt_buckets.tsv](prompt_buckets.tsv), `agent/c600a/turns.csv`): 12
  turns read 601 to 1,400 new tokens (610, 704, 775, 803, 815, 896, 991, 1,047, 1,086, 1,130, 1,237 and 1,327), 11,421
  tokens in 81.85 s; in R225's replay every such call read faster in 64s. The other 48 turns read 24,083 tokens in
  158.22 s.
- **One reply re-read** ([rereads_turns.tsv](rereads_turns.tsv)): Python `list-ops` turn 4 read the previous reply's
  195 tokens again (1.91 s). The client sent the reply back with the tool call's arguments re-serialized, its tokens
  left the reply's at the tool call's start (token 108 of 195), and the cache kept none of the reply. Text reuse kept the cached tokens on Python `pig-latin` turns 4
  and 5 ([text_reuse_points.tsv](text_reuse_points.tsv)); the time from choosing the slot to processing a request was
  0.67 ms on the median ([slot_gap.tsv](slot_gap.tsv)).
- **Answers sound:** every task passed with its hidden tests; 0 of 54 calls malformed, none inside the reasoning, no
  NaN or inf, 12 checkpoints restored, 0 full re-processing, 0 graph reallocations. The report's gates are all GO
  except memory (NO-GO on the 318 MB low), as in every Flash-Next run of this preset.
- **Memory:** swap stayed at 50 MiB the whole run; the lowest MemAvailable was 318 MiB at 16:46:23, after the last
  task. The RAM watchdog stopped nothing.
- **The kit's watchdog** sampled 2,030 times a card over the tasks: 0 trips, never two samples in a row above 80 W,
  the offsets the same in every sample.

## Run 2, `t1400a` (R227b): `short=1400:64`, 17:15–17:41

**The preset with `short=1400:64` ran 12 of 12 tasks in 21.5 minutes.** Its started command (1,717 bytes, md5
`40399f44f30ffa211bec0e89f0362b8f`) equals [run/dry_line.sh](run/dry_line.sh)'s line for `t1400a` byte for byte, so it
differs from the control's in the taper's value only, and passes [run/preset_check.sh](run/preset_check.sh) with
`short=1400:64` (40 switches, `PRESET_ARGS` with `--ubatch-taper short=1400:64`, plus the harness's `--metrics
--no-webui` and the draft). The server's `switches:` line is the control's byte for byte (40 set), and its load logged
`micro-batch taper of prompt calls: short=1400:64 (n_ubatch 512)` ([taper_lines.tsv](taper_lines.tsv)). The server
started at 17:15:18 and was up after 226 s; the tasks ran 17:19:08–17:40:49 and the report ended at 17:40:58. Before and
after, every card was at the profile's offsets (75 W), the kit's watchdog was running, and the kernel log held no Xid
line.

| | `c600a` (R227a, the control) | **`t1400a` (R227b)** |
|---|---|---|
| passed | 12 / 12 | **12 / 12** |
| minutes a task at equal work (R90 `v3`'s work) | 2.26 | **2.25** |
| tasks that ran to the 12-minute cap | none | **none** |
| generation tok/s | 31.05 | **30.71** |
| prompt tok/s, the whole session | 147.9 | **157.9** |
| prompt s a task (all prompt s) | 20.01 (240.07) | **20.37 (244.47)** |
| prompt tokens read a task | 2,959 | **3,216** |
| tokens generated a task | 1,856 | **2,613** |
| prompt s by the turn's new tokens: under 600 (turns) | 73.36 (40) | **79.93 (45)** |
| 600 to 1,023 | 45.46 (7) | **32.84 (7)** |
| 1,024 to 1,999 | 59.97 (8) | **65.20 (9)** |
| 2,000 and more | 61.28 (5) | **66.51 (5)** |
| ms a prompt token in those buckets: under 600; 600 to 1,023; 1,024 to 1,999; 2,000 and more | 9.249; 8.127; 5.595; 5.442 | **9.061; 6.023; 5.378; 5.450** |
| the turns of 601 to 1,400 new tokens, where `short=1400:64` acts: turns; tokens; prompt s | 12; 11,421; 81.85 | **12; 11,285; 66.90** |
| the 12 first turns, the same prompt on the same cached prefix in both runs, prompt s | 86.07 | **76.01** |
| of them, the 7 of 601 to 1,400 new tokens, prompt s | 45.78 | **36.80** |
| first token median s; p90 s | 2.44; 7.94 | **2.32; 7.88** |
| first token on a task's first turn, median s | 6.38 | **4.67** |
| the longest first-token wait s (`affine-cipher` turn 1, 2,780 tokens, nothing cached) | 27.46 | **27.37** |
| draft acceptance; tokens a decode step | 0.767; 3.30 | **0.762; 3.29** |
| ms a decode step, all requests | 106.56 | **107.18** |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 1; 195; 1.9 of 240.1 | **0; 0; 0.0 of 244.5** |
| requests on which text reuse kept the cached tokens | 2 (1 task) | **1 (1 task)** |
| minutes, all 12 tasks; turns | 16.7; 60 | **21.5; 66** |
| tool calls / failed / malformed | 54 / 0 / 0 | **60 / 1 / 0** |
| graph reallocations | 0 in 60 requests | **0 in 66 requests** |
| busy SM clock medians, cards 0 to 8, MHz (mean) | 2113 2151 2151 2100 2113 2100 2151 2151 2126 (2,128.4) | **2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0)** |
| lowest MemAvailable MiB; swap highest MiB | 318; 50 | **309; 50** |

Each value is one run (the same tables as run 1; the agent check's own report:
[t1400a_report_table.md](t1400a_report_table.md), [t1400a_compare.md](t1400a_compare.md),
[t1400a_gonogo.md](t1400a_gonogo.md)). R220's two reference runs are in run 1's table and in the same files.

- **More work than the control:** the agent took 66 turns and generated 31,360 tokens (60 and 22,271 in the control's
  run) and read 3,094 more prompt tokens, so the plain prompt sums and the 21.5 minutes compare different work; "a task
  at equal work" prices each run's work at R90 `v3`'s, and the matched first turns compare the same prompts.
- **The turns `short=1400:64` changes** ([prompt_buckets.tsv](prompt_buckets.tsv), `agent/t1400a/turns.csv`): 12 turns
  read 601 to 1,400 new tokens (610, 622, 775, 803, 815, 896, 931, 1,051, 1,086, 1,130, 1,239 and 1,327), 11,285 tokens
  in 66.90 s. The other 54 turns read 27,313 tokens in 177.56 s.
- **The matched first turns** ([same_prompts.md](same_prompts.md)): the 7 of 601 to 1,400 new tokens each read faster
  than in the control's run (610 tokens 6.33 → 3.84 s, 775 6.17 → 4.72, 803 6.44 → 4.79, 815 6.22 → 4.54, 896 6.51 →
  6.00, 1,086 6.86 → 6.64, 1,130 7.25 → 6.27), 45.78 s → 36.80 s. The 2,780-token turn, read in 512s by both arms, took
  27.33 s (27.41). Of the four under 600 tokens, which both arms read in 64s, three are within 0.06 s of the control;
  the fourth, JavaScript `variable-length-quantity`'s 577 tokens, took 3.39 s against 4.37 (R220's two runs: 4.31 and
  4.42), which the taper does not change.
- **Prompt seconds a task** ([prompt_tasks.tsv](prompt_tasks.tsv)): from 13.64 s (Python `variable-length-quantity`, 6
  turns) to 45.27 s (Python `affine-cipher`, 7 turns, its first reading 2,780 tokens with nothing cached); every other
  task 14.52 to 30.72 s.
- **No reply re-read** ([rereads_runs.tsv](rereads_runs.tsv)): 0 of 54 later turns. Text reuse kept the cached tokens
  on Python `variable-length-quantity` turn 6 ([text_reuse_points.tsv](text_reuse_points.tsv)); the time from choosing
  the slot to processing a request was 0.65 ms on the median ([slot_gap.tsv](slot_gap.tsv)).
- **Answers sound:** every task passed with its hidden tests; 0 of 60 calls malformed, none inside the reasoning; one
  tool call failed (in Python `affine-cipher`, which passed); no NaN or inf, 11 checkpoints restored, 0 full
  re-processing, 0 graph reallocations. The report's gates are all GO except memory (NO-GO on the 309 MB low), as in
  every Flash-Next run of this preset.
- **Memory:** swap stayed at 50 MiB the whole run; the lowest MemAvailable was 309 MiB at 17:35:19. The RAM watchdog
  stopped nothing.
- **The kit's watchdog** sampled 2,603 times a card over the tasks: 0 trips, never two samples in a row above 80 W,
  the offsets the same in every sample.

## Run 3, `c600b` (R227c): the second control, 17:49–18:14

**The v5 preset ran 12 of 12 tasks in 20.5 minutes.** Its started command (1,716 bytes, md5
`f588628a98483a96e8ceb8f2969d27dc`) equals [run/dry_line.sh](run/dry_line.sh)'s line for `c600b` and the first
control's started command byte for byte, and passes [run/preset_check.sh](run/preset_check.sh) against `settings.env`
(40 switches, `PRESET_ARGS` with `--ubatch-taper short=600:64`, plus the harness's `--metrics --no-webui` and the
draft). The server's `switches:` line is the first control's byte for byte (40 set), and its load logged `micro-batch
taper of prompt calls: short=600:64 (n_ubatch 512)` ([taper_lines.tsv](taper_lines.tsv)). The server started at
17:49:33 and was up after 224 s; the tasks ran 17:53:21–18:14:04 and the report ended at 18:14:12. Before and after,
every card was at the profile's offsets (75 W), the kit's watchdog was running, and the kernel log held no Xid line.

The last column is how far the two controls differ: the spread `short=1400:64`'s two runs are judged against once the
fourth run is in.

| | `c600a` (R227a, the first control) | `t1400a` (R227b, `short=1400:64`) | **`c600b` (R227c, the second control)** | the two controls differ by |
|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | **12 / 12** | |
| minutes a task at equal work (R90 `v3`'s work) | 2.26 | 2.25 | **2.24** | 0.02 |
| tasks that ran to the 12-minute cap | none | none | **none** | |
| generation tok/s | 31.05 | 30.71 | **31.46** | 0.41 |
| prompt tok/s, the whole session | 147.9 | 157.9 | **154.3** | 6.4 |
| prompt s a task (all prompt s) | 20.01 (240.07) | 20.37 (244.47) | **21.41 (256.87)** | 1.40 (16.80) |
| prompt tokens read a task | 2,959 | 3,216 | **3,302** | 343 |
| tokens generated a task | 1,856 | 2,613 | **2,479** | 623 |
| prompt s by the turn's new tokens: under 600 (turns) | 73.36 (40) | 79.93 (45) | **85.36 (48)** | |
| 600 to 1,023 | 45.46 (7) | 32.84 (7) | **38.09 (6)** | |
| 1,024 to 1,999 | 59.97 (8) | 65.20 (9) | **59.23 (8)** | |
| 2,000 and more | 61.28 (5) | 66.51 (5) | **74.19 (6)** | |
| ms a prompt token in those buckets: under 600; 600 to 1,023; 1,024 to 1,999; 2,000 and more | 9.249; 8.127; 5.595; 5.442 | 9.061; 6.023; 5.378; 5.450 | **9.567; 7.791; 5.294; 5.072** | 0.318; 0.336; 0.301; 0.370 |
| the turns of 601 to 1,400 new tokens, where `short=1400:64` acts: turns; tokens; prompt s | 12; 11,421; 81.85 | 12; 11,285; 66.90 | **10; 9,483; 66.17** | |
| ms a prompt token on those turns | 7.166 | 5.929 | **6.978** | 0.188 |
| the 12 first turns, the same prompt on the same cached prefix in every run, prompt s | 86.07 | 76.01 | **85.71** | 0.36 |
| of them, the 7 of 601 to 1,400 new tokens, prompt s | 45.78 | 36.80 | **45.61** | 0.17 |
| first token median s; p90 s | 2.44; 7.94 | 2.32; 7.88 | **2.26; 8.02** | 0.18; 0.08 |
| first token on a task's first turn, median s | 6.38 | 4.67 | **6.38** | 0.00 |
| the longest first-token wait s (`affine-cipher` turn 1, 2,780 tokens, nothing cached) | 27.46 | 27.37 | **27.28** | 0.18 |
| draft acceptance; tokens a decode step | 0.767; 3.30 | 0.762; 3.29 | **0.785; 3.35** | 0.018; 0.05 |
| ms a decode step, all requests | 106.56 | 107.18 | **106.88** | 0.32 |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 1; 195; 1.9 of 240.1 | 0; 0; 0.0 of 244.5 | **0; 0; 0.0 of 256.9** | |
| requests on which text reuse kept the cached tokens | 2 (1 task) | 1 (1 task) | **5 (2 tasks)** | |
| minutes, all 12 tasks; turns | 16.7; 60 | 21.5; 66 | **20.5; 68** | |
| tool calls / failed / malformed | 54 / 0 / 0 | 60 / 1 / 0 | **64 / 2 / 0** | |
| graph reallocations | 0 in 60 requests | 0 in 66 requests | **0 in 68 requests** | |
| busy SM clock medians, cards 0 to 8, MHz (mean) | 2113 2151 2151 2100 2113 2100 2151 2151 2126 (2,128.4) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | **2113 2138 2151 2100 2113 2100 2164 2151 2126 (2,128.4)** | |
| lowest MemAvailable MiB; swap highest MiB | 318; 50 | 309; 50 | **278; 50** | |

Each value is one run (the same tables as run 1; the agent check's own report:
[c600b_report_table.md](c600b_report_table.md), [c600b_compare.md](c600b_compare.md),
[c600b_gonogo.md](c600b_gonogo.md)). R220's two reference runs are in run 1's table and in the same files.

- **The most work of the three runs:** 68 turns, 29,745 tokens generated and 39,629 prompt tokens read, so the plain
  prompt sums compare different work again; "a task at equal work" prices each run's work at R90 `v3`'s, and the
  matched first turns compare the same prompts.
- **The matched first turns** ([same_prompts.md](same_prompts.md)): the two controls read each of the 7 of 601 to
  1,400 new tokens within 0.15 s of each other (610 tokens 6.33 and 6.44 s, 775 6.17 and 6.15, 803 6.44 and 6.44, 815
  6.22 and 6.16, 896 6.51 and 6.36, 1,086 6.86 and 6.87, 1,130 7.25 and 7.21); `short=1400:64`'s run read each faster
  than either control, by 0.22 s at the least (1,086 tokens) and 2.49 s at the most (610 tokens). The 2,780-token turn,
  read in 512s by every run, took 27.22 s (27.41 and 27.33 in the first two). JavaScript `variable-length-quantity`'s
  577 tokens, read in 64s by both arms, took 4.41 s (4.37 in the first control, 3.39 in `short=1400:64`'s run), so that
  3.39 is an oddity of that run, not of the setting.
- **The turns `short=1400:64` would change** ([prompt_buckets.tsv](prompt_buckets.tsv), `agent/c600b/turns.csv`): 10
  turns read 601 to 1,400 new tokens (610, 775, 803, 815, 896, 990, 1,051, 1,086, 1,130 and 1,327), 9,483 tokens in
  66.17 s. The other 58 turns read 30,146 tokens in 190.70 s.
- **Prompt seconds a task** ([prompt_tasks.tsv](prompt_tasks.tsv)): from 13.79 s (JavaScript `triangle`, 4 turns) to
  48.35 s (Python `affine-cipher`, 9 turns, its first reading 2,780 tokens with nothing cached); every other task 15.07
  to 27.49 s.
- **No reply re-read** ([rereads_runs.tsv](rereads_runs.tsv)): 0 of 56 later turns. Text reuse kept the cached tokens
  on JavaScript `grade-school` turns 4 and 5 and Python `pig-latin` turns 3, 4 and 5
  ([text_reuse_points.tsv](text_reuse_points.tsv)); the time from choosing the slot to processing a request was 0.65 ms
  on the median ([slot_gap.tsv](slot_gap.tsv)).
- **Answers sound:** every task passed with its hidden tests; 0 of 64 calls malformed, none inside the reasoning; two
  tool calls failed (one each in Python `affine-cipher` and JavaScript `variable-length-quantity`, which both passed);
  no NaN or inf, 11 checkpoints restored, 0 full re-processing, 0 graph reallocations. The report's gates are all GO
  except memory (NO-GO on the 279 MB low), as in every Flash-Next run of this preset.
- **Memory:** swap stayed at 50 MiB the whole run; the lowest MemAvailable was 278 MiB at 18:14:07, after the last
  task. The RAM watchdog stopped nothing.
- **The kit's watchdog** sampled 2,487 times a card over the tasks: 0 trips, never two samples in a row above 80 W,
  the offsets the same in every sample.

## Files

| File | What it holds |
|---|---|
| [agent.tsv](agent.tsv) | per run: tasks passed, turns, prompt and generation tok/s, first token median and p90, the draft's acceptance, minutes, a task at equal work (R90 `v3`'s work) |
| `agent_nocap.tsv` | the same without every task that ran to its cap in any R227 run; written only when one did ([capped.tsv](capped.tsv) lists them; empty so far) |
| [work.md](work.md), [work.tsv](work.tsv) | per run: prompt tokens read and generated, reply re-reads, prompt seconds, the longest first-token wait |
| [prompt_tasks.tsv](prompt_tasks.tsv) | per task and run: prompt seconds, prompt tokens read, turns; the mean a task |
| [prompt_buckets.tsv](prompt_buckets.tsv) | per run: turns, new tokens and prompt seconds by the turn's new tokens (under 600, 600 to 1,023, 1,024 to 1,999, 2,000 and more), all tasks and, when a task ran to its cap, without it |
| [taper_lines.tsv](taper_lines.tsv) | the micro-batch taper each server read at its load |
| [same_prompts.md](same_prompts.md) | the requests that read the same prompt on the same cached prefix in every run named, prompt ms each |
| [first_token.tsv](first_token.tsv) | the longest first-token wait of each run, and the median and p90 of the turns after a task's first |
| [steps.md](steps.md) | ms a decode step by the context at each request's middle (R203's buckets) |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv) | turns whose cached prefix stops inside the previous reply, per run and per turn (C854's [reply_rereads.py](../2026-11-18_c854b_p106_check/run/reply_rereads.py)) |
| [text_reuse_points.tsv](text_reuse_points.tsv) | the turns on which text reuse kept the cached tokens |
| [slot_gap.tsv](slot_gap.tsv) | the time from choosing the slot to processing each request (text reuse's check runs there) |
| [clocks.tsv](clocks.tsv) | busy SM clock medians per card over the tasks |
| [host.tsv](host.tsv) | lowest MemAvailable, swap, card memory peaks |
| [reallocs.tsv](reallocs.tsv) | graph reallocations by prompt size |
| [agent/](agent/) | each run's per-turn and per-attempt tables (`turns.csv`, `attempts.csv`), the harness's arm `v4r` renamed to the run |
| `<run>_report_table.md`, `<run>_compare.md`, `<run>_gonogo.md` | the agent check's own report of each run |
| [run/](run/) | the scripts below |

## Commands

From the repository root on the machine that drives the rig and is the agent's client, one run a session, each with
its own `OUT`, rig folder, client folder and container prefix; only `ARM` and those names change between the four runs:

```bash
export ARM=c600a PREFIX=acrel-r227a RIG=<rig alias> CLIENT=local \
  MODEL=<models>/flash_next/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483 REF=<v4.0's build folder, unused by this arm> \
  RIGDIR=<a folder on the rig>/r227a OUT=<run folder>/out RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client>/bench CLIENT_OUT=<client>/r227a
unset GAINS_ENV V4R_ENV V4C2_TAPER
bash docs/data/2026-11-20_r227_flash_taper1400_agent/run/flash_taper.sh
```

The runs' `ARM`, `PREFIX` and folder names: R227a `c600a` `r227a`, R227b `t1400a` `r227b`, R227c `c600b` `r227c`, R227d
`t1400b` `r227d`. Before a run, the command it will start, and its check against the preset:

```bash
ARM=t1400a WORK=<an empty scratch folder> NEW=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483 \
  DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  bash docs/data/2026-11-20_r227_flash_taper1400_agent/run/dry_line.sh > dry_line_t1400a.txt
bash docs/data/2026-11-20_r227_flash_taper1400_agent/run/preset_check.sh dry_line_t1400a.txt short=1400:64   # the control: no taper
```

The two arms' dry lines are 1,716 bytes (md5 `f588628a98483a96e8ceb8f2969d27dc`, the control) and 1,717 bytes (md5
`40399f44f30ffa211bec0e89f0362b8f`, `short=1400:64`), different in the taper's value only.

The tables, with Python 3's standard library only, from the runs' collected logs (the inputs are listed at the top of
[run/tables.sh](run/tables.sh)); the client's `res/` files go into `agent/<run>/` first:

```bash
python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <run folder>/out/agent/res \
  docs/data/2026-11-20_r227_flash_taper1400_agent/agent/c600a v4r c600a
bash docs/data/2026-11-20_r227_flash_taper1400_agent/run/tables.sh <tables folder> c600a [t1400a c600b t1400b]
```

| Script | What it does |
|---|---|
| [run/flash_taper.sh](run/flash_taper.sh) | runs one arm: the v5 preset, or the same with `--ubatch-taper short=1400:64` |
| [run/dry_line.sh](run/dry_line.sh) | prints the command an arm would start, without loading anything |
| [run/preset_check.sh](run/preset_check.sh) | checks a started command against `release/flash-next-v5/settings.env`, with the taper's value given |
| [run/tables.sh](run/tables.sh) | builds this folder's tables from the runs done so far |

## Provenance and licence

Measured on the project's own rig on 2026-11-20 with the repository's own scripts; the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/<run>/attempts.csv`). No
model file or task text is in this folder. The tables and scripts here are under the repository's licence.
