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
second control read JavaScript `variable-length-quantity`'s 577-token first turn in 4.41 s (the first: 4.37;
`short=1400:64`'s run: 3.39). The started command is the first control's byte for byte and the server read
`short=600:64`. No class is given yet: the verdict waits for the fourth run.

**Run 4 of 4 done: `short=1400:64` again passed 12 of 12 tasks.** A task took 2.24 minutes at equal work, the model
generated at 30.87 tok/s, and the prompts took 20.80 s a task, 249.63 s in all for 39,649 tokens read (67 turns,
18,328 tokens generated). No task ran to its cap. The 12 turns of 601 to 1,400 new tokens took 64.09 s for 10,875
tokens (5.894 ms a token; the controls 7.166 and 6.978, the first `short=1400:64` run 5.929), and the 12 first turns
took 74.66 s (the controls 86.07 and 85.71, the first `short=1400:64` run 76.01). The started command is the first
`short=1400:64` run's byte for byte and the server read `short=1400:64`; every card held the profile, the watchdog never
tripped and the boot logged no GPU error.

**Verdict: negligible on the agent check, and a gain on the prompts it changes.** A task at equal work took 2.25 and
2.24 minutes with `short=1400:64` against the controls' 2.26 and 2.24 (the controls 0.02 apart), and the prompts 20.37
and 20.80 s a task against 20.01 and 21.41 (1.40 apart): within the spread, as expected for a saving of about 1 s a
task (est. 0.8 %). Where the setting acts it is faster beyond the spread: the turns of 601 to 1,400 new tokens took
5.929 and 5.894 ms a token against 7.166 and 6.978 (0.189 apart); the 7 first turns of that size, the same prompt on
the same cached prefix in every run, took 36.80 and 35.39 s against 45.78 and 45.61 (0.17 apart), each of the 7 faster
in both `short=1400:64` runs than in either control; and the first token of a task's first turn came after 4.67 and
4.62 s on the median against 6.38 and 6.38. Generation ran at 30.71 and 30.87 tok/s against 31.05 and 31.46, 0.465
tok/s lower on the means against a 0.41 spread, while a decode step took as long in both arms (106.71 and 106.70 ms
against 106.67 and 106.53 at 4,000 to 8,000 tokens of context): the gap is the draft's acceptance on each run's own
replies, which the setting, a change to how prompt calls are cut into micro-batches, does not touch. All four runs
passed 12 of 12. **Recommendation:** by the rule registry row Q1641 set before the runs (a gain, or negligible and
slower on no figure, puts `short=1400:64` into the v5 preset), it qualifies; `release/flash-next-v5/settings.env`
keeps `short=600:64` until the owner's word.

## What runs

Four runs of Flash-Next's agent check ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 40-minute budget) on one rig, one engine and one boot,
alternated so that each arm has one run in each half:

| # | Run | Arm | `--ubatch-taper` | Status |
|---|---|---|---|---|
| 1 | R227a | `c600a` | `short=600:64`, the v5 preset as it is (the control) | done 16:25–16:46 |
| 2 | R227b | `t1400a` | `short=1400:64` | done 17:15–17:41 |
| 3 | R227c | `c600b` | `short=600:64` (the control) | done 17:49–18:14 |
| 4 | R227d | `t1400b` | `short=1400:64` | done 18:23–18:41 |

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
  577 tokens, read in 64s by both arms, took 4.41 s (4.37 in the first control, 3.39 in `short=1400:64`'s run).
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

## Run 4, `t1400b` (R227d): `short=1400:64` again, 18:23–18:41

**The preset with `short=1400:64` ran 12 of 12 tasks in 14.6 minutes.** Its started command (1,717 bytes, md5
`40399f44f30ffa211bec0e89f0362b8f`) equals [run/dry_line.sh](run/dry_line.sh)'s line for `t1400b` and the first
`short=1400:64` run's started command byte for byte, and passes [run/preset_check.sh](run/preset_check.sh) with
`short=1400:64` (40 switches, `PRESET_ARGS` with `--ubatch-taper short=1400:64`, plus the harness's `--metrics
--no-webui` and the draft). The server's `switches:` line is the other three runs' byte for byte (40 set), and its load
logged `micro-batch taper of prompt calls: short=1400:64 (n_ubatch 512)` ([taper_lines.tsv](taper_lines.tsv)). The
server started at 18:23:12 and was up after 227 s; the tasks ran 18:27:02–18:41:46 and the report ended at 18:41:52.
Before and after, every card was at the profile's offsets (75 W), the kit's watchdog was running, and the kernel log
held no Xid line.

| | `c600a` (R227a, control) | `t1400a` (R227b, `short=1400:64`) | `c600b` (R227c, control) | **`t1400b` (R227d, `short=1400:64`)** | the two controls differ by |
|---|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 12 / 12 | **12 / 12** | |
| minutes a task at equal work (R90 `v3`'s work) | 2.26 | 2.25 | 2.24 | **2.24** | 0.02 |
| tasks that ran to the 12-minute cap | none | none | none | **none** | |
| generation tok/s | 31.05 | 30.71 | 31.46 | **30.87** | 0.41 |
| prompt tok/s, the whole session | 147.9 | 157.9 | 154.3 | **158.8** | 6.4 |
| prompt s a task (all prompt s) | 20.01 (240.07) | 20.37 (244.47) | 21.41 (256.87) | **20.80 (249.63)** | 1.40 (16.80) |
| prompt tokens read a task | 2,959 | 3,216 | 3,302 | **3,304** | 343 |
| tokens generated a task | 1,856 | 2,613 | 2,479 | **1,527** | 623 |
| prompt s by the turn's new tokens: under 600 (turns) | 73.36 (40) | 79.93 (45) | 85.36 (48) | **81.21 (45)** | |
| 600 to 1,023 | 45.46 (7) | 32.84 (7) | 38.09 (6) | **37.43 (8)** | |
| 1,024 to 1,999 | 59.97 (8) | 65.20 (9) | 59.23 (8) | **73.24 (10)** | |
| 2,000 and more | 61.28 (5) | 66.51 (5) | 74.19 (6) | **57.76 (4)** | |
| ms a prompt token in those buckets: under 600; 600 to 1,023; 1,024 to 1,999; 2,000 and more | 9.249; 8.127; 5.595; 5.442 | 9.061; 6.023; 5.378; 5.450 | 9.567; 7.791; 5.294; 5.072 | **8.826; 5.958; 5.274; 5.618** | 0.318; 0.336; 0.301; 0.370 |
| the turns of 601 to 1,400 new tokens, where `short=1400:64` acts: turns; tokens; prompt s | 12; 11,421; 81.85 | 12; 11,285; 66.90 | 10; 9,483; 66.17 | **12; 10,875; 64.09** | |
| ms a prompt token on those turns | 7.166 | 5.929 | 6.978 | **5.894** | 0.188 |
| the 12 first turns, the same prompt on the same cached prefix in every run, prompt s | 86.07 | 76.01 | 85.71 | **74.66** | 0.36 |
| of them, the 7 of 601 to 1,400 new tokens, prompt s | 45.78 | 36.80 | 45.61 | **35.39** | 0.17 |
| first token median s; p90 s | 2.44; 7.94 | 2.32; 7.88 | 2.26; 8.02 | **2.83; 7.90** | 0.18; 0.08 |
| first token on a task's first turn, median s | 6.38 | 4.67 | 6.38 | **4.62** | 0.00 |
| the longest first-token wait s (`affine-cipher` turn 1, 2,780 tokens, nothing cached) | 27.46 | 27.37 | 27.28 | **27.53** | 0.18 |
| draft acceptance; tokens a decode step | 0.767; 3.30 | 0.762; 3.29 | 0.785; 3.35 | **0.753; 3.26** | 0.018; 0.05 |
| ms a decode step, all requests | 106.56 | 107.18 | 106.88 | **105.95** | 0.32 |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 1; 195; 1.9 of 240.1 | 0; 0; 0.0 of 244.5 | 0; 0; 0.0 of 256.9 | **1; 215; 2.9 of 249.6** | |
| requests on which text reuse kept the cached tokens | 2 (1 task) | 1 (1 task) | 5 (2 tasks) | **0** | |
| minutes, all 12 tasks; turns | 16.7; 60 | 21.5; 66 | 20.5; 68 | **14.6; 67** | |
| tool calls / failed / malformed | 54 / 0 / 0 | 60 / 1 / 0 | 64 / 2 / 0 | **64 / 1 / 0** | |
| graph reallocations | 0 in 60 requests | 0 in 66 requests | 0 in 68 requests | **0 in 67 requests** | |
| busy SM clock medians, cards 0 to 8, MHz (mean) | 2113 2151 2151 2100 2113 2100 2151 2151 2126 (2,128.4) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | 2113 2138 2151 2100 2113 2100 2164 2151 2126 (2,128.4) | **2113 2151 2151 2100 2113 2100 2151 2151 2126 (2,128.4)** | |
| lowest MemAvailable MiB; swap highest MiB | 318; 50 | 309; 50 | 278; 50 | **311; 50** | |

Each value is one run (the same tables as run 1; the agent check's own report:
[t1400b_report_table.md](t1400b_report_table.md), [t1400b_compare.md](t1400b_compare.md),
[t1400b_gonogo.md](t1400b_gonogo.md)). R220's two reference runs are in run 1's table and in the same files.

- **Less generated, as much read:** the agent generated 18,328 tokens, the least of the four runs, and read 39,649
  prompt tokens, about as many as the second control; "a task at equal work" prices each run's work at R90 `v3`'s,
  and the matched first turns compare the same prompts.
- **The matched first turns** ([same_prompts.md](same_prompts.md), [verdict_turns.tsv](verdict_turns.tsv)): the 7 of
  601 to 1,400 new tokens each read faster than in either control (610 tokens 3.81 s, 775 4.65, 803 5.24, 815 4.52,
  896 5.02, 1,086 5.97, 1,130 6.19), 35.39 s. The 2,780-token turn, read in 512s by both arms, took 27.41 s. Of the four
  under 600 tokens, three are within 0.04 s of both controls; JavaScript `variable-length-quantity`'s 577 tokens took
  3.37 s, as in the first `short=1400:64` run (3.39; the controls 4.37 and 4.41).
- **The turns `short=1400:64` changes** ([prompt_buckets.tsv](prompt_buckets.tsv), `agent/t1400b/turns.csv`): 12 turns
  read 601 to 1,400 new tokens (610, 622, 775, 803, 815, 830, 896, 931, 1,051, 1,086, 1,130 and 1,326), 10,875 tokens
  in 64.09 s. The other 55 turns read 28,774 tokens in 185.54 s.
- **Prompt seconds a task** ([prompt_tasks.tsv](prompt_tasks.tsv)): from 15.37 s (Python `list-ops`, 5 turns) to
  41.58 s (Python `affine-cipher`, 5 turns, its first reading 2,780 tokens with nothing cached); every other task 15.84
  to 23.23 s.
- **One reply re-read** ([rereads_turns.tsv](rereads_turns.tsv)): Python `pig-latin` turn 7 read the previous reply's
  215 tokens again (2.88 s); its cached prefix stopped at the tool call's start. Text reuse kept the cached tokens on no
  request ([text_reuse_points.tsv](text_reuse_points.tsv)); the time from choosing the slot to processing a request was
  0.64 ms on the median ([slot_gap.tsv](slot_gap.tsv)).
- **Answers sound:** every task passed with its hidden tests; 0 of 64 calls malformed, none inside the reasoning; one
  tool call failed (in Python `pig-latin`, which passed); no NaN or inf, 12 checkpoints restored, 0 full re-processing,
  0 graph reallocations. The report's gates are all GO except memory (NO-GO on the 311 MB low), as in every Flash-Next
  run of this preset.
- **Memory:** swap stayed at 50 MiB the whole run; the lowest MemAvailable was 311 MiB at 18:38:34. The RAM watchdog
  stopped nothing.
- **The kit's watchdog** sampled 1,769 times a card over the tasks: 0 trips, never two samples in a row above 80 W,
  the offsets the same in every sample.

## The verdict: `short=1400:64` is negligible on the agent check, and a gain on the prompts it changes

Each arm's two runs against each other, the controls' difference as the spread (each run by value; "means" is the mean
of an arm's two runs). The turn sizes are the taper's: up to 600 new tokens both arms read in micro-batches of 64; 601
to 1,400 `short=1400:64` reads in 64s and the control in 512s; from 1,401 both read in 512s. Every turn's figures come
from [run/verdict.py](run/verdict.py) ([verdict_buckets.tsv](verdict_buckets.tsv),
[verdict_matched.tsv](verdict_matched.tsv), [verdict_close.tsv](verdict_close.tsv),
[verdict_first_token.tsv](verdict_first_token.tsv)); the session figures from [agent.tsv](agent.tsv),
[work.tsv](work.tsv) and [steps.md](steps.md).

| | `short=1400:64`: `t1400a`, `t1400b` | control: `c600a`, `c600b` | `short=1400:64` − control, on the means | the controls' spread | class |
|---|---|---|---|---|---|
| tasks passed | 12, 12 | 12, 12 | | | |
| minutes a task at equal work | 2.25, 2.24 | 2.26, 2.24 | −0.005 | 0.02 | within |
| prompt s a task | 20.37, 20.80 | 20.01, 21.41 | −0.125 | 1.40 | within |
| ms a prompt token, every turn | 6.334, 6.296 | 6.762, 6.482 | −0.307 | 0.280 | faster |
| every turn by its new tokens, ms a prompt token (turns): up to 600 | 9.061 (45), 8.826 (45) | 9.249 (40), 9.567 (48) | −0.464 | 0.318 | faster (other turns) |
| 601 to 1,400, where the setting acts | 5.929 (12), 5.894 (12) | 7.166 (12), 6.978 (10) | −1.161 | 0.189 | **gain** |
| 1,401 to 2,047 | 4.722 (5), 4.854 (7) | 4.408 (6), 4.482 (6) | +0.343 | 0.074 | other turns; the same turns alike |
| 2,048 and more | 5.738 (4), 5.984 (3) | 7.066 (2), 5.450 (4) | −0.397 | 1.616 | within |
| the 12 first turns matched in every run, prompt s | 76.01, 74.66 | 86.07, 85.71 | −10.55 | 0.36 | **gain** |
| of them, the 7 of 601 to 1,400 new tokens | 36.80, 35.39 | 45.78, 45.61 | −9.60 | 0.17 | **gain** |
| the 4 of up to 600 | 11.88, 11.86 | 12.88, 12.87 | −1.01 | 0.01 | faster (one turn) |
| the one of 2,780 | 27.33, 27.41 | 27.41, 27.22 | +0.06 | 0.18 | within |
| the 2 turns of 1,401 to 2,047 new tokens that are the same task and turn in every run, prompt s | 15.52, 15.67 | 15.69, 15.37 | +0.06 | 0.32 | within |
| first token on a task's first turn, median s | 4.67, 4.62 | 6.38, 6.38 | −1.735 | 0.00 | faster |
| first token, every turn: median s | 2.32, 2.83 | 2.44, 2.26 | +0.225 | 0.18 | the median in a gap |
| first token, every turn: mean s | 3.764, 3.794 | 4.063, 3.860 | −0.183 | 0.203 | within |
| generation tok/s over the session | 30.71, 30.87 | 31.05, 31.46 | −0.465 | 0.41 | lower: other replies |
| ms a decode step, all requests | 107.18, 105.95 | 106.56, 106.88 | −0.155 | 0.32 | within |
| ms a decode step, 4,000 to 8,000 tokens of context at the request's middle | 106.71, 106.70 | 106.67, 106.53 | +0.105 | 0.14 | within |
| tokens a decode step (draft acceptance) | 3.29 (0.762), 3.26 (0.753) | 3.30 (0.767), 3.35 (0.785) | −0.05 (−0.0185) | 0.05 (0.018) | other replies |

- **Negligible on the session:** a task at equal work and the prompt seconds a task lie within the controls' spread,
  on the faster side. The replay's saving, about 1 s of prompt reading a task, est. 0.8 % of a task, is smaller than
  four runs resolve: the controls alone are 0.02 minutes (1.2 s) a task apart. Counted on this check's own turns, the
  12 turns of 601 to 1,400 new tokens in each `short=1400:64` run would have taken est. 79.8 and 76.9 s at the
  controls' mean 7.072 ms a token instead of 66.90 and 64.09 s: est. 12.9 and 12.8 s a run, about 1.1 s a task.
- **A gain where it acts:** on the turns of 601 to 1,400 new tokens both `short=1400:64` runs read faster a token than
  both controls, by 1.161 ms on the means against a 0.189 ms spread. On the 7 matched first turns of that size each turn
  is faster in both runs than in either control (610 tokens 3.84 and 3.81 s against 6.33 and 6.44; 1,086 tokens, the
  closest, 6.64 and 5.97 against 6.86 and 6.87). A task's first turn is the one a user waits through before the agent
  starts: its first token came 1.735 s sooner on the median.
- **Turns of 1,401 to 2,047 new tokens** both arms read in 512s. Their cost a token is higher in both `short=1400:64`
  runs because the bucket holds other turns there: later turns of 1,437 to 1,549 tokens on 3,117 to 3,472 cached
  tokens, one in the first run (5.448 ms a token) and three in the second (5.038, 5.197 and 5.373), against the
  controls' turns 2 of 1,557 to 2,047 tokens on caches of 2,010 to 2,717. The same turns read alike in all four runs: Python `list-ops` turn 2
  (1,557 tokens) 7.78 and 7.89 s against 7.80 and 7.71, Python `scale-generator` turn 2 (1,590) 7.75 and 7.78 against
  7.89 and 7.66, JavaScript `triangle` turn 2 (1,705) 7.78 and 7.82 against the second control's 7.91
  ([verdict_close.tsv](verdict_close.tsv)).
- **Up to 600 new tokens** both arms read in micro-batches of 64, yet two calls of this size took longer in the
  controls: JavaScript `variable-length-quantity`'s 577-token first turn, 4.37 and 4.41 s against 3.39 and 3.37, and
  Python `affine-cipher`'s 561-token turn 3, 4.99 s in the second control against 3.45 and 3.45 (the first control had
  no such turn). Python `scale-generator`'s 582-token turn 3 took 3.72 s in the first control and 3.73 and 3.71 s with
  `short=1400:64`. R225's replay read the same 577 tokens in the same micro-batches with either setting, 3.44, 3.44 and
  3.43 s with the control and 3.44 and 3.43 s with `short=1400:64`. The agent check's server log does not list each
  call's micro-batches, so what made those two calls slower in the controls is not shown here; neither is a
  restored-checkpoint request alone (the 561-token one restored none).
- **Generation speed** follows the tokens each decode step yields (3.29 and 3.26 in the `short=1400:64` runs, 3.30 and
  3.35 in the controls), which follows each run's replies, while the step itself took as long in both arms. The setting
  changes how a prompt call is cut into micro-batches and no decode kernel, so the 0.465 tok/s on the means, 0.055
  beyond the controls' spread, is the replies'. The first-token median of the second `short=1400:64` run falls in the
  gap between 2.27 and 2.81 s among its turns ([verdict_first_token.tsv](verdict_first_token.tsv)); the mean is within
  the spread and lower than both controls'.
- **Quality:** all four runs passed 12 of 12 with 0 malformed calls; failed tool calls 1 and 1 with `short=1400:64`, 0
  and 2 in the controls, each in a task that passed.
- **Recommendation:** registry row Q1641 set the rule before the runs: a gain, or negligible and slower on no figure,
  puts `short=1400:64` into Flash-Next's v5 preset. On time a task it is negligible, on the prompts it changes a gain,
  and no figure of what it changes is slower, so it qualifies; `release/flash-next-v5/settings.env` keeps
  `short=600:64` until the owner's word.

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
| [verdict_buckets.tsv](verdict_buckets.tsv) | per run, every turn by its new tokens in the taper's sizes (up to 600, 601 to 1,400, 1,401 to 2,047, 2,048 and more): turns, new tokens, prompt seconds, ms a prompt token |
| [verdict_matched.tsv](verdict_matched.tsv), [verdict_turns.tsv](verdict_turns.tsv) | the 12 requests that read the same prompt on the same cached prefix in all four runs, by size and one by one |
| [verdict_close.tsv](verdict_close.tsv) | the requests that are the same task and turn in all four runs with new and cached tokens within 2 % and in one size, one by one |
| [verdict_first_token.tsv](verdict_first_token.tsv) | per run, the first token of every turn: median, mean and the waits around the median |
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

The verdict's tables, from the four runs' `agent/<run>/turns.csv`:

```bash
D=docs/data/2026-11-20_r227_flash_taper1400_agent
python3 -I $D/run/verdict.py $D c600a=$D/agent/c600a/turns.csv t1400a=$D/agent/t1400a/turns.csv \
  c600b=$D/agent/c600b/turns.csv t1400b=$D/agent/t1400b/turns.csv
```

| Script | What it does |
|---|---|
| [run/flash_taper.sh](run/flash_taper.sh) | runs one arm: the v5 preset, or the same with `--ubatch-taper short=1400:64` |
| [run/dry_line.sh](run/dry_line.sh) | prints the command an arm would start, without loading anything |
| [run/preset_check.sh](run/preset_check.sh) | checks a started command against `release/flash-next-v5/settings.env`, with the taper's value given |
| [run/tables.sh](run/tables.sh) | builds this folder's tables from the runs done so far |
| [run/verdict.py](run/verdict.py) | the verdict's tables: every turn, the matched and the close requests by the taper's sizes, and the first token, for the four runs |

## Provenance and licence

Measured on the project's own rig on 2026-11-20 with the repository's own scripts; the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/<run>/attempts.csv`). No
model file or task text is in this folder. The tables and scripts here are under the repository's licence.
