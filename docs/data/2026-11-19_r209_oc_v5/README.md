# R209: the full overclock on the v5 engine, Ornith and Flash-Next on one boot (Q1603, 2026-11-19)

**In plain words.** Two separate gains landed on 2026-11-18. The second rig's full clock profile (a memory and a core
offset on every card, [R202b](../2026-11-18_r202b_core_soak/README.md)) made Ornith's coding agent about 8 % faster in
a first look on the v4.0 engine ([R203e](../2026-11-18_r203_ornith_offsets/README.md#the-full-profile-first-look-o4oc2-2026-11-18)),
and the v5 engine with Ornith's v5 preset made each decode step 6.3 % shorter at stock
([R204](../2026-11-18_r204_ornith_glu_agent/README.md)). This row asks whether they add up. On one fresh boot of the
second rig it runs five steps in order: Ornith at stock, Flash-Next at stock, the kit's `apply` of the full profile
with Ornith's whole-memory check, Flash-Next with the offsets, Ornith with the offsets; then every offset back to 0.
The comparison of the row is each model with the offsets against the same model at stock on this boot.

**The verdict: they add up, for both models.** With every card at the full profile, on the v5 engine, Ornith's coding
agent took 1.90 minutes a task at equal work against 2.00 at stock on the same boot (−5.00 %) and generated 19.26
against 17.87 tok/s (+7.78 %). Flash-Next's took 2.26 against 2.36 minutes (−4.24 %) and generated 30.69 against
27.82 tok/s (+10.32 %). Each moved beyond the spread of two stock runs of its preset, so for both models the
overclock is a **gain** on top of the v5 engine. Where the runs did the same work, a decode step at the same context
was 8.21 %, 9.17 %, 8.69 % and 8.77 % shorter for Ornith (its four context buckets) and 8.21 %, 8.49 % and 8.38 %
shorter for Flash-Next (its three), and the requests read on the same prompt were 5.02 % and 5.78 % faster. The
offsets shortened Ornith's decode step as much on the v5 engine as on v4.0's (R203e against R203a, two boots of
2026-11-18: 8.91 %, 8.75 % and 6.25 % in R203's three buckets), so neither gain takes from the other. The answers stayed sound in every run, the
hottest card reached 45 °C, the watchdog never tripped and the boot logged no GPU error.

Against v4.0's released engine at stock on this rig a day earlier (another boot, so context only): Ornith 2.13 minutes
a task and 16.42 tok/s ([R203a](../2026-11-18_r203_ornith_offsets/README.md#the-stock-half-o4oc0-2026-11-18)) against
1.90 and 19.26 with the whole stack (−10.80 % and +17.30 %), a decode step at R204's mix of contexts 59.85 against
51.12 ms (−14.59 %: −6.33 % from the engine and preset, −8.81 % from the offsets); Flash-Next 2.36 minutes and
28.08 tok/s ([R203c](../2026-11-18_r203_ornith_offsets/README.md#flash-next-at-stock-fnoc0-2026-11-18)) against 2.26
and 30.69 (−4.24 % and +9.29 %), its whole gain from the offsets, since its preset uses no v5 switch. These are this
rig's numbers with this rig's soaked profile: each card's offsets come from its own soak
([R202b](../2026-11-18_r202b_core_soak/README.md)), so another rig needs its own.

**Run 1, Ornith at stock on the v5 engine: 12 of 12 tasks passed, 2.00 minutes a task at equal work, generation
17.87 tok/s over the session and 17.84 tok/s at R204's mix of contexts.** This is the stock number the offset run on
the same boot will be compared with. As context only (another engine, or another rig): on the same rig at stock with
v4.0's engine and preset ([R203a](../2026-11-18_r203_ornith_offsets/README.md#the-stock-half-o4oc0-2026-11-18), a
day earlier) a decode step at the same mix of contexts took 6.33 % longer; on the first rig with this engine and preset
(R204's two runs) it took 0.64 % and 0.99 % longer.

**Run 2, Flash-Next at stock on the v5 engine: 12 of 12 tasks passed, 2.36 minutes a task at equal work, generation
27.82 tok/s.** This is the stock number for Flash-Next's offset run on the same boot. As context only: on the same rig
at stock with v4.0's engine a day earlier ([R203c](../2026-11-18_r203_ornith_offsets/README.md#flash-next-at-stock-fnoc0-2026-11-18))
a decode step at the same context took as long (this run +0.35 %, +0.08 % and −0.66 % in the three context buckets);
the v5 engine's gain is Ornith's own switch, which Flash-Next's preset does not use.

**Run 3, the kit's `apply` of the full profile with Ornith's whole-memory check: passed.** Every card was set to the
profile, read back as set, and gave its stock answer on its own. Then Ornith, served on all nine cards with the
offsets for 30 minutes, gave exactly the stock answer in all 58 rounds and generated 8.04 % faster on both of the
check's prompts (18.00 against 16.66 tok/s, the same tokens). The hottest card reached 39 °C, the watchdog never
tripped, and the boot logged no GPU error. This single-request number is a diagnostic; the verdict of the row is the
agent checks with the offsets that follow on the same boot.

**Run 4, Flash-Next with the offsets: a gain. 12 of 12 tasks passed, 2.26 minutes a task at equal work and generation
30.69 tok/s, against 2.36 minutes and 27.82 tok/s at stock on the same boot (−4.24 % and +10.32 %).** Both moved more
than two stock runs of the same preset move apart (2.53 % and 3.52 %). Where the two runs did the same work the cards
were faster too: a decode step took 8.21 %, 8.49 % and 8.38 % less time at the same context in the three context
buckets, and the 12 requests read on the same prompt took 5.78 % less time. With yesterday's profile, which raised the
core on three cards only, the same test on v4.0's engine was negligible
([R203d](../2026-11-18_r203_ornith_offsets/README.md#flash-next-with-the-offsets-fnoc1-2026-11-18)). The answers
stayed sound, the hottest card reached 42 °C, the watchdog never tripped and the boot logged no GPU error.

**Run 5, Ornith with the offsets: a gain. 12 of 12 tasks passed, 1.90 minutes a task at equal work and generation
19.26 tok/s, against 2.00 minutes and 17.87 tok/s at stock on the same boot (−5.00 % and +7.78 %).** Both moved more
than two stock runs of the same preset move apart (0.95 % and 2.42 %). Where the two runs did the same work the cards
were faster too: a decode step took 8.21 %, 9.17 %, 8.69 % and 8.77 % less time at the same context in the four
context buckets (8.81 % at R204's mix), and the 20 requests read on the same prompt took 5.02 % less time. The answers
stayed sound, the hottest card reached 45 °C, the watchdog never tripped and the boot logged no GPU error. Then every
offset went back to 0 and the watchdog stopped ([the rig at the end](#ornith-with-the-offsets-o5oc1-2026-11-19)).

**After the row: the profile again on the boot of 14:04 (R211, Q1621): passed.** The owner restarted the rig at 14:04
to keep the overclock for the afternoon's runs, since offsets set this way last only until a reboot. The kit set the
same profile again and checked every card on its own; all nine passed, with the same offsets read back. Ornith then
answered the check's two prompts twice with the offsets: both rounds gave this morning's stock answer token for token
and generated 18.01 and 18.01 tok/s in round 2, against 18.02 and 18.01 with the offsets this morning and 16.66 and
16.66 at stock. The offsets stay set for the afternoon's runs ([the section](#the-profile-again-on-the-boot-of-1404-2026-11-19)).

Registry rows Q1603 and Q1621 in [quants/experiments.md](../../../quants/experiments.md).

## The method

The agent check of [R166](../2026-11-16_r166_ornith_stack_no_taper/README.md) and
[R204](../2026-11-18_r204_ornith_glu_agent/README.md) ([bench/agent_check](../../../bench/agent_check/README.md):
12 bug-fix tasks, Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget, the makers' sampling
at temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF (valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`,
48,196,722,592 bytes) at 49,152 tokens on the second rig's nine P106-090 cards, one load a run:

- **The engine:** the v5 engine, `pascal-p106-v5` at `602691153` (upstream's b12483 merged); the rig tree's
  `llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`, checked before the run.
- **The preset:** Ornith's v5 preset ([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)): v4.0's
  preset plus `GGML_CUDA_MMID_GLU_UNFUSE=2`, through R204's harness call
  ([ornith_agent_glu.sh](../2026-11-18_r204_ornith_glu_agent/run/ornith_agent_glu.sh)). The started command is R204's
  `o4g2a` byte for byte (549 bytes with its newline, md5 `a0a97e752b2acbfef50bbaf86250e1e5`), and the server's
  `switches:` line is the same: `7 set: GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_KEEPALIVE=decode
  GGML_CUDA_KEEPALIVE_EAGER=1 GGML_CUDA_MMID_GLU_UNFUSE=2 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SERVER_YIELD=2`, 889 names
  read.
- **Flash-Next:** R203c's run ([run/flash_run.sh](../2026-11-18_r203_ornith_offsets/run/flash_run.sh): R156's driver,
  the released v4.0 preset) with the v5 engine's `llama-server`.
- **The offsets:** R202b's full profile ([use-profile.json](../2026-11-18_r202b_core_soak/use-profile.json): a memory
  and a core offset on every card), set by the offset kit's `apply`.

Arms: `o5oc0` (Ornith at stock), `f5oc0` (Flash-Next at stock), `f5oc1` and `o5oc1` (with the offsets). Beside them,
as context only: R203a's `o4oc0` (the same rig at stock on v4.0's engine and preset, 2026-11-18) and R204's `o4g2a` and
`o4g2b` (this engine and preset on the first rig, 2026-11-18).

## Ornith at stock: `o5oc0` (2026-11-19)

The rig was booted at 09:18:21. Before the run the offset kit's read-only start check passed: every card at 0 offset,
75 W, 1,911 / 4,004 MHz maximum, P8, 0 MiB, no load process, no GPU Xid, no step-down this boot, 3,168 MiB of host
memory available. The server started at 09:41:22 and was up after 585 s (R203a on this rig: 642 s; R204 on the first
rig: 683 and 681 s); the tasks ran 09:51:09–10:11:21. After the run the same check passed again, every offset still 0.

| | R203a `o4oc0` (rig 2, v4.0's engine and preset, 2026-11-18) | R204a `o4g2a` (rig 1, this engine and preset, 2026-11-18) | R204b `o4g2b` (rig 1, the same) | **R209a `o5oc0`, stock** |
|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.13 | 2.04 | 2.05 | **2.00** |
| generation tok/s, the whole session | 16.42 | 17.66 | 17.28 | **17.87** |
| ms a decode step at R204's mix of contexts (four buckets) | 59.85 | 56.42 | 56.62 | **56.06** |
| generation tok/s at that mix | 16.71 | 17.72 | 17.66 | **17.84** |
| prompt tok/s, the whole session | 102.2 | 98.5 | 102.8 | **101.9** |
| first token median s; p90 s | 7.07; 11.51 | 6.99; 11.18 | 6.97; 11.74 | **6.78; 11.30** |
| first token on a task's first turn, median s | 8.48 | 8.53 | 8.51 | **8.43** |
| the 20 requests read on the same prompt in all four runs, prompt s | 203.81 | 206.32 | 206.08 | **201.77** |
| of them, the 12 first turns, prompt s | 116.87 | 118.71 | 118.68 | **115.49** |
| prompt tokens read a task | 3,725 | 3,235 | 3,322 | **3,213** |
| tokens generated a task | 1,980 | 1,376 | 1,701 | **1,188** |
| reply tokens re-read (turns; their prompt s of all prompt s) | 8,860 (5; 65.6 of 437.3) | 3,406 (3; 31.2 of 394.3) | 4,592 (2; 31.4 of 387.5) | **3,431 (2; 22.8 of 378.4)** |
| minutes, all 12 tasks; turns | 31.9; 67 | 22.6; 62 | 26.6; 59 | **20.0; 60** |
| tool calls / failed / malformed | 67 / 2 / 0 | 62 / 1 / 0 | 59 / 0 / 0 | **60 / 2 / 0** |
| checkpoints restored / created / erased; full re-processing | 16 / 0 / 0; 0 | 14 / 0 / 0; 0 | 13 / 0 / 0; 0 | **13 / 0 / 0; 0** |
| graph reallocations | 0 in 67 requests | 0 in 62 | 0 in 59 | **0 in 60** |
| busy SM clock, mean of the nine cards' medians, MHz | 1,883.9 | 1,879.8 | 1,881.2 | **1,886.8** |
| lowest MemAvailable MiB; swap highest MiB | 1,207; 334 | 1,223; 343 | 1,205; 342 | **1,248; 336** |

Each value is one run ([agent.tsv](agent.tsv), [decode_r204_mix.md](decode_r204_mix.md),
[same_prompts.md](same_prompts.md), [work.md](work.md), [clocks.tsv](clocks.tsv), [host.tsv](host.tsv),
[reallocs.tsv](reallocs.tsv), [o5oc0_gonogo.md](o5oc0_gonogo.md); the other runs' tool calls and checkpoints from their
folders). Minutes a task at equal work price `ka`'s work (v3.0's 2026-11-06 run on the second rig: 2.52) at each run's
own prompt and generation speeds, so they compare the engine and the cards on the same work. The minutes for all tasks
follow what the agent wrote at temperature 0.6: this run's replies were the shortest of the four (1,188 tokens a task,
none past 8,000 tokens of context; the largest context any task reached was 7,155 tokens).

- **Decode by context** ([decode_r204_mix.md](decode_r204_mix.md)): this run had no request whose middle context
  reached 8,000 tokens, so it cannot be read at R204's five-bucket mix. The table uses R204's four runs' tokens over the
  four buckets every run here decoded in (under 3,000 5.0 %, 3,000 to 4,500 27.4 %, 4,500 to 6,000 49.9 %, 6,000 to
  8,000 17.7 %), and R204's runs are recomputed on that mix (their five-bucket figures in R204's
  [r204b_decode.md](../2026-11-18_r204_ornith_glu_agent/r204b_decode.md) are 57.22 and 57.70 ms). Bucket by bucket, ms
  a token, this run against `o4oc0`: 52.76 against 56.66 under 3,000 tokens (−6.88 %), 54.73 against 58.48 at 3,000 to
  4,500 (−6.41 %), 56.30 against 60.22 at 4,500 to 6,000 (−6.51 %), 58.35 against 61.86 at 6,000 to 8,000 (−5.67 %).
  Against `o4g2a` and `o4g2b`: −0.57 % and −0.68 %, −0.35 % and −0.71 %, −0.81 % and −1.09 %, −0.66 % and −1.19 %.
- **The same prompts** ([same_prompts.md](same_prompts.md)): 20 requests read the same number of new tokens on the same
  cached prefix in all four runs (every task's first turn and eight later turns). They took 201.77 s here, against
  203.81 s in `o4oc0` (−1.00 %) and 206.32 and 206.08 s in `o4g2a` and `o4g2b` (−2.21 %, −2.09 %). With the cached
  tokens matched within one ([same_prompts_tol1.md](same_prompts_tol1.md)) 22 requests: 219.14 s against 221.23,
  224.03 and 223.88 s.
- **The cards' busy clocks** ([clocks.tsv](clocks.tsv), from the card log `scripts/card_log.py` writes every 0.5 s,
  busy = utilization above 0 %, 09:51:09–10:11:21): medians of 1,860 1,885 1,885 1,873 1,911 1,898 1,873 1,898 1,898
  MHz on cards 0 to 8 (`o4oc0` on this rig: 1,860 1,885 1,885 1,860 1,898 1,898 1,873 1,898 1,898); busy samples below
  1,800 MHz 3.50 3.17 3.29 2.49 3.94 3.87 5.38 3.76 2.83 %.
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,248 MiB at 10:10:12, during the last task; swap
  0 MiB at the start and 336 MiB at its highest at 09:50:41, during the load. The RAM watchdog stopped nothing. Card
  peaks from the card log, 5,808 5,828 5,754 5,828 5,436 5,544 5,706 5,782 5,188 MiB, are `o4g2a`'s card for card. The
  report's memory gate is NO-GO on the swap, as for every Ornith run of this preset; every other gate is GO
  ([o5oc0_gonogo.md](o5oc0_gonogo.md)). Every task also passed its hidden tests; no NaN or inf.
- **The client:** the machine that drives the rig was also the agent's client; no other agent check used it as a
  client during the run.
- **The two failed tool calls** were two edits in Python `affine-cipher` whose old text had the wrong indentation;
  the agent read the file's exact line and redid the edit, and the task passed.

## Flash-Next at stock: `f5oc0` (2026-11-19)

**12 of 12 tasks passed, 2.36 minutes a task at equal work, generation 27.82 tok/s.** This is the stock number the
Flash-Next run with the offsets on the same boot will be compared with. A decode step at the same context took as long
as on v4.0's engine on this rig a day earlier (R203c `fnoc0`): +0.35 %, +0.08 % and −0.66 % in the three context
buckets both runs reached. The v5 engine's gain is Ornith's own switch, and Flash-Next's preset does not use it. The
generation speed is 0.93 % under `fnoc0`'s because the target accepted fewer drafted tokens (0.754 against 0.765).

The run is R203c's ([run/flash_run.sh](../2026-11-18_r203_ornith_offsets/run/flash_run.sh): R156's driver, the
harness's arm `v4r`, the released v4.0 preset
[release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env)) on the maker's Swift GSQ-RCO Q2_0
(`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`) with the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`)
at 65,536 tokens, one load, with the v5 engine's `llama-server` (md5 `86607661d5a19d1a90d19ee4db10c90b`, checked on the
rig before the run). The tables call it `f5oc0`. Its started command is R203c's (1,689 bytes) with only the engine's
folder name changed, and the server's `switches:` line names R156's 39 switches with the same values (this build reads
889 names, v4.0's 880).

It ran right after Ornith's stock run on the same boot. At 10:21:38 the kit's read-only state showed every card at 0
offset, 0 MiB, 75 W, 1,911 / 4,004 MHz maximum, P8, no load, no GPU Xid, no step-down this boot, 3,176 MiB of host
memory available. The server started at 10:21:47 and was up after 226 s (R203c on this rig: 227 s); the tasks ran
10:25:37–10:44:14. At 10:44:29 the kit's start check passed again, every offset still 0. The machine that drives the
rig was the agent's client; no other agent check used it as a client during the run.

Beside it, as context only: R203c's `fnoc0` (the same rig, preset and model at stock on v4.0's engine, 2026-11-18) and
C854's `v4rc2` (this engine, preset and model on the first rig, 2026-11-18,
[C854's part (6)](../2026-11-18_c854b_p106_check/README.md#part-6-the-agent-check-on-flash-next-2026-11-18)).

| | R203c `fnoc0` (rig 2, v4.0's engine, 2026-11-18) | C854 `v4rc2` (rig 1, this engine, 2026-11-18) | **R209b `f5oc0`, stock** |
|---|---|---|---|
| passed | 12 / 12 | 11 / 12 | **12 / 12** |
| minutes a task at equal work (R90 `v3`'s work) | 2.36 | 2.40 | **2.36** |
| generation tok/s | 28.08 | 27.00 | **27.82** |
| ms a decode step, middle context under 4,000 / 4,000 to 8,000 / 8,000 to 16,000 | 113.06 / 117.94 / 121.89 | 113.14 / 117.96 / 119.01 | **113.46 / 118.04 / 121.08** |
| draft acceptance (accepted of drafted tokens); tokens a decode step | 0.765; 3.30 | 0.740; 3.23 | **0.754; 3.27** |
| prompt tok/s, the whole session | 141.2 | 139.2 | **144.1** |
| first token median s; p90 s | 2.63; 8.34 | 2.52; 9.54 | **2.52; 8.32** |
| first token on a task's first turn, median s | 6.66 | 6.69 | **6.57** |
| the 13 requests read on the same prompt in all three runs, prompt s | 99.49 | 97.66 | **98.74** |
| of them, the 11 first turns, prompt s | 84.27 | 82.28 | **83.60** |
| prompt tokens read a task | 3,082 | 4,802 | **3,425** |
| tokens generated a task | 1,905 | 3,306 | **1,846** |
| reply tokens re-read (turns; their prompt s of all prompt s) | 0 (0; 0.0 of 261.9) | 4,854 (7; 51.6 of 413.9) | **1,876 (3; 17.0 of 285.2)** |
| minutes, all 12 tasks; turns | 18.4; 60 | 32.6; 92 | **18.5; 66** |
| minutes a task, median | 1.28 | 1.34 | **1.21** |
| tool calls / failed / malformed | 54 / 0 / 0 | 90 / 3 / 0 | **60 / 3 / 0** |
| graph reallocations | 0 in 60 requests | 0 in 91 | **0 in 66** |
| busy SM clock, mean of the nine cards' medians, MHz | 1,885.3 | 1,878.3 | **1,888.2** |
| lowest MemAvailable MiB; swap highest MiB | 414; 426 | 387; 464 | **382; 399** |

Each value is one run ([flash_agent.tsv](flash_agent.tsv), [flash_steps.md](flash_steps.md),
[flash_same_prompts_3runs.md](flash_same_prompts_3runs.md), [flash_work.md](flash_work.md), [flash_clocks.tsv](flash_clocks.tsv),
[flash_host.tsv](flash_host.tsv), [flash_reallocs.tsv](flash_reallocs.tsv), [f5oc0_gonogo.md](f5oc0_gonogo.md); the
other runs' tool calls from their `attempts.csv`). The two context rows reproduce their own folders' published values.
Minutes a task at equal work price R90 `v3`'s work (v3.0 on the first rig, 2026-11-11: 3.08) at each run's own prompt
and generation speeds.

- **Decode at the same context** ([flash_steps.md](flash_steps.md), from the server logs' timings, R203's
  [run/flash_steps.py](../2026-11-18_r203_ornith_offsets/run/flash_steps.py)): a decode step is one verify pass of the
  target plus the draft's passes, so its time does not depend on how many drafted tokens were accepted. Against
  `fnoc0`: +0.35 % under 4,000 tokens (19 and 16 requests), +0.08 % at 4,000 to 8,000 (42 and 38), −0.66 % at 8,000 to
  16,000 (5 and 6). Against `v4rc2` on the first rig: +0.28 %, +0.07 %, +1.74 % (5 and 16 requests). The tokens
  generated a second follow the acceptance: 0.722 under 4,000 tokens here, 0.757 in `fnoc0`.
- **The same prompts** ([flash_same_prompts_3runs.md](flash_same_prompts_3runs.md)): 13 requests read the same number of new
  tokens on the same cached prefix in all three runs (11 first turns, and the second turns of JavaScript `space-age`
  and `triangle`). They took 98.74 s here, against 99.49 s in `fnoc0` (−0.76 %) and 97.66 s in `v4rc2` (+1.11 %). One
  of them, JavaScript `variable-length-quantity`'s first turn, varies between runs (4.67 s here, 4.86 s and 3.66 s);
  without it the other 12 took 94.07 s against 94.63 s (−0.60 %) and 93.99 s (+0.08 %).
- **Work:** the agent took 66 turns (every task 4 to 7 turns but Python `affine-cipher`, 9) and generated 1,846
  tokens a task; the longest context was 11,316 tokens. The server read a reply again on 3 turns (Python `pig-latin`
  twice, Python `variable-length-quantity` once: 1,876 tokens, 17.0 of 285.2 prompt seconds); `fnoc0` on none.
- **The three failed tool calls** were the agent's own extra checks after the task's tests had passed: two scripts
  in Python `affine-cipher` that tested edge cases the tests do not cover and stopped on an error, and one in Python
  `variable-length-quantity` with a syntax error. Every task passed, also its hidden tests, with the test files
  untouched; 0 of 60 calls malformed, none inside the reasoning; no NaN or inf; 14 checkpoints restored, none created
  or erased, 0 full re-processing.
- **The cards' busy clocks** ([flash_clocks.tsv](flash_clocks.tsv), from the card log `scripts/card_log.py` writes
  every 0.5 s, busy = utilization above 0 %, 10:25:37–10:44:14): medians of 1,860 1,898 1,885 1,873 1,911 1,898 1,873
  1,898 1,898 MHz on cards 0 to 8 (mean 1,888.2), against 1,860 1,885 1,885 1,873 1,898 1,898 1,873 1,898 1,898 in
  `fnoc0` (1,885.3); busy samples below 1,800 MHz 3.56 1.96 2.91 2.06 2.13 1.97 2.70 2.73 1.06 %. These are the stock
  numbers `f5oc1`'s clocks compare with.
- **Host and cards** ([flash_host.tsv](flash_host.tsv)): lowest MemAvailable 382 MiB at 10:26:34, during the first
  task; swap 24 MiB at the start (left from Ornith's run) and 399 MiB at its highest at 10:31:20. The RAM watchdog
  stopped nothing. The card log's peaks, 5,190 5,234 5,092 5,250 4,348 5,108 4,502 5,120 5,698 MiB, are 16, 14, 16, 14,
  14, 16, 10, 16 and 4 MiB above `fnoc0`'s on cards 0 to 8. The report's memory gate is NO-GO on the 382 MB low and the swap, as for R156,
  C854's part (6) and R203c; every other gate is GO ([f5oc0_gonogo.md](f5oc0_gonogo.md)).
- **No class:** this run is the stock reference of R209's Flash-Next comparison, `f5oc1` against `f5oc0` on this
  boot. The differences to `fnoc0` and `v4rc2` mix the engine with the day or the rig.

## The full profile's `apply` and Ornith's whole-memory check (2026-11-19)

**In plain words.** Run 3 set the full clock profile on all nine cards and checked it on the model the rig serves.
First Ornith answered the kit's two fixed prompts twice at stock, the reference. Then the kit set each card's memory
and core offset, read each one back and checked every card on its own with a small model. Then Ornith answered the
same two prompts again and again for 30 minutes with the offsets set. All 58 rounds gave exactly the stock answer,
token for token, and generated 8.04 % faster on both prompts: 18.00 tok/s against 16.66. No card went above 39 °C, the
watchdog never tripped, and the boot logged no GPU error. The offsets stay set for runs 4 and 5.

[run/apply_check.sh](run/apply_check.sh) ran it on the rig, detached, in the kit's folder: `soak_chain.sh`'s steps 1,
2 and 4 without the soak, since the profile held its soak in [R202b](../2026-11-18_r202b_core_soak/README.md).

1. **The start check** at 11:00:44 (every offset 0, P8, 0 MiB, no load process, no GPU Xid, no step-down this boot,
   3,143 MiB of host memory available a minute before).
2. **Ornith at stock** with the v5 engine and Ornith's v5 preset ([release/ornith-v5](../../../release/ornith-v5/settings.env),
   copied into the kit, md5 `bc73c0439a83f0556f88a2dfc7ea9292`), the kit's `model_check.sh` with `ROUNDS=2 MINUTES=0`:
   loaded in 658 s, two rounds 11:11:50–11:13:07. Each round is the kit's greedy check, two prompts of 256 tokens at
   temperature 0 with every token's top-10 probabilities hashed.
3. **The start check again, the watchdog** (11:13:10: every 0.5 s, a card trips above 70 °C or above 80 W held for
   5 s, and on its own Xid) **and `apply`** of R202b's [use-profile.json](../2026-11-18_r202b_core_soak/use-profile.json)
   (md5 `b32956cd064d8b449bdd6ac2b8a6d4f7` in the kit folder), 11:13:17–11:18:35: a check server on every card with the
   kit's check model ([R200](../2026-11-17_r200_climb_rig2/README.md)'s, about 2 GB of each card) on the kit's engine
   (v4.0's build, the same for every round of the step), two stock rounds, every card's offsets set 11:17:36–11:17:43
   and each read back as set, then one round with the offsets: every card PASS at 11:18:28.
4. **Ornith with the offsets**, the same engine, preset and output folder as step 2: loaded in 663 s, then 58 rounds
   in 30 minutes, 11:29:43–12:00:12, each compared with stock's round 1. The two checks started the same server
   command, and both servers' `switches:` lines are the v5 preset's (7 set, 889 names read).

**Ornith's whole-memory check** ([model_check.md](model_check.md); every round in
[model_rounds.tsv](model_rounds.tsv)):

| | stock | with the offsets | change |
|---|---|---|---|
| load, s | 658 | 663 | |
| host memory available when loaded, MiB | 1,496 | 1,492 | |
| rounds; the same hash as stock's round 1 | 2; 1 (round 1 is the reference) | 58; 58 | |
| hash | `adf47784f9b02d4b` | `adf47784f9b02d4b` in every round | |
| generation tok/s in round 1 (prompts 1, 2) | 15.27, 16.15 | 16.42, 17.44 | +7.53 %, +7.99 % |
| generation tok/s in round 2 (prompts 1, 2) | 16.66, 16.66 | 18.02, 18.01 | +8.16 %, +8.10 % |
| generation tok/s, median of rounds 2 on (prompts 1, 2) | 16.66, 16.66 (round 2) | 18.00, 18.00 (57 rounds) | **+8.04 %, +8.04 %** |
| prompt processing tok/s in round 2 (prompts 1, 2) | 26.95, 29.43 | 29.09, 31.84 | +7.94 %, +8.19 % |
| prompt processing tok/s, median of rounds 2 on (prompts 1, 2) | 26.95, 29.43 (round 2) | 29.05, 32.06 (57 rounds) | +7.79 %, +8.94 % |

- **Round 1 is each load's first request** and runs slower in both checks; prompt 1's prompt processing in round 1
  read 13.05 tok/s at stock and 12.05 with the offsets. From round 2 on the check barely moves: with the offsets the
  57 rounds read 17.99 tok/s at the lowest and 18.02 at the highest on prompt 1, 17.99 and 18.01 on prompt 2, so
  stock's one round after the warm-up is the reference.
- **Both loads used the same memory:** 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 MiB on cards 0 to 8.
- The hash is the one R200 and R201 recorded with v4.0's engine and preset
  ([R201](../2026-11-18_r201_soak_ladder/README.md)): the v5 engine with Ornith's v5 preset answers these two prompts
  as v4.0 did.
- As context only (another boot, profile, engine and preset): R201b's single-request check of the three-card use
  profile on v4.0's engine gave +4.42 % (16.29 against 15.60 tok/s, medians of all rounds); the first look at this
  profile in Ornith's agent check, R203e on v4.0's engine, gave +8.95 % generation against a stock run on an earlier
  boot.

**`apply`, card by card** ([apply.md](apply.md); the kit's own table [o5_apply_steps.csv](o5_apply_steps.csv) with
both stock rounds' times and limits):

| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | change | peak core MHz at stock | with the offsets | greedy check | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | +1,300 | +256 | 179.11 | 206.96 | +15.55 % | 1,873 | 2,126 | same as stock | PASS |
| 1 | +1,000 | +256 | 179.21 | 199.44 | +11.29 % | 1,898 | 2,164 | same as stock | PASS |
| 2 | +1,000 | +269 | 179.11 | 199.52 | +11.40 % | 1,885 | 2,151 | same as stock | PASS |
| 3 | +1,100 | +231 | 179.18 | 201.98 | +12.72 % | 1,873 | 2,100 | same as stock | PASS |
| 4 | +1,100 | +206 | 179.27 | 201.96 | +12.66 % | 1,911 | 2,113 | same as stock | PASS |
| 5 | +1,200 | +206 | 179.12 | 204.62 | +14.24 % | 1,898 | 2,100 | same as stock | PASS |
| 6 | +900 | +282 | 179.27 | 197.34 | +10.08 % | 1,885 | 2,164 | same as stock | PASS |
| 7 | +900 | +256 | 179.24 | 197.49 | +10.18 % | 1,898 | 2,151 | same as stock | PASS |
| 8 | +1,300 | +231 | 179.29 | 207.41 | +15.68 % | 1,911 | 2,126 | same as stock | PASS |

Every card's offsets are the profile's, read back as set; every card's check model gave its stock hash
(`60ee0f178cfd67d5` on all nine) and passed the 31 kernel checks with the offsets. As context, the same `apply` of the
same profile on the boot of 2026-11-18 20:47 ([R203e](../2026-11-18_r203_ornith_offsets/README.md#the-full-profile-first-look-o4oc2-2026-11-18))
also passed on every card; its peaks with the offsets were the same on cards 0, 2, 3, 4, 5 and 7, one driver step
lower on cards 1 and 6 (2,151 MHz) and one higher on card 8 (2,138 MHz).

**Heat, power and clocks** ([model_cards.md](model_cards.md), the watchdog's 3,658 samples a card over the 30 minutes
with the offsets; the watchdog started after the stock check, so there are no stock samples):

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| highest °C | 39 | 36 | 33 | 31 | 36 | 32 | 39 | 34 | 34 |
| highest W | 77.82 | 83.53 | 81.77 | 82.29 | 78.66 | 77.99 | 78.78 | 80.30 | 80.46 |
| samples above 80 W | 0 | 8 | 4 | 4 | 0 | 0 | 0 | 2 | 2 |
| SM clock median, MHz | 2,113 | 2,151 | 2,151 | 2,100 | 2,113 | 2,100 | 2,164 | 2,151 | 2,126 |
| memory clock median, MHz | 4,657 | 4,498 | 4,498 | 4,551 | 4,551 | 4,608 | 4,455 | 4,455 | 4,657 |

The hottest cards were cards 0 and 6 at 39 °C. Power went above 80 W only in single samples, never for the 5 s that
trip the watchdog; the highest was 83.53 W on card 1. No sample tripped, and the boot logged no GPU Xid.

**The rig at the end** (12:00:16, the kit's `state`): every card at its profile value, P0, no load process, the
watchdog running, 0 GPU Xid, 0 step-downs, 3,121 MiB of host memory available. Runs 4 and 5 start from this state.

## Flash-Next with the offsets: `f5oc1` (2026-11-19)

**12 of 12 tasks passed, 2.26 minutes a task at equal work, generation 30.69 tok/s. Against `f5oc0` on the same boot
(2.36 minutes, 27.82 tok/s) that is −4.24 % and +10.32 %, both beyond the spread of two stock runs of the same preset
(R156a and R156b: 2.43 and 2.37 minutes, 2.53 % apart; 26.67 and 27.61 tok/s, 3.52 % apart): a gain on the agent's
speed.** It holds where the two runs did the same work. At the same context a decode step took 8.21 %, 8.49 % and
8.38 % less time in the three context buckets, and the 12 requests read on the same prompt took 5.78 % less time,
every one of them faster.

**The run:** `f5oc0`'s launcher with its own output, rig and client folders, and the kit's state read with run 3's
work folder, where the watchdog runs. No `apply` of its own: the offsets `apply` set in run 3 stayed set. At 12:10:09
the kit's state showed every card at its profile value, P8, 0 MiB, no load, the watchdog running, no GPU Xid, no
step-down this boot and 3,159 MiB of host memory available. The card log's start line read the profile's offsets on
every card. The started command (1,689 bytes) is `f5oc0`'s byte for byte, and so is the server's `switches:` line
after `switches:`. The server started at 12:10:18 and was up after 227 s (`f5oc0`: 226 s), and the tasks ran
12:14:08–12:30:57. At 12:31:16 every card still read its profile value, with the watchdog running, no GPU Xid and no
step-down. The kit's start check said FAIL, as it must with offsets set. An agent check on the first rig used the same
client during these tasks, with its own ports and folders.

| | R209b `f5oc0`, stock | **R209d `f5oc1`, the full profile** | change | as context: R203d's change (the three-card profile on v4.0's engine) |
|---|---|---|---|---|
| passed | 12 / 12 | **12 / 12** | | 12 / 12 → 11 / 12 |
| minutes a task at equal work (R90 `v3`'s work) | 2.36 | **2.26** | −4.24 % | +1.27 % |
| generation tok/s | 27.82 | **30.69** | +10.32 % | −3.38 % |
| ms a decode step, middle context under 4,000 / 4,000 to 8,000 / 8,000 to 16,000 | 113.46 / 118.04 / 121.08 | **104.15 / 108.02 / 110.93** | −8.21 / −8.49 / −8.38 % | −2.65 / −1.92 / −0.66 % |
| requests in those buckets | 19 / 42 / 5 | **17 / 39 / 5** | | |
| ms a decode step, all requests | 117.51 | **107.75** | −8.31 % | |
| draft acceptance (accepted of drafted tokens); tokens a decode step | 0.754; 3.27 | **0.767; 3.31** | | 0.765 → 0.737 |
| prompt tok/s, the whole session | 144.1 | **152.5** | +5.83 % | +2.20 % |
| first token median s; p90 s | 2.52; 8.32 | **2.66; 7.91** | +5.56 %; −4.93 % | +13.31 %; −0.96 % |
| first token on a task's first turn, median s | 6.57 | **6.25** | −4.87 % | −2.40 % |
| first token on the later turns, median s | 2.16 | **1.88** | −13.25 % | |
| the 12 requests read on the same prompt in all four runs, prompt s | 91.84 | **86.53** | −5.78 % | |
| of them, the 11 first turns, prompt s | 83.60 | **78.69** | −5.86 % | −0.99 % |
| prompt tokens read a task | 3,425 | **3,145** | −8.18 % | +5.84 % |
| tokens generated a task | 1,846 | **1,855** | | |
| reply tokens re-read (turns; their prompt s of all prompt s) | 1,876 (3; 17.0 of 285.2) | **941 (1; 6.9 of 247.4)** | | |
| minutes, all 12 tasks; turns | 18.5; 66 | **16.7; 61** | | |
| minutes a task, median | 1.21 | **1.04** | | |
| tool calls / failed / malformed | 60 / 3 / 0 | **56 / 1 / 0** | | |
| checkpoints restored / created / erased; full re-processing | 14 / 0 / 0; 0 | **12 / 0 / 0; 0** | | |
| graph reallocations | 0 in 66 requests | **0 in 61** | | |
| busy SM clock, mean of the nine cards' medians, MHz | 1,888.2 | **2,127.0** | +12.65 % | +3.95 % |
| highest card temperature °C; highest power W | 42; 79.74 | **42; 87.80** | | |
| watchdog trips; GPU Xid | no watchdog at stock; 0 | **0; 0** | | 0; 0 |
| lowest MemAvailable MiB; swap highest MiB | 382; 399 | **384; 589** | | |

Each value is one run ([flash_agent.tsv](flash_agent.tsv), [flash_steps.md](flash_steps.md),
[flash_same_prompts.md](flash_same_prompts.md), [flash_work.md](flash_work.md), [flash_clocks.tsv](flash_clocks.tsv),
[flash_cards.md](flash_cards.md), [flash_host.tsv](flash_host.tsv), [flash_reallocs.tsv](flash_reallocs.tsv),
[f5oc1_gonogo.md](f5oc1_gonogo.md); the tool calls from `attempts.csv`, the later turns' first token from
`turns.csv`). R203d's column is the change its run `fnoc1` showed against its own stock run `fnoc0` on 2026-11-18
([R203](../2026-11-18_r203_ornith_offsets/README.md#flash-next-with-the-offsets-fnoc1-2026-11-18)).

- **The verdict: gain.** The two numbers the class is set on moved beyond the spread of two stock runs: minutes a task
  at equal work by 4.24 % (the spread 2.53 %), generation by 10.32 % (3.52 %). Both are session-wide, so they also
  carry the work: this run read 8.18 % fewer prompt tokens a task and accepted more drafted tokens (0.767 against
  0.754). The like-for-like measures do not carry the work, and they agree. A decode step at the same context was
  8.21 %, 8.49 % and 8.38 % shorter, where two stock runs of this preset on this rig, on two engines a day apart, differ
  by +0.35 %, +0.08 % and −0.66 % (`f5oc0` against R203c's `fnoc0`). The same prompts were 5.78 % faster, where those
  two stock runs differ by 0.76 %. So one run of each arm is enough to set the class. The size of the generation gain
  is less sure than its sign: the shorter decode step alone gives +9.06 % est. at equal acceptance, and the higher
  acceptance the rest.
- **Why yesterday's test was negligible:** R203d's profile raised the core on cards 0, 4 and 8 only, and its decode
  step was 2.65 %, 1.92 % and 0.66 % shorter. This profile raises the core on every card (busy SM clocks +12.65 % on
  the cards' mean, against +3.95 %), and the step is about 8 % shorter in every bucket. The engine is not the
  difference: Flash-Next's preset decodes as fast on v4.0's engine as on this one (`f5oc0` above). A decode step passes
  through all nine cards in turn, so with three cores raised the other six likely held it back.
- **Decode at the same context** ([flash_steps.md](flash_steps.md), from the server logs' timings, R203's
  [run/flash_steps.py](../2026-11-18_r203_ornith_offsets/run/flash_steps.py)): under 4,000 tokens 104.15 against
  113.46 ms a step (17 and 19 requests), 32.87 against 28.04 tok/s; at 4,000 to 8,000 tokens, where 39 and 42 of the
  requests fall, 108.02 against 118.04 ms, 30.41 against 27.85 tok/s; at 8,000 to 16,000 tokens 110.93 against
  121.08 ms (5 and 5 requests), 31.05 against 26.71 tok/s. Tokens a second move with the acceptance too (0.800
  against 0.722 under 4,000 tokens); a step's time does not.
- **The same prompts** ([flash_same_prompts.md](flash_same_prompts.md)): 12 requests read the same number of new
  tokens on the same cached prefix in all four runs (11 first turns and JavaScript `triangle`'s second turn; this run's
  `space-age` took a different second turn, so the set is one smaller than `f5oc0`'s 13,
  [flash_same_prompts_3runs.md](flash_same_prompts_3runs.md)). They took 86.53 s against 91.84 s (−5.78 %). All 12
  were faster: by 3.55 % (Python `variable-length-quantity`) to 12.32 % (JavaScript `triangle`'s first turn); Python
  `affine-cipher`'s 2,780-token first turn, read with nothing cached, 27.49 s against 28.66 s (−4.07 %). Without
  JavaScript `variable-length-quantity`'s first turn, which varies between runs, the other 11 took 82.17 s against
  87.17 s (−5.73 %). Prompt reading is mostly compute, so it follows the cores: with three cores raised it was 0.99 %
  faster.
- **Work:** the agent took 61 turns (every task 4 to 7 turns) and generated 1,855 tokens a task; the longest context
  was 12,712 tokens (JavaScript `variable-length-quantity`'s seventh turn). The server read a reply again on 1 turn
  (Python `pig-latin`, 941 tokens, 6.9 of 247.4 prompt seconds); `f5oc0` on 3.
- **First token follows the turn mix.** A task's first turn took a median 6.25 s (6.57 s) and the later turns 1.88 s
  (2.16 s), both faster. This run had 49 later turns against 54, so its 12 first turns, the slowest, weigh more in the
  median of all turns, which rose to 2.66 s (2.52 s).
- **The answers looked sound.** Every task passed, also its hidden tests, with the test files untouched. None of the
  56 tool calls was malformed, none was inside the reasoning, and there was no NaN or inf in the server log; 12
  checkpoints restored, none created or erased, 0 full re-processing. The one failed call was the agent's own extra
  check in JavaScript `variable-length-quantity` after the task's tests had passed: a scratch script outside the task
  folder imported the solution by a relative path that did not resolve, and the next call fixed the path and ran it.
  The report's memory gate is NO-GO on the 385 MB low and the swap, as for `f5oc0`; every other gate is GO
  ([f5oc1_gonogo.md](f5oc1_gonogo.md)).
- **The cards' busy clocks** ([flash_clocks.tsv](flash_clocks.tsv), busy = utilization above 0 %, 12:14:08–12:30:57):
  medians of 2,113 2,138 2,151 2,100 2,113 2,100 2,151 2,151 2,126 MHz on cards 0 to 8 (mean 2,127.0), against
  `f5oc0`'s 1,860 1,898 1,885 1,873 1,911 1,898 1,873 1,898 1,898: +253, +240, +266, +227, +202, +202, +278, +253 and
  +228 MHz while busy, against core offsets of +256, +256, +269, +231, +206, +206, +282, +256 and +231 MHz. Busy
  samples below 1,800 MHz fell to 0.67 0.72 0.67 0.33 0.84 0.89 0.84 0.79 0.58 % (`f5oc0`: 3.56 1.96 2.91 2.06 2.13
  1.97 2.70 2.73 1.06 %). The memory ran at busy medians of 4,657 4,498 4,498 4,551 4,551 4,608 4,455 4,455 4,657 MHz
  against 4,006 on every card at stock ([flash_cards.md](flash_cards.md)).
- **Heat and power** ([flash_cards.md](flash_cards.md), the card log every 0.5 s over the tasks): the hottest cards
  were cards 0 and 6 at 42 °C (`f5oc0`: card 0 at 42 °C). Power went above 80 W only in single samples: 1, 0, 2, 2, 1,
  0, 1, 0 and 3 samples on cards 0 to 8, of 2,020 each. The highest was 87.80 W on card 8 at 12:26:57 (`f5oc0`:
  79.74 W on card 1, none above 80 W). The watchdog trips at 80 W held for 5 s; it never tripped, and the boot logged
  no GPU Xid.
- **Host and cards** ([flash_host.tsv](flash_host.tsv)): lowest MemAvailable 384 MiB at 12:17:04, during the third
  task; swap 52 MiB at the start (left from the earlier runs) and 589 MiB at its highest at 12:24:24. The RAM
  watchdog stopped nothing. The card log's peaks, 5,190 5,234 5,092 5,250 4,348 5,108 4,502 5,120 5,698 MiB on cards 0
  to 8, are `f5oc0`'s on every card.
- **Graph reallocations:** 0 in 61 requests, in every prompt-size bucket ([flash_reallocs.tsv](flash_reallocs.tsv)).
- **The rig at the end:** the offsets stayed set and the watchdog kept running for Ornith's offset run on the same
  boot (below).

## Ornith with the offsets: `o5oc1` (2026-11-19)

**12 of 12 tasks passed, 1.90 minutes a task at equal work, generation 19.26 tok/s. Against `o5oc0` on the same boot
(2.00 minutes, 17.87 tok/s) that is −5.00 % and +7.78 %, both beyond the spread of two stock runs of the same preset
(R166a and R166b: 2.11 and 2.13 minutes, 0.95 % apart; 16.52 and 16.13 tok/s, 2.42 % apart): a gain on the agent's
speed.** It holds where the two runs did the same work. At the same context a decode step took 8.21 %, 9.17 %, 8.69 %
and 8.77 % less time in the four context buckets, 8.81 % at R204's mix, and the 20 requests read on the same prompt
took 5.02 % less time.

**The run:** `o5oc0`'s launcher with its own run and rig folders and the same client folder, and the kit's state read
with run 3's work folder, where the watchdog runs. No `apply` of its own: the offsets `apply` set in run 3 stayed set
through run 4 and this run. At 12:46:24 the kit's state showed every card at its profile value, P8, 0 MiB, no load,
the watchdog running, no GPU Xid, no step-down this boot and 3,125 MiB of host memory available. The card log's start
line read the profile's offsets on every card. The started command (549 bytes) is `o5oc0`'s byte for byte, and so is
the server's `switches:` line. The server started at 12:46:33 and was up after 661 s (`o5oc0`: 585 s), and the tasks
ran 12:57:36–13:21:55. At 13:22:31 every card still read its profile value, with the watchdog running, no trip, no
GPU Xid and no step-down. An agent check on the first rig used the same client from 12:50:12 to 13:17:59, with its own
ports, folders and container names; that touches only the time outside the server, which minutes a task at equal
work leave out.

| | R209a `o5oc0`, stock | **R209e `o5oc1`, the full profile** | change | as context: R203e's change against R203a (the full profile on v4.0's engine, two boots) |
|---|---|---|---|---|
| passed | 12 / 12 | **12 / 12** | | 12 / 12 → 12 / 12 |
| minutes a task at equal work (`ka`'s work) | 2.00 | **1.90** | −5.00 % | −7.98 % |
| generation tok/s, the whole session | 17.87 | **19.26** | +7.78 % | +8.95 % |
| ms a decode step at R204's mix of contexts (four buckets) | 56.06 | **51.12** | −8.81 % | |
| generation tok/s at that mix | 17.84 | **19.56** | +9.64 % | |
| ms a decode step, middle context under 3,000 / 3,000 to 4,500 / 4,500 to 6,000 / 6,000 to 8,000 | 52.76 / 54.73 / 56.30 / 58.35 | **48.43 / 49.71 / 51.41 / 53.23** | −8.21 / −9.17 / −8.69 / −8.77 % | in R203's three buckets −8.91 / −8.75 / −6.25 % |
| requests in those buckets | 12 / 14 / 28 / 6 | **12 / 18 / 22 / 12** (and 3 at 8,000 and over) | | |
| prompt tok/s, the whole session | 101.9 | **101.6** | −0.29 % | +7.53 % |
| first token median s; p90 s | 6.78; 11.30 | **5.37; 10.04** | −20.80 %; −11.15 % | +4.53 %; −6.60 % |
| first token on a task's first turn, median s | 8.43 | **7.94** | −5.81 % | −5.54 % |
| first token on the later turns, median s | 4.77 | **4.40** | −7.76 % | |
| the 20 requests read on the same prompt in all five runs, prompt s | 201.77 | **191.64** | −5.02 % | 19 requests in R203's three runs: −6.45 % |
| of them, the 12 first turns, prompt s | 115.49 | **110.65** | −4.19 % | −6.39 % |
| prompt tokens read a task | 3,213 | **3,138** | | |
| tokens generated a task | 1,188 | **1,685** | | |
| reply tokens re-read (turns; their prompt s of all prompt s) | 3,431 (2; 22.8 of 378.4) | **2,078 (3; 23.4 of 370.6)** | | |
| minutes, all 12 tasks; turns | 20.0; 60 | **24.1; 67** | | |
| minutes a task, median | 1.43 | **1.29** | | |
| tool calls / failed / malformed | 60 / 2 / 0 | **67 / 1 / 0** | | |
| checkpoints restored / created / erased; full re-processing | 13 / 0 / 0; 0 | **14 / 0 / 0; 0** | | |
| graph reallocations | 0 in 60 requests | **0 in 67** | | |
| busy SM clock, mean of the nine cards' medians, MHz | 1,886.8 | **2,127.0** | +12.73 % | +12.83 % |
| highest card temperature °C; highest power W | 44; 80.86 | **45; 89.80** | | |
| watchdog trips; GPU Xid | no watchdog at stock; 0 | **0; 0** | | 0; 0 |
| lowest MemAvailable MiB; swap highest MiB | 1,248; 336 | **1,183; 362** | | |

Each value is one run ([agent.tsv](agent.tsv), [decode_r204_mix.md](decode_r204_mix.md),
[same_prompts.md](same_prompts.md), [work.md](work.md), [clocks.tsv](clocks.tsv), [ornith_cards.md](ornith_cards.md),
[host.tsv](host.tsv), [reallocs.tsv](reallocs.tsv), [o5oc1_gonogo.md](o5oc1_gonogo.md); the tool calls from
`attempts.csv`, the later turns' first token from `turns.csv`). R203e's column is the change its run `o4oc2` showed
against R203a's stock run `o4oc0`, a different boot of 2026-11-18
([R203](../2026-11-18_r203_ornith_offsets/README.md#the-full-profile-first-look-o4oc2-2026-11-18)), in R203's own
buckets and same-prompt set.

- **The verdict: gain.** The two numbers the class is set on moved beyond the spread of two stock runs: minutes a task
  at equal work by 5.00 % (the spread 0.95 %), generation by 7.78 % (2.42 %). The like-for-like measures agree. A
  decode step at the same context was 8.21 %, 9.17 %, 8.69 % and 8.77 % shorter; at R204's mix it was 8.81 % shorter,
  where two stock runs of this engine and preset on the first rig differ by 0.35 % (R204's `o4g2a` and `o4g2b`), and
  this rig's stock run differs from them by 0.64 % and 0.99 %. The same prompts were 5.02 % faster, where those two
  stock runs differ by 0.12 %.
- **Generation over the session gained less than at the same mix** (+7.78 % against +9.64 %) because this run's
  replies sat deeper: 12 requests with their middle context at 6,000 to 8,000 tokens and 3 above 8,000, against 6 and
  none, and a decode step is slower at a longer context. The run also did more work: 1,685 tokens generated a task
  against 1,188, most of the difference in Python `scale-generator` (6,172 tokens against 2,903, one reply of 5,405
  tokens; 362 s against 197 s). Minutes a task at equal work price the same work at each run's own speeds, so this
  does not move them.
- **Decode at the same context** ([decode_r204_mix.md](decode_r204_mix.md)): under 3,000 tokens 48.43 against
  52.76 ms a token (12 and 12 requests), 20.65 against 18.95 tok/s; at 3,000 to 4,500 tokens 49.71 against 54.73 ms
  (18 and 14), 20.12 against 18.27 tok/s; at 4,500 to 6,000 tokens 51.41 against 56.30 ms (22 and 28), 19.45 against
  17.76 tok/s; at 6,000 to 8,000 tokens 53.23 against 58.35 ms (12 and 6), 18.79 against 17.14 tok/s. Ornith decodes
  one token a step (no drafts), so its step and its token are the same.
- **The same prompts** ([same_prompts.md](same_prompts.md)): the same 20 requests as run 1's table (every task's
  first turn and eight later turns), now read on the same prompt in all five runs. They took 191.64 s against 201.77 s
  (−5.02 %); 18 of the 20 were faster, by 4.43 % (Python `affine-cipher`'s 2,742-token first turn, read with nothing
  cached, 25.94 s against 27.15 s) to 6.49 % (Python `variable-length-quantity`'s second turn). The two that were not
  are short first turns: JavaScript `grade-school` (403 new tokens) +3.57 % and `triangle` (430) +0.04 %. With the
  cached tokens matched within one ([same_prompts_tol1.md](same_prompts_tol1.md)) 22 requests: 208.06 s against
  219.14 s (−5.05 %).
- **Prompts over the session look flat** (101.6 against 101.9 tok/s) only through the mix: 43 of this run's 67
  requests read fewer than 512 new tokens, against 36 of 60, and such short reads are dominated by a request's fixed
  cost ([reallocs.tsv](reallocs.tsv)). The same prompts above are the like-for-like measure.
- **First token follows the turn mix.** A task's first turn took a median 7.94 s (8.43 s) and the later turns 4.40 s
  (4.77 s), both faster. This run had 55 later turns against 48, so the fast later turns weigh more in the median of
  all turns, which fell 20.80 % to 5.37 s.
- **The answers looked sound.** Every task passed, also its hidden tests, with the test files untouched. None of the
  67 tool calls was malformed, none was inside the reasoning, and there was no NaN or inf in the server log; 14
  checkpoints restored, none created or erased, 0 full re-processing. The one failed call was the agent's own probe in
  Python `variable-length-quantity`: it fed the decoder an incomplete byte sequence, which raised the error it should,
  and the agent then fixed the bug and all 31 tests passed. The report's memory gate is NO-GO on the swap, as for every
  Ornith run of this preset; every other gate is GO ([o5oc1_gonogo.md](o5oc1_gonogo.md), over both arms, as the
  client's results folder held both).
- **The cards' busy clocks** ([clocks.tsv](clocks.tsv), busy = utilization above 0 %, 12:57:36–13:21:55): medians of
  2,113 2,138 2,151 2,100 2,113 2,100 2,151 2,151 2,126 MHz on cards 0 to 8 (mean 2,127.0), against `o5oc0`'s
  1,860 1,885 1,885 1,873 1,911 1,898 1,873 1,898 1,898: +253, +253, +266, +227, +202, +202, +278, +253 and +228 MHz
  while busy, against core offsets of +256, +256, +269, +231, +206, +206, +282, +256 and +231 MHz. Busy samples below
  1,800 MHz fell to 0.65 0.64 0.56 0.52 1.05 1.02 1.15 1.21 1.02 % (`o5oc0`: 3.50 3.17 3.29 2.49 3.94 3.87 5.38 3.76
  2.83 %). The memory ran at busy medians of 4,657 4,498 4,498 4,551 4,551 4,608 4,455 4,455 4,657 MHz against 4,006
  on every card at stock ([ornith_cards.md](ornith_cards.md)).
- **Heat and power** ([ornith_cards.md](ornith_cards.md), the card log every 0.5 s over the tasks, 2,920 samples a
  card): the hottest card was card 0 at 45 °C (`o5oc0`: card 0 at 44 °C). Power went above 80 W only in single
  samples, never two in a row: 0, 3, 2, 7, 1, 0, 1, 2 and 9 samples on cards 0 to 8. The highest was 89.80 W on card 7
  at 13:07:15 (`o5oc0`: 80.86 W on card 7, 2 samples above 80 W). The watchdog's own samples over the same window
  (2,917 a card) had 0, 3, 4, 3, 2, 0, 1, 1 and 2 single samples above 80 W, the highest 87.92 W on card 4. It trips at
  80 W held for 5 s; it never tripped, and the boot logged no GPU Xid.
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,183 MiB at 13:22:10, as the server stopped
  (1,188 MiB at 13:21:40 during the last task); swap 52 MiB at the start (left from the earlier runs) and 362 MiB at
  its highest at 12:53:30, during the load. The RAM watchdog stopped nothing. The card log's peaks, 5,830 5,846 5,776
  5,846 5,454 5,562 5,724 5,796 5,202 MiB on cards 0 to 8, are R204's `o4g2b`'s card for card.
- **Graph reallocations:** 0 in 67 requests, in every prompt-size bucket ([reallocs.tsv](reallocs.tsv),
  [o5oc1_requests_reallocs.tsv](o5oc1_requests_reallocs.tsv)).
- **The rig at the end:** at 13:22:54–13:23:04 the kit's `reset-all` set every card's offsets to 0 and read them back,
  and the watchdog stopped after 0 trips. The kit's state then showed every offset 0, P8, no load, no GPU Xid, and
  1,911 / 4,004 MHz maximum on every card. Its start check now refuses a new offset until a reboot, since the ledger
  holds the nine step-downs to 0 of this boot: a card stepped down can stay slower until the rig restarts.

## The profile again on the boot of 14:04 (2026-11-19)

**In plain words.** Offsets set through the driver last only until the next reboot, and the kit raises no offset again
in a boot once one went down, so after run 5's reset the rig was restarted (up at 14:04:12) and the profile set once
more for the afternoon's runs. This is run 3 again without its stock check and its 30 minutes: the profile already
held its soak ([R202b](../2026-11-18_r202b_core_soak/README.md)) and its first check on this engine (run 3). It passed:
every card took the profile's offsets and gave its stock answer on its own, and Ornith with the offsets gave this
morning's stock answer in both rounds at the same speed as this morning. The offsets stay set and the watchdog runs.

[run/apply_again.sh](run/apply_again.sh) ran it on the rig, detached, in the kit's folder, with the same engine,
preset, profile and kit as run 3:

1. **The stock reference** is run 3's: its `greedy-stock1.sha` (`adf47784f9b02d4b`) and the responses behind it,
   copied into a fresh output folder, so each round is compared with the same stock answer (the kit's
   `model_check.sh` with the offsets needs that reference beside it).
2. **The start check** at 14:07:38 (every offset 0, P8, 0 MiB, no load process, no GPU Xid, no step-down this boot,
   3,190 MiB of host memory available) and **the watchdog** (14:07:45, the same limits as run 3: 70 °C, 80 W held for
   5 s, each card's own Xid, a sample every 0.5 s).
3. **`apply`** of the same [use-profile.json](../2026-11-18_r202b_core_soak/use-profile.json) (md5
   `b32956cd064d8b449bdd6ac2b8a6d4f7`), 14:07:49–14:13:28: check servers on every card with the kit's engine (v4.0's
   build) and check model, two stock rounds, every card's offsets set 14:12:24–14:12:31 and each read back as set,
   then one round with the offsets: every card PASS at 14:13:16.
4. **Ornith with the offsets**, `model_check.sh` with the v5 engine, Ornith's v5 preset (md5
   `bc73c0439a83f0556f88a2dfc7ea9292`) and `ROUNDS=2 MINUTES=0`: loaded in 543 s, two rounds 14:22:36–14:23:47.

**`apply`, card by card** ([r211_apply.md](r211_apply.md); the kit's own table
[r211_apply_steps.csv](r211_apply_steps.csv)), beside run 3's on the boot of 09:18 ([apply.md](apply.md)):

| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | run 3 with the offsets | peak core MHz at stock | with the offsets | run 3 with the offsets | greedy check | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | +1,300 | +256 | 179.09 | 207.02 | 206.96 | 1,873 | 2,126 | 2,126 | same as stock | PASS |
| 1 | +1,000 | +256 | 179.43 | 199.58 | 199.44 | 1,898 | 2,151 | 2,164 | same as stock | PASS |
| 2 | +1,000 | +269 | 179.00 | 199.60 | 199.52 | 1,898 | 2,164 | 2,151 | same as stock | PASS |
| 3 | +1,100 | +231 | 179.27 | 202.06 | 201.98 | 1,873 | 2,100 | 2,100 | same as stock | PASS |
| 4 | +1,100 | +206 | 179.29 | 202.21 | 201.96 | 1,911 | 2,113 | 2,113 | same as stock | PASS |
| 5 | +1,200 | +206 | 179.38 | 204.44 | 204.62 | 1,898 | 2,100 | 2,100 | same as stock | PASS |
| 6 | +900 | +282 | 179.04 | 197.25 | 197.34 | 1,885 | 2,164 | 2,164 | same as stock | PASS |
| 7 | +900 | +256 | 179.14 | 197.49 | 197.49 | 1,911 | 2,151 | 2,151 | same as stock | PASS |
| 8 | +1,300 | +231 | 179.32 | 207.00 | 207.41 | 1,911 | 2,126 | 2,126 | same as stock | PASS |

Every card's offsets are the profile's, read back as set; every card's check model gave its stock hash
(`60ee0f178cfd67d5` on all nine) and passed the 31 kernel checks with the offsets. The streaming reads with the
offsets are within 0.41 GB/s of run 3's on every card. The peaks with the offsets are run 3's on seven cards; card 1
peaked one driver step lower (2,151 against 2,164 MHz) and card 2 one step higher (2,164 against 2,151), with the same
offsets.

**Ornith's whole-memory check with the offsets** ([r211_model_check.md](r211_model_check.md), both rounds in
[r211_model_rounds.tsv](r211_model_rounds.tsv)):

| | run 3, stock (boot of 09:18) | run 3, with the offsets (boot of 09:18) | **this boot, with the offsets** | against run 3's offsets | against stock |
|---|---|---|---|---|---|
| load, s | 658 | 663 | **543** | | |
| host memory available when loaded, MiB | 1,496 | 1,492 | **1,502** | | |
| rounds; the stock hash in | 2; round 1 is the reference | 58; 58 | **2; 2** | | |
| hash | `adf47784f9b02d4b` | `adf47784f9b02d4b` in every round | **`adf47784f9b02d4b`, `adf47784f9b02d4b`** | | |
| generation tok/s in round 1 (prompts 1, 2) | 15.27, 16.15 | 16.42, 17.44 | **16.54, 17.47** | +0.73 %, +0.17 % | +8.32 %, +8.17 % |
| generation tok/s in round 2 (prompts 1, 2) | 16.66, 16.66 | 18.02, 18.01 | **18.01, 18.01** | −0.06 %, +0.00 % | +8.10 %, +8.10 % |
| prompt processing tok/s in round 1 (prompts 1, 2) | 13.05, 27.45 | 12.05, 29.40 | **13.31, 29.58** | +10.46 %, +0.61 % | +1.99 %, +7.76 % |
| prompt processing tok/s in round 2 (prompts 1, 2) | 26.95, 29.43 | 29.09, 31.84 | **29.10, 32.01** | +0.03 %, +0.53 % | +7.98 %, +8.77 % |

- **Round 2 is the steady one** (round 1 is each load's first request): 18.01 and 18.01 tok/s, run 3's round 2 within
  0.01 tok/s on prompt 1 and equal on prompt 2. Run 3's 57 rounds after its warm-up read 17.99 tok/s at the lowest
  and 18.02 at the highest on prompt 1, so this round sits among them.
- **The load used the same memory as run 3's:** 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 MiB on cards
  0 to 8.

**Heat, power and clocks** ([r211_watchdog.md](r211_watchdog.md), the watchdog's 1,954 samples a card from its start at
14:07:45 to 14:24:15, `apply`'s check servers and the load included; [r211_model_cards.md](r211_model_cards.md), its 144
samples a card over the two rounds):

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| highest °C, whole run | 41 | 37 | 35 | 32 | 38 | 34 | 42 | 37 | 34 |
| highest W, whole run | 78.35 | 72.96 | 75.44 | 72.75 | 73.62 | 73.61 | 76.72 | 76.18 | 75.74 |
| SM clock median over the two rounds, MHz | 2,126 | 2,151 | 2,164 | 2,100 | 2,113 | 2,100 | 2,164 | 2,164 | 2,126 |
| memory clock median over the two rounds, MHz | 4,657 | 4,498 | 4,498 | 4,551 | 4,551 | 4,608 | 4,455 | 4,455 | 4,657 |

The hottest card was card 6 at 42 °C. No sample went above 80 W; the highest was 78.35 W on card 0. No sample tripped,
and the boot logged no GPU Xid.

**The rig at the end** (14:24:02, the kit's `state`): every card at its profile value, P8, 0 MiB, no load process, the
watchdog running, 0 GPU Xid, 0 step-downs, 3,156 MiB of host memory available; no model server left running. The
afternoon's runs start from this state.

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv) | per arm (`ka`'s reference `v3ka`, `o4oc0`, `o4g2a`, `o4g2b`, `o5oc0`, `o5oc1`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes for all tasks, a task's median, minutes a task at equal work against `v3ka` |
| [decode.md](decode.md) | `o4oc0`, `o4g2a`, `o4g2b`, `o5oc0` and `o5oc1` by the context at a request's middle (requests, tokens, decode s, tok/s, ms a token), and at their own shared mix of contexts |
| [decode_r204_mix.md](decode_r204_mix.md) | `o4oc0`, `o5oc0` and `o5oc1` by context, and every run at R204's mix (R204's four runs' tokens) over the buckets all of them decoded in |
| [same_prompts.md](same_prompts.md), [same_prompts_tol1.md](same_prompts_tol1.md) | the requests read on the same prompt in `o4oc0`, `o4g2a`, `o4g2b`, `o5oc0` and `o5oc1` (task, turn, new tokens, cached tokens, each run's prompt ms), exactly and with the cached tokens within one; each run's generation over all its requests |
| [work.md](work.md), [work.tsv](work.tsv) | the work of each run: prompt tokens read, tokens generated, reply tokens re-read and the turns that re-read a reply |
| [clocks.tsv](clocks.tsv) | each card's busy SM clock median, their mean and each card's share of busy samples below 1,800 MHz, over each run's tasks |
| [host.tsv](host.tsv) | lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the log's end, each card's peak memory from the card log |
| [reallocs.tsv](reallocs.tsv), [o5oc0_requests_reallocs.tsv](o5oc0_requests_reallocs.tsv), [o5oc1_requests_reallocs.tsv](o5oc1_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size (the five runs), and per request (`o5oc0`, `o5oc1`) |
| [ornith_cards.md](ornith_cards.md) | per card over `o5oc0`'s and `o5oc1`'s tasks: highest temperature and power, samples above 80 W, the busy memory clock's median |
| [o5oc0_compare.md](o5oc0_compare.md), [o5oc0_report_table.md](o5oc0_report_table.md), [o5oc0_gonogo.md](o5oc0_gonogo.md) | the agent check's report on `o5oc0`: per task and first token per turn; per arm over the client's attempts; the gates and the server log's counts |
| [o5oc1_compare.md](o5oc1_compare.md), [o5oc1_report_table.md](o5oc1_report_table.md), [o5oc1_gonogo.md](o5oc1_gonogo.md) | the same report after `o5oc1`, over both arms (the client's results folder held both) |
| [agent/o5oc0/](agent/o5oc0/) | `turns.csv` (one row a request) and `attempts.csv` (one row a task) |
| [agent/o5oc1/](agent/o5oc1/) | the same of `o5oc1` |
| [run/ornith_run.sh](run/ornith_run.sh) | one Ornith run: the rig's and the kit's read-only state, then R204's harness call, the arm and the machine-specific values from the environment |
| [run/tables.sh](run/tables.sh) | Ornith's tables, from a folder holding the rows, server logs, card logs and host logs it lists |
| [run/decode_at_mix.py](run/decode_at_mix.py) | decode speed at a fixed mix of contexts taken from earlier runs, so a new run sits beside an earlier table without moving it |
| [flash_agent.tsv](flash_agent.tsv) | Flash-Next per arm (R90's `r90v3`, the equal-work reference, the context runs `fnoc0` and `v4rc2`, `f5oc0` and `f5oc1`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, the draft's acceptance, tokens a decode step, minutes for all tasks, a task's median, minutes a task at equal work against `r90v3` |
| [flash_steps.md](flash_steps.md) | `fnoc0`, `v4rc2`, `f5oc0` and `f5oc1` by the context at a request's middle: requests, tokens, decode s, tok/s, the draft's acceptance, tokens a decode step and ms a decode step |
| [flash_same_prompts.md](flash_same_prompts.md) | the 12 requests read on the same prompt in `fnoc0`, `v4rc2`, `f5oc0` and `f5oc1` (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [flash_same_prompts_3runs.md](flash_same_prompts_3runs.md) | the same for the 13 requests read on the same prompt in `fnoc0`, `v4rc2` and `f5oc0`, the table of run 2 |
| [flash_work.md](flash_work.md), [flash_work.tsv](flash_work.tsv) | the work of `fnoc0`, `v4rc2`, `f5oc0` and `f5oc1`: prompt tokens read, tokens generated, reply tokens re-read and the turns that re-read a reply |
| [flash_clocks.tsv](flash_clocks.tsv) | each card's busy SM clock median, their mean and each card's share of busy samples below 1,800 MHz, over each Flash-Next run's tasks |
| [flash_host.tsv](flash_host.tsv) | as `host.tsv`, for `fnoc0`, `v4rc2`, `f5oc0` and `f5oc1` (card peaks from the card log) |
| [flash_reallocs.tsv](flash_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, the four Flash-Next runs |
| [flash_cards.md](flash_cards.md) | per card over `f5oc0`'s and `f5oc1`'s tasks: highest temperature and power, samples above 80 W, the busy memory clock's median |
| [f5oc0_compare.md](f5oc0_compare.md), [f5oc0_report_table.md](f5oc0_report_table.md), [f5oc0_gonogo.md](f5oc0_gonogo.md) | the agent check's report on `f5oc0`, under the harness's arm name `v4r` |
| [f5oc1_compare.md](f5oc1_compare.md), [f5oc1_report_table.md](f5oc1_report_table.md), [f5oc1_gonogo.md](f5oc1_gonogo.md) | the same on `f5oc1` |
| [agent/f5oc0/](agent/f5oc0/) | `turns.csv` and `attempts.csv` of `f5oc0` (the harness's arm `v4r` renamed) |
| [agent/f5oc1/](agent/f5oc1/) | the same of `f5oc1` |
| [run/flash_tables.sh](run/flash_tables.sh) | Flash-Next's tables, from a folder holding the server, card and host logs it lists |
| [model_check.md](model_check.md), [model_rounds.tsv](model_rounds.tsv) | run 3's whole-memory check: stock against the offsets (loads, hashes, generation and prompt tok/s in rounds 1 and 2 and the median of rounds 2 on), and every round of both checks (hash, compared with stock's round 1, tok/s of both prompts) |
| [apply.md](apply.md), [o5_apply_steps.csv](o5_apply_steps.csv) | run 3's `apply`, card by card (the offsets, the streaming read and peak core clock at stock and with the offsets, the greedy check), and the kit's own table (both stock rounds' times and limits, the round with the offsets) |
| [model_cards.md](model_cards.md) | each card's highest temperature and power, samples above 80 W, SM and memory clock medians and the offsets the watchdog read, over the 30 minutes with the offsets |
| [run/apply_check.sh](run/apply_check.sh) | run 3 on the rig: the start check, the whole-memory check at stock, the watchdog, `apply`, the whole-memory check with the offsets; on any failure every offset back to 0 |
| [run/model_rounds.py](run/model_rounds.py) | `model_rounds.tsv`, `model_check.md` and `model_cards.md` from `model_check.sh`'s log and the watchdog's samples; `again` and `whole` for the boot of 14:04 |
| [r211_apply.md](r211_apply.md), [r211_apply_steps.csv](r211_apply_steps.csv) | `apply` on the boot of 14:04, card by card, and the kit's own table |
| [r211_model_check.md](r211_model_check.md), [r211_model_rounds.tsv](r211_model_rounds.tsv) | Ornith's whole-memory check with the offsets on the boot of 14:04 beside run 3's stock and offset rounds, and its two rounds |
| [r211_model_cards.md](r211_model_cards.md), [r211_watchdog.md](r211_watchdog.md) | each card's heat, power and clocks over the two rounds, and its highest temperature and power over every watchdog sample of the boot of 14:04's run |
| [run/apply_again.sh](run/apply_again.sh) | the profile again on a later boot: the start check, the watchdog, `apply`, the whole-memory check with the offsets against an earlier check's stock reference; on any failure every offset back to 0 |

## Commands

From the repository root on the machine that drives the rig (here also the client), with the rig freshly booted, no
llama process, 0 MiB on every card, and the offset kit's read-only start check passing
(`python3 climb.py state && python3 climb.py start-check` in the kit's folder: every offset 0, no step-down this
boot). The bench files must be on the client in `CLIENT_DIR` (the check's `prep` step copies them).

```bash
# Ornith at stock (~10 min load, 12 tasks), then its report; another check may share the local client, so other ports
RIG=$RIG CLIENT=local ARM=o5oc0 MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN=<the v5 tree>/bin/llama-server KIT=<the offset kit's folder> RIG_LOGDIR=<rig folder>/r209a/agent-logs \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r209 RELAY_PORT=8296 PROXY_PORT=18793 \
  OUT=<out> bash docs/data/2026-11-19_r209_oc_v5/run/ornith_run.sh
RIG=$RIG CLIENT=local ARM=o5oc0 MODEL=<the same> BIN=<the same> KIT=<the same> RIG_LOGDIR=<the same> \
  CLIENT_DIR=<the same> CLIENT_OUT=<the same> RELAY_PORT=8296 PROXY_PORT=18793 OUT=<out> STEPS="report collect" \
  bash docs/data/2026-11-19_r209_oc_v5/run/ornith_run.sh

# the tables, on a folder holding the rows and logs run/tables.sh lists; o5oc0's window (rig_o5oc0/window):
# 2026-11-19 09:51:09|2026-11-19 10:11:21 (the driver's log; its clock and the rig's agreed)
bash docs/data/2026-11-19_r209_oc_v5/run/tables.sh <tables folder> o5oc0
```

`agent/o5oc0/` is the client's `res/turns.csv` rows of `o5oc0` and its `res/attempts.csv` rows whose file is `o5oc0`.
`res/` in the tables folder holds those rows plus the rows of `v3ka` and `o4oc0`
([R203](../2026-11-18_r203_ornith_offsets/README.md)) and of `o4g2a` and `o4g2b`
([R204](../2026-11-18_r204_ornith_glu_agent/README.md)); their server, host and card logs are the ones those folders'
tables used.

Ornith with the offsets ran the same two commands with `ARM=o5oc1`, its own rig folder (`r209e`) and the same
client folder, after run 3's `apply` and run 4, with the kit's state read through a kit file that sets run 3's work
folder (`WORK`, where the watchdog runs); then its tables:

```bash
# agent/o5oc1/ as agent/o5oc0/ (the client's res/turns.csv rows of o5oc1, its res/attempts.csv rows whose file is
# o5oc1); in the tables folder of o5oc0, add those rows to res/, o5oc1's server log as rig/server-o5oc1.log, and in
# rig_o5oc1/ its own lines of the client's rig/cards.csv and rig/meminfo.log (both arms append to them; o5oc1's begin
# at its server's start, 12:46:33) and its window: 2026-11-19 12:57:36|2026-11-19 13:21:55
bash docs/data/2026-11-19_r209_oc_v5/run/tables.sh <tables folder> o5oc0 o5oc1
python3 -I docs/data/2026-11-18_r203_ornith_offsets/run/flash_cards.py cards \
  "o5oc0=<tables>/rig_o5oc0/cards.csv=2026-11-19 09:51:09=2026-11-19 10:11:21" \
  "o5oc1=<tables>/rig_o5oc1/cards.csv=2026-11-19 12:57:36=2026-11-19 13:21:55" > docs/data/2026-11-19_r209_oc_v5/ornith_cards.md
```

`o5oc1_compare.md`, `o5oc1_report_table.md` and `o5oc1_gonogo.md` are the client's `compare.md`, `table.md` and
`gonogo.md` after `o5oc1`'s report. After the run, on the rig as root, in the kit's folder with run 3's work folder
set: `python3 climb.py reset-all && python3 climb.py watchdog stop && python3 climb.py state` (every offset 0 read
back; the start check then asks for a reboot before the next offset).

Flash-Next at stock, R203c's run ([run/flash_run.sh](../2026-11-18_r203_ornith_offsets/run/flash_run.sh)) with the v5
engine's build folder as `NEW` (the run itself was started from a launcher that did the same three steps with
this machine's values written in):

```bash
# Flash-Next at stock (~4 min load, 12 tasks; the report is part of the run); its own client folder, since the
# driver fixes the arm at v4r and skips attempts it finds finished in its client folder
RIG=$RIG CLIENT=local KIT=<the kit's folder on the rig> \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf NEW=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483 \
  REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern RIGDIR=<rig folder>/r209b \
  RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r209b OUT=<out>/R209b \
  bash docs/data/2026-11-18_r203_ornith_offsets/run/flash_run.sh

# agent/f5oc0/ from the client's res/turns.csv and res/attempts.csv, the arm v4r renamed f5oc0 (the arm and file
# columns, the tag's first part) and the line ends made LF; then the tables, on a folder holding R90's v3 server log,
# each run's server log as rig/server-<run>.log and its card and host logs in rig_<run>/ (fnoc0's and v4rc2's are the
# ones R203's and C854's tables used; v4rc2's window 2026-11-18 15:00:11|2026-11-18 15:32:58, fnoc0's
# 2026-11-18 16:04:59|2026-11-18 16:23:34); f5oc0's window: 2026-11-19 10:25:37|2026-11-19 10:44:14 (the first
# task's start and the last task's end in the driver's log)
sed 's/v4r/f5oc0/g; s/\r$//' <client folder>/r209b/res/turns.csv > docs/data/2026-11-19_r209_oc_v5/agent/f5oc0/turns.csv
sed 's/v4r/f5oc0/g; s/\r$//' <client folder>/r209b/res/attempts.csv > docs/data/2026-11-19_r209_oc_v5/agent/f5oc0/attempts.csv
bash docs/data/2026-11-19_r209_oc_v5/run/flash_tables.sh <Flash-Next tables folder> f5oc0
```

Flash-Next with the offsets ran the same command with its own rig, client and output folders (`r209d`), after run 3's
`apply`, with the kit's state read through a kit file that sets run 3's work folder (`WORK`, where the watchdog runs);
then its tables:

```bash
# agent/f5oc1/ as agent/f5oc0/ (the arm v4r renamed f5oc1); f5oc1's server log as rig/server-f5oc1.log and its card
# and host logs in rig_f5oc1/ of f5oc0's tables folder; f5oc1's window: 2026-11-19 12:14:08|2026-11-19 12:30:57
sed 's/v4r/f5oc1/g; s/\r$//' <client folder>/r209d/res/turns.csv > docs/data/2026-11-19_r209_oc_v5/agent/f5oc1/turns.csv
sed 's/v4r/f5oc1/g; s/\r$//' <client folder>/r209d/res/attempts.csv > docs/data/2026-11-19_r209_oc_v5/agent/f5oc1/attempts.csv
bash docs/data/2026-11-19_r209_oc_v5/run/flash_tables.sh <Flash-Next tables folder> f5oc0 f5oc1
python3 -I docs/data/2026-11-18_r203_ornith_offsets/run/flash_cards.py cards \
  "f5oc0=<tables>/rig_f5oc0/cards.csv=2026-11-19 10:25:37=2026-11-19 10:44:14" \
  "f5oc1=<tables>/rig_f5oc1/cards.csv=2026-11-19 12:14:08=2026-11-19 12:30:57" > docs/data/2026-11-19_r209_oc_v5/flash_cards.md
```

`flash_same_prompts_3runs.md` is `run/flash_tables.sh <tables> f5oc0`'s `flash_same_prompts.md`, run 2's table.

Run 3, on the rig as root, with the kit's start check passing (after runs 1 and 2 on this boot every offset was still
0 and no step-down was in the ledger); about 60 minutes (two ~11-minute loads, `apply` 5 minutes, 30 minutes with the
offsets):

```bash
# in the kit's folder: Ornith's v5 preset and R202b's use profile beside the kit
cp release/ornith-v5/settings.env $KIT/ornith-v5.env           # md5 bc73c0439a83f0556f88a2dfc7ea9292
cp docs/data/2026-11-18_r202b_core_soak/use-profile.json $KIT/r202b-use.json   # md5 b32956cd064d8b449bdd6ac2b8a6d4f7
cp docs/data/2026-11-19_r209_oc_v5/run/apply_check.sh $KIT/
KIT=$KIT BIN=<the v5 tree>/bin MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  SETTINGS=$KIT/ornith-v5.env PROFILE=$KIT/r202b-use.json WORK=<work folder> OUT=<model check folder> \
  setsid -f nohup bash $KIT/apply_check.sh > <console> 2>&1 < /dev/null
cat <work folder>/R209C_DONE      # "done 0": passed, the offsets set and the watchdog running

# the tables, from the two folders copied off the rig
python3 -I docs/data/2026-11-19_r209_oc_v5/run/model_rounds.py rounds <model check folder>/model.log > model_rounds.tsv
python3 -I docs/data/2026-11-19_r209_oc_v5/run/model_rounds.py summary <model check folder>/model.log > model_check.md
python3 -I docs/data/2026-11-19_r209_oc_v5/run/model_rounds.py cards <model check folder>/model.log <work folder>/wd.csv > model_cards.md
cp <work folder>/out/apply_steps.csv o5_apply_steps.csv
python3 -I docs/data/2026-11-18_r203_ornith_offsets/run/flash_cards.py apply o5_apply_steps.csv > apply.md
```

The kit's `env.sh` (R200's, [run/env.sh](../2026-11-17_r200_climb_rig2/run/env.sh) with v4.0's build as its engine)
gives `apply` its check model and memory steps; `apply_check.sh` sets the work folder after it and passes the v5
tree only to `model_check.sh`. The run's script was this one with the machine's paths written in.

The profile again on the boot of 14:04, on the rig as root, with the kit's start check passing (a fresh boot) and run
3's model check folder still on the rig; about 17 minutes (`apply` 6 minutes, a 9-minute load, two rounds):

```bash
cp docs/data/2026-11-19_r209_oc_v5/run/apply_again.sh $KIT/
KIT=$KIT BIN=<the v5 tree>/bin MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  SETTINGS=$KIT/ornith-v5.env PROFILE=$KIT/r202b-use.json REF=<run 3's model check folder> REF_HASH=adf47784f9b02d4b \
  WORK=<fresh work folder> OUT=<fresh model check folder> ROUNDS=2 \
  setsid -f nohup bash $KIT/apply_again.sh > <console> 2>&1 < /dev/null
cat <work folder>/APPLY_AGAIN_DONE   # "done 0": passed, the offsets set and the watchdog running

# the tables, from the folders copied off the rig
cp <work folder>/out/apply_steps.csv r211_apply_steps.csv
python3 -I docs/data/2026-11-18_r203_ornith_offsets/run/flash_cards.py apply r211_apply_steps.csv > r211_apply.md
python3 -I docs/data/2026-11-19_r209_oc_v5/run/model_rounds.py again <run 3's model check folder>/model.log <model check folder>/model.log > r211_model_check.md
python3 -I docs/data/2026-11-19_r209_oc_v5/run/model_rounds.py rounds <model check folder>/model.log > r211_model_rounds.tsv
python3 -I docs/data/2026-11-19_r209_oc_v5/run/model_rounds.py cards <model check folder>/model.log <work folder>/wd.csv > r211_model_cards.md
python3 -I docs/data/2026-11-19_r209_oc_v5/run/model_rounds.py whole <work folder>/wd.csv > r211_watchdog.md
```

The run's script was `apply_again.sh` as committed, called with the machine's paths; the watchdog's `wd.csv` was
copied at 14:24:15, while the watchdog kept running.

## Provenance and licence

Measured on the project's own rig on 2026-11-19 with the repository's own scripts; the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/o5oc0/attempts.csv`, `agent/o5oc1/attempts.csv`, `agent/f5oc0/attempts.csv` and `agent/f5oc1/attempts.csv`). The
tables and scripts here are under the repository's licence.
