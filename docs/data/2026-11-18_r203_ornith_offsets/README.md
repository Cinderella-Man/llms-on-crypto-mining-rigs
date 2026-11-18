# R203: the agent check at stock and with the second rig's clock offsets, Ornith and Flash-Next on one boot (Q1592, 2026-11-18)

**In plain words.** This measures the speed that counts, a coding agent's, with the memory and core offsets that held
on the second rig ([R201](../2026-11-18_r201_soak_ladder/README.md#the-second-boot-the-use-profile-held-and-ornith-generated-442--faster),
Q1587). On one fresh boot it runs each model's agent check twice: first with every card at stock, then, after the use
profile is applied, the same run again. The order on the boot was Ornith at stock, Flash-Next at stock, the kit's
`apply`, Flash-Next with the offsets, Ornith with the offsets, and then every offset back to 0.

**What the second rig's overclock is worth to a coding agent: a gain of about 4 to 5 % on Ornith, and a negligible
one on Flash-Next.**

- **Ornith with the offsets** passed 12 of 12 tasks at 2.05 minutes a task at equal work and generated 17.31 tok/s.
  At stock on the same boot it took 2.13 minutes and generated 16.42 tok/s: −3.76 % and +5.42 %, both larger than
  the gap between two stock runs of the same preset (0.95 % and 2.42 %), so a **gain**. At the same context a decode
  step took 4.83 % and 4.22 % less time, close to the +4.42 % of R201's single-request check.
- **Flash-Next with the offsets** passed 11 of 12 at 2.39 minutes a task at equal work and 27.13 tok/s, against 2.36
  minutes and 28.08 tok/s at stock: +1.27 % and −3.38 %, inside the gap between two stock runs (2.53 % and 3.52 %),
  so **negligible**. Its decode steps at the same context took 2.65 % and 1.92 % less time. One long reply on a task
  that ran out of time pulled the session's figures down.
- **The answers stayed sound** in both runs with the offsets: no malformed tool call, no NaN or inf, and every passed
  task also passed its hidden tests. The watchdog never tripped, no card logged a GPU Xid, and the hottest card
  reached 50 °C. After the last run every offset read 0 again.
- **The full profile, a first look** ([section](#the-full-profile-first-look-o4oc2-2026-11-18)): with a core offset
  on every card ([R202b](../2026-11-18_r202b_core_soak/README.md)), one Ornith run on a later boot passed 12 of 12 at
  1.96 minutes a task at equal work and 17.89 tok/s: −4.39 % and +3.35 % against the three-card profile, −7.98 % and
  +8.95 % against stock. At the same context a decode step took 4.29 % and 4.73 % less time than with the three-card
  profile, and the same 19 prompts 5.08 % less. The references ran on another boot, so the class waits for a
  same-boot test (Q1603). Answers sound, hottest card 46 °C, no trip, no Xid; every offset 0 after it.

At stock both models ran where they ran before on the same rig. Ornith: 12 of 12, 2.13 minutes, 16.42 tok/s (R166 two
days earlier: 2.11 and 2.13 minutes, 16.52 and 16.13 tok/s). Flash-Next: 12 of 12, 2.36 minutes, 28.08 tok/s (R156a
five days earlier: 2.43 minutes, 26.67 tok/s, on longer tasks).

Registry row Q1592 in [quants/experiments.md](../../../quants/experiments.md). The single-request diagnostic on
R201's second boot was +4.42 % generation with the profile (16.29 against 15.60 tok/s).

## The method

The agent check of [R166](../2026-11-16_r166_ornith_stack_no_taper/README.md)
([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks, Python and JavaScript alternating, a
12-minute cap a task, a 50-minute budget, the makers' sampling at temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF
(valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,592 bytes) at 49,152 tokens on the second rig's
nine P106-090 cards, one arm a half, one load an arm:

- **`o4oc0`, the stock half:** R166a's `o4sna`, the released v4.0 preset
  ([release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)) on v4.0's build (`pascal-p106-v4`
  `eb82916d3`; the rig tree's `llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`, checked before the run). The
  started command is R166a's byte for byte (521 bytes, md5 `211f8db2edeacd267705795e4911d8c9`), and the server's
  `switches:` line is R166's: `6 set: GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_KEEPALIVE=decode
  GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SERVER_YIELD=2`.
- **`o4oc1`, the offset half:** the same command after the offset kit's `apply` of R201's use profile
  ([use-profile.json](../2026-11-18_r201_soak_ladder/use-profile.json): memory +1,300, +1,000, +1,000, +1,100, +1,100,
  +1,200, +900, +900, +1,300 MT/s on cards 0 to 8, core +256, +206 and +231 MHz on cards 0, 4 and 8).

Both halves run on the same boot, so the rig, the day and the file's place on disk are the same and only the offsets
differ. The comparison of this row is `o4oc1` against `o4oc0`. R166a `o4sna` and R166b `o4snb` (the same preset,
build and rig, 2026-11-16) are beside it as context.

## The stock half: `o4oc0` (2026-11-18)

The rig was rebooted at 14:57:26. Before the run the kit's read-only start check passed: every card at 0 offset, 75 W,
1,911 / 4,004 MHz maximum, P8, no load, no GPU Xid, no step-down this boot, 3,192 MiB of host memory available. The
server started at 15:08:35 and was up after 642 s (R166's loads: 539 and 538 s); the tasks ran 15:19:19–15:51:25.
After the run the same check passed again, every offset still 0.

| | R166a `o4sna` (2026-11-16) | R166b `o4snb` (2026-11-16) | **R203a `o4oc0`, stock** |
|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.11 | 2.13 | **2.13** |
| generation tok/s | 16.52 | 16.13 | **16.42** |
| prompt tok/s, the whole session | 103.0 | 105.9 | **102.2** |
| first token median s | 6.69 | 6.06 | **7.07** |
| first token p90 s | 11.29 | 11.32 | **11.51** |
| first token on a task's first turn, median s | 8.43 | 8.45 | **8.48** |
| the 19 requests read on the same prompt in all three runs, prompt s | 190.8 | 190.3 | **192.5** |
| of them, the 12 first turns, prompt s | 115.8 | 115.3 | **116.9** |
| prompt tokens read a task | 3,405 | 4,130 | **3,725** |
| tokens generated a task | 1,818 | 2,300 | **1,980** |
| reply tokens re-read (turns; their prompt s of all prompt s) | 4,831 (2; 30.6 of 396.6) | 12,332 (3; 71.9 of 467.8) | **8,860 (5; 65.6 of 437.3)** |
| minutes, all 12 tasks; turns | 29.1; 63 | 36.8; 73 | **31.9; 67** |
| minutes a task, median | 2.15 | 2.34 | **2.52** |
| tool calls / failed / malformed | 63 / 2 / 0 | 73 / 2 / 0 | **67 / 2 / 0** |
| checkpoints restored / created / erased; full re-processing | 13 / 0 / 0; 0 | 14 / 0 / 0; 0 | **16 / 0 / 0; 0** |
| graph reallocations | 0 in 63 requests | 0 in 73 | **0 in 67** |
| busy SM clock, mean of the nine cards' medians, MHz | 1,888.2 | 1,888.2 | **1,883.9** |
| lowest MemAvailable MiB | 1,225 | 1,228 | **1,207** |

Each value is one run ([agent.tsv](agent.tsv), [same_prompts.md](same_prompts.md), [work.md](work.md),
[clocks.tsv](clocks.tsv), [host.tsv](host.tsv), [reallocs.tsv](reallocs.tsv), [o4oc0_gonogo.md](o4oc0_gonogo.md);
R166's rows from its folder). Minutes a task at equal work price `ka`'s work (v3.0's 2026-11-06 run on the same rig:
2.52) at each run's own prompt and generation speeds, so they compare the engine and the cards on the same work. The
minutes for all tasks and a task's median follow what the agent wrote at temperature 0.6: this run read a reply again
on five turns ([work.md](work.md)).

- **The cards' busy clocks** ([clocks.tsv](clocks.tsv), from the card log `scripts/card_log.py` writes every 0.5 s,
  busy = utilization above 0 %, 15:19:19–15:51:25): medians of 1,860 1,885 1,885 1,860 1,898 1,898 1,873 1,898 1,898
  MHz on cards 0 to 8, against 1,860 1,898 1,885 1,873 1,911 1,898 1,873 1,898 1,898 in both R166 runs; busy samples
  below 1,800 MHz 3.73 2.52 2.80 1.96 3.32 3.30 3.75 2.96 2.36 %. These are the stock numbers the offset half's clocks
  compare with.
- **The same prompts, two days apart:** the 19 requests that read the same number of new tokens on the same cached
  prefix in all three runs (every task's first turn and seven second turns) took 192.5 s, against 190.8 and 190.3 s
  in R166 (+1.1 % on their mean; R166's two runs are 0.3 % apart). 17 of the 19 are slower than in both R166 runs, by
  0.25 to 4.72 %. Same rig, file, preset and build, so this is how far one rig moves between days; the offset half on
  the same boot does not carry it.
- **Generation** 16.42 tok/s over all 67 requests (23,764 tokens in 1,446.9 s of decoding).
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,207 MiB at 15:50:39, during the last task; swap
  0 MiB at the start and 334 MiB at its highest at 15:18:49, during the load (R166: 338 and 337). The RAM watchdog
  stopped nothing. Card peaks from the card log, 5,808 5,828 5,754 5,828 5,438 5,546 5,706 5,782 5,188 MiB, are
  R166a's 5,733 5,753 5,679 5,753 5,363 5,471 5,631 5,707 5,113 from `nvidia-smi` plus the 75 MiB `card_log.py`
  counts on each idle card, card for card. The report's memory gate is NO-GO on the swap, as for R166; every other
  gate is GO ([o4oc0_gonogo.md](o4oc0_gonogo.md)).
- **The client:** the machine that drives the rig was also the agent's client. Another agent check, on the first rig,
  used the same client until 15:32:58, the first 13.7 minutes of these tasks, with its own ports, arm names and
  results folder. It can show only in the minutes outside the server (tools, tests); every server figure above is
  the server's own timing, and the equal-work minutes take the time outside the server from `ka`'s run.

## Flash-Next at stock: `fnoc0` (2026-11-18)

**12 of 12 tasks passed, 2.36 minutes a task at equal work, generation 28.08 tok/s, no reply read again.** This is the
stock number for Flash-Next's offset half `fnoc1` on the same boot.

The run is R156's agent check ([R156](../2026-11-13_r156_agent_v4r/README.md),
[bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks, Python and JavaScript alternating, a
12-minute cap a task, a 40-minute budget) on the maker's Swift GSQ-RCO Q2_0
(`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, the n-gram table in the third shard) with the MTP draft
(`mtp-Qwen3.8-Flash-Next-draft.gguf`) at 65,536 tokens, one arm, one load: the harness's arm `v4r`, **the released
v4.0 preset ([release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env)) on v4.0's build**
(`llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`, checked on the rig before the run). The tables call it
`fnoc0`. Its started command is R156a's byte for byte (1,689 bytes, md5 `6b5dac04b78dfe51cfc94cf55bda79fc`), and the
server's `switches:` line names R156's 39 switches with the same values.

It ran right after Ornith's stock half on the same boot. At 16:01:02 the kit's read-only state showed every card at 0
offset, 0 MiB, 75 W, 1,911 / 4,004 MHz maximum, no load, no GPU Xid, no step-down this boot, 3,173 MiB of host memory
available. The server started at 16:01:11 and was up after 227 s (R156a on this rig: 225 s); the tasks ran
16:04:59–16:23:34. At 16:23:54 the kit's start check passed again, every offset still 0. The machine that drives the
rig was the agent's client. An Ornith agent check on the first rig used the same client through all of these tasks
(from 16:06:27), with its own ports and folders; that shows only in the time outside the server.

R156a `v4r` is beside it as context: the same preset, build and rig on 2026-11-13, with the 7-GPU quant machine's CPU as
the agent's client. R203's comparison is `fnoc1` against `fnoc0`.

| | R156a `v4r` (2026-11-13) | **R203c `fnoc0`, stock** |
|---|---|---|
| passed | 12 / 12 | **12 / 12** |
| minutes a task at equal work (R90 `v3`'s work) | 2.43 | **2.36** |
| generation tok/s | 26.67 | **28.08** |
| prompt tok/s, the whole session | 133.9 | **141.2** |
| first token median s | 2.42 | **2.63** |
| first token p90 s | 9.00 | **8.34** |
| first token on a task's first turn, median s | 6.68 | **6.66** |
| the 11 first turns read on the same prompt in both runs, prompt s | 81.84 | **84.27** |
| prompt tokens read a task | 4,400 | **3,082** |
| tokens generated a task | 3,486 | **1,905** |
| reply tokens re-read (turns; their prompt s of all prompt s) | 4,863 (2; 31.0 of 394.2) | **0 (0; 0.0 of 261.9)** |
| draft acceptance (accepted of drafted tokens) | 0.749 | **0.765** |
| minutes, all 12 tasks; turns | 33.4; 95 | **18.4; 60** |
| minutes a task, median | 1.20 | **1.28** |
| tool calls / failed / malformed | 90 / 3 / 0 | **54 / 0 / 0** |
| checkpoints restored / created / erased; full re-processing | 12 / 0 / 1; 1 | **11 / 0 / 0; 0** |
| graph reallocations | 0 in 95 requests | **0 in 60** |
| busy SM clock, mean of the nine cards' medians, MHz | 1,886.8 | **1,885.3** |
| lowest MemAvailable MiB; swap highest MiB | 312; 21 | **414; 426** |

Each value is one run ([flash_agent.tsv](flash_agent.tsv), [flash_same_prompts.md](flash_same_prompts.md),
[flash_work.md](flash_work.md), [flash_clocks.tsv](flash_clocks.tsv), [flash_host.tsv](flash_host.tsv),
[flash_reallocs.tsv](flash_reallocs.tsv), [fnoc0_gonogo.md](fnoc0_gonogo.md); R156a's session figures from
[its folder](../2026-11-13_r156_agent_v4r/agent.tsv), its per-request table rebuilt from its server log in
[C854's part (6)](../2026-11-18_c854b_p106_check/README.md#part-6-the-agent-check-on-flash-next-2026-11-18)).
Minutes a task at equal work price R90 `v3`'s work (v3.0 on the first rig, 2026-11-11: 3.08) at each run's own
prompt and generation speeds.

- **Shorter tasks than R156a's.** At the preset's sampling the agent took 60 turns against 95 and generated 1,905
  tokens a task against 3,486. JavaScript `complex-numbers` passed in 5 turns (1.1 minutes); R156a took 29 turns and
  9.7 minutes there. Every task ended in 4 to 6 turns, and the longest context was 9,423 tokens (R156a: 25,543). So the
  minutes for all tasks (18.4 against 33.4) follow the work; the equal-work column compares the speeds.
- **Generation** 28.08 tok/s over all 60 requests (22,860 tokens in 814.1 s of decoding). Part of the gap to R156a is
  the context mix: R156a generated 5,203 tokens at contexts over 16,000 tokens, at 23.70 tok/s, where this run never
  went. At 4,000 to 8,000 tokens of context the two runs generated at 27.58 and 27.38 tok/s; under 4,000, at 28.86
  and 27.05. The draft's acceptance was 0.765 against 0.749.
- **The same prompts, five days apart:** the 11 requests that read the same number of new tokens on the same cached
  prefix in both runs (every task's first turn except `pig-latin`'s) took 84.27 s against 81.84 s (+2.97 %). One of
  them, JavaScript `variable-length-quantity`'s first turn, took 4.86 s against 3.69 s; that request also took 5.39 s
  in R156b on the first rig, against 3.66 s in C854's run there. Without it the other 10 took 79.41 s against
  78.15 s (+1.61 %): 9 slower by 0.53 to 2.62 %, JavaScript `phone-number` 2.56 % faster. Ornith's same prompts on
  this boot were 1.1 % slower than two days earlier; the offset half on the same boot does not carry this.
  `pig-latin`'s first turn is not in the set because R156a read its whole prompt again (C854's part (6)).
- **First token follows the turn mix.** The turns after the first read a median of 215.5 new tokens (R156a: 215),
  with a median server prefill of 2.00 s (2.16 s). A task's first turn, the same request in both runs, took a median
  6.66 s (6.68 s). Those 12 first turns are 12 of this run's 60 requests and 12 of R156a's 95, so they lift the median
  of all turns more here.
- **No reply read again** ([flash_work.md](flash_work.md)): on every turn the cached prefix reached the previous
  reply's end. R156a re-read on 2 turns, and C854's run of the b12483 candidate on the first rig on 7.
- **The cards' busy clocks** ([flash_clocks.tsv](flash_clocks.tsv), from the card log `scripts/card_log.py` writes
  every 0.5 s, busy = utilization above 0 %, 16:04:59–16:23:34): medians of 1,860 1,885 1,885 1,873 1,898 1,898 1,873
  1,898 1,898 MHz on cards 0 to 8 (mean 1,885.3), against 1,860 1,898 1,885 1,873 1,898 1,898 1,873 1,898 1,898 in
  R156a (1,886.8); busy samples below 1,800 MHz 3.46 2.66 3.01 2.40 1.67 2.63 2.64 2.32 1.05 %. These are the stock
  numbers `fnoc1`'s clocks compare with.
- **Host and cards** ([flash_host.tsv](flash_host.tsv)): lowest MemAvailable 414 MiB at 16:06:47, as the second task
  started; swap 23 MiB at the start and 426 MiB at its highest at 16:12:58 (R156a: 21 and 21). The RAM watchdog stopped
  nothing. The card log's peaks, 5,174 5,220 5,076 5,236 4,334 5,092 4,492 5,104 5,694 MiB, less the 75 MiB
  `card_log.py` counts on each idle card, are 4 to 18 MiB under R156a's `nvidia-smi` peaks, card for card. The
  report's memory gate is NO-GO on the 414 MB low and the 427 MB of swap, as R156's and C854's part (6) were; every
  other gate is GO ([fnoc0_gonogo.md](fnoc0_gonogo.md)).
- **Graph reallocations:** 0 in 60 requests, in every prompt-size bucket ([flash_reallocs.tsv](flash_reallocs.tsv)).

## Flash-Next with the offsets: `fnoc1` (2026-11-18)

**11 of 12 tasks passed, 2.39 minutes a task at equal work, generation 27.13 tok/s. Against `fnoc0` on the same boot
(2.36 minutes, 28.08 tok/s) that is +1.27 % and −3.38 %, both inside the spread of two stock runs of the same preset
(R156a and R156b: 2.43 and 2.37 minutes, 2.53 % apart; 26.67 and 27.61 tok/s, 3.52 % apart): a negligible gain on
the agent's speed.** The cards were faster where the two runs did the same kind of work. A decode step took 2.65 %
less time at contexts under 4,000 tokens and 1.92 % less at 4,000 to 8,000, and the 11 identical first-turn prompts
took 0.99 % less time. One reply of 15,807 tokens, with its context growing from 4,307 to about 20,100 tokens and
fewer drafted tokens accepted, pulled the session's generation figure down. The task that failed ran out of its 12
minutes on that reply. The reply reads as sound reasoning, and stock runs lose tasks the same way (below).

**The kit's `apply`** (16:32–16:37, [flash_apply.md](flash_apply.md), [fnoc1_apply_steps.csv](fnoc1_apply_steps.csv)),
in the kit's folder on the rig, with a fresh work folder:

```bash
python3 climb.py start-check && python3 climb.py watchdog start   # every offset 0, no step-down this boot: PASS
python3 climb.py apply use-profile.json                            # detached; APPLY_DONE "exit 0" after 5 min
python3 climb.py state                                             # every card at the profile's values
```

The start check passed, and the watchdog started (70 °C, 80 W for 5 s, an Xid that names a card resets that card).
The nine check servers loaded, and two stock rounds gave every card the same greedy hash. Then each card's offsets
from [use-profile.json](../2026-11-18_r201_soak_ladder/use-profile.json) were set and read back as set, and one check
round passed on every card: the greedy hash the same as at stock, 31 of 31 kernel checks.

| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | change | peak core MHz at stock | with the offsets |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +1,300 | +256 | 179.32 | 206.80 | +15.32 % | 1,873 | 2,126 |
| 1 | +1,000 | 0 | 179.06 | 199.69 | +11.52 % | 1,898 | 1,898 |
| 2 | +1,000 | 0 | 179.18 | 199.67 | +11.44 % | 1,885 | 1,885 |
| 3 | +1,100 | 0 | 179.23 | 201.90 | +12.65 % | 1,873 | 1,873 |
| 4 | +1,100 | +206 | 179.26 | 202.08 | +12.73 % | 1,898 | 2,113 |
| 5 | +1,200 | 0 | 179.23 | 204.36 | +14.02 % | 1,898 | 1,898 |
| 6 | +900 | 0 | 179.23 | 197.47 | +10.18 % | 1,873 | 1,873 |
| 7 | +900 | 0 | 179.21 | 197.73 | +10.33 % | 1,898 | 1,898 |
| 8 | +1,300 | +231 | 179.17 | 207.21 | +15.65 % | 1,898 | 2,126 |

**The run:** `fnoc0`'s launcher with its own output, rig and client folders, started at 16:38:07, 25 s after `apply`
ended. The kit's state showed every card at its profile values, no load, no GPU Xid, no step-down this boot and
3,143 MiB of host memory available. The card log's start line read the same offsets on every card. The started
command and the server's `switches:` line are `fnoc0`'s byte for byte. The server was up after 226 s (`fnoc0`:
227 s), and the tasks ran 16:42:03–17:12:42. At 17:13:01 every card still read its profile values, with no GPU Xid
and no step-down. The kit's start check said FAIL, as it must with offsets set. The watchdog ran through the whole
run and never tripped. The launcher's `state` looks for a watchdog by its work folder; it was given another folder,
so it printed "watchdog running: no". Read with `apply`'s folder, the kit showed it running at 17:13:21. An Ornith
agent check on the first rig used the same client during these tasks, with its own ports and folders.

| | R203c `fnoc0`, stock | **R203d `fnoc1`, the use profile** | change |
|---|---|---|---|
| passed | 12 / 12 | **11 / 12** | |
| minutes a task at equal work (R90 `v3`'s work) | 2.36 | **2.39** | +1.27 % |
| generation tok/s | 28.08 | **27.13** | −3.38 % |
| prompt tok/s, the whole session | 141.2 | **144.3** | +2.20 % |
| first token median s | 2.63 | **2.98** | +13.31 % |
| first token p90 s | 8.34 | **8.26** | −0.96 % |
| first token on a task's first turn, median s | 6.66 | **6.50** | −2.40 % |
| the 11 first turns read on the same prompt, prompt s | 84.27 | **83.44** | −0.99 % |
| prompt tokens read a task | 3,082 | **3,262** | +5.84 % |
| tokens generated a task | 1,905 | **3,298** | |
| reply tokens re-read (turns; their prompt s of all prompt s) | 0 (0; 0.0 of 261.9) | **80 (1; 3.5 of 271.2)** | |
| draft acceptance (accepted of drafted tokens) | 0.765 | **0.737** | |
| minutes, all 12 tasks; turns | 18.4; 60 | **30.4; 63** | |
| minutes a task, median | 1.28 | **1.02** | |
| tool calls / failed / malformed | 54 / 0 / 0 | **61 / 2 / 0** | |
| checkpoints restored / created / erased; full re-processing | 11 / 0 / 0; 0 | **13 / 0 / 0; 0** | |
| graph reallocations | 0 in 60 requests | **0 in 62** | |
| busy SM clock, mean of the nine cards' medians, MHz | 1,885.3 | **1,959.8** | +3.95 % |
| highest card temperature °C; highest power W | 47; 80.52 | **50; 83.88** | |
| lowest MemAvailable MiB; swap highest MiB | 414; 426 | **424; 873** | |

Each value is one run ([flash_agent.tsv](flash_agent.tsv), [flash_same_prompts.md](flash_same_prompts.md),
[flash_work.md](flash_work.md), [flash_steps.md](flash_steps.md), [flash_clocks.tsv](flash_clocks.tsv),
[flash_cards.md](flash_cards.md), [flash_host.tsv](flash_host.tsv), [flash_reallocs.tsv](flash_reallocs.tsv),
[fnoc1_gonogo.md](fnoc1_gonogo.md)). Minutes a task at equal work price R90 `v3`'s work at each run's own prompt and
generation speeds over the whole session. So this run's one long reply, at a context `fnoc0` never reached, lowers
its equal-work speed too.

- **The verdict: negligible.** The two numbers the class is set on, minutes a task at equal work and generation, moved
  less than the same preset moves between two stock runs, so the change is within the spread. It is not slower: at
  the same context the cards decode faster, and the same prompts are read faster. It stays in the stacked line as a
  negligible gain. Ornith's offset half `o4oc1` ran after it on the same boot.
- **Decode at the same context** ([flash_steps.md](flash_steps.md), from the server logs' timings:
  [run/flash_steps.py](run/flash_steps.py)). Each request is put in the bucket of its middle context: its start plus
  half the tokens it generated. A decode step is one verify pass of the target plus the draft's passes, so its time
  leaves out how many drafted tokens were accepted. Under 4,000 tokens: 110.06 against 113.06 ms a step (−2.65 %),
  30.23 against 29.07 tok/s (+3.99 %). At 4,000 to 8,000 tokens, where 41 and 38 of the requests fall: 115.67 against
  117.94 ms a step (−1.92 %), 28.79 against 27.93 tok/s (+3.08 %). At 8,000 to 16,000 tokens: 121.09 against
  121.89 ms a step (−0.66 %), but 25.75 against 27.74 tok/s. That bucket holds this run's 15,807-token reply, at an
  acceptance of 0.705, while `fnoc0` has six short requests there. Over 16,000 tokens, where `fnoc0` never went: three
  requests, 25.37 tok/s, 130.92 ms a step.
- **The same prompts** ([flash_same_prompts.md](flash_same_prompts.md)): the 11 first turns took 83.44 s against
  84.27 s (−0.99 %). Ten of the 11 were faster, the most JavaScript `variable-length-quantity`'s (4,625.2 against
  4,861.5 ms, −4.86 %). Python `scale-generator`'s first turn was 7.52 % slower (7,809.2 against 7,263.3 ms). Prompt
  reading is mostly compute, and only three cards have a core offset.
- **The task that failed:** Python `scale-generator`, ended by the 12-minute cap ("runaway time", 3 turns). Its second
  turn reasoned for 15,807 tokens (47,228 characters, 635.6 s at 24.87 tok/s), working through the exercise's own
  ambiguity (a lowercase `d` wants sharps in the dorian test and flats in the harmonic minor one), and ended in a
  well-formed call to run the tests. The cap fell during the third turn. At stock the same preset has lost tasks the
  same way: JavaScript `complex-numbers` in C854's run of v4.0 on the first rig (34 turns, 13,254 tokens). The tasks
  are sampled at temperature 1.0, so a run's work differs. `fnoc0`'s `scale-generator` took 5 turns and 1.9 minutes.
- **The answers looked sound.** None of the 61 tool calls was malformed, there were no calls inside the reasoning, and
  there was no NaN or inf in the server log. The 2 failed calls were the agent's own scratch Python scripts exiting
  with an error inside `bash` (Python `pig-latin` and `variable-length-quantity`), which then passed. The report's
  proxy gate is NO-GO only because the capped request has no timings, and its memory gate is NO-GO on the swap, as
  for `fnoc0` ([fnoc1_gonogo.md](fnoc1_gonogo.md)).
- **First token follows the turn mix.** The turns after the first read a median of 231.5 new tokens (`fnoc0`: 215.5),
  with a median server prefill of 2.10 s (2.00 s), so the median of all turns rose. A task's first turn, the same
  request in both runs, took a median 6.50 s (6.66 s).
- **The cards' busy clocks** ([flash_clocks.tsv](flash_clocks.tsv), busy = utilization above 0 %, 16:42:03–17:12:42):
  medians of 2,113 1,885 1,885 1,860 2,100 1,898 1,873 1,898 2,126 MHz on cards 0 to 8 (mean 1,959.8), against `fnoc0`'s
  1,860 1,885 1,885 1,873 1,898 1,898 1,873 1,898 1,898. The core offsets put cards 0, 4 and 8 at +253, +202 and
  +228 MHz while busy; their busy samples below 1,800 MHz fell to 0.39, 0.89 and 0.52 % (`fnoc0`: 3.46, 1.67 and
  1.05 %). The memory ran at busy medians of 4,657 4,498 4,498 4,551 4,551 4,608 4,455 4,455 4,657 MHz against 4,006
  on every card at stock ([flash_cards.md](flash_cards.md)).
- **Heat and power** ([flash_cards.md](flash_cards.md), the card log every 0.5 s over the tasks): the hottest card was
  card 6 at 50 °C (`fnoc0`: card 0 at 47 °C). Power went above 80 W only in single samples: 3, 1, 1, 0, 2, 0, 1, 0 and
  7 samples on cards 0 to 8, of about 3,700 each. The highest was 83.88 W on card 8 (`fnoc0`: 80.52 W on card 1, one
  sample). The watchdog trips at 80 W held for 5 s.
- **Host and cards** ([flash_host.tsv](flash_host.tsv)): lowest MemAvailable 424 MiB at 16:43:09, as the first task
  ran; swap 37 MiB at the start and 873 MiB at its highest at 17:12:10, during the last task (`fnoc0`: 23 and 426).
  The RAM watchdog stopped nothing. The card log's peaks, 5,192 5,234 5,094 5,250 4,350 5,108 4,504 5,120 5,858 MiB,
  are 18, 14, 18, 14, 16, 16, 12 and 16 MiB above `fnoc0`'s on cards 0 to 7, and 164 MiB above on card 8, at this
  run's longer contexts.
- **Graph reallocations:** 0 in 62 requests, in every prompt-size bucket ([flash_reallocs.tsv](flash_reallocs.tsv)).
- **The rig at the end:** the offsets stayed set and the watchdog kept running for Ornith's offset half on the same
  boot (below).

## Ornith with the offsets: `o4oc1` (2026-11-18)

**12 of 12 tasks passed, 2.05 minutes a task at equal work, generation 17.31 tok/s. Against `o4oc0` on the same boot
(2.13 minutes, 16.42 tok/s) that is −3.76 % and +5.42 %, both beyond the spread of two stock runs of the same preset
(R166a and R166b: 2.11 and 2.13 minutes, 0.95 % apart; 16.52 and 16.13 tok/s, 2.42 % apart): a gain on the agent's
speed.** Part of the session's generation figure comes from the work, since this run's replies were shorter and
stayed at smaller contexts. The gain holds where the two runs did the same kind of work: a decode step took 4.83 %
less time at contexts under 4,000 tokens and 4.22 % less at 4,000 to 8,000 tokens, where most of both runs' tokens
were generated, and the 19 identical prompts took 1.65 % less time.

**The run:** no `apply` of its own. The offsets that `apply` set before `fnoc1` stayed set, with the watchdog
running, through Flash-Next's run and this one. At 17:26:27 the kit's state showed every card at its profile values,
no load, no GPU Xid, no step-down this boot and 3,150 MiB of host memory available; the card log's start line read the
same offsets on every card. The launcher is `o4oc0`'s with only the arm and its own rig folder changed. The started
command is `o4oc0`'s byte for byte (521 bytes, md5 `211f8db2edeacd267705795e4911d8c9`), and so is the server's
`switches:` line. The server was up after 662 s (`o4oc0`: 642 s), and the tasks ran 17:37:39–18:01:59. At 18:02:34
every card still read its profile values, the watchdog was running with no trip, and there was no GPU Xid. The
machine that drives the rig was the agent's client. An Ornith agent check on the first rig used the same client from
17:50:54, the last 11.1 minutes of these tasks, with its own ports and folders; that shows only in the time outside
the server.

| | R203a `o4oc0`, stock | **R203b `o4oc1`, the use profile** | change |
|---|---|---|---|
| passed | 12 / 12 | **12 / 12** | |
| minutes a task at equal work (`ka`'s work) | 2.13 | **2.05** | −3.76 % |
| generation tok/s | 16.42 | **17.31** | +5.42 % |
| prompt tok/s, the whole session | 102.2 | **101.6** | −0.59 % |
| first token median s | 7.07 | **6.60** | −6.65 % |
| first token p90 s | 11.51 | **10.82** | −5.99 % |
| first token on a task's first turn, median s | 8.48 | **8.30** | −2.12 % |
| the 19 requests read on the same prompt, prompt s | 192.5 | **189.4** | −1.65 % |
| of them, the 12 first turns, prompt s | 116.9 | **115.4** | −1.30 % |
| prompt tokens read a task | 3,725 | **3,296** | −11.52 % |
| tokens generated a task | 1,980 | **1,418** | |
| reply tokens re-read (turns; their prompt s of all prompt s) | 8,860 (5; 65.6 of 437.3) | **4,074 (1; 28.8 of 389.2)** | |
| ms a decode step, middle context under 4,000 / 4,000–8,000 / 8,000–16,000 | 57.82 / 60.71 / 64.83 | **55.03 / 58.15 / 63.30** | −4.83 / −4.22 / −2.36 % |
| minutes, all 12 tasks; turns | 31.9; 67 | **24.1; 62** | |
| minutes a task, median | 2.52 | **1.30** | |
| tool calls / failed / malformed | 67 / 2 / 0 | **62 / 2 / 0** | |
| checkpoints restored / created / erased; full re-processing | 16 / 0 / 0; 0 | **12 / 0 / 0; 0** | |
| graph reallocations | 0 in 67 requests | **0 in 62** | |
| busy SM clock, mean of the nine cards' medians, MHz | 1,883.9 | **1,959.8** | +4.03 % |
| highest card temperature °C; highest power W | 48; 82.77 | **47; 85.34** | |
| lowest MemAvailable MiB; swap highest MiB | 1,207; 334 | **1,183; 358** | |

Each value is one run ([agent.tsv](agent.tsv), [same_prompts.md](same_prompts.md), [work.md](work.md),
[steps.md](steps.md), [clocks.tsv](clocks.tsv), [cards.md](cards.md), [host.tsv](host.tsv),
[reallocs.tsv](reallocs.tsv), [o4oc1_gonogo.md](o4oc1_gonogo.md)). Minutes a task at equal work price `ka`'s work at
each run's own prompt and generation speeds over the whole session, so this run's shorter contexts help its
equal-work figure a little; the decode steps at the same context are the like-for-like measure.

- **The verdict: gain.** Both numbers the class is set on moved more than the same preset moves between two stock
  runs, and in the direction of the offsets. The like-for-like pieces agree: decode steps at the same context took
  4.83 % and 4.22 % less time, and the same prompts 1.65 % less. It goes into the stacked line.
- **Decode at the same context** ([steps.md](steps.md), from the server logs' timings:
  [run/ornith_steps.py](run/ornith_steps.py), the buckets of [run/flash_steps.py](run/flash_steps.py)). Ornith runs
  without a draft, so a decode step is one token. Under 4,000 tokens, 21 requests in each run: 55.03 against
  57.82 ms a token (−4.83 %), 18.17 against 17.30 tok/s (+5.03 %). At 4,000 to 8,000 tokens, 39 and 43 requests with
  13,724 and 18,049 tokens: 58.15 against 60.71 ms (−4.22 %), 17.20 against 16.47 tok/s (+4.43 %). At 8,000 to
  16,000 tokens, 2 and 3 requests: 63.30 against 64.83 ms (−2.36 %). Over all requests: 57.76 against 60.89 ms
  (−5.14 %), which also carries the context mix.
- **The same prompts** ([same_prompts.md](same_prompts.md)): the 19 requests took 189.4 s against 192.5 s (−1.65 %),
  and against 190.8 and 190.3 s in R166 two days earlier. 17 of the 19 were faster, the most Python `list-ops`'s first
  turn (−4.08 %); JavaScript `triangle`'s first turn took 1.99 % more time and `grade-school`'s 4.07 % more. Prompt
  reading is mostly compute, and only three cards have a core offset.
- **The work** ([work.md](work.md)): at temperature 0.6 the agent wrote shorter replies this time, 1,418 tokens a task
  against 1,980, in 62 turns against 67, and read a reply again on 1 turn against 5. So the minutes for all tasks
  (24.1 against 31.9) follow the work; the equal-work row compares the speeds. Python `scale-generator`, the longest
  task in both runs, took 5 turns and 6.4 minutes (`o4oc0`: 8 turns, 6.7 minutes).
- **The answers looked sound.** All 12 tasks passed with the test files untouched, and the hidden tests passed too.
  None of the 62 tool calls was malformed, there were no calls inside the reasoning, and there was no NaN or inf in
  the server log. The 2 failed calls were two edits in Python `affine-cipher` whose old text had the wrong
  indentation; the agent printed the lines, made the edit again and the tests passed. The report's memory gate is
  NO-GO on the swap, as for `o4oc0`; every other gate is GO ([o4oc1_gonogo.md](o4oc1_gonogo.md), over both arms, as
  the client's results folder held both).
- **First token** fell with the work: a task's first turn, the same request in both runs, took a median 8.30 s
  (8.48 s, −2.12 %), close to the same prompts' −1.30 % on the 12 first turns. The median and p90 over all turns also
  follow the turn mix: `o4oc0` read a reply again on five turns, with first tokens of 7.46, 9.50, 12.56, 17.61 and
  18.92 s, and this run on one, at 28.94 s.
- **The cards' busy clocks** ([clocks.tsv](clocks.tsv), busy = utilization above 0 %, 17:37:39–18:01:59): medians of
  2,113 1,885 1,885 1,860 2,100 1,898 1,873 1,898 2,126 MHz on cards 0 to 8 (mean 1,959.8, as in `fnoc1`), against
  `o4oc0`'s 1,860 1,885 1,885 1,860 1,898 1,898 1,873 1,898 1,898. The core offsets put cards 0, 4 and 8 at +253, +202
  and +228 MHz while busy; their busy samples below 1,800 MHz fell to 0.77, 1.02 and 1.35 % (`o4oc0`: 3.73, 3.32 and
  2.36 %). The memory ran at busy medians of 4,657 4,498 4,498 4,551 4,551 4,608 4,455 4,455 4,657 MHz against 4,006
  on every card at stock ([cards.md](cards.md)).
- **Heat and power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): the hottest card was card 0 at
  47 °C (`o4oc0`: card 0 at 48 °C). Power went above 80 W only in single samples: 2, 1, 2, 1, 3, 0, 0, 2 and 5
  samples on cards 0 to 8, of 2,920 each (`o4oc0`: one each on cards 7 and 8, of 3,852). The highest was 85.34 W on
  card 8 (`o4oc0`: 82.77 W, also card 8). The watchdog trips at 80 W held for 5 s and never tripped.
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,183 MiB at 18:01:26, during the last task; swap
  42 MiB at the start, left from the earlier runs on the boot, and 358 MiB at its highest at 17:32:11, during the
  load (`o4oc0`: 0 and 334). The RAM watchdog stopped nothing. The card log's peaks, 5,810 5,830 5,756 5,830 5,438
  5,546 5,708 5,784 5,190 MiB, are 2 MiB above `o4oc0`'s on cards 0 to 3 and 6 to 8 and the same on cards 4 and 5.
- **Graph reallocations:** 0 in 62 requests, in every prompt-size bucket ([reallocs.tsv](reallocs.tsv),
  [o4oc1_requests_reallocs.tsv](o4oc1_requests_reallocs.tsv)).
- **The rig at the end** (18:02:44–18:02:55), in the kit's folder with `apply`'s work folder:

  ```bash
  python3 climb.py reset-all        # every card's memory and core offset to 0, one card a second
  python3 climb.py watchdog stop    # "watchdog stop after 0 trips"
  python3 climb.py state            # every offset 0, 1,911 / 4,004 MHz maximum, no load, no GPU Xid
  ```

  Every card read 0 back. The kit's ledger counts the nine resets as step-downs this boot, so its start check wants
  a fresh boot before the next offset is set.

**Both models together.** On the second rig the use profile, memory +900 to +1,300 MT/s on every card and the core
+206 to +256 MHz on three of them, is worth about 4 to 5 % to Ornith's agent: its decode steps took 4.22 to 4.83 %
less time at the same context, and the session ran 3.76 % faster at equal work. That is what R201's single-request
check found (+4.42 %). Flash-Next gained less, 1.92 to 2.65 % a decode step at the same context, and inside the spread
of stock runs over the whole session. Neither model's answers changed in kind: no malformed calls, no NaN, every task
but one passed with its hidden tests, and the one that failed ran out of time on a sound reply.

## The full profile (first look): `o4oc2` (2026-11-18)

**In plain words.** The use profile above gives only three cards a core offset. Later the same day the core soak
ladder ([R202b](../2026-11-18_r202b_core_soak/README.md), Q1588) found a core offset for every card that held
90 minutes with no differing check. This run is one Ornith agent check with that full profile,
[use-profile.json](../2026-11-18_r202b_core_soak/use-profile.json): memory as before (+1,300, +1,000, +1,000, +1,100,
+1,100, +1,200, +900, +900, +1,300 MT/s on cards 0 to 8), core +256, +256, +269, +231, +206, +206, +282, +256 and
+231 MHz. It is a first look and not a verdict. `o4oc0` and `o4oc1` ran on the boot of 14:57 and this run on the boot
of 20:47, so a change between boots is not separated from the offsets. The same-boot comparison is a later test
(Q1603).

**12 of 12 tasks passed, 1.96 minutes a task at equal work, generation 17.89 tok/s.** Against the three-card
profile's `o4oc1` (2.05 minutes, 17.31 tok/s) that is −4.39 % and +3.35 %. Against stock `o4oc0` (2.13 minutes,
16.42 tok/s) it is −7.98 % and +8.95 %. The like-for-like pieces point the same way:

- At the same context a decode step took 4.29 %, 4.73 % and 3.98 % less time than in `o4oc1` and 8.91 %, 8.75 % and
  6.25 % less than at stock.
- The 19 identical prompts took 5.08 % less time than in `o4oc1` and 6.45 % less than at stock. With a core offset on
  every card, prompt reading got faster too; the three-card profile had left it almost unchanged.

**The run.** Rig 2 was rebooted at 20:47:19, after R202b had set every offset back to 0. At 20:49 there was no llama
process, every card used 0 MiB, and the kit's read-only start check passed (every offset 0, no step-down this boot).
The watchdog started at 20:49:44 and `apply` of the full profile at 20:49:50. Both stock rounds gave every card the
hash `60ee0f178cfd67d5`. The offsets were set 20:54:05–20:54:12, each card "read back as set", and in the round with
the offsets every card passed its greedy check: `APPLY_DONE` "exit 0" at 20:55:04. At 20:55:18 the kit's state showed
every card at its profile value, the watchdog running, no GPU Xid, no step-down and 3,146 MiB of host memory
available. The launcher is `o4oc1`'s with only the arm and its own rig folder changed, on a copy of this repository at
`b1190e7b`. The started command is `o4oc0`'s and `o4oc1`'s byte for byte (521 bytes, md5
`211f8db2edeacd267705795e4911d8c9`), and so is the server's `switches:` line. The server was up after 582 s
(`o4oc0`: 642 s, `o4oc1`: 662 s), and the tasks ran 21:05:19–21:32:33. Nothing else used the client during the run.
At 21:33:11 every card still read its profile value, the watchdog was running with no trip, and there was no GPU Xid.

| | R203a `o4oc0`, stock | R203b `o4oc1`, three-card profile | **R203e `o4oc2`, the full profile** | change against `o4oc1` | change against `o4oc0` |
|---|---|---|---|---|---|
| boot | 14:57:26 | 14:57:26 | **20:47:19** | | |
| passed | 12 / 12 | 12 / 12 | **12 / 12** | | |
| minutes a task at equal work (`ka`'s work) | 2.13 | 2.05 | **1.96** | −4.39 % | −7.98 % |
| generation tok/s | 16.42 | 17.31 | **17.89** | +3.35 % | +8.95 % |
| prompt tok/s, the whole session | 102.2 | 101.6 | **109.9** | +8.17 % | +7.53 % |
| first token median s | 7.07 | 6.60 | **7.39** | +11.97 % | +4.53 % |
| first token p90 s | 11.51 | 10.82 | **10.75** | −0.65 % | −6.60 % |
| first token on a task's first turn, median s | 8.48 | 8.30 | **8.01** | −3.49 % | −5.54 % |
| the 19 requests read on the same prompt in these three runs, prompt s | 187.2 | 184.5 | **175.1** | −5.08 % | −6.45 % |
| of them, the 12 first turns, prompt s | 116.9 | 115.4 | **109.4** | −5.16 % | −6.39 % |
| prompt tokens read a task | 3,725 | 3,296 | **3,614** | | |
| tokens generated a task | 1,980 | 1,418 | **1,793** | | |
| reply tokens re-read (turns; their prompt s of all prompt s) | 8,860 (5; 65.6 of 437.3) | 4,074 (1; 28.8 of 389.2) | **8,130 (5; 59.1 of 394.5)** | | |
| ms a decode step, middle context under 4,000 / 4,000–8,000 / 8,000–16,000 | 57.82 / 60.71 / 64.83 | 55.03 / 58.15 / 63.30 | **52.67 / 55.40 / 60.78** | −4.29 / −4.73 / −3.98 % | −8.91 / −8.75 / −6.25 % |
| minutes, all 12 tasks; turns | 31.9; 67 | 24.1; 62 | **27.0; 63** | | |
| minutes a task, median | 2.52 | 1.30 | **1.45** | | |
| tool calls / failed / malformed | 67 / 2 / 0 | 62 / 2 / 0 | **63 / 0 / 0** | | |
| checkpoints restored / created / erased; full re-processing | 16 / 0 / 0; 0 | 12 / 0 / 0; 0 | **16 / 0 / 0; 0** | | |
| graph reallocations | 0 in 67 requests | 0 in 62 | **0 in 63** | | |
| busy SM clock, mean of the nine cards' medians, MHz | 1,883.9 | 1,959.8 | **2,125.6** | +8.46 % | +12.83 % |
| highest card temperature °C; highest power W | 48; 82.77 | 47; 85.34 | **46; 86.74** | | |
| lowest MemAvailable MiB; swap highest MiB | 1,207; 334 | 1,183; 358 | **1,238; 359** | | |

Each value is one run ([agent.tsv](agent.tsv), [fulloc_same_prompts.md](fulloc_same_prompts.md), [work.md](work.md),
[steps.md](steps.md), [clocks.tsv](clocks.tsv), [cards.md](cards.md), [host.tsv](host.tsv),
[reallocs.tsv](reallocs.tsv), [o4oc2_gonogo.md](o4oc2_gonogo.md)). The same-prompt row matches requests over these
three runs only, so its 19 requests are not quite the 19 that `o4oc1`'s section matched over four runs with R166's
two; [same_prompts.md](same_prompts.md) is that four-run table, unchanged.

- **The class waits for the same-boot test.** Against `o4oc1`, both numbers the class is set on moved more than the
  same preset moves between runs on this rig: R166a and R166b are 0.95 % and 2.42 % apart, and `o4oc0`, two days
  after R166, took 2.13 minutes against 2.11 and 2.13 and generated 16.42 tok/s against 16.52 and 16.13. The
  like-for-like pieces moved too, and in the direction of the offsets. So this looks like a gain on top of the three-card profile. It still has to be measured with stock and
  the offsets on one boot (Q1603).
- **Decode at the same context** ([steps.md](steps.md)). Under 4,000 tokens, 21 requests: 52.67 ms a token against
  55.03 and 57.82 (18.99 tok/s against 18.17 and 17.30: +4.51 % and +9.77 %). At 4,000 to 8,000 tokens, 36 requests
  with 14,319 tokens: 55.40 ms against 58.15 and 60.71 (18.05 tok/s: +4.94 % and +9.59 %). At 8,000 to 16,000 tokens,
  6 requests with 3,706 tokens: 60.78 ms against 63.30 and 64.83 (16.45 tok/s: +4.11 % and +6.68 %). Over all
  requests: 55.88 ms against 57.76 and 60.89 (−3.25 % and −8.23 %), which also carries the context mix. This run
  generated 3,706 tokens above 8,000 tokens of context, against 428 and 2,965.
- **The same prompts** ([fulloc_same_prompts.md](fulloc_same_prompts.md)): the 19 requests took 175.1 s against
  184.5 s and 187.2 s. All 19 were faster than in `o4oc1`, most of all JavaScript `triangle`'s first turn (−10.62 %)
  and `variable-length-quantity`'s (−9.30 %). 18 were faster than at stock; JavaScript `grade-school`'s first turn
  took 0.07 % more time. The session's prompt speed rose 8.17 %, where the three-card profile had moved it −0.59 %.
  Prompt reading is mostly compute, and now every card has a core offset.
- **The work** ([work.md](work.md)): 1,793 tokens generated a task, between `o4oc1`'s 1,418 and `o4oc0`'s 1,980, in 63
  turns. The agent read a reply again on 5 turns, as at stock. Python `scale-generator`, the longest task again, took
  8 turns and 7.98 minutes (`o4oc1`: 5 turns, 6.4 minutes; `o4oc0`: 8 turns, 6.7 minutes).
- **The answers looked sound.** All 12 tasks passed with the test files untouched, and the hidden tests passed too.
  None of the 63 tool calls failed or was malformed, there were no calls inside the reasoning, and there was no NaN
  or inf in the server log. The report's memory gate is NO-GO on the swap, as for the other two runs; every other
  gate is GO ([o4oc2_gonogo.md](o4oc2_gonogo.md), over all three arms, as the client's results folder held them all).
- **First token** follows the turn mix again. A task's first turn, the same request in every run, took a median
  8.01 s (−3.49 % and −5.54 %), close to the same 12 first turns' −5.16 % and −6.39 %. The median over all turns rose
  to 7.39 s because the agent read a reply again on five turns (first tokens of 8.08, 9.02, 11.02, 17.46 and
  14.04 s), against one in `o4oc1`.
- **The cards' busy clocks** ([clocks.tsv](clocks.tsv), busy = utilization above 0 %, 21:05:19–21:32:33): medians of
  2,113 2,138 2,151 2,100 2,100 2,100 2,151 2,151 2,126 MHz on cards 0 to 8 (mean 2,125.6), against `o4oc1`'s 2,113
  1,885 1,885 1,860 2,100 1,898 1,873 1,898 2,126 and `o4oc0`'s 1,860 1,885 1,885 1,860 1,898 1,898 1,873 1,898
  1,898. Busy samples below 1,800 MHz: 0.68 0.59 0.63 0.56 0.68 0.90 1.08 0.85 1.24 %. The memory ran at the same busy
  medians as in `o4oc1`: 4,657 4,498 4,498 4,551 4,551 4,608 4,455 4,455 4,657 MHz ([cards.md](cards.md)).
- **`apply`** ([fulloc_apply.md](fulloc_apply.md), the kit's own table [o4oc2_apply_steps.csv](o4oc2_apply_steps.csv)):
  every card's greedy check at the offsets equalled its stock hash. Streaming reads at stock and with the offsets,
  GB/s: 179.23 → 207.33, 179.15 → 199.33, 179.12 → 199.71, 179.14 → 201.88, 179.15 → 202.17, 179.21 → 204.90,
  179.12 → 197.47, 179.09 → 197.27, 179.24 → 207.21 on cards 0 to 8, as with the three-card profile, since the memory
  offsets are the same. Peak core clocks with the offsets: 2,126 2,151 2,151 2,100 2,113 2,100 2,151 2,151 2,138 MHz.
- **Heat and power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): the hottest card was card 0 at
  46 °C. Power went above 80 W only in single samples: 4, 6, 4, 0, 1, 1, 7, 4 and 3 samples on cards 0 to 8, of
  3,270 each. The highest was 86.74 W on card 1. The watchdog trips at 80 W held for 5 s and never tripped.
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,238 MiB at 21:30:34, during the last task. Swap
  was 43 MiB at the server's start, left from `apply`'s check servers on this boot, and 359 MiB at its highest at
  20:58:41, during the load. The RAM watchdog stopped nothing. The card log's peaks, 5,830 5,846 5,776 5,846 5,456
  5,564 5,724 5,796 5,202 MiB, are 20, 16, 20, 16, 18, 18, 16, 12 and 12 MiB above `o4oc1`'s; this session reached
  deeper contexts (11,725 tokens on its longest task).
- **Graph reallocations:** 0 in 63 requests, in every prompt-size bucket ([reallocs.tsv](reallocs.tsv),
  [o4oc2_requests_reallocs.tsv](o4oc2_requests_reallocs.tsv)).
- **The rig at the end** (21:33:15–21:33:27), in the kit's folder with this boot's `apply` work folder: `reset-all`
  (each card to 0 0), `watchdog stop` ("watchdog stop after 0 trips"), `state`: every offset 0, 1,911 / 4,004 MHz
  maximum on every card, no load, no GPU Xid. The kit's ledger counts the nine resets as step-downs this boot, so its
  start check wants a fresh boot before the next offset is set.

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv) | per arm (`ka`'s reference `v3ka`, R166a `o4sna`, R166b `o4snb`, `o4oc0`, `o4oc1`, `o4oc2`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes for all tasks, a task's median, minutes a task at equal work against `v3ka` |
| [same_prompts.md](same_prompts.md) | the 19 requests read on the same prompt in `o4sna`, `o4snb`, `o4oc0` and `o4oc1` (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [work.md](work.md), [work.tsv](work.tsv) | the work of each run: prompt tokens read, tokens generated, reply tokens re-read and the turns that re-read a reply |
| [steps.md](steps.md) | `o4oc0`, `o4oc1` and `o4oc2` by the context at a request's middle: requests, tokens, decode s, tok/s and ms a decode step (one token, no draft) |
| [clocks.tsv](clocks.tsv) | each card's busy SM clock median, their mean and each card's share of busy samples below 1,800 MHz, over each run's tasks |
| [cards.md](cards.md) | per card over `o4oc0`'s, `o4oc1`'s and `o4oc2`'s tasks: highest temperature and power, samples above 80 W, the busy memory clock's median |
| [host.tsv](host.tsv) | lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the log's end, each card's peak memory (`o4oc0`'s, `o4oc1`'s and `o4oc2`'s from the card log, 75 MiB over `nvidia-smi`'s on an idle card) |
| [reallocs.tsv](reallocs.tsv), [o4oc0_requests_reallocs.tsv](o4oc0_requests_reallocs.tsv), [o4oc1_requests_reallocs.tsv](o4oc1_requests_reallocs.tsv), [o4oc2_requests_reallocs.tsv](o4oc2_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size (the five runs), and per request (`o4oc0`, `o4oc1`, `o4oc2`) |
| [o4oc0_compare.md](o4oc0_compare.md), [o4oc0_report_table.md](o4oc0_report_table.md), [o4oc0_gonogo.md](o4oc0_gonogo.md) | the agent check's report on `o4oc0`: per task, per arm and first token per turn; per arm over the client's attempts; the gates and the server log's counts |
| [o4oc1_compare.md](o4oc1_compare.md), [o4oc1_report_table.md](o4oc1_report_table.md), [o4oc1_gonogo.md](o4oc1_gonogo.md) | the same on `o4oc1`; its report table and gates cover both arms, since the client's results folder held both |
| [o4oc2_compare.md](o4oc2_compare.md), [o4oc2_report_table.md](o4oc2_report_table.md), [o4oc2_gonogo.md](o4oc2_gonogo.md) | the same on `o4oc2`, the full profile; its report table and gates cover all three arms |
| [fulloc_same_prompts.md](fulloc_same_prompts.md) | the 19 requests read on the same prompt in `o4oc0`, `o4oc1` and `o4oc2` (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [fulloc_apply.md](fulloc_apply.md), [o4oc2_apply_steps.csv](o4oc2_apply_steps.csv) | the kit's `apply` of the full profile before `o4oc2`, card by card (as `flash_apply.md`), and the kit's own table |
| [agent/o4oc0/](agent/o4oc0/), [agent/o4oc1/](agent/o4oc1/), [agent/o4oc2/](agent/o4oc2/) | `turns.csv` (one row a request) and `attempts.csv` (one row a task) of each run |
| [run/ornith_agent.sh](run/ornith_agent.sh) | the harness call of both halves, the arm and the machine-specific values from the environment |
| [run/tables.sh](run/tables.sh) | the tables, from a folder holding R166's rows and each arm's turn tables, server log, card log and host log |
| [run/ornith_steps.py](run/ornith_steps.py) | `steps.md`, from each arm's server log and `agent/<arm>/turns.csv` |
| [flash_agent.tsv](flash_agent.tsv) | Flash-Next per arm (R90's `r90v3`, the equal-work reference, `fnoc0` and `fnoc1`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, the draft's acceptance, minutes for all tasks, a task's median, minutes a task at equal work against `r90v3` |
| [flash_same_prompts.md](flash_same_prompts.md) | the 11 requests read on the same prompt in `fnoc0`, `fnoc1` and R156a `v4r` (cached counts matched within one token); each run's generation over all its requests |
| [flash_work.md](flash_work.md), [flash_work.tsv](flash_work.tsv) | the work of `fnoc0`, `fnoc1` and R156a `v4r`: prompt tokens read, tokens generated, reply tokens re-read and the turns that re-read a reply |
| [flash_clocks.tsv](flash_clocks.tsv) | each card's busy SM clock median, their mean and each card's share of busy samples below 1,800 MHz, over `fnoc0`'s, `fnoc1`'s and R156a's tasks |
| [flash_host.tsv](flash_host.tsv) | as `host.tsv`, for `fnoc0` and `fnoc1` (their card peaks from the card log) and R156a |
| [flash_reallocs.tsv](flash_reallocs.tsv) | `fnoc0`'s and `fnoc1`'s graph reallocations and prompt seconds per prompt size |
| [flash_steps.md](flash_steps.md) | `fnoc0` and `fnoc1` by the context at a request's middle: requests, tokens, decode s, tok/s, the draft's acceptance, tokens a decode step and ms a decode step |
| [flash_apply.md](flash_apply.md), [fnoc1_apply_steps.csv](fnoc1_apply_steps.csv) | the kit's `apply` before `fnoc1`, card by card: the offsets, the streaming read and the peak core clock at stock and with the offsets, the greedy check; the kit's own table (`out/apply_steps.csv`: both stock rounds' times and limits, the apply round's checks) |
| [flash_cards.md](flash_cards.md) | per card over `fnoc0`'s and `fnoc1`'s tasks: highest temperature and power, samples above 80 W, the busy memory clock's median |
| [fnoc0_compare.md](fnoc0_compare.md), [fnoc0_report_table.md](fnoc0_report_table.md), [fnoc0_gonogo.md](fnoc0_gonogo.md) | the agent check's report on `fnoc0`, under the harness's arm name `v4r` |
| [fnoc1_compare.md](fnoc1_compare.md), [fnoc1_report_table.md](fnoc1_report_table.md), [fnoc1_gonogo.md](fnoc1_gonogo.md) | the same on `fnoc1` |
| [agent/fnoc0/](agent/fnoc0/), [agent/fnoc1/](agent/fnoc1/) | `turns.csv` and `attempts.csv` of each run, the arm renamed from `v4r` |
| [run/flash_run.sh](run/flash_run.sh) | one Flash-Next half: the rig's and the kit's state, C854's `run/flash_agent.sh` (R156's driver, arm `v4r`), the state and the kit's start check again |
| [run/flash_tables.sh](run/flash_tables.sh) | Flash-Next's tables, from a folder of their own holding the server logs, card logs and host logs it lists |
| [run/flash_steps.py](run/flash_steps.py) | `flash_steps.md`, from each arm's server log and `agent/<arm>/turns.csv` |
| [run/flash_cards.py](run/flash_cards.py) | `flash_apply.md` from the kit's `apply_steps.csv`, and `flash_cards.md` from each run's card log |

## Commands

From the repository root on the machine that drives the rig (here also the client), with the rig freshly booted, no
llama process, 0 MiB on every card, and the offset kit's read-only start check passing
(`python3 climb.py state && python3 climb.py start-check` in the kit's folder: every offset 0, no step-down this
boot). The bench files must be on the client in `CLIENT_DIR` (the check's `prep` step copies them).

```bash
# the stock half (~11 min load, 12 tasks), then its report; another check shared the local client, so other ports
RIG=$RIG CLIENT=local ARM=o4oc0 MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN=<v4.0's tree>/bin/llama-server RIG_LOGDIR=<rig folder>/agent-logs \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r203 RELAY_PORT=8296 PROXY_PORT=18793 \
  OUT=<out>/agent-o4oc0 bash docs/data/2026-11-18_r203_ornith_offsets/run/ornith_agent.sh
RIG=$RIG CLIENT=local ARM=o4oc0 MODEL=<the same> BIN=<the same> RIG_LOGDIR=<the same> CLIENT_DIR=<the same> \
  CLIENT_OUT=<the same> RELAY_PORT=8296 PROXY_PORT=18793 OUT=<out>/agent STEPS="report collect" \
  bash docs/data/2026-11-18_r203_ornith_offsets/run/ornith_agent.sh

# the tables, on a folder holding R166's rows and o4oc0's turn tables and logs (run/tables.sh lists them);
# o4oc0's window: 2026-11-18 15:19:19|2026-11-18 15:51:25 (the driver's log; its clock and the rig's agreed)
bash docs/data/2026-11-18_r203_ornith_offsets/run/tables.sh <tables folder> o4oc0
```

`res/turns.csv` and `res/attempts.csv` in the tables folder are the rows of `v3ka`, `o4sna` and `o4snb` that
[C854's part (5)](../2026-11-18_c854b_p106_check/README.md#part-5-the-agent-check-on-ornith-2026-11-18) used, plus
`o4oc0`'s; R166's server, host and card logs are the ones that table used.

Ornith's offset half, last on the same boot, with no `apply` of its own (the offsets set before `fnoc1` stayed set,
with the watchdog running), then every offset back to 0:

```bash
# the run and its report, as for o4oc0 with only the arm and the rig folder changed
RIG=$RIG CLIENT=local ARM=o4oc1 MODEL=<the same> BIN=<the same> RIG_LOGDIR=<rig folder>/r203b/agent-logs \
  CLIENT_DIR=<the same> CLIENT_OUT=<client folder>/r203 RELAY_PORT=8296 PROXY_PORT=18793 \
  OUT=<out>/agent-o4oc1 bash docs/data/2026-11-18_r203_ornith_offsets/run/ornith_agent.sh
RIG=$RIG CLIENT=local ARM=o4oc1 MODEL=<the same> BIN=<the same> RIG_LOGDIR=<the same> CLIENT_DIR=<the same> \
  CLIENT_OUT=<the same> RELAY_PORT=8296 PROXY_PORT=18793 OUT=<out>/agent STEPS="report collect" \
  bash docs/data/2026-11-18_r203_ornith_offsets/run/ornith_agent.sh

# on the rig, in the kit's folder with apply's work folder: every offset to 0, the watchdog stopped, read back
python3 climb.py reset-all && python3 climb.py watchdog stop && python3 climb.py state

# agent/o4oc1/ from the client's res/turns.csv and its res/attempts.csv rows of o4oc1; then o4oc0's tables folder
# with o4oc1's rows added to res/, its server log as rig/server-o4oc1.log, and rig_o4oc1/ holding its lines of the
# rig's meminfo.log (from its server's start) and card log; o4oc1's window: 2026-11-18 17:37:39|2026-11-18 18:01:59
D=docs/data/2026-11-18_r203_ornith_offsets
bash $D/run/tables.sh <tables folder> o4oc0 o4oc1
python3 -I $D/run/ornith_steps.py o4oc0=<tables>/rig/server-o4oc0.log=$D/agent/o4oc0/turns.csv \
  o4oc1=<tables>/rig/server-o4oc1.log=$D/agent/o4oc1/turns.csv > $D/steps.md
python3 -I $D/run/flash_cards.py cards "o4oc0=<tables>/rig_o4oc0/cards.csv=2026-11-18 15:19:19=2026-11-18 15:51:25" \
  "o4oc1=<tables>/rig_o4oc1/cards.csv=2026-11-18 17:37:39=2026-11-18 18:01:59" > $D/cards.md
```

The full profile's run (`o4oc2`), on a later fresh boot of the same rig, with its own `apply`:

```bash
# on the rig, in the kit's folder, with a fresh work folder: the read-only start check, the watchdog, apply (detached;
# it ends with APPLY_DONE in the work folder), then the state read back
export WORK=<fresh work folder>
python3 climb.py start-check && python3 climb.py watchdog start
python3 climb.py apply use-profile.json     # docs/data/2026-11-18_r202b_core_soak/use-profile.json
cat $WORK/APPLY_DONE && python3 climb.py state

# the run and its report, as for o4oc1 with only the arm and the rig folder changed
RIG=$RIG CLIENT=local ARM=o4oc2 MODEL=<the same> BIN=<the same> RIG_LOGDIR=<rig folder>/r203e/agent-logs \
  CLIENT_DIR=<the same> CLIENT_OUT=<client folder>/r203 RELAY_PORT=8296 PROXY_PORT=18793 \
  OUT=<out>/agent-o4oc2 bash docs/data/2026-11-18_r203_ornith_offsets/run/ornith_agent.sh
RIG=$RIG CLIENT=local ARM=o4oc2 MODEL=<the same> BIN=<the same> RIG_LOGDIR=<the same> CLIENT_DIR=<the same> \
  CLIENT_OUT=<the same> RELAY_PORT=8296 PROXY_PORT=18793 OUT=<out>/agent STEPS="report collect" \
  bash docs/data/2026-11-18_r203_ornith_offsets/run/ornith_agent.sh

# on the rig, as after o4oc1: every offset to 0, the watchdog stopped, read back
python3 climb.py reset-all && python3 climb.py watchdog stop && python3 climb.py state

# agent/o4oc2/ from the client's res/turns.csv and res/attempts.csv rows of o4oc2; then o4oc1's tables folder with
# o4oc2's rows added to res/, its server log as rig/server-o4oc2.log, and rig_o4oc2/ holding its lines of the rig's
# meminfo.log (from its server's start) and card log; o4oc2's window: 2026-11-18 21:05:19|2026-11-18 21:32:33
D=docs/data/2026-11-18_r203_ornith_offsets
bash $D/run/tables.sh <tables folder> o4oc0 o4oc1 o4oc2    # same_prompts.md stays the four-run table above
python3 -I docs/data/2026-11-18_c854b_p106_check/run/same_prompts.py o4oc0=$D/agent/o4oc0/turns.csv \
  o4oc1=$D/agent/o4oc1/turns.csv o4oc2=$D/agent/o4oc2/turns.csv > $D/fulloc_same_prompts.md
python3 -I $D/run/ornith_steps.py o4oc0=<tables>/rig/server-o4oc0.log=$D/agent/o4oc0/turns.csv \
  o4oc1=<tables>/rig/server-o4oc1.log=$D/agent/o4oc1/turns.csv \
  o4oc2=<tables>/rig/server-o4oc2.log=$D/agent/o4oc2/turns.csv > $D/steps.md
python3 -I $D/run/flash_cards.py cards "o4oc0=<tables>/rig_o4oc0/cards.csv=2026-11-18 15:19:19=2026-11-18 15:51:25" \
  "o4oc1=<tables>/rig_o4oc1/cards.csv=2026-11-18 17:37:39=2026-11-18 18:01:59" \
  "o4oc2=<tables>/rig_o4oc2/cards.csv=2026-11-18 21:05:19=2026-11-18 21:32:33" > $D/cards.md
python3 -I $D/run/flash_cards.py apply $D/o4oc2_apply_steps.csv > $D/fulloc_apply.md
```

Flash-Next's stock half the same way (the rig's paths written as `$MODELS` and `/opt/llama-builds`, as in R156; rig 1's
Ornith check held the default client ports that afternoon):

```bash
# the run (~4 min load, 12 tasks) and its report: the kit's state, r156.sh's agent and report steps, arm v4r, v4.0's build
RIG=$RIG CLIENT=local KIT=<the kit's folder on the rig> \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf NEW=llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828 \
  REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern RIGDIR=<rig folder>/r203c \
  RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r203c OUT=<out>/R203c \
  bash docs/data/2026-11-18_r203_ornith_offsets/run/flash_run.sh

# agent/fnoc0/ from the client's res/turns.csv and res/attempts.csv, the arm v4r renamed fnoc0 (the arm and file
# columns, the tag's first part); then the tables, on a folder holding R90's v3 server log, fnoc0's server log, card
# log and host log, and R156a's host and card logs (run/flash_tables.sh lists them);
# fnoc0's window: 2026-11-18 16:04:59|2026-11-18 16:23:34 (the driver's "client reaches" line and the last task's end)
bash docs/data/2026-11-18_r203_ornith_offsets/run/flash_tables.sh <Flash-Next tables folder> fnoc0
```

The run itself was started from a launcher that did the same three steps with this machine's values written in.

Flash-Next's offset half on the same boot, after the kit's `apply` (section "Flash-Next with the offsets"):

```bash
# the same run with its own folders; the offsets stay set while the watchdog runs
RIG=$RIG CLIENT=local KIT=<the kit's folder on the rig> MODEL=<the same> DRAFT=<the same> NEW=<the same> \
  REF=<the same> RIGDIR=<rig folder>/r203d RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r203d OUT=<out>/R203d \
  bash docs/data/2026-11-18_r203_ornith_offsets/run/flash_run.sh

# agent/fnoc1/ as agent/fnoc0/ (the arm v4r renamed fnoc1); fnoc1's server log as rig/server-fnoc1.log and its card
# and host logs in rig_fnoc1/ of fnoc0's tables folder; fnoc1's window: 2026-11-18 16:42:03|2026-11-18 17:12:42
bash docs/data/2026-11-18_r203_ornith_offsets/run/flash_tables.sh <Flash-Next tables folder> fnoc0 fnoc1
D=docs/data/2026-11-18_r203_ornith_offsets
python3 -I $D/run/flash_steps.py fnoc0=<tables>/rig/server-fnoc0.log=$D/agent/fnoc0/turns.csv \
  fnoc1=<tables>/rig/server-fnoc1.log=$D/agent/fnoc1/turns.csv > $D/flash_steps.md
python3 -I $D/run/flash_cards.py apply $D/fnoc1_apply_steps.csv > $D/flash_apply.md
python3 -I $D/run/flash_cards.py cards "fnoc0=<tables>/rig_fnoc0/cards.csv=2026-11-18 16:04:59=2026-11-18 16:23:34" \
  "fnoc1=<tables>/rig_fnoc1/cards.csv=2026-11-18 16:42:03=2026-11-18 17:12:42" > $D/flash_cards.md
```

## Provenance and licence

The tasks are the agent check's 12 bug fixes, derived from Exercism's Python and JavaScript exercises (MIT; source,
licences and commits in [bench/agent_check](../../../bench/agent_check/README.md)); this folder holds only their IDs
(in [agent/o4oc0/attempts.csv](agent/o4oc0/attempts.csv), [agent/o4oc1/attempts.csv](agent/o4oc1/attempts.csv),
[agent/fnoc0/attempts.csv](agent/fnoc0/attempts.csv) and [agent/fnoc1/attempts.csv](agent/fnoc1/attempts.csv)). The tables summarize this project's own runs and are
published under the repository's licence.
