# R203: the agent check at stock and with the second rig's clock offsets, Ornith and Flash-Next on one boot (Q1592, 2026-11-18)

**In plain words.** This measures the speed that counts, a coding agent's, with the memory and core offsets that held
on the second rig ([R201](../2026-11-18_r201_soak_ladder/README.md#the-second-boot-the-use-profile-held-and-ornith-generated-442--faster),
Q1587). On one fresh boot it runs each model's agent check twice: first with every card at stock, then, after the use
profile is applied, the same run again. The order on the boot is Ornith at stock, Flash-Next at stock, the kit's
`apply`, Flash-Next with the offsets, Ornith with the offsets. **Both stock halves have run.** Ornith: 12 of 12 tasks
passed, 2.13 minutes a task at equal work, generation 16.42 tok/s, where the released v4.0 preset stood on the same rig
two days earlier (R166: 2.11 and 2.13 minutes, 16.52 and 16.13 tok/s). Flash-Next: 12 of 12, 2.36 minutes a task at
equal work, generation 28.08 tok/s (R156a, the same preset and build on the same rig five days earlier: 12 of 12, 2.43
minutes, 26.67 tok/s, on longer tasks). The offset halves, the comparisons this row is for, come next on the same
boot.

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

## Flash-Next with the offsets: `fnoc1` (to come)

On the same boot: the kit's `apply` with the use profile, then [run/flash_run.sh](run/flash_run.sh) again with its
own output, rig and client folders, then [run/flash_tables.sh](run/flash_tables.sh) with `fnoc0 fnoc1`.

## The offset half: `o4oc1` (to come)

On the same boot: the offset kit's `apply` with the use profile, then the same command with `ARM=o4oc1` and the same
client folder, then [run/tables.sh](run/tables.sh) with both arms. This section gets its table, its busy clocks, its
temperatures and power, the kit's state before and after, and the comparison with `o4oc0`.

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv) | per arm (`ka`'s reference `v3ka`, R166a `o4sna`, R166b `o4snb`, `o4oc0`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes for all tasks, a task's median, minutes a task at equal work against `v3ka` |
| [same_prompts.md](same_prompts.md) | the 19 requests read on the same prompt in `o4sna`, `o4snb` and `o4oc0` (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [work.md](work.md), [work.tsv](work.tsv) | the work of each run: prompt tokens read, tokens generated, reply tokens re-read and the turns that re-read a reply |
| [clocks.tsv](clocks.tsv) | each card's busy SM clock median, their mean and each card's share of busy samples below 1,800 MHz, over each run's tasks |
| [host.tsv](host.tsv) | lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the log's end, each card's peak memory (`o4oc0`'s from the card log, 75 MiB over `nvidia-smi`'s on an idle card) |
| [reallocs.tsv](reallocs.tsv), [o4oc0_requests_reallocs.tsv](o4oc0_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size (the three runs), and per request (`o4oc0`) |
| [o4oc0_compare.md](o4oc0_compare.md), [o4oc0_report_table.md](o4oc0_report_table.md), [o4oc0_gonogo.md](o4oc0_gonogo.md) | the agent check's report on `o4oc0`: per task, per arm and first token per turn; per arm over the client's attempts; the gates and the server log's counts |
| [agent/o4oc0/](agent/o4oc0/) | `turns.csv` (one row a request) and `attempts.csv` (one row a task) |
| [run/ornith_agent.sh](run/ornith_agent.sh) | the harness call of both halves, the arm and the machine-specific values from the environment |
| [run/tables.sh](run/tables.sh) | the tables, from a folder holding R166's rows and each arm's turn tables, server log, card log and host log |
| [flash_agent.tsv](flash_agent.tsv) | Flash-Next per arm (R90's `r90v3`, the equal-work reference, and `fnoc0`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, the draft's acceptance, minutes for all tasks, a task's median, minutes a task at equal work against `r90v3` |
| [flash_same_prompts.md](flash_same_prompts.md) | the 11 requests read on the same prompt in `fnoc0` and R156a `v4r` (cached counts matched within one token); each run's generation over all its requests |
| [flash_work.md](flash_work.md), [flash_work.tsv](flash_work.tsv) | the work of `fnoc0` and R156a `v4r`: prompt tokens read, tokens generated, reply tokens re-read and the turns that re-read a reply |
| [flash_clocks.tsv](flash_clocks.tsv) | each card's busy SM clock median, their mean and each card's share of busy samples below 1,800 MHz, over `fnoc0`'s and R156a's tasks |
| [flash_host.tsv](flash_host.tsv) | as `host.tsv`, for `fnoc0` (its card peaks from the card log) and R156a |
| [flash_reallocs.tsv](flash_reallocs.tsv) | `fnoc0`'s graph reallocations and prompt seconds per prompt size |
| [fnoc0_compare.md](fnoc0_compare.md), [fnoc0_report_table.md](fnoc0_report_table.md), [fnoc0_gonogo.md](fnoc0_gonogo.md) | the agent check's report on `fnoc0`, under the harness's arm name `v4r` |
| [agent/fnoc0/](agent/fnoc0/) | `turns.csv` and `attempts.csv` of the run, the arm renamed from `v4r` |
| [run/flash_run.sh](run/flash_run.sh) | one Flash-Next half: the rig's and the kit's state, C854's `run/flash_agent.sh` (R156's driver, arm `v4r`), the state and the kit's start check again |
| [run/flash_tables.sh](run/flash_tables.sh) | Flash-Next's tables, from a folder of their own holding the server logs, card logs and host logs it lists |

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

## Provenance and licence

The tasks are the agent check's 12 bug fixes, derived from Exercism's Python and JavaScript exercises (MIT; source,
licences and commits in [bench/agent_check](../../../bench/agent_check/README.md)); this folder holds only their IDs
(in [agent/o4oc0/attempts.csv](agent/o4oc0/attempts.csv) and [agent/fnoc0/attempts.csv](agent/fnoc0/attempts.csv)). The tables summarize this project's own runs and are
published under the repository's licence.
