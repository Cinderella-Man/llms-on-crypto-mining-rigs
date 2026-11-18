# R203: Ornith's agent check at stock and with the second rig's clock offsets, on one boot (Q1592, 2026-11-18)

**In plain words.** This measures the speed that counts, a coding agent's, with the memory and core offsets that held
on the second rig ([R201](../2026-11-18_r201_soak_ladder/README.md#the-second-boot-the-use-profile-held-and-ornith-generated-442--faster),
Q1587). On one fresh boot it runs Ornith's agent check twice: first with every card at stock, then, after the use
profile is applied, the same run again. **The stock half has run: 12 of 12 tasks passed, 2.13 minutes a task at equal
work, generation 16.42 tok/s.** That is where the released v4.0 preset stood on the same rig two days earlier (R166:
2.11 and 2.13 minutes, 16.52 and 16.13 tok/s). The offset half, the comparison this row is for, comes next on the same
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

## Provenance and licence

The tasks are the agent check's 12 bug fixes, derived from Exercism's Python and JavaScript exercises (MIT; source,
licences and commits in [bench/agent_check](../../../bench/agent_check/README.md)); this folder holds only their IDs
(in [agent/o4oc0/attempts.csv](agent/o4oc0/attempts.csv)). The tables summarize this project's own runs and are
published under the repository's licence.
