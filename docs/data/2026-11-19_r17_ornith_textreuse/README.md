# Text reuse in Ornith's v5 preset: four alternated agent checks on the second rig (R17's Ornith arm, Q260; 2026-11-19)

**In plain words.** On a coding agent's later turns the server sometimes reads the agent's previous reply again as
prompt: the model wrote a word in other tokens than its tokenizer would choose, the agent sends the reply back as text,
the server splits it its own way, and its cache match stops inside the reply. Text reuse (`LLAMA_SERVER_TEXT_REUSE=1`)
keeps the cached tokens wherever the new prompt spells the same text, so the reply is not read again. In Flint's two
runs it removed every such re-read, a gain on the work and neutral on time
([R17a and R17b](../2026-11-17_r17_flint_textreuse/README.md)), and it is in Flint's and Flash-Next's v5 presets. This
folder decides it for Ornith: four agent checks on one boot of the second rig, every card at its soaked clock profile,
alternating Ornith's v5 preset as it is (control) and the same plus text reuse, so the two settings share the boot, the
clocks and the hour.

**Control run a (R17c, `o5t0a`, 2026-11-19 18:21–18:58): 12 of 12 tasks passed, 1.87 minutes a task at equal work,
19.12 tok/s generation, and the server read two replies again: 5,321 tokens, 29.9 of the run's 370.4 seconds of
prompt reading.** As context only, the same setting on the same rig on the morning's boot (R209e's `o5oc1`): 1.90
minutes a task, 19.26 tok/s, and 2,078 reply tokens read again in 3 turns (23.4 of 370.6 s).

**Text reuse run a (R17d, `o5t1a`, 2026-11-19 19:07–19:40): 12 of 12 tasks passed, 1.89 minutes a task at equal work,
19.41 tok/s generation, and no reply was read again: 0 tokens, 0.0 of 360.4 seconds of prompt reading.** The switch
kept 2,020, 2,110, 2,225 and 2,351 cached tokens on four later turns of the two tasks whose replies control run a read
again (Python `scale-generator` and `pig-latin`), and those turns read 28, 52, 32 and 36 new tokens. Prompt tokens
read fell from 3,407 to 3,080 a task. On the measures of time this one run sits beside control run a (1.89 against
1.87 minutes a task at equal work; 19.43 against 19.45 tok/s at the runs' shared mix of contexts). No class before the
fourth run.

## The runs

R17's Ornith arm runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md):
12 bug-fix tasks, Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget a run) on
Ornith-1.5-122B Pascal N2P GGUF (valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`) at 49,152 tokens on nine
P106-090 cards, on the v5 engine (`pascal-p106-v5` at `602691153`, upstream b12483 merged) with Ornith's v5 preset
([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)). The four runs differ only in the arm's name
and one variable:

| Run | Arm | Server variables added to the v5 preset | Status |
|---|---|---|---|
| R17c, control a | `o5t0a` | none | done 18:21–18:58, 12/12 |
| R17d, text reuse a | `o5t1a` | `LLAMA_SERVER_TEXT_REUSE=1` | done 19:07–19:40, 12/12 |
| R17e, control b | `o5t0b` | none | to run |
| R17f, text reuse b | `o5t1b` | `LLAMA_SERVER_TEXT_REUSE=1` | to run |

The class comes after the fourth run, two runs against two by
[class_arms.py](../../../bench/agent_check/class_arms.py), as for Flint: a measure is a gain or a loss only where the
difference of the two arms' means is beyond the larger of their spreads. Beside them, as context, R209e's `o5oc1`: the
same rig, clock profile, engine, preset and started command on the morning's boot
([R209](../2026-11-19_r209_oc_v5/README.md#ornith-with-the-offsets-o5oc1-2026-11-19)).

## Control run a: `o5t0a` (R17c, 2026-11-19)

| | R209e `o5oc1` (context: the same setting, the morning's boot) | **R17c `o5t0a`** | change |
|---|---|---|---|
| passed | 12/12 | **12/12** | |
| minutes a task at equal work (`v3ka`'s work) | 1.90 | **1.87** | −1.58 % |
| generation tok/s, session | 19.26 | **19.12** | −0.73 % |
| ms a decode step at the two runs' shared mix of contexts; tok/s at it | 52.37; 19.10 | **52.00; 19.23** | −0.71 % |
| prompt tok/s, session | 101.6 | **110.4** | +8.66 % (fewer short reads) |
| the 22 requests read on the same prompt in both, prompt s | 211.80 | **210.42** | −0.65 % |
| of them the 12 first turns, prompt s | 110.65 | **109.19** | −1.32 % |
| first token median; p90 s | 5.37; 10.04 | **7.03; 10.60** | +30.91 %; +5.58 % (the turn mix) |
| first token on a task's first turn, median s | 7.94 | **7.95** | |
| prompt tokens read (a task) | 37,657 (3,138) | **40,888 (3,407)** | |
| reply tokens re-read (turns; their prompt s of all prompt s) | 2,078 (3; 23.4 of 370.6) | **5,321 (2; 29.9 of 370.4)** | |
| reply tokens re-read a task; their prompt s a task | 173; 1.95 | **443; 2.49** | |
| longest first-token wait s (task, turn) | 26.88 (Python `affine-cipher`, 1) | **26.47 (Python `affine-cipher`, 1)** | |
| tokens generated (a task) | 20,222 (1,685) | **20,554 (1,713)** | |
| minutes, all 12 tasks; turns | 24.1; 67 | **24.5; 58** | |
| tool calls / failed / malformed | 67 / 1 / 0 | **58 / 1 / 0** | |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 14 / 0 / 0; 0; 0 in 67 | **13 / 0 / 0; 0; 0 in 58** | |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | **the same on every card (2,127.0)** | |
| hottest card; highest power | 45 °C (card 0); 89.80 W (card 7) | **47 °C (card 0); 85.04 W (card 6)** | |
| watchdog trips; GPU Xid | 0; 0 | **0; 0** | |
| lowest MemAvailable MiB; swap MiB during the load | 1,183; 52 → 362 | **1,233; 52 → 362** | |

- **The two re-read turns** ([work.md](work.md)): Python `pig-latin` turn 4 (1,441 reply tokens, 9.87 s of prompt) and
  Python `scale-generator` turn 5 (3,880 reply tokens, 20.07 s, a first token after 20.17 s). In both the cache match
  stopped 4 tokens before the previous prompt's end, so the whole reply was read again. 44 of the 46 later turns found
  the whole previous prompt and reply in the cache (median hit 94.2 %, [o5t0a_gonogo.md](o5t0a_gonogo.md)).
- **The longest wait was not a re-read:** Python `affine-cipher`'s first turn reads a 2,742-token prompt with nothing
  cached, 26.47 s here and 26.88 s in `o5oc1`; text reuse cannot shorten it.
- **Like for like the speed is `o5oc1`'s:** a decode step at the shared mix 0.71 % shorter, the 22 same-prompt
  requests 0.65 % faster ([decode.md](decode.md), [same_prompts.md](same_prompts.md)), every card's busy SM clock
  median the same to the MHz ([clocks.tsv](clocks.tsv)). Session prompts look faster (110.4 against 101.6 tok/s) only
  through the mix: 33 of 58 requests read fewer than 512 new tokens, against 43 of 67 ([reallocs.tsv](reallocs.tsv)).
  The first-token median is higher for the same reason: 46 later turns against 55.
- **Answers sound:** every task passed, also its hidden tests, test files untouched; 0 of 58 calls malformed, none
  inside the reasoning; the one failed call (Python `affine-cipher`) is one the tool answered with an error. No NaN or
  inf, no watchdog kill. The report's gates: all GO except memory (NO-GO on the swap growth during the load, as for
  every Ornith run of this preset).
- **Heat and power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): above 80 W in 3 4 4 1 5 5 4 4 2
  single samples on cards 0 to 8; the clock watchdog (70 °C, 80 W for 5 s) never tripped and the boot logged no GPU
  Xid.

The same-prompt and shared-mix figures in this section are those of `o5oc1` and `o5t0a` alone (22 shared requests,
five context buckets). From text reuse run a on, [same_prompts.md](same_prompts.md) and [decode.md](decode.md) compare
every run so far over what all of them share (21 requests and four buckets with three runs).

## Text reuse run a: `o5t1a` (R17d, 2026-11-19)

| | R17c `o5t0a` (control a) | **R17d `o5t1a`** | change |
|---|---|---|---|
| passed | 12/12 | **12/12** | |
| minutes a task at equal work (`v3ka`'s work) | 1.87 | **1.89** | +1.07 % |
| generation tok/s, session | 19.12 | **19.41** | +1.52 % |
| ms a decode step at the three runs' shared mix of contexts; tok/s at it | 51.42; 19.45 | **51.47; 19.43** | +0.10 % |
| prompt tok/s, session | 110.4 | **102.5** | −7.16 % (the mix) |
| the 21 requests read on the same prompt in all three runs, prompt s | 199.77 | **201.05** | +0.64 % |
| of them the 12 first turns, prompt s | 109.19 | **110.39** | +1.10 % |
| first token median; p90 s | 7.03; 10.60 | **6.13; 9.58** | −12.80 %; −9.62 % |
| first token on a task's first turn, median s | 7.95 | **8.00** | +0.63 % |
| prompt tokens read (a task) | 40,888 (3,407) | **36,956 (3,080)** | −9.62 % |
| reply tokens re-read (turns; their prompt s of all prompt s) | 5,321 (2; 29.9 of 370.4) | **0 (0; 0.0 of 360.4)** | −100 % |
| reply tokens re-read a task; their prompt s a task | 443; 2.49 | **0; 0.00** | |
| turns where text reuse kept cached tokens | | **4** | |
| longest first-token wait s (task, turn) | 26.47 (Python `affine-cipher`, 1) | **26.70 (Python `affine-cipher`, 1)** | |
| tokens generated (a task) | 20,554 (1,713) | **17,458 (1,455)** | −15.06 % |
| minutes, all 12 tasks; turns | 24.5; 58 | **21.4; 64** | |
| tool calls / failed / malformed | 58 / 1 / 0 | **64 / 2 / 0** | |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 13 / 0 / 0; 0; 0 in 58 | **11 / 0 / 0; 0; 0 in 64** | |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | **2113 2138 2138 2088 2100 2100 2151 2151 2126 (2,122.8)** | cards 2, 3, 4 one step lower |
| hottest card; highest power | 47 °C (card 0); 85.04 W (card 6) | **47 °C (card 0); 85.84 W (card 7)** | |
| watchdog trips; GPU Xid | 0; 0 | **0; 0** | |
| lowest MemAvailable MiB; swap MiB during the load | 1,233; 52 → 362 | **1,189; 53 → 363** | |

- **The switch's four turns** (the server's log line `prompt ids differ from the cache at <n>, same text: <k> more
  cached tokens kept`): Python `scale-generator` turns 4 and 5 (ids differ at token 5,511; 2,020 and 2,110 more tokens
  kept; 28 and 52 new tokens read in 1.09 and 1.63 s; first token after 1.18 and 1.70 s) and Python `pig-latin` turns
  4 and 5 (at 3,798; 2,225 and 2,351 kept; 32 and 36 read in 1.19 and 1.22 s; first token after 1.27 and 1.28 s). In
  each pair the second turn meets the same point again: the cache holds the model's own tokens for the earlier reply,
  and the agent sends that reply back as the same text in other tokens. These are the two tasks whose replies control
  run a read again (`pig-latin` turn 4, 1,441 tokens; `scale-generator` turn 5, 3,880 tokens). All 52 later turns
  found the whole previous prompt and reply in the cache (median hit 96.9 %, [o5t1a_gonogo.md](o5t1a_gonogo.md)).
- **Fewer tokens, not faster ones:** the run generated 17,458 tokens against 20,554, so the 12 tasks took 21.4 minutes
  against 24.5; at equal work (each run scaled to `v3ka`'s tokens) 1.89 against 1.87 minutes a task. Like for like the
  decode step at the shared mix is 51.47 against 51.42 ms and the 21 same-prompt requests 201.05 against 199.77 s; for
  scale, the same setting on two boots (`o5oc1` and `o5t0a`) differed by 0.71 % and 0.65 % on those two measures. The
  class waits for the fourth run.
- **First tokens:** the median and p90 are lower (6.13 and 9.58 s against 7.03 and 10.60): no turn read 1,441 or 3,880
  reply tokens, and the turn mix differs (52 later turns against 46). The longest wait is again Python
  `affine-cipher`'s first turn (2,742 prompt tokens, nothing cached), 26.70 s; text reuse cannot shorten it.
- **Answers sound:** every task passed, also its hidden tests, test files untouched; 0 of 64 calls malformed, none
  inside the reasoning; the two failed calls (Python `affine-cipher`) are calls the tool answered with an error. No
  NaN or inf, no watchdog kill. The report's gates: all GO except memory (NO-GO on the swap growth during the load).
- **Heat and power** ([cards.md](cards.md)): above 80 W in 1 3 4 2 3 0 2 3 2 single samples on cards 0 to 8; the
  peak temperature 1 °C above control a's on cards 1 to 8 and the same on card 0; the clock watchdog never tripped and
  the boot logged no GPU Xid. The busy SM clock median is control a's on six cards and one driver step lower on cards 2,
  3 and 4 ([clocks.tsv](clocks.tsv)).
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,189 MiB (19:40:37, as the run ended); each card's
  peak memory 22 18 22 18 18 18 18 14 14 MiB below control a's on cards 0 to 8, with a longest context of 7,916 tokens
  against 10,752.

## How it ran

- **The rig:** the second rig, nine P106-090, booted at 14:04 and set to its soaked clock profile at 14:13 by the
  offset kit's `apply` ([R211](../2026-11-19_r209_oc_v5/README.md#the-profile-again-on-the-boot-of-1404)): memory
  +1,300 +1,000 +1,000 +1,100 +1,100 +1,200 +900 +900 +1,300 MT/s and core +256 +256 +269 +231 +206 +206 +282 +256 +231
  MHz on cards 0 to 8, the 75 W power limit, the kit's watchdog running. Before and after each run every card read back
  at those values, no load process, 0 Xid, 0 step-downs. Nothing was set or reset for these runs.
- **The build:** the v5 engine `602691153` (`pascal-p106-v5`, upstream b12483 merged), the rig's tarball
  `llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483` (`llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`,
  checked before and after).
- **The started command** (549 bytes with its newline, md5 `a0a97e752b2acbfef50bbaf86250e1e5`) is R209e's `o5oc1`
  byte for byte, and the server's `switches:` line the same after `switches:` (7 set, the v5 preset's):

  ```
  GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
    GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2
  llama-server -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --temp 0.6 -b 4096 -ub 512
    -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall
  ```

  The text-reuse runs' command, dry-run through the same launcher with `EXTRA_ENV=LLAMA_SERVER_TEXT_REUSE=1` (no load:
  the rig alias did not resolve, so the harness printed it and stopped), is this one with ` LLAMA_SERVER_TEXT_REUSE=1`
  after `GGML_CUDA_MMID_GLU_UNFUSE=2`, 575 bytes, nothing else changed (`cmp`). Text reuse run a started exactly that
  line (575 bytes, md5 `67aae3dba6542ab62ac9f096e24d3ae3`, `cmp` against the dry run), and its server logged
  `switches: 8 set`, the preset's seven and `LLAMA_SERVER_TEXT_REUSE=1`.
- **The client** ran on the machine that drives the rig (`CLIENT=local`) in the sandbox image `agent-check:0.87.1` at
  `--cpu-shares 256`, with the bench files R209's runs used (the agent check as of C842, 2026-11-14; since then only
  comments changed in the client's code). Each run has its own client folder and container prefix
  (`PREFIX=acr-<run>`), so another check on the same client at the same time cannot remove its containers. The work
  tables come from v5's [report.py](../../../bench/agent_check/report.py) `work` (C852) on the published rows.
- **Timeline:** control run a's server started at 18:21:56 and the client reached it at 18:32:54 (658 s; R209e:
  661 s); the 12 tasks ran 18:32:54–18:57:39; the rig's logs went to the client at 18:57:47. Text reuse run a's server
  started at 19:07:54 and was up after 653 s; the client reached it at 19:18:49, the 12 tasks ran 19:18:49–19:40:25 and
  the logs went to the client at 19:40:43. Nothing else ran on the rig.

## Commands

The run, from the repository's root on the machine that drives the rig:

```bash
RIG=<rig alias> CLIENT=local ARM=o5t0a EXTRA_ENV= PREFIX=acr-r17c \
  MODEL=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN=<rig path>/llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r17c OUT=<out>/agent-o5t0a \
  bash docs/data/2026-11-19_r17_ornith_textreuse/run/ornith_switch.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
# a text-reuse run: ARM=o5t1a EXTRA_ENV=LLAMA_SERVER_TEXT_REUSE=1, its own rig folder, client folder and PREFIX
```

The tables, with Python 3 (no packages; a few seconds), in a folder holding `res/turns.csv` and `res/attempts.csv`
with the rows of `v3ka` and `o5oc1` (R209's tables folder: `v3ka`'s rows as [R203](../2026-11-18_r203_ornith_offsets/README.md)
published them, `o5oc1`'s = [R209's agent/o5oc1/](../2026-11-19_r209_oc_v5/agent/o5oc1/)) and each run's rows
([agent/o5t0a/](agent/o5t0a/), [agent/o5t1a/](agent/o5t1a/)), each run's server log as `rig/server-<arm>.log`, and in
`rig_<arm>/` its `meminfo.log`, `cards.csv` and `window` (`o5oc1`'s from R209's tables; from the driver's log, `o5t0a`'s
window `2026-11-19 18:32:54|2026-11-19 18:57:39` and `o5t1a`'s `2026-11-19 19:18:49|2026-11-19 19:40:25`):

```bash
bash docs/data/2026-11-19_r17_ornith_textreuse/run/tables.sh <tables folder> o5t0a o5t1a
```

The server's text-reuse lines: `grep 'same text:' rig/server-o5t1a.log` (each names the token where the prompt's ids
leave the cache's and how many more cached tokens it kept; the turn is the one whose prompt has that many tokens).

## Files

| File | Content |
|---|---|
| [agent.tsv](agent.tsv) | per run (`v3ka`, `o5oc1`, `o5t0a`, `o5t1a`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes, minutes a task at equal work against `v3ka`, then the work: prompt tokens read a task, reply tokens re-read a task and their prompt s a task, the re-read turns, the longest first-token wait |
| [work.md](work.md), [work.tsv](work.tsv) | `report.py work` on `o5oc1`, `o5t0a` and `o5t1a`: prompt tokens read, tokens generated, reply tokens re-read and their prompt seconds, the longest first-token wait; then every turn that read a reply again |
| [same_prompts.md](same_prompts.md) | the requests read on the same prompt in every run (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [decode.md](decode.md) | each run by the context at a request's middle (requests, tokens, decode s, tok/s, ms a token), and at their shared mix of contexts |
| [reallocs.tsv](reallocs.tsv), [o5t0a_requests_reallocs.tsv](o5t0a_requests_reallocs.tsv), [o5t1a_requests_reallocs.tsv](o5t1a_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, and per request |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, and each card's peak memory |
| [clocks.tsv](clocks.tsv), [cards.md](cards.md) | each card's busy SM clock median; its highest temperature and power, samples above 80 W and busy memory clock median, over each run's tasks |
| [o5t0a_compare.md](o5t0a_compare.md), [o5t0a_report_table.md](o5t0a_report_table.md), [o5t0a_gonogo.md](o5t0a_gonogo.md); the same for `o5t1a` ([o5t1a_compare.md](o5t1a_compare.md), [o5t1a_report_table.md](o5t1a_report_table.md), [o5t1a_gonogo.md](o5t1a_gonogo.md)) | the harness's report of each run: per task, per turn and the go/no-go |
| [agent/o5t0a/](agent/o5t0a/), [agent/o5t1a/](agent/o5t1a/) | per request (`turns.csv`) and per attempt (`attempts.csv`) rows of each run |
| [run/ornith_switch.sh](run/ornith_switch.sh), [run/tables.sh](run/tables.sh) | the launcher of the four runs and the tables |

## Provenance and licence

Measured on the project's own rig on 2026-11-19 with the repository's own scripts and Miner Llama's v5 engine (MIT, like
llama.cpp); the agent check's tasks are [bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are
in `agent/<arm>/attempts.csv`); the tables hold times, counts and task ids, no text. The tables and scripts here are
under the repository's licence.
