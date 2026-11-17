# Text reuse in Flint's v4 preset: the agent check on v4.0, runs a and b (R17, Q260; 2026-11-17)

R17 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix
tasks, Python and JavaScript alternating, a 12-minute cap a task, a 40-minute budget an arm) on Flint Pascal Q8 GGUF
(p065, `flint-near-q8-p065.gguf`, 40,431,236,544 bytes) at 262,144 tokens on nine P106-090 cards, with Flint's released
v4 preset and one variable added: `LLAMA_SERVER_TEXT_REUSE=1`. Arm `f4tr`; this folder holds its two runs, R17a and
R17b, named `f4tr` and `f4trb` in the tables.

**Why:** on v4.0 a reply is sometimes read again as prompt on the next turn. Flint samples at temperature 1.0 and now
and then writes a word in token ids its tokenizer would not choose; the client sends the reply back as text, the
tokenizer splits it its own way, and the server's cache match stops inside the reply, so the server restores the
checkpoint at the previous prompt's end and reads the whole reply again. In the release check's two runs of this
preset (R159e `f4ta`, R159f `f4tb`), those turns took 67.6 of 222.6 s and 141.7 of 299.5 s of prompt time
([2026-11-17_c852_agent_work_counts](../2026-11-17_c852_agent_work_counts/README.md)). `LLAMA_SERVER_TEXT_REUSE=1`
keeps the cached ids wherever the new prompt spells the same text, so the reply is not read again (Q260,
[2026-10-28_reply_rereads](../2026-10-28_reply_rereads/README.md)). The switch is in v4.0's build, off by default.

Registry row Q260 in [quants/experiments.md](../../../quants/experiments.md). **Classed on two runs against the release
check's two runs of the preset without the switch** ([class.tsv](class.tsv), the rule of
[class_arms.py](../../../bench/agent_check/class_arms.py)): **a gain on the work it removes** (prompt tokens read a
task, reply tokens re-read a task and their prompt seconds) **and negligible on every measure of time.**

## The result

| | R17a `f4tr` (text reuse) | R17b `f4trb` (text reuse) | R159e `f4ta` | R159f `f4tb` |
|---|---|---|---|---|
| passed | **12 / 12** | **12 / 12** | 11 / 12 | 12 / 12 |
| minutes a task at equal work (`v3base`'s work) | **1.81** | **1.74** | 1.72 | 1.81 |
| minutes, all 12 tasks (wall) | **26.7** | **19.1** | 28.8 | 23.3 |
| minutes a task, median | **1.30** | **1.15** | 1.10 | 1.13 |
| turns | **77** | **70** | 73 | 71 |
| generation tok/s | **46.74** | **49.42** | 49.32 | 46.99 |
| prompt tok/s | **210.0** | **207.6** | 219.7 | 204.1 |
| first token median, p90 s | **1.76, 5.31** | **1.81, 6.18** | 1.74, 6.61 | 1.82, 6.35 |
| first token on a task's first turn, median s | **3.21** | **3.26** | 3.24 | 3.18 |
| prompt tokens read (a task) | **35,635 (2,970)** | **35,309 (2,942)** | 48,907 (4,076) | 61,112 (5,093) |
| reply tokens re-read (turns; their prompt s of all prompt s) | **0 (0; 0.0 of 169.7)** | **0 (0; 0.0 of 170.1)** | 14,632 (4; 67.6 of 222.6) | 24,755 (5; 141.7 of 299.5) |
| longest first-token wait s (task, turn) | **11.68 (JavaScript `complex-numbers`, 2)** | **12.57 (JavaScript `complex-numbers`, 2)** | 24.70 (Python `affine-cipher`, 3, a re-read) | 93.88 (Python `scale-generator`, 4, a re-read) |
| tokens generated (a task) | **65,384 (5,449)** | **46,509 (3,876)** | 37,846 (3,154) | 48,656 (4,055) |
| tool calls / failed / malformed | **71 / 0 / 0** | **64 / 1 / 0** | 68 / 3 / 0 | 68 / 1 / 0 |
| graph reallocations (target) | **200, 0 changed, in 76 of 77 requests** | **191, 0 changed, in 69 of 70** | 250 in 71 of 72 | 290 in 70 of 71 |
| lowest MemAvailable MiB; swap during the load | **1,081; 27 → 293** | **1,100; 0 → 296** | 1,084; 30 → 296 | 1,100; 30 → 299 |
| card peaks MiB, cards 0 to 8 | **5,433 5,979 5,129 5,639 5,979 5,129 5,593 5,349 4,853** | **5,437 5,985 5,133 5,643 5,985 5,133 5,597 5,355 4,859** | R17a's to the MiB | 5,439 5,985 5,133 5,643 5,985 5,133 5,597 5,355 4,859 |

## The class: two runs against two

`class_arms.py agent.tsv f4tr=f4tr,f4trb f4t=f4ta,f4tb` ([class.tsv](class.tsv)). A measure is a gain or a loss only
where the difference of the two arms' means is beyond the larger of their spreads (each arm's highest run minus its
lowest); otherwise it is negligible.

| Measure | `f4tr`: R17a, R17b | `f4t`: R159e, R159f | Difference of the means (the larger spread) | Class |
|---|---|---|---|---|
| minutes a task at equal work | 1.81, 1.74 | 1.72, 1.81 | +0.01 (0.09) | negligible |
| generation tok/s | 46.74, 49.42 | 49.32, 46.99 | −0.075 (2.68) | negligible |
| prompt tok/s | 210.0, 207.6 | 219.7, 204.1 | −3.1 (15.6) | negligible |
| first token median s | 1.76, 1.81 | 1.74, 1.82 | +0.005 (0.08) | negligible |
| first token p90 s | 5.31, 6.18 | 6.61, 6.35 | −0.735 (0.87) | negligible |
| first token on a task's first turn, median s | 3.21, 3.26 | 3.24, 3.18 | +0.025 (0.06) | negligible |
| **prompt tokens read a task** | **2,970, 2,942** | 4,076, 5,093 | −1,628, −35.5 % (1,017) | **gain** |
| **reply tokens re-read a task** | **0, 0** | 1,219, 2,063 | −1,641, −100 % (844) | **gain** |
| **prompt s of the turns that re-read a reply, a task** | **0, 0** | 5.63, 11.81 | −8.72, −100 % (6.18) | **gain** |
| longest first-token wait s | 11.68, 12.57 | 24.70, 93.88 | −47.16 (69.18) | negligible |

- **Neither run read a reply again.** The server read 0 reply tokens a second time in the 65 and 58 turns after a
  task's first, against 14,632 in 4 turns and 24,755 in 5 turns in the two runs without the switch ([work.md](work.md)).
  Every one of those turns found the whole previous prompt and reply in the cache: their cache paths are 65 and 58
  exact, 0 older checkpoint, 0 reply re-read, with a median hit of 95.4 and 95.1 % from the second turn
  ([f4tr_gonogo.md](f4tr_gonogo.md), [f4trb_gonogo.md](f4trb_gonogo.md)). The prompt time was 169.7 and 170.1 s for
  the 12 tasks against 222.6 and 299.5 s.
- **Time is negligible on every measure, equal work included** (1.81 and 1.74 minutes a task against 1.72 and 1.81).
  Equal work prices `v3base`'s token counts at each run's speeds, so it holds the tokens fixed and cannot show work
  left out; the work counts above are the measure the switch changes (C852, Q1521; C855, Q1526).
- **R17a's gain on the first-token p90 does not hold on two runs.** R17a alone gave 5.31 s against 6.61 and 6.35, a
  gain on `f4t`'s spread; R17b's 6.18 s widens the arm's own spread to 0.87 s, beyond the difference of the means
  (0.735 s).
- **The longest wait for a first token** was 11.68 and 12.57 s, both on JavaScript `complex-numbers` turn 2 and
  neither a re-read, against 24.70 and 93.88 s, both re-reads. It is negligible by the rule because `f4t`'s two runs
  spread by 69.18 s.
- **Chance and the work counts.** The agent sends no seed, so each run does different work, and two runs an arm are a
  narrow guard for a work count (Q1575, [C855](../2026-11-17_c855_class_work_counts/README.md)). Text reuse has a cause
  that removes the tokens, and the counts agree with it: the seven Flint Pascal Q8 GGUF runs without the switch re-read
  a reply on 41 of their 425 turns after a task's first; at that rate 0 in R17b's 58 such turns has a chance of
  0.28 % (est.), and 0 in both runs' 123 such turns 0.0004 % (est.).
- **Each run generated a different amount by chance** (Q1488): 65,384 and 46,509 tokens against 37,846 and 48,656
  (decode 1,399 and 941 s against 767 and 1,035). Python `scale-generator` wrote 28,294 tokens in 8 turns in R17a
  (645 s of decode, context to 32,424 tokens, passed in 667.5 s) and 13,358 in 6 turns in R17b (277 s, context to
  18,536 tokens, passed in 301.4 s). So the wall minutes run 26.7 and 19.1 against 28.8 (with one capped task) and
  23.3.
- **Prompt tok/s is a mix of sizes.** Without re-reads the two runs have fewer long prompts: 2 and 2 of their 77 and
  70 requests read 2,000 tokens or more, against 5 and 5 ([prompt_by_size.tsv](prompt_by_size.tsv)). By size, R17a,
  R17b (R159e, R159f): under 512 tokens 173.4, 164.0 (172.7, 183.8); 512–1,023 216.2, 202.0 (220.3, 217.5);
  1,024–1,999 225.1, 230.9 (236.9, 230.7); 2,000 and more 267.3 and 257.3 in 2 requests each (236.2 and 196.0 in 5
  each).
- **Without Python `scale-generator`** ([gen_without_scale.tsv](gen_without_scale.tsv); its long replies at deep
  context count only where they were timed): generation 49.23 and 49.94 tok/s against 49.35 and 47.80, prompts 214.1
  and 209.3 against 218.8 and 221.2.
- **Checkpoints:** 11 restores from the cards in each run, each in 2.1 or 2.2 ms; none created on the host or erased,
  no full re-processing, no NaN or inf, no watchdog kill. Both loads printed the same plans as the release runs
  (checkpoints on the devices, the newest 4 of each slot; the micro-batch taper `short=600:64`), fitted the first time
  and were up after 588 s (R159e and R159f: 587 s).
- **Graph reallocations** ([f4tr_reallocs.tsv](f4tr_reallocs.tsv), [f4trb_reallocs.tsv](f4trb_reallocs.tsv), per
  request in [f4tr_requests_reallocs.tsv](f4tr_requests_reallocs.tsv) and
  [f4trb_requests_reallocs.tsv](f4trb_requests_reallocs.tsv)), the same size in every case, none changed. R17a, R17b:
  under 512 prompt tokens 93 in 54 of 54 requests, 78 in 46 of 46; 512–1,023 49 in 13 of 13, 43 in 12 of 12;
  1,024–1,999 47 in 8 of 8, 58 in 10 of 10; 2,000 and more 11 in 1 of 2, 12 in 1 of 2.
- **Host and cards** ([host.tsv](host.tsv)): MemAvailable lowest 1,081 MiB (R17a, 10:45:48) and 1,100 MiB (R17b,
  16:23:09), both during the last task; 1,375 and 1,374 MiB at the server's stop. Swap grew during the load from 27 to
  293 MiB and from 0 to 296 MiB (R17b started seven minutes after the rig's restart), as in every run of this preset. R17a's card peaks are R159e's to the MiB; R17b's are R159f's on cards 1 to 8, and 5,437 against 5,439
  MiB on card 0.
- **Go/no-go** ([f4tr_gonogo.md](f4tr_gonogo.md), [f4trb_gonogo.md](f4trb_gonogo.md)): infrastructure, proxy rows,
  first token against prefill, tool calls (0 malformed of 71 and of 64) and cache GO; memory NO-GO on the swap growth
  during the load (the gate allows 16 MiB), with MemAvailable above its 600 MB floor, as in every release run. R17b's
  one failed tool call (JavaScript `phone-number`) is a call the tool answered with an error, not a malformed one.

## How it ran

- **The build:** Miner Llama v4.0 (tag `miner-llama-v4.0`, `pascal-p106-v4` `eb82916d3`, upstream b12360), the rig
  tarball the release tests ran (`llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`, checked before each run). The
  switch is in this source (`server_text_reuse_enabled` in `tools/server/server-context.cpp`) and its name is in the
  binary's strings.
- **The model:** `flint-near-q8-p065.gguf`, md5 `8387b2e585a7b9cc10f40cb6b4531694`, checked on the rig before each
  run.
- **The preset:** [release/flint-v4/settings.env](../../../release/flint-v4/settings.env), whose line and switches are
  R159's tuned arm `f4t`, with `LLAMA_SERVER_TEXT_REUSE=1` added to its environment. Both runs' started command (535
  bytes, md5 `d061d86f9f3549eccd68a179ac9e1735`; R17b's is R17a's byte for byte, `cmp`) is R159f's started command
  (509 bytes, md5 `85b1ebffdbee2705ba57d648df2bded0`) with ` LLAMA_SERVER_TEXT_REUSE=1` after its five variables, byte
  for byte (`cmp`):

  ```
  LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_GRAPHS_FORCE=1
    LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_TEXT_REUSE=1
  llama-server -ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --alias flint -b 512 -ub 128
    -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64
  ```

  Each server logged `switches: 6 set`, the preset's five and `LLAMA_SERVER_TEXT_REUSE=1`; its process held those six
  variables and these flags plus `-m`, `--host` and `--port`.
- **The cards at stock:** the rig ran a clock-offset test earlier the same day, so before R17b it was restarted and
  read back: every card's memory and core clock offsets 0 through NVML (36 readings, [bench/mem_offset](../../../bench/mem_offset/)'s
  `nvml_memoc.py probe`, read-only), the 75 W power limit and 1,911 / 4,004 MHz maximum clocks on all nine, no Xid. No
  clock, power or offset setting was changed for either run.
- **The client** ran on the machine that drives the rig (`CLIENT=local`), in the sandbox image `agent-check:0.87.1` at
  `--cpu-shares 256`, with public `v5`'s bench files at `9626fa8`: the release runs' client code (only comments changed
  since) and C852's report, which counts the work. Each run had its own client folder (the same files) and results
  folder. Nothing else ran on the rig during the tasks.
- **Timeline:** R17a's server started at 10:09:26, the 12 tasks ran 10:19:18–10:46:11; R17b's started at 15:56:57,
  the tasks ran 16:06:49–16:26:01, and the rig's logs went to the client at 16:26:09.
- **The tables** were made on the driving machine with the release runs' published per-request rows: `v3base`
  ([agent/flint-p065-v3base](../2026-11-05_release_regression/agent/flint-p065-v3base/), arm `rel` renamed `v3p065`),
  `f4ta` and `f4tb` ([R159's agent folder](../2026-11-14_r159_flint_release_agent/agent/)) beside R17a's published
  rows ([agent/f4tr/](agent/f4tr/)) and R17b's collected rows, arm `f4tr` renamed `f4trb`
  ([run/merge_rows.py](run/merge_rows.py)). `agent.tsv` comes from
  [bench/agent_check/agent_tables.py](../../../bench/agent_check/agent_tables.py) (C855): its first 14 columns are the
  older script's table ([docs/data/2026-11-05_keepalive/run/agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)),
  whose `v3p065`, `f4ta`, `f4tb` and `f4tr` rows are R159's and R17a's published rows unchanged, and its last five
  columns are the work counts the class uses. R17a's other tables come out the same from the published rows.

## Commands

The runs, from the repository's root on the machine that drives the rig ([run/r17.sh](run/r17.sh) checks the preset's
line and switches first). Run b used the same arm with its own ports, rig folder and client folders:

```bash
RIG=<rig alias> CLIENT=local MODEL=<rig path>/flint-near-q8-p065.gguf \
  BIN=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8292 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r17 OUT=<out>/agent-f4tr \
  bash docs/data/2026-11-17_r17_flint_textreuse/run/r17.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
# run b: RIG_LOGDIR=<another rig folder>/agent-logs RIG_PORT=8302 RELAY_PORT=8306 PROXY_PORT=18803
#   CLIENT_DIR=<client folder>/bench-b CLIENT_OUT=<client folder>/r17b, the rest the same
```

That is this call, as it ran:

```bash
. release/flint-v4/settings.env
BENCH=$PWD/bench/agent_check RIG=<rig alias> CLIENT=local CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=256 \
  MODEL=<rig path>/flint-near-q8-p065.gguf RIG_LOGDIR=<rig folder>/agent-logs \
  RIG_PORT=8292 RELAY_PORT=8296 PROXY_PORT=18793 CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r17 \
  REL_BIN=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  REL_ENV="$PRESET_ENV LLAMA_SERVER_TEXT_REUSE=1" \
  ARGS_f4tr="--alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64" \
  BUDGET_MIN_f4tr=40 ARMS=f4tr STEPS=run \
  bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh <out>/agent-f4tr
```

The tables, with Python 3 (no packages; a few seconds), in a folder holding `v3base`'s `turns.csv` and
`attempts.csv` in `in/v3base/`, R159e's and R159f's in `in/f4ta/` and `in/f4tb/`, R17a's in `in/f4tr/`, run b's
collected folder (its `res/`) in `in/colb/`, and run b's `server-f4tr.log`, `meminfo.log` and `gpu.csv` in `rig/`, the
server log also as `rig/server-f4trb.log`:

```bash
python3 -I merge_rows.py . v3p065=in/v3base:rel f4ta=in/f4ta f4tb=in/f4tb f4tr=in/f4tr f4trb=in/colb:f4tr
python3 -I bench/agent_check/agent_tables.py . v3p065 f4ta f4tb f4tr f4trb               # agent.tsv
python3 -I bench/agent_check/class_arms.py agent.tsv f4tr=f4tr,f4trb f4t=f4ta,f4tb > class.tsv
python3 -I bench/agent_check/report.py work f4ta=agent/f4ta/turns.csv f4tb=agent/f4tb/turns.csv \
  f4tr=agent/f4tr/turns.csv f4trb=agent/f4trb/turns.csv --tsv work.tsv > work.md
python3 -I realloc_counts.py f4trb=rig/server-f4trb.log > f4trb_reallocs.tsv
python3 -I realloc_counts.py --requests f4trb=rig/server-f4trb.log > f4trb_requests_reallocs.tsv
python3 -I host_summary.py rig R17b f4tr                                                 # host.tsv's R17b row
python3 -I prompt_by_size.py . f4ta f4tb f4tr f4trb > prompt_by_size.tsv
python3 -I gen_without_task.py . bugfix-python-scale-generator f4ta f4tb f4tr f4trb > gen_without_scale.tsv
```

R17a's `f4tr_reallocs.tsv`, `f4tr_requests_reallocs.tsv` and host row came from its own rig logs the same way.
`realloc_counts.py` is [scripts/realloc_counts.py](../../../scripts/realloc_counts.py), `host_summary.py` and
`prompt_by_size.py` are in [R159's run folder](../2026-11-14_r159_flint_release_agent/run/), `gen_without_task.py` in
[R170's](../2026-11-16_r170_flint_q5/run/gen_without_task.py). The R159 rows of [host.tsv](host.tsv) are copied from
[R159's](../2026-11-14_r159_flint_release_agent/host.tsv).

## Files

| File | Content |
|---|---|
| [agent.tsv](agent.tsv) | per run (`v3p065`, `f4ta`, `f4tb`, `f4tr`, `f4trb`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes, minutes a task at equal work (`v3base`'s work), then the work: prompt tokens read a task, reply tokens re-read a task and their prompt s a task, the re-read turns, the longest first-token wait |
| [class.tsv](class.tsv) | the class of `f4tr` (R17a, R17b) against `f4t` (R159e, R159f) on each measure |
| [work.md](work.md), [work.tsv](work.tsv) | `report.py work` on `f4ta`, `f4tb`, `f4tr` and `f4trb`: prompt tokens read, tokens generated, reply tokens re-read and their prompt seconds, the longest first-token wait; then every turn that read a reply again |
| [prompt_by_size.tsv](prompt_by_size.tsv) | prompt tok/s by prompt size, the four runs |
| [gen_without_scale.tsv](gen_without_scale.tsv) | generation and prompt tok/s per run without Python `scale-generator`, and that task's turns, tokens, speed and deepest context |
| [f4tr_reallocs.tsv](f4tr_reallocs.tsv), [f4tr_requests_reallocs.tsv](f4tr_requests_reallocs.tsv), [f4trb_reallocs.tsv](f4trb_reallocs.tsv), [f4trb_requests_reallocs.tsv](f4trb_requests_reallocs.tsv) | graph reallocations by prompt size and per request, R17a and R17b |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, and each card's peak memory, R159e, R159f, R17a and R17b |
| [f4tr_compare.md](f4tr_compare.md), [f4tr_report_table.md](f4tr_report_table.md), [f4tr_gonogo.md](f4tr_gonogo.md) | R17a's report from the harness: per task, per turn and the go/no-go |
| [f4trb_compare.md](f4trb_compare.md), [f4trb_report_table.md](f4trb_report_table.md), [f4trb_gonogo.md](f4trb_gonogo.md) | R17b's (its arm is named `f4tr` inside them) |
| [agent/f4tr/](agent/f4tr/), [agent/f4trb/](agent/f4trb/) | per request (`turns.csv`) and per attempt (`attempts.csv`) rows, R17a and R17b |
| [run/r17.sh](run/r17.sh), [run/merge_rows.py](run/merge_rows.py) | the run and the table folder |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), v4.0. Model: [Flint](https://huggingface.co/RecursiveSI/flint) (Apache-2.0)
as my Flint Pascal Q8 GGUF ([models/flint](../../../models/flint/README.md)). The tasks are the agent check's bug-fix set
([bench/agent_check](../../../bench/agent_check/README.md)); the tables hold times, counts and task ids, no text.
Measured on a Manli M-P106L9-N6G (nine P106-090).
