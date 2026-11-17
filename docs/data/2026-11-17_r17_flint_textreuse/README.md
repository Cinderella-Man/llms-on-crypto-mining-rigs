# Text reuse in Flint's v4 preset: the agent check on v4.0, run a (R17, Q260; 2026-11-17)

R17 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix
tasks, Python and JavaScript alternating, a 12-minute cap a task, a 40-minute budget an arm) on Flint Pascal Q8 GGUF
(p065, `flint-near-q8-p065.gguf`, 40,431,236,544 bytes) at 262,144 tokens on nine P106-090 cards, with Flint's released
v4 preset and one variable added: `LLAMA_SERVER_TEXT_REUSE=1`. Arm `f4tr`; this folder holds its first run, R17a.

**Why:** on v4.0 a reply is sometimes read again as prompt on the next turn. Flint samples at temperature 1.0 and now
and then writes a word in token ids its tokenizer would not choose; the client sends the reply back as text, the
tokenizer splits it its own way, and the server's cache match stops inside the reply, so the server restores the
checkpoint at the previous prompt's end and reads the whole reply again. In the release check's two runs of this
preset (R159e `f4ta`, R159f `f4tb`), those turns took 67.6 of 222.6 s and 141.7 of 299.5 s of prompt time
([2026-11-17_c852_agent_work_counts](../2026-11-17_c852_agent_work_counts/README.md)). `LLAMA_SERVER_TEXT_REUSE=1`
keeps the cached ids wherever the new prompt spells the same text, so the reply is not read again (Q260,
[2026-10-28_reply_rereads](../2026-10-28_reply_rereads/README.md)). The switch is in v4.0's build, off by default.

Registry row Q260 in [quants/experiments.md](../../../quants/experiments.md). **This is one run, given by value against
the release check's two runs of the preset without the switch, and not classed:** a class needs the arm's further
runs, starting with run b.

## The result

| | R17a `f4tr` (text reuse) | R159e `f4ta` | R159f `f4tb` |
|---|---|---|---|
| passed | **12 / 12** | 11 / 12 | 12 / 12 |
| minutes a task at equal work (`v3base`'s work) | **1.81** | 1.72 | 1.81 |
| minutes, all 12 tasks (wall) | **26.7** | 28.8 | 23.3 |
| minutes a task, median | **1.30** | 1.10 | 1.13 |
| turns | **77** | 73 | 71 |
| generation tok/s | **46.74** | 49.32 | 46.99 |
| prompt tok/s | **210.0** | 219.7 | 204.1 |
| first token median, p90 s | **1.76, 5.31** | 1.74, 6.61 | 1.82, 6.35 |
| first token on a task's first turn, median s | **3.21** | 3.24 | 3.18 |
| prompt tokens read (a task) | **35,635 (2,970)** | 48,907 (4,076) | 61,112 (5,093) |
| reply tokens re-read (turns; their prompt s of all prompt s) | **0 (0; 0.0 of 169.7)** | 14,632 (4; 67.6 of 222.6) | 24,755 (5; 141.7 of 299.5) |
| longest first-token wait s (task, turn) | **11.68 (JavaScript `complex-numbers`, 2)** | 24.70 (Python `affine-cipher`, 3, a re-read) | 93.88 (Python `scale-generator`, 4, a re-read) |
| tokens generated (a task) | **65,384 (5,449)** | 37,846 (3,154) | 48,656 (4,055) |
| tool calls / failed / malformed | **71 / 0 / 0** | 68 / 3 / 0 | 68 / 1 / 0 |
| graph reallocations (target) | **200, 0 changed, in 76 of 77 requests** | 250 in 71 of 72 | 290 in 70 of 71 |
| lowest MemAvailable MiB; swap during the load | **1,081; 27 → 293** | 1,084; 30 → 296 | 1,100; 30 → 299 |
| card peaks MiB, cards 0 to 8 | **5,433 5,979 5,129 5,639 5,979 5,129 5,593 5,349 4,853** | the same to the MiB | 5,439 5,985 5,133 5,643 5,985 5,133 5,597 5,355 4,859 |

- **No turn read a reply again.** The server read 0 reply tokens a second time, against 14,632 in 4 turns and 24,755
  in 5 turns in the two runs without the switch ([work.md](work.md)). Its prompt time was 169.7 s for the 12 tasks
  (14.1 s a task) against 222.6 and 299.5 s (18.6 and 25.0 s a task). Every turn from the second on found the whole
  previous prompt and reply in the cache: the go/no-go's cache paths are 65 exact, 0 older checkpoint, 0 reply
  re-read, with a median hit of 95.4 % from the second turn ([f4tr_gonogo.md](f4tr_gonogo.md)).
- **The longest wait for a first token was 11.68 s,** on a turn that was not a re-read, against 24.70 and 93.88 s,
  both re-reads. The p90 was 5.31 s against 6.61 and 6.35; the median 1.76 s against 1.74 and 1.82.
- **Equal work does not show the switch.** "Minutes a task at equal work" prices `v3base`'s token counts at each run's
  speeds, so it holds the tokens fixed: 1.81 against 1.72 and 1.81. The work counts above are the measure the switch
  changes (C852, Q1521).
- **This run generated more by chance.** It wrote 65,384 tokens against 37,846 and 48,656 (decode 1,399 s against 767
  and 1,035): the agent sends no seed, so the same task ends in a short tool call or a long reply by chance (Q1488).
  Most of the difference is two tasks: Python `scale-generator` (28,294 tokens in 8 turns, 645 s of decode, context to
  32,424 tokens, passed in 667.5 s, 52.5 s under the cap) and Python `list-ops` (11,036 tokens, 217 s). So its wall
  minutes (26.7) sit between the other two runs' (28.8 with one capped task, 23.3).
- **Prompt tok/s is a mix of sizes.** The re-reads were long prompts read at high speed; without them this run's
  prompts are shorter: 54 of its 77 requests read under 512 tokens and 2 read 2,000 or more, against 48 and 5, and 43
  and 5 ([prompt_by_size.tsv](prompt_by_size.tsv)). By size: under 512 tokens 173.4 tok/s (172.7, 183.8), 512–1,023
  216.2 (220.3, 217.5), 1,024–1,999 225.1 (236.9, 230.7), 2,000 and more 267.3 in 2 requests (236.2 and 196.0 in 5
  each).
- **Without Python `scale-generator`** ([gen_without_scale.tsv](gen_without_scale.tsv); its long replies at deep
  context count only where they were timed): generation 49.23 tok/s against 49.35 and 47.80, prompts 214.1 against
  218.8 and 221.2.
- **Checkpoints:** 11 restores from the cards, each in 2.1 or 2.2 ms; none created on the host or erased, no full
  re-processing, no NaN or inf, no watchdog kill. The load printed the same plans as the release runs (checkpoints on
  the devices, the newest 4 of each slot; the micro-batch taper `short=600:64`) and fitted the first time.
- **Graph reallocations** ([f4tr_reallocs.tsv](f4tr_reallocs.tsv), per request in
  [f4tr_requests_reallocs.tsv](f4tr_requests_reallocs.tsv)), the same size in every case, none changed: under 512
  prompt tokens 93 in 54 of 54 requests, 512–1,023 49 in 13 of 13, 1,024–1,999 47 in 8 of 8, 2,000 and more 11 in 1
  of 2.
- **Host and cards** ([host.tsv](host.tsv)): MemAvailable lowest 1,081 MiB at 10:45:48, during the last task, and
  1,375 MiB at the server's stop; swap grew from 27 to 293 MiB during the load, as in every run of this preset. The
  cards' peaks are R159e's to the MiB.
- **Go/no-go** ([f4tr_gonogo.md](f4tr_gonogo.md)): infrastructure, proxy rows, first token against prefill, tool
  calls (0 malformed of 71) and cache GO; memory NO-GO on the swap growth during the load (the gate allows 16 MiB),
  with MemAvailable above its 600 MB floor, as in every release run.

## How it ran

- **The build:** Miner Llama v4.0 (tag `miner-llama-v4.0`, `pascal-p106-v4` `eb82916d3`, upstream b12360), the rig
  tarball the release tests ran (`llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`). The switch is in this source
  (`server_text_reuse_enabled` in `tools/server/server-context.cpp`) and its name is in the binary's strings.
- **The model:** `flint-near-q8-p065.gguf`, md5 `8387b2e585a7b9cc10f40cb6b4531694`, checked on the rig before the run.
- **The preset:** [release/flint-v4/settings.env](../../../release/flint-v4/settings.env), whose line and switches are
  R159's tuned arm `f4t`, with `LLAMA_SERVER_TEXT_REUSE=1` added to its environment. The started command (535 bytes,
  md5 `d061d86f9f3549eccd68a179ac9e1735`) is R159f's started command (509 bytes, md5
  `85b1ebffdbee2705ba57d648df2bded0`) with ` LLAMA_SERVER_TEXT_REUSE=1` after its five variables, byte for byte (`cmp`):

  ```
  LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_GRAPHS_FORCE=1
    LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_TEXT_REUSE=1
  llama-server -ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --alias flint -b 512 -ub 128
    -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64
  ```

  The server logged `switches: 6 set`, the preset's five and `LLAMA_SERVER_TEXT_REUSE=1`; its process held those six
  variables and these flags plus `-m`, `--host` and `--port`.
- **The client** ran on the machine that drives the rig (`CLIENT=local`), in the sandbox image `agent-check:0.87.1` at
  `--cpu-shares 256`, with public `v5`'s bench files: the release runs' client code (only comments changed since) and
  C852's report, which counts the work. Nothing else ran on the rig during the tasks.
- **Timeline:** the server started at 10:09:26 and was up after 588 s (R159e and R159f: 587 s); the 12 tasks ran
  10:19:18–10:46:11; the rig's logs went to the client at 10:46:24.
- **The tables** were made with the release runs' published per-request rows: `v3base`
  ([agent/flint-p065-v3base](../2026-11-05_release_regression/agent/flint-p065-v3base/), arm `rel` renamed `v3p065`),
  `f4ta` and `f4tb` ([R159's agent folder](../2026-11-14_r159_flint_release_agent/agent/)) beside `f4tr`'s collected
  rows ([run/merge_rows.py](run/merge_rows.py)). `agent_tables.py` on that folder gives the `v3p065`, `f4ta` and
  `f4tb` rows of [R159's agent.tsv](../2026-11-14_r159_flint_release_agent/agent.tsv) unchanged.

## Commands

The run, from the repository's root on the machine that drives the rig ([run/r17.sh](run/r17.sh) checks the preset's
line and switches first):

```bash
RIG=<rig alias> CLIENT=local MODEL=<rig path>/flint-near-q8-p065.gguf \
  BIN=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8292 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r17 OUT=<out>/agent-f4tr \
  bash docs/data/2026-11-17_r17_flint_textreuse/run/r17.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
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

The tables, with Python 3 (no packages; they ran on the rig after the run), in a folder holding `v3base`'s
`turns.csv` and `attempts.csv` in `in/v3base/`, R159e's and R159f's in `in/f4ta/` and `in/f4tb/`, the collected
`res/` in `in/col/`, and the rig's `server-f4tr.log`, `meminfo.log` and `gpu.csv` in `rig/`:

```bash
python3 -I merge_rows.py . v3p065=in/v3base:rel f4ta=in/f4ta f4tb=in/f4tb f4tr=in/col
python3 -I agent_tables.py . v3p065 f4ta f4tb f4tr                         # agent.tsv
python3 -I realloc_counts.py f4tr=rig/server-f4tr.log > f4tr_reallocs.tsv
python3 -I realloc_counts.py --requests f4tr=rig/server-f4tr.log > f4tr_requests_reallocs.tsv
python3 -I host_summary.py rig R17a f4tr                                   # host.tsv's R17a row
python3 -I prompt_by_size.py . f4ta f4tb f4tr > prompt_by_size.tsv
python3 -I gen_without_task.py . bugfix-python-scale-generator f4ta f4tb f4tr > gen_without_scale.tsv
python3 -I bench/agent_check/report.py work f4ta=agent/f4ta/turns.csv f4tb=agent/f4tb/turns.csv \
  f4tr=agent/f4tr/turns.csv --tsv work.tsv > work.md
```

`agent_tables.py` is [docs/data/2026-11-05_keepalive/run/agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py),
`realloc_counts.py` [scripts/realloc_counts.py](../../../scripts/realloc_counts.py), `host_summary.py` and
`prompt_by_size.py` are in [R159's run folder](../2026-11-14_r159_flint_release_agent/run/), `gen_without_task.py` in
[R170's](../2026-11-16_r170_flint_q5/run/gen_without_task.py). The R159 rows of [host.tsv](host.tsv) are copied from
[R159's](../2026-11-14_r159_flint_release_agent/host.tsv).

## Files

| File | Content |
|---|---|
| [agent.tsv](agent.tsv) | per run (`v3p065`, `f4ta`, `f4tb`, `f4tr`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes, minutes a task at equal work (`v3base`'s work) |
| [work.md](work.md), [work.tsv](work.tsv) | `report.py work` on `f4ta`, `f4tb` and `f4tr`: prompt tokens read, tokens generated, reply tokens re-read and their prompt seconds, the longest first-token wait; then every turn that read a reply again |
| [prompt_by_size.tsv](prompt_by_size.tsv) | prompt tok/s by prompt size, the three runs |
| [gen_without_scale.tsv](gen_without_scale.tsv) | generation and prompt tok/s per run without Python `scale-generator`, and that task's turns, tokens, speed and deepest context |
| [f4tr_reallocs.tsv](f4tr_reallocs.tsv), [f4tr_requests_reallocs.tsv](f4tr_requests_reallocs.tsv) | graph reallocations by prompt size and per request |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, and each card's peak memory, R159e, R159f and R17a |
| [f4tr_compare.md](f4tr_compare.md), [f4tr_report_table.md](f4tr_report_table.md), [f4tr_gonogo.md](f4tr_gonogo.md) | the harness's report: per task, per turn and the go/no-go |
| [agent/f4tr/](agent/f4tr/) | per request (`turns.csv`) and per attempt (`attempts.csv`) rows |
| [run/r17.sh](run/r17.sh), [run/merge_rows.py](run/merge_rows.py) | the run and the table folder |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), v4.0. Model: [Flint](https://huggingface.co/RecursiveSI/flint) (Apache-2.0)
as my Flint Pascal Q8 GGUF ([models/flint](../../../models/flint/README.md)). The tasks are the agent check's bug-fix set
([bench/agent_check](../../../bench/agent_check/README.md)); the tables hold times, counts and task ids, no text.
Measured on a Manli M-P106L9-N6G (nine P106-090).
