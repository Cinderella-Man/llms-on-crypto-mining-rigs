# Ornith's v4 preset without the eager keep-alive stop on the agent check (R169, Q1496; 2026-11-16)

R169 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix
tasks, Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget an arm, the makers' sampling at
temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF (valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`,
48,196,722,592 bytes) at 49,152 tokens on nine P106-090 cards. It uses R166's method and started command
([2026-11-16_r166_ornith_stack_no_taper](../2026-11-16_r166_ornith_stack_no_taper/README.md)). Two runs of one arm
are planned, one load a run: **R169a `o4nea`** (below) and R169b `o4neb` (not yet run).

- `o4ne*` is the released v4 preset ([release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)) less
  `GGML_CUDA_KEEPALIVE_EAGER=1`, and nothing else. It runs on the release candidate `v4-rc2` (`pascal-p106-v4`
  `eb82916d3`, upstream b12360; the rig tree's `llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`). The
  keep-alive itself (`GGML_CUDA_KEEPALIVE=decode`, v3.0's) stays. R166's `o4sn*` runs are the released preset: the
  same command with the eager stop.
- Why: at depth, the preset without the eager stop read all prompt chunks 2.4 % faster and decoded 0.2 % faster, with
  the same greedy replies ([2026-11-16_r167_keepalive_depth](../2026-11-16_r167_keepalive_depth/README.md#r167a-ornith),
  R167a). The deep lane is a diagnostic; the agent check decides. A gain or a negligible result over both runs argues
  for dropping the eager stop from Ornith's preset; a loss keeps it.

Registry row Q1496 in [quants/experiments.md](../../../quants/experiments.md).

**R169a, one run of two (not classed):** 12 of 12 passed, **2.12 minutes a task at equal work**, between R166's two
runs of the released preset (`o4sna` 2.11, `o4snb` 2.13). Generation 16.58 tok/s (16.52, 16.13), prompts 100.2 tok/s
(103.0, 105.9), first token median 6.74 s (6.69, 6.06), p90 10.71 s (11.29, 11.32), a task's first turn 8.30 s (8.43,
8.45). The class waits for R169b.

## The runs

| Run | Arm | Rig | Server started, up | Tasks | Passed | Started command |
|---|---|---|---|---|---|---|
| R169a | `o4nea` | rig 2 | 19:57:22, up after 656 s | 20:08:22–20:34:49 | 12 / 12 | 493 bytes, md5 `d4d0418461573353dc37d7dba8e0fcbb` |

## R169a: `o4nea`, the released preset without the eager stop, first run (2026-11-16)

- **The started command** is R166b's (`o4snb`, 521 bytes, md5 `211f8db2edeacd267705795e4911d8c9`) less
  ` GGML_CUDA_KEEPALIVE_EAGER=1` and nothing else: with that variable put back after `GGML_CUDA_KEEPALIVE=decode` it
  is R166b's byte for byte. The environment is `GGML_CUDA_KEEPALIVE=decode LLAMA_SERVER_YIELD=2
  LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1`. The harness's `COMMON` line follows,
  then `--temp 0.6 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall`,
  with one `-b`, 4096. The server's `switches:` line, read once during the load, says `5 set: GGML_CUDA_GRAPHS_FORCE=1
  GGML_CUDA_HOP_STAGED=1 GGML_CUDA_KEEPALIVE=decode LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SERVER_YIELD=2`: R166's six
  without the eager stop. The harness and the bench are R166b's (`diff` empty). Since R166, the settings file holds the
  staged hops and the prompt graphs in `PRESET_ENV` and the stack's flags in `PRESET_ARGS`, so the environment is
  taken from the file less the one variable, instead of being added to it as R166 did.

**Numbers** (one run each; [agent.tsv](agent.tsv), [host.tsv](host.tsv), [r169_reallocs.tsv](r169_reallocs.tsv);
R166's rows from its [agent.tsv](../2026-11-16_r166_ornith_stack_no_taper/agent.tsv)):

| | released preset `o4sna` (R166a) | released preset `o4snb` (R166b) | **no eager stop `o4nea` (R169a)** |
|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.11 | 2.13 | **2.12** |
| minutes, all 12 tasks (wall) | 29.1 | 36.8 | **26.3** |
| minutes a task, median | 2.15 | 2.34 | **1.96** |
| turns | 63 | 73 | **62** |
| generation tok/s | 16.52 | 16.13 | **16.58** |
| prompt tok/s | 103.0 | 105.9 | **100.2** |
| first token median, p90 s | 6.69, 11.29 | 6.06, 11.32 | **6.74, 10.71** |
| first token on a task's first turn, median s | 8.43 | 8.45 | **8.30** |
| prompt s for the requests under 512 tokens; requests | 116.3; 37 | 144.8; 46 | **126.2; 37** |
| tool calls / failed / malformed | 63 / 2 / 0 | 73 / 2 / 0 | **62 / 1 / 0** |
| graph reallocations | 0 in 63 requests | 0 in 73 | **0 in 62 requests** |
| lowest MemAvailable MiB | 1,225 | 1,228 | **1,213** |
| server up after, s | 539 | 538 | **656** |

- **Minutes a task at equal work** price the same work, `ka`'s (v3.0's 2026-11-06 run on the same rig, file and
  line: 2.52), at each run's own speeds. `o4nea` took 2.12, between R166's two runs of the released preset (2.11 and
  2.13). The class needs R169b: R160's rule compares the means of an arm's two runs against the larger spread.
- **Where the prompt time went** ([o4nea_reallocs.tsv](o4nea_reallocs.tsv), [r169_reallocs.tsv](r169_reallocs.tsv),
  the server's prompt seconds per size of prompt): requests under 512 tokens took 126.2 s for 37 requests, 3.41 s
  each on average (R166a 116.3 s for 37, 3.14 s; R166b 144.8 s for 46, 3.15 s). Prompts of 512–1,023 tokens took
  92.3 s for 11 (93.5 s for 11, 95.0 s for 11), 1,024–1,999 tokens 99.7 s for 10 (101.6 s for 10 in both), 2,000 or
  more 77.0 s for 4 (85.3 s for 5, 126.4 s for 6). All 62 requests: 395.1 s of prompt (396.6 s for 63, 467.8 s for
  73).
- **First token per turn** ([o4nea_compare.md](o4nea_compare.md), the median over tasks): turn 1 8.30 s for 789 new
  prompt tokens, turn 2 9.99 s for 1,452, turn 3 6.38 s for 340, turn 4 3.26 s for 133, turn 5 1.64 s for 44.
- **Where the wall minutes went** ([o4nea_report_table.md](o4nea_report_table.md), the client's attempts over R160's
  six runs, R166's two and R169a): `o4nea`'s attempts spent a mean 96 s a task generating (`o4sna` 110, `o4snb` 143)
  and 33 s reading prompts (33, 39), and processed 3,299 prompt tokens a task (3,405, 4,130). Python
  `scale-generator` took 5.2 minutes in 5 turns (`o4sna` 4.5, `o4snb` 10.8), Python `list-ops` 3.6 in 5, Python
  `affine-cipher` 3.0 in 9 and Python `pig-latin` 2.9 in 6; every other task took 2.3 minutes or less. No task came
  near the 12-minute cap. The wall minutes (26.3 against 29.1 and 36.8) follow the work done, which the minutes a task
  at equal work take out.
- **Graph reallocations** ([o4nea_reallocs.tsv](o4nea_reallocs.tsv),
  [o4nea_requests_reallocs.tsv](o4nea_requests_reallocs.tsv)): 0 in every bucket.
- **Host** ([host.tsv](host.tsv), R169a's own lines of `meminfo.log` and `gpu.csv`, from 19:57:23): lowest
  MemAvailable 1,213 MiB at 20:34:20, during the last task, 3,085 MiB at the end; swap 31 MiB at the start, 337 MiB at
  its highest at 20:07:34, during the load. The report's memory gate ([o4nea_gonogo.md](o4nea_gonogo.md)) reads the
  client's log shared with R160's and R166's runs: its 718 MB low is R160d's, its 341 MB swap high R160c's. NO-GO on
  the swap growth, as for those runs. The RAM watchdog did not stop the server. Card peaks MiB: 5,733 5,753 5,679
  5,753 5,363 5,471 5,631 5,707 5,113, R166a's to the MiB.
- **The load took 656 s,** 118 s more than R166's 539 and 538 s on the same rig, file and build. The server's start
  is outside the tasks' time.
- **Server log:** 13 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill ([o4nea_gonogo.md](o4nea_gonogo.md)).
- The report's files: [o4nea_compare.md](o4nea_compare.md), [o4nea_report_table.md](o4nea_report_table.md),
  [o4nea_gonogo.md](o4nea_gonogo.md); the per-request and per-attempt rows: [agent/o4nea/](agent/o4nea/).

## Files

| File | What it holds |
|---|---|
| [agent.tsv](agent.tsv) | per arm: passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes for all tasks, a task's median, minutes a task at equal work against `v3ka` (R160's rows, R166's two and `o4nea`'s) |
| [host.tsv](host.tsv) | R166a's, R166b's and R169a's lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the end, each card's peak memory |
| [r169_reallocs.tsv](r169_reallocs.tsv) | R166's two runs and R169a: graph reallocations and prompt seconds per prompt size |
| [o4nea_compare.md](o4nea_compare.md) | the report: per task and per arm, first token per turn |
| [o4nea_report_table.md](o4nea_report_table.md) | per arm over the client's attempts (R160's six runs, R166's two and R169a): pass@1, minutes, prefill, decode, prompt tokens, cache hit |
| [o4nea_gonogo.md](o4nea_gonogo.md) | the report's gates and each server log's counts |
| [o4nea_reallocs.tsv](o4nea_reallocs.tsv), [o4nea_requests_reallocs.tsv](o4nea_requests_reallocs.tsv) | R169a's graph reallocations and prompt seconds per prompt size, and per request |
| [agent/o4nea/](agent/o4nea/) | `turns.csv` (one row a request) and `attempts.csv` (one row a task) |
| [run/r169.sh](run/r169.sh) | runs one arm, `o4nea` or `o4neb` |
| [run/host.py](run/host.py) | the host table from the rig's `meminfo.log` and `gpu.csv` (R166's) |

## How to run it

[run/r169.sh](run/r169.sh) runs one arm. It reads the v4 settings file, stops if its `PRESET_ARGS` or `PRESET_ENV`
is not the released Ornith preset's, and passes the preset's environment less `GGML_CUDA_KEEPALIVE_EAGER=1`. Run it
from the repository's root on the machine that drives the rig; the machine-specific values go in the environment:

```bash
ARM=o4nea RIG=<rig alias> CLIENT=local MODEL=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN_V4=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r160 OUT=<out>/agent-o4nea \
  bash docs/data/2026-11-16_r169_ornith_no_eager/run/r169.sh
```

R169a was started with the same call written out, from a copy of `v4` at `eb0a9c4`: the harness
[release_check_s12.sh](../2026-11-03_flash_next_mtp/run/release_check_s12.sh) with `STEPS=run`,
`REL_ENV="${PRESET_ENV/ GGML_CUDA_KEEPALIVE_EAGER=1/}"`, `ARGS_o4nea="--temp 0.6 -b 4096 -ub 512 -ts
6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"`, `BUDGET_MIN_o4nea=50`, `CTX=49152`,
`TIME_CAP_MIN=12`, `SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}'`, and the client's results
folder R160's and R166's runs used. Its report came from the same environment with `STEPS="report collect"` and
`ARMS=o4nea`. The client's results folder is shared with those runs and the rig's logs are appended to it, so the
tables used R169a's own rows. They ran in the sandbox image (`agent-check:0.87.1`, no network) from R166b's table
folder plus `o4nea`'s rows of `res/turns.csv` and of `res/attempts.csv` (only those whose `tag` starts with `o4nea/`),
its server log and the lines of `meminfo.log` and `gpu.csv` that R169a added (from 19:57:23):
`agent_tables.py OUT v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb o4nea`
([docs/data/2026-11-05_keepalive/run/agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py); the earlier
arms' rows come out as R166b's), `scripts/realloc_counts.py o4nea=rig/server-o4nea.log` (default and `--requests`),
`scripts/realloc_counts.py o4sna=… o4snb=… o4nea=…` for [r169_reallocs.tsv](r169_reallocs.tsv), and
`run/host.py <those lines> R169a o4nea`.

## Provenance and licence

The tasks are the agent check's 12 bug fixes, derived from Exercism's Python and JavaScript exercises (MIT; source,
licences and commits in [bench/agent_check](../../../bench/agent_check/README.md)); this folder holds only their IDs
(in [agent/o4nea/attempts.csv](agent/o4nea/attempts.csv)). The tables summarize this project's own runs and are
published under the repository's licence.
