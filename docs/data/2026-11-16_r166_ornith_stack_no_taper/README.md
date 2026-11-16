# Ornith's candidate stack without the micro-batch taper on the agent check (R166, Q1491; 2026-11-16)

R166 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix
tasks, Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget an arm, the makers' sampling at
temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF (valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`,
48,196,722,592 bytes) at 49,152 tokens on nine P106-090 cards, with R160's method
([2026-11-14_r160_ornith_release_agent](../2026-11-14_r160_ornith_release_agent/README.md)). Two runs of one arm, one
load a run: **R166a `o4sna`, R166b `o4snb`**.

- `o4sn*` is R160's stack arm `o4s*` without `--ubatch-taper short=600:64`: the v4 preset
  ([release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)) on the release candidate `v4-rc2`
  (`pascal-p106-v4` `eb82916d3`, upstream b12360; the rig tree's `llama-server` md5
  `e7afd26db01f4c671549ef12c504ada3`), plus `GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1`,
  `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall` and `-b 4096` in place of `-b 2048`.
- Why: the taper is the one switch of the stack that changes Ornith's replies. On 512-token calls it moves the
  next-token distributions by a mean KLD of 0.047 with the same perplexity
  ([2026-11-16_fn93c_ornith_taper_kld](../2026-11-16_fn93c_ornith_taper_kld/README.md)). R166 measures how much of the
  stack's gain holds without it. The comparison arms are R160's: the stack with the taper (`o4sa`, `o4sb`) and the v4
  preset (`o4a`, `o4b`).

Registry row Q1491 in [quants/experiments.md](../../../quants/experiments.md). R166a ran on 2026-11-16; R166b, and
the class of `o4sn` against the stack and the preset, follow.

## The runs

| Run | Arm | Rig | Server started, up | Tasks | Passed | Started command |
|---|---|---|---|---|---|---|
| R166a | `o4sna` | rig 2 | 13:51:30, up after 539 s | 14:00:33–14:29:45 | 12 / 12 | 521 bytes, md5 `211f8db2edeacd267705795e4911d8c9` |

## R166a: `o4sna`, the stack without the taper, first run (2026-11-16)

- **The started command** is R160b's and R160e's (`o4sa`, `o4sb`) less ` --ubatch-taper short=600:64` and nothing
  else: the preset's four switches and the stack's two, the harness's `COMMON` line, then `--temp 0.6 -b 4096 -ub 512
  -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall`, with one `-b`, 4096. The
  server's `switches:` line, read during the load, says `6 set: GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_HOP_STAGED=1
  GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SERVER_YIELD=2`, as for
  `o4sa` and `o4sb` (the taper is a command-line flag, not a switch).

**Numbers** (one run each; [agent.tsv](agent.tsv), [host.tsv](host.tsv); R160's rows from its
[agent.tsv](../2026-11-14_r160_ornith_release_agent/agent.tsv) and
[host.tsv](../2026-11-14_r160_ornith_release_agent/host.tsv)):

| | v4 `o4a` | v4 `o4b` | stack `o4sa` | stack `o4sb` | **no taper `o4sna`** |
|---|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.57 | 2.45 | 2.08 | 2.06 | **2.11** |
| minutes, all 12 tasks (wall) | 29.7 | 31.0 | 27.6 | 25.9 | **29.1** |
| minutes a task, median | 2.07 | 1.97 | 1.94 | 1.50 | **2.15** |
| turns | 62 | 57 | 63 | 65 | **63** |
| generation tok/s | 16.53 | 16.37 | 16.30 | 16.52 | **16.52** |
| prompt tok/s | 56.6 | 65.4 | 114.5 | 113.7 | **103.0** |
| first token median, p90 s | 8.65, 19.22 | 9.58, 20.29 | 3.05, 10.97 | 3.27, 11.28 | **6.69, 11.29** |
| first token on a task's first turn, median s | 16.32 | 16.26 | 8.45 | 8.44 | **8.43** |
| tool calls / failed / malformed | 62 / 0 / 0 | 57 / 0 / 0 | 64 / 0 / 0 | 65 / 1 / 0 | **63 / 2 / 0** |
| graph reallocations | 0 in 62 requests | 0 in 57 | 0 in 63 | 0 in 65 | **0 in 63 requests** |
| lowest MemAvailable MiB | 803 | 718 | 1,252 | 1,260 | **1,225** |

- **Minutes a task at equal work** price the same work, `ka`'s (v3.0's 2026-11-06 run on the same rig, file and
  line: 2.52), at each run's own speeds. `o4sna` took 2.11, against the stack's 2.08 and 2.06 and the v4 preset's 2.57
  and 2.45. This is one run of two; the class waits for R166b.
- **Where the prompts slowed** ([o4sna_reallocs.tsv](o4sna_reallocs.tsv), the server's prompt seconds per size of
  prompt): requests under 512 tokens took 116.3 s for 37 requests, against 80.3 s for 39 (`o4sa`) and 83.7 s for 39
  (`o4sb`) with the taper, and 199.2 s for 38 (`o4a`) and 193.0 s for 34 (`o4b`) on the preset. Prompts of 1,024–1,999
  tokens took 101.6 s for 10 requests, as `o4sb`'s 101.6 s for 10 (`o4sa` 112.0 s for 11); prompts of 2,000 or more
  85.3 s for 5 (`o4sb` 86.8 s for 5). The taper only acts on prompts of at most 600 tokens, so its share of the
  stack's prompt speed is in the short prompts of the later turns: the first token median is 6.69 s against 3.05 and
  3.27 s, while a task's first turn (a long prompt) is 8.43 s against 8.45 and 8.44 s.
- **Where the wall minutes went** ([o4sna_report_table.md](o4sna_report_table.md), the seven R160 and R166 runs'
  attempts): `o4sna`'s attempts spent a mean 110 s a task generating (`o4sa` 107, `o4sb` 97) and 33 s reading prompts
  (29, 30; `o4a` 53, `o4b` 54), and processed 3,405 prompt tokens a task (3,316, 3,439). Python `scale-generator` took
  4.5 minutes in 5 turns (`o4sa` 8.6, `o4sb` 5.0), JavaScript `variable-length-quantity` 4.2 in 5 (2.0, 2.6), Python
  `pig-latin` 3.8 in 7 (2.2, 4.7) and `list-ops` 3.6 in 5 (1.9, 1.1); every other task took 3.2 minutes or less. No
  task came near the 12-minute cap.
- **Graph reallocations** ([o4sna_reallocs.tsv](o4sna_reallocs.tsv),
  [o4sna_requests_reallocs.tsv](o4sna_requests_reallocs.tsv)): 0 in every bucket (37 requests under 512 tokens, 11 at
  512–1,023, 10 at 1,024–1,999, 5 at 2,000+).
- **Host** ([host.tsv](host.tsv), R166a's own lines of `meminfo.log` and `gpu.csv`, from 13:51:31): lowest
  MemAvailable 1,225 MiB at 14:28:59, during the last task (checkpoints on the cards, as `o4sa` and `o4sb`), 3,187 MiB
  at the end; swap 31 MiB at the start, 338 MiB at its highest at 13:57:08, during the load. The report's memory gate
  ([o4sna_gonogo.md](o4sna_gonogo.md)) reads the client's log shared with R160's runs: its 718 MB low is R160d's, its
  341 MB swap high R160c's. NO-GO on the swap growth, as for R160's runs. The RAM watchdog did not stop the server.
  Card peaks MiB: 5,733 5,753 5,679 5,753 5,363 5,471 5,631 5,707 5,113.
- **Server log:** 13 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill ([o4sna_gonogo.md](o4sna_gonogo.md)).
- The report's files: [o4sna_compare.md](o4sna_compare.md), [o4sna_report_table.md](o4sna_report_table.md),
  [o4sna_gonogo.md](o4sna_gonogo.md); the per-request and per-attempt rows: [agent/o4sna/](agent/o4sna/).

## Files

| File | What it holds |
|---|---|
| [agent.tsv](agent.tsv) | per arm: passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes for all tasks, a task's median, minutes a task at equal work against `v3ka` (R160's rows and `o4sna`'s) |
| [host.tsv](host.tsv) | R166a's lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the end, each card's peak memory |
| [o4sna_compare.md](o4sna_compare.md) | the report: per task and per arm, first token per turn |
| [o4sna_report_table.md](o4sna_report_table.md) | per arm over the client's attempts (R160's six runs and R166a): pass@1, minutes, prefill, decode, prompt tokens, cache hit |
| [o4sna_gonogo.md](o4sna_gonogo.md) | the report's gates and each server log's counts |
| [o4sna_reallocs.tsv](o4sna_reallocs.tsv), [o4sna_requests_reallocs.tsv](o4sna_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, and per request |
| [agent/o4sna/](agent/o4sna/) | `turns.csv` (one row a request) and `attempts.csv` (one row a task) |
| [run/r166.sh](run/r166.sh) | runs one arm, `o4sna` or `o4snb` |
| [run/host.py](run/host.py) | the host table from the rig's `meminfo.log` and `gpu.csv` |

## How to run it

[run/r166.sh](run/r166.sh) runs one arm. It reads the v4 settings file and stops if its `PRESET_ARGS` is not v3.0's
Ornith line. Run it from the repository's root on the machine that drives the rig; the machine-specific values go in
the environment:

```bash
ARM=o4sna RIG=<rig alias> CLIENT=local MODEL=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN_V4=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r160 OUT=<out>/agent-o4sna \
  bash docs/data/2026-11-16_r166_ornith_stack_no_taper/run/r166.sh
```

R166a was started with the same call written out: the harness
[release_check_s12.sh](../2026-11-03_flash_next_mtp/run/release_check_s12.sh) with `STEPS=run`,
`REL_ENV="$PRESET_ENV GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1"`, `ARGS_o4sna="--temp 0.6 -b 4096 -ub 512 -ts
6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall"`, `BUDGET_MIN_o4sna=50`, `CTX=49152`,
`TIME_CAP_MIN=12`, `SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}'`, and the client's results
folder R160's runs used. Its report came from the same environment with `STEPS="report collect"` and `ARMS=o4sna`.
The client's results folder is shared with R160's runs and the rig's logs are appended to it, so the tables used
R166a's own rows. They ran in the sandbox image (`agent-check:0.87.1`) from a folder holding R160f's table rows
(`v3ka`, `o4a`, `o4sa`, `o3a`, `o4b`, `o4sb`, `o3b`) plus `o4sna`'s rows of `res/turns.csv` and of `res/attempts.csv`
(only those whose `tag` starts with `o4sna/`), the server logs, and the lines of `meminfo.log` and `gpu.csv` that
R166a added (from 13:51:31): `agent_tables.py OUT v3ka o4a o4sa o3a o4b o4sb o3b o4sna`
([docs/data/2026-11-05_keepalive/run/agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)),
`scripts/realloc_counts.py o4sna=rig/server-o4sna.log` (default and `--requests`), `run/host.py <those lines> R166a
o4sna`.

## Provenance and licence

The tasks are the agent check's 12 bug fixes, derived from Exercism's Python and JavaScript exercises (MIT; source,
licences and commits in [bench/agent_check](../../../bench/agent_check/README.md)); this folder holds only their IDs
(in [agent/o4sna/attempts.csv](agent/o4sna/attempts.csv)). The tables summarize this project's own runs and are
published under the repository's licence.
