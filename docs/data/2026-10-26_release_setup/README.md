# The v2.0 release setup, chosen on agentic speed (2026-10-26)

Every run behind the agentic table of [results.md](../../../release/flint-v2/results.md#agentic-speed-a-coding-agent-at-work-2026-10-26):
the pi coding agent fixing the twelve small bugs of [bench/agent_check](../../../bench/agent_check/README.md)'s bugfix
set (six Python, six JavaScript, one attempt each, real test runs) against llama-server on nine P106-090 cards, one run
per server setup, 15 runs and 161 scored attempts. Next to them, what fits at the full 262,144-token context and how
fast the server reads an agent's prompt. The release command line (Flint-Pascal-Q5 or Q8_0 at `-c 262144 -b 512 -ub 128
--pipe-depth 4`) was picked from these runs; registry rows Q131 (the agent check), Q111 (the agent prompt cache) and
Q140 (the command lines at 262k) in the [registry](../../../quants/experiments.md#registry).

| Folder | Engine | File | Context | Flags beyond the common ones | Row of results.md |
|---|---|---|---|---|---|
| `k2h-d4` | release `37e717ff6` | Flint-Pascal-Q5 | 262,144 | `-b 512 -ub 128 --pipe-depth 4` | the release command line |
| `k2h` | release | Flint-Pascal-Q5 | 262,144 | `-b 512 -ub 128 --pipe-depth 3` | Flint-Pascal-Q5 at depth 3 |
| `stock-k2h` | stock `d77256999` | Flint-Pascal-Q5 | 65,536 | `-b 2048 -ub 512` | Flint-Pascal-Q5 on stock |
| `ud-d4` | release | UD-Q4_K_XL | 262,144 | `-b 512 -ub 128 --pipe-depth 4` | UD-Q4_K_XL, release |
| `ud5-d4` | release | UD-Q5_K_XL | 262,144 | `-b 512 -ub 128 --pipe-depth 4` | UD-Q5_K_XL, release |
| `stock-ud` | stock | UD-Q4_K_XL | 65,536 | `-b 2048 -ub 512` | UD-Q4_K_XL on stock |
| `d4`, `q8-d4b` | release | Q8_0 | 262,144 | `-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4` | Q8_0, release (two runs) |
| `page`, `q8-d3b` | release | Q8_0 | 262,144 | the same at `--pipe-depth 3` | Q8_0 at depth 3 (two runs) |
| `d2` | release | Q8_0 | 262,144 | the same at `--pipe-depth 2` | Q8_0 at depth 2 |
| `noguess` | release | Q8_0 | 262,144 | the same without `--pipe-depth` | without pipelined speculation |
| `cache` | release | Q8_0 | 262,144 | depth 3 with `--prefix-cache DIR --checkpoint-gen` | with the agent prompt cache |
| `stock` | stock | Q8_0 | 65,536 | `-b 2048 -ub 512 -ts 6,6,5,5,6,5,5,6,6` | Q8_0 on stock |
| `v1-q8` | v1 `eeaa59660` | Q8_0 | 65,536 | `-b 2048 -ub 512 -ts 6,6,5,5,6,5,5,6,6` | Q8_0 on v1 |

Common flags: `-ngl 99 -fa on -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek`, the
makers' sampling (temperature 1.0, top-p 0.95, top-k 20, min-p 0) and `--metrics`. Each attempt is capped at 12 minutes
and 40 turns; no attempt starts after 25 minutes on the release engine or 35 on stock and v1. A sixteenth candidate,
`k2h-ub512` (Flint-Pascal-Q5 with `-b 2048 -ub 512` at 262,144 tokens), never served: the 4 GB host ran out of memory
while the server loaded, and the server was killed. Flint-Pascal-Q5 is the file built with recipe K2h-r16.

| File | What |
|---|---|
| `arms.tsv` | one row per run, the sixteen candidates: engine, build, file, context, flags, date and start time, budget; tasks scored and passed, minutes for all tasks, minutes per task (median, p90), generation and prompt tok/s over all requests, time to first token (median and p90 over all turns, and the median of each attempt's median), tool calls (all, failed, malformed), runaways, turns, prompt cache hit, the mean seconds per task in prefill / decode / fixed cost / other, and the host's lowest MemAvailable and swap at the start and end |
| `<run>/compare.md` | the run's report: per task the result, minutes, turns and tool calls; per arm the totals; the median time to first token and new prompt tokens per turn |
| `<run>/table.md` | the per-arm row with the prefill / decode / fixed split, prompt tokens processed per task and the cache hit rate |
| `<run>/gonogo.md` | the check's own list: harness, tool calls, cache paths, host memory, server log counts |
| `<run>/attempts.csv` | one row per attempt (the layout of [bench/task_results.py](../../../bench/task_results.py)): pass, tests passed, turns, requests, wall time and its split, prompt tokens new and cached, peak context, time to first token, tool calls and errors, runaway |
| `<run>/turns.csv` | one row per request: task, turn, time to first token, end time, new and cached prompt tokens, the server's prompt time, generated tokens and time, finish reason, tool calls |
| `patches/<run>/<task>.patch` | the agent's change to the task's files, as `diff -ruN` gave it, with paths relative to the task's workspace; six are empty (runaways that changed nothing) |
| `fit_prefill.tsv` | Q8_0 at 262,144 tokens (and a few smaller contexts) in llama-batched-bench: a 16,384-token prompt, then 64 tokens, per batch / micro-batch size and mask placement; whether pipeline parallelism survived |
| `serve_prefill.tsv` | Q8_0 on llama-server at depth 3, per batch setting and context: a fresh 13,404-token prompt, the next turn (+1,951 tokens, 13,390 cached) and generation at 15,341 tokens of context |
| `run/agent_search.sh`, `run/agent_v1.sh`, `run/fit_prefill.sh` | the commands; [scripts/serve_prefill.sh](../../../scripts/serve_prefill.sh) made `serve_prefill.tsv` |

`compare.md` and `table.md` name `res/attempts.csv` and `res/turns.csv`, their place in a run's output folder; here
they sit next to them. A run's arm is called `rel` (the release engine) or `stock` (stock llama.cpp, and v1, which ran
through the same arm with its own binary).

## What it shows beyond results.md

- **Host memory is tight in every run.** The lowest MemAvailable was 643–918 MB on the release engine at 262,144 tokens
  and 931–984 MB on stock and v1 at 65,536, and swap grew by 255–289 MB per run (533 MB in `k2h-d4`, which started with
  120 MB in swap). That fails the check's memory item (at least 600 MB and no swap growth) in all fifteen runs, so each
  `gonogo.md` ends in NO-GO; every other item passed, except the proxy-row item in the nine runs with a runaway, where
  the request cut at the 12-minute cap has a row without timings. No server was stopped by the RAM watchdog.
- **Every runaway is the same task,** `bugfix-python-scale-generator`: it reached the 12-minute cap in nine of the
  fifteen runs (one of them, `k2h`, still passed its tests). In six the agent had changed nothing by then, so those
  patches are empty.
- **At 262,144 tokens only `-ub 128` keeps Q8_0's pipeline parallelism:** with `-b 2048 -ub 256` card 1 has no room for
  its 561 MiB compute buffer, the scheduler falls back to running without pipeline parallelism, and a 16k prompt reads
  at 133 against 214 tok/s. `-b 8192 -ub 128` reads at the same 214 as `-b 512`.
- **llama-batched-bench understates the server's prompt speed about 2.3×:** 214 tok/s for a 16k prompt against
  485 tok/s for a 13.4k prompt on llama-server with the same flags. Prompt speeds for agents come from the server.
- **`-ub 512` at 131,072 tokens reads faster but decodes at 21.6 tok/s** (554 tok/s fresh prompt, 7.3 s for the next
  turn); at 65,536 tokens it decodes at 38.7 (570 tok/s, 6.4 s). The release line at 262,144 tokens: 485 tok/s, 5.8 s,
  37.2 tok/s; `-b 2048 -ub 128` gives the same (489, 5.7 s, 37.1).

Made on 2026-10-26: the agent runs from the controlling machine (only ssh), the client in Docker on a second machine
and llama-server on the rig, as [bench/agent_check](../../../bench/agent_check/README.md#the-release-check-one-command-1-h)
describes; the other two on the rig alone:

```bash
# the agent check per candidate; ENV_FILE holds the machine values (bench/agent_check/release.env.example)
ENV_FILE=my-release.env CLIENT_ROOT=/data/agent-check Q8_MODEL=/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf \
  K2H_MODEL=/mnt/models/flint/flint-q113-K2h-r16.gguf UD_MODEL=/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf \
  UD5_MODEL=/mnt/models/flint/flint-UD-Q5_K_XL.gguf OUT=agent bash docs/data/2026-10-26_release_setup/run/agent_search.sh
ENV_FILE=my-release.env CLIENT_ROOT=/data/agent-check Q8_MODEL=/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf OUT=agent \
  bash docs/data/2026-10-26_release_setup/run/agent_v1.sh

# what fits at 262k (fit_prefill.tsv)
RIG=rig OUT=out bash docs/data/2026-10-26_release_setup/run/fit_prefill.sh
RIG=rig OUT=out GRID='8192/128 8192/192' bash docs/data/2026-10-26_release_setup/run/fit_prefill.sh

# the server's prompt reading (serve_prefill.tsv)
RIG=rig BIN=/opt/llama-builds/llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4/bin \
  MODEL=/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf TS=6,6,5,5,6,5,5,6,6 scripts/serve_prefill.sh serve_prefill.md
```

The runs went in the order of `arms.tsv`, from 12:16 to 21:50. The tables come from each run's `compare.md`,
`table.md` and `gonogo.md`; `fit_prefill.tsv` and `serve_prefill.tsv` from the two scripts' logs. In `fit_prefill.tsv`
three settings were stopped before a result and the batch size of the four context and mask rows was not recorded
(the micro-batch was). Not published: the agent's event streams and sessions, the request bodies, the proxy's
per-request log and the rig's monitors (the tables above summarize them).

The tasks are Exercism exercises (python and javascript tracks, MIT; `bench/agent_check/tasks/polyglot/LICENSE.exercism-*`)
with one injected bug each; the patches are Flint's edits to them, made through pi (Flint is Apache-2.0). The serve
probe's prompt is Python's standard library source on the rig (PSF licence) and is not stored here.
