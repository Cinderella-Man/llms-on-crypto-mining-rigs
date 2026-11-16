# Flint Pascal Q5 GGUF on Miner Llama v4's preset and on v3.0: the agent check (R170, Q1497; 2026-11-16)

The 5-bit file, Flint Pascal Q5 GGUF (K2h-r16 with the release header, 29.4 GB), had no v4 numbers. In v3's release it
ran v3's line with layers by count (no `-ts`): 12 of 12 tasks, 49.4 tok/s, prompts 168 tok/s, first token 3.02 s median
([2026-11-05_release_regression](../2026-11-05_release_regression/README.md), arm `v3base` on the 5-bit file,
measured 2026-11-06). R170 runs the agent check on it twice, each run with its own load on nine P106-090 cards at
262,144 tokens:

- **R170a, arm `q5f4t`**: Flint's v4 preset ([release/flint-v4/settings.env](../../../release/flint-v4/settings.env),
  `f4t`) on the release candidate `v4-rc2`, without `-ts`, as [release/flint-v4](../../../release/flint-v4/README.md#run-it)
  says the 5-bit file runs;
- **R170b, arm `q5f3`**: v3.0 ([release/flint-v3/settings.env](../../../release/flint-v3/settings.env), no switch) on
  the same file, without `-ts` as in v3's Q5 run, on the same rig the same night.

The agent check ([bench/agent_check](../../../bench/agent_check/README.md)) is a coding agent fixing 12 small bugs, Python
and JavaScript alternating, with real tool runs and tests, one attempt a task, a 12-minute cap a task.

## Result: v4 against v3.0 on the 5-bit file, the same night

**On Flint Pascal Q5 GGUF, v4's preset does the same work in 3.9 % less time than v3.0 (est. from 2.45 and 2.55
minutes a task at v3's Q5 run's work), one run each on the same rig the same night.** At R170b's own work the two
take 1.61 and 1.67 minutes a task (3.6 % less). The difference is in reading prompts (195.6 against 169.1 tok/s) and
in the first token (median 1.82 against 2.81 s). Generation is the same: 52.01 against 51.77 tok/s. The two are not
classed: one run each.

| | v3.0, R170b `q5f3` | **v4's preset, R170a `q5f4t`** | v3.0, v3's Q5 run (2026-11-06) |
|---|---|---|---|
| tasks passed | 12 of 12 | **12 of 12** | 12 of 12 |
| minutes a task at equal work (v3's Q5 run's work) | 2.55 | **2.45** | 2.64 |
| minutes a task at equal work (R170b's work) | 1.67 | **1.61** | |
| minutes, all 12 tasks | 20.1 | **23.0** | 31.7 |
| minutes a task, median | 0.93 | **1.11** | 1.66 |
| turns | 70 | **74** | 99 |
| new prompt tokens; generated tokens | 48,939; 45,627 | **60,681; 53,673** | 79,614; 67,479 |
| generation tok/s | 51.77 | **52.01** | 49.43 |
| prompt tok/s | 169.1 | **195.6** | 168.2 |
| first token per turn, median and p90 s | 2.81, 8.84 | **1.82, 7.13** | 3.02, 7.94 |
| first token on a task's first turn, median s | 4.27 | **3.39** | 4.74 |
| tool calls / failed / malformed | 64 / 0 / 0 | **70 / 1 / 0** | 90 / 1 / 0 |
| checkpoint restores | 15, from the host: 8 in 292.4 ms, 5 in 292.5, 1 in 293.2, 1 in 294.1 | **14, from the cards: each in 2.0, 2.1, 2.2 or 2.3 ms** | not in that run's published data |
| graph reallocations (target) | 242 with the graph's size unchanged, 0 changed, in 69 of 70 requests | **295, 0 changed, in 73 of 74** | not in that run's published data |
| lowest MemAvailable; swap used | 931 MiB (21:25:34); 26 → 294 MiB | **1,104 MiB (20:52:19); 17 → 290 MiB** | 908 MiB; 32 → 293 MiB |
| each card's peak memory, MiB (cards 0–8) | 5,007 4,787 3,663 4,731 3,653 4,707 3,617 4,165 4,095 | **5,027 4,813 3,683 4,751 3,673 4,727 3,637 4,191 4,105** | not in that run's published data |

- **Why v3.0 finished sooner on the clock:** its agent read and wrote less (45,627 generated tokens against 53,673).
  Python `scale-generator` was the longest task in both runs: 440 s and 17,393 tokens on v3.0, 561 s and 20,867 tokens
  on v4. Minutes a task at equal work price the same tokens at each run's speeds, so they compare the engines
  ([agent.tsv](agent.tsv) at v3's Q5 run's work, [agent_ref_q5f3.tsv](agent_ref_q5f3.tsv) at R170b's).
- **Prompts by size** ([prompt_by_size.tsv](prompt_by_size.tsv)), v4 against v3.0 the same night: under 512 tokens
  165.3 against 107.8 tok/s (49 and 42 requests), 512–1,023 209.2 against 169.2, 1,024–1,999 225.9 against 208.4,
  2,000 and more 191.5 against 182.3 (5 and 6 requests). Short prompts gain the most: they are most of an agent's
  turns.
- **Generation without JavaScript `complex-numbers`** ([gen_without_complex.tsv](gen_without_complex.tsv)): 52.10
  tok/s on v4 and 51.84 on v3.0 the same night. v3's run of 2026-11-06 read 53.64 without that task.
- **Checkpoints:** v3.0 keeps the context checkpoints in host memory and copies one back in 292.4, 292.5, 293.2 or 294.1 ms;
  v4's preset keeps the newest four on the cards (`--ctx-checkpoints-vram 4`), and a restore takes 2.0, 2.1, 2.2 or 2.3 ms.
- **Host and cards** ([host.tsv](host.tsv)): on v3.0 swap grew from 26 to 294 MiB during the load (highest at
  21:05:42, before the first task), as on v4 (17 → 290); MemAvailable was lowest at 931 MiB at 21:25:34, in
  `scale-generator`'s last turn (context 25,995 tokens). v4's preset holds 20, 26, 20, 20, 20, 20, 20, 26 and 10 MiB
  more on cards 0 to 8.
- **The report's go/no-go** ([q5f3_gonogo.md](q5f3_gonogo.md)): as for R170a, every line GO but memory, NO-GO on the
  swap the load moved (294 MiB); 0 of 64 tool calls malformed, median cache hit 93.8 %.

## Result: R170a, v4's preset (`q5f4t`)

One run, 2026-11-16: the server started 20:21:52 and was up after 427 s; the 12 tasks ran 20:29:02–20:52:15. Beside it,
v3's Q5 run of the release regression, made on v3.0 (`1551f5fe1`) on 2026-11-06. The two runs are not classed: they
were made ten days apart, and R170b (above) is the same-night comparison.

| | v3.0, v3's Q5 run (2026-11-06) | **v4's preset, R170a `q5f4t`** |
|---|---|---|
| tasks passed | 12 of 12 | **12 of 12** |
| minutes a task at equal work (v3's Q5 run's work) | 2.64 | **2.45** |
| minutes, all 12 tasks | 31.7 | **23.0** |
| minutes a task, median | 1.66 | **1.11** |
| turns | 99 | **74** |
| generation tok/s | 49.43 | **52.01** |
| prompt tok/s | 168.2 | **195.6** |
| first token per turn, median and p90 s | 3.02, 7.94 | **1.82, 7.13** |
| first token on a task's first turn, median s | 4.74 | **3.39** |
| tool calls / failed / malformed | 90 / 1 / 0 | **70 / 1 / 0** |
| graph reallocations (target) | not in that run's published data | **295 with the graph's size unchanged, 0 changed, in 73 of 74 requests** |
| lowest MemAvailable; swap used | 908 MiB; 32 → 293 MiB | **1,104 MiB (20:52:19); 17 → 290 MiB** |
| each card's peak memory, MiB (cards 0–8) | not in that run's published data | **5,027 4,813 3,683 4,751 3,673 4,727 3,637 4,191 4,105** |

- **Equal work** means v3's Q5 run's work (its 79,614 prompt tokens and 67,479 generated tokens, and its 61.4 s outside
  the server) at each run's own prompt and generation speed, over its 12 tasks
  ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py), [agent.tsv](agent.tsv)). On v3's own run it gives
  back that run's report exactly (168.2 and 49.43 tok/s, 3.02 and 7.94 s). Each speed is the session's average, and
  the two sessions reached different depths: v3's JavaScript `complex-numbers` ran 31 turns with its context to
  32,703 tokens and generated 14,024 tokens at 38.04 tok/s, while R170a's deepest task, Python `scale-generator`, reached
  25,775 tokens.
- **Without JavaScript `complex-numbers`** ([gen_without_complex.tsv](gen_without_complex.tsv)): v3's Q5 run generates
  at 53.64 tok/s over 68 turns and R170a at 52.10 over 69; prompts read at 182.3 and 194.2 tok/s. R170a ran that task
  in 5 turns and 0.8 minutes (1,294 tokens at 48.71 tok/s), v3's run in 31 turns and 8.9 minutes.
- **Long tasks:** Python `scale-generator` took 9.35 minutes (561 s) in 8 turns and passed, 159 s under the cap: 20,867
  tokens in 424.77 s of generation (49.13 tok/s), its context to 25,775 tokens (v3's run: 5.6 minutes, 13,126 tokens).
  JavaScript `variable-length-quantity` took 2.5 minutes in 11 turns, Python `list-ops` 2.4 minutes; every other task
  took under 2 minutes ([q5f4t_compare.md](q5f4t_compare.md)).
- **Prompt by size** ([prompt_by_size.tsv](prompt_by_size.tsv)), R170a against v3's run: under 512 tokens 165.3
  against 112.5 tok/s (49 and 66 requests), 512–1,023 209.2 against 154.9, 1,024–1,999 225.9 against 186.1, 2,000 and
  more 191.5 against 200.6 (5 and 7 requests; v3's included `complex-numbers`' prompts at depth).
- **Graph reallocations** ([q5f4t_reallocs.tsv](q5f4t_reallocs.tsv), per request in
  [q5f4t_requests_reallocs.tsv](q5f4t_requests_reallocs.tsv)): under 512 prompt tokens 84 in 49 of 49 requests;
  512–1,023, 38 in 10 of 10; 1,024–1,999, 60 in 10 of 10; 2,000 and more, 113 in 4 of 5.
- **Checkpoints on the cards:** 14 restores, each in 2.0, 2.1, 2.2 or 2.3 ms; none created on the host or erased, no
  full re-processing, no NaN or inf, no watchdog kill (the server log's line in [q5f4t_gonogo.md](q5f4t_gonogo.md)).
- **Host and cards** ([host.tsv](host.tsv), [run/host_summary.py](../2026-11-14_r159_flint_release_agent/run/host_summary.py)):
  swap grew from 17 to 290 MiB during the load (its highest at 20:28:59, before the first task) and did not grow
  during the tasks; MemAvailable was lowest at 1,104 MiB at 20:52:19, after the last task's end at 20:52:15 and before
  the server stopped. The RAM watchdog did not stop the server. Layers by count leave cards 2, 4 and 6 peaking at
  3,683, 3,673 and 3,637 MiB and card 0 at 5,027 MiB of 6,144; Flint Pascal Q8 GGUF on the same preset with `-ts` peaks at 5,985 MiB on two cards
  ([R159f](../2026-11-14_r159_flint_release_agent/README.md#r159f-f4tb-the-tuned-preset-f4ts-second-run-2026-11-16)).
- **The report's go/no-go** ([q5f4t_gonogo.md](q5f4t_gonogo.md)): infrastructure, proxy rows, first token against
  prefill, tool calls (0 of 70 malformed) and cache (median hit 94.0 % on turns from the second) GO; memory NO-GO on
  the swap the load moved (290 MiB), as on v3's Q5 run (293 MiB).

## Result: R170b, v3.0 (`q5f3`)

One run, 2026-11-16, after R170a on the same rig: the server started 21:03:25 and was up after 432 s; the 12 tasks ran
21:10:39–21:30:56. Its numbers are in the first table.

- **Long tasks** ([q5f3_compare.md](q5f3_compare.md)): Python `scale-generator` passed in 7.3 minutes (440 s) in 8 turns,
  17,393 tokens at 48.18 tok/s (est. from 360.98 s of generation), its context to 25,995 tokens; JavaScript
  `variable-length-quantity` took 2.8 minutes in 7 turns, Python `list-ops` 2.0 minutes and `affine-cipher` 1.8; every
  other task took under 2 minutes. JavaScript `complex-numbers` took 0.9 minutes in 6 turns.
- **Graph reallocations** ([q5f3_reallocs.tsv](q5f3_reallocs.tsv), per request in
  [q5f3_requests_reallocs.tsv](q5f3_requests_reallocs.tsv)): under 512 prompt tokens 65 in 42 of 42 requests;
  512–1,023, 45 in 12 of 12; 1,024–1,999, 59 in 10 of 10; 2,000 and more, 73 in 5 of 6.
- **The server log** ([q5f3_gonogo.md](q5f3_gonogo.md)): 15 checkpoints restored, none created on the host or erased,
  no full re-processing, no NaN or inf, no watchdog kill.

## Method

- **Build:** `v4-rc2` (the fork's `pascal-p106-v4` at `eb82916d3`), the rig tarball
  `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`, `llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`.
- **File:** `flint-pascal-Q5.gguf`, 29,390,812,736 bytes, md5 `54e0f28553d0c18ba940d1e8d894dfcf` (checked on the rig
  before the run), sha256 `0163ec54…df6003`.
- **The line:** the preset's switches (`switches: 5 set: GGML_CUDA_GRAPHS_FORCE=1 LLAMA_KV_SEQ_RM_BOUND=1
  LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_YIELD=2`, the server's own line) and its
  flags less `-ts 6,6,5,5,6,5,5,6,6`:

  ```
  llama-server -ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --alias flint -b 512 -ub 128 --pipe-depth 4
    --ctx-checkpoints-vram 4 --ubatch-taper short=600:64
  ```

  It is the command of R159's `f4t` runs on Flint Pascal Q8 GGUF without `-ts`, byte for byte. The server logged its
  plans at load: `context checkpoints on the devices: the newest 4 of each slot` and `micro-batch taper of prompt
  calls: short=600:64 (n_ubatch 128)`; the load fitted the first time.
- **v3's Q5 run** used v3.0's line for the 5-bit files, `-b 512 -ub 128 --pipe-depth 4` after the same common flags, no
  switches ([its run script](../2026-11-05_release_regression/run/agent_flint.sh), `REL_LINE`).
- **R170b** ran that line on v3.0 itself: the release build `pascal-p106-v3-b12076-ladder5` `1551f5fe1` (tag
  `miner-llama-v3.0`), the rig tarball `llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5`, `llama-server`
  md5 `c735e40f184f3de61f16ffd5e2ab5ea5`, with v3's preset ([release/flint-v3/settings.env](../../../release/flint-v3/settings.env):
  no switch) less `-ts 6,6,5,5,6,5,5,6,6`. Its command is R170a's less the five switches, `--ctx-checkpoints-vram 4` and
  `--ubatch-taper short=600:64`, byte for byte; the server's process had no `LLAMA_` or `GGML_` variable:

  ```
  llama-server -ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --alias flint -b 512 -ub 128 --pipe-depth 4
  ```

  The file's md5 was checked on the rig again before R170b. Each run loaded the model on its own server start.
- **The client** ran on the machine that drives the rig (`CLIENT=local`), in the sandbox image `agent-check:0.87.1` at
  `--cpu-shares 256`. Nothing else ran on the rig during the tasks.
- **The tables** were made in the same image: v3's rows
  ([agent/flint-pq5-v3base](../2026-11-05_release_regression/agent/flint-pq5-v3base/), arm `rel` renamed `v3q5`) beside
  R170a's and R170b's ([run/merge_rows.py](run/merge_rows.py)), then `agent_tables.py`, `realloc_counts.py`,
  `host_summary.py`, `prompt_by_size.py`, [run/gen_without_task.py](run/gen_without_task.py) and
  [run/per_task.py](run/per_task.py). Adding R170b's rows left every R170a number as it was.

## Commands

The run, from the repository's root on the machine that drives the rig ([run/r170a.sh](run/r170a.sh) checks the preset's
line first):

```bash
RIG=<rig alias> CLIENT=local MODEL=<rig path>/flint-pascal-Q5.gguf \
  BIN=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8272 RELAY_PORT=8276 PROXY_PORT=18773 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r170 OUT=<out>/agent-q5f4t \
  bash docs/data/2026-11-16_r170_flint_q5/run/r170a.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
```

That is this call, as it ran:

```bash
. release/flint-v4/settings.env
BENCH=$PWD/bench/agent_check RIG=<rig alias> CLIENT=local CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=256 \
  MODEL=<rig path>/flint-pascal-Q5.gguf RIG_LOGDIR=<rig folder>/agent-logs \
  RIG_PORT=8272 RELAY_PORT=8276 PROXY_PORT=18773 CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r170 \
  REL_BIN=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  REL_ENV="$PRESET_ENV" \
  ARGS_q5f4t="--alias flint -b 512 -ub 128 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64" \
  BUDGET_MIN_q5f4t=40 ARMS=q5f4t STEPS=run \
  bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh <out>/agent-q5f4t
```

R170b, the same way with v3.0 ([run/r170b.sh](run/r170b.sh) checks v3's preset first; its own `CLIENT_OUT`, since the
collect step appends the rig's logs to the folder's):

```bash
RIG=<rig alias> CLIENT=local MODEL=<rig path>/flint-pascal-Q5.gguf \
  BIN=<rig path>/llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8272 RELAY_PORT=8276 PROXY_PORT=18773 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r170b OUT=<out>/agent-q5f3 \
  bash docs/data/2026-11-16_r170_flint_q5/run/r170b.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
```

That is this call, as it ran:

```bash
. release/flint-v3/settings.env
BENCH=$PWD/bench/agent_check RIG=<rig alias> CLIENT=local CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=256 \
  MODEL=<rig path>/flint-pascal-Q5.gguf RIG_LOGDIR=<rig folder>/agent-logs \
  RIG_PORT=8272 RELAY_PORT=8276 PROXY_PORT=18773 CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r170b \
  REL_BIN=<rig path>/llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5/bin/llama-server \
  REL_ENV="$PRESET_ENV" ARGS_q5f3="--alias flint -b 512 -ub 128 --pipe-depth 4" \
  BUDGET_MIN_q5f3=40 ARMS=q5f3 STEPS=run \
  bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh <out>/agent-q5f3
```

The tables, in the sandbox image (`docker run --rm --network none --user <uid>:<gid> -v <folder>:/w -w /w
agent-check:0.87.1 python3 -I …`), in a folder holding v3's two CSVs in `v3/`, R170a's collected `res/turns.csv` and
`res/attempts.csv` in `colA/res/`, R170b's in `col/res/`, and the rig's `server-q5f4t.log` (R170a's) and
`server-q5f3.log`, `meminfo.log` and `gpu.csv` (R170b's) in `rig/`. R170a's own tables were made the same way with one
pair (`merge_rows.py v3 col . q5f4t`, `host_summary.py rig R170a q5f4t` on its own `rig/`):

```bash
python3 -I merge_rows.py v3 colA . q5f4t col q5f3
python3 -I agent_tables.py . v3q5 q5f4t q5f3                              # agent.tsv
OUT_TAG=_ref_q5f3 python3 -I agent_tables.py . q5f3 q5f4t                 # agent_ref_q5f3.tsv
python3 -I realloc_counts.py q5f3=rig/server-q5f3.log > q5f3_reallocs.tsv
python3 -I realloc_counts.py --requests q5f3=rig/server-q5f3.log > q5f3_requests_reallocs.tsv
python3 -I host_summary.py rig R170b q5f3                                 # host.tsv's row
python3 -I prompt_by_size.py . v3q5 q5f4t q5f3 > prompt_by_size.tsv
python3 -I gen_without_task.py . bugfix-javascript-complex-numbers v3q5 q5f4t q5f3 > gen_without_complex.tsv
python3 -I per_task.py res/attempts.csv v3q5 q5f4t q5f3
```

`agent_tables.py` is [docs/data/2026-11-05_keepalive/run/agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py),
`realloc_counts.py` [scripts/realloc_counts.py](../../../scripts/realloc_counts.py), `host_summary.py` and
`prompt_by_size.py` are in [R159's run folder](../2026-11-14_r159_flint_release_agent/run/).

## Files

| File | Content |
|---|---|
| [agent.tsv](agent.tsv) | per run: passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes, minutes a task at equal work (v3's Q5 run's work) |
| [agent_ref_q5f3.tsv](agent_ref_q5f3.tsv) | the same for R170b and R170a at R170b's work |
| [gen_without_complex.tsv](gen_without_complex.tsv) | generation and prompt tok/s per run without JavaScript `complex-numbers`, and that task's turns, tokens, speed and deepest context |
| [prompt_by_size.tsv](prompt_by_size.tsv) | prompt tok/s by prompt size, all three runs |
| [q5f4t_reallocs.tsv](q5f4t_reallocs.tsv), [q5f4t_requests_reallocs.tsv](q5f4t_requests_reallocs.tsv), [q5f3_reallocs.tsv](q5f3_reallocs.tsv), [q5f3_requests_reallocs.tsv](q5f3_requests_reallocs.tsv) | graph reallocations by prompt size and per request, R170a and R170b |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, and each card's peak memory, R170a and R170b |
| [q5f4t_compare.md](q5f4t_compare.md), [q5f4t_report_table.md](q5f4t_report_table.md), [q5f4t_gonogo.md](q5f4t_gonogo.md), [q5f3_compare.md](q5f3_compare.md), [q5f3_report_table.md](q5f3_report_table.md), [q5f3_gonogo.md](q5f3_gonogo.md) | the harness's report for each run: per task, per turn and the go/no-go |
| [agent/q5f4t/](agent/q5f4t/), [agent/q5f3/](agent/q5f3/) | per request (`turns.csv`) and per attempt (`attempts.csv`) rows |
| [run/r170a.sh](run/r170a.sh), [run/r170b.sh](run/r170b.sh) | the runs |
| [run/merge_rows.py](run/merge_rows.py), [run/gen_without_task.py](run/gen_without_task.py), [run/per_task.py](run/per_task.py) | the table folder (v3's rows beside each run's), generation without one task, per task sums |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2` and v3.0. Model: [Flint](https://huggingface.co/RecursiveSI/flint)
(Apache-2.0) as my Flint Pascal Q5 GGUF ([models/flint](../../../models/flint/README.md)). The tasks are the agent
check's bug-fix set ([bench/agent_check](../../../bench/agent_check/README.md)); the tables hold times, counts and task
ids, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
