# Q551 step 1: pipelined speculation for Flash-Next on the rig, the agent check (2026-11-03)

The first step of the Flash-Next speed ladder ([Q551](../../../quants/experiments.md)): a coding agent fixing small
bugs against Swift-1.5 GSQ-RCO Q2_0 on the rig's nine P106 cards, with pipelined speculation at depth 3
([Q550](../2026-11-03_flash_next_pipelined/README.md): guessed tokens run on the cards that would otherwise wait)
against the same build without it. Before the timing, the same build passed the token-equality check on the rig's
cards and its memory per depth was measured; both are in [Q550's folder](../2026-11-03_flash_next_pipelined/README.md)
("Pipelined output = plain output" and "Measured on the rig").

## Result: depth 3 decodes 1.6× slower, the sessions take as long

The [agent check](../../../bench/agent_check/README.md): the pi coding agent fixes 12 small bugs (six Python, six
JavaScript), one attempt per task, at most 12 minutes each, no attempt started after an arm's 40 minutes. The client
ran pi, the proxy and the tasks' tests in containers on another machine, its requests relayed to the rig's loopback.

| Arm | Tasks done (passed) | Minutes, all tasks | Minutes per task, median | First token per turn, median (p90) | Prompt tok/s | Generation tok/s | Tool calls (malformed) | Runaways |
|---|---|---|---|---|---|---|---|---|
| no pipelining (`pipe0`) | 8 (7) | 42.3 | 3.2 | 7.6 s (18.2) | 61 | **7.9** | 36 (0) | 1 |
| `--pipe-depth 3` (`pipe3b`) | 8 (8) | 41.2 | 3.7 | 8.8 s (19.3) | 60 | **4.9** | 37 (0) | 0 |

- **Generation is 1.6× slower with pipelining:** 9,769 tokens in 1,990 s of decoding against 10,142 in 1,287 s
  (4.91 against 7.88 tok/s over the session's requests, from the server's timings). Per task the sessions come out
  even (0.96×, geometric mean over the 8 tasks both arms ran, range 0.63–1.39×; 41.2 against 42.3 minutes), because
  prompt processing, the tools and the fixed cost per request do not change and the arm without pipelining had one
  runaway (list-ops, to its 12-minute cap). Without that task: 32.6 against 30.3 minutes, 0.93×.
- **The arm without pipelining reproduces Q215's v3 arm** (7.8 tok/s, 3.0 minutes per task, first token 7.7 s,
  [Q215](../2026-11-03_flash_next_baseline/README.md)): the branch costs nothing when pipelining is off.
- **The guesses** ([guesses.csv](agent/guesses.csv), [requests.csv](requests.csv)): `cont` (copy ≥ 4, else the
  request's continuation counts) made 14,720 guesses at depth 3, 2,900 right and 3,900 wrong; the rest were dropped
  behind a wrong one (two per wrong guess at depth 3). Right guesses came to 29.7 % of the generated tokens and wrong
  ones to 39.9 %; C55's event model had predicted 39.2 % and 40.3 % for this guesser at depth 3, and 1.31× for agent
  sessions.
- **A wrong guess costs about three plain tokens.** Fitted over the 40 requests (R² 0.967), a request's decode time
  is 103 ms per generated token, −145 ms per right guess and **+372 ms per wrong guess**. A plain token takes 127 ms
  here, so a wrong guess delays the stream by ~2.9 tokens' time, about what the work queued behind it at depth 3 (the
  wrong guess and the two behind it) would take if it still ran on every card ahead of the corrected token. The event
  model let dropped guesses use idle cards instead (117 card-ms per token, overlapped). Where the time goes is not
  measured yet: the cards' queues, the rollback, the host loop on the two Celeron cores (Q593).
- **The tool calls run unpipelined:** llama-server leaves the pipeline while a lazy grammar is active, so the tool-call
  bodies, 23 % of the generated tokens in both arms (2,359 of 10,142; 2,141 of the 2,181 tokens sampled outside the
  pipeline at depth 3), decode at the plain rate. They include the edits, the text that copy guesses get right most
  often (Q594).
- **Host memory is tighter:** the lowest MemAvailable was 227 MB without pipelining and 138 MB with it; MemAvailable
  minus CmaFree (what pinned buffers can use, Q560) averaged 205 against 114 MiB, swap peaked at 845 against
  1,484 MiB, memory pressure stayed low (PSI full avg10 at most 13 and 16 %) ([host_memory.csv](host_memory.csv)).
  The first depth-3 arm (`pipe3`) was stopped 18 s into its second request by the harness's watchdog
  (MemAvailable 139 MiB, below its 150); `pipe3b` is the same server line with the watchdog at 60 MiB and
  `mem_guard.sh` as the stop, which never fired.
- The report's NO-GO flags ([gonogo.md](agent/gonogo.md)): memory (above), and the runaway's proxy row without a
  reply. Every tool call parsed (73, none malformed); the cache hit 96.1 % of the prompt from the second turn on.

**Pipelining stays off on the release line** until a wrong guess costs less than a right one saves (Q593). Depths 1
and 2 and a guesser that guesses less often but more surely were not run; they are timed once Q593 shows where a
wrong guess's time goes (Q595).

## Files

| File | What |
|---|---|
| `agent/compare.md` | the report: per task result, minutes, turns and tool calls per arm; per arm the summary; first token per turn |
| `agent/table.md`, `agent/gonogo.md` | the harness's per-arm table (prefill, decode, fixed and other seconds per task) and its go/no-go checks |
| `agent/attempts.csv`, `agent/turns.csv` | per attempt and per turn: tokens, cache, times, tool calls (bench/agent_check's columns) |
| `agent/guesses.csv` | per arm the requests, those pipelined, generated tokens, guesses, right, wrong |
| `requests.csv` | per request: prompt tokens and ms, generated tokens and ms, ms per token, guesses, right, wrong, tokens sampled outside the pipeline and how many with the tool grammar active (from the server logs; `pipe3` is the one request before the watchdog stopped it) |
| `host_memory.csv` | per arm while the server ran: MemAvailable minus CmaFree (mean, lowest), MemAvailable (lowest), swap (highest), the most swapped out in 10 s, memory pressure |
| `run/agent_check.sh` | the check: a quiet check of the rig and `mem_guard.sh` per arm, each arm through `release_check_s12.sh`, the report, the guess counts |
| `run/rig_tables.py` | `requests.csv`, `host_memory.csv` and the fit, from the rig's server logs and the guard's logs |

## How it was made

On 2026-11-03, 17:52–19:44, on the first rig (nine P106-090 6 GB cards on PCIe 1.0 x1, a two-core Celeron, 4 GB of
host RAM, the desktop running), nothing else running on it (a quiet check before each arm, logged). The engine is
the fork branch `pascal-p106-v3-hybrid` at `f377294d6`, the rig tarball
`llama.cpp-20261103-f377294d6-pascal-p106-v3-hybrid.tar.gz` (md5 `8e53e99a8293d18bc1c8df37ad2c0815`). Both arms run
Q215's v3 line, `-ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format
deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio`, the
depth-3 arm with `--pipe-depth 3` (guesser `cont`, the default). The model is Swift-1.5 GSQ-RCO Q2_0 as three shards
(the n-gram table in the third, on the rig's internal SSD), from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
`99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d` (Swift Open License 1.0 and qwen-community-1.0). The tasks are
bench/agent_check's bugfix set; the client image `agent-check:0.87.1`.

```bash
# the machine values in a private env file: RIG, CLIENT, ports, CLIENT_DIR, CLIENT_OUT, RIG_LOGDIR, REL_BIN, MODEL,
# CTX=65536, TIME_CAP_MIN=12, BUDGET_MIN_<arm>=40, CPU_SHARES=1024, MIN_AVAIL_MB=150, GUARD_LOG=/dev/shm/...,
# BENCH=bench/agent_check, and per arm
#   ARGS_pipe0="-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio"
#   ARGS_pipe3b="$ARGS_pipe0 --pipe-depth 3"
set -a; . <env>; set +a
# pipe0, then pipe3 (ARGS_pipe3 = ARGS_pipe3b), which the watchdog stopped at 150 MiB
ARMS='pipe0 pipe3' STEPS='prep arms' bash docs/data/2026-11-03_flash_next_pipelined_rig/run/agent_check.sh <out>
STEPS=arms ARMS=pipe3b MIN_AVAIL_MB=60 bash docs/data/2026-11-03_flash_next_pipelined_rig/run/agent_check.sh <out>
STEPS=report ARMS='pipe0 pipe3b' bash docs/data/2026-11-03_flash_next_pipelined_rig/run/agent_check.sh <out>
# on the client, over the rig's logs the check collected and the guard's logs
python3 -I docs/data/2026-11-03_flash_next_pipelined_rig/run/rig_tables.py <out>/rig <out> <tables> pipe0 pipe3 pipe3b
```

`agent_check.sh` uses `release_check_s12.sh` and `mem_guard.sh` from
[2026-11-03_flash_next_mtp/run/](../2026-11-03_flash_next_mtp/run/).
