# Ornith's v4 preset without the eager keep-alive stop on the agent check (R169, Q1496; 2026-11-16)

R169 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix
tasks, Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget an arm, the makers' sampling at
temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF (valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`,
48,196,722,592 bytes) at 49,152 tokens on nine P106-090 cards. It uses R166's method and started command
([2026-11-16_r166_ornith_stack_no_taper](../2026-11-16_r166_ornith_stack_no_taper/README.md)). Two runs of one arm,
one load a run: **R169a `o4nea`** and **R169b `o4neb`**.

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

**The answer: dropping the eager stop changes nothing on the agent check's time per task.** On the same work a task
took 2.12 and 2.18 minutes without it (R169a, R169b) and 2.11 and 2.13 minutes with it (R166a, R166b): +1.4 % on the
means, inside the 0.06-minute gap between the two runs without it, so negligible. Underneath, two effects cancel out.
Short prompts got slower: a request under 512 tokens took 3.41 and 3.33 s of prompt against 3.14 and 3.15 s, and the
prompt rate, which these many short requests dominate, fell from 103.0 and 105.9 to 100.2 and 96.8 tok/s (−5.7 %, a
loss). Long prompts got a little faster, as in R167a's deep lane: the slowest first tokens (p90) came at 10.71 and
10.08 s against 11.29 and 11.32 s (−8.0 %, a gain), and a task's first turn at 8.30 and 8.35 s against 8.43 and
8.45 s (−1.4 %, a gain). Generation did not change (16.58 and 16.17 tok/s against 16.52 and 16.13). All four runs
passed 12 of 12. The settings file keeps the eager stop until the owner decides.

## The runs

| Run | Arm | Rig | Server started, up | Tasks | Passed | Started command |
|---|---|---|---|---|---|---|
| R169a | `o4nea` | rig 2 | 19:57:22, up after 656 s | 20:08:22–20:34:49 | 12 / 12 | 493 bytes, md5 `d4d0418461573353dc37d7dba8e0fcbb` |
| R169b | `o4neb` | rig 2 | 20:42:09, up after 653 s | 20:53:05–21:23:43 | 12 / 12 | 493 bytes, md5 `d4d0418461573353dc37d7dba8e0fcbb` (R169a's byte for byte) |

## The class over four runs

R169's two runs without the eager stop (`o4nea`, `o4neb`) against R166's two runs of the released preset (`o4sna`,
`o4snb`), with the same started command apart from that one variable, on the same rig, file and build. Each value is
one run ([agent.tsv](agent.tsv), [class.tsv](class.tsv), made by [run/class_r169.py](run/class_r169.py)). Minutes a
task at equal work price the same work, `ka`'s (v3.0's 2026-11-06 run on the same rig, file and line: 2.52), at each
run's own speeds. The class compares the means of an arm's two runs, as R160's and R166's: a gain or a loss when the
difference is larger than the larger of the two arms' spreads (the gap between an arm's two runs), negligible
otherwise.

| | released preset `o4sna` (R166a) | released preset `o4snb` (R166b) | no eager stop `o4nea` (R169a) | no eager stop `o4neb` (R169b) | no eager stop against the released preset |
|---|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | |
| minutes a task at equal work | 2.11 | 2.13 | 2.12 | 2.18 | **negligible (+1.4 %; spreads 0.02, 0.06)** |
| generation tok/s | 16.52 | 16.13 | 16.58 | 16.17 | negligible (+0.3 %; 0.39, 0.41) |
| prompt tok/s | 103.0 | 105.9 | 100.2 | 96.8 | **loss (−5.7 %; 2.9, 3.4)** |
| first token median s | 6.69 | 6.06 | 6.74 | 6.43 | negligible (+3.3 %; 0.63, 0.31) |
| first token p90 s | 11.29 | 11.32 | 10.71 | 10.08 | **gain (−8.0 %; 0.03, 0.63)** |
| first token on a task's first turn, median s | 8.43 | 8.45 | 8.30 | 8.35 | **gain (−1.4 %; 0.02, 0.05)** |
| prompt s for the requests under 512 tokens; requests | 116.3; 37 | 144.8; 46 | 126.2; 37 | 133.0; 40 | |
| minutes, all 12 tasks; turns | 29.1; 63 | 36.8; 73 | 26.3; 62 | 30.5; 63 | different work |
| tool calls / failed / malformed | 63 / 2 / 0 | 73 / 2 / 0 | 62 / 1 / 0 | 63 / 1 / 0 | |
| graph reallocations | 0 in 63 requests | 0 in 73 | 0 in 62 | 0 in 63 | |
| lowest MemAvailable MiB | 1,225 | 1,228 | 1,213 | 1,268 | |

- **Time per task: no change.** The means are 2.15 minutes without the eager stop and 2.12 with it, a difference of
  0.03 against spreads of 0.02 and 0.06. Both runs without it sit at or above the released preset's mean, so it does
  not help either.
- **Short prompts are slower without the eager stop** ([r169_reallocs.tsv](r169_reallocs.tsv), the server's prompt
  seconds per size of prompt; averages over different prompts, not classed). A request under 512 tokens took 3.41 and
  3.33 s (126.2 s for 37, 133.0 s for 40) against 3.14 and 3.15 s (116.3 s for 37, 144.8 s for 46). These are most of
  the requests after a task's second turn, so they set the prompt rate (the loss above).
- **Longer prompts are a little faster without it**, in the same direction as the deep lane (+2.4 % on all prompt
  chunks from 4,096 tokens, [R167a](../2026-11-16_r167_keepalive_depth/README.md#r167a-ornith)). Prompts of
  512–1,023 tokens took 8.39 and 8.43 s each (92.3 s for 11, 84.3 s for 10) against 8.50 and 8.64 s (93.5 s for 11,
  95.0 s for 11), and 1,024–1,999 tokens 9.97 and 10.01 s (99.7 s for 10, 100.1 s for 10) against 10.16 and 10.16 s
  (101.6 s for 10 in both). The slowest first tokens (p90) and a task's first turn (a median 789 new prompt tokens)
  are gains.
- **What it means for the preset:** the agent check prices the two effects together, and they cancel out. By R169's
  rule a negligible result argues for dropping the eager stop; the prompt loss on short requests argues for keeping
  it. [release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env) is unchanged: the owner decides.
- **The longer loads (656 and 653 s, against R166's 539 and 538 s) are not the eager stop's.** On the same rig, R167a's
  deep lane loaded the released preset in 541 s at 17:34 and in 658 s at 18:41: from 18:18 on, every load there took
  648, 658, 654 or 657 s whatever the arm. The cause was not looked at. The server's start is outside the tasks'
  time.
- **Host** ([host.tsv](host.tsv)): every run's swap peaked during the load (338, 337, 337 and 337 MiB, from 31, 31,
  31 and 30); the lowest MemAvailable came during the last task. Card peaks are the released preset's to the MiB
  (`o4nea` = `o4sna`, `o4neb` = `o4snb`). The RAM watchdog never stopped a server; no graph reallocation in 261
  requests.

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
  2.13). The class over both runs is in "The class over four runs" above.
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

## R169b: `o4neb`, the released preset without the eager stop, second run (2026-11-16)

- **The started command** is R169a's byte for byte (493 bytes, md5 `d4d0418461573353dc37d7dba8e0fcbb`; with
  ` GGML_CUDA_KEEPALIVE_EAGER=1` put back after `GGML_CUDA_KEEPALIVE=decode` it is R166b's, md5
  `211f8db2edeacd267705795e4911d8c9`), with one `-b`, 4096. The server's `switches:` line, read once during the load,
  says `5 set: GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_KEEPALIVE=decode LLAMA_SAMPLING_TOP_K_ROWS=1
  LLAMA_SERVER_YIELD=2`, as R169a's, and the eager stop appears nowhere in the server's log. The run copy is `v4` at
  `c981c3f`, whose harness, bench and settings file equal R169a's copy (`diff -r` empty).
- **Timeline:** server started 20:42:09, up after 653 s; the client reached it 20:53:03; 12 tasks 20:53:05–21:23:43;
  rig logs copied 21:23:57; report and collect 21:24:31; rig clean 21:24:34.

**Numbers** (one run each; [agent.tsv](agent.tsv), [host.tsv](host.tsv), [r169_reallocs.tsv](r169_reallocs.tsv)):

| | released preset `o4sna` (R166a) | released preset `o4snb` (R166b) | no eager stop `o4nea` (R169a) | **no eager stop `o4neb` (R169b)** |
|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.11 | 2.13 | 2.12 | **2.18** |
| minutes, all 12 tasks (wall) | 29.1 | 36.8 | 26.3 | **30.5** |
| minutes a task, median | 2.15 | 2.34 | 1.96 | **1.72** |
| turns | 63 | 73 | 62 | **63** |
| generation tok/s | 16.52 | 16.13 | 16.58 | **16.17** |
| prompt tok/s | 103.0 | 105.9 | 100.2 | **96.8** |
| first token median, p90 s | 6.69, 11.29 | 6.06, 11.32 | 6.74, 10.71 | **6.43, 10.08** |
| first token on a task's first turn, median s | 8.43 | 8.45 | 8.30 | **8.35** |
| prompt s for the requests under 512 tokens; requests | 116.3; 37 | 144.8; 46 | 126.2; 37 | **133.0; 40** |
| tool calls / failed / malformed | 63 / 2 / 0 | 73 / 2 / 0 | 62 / 1 / 0 | **63 / 1 / 0** |
| graph reallocations | 0 in 63 requests | 0 in 73 | 0 in 62 | **0 in 63 requests** |
| lowest MemAvailable MiB | 1,225 | 1,228 | 1,213 | **1,268** |
| server up after, s | 539 | 538 | 656 | **653** |

- **Where the prompt time went** ([o4neb_reallocs.tsv](o4neb_reallocs.tsv)): requests under 512 tokens took 133.0 s
  for 40 requests (3.33 s each), 512–1,023 tokens 84.3 s for 10, 1,024–1,999 tokens 100.1 s for 10, 2,000 or more
  55.0 s for 3. All 63 requests: 372.4 s of prompt.
- **First token per turn** ([o4neb_compare.md](o4neb_compare.md), the median over tasks): turn 1 8.35 s for 789 new
  prompt tokens, turn 2 10.03 s for 1,452, turn 3 3.05 s for 126, turn 4 3.98 s for 172, turn 5 1.43 s for 34.
- **Where the wall minutes went** ([o4neb_report_table.md](o4neb_report_table.md)): `o4neb`'s attempts spent a mean
  119 s a task generating (`o4nea` 96, `o4sna` 110, `o4snb` 143) and 31 s reading prompts (33, 33, 39), and processed
  3,003 prompt tokens a task (3,299, 3,405, 4,130). Python `scale-generator` took 10.1 minutes in 9 turns (`o4nea`
  5.2 in 5, `o4sna` 4.5, `o4snb` 10.8), JavaScript `variable-length-quantity` 3.3 in 9, Python `list-ops` 3.2 in 4,
  Python `pig-latin` 2.6 in 5 and Python `affine-cipher` 2.5 in 6; every other task took 1.8 minutes or less. No task
  came near the 12-minute cap.
- **Graph reallocations** ([o4neb_reallocs.tsv](o4neb_reallocs.tsv),
  [o4neb_requests_reallocs.tsv](o4neb_requests_reallocs.tsv)): 0 in every bucket.
- **Host** ([host.tsv](host.tsv), R169b's own lines of `meminfo.log` and `gpu.csv`, from 20:42:09): lowest
  MemAvailable 1,268 MiB at 21:20:41, during the last task; 1,536 MiB on the last line (21:23:55, as the server
  stopped). Swap 30 MiB at the start, 337 MiB at its highest at 20:48:45, during the load. The report's memory gate
  ([o4neb_gonogo.md](o4neb_gonogo.md)) reads the client's log shared with R160's, R166's and R169a's runs: its
  718 MB low is R160d's, its 341 MB swap high R160c's. NO-GO on the swap growth, as for those runs. The RAM watchdog
  did not stop the server. Card peaks MiB: 5,755 5,771 5,701 5,771 5,381 5,489 5,649 5,721 5,127, R166b's to the MiB.
- **Server log:** 11 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill ([o4neb_gonogo.md](o4neb_gonogo.md)).
- The report's files: [o4neb_compare.md](o4neb_compare.md), [o4neb_report_table.md](o4neb_report_table.md),
  [o4neb_gonogo.md](o4neb_gonogo.md); the per-request and per-attempt rows: [agent/o4neb/](agent/o4neb/).

## Files

| File | What it holds |
|---|---|
| [agent.tsv](agent.tsv) | per arm: passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes for all tasks, a task's median, minutes a task at equal work against `v3ka` (R160's rows, R166's two, `o4nea`'s and `o4neb`'s) |
| [class.tsv](class.tsv) | the class over four runs: each measure's four values, the two arms' means and spreads, the difference and the class |
| [host.tsv](host.tsv) | R166a's, R166b's, R169a's and R169b's lowest MemAvailable and its time, swap at the start and highest, MemAvailable on the run's last line, each card's peak memory |
| [r169_reallocs.tsv](r169_reallocs.tsv) | R166's two runs, R169a and R169b: graph reallocations and prompt seconds per prompt size |
| [o4nea_compare.md](o4nea_compare.md) | the report: per task and per arm, first token per turn |
| [o4nea_report_table.md](o4nea_report_table.md) | per arm over the client's attempts (R160's six runs, R166's two and R169a): pass@1, minutes, prefill, decode, prompt tokens, cache hit |
| [o4nea_gonogo.md](o4nea_gonogo.md) | the report's gates and each server log's counts |
| [o4nea_reallocs.tsv](o4nea_reallocs.tsv), [o4nea_requests_reallocs.tsv](o4nea_requests_reallocs.tsv) | R169a's graph reallocations and prompt seconds per prompt size, and per request |
| [agent/o4nea/](agent/o4nea/) | `turns.csv` (one row a request) and `attempts.csv` (one row a task) |
| [o4neb_compare.md](o4neb_compare.md), [o4neb_report_table.md](o4neb_report_table.md), [o4neb_gonogo.md](o4neb_gonogo.md) | R169b's report: as R169a's above (the report table over R160's six runs, R166's two, R169a and R169b) |
| [o4neb_reallocs.tsv](o4neb_reallocs.tsv), [o4neb_requests_reallocs.tsv](o4neb_requests_reallocs.tsv) | R169b's graph reallocations and prompt seconds per prompt size, and per request |
| [agent/o4neb/](agent/o4neb/) | `turns.csv` (one row a request) and `attempts.csv` (one row a task) |
| [run/r169.sh](run/r169.sh) | runs one arm, `o4nea` or `o4neb` |
| [run/host.py](run/host.py) | the host table from the rig's `meminfo.log` and `gpu.csv` (R166's) |
| [run/class_r169.py](run/class_r169.py) | the class over four runs from `agent.tsv` |

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

R169b was started the same way with `o4neb` in place of `o4nea`, from a copy of `v4` at `c981c3f`, and its report with
`ARMS=o4neb`. Its tables ran in the same image from R169a's table folder plus `o4neb`'s rows of `res/turns.csv` and
of `res/attempts.csv` (only those whose `tag` starts with `o4neb/`), its server log and the lines of `meminfo.log` and
`gpu.csv` that R169b added (from 20:42:09): `agent_tables.py OUT v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb o4nea
o4neb` (the earlier arms' rows come out as R169a's), `scripts/realloc_counts.py o4neb=rig/server-o4neb.log` (default
and `--requests`), `scripts/realloc_counts.py o4sna=… o4snb=… o4nea=… o4neb=…` for
[r169_reallocs.tsv](r169_reallocs.tsv) (its first three arms' rows come out as R169a's), `run/host.py <those lines>
R169b o4neb` (over R169a's, R166a's and R166b's lines it gives their rows again), and `run/class_r169.py OUT` for
[class.tsv](class.tsv).

## Provenance and licence

The tasks are the agent check's 12 bug fixes, derived from Exercism's Python and JavaScript exercises (MIT; source,
licences and commits in [bench/agent_check](../../../bench/agent_check/README.md)); this folder holds only their IDs
(in [agent/o4nea/attempts.csv](agent/o4nea/attempts.csv)). The tables summarize this project's own runs and are
published under the repository's licence.
