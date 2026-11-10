# A generated-token mark with a fresh attempt, priced on the recorded agent checks (FN58, Q1200; 2026-11-11)

What this is: the price of a stop rule before any run. An attempt of the agent check
([bench/agent_check](../../../bench/agent_check/README.md)) is stopped once it has generated a set number of tokens
(the mark), and the task starts again from nothing. Published on SWE-bench Verified, a stop on the step count alone
saved 11.4 % of the tokens at 5 % false stops, and a fresh attempt after the stop kept the resolve rate
([tokens per solved task](../../research/notes/tokens_per_solved_task.md), section 4). Registry row Q1200 in
[quants/experiments.md](../../../quants/experiments.md).

The rule is in the runner now (`run.py --gen-mark`, `--gen-budget`, `--mark-tries`; the README's
[token mark](../../../bench/agent_check/README.md#the-token-mark-and-a-fresh-attempt)). This folder holds what the
rule would have done to every attempt the agent check recorded on the rigs from 2026-10-26 to 2026-11-06 and on the
quant machine on 2026-11-10, read back from each attempt's own event log by
[token_mark.py](../../../bench/agent_check/token_mark.py) `price`. No model ran for it.

## Result

**On the agent check as recorded, a mark of 16,384 to 65,536 generated tokens has almost nothing to stop. Of 357
Flash-Next attempts (three Swift Q2 files, the 12 bug-fix tasks), the longest generated 19,770 tokens, a pass. A mark
of 16,384 stops that one pass and saves 0.45 % of the tokens and no rig minute; 24,576 or more stops nothing. Lower
marks stop more passes than failures: 8,192 stops 8 passes and 4 failures and saves 4.59 % of the tokens and 1.72 %
of the rig minutes. With a fresh try at the same budget the estimate keeps the passes (est. 346.64 against 346) but
costs 2.76 % more tokens (est. 733,146 against 713,462) and 3.29 % more rig minutes (est. 1,141.1 against 1,104.8). So
the rule shows no gain on this task set, and arms at equal budget here could not show one either (Q1296). It belongs
on tasks whose attempts run long: the 82 polyglot tasks (Q1146), the deep agent lane (Q1144) or the
[Elixir SWE Bench](https://github.com/Cinderella-Man/elixir-swe-bench), where 7 and 5 of 20 attempts end at its
65,536-token budget ([tokens per solved task](../../research/notes/tokens_per_solved_task.md)).**

### Flash-Next per mark (Swift GSQ-RCO Q2_0, its hcq8 retype and the Pascal Q2 GGUF)

357 attempts (309 on the rigs, 48 on the quant machine), 346 passed, 14 stopped by the time cap, 713,462 tokens
generated, 1,104.8 minutes of rig attempts. The fresh try is est., with two tries per attempt (a budget of twice the
mark).

| Mark | Attempts over it | Passes lost | Failures cut | Tokens saved | Rig minutes saved | Budget with a fresh try | Passes, est. | Tokens, est. | Rig minutes, est. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 110 | 100 | 10 | 230,280 (32.28 %) | 246.7 (22.33 %) | 4,096 | 281.87 | 693,697 | 1,183.5 |
| 4,096 | 33 | 24 | 9 | 113,211 (15.87 %) | 96.5 (8.73 %) | 8,192 | 344.59 | 696,671 | 1,131.8 |
| 8,192 | 12 | 8 | 4 | 32,724 (4.59 %) | 19.0 (1.72 %) | 16,384 | 346.64 | 733,146 | 1,141.1 |
| 12,288 | 2 | 2 | 0 | 7,606 (1.07 %) | 0.3 (0.03 %) | 24,576 | 345.43 | 718,112 | 1,115.2 |
| 16,384 | 1 | 1 | 0 | 3,245 (0.45 %) | 0.0 | 32,768 | 345.76 | 716,553 | 1,110.2 |
| 24,576, 32,768, 49,152, 65,536 | 0 | 0 | 0 | 0 | 0.0 | 49,152 to 131,072 | 346 | 713,462 | 1,104.8 |

- **The long attempts are few, and the caps' time is not long generation.** 12 of 357 attempts reached 8,192
  tokens; 8 of them passed. The 14 attempts the time cap stopped hold 171.9 of the 1,104.8 rig minutes and 102,320
  of the 584,492 tokens generated on the rigs, and 8 of them stopped before 8,192 tokens: on the rig the minutes go to
  reading the growing context and to the turns, so a token mark low enough to cut them stops passes first.
- **The files are alike.** Swift GSQ-RCO Q2_0 (178 attempts, the bug-fix tasks 7 to 18 times each): the longest
  attempt 19,770 tokens on the quant machine (FN50's `pv1` 1), the longest on the rig 10,106. Swift Qwen3.8-Flash-Next
  Pascal Q2 GGUF (pmix-q2-fast; 167 attempts, 13 to 14 times each): the longest 12,534, a pass the time cap stopped. The
  hcq8 retype of Swift GSQ-RCO Q2_0 (one run, 12 attempts): none reached 4,096. Per file in [price.md](price.md).
- **Why the fresh try costs tokens here:** a stopped attempt has spent the mark before it starts again, and on this
  set most stopped attempts would have passed (8 of 12 at 8,192), so the second try buys back passes the first try
  was about to make.

### Flint and Ornith

- **Flint** (360 attempts of five files, 348 passed, the bug-fix tasks): the longest attempt generated 28,180 tokens
  (Flint Pascal Q5 GGUF, a pass the time cap stopped). A mark of 16,384 stops 18 attempts, 13 passes and 5 failures,
  and saves 4.57 % of the tokens and 4.51 % of the rig minutes; with a fresh try at 32,768 the estimate gives 341.91
  passes against 348 at 11.09 % more tokens (1,705,130 against 1,534,929) and 9.91 % more rig minutes (1,002.2
  against 911.8). A mark of 24,576 stops 2 attempts, one pass and one failure.
- **Ornith** (159 attempts of three files, 152 passed): the longest attempt generated 7,472 tokens, so no mark from
  8,192 up stops anything.

## Method

- **Input:** every scored attempt of 93 recorded runs ([run/runs.tsv](run/runs.tsv): the run, the model file, its
  family, the machine and the run's folder in the raw results). Each attempt's `result.json` (pass, caps, wall time)
  and `events.jsonl` (pi's event stream with arrival times). Report folders that copied an earlier run's attempts
  list them again; an attempt already seen (same arm, task, attempt, start and wall time) is read once, so 80 runs
  remain. Every run is the agent check's 12 bug-fix tasks, one attempt per task, with the caps of its day: 40 turns
  and 12 or 15 minutes on the rig, 12 minutes on the quant machine.
- **The count:** the one run.py's mark watches. The server's own token count of every finished reply (pi's
  `usage.output`, llama-server's `completion_tokens`), plus one per streamed piece (`*_delta` event) of the reply in
  flight. Over all attempts a piece held 1.037 tokens (a reply cut off by a cap counted at one token a piece), so
  the count follows the server's closely and does not run ahead of it. A piece holds more than one token where a
  reply is short and its markup streams no piece (up to 2.6 in one attempt), which is why a reply cut off counts one
  token a piece rather than its attempt's rate.
- **The stop:** for each mark, the first event at which the count reaches it. A passed attempt stopped there is a
  pass lost: its workspace at the stop is counted a fail (the recorded logs cannot rebuild it). A failed one is a
  failure cut short. Tokens saved are what the attempt generated after the stop, and on rig runs the minutes saved
  are its wall time after the stop. Attempts a cap stopped (`capped`) would have run longer without it, and what they
  would have generated is not known.
- **The fresh try, est.:** with two tries per attempt (a budget of twice the mark), per task the share a of the
  model's recorded attempts that pass under the mark and the share s that reach it, the tries taken as independent:
  expected passes a (1 + s), tokens and minutes the mean cost of a try cut at the mark times (1 + s). A second try
  stopped at the mark counts a fail. Tasks repeat across runs of one file (each of the 12 tasks 7 to 18 times on
  Swift GSQ-RCO Q2_0, 13 to 14 times on the Pascal Q2), so the shares are per task, not pooled.
- **Not here:** speed. Rig minutes are the recorded wall time of rig attempts, summed; the quant machine's attempts
  count for tokens and passes only.

## Files

| File | What it holds |
|---|---|
| [price.md](price.md) | the tables per model file and per family, all marks (the printout of `token_mark.py price`) |
| [price.tsv](price.tsv) | the same rows: attempts, passed, capped, generated, attempts over the mark, passes lost, failures cut, tokens and rig minutes saved, and the fresh-try estimate per mark |
| [attempts.tsv](attempts.tsv) | one row per attempt: run, model, family, machine, task, pass, cap, turns, generated tokens, wall seconds, streamed pieces, tokens per piece of its finished replies, and per mark the tokens and seconds at the stop (empty where the mark is not reached) |
| [run/runs.tsv](run/runs.tsv) | the run list: run, model, family, machine, folder |
| [run/price.sh](run/price.sh) | the pricing and the agent check's self-test, as run |

## Commands

On a machine with the recorded runs' folders under `DATA` (as `run/runs.tsv` names them) and Docker for the
self-test's sandbox image (`agent-check:0.87.1`, built from
[bench/agent_check/docker/Dockerfile](../../../bench/agent_check/docker/Dockerfile)):

```bash
REPO=<checkout of this repository> DATA=<recorded runs> OUT=<output folder> SANDBOX=docker \
  nice -n 19 bash docs/data/2026-11-11_fn58_stop_and_retry/run/price.sh
```

It writes `OUT/price/` (this folder's three tables) in seconds, then runs
[selftest.sh](../../../bench/agent_check/selftest.sh) (about 1.5 minutes), which passed all 31 checks; the 15 of the
token mark run pi against a mock server whose agent loops without writing the fix on chosen tries:

- `--gen-budget 9000 --mark-tries 3` stops each looping try at 3,000 tokens (3,090, 3,088, 3,089 and 3,102 counted);
- a fresh try waits behind every try already queued (starts in the order: task 1, task 2, task 3, task 2's second
  try, task 3's second, task 3's third);
- a task looping on its first try only passes on try 2, one looping on every try is scored as it stands after the
  third, and a task under the mark passes on try 1;
- the row sums the tokens and seconds of all tries, the log counts the passes by try, and a second run resumes;
- `--gen-mark` alone stops the try without a fresh one, and without a mark the looping task runs to the turn cap;
- `token_mark.py price` on that run stops the looping attempt at 3,067 tokens where run.py stopped at 3,100, loses
  no pass at 3,000 and the passing task's pass at 50, and gives one group per model and family of its run list;
- `token_mark.py arms` counts passes by the try that passed.

## Provenance and licence

The attempts are this project's own agent-check runs: the pi coding agent (0.87.1) on Exercism's exercises (MIT),
with Flint, Ornith-1.5-122B and Qwen3.8-Flash-Next files served by llama.cpp builds of this project. The tables
hold counts and times only, no task text and no model output. The raw event logs stay out of the repository (about
860 MB). Tables and scripts: this repository's licence.
