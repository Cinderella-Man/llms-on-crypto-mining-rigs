# Probabilistic verification on the agent check: passes, turns and tokens per task (Q1183; 2026-11-10)

What this is: whether a coding agent still gets its tasks done, and in how many turns and tokens, when the Miner
Llama v3 release preset ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env):
Qwen3.8-Flash-Next, Swift Q2_0, an MTP draft of three tokens) verifies its drafts by rejection sampling:

- **`pv0`** is the preset: the draft takes its first choice, and the main model keeps a drafted token only when its
  own sample is the same token;
- **`pv1`** adds `--spec-draft-sampling probabilistic --spec-draft-p-min 0`: the draft samples its tokens, and the
  main model accepts a drafted token with probability min(1, p/q).

The agent check's 12 bug-fix tasks ([bench/agent_check](../../../bench/agent_check/README.md), `tasks/bugfix`) ran
live, one slot and one task at a time, in four runs alternated `pv0 pv1 pv0 pv1`, each on a fresh server load.
Registry row Q1183 in [quants/experiments.md](../../../quants/experiments.md), the quality half of Q731.

Why: on the release session's recorded requests `pv1` yields 7.74 % more tokens a step
([2026-11-10_prob_verify_acceptance](../2026-11-10_prob_verify_acceptance/README.md), Q1173), and its replays also
generated more tokens for the same requests. A replay caps each reply at the recorded length, so it cannot say
whether a task takes more tokens or turns. Passes, turns and tokens depend on the models, the sampler and the text,
not on the card, so they were counted on a machine with newer cards. No time is read or reported here.

## Result

**Every task passed in all four runs, and `pv1` took no more turns: 64 and 60 against the preset's 92 and 67. In
generated tokens `pv1`'s runs are 35,406 and 36,683 against 34,945 and 21,936 (+26.74 % between the means), which
is inside the 13,009 tokens between the preset's own two runs. The whole difference is one task's second reply:
17,785 and 9,445 tokens with `pv1` against 2,872 and 4,071. Without that task the runs generated 15,636 and 25,105
tokens against 26,086 and 17,059. Twelve tasks and two runs an arm cannot tell that from chance, so the reply
length under `pv1` is its own row, Q1184.**

### Every run

| Run | Passed | Turns | Generated | Tool calls | Tool errors | Accepted of drafted | Share accepted | Tokens a step |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `pv0` 1 | 12 of 12 | 92 | 34,945 | 87 | 0 | 23,129 of 35,229 | 0.6565 | 2.957 |
| `pv1` 1 | 12 of 12 | 64 | 35,406 | 59 | 1 | 24,874 of 31,644 | 0.7861 | 3.362 |
| `pv0` 2 | 12 of 12 | 67 | 21,936 | 60 | 2 | 14,752 of 21,399 | 0.6894 | 3.053 |
| `pv1` 2 | 12 of 12 | 60 | 36,683 | 56 | 0 | 25,665 of 33,114 | 0.7750 | 3.329 |
| the release session on the rig (the preset, live) | 12 of 12 | 66 | 27,067 | | | 18,447 of 25,704 | 0.7177 | 3.140 |

- No attempt hit the turn cap, the time cap or the reply limit, no tool call was malformed, and no attempt was
  re-run for an infrastructure failure. The three tool errors are single failed calls in passed attempts.
- Tokens a step is generated tokens over steps (generated minus accepted draft tokens), summed over the run's
  requests before dividing.
- By the server logs every sampled token was drawn on the device (34,964 of 34,964 and 21,952 of 21,952 in the
  `pv0` runs, 35,486 of 35,486 and 36,763 of 36,763 in the `pv1` runs), and the grammar rejected 4 and 2 device
  draws in the `pv0` runs and 22 and 26 in the `pv1` runs. No log holds an error line.
- The release session's row is the recorded session of
  [2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md), arm `t1swift`.

[runs.tsv](runs.tsv); every task of every run in [tasks.tsv](tasks.tsv), every request in
[requests.tsv](requests.tsv).

### `pv1` against the preset

An arm's value is the mean of its two runs. "Apart" is the difference between an arm's two runs. The interval is a
95 % bootstrap over the 12 tasks of the difference between the arms' sums. Higher, lower, equal: the tasks where
`pv1`'s mean is above, below or at the preset's. [compare.tsv](compare.tsv).

| Measure (sum over the 12 tasks) | `pv0` 1, 2 | `pv1` 1, 2 | Difference | % | `pv0`'s runs apart | `pv1`'s runs apart | Inside `pv0`'s | 95 % over tasks | Higher, lower, equal |
|---|---|---|---:|---:|---:|---:|---|---|---|
| tasks passed | 12, 12 | 12, 12 | 0 | 0 | 0 | 0 | yes | | 0, 0, 12 |
| turns | 92, 67 | 64, 60 | −17.5 | −22.01 | 25 | 4 | yes | −45.5 to −1.0 | 1, 7, 4 |
| generated tokens | 34,945, 21,936 | 35,406, 36,683 | +7,604.0 | +26.74 | 13,009 | 1,277 | yes | −9,418.5 to +29,236.5 | 7, 5, 0 |
| tool calls | 87, 60 | 59, 56 | −16.0 | −21.77 | 27 | 3 | yes | −42.5 to 0.0 | 3, 7, 2 |
| tool errors | 0, 2 | 1, 0 | −0.5 | | 2 | 1 | yes | −2.0 to +1.0 | 1, 2, 9 |
| tokens of the first replies | 792, 797 | 801, 861 | +36.5 | +4.59 | 5 | 60 | no | −21.0 to +109.5 | 6, 4, 2 |
| tokens of the second replies | 11,826, 10,783 | 25,457, 27,222 | +15,035.0 | +133.00 | 1,043 | 1,765 | no | +1,712.0 to +36,570.0 | 9, 3, 0 |
| tokens of the later replies | 22,327, 10,356 | 9,148, 8,600 | −7,467.5 | −45.70 | 11,971 | 548 | yes | −17,321.5 to −813.5 | 3, 9, 0 |
| tokens a step | 2.957, 3.053 | 3.362, 3.329 | +0.3405 | +11.33 | 0.096 | 0.033 | no | | |
| share accepted | 0.6565, 0.6894 | 0.7861, 0.7750 | +0.1076 | +15.99 | 0.0329 | 0.0111 | no | | |

- **Passes:** the same, 12 of 12 in each run.
- **Turns and tool calls:** `pv1` is not higher. The preset's first run holds one attempt of 29 turns
  (`bugfix-javascript-complex-numbers`, passed); its other 11 tasks took 63 turns.
- **Generated tokens:** both `pv1` runs are above both preset runs; the nearest two, `pv1` 1 and `pv0` 1, are 461
  tokens apart. The difference between the means is inside the preset's own spread, and the interval over tasks
  spans zero. The median task generated 1,900 and 1,168 tokens with the preset and 1,264 and 1,467.5 with `pv1`.
- **Where the tokens are:** a task's first reply is a short tool call (792, 801, 797 and 861 tokens over the 12
  tasks in the four runs). The second reply, written after the first test run, holds the long reasoning, and there `pv1`'s runs wrote 25,457 and
  27,222 tokens against 11,826 and 10,783. One task, `bugfix-javascript-variable-length-quantity`, is 10,143.5 of
  that difference of 15,035.0: its second reply was 17,785 and 9,445 tokens with `pv1` and 2,872 and 4,071 with the
  preset. Without it the second replies are 7,672 and 17,777 against 8,954 and 6,712. The preset then spent more
  in later turns (22,327 and 10,356 tokens against 9,148 and 8,600).
- **Tokens a step:** higher with `pv1` in every task and run, 3.362 and 3.329 against 2.957 and 3.053 (+11.33 %);
  the replay measured +7.74 % on fixed prompts. The texts differ between the arms here, so this is not a paired
  number.

### Per task

Turns and generated tokens in run order `pv0` 1, `pv1` 1, `pv0` 2, `pv1` 2; all 48 attempts passed.
[paired.tsv](paired.tsv) adds the differences; [tasks.tsv](tasks.tsv) has each attempt's tool calls, tool errors,
drafted and accepted tokens, tokens a step and tests passed.

| Task | Turns | Generated tokens | Second reply's tokens |
|---|---|---|---|
| bugfix-python-affine-cipher | 6, 5, 8, 5 | 1,895, 1,721, 2,602, 1,941 | 378, 750, 272, 687 |
| bugfix-javascript-space-age | 5, 5, 5, 5 | 662, 791, 656, 646 | 189, 381, 184, 161 |
| bugfix-python-phone-number | 6, 5, 5, 5 | 969, 903, 1,085, 994 | 112, 439, 238, 177 |
| bugfix-javascript-triangle | 5, 5, 5, 4 | 540, 669, 662, 600 | 229, 253, 174, 270 |
| bugfix-python-variable-length-quantity | 6, 5, 6, 5 | 4,656, 2,002, 1,251, 3,254 | 3,943, 564, 529, 2,674 |
| bugfix-javascript-phone-number | 5, 5, 5, 5 | 633, 715, 763, 687 | 203, 110, 171, 38 |
| bugfix-python-list-ops | 5, 5, 5, 5 | 2,295, 2,514, 1,431, 2,137 | 1,739, 1,863, 158, 1,580 |
| bugfix-javascript-grade-school | 7, 5, 4, 5 | 890, 544, 628, 625 | 102, 119, 283, 100 |
| bugfix-python-scale-generator | 5, 7, 5, 5 | 1,905, 3,249, 4,740, 7,726 | 1,294, 2,012, 4,069, 7,145 |
| bugfix-javascript-complex-numbers | 29, 5, 8, 5 | 9,427, 1,034, 760, 742 | 42, 428, 36, 246 |
| bugfix-python-pig-latin | 5, 5, 6, 5 | 2,214, 1,494, 2,481, 5,753 | 723, 753, 598, 4,699 |
| bugfix-javascript-variable-length-quantity | 8, 7, 5, 6 | 8,859, 19,770, 4,877, 11,578 | 2,872, 17,785, 4,071, 9,445 |

**What it decides.**
- **Q731 (probabilistic verification on this line):** nothing here speaks against `pv1` in passes, turns or tool
  calls, and tokens per task are inside the preset's own spread, so the rig pair for the time of a step decides the
  next move. If a task did take 26.74 % more tokens, the 11.33 % more tokens a step would not pay for it (est.),
  so the agent-check arm on the rig that follows a gain there reads minutes per task, not tokens per second.
- **Q1184 (new):** whether replies are longer under `pv1`, on fixed prompts with many seeds, and what the source
  samples after a rejection.

**Limits.**
- **Twelve tasks, two runs an arm.** One spread per arm. Flash-Next's reply lengths have a long tail (the longest
  tenth of the replies holds 63.20 % of the tokens, Q1152), so a run's token sum is a few long replies: 7, 5, 2
  and 6 replies of 1,000 tokens or more in the four runs here. A real difference of a quarter in tokens per task would look like
  this result, and so would none.
- **No seed is sent,** as in the release check, so each run draws its own text; the tasks are paired, the texts are
  not.
- **Counts, not time.** On the rig a task also hits the 12-minute cap sooner than on these cards; no attempt was
  capped here.
- Passing is the task's test file; the quality of the fix beyond that is not scored.
- One model file, one draft, one draft length, the sm_61 build's compute 6.1 PTX on newer cards.

## Method

- **The line:** the release preset without its nine-card split (`-ts`), the draft's device override (`-otd`) and the
  keep-alive variable, with `--metrics --no-webui` as in the release check, on three RTX 3090 cards, llama-server
  choosing the split. The build (`pascal-p106-v3-b12076-ladder5`, `1551f5fe1`) holds only sm_61 code and compute
  6.1 PTX, so the newer cards' driver compiles the PTX; `CUDA_FORCE_PTX_JIT` is not set. After loading, the cards
  held 13,938, 13,162 and 15,042 MiB in every run. One slot (`-np 1`), 65,536 tokens of context.
- **The runs:** `pv0 pv1 pv0 pv1`, one llama-server container per run; a one-task run of `pv1` came first as a
  smoke test and is not counted.
- **The client:** the release check's client part, [client_arm.sh](../../../bench/agent_check/client_arm.sh), on
  the same machine: the logging proxy in a container, and pi 0.87.1 with the tests in one container per attempt
  (the sandbox image of [docker/Dockerfile](../../../bench/agent_check/docker/Dockerfile)). One attempt per task,
  one task at a time, in the set's order. The release check's limits: 40 turns, 12 minutes and 32,768 reply
  tokens per attempt, no attempt started after 40 minutes of a run, and the makers' sampling on every request
  (temperature 1.0, top-p 0.95, top-k 20, min-p 0). The 12 tasks were validated in the image first (12 of 12).
- **Not `quant_run.sh`:** that script's server line is fixed (six slots, 262,144 tokens, no draft) and it runs the
  polyglot set, so [run/run_pv_agent.sh](run/run_pv_agent.sh) starts the server on the preset's line itself and
  calls the client part.
- **The counters:** each attempt's `result.json` (passed, turns, tool calls and errors, caps) joined with the
  server's `timings` of its requests in the proxy's log (`predicted_n`, `draft_n`, `draft_n_accepted`,
  `sampled_on_device`, `grammar_rejected_n`).
- **The tables:** [run/pv_agent_table.py](run/pv_agent_table.py); 10,000 bootstrap resamples of the 12 tasks,
  seed 1183.

## Files

| File | Holds |
|---|---|
| [summary.md](summary.md) | every table of `pv_agent_table.py` as Markdown |
| [runs.tsv](runs.tsv) | per run: tasks passed, turns, requests, generated, tool calls and errors, malformed calls, capped attempts, drafted, accepted, share accepted, tokens a step, tokens a turn, the median and the largest reply, replies of 1,000 tokens or more, tokens of the first, second and later replies, tokens drawn on the device, grammar rejections |
| [tasks.tsv](tasks.tsv) | per run and task: passed, tests passed, turns, requests, generated, drafted, accepted, share accepted, tokens a step, tool calls and errors, caps, the context at the last request, tokens of the first, second, later and largest reply |
| [requests.tsv](requests.tsv) | per request: prompt tokens, generated, drafted and accepted tokens, tool calls, finish reason |
| [paired.tsv](paired.tsv) | per task: passes, turns, generated tokens, tokens a step and the first and second reply's tokens in every run; `pv1`'s mean minus the preset's; each arm's second run minus its first |
| [compare.tsv](compare.tsv) | per measure: every run by value, the arms' means, the difference, each arm's two runs apart, whether the difference is inside the preset's, the interval over tasks, tasks higher, lower and equal |
| [run/run_pv_agent.sh](run/run_pv_agent.sh) | the runs: task validation, then per run the server container, the client, a check that every attempt was scored |
| [run/pv_agent_table.py](run/pv_agent_table.py) | the tables |

The tables name the tasks by id and hold no prompt or reply text;
[build_bugfix.py](../../../bench/agent_check/build_bugfix.py) rebuilds the task files.

## Commands

On the machine with the cards. `ROOT` is a work folder, `REPO` a checkout of this repository, `BIN` the build's
`llama-server`, `MODEL` the first shard of Swift Q2_0 (the other two beside it), `MODEL_DRAFT` the MTP draft, `GPUS`
the three cards as Docker names them, `MOUNTS` the folders the server container reads. The sandbox image
`agent-check:0.87.1` is built beforehand
(`docker build -t agent-check:0.87.1 -f docker/Dockerfile .` in `bench/agent_check`).

```bash
RUN=$REPO/docs/data/2026-11-10_prob_verify_agent_check/run/run_pv_agent.sh
mkdir -p "$ROOT/smoke" "$ROOT/chain" && cd "$ROOT"
# the smoke test: one task on pv1
WORK=$ROOT/smoke REPO=$REPO BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT GPUS=$GPUS MOUNTS="$MOUNTS" \
  NAME=fn50-srv PREFIX=fn50 PORT=18150 PROXY_PORT=18151 JIT=$ROOT/jit RUNS=pv1 TASKS=1 \
  setsid -f nohup nice -n 19 bash "$RUN" > smoke/console.txt 2>&1 < /dev/null
# the four runs, once smoke/done exists; a second pair would not have started after 07:08
WORK=$ROOT/chain REPO=$REPO BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT GPUS=$GPUS MOUNTS="$MOUNTS" \
  NAME=fn50-srv PREFIX=fn50 PORT=18150 PROXY_PORT=18151 JIT=$ROOT/jit PAIR_START_BY=07:08 \
  setsid -f nohup nice -n 19 bash "$RUN" > chain/console.txt 2>&1 < /dev/null
```

It is done when `$ROOT/chain/done` exists (`$ROOT/chain/failed` otherwise); the four runs ended 37 minutes after
the start. Then:

```bash
nice -n 19 python3 -I "$REPO/docs/data/2026-11-10_prob_verify_agent_check/run/pv_agent_table.py" \
  "$ROOT/chain/out" --out "$ROOT/chain/tables"
```

## Provenance

- The tasks are this project's bug-fix set, built from Exercism's Python and JavaScript tracks (MIT; the licences
  are in `bench/agent_check/tasks/bugfix/`). The tables hold counters only.
- The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, as in the release preset.
- Raw output (the agent's transcripts, the proxy's log with its request bodies and times, the server logs, the
  cards' memory) is not published.
