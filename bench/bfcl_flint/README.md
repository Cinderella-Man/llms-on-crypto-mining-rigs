# BFCL tool-calling check for Flint

A harness for checking whether a quantized Flint file still calls tools correctly, run the way Flint's makers
evaluate it. It uses the Berkeley Function Calling Leaderboard v4 (`bfcl-eval` 2026.3.23) in two lanes, both scored
with BFCL's own checkers, unmodified:

- **Single-turn** (lane E1 of [Q41](../../quants/experiments.md)): 3,477 of the 3,491 judge-free Python tasks
  (simple_python, multiple, parallel, parallel_multiple, irrelevance and the six live categories).
- **Multi-turn** (lane E2): 799 of the 800 tasks of base, miss_func, miss_param and long_context. These are 200
  scenarios run as conversations against BFCL's local Python backends (file system, trading, travel, vehicle,
  messaging and others), checked on backend state and call results after every turn.

The protocol and its statistics are in
[task_level_quality_checks.md](../../docs/research/notes/task_level_quality_checks.md). The makers' serving settings
are in [rec_engine.md](../../docs/research/notes/rec_engine.md) (§4). Results go into one CSV row per task, file and
attempt. [paired_tasks.py](../paired_tasks.py) compares files on those rows.

## How it works

### Prompts

- Prompts are rendered on the client with the chat template embedded in the GGUF; [gguf_meta.py](gguf_meta.py)
  extracts it. UD-Q4_K_XL, the makers' Q4_1 and Unsloth's Q8_0 carry the same template (sha256 `7940f4ff…`).
- The template is rec's copy plus a macro that sanitises user and tool text. The tool list goes in its own `<|system|>`
  block. Assistant turns are `<|assistant|><think>reasoning</think>content<tool_call>…`, tool results are
  `<|observation|><tool_response>…</tool_response>`, and the generation prompt is `<|assistant|>` alone.
- Tools are BFCL's conversion of each task's functions for an OpenAI chat-completions client. That means
  `convert_to_tool` with `ModelStyle.OPENAI_COMPLETIONS`, "." in names replaced by "_", and BFCL's Python-syntax hint
  added to every description.

### Generation

- Each request goes to llama-server's native `/completion` endpoint as a pre-rendered prompt, with no `--jinja`,
  server-side tool parsing or grammar. This is the method of the Q30 harness, so the raw text, `<think>` block
  included, is what the model wrote.
- Every request carries the makers' sampling explicitly: temperature 1.0, top-p 0.95, top-k 20, min-p 0. The other
  llama-server samplers are set to neutral values and the chain order is pinned to top-k → top-p → min-p → temperature
  ([flint_client.py](../flint_client.py)).
- **Attempts.** Each task runs K times (`--attempts 1-5`, the default, matching the makers' "pass@1 averaged over at
  least five attempts"). Attempt n samples with seed n for every task and every file, so two files' attempts share
  their random numbers, which lowers the variance of paired differences.
  - Attempts run in order: all tasks of attempt 1, then attempt 2, and so on.
  - Runs resume where they stopped, and K can be raised later without repeating work.
  - Attempts can be split across GPU sets (`--attempts 1-3` on one server, `--attempts 4-5` on another).
- **Token limits.** `n_predict` is 4,096 per turn, as in BFCL's local-model handler. Single-turn runs use
  `cache_prompt` false; multi-turn runs keep the prompt cache, so each step processes only the new messages.

### Parsing

[bfcl_common.py](bfcl_common.py) parses Flint's output the way rec does (`engine/src/rec_mlx/chat.py`,
`parse_tool_call`):

- **Reasoning and calls.** The reasoning is everything up to the last `</think>`. Calls are the
  `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>` blocks after it.
- **Typing.** A value stays a raw string when the tool's schema accepts a string, and is parsed as JSON otherwise
  (kept raw if that fails). `null` becomes None only when the schema accepts null.
- **Undecodable turns.** An unterminated call, a call rec's parser rejects, or a stray protocol marker outside the
  reasoning make the turn undecodable. BFCL then scores it as a decoder failure: wrong on AST tasks, "no call" on
  irrelevance tasks.

Process flags are recorded per attempt, separately from the verdict:

| Flag | Meaning |
|---|---|
| `undecodable` | the turn could not be decoded, as above |
| `unknown_function` | calls to a function that is not offered |
| `unknown_arg` | argument names not in the function's schema |
| `call_in_think` | a `<tool_call>` inside the reasoning (it is not a call) |
| `extra_think_close` | `</think>` written more than once |
| `runaway` | stopped by the token limit, or the reasoning never closed |

`malformed` is any of the first four. Q41's format gate pools `malformed` and `runaway`.

### Multi-turn loop

[multi_turn.py](multi_turn.py) runs BFCL's `inference_multi_turn_FC` loop with Flint in place of an API:

- Each step renders the whole history and samples one assistant turn.
- Earlier assistant turns are sent back with their `reasoning_content`, rendered as `<think>…</think>` as the makers
  ask. Calls are sent back as parsed arguments, the way a client of rec's API returns them.
- Calls run on BFCL's backends and each result becomes a tool message.
- A turn ends when the model answers without a call or with undecodable calls. More than 20 steps in one turn ends
  the task as failed.
- miss_func turns add the held-out functions with BFCL's "I have updated some more functions" message.
- Scoring decodes the stored raw text again and runs BFCL's `multi_turn_checker`: backend state and call results
  after every turn must match the ground truth's.

### Output

- `score.py` (single-turn) and `multi_turn.py score` write `per_attempt.csv` and `summary.md`. The CSV columns are
  defined in [task_results.py](../task_results.py): lane, task, cluster, category, file, attempt, seed, pass,
  malformed, runaway, error type, tokens and the process flags.
- `cluster` groups tasks that share a source: the source key of live tasks (`live_multiple_915-191-3` → source 191),
  and the scenario number of multi-turn tasks (the four categories reuse the same 200 scenarios).
- [paired_tasks.py](../paired_tasks.py) reports each file against the reference:
  - pass@1 per task (the mean over attempts) and the paired difference per task;
  - the plain and clustered standard error, with 90 % and 95 % intervals and a bootstrap interval;
  - the non-inferiority verdict against the lane's margin (1.5 points single-turn, 3 multi-turn);
  - the format gate (malformed + runaway, upper 90 % bound ≤ +0.5 points);
  - E[p(1−p)] and the attempt-1 against attempt-2 discordance.

## Validation without a model

- **[selftest.py](selftest.py)** renders every single-turn task's ground truth as an assistant turn through the
  template, then parses and checks it. The turn has a reasoning block and one allowed value per parameter, with
  optional parameters left out. **All 3,477 tasks pass.**
  - The 14 excluded tasks ([excluded_single_turn.txt](excluded_single_turn.txt)) fail for every possible answer.
  - In 11 of them the ground truth contradicts the function's own schema: a required parameter is missing, or a
    string enum such as `"True"/"False"/"dontcare"` is expected as a boolean.
  - In 3, rec's parser cannot return the expected type: an array schema that lists string enum values keeps its
    value as a raw string, and a string parameter whose only accepted value is None gets the string `"null"`.
- **`multi_turn.py selftest`** replays every multi-turn task's ground truth through the conversation loop:
  - in each turn the "model" makes the ground-truth calls, one per step, through the template with a reasoning
    block, and then answers in plain text;
  - the calls run on the backends, and BFCL's checker scores the conversation.

  **799 of 800 pass with no process flag.** `multi_turn_miss_func_49` is excluded
  ([excluded_multi_turn.txt](excluded_multi_turn.txt)): its ground truth calls `tail` two turns before `tail` is
  offered. The longest multi-turn prompt is 92,479 characters (~22k tokens, a long_context task).
- **Q30's raw outputs** (300 v3 tasks, greedy) re-scored with this harness give the same verdict as Q30's decoder on
  all 600 BF16 rows. The makers' Q4_1 differs on 2 of 300 tasks:
  - one answer writes a list as a Python literal (`['Sam', 'Alice']`), which is not JSON;
  - one writes `null` for a string argument.

  The makers' Q4_1 closes its reasoning twice in 9 of 300 answers, BF16 in 2 of 600.

## Running it

Needs Python ≥ 3.10 with jinja2, requests, numpy, tqdm, pyyaml, tenacity, overrides, filelock and mpmath, plus the
unpacked wheel. The full `bfcl-eval` install pulls in every API SDK, vLLM and sentence-transformers, which the
Python categories never use.

```bash
pip download --no-deps bfcl-eval==2026.3.23 && python -m zipfile -e bfcl_eval-2026.3.23-py3-none-any.whl bfcl-v4
export BFCL_ROOT=$PWD/bfcl-v4
GGUF_PY=/path/to/llama.cpp/gguf-py python gguf_meta.py flint.gguf > template.jinja
python prepare.py template.jinja tasks.jsonl                 # 3,477 tasks
python selftest.py template.jinja tasks.jsonl                # expect: 3477 pass, 0 fail
python multi_turn.py selftest template.jinja                 # expect: 799 pass, 1 fail (the excluded one)
```

Single-turn, then multi-turn. Each server needs a context of at least ~32k tokens per slot: the longest prompt plus
4,096.

```bash
llama-server -m flint.gguf -ngl 99 -fa on -np 8 -kvu -c 262144 -b 2048 -ub 512 --cache-ram 0 &
python run.py http://127.0.0.1:8080 tasks.jsonl raw/st-myfile.jsonl --attempts 1-5 --conc 8
python multi_turn.py run http://127.0.0.1:8080 template.jinja raw/mt-myfile.jsonl --attempts 1-5 --conc 8
python score.py tasks.jsonl out/st myfile=raw/st-myfile.jsonl ref=raw/st-ref.jsonl
python multi_turn.py score out/mt myfile=raw/mt-myfile.jsonl ref=raw/mt-ref.jsonl
python ../paired_tasks.py --ref ref out/st/per_attempt.csv out/mt/per_attempt.csv
```

Each run writes a `.manifest.json` next to its raw output. It holds the server's model path, build and slot context,
the sampling, the task-list hash and the harness commit. Files with the rescale switch run with
`LLAMA_NO_MOE_DOWN_RESCALE=1` in the server's environment.

## Where this differs from the makers' engine

- **Grammar.** rec builds a grammar for every request:
  1. an optional reasoning block that contains `</think>` only at its end;
  2. text without any protocol marker;
  3. the end of the turn, or calls in the GLM-4.7 layout followed by `<|observation|>`.

  With tools, function names are limited to the offered ones and arguments follow the schemas. llama-server
  constrains nothing on this path. With `--jinja` and tools it constrains only after the model writes the tool-call
  trigger (a lazy grammar), never the reasoning or the text before the call.
- **What this lane counts.** This lane measures the unconstrained model. Undecodable calls, invented function and
  argument names, non-JSON values, a second `</think>`, and calls inside the reasoning all show up here, where rec
  would have prevented them. The first three count as wrong answers, as BFCL counts them for any model. The process
  flags keep them visible separately for the format gate.
  - Answers that close their reasoning twice are scored after the last `</think>`. That is what llama-server's
    parser returns, and what rec's grammar would have forced.
- **Sampler details.** mlx-lm, which rec uses, applies top-p to the top-k tokens without renormalizing them.
  llama-server renormalizes after top-k. At top-k 20 / top-p 0.95 this only matters for flat distributions.
- **Prompt cache.** rec reuses the shared prompt prefix. llama-server's `cache_prompt` can change logits slightly,
  because prompt and generation batches differ. Every file runs with the same cache setting, so the effect is part
  of the measured noise.

The 300-task BFCL v3 harness used for Q30 (greedy, one run) is at commit `4d216de`.

## Pilot plan (Q41)

**Pilot, before any full run.** One file, UD-Q4_K_XL (the bar), at K = 5 on one 8-slot server:

| Lane | Tasks | Requests | Server-hours (est.) |
|---|---|---|---|
| E1 | `prepare.py --per-category 40`: 376 tasks (all of live_parallel, live_parallel_multiple and live_relevance) | 1,880 | ~0.4 |
| E2 | `multi_turn.py run --scenarios 25`: 25 scenarios × 4 categories = 100 tasks | ~4,500 steps | ~1.6 |

Also run S4-L0 on the E1 pilot (~0.4 h). That gives a first paired difference and checks that two files differ no
more than reruns do.

**What the pilot fixes before the main runs:**

1. **E[p(1−p)] per lane under sampling** (`paired_tasks.py`), which sets K per lane. N·K must reach
   12.4 · E[p(1−p)] / m² for 80 % power at margin m when the files are equal (the note's Table C). With
   N = 3,477 and m = 1.5 points, single-turn needs K = 1 at E[p(1−p)] 0.045 and K = 2 at 0.08. With N = 799 and
   m = 3 points, multi-turn needs K = 2 at 0.08 and K = 3 at 0.12 (more if clustering by scenario inflates the
   error). K = 5 stays the default; a lower K is recorded in Q41's row before the main runs.
2. **Runaway rate at 4,096 tokens.** If more than 1 % of turns hit the limit, raise `--n-predict` for every file.
3. **Malformed rate:** the format gate's baseline.
4. **Tokens and job time per task,** which replace the estimates below.

**Full runs, per file at K = 5 (est.).**

| Lane | Work | Server-hours |
|---|---|---|
| E1 | 17,385 requests: ~11 M prompt tokens (~630 per request, no prompt cache), ~3.5 M output | ~3.4 |
| E2 | 3,995 conversations: ~11 M output tokens, ~40 M new prompt tokens | ~12 |

The assumptions are the note's §7.5, per 8-slot server: ~400 tokens/s aggregate decode and ~3,000 tokens/s prompt
processing; output ~200 tokens per single-turn request; ~2,700 output tokens per multi-turn task. These are job
times for scheduling on the quant machine, not speed results.
