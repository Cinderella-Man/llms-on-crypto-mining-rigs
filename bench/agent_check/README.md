# Agent check: the pi coding agent driving Flint through real coding tasks

Microbenchmarks and token-level scores don't show what a user of a coding agent feels: minutes per task, the wait
before each turn's first token while re-sent context is processed, tool calls that fail, and whether the task gets
done. This folder measures that. The [pi](https://github.com/earendil-works/pi) coding agent (0.87.1) runs on a
laptop and talks to llama-server on the rig. It works through Exercism exercises with hidden tests: Aider's polyglot
selection, Python and JavaScript tracks. Each attempt is timed and scored. A logging proxy records llama-server's
own counters for every request, and a replay tool re-sends the recorded requests to other builds, flags or files, so
they are compared on identical prompts.

The design, the statistics and the run plan are in
[agent_workload_check.md](../../docs/research/notes/agent_workload_check.md). Of its suite, this folder implements
the polyglot family, plus a set of 12 small bug fixes derived from it for the release check
([below](#the-release-check-one-command-1-h)). Asking for missing information is covered by the BFCL multi-turn lane
([below](#the-bfcl-multi-turn-lane)). Nothing here needs a GPU on the client: pi, the proxy and the exercises' own
small test runs are all it does.

## The task set

- **Source:** the exercises are taken from Exercism's [python](https://github.com/exercism/python) and
  [javascript](https://github.com/exercism/javascript) tracks (MIT; licences in `tasks/polyglot/LICENSE.exercism-*`,
  commits in `tasks/polyglot/commits.json`). They are not taken from Aider's repository, which has no licence file.
- **Selection:** Aider's polyglot list for the two tracks ([polyglot_exercises.txt](polyglot_exercises.txt)): 34
  Python and 49 JavaScript exercises. JavaScript's `ledger` is excluded
  ([excluded_exercises.txt](excluded_exercises.txt)): it is a refactoring exercise whose untouched stub already passes
  every test. **82 tasks** remain.
- **What the agent sees** (`tasks/polyglot/<track>-<slug>/`):
  - `prompt.md`: Exercism's introduction, instructions and track appendix, then the task statement: implement the
    stub, keep the names, standard library only, which test file holds some of the tests and how to run it (`pytest`
    or `npm test`), and that a hidden suite will be run.
  - `workspace/`: the stub, helper and data files, and the visible tests. JavaScript workspaces also get a
    package.json, Exercism's babel and jest configuration, and a `node_modules` link to the shared environment.
- **Visible and hidden tests:**
  - The visible tests are about a third of the canonical cases: positions 0, 3, 6, … within each test class
    (Python) or innermost `describe` block (JavaScript), so every section of the suite shows its API and error format.
    That makes 574 of 1,574 cases visible (36 %).
  - `hidden/` holds the full canonical test file, every skipped JavaScript test switched on. Only the two cases
    Exercism marks `test.skip` because they run too long stay off.
  - `reference/` holds Exercism's example solution, used only for validation. Neither `hidden/` nor `reference/`
    enters a workspace.
- **Validation** ([validate.py](validate.py), as Harbor validates its tasks): for every task, the reference solution
  passes the hidden tests, the untouched workspace fails them, and the reference also passes the visible tests, which
  are a strict subset. All 82 tasks pass (`tasks/polyglot/validation.json`, laptop sandbox, Python 3.14, node 26). The
  quant machine's runs repeat the check in their Docker image before starting a server.
- **Run order** ([make_order.py](make_order.py), `tasks/polyglot/order.txt`): every prefix of the order is a
  proportional sample of track × size tercile (reference solution length), and the tracks alternate from the start.
  A pilot that runs the first few tasks is therefore representative, and so are the two halves of a full run. After
  the calibration, `make_order.py --calibration attempts.csv` re-strata by pass rate.
- **Rebuilding the set:** check out the two tracks at the pinned commits and run
  `AC_NODE_MODULES=$AC_HOME/node/node_modules python3 build_tasks.py <python checkout> <javascript checkout>`, then
  `make_order.py` and `validate.py`.

## The bugfix task set

Small bug fixes in working code, the everyday job of a coding agent: 12 tasks, 6 Python and 6 JavaScript, tracks
alternating (`tasks/bugfix/`, built by [build_bugfix.py](build_bugfix.py) from [bugfix_bugs.json](bugfix_bugs.json)).

- **The bugs:** each task is a polyglot task's reference solution (Exercism's example solution) with one small change
  that breaks it, of the kind people make: parentheses lost in a formula, an off-by-one length check, 365 days to the
  year instead of 365.25, a missing third side in the triangle inequality, a check that looks at the wrong digit,
  arguments in the wrong order, a duplicate check limited to one group, a lookup after a value was normalized, a sign
  flipped in a product, a regular expression that lost a group, a signed result where an unsigned one is needed.
  Each breaks 1 to 8 of the task's tests.
- **What the agent sees:** the broken file and the task's full test file (every JavaScript test switched on), with
  Exercism's instructions and the statement: some tests fail, run them, find the bug and fix it, keep the fix small,
  don't change the tests. A typical fix takes 5 to 10 tool calls: run the tests, read the code, edit, run the tests
  again.
- **Success:** the full test file passes and is unchanged, as in the polyglot set.
- **Validation:** `validate.py --task-set bugfix` checks that the reference solution passes, the broken workspace
  fails, and the workspace's test file is the full one. All 12 pass (`tasks/bugfix/validation.json`, the Docker image:
  Python 3.12, node 24).

## Files

| File | What it does |
|---|---|
| [run.py](run.py) | The runner: one arm (server configuration) at a time, fresh workspace per attempt, pi headless, caps, hidden-test scoring, resumable; infrastructure failures re-run, never scored |
| [proxy.py](proxy.py) | Pass-through logging proxy between pi and llama-server: timings, usage, cache, first token, tools, the request bodies |
| [replay.py](replay.py) | Re-sends recorded traces to any server (same prompts, seeds and `max_tokens`); picks traces; summarizes and compares per-token costs |
| [report.py](report.py) | The per-attempt CSV (the layout of [task_results.py](../task_results.py), read by [paired_tasks.py](../paired_tasks.py)), the per-arm table, the pilot's go/no-go list, arms side by side with the time to first token per turn |
| [validate.py](validate.py) | Oracle, empty-change and visible-subset checks of every task, no model |
| [common.py](common.py) | Tasks, the sandbox, the test check, the stream parser |
| [pi/](pi/) | The agent folder template: `models.json` (Flint as an OpenAI-compatible provider), `settings.json`, the `check.ts` extension |
| [pilot.sh](pilot.sh), [full_run.sh](full_run.sh), [rig_lib.sh](rig_lib.sh), [rig_server.sh](rig_server.sh) | The rig pilot and full run, one command each, run on the laptop; the server launcher with monitors and the RAM watchdog, run on the rig over ssh |
| [quant_run.sh](quant_run.sh), [docker/](docker/) | Calibration and file-quality runs on a GPU machine with Docker; the sandbox image |
| [release_check.sh](release_check.sh), [client_arm.sh](client_arm.sh), [client_report.sh](client_report.sh) | The release check: stock against the release engine on the bugfix set, the client in Docker on a second machine, started from a third that only runs ssh and relays the requests |
| [build_bugfix.py](build_bugfix.py), [bugfix_bugs.json](bugfix_bugs.json) | Build the bugfix set from the polyglot tasks and the list of injected bugs |
| [mock_server.py](mock_server.py), [selftest.sh](selftest.sh), [selftest_pilot.sh](selftest_pilot.sh) | A stand-in for llama-server and the end-to-end tests that use it (no model) |
| [build_tasks.py](build_tasks.py), [cut_tests.mjs](cut_tests.mjs), [make_order.py](make_order.py) | Build the task set from the Exercism checkouts |
| [env/](env/) | The shared test environments: jest (locked) and pytest; `env/setup.sh` creates them |

## Setup on the laptop

- **pi 0.87.1**, the version the check pins. Its managed install keeps each release in a folder holding
  `node_modules/@earendil-works/pi-coding-agent/` (`AC_PI_ROOT`). `node` ≥ 22.19 (`AC_NODE_DIR`, the install root
  holding `bin/node`).
- **The shared test environments:** `env/setup.sh $AC_HOME` puts jest and Exercism's babel preset in
  `$AC_HOME/node/node_modules` and a relocatable venv with pytest in `$AC_HOME/py` (~140 MB together). Nothing is
  installed per task.
- **bwrap** (bubblewrap) for the sandbox, `uv`, Python ≥ 3.10.
- **Disk:** workspaces live under the system temp folder and are deleted after each attempt. Per attempt the results
  keep pi's events and session, the diff, the test output and a copy of the agent folder (~0.1–0.5 MB). The proxy's
  request bodies are gzipped (~10–50 kB per request).

The sandbox is the same on every machine:

| Path inside | What |
|---|---|
| `/work` | the attempt's workspace, writable; the same path for every task, so pi's system prompt (which names the working directory) is byte-identical and its prefix reusable by the server |
| `/session` | pi's session folder |
| `/opt/pi-agent` | the attempt's copy of the agent folder (pi writes a settings lock and `auth.json` there) |
| `/opt/pi`, `/opt/node`, `/opt/ac-node`, `/opt/ac-py` | pi, node, the shared node_modules, the shared venv (read-only) |
| `/home/agent`, `/tmp` | empty, private |

On the laptop `bwrap` builds it: only these paths, a cleared environment, its own process namespace; the hidden-test
check also has no network. On the quant machine each attempt runs in its own container of the
[docker/Dockerfile](docker/Dockerfile) image.

## How an attempt runs

1. The task's workspace is copied fresh and mounted at `/work`.
2. pi runs in JSON mode with only the check's configuration:

   ```
   pi --mode json --provider agent-check --model flint --no-context-files --no-extensions -e /opt/pi-agent/check.ts
      --no-skills --no-prompt-templates --no-themes --no-approve --offline --session-dir /session -- "<prompt.md>"
   ```

   - `models.json` declares Flint as the makers declare it for pi (`reasoning: false`, no reasoning controls, compat
     flags off), with `contextWindow` equal to the server's `-c` (262,144), so pi's compaction never fires in the suite,
     and `maxTokens` 32,768.
   - `settings.json` turns pi's retries off and sets a 30-minute HTTP idle timeout: a long prefill on v1 is not cut,
     and every infrastructure failure shows up.
   - `check.ts` sets the makers' sampling on every request (temperature 1.0, top-p 0.95, top-k 20, min-p 0), so all
     arms sample alike whatever their command-line defaults. It also sets `max_tokens` 32,768 and tags each request
     with an `X-Agent-Check: <arm>/<task>/a<k>/t<try>/<turn>` header for the proxy.
   - pi sends each reply's `reasoning_content` back in the history, as the makers ask, and its four default tools
     (read, bash, edit, write) with every request.
3. Every event line pi prints is written to `events.jsonl` with its arrival time. The runner stops pi after 40 turns
   or `--time-cap-min` minutes and flags the attempt a runaway.
4. The final workspace is copied, the full hidden test file put in place, and the tests run without network
   (timeout 180 s). **Success = the hidden tests pass and the visible test file is unchanged.** `hidden_pass` and
   `tests_untouched` are reported apart. A runaway's workspace is scored as it stands.
5. `result.json` gets the attempt's row:
   - success, tests passed and total, turns, tool calls by tool, tool errors;
   - malformed calls: an unknown tool, arguments that fail the tool's schema, or tool-call markup left in the reply
     text; calls inside the reasoning;
   - replies stopped at `max_tokens`, runaways, compactions;
   - wall-clock, pi's own token usage.

   `diff.patch`, `check.txt`, `stderr.txt`, `session/` and the server's `/metrics` before and after are kept next to
   it.
6. **Infrastructure failures are re-run, never scored:** the server or tunnel unreachable, a broken stream, an HTTP
   5xx, or pi exiting before any reply. The try's folder is renamed `a<k>-t<n>-infra`, the runner waits for `/health`,
   and the attempt runs again (up to 3 tries). A finished attempt has a `result.json` and is skipped, so a stopped run
   resumes with the same command.

## The proxy and the replay

- **The proxy** writes one row per chat request, and pi itself drops all of it:
  - the tag;
  - whether the request carried tools (backend sampling is off for those, Q55) or a grammar;
  - `max_tokens` and the sampling fields;
  - response headers, first chunk, first token and end times;
  - llama-server's `timings` (`prompt_n`, `prompt_ms`, `cache_n`, `predicted_n`, `predicted_ms`) and `usage`;
  - finish reason and tool calls.

  Each request body and the assembled reply (reasoning, text, tool calls) go to `proxy.jsonl.bodies.gz`.
- **The cache path of a turn** follows from consecutive rows, without a tokenizer:
  - "exact": `cache_n` reaches the previous prompt plus its reply;
  - "reply re-read": it stops at the previous prompt's end;
  - "older checkpoint": it falls short of the previous prompt, which is Q132's eviction case.
- **`replay.py run`** re-sends one or more traces (an attempt's requests) in order, straight to a server. It keeps
  their messages, tools and sampling, sets `max_tokens` to the recorded reply's length and a seed fixed per trace and
  request (the same for every server), and erases the slot before each trace. Options:
  - `--stream off`: the request non-streamed;
  - `--drop-reasoning`: the earlier replies' reasoning removed from the history;
  - `--ignore-eos`: exactly the recorded length generated.
- **`replay.py summary`** prints, per label: ms per prompt token, ms per generated token, the fixed cost per request,
  cache and time to first token. It gives each label's ratio to a reference with a 95 % bootstrap interval over
  traces, and "same work" minutes (the reference's token counts priced at each label's costs). The live rows of the
  same traces appear as `live:<arm>`.

## Calibration on the quant machine

Before any rig time, the whole set runs twice with UD-Q4_K_XL on one llama-server (6 slots, the same template,
sampling, context and checkpoint flags as the rig). This sizes pass rates, turns and tokens per task, and marks the
tasks every attempt passes or fails. Copy this folder to the machine, then:

```
MODEL=<gguf in the container> LLAMA_SERVER=<llama-server in the container> MOUNTS="-v <dir>:<dir>:ro …" \
GPUS=<a 2-GPU set> PORT=<free port> bench/agent_check/quant_run.sh <out folder>
```

- **Steps:** it builds the sandbox image (cached), validates the 82 tasks in it, starts the server container, the
  proxy and `run.py --sandbox docker --parallel 6`, and writes `table.md` and `attempts.csv`. Every container is
  removed at the end. Running it again with the same folder resumes.
- **Duration:** est. ~2 h for 164 attempts (the note's ~90 attempts per hour per 6-slot server).
- **File comparison:** the same script runs it, UD-Q4_K_XL against the lead file, 10 attempts per task each:
  `ARM=q-ud ATTEMPTS=1-10 …` and `ARM=q-k2h MODEL=<lead file> ATTEMPTS=1-10 …`. Then
  `python ../paired_tasks.py --ref q-ud --margin agent_polyglot=6 --gate agent_polyglot=2 q-ud/attempts.csv q-k2h/attempts.csv`.
- **Times recorded there are not speed results.**

## The rig pilot (one command, ~1 h 30 min)

On the laptop, with the machine-specific values in the environment (`rig_lib.sh` lists them: the rig's ssh alias,
three ports, the v1 and v2 llama-server binaries and the model path on the rig, the laptop's environments):

```
bench/agent_check/pilot.sh <out folder>
```

| Step | Rig time (est.) |
|---|---|
| v2 with UD-Q4_K_XL at 262,144 tokens, `-b 2048 -ub 512`, monitors and watchdog on | ≤ 10 min |
| Smoke task, then Q132's check first: each turn's cache path and the server's checkpoint lines, printed before anything else runs; the pilot stops if the task could not be scored | 4 min |
| Tasks 1–6 on v2 (task 1 already done) | 20 min |
| Replays on v2 of the two shortest v2 traces among tasks 1–3, streamed and not streamed | 14 min |
| v1 as released (`-b 512 -ub 128`); tasks 1–3 on v1; the same two traces replayed on v1 | 10 + 17 + 10 min |
| Rig logs copied back and removed there; table, replay summary, go/no-go | 2 min |
| **Total** | **~1 h 30 min** |

- **Where the times come from:** 3.9 min per task on v2 and 5.5 on v1, from the note's cost model. If Flint reasons
  twice as long per turn as assumed, the pilot takes ~2 h 30 min. Each attempt is capped at 20 minutes
  (`PILOT_TIME_CAP_MIN`) and 40 turns.
- **`gonogo.md`** applies the note's list (§6.3):
  - harness: at least 5 of 6 attempts without an infrastructure failure, every request with a proxy row and timings,
    first token consistent with the server's prefill;
  - tool calls: malformed calls under 5 %, none inside the reasoning;
  - cache: median hit rate from turn 2 ≥ 80 %, no older-checkpoint fallbacks;
  - memory: MemAvailable ≥ 600 MB, no swap growth;
  - time: within 2× of the estimates;
  - replay: within 5 % of the live per-token costs;
  - the server logs' counts of backend-sampling, checkpoint, re-processing and NaN lines.
- A failing item means fixing and repeating that step, not the whole pilot.

## The release check (one command, ~1 h)

Does a coding agent work on the release engine, and how much sooner is the job done than on stock llama.cpp? The
12 bugfix tasks run once on each engine, with UD-Q4_K_XL on nine cards and the same template, sampling, context and
checkpoints:

| Arm | Server | Flags beyond the common ones |
|---|---|---|
| `stock` | stock llama.cpp | `-b 2048 -ub 512` |
| `rel` | the release engine | `-b 2048 -ub 512 --pipe-depth 3` |

- **Common flags:** `-ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format
  deepseek`, the makers' sampling, `--metrics`. The context is 65,536 tokens because stock llama.cpp keeps an attention
  mask of context × micro-batch entries in pinned host memory: 264 MiB at 65,536 tokens with `-ub 512`, ~1 GiB at
  262,144, more than a 4 GB host can spare ([Qwen baselines](../../docs/benchmarks/2026-10-22-qwen-baselines.md#maximum-context-and-how-to-avoid-crashing-a-4-gb-host)).
  pi's context window is set to match, and a bug fix stays far below it.
- **Three machines** (any two can be the same):
  - the rig runs llama-server on its loopback only ([rig_server.sh](rig_server.sh), with its monitors and RAM
    watchdog);
  - a client, any Linux machine with Docker, runs pi, the logging proxy and the tests, each in a container
    ([client_arm.sh](client_arm.sh)), labelled and at a low CPU share (`--cpu-shares 256`), so it can share a machine
    with other jobs;
  - the controlling machine runs only ssh: it copies the files over, starts and stops everything, and relays the
    client's requests to the rig through two forwards, `ssh -L` to the rig and `ssh -R` to the client. Neither the rig
    nor the client has to reach the other, and the server never listens beyond the rig's loopback.
- **Run it** from the controlling machine, with the machine-specific values in the environment (the header of
  [release_check.sh](release_check.sh) lists them: ssh aliases, four ports, folders on the client, the two binaries and
  the model on the rig):

  ```
  STEPS=prep bench/agent_check/release_check.sh <out>    # untimed: files, image, task validation; no rig
  bench/agent_check/release_check.sh <out>               # the check; prep again is a quick no-op
  ```

| Step | Time (est.) |
|---|---|
| Prep: files to the client, the sandbox image (cached), the 12 tasks validated in it | 1 min, before the rig slot |
| Stock: load, 12 tasks | 2–4 + 25–40 min |
| Release engine: load, 12 tasks | 2–4 + 12–20 min |
| Rig logs to the client, reports, results copied to `<out>` | 1 min |
| **Total on the rig** | **~45–70 min** |

- **Where the times come from:** a fix takes about 6 turns. At ~400 generated tokens per turn, stock's 16.5 tok/s and
  ~500 prompt tok/s give 2–3.5 min per task; the release engine's ~45 tok/s pipelined gives 1–1.7 min. `TIME_CAP_MIN`
  (default 15) caps each attempt, and `BUDGET_MIN_stock` / `BUDGET_MIN_rel` stop an arm from starting more tasks after
  that many minutes, so a slow arm cannot overrun the slot. The comparison then uses the tasks both arms scored.
- **Outputs** in `<out>`: `compare.md` (per task: result, minutes, turns, tool calls all / failed / malformed and the
  time ratio; per arm: tasks passed, minutes, time to first token over all turns, the server's prompt and generation
  speeds; the ratio of total and per-task times; the median time to first token per turn), `res/turns.csv` (every
  request), `table.md` and `res/attempts.csv`, `gonogo.md` (harness, tool calls, cache paths, the rig's memory and
  server log lines), the rig's logs in `rig/`, and every attempt's events, session, diff and test output under `res/`.
- **Times through the relay:** the time to first token is measured at the client, so it includes the relay's two
  hops (the same for both arms); the server's own prompt time is in `turns.csv` (`prompt_ms`).
- **Dry run without a model:** point `RIG` at the client and both binaries at [mock_server.py](mock_server.py) (it
  ignores llama-server's flags), with its failure options among the arm flags, e.g.
  `STOCK_ARGS='--fail-every 4 --decode-ms 4'`, `REL_ARGS='--drop-every 5 --malformed-every 4'`, `MODEL=/dev/null`.
  The whole path runs: the relay, the server start and stop with monitors, the client's containers, the re-run of
  dropped streams, the reports and the copy back.

## The full run (one command, ~20 h)

```
bench/agent_check/full_run.sh <out folder> [live] [replay]
```

- **Live:** all 82 tasks once on each engine with UD-Q4_K_XL, in crossover order (v2 on tasks 1–41, v1 on 1–41, v1 on
  42–82, v2 on 42–82), est. 13 h.
- **Replays:** 8 traces from v2's first half spanning its range of task times (`replay.py pick`), 4 of them for the
  variants. In priority order:
  - v2 (8) and v1 (4);
  - v2 at v1's `-b 512 -ub 128` (4), which separates the flag change from the code;
  - v2 not streamed (4) and v2 with the reasoning dropped from the history (4);
  - v2 with 6 checkpoints and no minimum spacing (4);
  - the lead file on v2 (8, with `MODEL_K2H`);
  - stock llama.cpp at v1's flags (4, with `STOCK_BIN`);
  - v2 with `-t 1` (4).

  Est. 6–7 h with the server loads.
- Both stages resume where they stopped. The quality comparison of the engines is the live pairs:
  `python ../paired_tasks.py --ref v1-ud --margin agent_polyglot=15 <out>/res/attempts.csv` is the breakage gate,
  since one attempt per task cannot see a small loss.

## The BFCL multi-turn lane

Asking for missing information is the behaviour where Q41 found its only replicated quality gap: BFCL miss_param, where
a file asks as often as UD-Q4_K_XL but fails more often after the answer. It is measured by the existing BFCL harness,
not by new tasks: [bench/bfcl_flint](../bfcl_flint/README.md), lane E2 (799 multi-turn tasks: base, miss_func,
miss_param, long_context).

- **Where and how:** E2 runs on the quant machine with the same files and the makers' sampling as the polyglot file
  comparison. [ask_rate.py](../bfcl_flint/ask_rate.py) splits miss_param into asked, acted and passed after asking.
- **Pairing with the polyglot results:** both lanes write the same per-attempt CSV layout, so one call of
  [paired_tasks.py](../paired_tasks.py) gives each lane's paired difference, margin verdict and process gate:

  ```
  python paired_tasks.py --ref <UD file> --margin agent_polyglot=6 --gate agent_polyglot=2 \
      <polyglot attempts.csv> <bfcl multi-turn per_attempt.csv>
  ```

  The one-page result reports them side by side: polyglot for coding-agent work, E2 for tool calls over many steps
  and for asking.
- **What each lane covers:** BFCL requests are rendered on the client and sent to the native `/completion` endpoint
  without a grammar, so E2 says nothing about the served tool-call path's speed. The rig's speed numbers come from the
  polyglot traces only.

## Self-tests (no model, light)

```
AC_HOME=… AC_NODE_DIR=… AC_PI_ROOT=… bench/agent_check/selftest.sh <scratch folder>
AC_HOME=… AC_NODE_DIR=… AC_PI_ROOT=… bench/agent_check/selftest_pilot.sh <scratch folder>   # FULL=1 for full_run.sh
```

- **`selftest.sh`** drives pi in the sandbox through 12 tasks against [mock_server.py](mock_server.py), 3 at a time.
  The mock speaks llama-server's streamed chat endpoint with `timings`, solves the tasks with the reference solutions
  and fails some on purpose; it also drops some streams and writes tool-call markup as text. The test checks that:
  - exactly the intended tasks fail;
  - every dropped stream becomes a re-run, not a score;
  - the markup counts as malformed;
  - a second run resumes without redoing anything;
  - the turn cap stops an attempt;
  - every turn has a proxy row with timings;
  - replay re-sends traces with the recorded `max_tokens`.
- **`selftest_pilot.sh`** runs `pilot.sh` (or `full_run.sh`) against a fake rig on the laptop. An `ssh` stand-in runs
  the rig-side script locally and forwards the port, and the "llama-server" is the mock. That checks the server
  starts and stops, the monitors, the log collection and cleanup, and the reports.
