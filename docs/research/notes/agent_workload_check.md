# A realistic agent check: the pi coding agent driving Flint on the rig, timed and scored on real coding work (2026-10-24)

Scope: every measurement of Flint on this rig so far is a microbenchmark (llama-bench tok/s), a token-level fidelity
score (KLD), or a task check scored on the 7-GPU quant machine (BFCL, LiveCodeBench). None of them shows what a user of
a coding agent feels: minutes per task, the wait before each turn's first token while re-sent context is processed,
tool calls that fail, and quality over many steps. This note designs that check. The agent is pi (`pi` 0.87.1, the
harness Flint's makers ship a configuration for), running on a laptop and talking to llama-server on the rig through
an SSH tunnel. The check has to answer two questions on actual work: how much faster the v2 engine is than v1, and
what quality, if any, v2 or our release file loses. The note covers:
- what others do: agentic benchmarks and harnesses, published quantized-against-full results on agentic tasks, how
  people time agents, and how many tasks separate two models at our measured noise;
- pi's non-interactive modes, logs, context and compaction, read from its documentation and source, and the makers'
  pi configuration read from their engine's repository;
- a task suite that fits a day of rig time, with token and time estimates per task and script-checked pass criteria;
- the comparisons (engine, file, serving settings), how to pair them, and how many tasks and attempts each needs;
- what to measure, a one-page result format, and the runnable plan: setup, runner, pilot, full run, risks.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (code, papers, official docs, leaderboards) and secondary.
- "Laptop", "quant machine" and "rig" are the three places work runs. The laptop does no heavy compute: pi, a logging
  proxy and the tasks' own small test runs only. The quant machine gives quality numbers, never speed. Speed is
  measured on the rig only.
- "v1" is the released engine (fork branch `pascal-p106`), "v2" the combined test build of every v2 engine piece
  (Q117, branch `pascal-p106-v2-combined`), "UD" Unsloth's UD-Q4_K_XL.
- Nothing here ran on the rig or the quant machine. pi was inspected with `pi --help`, its bundled documentation and
  its installed source; no agent session was started.

Starting point, measured ([registry](../../../quants/experiments.md)):
- **Served speed on v1, nine cards, 262,144-token context, `-bs`:** UD 27.97 tok/s and S4-L0 31.86 tok/s in a
  five-turn chat (Q58). Prompt processing in that server: ~7.1 ms per new token plus ~2.2 s per request (80–85 tok/s;
  [prompt processing](prefill_pascal.md) §1.5).
- **Tool requests sample on the host:** every request that carries tools switches backend sampling off, est. +3.3 ms
  per generated token (Q55, [tool-request sampling](tool_request_sampling.md)). pi sends its tools with every request.
- **v2 pieces:** FP32 activation sums +9.6 % tg128 for UD on six cards (Q76). The combined build (Q117) passes every
  correctness check and has no rig timing yet; its prompt pieces are est. 130–190 instead of 56–72 generated tok/s for
  tool-heavy agents ([prompt processing](prefill_pascal.md)).
- **Agent traffic:** in 200 BFCL multi-turn sessions each request re-sends 39 prompt tokens per generated token and
  llama-server's prefix reuse with checkpoints serves 89.6 % of them (Q111). Round 9 found llama-server ~8 ms per token
  slower than llama-bench before the fork, and every published speed here non-streaming, while agents stream
  ([host in the decode loop](host_decode_loop.md)). Round 10 found pass rate the largest term of time per correct
  answer ([answer time](answer_time.md)).
- **Quality noise and gaps:** two sampled attempts of one file disagree on 15.5 % (single-turn) and 20 % (multi-turn)
  of BFCL tasks; G3 trails UD on BFCL miss_param (14 against 24 of 47 at 199 tasks), and our files are 3.5–4 % worse
  on the tokens Flint reads in long agent text (Q41, Q106).

## Verdict

- **Build a small runner around `pi --mode json`, a pass-through logging proxy and llama-server's own counters; do not
  adopt a Docker harness.** pi's JSON mode streams every event (thinking and text deltas, tool calls and their
  results, usage per message, compaction and retries) and its session file records the full prompt, tools and
  messages. llama-server reports per-request `timings` (`prompt_n`, `prompt_ms`, `cache_n`, `predicted_n`,
  `predicted_ms`) on the last streamed chunk, which pi drops, so a proxy on the laptop records them. Harbor already
  wraps pi the same way (`pi --print --mode json` with a custom `models.json` endpoint), but it runs every task in its
  own Docker container, which the laptop should not do.
- **Suite: 48 tasks in six families, all checked by scripts.** Missing information (ask, then finish) 8; a feature
  across files of a small project 8; read the docs of an unfamiliar tool and use it 6; refactor under unit tests 6;
  fix a real reported bug in a small Python package (SWE-bench Verified) 6; implement against partial tests (Exercism
  exercises of Aider's polyglot set, Python and JavaScript) 14. One pass of the suite takes est. 8.1 h of rig time on
  v1 and 5.5 h on v2 (1.47×), from the cost model in §3.2.
- **Speed is answered by per-request timings and trace replay, not by raw minutes per task.** Two runs of one agent on
  one task take different paths, so 48 paired tasks pin a time-per-task ratio only to about ±20 % (est.). Re-sending
  recorded requests from a live run to each server configuration measures ms per prompt token, ms per generated token
  and the fixed cost per request on identical prompts to within a few percent, and a cost model turns them into "the
  same work" in minutes. The live runs supply the realistic token mix and the felt numbers (minutes per task, time to
  first token per turn).
- **Quality has two read-outs of very different power.** The rig's live runs give one attempt per task per arm: a
  paired standard error of ~6 points, enough to catch a broken engine, not a small loss (est.). The file comparison
  (UD against the lead file) runs on the quant machine at 48 tasks × 10 attempts per file: paired standard error
  ~2.4 points, a non-inferiority margin of 6 points with 80 % power if the files are equal (est., Miller's formula at
  our noise). The missing-information family reports ask rate, false-ask rate and pass after asking, the
  decomposition that located Q41's miss_param gap.
- **pi is a light client for this rig, with four traps.**
  - Its default system prompt and four tool schemas are ~5.5k characters, est. ~1.4–1.8k tokens, against BFCL's
    ~5k-token first prompt and OpenCode's est. 10–15k.
  - It always streams, so the check measures the streamed path that no published number here has used.
  - It sends each reply's `reasoning_content` back in the history, as the makers ask. A history that re-renders
    differently from the generated tokens therefore costs the whole previous reasoning in prefill.
  - It has no turn limit, its bash tool has no default timeout, and a compaction (at context window minus 16,384
    tokens) costs one full re-read of the summarized span on the rig. The runner caps turns and wall time, and the
    context window is set to the server's so compaction never fires in the suite.
- **The one mechanism the pilot must settle first is cache reuse per turn.** From llama-server's code (est.): once a
  slot's checkpoint list is full, older checkpoints within 8,192 tokens of an earlier one are evicted
  (`--checkpoint-min-step`, default 8,192). If pi's re-rendered history differs from what Flint generated, the server
  can then fall back to a checkpoint thousands of tokens back, est. 60–100 s of extra prefill per turn on v1. `cache_n`
  per request shows which case holds.
- **Tool requests keep two v2 levers switched off.** Backend sampling (Q55) and, by its documented rule, the screened
  head (Q92) are off for any request with a grammar, and every pi request carries the tool-call grammar. On agent
  traffic v2's decode gain is therefore the FP32 sums and the norm fusion, est. 1.10× per token, while prefill gains
  est. 3.5× per new token. The same rule makes K2h-r16's GPTQ head, which needs no screen, about 3 % cheaper per token
  than UD on tool requests (est.), where screened files would gain nothing.
- **Plan:** a quant-machine calibration of the suite first (no rig time), a rig pilot of ≤ 2 h with a go/no-go list,
  then a full rig day of ~19–20 h: live v1 and v2 passes on UD (13.6 h, crossover order), and replays on 4–8 recorded
  traces for stock llama.cpp, v1, v2 at v1's flags, streaming off, the checkpoint policy, the host flags and the lead
  file. The file-quality runs take est. 4–7 h of the quant machine.
- **First concrete step:** write the six families' task directories with oracle and empty-change validation, the
  proxy and the runner (laptop, no model), then run the suite twice with UD on one quant-machine server to calibrate
  pass rates and tokens per task.

## The suite, ranked

Ranked by information per rig minute for the open decisions, then realism. Minutes per task are rig time on v1 / v2
with UD, from the cost model in §3.2 (est.); "new" prompt tokens are those the server processes, "re-sent" the context
sent over all turns.

| # | Family | Why it is realistic, and what it tests | Turns; generated / new prompt / re-sent tokens per task (est.) | Rig min per task v1 / v2 (est.) | Tasks | Pass check (script) | Source and licence |
|---|---|---|---|---|---|---|---|
| 1 | **Missing information: ask, then finish** (6 tasks whose prompt omits a value the code needs, 2 complete twins as controls) | Real requests leave things out; Q41's only replicated quality gap is BFCL miss_param, where G3 asks as often as UD but fails more often after the answer | ~10; 6k / 13k / ~80k | 6.8 / 4.8 | 8 | Phase 1 must end with a question naming the missing item and no edit to the target files (`git diff` on them empty); the runner then sends the scripted answer in the same session; pass = hidden tests green. Controls pass only if the agent finishes without asking | Written for this project |
| 2 | **Feature across files of a small project** (a ~2k-line Python package with tests, written for the suite) | The commonest agent job: read, plan, edit three or four modules, run the tests | ~15; 12k / 27k / ~230k | 15.0 / 10.3 | 8 | Hidden feature tests plus the existing suite pass; `ruff check` clean | Written for this project |
| 3 | **Read the docs of an unfamiliar tool and use it** (a small CLI shipped with a 4–8k-token manual; the tool is new, so recall cannot help) | Agents spend most input tokens reading tool output and docs; Q106 found our files weakest on the tokens Flint reads | ~8; 4k / 15k / ~80k | 5.8 / 3.6 | 6 | The produced file equals the expected one byte for byte (or as parsed JSON/CSV) | Written for this project |
| 4 | **Refactor under unit tests** (rename across files, extract a module, change a signature with all callers) | Behaviour must not change; long `edit` arguments (pi's `edits[]` array of objects) stress Flint's typed tool arguments | ~13; 10k / 22k / ~160k | 11.9 / 8.3 | 6 | Existing tests pass; structural checks (old symbol absent by `grep`, new module importable); no test file modified | Written for this project |
| 5 | **Fix a reported bug in a real package** (SWE-bench Verified instances marked "<15 min fix" from `psf/requests`, `pytest-dev/pytest`, `pallets/flask`, `sympy/sympy`, set up without Docker) | Real issue text in a real codebase; repository search and the longest contexts | ~20; 14k / 40k / ~450k | 21.0 / 13.7 | 6 | The instance's FAIL_TO_PASS and PASS_TO_PASS tests after applying its test patch | Harness MIT; dataset card states no licence; repos Apache-2.0, MIT, BSD-3; instance IDs only, downloaded at run time |
| 6 | **Implement against partial tests** (Exercism exercises from Aider's polyglot selection: 8 Python, 6 JavaScript; the agent sees the instructions, the stub and about a third of the test cases) | Cheap, public, comparable with Aider's and Unsloth's polyglot ladders; the hidden cases test generalisation | ~8; 5k / 10k / ~55k | 5.5 / 3.9 | 14 | The full canonical test file passes (pytest, jest); the visible test file unchanged | Exercism tracks, MIT (taken from `exercism/python` and `exercism/javascript`, not from Aider's repository, which has no licence file) |
| | **Suite** | | | **8.1 h / 5.5 h per pass** | **48** | | |

- **Families 1–4 are new tasks**, so Flint cannot have seen them; families 5 and 6 predate its June 2026 knowledge
  cutoff. Contamination inflates absolute pass rates there but not comparisons between files or engines of the same
  weights ([task-level checks](task_level_quality_checks.md) §5).
- **Every task is validated before any model time:** its reference solution passes the check (oracle) and an
  untouched workspace fails it (empty change), as Harbor validates its tasks with its `oracle` and `nop` agents.

## The comparison design

| Comparison | Arms | Where | Tasks × attempts | Read-out | What it decides |
|---|---|---|---|---|---|
| **Engine speed** | v1 as released (its documented flags) against v2 (combined build, `-b 2048 -ub 512`), same UD file | Rig: live passes; replay of recorded traces | Live 48 × 1 per arm, crossover order; replay 4–8 traces | Minutes per task and time per successful task (task bootstrap); time to first token per turn; ms per prompt and generated token and fixed cost per request on identical prompts (replay, ±1–3 % est.); "same work" minutes from the cost model | The headline: how much faster v2 is on agent work |
| **Engine quality** | Same pairs | Rig, live | 48 × 1 | Paired pass difference (SE ~6 points est.); malformed calls, runaways, NaN | Breakage gate: lower 90 % bound above −15 points and process failures not worse; the identity tests of Q117 are the real evidence |
| **Stock baseline** | Stock llama.cpp at v1's flags | Rig, replay | 4 traces | Same per-token read-outs | The README's stock-against-fork claim on agent work |
| **File quality** | UD against the lead file, K2h-r16; Unsloth Q8_0 as an anchor if time allows | Quant machine | 48 × 10 per file, files interleaved per task, attempts split across GPU sets | Paired difference with task-clustered SE and bootstrap; non-inferiority at 6 points; ask rate, false-ask rate, pass after asking; length ratio; process failures | Whether the lead file holds UD's quality on agent work |
| **File speed** | Same two files | Rig, replay of the same 8 traces | 8 traces each | ms per token on identical prompts | Time per successful task of each file (with the quant machine's tokens and pass rates) |
| **Serving settings** | v2 with: streaming off; Q119's host flags (`-t 1`); `--ctx-checkpoints`/`--checkpoint-min-step` policy; v1's `-b 512 -ub 128`; reasoning dropped from the history (speed only) | Rig, replay | 4 traces each | ms per token, time to first token, `cache_n` | v2's setup guide |

## 1. What others do

### 1.1 Agentic benchmarks and harnesses

| Set or harness | Tasks | What the agent does; how it is checked | Infrastructure | Published size of a task | Licence | Use here |
|---|---|---|---|---|---|---|
| Terminal-Bench 2.0 / 2.1 | 89 (4 easy, 55 medium, 30 hard); 2.1 repairs 28 of them | Terminal tasks in a container, checked by tests | Docker image per task; Harbor | Most trials under 25 model calls and 20 minutes on API models, some up to 2 h; ≥ 5 runs per agent–model pair | Apache-2.0 | No: Docker per task, and est. hours per task on the rig. Flint's makers report 54.9 (BF16) at up to 5 h per task |
| OpenThoughts-TBLite | 100 (40 easy, 26 medium, 26 hard, 8 extreme) | As Terminal-Bench; correlates with it at r = 0.911 | Harbor, Docker | — | Apache-2.0 | Optional external anchor, quality only, on the quant machine |
| SWE-bench Verified (and mini-SWE-agent) | 500, of which 194 "<15 min fix" | Fix a GitHub issue; FAIL_TO_PASS and PASS_TO_PASS tests | Docker images of 1–3 GB per instance, ~100 GB cache | mini-SWE-agent: 27–88 model calls per instance; SWE-rebench: Qwen3.6-35B-A3B 3.73 M tokens per problem, 87.5 % cached | Harness MIT; dataset card states no licence | A few easy instances from small pure-Python repositories, set up without Docker (family 5) |
| Aider polyglot | 225 Exercism exercises (Python 34, JavaScript 49, Go 39, Java 47, C++ 26, Rust 30), the hardest of 697 | Edit a stub until hidden tests pass; two tries | Docker recommended (runs model code) | ~10–15k prompt tokens per case; completion ~1.6k for non-reasoning models, 8–12k for reasoning ones (Kimi K2 10.5k / 1.6k; DeepSeek R1-0528 11.8k / 8.2k at 717 s per case) | Exercism MIT; Aider's repository has no licence file | Python and JavaScript subset with partial visible tests (family 6) |
| τ-bench, τ²-bench | 50 / 115 / 114 per domain | Customer-service tools; pass^k | An LLM user simulator | — | MIT | No: the simulator would share the rig and set the score |
| BFCL v4 multi-turn | 800 | Tool calls against local Python backends; judge-free | None | ~12 steps of ~200 tokens | Apache-2.0 | Already lane E2 of Q41 |
| Ambig-SWE | Underspecified variants of SWE-bench Verified issues | Detect the gap, ask, use the answer; LLM user proxy | OpenHands, Docker | Interaction gains up to 74 % over non-interactive runs | — | The idea behind family 1, with a scripted answer instead of an LLM user |
| Harbor | Framework; 98 adapters (SWE-bench, Aider polyglot, LiveCodeBench, BFCL, τ³, …) | Installs a CLI agent in each task container and runs it headless | Docker by default | — | Apache-2.0 | Its pi adapter confirms the invocation used here (`pi --print --mode json --session-dir …`, a `models.json` provider pointing at any OpenAI-compatible endpoint); its usage accounting counts prompt tokens as input + cache read + cache write |

- **One-shot coding scores say little about agent time or agent scores.** LiveCodeBench is one long reasoning per
  problem (Flint averages 12.9k tokens per sample, Q71); an agent task is many short turns with re-sent context.
  The same model moves 10–20 points with the scaffold: Qwen3-Coder-480B resolves 69.6 % of SWE-bench Verified with
  OpenHands and 55.4 % with mini-SWE-agent, and scores 23.9 % on Terminal-Bench 2.0.
- **What carries over:** Harbor's oracle and empty-change validation of every task, isolated per-task workspaces,
  pinned agent versions, and several attempts per task with pass^k next to pass@1 (τ-bench). Anthropic's evaluation
  guidance logs turns, tokens, time to first token, output tok/s and time to last token per trial.

### 1.2 Quantized against full precision on agentic work

- **Unsloth, DeepSeek-V3.1 on Aider polyglot (pass-2, ~3 runs per score, median, no spread published):** full
  precision 71.6, 5-bit 70.7, 4-bit 69.7, 3-bit 68.4, 2-bit 65.8, 1-bit 55.7 (non-thinking); thinking 76.1 full,
  75.6 at 3 bits. At 225 cases one run's standard error is ~3 points, so the 4–5-bit steps are within noise (est.).
- **Aider, Qwen2.5-Coder-32B across servers (133 cases, single runs):** BF16 providers 72.2; Ollama q4_K_M 66.9, q2_K
  61.7; Ollama fp16 at its 2k default context 51.9 with only 46.2 % well-formed edits. A truncated context cost more
  than any quantization: the server's context and template settings must be fixed and checked in every arm.
- **Jang et al., τ²-bench (114 tasks × 4 trials per cell):** Qwen3.6-35B-A3B AWQ INT4 against BF16, retail 59.9 → 60.4,
  telecom 98.2 → 96.3; no cell significant after correction, equivalence at ±7.5 points in 7 of 10 cells, while 4-bit
  amplified tool-name hallucinations up to 2.5×.
- **ACBench:** 4-bit GPTQ/AWQ within 1–3 % on workflow and tool use, 10–15 % lower on real-world agent tasks. Red Hat's
  Gemma-4-26B-A4B NVFP4: BFCL multi-turn unchanged, agentic −11.8 points ([task-level checks](task_level_quality_checks.md) §1).
- **Community, single runs (S):** SWE-bench Verified's first 100 tasks with mini-SWE-agent, Qwen3.5-35B-A3B Q4_K_M,
  UD-Q6_K_XL and Q8_0 each resolved 59.
- **Consequence:** at 4–5 bits on a good model, agentic score differences sit inside the noise of single runs of a few
  hundred tasks, while process failures (wrong tool names, malformed calls) and serving configuration move more. The
  check logs process failures per attempt and holds every server setting fixed across arms.

### 1.3 How people time agents

- **Datacenter traces:** with working prefix caching, decode is 91.0–98.6 % of LLM time for Claude Code (through
  Harbor) driving Qwen3.6-27B and Gemma4-31B on vLLM with two H100s; mean input:output ratios are 53.9–559.8×, but new
  input per output token only 1.5–7.3×; cache hit rates 84.6–99.5 %; tools 2–29 % of runtime; cache thrashing cut
  generation throughput by up to 66 % (Yuan et al., IISWC 2026). TraceLab's 4,300 Claude Code and Codex sessions: 294×
  more input than output, 95.7 % prefix-cache hits (84.4 % on new user turns), ~13 s of model time and ~18 s of tool
  time per step. Manus calls the KV-cache hit rate the single most important metric of an agent at ~100:1
  input:output.
- **This rig is different:** today prompt processing is about half of a multi-turn episode's time (decode 49 % of a
  BFCL episode, [answer time](answer_time.md) §3.1), est. a quarter after the v2 prompt pieces. Cache hits and the
  speed of the fresh tokens therefore set the agent's speed here far more than on datacenter GPUs.
- **Local agents:** published wall-clock numbers with a prefill/decode split for local agents on consumer hardware
  were not found. Aider records seconds per case (Ollama runs 35.7–188.6 s per case, hardware not stated).
- **Practice to copy:** server-side counters rather than client wall time alone; the cache hit rate per request; time
  to first token per turn; time and cost per resolved task as a ratio of sums with a bootstrap interval (Cost-of-Pass);
  the same infrastructure for both arms, since resource limits alone moved Terminal-Bench by 6 points and
  infrastructure errors reached 5.8 % (Anthropic); infrastructure failures re-run, not scored.
- **Streaming:** llama-server's `timings` do not depend on whether the client streams; streaming adds the server's
  per-token re-parse of the reply ([host in the decode loop](host_decode_loop.md) §4) and can change parsed tool calls
  (llama.cpp issue #19382). The stream setting is fixed per arm and compared on its own.

### 1.4 How many tasks separate two models at our noise

Miller's sample size for paired comparisons with K attempts per task: n = (z₀.₉₇₅ + z₀.₈)² (ω² + 2σ²/K) / δ², where σ²
is the mean within-task variance and ω² the variance of the true per-task difference between the arms. Two attempts
disagree on a task with probability E[2p(1−p)] = 2σ², so Q41's 15–20 % disagreement means σ² ≈ 0.075–0.10. For two
quants of one model ω² is near 0 (Q30), with some category effect (miss_param), taken here as 0.01.

| σ² | ω² | Tasks for 10 points at K = 1 / 3 / 5 / 10 | Tasks for 5 points at K = 1 / 3 / 5 / 10 |
|---|---|---|---|
| 0.075 | 0 | 118 / 40 / 24 / 12 | 471 / 157 / 95 / 48 |
| 0.10 | 0.01 | 165 / 61 / 40 / 24 | 660 / 241 / 157 / 95 |
| 0.10 | 0.02 | 173 / 69 / 48 / 32 | 691 / 273 / 189 / 126 |

- **Minimum detectable difference of this design** (σ² 0.09, 80 % power, two-sided α 0.05, est.): the rig's 48 × 1 per
  arm, 17 points; the quant machine's 48 × 10 per file, 6.8 points with ω² = 0.01 (5.4 with ω² = 0).
- **Tasks every arm always passes or always fails carry no paired information.** The calibration run marks them; the
  suite keeps them for the user-facing pass rate, and the power figures above already average over them.
- **Published run-to-run noise agrees:** SWE-bench Verified pass rates vary by more than 1.5 points per run even at
  temperature 0, and a single run's estimate by 2.2–6.0 points depending on which run is taken (60,000 trajectories);
  Anthropic distrusts leaderboard gaps under 3 points.

## 2. Pi specifics

### 2.1 Driving pi non-interactively and reproducibly

- **Modes:** `--print` writes only the final text; `--mode json` runs the supplied prompt and streams a session header
  and every event as JSONL until the run settles; `--mode rpc` keeps a process for many prompts. The check uses JSON
  mode: one process per task phase, events timestamped by the runner on arrival.
- **Continuing a session:** `--session <file>` continues a saved session, which is how the missing-information
  family's scripted answer enters the same conversation; `--session-dir` puts each attempt's session file in its own
  results folder.
- **Isolation flags:** `--no-context-files` (no AGENTS.md or CLAUDE.md picked up from parent directories),
  `--no-extensions` with one explicit `-e` extension, `--no-skills`, `--no-prompt-templates`, `--no-approve` (ignore
  any project-local `.pi` settings in a task), `--offline`, and a private agent directory (`PI_CODING_AGENT_DIR`) that
  holds only the check's `models.json` and `settings.json`, so no personal credentials or settings reach a run.
- **Workspace:** a fresh copy of the task per attempt, always mounted at the same path. pi writes the working
  directory into its system prompt, so a fixed path keeps the system prompt byte-identical across tasks and its prefix
  reusable by the server.
- **Sampling and seeds:** pi's OpenAI-compatible provider sends `temperature` only when set and never `top_p`,
  `top_k` or a seed. The makers' settings (temperature 1.0, top-p 0.95, top-k 20, min-p 0) therefore go on the
  server command line, and a small extension deletes any sampling field from the payload, as the makers' own pi
  transport does. Seeds stay random: at temperature 1.0 attempts are independent samples, and the rig's results are
  not bit-reproducible anyway (prompt caching and batch shapes change summation order).
- **Tool calls per reply:** llama-server takes `parallel_tool_calls` from the chat template's capability when the
  client sends none (`server-common.cpp:1442`), so Flint can make several calls in one reply, as the makers' engine
  allows; the pilot's logs show whether it does.
- **No limits of its own:** pi has no turn limit, and its bash tool has no default timeout. The runner kills a run at
  a turn cap (40) or a wall-clock cap per family (est. 3× the family's median), and records it as a runaway.
- **Failure handling:** pi retries transient provider errors three times by default and times out an idle HTTP
  response after 300 s. The check turns retries off and raises the timeout to 30 min, so a long prefill on v1 is not
  cut and every infrastructure failure is visible; the runner re-runs such attempts instead of scoring them.

### 2.2 The makers' pi configuration

Recursive's engine installs a pinned pi (0.73.1) and registers Flint through a small extension
([rec](rec_engine.md), `cli/clients/pi/`):
- the model is declared with `reasoning: false`, `contextWindow` and `maxTokens` both equal to the engine's context
  (262,144), and compat flags `supportsStore`, `supportsDeveloperRole`, `supportsReasoningEffort` and
  `supportsStrictMode` false, `maxTokensField: "max_tokens"`;
- its transport deletes `max_completion_tokens`, `temperature`, `top_p` and `top_k` from every request, so the
  engine's defaults (1.0 / 0.95 / 20) always apply;
- it keeps pi's default system prompt and tools.

Flint has no thinking switch, so no thinking-level map is needed. pi still records the `reasoning_content` that
llama-server returns as a thinking block and sends it back as `reasoning_content` on the same model, whatever the
`reasoning` flag (pi-ai's OpenAI-completions provider), which is the history the makers ask for. The check's provider
entry copies the makers' declaration with one change: `maxTokens` 32,768 per request, the makers' minimum for hard
problems, so a looping reply cannot run for hours.

### 2.3 What pi logs, and what the server has to add

| Quantity | pi session file / JSON events | Logging proxy (from llama-server's stream) | `/metrics`, server log |
|---|---|---|---|
| Messages, tool calls and arguments, tool results and errors, full system prompt and tool schemas | yes | request bodies | — |
| Prompt tokens per request: new and cached | `usage.input` (= prompt − cached), `usage.cacheRead` | `timings.prompt_n`, `timings.cache_n`; `usage.prompt_tokens_details.cached_tokens` | `llamacpp:prompt_tokens_total` |
| Generated tokens | `usage.output` | `timings.predicted_n` | `llamacpp:tokens_predicted_total` |
| Prefill and decode time per request | — | `timings.prompt_ms`, `timings.predicted_ms` | `prompt_seconds_total`, `tokens_predicted_seconds_total` |
| Time to first token per turn | arrival time of the first delta | first content or reasoning chunk after the request | — |
| Tool execution time | `tool_execution_start` / `_end` arrival times | — | — |
| Compactions, retries, aborted or errored turns | `compaction_*`, `auto_retry_*`, `stopReason` | HTTP status, truncated streams | — |
| Checkpoint restores, full re-processing, backend-sampling and screened-head decisions | — | — | server log lines |

- **Server additions:** `--metrics` (the endpoint is off by default), the server log kept per run, and nothing else in
  the request path. llama-server already attaches `timings` and the OpenAI `usage` object to the last streamed chunk;
  `return_progress` and `timings_per_token` stay off so that pi sees exactly what users get.
- **The proxy** listens on the laptop, forwards every byte to the tunnel unchanged, and writes one JSON line per
  request: task, attempt and phase (from a header the extension adds), request size, send time, first byte, first
  content chunk, end, the final `timings` and `usage`, the finish reason, and the gzipped request body for replay.
- **Cache diagnosis without a tokenizer:** a turn reused everything Flint generated when its `cache_n` equals the
  previous request's `cache_n + prompt_n + predicted_n`; it re-read the previous reply when `cache_n` is about the
  previous prompt's length; it fell back to an older checkpoint when `cache_n` is smaller still.
- **Slot files** (`--slot-save-path`, `/slots/<id>?action=save|restore`) are needed only for the saved-prefix variant;
  pi's shared prefix is small (§2.4), so it is a replay variant, not part of the default setup.

### 2.4 The context pi re-sends, and compaction

- **The fixed prefix:** pi's default system prompt for its four tools (read, bash, edit, write) is 2,664 characters:
  a preamble, one line per tool, ten rules, a paragraph pointing at pi's own documentation, and the working directory.
  The four tool schemas are 2,841 characters of JSON. Together est. ~1.4–1.8k tokens with Flint's template, about a
  third of BFCL's first prompt (median ~5k tokens, Q111). On v1 it costs est. ~13 s to process cold and on v2 est.
  ~4 s; across tasks it is reusable only if a checkpoint at its end survives (§6.3 measures it).
- **Per turn:** a turn's new tokens are its tool result plus whatever of the previous reply the server cannot reuse.
  Tool outputs are capped by pi at 2,000 lines or 50 KB (est. ≤ 12k tokens) per call. With reasoning kept in the
  history, a reply's reasoning stays in the context for the rest of the task, so contexts grow by every turn's
  reasoning (est. 400–600 tokens per coding turn, unmeasured).
- **Two cache paths:**
  - **Exact reuse:** llama-server's live sliding-window cells hold the end of the generated reply, so a next request
    that matches the generated tokens needs no checkpoint and processes only the tool result.
  - **Re-render mismatch:** the server parses Flint's `<tool_call>` XML into JSON, pi sends JSON back, and the
    template writes XML again (`tojson` for non-string values; pi's `edit` tool takes an array of objects). Any
    difference moves the divergence back to the previous prompt's end or to an older checkpoint.
    [Prompt processing](prefill_pascal.md) §4.2 found the BFCL harness re-reading the previous reply on every turn.
    Combined with the eviction rule above, the worst case is up to ~8k tokens per turn (est.).
- **Compaction:** pi compacts when the context exceeds the model's context window minus 16,384 tokens (configurable),
  keeps the last 20,000 tokens and summarizes the rest with an extra model request.
  - **On the rig** that request is a fresh prompt of the whole summarized span: at today's ~7–30 ms per prompt token
    (est., by depth), a 100k-token span is est. 12–50 minutes before the summary starts, followed by a new prefix of
    summary plus 20k kept tokens.
  - **Users of this rig** should set the context window to the server's `-c` and keep compaction rare. The suite's
    tasks stay far below it, and any compaction is logged as an event of its own.

## 3. The task suite

### 3.1 The families in detail

- **Common to all tasks:**
  - a directory with the prompt, a workspace builder (a template directory, or a repository URL, commit and patch),
    a setup step (a `uv` virtual environment or a shared `node_modules`), the hidden checks, the reference solution,
    a turn cap and a wall-clock cap;
  - the hidden tests and the reference solution never enter the workspace;
  - a short addendum to pi's system prompt, identical for every task: "If the task lacks information you need, ask
    and stop." Users of agents commonly carry such a rule, and the same text keeps the prefix shared.
- **Family 1, missing information:**
  - the six must-ask tasks omit a value that is nowhere in the repository (a rate, a code, the name of a target among
    two candidates) and that a guess cannot pass;
  - the two controls give everything;
  - the scripted answer is fixed text, so no LLM plays the user;
  - reported separately: ask rate on must-ask tasks, false-ask rate on controls, pass after asking, pass after a
    guess (always 0 by construction).
- **Family 2, feature across files:** one purpose-built package (a small inventory or log-processing tool with CLI,
  storage, report and config modules and ~60 tests). Each task adds a feature that touches 3–4 modules and its tests.
- **Family 3, docs and tool:** a small command-line tool (a query language over TSV files, delivered as a Python
  zipapp) with a reference manual of 4–8k tokens. Each task needs flags that only the manual explains, and the result
  is a file the check compares exactly.
- **Family 4, refactor:** on the same package as family 2, behaviour-preserving changes whose correctness the
  existing tests plus structural checks decide.
- **Family 5, real bugs:**
  - **Selection:** SWE-bench Verified instances labelled "<15 min fix" from four light pure-Python repositories
    (`psf/requests` has 6 such instances; `pytest-dev/pytest`, `pallets/flask` and `sympy/sympy` supply the rest).
  - **Environment:** each instance gets the Python version and install commands from SWE-bench's harness
    specifications, built with `uv` instead of Docker.
  - **Validation:** an instance stays only if its gold patch passes FAIL_TO_PASS and an untouched checkout fails it on
    the laptop.
- **Family 6, partial tests:**
  - **Selection:** Aider's polyglot list of exercises, taken from Exercism's MIT-licensed tracks at a pinned commit:
    8 Python and 6 JavaScript, chosen by the calibration run to avoid exercises every attempt passes.
  - **Workspace:** the instructions, the stub and the first third of the canonical test cases; the reference
    solution (`.meta/`) and the remaining cases are withheld.
  - **Check:** the hidden full test file.

### 3.2 Cost model and per-task estimates

Time per task (est.):

  T = Σ over requests [ r + n_new · p(d) + n_gen · t(d) ] + T_tools + T_client

| Term | v1 (released flags) | v2 (combined, `-ub 512`) | Basis |
|---|---|---|---|
| r, fixed per request | 2.2 s | ~1.0 s | Q58's fit; [answer time](answer_time.md) §3.1 after Q108–Q110 |
| p(d), per new prompt token at context depth d | 7.1 ms + 0.3 ms per 1k of depth | 2.0 ms + 0.1 ms per 1k | Q58 at ≤ 2k; Q79's fill slope (0.19 ms per 1k) × the server's 1.6× over llama-bench; [prompt processing](prefill_pascal.md) §1.4 after rows 1–3 |
| t(d), per generated token, tool requests, UD | 39 ms + 0.3 ms per 1k | 35.5 ms + 0.3 ms per 1k | Q58's 35.8 ms + 3.3 ms of host sampling (Q55); v2: −2.6 ms FP32 sums (Q76), −1.0 ms norm fusion (Q100, est. 3 %); screened head off for tool requests; depth slope from Q54 |
| T_tools | 1–3 s per turn | same | the tasks' own test runs on the laptop |

Per family (turns, tokens and depth as in the suite table; est.):

| Family | v1: prefill / fixed / decode / tools (s) | v1 min | v2: prefill / fixed / decode / tools (s) | v2 min | v2 speed-up |
|---|---|---|---|---|---|
| 1 Missing information | 124 / 22 / 248 / 15 | 6.8 | 36 / 10 / 227 / 15 | 4.8 | 1.42× |
| 2 Feature across files | 313 / 33 / 522 / 30 | 15.0 | 95 / 15 / 480 / 30 | 10.3 | 1.45× |
| 3 Docs and tool | 152 / 18 / 168 / 8 | 5.8 | 45 / 8 / 154 / 8 | 3.6 | 1.61× |
| 4 Refactor | 235 / 29 / 426 / 26 | 11.9 | 70 / 13 / 391 / 26 | 8.3 | 1.43× |
| 5 Real bugs | 524 / 44 / 630 / 60 | 21.0 | 160 / 20 / 581 / 60 | 13.7 | 1.53× |
| 6 Partial tests | 89 / 18 / 204 / 16 | 5.5 | 26 / 8 / 186 / 16 | 3.9 | 1.39× |

- **What dominates:** on v1 prefill and fixed costs are a third to a half of a task, on v2 14–25 %. Decode, which v2
  speeds up by only ~10 % on tool requests, becomes most of v2's time.
- **Time to first token per turn** (est.): v1 ~13 s on short turns and ~23 s on family 2's ~1.8k new tokens per
  turn; v2 ~4 s and ~7 s.
- **The largest uncertainties:**
  - Flint's reasoning per coding turn (assumed 400–600 tokens; BFCL steps average ~200 and a LiveCodeBench sample
    12.9k);
  - the cache path of §2.4, which can add est. 60–100 s per turn on v1 in its worst case;
  - the served overheads on v2 (Q118 and Q119 are unmeasured).

  The calibration run measures the first, the pilot the other two, and the suite is rescaled before the full run.

### 3.3 Public sets: usable as they are or not

| Set | Usable as is | Licence | Note |
|---|---|---|---|
| Exercism Python and JavaScript tracks (Aider's polyglot selection) | Yes, with the visible tests cut to a third | MIT | Take the files from Exercism, not from Aider's repository (no licence file) |
| SWE-bench Verified | Yes, by instance ID, environment without Docker | Harness MIT; dataset card states no licence | Light pure-Python repositories only; validate the gold patch locally |
| OpenThoughts-TBLite, Terminal-Bench 2.x | Yes through Harbor, Docker per task | Apache-2.0 | Quant machine only, optional |
| BFCL v4 multi-turn | Yes (Q41's lane E2) | Apache-2.0 | Tool calling, not coding |
| τ²-bench, Ambig-SWE | No: an LLM user is part of the task | MIT / — | Family 1 uses a scripted answer instead |
| Aider's polyglot repository itself | Not redistributed | No licence file | — |

## 4. Comparisons

### 4.1 Engine: v1 against v2, the same file

- **File:** UD, the file users run and the quality bar. On v1 it runs with the MoE down-projection rescale (it has no
  FP32 sums); on v2 with `LLAMA_NO_MOE_DOWN_RESCALE=tg` (FP32 sums on one-token steps).
- **Flags:** each engine as a user would run it. v1 with its documented chat flags (`-c 262144 -b 512 -ub 128 -np 1
  -bs --ctx-checkpoints 4 --cache-ram 0`); v2 with the same except `-b 2048 -ub 512`, which the on-card mask (Q110)
  makes affordable at 262,144 tokens. Both with `--jinja`, the makers' sampling and `--metrics`. Both builds share
  upstream base `d77256999`, so the chat template, tool-call parser and checkpoint code are the same in both arms.
- **Live passes:** all 48 tasks once per arm, in crossover order (v2 on half H1, v1 on H1, v1 on H2, v2 on H2; the
  halves balanced by family), so drift over the day falls equally on both arms; task order randomized within a block.
- **Pieces:** the per-piece shares of Q117 come from its llama-bench grid (runbook step 5; [kernel note](pascal_mmvq_integration.md)
  §16). The agent check adds what llama-bench cannot show: v2 at v1's `-b 512 -ub 128` in replay separates the flag
  change from the code, and the decode pieces show up in t(d).
- **Quality:** the live pairs are the breakage gate of the comparison design table. A quality loss of v2 beyond
  summation-order noise would contradict the byte-identical prompt logits and decode text of Q117's checks; if the
  gate's point estimate is below −5 points, the same pairs run on the quant machine with the Pascal kernels forced
  (`GGML_CUDA_MMVQ_PASCAL=2`) at K = 5.

### 4.2 File: UD against the lead file

- **The pair:** UD (Q8_0 head) against K2h-r16 (GPTQ Q5_0 head, F16 router; 2.091 against 2.213 GB per token).
- **Head treatment:** Q125 asks for the same head treatment on both, and K2sr (Q126, the screened Q8_0 head) is the
  lead for the fork on chat. On tool requests the screen is off by its rule (any grammar disables it), so on agent
  traffic K2sr runs a full Q8_0 head and reads about as many bytes as UD (2.206 against 2.213 GB per token, est.
  from Q126's 2.053 with the screen). K2sr joins the file arm only if
  the pilot's server log shows the screen on for tool requests, or once GPU sampling with a host grammar check (Q67)
  lets tool requests keep it.
- **Quality on the quant machine:**
  - 48 tasks × 10 attempts per file, attempts interleaved per task, split across the 3090-based and V100 GPU sets as
    in Q41;
  - read-out: non-inferiority at 6 points (lower 90 % bound of K2h-r16 − UD above −6), family 1's decomposition,
    process failures (upper 90 % bound ≤ +2 points), and the length ratio, the length gate of
    [answer time](answer_time.md) §5.1.
- **Speed on the rig:** replay of the same 8 traces with each file on v2.
- **Time per successful task:** (per-token cost ratio from the replay, applied to each file's token counts from the
  quant machine) / (pass ratio from the quant machine), with a task bootstrap. This is the TPCA metric of
  [answer time](answer_time.md) on agent work.

### 4.3 Serving settings

All on v2 in replay, 4 traces each:
- **Streaming off:** the cost of the streamed path (Q118, Q122).
- **Q119's host flags** (`-t 1`, or `OMP_WAIT_POLICY=passive`): only if Q118's measurement has not already folded
  them into v2's flags.
- **Checkpoint policy:** the default (`--ctx-checkpoints 4`, `--checkpoint-min-step 8192`) against more checkpoints
  without the minimum spacing (e.g. `--ctx-checkpoints 6 --checkpoint-min-step 0`), within the host RAM budget of
  §6.5. Replay is the hard case here: a replayed reply never matches the recorded one, so every turn needs a
  checkpoint.
- **Reasoning dropped from the history:** what keeping reasoning costs in prefill. Speed only; the makers' advice to
  keep it stands until a quality run says otherwise.
- **Prompt cache off:** not run. It would re-read every request's whole context, est. 20–60 minutes per task on v1.

### 4.4 The split between the rig and the quant machine

| Question | Rig | Quant machine |
|---|---|---|
| Engine speed (v1, v2, stock, pieces, settings) | live passes and replays | — (never speed) |
| Engine quality | breakage gate from the live pairs | only if the gate's point estimate is below −5 points: forced Pascal kernels, K = 5 |
| File quality | the K2h-r16 replay gives speed only | 48 × 10 per file, the decision |
| Calibration (task pass rates, tokens per task, harness bugs) | — | suite × 2 with UD before any rig time |

- **On the quant machine** pi runs in a small container there (Node 22+, Python, pytest, jest), one workspace per
  attempt, three llama-server instances (one per GPU set) with ~6 slots each and the same template, sampling, context
  and checkpoint flags as the rig.
- **Cost** (est.): ~4 minutes per attempt per slot (~8k generated tokens at ~50 tok/s per slot, plus fast prefill and
  tool time), ~270 attempts per hour on three servers. The calibration (96 attempts) takes ~0.5 h, the file
  comparison (960 attempts) ~3.6 h, and each optional extra file (the Q8_0 anchor, K2sr) ~1.8 h more. Its job times
  are for scheduling only.

## 5. What to measure and report

### 5.1 Measures

- **Per request:**
  - client: time to first token and end time;
  - server: `prompt_n`, `cache_n`, `prompt_ms`, `predicted_n`, `predicted_ms` and the depth (`cache_n + prompt_n`);
  - cache path (exact reuse, previous reply re-read, older checkpoint);
  - tool calls in the reply, finish reason.
- **Per task attempt:**
  - wall-clock, split into prefill, decode, fixed cost (request time minus prefill and decode), tool time and client
    time;
  - turns, prompt and generated tokens (new and re-sent), peak context, mean cache hit rate (`cache_n` over
    `cache_n + prompt_n`);
  - success, and the process failures:
    - tool errors (pi's `isError` results);
    - malformed calls (tool-call markup returned as text, or arguments pi cannot parse);
    - a call inside the reasoning (llama.cpp issue #31000);
    - replies stopped at `maxTokens`, runaways stopped by the caps, compactions, infrastructure failures.
- **Per comparison:**
  - speed:
    - ratios of ms per prompt token, ms per generated token and fixed cost per request on identical prompts
      (replay; intervals by bootstrap over requests clustered by trace);
    - "same work" minutes on the reference trace from the cost model;
    - live minutes per task (geometric mean of paired ratios, task bootstrap);
    - time per successful task as Σ time / Σ successes per arm, with a paired task bootstrap;
  - quality: pass@1 per task (mean over attempts) and the paired difference with plain and task-clustered standard
    errors, 90 % and 95 % intervals, a 10,000-resample task bootstrap, the exact McNemar test on attempt 1, and pass^k
    for k = 2–4 where K = 10.
- **Tooling:** the per-attempt table uses the columns of [task_results.py](../../../bench/task_results.py) plus the
  agent columns, so [paired_tasks.py](../../../bench/paired_tasks.py) runs on it unchanged with an `agent_check` lane
  and its margins.

### 5.2 The one-page result (`docs/benchmarks/<date>-flint-agent-check.md`)

1. **One sentence of setup:** pi version, the file, the engines, nine cards, context, sampling, the number of tasks
   and attempts, where quality was scored.
2. **Headline table**, one row per arm (stock, v1, v2; each file on v2):

   | Arm | Minutes per task (median, p90) | Time per successful task | Pass@1 | First token per turn (p50, p90) | Prompt tokens processed per task | Cache hit rate | Prefill tok/s | Decode tok/s (tool requests) |
   |---|---|---|---|---|---|---|---|---|

3. **Comparisons table:** each ratio with its 95 % interval ("v2 against v1, same work: 1.xx×"), each quality
   difference with its 90 % interval and the margin verdict ("K2h-r16 − UD: −x.x points, within the 6-point margin").
4. **Per family:** minutes per task and pass@1 per arm.
5. **Process failures:** rates per arm (tool errors, malformed calls, runaways, compactions, infrastructure re-runs).
6. **What the numbers do not cover:** one agent at a time, the tasks' sizes, the quant machine's role.
7. **How to reproduce:** the server command lines, the pi configuration and the runner command.

Raw sessions, proxy logs and server logs stay outside the repository; the per-attempt tables go with the page.

## 6. The plan

### 6.1 Setup

1. **Rig:** the v2 combined tarball built on the quant machine and unpacked next to the v1 build; UD and K2h-r16 on
   the rig's model storage. One llama-server on the rig's loopback interface per arm, with the
   flags of §4.1 plus `--metrics`, the server log kept, and the watchdog of `scripts/server_bench.sh` (kill the server
   when MemAvailable drops below 400 MB).
2. **Laptop:**
   - an SSH local forward to the rig's server port, restarted in a loop with keep-alives;
   - the logging proxy on a loopback port in front of it;
   - a private pi agent directory with:
     - a `models.json` provider pointing at the proxy (the makers' declaration of §2.2, `contextWindow` 262,144,
       `maxTokens` 32,768);
     - a `settings.json` with retries off, a 30-minute HTTP idle timeout, the default four tools, telemetry off;
     - the check's extension (delete sampling fields, set `max_tokens`, add the task header).
3. **Sandbox:** pi and its tools run under `bwrap` with only the attempt's workspace writable, a private home
   directory, a cleared environment and the agent directory read-only, so a model-written command cannot touch
   anything else on the laptop.

### 6.2 The runner (`bench/agent_check/`, to be written; laptop only)

- **`tasks/<family>/<task>/`:** prompt, builder, setup, hidden checks, reference solution, scripted answer (family 1),
  caps.
- **`validate.py`:** oracle and empty-change checks for every task, no model.
- **`proxy.py`:** the pass-through logger of §2.3 (standard-library Python, a few MB of memory).
- **`run.py`:**
  - for each arm, task and attempt: build the workspace;
  - run pi in the sandbox (`pi --mode json --provider … --model flint --no-context-files --no-extensions -e
    check.ts --no-skills --no-prompt-templates --no-approve --offline --session-dir <attempt>/session
    --append-system-prompt <addendum> "<prompt>"`);
  - timestamp every event and enforce the caps;
  - run phase 2 for family 1 (`--session <file> "<answer>"`);
  - snapshot `/metrics` before and after;
  - run the hidden checks in the sandbox without network, and write the per-attempt row.
  - Resumable: finished attempts are skipped, so a crash or a tunnel drop loses one attempt.
- **`replay.py`:** re-sends the recorded request bodies of chosen traces in order, streamed or not, with `max_tokens`
  capped at each request's recorded completion length; records the same per-request fields.
- **`analyze.py`:** the measures of §5.1, the cost-model fit and the one-page tables; the quality statistics through
  `paired_tasks.py`.
- **Light by construction:** CPU on the laptop goes to pi (Node), the proxy and the tasks' own tests (seconds per
  turn). Workspaces are small (the largest is a sympy checkout) and deleted after each attempt; only diffs, sessions
  and logs are kept.

### 6.3 Pilot (≤ 2 h of rig time) and its go/no-go

Before the pilot, off the rig: every task validated; the suite run twice with UD on one quant-machine server
(calibration); the runner and proxy exercised against that server.

| Step | Rig time (est.) |
|---|---|
| Start v2 with UD at `-c 262144 -b 2048 -ub 512`; memory per card, host MemAvailable and swap logged | 10 min |
| One smoke task: proxy rows complete, `cache_n` per turn, server log lines for backend sampling, the screened head and checkpoints | 5 min |
| Six tasks on v2, one per family | ~45 min |
| Restart on v1; three of the same tasks (families 1, 3, 6) | ~25 min |
| Replay two recorded traces: v2 streamed, v2 not streamed, v1 | ~25 min |
| **Total** | **~1 h 50 min** |

Go, if all of these hold:
- **Harness:** at least 5 of 6 tasks finish without an infrastructure error (no HTTP timeout, broken stream or retry);
  every request has a proxy row with `timings`; client and server time to first token agree within 0.5 s plus the
  tunnel's round trip.
- **Tool calls:** malformed calls under 5 % of all calls; no call parsed from inside the reasoning.
- **Cache:** on turns 2 and later the median cache hit rate is ≥ 80 % and no turn re-reads more than its previous
  reply plus tool output. Otherwise the checkpoint policy of §4.3 goes into both arms' flags before the full run,
  chosen from a replay comparison.
- **Memory:** host MemAvailable never below 600 MB, no swap growth, no out-of-memory kill. Otherwise v2 drops to
  `-ub 256`.
- **Time:** measured minutes per task within 2× of §3.2 per family. Otherwise the suite is rescaled: fewer family-5
  tasks, or lower caps.
- **Replay:** reproduces the live per-token costs within 5 %.

No-go on any item means fix and repeat the failing step, not the whole pilot.

### 6.4 The full run (one rig day, ~19–20 h)

| Block | Rig time (est.) |
|---|---|
| v2, UD, half H1 (24 tasks), then v1 on H1, v1 on H2, v2 on H2 | 13.6 h (v1 8.1, v2 5.5) |
| Replays of 8 traces chosen from v2's H1 run (two each from families 2 and 5, one each from 1, 3, 4 and 6): v2 reference (8) | 1.0 h |
| v1 (4) and stock llama.cpp at v1's flags (4) | 1.75 h |
| v2 at v1's `-b 512 -ub 128` (4); streaming off (4); reasoning dropped (4) | 1.5 h |
| Checkpoint policy (4), Q119's host flags (4, if still open) | 1.0 h |
| K2h-r16 on v2 (8) | 1.0 h |
| Server restarts and loads (~12 × 3 min) | 0.6 h |
| **Total** | **~20 h** |

- If time runs short, the stock replay and the host-flag replay go first; the live passes and the v2 reference
  replay are the core.
- File quality runs on the quant machine whenever the owner books it (est. 4–7 h, §4.4); its results join the rig's
  for the one-page result.
- Q118 (M0) and Q117's llama-bench grid run in an earlier rig session if possible, so that v2's live arm already has
  the right host flags.

### 6.5 Risks

- **4 GB of host RAM:**
  - each context checkpoint holds the sliding-window cells, ~52.5 MiB at `-ub 128` and ~87.5 MiB at `-ub 512`, so
    `--ctx-checkpoints 4` takes up to ~350 MiB on v2 and 6 take ~525 MiB;
  - the mapped embedding table, pinned buffers and the desktop share the remaining ~2.6 GiB;
  - mitigations: `--cache-ram 0` always, the watchdog, `vmstat` and major page faults logged per run, and no relays or
    other jobs on the rig during the run.
- **Context checkpoints:** the eviction rule of §2.4 can turn a re-render mismatch into thousands of re-read tokens
  per turn. The pilot measures it, and the checkpoint policy is fixed before the full run, the same for both arms.
- **Rig stability:**
  - a crash can cost the rest of the day if restarting the rig needs someone at its keyboard;
  - mitigations: a resumable runner; per-card temperature, power and clocks logged every 10 s; long blocks split at
    task boundaries; the full run started while someone is near the rig.
- **A laptop on a wireless network:**
  - a dropped tunnel breaks the request in flight;
  - mitigations: the tunnel restarts itself; the runner marks the attempt as an infrastructure failure and re-runs it;
    speed numbers come from the server's own timings, so network latency only enters client-side time to first
    token;
  - the bandwidth is irrelevant (a request is at most a few hundred kB);
  - on the 2-core host, the tunnel's encryption competes with the decode loop: bulk file relays took ~90 % of a core.
    The pilot checks its CPU share under agent traffic, which is far lighter.
- **Laptop heat and disk:** no builds, no model work; small test runs only; workspaces deleted after each attempt;
  free disk checked before the run.
- **pi itself:**
  - version pinned (0.87.1 on the laptop and in the quant machine's container) with update checks off;
  - the agent runs shell commands, hence the sandbox;
  - a hung test command is ended by the wall-clock cap.
- **Parser and template issues:** tool-call XML inside the reasoning (#31000) and unbounded optional arguments (#31040)
  are counted as process failures. They hit both arms equally, since both builds share the server code.
- **Contamination:** families 5 and 6 are older than Flint's knowledge cutoff; their absolute pass rates are reported
  as such, and families 1–4 carry the fresh-task numbers.

## What this changes in the plan

- **The serving-path work (priority 3) gets its end-to-end yardstick:** v2's served gain on agent traffic is set by
  prefill and the fixed cost per request on v1, and by decode on v2. Q55/Q67 (GPU sampling for tool requests) is the
  lever that the agent check isolates, because neither backend sampling nor the screened head applies to tool
  requests today.
- **The file decision (priority 1) gains a factor:** on tool requests the GPTQ head of K2h-r16 saves bytes that the
  screened head cannot (until Q67). The release comparison should report agent traffic separately from chat.
- **Suggested registry rows (not added here):** the agent check as one experiment (calibration, pilot, full run), and
  the checkpoint-policy replay as a serving item under priority 3.

## Already covered

| Topic | Where |
|---|---|
| Task-check statistics, BFCL and LiveCodeBench lanes, non-inferiority margins | [task-level checks](task_level_quality_checks.md), Q41, [paired_tasks.py](../../../bench/paired_tasks.py) |
| The makers' engine, sampling, history and pi/OpenCode clients | [rec](rec_engine.md) |
| Time per correct answer, length gate, reasoning length by file | [answer time](answer_time.md), Q123 |
| Prompt processing, per-request fixed cost, cache replay of BFCL sessions (Q111), saved-prefix states | [prompt processing](prefill_pascal.md), Q108–Q111, Q116 |
| Host costs per served token, streaming re-parse, `-t 1` | [host in the decode loop](host_decode_loop.md), Q118–Q122 |
| Tool requests on the host sampling path | [tool-request sampling](tool_request_sampling.md), Q55, Q67, Q68 |
| Read-token and miss_param gaps of our files | [long-context quality](long_context_quality.md), [long-context mix](long_context_mix.md), Q106, Q126–Q130 |
| Several conversations at once (sub-agents, parallel users) | [multi-conversation serving](multi_conversation_serving.md), Q104; this check runs one agent at a time |
| The combined v2 build and its switches | [kernel note](pascal_mmvq_integration.md) §16, Q117 |

## Not verified

- **Every time and token figure in §3.2 and the suite table:** a cost model with assumed reasoning per turn; the
  calibration and the pilot replace them.
- **The eviction behaviour of §2.4:** read from llama-server's code at the fork's base, not observed. Neither is
  whether pi's re-rendered history matches Flint's generated tokens.
- **The pi prefix:** ~1.4–1.8k tokens is a character-based estimate, not a tokenizer count.
- **The screened head on tool requests:** off by its documented rule, not checked in a server log.
- **The 1.10× decode gain of v2 on tool requests:** from the components, not timed. Neither is v2's prompt speed on
  the rig.
- **`-ub 512` at 262,144 tokens on v2:** whether it fits the cards and the host with checkpoints.
- **Disconnects and headers:** whether llama-server stops generating when the client disconnects mid-stream (its HTTP
  layer checks for closed connections), and whether it sends response headers before a long prefill ends.
- **The quant machine:** throughput with pi in a container (~4 minutes per attempt per slot).
- **Family 5's environments:** that SWE-bench's environment specifications for the chosen instances build with `uv`
  on the laptop without Docker.
- **The noise of agent tasks:** σ² ≈ 0.09 and ω² ≈ 0.01 are taken from BFCL; the calibration run estimates them.

## Sources

Agentic benchmarks and harnesses (P):
- Terminal-Bench 2.0 paper, [arXiv 2601.11868](https://arxiv.org/abs/2601.11868); task metadata in
  [harbor-framework/terminal-bench-2](https://github.com/harbor-framework/terminal-bench-2) (Apache-2.0).
- Harbor, [repository](https://github.com/laude-institute/harbor) and its pi agent,
  [`src/harbor/agents/installed/pi.py`](https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/agents/installed/pi.py)
  (read 2026-10-24); [adapters](https://github.com/laude-institute/harbor/tree/main/adapters).
- [OpenThoughts-TBLite](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TBLite) (100 tasks, Apache-2.0,
  r = 0.911 with TB2).
- SWE-bench: [datasets guide](https://www.swebench.com/SWE-bench/guides/datasets/),
  [SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) (fields incl. `difficulty`;
  194 "<15 min fix"; repository counts via the Hugging Face datasets server), [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent),
  [SWE-rebench leaderboard](https://swe-rebench.com/).
- Aider: [polyglot benchmark](https://aider.chat/2024/12/21/polyglot.html), [leaderboard data](https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml),
  ["Details matter with open source models"](https://aider.chat/2024/11/21/quantization.html) and its
  [data](https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/quant.yml).
- Exercism [python](https://github.com/exercism/python/blob/main/LICENSE) and [javascript](https://github.com/exercism/javascript/blob/main/LICENSE)
  tracks (MIT).
- τ-bench [arXiv 2406.12045](https://arxiv.org/abs/2406.12045); τ²-bench [arXiv 2506.07982](https://arxiv.org/abs/2506.07982).
- Vijayvargiya et al., "Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering", ICLR 2026,
  [arXiv 2502.13069](https://arxiv.org/abs/2502.13069).
- SWE-bench Verified results by scaffold: [leaderboards.json](https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json).

Quantization on agentic work (P unless marked):
- Unsloth, [Dynamic GGUFs on Aider polyglot](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot).
- Jang et al., [Flat Score, Amplified Failures](https://arxiv.org/abs/2607.27275); Dong et al., ACBench,
  [arXiv 2505.19433](https://arxiv.org/abs/2505.19433); Red Hat's Gemma-4 NVFP4 card and Kurtic et al., as cited in
  [task-level checks](task_level_quality_checks.md).
- Community (S): [SWE-bench Verified with mini-SWE-agent on Qwen3.5/3.6-35B-A3B GGUFs](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/10).

Timing agents and evaluation noise (P):
- Yuan, Nayak, Kundu, Talati, "Agentic AI Workload Characteristics", IISWC 2026, [arXiv 2605.26297](https://arxiv.org/html/2605.26297v2).
- [TraceLab](https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/) (Claude Code and Codex traces).
- Manus, [Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus).
- Anthropic, [Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/engineering/infrastructure-noise)
  (2026-02-05) and [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
  (2026-01-09).
- [On Randomness in Agentic Evals](https://arxiv.org/abs/2602.07150); Miller, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640);
  Cost-of-Pass, [arXiv 2504.13359](https://arxiv.org/abs/2504.13359).

pi and the makers' client (P):
- pi 0.87.1 (`@earendil-works/pi-coding-agent`), its bundled docs (`cli.md`, `json.md`, `session-format.md`,
  `message-types.md`, `compaction.md`, `settings.md`, `extensions.md`, `containerization.md`, `llama-cpp.md`) and
  source: `dist/core/system-prompt.js`, `dist/core/tools/{read,bash,edit,write,truncate}.js`, pi-ai's
  `dist/api/openai-completions.js` (streaming, reasoning fields, usage mapping, replay) and `transform-messages.js`;
  [repository](https://github.com/earendil-works/pi).
- rec at `0dcf43c`: [`cli/clients/pi/flint-pi.ts`, `pi-transport.mjs`, `package.json`](https://github.com/recursive-oss/rec/tree/0dcf43c765195c44d72e3c59c6849a228ebe638b/cli/clients/pi)
  and `cli/pi.go` (pi 0.73.1 pinned; context window and max tokens set to the engine's context).

llama.cpp (P):
- [`tools/server/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (`timings`,
  `/metrics` fields, slot save and restore, `return_progress`).
- The fork's base `d77256999`: `tools/server/server-context.cpp` (`create_checkpoint`, eviction within
  `checkpoint_min_step`), `common/common.h:641` (`checkpoint_min_step = 8192`), `common/arg.cpp` (`-cms`),
  `tools/server/server-common.cpp:1442` (`parallel_tool_calls` defaults to the template's capability),
  `tools/server/server-task.cpp` (`timings` and `cached_tokens` on the final chunk), `tools/server/server-http.cpp`
  (closed-connection checks); issues [#19382](https://github.com/ggml-org/llama.cpp/issues/19382),
  [#31000](https://github.com/ggml-org/llama.cpp/issues/31000), [#31040](https://github.com/ggml-org/llama.cpp/pull/31040).

This repository:
- [quants/experiments.md](../../../quants/experiments.md): Q30, Q41, Q54, Q55, Q58, Q67, Q71, Q76, Q79, Q92, Q100,
  Q104, Q106, Q108–Q111, Q116–Q126.
- Notes: [rec](rec_engine.md), [task-level checks](task_level_quality_checks.md), [tool-request sampling](tool_request_sampling.md),
  [host in the decode loop](host_decode_loop.md), [answer time](answer_time.md), [long-context quality](long_context_quality.md),
  [multi-conversation serving](multi_conversation_serving.md), [prompt processing](prefill_pascal.md),
  [Flint speed-ups](flint_speedups.md), [kernel note](pascal_mmvq_integration.md) §16,
  [gptq_head](../../../quants/gptq_head/README.md) (screened head rules).
- [scripts/server_bench.sh](../../../scripts/server_bench.sh) (served chat timing, the RAM watchdog),
  [bench/bfcl_flint](../../../bench/bfcl_flint/README.md), [task_results.py](../../../bench/task_results.py).
