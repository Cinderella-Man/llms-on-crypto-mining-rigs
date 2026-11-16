# Several whole attempts at once: how the winner is picked without hidden tests (2026-11-17)

The attempt race ([registry](../../../quants/experiments.md) Q1498) gives one coding agent several rigs: each rig runs
the whole task with its own seed, and the first finished attempt is kept. Its estimate is 0.71 of one attempt's wall
time with two rigs ([the task-level review](v5_review_task_level_2026-11.md)), and its open question is that a
finished attempt is not a passed one. This note lists what is published on running several whole attempts of a coding
agent at once, on what "first finished" does to the pass rate, and on picking a winner when no hidden tests exist. It
ends with a rule for two to five rigs and with what the rig run should record. Races between samples of one reply are
in [tokens per solved task](tokens_per_solved_task.md), section 5, and are not repeated here.

Conventions:
- "est." marks arithmetic or inference of this note, with its basis beside it. Every other number is measured and
  quoted from the linked source, with its date, model and benchmark.
- No source uses this project's models or files. Every published number is therefore a claim to test here, not a
  result for the rig.
- "The agent check" is [bench/agent_check](../../../bench/agent_check/README.md), 12 bug-fix tasks under the pi coding
  agent. "The bracket" is the sister repository's run of the rig's own file on 20 real repository tasks
  ([elixir-agents-on-mining-rigs](https://github.com/Cinderella-Man/elixir-agents-on-mining-rigs),
  `docs/data/2026-11-10_bracket`).
- "Oracle" is the share of tasks where at least one of the N attempts passes the hidden tests: the ceiling for any way
  of picking.
- Nothing ran on any machine for this note. Sources were read on 2026-11-17.

## The rule I would run on two rigs

Both rigs run the whole task, each with its own seed, and when an attempt ends, the project's own tests run on its
result while the other rig keeps working. If the tests pass and the diff is not empty, that attempt is kept and the
other rig is stopped. If they fail, the attempt is dropped and the other rig's attempt is awaited and checked the same
way; if both fail, the one that fails fewer tests is handed back, marked as failed.

What the published evidence says this costs or gains:

- **Time: nothing published measures it.** No source I found times a race of whole coding-agent attempts. The est.
  0.71 of one attempt's wall time is this project's own. Published run times support the direction: failed runs
  take about twice as long as solved ones (544.2 s against 274.7 s in one study), so the attempt that ends first is
  more often a pass than the one that ends last.
- **Passes without the check: between p² and 1 − (1 − p)².** For a task that one attempt passes with probability p,
  the first finisher of two passes with est. p² + 2p(1 − p)q, where q is the chance that a passing attempt ends
  before a failing one. With q = 0.5 the race neither gains nor loses passes. No published data gives q for one task.
- **Passes with the check: the published gain is small with two attempts.** A trained critic that picks one of two
  finished attempts raised the pass rate from 57.8 % to 66.7 % on the 148 of 500 SWE-bench Verified tasks where the
  attempts disagreed, which is est. 2.6 points over all 500. Dropping patches that break the visible tests and
  ranking the rest took Claude 3.7 Sonnet from 63.7 % to 70.3 % with an unstated number of attempts.
- **The check itself is cheap, and it is not proof.** Running a project's tests took about 50 s an instance on
  SWE-bench in one measurement, and it runs while the other rig still works. Of the patches that pass a benchmark's
  chosen tests, 7.8 % fail the developers' full suite.
- **Rig time doubles.** Both rigs work until the kept attempt is checked: est. 1.42 attempts' worth of rig time for
  one kept attempt (2 × 0.71). The race buys wall time for one agent with rig time other agents could have used.

## 1. What several attempts gain, with N

Every result below runs N whole attempts and waits for all of them before picking. All are measured on the source's
own model.

| Source | Model and benchmark | One attempt | N | Oracle | Picked without hidden tests | How it is picked |
|---|---|---:|---:|---:|---:|---|
| [Large Language Monkeys](https://arxiv.org/abs/2407.21787), 2024-07 | DeepSeek-Coder-V2 under Moatless Tools, SWE-bench Lite | 15.9 % | 5; 250 | 29.62 %; 56 % | — | by the hidden tests only |
| [DEI](https://arxiv.org/abs/2408.07060), 2024-08 | ten runs each of Agentless, Aider and Moatless, SWE-bench Lite | 20.4 %, 21.7 %, 15.9 % (the runs' average) | 10 | 34.7 %, 38.0 %, 35.3 % | 26.0 %, 24.7 %, 26.3 % | a reviewing model (GPT-4o) scores each patch, ten votes a patch |
| [SWE-Gym](https://arxiv.org/abs/2412.21139), 2024-12 | fine-tuned Qwen2.5-Coder-32B under OpenHands, SWE-bench Verified | 20.6 % | 8; 16 | 37.8 %; 42.8 % | 29.8 %; 32.0 % | a trained model reads each attempt's whole trajectory |
| [CodeMonkeys](https://arxiv.org/abs/2501.14723), 2025-01 | Claude 3.5 Sonnet, SWE-bench Verified | 45.8 % (a random one of the ten) | 10 | 69.8 % | 57.4 % | tests written by the model vote, then the model picks among the top three |
| [Anthropic](https://www.anthropic.com/news/claude-3-7-sonnet), 2025-02 | Claude 3.7 Sonnet, 489 tasks of SWE-bench Verified | 63.7 % | not stated | — | 70.3 % | patches that break the visible tests are dropped, a scoring model ranks the rest |
| [R2E-Gym](https://arxiv.org/abs/2504.07164), 2025-04 | fine-tuned Qwen2.5-Coder-32B, SWE-bench Verified | 34.4 % | 26 | 64.4 % | 51.0 % | written tests and a trained model together; either alone stops near 43.7 % and 42.8 % |
| [OpenHands](https://openhands.dev/blog/sota-on-swe-bench-verified-with-inference-time-scaling-and-critic-model), 2025-04 | Claude 3.7 Sonnet at temperature 1.0, SWE-bench Verified | 60.6 % | 5 | — | 66.4 % | patches failing the tests are dropped, a trained critic picks |
| [Trae Agent](https://arxiv.org/abs/2507.23370), 2025-07 | Claude 3.7 Sonnet, SWE-bench Verified | 61.33 % (the runs' average) | 3 | 70.00 % | 66.40 % | duplicates and patches that break the tests are dropped, a selector agent votes |
| [Agentic Rubrics](https://arxiv.org/abs/2601.04171), 2026-01 | Qwen3-32B and Qwen3-Coder-30B-A3B, SWE-bench Verified | 22.6 % and 39.6 % (a random one) | 16 | 51.4 % and 65.6 % | 40.6 % and 54.2 % | a stronger model writes a checklist for the task and each patch is graded against it |
| [OpenHands critic](https://arxiv.org/abs/2603.03800), 2026-03 | Claude Sonnet 4.5 and Opus 4.5, the 148 tasks of SWE-bench Verified where runs disagree | 57.8 % (a random one) | 2; 4; 8 | 100 % by construction | 66.7 %; 72.6 %; 73.8 % | a trained 4B critic reads each trajectory |

What the table shows:

- **Picking recovers about half of what the attempts hold.** CodeMonkeys picks 57.4 % between a random 45.8 % and
  an oracle of 69.8 %. SWE-Gym picks 29.8 % at 8 attempts between 20.6 % and 37.8 %. R2E-Gym picks 51.0 % at 26
  between 34.4 % and 64.4 %.
- **The gain grows with the logarithm of N.** SWE-Gym's picked rate is 20.6 %, 29.8 % and 32.0 % at 1, 8 and 16
  attempts, and its authors and OpenHands both describe the curve as log-linear. Most of the gain comes from the
  first few attempts.
- **Attempts of one model fail together.** If SWE-Gym's eight attempts were independent, its oracle at 8 would be
  est. 84.2 % (1 − 0.794⁸). It is 37.8 %. In the OpenHands critic's data only 148 of 500 tasks have both a passing
  and a failing run among eight. More attempts therefore help on a minority of tasks: the ones the model sometimes
  solves.
- **Two attempts gain little in passes.** The one result at N = 2 is the critic's 57.8 % to 66.7 % on the 148
  mixed tasks, est. 2.6 points over all 500 (8.9 points × 148 / 500, the other tasks unchanged).
- **None reports wall time.** Cost is given in dollars where it is given: CodeMonkeys spent $2,291.90 on 500 tasks,
  of which $1,366.02 on the ten edits, $439.99 on writing tests and $131.95 on picking. One author group
  ([Augment](https://www.augmentcode.com/blog/1-open-source-agent-on-swe-bench-verified-by-combining-claude-3-7-and-o1),
  2025-03) puts the gain of picking among several runs at 3 to 8 % and calls it too expensive for real use.
- **Stopping at the first accepted attempt is published once.** The OpenHands critic accepts the first attempt
  whose score passes 0.5 and otherwise tries the next, up to eight. It uses 1.35 attempts on average and scores
  17.7 points above a random pick on the mixed tasks. This is the shape of the rule above, with a critic in place of
  the tests.

## 2. "First finished wins"

### How long failed and passed runs take

| Source | Setup | Passed runs | Failed runs |
|---|---|---|---|
| [SWE-agent](https://arxiv.org/abs/2405.15793), 2024-05 | GPT-4 Turbo, SWE-bench | median 12 steps and $1.21 | mean 21 steps and $2.52; 93.0 % of the passed runs ended before the cost limit, against 69.0 % of all runs |
| [Majgaonkar et al.](https://arxiv.org/abs/2511.00197), 2025-10 | OpenHands, SWE-agent and Prometheus on SWE-bench Verified | 54.05, 151.95 and 146.59 steps on average | 98.62, 180.10 and 220.92 steps; on SWE-bench Lite 61.01, 158.36 and 87.16 against 79.90, 178.31 and 136.47 |
| [AgentStop](https://arxiv.org/abs/2605.15206), 2026-05 | Qwen3-Coder-30B-A3B under mini-swe-agent, SWE-bench Verified, 100-step limit | 94 runs: 274.7 s, 4.5k output tokens | 406 runs: 544.2 s, 8.0k output tokens; no run passed after step 60 |
| [Agentic AI Workload Characterization](https://arxiv.org/abs/2605.26297), 2026-05 | Qwen3.6-27B under Claude Code, Terminal-Bench 2.0 | 44.4K and 45.9K tokens of context (thinking, instant) | 79.2K and 76.6K; one reverse case: Gemma4-31B instant on SWE-bench Pro, 82.2K passed against 72.3K failed |
| This project, [recorded attempts](../../data/2026-11-11_fn58_stop_and_retry/README.md) | Flash-Next on the agent check, 357 attempts | 346 passed | 11 failed; the time limit stopped 14 attempts, passes among them |

- **Failed runs are longer on average in every source but one case.** The ratios are est. 1.98 in seconds
  (AgentStop), and 1.82, 1.19 and 1.51 in steps (Majgaonkar et al., SWE-bench Verified). Failed attempts also
  spread wider: the standard deviation of Prometheus's steps is 100.79 for failures against 34.62 for passes on
  SWE-bench Lite. [SWE-Effi](https://arxiv.org/abs/2509.09853) gives the same direction in tokens
  ([tokens per solved task](tokens_per_solved_task.md), section 4).
- **Failures come in two kinds.** Majgaonkar et al. describe one group that fails about as fast as the passes end,
  and a second that runs on to the limit. AgentStop's failed runs often go on to the step limit. An attempt that
  ends at a limit ends last by construction, so it loses a race to any attempt that ends by itself.
- **The risk is the attempt that ends early and wrong.** An agent's own word that it is done carries little: asked
  after finishing whether their patch is right, three models predicted success for 73 %, 77 % and 61 % of their
  patches while 35 %, 22 % and 27 % passed ([Agentic Uncertainty](https://arxiv.org/abs/2602.06948), 2026-02,
  SWE-bench Pro, 100 tasks).

### What keeping the first finisher does to the pass rate

- **No published measurement exists for whole coding-agent attempts.** I found none that picks the attempt that ends
  first, or the shortest, and reports its pass rate. On single replies of mathematics and science questions the
  first finisher scored the same or better ([First Finish Search](https://arxiv.org/abs/2505.18149)), and on this
  project's tool-call test the first to end its reasoning scored +0.51 points (Q93).
- **The arithmetic, est.** Take one task, two independent attempts that each pass with probability p, and q the
  chance that a passing attempt ends before a failing one. The first finisher passes with p² + 2p(1 − p)q. It
  equals p at q = 0.5, the oracle 1 − (1 − p)² at q = 1 and p² at q = 0.
- **An illustration from published means, est.** If run times were exponential with AgentStop's means, q would be
  0.665 (544.2 / (544.2 + 274.7)), and at its p of 0.188 the first finisher would pass 23.8 % against 18.8 % for
  one attempt and 34.1 % for the oracle. This overstates it: the means pool easy and hard tasks, and the rule needs
  q within one task.
- **On the agent check the effect is bounded at est. 3.0 points.** With 346 of 357 attempts passed, p is est.
  0.969, so the first finisher passes between est. 93.9 % and 99.9 %. One task of the 12 is 8.3 points, so a run of
  the check cannot show this effect.
- **On real tasks it can go either way.** The bracket's first seed passed 14 of 20, so the first finisher passes
  between est. 49 % and 91 %. Attempts there that end at the generation budget end last, which pushes q above 0.5.
  The share of failures that end by themselves is what decides it, and it is a count over recorded attempts (Q1500,
  Q1537).

## 3. Picking without hidden tests

### The project's own tests

- **What it does:** a patch that breaks a test which passed before is dropped.
- **Published gain:** [Agentless](https://arxiv.org/abs/2407.01489) (GPT-4o, SWE-bench Lite, 2024-07) solves 77 of
  300 by agreement alone and 81 with this filter, at $0.01 a task. In Trae Agent's table the filter with duplicate
  removal adds 1.20 and 1.54 points to two judge-based selectors at N = 3 (63.13 % to 64.33 %, and 62.33 % to
  63.87 %). Anthropic's and OpenHands's results above both start with it.
- **Cost:** the tests' run time and no model tokens. About 50 s an instance on a 64-core server for SWE-bench's
  tests ([R4P](https://arxiv.org/abs/2510.22775), 2025-10). The filter with tests takes about 386 s a bug in
  [PatchFusion](https://arxiv.org/abs/2607.01597)'s comparison (2026-07).
- **Limits:** it only catches what the existing tests cover. Of patches that pass SWE-bench Verified's chosen tests,
  7.8 % fail when all the developers' tests run, 29.6 % behave differently from the developers' fix, and est. 11.0 %
  are wrong ([Wang et al.](https://arxiv.org/abs/2503.15223), 2025-03). Tests can also fail by chance: 11.3 % of
  SWE-bench Lite's tasks have suites that give different results on the same code (Large Language Monkeys).

### Tests the agent writes

- **What it does:** a test that fails before the fix and should pass after it. Attempts are ranked by the tests they
  pass.
- **Published gain:** Agentless goes from 81 to 96 of 300 with such tests, at $0.25 a task. Model-written tests
  alone take R2E-Gym from 34.4 % to about 43.7 % at 26 attempts and Agentic Rubrics' two models from 22.6 % and
  39.6 % to 33.6 % and 49.0 % at 16. [CodeT](https://arxiv.org/abs/2207.10397) (2022-07) is the origin: 65.8 % on
  HumanEval, 18.8 points above the same model's first sample.
- **Cost:** a second agent run per test. CodeMonkeys spent $439.99 on tests beside $1,366.02 on edits. Agentic
  Rubrics prices it at $0.515 a task with a stronger model writing the tests.
- **Limits:** most written tests say nothing. In R2E-Gym fewer than 20 % of them tell a right patch from a wrong one
  for most tasks, and for a few tasks up to 10 % pass wrong patches and fail right ones. In Agentless 213 of 300
  tasks get a test that shows the bug, and 94 of those tests pass on the true fix. Used as a gate on SWE-agent's
  fixes, written tests raise precision to 47.8 % and keep about 20 % of the correct fixes
  ([SWT-bench](https://arxiv.org/abs/2406.12952), GPT-4, 2024-06).

### A model's review of the diffs

- **What it does:** a model reads the task and the patch, or the whole trajectory, and scores or ranks.
- **Published accuracy:** between 61.9 % and 72.7 % for judging one patch right or wrong (R4P's table: Qwen2.5-Coder-32B
  61.9 %, Claude 3.7 Sonnet 68.1 %, o3 71.5 %, Gemini 2.5 Pro 72.7 %, its own trained model 72.2 %). R2E-Gym's
  trained model reaches 71.82 % with the trajectory and 68.01 % with the patch alone.
- **Published gain:** in the table of section 1, the rows of DEI, SWE-Gym and the two critics. A plain judge prompt
  is weaker: on six agents' patches for the same bug, an LLM judge picks 396 of 500 against 388.0 for a random pick
  and an oracle of 443 (PatchFusion). Trae Agent's authors report that judge prompts get worse as N grows.
- **The same model reviewing its own patch is near chance.** Area under the curve 0.58, 0.51 and 0.55 for three
  models on SWE-bench Pro, and 0.55, 0.57 and 0.64 when told to look for bugs (Agentic Uncertainty).
- **Cost:** one model call or one short agent run. About 1 s a patch for R4P on two GPUs, 1.1 s a trajectory for the
  4B critic, 1.26 s and about 4.0k input tokens for PatchFusion's judge call, and $0.23 or $0.52 a task for a review
  run of 12.7 or 23.4 steps (Agentic Uncertainty).
- **In products:** Cursor 2.2 (2025-12-10) reviews parallel runs and suggests one, only after all of them have
  finished, with no published accuracy ([changelog](https://cursor.com/changelog/2-2)).

### Agreement between attempts

- **What it does:** the patch that most attempts agree on is kept, after formatting differences are removed.
- **Published gain:** Agentless's 77 of 300 by agreement alone, among 40 patches. At 16 attempts agreement gives
  33.2 % and 47.6 % against a random 22.6 % and 39.6 % (Agentic Rubrics). On six different agents' patches,
  agreement on whole diffs picks 384 of 500, below the random 388.0, and agreement on the single edits inside the
  diffs picks 426 of an oracle 443 (PatchFusion).
- **Cost:** none in tokens, milliseconds in time (3.28 ms a bug for PatchFusion).
- **Limits:** it needs enough attempts to agree. Two attempts can agree or differ and no more. It also stops
  improving: on mathematics sets, voting stopped gaining near 100 samples while the oracle kept rising (Large
  Language Monkeys).
- **A related signal:** the mean confidence of the attempt's own tokens. One 8B model reaches 45.2 % on SWE-bench
  Verified with eight candidates picked this way ([Klear-AgentForge](https://arxiv.org/abs/2511.05951), 2025-11); its
  text gives no single-attempt figure beside it.

### Side by side

| Way of picking | Works with two attempts | Can run before the others finish | Time and tokens | Main limit |
|---|---|---|---|---|
| The project's own tests | yes | yes | the tests' run time; no tokens | only what the tests cover |
| Tests the agent writes | yes | yes | a second agent run | most written tests do not separate right from wrong |
| A model's review | yes | yes | one model call or a short run | about 62 to 73 % accurate; near chance on its own patch |
| Agreement | no: it needs three or more | no: it waits for several | none | the attempts must agree |

## 4. One agent on two to five rigs

- **Only a rule that accepts the first good attempt keeps the race's time.** Every published selector except one
  waits for all N attempts, and then the time is the slowest attempt's, not the first's. The rule above checks
  each attempt as it ends and stops the rest on the first that passes.
- **The check costs almost no wall time.** It runs on the machine that runs the agent's tools while the other rigs
  still generate. If it passes, the task is over at the first finisher's time plus the check. If it fails, the other
  attempts were never interrupted.
- **Three to five rigs run the same rule.** The first attempt that ends and passes its tests is kept. The time
  falls to est. 0.58, 0.52 and 0.49 of one attempt's (the task-level review) and the rig time rises to est. 1.74,
  2.08 and 2.45 attempts' worth (N times each share). The pass rate gains little beyond the second rig: attempts of
  one model fail together (section 1).
- **With three or more rigs, agreement becomes possible** for a project without tests: wait for two attempts that
  change the same lines (Q1541). With two rigs and no tests, the first finisher with a non-empty diff is kept,
  unchecked: that is Q1498 as it stands.
- **What the check means on the agent check.** Its bug-fix tasks give the agent the full test file, and success is
  that file passing unchanged. The rule's check is therefore the score itself there, and the checked race cannot keep
  a failed attempt. A real bug report often has no failing test yet, and then the project's tests pass before and
  after any fix: the check only catches breakage. A test that shows the bug, written first, would close that gap
  (Q1539).
- **Claims to test on my models,** in the order the evidence is thinnest: that failed attempts of Flash-Next end
  later than passed ones on one task (q above 0.5); that an attempt which ends by itself has passed its tests;
  that a second attempt passes where the first failed often enough to matter.

### What the rig run should record (Q1498)

The run is two runs of the same 12 tasks, one on each rig, each attempt to its end. Nothing is stopped, so both
rules are priced from one run.

1. **Per attempt:** wall seconds, how it ended (by itself, or at the turn limit, the time limit or a token mark),
   pass, whether the test file is unchanged, whether the diff is empty, generated tokens, turns. The attempt's
   `result.json` and `diff.patch` hold all of these already.
2. **The check's seconds:** how long the task's tests take on the finished workspace. This is new, and it is the
   only cost the rule adds.
3. **Per task, from the pair:** each rig alone, the first finisher (its rig, seconds and pass), the first finisher
   whose tests pass (its seconds, check included), and whether the two diffs change the same lines.
4. **Per run:** the four sums of seconds (rig 1 alone, rig 2 alone, first finisher, checked first finisher), the
   passes of each, and the rig seconds both rigs spent until the kept attempt ended.
5. **Counts of the cases the check exists for:** attempts that ended by themselves and failed, attempts with an
   empty diff, attempts that changed the test file.
6. **No verdict on passes from this run.** The effect is bounded at est. 3.0 points and one task is 8.3. The run
   prices time. Passes are priced on recorded attempts: q for each task and the failures split by how they ended
   (Q1500), then on tasks that fail (Q1537).

## 5. New registry rows

| Row | What |
|---|---|
| Q1536 | The attempt race with a check before the win: the pairs' table for Q1498's run, then a runner that stops the other rig on a checked win |
| Q1537 | The first finisher's pass share on tasks that fail: the bracket's seeds, paired per case |
| Q1538 | The second attempt started late: the second rig joins only once the first has run past a mark |
| Q1539 | Parked: a test that shows the bug, written first, as part of the check |
| Q1540 | Parked: a model's review of the first finisher's diff |
| Q1541 | Parked: agreement between attempts as the check, with three or more rigs |
| Q1542 | Parked: an attempt's token confidence as the pick |

## Not verified

- Whether q is above 0.5 within one task on any model. Every published length comparison pools tasks.
- The wall time of a race of whole attempts. No source times one.
- The numbers read from figures by their sources' text: R2E-Gym's 43.7 % and 42.8 %.
- [An ESEM 2026 study](https://drops.dagstuhl.de/storage/00lipics/lipics-vol394-esem2026/LIPIcs.ESEM.2026.21/LIPIcs.ESEM.2026.21.pdf)
  of LLM judges for patch correctness: its text could not be read, so it is not used.
- How long the agent check's tests take on a finished workspace. It is not in the recorded results.
- Whether pi's attempts on the rig differ enough between seeds: the agent sends no seed today, so two runs already
  differ, and the est. 0.71 rests on that.

## Sources

Project data and notes: [the task-level review](v5_review_task_level_2026-11.md),
[tokens per solved task](tokens_per_solved_task.md), the
[recorded attempts](../../data/2026-11-11_fn58_stop_and_retry/README.md),
[bench/agent_check](../../../bench/agent_check/README.md), the [registry](../../../quants/experiments.md) (Q93, Q1195,
Q1498, Q1500, Q1528).

Outside, read 2026-11-17:
- several attempts and how they are picked: [Large Language Monkeys](https://arxiv.org/abs/2407.21787),
  [Agentless](https://arxiv.org/abs/2407.01489), [DEI](https://arxiv.org/abs/2408.07060),
  [SWE-Gym](https://arxiv.org/abs/2412.21139), [CodeMonkeys](https://arxiv.org/abs/2501.14723),
  [Anthropic's Claude 3.7 Sonnet post](https://www.anthropic.com/news/claude-3-7-sonnet), appendix,
  [R2E-Gym](https://arxiv.org/abs/2504.07164),
  [OpenHands's critic post](https://openhands.dev/blog/sota-on-swe-bench-verified-with-inference-time-scaling-and-critic-model)
  and [model card](https://huggingface.co/OpenHands/openhands-critic-32b-exp-20250417),
  [Augment's post](https://www.augmentcode.com/blog/1-open-source-agent-on-swe-bench-verified-by-combining-claude-3-7-and-o1),
  [Trae Agent](https://arxiv.org/abs/2507.23370), [R4P](https://arxiv.org/abs/2510.22775),
  [Klear-AgentForge](https://arxiv.org/abs/2511.05951), [Agentic Rubrics](https://arxiv.org/abs/2601.04171),
  [the OpenHands critic paper](https://arxiv.org/abs/2603.03800), [PatchFusion](https://arxiv.org/abs/2607.01597),
  [Cursor 2.2's changelog](https://cursor.com/changelog/2-2);
- tests as a check: [CodeT](https://arxiv.org/abs/2207.10397), [SWT-bench](https://arxiv.org/abs/2406.12952),
  [Are "Solved Issues" in SWE-bench Really Solved Correctly?](https://arxiv.org/abs/2503.15223);
- how long runs take and what an agent knows of its own result: [SWE-agent](https://arxiv.org/abs/2405.15793),
  [Understanding Code Agent Behaviour](https://arxiv.org/abs/2511.00197),
  [AgentStop](https://arxiv.org/abs/2605.15206),
  [Agentic AI Workload Characterization](https://arxiv.org/abs/2605.26297),
  [Agentic Uncertainty Reveals Agentic Overconfidence](https://arxiv.org/abs/2602.06948),
  [SWE-Effi](https://arxiv.org/abs/2509.09853), [First Finish Search](https://arxiv.org/abs/2505.18149).
