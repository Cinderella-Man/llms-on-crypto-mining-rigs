# Tokens per solved task: what a thinking model can spend less of without retraining (2026-11-10)

A coding agent on the rig spends most of its time decoding reasoning, and more than half of the rig time of a real
task goes to attempts that end at the generation budget. This note lists what is published on spending fewer tokens
per solved agentic coding task without changing the weights, with each source's effect on tokens and on success, and
says how each lever runs on Miner Llama's `llama-server` with the pi coding agent.

Conventions:
- "est." marks arithmetic or inference of this note, with its basis beside it. Every other number is measured and
  quoted from the linked source, one run unless repeats are listed.
- "The release line" is [release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env) on build
  `1551f5fe1`: UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next, nine P106-090 cards, the MTP draft.
- "The bench" is the [Elixir SWE Bench](https://github.com/Cinderella-Man/elixir-swe-bench): 20 repository-level cases
  under pi, a 262,144-token window, no compaction, effort `xhigh`, 8,192 tokens a reply and 65,536 generated tokens an
  attempt ([second review](flash_next_review2_2026-11.md), section 2).
- "The agent check" is [bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks under pi.
- Server flags and request fields were read in the release tree `1551f5fe1` of
  [Miner Llama](https://github.com/Cinderella-Man/miner-llama); pi's settings in the documentation and type
  declarations of pi 0.87.1, the version the agent check pins. Nothing ran on any machine for this note.

## The short answer

- **Three settings need no code:** a reasoning budget per reply with a stop message, the effort level, and the mark
  at which an attempt is stopped. The first has never been tried here, and its published effect depends on one detail:
  the message.
- **A forced end of reasoning needs a stop message.** In llama.cpp's own test of its budget sampler (Qwen3.5 9B Q8_0,
  HumanEval), budgets of 1,000 and 400 tokens scored about 89 % with a message written before the closing tag and
  about 79 % without it, against about 93 % with unlimited reasoning and about 88 % with none. A forced end without
  the message is worse than not reasoning at all.
- **A reply limit without a reasoning budget wastes the whole reply.** A reply that reaches `max_tokens` inside its
  reasoning returns no tool call. A reasoning budget set below the reply limit turns that reply into a forced action.
- **`preserve_thinking` off drops nothing inside one attempt.** The model's chat template keeps the reasoning of every
  assistant message that follows the last user message, whatever the switch says. An agent attempt is one user message
  followed by replies and tool results, so dropping earlier reasoning needs the harness to remove it.
- **Published context policies keep the reasoning and drop old tool results.** On SWE-bench Verified, masking tool
  results older than ten turns cut the cost per case by 50.9, 52.7, 56.1 and 57.1 % in four of five model settings,
  with the solve rate up in three of the five (+1.4, +1.6 and +2.8 points) and down in two (−2.0 and −4.0). Summaries
  written by the model lengthened the runs by 15 %.
- **Failed attempts are the cost, and a plain length rule already catches part of them.** On SWE-bench Verified a
  stop rule that reads only the step count saved 11.4 % of the tokens at 5 % false stops (a trained monitor 20.4 %),
  and a fresh attempt after the stop kept the resolve rate (66.6 → 67.2 and 66.8).
- **More effort is not monotone.** In the one same-model sweep on hard SWE-bench Verified cases, the highest effort
  resolved fewer cases than the level below it on both models (17 against 19, and 16 against 18, of 25), at 1.29 and
  1.92 times the cost per resolved case, and timeouts and empty patches appeared only at the two higher levels.
- **A first-finish race is published only on mathematics and science questions.** There the first of four samples
  generated 0.74, 0.68, 0.79 and 0.76 of one sample's tokens in sequence on four models, with equal or better
  accuracy on most sets. On this project's replies the first of four generates 0.436: the tail here is heavier, so
  the race is worth more, and it pays across rigs (registry Q1195).

## 1. What the tokens are spent on

Measured in this project:

| Fact | Value | Source |
|---|---|---|
| Reasoning as a share of generated tokens, agent check, release arms | 79.42 % (125,356 of 157,837 tokens in 412 replies) | [reasoning census](../../data/2026-11-09_reasoning_census/README.md) |
| The same in replies that follow a tool result | 86.38 %; these replies hold 86.34 % of all reasoning tokens | the same |
| The same in the tasks' second replies, one load of 60 replies | 95.52 % (63,812 reasoning tokens against 2,990 for the answer) | [reply lengths](../../data/2026-11-10_prob_verify_reply_length/summary.md), load `pv0-a` |
| The answer after the reasoning in those replies | mean 49.8 tokens, median 45.5 | the same |
| Share of generated tokens in the longest 10 % of replies | 63.20 % (42 of 412 replies; 38 of the 42 sit in attempts that passed) | reasoning census; registry Q266 |
| One prompt, five seeds, four loads: median reply | 444.0, 348.0, 378.5 and 338.0 tokens | reply lengths |
| The same: mean reply | 1,113.4, 1,519.4, 1,594.5 and 1,209.4 tokens | the same |
| First finisher of 2, 3, 4 and 5 samples of one request | 0.621, 0.498, 0.436 and 0.399 of one sample's tokens | [race price](../../data/2026-11-10_prob_verify_reply_length/race_price.md) |
| A cap on a reply at 2,000, 4,000 and 8,000 tokens | keeps 0.563, 0.744 and 0.897 of all tokens; touches 40, 19 and 7 of 240 replies | the same |
| Replies that loop word for word | 0 of 2,745 in the release and Flash-Next arms | [loop census](../../data/2026-11-09_loop_census/README.md) |
| Bench attempts that end at the 65,536-token budget | 7 of 20 (ISTA GSQ-RCO-Coder IQ1_M) and 5 of 20 (Gyro-L TQ3_0), est. 56 % of rig time in the first mix | [second review](flash_next_review2_2026-11.md), section 2 |
| First to finish its reasoning among k attempts, another model, BFCL | no loss: +0.51 points at k = 2 with 0.72 of the mean reasoning | registry Q93 |

Three things follow for the levers below.
- **The tokens are reasoning after tool results, in a few long replies.** A lever that shortens the typical reply
  saves little. A lever that acts on the long tail (a budget, a race, an early exit) acts on 63.20 % of the tokens.
- **Length is mostly a property of the prompt, with a wide spread on top.** Three of the 12 prompts have median
  replies of 4,044.5, 2,421.0 and 1,792.5 tokens; the other nine have medians of 38.0 to 1,055.5. Within one prompt
  the shortest and the longest of 20 samples are 1,073 and 13,639 tokens, 1,244 and 19,994, 770 and 7,452.
- **After the reasoning the reply is a tool call of about 50 tokens.** The reasoning's end and the reply's end nearly
  coincide, and a probe that asks "which tool call would follow now" costs about 50 tokens.

## 2. The levers, ranked

Ranked by the published effect, weighted by how directly the evidence carries over to a 2-bit file in an agent loop.
"Check" is the 12-task agent check, two runs an arm. "Bench arm" is 20 cases with three seeds on the Elixir SWE
Bench, which the sister repository runs (section 9).

| # | Lever | Published effect on tokens | Published effect on success | Risk | How to measure it here | Needs | Row |
|---|---|---|---|---|---|---|---|
| 1 | A reasoning budget per reply, below the reply limit, with a stop message | here: a cap at 4,000 keeps 0.744 of second-reply tokens, at 8,000 0.897, before what is written after the forced end | HumanEval, Qwen3.5 9B: about 93 % unlimited, about 89 % at 1,000 and 400 tokens with the message, about 79 % without. SWE-bench Verified subset, Claude 4 Sonnet: 67.3, 68.4 and 68.7 % at 8k, 16k and 32k | 38 of the 42 longest replies here sit in passing attempts; a code model without budget training stays under 10 % on LiveCodeBench below a 4K total budget | the forced-end curve offline (Q1197); check arms at 4,000 and 8,000 with the message and 4,000 without; then a bench arm | server flags or a pi setting; no code | Q1198, Q266, Q1147 |
| 2 | An attempt stopped at a generated-token mark, then a fresh attempt | SWE-bench Verified: 11.4 % saved by a step-count rule and 20.4 % by a trained monitor at 5 % false stops (Qwen3.6-27B); 26.0 % fewer steps and 28.7 % fewer output tokens with 34.8 % of runs halted | a fresh attempt after the stop: 66.6 → 67.2 and 66.8; with the stopped run's diff offered: 69.8 and 71.8 | a pass that needed more than the mark is lost; the fresh attempt costs +18.2 % tokens at 25 % false stops | the mark's curve from the bench's saved attempts; a bench arm; on the check the 12-minute cap as a token mark (Q1200) | a harness setting | Q1200; the sister repository |
| 3 | A first-finish race across rigs, the second sample started late | first of 4: 0.74, 0.68, 0.79 and 0.76 of one sample's tokens in sequence on four models, at 2.9, 2.7, 3.2 and 3.0 times the total tokens | AIME and GPQA: equal or better on most sets (DeepSeek-R1, AIME24: 75.0 → 86.7) | no published test on code or tool use; the losing rig reads the winner's reply | Q1195's order: two servers off the rig, then two rigs; the late start priced offline first (Q1199) | a proxy | Q1195, Q1199, Q93 |
| 4 | Effort `medium` instead of `xhigh` | Kimi K3, mini-swe-agent: $0.117 a run at low against $0.561 at max; Claude Code, xhigh against low: 12 to 16 thousand more thinking tokens and 23 to 30 more turns | 25 hard cases: 12, 19 and 17 resolved at low, high and xhigh (Sonnet 5); 17, 18 and 16 (Opus 5). One case at low: 47 % against 99 % | the model card and the GSQ-RCO maintainer warn that lower effort costs more on these files | the bench arm E16 | a server flag; no code | Q1147 |
| 5 | An early exit when the next tool call no longer changes | 11 to 29 % at equal accuracy on mathematics (Dynasor); on code, DEER leaves 8.7 % to 89.2 % of the tokens | DEER on code: between −1.2 and +3.2 points over 21 model and benchmark pairs | published for final answers, not for tool calls; a probe costs about 50 tokens and a copy of the state | the forced-end curve offline (Q1197) is its ceiling | the server's control endpoint, a proxy, and engine code for the probe | Q1204, Q268 |
| 6 | Old tool results masked, earlier reasoning removed, both in blocks | masking results older than 10 turns: cost −52.7 % (Qwen3-Coder 480B), −56.1 % (Gemini 2.5 Flash), −9.8 % (Qwen3-32B thinking); trajectory reduction: input tokens −39.9, −40.9, −40.4 and −59.7 % | masking: 53.4 → 54.8 %, 32.8 → 35.6 %, 23.0 → 24.6 %, but 40.4 → 36.4 % (Gemini 2.5 Flash thinking); keeping reasoning between tool calls: about +3 % on SWE-bench | on the rig the gain is depth, not tokens, and every edit of history is re-read; the makers ask to keep the reasoning | replay arms for the re-read; a bench arm for the tasks | a pi extension | Q1201, Q1202, Q1166 |
| 7 | A line that asks for bounded effort; the failing test's output in the task | wording that asks for several approaches: 2.4 to 7.4 times the reasoning tokens; bounded-efficiency wording: no cost penalty; a bare user story: +29.7 % cost | comparable success in both studies | a changed prompt makes earlier series incomparable | one check arm | a harness setting | Q267 |
| 8 | A trained monitor that predicts a failing attempt | see lever 2: 20.4 % against the step count's 11.4 % | 30.5 % of failures caught at 5 % false stops | needs labelled attempts of this model and harness | on the bench's saved attempts | training | the sister repository |

Parked, with the reason (section 8): soft warnings before the budget, a rising bias on the end-of-reasoning token,
a presence penalty, thinking turned off for some turns, and pi's summary compaction as a token lever.

## 3. A reasoning budget per reply

### What is published

- **llama.cpp's budget sampler** ([PR #20297](https://github.com/ggml-org/llama.cpp/pull/20297), merged 2026-03-11)
  counts the tokens inside the reasoning block and, at the budget, forces a configured message and the closing tag.
  Its author's test on Qwen3.5 9B Q8_0 with HumanEval: about 93 % with unlimited reasoning, about 88 % with none,
  about 89 % at budgets of 1,000 and 400 tokens with the message, about 79 % without the message.
- **The Qwen makers describe the same mechanism for this family.** The
  [Qwen3 technical report](https://arxiv.org/abs/2505.09388) halts the thinking at a user-set length and inserts a
  first-person instruction that says the time is limited and the answer must now be given from the thinking so far.
  It reports smooth gains with the budget on four benchmarks, coding among them; the values are in a figure.
- **A per-turn budget on an agent, a closed model:** the [Confucius Code Agent](https://arxiv.org/abs/2512.10398)
  ran Claude 4 Sonnet at thinking budgets of 8k, 16k and 32k on a SWE-bench Verified subset: 67.3, 68.4 and 68.7 %.
  The budget there does not control the hidden reasoning exactly.
- **Tight budgets break code models that were not trained for them.**
  [Elastic Reasoning](https://arxiv.org/abs/2505.05315) forces the closing tag at a thinking budget and gives the
  answer its own budget. The untrained DeepCoder-14B stays under 10 % on LiveCodeBench when the total budget is under
  4K tokens. Its trained model scores 37.3, 41.6, 44.1 and 47.0 at thinking budgets of 1K to 4K against 58.4
  unconstrained.
- **No reasoning at all is the wrong end for code.** In
  [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858) an empty reasoning block
  uses 2.0 to 5.1 times fewer tokens and wins at low budgets on mathematics, but LiveCodeBench is the exception:
  thinking wins there even at low budgets.
- **On these files:** in a 30-task check with a 2,048-token budget, the GSQ-RCO Q2_0 hit its cap on all 11 of its
  misses ([thread](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/71)), and testers of
  the neighbouring files run budgets of 12k or more ([variants note](flash_next_variants.md), section 3).

### What it would do here

- **The size of the saving.** A cap at 4,000 tokens keeps 0.744 of the tokens of 240 second replies and touches 19
  of them; a cap at 8,000 keeps 0.897 and touches 7. The saving is est. at most 25.6 % and 10.3 % of those replies'
  tokens, less whatever the model writes or re-derives after a forced end.
- **The reply limit.** The bench gives a reply 8,192 tokens. A reply that reaches the limit inside its reasoning ends
  with no tool call ([SGLang's cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
  describes the empty content). In the 240 replies above, 7 are longer than 8,000 tokens. A reasoning budget of about
  6,000 under an 8,192 limit leaves the answer room and removes that case.
- **The risk.** 38 of the 42 longest replies of the release arms sit in attempts that passed, so the long replies
  are not waste by default. Whether the tool call written after a forced end is the one the full reasoning would have
  reached is measurable offline (Q1197): recorded replies cut at 1,000, 2,000, 4,000 and 8,000 reasoning tokens, the
  end forced with and without the message, and the tool call compared with the full reply's.

### How it runs

| What | Where | Checked in |
|---|---|---|
| `--reasoning-budget N` (−1 unlimited, 0 an immediate end, N a token budget) and `--reasoning-budget-message TEXT` | server command line | `common/arg.cpp` |
| `reasoning_budget_tokens` or `thinking_budget_tokens`, and `reasoning_budget_message`, per request | request body | `tools/server/server-common.cpp` |
| The sampler: counts after the opening tag, waits for a whole UTF-8 character at the budget, then forces the message and the closing tag token by token | — | `common/reasoning-budget.cpp` |
| Sampling on the device stays on with a budget; a forced token replaces the device's draw on the host | — | `common/sampling.cpp` |
| pi: `compat.thinkingTokenBudgetField: "thinking_budget_tokens"` in `models.json` with `thinkingBudgets` in `settings.json` (budgets exist for `minimal`, `low`, `medium` and `high`) | pi 0.87.1 | its `settings.md` and type declarations |
| The agent check: its extension already rewrites every request, so the two fields are one line each | [check.ts](../../../bench/agent_check/pi/check.ts) | this repository |

Upstream's draft [PR #25961](https://github.com/ggml-org/llama.cpp/pull/25961) adds a soft warning at a fraction of
the budget and an opening message that names the budget. It is not in the release tree, and its one benchmark argues
against both (section 8).

## 4. Attempts that run to the budget

### What is published

- **Failed attempts cost more than solved ones.** [SWE-Effi](https://arxiv.org/abs/2509.09853) (50 SWE-bench
  Verified issues, five scaffolds, three models): an unresolved attempt takes 4.8 times the tokens of a resolved one
  for SWE-agent with GPT-4o-mini, 7.9 times with Llama-3.3-70B and 1.4 times with Qwen3-32B; OpenHands 1.6, 4.5 and
  2.0 times. [AgentStop](https://arxiv.org/abs/2605.15206) (Qwen3-Coder-30B-A3B, mini-swe-agent, 500 tasks): failed
  runs average 544.2 s and 8.0k output tokens against 274.7 s and 4.5k for solved ones.
- **A stop rule on the trajectory.**
  [Fail-Fast, Restart-Smart](https://arxiv.org/abs/2608.03222) (mini-swe-agent, 100 steps, SWE-bench Verified): a
  0.6B monitor that reads the issue and the last 8 steps saves 15.7, 20.4, 14.6 and 16.0 % of execution tokens at 5 %
  false stops on Qwen3.5-9B, Qwen3.6-27B, Gemma4-31B and Gemini 3 Flash. Its "Duration" baseline, the step count
  alone, saves 11.4 % on Qwen3.6-27B, and AgentStop's features 12.5 %. No decision is taken before a fifth of the
  step limit.
- **A fresh attempt after the stop.** The same paper, resolve rate against 66.6 without any stop (Qwen3.6-27B): a
  cold restart 67.2 and 66.8 at 10 % and 25 % false stops; a restart that offers the stopped run's diff 69.8 and 71.8.
  Token overhead at 25 %: +18.2 % and +43.8 %. It has no arm that lets the stopped run continue.
- **Classifiers on behaviour.** [EarlyEval](https://arxiv.org/abs/2609.02783) (LightGBM on about 500 trajectory
  features, 16 models under mini-swe-agent): 34.8 % of runs halted at 95.0 % accuracy, 26.0 % fewer steps, 32.7 %
  fewer input and 28.7 % fewer output tokens, pass rates off by 1.1 points on average. Removing the behavioural
  features (activity counts, repeated actions, error and test status) cuts the halted share to 23.4 %.
- **The weaker result.** [StopLoss](https://arxiv.org/abs/2610.25525): on SWE-bench Verified a 7B monitor stopped 5.0 %
  and 0.0 % of failures (cost per solve −4.5 % and +0.2 %); on a SWE-bench Pro subset it saved 24.2 % and 30.6 % and
  lost 27.8 % and 8.4 % of the successes.
- **Why a retry beats waiting, in theory.** [Reset-and-Discard](https://arxiv.org/abs/2601.21522) proves, for
  independent attempts with a verifier, that moving on after every failed attempt and coming back later solves the
  most tasks per unit of cost; its test is HumanEval. The bench has a verifier (the tests), so the result applies to a
  series: a stopped case goes to the back of the queue.
- **Loops.** The detector of [arXiv 2606.02011](https://arxiv.org/abs/2606.02011) (a 20-gram at least 4 times in the
  last 1,024 tokens) found loops in 18.0 % of a 2-bit Qwen3-32B's replies on question benchmarks. On this project's
  replies none loops word for word (Q1138), so the budget-cut attempts here are long reasoning, and the levers are
  the budget and the stop mark.

### What it would do here

- **The cost today.** A budget-cut attempt is est. 99 minutes of rig time against est. 46 for a passed case
  ([second review](flash_next_review2_2026-11.md), section 2.2).
- **A lower mark, est.** With the same model, an attempt stopped at 49,152 generated tokens costs est. 68 minutes
  (49,152 tokens at est. 69.9 ms and est. 82 calls at 8.0 s). If no pass needs more than the mark, minutes per solved
  case fall from est. 123 to est. 101 in the first series' outcome mix and from est. 100 to est. 86 in the second's.
  One pass lost to the mark takes most of that back (est. 115 in the first mix).
- **What decides it** is the bench's own data: the generated tokens of each passed case. Passed cases average 39,421
  and 32,639 tokens, so a mark near 49,152 is above the average pass; the distribution is in the bench's saved
  attempts (section 9).
- **The agent check** stops an attempt at 12 minutes, and attempts stopped there take 13.11 % of Flash-Next's task
  time in the release arms. A token mark with a fresh attempt from a clean workspace is the same rule in the unit
  that does not depend on the machine (Q1200).

### How it runs

The bench's `max_tokens` per attempt is the mark. In the agent check, `run.py` runs `--attempts` per task from fresh
copies of the workspace, every task's first attempt before any second one (the order Reset-and-Discard favours), and
stops an attempt at `--turn-cap` turns or `--time-cap-min` minutes. A generated-token cap is one more stop of that
kind. No server change.

## 5. First finish among several samples

### What is published

- **[First Finish Search](https://arxiv.org/abs/2505.18149)** starts n samples and returns the first to complete.
  At n = 4 on AIME24, AIME25 and GPQA Diamond the tokens generated in sequence fall from 10.6k to 7.8k (DeepSeek-R1),
  11.4k to 7.8k (R1-Distill-Qwen-32B), 14.9k to 11.8k (QwQ-32B) and 14.7k to 11.2k (Phi-4-Reasoning-Plus), while the
  total over the four samples is 31.1k, 31.3k, 47.2k and 44.8k. Accuracy for DeepSeek-R1: GPQA 72.0 → 74.2, AIME24
  75.0 → 86.7. Correct traces are shorter: 7.2k against 15.4k tokens on average for R1-Distill, 10.7k against 21.3k
  for QwQ-32B.
- **[short-m@k](https://arxiv.org/abs/2505.17813)** stops k samples when the first m finish and votes among them.
  For the same question the shortest chain is up to 34.5 points more accurate than the longest; short-1@k matches
  majority voting at low compute with up to 40 % fewer thinking tokens.
- **Neither paper has a coding or tool-use benchmark.** This project's own gate is the only evidence on tool calls:
  on BFCL the first of two attempts to finish its reasoning scored +0.51 points with 0.72 of the mean reasoning, and
  choosing by total output instead cost −0.5 to −2.6 points (Q93).
- **Restarts.** For a run time with a known distribution the best restart policy is one fixed cutoff, and without
  that knowledge a universal sequence of cutoffs is within a logarithmic factor of it (Luby, Sinclair and Zuckerman,
  1993). A second request sent only once the first is slow is the "hedged request" of large services (Dean and
  Barroso, 2013).

### What it would do here

- **The race is priced** on this project's replies (Q1195): 0.621 of one sample's tokens for the first of two, and
  est. 0.95 of one sample's time when both samples share one rig, so it pays across rigs.
- **The published races save less than this.** Their first of four generates 0.74, 0.68, 0.79 and 0.76 of one
  sample's tokens; here the first of four generates 0.436. The replies here have a heavier tail than those sets.
- **A late start for the second sample.** The median reply is 444.0, 348.0, 378.5 and 338.0 tokens in the four
  loads, and on a reply that short a second sample has nothing to win. Started only when the first sample passes a
  mark, the second rig stays free for most replies and joins on the long ones: 79 of 240 replies pass 1,000 tokens,
  40 pass 2,000. The price of each mark, in sequential tokens and in the second rig's tokens, is a count over the
  same 240 replies (Q1199).
- **A restart on one rig** (abandon a reply at a cutoff and sample again) discards the reasoning already paid for,
  and length here is mostly the prompt's: it helps only where one prompt's own samples spread widely. The same count
  prices it (Q1199).
- **Where to stop.** In second replies the answer after the reasoning is 49.8 tokens on average, so "first to close
  its reasoning" and "first complete reply" pick the same sample almost always.

### How it runs

A proxy in front of two or more servers sends each request to all of them and closes the losers' streams, which
cancels their generation. Within one server, `n` in the request body gives several completions of one prompt, the
children starting when the parent's prompt has been read (`tools/server/server-context.cpp`); it needs as many slots,
and two or more slots with the draft need upstream's fix for the slot finder
([second review](flash_next_review2_2026-11.md), section 4).

## 6. Effort

- **The model card** lists `xhigh` (the default), `medium` and `low`, and says that lower effort does not always
  shorten a multi-turn task, because failures and retries add tokens
  ([card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).
- **The template:** the effort text sits at the top of the system message. `xhigh` and `low` each add a sentence;
  `medium` adds none. So `medium` is the model with no effort instruction, and a change of effort within a session
  re-reads the whole prompt.
- **A same-model sweep on hard cases**
  ([Prompt-Induced Waste in Coding Agents](https://arxiv.org/abs/2608.01347), Claude Code, 25 SWE-bench Verified
  cases, one run a cell): Sonnet 5 resolved 12, 19 and 17 at low, high and xhigh, at $1.76, $2.32 and $3.00 per
  resolved case; Opus 5 resolved 17, 18 and 16 at $0.75, $1.89 and $3.62. Timeouts and empty patches occurred only at
  high and xhigh. With 25 cases a difference of two is within chance; the direction is that the top level buys
  nothing.
- **A sweep on easy cases** ([Can your AI agent be cheaper?](https://arxiv.org/abs/2608.25399), Kimi K3,
  mini-swe-agent, 5 cases, 15 repeats a cell): a run costs $0.117 at low and $0.561 at max (geometric means), and the
  cases are solved 86.3 % to 99.8 % of the time over all efforts. One case with the raw issue text fell to 47 % at
  low against 99 % at high and max.
- **On these files** the GSQ-RCO maintainer reports medium effort degraded
  ([variants note](flash_next_variants.md), section 3).

How it runs: `--reasoning-effort LEVEL` on the server, or `reasoning_effort` in the request, both passed to the
template (`common/arg.cpp`, `tools/server/server-common.cpp`). In pi a model's `thinkingLevelMap` maps pi's levels to
the template's three. The bench arm is the sister repository's E16; the registry row is Q1147.

## 7. Context: what to keep of earlier turns

### What the makers say, and what the template does

- **The card:** `preserve_thinking` is on by default and keeps the reasoning of all earlier messages; the makers
  name agents as the case it helps most, through more consistent decisions and less repeated reasoning.
- **The template** renders an assistant message's reasoning when `preserve_thinking` is true or undefined, or when
  the message comes after the last user message that is not a tool result
  ([chat_template.jinja](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/chat_template.jinja)). With the
  switch off, only messages before the last user message lose their reasoning.
- **So inside one attempt the switch changes nothing.** A bench attempt, like a task of the agent check, is one user
  message followed by replies and tool results: every reply comes after the last user message. `preserve_thinking`
  off matters for a chat of several user turns. An arm that drops earlier reasoning within an attempt has to remove
  `reasoning_content` from the messages before they are sent.
- **Other makers:** OpenAI's guide for its reasoning models reports about 3 % more on SWE-bench when the reasoning
  items between tool calls are passed back, and drops them after each user turn
  ([cookbook](https://developers.openai.com/cookbook/examples/responses_api/reasoning_items)). That is the same rule
  as this template with the switch off.

### What is measured on coding agents

- **[The Complexity Trap](https://arxiv.org/abs/2508.21433)** (SWE-agent, 500 SWE-bench Verified cases, 250 turns):
  tool results older than the last 10 turns are replaced by a one-line placeholder; reasoning and actions stay whole.

  | Model | Solve rate: all kept → masked | Cost per case: all kept → masked | With summaries by the model |
  |---|---|---|---|
  | Qwen3-Coder 480B | 53.4 → 54.8 % | $1.29 → $0.61 | 53.8 %, $0.64 |
  | Gemini 2.5 Flash | 32.8 → 35.6 % | $0.41 → $0.18 | 36.0 %, $0.24 |
  | Gemini 2.5 Flash, thinking | 40.4 → 36.4 % | $0.56 → $0.24 | 31.4 %, $0.25 |
  | Qwen3-32B, thinking | 23.0 → 24.6 % | $0.51 → $0.46 | 24.8 %, $0.51 |
  | Qwen3-32B | 17.0 → 15.0 % | $1.12 → $0.55 | 16.0 %, $0.50 |

  Summaries lengthened the runs: 52 turns against 44 with masking for Gemini 2.5 Flash, and +15 % against keeping
  everything for Qwen3-Coder 480B. Tool results were about 84 % of a turn's tokens.
- **[AgentDiet](https://arxiv.org/abs/2509.23586)** (Trae Agent, Claude 4 Sonnet and Gemini 2.5 Pro): a cheaper model
  rewrites each step two steps later, removing useless, repeated and outdated content. Input tokens −39.9, −40.9,
  −40.4 and −59.7 %; pass rates 64.5 → 66.5, 50.5 → 52.0, 40.0 → 39.0 and 21.7 → 22.7 %.
- **[Confucius Code Agent](https://arxiv.org/abs/2512.10398):** context management on a 100-case SWE-bench Pro
  subset: 42.0 → 48.6 % (Claude 4 Sonnet) and 51.0 → 51.6 % (Claude 4.5 Sonnet).
- **A vendor's number:** Anthropic reports 84 % fewer tokens in a 100-turn web-search evaluation from clearing old
  tool calls and results ([post](https://claude.com/blog/context-management)).

### What it would do here

The papers count input tokens in dollars. On the rig a cached prefix is free, and context costs in two other ways:
every decoding step reads the cache (est. 1.75 ms per 1,000 cells on the release line), and an attempt has to fit
the window. Any edit of earlier messages is paid by reading everything after the edit again, at 69.4 tok/s on the
release line.

- **Removing earlier reasoning, est.** In the agent check 79.42 % of generated tokens are reasoning, so of the
  32,639 tokens a passed bench case generated est. 25,900 are reasoning that later turns read at every step. Without
  them the mean depth of a passed case falls from est. 35k to est. 22k and a step from est. 216 to est. 193 ms: est.
  −10.5 % of decoding, est. 3.9 of 46 minutes. Removed at every turn, each call restores a checkpoint and reads the
  reply's answer again (est. 1.3 s, est. 1.4 minutes over 62 calls). Net est. −5 % of a passed case, against the
  makers' advice. The larger effect is the window: such an attempt fits 65,536 tokens more often.
- **Masking old tool results, est.** A window that slides every turn changes the prefix every turn. Done in blocks
  (everything older than the last M turns once the context passes a mark), it costs one re-read per block: est. 4.8
  minutes for 20,000 kept tokens.
- **pi's compaction** triggers when the context exceeds the window minus `reserveTokens` (16,384), keeps the last
  `keepRecentTokens` (20,000) and asks the model for a summary of the rest, with tool results cut to 2,000 characters
  in the summary's input. At a 65,536-token window that is est. 7 minutes to read the old history, the summary's own
  tokens, and est. 4.8 minutes to read the kept part again: est. 12 minutes a compaction, and the published effect of
  summaries is longer runs. The bench runs with compaction off.

### How it runs

| What | Where |
|---|---|
| `--reasoning-preserve` / `--no-reasoning-preserve`: sets the template's `preserve_thinking` and `clear_thinking`; on by default | `common/arg.cpp`, `common/jinja/caps.cpp` |
| pi: the `context` event transforms the messages before each request; `tool_result` handlers can change a result; `turn_end` can add a context edit or a compaction | pi 0.87.1, `extensions.md` |
| pi: `compaction.enabled`, `reserveTokens`, `keepRecentTokens`; `session_before_compact` can replace the summary | pi 0.87.1, `settings.md`, `compaction.md` |
| The prompt side of dropped reasoning: `replay.py --drop-reasoning` | [bench/agent_check](../../../bench/agent_check/README.md) |

## 8. Early exit, sampler levers and what is parked

- **Early exit by a probe.** [Dynasor](https://arxiv.org/abs/2412.20993) asks the model for its current answer every
  fixed number of tokens, discards the probe and stops when the answers agree: 11 to 29 % fewer tokens at equal
  accuracy on mathematics with distilled Qwen models. [DEER](https://arxiv.org/abs/2504.15895) probes at words such as
  "Wait" and exits on a confident trial answer. Its code table (seven models; HumanEval, BigCodeBench,
  LiveCodeBench): Qwen3-14B on LiveCodeBench 8,203 → 5,437 tokens at 73.4 → 74.1; R1-Distill-Qwen-32B 9,109 → 3,677
  at 56.0 → 56.6; the worst change is −1.2 points (Qwen3-4B, LiveCodeBench) and the best +3.2 (Qwen3-8B,
  BigCodeBench). Both are single-answer tasks. In an agent turn the answer is the next tool call, about 50 tokens
  here; Flash-Next has 2.74 reflection words per 1,000 reasoning tokens, so DEER's trigger is rare and a probe would
  run on a token interval. The server already ends a running reply's reasoning on request: `reasoning_control: true`
  in the request, then `POST /v1/chat/completions/control` with the completion's id and `action: "reasoning_end"`
  (`tools/server/README.md`, `tools/server/server-context.cpp`). A probe that lets the reasoning go on afterwards
  needs a copy of the sequence's state, which is engine code. The forced-end curve (Q1197) gives the share of replies
  whose tool call is already fixed at a given point, the ceiling of any such rule (Q1204).
- **Parked: soft warnings.** The one benchmark on PR #25961 (20 hard MMLU-Pro items, a 3,000-token budget): on
  Gemma4-26B-A4B strict accuracy went from 4/20 with the budget alone to 2/20 with the opening message and 0/20 with
  the message and the warning, and on DeepSeek-V4-Flash one item ran to 7,242 tokens under the 3,000-token budget.
  Reopens with a paired run in which the hard budget is reached less often and the score holds (Q1205).
- **Planned, low: a bias on the end-of-reasoning token** that rises as the budget nears, so the model closes at a
  place of its own choosing. No published measurement was found. A sampler switch; after Q1197 and Q1198 (Q1206).
- **Parked: a presence penalty.** The card allows 0 to 2 against repetition and warns of mixed languages and a
  small loss. No reply here loops, and Q2_0 already switches language at times
  ([variants note](flash_next_variants.md)). Reopens if a census of real cases finds loops (Q1207).
- **Parked: thinking off for some turns** (budget 0 or `enable_thinking` false). HumanEval about 88 % against 93 %,
  and thinking wins on LiveCodeBench at every budget in the no-thinking study. The turns that reason little by
  themselves (the first turn 18.40 %, the final answer 24.91 %) hold little to save. Reopens if Q1197 shows tool
  calls that do not depend on the reasoning (Q1208).
- **Parked: pi's summary compaction as a token lever** (section 7). Reopens for attempts that do not fit the window
  after masking (Q1203).
- **Already rows:** a bias against reflection words is not for Flash-Next (2.74 per 1,000 against the ~5 where it
  pays; [NoWait](https://arxiv.org/abs/2506.08343) cut 13 to 31 % of the reasoning on mathematics with accuracy
  changes of −2.67 to +6.00 points on its three main models; Q94, Q1174). The loop detector is Q266 and Q1139. A
  terse-reasoning line in the system prompt is Q267's zero-cost arm; the wording study above adds that asking for
  several approaches multiplies reasoning tokens by 2.4 to 7.4 and that pi's fixed prefix is 12 to 15 times smaller
  than Claude Code's.

## 9. Asks for the sister repository

The bench and its runs belong to
[elixir-agents-on-mining-rigs](https://github.com/Cinderella-Man/elixir-agents-on-mining-rigs); its registry rows E14,
E15, E16 and E20 are the neighbours.

1. **The mark's curve, a desk read of the saved attempts (with E14):** the generated tokens of every passed case in
   every Flash-Next series, and for marks of 32,768, 40,000 and 49,152 tokens the passes lost and the tokens saved.
2. **Two counts for the failure census (E20):** replies that reached the 8,192-token reply limit inside their
   reasoning, and for each failed attempt the generated-token count at its last new edit.
3. **A bench arm with a reasoning budget:** 6,000 tokens a reply with a stop message, under the 8,192 limit, against
   the bar (E1), three seeds.
4. **A bench arm with a lower mark and a fresh attempt:** the mark from ask 1, one fresh attempt for a stopped case,
   against one attempt to 65,536, at equal total tokens. It belongs beside E15, which doubles the budget.
5. **For E16:** `medium` is the template without an effort sentence; the levels are `xhigh`, `medium` and `low`.
6. **For the arm with `preserve_thinking` off:** inside one attempt the switch drops nothing. The arm needs a pi
   extension that removes `reasoning_content` from replies older than the last K, in blocks (Q1201), and a second arm
   that masks tool results older than M turns (Q1202).
7. **A failure monitor** trained on the bench's saved attempts, after the step-count rule of ask 4: the features of
   EarlyEval's behavioural family first.
8. **The task statement:** in the specification study above, the raw failing-test output was the cheapest variant
   at low effort on four of five cases (0.73 to 1.10 times the full specification's cost).

## 10. New registry rows

In [quants/experiments.md](../../../quants/experiments.md):

| Row | Idea | Status |
|---|---|---|
| Q1197 | The forced-end curve: recorded replies cut at fixed reasoning lengths, the end forced, the tool call compared with the full reply's | planned |
| Q1198 | A reasoning budget with a stop message, below the reply limit: check arms at 4,000 and 8,000, and 4,000 without the message | planned |
| Q1199 | A late start for the race's second sample, and a restart cutoff on one rig, priced on Q1184's replies | planned |
| Q1200 | A generated-token mark per attempt with a fresh attempt, in the agent check | planned |
| Q1201 | Earlier replies' reasoning removed by the harness, in blocks | planned |
| Q1202 | Tool results older than M turns masked, in blocks | planned |
| Q1203 | pi's summary compaction as a token lever | planned, parked |
| Q1204 | Early exit by a tool-call probe through the server's reasoning control | planned |
| Q1205 | Soft warnings and an opening message for the budget | planned, parked |
| Q1206 | A rising bias on the end-of-reasoning token | planned |
| Q1207 | A presence penalty as a length lever | planned, parked |
| Q1208 | Thinking off for some turns | planned, parked |

## Not verified

- **Web sources** were read through summarising fetches of their pages. Values that a paper shows only in figures
  were not read: the Qwen3 report's budget curves, First Finish Search's accuracy by n, SWE-Effi's success by budget.
- **Not opened:** Luby, Sinclair and Zuckerman (1993) and Dean and Barroso (2013); their results are stated from
  memory and from search summaries. Prompt-Induced Waste's per-level token counts are partial in its text.
- **The chat template** was read in the base model's repository. The template inside the Swift GGUF was not read;
  the server's `/props` shows it.
- **The budget sampler on the release line** has not run in this project: its forced tokens with the MTP draft and
  with sampling on the device are read from the source, not measured.
- **pi:** what `thinkingFormat: "qwen-chat-template"` sends for `preserve_thinking`, the exact handler types of the
  `context` and `tool_result` events, and pi's own truncation of tool output were not read in its source.
- **The bench:** how many of its replies reach the 8,192-token limit inside reasoning, and what pi does with such a
  reply.
- **Every "est."** is arithmetic on the second review's model, which is itself unmeasured beyond 32k of context.

## Sources

Project data and notes: [reasoning census](../../data/2026-11-09_reasoning_census/README.md),
[reply lengths and the race's price](../../data/2026-11-10_prob_verify_reply_length/README.md),
[loop census](../../data/2026-11-09_loop_census/README.md), the [second review](flash_next_review2_2026-11.md),
[answer time](answer_time.md), the [variants note](flash_next_variants.md), the
[registry](../../../quants/experiments.md) (Q93, Q94, Q266, Q267, Q268, Q1138, Q1147, Q1152, Q1166, Q1184, Q1195).
The release tree `1551f5fe1` of [Miner Llama](https://github.com/Cinderella-Man/miner-llama): `common/arg.cpp`,
`common/reasoning-budget.cpp`, `common/reasoning-budget.h`, `common/sampling.cpp`, `common/jinja/caps.cpp`,
`tools/server/server-common.cpp`, `tools/server/server-context.cpp`, `tools/server/server-schema.cpp`,
`tools/server/README.md`.

Outside, read 2026-11-10:
- the model: [card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
  [chat template](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/chat_template.jinja),
  [SGLang's cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next),
  [Swift's card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), the
  [Qwen3 technical report](https://arxiv.org/abs/2505.09388);
- llama.cpp: [PR #20297](https://github.com/ggml-org/llama.cpp/pull/20297),
  [PR #25961](https://github.com/ggml-org/llama.cpp/pull/25961);
- pi 0.87.1: [settings](https://cdn.jsdelivr.net/npm/@earendil-works/pi-coding-agent@0.87.1/docs/settings.md),
  [compaction](https://cdn.jsdelivr.net/npm/@earendil-works/pi-coding-agent@0.87.1/docs/compaction.md),
  [extensions](https://cdn.jsdelivr.net/npm/@earendil-works/pi-coding-agent@0.87.1/docs/extensions.md),
  [type declarations](https://cdn.jsdelivr.net/npm/@earendil-works/pi-ai@0.87.1/dist/types.d.ts);
- reasoning length: [First Finish Search](https://arxiv.org/abs/2505.18149),
  [short-m@k](https://arxiv.org/abs/2505.17813), [NoWait](https://arxiv.org/abs/2506.08343),
  [Dynasor](https://arxiv.org/abs/2412.20993), [DEER](https://arxiv.org/abs/2504.15895),
  [Elastic Reasoning](https://arxiv.org/abs/2505.05315),
  [Reasoning Models Can Be Effective Without Thinking](https://arxiv.org/abs/2504.09858),
  [Extreme Low-Bit Inference in Reasoning Models](https://arxiv.org/abs/2606.02011);
- effort and prompts: [Prompt-Induced Waste in Coding Agents](https://arxiv.org/abs/2608.01347),
  [Can your AI agent be cheaper?](https://arxiv.org/abs/2608.25399),
  [Confucius Code Agent](https://arxiv.org/abs/2512.10398);
- context: [The Complexity Trap](https://arxiv.org/abs/2508.21433), [AgentDiet](https://arxiv.org/abs/2509.23586),
  [OpenAI's reasoning items guide](https://developers.openai.com/cookbook/examples/responses_api/reasoning_items),
  [Anthropic's context management post](https://claude.com/blog/context-management);
- failing attempts: [SWE-Effi](https://arxiv.org/abs/2509.09853),
  [Fail-Fast, Restart-Smart](https://arxiv.org/abs/2608.03222), [EarlyEval](https://arxiv.org/abs/2609.02783),
  [StopLoss](https://arxiv.org/abs/2610.25525), [AgentStop](https://arxiv.org/abs/2605.15206),
  [Reset-and-Discard](https://arxiv.org/abs/2601.21522).
