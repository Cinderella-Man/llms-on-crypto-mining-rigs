# 11. After v3: finding where the time goes

> **In short**
> - After v3 I did not start the next idea. First I asked two questions: what does real agent work look like, and
>   where does a token's time go on the released build?
> - Real work is long. A coding agent on a real repository writes tens of thousands of tokens in one attempt, so
>   speed deep into a conversation matters more than speed at its start.
> - A new map of the release build says 93 % of a pass is GPU time. The next milliseconds are in the kernels and in
>   what grows with the conversation, not in the host.
> - Three working rules came out of it: measure the map before the idea, compare the time of one step and not only
>   tokens per second, and prove a change exact in four ways before a mining card times it.
> - Everything built since sits in one build, each change behind a switch that is off by default.

Facts in this chapter are from 2026-11-11. The model is Qwen3.8-Flash-Next on the v3 release preset
([chapter 10](10-v3.md)) unless a section says otherwise. Chapters 12 to 15 tell what the new maps led to.

## 11.1 Two reviews

**The idea.** v3 shipped on 2026-11-06. By then the project held more than a thousand recorded ideas, and most of
their estimates had been written while the engine was changing under them. Before picking the next one, check the
list against the build that shipped.

**What I did.** I reviewed the whole project twice on 2026-11-09: once for mistakes in my own thinking, and once to
check the first review claim by claim against the data folders, the release source and what others had published
that week.

**What happened.** Three findings changed the order of work.

- **Every estimate was priced on an old token.** Rows said "saves 5 ms" against a token of 136 ms or a step of 179 ms.
  The release build's token and step were much shorter, and nobody had profiled them with every v3 feature on. An
  estimate against the wrong total ranks ideas wrongly.
- **Things were built and never measured.** The largest was the sparse attention of [chapter 6](06-gpu-code.md): in
  the release build behind a switch, checked on a CPU, and never run on a GPU.
- **The test was not the workload.** The agent check's twelve small bugs end after a few turns. Real work goes much
  deeper into a conversation, as the next section shows.

**Why.** A to-do list is a cache of old decisions. After a release the thing it was computed from has changed, so it
needs a refresh before it is trusted.

## 11.2 What real work looks like

The agent check of [chapter 3](03-measuring.md) fixes one-line bugs in single files. Its conversations end at about
2,800 to 10,000 tokens. That makes it a good regression test, and a poor picture of a working day.

A harder test exists: the [Elixir SWE Bench](https://github.com/Cinderella-Man/elixir-swe-bench), 20 tasks taken
from real repositories. Its published runs show what a real attempt costs:

| What a real attempt looks like | Measured |
|---|---|
| Tokens Flash-Next writes in one attempt, on average (two files) | 40,557 and 47,293 |
| The window the bench gives the model | 262,144 tokens |
| The v3 preset's window | 65,536 tokens |
| Share of all written tokens that are reasoning, on the agent check | 79.42 % |
| Share of the written tokens that sit in the longest 10 % of replies | 63.20 % |

Two things follow. A typical attempt does not fit the preset's window at all. And priced on v3's speeds by depth, a
solved task costs est. 100 to 123 minutes of rig time, with more than half of that spent in attempts that end at the
token budget.

So the goal got a second number beside the agent check: minutes of rig time per solved real task. Three kinds of
work serve it, and the next chapters take them in turn: speed deep into a conversation
([chapter 12](12-long-conversations.md)), the cost of one step ([chapter 13](13-the-check-pass.md)), and reading
prompts ([chapter 14](14-reading-prompts.md)). Making the model write fewer tokens is the fourth, and it is mostly
outside the engine.

One worry turned out empty. Long replies could have been loops, the model repeating itself until the limit. A count
over 4,706 recorded replies found one loop. The long replies are long reasoning.

## 11.3 Story: the map of one pass

**The idea.** Profile the release build as it ships, at the two pass sizes it really runs: one token, and the four
tokens of a check (the newest token plus three guesses, [chapter 8](08-guessing-ahead.md)).

**How it should work.** The profiler of [chapter 6](06-gpu-code.md) puts a timestamp between groups of GPU launches.
Its own cost, 7.6 µs per timestamp, is subtracted. Whatever the nine cards do not account for is hops, host work and
gaps.

**What went wrong first.** The first run measured the wrong thing. With the draft loaded, the "one-token" and
"four-token" requests both ran as five-token passes. A recurrent layer cannot be rewound, so the engine keeps the
last few tokens of a prompt together in one batch, and with a draft of three that batch is five tokens. I found the
cause in the source and ran again without the draft.

**What happened.** GPU time per pass, by part, on real agent text:

| Part of the pass (ms) | 1 token | 4 tokens | 5 tokens |
|---|---:|---:|---:|
| Attention and recurrent layers' matrices | 15.81 | 34.13 | 39.00 |
| Routed experts | 6.97 | 19.68 | 24.21 |
| Mixing matrices between the model's four streams (16-bit) | 9.49 | 12.73 | 18.20 |
| Operations without weights | 4.72 | 6.10 | 6.23 |
| Shared expert | 2.56 | 4.79 | 5.13 |
| Recurrent state updates | 2.65 | 3.79 | 4.25 |
| Router | 1.79 | 2.35 | 4.57 |
| Attention over the cache | 0.59 | 1.01 | 1.50 |
| Other weights | 0.47 | 0.78 | 0.94 |
| **All nine cards** | **45.05** | **85.36** | **104.03** |
| The whole pass, unprofiled | 48.31 | 91.20 | 111.00 |
| **Hops, host and gaps** | **3.26** | **5.84** | **6.97** |

**Why it matters.** Two readings of this table drove everything after it.

- **The cards hold 93 % of a pass.** Hops, host work and gaps are 3.26 ms of a one-token pass. Six days earlier that
  line had been 11.60 ms ([chapter 6](06-gpu-code.md)). Ideas that speed up the hand-offs have at most 3.26 ms of a
  one-token pass and 5.84 ms of a four-token pass left to win.
- **A fourth token is cheap, a fifth is not.** From one to four tokens a pass grows 14.30 ms per added token. The
  fifth adds 19.80 ms: the mixing matrices and the router jump at five tokens (12.73 to 18.20 ms and 2.35 to
  4.57). That step in the curve decides how long a draft can pay, and [chapter 13](13-the-check-pass.md) comes
  back to it.

## 11.4 Story: the map of one step

**The idea.** A pass is not what a user waits for. With the draft on, the unit is a *step*: one check pass over four
tokens, then the draft's own small passes that prepare the next three guesses. A step yields one sure token plus
every guess the model agrees with, a little over three tokens on average. Map the step.

**What I did.** I profiled real generation on the release preset, with a second profiler mode that timestamps the
host side of every hand-off.

**What happened.** One step at the start of a conversation, 82 steps averaged:

| Part of a step | ms |
|---|---:|
| The check pass: GPU time on the nine cards | 102.10 |
| The check pass: crossings, results back, the accept test | 4.98 |
| The draft: four small passes on the last card, with their crossings and hand-backs | 18.29 |
| The server between the last draft pass and the next check | 7.82 |
| **One step** | **133.18** |

Cards account for 115.76 ms of it, 86.9 %. A step deeper into the conversation is longer: 140.46 ms at 2,760 tokens
of context against 131.57 at the start (medians).

Then the surprise. A check pass inside a step takes 15.68 ms longer than a plain four-token pass, and all of it is
GPU time. A second profile split that excess:

| Where a check pass costs more than a plain four-token pass | ms, in two requests |
|---|---:|
| The output layer computed for four tokens instead of none | 8.05 and 8.06 |
| Sampling the four results on the card | 0.68 and 0.68 |
| Snapshots of the recurrent state, so a wrong guess can be undone (cards 0 to 7) | 5.37 and 5.39 |

**Why.** A plain pass over a prompt needs no output: only the last token's scores matter. A check needs the scores at
every guessed position, and it must be able to roll back. Those two needs cost 14.10 ms in the first request, and no table of "pass
time by size" had ever shown them. They became the first targets of [chapter 13](13-the-check-pass.md).

## 11.5 A lane for depth

The agent check cannot say how fast the model is 40,000 tokens into a conversation, because its conversations never
get there. So there is a second test, the *deep lane*.

It fills the context with recorded agent text to a set depth, then asks for the same 256 tokens twice. Depths are 0,
8,192, 16,384, 32,768 and 49,152 tokens. Every setting gets two full loads of the model, and the settings' loads
alternate, so a slow hour hits all of them alike. It reports writing speed, prompt reading speed and
the time of a step at each depth.

**Why the step and not only tokens per second.** Two settings can write different text. A change of rounding flips a
near-tie, the text goes another way, and the new text happens to be easier or harder to guess. Then tokens per
second moves by 10 % for no reason the engine controls. The time of one step does not depend on the text. So a
change is judged on the step, and tokens per second is reported beside it.

The second of the two requests is the cleaner one. The first follows the long fill of the context, when the 4 GB
host is still swapping memory back in. On v3 at 8,192 tokens the first request's step took 234.86 and 217.84 ms in
two loads, the second request's 158.48 and 152.65.

## 11.6 Three verdicts, never "too small"

Every timed change now gets one of three verdicts, against the spread between two loads of the same setting:

| Verdict | Rule | What happens to it |
|---|---|---|
| Gain | Faster by more than the spread | It joins the stacked line |
| Negligible | Inside the spread, and not measurably slower | It stays in the build behind its switch and is tested later with the other small ones, as a group |
| Loss | Slower by more than the spread | It is parked, with its numbers |

There is no verdict "not worth it". A change that saves half a millisecond is kept. [Chapter 10](10-v3.md) made the
case for this, and the weeks since have confirmed it: the largest result so far is a stack of eleven
changes ([chapter 15](15-toward-v4.md)).

## 11.7 A newer base under everything

**The idea.** v3 was built on upstream llama.cpp's release b12076. By 2026-11-10 upstream stood at b12246 and had
merged work this project had planned to write itself. The largest piece keeps the draft's input, the main model's internal
state at each position, on the card instead of copying it to the host and back. Rebase first, then build on it.

**What I did.** I merged upstream's newer release under v3's tree, with two upstream bug fixes for the recurrent
state that the reviews had found missing. Then I ran every feature's check again, and the agent check on a rig.

**What happened.** With every new switch off, the rebased build ran the agent check at v3's speed: 2.75 against 2.75
minutes a task at equal work, 12 of 12 against 11 of 12 tasks. Upstream's change for the draft's input was on by
default. Turning it off cost 4.18 % of writing speed over three depths and 3.39 % of prompt speed, so it stays on. It
holds 82 MiB more on the last card.

**Why.** Other people's work arrived for the price of one merge and a day of checks. Read what upstream merged
before writing the same thing.

## 11.8 Four checks before a card's clock

A timed run needs a quiet rig, one run at a time. Rig time is the scarcest thing in the project. So a change must be proven correct before it gets any, and everything that can be
proven elsewhere is proven elsewhere.

| Check | Where | What it proves |
|---|---|---|
| Switch off, the build gives the old build's numbers bit for bit, on all 146 model types in llama.cpp's test suite | Any CPU | The change touches nothing until it is switched on |
| Switch on, the new path gives the old path's results bit for bit; for a GPU kernel, in a CPU emulation of its own arithmetic | Any CPU | The new code adds the same numbers in the same order |
| The rig build runs a real model on newer NVIDIA cards, which can load the same portable GPU code the P106's driver compiles | A PC with newer cards | Whole-model output and quality at depth, with no rig time |
| Recorded test cases are replayed on one P106, with a planted bug that must fail | The rig, minutes | The code is right on the real card |

Only after those does a change get a timed run.

**Story: a check that never touched the card.** One of these checks passed 15 of 15 cases on a newer card in a
minute. Its log showed the card's memory had not moved. The test built its model with an empty device list, so
everything had run on the CPU. A passing test proved nothing about the GPU path it was meant to cover. The rig's own
check covered that path later, and a device option for the test is on the list.

**Story: a planted bug that crashed the checker.** Each replay set includes a run with a deliberate fault, to prove
the test can fail. One fault made the checker itself abort after 104 cases, on an assertion in the fault's own
code. The kernel under test was fine. The fault code got its own fix; its rerun on a card is queued.

**Why.** A test is a program too. It can pass for the wrong reason or fail for the wrong reason. A control that must
fail, and a look at whether the card actually worked, cost minutes and catch both.

## 11.9 Many small changes, one build

The work since v3 follows one pattern. A map names a part of the step that costs more than it should. One small
change attacks that part, on its own branch, behind its own switch. It passes the four checks. Then all finished
branches are merged into one build, with every switch off, and that build must still give the base's numbers bit
for bit.

Seventeen branches were merged one night, seventeen more the next day, and seven after that. One real conflict came
out of the merging: two changes that each worked alone lost the server's checkpoints when both were on. A check of the two
together found it, and the fix went in before they were timed together.

Then each switch gets its turn on a rig, alone on the merged build, against the same build with no switch set.

## What to remember

- After a release, refresh the plan against what shipped. Estimates made against an old total rank ideas wrongly.
- Measure the map before the idea. On v3 the cards hold 93 % of a pass; that one number retired a whole family of
  host-side ideas.
- A check pass is not a plain pass. It computes the output at every guessed position and keeps rollback copies:
  14 ms that only a profile of real generation showed.
- Judge a change on the time of one step. Tokens per second also moves when the text changes.
- Give every result one of three verdicts against the spread of repeated loads. Keep the small gains.
- Prove a change exact on a CPU, on newer cards and on one old card before spending timed rig hours on it. Make sure
  each check can fail.

## Where the details are

- The two reviews: [the first](../research/notes/flash_next_review_2026-11.md) and
  [the second](../research/notes/flash_next_review2_2026-11.md); a third, on 2026-11-11, is in
  [chapter 15](15-toward-v4.md).
- Real attempts: the bench's runs as read in the second review (section 2), the headline number Q1163; the
  reasoning share, Q1152 ([data](../data/2026-11-09_reasoning_census/README.md)); the loop count, Q1138 and Q1158
  ([data](../data/2026-11-09_loop_census/README.md)).
- The map of a pass: Q1191 ([data](../data/2026-11-10_release_profile/README.md)); the batch a draft forces on a
  prompt's tail is Q1192. The earlier 11.60 ms line is [chapter 6](06-gpu-code.md)'s profile, Q822.
- The map of a step: Q961 ([data](../data/2026-11-10_served_step_split/README.md)); the check pass's excess: Q1211
  ([data](../data/2026-11-10_served_verify_parts/README.md)).
- The deep lane: [bench/deep_lane](../../bench/deep_lane/README.md) and its runs in
  [2026-11-10_deep_lane](../data/2026-11-10_deep_lane/README.md); the first and second request at 8,192 tokens are
  from the stacked run, Q1322 ([data](../data/2026-11-11_r90_v4_vs_v3/README.md)).
- The three verdicts and the group of small gains: Q1023 and [docs/switches.md](../switches.md).
- The newer base: Q1150 ([data](../data/2026-11-10_ladder6/README.md)); on a rig,
  [2026-11-11_r57_v4night_p106](../data/2026-11-11_r57_v4night_p106/README.md); upstream's defaults timed, Q1367
  ([data](../data/2026-11-11_r92_upstream_defaults/README.md)).
- The checks: the replay tool [scripts/replay_checks.sh](../../scripts/replay_checks.sh); the night's replay on a
  P106, Q1321 ([data](../data/2026-11-11_v4night_replay/README.md)); the check that ran on the CPU, Q1280
  ([data](../data/2026-11-11_fn68_c620_ptx/README.md)); the fault that aborted, Q1324
  ([data](../data/2026-11-11_c749_lid_rows_fault/README.md)).
- The merges: Q1319 ([night](../data/2026-11-11_v4_night_merge/README.md)), Q1359
  ([day](../data/2026-11-11_v4_day_merge/README.md)), [late](../data/2026-11-11_v4_late_merge/README.md), Q1391
  ([kernels](../data/2026-11-11_v4_kernel_merge/README.md)); the two changes that collided, Q1327
  ([data](../data/2026-11-11_c735_c217_c703/README.md)).
- Registry: [quants/experiments.md](../../quants/experiments.md).
