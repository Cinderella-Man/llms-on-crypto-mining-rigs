# 15. Toward v4: the stack so far

> **In short**
> - v4 is not released. On 2026-11-11 its build holds every change since v3, each behind a switch that is off by
>   default.
> - Eleven of those switches, on together, against v3 on the same rig the same day: Flash-Next writes 45.97 % faster
>   over five depths, 86.51 % faster at 49,152 tokens, and reads prompts 10.74 % faster.
> - On the agent check the same line takes 9.4 % less time per task at equal work, 2.79 against 3.08 minutes, with
>   12 of 12 tasks passed.
> - The agent check gained far less than the deep lane because an agent on small tasks works near the start of a
>   conversation and spends a third of its time reading.
> - 21 more switches are written and checked and have never been timed on a mining card. What comes next is ranked
>   by where an agent task's time goes.

Facts in this chapter are from 2026-11-11. The model is Qwen3.8-Flash-Next in UkisAI's Swift Q2_0 file on nine
cards with its draft, as in the v3 release ([chapter 10](10-v3.md)). "The v4 line" means the release preset plus the
switches named in section 15.2.

## 15.1 The first stack: five switches that needed no code

**The idea.** By the evening of 2026-11-10 five switches had each been timed as a gain on its own. All five were
already in the v3 release build, switched off. Before writing anything new, run them together.

| Switch | Its own result | Chapter |
|---|---|---|
| The attention reads only the cells the model picks | a step at 49,152 tokens 180.77 ms against 301.90 | [12](12-long-conversations.md) |
| Guesses accepted by probability | writing 2.97 % faster | [13](13-the-check-pass.md) |
| Checkpoints kept in card memory | reading time 15.40 % shorter | [14](14-reading-prompts.md) |
| Small batches at the edges of a prompt | reading time 14.28 % shorter | [14](14-reading-prompts.md) |
| The draft reads prompts in batches of 512 | reading 3.7 % faster | [14](14-reading-prompts.md) |

**Why run them together at all.** A fair question came up that day: the reading switches had been timed only on
reading, the writing switches only on writing. Nobody had checked that a switch which speeds up reading does not
slow down writing. Gains measured apart do not have to add, and some of these compete for the same card memory.

**What happened.** On the deep lane against the release preset, two loads each: writing 20.66 and 20.10 tokens per
second against 14.78 and 15.85 over the five depths, 33.08 % more, and 19.08 and 18.34 against 10.40 and 10.70 at
49,152 tokens. Reading 101.4 and 101.9 tokens per second against 88.7 and 88.7, 14.55 % more. A loss at no depth, in
neither direction.

Three things got worse and were written down with the gain: the host's free memory fell to 106 MiB in one load, the
last card held 180 MiB more, and one request at the start accepted fewer guesses than the release's.

**Why.** The five add up because they attack different parts: one the attention, one the accept rule, three the
prompt path. And a release build that carries its experiments behind switches ([chapter 10](10-v3.md)) made this a
change of settings, not a build.

## 15.2 The second stack: everything proven, against v3

**What I did.** One day later the rebased build of [chapter 11](11-after-v3.md) held the day's new kernels, and six
of them had been timed as gains. So one run, three settings, on one rig in one afternoon:

- **v3:** the v3.0 release build with its preset.
- **The new build, no switch:** to see what the rebase alone did.
- **The v4 line:** the new build with the five switches above and six more: the two check-pass kernels and the
  single top-k filter of [chapter 13](13-the-check-pass.md), the in-place index read and the index tables on the
  cards of [chapter 12](12-long-conversations.md), and the checkpoints taken inside the call of
  [chapter 14](14-reading-prompts.md).

First the deep lane, two loads each, then the agent check, v3 against the v4 line.

**What happened on the deep lane.** Over all five depths:

| | v3 | New build, no switch | The v4 line | v4 line against v3 |
|---|---|---|---|---:|
| Writing, tok/s | 15.87 and 14.77 | 15.26 and 15.67 | 22.50 and 22.24 | +45.97 % |
| One step, ms | 204.19 and 205.44 | 201.05 and 201.25 | 130.85 and 131.06 | −36.06 % |
| Writing with the makers' sampling, tok/s | 13.22 and 12.76 | 13.76 and 13.68 | 21.79 and 21.86 | +67.99 % |
| Reading, tok/s | 87.7 and 88.0 | 91.5 and 91.5 | 97.0 and 97.6 | +10.74 % |

The new build without its switches writes as fast as v3, 0.92 % apart, inside the spread, and reads 4.17 % faster.
So the gain in writing belongs to the switches, not to the rebase.

By depth, writing in tokens per second:

| Depth (tokens) | v3 | The v4 line | Change |
|---:|---|---|---:|
| 0 | 26.92 and 26.62 | 27.15 and 27.10 | +1.33 % |
| 8,192 | 19.43 and 19.30 | 20.86 and 20.90 | +7.81 % |
| 16,384 | 18.36 and 16.61 | 25.47 and 25.38 | +45.39 % |
| 32,768 | 11.58 and 10.83 | 19.71 and 19.67 | +75.76 % |
| 49,152 | 11.68 and 10.36 | 21.06 and 20.05 | +86.51 % |

At the start the step is 12.2 % shorter ([chapter 12](12-long-conversations.md) has the step by depth) but the
speed moved only 1.33 %. The new kernels changed the text at the start, and the new text accepted fewer guesses:
85 steps for its 256 tokens against 75. That is the effect [chapter 11](11-after-v3.md) describes, and the reason
a change is judged on the step.

**What happened on the agent check.** The same twelve bugs, one run each:

| | v3 | The v4 line |
|---|---:|---:|
| Tasks passed | 11 of 12 | 12 of 12 |
| Minutes per task, at equal work | 3.08 | 2.79 |
| Writing, tok/s | 20.66 | 24.85 |
| Reading, tok/s | 65.5 | 73.9 |
| First word of a turn, median | 6.52 s | 4.55 s |
| First word of a turn, one turn in ten waits longer than | 16.17 s | 24.48 s |
| Share of guesses accepted | 0.696 | 0.757 |
| The host's free memory at its lowest | 66 MiB | 415 MiB |

Two earlier runs of v3's preset on the other rig took 3.03 and 3.06 minutes a task, so 2.79 is outside v3's own
spread. It is still one run, and the next stacked check repeats it.

**What got worse.** Three things, each with a row on the list:

- Turns with 1,024 or more new tokens read slower than on v3, and the slowest tenth of first words got slower
  ([chapter 14](14-reading-prompts.md)).
- The last card holds 254 MiB more, 4,881 of its 6,144 MiB.
- The text at temperature 0 differs from v3's, and whether the new kernels change how many guesses are accepted is
  not settled ([chapter 13](13-the-check-pass.md)).

One load was also stopped by the memory watchdog of [chapter 5](05-nine-cards.md): the new build without switches
ran the host down to 54 MiB at 32,768 tokens. v3 itself reached 80 MiB in the same run. The v4 line, with its
checkpoints on the cards, never went below 304 MiB on the lane.

## 15.3 Why 46 % became 9 %

The deep lane says +45.97 %. The agent check says −9.4 % of a task's time. Both are right, and the gap between them
is the most useful result of the day.

An agent task on the v4 line takes 143 s on average:

| Part of a task | Seconds | Share |
|---|---:|---:|
| Writing | 74 | 52 % |
| Reading prompts | 47 | 33 % |
| Everything else: tools, the client, fixed costs | 22 | 15 % |

Three things follow.

- **The agent works shallow.** The twelve bugs end at 2,800 to 10,000 tokens of context. The deep lane's large gains
  are at 16,384 tokens and beyond. At the start the gain is 1.33 % in speed and 12.2 % in the step.
- **A step in an agent turn is est. 131.6 ms**, from 24.85 tokens per second at 3.27 tokens a step. At the start of a
  conversation it is 108.42 and 108.80 ms, at 8,192 tokens 127.22 and 126.80. So even with every gain on, the step
  grows by about a sixth across exactly the stretch where the agent works, and nothing has mapped that stretch yet.
- **A third of the time is reading,** and reading got slower for the largest turns.

The bench of [chapter 11](11-after-v3.md), with attempts of 40,000 tokens, sits at the other end: there the deep
lane's numbers apply almost directly. Small tasks and real tasks need different parts of the same work, and one
number cannot stand for both.

So the ranking changed. One percent off the step is worth est. 0.5 % of a small task, one percent off reading est.
0.3 %. Work on depth beyond 16,384 tokens moves the bench and barely moves the agent check.

## 15.4 A better file at the same size

The engine work left one question from [chapter 4](04-smaller-models.md) open: can the file be better at the same
size? v3 ships UkisAI's file unchanged, because my faster mix of it no longer made a difference in speed
([chapter 10](10-v3.md)). Quality is another matter.

**What I did.** The method of [chapter 4](04-smaller-models.md) again, on Flash-Next: a list of switches, each
changing one kind of tensor in one group of eight layers to another format, each scored alone on three texts; then a
solver picks the set with the least damage for a byte budget. With my Pascal mix as the start and its own size as
the budget, the solver picked five switches.

**What happened.** The new mix is 25.9 MB smaller than the start. KL divergence against the 8-bit reference, lower
is better:

| Text | The start | Predicted for the new mix | Measured |
|---|---:|---:|---:|
| Agent sessions | 0.279228 | 0.21610 | 0.236062 |
| Wikipedia | 0.426376 | 0.41379 | 0.414086 |
| Code | 0.108328 | 0.09212 | 0.095812 |

A gain on all three, at no cost in size. On agent text the gain is 0.684 of the predicted one. Most likely two
of the five switches overlap: both change the same group of layers, and both gain on agent text only. A run that
builds the file without one switch at a time will say.

**What is open.** Speed. The kernel timings behind the solver's estimate, est. 78.35 ms a token against 72.44, are
from before the kernels of [chapter 13](13-the-check-pass.md), so the estimate is stale. The file is on a rig and
is timed next, as a pair of steps against the start.

**Why.** On Flint the costs of separate switches added up well, and a cheap search found the best file. Here they
mostly did, and where they did not, the measurement said so the same evening. A prediction is a reason to build
the file, not a substitute for scoring it.

## 15.5 Fewer tokens, not faster ones

Since 79.42 % of what the model writes on the agent check is reasoning ([chapter 11](11-after-v3.md)), the largest
lever on a task's time may not be the engine at all. Two ideas were priced. Neither cuts the reasoning short, which
costs answers.

- **A race.** Two rigs answer the same request with different random draws, and the agent takes whichever reply is
  complete first. On recorded replies, the shorter of two samples has est. 0.621 of one sample's tokens. The proxy
  that runs the race is written and tested against mock servers; its first counted run is queued.
- **A fresh attempt.** Stop an attempt that has written far more than a solved task usually needs, and start again.
  On the agent check as recorded there was almost nothing to stop: the longest of 357 attempts wrote 19,770 tokens
  and passed. It is parked until a workload shows runaways that this would catch.

## 15.6 What is written and waits

Rig time is what limits the work now. Code is ahead of measurement:

| Group | Switches written, checked, not yet timed | Chapter |
|---|---:|---|
| The server's work between two steps | 8 | [13](13-the-check-pass.md) |
| Kernels for the check pass | 8 | [13](13-the-check-pass.md) |
| The prompt path and the attention at depth | 5 | [12](12-long-conversations.md), [14](14-reading-prompts.md) |

Timed one by one, each on its own build, they would need est. 31 hours of rig time. One build holds them all, so
they are timed as *ladders*: the first load runs the line as it is, the next adds one switch, the next another, and
the whole round runs twice. Each rung is judged against the one before it, and the top against the bottom. A rung
that loses, or changes a reply, goes back to a run of its own. Est. 12.5 hours for the first pass.

## 15.7 The order from here

| # | What | Why it is next |
|---|---|---|
| 1 | The taper's rule by prompt size | done: the taper without its head rule, `short=600:64`; the loss itself comes from graph rebuilds ([14.9](14-reading-prompts.md#149-the-loss-that-is-still-open)) |
| 2 | The eight switches between two steps, as a ladder | est. up to 8 ms of a 131.6 ms step, and no new code |
| 3 | The prompt switches on an agent session's own prompts, as a ladder | est. 10 to 20 % of reading time |
| 4 | The check-pass kernels: the checks on a card, then one timed pair with all of them on | the largest group of products left in a check pass has no fast kernel in use yet |
| 5 | A map of the step at 0, 4,096 and 8,192 tokens, for passes of one to eight tokens | names what grows where the agent works, and prices a longer draft |
| 6 | The kernel for a check's four tokens at the agent's depth | part of the 18.4 ms between the start and 8,192 tokens |
| 7 | A longer draft, fixed or chosen step by step | only if the map shows a checked token under est. 13 ms |
| 8 | The stacked agent check again, with everything proven since | the day's bottom line, and the second run this line still lacks |

## 15.8 What v4 will be

v4 will be cut the way v3 was ([chapter 10](10-v3.md)): one build, every feature behind a switch, a preset per
model with the switches its own agent check ran with. The plan for the cut is written down:

1. No new feature starts. Every finished branch is merged once, and that commit becomes the release candidate.
2. The candidate gets the full set of checks on a mining card: the kernel cases whose machine code changed, and the
   same answers as v3 for Flash-Next, Flint and Ornith-1.5-122B with every switch off.
3. The deep lane and the agent check run with the v4 preset against v3 on the same rig.
4. The preset files and the release notes are written from those numbers. A fix during the freeze makes a second
   candidate and repeats only the affected checks.

Until then the numbers in this chapter are a line under test, not a release.

## What to remember

- Measure switches together as well as apart, on writing and reading both. Gains measured apart do not have to add.
- A benchmark at depth and a benchmark of small tasks answer different questions. +45.97 % on one was −9.4 % of a
  task's time on the other, and both were correct.
- Split a task's time into its parts before ranking the next idea. Here it is 52 % writing, 33 % reading.
- Write down what got worse beside what got better, in the same table.
- One run is a result with no spread. Repeat it before it becomes a headline.
- A predicted file is a reason to build it. The measured gain was 0.684 of the predicted one on the text that
  matters most.
- When code is ahead of measurement, measure in ladders: one build, one switch more per rung.

## Where the details are

- The first stack: Q1254 ([data](../data/2026-11-10_v4_candidate/README.md)).
- The second stack: Q1322 ([data](../data/2026-11-11_r90_v4_vs_v3/README.md)), on the day's merged build, Q1359
  ([data](../data/2026-11-11_v4_day_merge/README.md)); the rebased build alone on the agent check,
  [2026-11-11_r57_v4night_p106](../data/2026-11-11_r57_v4night_p106/README.md); the watchdog stop, Q1366. The next
  stacked check is Q1261.
- Where a task's time goes, and the order of section 15.7:
  [the third review](../research/notes/flash_next_review3_2026-11.md); the ladders, Q1400 and Q1401; the taper,
  Q1235; the kernels on a card, Q1391; the map, Q1393; the four-token attention kernel, Q1315; longer drafts,
  Q1394 and Q1402.
- The new mix (`gmix-q2`, five switches on `pmix-q2-fast`): the solver and the pick, Q219
  ([data](../data/2026-11-04_fmt_mixes/README.md#the-v31-pick-2026-11-11)); its measured scores, Q1381; the
  overlap, Q1392; on a rig, Q1382. Code text scored for every format file: Q219 and Q1185.
- Fewer tokens: [tokens_per_solved_task.md](../research/notes/tokens_per_solved_task.md); the race, Q1195
  ([bench/agent_check](../../bench/agent_check/README.md)); the fresh attempt, Q1200
  ([data](../data/2026-11-11_fn58_stop_and_retry/README.md)).
- Every switch and its default: [docs/switches.md](../switches.md).
- Registry: [quants/experiments.md](../../quants/experiments.md).
