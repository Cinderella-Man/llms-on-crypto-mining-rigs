# Flash-Next on the v4 line, third review: where an agent task's time goes now, and the order of work (2026-11-11)

This note follows [the first](flash_next_review_2026-11.md) and [the second](flash_next_review2_2026-11.md) review
of 2026-11-09. It reads everything measured on 2026-11-10 and 2026-11-11 (the rig and build-machine rows of both
days, and the engine branches merged into `pascal-p106-v4`), checks which results rest on a stale base, and orders
the open rows by how much of an agent task's time each can take away. Where it differs from the second review's order, this one holds.

Conventions:
- "est." marks my arithmetic or inference, with its basis beside it. Every other number is measured and quoted from
  the linked folder or registry row, one run unless repeats are listed.
- "The v4 line" is the release preset plus the switches of
  [the stacked run](../../data/2026-11-11_r90_v4_vs_v3/README.md) (its `gains` arm), on the fork branch
  `pascal-p106-v4`. Registry rows are in [quants/experiments.md](../../../quants/experiments.md).
- Nothing ran on any machine for this note: file reads only.

## The short answer

- **v4 is real:** with every switch the rigs proved, the step is 108.42 and 108.80 ms at depth 0 against v3.0's
  123.66 and 123.70, and 149.81 and 150.10 ms at 49,152 tokens against 302.25 and 302.37 (Q1322, second request).
  An agent task takes 2.79 min at equal work against 3.08 (one run an arm).
- **The agent check gained far less than the deep lane** (−9.4 % of a task's time against +45.97 % decode) because
  an agent turn runs at 3 to 10 thousand tokens of context, where the depth work gains little, and because a third
  of a task is prompt reading, which got slower for prompts of 1,024 or more new tokens.
- **So the next rows are ranked on the agent's own conditions:** the prompt path on an agent session's prompts by
  size, the step's host work, the verify pass's remaining kernels, and the step between depth 0 and 8,192.
- **Most of that is already written.** 21 switches in `pascal-p106-v4` have never been timed on a P106 (8 in the
  step's host work, 8 kernels, 5 on the prompt and attention path). As single rows on their own tarballs they are
  est. 31 hours of rig time (the rows' own padded estimates). One build holds them all now, so the first pass is
  two ladders, one stacked pair, one short lane and the two native checks, est. 12.5 hours; a single row runs only
  where its rung fails.
- **Five results rest on v3.0's step times or kernels** and are measured again before anything is built on them
  (the table in "Results on a stale base").

## Where an agent task's time goes under v4

From Q1322's agent check, the `gains` run, means over its 12 tasks:

| Part of a task | Seconds | Share |
|---|---:|---:|
| generation | 74 | 52 % |
| prompt reading | 47 | 33 % |
| the rest (tools, the client, fixed costs) | 22 | 15 % |

Prompt reading by the size of a turn's new prompt (the same run, 66 turns):

| New prompt tokens | Turns | Seconds | Share of prompt time | tok/s, v4 line | tok/s, v3.0 (the same day) |
|---|---:|---:|---:|---:|---:|
| under 512 | 42 | 119.7 | 21 % | 64.4 | 39.4 |
| 512 to 1,023 | 9 | 94.9 | 17 % | 71.2 | 61.3 |
| 1,024 to 1,999 | 8 | 135.7 | 24 % | 80.1 | 90.5 |
| 2,000 and more | 7 | 216.7 | 38 % | 76.4 | 93.3 |

What follows from the two tables:
- **Turns of 1,024 or more new tokens hold 62 % of the prompt time (352.4 of 567.1 s) and are the one place where
  the v4 line is slower than v3.0.** At v3.0's speed for those sizes the run would read them in est. 297 s (10,867
  tokens at 90.5 tok/s and 16,553 at 93.3), est. 55 s less, 3.2 % of the run's 28.7 minutes. Q1235 holds the lead
  (the taper loses at 1,024 to 1,999 new tokens in a replay too).
- **The step in an agent turn is est. 131.6 ms** (24.85 tok/s at 3.27 tokens a step), not the 108.6 ms of depth 0.
  On the deep lane the same line's step is 127.22 and 126.80 ms at 8,192 tokens: 18.4 ms grow between depth 0 and
  8,192, before the depth switches act. No row has mapped that stretch with the v4 switches on.
- **A 1 % cut of the step is worth est. 0.5 % of a task; a 1 % cut of prompt time est. 0.3 %.** Rows are ranked
  on that below.

## What the two days proved on a P106

Each row is one switch on its own loads, against the same build without it (the step on the second request unless
said otherwise). The two later k-pool switches ran on top of the tables' switch.

| Switch | Row | Result |
|---|---|---|
| read-once IQ4_XS and Q4_K products of 2 to 4 tokens | Q1277 | step 114.01 and 113.62 ms against 122.10 and 122.03 |
| four-row Q5_K output head | Q1222 | step 116.37 and 116.25 ms against 122.11 and 121.81 |
| one top-k over the output rows | Q1300 | step 121.05 and 121.26 ms against 121.78 and 121.98 |
| in-place indexer | Q1282 | step at 49,152: 168.94 and 169.16 ms against 179.58 and 180.15 |
| k-pool tables on the cards | Q1273 | step at 49,152: 173.97 ms against 179.96 |
| small k-pool inputs in one update | Q1293 | step at 49,152: 173.89 ms against 174.00 |
| pool mask built on the cards | Q1284 | step at 49,152: 172.04 ms against 173.87 |
| checkpoints inside the decode call | Q1306 | prompt replay 338.96 and 331.09 s a pass against 416.56 and 410.60 |
| prompt micro-batches as CUDA graphs | Q1297 | fresh 4,097-token prompt 24,932.3 and 24,305.5 ms against 27,783.8 and 27,050.6 |
| KV-only catch-up of the draft | Q1278 | a load's four prompt chunks 101.4 and 101.8 tok/s against 92.2 and 92.6 |
| upstream's device rows for the draft | Q1367 | stay on: turning them off is −4.18 % decode |

Counted on newer cards (no time): the end-of-turn stop takes grammar rejections from 29 and 24 a load to 0 and 0
(Q1262); a longer draft keeps 3.7028, 4.1257 and 4.5037 tokens a step at 4, 5 and 6 tokens against 3.2401 at 3
(Q1396); every token tree loses to the plain draft of the same width (Q1397).

The stacked run (Q1322) held the first five and the checkpoint switch. The prompt graphs, the KV-only catch-up, the
two later k-pool switches and the stop have not run together with them yet.

## Results on a stale base

| Result | What it rests on | What settles it |
|---|---|---|
| A longer draft pays only below a verify column of 14.77 to 13.11 ms (Q1396) | v3.0's columns: 15.36 ms from 1 to 4 tokens, 21.36 for the fifth (Q1191). The read-once kernels took 7.92 ms off the IQ4_XS and Q4_K products and 4.96 ms off the head in a four-token verify pass (Q1277, Q1222) | the v4 step's map with widths 1 to 8 (Q1393) |
| The draft is at break-even past 64k (Q1251), so a switch turns it off there (Q1399) | v3.0's step times at 65,536 and 120,000 (Q1218); the step at 49,152 has since halved | the draft on against off on the v4 line at both depths (Q1403), before the switch is written |
| The v3.1 file costs est. 5.91 ms a token (Q1382) | v3.0's kernel times per type | a step pair of the two files on the v4 line, two loads each |
| The KV-only catch-up's check through the compute 6.1 PTX ran on the CPU (Q1280) | no device was given to the check | the rig's own identity step has since passed on a P106 (Q1278: 39 of 39 verify steps the same), so the PTX rerun is no longer a gate |
| The v4 line's agent time (Q1322) | one run an arm; five switches classed since are not in it | the next stacked check (Q1261), with the `gains` line as its control |

Two more things the stacked run leaves open:
- **The greedy replies changed with the read-once kernels, and the new replies accept less of the draft:** 85 steps
  against 75 for 256 tokens at depth 0, 84 against 67 and 71 at 8,192. The earlier candidate without those kernels
  wrote v3.0's reply at depth 0 with 75 steps (Q1254). One text per depth cannot tell a flipped near-tie from a
  lower yield. A count on 66 recorded requests can, in minutes, on the build machine's cards (Q1404).
- **The prompt switches were timed on a fresh 4,097-token prompt or on 8,128-token chunks**, never on an agent
  session's prompts by size, where the loss above sits (Q1401).

## The order

Rows that only time code already in `pascal-p106-v4` come first: they need no new code and each has a measured
or estimated size. "Worth" is est. share of an agent task's time by the tables above.

| # | Row | What it times | Est. rig time | Worth, est. |
|---|---|---|---|---|
| 1 | Q1235 | the taper's rule on the v4 line, by prompt size, with the 5 to 8-token read-once kernels as one more pass | 1 h 15 min | up to 3.2 % (the loss at 1,024+ tokens undone) |
| 2 | Q1400 | the step's eight host switches as one ladder | 1 h 30 min | est. up to 8 ms of a 131.6 ms step (the rows' own estimates added): up to 3 % |
| 3 | Q1401 | the prompt switches as a ladder on an agent session's prompts | 2 × 1 h 30 min | est. 10 to 20 % of prompt time: 3 to 7 % |
| 4 | Q1391 (R120a), Q1359 | the native checks of the kernel merge and of the day merge | 2 h + 1 h 20 min | gates for row 5 |
| 5 | Q1391 (R120b) | the verify pass's six unmeasured kernels in one served pair | 1 h 40 min | est. 1.3 to 2.1 ms a step from the two that carry an estimate (Q1287, Q1330); the Q3_K products are the largest part left unpriced (14.535 ms of a four-token pass against 8.69 at one token on v3.0, Q1191) |
| 6 | Q1393 | the v4 step's map at depth 0, 4,096 and 8,192, widths 1 to 8 | 1 h 30 min | prices rows 7 to 9 |
| 7 | Q1315 | the union kernel at 4,096, 8,192 and 16,384 | 1 h 50 min | part of the 18.4 ms between depth 0 and 8,192 |
| 8 | Q1394, Q1402 | a longer draft, fixed or chosen per step | 1 h 30 min offline, then a rig pair | est. up to 5 % of decode if a column is under 10 ms |
| 9 | Q1261 | the stacked agent check of everything classed a gain | 2 h 20 min | the day's bottom line |

Rows moved down, with the reason:
- **The single-switch rows the ladders stand in for** (Q1334, Q1344, Q1345, Q1350, Q1354, Q1361; the served loads
  of Q1285, Q1330, Q1333; the timing halves of Q1239, Q1241, Q1304, Q1349, Q1351): each runs only if its rung is a
  loss, changes a reply, or the stack does not add up.
- **The end-of-turn stop's own timing (Q1384):** it changes no reply and removes rejections of est. 389.65 ms each
  (Q1187), so it rides in the stacked check; its own pair is a diagnostic.
- **Depth beyond 16,384 tokens** (Q1356, Q1342, the whole lane of Q1315, Q1272's deep lane): the step at 49,152 is
  within 41.4 ms of depth 0 now; an agent turn is not there, the bench's long attempts are.
- **Exactness rows with no speed claim** (Q1325, Q1335) and one-token kernels that act only without the draft
  (Q1348, Q1360).

The first-finish race on two rigs (Q1195) stays the largest lever on a task's time that needs no engine code (est.
0.621 of one sample's tokens), and its count on the build machine's cards can run any night. It speeds up neither
decode nor prompt reading, so it is not ranked in the table.
