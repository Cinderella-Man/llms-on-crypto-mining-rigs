# v5 review, task level: the race across rigs, retries, several rigs, and how v5 measures itself (2026-11-16)

One of five reviews for the v5 order, after [v4.0](../../book/16-v4.md): what changes a task's time without engine
code. "est." marks my arithmetic, with its basis. Every other number is quoted from the linked folder
or from a row of [the registry](../../../quants/experiments.md), one run unless repeats are listed. Nothing ran for
this note. "The bracket" is the sister repository's run of the rig's own file on 20 real repository tasks
([elixir-agents-on-mining-rigs](https://github.com/Cinderella-Man/elixir-agents-on-mining-rigs),
`docs/data/2026-11-10_bracket`).

## The short answer

- **The race is the largest lever that needs no engine code, and it has never met a real server.** On v4's sampling
  the first of two samples writes est. 0.593 of one sample's tokens. That is est. up to 21 % and 32 % of a task's
  time on v4's two release runs, before the loser's catch-up read. Flash-Next goes first: only its reply lengths
  are counted.
- **A second kind of race needs no proxy and no catch-up.** Each rig runs the whole task and the first finished
  attempt wins: est. 0.71 of one attempt's wall time with two rigs.
- **Stopping and retrying has no workload here.** The check's longest of 357 attempts wrote 19,770 tokens. In the
  bracket a mark at 49,152 tokens stops 5 passes and 4 failures (my count).
- **v5 can report a real-task number without new tasks:** the bracket's per-call table priced at the deep lane's
  speeds, then three recorded attempts replayed on a rig.
- **With five rigs,** two or three race for the agent someone waits on and the others run one agent each. Rigs
  joined into one machine serve one agent worst.

## Where an agent task's time and tokens go

- **Writing is most of a task on v4.** A task's mean seconds of reading, writing, fixed cost and the rest are 33,
  131, 1 and 3 on rig 2, and 23, 78, 1 and 51 on rig 1
  ([R156](../../data/2026-11-13_r156_agent_v4r/README.md), its two report tables). Writing is est. 78 % and 51 %.
  Most of rig 1's 51 s is est. one reply that the 12-minute cap cut off.
- **The work varies more than any switch.** Six runs of the v4 line took 31.3, 19.9, 16.4, 32.2, 33.4 and 30.6
  minutes for the 12 tasks (the same folder, `v4r2_compare.md`). `complex-numbers` alone took 12.0, 1.0, 0.9, 10.8,
  9.7 and 1.0.
- **At equal work, two runs of one setting agree better:** 3.08 and 3.19 minutes a task on v3.0, then 2.44 and 2.37,
  2.35 and 2.44, and 2.37 and 2.43 on three v4 lines. Without `complex-numbers`: 3.13 and 3.09, 2.45 and 2.43,
  2.46 and 2.46.
- **Replies to one prompt vary widely:** the shortest and longest of one prompt's 20 samples are 1,073 and
  13,639 tokens ([reply lengths](../../data/2026-11-10_prob_verify_reply_length/README.md),
  `race_price.md`). The first finisher of 2, 3, 4 and 5 samples writes 0.621, 0.498, 0.436 and 0.399 of one
  sample's tokens on v3's sampling. With probabilistic verification, which v4's preset uses, it writes 0.593,
  0.472, 0.417 and 0.384. A fourth and a fifth rig add est. 0.055 and 0.033.
- **Real attempts are long and deep.** The bracket's first seed with Swift Q2_0: 14 of 20 passed, with 42,230.7
  written tokens and 95.1 calls a passed case. 16 of its 20 attempts peak above the preset's 65,536-token window (my
  count from its `attempts.tsv`).

## What is built, what is measured, what is only estimated

| What | State |
|---|---|
| The race proxy and its runners ([bench/agent_check](../../../bench/agent_check/README.md#the-first-finish-race-across-servers)) | Built. The self-test passes against mock servers. It never ran against llama-server. `race_rigs.sh` defaults to v3's preset and reaches its client over ssh only |
| The race's share | est. from 240 replies: 12 prompts, each task's second reply only, ten samples an arm, counted on newer cards with the v3.0 build. No interval. Three prompts hold est. 74.9 % of the tokens. Two loads of one arm differ by 36.47 % in mean length |
| Passes under a race | Not measured. On another model (Q93): the first to end its reasoning scores +0.51 points with two samples; the shortest whole output −0.5 to −2.6 |
| The loser's catch-up | The restore is measured: 2.3 and 2.4 ms. The re-read is est. 8.9 and 8.1 s per 1,000 tokens of the winner's reply (112.3 and 123.0 tok/s at 512 to 1,023 new tokens, R156) |
| Stop and retry ([Q1200](../../data/2026-11-11_fn58_stop_and_retry/README.md)) | Built, and priced on 876 recorded attempts: nothing to stop |
| Two rigs as one machine ([Q505](../../data/2026-11-04_two_rigs/README.md)) | The RPC build gives the local tokens on a CPU (12 of 12 requests). Never run on rigs. est. 10.7 to 12.3 tok/s plain (Q1169) |
| A real-task time on a rig | None. The bracket counts tokens on newer cards, and Q1163 is a model |
| Stock llama.cpp and Unsloth's files | Measured on v3.0 only ([Flash-Next](../../data/2026-11-05_release_flashnext/README.md), [Flint and Ornith](../../data/2026-11-05_release_regression/README.md)) |

## Candidates for v5

"Worth" is est. share of one agent task's time. Rig hours are est.

| # | Row | What | Evidence | Worth | Cost | Needs | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Q1195 (R48) | The reply race on two rigs, Flash-Next on v4's preset: `one race2 one race2` | est. 0.593 of one sample's tokens | up to 21 % and 32 % (0.407 of the writing time above), before the catch-up | code: small (v4's preset, a local client); 2 h 30 min, two rigs | both rigs idle | first |
| 2 | NEW | The attempt race: each rig runs the whole task, the first finished attempt wins | est. 0.71, 0.58, 0.52 and 0.49 of one attempt's wall time with 2 to 5 rigs (each task's minutes in the six runs, every subset) | 29 % | code: small; 1 h, two rigs | row 1's slot | first |
| 3 | NEW | A count mode for the race proxy: both samples finish, one pair per request | wall time has est. sd 7.2 min over the six runs, so est. 7 runs an arm show 29 % | makes rows 1 and 2 measurable | code: small; 45 min, two rigs | — | first |
| 4 | Q1163 (C667) | Minutes per solved real case, from the bracket's per-call table and the deep lane | est. 54.0 min a passed case on v3.0 and 31.5 on v4.0 (42,230.7 written and est. 60,484.5 new prompt tokens at 16.41 and 91.2, then 26.15 and 218.5 tok/s) | 0 %: a second headline | code: small; no rig | — | first |
| 5 | Q1199 (FN57) | The second sample started late, and the catch-up as a handicap, on the 240 replies | 79 of 240 replies pass 1,000 tokens | prices rows 1 and 11 | code: small; no rig | — | first |
| 6 | Q1145 (C659) | The comparison rule, written down | pairs at equal work differ by 0.11, 0.07, 0.09 and 0.06 min; without `complex-numbers` by 0.04, 0.02 and 0.00 | two runs an arm resolve est. 6 %, or est. 1.4 % without that task | code: none | the owner's word | first |
| 7 | NEW | Baselines in v5's freeze: stock llama.cpp on three public files, v5 on Unsloth's Flint Q8_0 and Ornith UD-Q2_K_XL | v3.0's arms took 40 (its budget), 52.6, 56.5, 19.2 and 28.3 min | 0 %: closes both gaps | code: none; 3 h 40 min, one rig | a stock build at v5's base | next |
| 8 | Q1144 (C658) | Three recorded real attempts replayed on a rig: the passed cases that fit 65,536 | 191 calls, 48,578 written and 99,239 new prompt tokens | 0 %: a measured real-task time, est. 39 min a pass | code: small; 3 h, one rig | the request bodies, from the sister repository | next |
| 9 | Q1271 (FN67) | The race to the end of reasoning, `race2r` | Q93, another model | keeps passes; the first token comes before the reply ends | code: none; 1 h 15 min, two rigs | row 1 | next |
| 10 | Q1403 (R124) | The 131,072-token line on v4 | 3 of the bracket's 14 passed cases fit 65,536 and 11 fit 131,072 (my count) | lets a rig hold real tasks | code: none; 2 h 30 min, one rig | — | next |
| 11 | Q1270 (C718) | The loser reads the winner's reply while the tool runs | "the rest" of a task is 3 s on rig 2 | under 2 % on the check | code: small | row 3's counts | later |
| 12 | Q816, Q553, Q1164 | Rigs as one machine; several slots | est. 10.7 to 12.3 tok/s for IQ3_S on 18 cards against 26.15 on one rig; the bracket: 14 and 14 of 20 | negative for one agent | code: large; both rigs | bracket seeds 102 and 103 | park |

Bases of my estimates:
- **Row 2** uses each task's minutes in those six runs: three v4 lines that were classed level, two runs each. With
  five rigs it leans on each task's best run.
- **Row 4** prices the 14 passed cases at the deep lane's means over all depths
  ([R157](../../data/2026-11-13_r157_release_deep/README.md)). It is low: nothing is timed past 49,152 tokens.
- **Row 6** takes the sd from the pairs' differences (est. 0.073 and 0.018 min) and asks for two standard errors.
  3 % on all 12 tasks needs est. 9 runs an arm.
- **Row 7:** one run each is enough, since v3.0's ratios to stock were 2.48, 3.18, 3.38 and 2.10. With Flint's
  5-bit pair (62.7 and 24.5 min) it is est. 5 h 15 min.

## Rows to close, merge or re-rank

- **FN53's run:** merge into R48. The rigs give the counts themselves (row 3).
- **R48:** its command names v3.0's build and preset. Rewrite it for v4.0.
- **Q93:** status "running" is stale. Its gate passed, and inside one rig it does not pay (est. 0.95 of one
  sample's time). Close; Q1195 and Q1271 carry it.
- **Q94:** dropped already. Its agent form (Q1174) changes the text: behind the race.
- **Q85:** a reasoning budget for Flint: Q1198 for another model. Both stay below every lever that keeps the
  whole reasoning.
- **Q1200:** done. Q1296 stays parked. Its workload is the sister repository's, with the count above.
- **Q1163:** up to first. Its missing inputs are public now.
- **Q1144:** narrow to the replay. New tasks and live bench runs are the sister repository's.
- **Q505, Q816, Q1169, Q553, Q817 to Q819, the two-rig network baseline:** park together. Q1165 asked for 3 cases
  between the files. Seed 101 gave 14 and 14, and IQ3_S wrote 862,612 tokens against 895,310.
- **Q1164, Q1268:** last. Slots matter only when agents outnumber rigs.
- **Q1016:** close. R156 ran one line on both rigs: 2.46 and 2.46 minutes without `complex-numbers`.
- **Q1197, Q1198, Q1204, Q1206:** caps and forced ends. Below, by the owner's rule.
- **Q1201, Q1202:** context edits need the bench's depth: the sister repository's arms.
- **"The one goal":** stale as an order: its profile is v3's base (136 ms a token). Keep the goal's sentence and
  point to the v5 order.
- **"Rigs 3–5":** Q518's setup script is not written. A rig takes est. 2 to 3 hours by hand (Q517): write it first.

## New ideas

- **The attempt race** (row 2): no catch-up, no diverging caches, no proxy.
- **A count mode for the race proxy** (row 3): every request is a pair, so one run prices a race.
- **The attempt race priced on every recorded attempt** (the 357 of Q1200's folder): desk work before row 2.
- **Baselines in every freeze list** (row 7), so no release lacks them again.
- **A column without the runaway task, declared before a run** (row 6).
- **Reply lengths for Flint and Ornith** (Q1184's method), before either model races.
- **A local client for `race_rigs.sh`,** as the release harness has since Q1485.

## What I could not settle from the files

- Whether llama-server stops cleanly when the proxy closes a losing stream mid-reply with the draft on, and whether
  a seed per request changes the draw with sampling on the cards. Only mocks were tested.
- The race's share on whole sessions. Second replies held 11,826 of 34,945 and 10,783 of 21,936 written tokens in
  two runs, and 25,457 of 35,406 and 27,222 of 36,683 in two more
  ([data](../../data/2026-11-10_prob_verify_agent_check/README.md)).
- Whether a finished attempt has passed: real use has no hidden tests.
- The bracket's seeds 102 and 103, and whether its request bodies may leave that repository.
- Whether v4's line holds 131,072 tokens on nine cards and a 4 GB host. v3.0's did (Q1218).
