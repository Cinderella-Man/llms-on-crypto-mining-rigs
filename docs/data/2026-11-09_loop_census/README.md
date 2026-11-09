# Loop census of the agent check's replies (Q1138, Q1158; 2026-11-09)

What this is: a count of how many replies of the agent check ([bench/agent_check](../../../bench/agent_check/README.md))
are the model repeating itself, over every reply the check's proxy recorded on the rig from 2026-10-26 to 2026-11-06:
4,713 replies in 80 arms of Qwen3.8-Flash-Next, Flint and Ornith-1.5-122B. Registry rows Q1138 (the census)
and Q1158 (the repeated-token class) in [quants/experiments.md](../../../quants/experiments.md); the rules come from
[the weight-space LP note](../../research/notes/weight_space_lp.md).

Why: an attempt that runs into the 12-minute cap costs the most time per task, and nobody had checked whether those
replies loop or reason at length. A second question rides on the same pass: the release build resets the recurrent
state with a multiply by zero, so a NaN in the state would stay, and every reply after it would be one repeated token
(upstream llama.cpp [#31913](https://github.com/ggml-org/llama.cpp/pull/31913)).

## Result

**No reply ends in a repeated token, and the replies that run into the 12-minute cap are long reasoning, not loops.**
Of the 4,713 recorded replies 4,706 hold text, 2,338,024 tokens in all.

- **Class 3 is empty: 0 of 4,706.** No reply's last part ends in two equal token ids, and none ends in more than
  three equal characters (70 end in exactly three). The longest run of one character anywhere in a reply is 32, in two
  replies that ended on their own with a tool call. So no recorded reply shows the symptom of a NaN left in the
  recurrent state.
- **By the post's labels one reply loops, of 4,706:** a 355-token reply of Flint on `bugfix-javascript-grade-school`
  whose tool call says 67 tokens a second time (17 looping tokens by the post's label, every window twice), and it
  ended on its own. In the v3 release arms and the Flash-Next arms the count is 0 of 2,745 replies. The share of
  generated tokens that repeat a 16-token window at all is 2.60 % at most (per model in the table below).
- **The paper's rule trips on 82 replies, and 73 of them ended on their own,** 67 in attempts that passed. It trips
  in the reasoning in 74 and in a tool call in 8, on tasks whose reasoning lists the same values again and again:
  in the v3 release and Flash-Next arms 11 of the 36 are on `bugfix-javascript-variable-length-quantity`, 9 on
  `bugfix-python-scale-generator`, 8 on `bugfix-python-variable-length-quantity`, 4 on `bugfix-python-pig-latin`,
  2 on `bugfix-python-affine-cipher` and 1 each on `bugfix-python-list-ops` and `bugfix-python-phone-number`. A
  rescue that cut at the first trip would have cut 33 of those 36 replies before they ended on their own.
- **Of the 31 replies cut at the cap, none loops by the post's labels, 9 trip the paper's rule and none is in
  class 3.** 27 were cut in the reasoning, the longest at 16,660 tokens. At most 10.6 % of a cut reply's tokens
  repeat a 16-token window (10.2 % in the v3 release and Flash-Next arms). No request reached its `max_tokens`.

Per group and model ([by_model.tsv](by_model.tsv); the groups are described under "Files"):

| Group | Model | Replies | Generated tokens | Tokens that repeat a 16-token window | Rule 1: replies | Rule 2: replies (ended on their own) | Rule 2: tokens where it holds | Rule 2: tokens after the first trip | Rule 2: after the first trip, of task time | Class 3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 release | Qwen3.8-Flash-Next | 412 | 157,837 | 2.18 % | 0 | 8 (8) | 5.26 % | 15.18 % | 8.03 % | 0 |
| v3 release | Flint | 686 | 467,228 | 2.25 % | 0 | 16 (15) | 2.54 % | 10.37 % | 8.40 % | 0 |
| v3 release | Ornith-1.5-122B | 399 | 118,141 | 1.34 % | 0 | 0 | 0 | 0 | 0 | 0 |
| Flash-Next series | Qwen3.8-Flash-Next | 1,248 | 405,575 | 1.72 % | 0 | 12 (10) | 1.90 % | 7.25 % | 4.64 % | 0 |
| earlier | Flint | 1,590 | 1,078,648 | 2.60 % | 1 | 45 (40) | 4.05 % | 14.12 % | 10.49 % | 0 |
| earlier | Ornith-1.5-122B | 371 | 110,595 | 1.39 % | 0 | 1 (0) | 0.44 % | 1.27 % | 1.56 % | 0 |

Rule 1's loop tokens are 67 of the earlier Flint arms' 1,078,648 (0.01 %, 2.5 s of 37,308.9 s of task time) and 0
everywhere else. "After the first trip" is what a rescue at that point would cut; it is not time lost to loops,
since most of those replies went on to end and their attempts to pass.

The attempts the agent check stopped at its 12-minute cap, and the replies it cut ([runaways.tsv](runaways.tsv)):

| Group | Model | Scored attempts | Stopped at the cap | Their share of task time | Replies cut at the cap | Tokens in them | That repeat a 16-token window | Cut replies by rule 1 | By rule 2 | Class 3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 release | Qwen3.8-Flash-Next | 78 | 3 | 13.11 % | 2 | 12,350 | 384 | 0 | 0 | 0 |
| v3 release | Flint | 108 | 3 | 12.42 % | 3 | 19,594 | 596 | 0 | 1 | 0 |
| v3 release | Ornith-1.5-122B | 80 | 3 | 14.03 % | 3 | 3,511 | 124 | 0 | 0 | 0 |
| Flash-Next series | Qwen3.8-Flash-Next | 231 | 11 | 16.37 % | 8 | 35,081 | 1,684 | 0 | 2 | 0 |
| earlier | Flint | 252 | 10 | 19.31 % | 10 | 70,489 | 3,162 | 0 | 5 | 0 |
| earlier | Ornith-1.5-122B | 79 | 5 | 19.97 % | 5 | 19,478 | 483 | 0 | 1 | 0 |

An attempt stopped at the cap has no cut reply when the cap fell between two requests. The three cut replies of the
v3 release and Flash-Next arms that trip the paper's rule:

| Arm (result folder) | Task | Cut at | Rule 2 from token | Tokens where it holds | Tokens that repeat a 16-token window |
|---|---|---:|---:|---:|---:|
| Flint `rel` (`2026-11-05_release_regression/r1a-flint-q5/agent_q5stock/stock64k`) | `bugfix-python-scale-generator` | 8,817 | 5,241 | 871 | 347 |
| Flash-Next `mtp2d` (`2026-11-04_rig2_runner/s33/agent-report-d`) | `bugfix-javascript-variable-length-quantity` | 7,444 | 1,068 | 77 | 282 |
| Flash-Next `s34mtp` (`2026-11-04_rig2_runner/s34/agent-report`) | `bugfix-javascript-variable-length-quantity` | 9,457 | 3,770 | 1,482 | 898 |

The longest cut reply of the v3 release arms, 11,519 reasoning tokens of Flash-Next's `t1skip` on
`bugfix-python-scale-generator`, trips neither rule; 374 of its tokens repeat a 16-token window.

**What it decides.**
- Q1158: none of the recorded arms shows the symptom of the multiply-by-zero reset, so that bug has touched no
  agent-check result. The fix stays on the next engine merge's list.
- Q1139 and Q1140 (a loop-direction vector or weight edit): the post takes its direction from looping tokens, and
  there are 17 of them in 2,338,024. The gain is capped at the time loops take, which is 0.01 % or less.
- Q266 (a runaway guard): a repetition detector is the wrong guard here. The paper's rule trips on 73 replies
  that ended on their own and on 9 of the 31 the cap cut; on the other 22 cut replies it never trips. What the cut
  replies share is length, so the guard left to try is a reasoning budget per turn.

**Limits.** These are the 12 bug-fix tasks of the release check, at the makers' sampling settings. Both rules see
only word-for-word repetition; reasoning that goes over the same ground in new words is counted as new text.
The re-tokenized count over the server's own count (`tokens_per_server_count`) is 1.0045 and 1.0042 on Flint,
0.9493 and 0.9559 on Flash-Next and 0.9400 and 0.9445 on Ornith: the text holds no reasoning tags, tool-call wrappers
or end token, and a tool call is tokenized in a JSON form that is not the model's own.

## Method

- **Input:** the proxy's reply bodies (`proxy.jsonl.bodies.gz`: the reasoning, the answer and the tool calls of each
  request, as assembled from the stream), its per-request rows (`proxy.jsonl`) and each attempt's `result.json`. A
  proxy log is appended to across arms, so [run/arms.tsv](run/arms.tsv) names the one folder each arm is read from;
  the script stops if any recorded reply belongs to no listed arm.
- **Tokens:** each part of a reply is tokenized on its own with its model's vocabulary by llama.cpp's
  `llama-tokenize` (a CPU build, `--no-escape`), and no window crosses from one part into the next. Flint has its own
  vocabulary; Flash-Next and Ornith share one. The tokenizer's build cannot open Ornith's N2P file or a split with a
  missing shard, so Flash-Next and Ornith are tokenized through another file of the same model, and
  `loop_census.py vocab` compares the two files' tokenizer metadata: Flash-Next's is identical, Ornith's differs in
  one key (`tokenizer.ggml.normalizer.nfc`), which changes nothing for text that is already NFC.
- **Rule 1, the post's labels:** a token repeats when the 16 tokens ending at it appeared earlier in the reply. A loop
  starts at the first point after which at least 80 % of the remaining tokens repeat. The post counts as looping
  tokens the repeats at least 64 tokens into a loop, so a reply loops here when it has at least one of those. Its
  loop tokens are the repeats from the loop's start to the reply's end.
- **Rule 2, arXiv 2606.02011's loop rescue:** some 20-gram occurs at least 4 times within the last 1,024 tokens
  (occurrences may overlap). Its loop tokens are the tokens from the first position where that holds to the reply's
  end, which is what a rescue at that point would cut.
- **Class 3, one token repeated to the end:** the reply's last part ends in at least 16 copies of a unit of at most
  16 characters that is a single token of the model's vocabulary. The test is on the text, because a run of one
  character does not re-tokenize into the tokens that were generated: 64 slashes come back as one token in Flash-Next's
  vocabulary and two in Flint's. A run of 16 equal token ids at the end of the re-tokenized part counts as well.
- **Time:** a reply's loop tokens times its own milliseconds per generated token (the server's `predicted_ms` over
  `predicted_n`; for a reply cut mid-stream, the stream's duration over its tokens), over the arm's task time (the
  sum of `wall_s` of its scored attempts).
- **Runaways:** no request ever reached its `max_tokens` (32,768). The replies that did not end on their own are the
  ones the agent check cut at its 12-minute cap (`client_gone` in the proxy's row) and the ones a server failure
  broke (`server_error`); [runaways.tsv](runaways.tsv) lists both, and the last reply of every attempt stopped at a
  cap.

## Files

| File | What |
|---|---|
| [summary.md](summary.md) | the script's own tables: per group and model, and per arm |
| [by_model.tsv](by_model.tsv), [by_arm.tsv](by_arm.tsv), [by_task.tsv](by_task.tsv) | per group and model, per arm, per task: replies, tokens, replies by rule 1, rule 2, both and class 3, repeat and loop tokens and their share of generated tokens, task seconds, loop seconds and their share, attempts, runaway attempts and their seconds, cut replies and how many of them loop, the longest run at a reply's end |
| [loops.tsv](loops.tsv) | every reply that loops by rule 1 or rule 2 or is in class 3: arm, task, attempt, turn, how it ended, its tokens and how many repeat a 16-token window, where the loop starts and in which part, loop tokens and seconds, whether its attempt was stopped at the cap and passed |
| [runaways.tsv](runaways.tsv) | every reply that did not end on its own and the last reply of every attempt stopped at a cap, with the three classes and the attempt's loop seconds |
| [class3.tsv](class3.tsv) | every class 3 reply: arm, task, attempt, turn, the token, its copies, and how many later replies of the same arm are in class 3 |
| [run/loop_census.py](run/loop_census.py) | the census: `selftest` (the rules on built sequences, the tokenizer's escape handling), `vocab`, `tokenize`, `census` |
| [run/arms.tsv](run/arms.tsv) | the 80 arms: group, model, result folder, arm name |
| [run/run.sh](run/run.sh) | the run, with paths from the environment |

Columns shared by the three `by_*` tables:
- `tokens_per_server_count`: the re-tokenized count over the server's `predicted_n`, on the replies that have one;
- `repeat_tokens`: tokens that repeat a 16-token window, in a loop or not; `looping_tokens`: the post's label;
- `rule1_tokens`: the repeats from a loop's start on; `rule2_fire_tokens`: positions where rule 2 holds;
  `rule2_tokens`: tokens from its first trip to the reply's end;
- `rule1_finished`, `rule2_finished`: looping replies that ended on their own (`stop` or `tool_calls`);
  `rule2_in_passed`: rule 2 replies in attempts that passed;
- `runaway_attempts`, `runaway_s`, `runaway_time_pct`: scored attempts stopped at the 12-minute cap, their seconds
  and their share of `task_s`;
- `cut_*`: the replies cut at the cap, their tokens and how many of them fall under each rule;
- `longest_tail_same`, `longest_tail_char_run`: the longest run of one token id, and of one character, at the end
  of any reply's last part.

The groups: `v3-release` is the v3.0 release's arms (the folders
[2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md),
[2026-11-05_release_regression](../2026-11-05_release_regression/README.md),
[2026-11-06_flint_side_rig2](../2026-11-06_flint_side_rig2/README.md) and
[2026-11-06_ornith_mtp_rig2](../2026-11-06_ornith_mtp_rig2/README.md)); `flash-next-series` is Flash-Next's arms
before it, 2026-11-03 to 2026-11-05; `earlier` is Flint's and Ornith's arms of 2026-10-26 to 2026-10-28, the v2.0
release's arms and the experiments right after it. A result folder's page of the same name under `docs/data/` says
what its arms ran; the arms read from `2026-11-04_rig1_runner` and `2026-11-04_rig2_runner` are described in
[2026-11-04_mtp_redo](../2026-11-04_mtp_redo/README.md) (S33),
[2026-11-04_stacked_gains](../2026-11-04_stacked_gains/README.md) (S34),
[2026-11-04_s35_stack](../2026-11-04_s35_stack/README.md) (S35) and
[2026-11-04_adaptive_draft](../2026-11-04_adaptive_draft/README.md) (`fix2`, `adapt5`).

## Commands

On a machine with the result folders, a CPU build of llama.cpp and one GGUF file per model (CPU only, low priority,
about 5 minutes with 8 tokenizer processes, 12 s once the tokens are cached):

```bash
IN=results WORK=work OUT=out PRIVATE=private TOKENIZER=llama.cpp/build-cpu/bin/llama-tokenize \
VOCAB_FLASH_NEXT=Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00001-of-00005.gguf \
VOCAB_FLINT=flint-pascal-Q8.gguf \
VOCAB_ORNITH=Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf \
SAME_FLASH_NEXT=Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf \
SAME_ORNITH=Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
bash docs/data/2026-11-09_loop_census/run/run.sh
```

`OUT` is this folder's tables. `PRIVATE` gets `replies.tsv`, one row per reply with every measure; it is not
published, and neither are the replies.

## Provenance and licence

The replies are the models' output on the agent check's 12 bug-fix tasks (Exercism exercises, MIT; task ids as in
[bench/agent_check](../../../bench/agent_check/README.md)). The tables hold counts, positions and task ids only. The
script is under this repository's licence.
