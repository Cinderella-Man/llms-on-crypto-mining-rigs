# Reasoning census of the agent check's replies (Q1152; 2026-11-09)

What this is: a count of how much of what a model generates in the agent check
([bench/agent_check](../../../bench/agent_check/README.md)) is reasoning, how long the reasoning of one reply gets, and
how many reflection words ("Wait", "Hmm", "Actually") it holds. It covers the replies the
[loop census](../2026-11-09_loop_census/README.md) read: 4,713 replies in 80 arms of Qwen3.8-Flash-Next, Flint and
Ornith-1.5-122B, recorded on the rig from 2026-10-26 to 2026-11-06. Registry row Q1152 in
[quants/experiments.md](../../../quants/experiments.md); the reflection words and their counter are Q94's
([bench/reasoning_length](../../../bench/reasoning_length/README.md)).

Why: on the agent check most of the rig's time goes into decoding, so the number of tokens a task takes matters as
much as tokens per second. For Flint, Q94 counted 2.4 reflection words per 1,000 reasoning tokens in single requests,
below the ~5 per 1,000 where a logit bias against those words pays. Nobody had counted them for Flash-Next, or
measured what share of its tokens is reasoning.

## Result

**Flash-Next reasons a lot and reflects little. In the v3 release arms 79.42 % of its generated tokens are reasoning,
with 2.74 reflection words per 1,000 reasoning tokens (344 in 125,356), beside Flint's 2.4 in Q94 and below the ~5
line. The tokens sit in a few long replies: the longest 10 % of replies hold 63.20 % of all generated tokens.**

Per group and model ([by_model.tsv](by_model.tsv); the groups are the loop census's, described under "Files"):

| Group | Model | Replies | Generated tokens | Reasoning | Answer | Tool calls | Reasoning, of the server's count | Decoding, of task time | Reasoning, of task time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v3 release | Qwen3.8-Flash-Next | 412 | 157,837 | 79.42 % | 9.00 % | 11.58 % | 74.25 % | 73.11 % | 55.96 % |
| Flash-Next series | Qwen3.8-Flash-Next | 1,248 | 405,575 | 75.21 % | 10.88 % | 13.91 % | 69.17 % | 69.14 % | 49.92 % |
| v3 release | Flint | 686 | 467,228 | 86.65 % | 6.37 % | 6.98 % | 86.43 % | 83.85 % | 72.88 % |
| earlier | Flint | 1,590 | 1,078,648 | 86.28 % | 6.28 % | 7.43 % | 85.71 % | 80.20 % | 68.83 % |
| v3 release | Ornith-1.5-122B | 399 | 118,141 | 72.90 % | 12.13 % | 14.97 % | 68.20 % | 67.32 % | 47.42 % |
| earlier | Ornith-1.5-122B | 371 | 110,595 | 72.80 % | 12.78 % | 14.42 % | 62.97 % | 77.50 % | 55.11 % |

"Reasoning, of the server's count" divides by the server's own token count, which exists only for the replies that
ended and also holds the reasoning tags, the tool-call wrappers and the end token. "Decoding" is the time the server
spent generating the replies; the rest of a task's time is prompt processing and the agent's own tool runs.

### 1. Reasoning by turn type

Flash-Next, v3 release arms ([by_model_turn.tsv](by_model_turn.tsv); per arm in [by_arm_turn.tsv](by_arm_turn.tsv)):

| Turn type | Replies | Generated tokens | Of all generated tokens | Reasoning share | Of all reasoning tokens | Reasoning per reply: median | 90th percentile | Largest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| First turn | 78 | 3,961 | 2.51 % | 18.40 % | 0.58 % | 8 | 13 | 29 |
| After a tool result, with another tool call | 257 | 125,298 | 79.38 % | 86.38 % | 86.34 % | 78 | 924 | 11,953 |
| The final answer | 75 | 16,228 | 10.28 % | 24.91 % | 3.22 % | 44 | 107 | 226 |
| Unfinished (cut at the 12-minute cap) | 2 | 12,350 | 7.82 % | 100.00 % | 9.85 % | 6,175 | 11,519 | 11,519 |

- **The first turn is tool calls** (`bash` and `read`): 8 reasoning tokens at the median.
- **The second turn holds the reasoning.** It answers the first tool results, and its 78 replies hold 81,546
  reasoning tokens: 65.05 % of all reasoning tokens and 37.56 % of task time. Its median is 324.5 reasoning
  tokens and its 90th percentile 2,616 ([by_turn_number.tsv](by_turn_number.tsv)). In the Flash-Next series the second
  turn holds 65.71 % of the reasoning tokens (200,444 of 305,031).
- **The final answer is mostly answer text:** 24.91 % reasoning, 3.22 % of all reasoning tokens.
- Per arm, the reasoning share of the seven release arms is 78.26 % (`t1base`), 85.37 % (`t1skip`), 79.95 %
  (`t1neg`), 80.07 % (`t1swift`), 78.05 % (`stockq2`), 75.62 % (`stockq2t`) and 72.73 % (`stockpmix`)
  ([by_arm.tsv](by_arm.tsv)).

### 2. Reflection words

Q94's list and counter, per 1,000 reasoning tokens ([by_model.tsv](by_model.tsv), [words.tsv](words.tsv)):

| Group | Model | Reasoning tokens | Reflection words | Per 1,000 | At the start of a sentence, per 1,000 | Contrast words per 1,000 | Reasoning in sentences a reflection word opens | In paragraphs it opens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v3 release | Qwen3.8-Flash-Next | 125,356 | 344 | 2.74 | 2.07 | 0.96 | 2.87 % | 11.74 % |
| Flash-Next series | Qwen3.8-Flash-Next | 305,031 | 818 | 2.68 | 1.99 | 0.90 | 2.51 % | 9.32 % |
| v3 release | Flint | 404,874 | 4,213 | 10.41 | 8.05 | 6.44 | 9.09 % | 22.66 % |
| earlier | Flint | 930,676 | 9,888 | 10.62 | 8.08 | 6.55 | 8.89 % | 23.09 % |
| v3 release | Ornith-1.5-122B | 86,124 | 770 | 8.94 | 6.04 | 8.27 | 6.71 % | 17.46 % |
| earlier | Ornith-1.5-122B | 80,517 | 685 | 8.51 | 5.56 | 7.12 | 6.19 % | 16.88 % |
| Q94: single requests | Flint | 43,373 | 106 | 2.4 | 1.8 | 10.5 | | |

- **Flash-Next is below the line in every release arm:** 2.56 (`t1base`), 2.36 (`t1skip`), 2.48 (`t1neg`), 3.03
  (`t1swift`), 2.87 (`stockq2`), 3.52 (`stockq2t`) and 2.92 (`stockpmix`) per 1,000.
- **Its 344 reflection words** are "hmm" 147, "verify" 91, "wait" 66, "actually" 28, "ah" 6, "oh" 5 and
  "alternatively" 1. Only 20 of the 91 "verify" open a sentence; the other 71 stand inside one.
- **By turn type the rate stays low:** 2.78 per 1,000 after a tool result, 4.95 in the final answer (20 words in
  4,042 reasoning tokens), 1.86 in the two cut replies, 0 in the first turn.
- **Flint and Ornith reflect far more often here:** 10.41 and 8.94 per 1,000 in the v3 release arms, both above
  the ~5 line. Flint's words are "hmm" 1,674, "wait" 1,506 and "actually" 831. Q94 saw the same in the four
  agent turns of its transcripts (12.2 per 1,000) and 2.4 in its single requests; the agent check confirms the agent
  figure on 404,874 reasoning tokens.
- **Its sentences open with plan words.** The most frequent openers of Flash-Next's reasoning sentences are "So",
  "Let's", "Let", "The" and "Hmm" (2.03 % of sentences); Flint's are "Hmm" (4.76 %), "So", "For", "The", "But",
  "Let" and "Wait" ([openers.tsv](openers.tsv)).

The last two columns count the reasoning in sentences and paragraphs that a listed word opens. Q94's own rule also
counts those that open with a phrase such as "let me check" or "let me re…", and that prefix also matches "let me
read"; by that rule the shares are 3.52 % and 14.71 % for Flash-Next's v3 release arms
(`reflection_sentence_pct`, `reflection_paragraph_pct`), and were 2.5 % and 3.8 % for Flint's single requests in Q94.

### 3. Reasoning per reply

| Group | Model | Replies | Median | 90th percentile | Largest | Longest 10 % of replies | Their share of generated tokens | Share of reasoning tokens in the 10 % with the longest reasoning |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v3 release | Qwen3.8-Flash-Next | 412 | 48.5 | 550 | 11,953 | 42 | 63.20 % | 77.80 % |
| Flash-Next series | Qwen3.8-Flash-Next | 1,248 | 47 | 572 | 10,361 | 125 | 55.83 % | 72.00 % |
| v3 release | Flint | 686 | 8.5 | 1,499 | 14,334 | 69 | 68.85 % | 77.60 % |
| earlier | Flint | 1,590 | 8 | 1,446 | 18,086 | 159 | 68.44 % | 76.91 % |
| v3 release | Ornith-1.5-122B | 399 | 45 | 513 | 4,753 | 40 | 51.85 % | 65.48 % |
| earlier | Ornith-1.5-122B | 371 | 39 | 503 | 5,598 | 38 | 54.33 % | 69.86 % |

- **Half of Flash-Next's replies reason for under 50 tokens; one in ten for 550 or more.**
- **The 42 longest replies of the v3 release arms** hold 99,753 of the 157,837 generated tokens. 40 of them are
  replies after a tool result that ended with a tool call and 2 were cut at the cap; 32 are second turns; 38 sit in
  attempts that passed ([longest.tsv](longest.tsv)).
- **Reasoning piles up on a few tasks.** It is 94.09 % of the generated tokens on
  `bugfix-javascript-variable-length-quantity` (7,090 tokens per attempt), 90.88 % on
  `bugfix-python-scale-generator` (4,075) and 85.89 % on `bugfix-python-variable-length-quantity` (4,031); on
  `bugfix-javascript-grade-school` it is 36.68 % (581 per attempt) ([by_task.tsv](by_task.tsv)).
- **Flint writes no reasoning in 288 of its 686 replies** of the v3 release arms. Its median over the 398 that have
  reasoning is 235 tokens.
- **Per scored attempt** Flash-Next generates 2,024 tokens in the v3 release arms (1,607 of them reasoning), Flint
  4,326 (3,749) and Ornith 1,477 (1,077).

**What it decides.**
- Q1152, the reflection half: a logit bias against reflection words has little to remove from Flash-Next's reasoning.
  The rate is 2.74 per 1,000, and the sentences such a word opens are 2.87 % of the reasoning.
- Q1152, where the tokens are: in the second turn's long reasoning, most of all on the two
  `variable-length-quantity` tasks and `scale-generator`. A lever on tokens per task has to shorten those replies: a
  reasoning budget per turn (Q266), a lower reasoning effort (Q1147), or the first-finish race and early-stop rules
  that are the rest of Q1152.
- Q94 for agent work (Q1174): Flint's and Ornith's reasoning in the agent check is above the ~5 line, so the bias
  Q94 dropped on single requests is worth a test on those two models in agent sessions.

**Limits.** These are the 12 bug-fix tasks of the release check at the makers' sampling settings, Flash-Next at its
chat template's default reasoning effort (Q1147 tests lower ones). The counts are on re-tokenized text: the
reasoning's token ids give its text back in all 3,755 replies with reasoning, and the whole reply's count over the
server's is 0.9493 and 0.9559 on Flash-Next's two groups (the loop census's `tokens_per_server_count`), because a
tool call is tokenized in a JSON form that is not the model's own. The reflection list is English and fixed; a model
that re-checks in other words is counted as not reflecting, which is why the sentence openers are listed. Not done
here: the price of a first-finish race (Q93) and of early-stop or consensus rules, which need the transcripts of
Q1146's lane.

## Method

- **Input, tokens and arms** are the loop census's: the proxy's reply bodies, its per-request rows and each
  attempt's `result.json`; each part of a reply (the reasoning, the answer, each tool call) tokenized on its own with
  its model's vocabulary; the 80 arms of its `run/arms.tsv`. The script loads that census's reader and token cache
  and checks that its replies and tokens per group and model equal that census's `by_model.tsv`.
- **Turn type**, from the request's message roles and how the reply ended:
  - `first`: the request holds the system prompt and the task, and no reply yet;
  - `after_tool`: the request ends in a tool result and the reply makes another tool call;
  - `final`: the request ends in a tool result and the reply makes no tool call, which ends the attempt;
  - `unfinished`: the agent check cut the reply at its 12-minute cap, or a server failure broke it, at any turn.

  Every one of the 4,713 replies falls in one of the four.
- **Reflection words:** `count_block()` of
  [bench/reasoning_length/reflections.py](../../../bench/reasoning_length/reflections.py) gets the reasoning's token
  ids and the vocabulary's token texts from the GGUF header. It counts word-initial tokens whose text is one of
  Q94's reflection words (wait, hmm, actually, alternatively, oh, ah, double-check, recheck, verify, reconsider,
  rethink) or contrast words (but, however, maybe, perhaps), per 1,000 reasoning tokens. The run first reproduces
  Q94's published counts from Q51's transcripts with the same file, byte for byte.
- **Distribution:** over the replies that hold text. The 90th percentile is the nearest rank (the smallest value
  that 90 % of the replies do not exceed). "The longest 10 %" are the replies with the most generated tokens,
  rounded up to whole replies.
- **Time:** a reply's decode time is the server's `predicted_ms` (for a reply cut mid-stream, the stream's
  duration); its reasoning's part is its reasoning tokens times its own milliseconds per generated token. Shares
  are of the arm's task time, the sum of `wall_s` of its scored attempts.

## Files

| File | What |
|---|---|
| [summary.md](summary.md) | the script's own tables: per group and model, and per turn type |
| [by_model.tsv](by_model.tsv), [by_arm.tsv](by_arm.tsv), [by_task.tsv](by_task.tsv) | per group and model, per arm, per task: every column below |
| [by_model_turn.tsv](by_model_turn.tsv), [by_arm_turn.tsv](by_arm_turn.tsv) | the same per turn type |
| [by_turn_number.tsv](by_turn_number.tsv) | the same per turn of the attempt (1 to 5, and 6 or later) |
| [words.tsv](words.tsv) | per group and model: each reflection word, contrast word and phrase of Q94's list, its count, how often it opens a sentence, and its rate per 1,000 reasoning tokens |
| [tenths.tsv](tenths.tsv) | reflection words by tenth of the reasoning they fall in |
| [openers.tsv](openers.tsv) | the 40 most frequent first words of a reasoning sentence per group and model, with their share of sentences and whether Q94's list holds them |
| [longest.tsv](longest.tsv) | every reply in the longest 10 % of its group and model: arm, task, attempt, turn, turn type, how it ended, tokens by part, reflection words, whether its attempt was stopped at the cap and passed |
| [run/reasoning_census.py](run/reasoning_census.py) | the census: `selftest` (turn types, percentile and share on built values, the counter on a built block, the whole census on a built arm) and `census` |
| [run/run.sh](run/run.sh) | the run, with paths from the environment |

Columns of the `by_*` tables:
- `replies`, `replies_with_reasoning`, `tokens`, `reasoning_tokens`, `content_tokens`, `tool_tokens` and the three
  parts' shares of `tokens`; `of_all_tokens_pct` and `of_all_reasoning_pct`: a turn row's share of its group's (or
  arm's) tokens and reasoning tokens;
- `server_tokens`, `reasoning_pct_of_server`: the server's count over the replies that have one, and their reasoning
  tokens over it;
- `reflection_*`, `contrast_*`: words, words that open a sentence, and both per 1,000 reasoning tokens;
  `reflection_sentence_*` and `reflection_paragraph_*`: sentences and paragraphs that a reflection word or one of
  Q94's opening phrases opens, their reasoning tokens and share; `reflection_word_*`: those that a listed word opens;
- `reasoning_median`, `reasoning_p90`, `reasoning_max`, `reasoning_median_nonzero` (over the replies that have
  reasoning) and the same for `tokens`;
- `top10_replies`, `top10_token_pct`, `top10_reasoning_pct`: the row's own longest 10 % of replies, their share of
  its generated tokens, and the share of its reasoning tokens in the 10 % of replies with the longest reasoning;
- `group_top10_*`: how many of the row's replies are in the longest 10 % of their group and model, their tokens, how
  many were cut and how many sit in attempts that passed;
- `attempts`, `passed`, `tokens_per_attempt`, `reasoning_per_attempt`: scored attempts and tokens per scored
  attempt (a turn row uses its group's or arm's attempts, so the turn rows add up);
- `task_s`, `generated_s`, `generated_time_pct`, `reasoning_s`, `reasoning_time_pct`;
- `reasoning_roundtrip`: replies whose reasoning token ids decode to the reasoning text exactly.

The groups: `v3-release` is the v3.0 release's arms, `flash-next-series` Flash-Next's arms before it (2026-11-03 to
2026-11-05) and `earlier` Flint's and Ornith's arms of 2026-10-26 to 2026-10-28; the loop census's README names each
group's result folders.

## Commands

From a checkout of this repository, on a machine with the result folders, a CPU build of llama.cpp and one GGUF file
per model (CPU only, low priority, 7 s with the loop census's token cache; that census's README gives the time
without it):

```bash
IN=results WORK=work OUT=out PRIVATE=private TOKENIZER=llama.cpp/build-cpu/bin/llama-tokenize \
VOCAB_FLASH_NEXT=Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00001-of-00005.gguf \
VOCAB_FLINT=flint-pascal-Q8.gguf \
VOCAB_ORNITH=Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf \
bash docs/data/2026-11-09_reasoning_census/run/run.sh
```

`OUT` is this folder's tables. `PRIVATE` gets `replies.tsv`, one row per reply with every measure; it is not
published, and neither are the replies. The run was made on 2026-11-09 with `reflections.py` as this change leaves
it.

## Provenance and licence

The replies are the models' output on the agent check's 12 bug-fix tasks (Exercism exercises, MIT; task ids as in
[bench/agent_check](../../../bench/agent_check/README.md)). The tables hold counts, task ids and single words with
their counts. The scripts are under this repository's licence.
