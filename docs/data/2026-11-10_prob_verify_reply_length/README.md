# Does probabilistic verification lengthen the replies? The source's draw, and 240 replies to 12 fixed prompts (Q1184; 2026-11-10)

What this is: a check of whether the Miner Llama v3 release preset
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next, Swift Q2_0,
an MTP draft of three tokens) writes longer replies when it verifies its drafts by rejection sampling:

- **`pv0`** is the preset: the draft takes its first choice, and the main model keeps a drafted token only when its
  own sample is the same token;
- **`pv1`** adds `--spec-draft-sampling probabilistic --spec-draft-p-min 0`: the draft samples its tokens, and the
  main model accepts a drafted token with probability min(1, p/q).

Registry row Q1184 in [quants/experiments.md](../../../quants/experiments.md), from Q1183 and Q731.

Why: on the agent check `pv1`'s two runs generated 35,406 and 36,683 tokens against the preset's 34,945 and 21,936,
and the tasks' second replies, which hold the long reasoning, were 25,457 and 27,222 tokens against 11,826 and 10,783
([2026-11-10_prob_verify_agent_check](../2026-11-10_prob_verify_agent_check/README.md), Q1183). Two live runs an arm
cannot tell that from chance: no seed is sent, every run writes its own history, and one task's reply carried most of
the difference. Rejection sampling keeps the main model's distribution only if the token after a rejection is drawn
from the residual max(0, p − q), so a wrong draw there could change what the model writes. `pv1` yields 7.74 % more
tokens a step ([2026-11-10_prob_verify_acceptance](../2026-11-10_prob_verify_acceptance/README.md), Q1173); replies
a quarter longer would cost more than that gains.

Two steps: the source read for what is drawn, and the same 12 prompts sent with the same five seeds to two loads of
each arm, with the reply limit the agent itself sends (32,768 tokens). Reply lengths depend on the models, the
sampler and the text, not on the card, so they were counted on a machine with newer cards. No time is read or
reported here.

## Result

**The draw after a rejection is the residual, and the replies are not longer beyond what two loads of one arm
differ by. `pv1`'s two loads generated 95,670 and 72,564 tokens for the 60 replies, the preset's 66,802 and 91,163:
+6.50 % between the arms' means, where the preset's own two loads are 36.47 % apart and `pv1`'s 24.15 %. The typical
reply is no longer: medians 378.5 and 338.0 tokens against 444.0 and 348.0, and `pv1` is the shorter one in 34 of
the 60 pairs. No reply reached the limit. So by the row's rule nothing here keeps `pv1` out of the preset, and the
time of a step on the rig decides (Q731).**

What 240 replies cannot settle is the far tail: the three longest replies are `pv1`'s (19,994, 16,009 and 13,639
tokens against the preset's longest 11,619), and replies of 10,000 tokens or more are 4 of 120 with `pv1` against 1
of 120. That is its own row, Q1188.

### Every load

| Load | Card set, order | Replies | Generated | Median | Mean | Longest | 4,000 or more | Share | At the limit | Reasoning | Answer | Reasoning share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pv0-a` | a, first | 60 | 66,802 | 444.0 | 1,113.4 | 8,619 | 3 | 0.0500 | 0 | 63,812 | 2,990 | 0.9552 |
| `pv0-b` | b, second | 60 | 91,163 | 348.0 | 1,519.4 | 11,619 | 7 | 0.1167 | 0 | 87,630 | 3,533 | 0.9612 |
| `pv1-a` | a, second | 60 | 95,670 | 378.5 | 1,594.5 | 19,994 | 6 | 0.1000 | 0 | 92,761 | 2,909 | 0.9696 |
| `pv1-b` | b, first | 60 | 72,564 | 338.0 | 1,209.4 | 13,639 | 3 | 0.0500 | 0 | 69,337 | 3,227 | 0.9555 |

- Every reply ended in a tool call, none at the 32,768-token limit, and none holds text beside the reasoning and
  the call. The answer (the call and the markup) is 48.5 to 58.9 tokens in a load's mean; the length is the
  reasoning.
- A load's sum is a few long replies: `pv1-a`'s two longest are 36,003 of its 95,670 tokens. Replies of 1,000
  tokens or more: 17 and 23 with the preset, 20 and 19 with `pv1`; of 2,000 or more: 10 and 12 against 9 and 9.
- Two loads with the same prompts and seeds wrote the same reply in 4 of 60 pairs (the preset's two), 3 of 60
  (`pv1`'s two) and 1, 2, 1 and 2 of 60 (a `pv1` load and a preset load).
- No server log holds an error line.

[loads.tsv](loads.tsv); every reply in [replies.tsv](replies.tsv).

### `pv1` against the preset, beside two loads of one arm

Paired by prompt and seed index, 60 pairs over 12 prompts. With two loads on a side, a pair's value is their mean.
The interval is a 95 % bootstrap over prompts. The ratio is the geometric mean of the pairs' ratios (of length + 1),
which a single long reply moves little. [compare.tsv](compare.tsv) has the same for the reasoning and the answer.

| Length in tokens | Mean, first | Mean, second | Difference | % | 95 % over prompts | Ratio (95 %) | First longer, shorter, equal |
|---|---:|---:|---:|---:|---|---|---|
| **`pv1` against `pv0`, both loads each** | 1,402.0 | 1,316.4 | **+85.6** | **+6.50** | −498.7 to +761.7 | 0.9176 (0.7739 to 1.0764) | 25, 34, 1 |
| **`pv0-b` against `pv0-a`: two loads of the preset** | 1,519.4 | 1,113.4 | **+406.0** | **+36.47** | +107.5 to +780.5 | 1.1237 (0.9855 to 1.2887) | 34, 22, 4 |
| **`pv1-b` against `pv1-a`: two loads of `pv1`** | 1,209.4 | 1,594.5 | **−385.1** | **−24.15** | −1,494.1 to +272.4 | 0.9796 (0.8249 to 1.1254) | 30, 27, 3 |
| `pv1-a` against `pv0-a` (card set a) | 1,594.5 | 1,113.4 | +481.1 | +43.21 | −387.1 to +1,793.0 | 0.9882 (0.7943 to 1.2763) | 28, 31, 1 |
| `pv1-b` against `pv0-b` (card set b) | 1,209.4 | 1,519.4 | −310.0 | −20.40 | −787.4 to +109.3 | 0.8615 (0.7369 to 0.9920) | 27, 31, 2 |
| `pv1-a` against `pv0-b` | 1,594.5 | 1,519.4 | +75.1 | +4.94 | −796.7 to +1,183.5 | 0.8794 (0.6965 to 1.0740) | 20, 38, 2 |
| `pv1-b` against `pv0-a` | 1,209.4 | 1,113.4 | +96.0 | +8.63 | −216.0 to +437.8 | 0.9680 (0.8764 to 1.0740) | 27, 30, 3 |

- **The difference follows the loads, not the arm.** On card set a `pv1` is 43.21 % longer than the preset, on set
  b 20.40 % shorter. Each set's second load is the longer one (`pv1-a`, `pv0-b`), whichever arm it is; two sets
  cannot say whether that is chance.
- **The arms' +85.6 tokens a reply is about a fifth of what two loads of one arm differ by** (406.0 and 385.1).
- **The interval over prompts is too narrow a yardstick here:** for the preset's own two loads it leaves out zero
  (+107.5 to +780.5). A load is one draw of where the long replies fall, so the spread between loads is the
  measure, as the row's rule has it.
- **The reasoning carries all of it:** +88.8 tokens (+7.04 %; −490.3 to +759.8) between the arms; the answer −3.2
  tokens (−10.9 to +1.2).
- A lengthening the size of Q1183's second replies (26,339.5 against 11,304.5 tokens in the means, +133 %) is far
  outside the interval, whose upper end is +57.9 % of the preset's mean.

### Per prompt: the mean of five replies

| Prompt (the task's second request) | Prompt tokens | `pv0-a` | `pv0-b` | `pv1-a` | `pv1-b` |
|---|---:|---:|---:|---:|---:|
| `bugfix-javascript-complex-numbers` | 3,236 | 51.0 | 39.4 | 38.2 | 38.8 |
| `bugfix-javascript-grade-school` | 3,992 | 118.2 | 114.6 | 109.6 | 108.6 |
| `bugfix-javascript-phone-number` | 3,898 | 305.6 | 239.0 | 200.4 | 253.6 |
| `bugfix-javascript-space-age` | 3,278 | 212.6 | 169.6 | 187.8 | 201.8 |
| `bugfix-javascript-triangle` | 3,401 | 205.0 | 195.2 | 207.8 | 219.2 |
| `bugfix-javascript-variable-length-quantity` | 4,262 | 4,544.4 | 4,880.0 | 5,056.0 | 5,968.8 |
| `bugfix-python-affine-cipher` | 4,157 | 677.2 | 1,352.4 | 798.6 | 755.6 |
| `bugfix-python-list-ops` | 3,597 | 801.0 | 1,090.8 | 964.4 | 778.8 |
| `bugfix-python-phone-number` | 3,323 | 248.4 | 280.0 | 218.4 | 180.0 |
| `bugfix-python-pig-latin` | 3,430 | 1,128.2 | 1,598.6 | 1,069.4 | 1,206.2 |
| `bugfix-python-scale-generator` | 4,278 | 2,051.4 | 3,955.4 | 9,043.6 | 2,963.4 |
| `bugfix-python-variable-length-quantity` | 4,456 | 3,017.4 | 4,317.6 | 1,239.8 | 1,838.0 |

- Q1183's task with the long `pv1` replies (`bugfix-javascript-variable-length-quantity`: 17,785 and 9,445 tokens
  against 2,872 and 4,071, each run on its own history) is long in every load here: on the same history the four
  loads' means are within 31.34 % of the lowest.
- One prompt goes each way: `bugfix-python-scale-generator` is longest in `pv1-a` (9,043.6, with two replies of
  19,994 and 16,009 tokens; 2,963.4 in `pv1-b`), and `bugfix-python-variable-length-quantity` is longer in both of
  the preset's loads (3,017.4 and 4,317.6 against 1,239.8 and 1,838.0).

The five lengths of every prompt and load are in [tasks.tsv](tasks.tsv).

### The counters beside the lengths

| Load | Accepted of drafted | Share accepted | Tokens a step | Replies with a grammar rejection | Tokens sampled beyond the generated |
|---|---|---:|---:|---:|---:|
| `pv0-a` | 45,721 of 63,081 | 0.7248 | 3.169 | 0 | 6 |
| `pv0-b` | 62,326 of 86,367 | 0.7216 | 3.161 | 0 | 12 |
| `pv1-a` | 65,973 of 89,130 | 0.7402 | 3.222 | 33 | 73 |
| `pv1-b` | 50,543 of 66,090 | 0.7648 | 3.295 | 29 | 69 |

- **Tokens a step on these replies:** 3.222 and 3.295 with `pv1` against 3.169 and 3.161, +0.093 (+2.95 %) between
  the means, with `pv1`'s two loads 0.073 apart. That is less than the +7.74 % of Q1173's replay of a whole session.
  These replies are 95 to 97 % reasoning.
- **The grammar rejections are the end of the turn** (step 1, below): with `pv1` 33 and 29 of the 60 replies have
  exactly one, and each of those replies sampled two tokens more than it generated; no other reply of any load
  sampled more than one.

### What a first-finish race would save on these replies (Q1195)

Each prompt has ten samples per arm (five seeds in two loads), so the data prices a race: several samples of one
request run side by side and the first to finish is kept (Q93). [race_price.md](race_price.md) takes every subset of
N of a prompt's samples and the shortest reply in each.

| Arm | One sample | First of 2 | First of 3 | First of 4 | First of 5 |
|---|---:|---:|---:|---:|---:|
| `pv0` (the preset) | 15,796 tokens | 9,814 (0.621) | 7,860 (0.498) | 6,891 (0.436) | 6,296 (0.399) |
| `pv1` | 16,823 | 9,979 (0.593) | 7,944 (0.472) | 7,023 (0.417) | 6,454 (0.384) |

Expected tokens for the 12 prompts, with the share of one sample in brackets.

- **The tail is where the saving is.** On the six longest prompts a prompt's longest sample is 3.8 to 17.1 times
  its shortest (20 samples each, both arms together), and the mean is up to 1.86 times the median.
- **Across machines the saving is the share above:** each machine runs one sample at its own speed, so two machines
  finish a reply in est. 0.62 of the tokens and five in est. 0.40, before the cost of the losers reading the
  winner's reply as a prompt.
- **In slots of one machine it does not pay on this engine (est.):** two sequences in one pass take 77.9 and 78.4 ms
  on the release build against 51.1 and 52.3 ms for one token alone, so a race of 2 costs est. 0.621 × 1.52 = 0.95
  of one sample's time, and a race of 4 (101.6 and 106.3 ms a pass) est. 0.436 × 1.99 = 0.87, without a draft.
- **A cap instead of a race:** counting every reply at most at 4,000 tokens keeps 0.744 of all tokens and touches 19
  of the 240 replies; at 2,000 it keeps 0.563 and touches 40. What the model writes after a forced end of its
  reasoning is not in these numbers.
- **Limits:** 12 prompts, all second replies of bug-fix tasks, whole-reply lengths. Nothing here says whether the
  first finisher's answer is as good; Q93's offline gate found no loss on another model and benchmark.

## Step 1: what the source draws

Read in the fork's release build (`1551f5fe1`, branch `pascal-p106-v3-b12076-ladder5`).

**After a rejection the token is drawn from the residual, norm(max(0, p − q)), not from p.** The places:

- **The draft's token: `common_speculative_impl_draft_mtp::draft()`, `common/speculative.cpp`.** With the flag and a
  temperature above 0 (`server_slot::use_spec_rejection()`, `tools/server/server-context.cpp`) the server hands the
  drafter a place for its candidates (`result_q`). `spec_retune()` then rebuilds the draft's sampler as top-k 10,
  the request's temperature, and a random draw seeded with the request's seed mixed with a constant. Each drafting
  pass calls `common_sampler_sample()` on the draft's context and takes the token it drew. The draw runs on the
  host: the draft's device chain holds only the top-k 10. The 10 candidates with their probabilities q go to the
  verifier with the token. The draft's sampler is not reset between steps, so its random stream runs on. Without
  the flag the drafter takes its first candidate and passes no q.
- **The draft's candidates.** The draft's head reads a shortlist of the vocabulary (`LLAMA_DRAFT_SHORTLIST`). Its
  sampler runs the same check as any other (`common_sampler_head_bound()`, `common/sampling.cpp`): a shortlist row
  stands only if its bound rules out every token outside the list, otherwise the full head's row replaces it. For
  the verification this does not matter: rejection sampling is exact for any q, as long as the token was drawn from
  the q that is reported, and the candidate array passed on is the one the draw was made from.
- **The accept test and the residual: `common_sampler_sample_and_accept_n_rejection()`, `common/sampling.cpp`**,
  called from the verify step in `tools/server/server-context.cpp` when the request's temperature is above 0 and
  the drafter filled q. For each drafted token in turn:
  1. `common_sampler_sample()` runs the main sampler on that row and leaves the main model's distribution p in the
     candidate array. On this line the sampler chain runs on the device (top-k 20, top-p 0.95, temperature, the
     draw), and the array holds the device's candidates with their probabilities after the whole chain
     (`set_logits()` reads `llama_get_sampled_probs_ith()`; the device's draw is made from the same probabilities,
     `llama_sampler_dist_backend_apply()` in `src/llama-sampler.cpp`). So p is over at most 20 tokens, with 0 for
     those top-p cut.
  2. While the tools' grammar constrains the reply (not during the reasoning: `grammar_should_apply()`), a
     candidate the grammar refuses gets p = 0 and the rest is rescaled to sum to 1.
  3. The drafted token x is accepted if p(x) ≥ q(x), or else with probability p(x) / q(x). A drafted token outside
     the main model's candidates has p = 0 and is always rejected.
  4. On a rejection the token is drawn from the residual: every main-model candidate k with weight
     max(0, p(k) − q(k)), where a candidate outside the draft's 10 has q = 0 and keeps all of p. The step ends
     there. Only if the residual sums to 0 does the main sampler's own token stand in.
  5. If every drafted token is accepted, the token after them is the main sampler's own draw.
- **The random numbers.** The accept test and the residual draw use the main sampler's own generator
  (`common_sampler::rng`), seeded from the chain's seed mixed with a second constant, so they are independent of the
  draft's draws.

Two things in the same function differ from the preset's path. Neither touches the reasoning, and neither is a
wrong draw:

- **Under an active grammar the two arms resample differently.** The preset draws from the chain and, if the
  grammar refuses the token, draws again from the whole vocabulary with the grammar applied first. `pv1` rescales
  the chain's candidates that the grammar allows. The two agree whenever the grammar allows every candidate.
- **No stop at the end of a turn.** The preset's path stops verifying at an accepted end-of-turn token
  (`common_sampler_sample_and_accept_n()`: "do not accept draft tokens after an EOG"). The rejection path goes on
  to the drafted tokens behind it. They are never output, since the server ends the reply at the end-of-turn token,
  but the main sampler's token for such a row is checked against the grammar, fails, and is drawn again from the
  whole vocabulary on the host. This is the likely source of the grammar rejections after a device draw that the
  logs count with `pv1` (Q1173: 14 a load against 1; Q1183: 22 and 26 a run against 4 and 2). The loads here
  agree with it: 33 and 29 of `pv1`'s 60 replies have exactly one such rejection and sampled two tokens more than
  they generated, the preset's replies none. It costs one pass over the vocabulary with the grammar on the host at
  the end of a reply, which matters as time on a rig's processor. Adding the stop is registry row Q1187.

## Step 2: the method of the loads

- **Prompts:** the second request of each of the 12 bug-fix tasks, as the first preset run of Q1183 recorded it
  (`pv0-1`: the system prompt, the task, the agent's first reply with its tool calls, and their results; 3,236 to
  4,456 prompt tokens; seven prompts hold two tool results, five hold one). In a live session this is the reply
  that holds the long reasoning. The request is sent as recorded (messages, tools, the makers' sampling:
  temperature 1.0, top-p 0.95, top-k 20), with the agent's own `max_tokens` of 32,768.
- **Seeds:** five per prompt, the same in every load ([replay.py](../../../bench/agent_check/replay.py)
  `--requests 1 --repeat 5 --reply-limit request --count-parts`). A reply is paired across loads by prompt and seed
  index. The same seed does not give the same text on this line (Q1173), so the pairing removes the prompt's
  effect, not the draw's.
- **Loads:** each arm twice, on two sets of three cards side by side, in opposite order: set a `pv0` then `pv1`,
  set b `pv1` then `pv0`. Every load is a fresh server on the preset's line without `-ts`, `-otd` and
  `GGML_CUDA_KEEPALIVE` (the nine-card split and the rig's clock hold), plus `--metrics --no-webui`, one slot. A
  prompt's five replies run in a row, so the first reads the whole prompt and the other four its cache.
- **Lengths:** a reply's length is the server's count of generated tokens (`predicted_n`). The reasoning's tokens
  are the reasoning text tokenized again by the server (`/tokenize`); the answer is the rest (the text, the tool
  call and the markup).
- **The comparison:** the difference of the mean length between loads, over the pairs, with a 95 % bootstrap
  interval over the 12 prompts (prompts resampled, each with all its seeds; 10,000 resamples), beside the same
  difference between two loads of one arm. Lengths have a long tail, so the geometric mean of the paired ratios is
  given too.

## Files

| File | What |
|---|---|
| [loads.tsv](loads.tsv) | per load: replies, generated tokens, median and mean length, the longest, replies of 4,000 tokens or more, replies stopped at the limit, reasoning and answer tokens, draft and grammar counters |
| [compare.tsv](compare.tsv) | each group of loads against another, for length, reasoning and answer: means, difference, interval over prompts, geometric mean ratio, pairs longer and shorter |
| [tasks.tsv](tasks.tsv) | per prompt and load: the five lengths by seed index, their mean, the reasoning's tokens |
| [replies.tsv](replies.tsv) | one row per reply: load, task, seed index and seed, prompt tokens, length, reasoning, answer, how it ended, draft counters, the reply's MD5 |
| [summary.md](summary.md) | the tables as Markdown, as the script printed them |
| [run/run_pv_replies.sh](run/run_pv_replies.sh) | one chain of loads on one card set |
| [run/pv_replies_table.py](run/pv_replies_table.py) | the tables from the chains' work folders |
| [race_price.md](race_price.md) | from `replies.tsv`: the first and second finisher of N samples of one request, per prompt the shortest, median, mean and longest sample, and a cap on the reply's length |
| [run/race_price.py](run/race_price.py) | that table from `replies.tsv` (standard library) |

The replies' text is not kept: the rows hold counts and an MD5. The prompts are requests of
[2026-11-10_prob_verify_agent_check](../2026-11-10_prob_verify_agent_check/README.md)'s run `pv0-1`; its task ids are
in that folder's `tasks.tsv`, and the tasks are rebuilt with
[bench/agent_check/build_bugfix.py](../../../bench/agent_check/build_bugfix.py).

## Commands

`CLIENT` is the machine with the cards (the server and the replay run on it), `WORK` a scratch folder there, `REPO` a
checkout of this repository, `RECORDED` the folder with the agent-check run's `proxy.jsonl` and
`proxy.jsonl.bodies.gz`. `GPUS_A` and `GPUS_B` are the two card sets as Docker device lists.

```bash
# the tools' self-test (no model): replay.py's new options are among its checks
ssh $CLIENT "cd $WORK && AC_SANDBOX=docker AC_CPU_SHARES=256 AC_TEST_PORT=18180 bash $REPO/bench/agent_check/selftest.sh $WORK/selftest"

# the two chains, side by side
COMMON="REPO=$REPO BIN=$BIN MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf LOG=$RECORDED/proxy.jsonl START_BY=08:10 STOP_AT=08:40"
ssh $CLIENT "mkdir -p $WORK/a && cd $WORK && $COMMON WORK=$WORK/a GPUS=$GPUS_A MOUNTS='$MOUNTS' SET=a LOADS='pv0 pv1' \
  NAME=pv-replies-a PORT=18152 setsid -f nohup nice -n 19 bash $REPO/docs/data/2026-11-10_prob_verify_reply_length/run/run_pv_replies.sh > a/console.txt 2>&1 < /dev/null"
ssh $CLIENT "mkdir -p $WORK/b && cd $WORK && $COMMON WORK=$WORK/b GPUS=$GPUS_B MOUNTS='$MOUNTS' SET=b LOADS='pv1 pv0' \
  NAME=pv-replies-b PORT=18154 setsid -f nohup nice -n 19 bash $REPO/docs/data/2026-11-10_prob_verify_reply_length/run/run_pv_replies.sh > b/console.txt 2>&1 < /dev/null"

# the tables
ssh $CLIENT "cd $WORK && python3 -I $REPO/docs/data/2026-11-10_prob_verify_reply_length/run/pv_replies_table.py a b --out $WORK/tables"

# the race's price, from the published rows
ssh $CLIENT "cd $REPO/docs/data/2026-11-10_prob_verify_reply_length && python3 -I run/race_price.py replies.tsv" > race_price.md
```

`START_BY` and `STOP_AT` are the clock limits of this run (a load starts only before the first; the second cuts a
load in progress and keeps its rows); leave them out to run every load to its end.

The server line of a load, as the script built it from the preset (`pv1` adds the two flags in brackets):

```bash
env GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1 \
  LLAMA_DRAFT_SHORTLIST=32768,8192,2 CUDA_CACHE_PATH=$WORK/<set>/jit \
  $BIN -m $MODEL -ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -fit off -lm dio --spec-type draft-mtp --spec-draft-n-max 3 \
  --spec-draft-n-ubatch 128 --no-host --metrics --no-webui [--spec-draft-sampling probabilistic --spec-draft-p-min 0] \
  -md $MODEL_DRAFT --host 127.0.0.1 --port $PORT
```

The four loads ran in 36 minutes on two sets of three cards (three RTX 3090; one RTX 3080 12 GB and two RTX 3090),
without `-ts`: the server split the layers by the cards' free memory.

## Limits

1. **Two loads an arm, 60 replies a load.** A load's mean is carried by a few long replies, and two loads of one
   arm are 36.47 % and 24.15 % apart. A real difference of a third in the mean would not show here; the interval
   reaches from −37.9 % to +57.9 % of the preset's mean.
2. **The far tail is open.** Replies of 10,000 tokens or more: 0 and 1 in the preset's loads, 3 and 1 in `pv1`'s;
   the three longest of the 240 are `pv1`'s. Four against one in 120 each is within chance. More seeds on the
   prompts with long reasoning are Q1188.
3. **One history per task.** All 12 prompts come from one preset run, and only the second request of a task is
   sent. Later turns, with longer histories, are not covered.
4. **A seed index pairs the prompt, not the draw:** two loads of one arm wrote the same reply in 4 and 3 of 60
   pairs.
5. **The reasoning's tokens are counted by tokenizing its text again,** so the parts do not add up to the length
   exactly: reasoning plus tool-call arguments are 65,332 of `pv0-a`'s 66,802 generated tokens; the rest is the
   call's wrapper and the markup.
6. **Step 1 is a read of the source, not a run.** The fork carries a test for the function
   (`tests/test-speculative-rejection-sampling.cpp`); it was not run here.
7. Counts, not time.

## Provenance

- The prompts are requests recorded from this project's bug-fix set, built from Exercism's Python and JavaScript
  tracks (MIT; the licences are in `bench/agent_check/tasks/bugfix/`). The tables hold counters only.
- The engine is the Miner Llama v3.0 release build (`1551f5fe1`). The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
  Qwen3.8-Flash-Next with its MTP draft, as in the release preset.
- Raw output (the replay's rows with their times, the server logs, the cards' memory) is not published.
