# The end-of-turn stops on real loads: the rejection stop clears the grammar rejections, block verification never runs on the preset (Q1262, Q1291; 2026-11-11)

What this is: a count of what two switches of the v4 line do to probabilistic verification on real replies of the
Flash-Next preset ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env): Qwen3.8-Flash-Next,
Swift Q2_0, an MTP draft of three tokens). Five arms, each loaded once on each of two card sets:

| Arm | The preset plus |
|---|---|
| `pv0` | nothing: the draft takes its first choice, and the main model keeps a drafted token only when its own sample is the same token |
| `pv1` | `--spec-draft-sampling probabilistic --spec-draft-p-min 0`: the draft samples, and a drafted token is accepted with probability min(1, p/q) (token-by-token rejection) |
| `pv1s` | `pv1` + `LLAMA_SPEC_REJECT_STOP=1`: token-by-token rejection stops after an accepted end-of-generation token ([C675's design](../2026-11-11_c675_end_of_turn_stop/README.md), Q1187) |
| `pv1b` | `pv1` + `LLAMA_SPEC_BLOCK_VERIFY=1`: sampled drafts verified as a block (Q797) |
| `pv1bs` | `pv1b` + `LLAMA_SPEC_REJECT_STOP=1`: the block also ends at its first drafted end of generation ([C711's design](../2026-11-11_c711_block_verify_stop/README.md), Q1263) |

Registry rows Q1262 (the rejection stop's count) and Q1291 (the block cut's count) in
[quants/experiments.md](../../../quants/experiments.md), from Q1187 and Q1263.

Why: with `pv1` about half the replies that end in a tool call have one grammar rejection
([2026-11-10_prob_verify_reply_length](../2026-11-10_prob_verify_reply_length/README.md), Q1184: 33 and 29 of 60,
the preset's none). Token-by-token rejection goes on verifying the drafted tokens behind an accepted end of the turn;
the main sampler's token for such a row fails the tools' grammar and is drawn again from the whole vocabulary on the
host. On a P106 that costs est. 389.65 ms a time (a fit over the requests) ([2026-11-10_prob_verify_rig](../2026-11-10_prob_verify_rig/README.md),
Q731). Both switches are exact by their CPU checks; this run counts what they do on the model's own replies. Counts
are the same on any card, so they were taken on a machine with newer cards. No time is read or reported here.

## Result

**The rejection stop does its job and nothing else: `LLAMA_SPEC_REJECT_STOP=1` takes `pv1`'s grammar rejections
from 29 and 24 a load (each in a reply of its own) to 0 and 0, the preset's count, and every one of the 120 replies
is the same text as without it.** The drafted and generated tokens are the same to the token. What goes are the rows
behind the end of the turn: 58 and 48 accepted tokens, device draws and grammar checks, two in each reply that had a
rejection.

**Block verification never runs on this preset.** The preset samples on the device (`-bs`, backend sampling), and the
server verifies a sampled draft as a block only for requests without backend sampling; requests with it keep
token-by-token rejection, as the block path was built (Q797). So `pv1b` gives `pv1`'s counts and its 60 replies on
each set, and `pv1bs` gives `pv1s`'s. On the Flash-Next preset `LLAMA_SPEC_REJECT_STOP=1` acts through the
token-by-token stop alone, and the block cut (C711) is not reached; its count waits for a line that verifies as a
block, one without backend sampling (Q1383).

### Every load

| Load | Set, order | Generated | Drafted | Accepted | Tokens a step | Drawn on the device | Grammar checks | Grammar rejections | Replies with one |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pv0-a` | a, 1st | 78,984 | 75,468 | 53,776 | 3.133 | 78,992 | 2,946 | 0 | 0 of 60 |
| `pv1-a` | a, 2nd | 66,831 | 61,329 | 46,394 | 3.270 | 66,897 | 2,762 | 29 | 29 of 60 |
| `pv1s-a` | a, 3rd | 66,831 | 61,329 | 46,336 | 3.261 | 66,839 | 2,704 | 0 | 0 of 60 |
| `pv1b-a` | a, 4th | 66,831 | 61,329 | 46,394 | 3.270 | 66,897 | 2,762 | 29 | 29 of 60 |
| `pv1bs-a` | a, 5th | 66,831 | 61,329 | 46,336 | 3.261 | 66,839 | 2,704 | 0 | 0 of 60 |
| `pv1bs-b` | b, 1st | 81,751 | 74,427 | 56,891 | 3.288 | 81,760 | 2,934 | 0 | 0 of 60 |
| `pv1b-b` | b, 2nd | 81,751 | 74,427 | 56,939 | 3.295 | 81,808 | 2,982 | 24 | 24 of 60 |
| `pv1s-b` | b, 3rd | 81,751 | 74,427 | 56,891 | 3.288 | 81,760 | 2,934 | 0 | 0 of 60 |
| `pv1-b` | b, 4th | 81,751 | 74,427 | 56,939 | 3.295 | 81,808 | 2,982 | 24 | 24 of 60 |
| `pv0-b` | b, 5th | 86,706 | 82,089 | 59,296 | 3.163 | 86,719 | 2,725 | 0 | 0 of 60 |

- 60 replies a load, every one ended in a tool call, none at the 32,768-token limit, none with an error.
- Tokens a step is the table's generated ÷ (generated − accepted). `pv1` counts as accepted the two drafted tokens
  behind the end of the turn in each reply with a rejection, tokens that are never output; the stop leaves them out.
  So `pv1s` reads 3.261 and 3.288 against `pv1`'s 3.270 and 3.295 (−0.28 % and −0.21 %) on the same replies and the
  same drafts. That is within the spread of `pv1`'s two loads (3.270 and 3.295, 0.76 % apart).
- Drawn on the device less generated: 8 and 13 for `pv0`, 8 and 9 for `pv1s` and `pv1bs`, 66 and 57 for `pv1` and
  `pv1b` (sets a and b).

### The stopped arms against the same arm without the stop

| Set | First | Second | Replies | Same text | Same length | Rejections, first | Rejections, second | Accepted, first | Accepted, second |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| a | `pv1` | `pv1s` | 60 | 60 | 60 | 29 | 0 | 46,394 | 46,336 |
| a | `pv1b` | `pv1bs` | 60 | 60 | 60 | 29 | 0 | 46,394 | 46,336 |
| a | `pv1` | `pv1b` | 60 | 60 | 60 | 29 | 29 | 46,394 | 46,394 |
| a | `pv1s` | `pv1bs` | 60 | 60 | 60 | 0 | 0 | 46,336 | 46,336 |
| a | `pv0` | `pv1` | 60 | 1 | 1 | 0 | 29 | 53,776 | 46,394 |
| b | `pv1` | `pv1s` | 60 | 60 | 60 | 24 | 0 | 56,939 | 56,891 |
| b | `pv1b` | `pv1bs` | 60 | 60 | 60 | 24 | 0 | 56,939 | 56,891 |
| b | `pv1` | `pv1b` | 60 | 60 | 60 | 24 | 24 | 56,939 | 56,939 |
| b | `pv1s` | `pv1bs` | 60 | 60 | 60 | 0 | 0 | 56,891 | 56,891 |
| b | `pv0` | `pv1` | 60 | 2 | 2 | 0 | 24 | 59,296 | 56,939 |

On one card set a load is repeatable: the four probabilistic arms wrote the same 60 replies, to the MD5, whichever
order they ran in. Across the two sets the same arm writes other replies (other cards, another split), so the sets
are the spread, and on one set an exact switch can be checked reply for reply.

### The server's start lines

| Load | "block verification on" | "token-by-token rejection stops after an accepted end of generation" | "block verification ends each block at its first drafted end of generation" |
|---|---|---|---|
| `pv0-a`, `pv0-b`, `pv1-a`, `pv1-b` | no | no | no |
| `pv1s-a`, `pv1s-b` | no | yes | no |
| `pv1b-a`, `pv1b-b` | yes | no | no |
| `pv1bs-a`, `pv1bs-b` | yes | yes | yes |

The lines are printed once at start from the switches; "block verification on" does not mean that a request uses it.

### Reply lengths, beside Q1184

On this build `pv1`'s two loads generated 66,831 and 81,751 tokens for the 60 replies, the preset's 78,984 and
86,706: −10.33 % between the arms' means (95 % over prompts −444.3 to +123.9 tokens a reply), where the preset's own
two loads are 9.78 % apart and `pv1`'s 22.32 %. As in Q1184, the replies are not longer with `pv1`. Every table is in
[summary.md](summary.md).

## Method

The second request of each of the 12 bug-fix tasks of the agent-check run `pv0-1`
([2026-11-10_prob_verify_agent_check](../2026-11-10_prob_verify_agent_check/README.md)), sent with five seeds each and
the reply limit the agent sent (32,768 tokens): 60 replies a load, the same prompts and seeds in every load. Two
chains side by side on two sets of three cards, the arms in opposite orders: set a (three RTX 3090) `pv0 pv1 pv1s pv1b
pv1bs`, set b (one RTX 3080 12 GB, two RTX 3090) `pv1bs pv1b pv1s pv1 pv0`. Each load is one llama-server on the
preset's line without its nine-card split (`-ts`, `-otd`) and `GGML_CUDA_KEEPALIVE`, plus `--metrics --no-webui`; the
server split the layers by the cards' free memory. The chain and table scripts are Q1184's
([run_pv_replies.sh](../2026-11-10_prob_verify_reply_length/run/run_pv_replies.sh),
[pv_replies_table.py](../2026-11-10_prob_verify_reply_length/run/pv_replies_table.py)), unchanged;
[run/stop_pairs.py](run/stop_pairs.py) pairs the arms reply by reply and reads the start lines.

Where the server picks the path: the verify step in `tools/server/server-context.cpp` takes
`common_sampler_sample_and_accept_n_block` only when block verification is on and the request has no backend
sampling, and `common_sampler_sample_and_accept_n_rejection` for every other sampled draft. A request's backend
sampling is the server's `-bs` unless the request sets `backend_sampling` itself; these requests do not.

## Files

| File | What |
|---|---|
| [loads.tsv](loads.tsv) | per load: replies, generated tokens, median and mean length, the longest, long replies, reasoning and answer tokens, drafted and accepted tokens, tokens a step, draws on the device, grammar checks and rejections |
| [pairs.tsv](pairs.tsv) | per set and pair of arms: replies with the same text and length, and each load's rejections, drafted, accepted and generated tokens |
| [switches.tsv](switches.tsv) | per load: which of the three start lines its server log holds |
| [compare.tsv](compare.tsv) | `pv1` against `pv0` and two loads of one arm against each other, paired by prompt and seed index: length, reasoning and answer |
| [tasks.tsv](tasks.tsv) | per prompt and load: the five lengths by seed index and their mean |
| [replies.tsv](replies.tsv) | one row per reply: load, task, seed index and seed, length, reasoning, answer, how it ended, draft and grammar counters, the reply's MD5 |
| [summary.md](summary.md) | the tables as Markdown, as `pv_replies_table.py` printed them |
| [run/stop_pairs.py](run/stop_pairs.py) | `pairs.tsv` and `switches.tsv` from the chains' work folders (standard library) |

The replies' text is not kept: the rows hold counts and an MD5. The prompts are requests of Q1183's run `pv0-1`; its
task ids are in that folder's `tasks.tsv`, and the tasks are rebuilt with
[bench/agent_check/build_bugfix.py](../../../bench/agent_check/build_bugfix.py).

## Commands

`CLIENT` is the machine with the cards (the servers and the replays run on it), `WORK` a scratch folder there, `REPO`
a checkout of this repository, `RECORDED` the folder with the agent-check run's `proxy.jsonl` and
`proxy.jsonl.bodies.gz`, `MODELS` the folder with the model and the draft, `BIN` the build's `llama-server`, `MOUNTS`
the folders the containers read. `GPUS_A` and `GPUS_B` are the two card sets as Docker device lists.

```bash
RUN=$REPO/docs/data/2026-11-10_prob_verify_reply_length/run
PV='--spec-draft-sampling probabilistic --spec-draft-p-min 0'
COMMON="REPO=$REPO BIN=$BIN MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf LOG=$RECORDED/proxy.jsonl MOUNTS='$MOUNTS' \
  ARGS_pv1s='$PV' ENV_pv1s=LLAMA_SPEC_REJECT_STOP=1 \
  ARGS_pv1b='$PV' ENV_pv1b=LLAMA_SPEC_BLOCK_VERIFY=1 \
  ARGS_pv1bs='$PV' ENV_pv1bs='LLAMA_SPEC_BLOCK_VERIFY=1 LLAMA_SPEC_REJECT_STOP=1'"

# the two chains, side by side
ssh $CLIENT "mkdir -p $WORK/a && cd $WORK && $COMMON WORK=$WORK/a GPUS=$GPUS_A SET=a LOADS='pv0 pv1 pv1s pv1b pv1bs' \
  NAME=fn69-a PORT=18692 setsid -f nohup nice -n 19 bash $RUN/run_pv_replies.sh > a/console.txt 2>&1 < /dev/null"
ssh $CLIENT "mkdir -p $WORK/b && cd $WORK && $COMMON WORK=$WORK/b GPUS=$GPUS_B SET=b LOADS='pv1bs pv1b pv1s pv1 pv0' \
  NAME=fn69-b PORT=18694 setsid -f nohup nice -n 19 bash $RUN/run_pv_replies.sh > b/console.txt 2>&1 < /dev/null"

# the tables
ssh $CLIENT "cd $WORK && python3 -I $RUN/pv_replies_table.py a b --out $WORK/tables \
  && python3 -I $REPO/docs/data/2026-11-11_fn69_fn66_stops/run/stop_pairs.py a b --out $WORK/tables"
```

The server line of `pv1bs`, as the script built it from the preset (`pv0` has neither the two flags nor the two
switches; `pv1`, `pv1s` and `pv1b` have the flags and the switches their names give):

```bash
env GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1 \
  LLAMA_DRAFT_SHORTLIST=32768,8192,2 CUDA_CACHE_PATH=$JIT LLAMA_SPEC_BLOCK_VERIFY=1 LLAMA_SPEC_REJECT_STOP=1 \
  $BIN -m $MODEL -ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -fit off -lm dio --spec-type draft-mtp --spec-draft-n-max 3 \
  --spec-draft-n-ubatch 128 --no-host --metrics --no-webui --spec-draft-sampling probabilistic --spec-draft-p-min 0 \
  -md $MODEL_DRAFT --host 127.0.0.1 --port $PORT
```

The ten loads ran side by side, set a from 16:21:00 to 17:22:41 and set b to 17:52:31.

## Limits

1. **The block cut is not counted.** No load here verifies as a block, so C711's cut has only its CPU check. Its
   count needs a line without backend sampling (Q1383).
2. **What the stop saves on a P106 is not timed here.** It is one host draw over the whole vocabulary with the grammar
   in each reply that ends this way (est. 389.65 ms each by a fit over Q731's replay); the replay that times it is Q1384.
3. One history per task, the second request only, and 60 replies a load, as in Q1184.
4. Counts, not time.

## Provenance

- The prompts are requests recorded from this project's bug-fix set, built from Exercism's Python and JavaScript
  tracks (MIT; the licences are in `bench/agent_check/tasks/bugfix/`). The tables hold counters only.
- The engine is the v4 line's `pascal-p106-v4-b12246-c711` (`ca3c0b4fd`, C675's and C711's commits on
  `pascal-p106-v4`), built for Pascal with the PTX the newer cards compile at load. The model is UkisAI's Swift-1.5
  GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, as in the release preset.
- Raw output (the replay's rows with their times, the server logs, the cards' memory) is not published.
