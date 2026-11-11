# The MTP draft's tokens a step at 65,536 and 120,000 tokens with the preset's sampler (Q1251; 2026-11-11)

What this is: how many tokens one step of the main model yields with the MTP draft on the 131,072-token line, when the
context in front of the reply holds 65,536 and 120,000 tokens and the reply is written with the preset's own sampler.
The line is the Miner Llama v3 release preset ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env):
Qwen3.8-Flash-Next, Swift Q2_0, a draft of three tokens) at `-c 131072 -ub 256`, run with
[bench/deep_lane](../../../bench/deep_lane/README.md). Registry row Q1251 in
[quants/experiments.md](../../../quants/experiments.md).

Why: on the rig, the 131,072-token line with the draft was a gain in generation at 49,152, 65,536 and 120,000 tokens
and a loss in prompt reading at all three ([2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md), `ctx131k/`,
Q1218). Its replies were greedy, and they yielded 3.28, 3.71 and 3.82 tokens a step in one load and 3.56, 3.66 and
3.20 in the other. The preset writes with its sampler, which yields fewer: the only count with the sampler stopped at
49,152 tokens (2.88, [2026-11-10_draft_acceptance_by_depth](../2026-11-10_draft_acceptance_by_depth/README.md), Q1172).
Whether a 131,072-token preset keeps the draft depends on the sampler's yield at the deeper depths. The yield depends
on the two models and the text, not on the card, so it was counted on a machine with newer cards, where no speed is
measured. No time is read or reported here.

## Result

**With the preset's sampler, pooled over three tails, the draft yields 2.93 tokens a step at 65,536 tokens and 2.69 at
120,000 (506 of 769 and 483 of 841 drafted tokens accepted: 0.658 and 0.574). At the rig's step times on this line
that is est. +1.4 % against no draft on a first request at 65,536 and −1.3 % at 120,000. With the draft's cost in
prompt reading, it pays at 65,536 only in a turn that reads at most 0.23 new prompt tokens per token it writes, and at
120,000 in no turn.**

A tail is the recorded turn every prompt of a run ends in, so the three tails are three generation points per depth.
Per tail: draft tokens accepted of drafted (the share), then tokens a step. Pooled: the counters summed over the
tails, with a 95 % Wilson interval ([pooled.tsv](pooled.tsv), [by_tail.tsv](by_tail.tsv)).

### The preset's sampler (temperature 1.0, top-p 0.95, top-k 20), one request per depth and tail

| Depth (tokens) | Tail 1 | Tail 2 | Tail 3 | Pooled | Acceptance | 95 % interval | Tokens a step | Against 65,536 | 95 % interval |
|---:|---|---|---|---|---:|---|---:|---:|---|
| 65,536 | 169 of 256 (0.660), 2.94 | 175 of 237 (0.738), 3.16 | 162 of 276 (0.587), 2.72 | 506 of 769 | 0.658 | 0.624 to 0.691 | 2.93 | — | — |
| 120,000 | 157 of 293 (0.536), 2.59 | 162 of 278 (0.583), 2.72 | 164 of 270 (0.607), 2.78 | 483 of 841 | 0.574 | 0.541 to 0.607 | 2.69 | −0.084 | −0.131 to −0.036 |

### Greedy (temperature 0), the first request at each depth

| Depth (tokens) | Tail 1 | Tail 2 | Tail 3 | Pooled | Acceptance | 95 % interval | Tokens a step | Against 65,536 | 95 % interval |
|---:|---|---|---|---|---:|---|---:|---:|---|
| 65,536 | 176 of 235 (0.749), 3.20 | 181 of 219 (0.826), 3.41 | 189 of 197 (0.959), 3.82 | 546 of 651 | 0.839 | 0.808 to 0.865 | 3.46 | — | — |
| 120,000 | 188 of 198 (0.949), 3.76 | 161 of 280 (0.575), 2.69 | 190 of 193 (0.984), 3.88 | 539 of 671 | 0.803 | 0.772 to 0.832 | 3.35 | −0.035 | −0.077 to +0.006 |

The same greedy request sent a second time wrote the same reply in all six cells; its counters differ only at 65,536
on tails 1 and 2 (175 of 238 and 182 of 216), and pooled they are the same as pass 1's ([summary.md](summary.md)).

### What it means for the 131,072-token line on the rig

The rig's loads of this line (Q1218, two loads an arm) give the time of a step with the draft and of a token without
it. The draft pays when a step yields more tokens than its time is in plain tokens: the break-even. These yields at
those step times ([r55_at_this_yield.md](r55_at_this_yield.md), estimates, not measured speeds):

| Depth (tokens) | Yield | Tokens a step | Break-even on pass 1 | Est. on pass 1 | Break-even on pass 2 | Est. on pass 2 |
|---:|---|---:|---:|---:|---:|---:|
| 49,152 | the sampler (Q1172) | 2.88 | 2.87 | +0.3 % | 2.65 | +8.5 % |
| 65,536 | the sampler (this run) | 2.93 | 2.89 | +1.4 % | 2.67 | +9.8 % |
|  | greedy (this run) | 3.46 | 2.89 | +19.7 % | 2.67 | +29.5 % |
| 120,000 | the sampler (this run) | 2.69 | 2.73 | −1.3 % | 2.76 | −2.4 % |
|  | greedy (this run) | 3.35 | 2.73 | +22.8 % | 2.76 | +21.5 % |

Pass 1 is the first request at a depth, the one an agent's turn makes; pass 2 repeats it on a warm cache.

- **With the sampler the draft sits at its break-even from 49,152 tokens on.** The share of drafted tokens the step
  must keep to break even on a first request is 0.649 at 65,536 and 0.588 at 120,000 (Q1218); both lie inside this
  run's intervals (0.624 to 0.691 and 0.541 to 0.607).
- **The prompt cost decides it.** On the rig the draft reads prompts 29.6 % slower at 65,536 and 33.3 % slower at
  120,000 (4.49 and 6.81 ms more a prompt token). At the sampler's yield the draft saves 1.04 ms a generated token at
  65,536 and loses 1.22 at 120,000, so it pays in a turn only up to 0.23 new prompt tokens per generated token at
  65,536 and never at 120,000 ([r55_at_this_yield_turn.tsv](r55_at_this_yield_turn.tsv)). A turn that reads a
  2,000-token tool output and writes 256 tokens reads 7.8 per token written.
- **The rig's gains were the greedy text's.** The rig's draft arms wrote greedy replies; at the greedy yields counted
  here (3.46 and 3.35) the draft would be est. +19.7 % and +22.8 % on a first request.
- **The yield falls from 65,536 to 120,000 with the sampler:** the acceptance changes by −0.084, and its interval
  excludes zero (−0.131 to −0.036). Two tails fall (−0.124 and −0.156) and one rises (+0.020). Greedy the pooled
  change is −0.035 (−0.077 to +0.006), with one tail falling (−0.251) and two rising (+0.201 and +0.025).
- Against Q1172's sampler counts at 8,192, 16,384, 32,768 and 49,152 tokens (0.633, 0.762, 0.627 and 0.637), 0.658 at
  65,536 is level and 0.574 at 120,000 is the lowest at any depth with context in front. Two of the three tails differ
  between the runs, so this is a comparison of tail sets.

**Limits.**
- **The deep prompts are not sessions an agent wrote.** A prompt is a stream of recorded sessions cut at a fixed
  token count, followed by the last 64 tokens of another recorded turn. In 8 of the 12 replies the model opens by
  saying that the last tool output or message looks truncated or garbled (all four of tail 1, and tails 2 and 3 at
  65,536, read from each reply's first 160 characters). At 120,000, tails 2 and 3 continue the task in both passes.
- **Three tails, 256 tokens each, one sampler request per tail and depth with one seed:** 769 and 841 drafted tokens
  per depth with the sampler. The intervals count each drafted token as an independent trial; the tokens of one draft
  are accepted in order and all come from three replies, so the true uncertainty is wider. The per-tail columns show
  it.
- **The estimates pair this machine's yield with the rig's step times.** A step's time does not depend on the text,
  but the rig's two loads and this run write different greedy replies from the same prompts (tail 1's prompts here
  are the rig run's, token for token), as two loads of the line on the rig do ([Q1175](../2026-11-10_draft_line_exactness/README.md)).
- The stream is the recorded sessions of two arms of the release check, the same 12 tasks on two paths, so file
  contents repeat in the context. One model file, one draft length (three tokens), the sm_61 build's compute 6.1 PTX
  on newer cards.

## Method

- **The line:** the release preset at `-c 131072 -ub 256` (the rig's 131,072-token line, Q1154), without its nine-card
  split (`-ts`), the draft's device override (`-otd`) and the keep-alive variable, on three cards (an RTX 3080 12 GB
  and two RTX 3090). The build is the v3.0 release engine (`pascal-p106-v3-b12076-ladder5`, `1551f5fe1`), the one the
  rig's loads of this line ran (Q1218) and the one Q1172 used; it holds only sm_61 code and compute 6.1 PTX, so the
  newer cards' driver compiles the PTX. The rig's line also set `GGML_CUDA_FATTN_SPARSE_PASCAL=1`, which acts only on
  cards older than Turing; this run leaves it unset, as Q1172 did. After loading, the cards held 9,963, 17,050 and
  17,044 MiB in each of the three loads.
- **The prompts:** [deep_client.py](../../../bench/deep_lane/deep_client.py) on the release check's recorded requests
  of the arms `t1swift` and `t1base` (`TAG='^(t1swift|t1base)/'`: 24 sessions, the 12 bug-fix tasks of
  [bench/agent_check](../../../bench/agent_check/README.md) on two paths), rendered by the server's chat template with
  their tools and tokenized into one stream. A prompt of depth D is the stream's first tokens and a tail of 64 tokens,
  D tokens together. Tail 1 is the end of the recorded request whose reply was the longest (15,065 characters): the
  rig runs' tail, whose prompts at both depths have the same token ids here. Tail 2 is the second longest (8,921) and
  tail 3 the third (6,853, Q1172's tail 2). They come from three different sessions; the tail's session is left out of
  the stream.
- **The requests** per depth, in one server load per tail: a prime (the prompt without its tail, read on top of the
  cache), the whole prompt twice at temperature 0 with 256 tokens out and end of text ignored, and once with the
  server line's own sampler and a fixed seed. In every load the prime at 65,536 read 65,412 tokens on 60 cached and
  the one at 120,000 read 54,724 on 65,212, each from a context checkpoint; the rig's prime at 120,000 read the same.
- **The counters** are the server's `timings` of each request: `draft_n` (tokens the draft proposed) and
  `draft_n_accepted` (those the main model kept). A step is one pass of the main model: it yields one token of its own
  and the draft tokens it accepts, so steps are generated minus accepted tokens, and tokens a step is generated tokens
  over steps.
- **Pooling and intervals:** Q1172's [pool_tables.py](../2026-11-10_draft_acceptance_by_depth/run/pool_tables.py)
  with `--ref 65536`; Wilson's interval for an acceptance, Newcombe's (his method 10) for a difference, both at 95 %.
- **The estimate:** Q1218's [draft_pair.py](../2026-11-10_deep_lane/run/draft_pair.py) on that run's folder with this
  run's pooled tokens a step as `--yield`. On the folder with Q1218's own `--yield` it reproduces Q1218's published
  `draft_pair.md`, `draft_pair_turn.tsv` and `draft_pair_breakeven.tsv` byte for byte.
- **The run:** three loads one after another ([run/run_tails.sh](run/run_tails.sh)), 52 minutes. The quiet checks were
  off (`QUIET=0`), so every request is marked `valid=0` and the driver's tables print no speed. No CUDA error and no
  Xid.

## Files

| File | What |
|---|---|
| [summary.md](summary.md) | the pooled tables as `pool_tables.py` printed them: greedy pass 1, greedy pass 2, the sampler |
| [pooled.tsv](pooled.tsv) | per pass (`1`, `2`: greedy; `s`: sampler) and depth: tails, generated, drafted and accepted tokens, acceptance with its interval, steps, tokens a step, the difference to depth 65,536 with its interval |
| [by_tail.tsv](by_tail.tsv) | per tail, depth and pass: prompt tokens, generated, drafted and accepted tokens, acceptance, steps, tokens a step, the md5 of the reply's token ids |
| [tail1.md](tail1.md), [tail2.md](tail2.md), [tail3.md](tail3.md) | the driver's own table of each run: the draft and the step, the sampler pass, the checks (prompt sizes, tokens out, the greedy reply, what each prime read) |
| [r55_at_this_yield.md](r55_at_this_yield.md) | `draft_pair.py` on the rig's 131,072-token loads (Q1218) with this run's yields: Q1218's tables plus the estimate at another yield and the whole-turn rows |
| [r55_at_this_yield_turn.tsv](r55_at_this_yield_turn.tsv) | per depth and yield: what the draft adds per prompt token, the tokens a step, the time saved per generated token, the prompt tokens per generated token up to which the draft pays |
| [run/run_tails.sh](run/run_tails.sh) | the three runs, one after another, with hosts and paths from the environment |

Not published: the prompts' token ids, the replies and the server logs (recorded text and paths), and the per-request
times, which are not measurements on this machine.

## Commands

On the machine with the cards, from a checkout of this repository (`REPO`), with the recorded requests of the release
check (`proxy.jsonl` and `proxy.jsonl.bodies.gz` in one folder) and the model files:

```bash
WORK=<work folder> REPO=<checkout> BIN=<build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT=<recorded>/proxy.jsonl.bodies.gz GPUS=<three cards, as docker names them> \
MOUNTS="<the folders of WORK, the build, the models>" DOCKER_ARGS="--cpu-shares=256" NAME=fn65 PORT=18165 \
setsid -f nohup nice -n 19 bash docs/data/2026-11-11_fn65_draft_yield_deep/run/run_tails.sh > console.txt 2>&1 < /dev/null
```

Its defaults are this run's: `TAG='^(t1swift|t1base)/'`, `ARMS=draft`, `CTX=131072`, `UB=256`,
`DEPTHS="65536 120000"`. It writes `WORK/tail1`, `WORK/tail2` and `WORK/tail3` and ends with `WORK/chain.done`. Then
the tables, and the estimate on the rig's run (Q1218's `ctx131k` deep-lane folder as `<rig run>`):

```bash
cd <work folder>
nice -n 19 python3 -I <checkout>/docs/data/2026-11-10_draft_acceptance_by_depth/run/pool_tables.py \
  tail1 tail2 tail3 --out tables --ref 65536
nice -n 19 python3 -I <checkout>/docs/data/2026-11-10_deep_lane/run/draft_pair.py <rig run> \
  --yield "the sampler off the rig, at 49,152 (Q1172):49152=2.876" \
  --yield "the sampler off the rig (Q1251):65536=2.931,120000=2.695" \
  --yield "greedy off the rig (Q1251):65536=3.459,120000=3.354" \
  --out tables/r55_at_this_yield.md --tsv tables/r55_at_this_yield
```

Each run's server line is the preset's with `-c 131072 -ub 256`, `ARGS_DROP="-ts -otd"`,
`ENV_DROP=GGML_CUDA_KEEPALIVE` and a `CUDA_CACHE_PATH` for the compiled kernels, inside
`docker run --rm --init --network host --gpus "device=<cards>" … nvidia/cuda:12.9.2-devel-ubuntu24.04`.
`CUDA_FORCE_PTX_JIT` is not set.

## Provenance and licence

The recorded requests are this project's release check of 2026-11-05
([docs/data/2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)): the agent's sessions on 12
bug-fix tasks (Exercism exercises, MIT). The step times are the rig run of 2026-11-10
([docs/data/2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md)). The tables hold token counts, the server's
counters and hashes; no prompt or reply text. The scripts are under this repository's licence.
