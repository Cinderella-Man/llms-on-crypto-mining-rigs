# The MTP draft's acceptance by context depth (Q1172; 2026-11-10)

What this is: how many of the MTP draft's tokens the main model keeps, and how many tokens one step of the main model
yields, when the context in front of the reply holds 0, 8,192, 16,384, 32,768 and 49,152 tokens. The line is the
Miner Llama v3 release preset ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env):
Qwen3.8-Flash-Next, Swift Q2_0, a draft of three tokens), run with [bench/deep_lane](../../../bench/deep_lane/README.md).
Registry row Q1172 in [quants/experiments.md](../../../quants/experiments.md), the acceptance half of Q1153.

Why: the agent check's tasks end below 10,000 tokens of context, and its release run measured an acceptance of 0.718
(3.15 tokens a step) there. Real coding attempts run to 30,000–60,000 tokens, and other engines' published MTP runs
accept less at depth (one model card: 4.10 accepted tokens up to 32K, 3.46 beyond). Acceptance depends on the two
models and the text, not on the card, so it was counted on a machine with newer cards, where no speed is measured.
No time is read or reported here.

## Result

**Acceptance does not fall with depth up to 49,152 tokens on this text. With the preset's sampler, pooled over three
tails, the main model keeps 0.633 of the drafted tokens at 8,192, 0.762 at 16,384, 0.627 at 32,768 and 0.637 at
49,152 (2.87, 3.24, 2.84 and 2.88 tokens a step). The three tails differ more at one depth than the depths differ.**

A tail is the recorded turn every prompt of a run ends in, so the three tails are three generation points per depth.
Per tail: draft tokens accepted of drafted (the share), then tokens a step. Pooled: the counters summed over the
tails, with a 95 % Wilson interval ([pooled.tsv](pooled.tsv), [by_tail.tsv](by_tail.tsv)).

### The preset's sampler (temperature 1.0, top-p 0.95, top-k 20), one request per depth and tail

| Depth (tokens) | Tail 1 | Tail 2 | Tail 3 | Pooled | Acceptance | 95 % interval | Tokens a step | Against 8,192 | 95 % interval |
|---:|---|---|---|---|---:|---|---:|---:|---|
| 0 | 143 of 333 (0.429), 2.27 | 118 of 408 (0.289), 1.86 | 162 of 277 (0.585), 2.72 | 423 of 1,018 | 0.416 | 0.386 to 0.446 | 2.23 | −0.217 | −0.262 to −0.172 |
| 8,192 | 162 of 277 (0.585), 2.72 | 175 of 240 (0.729), 3.16 | 163 of 273 (0.597), 2.75 | 500 of 790 | 0.633 | 0.599 to 0.666 | 2.87 | — | — |
| 16,384 | 184 of 210 (0.876), 3.56 | 180 of 224 (0.804), 3.37 | 167 of 263 (0.635), 2.88 | 531 of 697 | 0.762 | 0.729 to 0.792 | 3.24 | +0.129 | +0.082 to +0.174 |
| 32,768 | 153 of 302 (0.507), 2.49 | 165 of 267 (0.618), 2.81 | 180 of 225 (0.800), 3.37 | 498 of 794 | 0.627 | 0.593 to 0.660 | 2.84 | −0.006 | −0.053 to +0.042 |
| 49,152 | 173 of 245 (0.706), 3.08 | 153 of 303 (0.505), 2.49 | 175 of 238 (0.735), 3.16 | 501 of 786 | 0.637 | 0.603 to 0.670 | 2.88 | +0.004 | −0.043 to +0.052 |

### Greedy (temperature 0), the first request at each depth

| Depth (tokens) | Tail 1 | Tail 2 | Tail 3 | Pooled | Acceptance | 95 % interval | Tokens a step | Against 8,192 | 95 % interval |
|---:|---|---|---|---|---:|---|---:|---:|---|
| 0 | 171 of 252 (0.679), 3.01 | 169 of 256 (0.660), 2.94 | 169 of 256 (0.660), 2.94 | 509 of 764 | 0.666 | 0.632 to 0.699 | 2.97 | +0.003 | −0.044 to +0.050 |
| 8,192 | 186 of 205 (0.907), 3.66 | 152 of 306 (0.497), 2.46 | 170 of 255 (0.667), 2.98 | 508 of 766 | 0.663 | 0.629 to 0.696 | 2.95 | — | — |
| 16,384 | 186 of 207 (0.899), 3.66 | 184 of 209 (0.880), 3.56 | 181 of 219 (0.826), 3.41 | 551 of 635 | 0.868 | 0.839 to 0.892 | 3.54 | +0.205 | +0.161 to +0.246 |
| 32,768 | 167 of 262 (0.637), 2.88 | 177 of 232 (0.763), 3.24 | 184 of 208 (0.885), 3.56 | 528 of 702 | 0.752 | 0.719 to 0.783 | 3.20 | +0.089 | +0.042 to +0.135 |
| 49,152 | 177 of 234 (0.756), 3.24 | 186 of 204 (0.912), 3.66 | 180 of 223 (0.807), 3.37 | 543 of 661 | 0.821 | 0.790 to 0.849 | 3.41 | +0.158 | +0.113 to +0.202 |

The same greedy request sent a second time wrote the same reply in all 15 cells. Its draft counters differ a little
in 9 of them, because the second request reads 4 prompt tokens instead of 64 and its steps fall elsewhere; pooled it
accepts 0.678, 0.691, 0.873, 0.757 and 0.832 at the five depths ([summary.md](summary.md)).

What the tables show:

- **No depth from 8,192 to 49,152 accepts less than 8,192 does.** With the sampler, 32,768 and 49,152 are within
  0.006 of 8,192 (intervals −0.053 to +0.042 and −0.043 to +0.052). Greedy, both are above it.
- **The spread between tails is the larger effect.** Greedy at 8,192 the three tails accept 0.907, 0.497 and 0.667.
  From 8,192 to 49,152 one tail falls and two rise, in both passes: greedy −0.151, +0.415 and +0.141, with the
  sampler +0.121, −0.224 and +0.138.
- **Depth 0 is the tail alone,** 64 tokens with no system prompt, tools or task in front. Its sampler acceptance,
  0.416, is the lowest of the run; it is not a point an agent session visits.
- **Every step drafts three tokens** (786 drafted in 267 steps at 49,152 with the sampler), so tokens a step is 1
  plus the accepted tokens per step: 501 accepted in 267 steps is 2.88.
- With the sampler the four filled depths accept 0.633, 0.762, 0.627 and 0.637 here; the agent check's own sessions
  measured 0.718. The level belongs to the text that is written (see the limits), and it does not fall with depth.

### The reply with and without the draft

The run of tail 1 also loaded the line without the draft. **Its greedy reply differs from the draft line's at all
five depths; each line repeats its own reply exactly on the second request** ([reply_check.tsv](reply_check.tsv)).
The driver keeps the first 160 characters of a reply; within them the two lines part at character 28 (depth 0),
0 (8,192), 35 (16,384) and 145 (32,768), and not at all at 49,152. The draft's counters are counted on the draft
line's own text either way. Whether this is rounding at a near-tie (a four-token verify takes other kernels than
one-token decoding) or a fault in the speculative path is registry row Q1175.

**What it decides.**
- Q1153's depth gate: acceptance gives no depth at which to switch the draft off. The draft's value at depth is then
  set by the time of a step alone, which the deep lane's rig arms and T(k) at depth (Q873) measure.
- The deep lane's draft arms on the rig can be read with these tokens a step at 8,192, 16,384, 32,768 and 49,152:
  2.87, 3.24, 2.84 and 2.88 with the sampler, 2.95, 3.54, 3.20 and 3.41 greedy.
- Q1176: the same count on whole recorded turns, because of the first limit below.

**Limits.**
- **The deep prompts are not sessions an agent wrote.** A prompt is a stream of recorded sessions cut at a fixed
  token count, followed by the last 64 tokens of another recorded turn. In 10 of the 12 greedy replies at depths
  above 0 the model opens by saying that the last tool output or message looks cut, truncated or garbled (read
  from each reply's first 160 characters). The counts are therefore acceptance on such turns at depth, not on the
  turns of a long real session. At depth 0 all three replies continue the tail's own turn.
- **Three tails, 256 tokens each:** at least 632 and at most 1,018 drafted tokens per depth and pass. The intervals
  count each drafted token as an independent trial. The tokens of one draft are accepted in order and all come from
  three replies, so the true uncertainty is wider; the per-tail columns show it.
- The context in front of the tail is the same recorded text for every depth of a tail (a deeper prompt extends a
  shallower one). Between tails it differs by one session, because the tail's own session is left out.
- One model file, one draft, one draft length (three tokens), the sm_61 build's compute 6.1 PTX on newer cards.

## Method

- **The line:** the release preset as it is, without its nine-card split (`-ts`), the draft's device override
  (`-otd`) and the keep-alive variable, on three cards (an RTX 3080 12 GB and two RTX 3090). The build
  (`pascal-p106-v3-b12076-ladder5`, `1551f5fe1`) holds only sm_61 code and compute 6.1 PTX, so the newer cards'
  driver compiles the PTX. After loading, the cards held 9,465, 16,322 and 16,284 MiB with the draft and 7,223,
  16,188 and 16,152 MiB without it.
- **The prompts:** [deep_client.py](../../../bench/deep_lane/deep_client.py) on the release check's recorded
  requests of the arm `t1swift` (12 sessions, the 12 bug-fix tasks of [bench/agent_check](../../../bench/agent_check/README.md)),
  rendered by the server's chat template with their tools and tokenized into one stream. A prompt of depth D is the
  stream's first tokens and a tail of 64 tokens, D tokens together. Tail 1 is the end of the recorded request whose
  reply was the longest (15,065 characters), tail 2 the second longest (6,853) and tail 3 the third (5,663); they
  come from three different sessions. The tail's session is left out of the stream.
- **The requests** per depth, in one server load per run: a prime (the prompt without its tail, read on top of the
  cache), the whole prompt twice at temperature 0 with 256 tokens out and end of text ignored, and once with the
  server line's own sampler and a fixed seed.
- **The counters** are the server's `timings` of each request: `draft_n` (tokens the draft proposed) and
  `draft_n_accepted` (those the main model kept). Acceptance is accepted over drafted. A step is one pass of the main
  model: it yields one token of its own and the draft tokens it accepts, so steps are generated minus accepted
  tokens, and tokens a step is generated tokens over steps.
- **Pooling and intervals:** [run/pool_tables.py](run/pool_tables.py) sums the counters over the tails; the interval
  of an acceptance is Wilson's, the interval of a difference to depth 8,192 is Newcombe's (his method 10), both at
  95 %.
- **Three runs:** tail 1 with the arms `draft` and `nodraft`, tails 2 and 3 with `draft` alone
  ([run/run_tails.sh](run/run_tails.sh)). The quiet checks were off (`QUIET=0`), so every request is marked
  `valid=0` and the driver's tables print no speed.
- **The driver on a real server:** this was its first run against llama-server. The server's template took the
  recorded requests with their tools, every prompt had exactly its depth in tokens, and the primes at 16,384, 32,768
  and 49,152 continued from a context checkpoint (8,708, 16,388 and 16,900 tokens read on 7,612, 16,316 and 32,188
  cached).

## Files

| File | What |
|---|---|
| [summary.md](summary.md) | the pooled tables as `pool_tables.py` printed them: greedy pass 1, greedy pass 2, the sampler, the reply check |
| [pooled.tsv](pooled.tsv) | per pass (`1`, `2`: greedy; `s`: sampler) and depth: tails, generated, drafted and accepted tokens, acceptance with its interval, steps, tokens a step, the difference to depth 8,192 with its interval |
| [by_tail.tsv](by_tail.tsv) | per tail, depth and pass of the `draft` arm: prompt tokens, generated, drafted and accepted tokens, acceptance, steps, tokens a step, the md5 of the reply's token ids |
| [reply_check.tsv](reply_check.tsv) | tail 1, per depth: whether the `draft` and `nodraft` arms wrote the same greedy reply, with each arm's reply hashes |
| [tail1.md](tail1.md), [tail2.md](tail2.md), [tail3.md](tail3.md) | the driver's own table of each run: the arms, the draft and the step per load, the sampler pass, the checks (prompt sizes, tokens out, the greedy reply, what each prime read) |
| [run/run_tails.sh](run/run_tails.sh) | the three runs, one after another, with hosts and paths from the environment |
| [run/pool_tables.py](run/pool_tables.py) | the pooled tables from the runs' `requests.csv` |

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
MOUNTS="<the folders of WORK, the build, the models>" DOCKER_ARGS="--cpu-shares=256" NAME=fn42 PORT=18142 \
setsid -f nohup nice -n 19 bash docs/data/2026-11-10_draft_acceptance_by_depth/run/run_tails.sh > console.txt 2>&1 < /dev/null
```

It writes `WORK/tail1`, `WORK/tail2` and `WORK/tail3` and ends with `WORK/chain.done` (27 minutes here, five server
loads). Then the tables:

```bash
cd <work folder> && nice -n 19 python3 -I <checkout>/docs/data/2026-11-10_draft_acceptance_by_depth/run/pool_tables.py \
  tail1 tail2 tail3 --out tables
```

Each run's server line is the preset's with `ARGS_DROP="-ts -otd"`, `ENV_DROP=GGML_CUDA_KEEPALIVE` and a
`CUDA_CACHE_PATH` for the compiled kernels, inside
`docker run --rm --init --network host --gpus "device=<cards>" … nvidia/cuda:12.9.2-devel-ubuntu24.04`. `run_tails.sh`
holds the whole call. `CUDA_FORCE_PTX_JIT` is not set.

## Provenance and licence

The recorded requests are this project's release check of 2026-11-05
([docs/data/2026-11-05_release_flashnext](../2026-11-05_release_flashnext/README.md)): the agent's sessions on 12
bug-fix tasks (Exercism exercises, MIT). The tables hold token counts, the server's counters and hashes; no prompt or
reply text. The scripts are under this repository's licence.
