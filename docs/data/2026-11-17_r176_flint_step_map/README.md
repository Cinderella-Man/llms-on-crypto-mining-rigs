# Flint's v4.0 decode step taken apart (R176, Flint's half; Q1510 with Q1543's passes; 2026-11-17)

What this measured: where a decode step of Flint Pascal Q8 GGUF (`p065`) goes on Miner Llama v4.0 with the released
preset, part by part and kernel by kernel, at 0 and 8,192 tokens of context, and what a pass of 2 to 8 tokens costs
it. Nine P106-090 cards, [release/flint-v4/settings.env](../../../release/flint-v4/settings.env) unchanged, plus the
profilers' variables and, for the passes of several tokens, the server's speculative flags. Registry rows Q1510 (the
map) and Q1543 (the published drafter these passes price) in [quants/experiments.md](../../../quants/experiments.md).
Ornith-1.5-122B's half of the same row is [2026-11-17_r176_ornith_step_map](../2026-11-17_r176_ornith_step_map/README.md);
the method is R121's on Flash-Next ([2026-11-17_r121_flash_step_map](../2026-11-17_r121_flash_step_map/README.md)).

Each value is one load or one request. Two values in a cell are two loads, in run order.

**In plain words.** A Flint token costs one whole pass through the nine cards, about 26 ms, unless the pipeline
guessed it right; the 80 right guesses in 256 tokens come almost free, which is how the preset gets to 18.09 ms a
token. Checking several tokens in one pass costs Flint more per token than writing them one at a time (at least 35.13
ms a token at every width from 2 to 8), so a drafter checked in batches cannot pay on today's kernels; a drafter can
only help Flint as the pipeline's guesser. At 8,192 tokens of context the attention kernel is the whole growth of a
pass (+5.34 and +5.31 ms).

## Result

### The step: one full pass for every token not guessed right

The preset decodes one token a pass and keeps up to four guesses of the next tokens in flight behind it
(`--pipe-depth 4`): a right guess is already on its way when its token is sampled, a wrong one is dropped with the
guesses behind it. The repeated request (pass 2), ms:

| Context | The step, no profiler (R168, ms a token) | Guesses, right, wrong (each load) | Tokens not guessed right | R168's generation time over those tokens | One-token pass, the cards' GPU time: group profile (`g1`, `g2`) | the same, host profile (`h1`, `h2`) | One-token step without pipelining (`s8`, `s4`, `s2`) |
|---:|---|---|---:|---|---|---|---|
| 0 | 18.09, 18.09 | 352, 80, 68 | 176 | 26.31, 26.32 | 25.35, 25.61 | 25.95, 25.94 | 27.29, 27.42, 28.12 |
| 8,192 | 19.48, 19.50 | 567, 93, 120 | 163 | 30.60, 30.63 | 30.30, 30.22 | 30.57, 29.94 | 40.36 (`s2`; 8 steps) |

The step without a profiler is [R168](../2026-11-16_r168_flint_f4t_deep/README.md)'s two v4 loads of 2026-11-16: the
same rig, binary (`llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`), preset, file and prompts. They wrote the same
replies as every load here (ids md5 `576dc466…` at depth 0, `e20d7b7e…` at 8,192) and their server logs count the
same guesses. This run has no load without a profiler (see "The profilers" below).

- **A token not guessed right costs a whole pass.** R168's 4,631.065 and 4,631.941 ms for 256 tokens are 26.31 and
  26.32 ms over each of the 176 tokens that were not right guesses, against 25.35 and 25.61 ms of the cards' GPU time
  in a one-token pass (group profile). At 8,192 tokens: 30.60 and 30.63 ms over 163 tokens, against 30.30 and 30.22.
  The right guesses therefore come nearly free (est.: the counts and the times leave them under 1 ms each).
- **What the guesses cost:** every card evaluates 2.06 one-token passes a generated token at depth 0 (529 and 528
  evaluations on card 0 for 256 tokens) and 2.85 at 8,192 (731 and 729): the dropped guesses. They run on cards that
  the pass in flight has already left, and the time model leaves them no room on the critical path (est.).
- **So, for the preset (est., from the counts):** a millisecond off a one-token pass takes 0.69 ms off a token at depth
  0 (176 of 256) and 0.64 at 8,192 (163 of 256); each more right guess in 256 tokens takes a whole pass off the reply
  (26.31 ms at depth 0, 30.60 at 8,192), 0.10 and 0.12 ms a token.
- **Without pipelining** a one-token step on the speculative path takes 27.29, 27.42 and 28.12 ms at depth 0 (the
  steps where the n-gram draft found nothing; the cards' GPU time 25.79, 25.90 and 26.61, the rest 1.50, 1.52 and
  1.55: the hops, the host's part, sampling and the n-gram lookup). The guesses take the token from there to 18.09.

### A one-token pass by part

GPU time of a one-token pass by part, all nine cards, ms, the repeated request (the group profile, [map.md](map.md)
table 3; the profiler's own 7.6 µs per launch group subtracted):

| Part | Depth 0 (`g1`, `g2`) | 8,192 (`g1`, `g2`) | 8,192 − 0 |
|---|---|---|---|
| attention products (q, k, v, output; Q8_0 and Q5_0) | 8.04, 8.13 | 8.02, 7.98 | −0.02, −0.15 |
| routed experts (gate, up and down; Q8_0 and Q5_0) | 7.82, 7.92 | 7.73, 7.71 | −0.09, −0.21 |
| shared expert | 2.62, 2.67 | 2.73, 2.73 | +0.11, +0.06 |
| output layer (Q5_0, 2,048 × 150,144) | 2.03, 2.01 | 2.03, 2.02 | 0.00, +0.01 |
| attention (`FLASH_ATTN_EXT`) | 1.97, 1.99 | 7.31, 7.30 | **+5.34, +5.31** |
| router (F16) | 0.91, 0.94 | 0.80, 0.79 | −0.11, −0.15 |
| dense feed-forward (the dense layers) | 0.55, 0.55 | 0.54, 0.54 | −0.01, −0.01 |
| other operations without weights | 1.40, 1.41 | 1.13, 1.14 | −0.27, −0.27 |
| **the pass** | **25.35, 25.61** | **30.30, 30.22** | **+4.95, +4.61** |

- **The products of weights are 21.97 and 22.22 ms of the 25.35 and 25.61** at depth 0: the attention products 8.04
  and 8.13, the routed experts 7.82 and 7.92 (the fused gate and up 4.17 and 4.24, the down 2.53 and 2.56 on the Q8_0
  layers), the shared expert, the head and the router.
- **The attention is the whole growth to 8,192 tokens:** +5.34 and +5.31 ms a pass, against +4.95 and +4.61 for the
  pass (map.md table 3b: no other label grows more than 0.07). On the first request at the depth it is 6.94 and 6.89.
  The trace's label does not name the kernel; Flint's own Pascal kernel (FAP) is off by default, so this is
  llama.cpp's.
- **The last card is the slowest:** card 8 takes 4.42 and 4.39 ms a pass (the head's 2.03 and 2.01 of it), the others
  2.33 to 3.02 (map.md table 3c); at 8,192 card 8 takes 5.14 and 5.14, card 0 3.97 and 3.98.

Each card's GPU ms a one-token pass (group profile, the repeated request):

| Card | 0 (`g1`, `g2`) | 8,192 (`g1`, `g2`) |
|---:|---|---|
| 0 | 3.02, 3.01 | 3.97, 3.98 |
| 1 | 2.81, 2.91 | 3.26, 3.27 |
| 2 | 2.45, 2.45 | 2.79, 2.87 |
| 3 | 2.51, 2.51 | 3.12, 3.12 |
| 4 | 2.61, 2.82 | 3.16, 3.06 |
| 5 | 2.45, 2.45 | 2.74, 2.74 |
| 6 | 2.34, 2.33 | 2.96, 2.96 |
| 7 | 2.74, 2.74 | 3.15, 3.08 |
| 8 | 4.42, 4.39 | 5.14, 5.14 |

### Passes of several tokens

How Flint gets one: its preset never runs a pass of more than one token (the pipelined guesses are one-token
evaluations). A pass of k tokens is a batched verify: the arms `s8`, `s4` and `s2` add the server's n-gram drafts
(`ngram-simple`, a one-token key, k − 1 drafted tokens) and synthetic acceptance of every draft
(`--spec-synth-len k`), so each server step verifies 1 + the drafted tokens through the path a drafter's verify takes
(logits for every row, the sampling chain, the acceptance); a slot that can speculate never pipelines. Their replies
are not valid text; only the times count. The step trace gives each step's width, so every step is paired with the
cards' GPU time of its pass ([widths.md](widths.md), `run/width_curve.py`). The repeated request, medians over the
steps, ms:

| Context | k | Load (steps) | The step | The cards' GPU time | The rest | The step a token |
|---:|---:|---|---|---|---|---|
| 0 | 1 | `s8` (18), `s4` (36), `s2` (45) | 27.29, 27.42, 28.12 | 25.79, 25.90, 26.61 | 1.50, 1.52, 1.55 | 27.29, 27.42, 28.12 |
| 0 | 2 | `s4` (63), `s2` (105) | 74.05, 75.28 | 71.60, 72.91 | 2.39, 2.40 | 37.03, 37.64 |
| 0 | 4 | `s4` (23) | 144.83 | 140.83 | 4.06 | 36.21 |
| 0 | 5 | `s8` (1) | 195.57 | 182.31 | 13.26 | 39.11 |
| 0 | 8 | `s8` (29) | 281.04 | 273.59 | 7.38 | 35.13 |
| 8,192 | 1 | `s2` (8) | 40.36 | 31.52 | 8.83 | 40.36 |
| 8,192 | 2 | `s2` (123) | 77.46 | 74.74 | 2.73 | 38.73 |
| 8,192 | 4 | `s4` (63) | 149.29 | 144.75 | 4.57 | 37.32 |
| 8,192 | 6 | `s8` (7) | 222.38 | 215.91 | 6.47 | 37.06 |
| 8,192 | 8 | `s8` (26) | 289.15 | 280.68 | 8.30 | 36.14 |

The first request at each depth: [widths.md](widths.md) (at depth 0, 74.02 at k = 2, 146.17 at k = 4, 280.99 at
k = 8).

- **A pass of k tokens costs more than k passes of one token.** At depth 0 a pass of 2 tokens takes 2.70 times a
  one-token step in the same load (74.05 against 27.42, `s4`) and 2.68 times (75.28 against 28.12, `s2`); 4 tokens
  5.28 times (144.83 against 27.42); 8 tokens 10.30 times (281.04 against 27.29, `s8`). Every card grows alike: card 8
  4.47, 12.02, 22.74 and 41.09 ms at k = 1, 2, 4 and 8, card 0 3.32, 8.34, 15.60 and 31.78 (widths.md).
- **Why, est.:** Flint's file is Q8_0 and Q5_0, and the preset sets none of the switches that read those types once for
  2 to 4 tokens (`GGML_CUDA_MMVQ_PASCAL_MT`, `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0`, `GGML_CUDA_MMVQ_PASCAL_GLU_MT`;
  [docs/switches.md](../../switches.md)), so these passes take llama.cpp's own small-batch paths, and each token picks
  its own experts. This run has no group profile at k > 1, so the kernels are not named (row R199, Q1582).
- **The rest grows about 0.84 ms a token** (1.50 at k = 1 to 7.38 at k = 8, `s8`): k rows cross each of the eight
  hops, and the sampling chain and the acceptance handle k rows.
- **For a drafter (Q1543):** a step verified in one pass of k tokens breaks even with the preset at T(k) / 18.09
  tokens a step at depth 0: 4.09 at k = 2 (74.05), 8.01 at k = 4 (144.83), 15.54 at k = 8 (281.04); at 8,192 against
  19.48: 3.98 (77.46), 7.66 (149.29) and 14.84 (289.15). Each is more than k, so no drafter verified in batches of up
  to 8 can pay on v4.0's kernels, even with every draft accepted: 35.13 ms a token at k = 8 against the preset's 18.09.

### The profilers

- **The host profile (`GGML_CUDA_PROFILE_HOST`) serializes Flint's pipeline.** At each card's next evaluation it waits
  for the card's previous one before it writes its time, so the host cannot queue guesses ahead: under it the preset
  takes 34.98 and 34.90 ms a token at depth 0 (`h1`, `h2`, the repeated request) and 47.95 and 47.03 at 8,192,
  against R168's 18.09 and 19.48 without it. Its GPU time per evaluation is right (its one-token pass sums to 25.95 and
  25.94 against the group profile's 25.35 and 25.61); its step times are not the preset's. On the speculative arms each
  step waits for its pass anyway, so there it costs little. Row C877 (Q1583) makes it wait-free.
- **The group profile synchronizes after every evaluation:** 71.71 and 72.24 ms a token at depth 0 (`g1`, `g2`),
  115.05 and 115.14 at 8,192. Its GPU times per launch group are what this folder uses from it.
- **No token changed:** every load of the preset wrote the same replies at both depths and passes as R168 (ids md5
  `576dc466…`, `e20d7b7e…`), and the speculative arms wrote one reply per width and depth in both passes
  ([requests.csv](requests.csv)).

### The cards' clocks (C832's card log)

Every card was at stock (no offset, 75 W, no clock lock; read before the run). The card log
([scripts/card_log.py](../../../scripts/card_log.py), every 0.5 s, 10,191 samples a card, no gap over 2 s) shows the
cards' own clock management at work: over the run 51.49 % (card 4) to 59.97 % (card 0) of each card's busy samples ran
below 1,800 MHz, many at 1,354 MHz, almost all with no throttle reason (power cap in 16 to 39 busy samples a card);
301 stretches of 2 s or more busy below 1,800 MHz ([cards/all.md](cards/all.md)). Per load (busy samples below
1,800 MHz, the lowest and the highest card; peak power; peak temperature):

| Load | Below 1,800 MHz | Peak W | Peak °C |
|---|---|---|---|
| `h1` | 46.26 % (card 4), 56.25 % (card 8) | 74.60 to 76.56 | 35 to 42 |
| `g1` | 64.69 % (card 4), 69.97 % (card 0) | 73.41 to 76.64 | 35 to 41 |
| `s8` | 36.31 % (card 7), 46.98 % (card 0) | 73.00 to 75.35 | 35 to 42 |
| `s4` | 41.97 %, 53.80 % | 74.13 to 77.27 | 35 to 42 |
| `s2` | 43.84 %, 52.60 % | 74.46 to 77.89 | 35 to 41 |
| `h2` | 44.29 %, 55.43 % | 74.25 to 78.18 | 35 to 41 |
| `g2` | 64.38 %, 70.51 % | 72.77 to 77.40 | 35 to 42 |

The group-profile loads, whose cards wait after every evaluation, spend the most busy time at low clocks. The same
behaviour was in R168's loads (same rig and settings), so the times here and there compare; what the clocks cost is
the clock rows' work (R186, R188).

### What the map prices

| Row | What it changes | Before this run | Now |
|---|---|---|---|
| C135 (Q675) | Flint's Pascal decode attention without local memory | ceiling est. 9.2 ms a token at 32,768 tokens (Flint Pascal Q5 GGUF); nothing at 8,192 | the attention takes 7.31 and 7.30 ms a pass at 8,192 (1.97 and 1.99 at depth 0), the whole growth (+5.34 and +5.31): est. up to 4.65 ms a token at 8,192 (163 of 256 × 7.31; 23.9 % of 19.48) if it took no time, 3.40 of it the growth. **Worth more than priced** |
| C848 (Q1511) | staged hops in decode only | 0.42 ms a token at the start (R165: 17.54 and 17.57 against 18.02 and 17.92) | a pass not guessed right spends est. 0.71 and 0.96 ms beyond the cards' GPU time at depth 0 (26.32 and 26.31 against 25.61 and 25.35); the staged hops' 0.42 a token is 0.61 a pass, most of it. **Stands at 0.42**; little else to win between the cards in decode |
| C640 (Q1125) | the Q8_0 cast of the data handed between cards | Flint est. up to 0.5 ms a token (by bytes) | on top of C848, est. at most 0.35 a pass (0.96 − 0.61), 0.24 a token. **Worth less** in decode; in passes of k tokens the rest grows 0.84 ms a token, but those passes do not pay on Flint |
| FN113 (Q1543) | Inco AI's DFlash2 drafter for Flint | needs Flint's passes of 2 to 8 tokens to be cheap; never timed on v4.0 | 74.05 and 75.28 ms at 2 tokens, 144.83 at 4, 281.04 at 8 (depth 0): a verify in one pass breaks even only at 4.09, 8.01 and 15.54 tokens a step. **The batched verify is a no-go on v4.0's kernels**; left is the drafter as the pipeline's guesser: every 10 more right guesses in 256 tokens take est. 1.03 ms off a token at depth 0 (26.31 × 10 / 256) |
| Q342 | an adaptive pipe depth by guess source | est. from guess rates | the dropped guesses (1.06 and 1.85 extra evaluations a token at depth 0 and 8,192) cost est. nothing on the critical path; what pays is more right guesses (80 of 256 at depth 0, 93 at 8,192) |
| R199 (Q1582), new | Flint's passes of 2, 4 and 8 tokens with the read-once switches for Q8_0 and Q5_0, by kernel | — | decides whether any drafter can be verified in batches on Flint (break-even above) |
| C877 (Q1583), new | the host profile without waiting, so a pipelined step can be profiled | — | today it turns Flint's 18.09 ms a token into 34.98 and 34.90 |

## What ran

- **Rig and build:** a nine-card P106-090 rig (Manli M-P106L9-N6G, the stock 4 GB host), nothing else running, every
  card at stock (no clock offset on any API, 75 W limit, no clock lock; read before the run); Miner Llama v4.0 (the
  rig build of `pascal-p106-v4-b12360-c828` `eb312b7bc`, whose tree is the tag `miner-llama-v4.0`'s), `llama-server`
  md5 `e7afd26db01f4c671549ef12c504ada3`.
- **Model and line:** Flint Pascal Q8 GGUF (`p065`, 40,431,236,544 bytes, md5 `8387b2e585a7b9cc10f40cb6b4531694`), the
  v4 preset exactly ([loads.tsv](loads.tsv) has each load's line).
- **The driver:** [bench/deep_lane](../../../bench/deep_lane/README.md) through [run/r176.sh](run/r176.sh), one
  detached run, 16:43:06 to 18:08:03. Per depth (0, 8,192) a load sends a prime (the prompt without its 64-token
  tail; none at depth 0), decode pass 1 (the first request at the depth: 256 tokens at temperature 0, end of text
  ignored) and decode pass 2 (the same request again), after the lane's 32-token warm-up request. The client waits
  4 s before every request (`PAUSE=4`), so each profile cuts into one piece per request.
- **The arms, each its own load, in this order:**

  | Load | What it adds to the preset |
  |---|---|
  | `h1` | `GGML_CUDA_PROFILE_HOST` (each card's host and GPU time per evaluation) and `LLAMA_STEP_TRACE` (the server's host marks per step) |
  | `g1` | `GGML_CUDA_PROFILE` with `GGML_CUDA_PROFILE_MODE=group` (every launch group timed, the stream synchronized after every evaluation) |
  | `s8`, `s4`, `s2` | `h1`'s profiler and `--spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m k−1 --spec-synth-len k`, k = 8, 4, 2 |
  | `h2`, `g2` | as `h1`, `g1` |

  Each load spent 587 to 590 s loading the 40 GB file from the rig's USB disk (`load_s` in loads.tsv). Before every
  load: no llama process and at most 100 MiB on any card; every load under the RAM watchdog (stop below 60 MiB
  available), which stopped none. Lowest MemAvailable per load: 1,153 (`h1`), 1,110 (`g1`), 1,214 (`s8`), 1,226
  (`s4`), 1,245 (`s2`), 1,186 (`h2`) and 1,126 MiB (`g2`).
- **No warm-up load:** a Flint load takes ten minutes, so the first timed load is the run's first; the comparisons
  here are within a load or against R168's loads.
- **The n-gram drafts' widths:** the drafts are not always k − 1 tokens long (the one-token key finds a shorter
  continuation, or none): `s8` verified 5.224 tokens a step at depth 0 and 7.314 at 8,192, `s4` 2.065 and 3.938, `s2`
  1.695 and 1.925 ([requests.csv](requests.csv)); widths.md times each width on its own.

## How it was read

On the operator's laptop after the run, from the copied results, with standard-library Python 3
([run/tables.sh](run/tables.sh), 7 s):

1. [run/flint_map.py](run/flint_map.py) cuts each profile and step trace into one piece per request at the idle gaps
   (a load's last N pieces are its N requests, warm-up first) and reads the server logs' guess counts. Per request: the
   host profile's evaluations per card and each card's median GPU time over its typical evaluations (within 20 % of the
   card's median), summed to a pass; the group profile's typical evaluations per card (the same rule), 7.6 µs
   subtracted per launch group, every part and launch-group label (graph node numbers folded, so each operation is one
   label over all layers) averaged and summed over the cards ([map.md](map.md), every value in `map.tsv.gz`).
2. [run/width_curve.py](run/width_curve.py) pairs each speculative step of `s8`, `s4` and `s2` (its width from the step
   trace) with each card's evaluation of that step (the k-th step with each card's k-th evaluation; prompt steps,
   which carry no drafted count, left out; the file's very last evaluation is never written, so the last request's last
   step is left out) and gives per width the step, the cards' GPU time and the rest ([widths.md](widths.md),
   `widths.tsv`, every step in `widths_steps.tsv.gz`).
3. [scripts/card_table.py](../../../scripts/card_table.py) summarizes C832's card log over the run and per load
   ([cards/](cards/)).

## Commands

```bash
# from the repository root on a machine that runs only ssh and tar; the rig's paths in the environment
RIG=<rig alias> WORK=<run folder on the rig> BIN=<v4.0 build on the rig>/bin/llama-server \
  MODEL=<models>/flint-pascal-Q8.gguf TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> DRY=1 STEP=start \
  bash docs/data/2026-11-17_r176_flint_step_map/run/r176.sh
# then the same without DRY=1 (about 1 h 25 min on the rig), STEP=wait, then STEP=collect OUT=<folder>

# the tables, on any machine with Python 3: S this repository, OUT the run's results, RES the tables (seconds)
S=<this repository> OUT=<results> RES=<tables> bash docs/data/2026-11-17_r176_flint_step_map/run/tables.sh
```

The two group-profile traces (262 MB together) are not published. Their sizes and md5s:

| Trace | Bytes | md5 |
|---|---:|---|
| `gp-g1.csv` | 131,214,767 | `95e6e9e6ece005121873b09f0b371957` |
| `gp-g2.csv` | 131,226,654 | `f5acd37db6c0e82a1de9b12301e147f0` |

## Files

| File | What it holds |
|---|---|
| [map.md](map.md), `map.tsv.gz` | `flint_map.py`: every request, the preset's one-token step under the host profile with the guesses, the one-token pass by part and label, what grows with depth, each card's time; every value in long form (table, load, depth, request, name, value) |
| [widths.md](widths.md), [widths.tsv](widths.tsv), `widths_steps.tsv.gz` | `width_curve.py`: passes of k tokens per load, depth, request and width, each card's time; every step |
| [requests.csv](requests.csv) | every request of every load: prompt and generation times, drafted and accepted tokens, the reply's md5 |
| [loads.tsv](loads.tsv) | every load's environment and server line as it ran (paths as `<OUT>`, `<MODELS>`, `<BIN>`) |
| [table.md](table.md), [deep_classes.tsv](deep_classes.tsv) | the deep lane's own tables of the run (its classes compare profiled loads and mean nothing here) |
| [cards/](cards/) | `card_table.py` over the run (`all.md`) and per load |
| [map_notes.txt](map_notes.txt) | how each file cut into requests |
| [run/r176.sh](run/r176.sh) | the run: the arms on the deep lane, start, wait, status, collect |
| [run/tables.sh](run/tables.sh), [run/flint_map.py](run/flint_map.py), [run/width_curve.py](run/width_curve.py) | the reading |

## Limits

- **One greedy reply per depth.** The guesses, and so the step, are this reply's: 80 right of 256 tokens at depth 0,
  93 at 8,192. R168 counted 129 and 165 right on other replies at other depths.
- **The preset's step is R168's**, measured a day earlier on the same rig, binary, file and prompts with the same
  replies and guesses; the split of it (a full pass per token not guessed right) is an estimate from those counts and
  this run's pass times.
- **No group profile at k > 1**, so the passes of several tokens are split into the cards' GPU time and the rest only.
- **At 8,192 tokens only 8 one-token steps** came in the speculative arms (`s2`), with a rest of 8.83 ms against 1.50
  to 1.55 at depth 0; the one-token pass's GPU time there (31.52) matches the profiles' 29.94 to 30.57.
- **The cards' clocks moved by themselves** during every load (above); the comparisons pair loads under the same
  conditions.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), v4.0. Model: [Flint](https://huggingface.co/RecursiveSI/flint) (Apache-2.0) as
my Flint Pascal Q8 GGUF ([models/flint](../../../models/flint/README.md)); the prompts are the deep lane's recorded
agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables hold times, counts, kernel labels and
md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
