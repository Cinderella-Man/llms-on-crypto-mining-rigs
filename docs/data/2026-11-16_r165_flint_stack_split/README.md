# Flint's candidate stack split at depth (R165, Q1486; 2026-11-16)

What this measures: which of Flint's candidate switches costs the deep prompts and decode that the candidate stack
lost in R161's deep lane ([2026-11-14_r161_flint_release_deep](../2026-11-14_r161_flint_release_deep/README.md)).
There, the v4 preset with every untried candidate of [release/flint-v4](../../../release/flint-v4/README.md) stacked
decoded −1.4 % and read prompts −14.8 % slower than the plain preset over six depths, −20.6 % on the prompt at
32,768 tokens, and gained only at depth 0. This run times the candidates one an arm against the plain preset, on nine
P106-090 cards, Flint Pascal Q8 GGUF (`p065`, 40.4 GB) at the full 262,144-token context, in two runs:

| Run | Arm | Settings | State |
|---|---|---|---|
| R165a, R165b | `v4` (the control) | the v4 preset ([release/flint-v4/settings.env](../../../release/flint-v4/settings.env)) as it is: v3.0's line and three switches | |
| R165a | `v4gf` | `v4` plus `GGML_CUDA_GRAPHS_FORCE=1` | done 2026-11-16 |
| R165a | `v4hs` | `v4` plus `GGML_CUDA_HOP_STAGED=1` | done 2026-11-16 |
| R165b | `v4ka` | `v4` plus `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1` | done 2026-11-16 |
| R165b | `v4ck` | `v4` plus `--ctx-checkpoints-vram 4` | done 2026-11-16 |
| R165b | `v4tp` | `v4` plus `--ubatch-taper short=600:64` | done 2026-11-16 |

**Result: the keep-alive is the stack's deep cost.** Alone, `GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1`
reads every prompt chunk from 4,096 tokens on slower than the plain preset: −20.1 % at 32,768 and −16.8 % at 49,152,
where the stack lost −20.6 % and −17.3 %. It also decodes slower at every depth from 4,096 on. Over all depths it
decodes −3.0 % and reads prompts −16.5 %, a loss on both. The other four candidates are no loss over all depths, decode
or prompts:
- forced graphs (`v4gf`) at no depth;
- staged hops (`v4hs`) at every depth but one, the 8,192-token chunk (−7.7 %);
- the checkpoints on the cards (`v4ck`) at no depth, with gains on the four deepest prompt chunks;
- the taper (`v4tp`) at no depth.

`--ctx-checkpoints-incall`, the stack's seventh switch, gets no arm: the server turns it off on Flint at load
(`not with sliding-window layers`, R161). `LLAMA_SCHED_PER_SIZE_RESERVE=2` was never in the stack: it acts only when the
scheduler re-plans, and no load of R161 re-planned. Every arm runs `v4-rc2` (`pascal-p106-v4` `eb82916d3`, upstream
b12360; the rig build's `llama-server` reports `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`, whose tree is
`v4-rc2`'s). Registry row Q1486 in [quants/experiments.md](../../../quants/experiments.md).

## R165a: forced graphs and staged hops

**Neither of the two is what cost the stack its deep prompts.** The stack lost most at 32,768 and 49,152 tokens:
prompts −20.6 % and −17.3 %, decode −2.0 % and −3.1 % (R161). At those depths both candidates read prompts within
the plain preset's spread:
- at 32,768: forced graphs 334.7 and 334.5 tok/s, staged hops 335.1 and 334.5, against `v4`'s 338.6 and 332.6;
- at 49,152: 218.8 and 217.9, and 217.9 and 218.3, against 220.0 and 218.2.

There, forced graphs decode as fast as `v4`, and staged hops decode faster. The stack's loss lies in R165b's three
candidates: the keep-alive, the checkpoints on the cards, or the taper.

**`v4gf` (`GGML_CUDA_GRAPHS_FORCE=1`) is slower at no depth.**
- Decode is negligible at every depth. Over all depths it is a gain: 40.50 and 40.55 tok/s against 40.42 and 40.40
  (+0.3 %, beyond the spreads 0.04 and 0.02).
- Prompts are a gain on the 16,384 chunk: 516.4 and 514.5 tok/s against 493.8 and 483.1 (+5.5 %). They are negligible
  at every other depth.
- All five prompt chunks (53,200 tokens) read at 325.7 and 321.9 tok/s against 324.2 and 320.3 (+0.5 %, negligible).

**`v4hs` (`GGML_CUDA_HOP_STAGED=1`) decodes faster, and reads one prompt chunk slower.**
- Decode is a gain at every depth but 8,192: +5.3 % at 0, +1.7 % at 4,096, +1.2 % at 16,384, +0.9 % at 32,768 and
  +0.8 % at 49,152. At 8,192 it is negligible (+0.5 %).
- Over all depths decode runs 40.99 and 41.03 tok/s against 40.42 and 40.40 (+1.5 %, a gain).
- Prompts are a gain on the 64-token prompt at depth 0: 88.7 and 89.4 tok/s against 86.4 and 87.1 (+2.6 %).
- They are a loss on the 8,192 chunk: 446.1 and 437.4 tok/s against 478.2 and 479.2 (−7.7 %).
- They are negligible at 4,096, 16,384, 32,768 and 49,152.
- All five prompt chunks read at 321.4 and 318.3 tok/s against 324.2 and 320.3 (−0.7 %, negligible: the spreads are
  3.1 and 3.8).

Every value is one load. Run on one rig on 2026-11-16, 10:30:04–12:09:07: a warm-up load of `v4`, then
`v41 gf1 hs1 v42 gf2 hs2`, every load through. The RAM watchdog did not act.

### Per depth

Decode: 256 greedy tokens twice per depth, a load's speed being both passes' tokens over their time. Prompts: the
prime's chunk, read on top of the previous depth's checkpoint (at depth 0 the 64-token tail on an empty cache; the
chunk read and the tokens cached are the same in every arm). `v4`'s loads are listed `v41`, `v42`; `v4gf`'s `gf1`,
`gf2`; `v4hs`'s `hs1`, `hs2`. Each class is against `v4`.

| Depth (tokens) | Decode tok/s, `v4` | `v4gf` | Class | `v4hs` | Class |
|---:|---|---|---|---|---|
| 0 | 50.55, 49.04 | 50.54, 50.55 | negligible (+1.5 %) | 52.31, 52.52 | gain (+5.3 %) |
| 4,096 | 77.34, 77.45 | 77.40, 77.44 | negligible (+0.0 %) | 78.65, 78.74 | gain (+1.7 %) |
| 8,192 | 48.88, 49.46 | 49.26, 49.42 | negligible (+0.3 %) | 49.47, 49.35 | negligible (+0.5 %) |
| 16,384 | 67.28, 67.48 | 67.48, 67.48 | negligible (+0.2 %) | 68.08, 68.26 | gain (+1.2 %) |
| 32,768 | 26.39, 26.50 | 26.46, 26.50 | negligible (+0.1 %) | 26.68, 26.70 | gain (+0.9 %) |
| 49,152 | 23.52, 23.57 | 23.51, 23.53 | negligible (−0.1 %) | 23.73, 23.74 | gain (+0.8 %) |
| all | 40.42, 40.40 | 40.50, 40.55 | gain (+0.3 %) | 40.99, 41.03 | gain (+1.5 %) |

| Depth (tokens) | Read, on cached | Prompt tok/s, `v4` | `v4gf` | Class | `v4hs` | Class |
|---:|---|---|---|---|---|---|
| 0 | 64 on 0 | 86.4, 87.1 | 85.5, 86.4 | negligible (−0.9 %) | 88.7, 89.4 | gain (+2.6 %) |
| 4,096 | 4,032 on 0 | 519.0, 523.4 | 541.9, 464.8 | negligible (−3.4 %) | 532.6, 447.6 | negligible (−6.0 %) |
| 8,192 | 4,100 on 4,028 | 478.2, 479.2 | 498.5, 483.6 | negligible (+2.6 %) | 446.1, 437.4 | loss (−7.7 %) |
| 16,384 | 12,292 on 4,028 | 493.8, 483.1 | 516.4, 514.5 | gain (+5.5 %) | 498.5, 498.4 | negligible (+2.1 %) |
| 32,768 | 16,388 on 16,316 | 338.6, 332.6 | 334.7, 334.5 | negligible (−0.3 %) | 335.1, 334.5 | negligible (−0.2 %) |
| 49,152 | 16,388 on 32,700 | 220.0, 218.2 | 218.8, 217.9 | negligible (−0.4 %) | 217.9, 218.3 | negligible (−0.5 %) |
| all primes | 53,200 | 324.2, 320.3 | 325.7, 321.9 | negligible (+0.5 %) | 321.4, 318.3 | negligible (−0.7 %) |

At 4,096 tokens each candidate's second load read the chunk slower than its first: 464.8 against 541.9 tok/s for
forced graphs, 447.6 against 532.6 for staged hops. That gives spreads of 77.0 and 84.9 tok/s, and both classes
there are negligible. The sampled request (the line's own sampler, fixed seed) at 49,152 tokens runs 22.64 and 22.67
tok/s for `v4`, 22.67 and 22.67 for `v4gf` and 22.85 and 22.85 for `v4hs` ([table.md](table.md), "The sampler pass").

### The replies, the host and the cards

- **Every arm writes the same greedy reply on every pass and load at every depth.** Both candidates change speed only.
- **No target graph reallocated** on any of the 24 requests of any load. Every timed request gave 256 tokens, and
  every load read the same prompt ids.
- **The host's lowest MemAvailable** was 958 and 960 MiB for `v4`, 952 and 949 for `v4gf`, and 937 and 931 for
  `v4hs`. MemAvailable after loading was 1,440 and 1,438, 1,436 and 1,436, and 1,414 and 1,417 MiB: staged hops left 26 and
  21 MiB less than `v4`'s loads, the pinned host ring they keep. Swap in use peaked at 293 or 294 MiB in every load. The RAM watchdog did not
  act ([prompt_depth.md](prompt_depth.md)).
- **The cards after loading** held the same memory in every load: 48,613 MiB over the nine, cards 1 and 4 at 5,923 MiB.
- **Each server printed the switches it was given:** `switches: 3 set` for `v4` (release/flint-v4's three), `4 set`
  for `v4gf` and `v4hs`, the fourth being the candidate.
- **The control beside R161's:** this run's `v4` decoded 40.42 and 40.40 tok/s over all depths (R161's `v4`: 40.25
  and 40.36). It read all primes at 324.2 and 320.3 tok/s (R161: 318.9 and 321.3). These are different runs, so they
  get no class.

## R165b: the keep-alive, the checkpoints on the cards and the taper

**The keep-alive (`v4ka`) costs the stack its deep prompts and decode.** It loses at every depth from 4,096 tokens on,
by a margin that matches the stack's own loss. On the prompt chunks:
- at 32,768 tokens, 270.0 and 269.4 tok/s against `v4`'s 338.5 and 337.0 (−20.1 %; the stack in R161: −20.6 %);
- at 49,152 tokens, 181.7 and 181.3 against 218.0 and 218.1 (−16.8 %; the stack: −17.3 %).

Its decode is a loss at 4,096, 8,192, 16,384, 32,768 and 49,152 tokens (−3.5, −4.0, −4.0, −4.3 and −5.0 %), each
beyond both arms' spreads. Its one gain is the 64-token
prompt at depth 0 (+27.3 %), as the stack's was (+31.3 %).

**`v4ka` (`GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1`)**
- Decode is a loss at 4,096 (−3.5 %), 8,192 (−4.0 %), 16,384 (−4.0 %), 32,768 (−4.3 %) and 49,152 (−5.0 %). At
  depth 0 it is negligible (+6.4 %, within `v4`'s spread of 4.49).
- Over all depths it decodes 39.06 and 38.94 tok/s against 40.55 and 39.86 (−3.0 %, a loss).
- Each decode step from 4,096 on takes longer: 13.9 and 14.0 ms against 13.4 and 13.3 at 4,096, 45.5 and 45.4 ms
  against 43.2 and 43.2 at 49,152.
- Prompts are a gain at depth 0: 96.1 and 105.7 tok/s against 85.3 and 73.2 (+27.3 %).
- They are a loss on every chunk from 4,096 on: −13.9, −9.7, −11.2, −20.1 and −16.8 %.
- All five chunks (53,200 tokens) read at 267.7 and 266.9 tok/s against 321.1 and 319.0 (−16.5 %, a loss).

**`v4ck` (`--ctx-checkpoints-vram 4`) is slower at no depth and reads the deep chunks faster.**
- Decode is negligible at every depth but 49,152. There it is a gain by the rule, 23.58 and 23.58 tok/s against
  23.57 and 23.57, with both spreads 0.00.
- Over all depths it decodes 40.67 and 40.62 against 40.55 and 39.86 (+1.1 %, negligible).
- Prompts are a gain on the four chunks read on a checkpoint: 8,192 (474.9 and 516.3 against 448.4 and 425.7,
  +13.4 %), 16,384 (+3.9 %), 32,768 (341.0 and 340.4 against 338.5 and 337.0, +0.9 %) and 49,152 (219.6 and 222.5
  against 218.0 and 218.1, +1.4 %).
- Prompts are negligible at depth 0 (+15.0 %) and on the 4,096 chunk (−5.3 %): there its two loads read 440.8 and
  544.1, a spread of 103.3.
- All five chunks read at 322.2 and 329.7 tok/s against 321.1 and 319.0 (+1.8 %, negligible).
- Each restore of the 52.5 MiB checkpoint took 2.0, 2.1 or 2.2 ms from the cards, against 291.9 or 292.0 ms from host
  memory in `v4`'s loads.
- The host keeps 212 and 216 MiB more at its lowest: 1,165 and 1,144 MiB against 949 and 932. That is about the four
  checkpoints of 52.5 MiB that now live on the cards.
- The cards' peak over a load rose by 32 MiB on cards 1, 4 and 7 and by 24 MiB on the other six: cards 1 and 4 reached
  5,985 MiB against 5,953 (49,147 MiB over the nine against 48,907).

**`v4tp` (`--ubatch-taper short=600:64`) is slower at no depth.**
- Decode is negligible at every depth. Over all depths: 40.28 and 40.41 tok/s against 40.55 and 39.86 (+0.4 %).
- Prompts are a gain on the 8,192 chunk (473.4 and 478.9 against 448.4 and 425.7, +9.0 %) and the 16,384 chunk
  (493.0 and 496.0 against 488.4 and 479.7, +2.2 %). They are negligible elsewhere. All five chunks: 321.0 and 324.5
  tok/s (+0.8 %, negligible).
- The taper reads a call in micro-batches of 64 when the call has at most 600 tokens, nothing is in flight when it
  starts, and the server waits for it (it has outputs, or is shorter than `-b`). A prime is read in calls of at most
  512 tokens (`-b 512`), so its last call, and a call cut short at a checkpoint, can qualify. The 64-token prompts are
  one micro-batch with or without the taper. The server keeps no count of the calls it split, so how many it split
  here is not recorded.
- Part of the 8,192 gain is the control's: here `v4` read that chunk at 448.4 and 425.7 tok/s, where R165a's `v4`
  read it at 478.2 and 479.2.
- Its greedy replies are `v4`'s at every depth. That does not show the taper leaves Flint's numbers alone, since the
  run does not record which calls it split. On Ornith the taper changes the replies (FN93c,
  [2026-11-16_fn93c_ornith_taper_kld](../2026-11-16_fn93c_ornith_taper_kld/README.md)). Calls of 65 to 600 tokens,
  where it acts most, come with the tool results of the agent check.

Every value is one load. Run on one rig on 2026-11-16, 12:21:13–14:31:07: a warm-up load of `v4`, then
`v41 ka1 ck1 tp1 v42 ka2 ck2 tp2`, every load through. The RAM watchdog did not act.

### Per depth (R165b)

As in R165a. `v4ka`'s loads are listed `ka1`, `ka2`; `v4ck`'s `ck1`, `ck2`; `v4tp`'s `tp1`, `tp2`. Each class is
against `v4`.

| Depth (tokens) | Decode tok/s, `v4` | `v4ka` | Class | `v4ck` | Class | `v4tp` | Class |
|---:|---|---|---|---|---|---|---|
| 0 | 50.83, 46.34 | 51.78, 51.63 | negligible (+6.4 %) | 51.35, 51.09 | negligible (+5.4 %) | 48.54, 49.70 | negligible (+1.1 %) |
| 4,096 | 77.52, 77.62 | 74.95, 74.76 | loss (−3.5 %) | 77.59, 77.64 | negligible (+0.1 %) | 77.62, 77.60 | negligible (+0.1 %) |
| 8,192 | 48.93, 47.56 | 46.65, 45.98 | loss (−4.0 %) | 49.41, 49.40 | negligible (+2.4 %) | 48.98, 48.98 | negligible (+1.5 %) |
| 16,384 | 67.43, 67.34 | 64.78, 64.64 | loss (−4.0 %) | 67.47, 67.42 | negligible (+0.1 %) | 67.35, 67.38 | negligible (−0.0 %) |
| 32,768 | 26.52, 26.48 | 25.37, 25.34 | loss (−4.3 %) | 26.52, 26.47 | negligible (−0.0 %) | 26.50, 26.47 | negligible (−0.0 %) |
| 49,152 | 23.57, 23.57 | 22.39, 22.40 | loss (−5.0 %) | 23.58, 23.58 | gain (+0.0 %) | 23.55, 23.57 | negligible (−0.1 %) |
| all | 40.55, 39.86 | 39.06, 38.94 | loss (−3.0 %) | 40.67, 40.62 | negligible (+1.1 %) | 40.28, 40.41 | negligible (+0.4 %) |

| Depth (tokens) | Read, on cached | Prompt tok/s, `v4` | `v4ka` | Class | `v4ck` | Class | `v4tp` | Class |
|---:|---|---|---|---|---|---|---|---|
| 0 | 64 on 0 | 85.3, 73.2 | 96.1, 105.7 | gain (+27.3 %) | 90.2, 92.1 | negligible (+15.0 %) | 85.7, 86.4 | negligible (+8.6 %) |
| 4,096 | 4,032 on 0 | 519.7, 520.2 | 445.7, 449.7 | loss (−13.9 %) | 440.8, 544.1 | negligible (−5.3 %) | 502.2, 523.4 | negligible (−1.4 %) |
| 8,192 | 4,100 on 4,028 | 448.4, 425.7 | 384.4, 404.8 | loss (−9.7 %) | 474.9, 516.3 | gain (+13.4 %) | 473.4, 478.9 | gain (+9.0 %) |
| 16,384 | 12,292 on 4,028 | 488.4, 479.7 | 436.4, 423.7 | loss (−11.2 %) | 498.4, 507.6 | gain (+3.9 %) | 493.0, 496.0 | gain (+2.2 %) |
| 32,768 | 16,388 on 16,316 | 338.5, 337.0 | 270.0, 269.4 | loss (−20.1 %) | 341.0, 340.4 | gain (+0.9 %) | 336.4, 337.4 | negligible (−0.3 %) |
| 49,152 | 16,388 on 32,700 | 218.0, 218.1 | 181.7, 181.3 | loss (−16.8 %) | 219.6, 222.5 | gain (+1.4 %) | 217.4, 220.4 | negligible (+0.4 %) |
| all primes | 53,200 | 321.1, 319.0 | 267.7, 266.9 | loss (−16.5 %) | 322.2, 329.7 | negligible (+1.8 %) | 321.0, 324.5 | negligible (+0.8 %) |

`v4`'s second load decoded the first pass at depth 0 at 39.80 tok/s against 47.02 in its first, and read the 64-token
prompt at 73.2 against 85.3. That widens `v4`'s spreads at depth 0 to 4.49 tok/s for decode and 12.1 for that prompt,
and to 0.69 over all depths. The keep-alive's losses lie beyond them. The sampled request at 49,152 tokens runs 22.68
and 22.69 tok/s for `v4`, 21.54 and 21.52 for `v4ka`, 22.70 and 22.69 for `v4ck` and 22.67 and 22.66 for `v4tp`
([r165b_table.md](r165b_table.md), "The sampler pass").

### The replies, the host and the cards (R165b)

- **Every arm writes the same greedy reply on every pass and load at every depth**, the same as R165a's.
- **No target graph reallocated** on any of the 24 requests of any load. Every timed request gave 256 tokens, and
  every load read the same prompt ids.
- **The host's lowest MemAvailable:**
  - `v4`: 949 and 932 MiB;
  - `v4ka`: 944 and 959;
  - `v4ck`: 1,165 and 1,144;
  - `v4tp`: 960 and 943.

  MemAvailable after loading was 1,433 and 1,436 MiB for `v4`, 1,438 and 1,427 for `v4ka`, 1,436 and 1,431 for
  `v4ck`, and 1,438 and 1,426 for `v4tp`. Swap in use peaked at 294 MiB in every load but `ck2` (293) and `tp2` (295).
  The RAM watchdog did not act ([r165b_prompt_depth.md](r165b_prompt_depth.md)).
- **The cards after loading** held the same memory in every load: 48,613 MiB over the nine, cards 1 and 4 at 5,923
  MiB. Over a load they peaked at 5,953 MiB on cards 1 and 4 in `v4`, `v4ka` and `v4tp`, and at 5,985 in `v4ck`.
- **Each server printed the switches it was given:** `switches: 3 set` for `v4`, `v4ck` and `v4tp`, whose candidates
  are flags; `5 set` for `v4ka`.
- **The control beside R165a's:** this run's `v4` decoded 40.55 and 39.86 tok/s over all depths (R165a's: 40.42 and
  40.40) and read all primes at 321.1 and 319.0 tok/s (R165a: 324.2 and 320.3). These are different runs, so they get
  no class.

## Method

- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) as R161 ran it, on one build: the sampler
  pass on, the recorded agent text of the release check's `t1swift` sessions, depths 0, 4,096, 8,192, 16,384, 32,768
  and 49,152 tokens.
- **Per depth and load:** the prime (the prompt without its 64-token tail, read on top of the previous depth's
  checkpoint), then 256 tokens at temperature 0 twice (decode pass 1, the first request at the new depth, and pass 2,
  the same request again on the same cache), then 256 tokens with the server line's own sampler
  (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`) and a fixed seed. End of text is ignored.
- **Loads:** one untimed warm-up load of `v4` at depth 0, then R165a's `v41 gf1 hs1 v42 gf2 hs2` and R165b's
  `v41 ka1 ck1 tp1 v42 ka2 ck2 tp2`, each a fresh server start after a quiet check (no llama process, at most 100 MiB
  on any card) and under the RAM watchdog with its default limits. Nothing else ran on the rig.
- **The lines:** the lane reads `v4`'s from release/flint-v4/settings.env (its `PRESET_ENV` and `PRESET_ARGS`); each
  candidate arm adds its own variables (`ARM_ENV_<arm>`) or flags (`ARM_ARGS_<arm>`) and nothing else. The run script
  refuses a control with switches of its own, a candidate with none, and a candidate variable the preset already sets.
  A dry run printed the lines first. Each candidate's line is `v4`'s plus its own change: one variable for `v4gf` and
  `v4hs`, the two keep-alive variables for `v4ka`, and one flag with its value for `v4ck` and `v4tp`. `loads.tsv` and
  `r165b_loads.tsv` hold them as they ran.
- **Classes** (the lane's rule), per depth and over all depths, on a load's speed (both decode passes' tokens over
  their time; for prompts, the prime's chunk over its time): the spread is the larger of the two arms' differences
  between their two loads; an arm is a gain when its mean is better than the control's by more than the spread, a
  loss when worse by more than it, else negligible.

## Commands

From the repository root on a machine that only runs ssh and tar; the machine values in the environment:

```bash
# R165a (DRY=1 first: prints each arm's line and starts nothing)
RIG=<rig alias> WORK=<run folder on the rig> BIN=<v4-rc2 tree on the rig>/bin/llama-server \
  MODEL=<models>/flint-pascal-Q8.gguf TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> \
  ARMS="v4 v4gf v4hs" ARM_ENV_v4gf=GGML_CUDA_GRAPHS_FORCE=1 ARM_ENV_v4hs=GGML_CUDA_HOP_STAGED=1 \
  LOADS="warm:v4 v41:v4 gf1:v4gf hs1:v4hs v42:v4 gf2:v4gf hs2:v4hs" \
  bash docs/data/2026-11-16_r165_flint_stack_split/run/r165.sh
RIG=<rig alias> WORK=<run folder on the rig> STEP=wait bash docs/data/2026-11-16_r165_flint_stack_split/run/r165.sh
RIG=<rig alias> WORK=<run folder on the rig> STEP=collect OUT=<folder> \
  bash docs/data/2026-11-16_r165_flint_stack_split/run/r165.sh
# R165b: the same with
#   ARMS="v4 v4ka v4ck v4tp" ARM_ENV_v4ka="GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1"
#   ARM_ARGS_v4ck="--ctx-checkpoints-vram 4" ARM_ARGS_v4tp="--ubatch-taper short=600:64"
#   LOADS="warm:v4 v41:v4 ka1:v4ka ck1:v4ck tp1:v4tp v42:v4 ka2:v4ka ck2:v4ck tp2:v4tp"
```

The tables (the driver writes `table.md` and `classes.tsv` at the end of the run; any machine with Python 3 builds the
same):

```bash
python3 -I bench/deep_lane/deep_table.py <folder> --control v4 --out table.md --tsv classes.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v4 --out prompt_depth.md \
  --tsv prompt_depth
```

## Files

| File | Content |
|---|---|
| [table.md](table.md), [classes.tsv](classes.tsv) | R165a, `deep_table.py`'s tables, control `v4`: decode by depth and over all depths, the step, prompts, the sampler pass, the checks |
| [prompt_depth.md](prompt_depth.md), [prompt_depth_prompt.tsv](prompt_depth_prompt.tsv), [prompt_depth_generation.tsv](prompt_depth_generation.tsv), [prompt_depth_memory.tsv](prompt_depth_memory.tsv) | R165a, `prompt_depth.py`'s tables, control `v4`: prompt reading per depth and over all primes, the host's memory and each card's per load, generation as a check |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv) | R165a, per load its line and binary as it ran; per request its counts, times and reply md5s |
| [r165b_table.md](r165b_table.md), [r165b_classes.tsv](r165b_classes.tsv) | R165b, `deep_table.py`'s tables, control `v4`, as above |
| [r165b_prompt_depth.md](r165b_prompt_depth.md), [r165b_prompt_depth_prompt.tsv](r165b_prompt_depth_prompt.tsv), [r165b_prompt_depth_generation.tsv](r165b_prompt_depth_generation.tsv), [r165b_prompt_depth_memory.tsv](r165b_prompt_depth_memory.tsv) | R165b, `prompt_depth.py`'s tables, control `v4`, as above |
| [r165b_loads.tsv](r165b_loads.tsv), [r165b_requests.csv](r165b_requests.csv) | R165b, per load its line and binary as it ran; per request its counts, times and reply md5s |
| [run/r165.sh](run/r165.sh) | the run, both parts: start, wait, status, collect; the arms from the environment |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2`. Model: [Flint](https://huggingface.co/RecursiveSI/flint)
(Apache-2.0) as my Flint Pascal Q8 GGUF ([models/flint](../../../models/flint/README.md)); the prompts are the deep
lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and `requests.csv` hold
times, counts and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
