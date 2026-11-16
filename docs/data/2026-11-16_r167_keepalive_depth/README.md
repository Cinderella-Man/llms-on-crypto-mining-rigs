# The keep-alive at depth: Flash-Next, Ornith and Flint (R167, Q1492; 2026-11-16)

What this measures: whether the keep-alive slows a model deep into a context. The keep-alive is the pair of variables
`GGML_CUDA_KEEPALIVE=decode` (a one-thread block per card holds the boost clock while a split decodes) and
`GGML_CUDA_KEEPALIVE_EAGER=1` (that block ends before a kernel's first launch on a card, so the launch does not wait
for it). On Flint, R165b found that the two together read prompts 16.5 % slower over all depths and decoded 3.0 %
slower ([2026-11-16_r165_flint_stack_split](../2026-11-16_r165_flint_stack_split/README.md)). The v4 presets of
Flash-Next and Ornith ship both. Each part of R167 times one model's v4 preset on the deep lane against the same
preset with the variables changed, two loads an arm, on one build (`v4-rc2`) and one rig of nine P106-090 cards.

| Part | Model | Arms | State |
|---|---|---|---|
| R167b | Swift Qwen3.8-Flash-Next Pascal Q2 GGUF with its MTP draft ([release/flash-next-v4](../../../release/flash-next-v4/README.md)) | `v4r` the preset as it is (control); `v4nk` the preset less both keep-alive variables; `v4ne` the preset less `GGML_CUDA_KEEPALIVE_EAGER=1` alone | done, below |
| R167a | Ornith-1.5-122B Pascal N2P GGUF ([release/ornith-v4](../../../release/ornith-v4/README.md)) | the same three arms on Ornith's v4 preset: `o4` the preset as it is (control); `o4nk` less both keep-alive variables; `o4ne` less `GGML_CUDA_KEEPALIVE_EAGER=1` alone | done, [below](#r167a-ornith) |
| R167c | Flint Pascal Q8 GGUF ([release/flint-v4](../../../release/flint-v4/README.md)) | the v4 preset (which sets neither variable) against it plus `GGML_CUDA_KEEPALIVE=decode` alone | not yet run |

Registry row Q1492 in [quants/experiments.md](../../../quants/experiments.md).

## R167b: Flash-Next

**On Flash-Next the keep-alive does not cost what it costs on Flint.** Without it (`v4nk`) prompts read a little
faster and decode is slower:
- All five prompt chunks of a load (55,248 tokens) read at 219.4 and 218.2 tok/s against `v4r`'s 212.6 and 215.3
  (+2.3 %, a gain). The gain is on the three deepest chunks: +3.0 % at 16,384, +3.2 % at 32,768 and +1.4 % at 49,152
  tokens. On Flint the keep-alive cost −16.5 %.
- Decode over all six depths runs 25.72 and 25.86 tok/s against 26.15 and 26.11 (−1.3 %, a loss). Without the
  keep-alive decode gains at 4,096 (+6.1 %) and 8,192 (+3.7 %) and loses at 32,768 (−1.0 %) and 49,152 (−9.9 %:
  21.64 and 21.96 against 24.17 and 24.21).

**The eager stop is not the part that costs.** Without it (`v4ne`) prompts are within the spread at every depth (all
chunks 214.4 and 216.2 tok/s, +0.6 %, negligible). Decode over all depths runs 25.90 and 25.88 tok/s (−0.9 %, a
loss). Nearly all of that loss is one request: the first one after the 4,096-token prompt, where each step took 137.0
and 136.6 ms against 114.0 and 113.9 (about 1.8 s more over its 79 steps). That is the kind of wait for the
keep-alive block that the eager stop was made to remove (C811, C817). On the same request run again, and on every
other greedy request, `v4ne`'s step was within 3.6 % of `v4r`'s.

In short, for Flash-Next's preset: without the keep-alive, prompts read 2.3 % faster at depth, but decode runs 1.3 %
slower over all depths and 9.9 % slower at 49,152 tokens. Without the eager stop alone, prompts read no faster and
decode runs 0.9 % slower over all depths.

Every arm wrote the same greedy reply on every pass and load at every depth, with the same draft counts, so the speed
differences are the step's time alone. Every value is one load. Run on one rig on 2026-11-16, 15:49:53–17:02:38: a
warm-up load of `v4r`, then `r1 n1 e1 r2 n2 e2`, every load through. The RAM watchdog did not act.

### Per depth (R167b)

`v4r`'s loads are listed `r1`, `r2`; `v4nk`'s `n1`, `n2`; `v4ne`'s `e1`, `e2`. Each class is against `v4r`. Decode
is a load's speed: both greedy passes' tokens over their time.

| Depth (tokens) | Decode tok/s, `v4r` | `v4nk` | Class | `v4ne` | Class |
|---:|---|---|---|---|---|
| 0 | 23.57, 23.46 | 23.37, 23.78 | negligible (+0.2 %) | 23.70, 23.72 | gain (+0.8 %) |
| 4,096 | 28.97, 28.99 | 30.64, 30.84 | gain (+6.1 %) | 26.28, 26.29 | loss (−9.3 %) |
| 8,192 | 30.77, 30.52 | 31.24, 32.33 | gain (+3.7 %) | 31.23, 31.24 | gain (+1.9 %) |
| 16,384 | 30.84, 30.88 | 30.55, 28.98 | negligible (−3.5 %) | 31.19, 31.20 | gain (+1.1 %) |
| 32,768 | 21.63, 21.61 | 21.35, 21.47 | loss (−1.0 %) | 21.72, 21.50 | negligible (−0.1 %) |
| 49,152 | 24.17, 24.21 | 21.64, 21.96 | loss (−9.9 %) | 24.20, 24.38 | negligible (+0.4 %) |
| all | 26.15, 26.11 | 25.72, 25.86 | loss (−1.3 %) | 25.90, 25.88 | loss (−0.9 %) |

| Depth (tokens) | Read, on cached | Prompt tok/s, `v4r` | `v4nk` | Class | `v4ne` | Class |
|---:|---|---|---|---|---|---|
| 0 | 64 on 0 | 53.5, 51.7 | 47.2, 48.5 | loss (−9.0 %) | 53.3, 53.9 | negligible (+1.9 %) |
| 4,096 | 4,032 on 0 | 198.0, 197.3 | 198.1, 200.1 | negligible (+0.7 %) | 198.8, 197.6 | negligible (+0.3 %) |
| 8,192 | 4,612 on 3,516 | 207.9, 220.0 | 221.9, 218.9 | negligible (+3.0 %) | 221.6, 223.4 | negligible (+4.0 %) |
| 16,384 | 12,804 on 3,516 | 275.7, 281.1 | 290.7, 282.6 | gain (+3.0 %) | 284.7, 279.5 | negligible (+1.3 %) |
| 32,768 | 16,900 on 15,804 | 221.3, 224.6 | 230.1, 230.0 | gain (+3.2 %) | 225.7, 225.4 | negligible (+1.2 %) |
| 49,152 | 16,900 on 32,188 | 178.7, 179.0 | 181.5, 181.4 | gain (+1.4 %) | 174.7, 180.4 | negligible (−0.7 %) |
| all chunks | 55,248 | 212.6, 215.3 | 219.4, 218.2 | gain (+2.3 %) | 214.4, 216.2 | negligible (+0.6 %) |

**The step, where decode moves** ([r167b_table.md](r167b_table.md), "The step on pass 1" and "on pass 2"):
- `v4nk` takes less time a step on the first request at 4,096 and 8,192 tokens (105.1 and 105.0 ms against 114.0 and
  113.9; 112.4 and 110.0 against 118.5 and 120.5) and more at 49,152 (150.6 and 149.5 against 141.0 and 140.7 on pass
  1, 160.7 and 157.2 against 137.7 and 137.5 on pass 2, +15.5 %).
- `v4ne`'s one large difference is pass 1 at 4,096 (137.0 and 136.6 ms against 114.0 and 113.9, +20.0 %); its pass 2
  there takes 109.6 and 110.0 ms against 109.7 and 109.7. Classed on pass 2 alone ([r167b_table_pass2.md](r167b_table_pass2.md)),
  `v4ne`'s decode over all depths is 26.60 and 26.63 tok/s against 26.65 and 26.66 (−0.2 %).
- The sampled request (the server line's own sampler, a fixed seed) at 49,152 tokens runs 23.53 and 23.59 tok/s for
  `v4r`, 20.20 and 18.90 for `v4nk`, 23.82 and 22.11 for `v4ne`.

### The replies, the host and the cards (R167b)

- **Every arm writes the same greedy reply on every pass and load at every depth**, and accepts the same number of
  draft tokens on it (for example 180 of 225 at 49,152 tokens).
- **No target graph reallocated** on any of the 24 requests of any load. Every timed request gave 256 tokens, and
  every load read the same prompt ids.
- **The host's lowest MemAvailable:** `v4r` 247 and 268 MiB, `v4nk` 307 and 288, `v4ne` 241 and 275. MemAvailable
  after loading was 866 and 840 MiB (`v4r`), 862 and 874 (`v4nk`), 866 and 854 (`v4ne`)
  ([r167b_prompt_depth.md](r167b_prompt_depth.md), "The host's memory per load").
- **Swap:** both `v4nk` loads and one `v4ne` load moved memory to swap during the prompts: the most in use was 891 and
  910 MiB for `v4nk` (+819 and +726 MiB during the prompts) and 47 and 672 for `v4ne` (+0 and +528), against 45 and
  52 for `v4r` (+14 and +5). The most written in 10 s was 125 and 137 MiB (`v4nk`), well under the watchdog's 600. This
  run does not show why: one `v4ne` load swapped and the other did not.
- **The cards after loading** held the same memory in every load: 43,983 MiB over the nine.
- **Each server printed the switches it was given:** `switches: 39 set` for `v4r`, the preset's 39 variables;
  `37 set` for `v4nk` and `38 set` for `v4ne`, each `v4r`'s list less exactly the variables the arm drops.
- **The control beside R157's `v4r`** ([2026-11-13_r157_release_deep](../2026-11-13_r157_release_deep/README.md)), the
  same preset and build on the same rig: decode over all depths 26.15 and 26.11 tok/s here against 26.15 and 26.14
  there, all chunks 212.6 and 215.3 against 218.5 and 218.8. These are different runs, so they get no class.

## R167a: Ornith

**On Ornith the keep-alive pays for itself; the eager stop costs a little at depth.**

Without both keep-alive variables (`o4nk`), prompts read a little faster and decode is clearly slower:
- All five prompt chunks of a load (54,736 tokens) read at 164.0 and 164.1 tok/s against `o4`'s 159.9 and 159.8
  (+2.6 %, a gain). Every chunk from 4,096 tokens on gains (+2.1, +1.5, +1.6, +3.1, +3.2 %). The 64-token prompt at depth 0 is the
  exception: it reads at 33.9 and 33.2 tok/s against 40.9 and 40.8 (−17.8 %, a loss).
- Decode over all six depths runs 12.45 and 12.63 tok/s against 12.97 and 12.98 (−3.4 %, a loss). It loses at 0,
  4,096, 16,384 and 32,768 tokens (−6.9, −7.9, −4.8, −3.5 %), gains at 8,192 (+1.1 %) and is negligible at 48,640.
- The two passes split apart. The first greedy request after a prompt runs faster without the keep-alive at 8,192,
  32,768 and 48,640 tokens: at 48,640 a step takes 105.9 and 102.0 ms against 115.6 and 115.4. Every later request at
  the same depth is slower, at every depth. Pass 2 at 48,640 takes 128.9 and 126.5 ms a step against 115.4 and 115.2.
  The sampled request at 48,640 runs at 7.54 and 7.54 tok/s against 8.66 and 8.68. On pass 2 alone,
  decode over all depths is 11.71 and 11.82 tok/s against 13.01 and 13.02 (−9.6 %;
  [r167a_table_pass2.md](r167a_table_pass2.md)). This run does not show why the first pass is faster without it.

Without the eager stop alone (`o4ne`), Ornith is a little faster on both prompts and decode:
- Prompts gain at every depth from 4,096 tokens on (+1.7, +2.5, +1.6, +2.8, +2.5 %). All chunks read at 163.6 and
  163.6 tok/s (+2.4 %, a gain). The 64-token prompt at depth 0 reads at 40.8 and 40.7 against 40.9 and 40.8 (−0.3 %,
  a loss by the rule).
- Decode gains at every depth, by +0.1 or +0.2 %. The loads of each arm differ by at most 0.02 tok/s, so even this
  small difference is beyond the spread. Over all depths decode is 13.00 and 13.00 tok/s (+0.2 %, a gain).
- The wait for the keep-alive block that the eager stop removes on Flash-Next (R167b, above) does not show here. At
  4,096 tokens `o4ne`'s first request takes 58.9 and 58.9 ms a step against 59.1 and 59.1.
- R162 found v4's host switches reading prompts 1.1 % slower than v3.0 from 8,192 tokens on, and v4 added the eager
  stop to v3.0's line ([2026-11-14_r162_ornith_release_deep](../2026-11-14_r162_ornith_release_deep/README.md)).
  This run is consistent with the eager stop being that cost, but R162 compared two builds, so it does not show it.

In short, for Ornith's preset:
- The keep-alive costs 2.6 % of prompt speed at depth and buys 3.4 % decode over all depths.
- The eager stop costs 2.4 % of prompt speed and 0.2 % of decode at depth, with the same replies.
- Whether the eager stop stays in Ornith's preset is the agent check's call (registry row Q1496).

Every arm wrote the same greedy reply on every pass and load at every depth. Every value is one load. Run on one rig on
2026-11-16, 17:15:05–19:39:49: a warm-up load of `o4`, then `o1 n1 e1 o2 n2 e2`, every load through. The RAM
watchdog did not act.

### Per depth (R167a)

`o4`'s loads are listed `o1`, `o2`; `o4nk`'s `n1`, `n2`; `o4ne`'s `e1`, `e2`. Each class is against `o4`. Decode
is a load's speed: both greedy passes' tokens over their time.

| Depth (tokens) | Decode tok/s, `o4` | `o4nk` | Class | `o4ne` | Class |
|---:|---|---|---|---|---|
| 0 | 18.51, 18.49 | 17.18, 17.28 | loss (−6.9 %) | 18.53, 18.53 | gain (+0.1 %) |
| 4,096 | 16.98, 16.98 | 15.39, 15.87 | loss (−7.9 %) | 17.01, 17.01 | gain (+0.2 %) |
| 8,192 | 15.60, 15.62 | 15.83, 15.73 | gain (+1.1 %) | 15.63, 15.64 | gain (+0.2 %) |
| 16,384 | 13.43, 13.42 | 12.66, 12.89 | loss (−4.8 %) | 13.45, 13.44 | gain (+0.2 %) |
| 32,768 | 10.47, 10.48 | 10.09, 10.12 | loss (−3.5 %) | 10.49, 10.50 | gain (+0.2 %) |
| 48,640 | 8.66, 8.67 | 8.52, 8.75 | negligible (−0.4 %) | 8.69, 8.69 | gain (+0.2 %) |
| all | 12.97, 12.98 | 12.45, 12.63 | loss (−3.4 %) | 13.00, 13.00 | gain (+0.2 %) |

| Depth (tokens) | Read, on cached | Prompt tok/s, `o4` | `o4nk` | Class | `o4ne` | Class |
|---:|---|---|---|---|---|---|
| 0 | 64 on 0 | 40.9, 40.8 | 33.9, 33.2 | loss (−17.8 %) | 40.8, 40.7 | loss (−0.3 %) |
| 4,096 | 4,032 on 0 | 170.7, 168.6 | 173.8, 172.5 | gain (+2.1 %) | 172.5, 172.6 | gain (+1.7 %) |
| 8,192 | 4,612 on 3,516 | 158.2, 158.8 | 160.3, 161.3 | gain (+1.5 %) | 162.4, 162.4 | gain (+2.5 %) |
| 16,384 | 12,804 on 3,516 | 203.0, 203.1 | 206.5, 206.1 | gain (+1.6 %) | 206.3, 206.1 | gain (+1.6 %) |
| 32,768 | 16,900 on 15,804 | 157.8, 157.8 | 162.5, 162.8 | gain (+3.1 %) | 162.3, 162.1 | gain (+2.8 %) |
| 48,640 | 16,388 on 32,188 | 137.2, 137.2 | 141.5, 141.7 | gain (+3.2 %) | 140.6, 140.6 | gain (+2.5 %) |
| all chunks | 54,736 | 159.9, 159.8 | 164.0, 164.1 | gain (+2.6 %) | 163.6, 163.6 | gain (+2.4 %) |

**The step on each pass** (ms; pass 1, pass 2 per load; [r167a_prompt_depth.md](r167a_prompt_depth.md), "Generation,
a check"):

| Depth (tokens) | `o4` | `o4nk` | `o4ne` |
|---:|---|---|---|
| 0 | 54.5, 53.5; 54.7, 53.5 | 58.4, 58.0; 58.2, 57.6 | 54.5, 53.5; 54.5, 53.5 |
| 4,096 | 59.1, 58.7; 59.1, 58.7 | 59.2, 70.7; 57.5, 68.5 | 58.9, 58.7; 58.9, 58.7 |
| 8,192 | 64.3, 64.0; 64.2, 63.9 | 61.3, 65.1; 61.8, 65.3 | 64.0, 63.9; 64.0, 63.9 |
| 16,384 | 74.6, 74.3; 74.6, 74.4 | 75.6, 82.4; 73.9, 81.2 | 74.4, 74.4; 74.4, 74.4 |
| 32,768 | 95.7, 95.4; 95.6, 95.3 | 91.0, 107.3; 89.3, 108.4 | 95.3, 95.2; 95.3, 95.2 |
| 48,640 | 115.6, 115.4; 115.4, 115.2 | 105.9, 128.9; 102.0, 126.5 | 115.1, 115.1; 115.1, 115.1 |

### The released preset at depth (R167a's `o4`)

`o4` is [release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env) as released. This run is the first
time the deep lane measures that preset. Beside it are R162's two controls
([2026-11-14_r162_ornith_release_deep](../2026-11-14_r162_ornith_release_deep/README.md)), on the same rig, model,
text and depths:
- `v4` is `v4-rc2` with v3.0's line and the three host switches (`-b 2048`, checkpoints in host memory).
- `v3` is the v3.0 release on v3.0's preset.

R162 ran two days earlier, so these columns get no class, and the percentages are est.

| Depth (tokens) | Decode tok/s, `o4` | R162 `v4` | R162 `v3` | Prompt tok/s, `o4` | R162 `v4` | R162 `v3` |
|---:|---|---|---|---|---|---|
| 0 | 18.51, 18.49 | 18.38, 18.39 | 18.34, 18.32 | 40.9, 40.8 | 24.6, 24.6 | 24.6, 24.3 |
| 4,096 | 16.98, 16.98 | 16.87, 16.86 | 16.82, 16.80 | 170.7, 168.6 | 133.8, 133.9 | 132.4, 132.2 |
| 8,192 | 15.60, 15.62 | 15.54, 15.54 | 15.46, 15.45 | 158.2, 158.8 | 143.2, 143.6 | 144.8, 144.5 |
| 16,384 | 13.43, 13.42 | 13.39, 13.38 | 13.34, 13.33 | 203.0, 203.1 | 191.9, 191.7 | 193.8, 193.4 |
| 32,768 | 10.47, 10.48 | 10.46, 10.44 | 10.43, 10.42 | 157.8, 157.8 | 152.3, 152.3 | 155.5, 155.5 |
| 48,640 | 8.66, 8.67 | 8.63, 8.62 | 8.61, 8.61 | 137.2, 137.2 | 119.7, 119.2 | 120.7, 120.5 |
| all | 12.97, 12.98 | 12.92, 12.92 | 12.88, 12.88 | 159.9, 159.8 | 145.2, 145.0 | 146.8, 146.7 |

Against v3.0 the released preset reads all five prompt chunks about 8.9 % faster (est.) and decodes about 0.7 % faster
(est.). R162's stack with the opt-in micro-batch taper (`v4s`) read all chunks at 167.2 and 166.9 tok/s. `o4` writes
v3.0's and R162's `v4`'s greedy reply at every depth (the same reply md5s). The taper is what wrote another reply in
R162's `v4s` ([FN93b](../2026-11-16_fn93b_ornith_greedy_rig/README.md)).

### The replies, the host and the cards (R167a)

- **Every arm writes the same greedy reply on every pass and load at every depth.**
- **No target graph reallocated** on any of the 24 requests of any load. Every timed request gave 256 tokens, and
  every load read the same prompt ids.
- **The host's lowest MemAvailable:** `o4` 1,228 and 1,248 MiB, `o4nk` 1,236 and 1,240, `o4ne` 1,245 and 1,237.
  MemAvailable after loading was 1,506 and 1,485 MiB (`o4`), 1,494 and 1,504 (`o4nk`), 1,497 and 1,508 (`o4ne`).
  With the checkpoints on the cards, Ornith keeps over 1.2 GB of host memory free at 48,640 tokens.
- **Swap:** the most swap in use was 338 and 338 MiB (`o4`), 338 and 338 (`o4nk`), 337 and 339 (`o4ne`), and none
  moved during the prompts (+0 MiB in every load), unlike R167b's `v4nk` loads on Flash-Next. The most written in
  10 s was 257 and 232 MiB (`o4`), 249 and 228 (`o4nk`), 252 and 229 (`o4ne`), under the watchdog's 600.
- **The cards after loading** held the same memory in every load: 49,139 MiB over the nine, as in R162.
- **Each server printed the switches it was given.** `o4` printed `switches: 6 set`, the preset's six variables.
  `o4nk` printed `4 set` and `o4ne` `5 set`, each `o4`'s list less exactly the variables the arm drops. Every
  started line held one `-b`, 4096, and both checkpoint flags ([r167a_loads.tsv](r167a_loads.tsv)).

## Method

- **The deep lane** ([bench/deep_lane](../../../bench/deep_lane/README.md)) as R157 ran it on Flash-Next
  ([2026-11-13_r157_release_deep](../2026-11-13_r157_release_deep/README.md)), on one build: the sampler pass on, the
  recorded agent text of the release check's `t1swift` sessions, depths 0, 4,096, 8,192, 16,384, 32,768 and 49,152
  tokens, Swift Q2_0 with its MTP draft.
- **Per depth and load:** the prime (the prompt without its 64-token tail, read on top of the previous depth's
  checkpoint), then 256 tokens at temperature 0 twice (decode pass 1, the first request at the new depth, and pass 2,
  the same request again on the same cache), then 256 tokens with the server line's own sampler
  (`--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`) and a fixed seed. End of text is ignored.
- **Loads:** one untimed warm-up load of `v4r` at depth 0, then `r1 n1 e1 r2 n2 e2`, each a fresh server start after
  a quiet check (no llama process, at most 100 MiB on any card) and under the RAM watchdog (stop below 60 MiB
  available). Nothing else ran on the rig.
- **The lines:** the lane reads the preset file (its `PRESET_ENV` and `PRESET_ARGS`). The run script takes per arm
  the variables to drop (`ARM_DROP_<arm>`) and to add (`ARM_ADD_<arm>`). It refuses a control with changes of its own,
  an arm with none, a drop of a variable the preset does not set and an add of one it sets. The lane's `ENV_DROP` is
  the same for every arm, so the script drops the union of the arms' drops and gives each arm back, with the preset's
  values, the ones it keeps; those stand at the end of the line. A dry run printed the three lines first: `v4r`'s
  environment is the preset's as a set, `v4nk`'s is `v4r`'s less the two keep-alive variables, `v4ne`'s less the eager
  stop, and the binary, model and flags are the same in all three. [r167b_loads.tsv](r167b_loads.tsv) holds the lines
  as they ran.
- **R167a (Ornith)** ran the same way, with R162's settings for Ornith:
  - the model is Ornith-1.5-122B Pascal N2P GGUF (`valid1`), no draft, a 49,152-token context;
  - the depths are 0, 4,096, 8,192, 16,384, 32,768 and 48,640 tokens (the deepest whose 256 written tokens fit the
    context);
  - the sampler pass uses the server line's own sampler (`--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0`);
  - the loads are one untimed warm-up load of `o4`, then `o1 n1 e1 o2 n2 e2`, under the RAM watchdog with its default
    limits.

  Its dry run printed the three lines first:
  - `o4`'s environment is the preset's six variables, and its flags are the preset's `PRESET_ARGS` (one `-b 4096`,
    `--ctx-checkpoints-vram 4` and `--ctx-checkpoints-incall`, no taper);
  - `o4nk`'s environment is `o4`'s less the two keep-alive variables, and `o4ne`'s is `o4`'s less the eager stop;
  - the binary, model and flags are the same in all three.

  [r167a_loads.tsv](r167a_loads.tsv) holds the lines as they ran.
- **Classes** (the lane's rule), per depth and over all depths: the spread is the larger of the two arms' differences
  between their two loads; an arm is a gain when its mean is better than the control's by more than the spread, a loss
  when worse by more than it, else negligible.

## Commands

From the repository root on a machine that only runs ssh and tar; the machine values in the environment:

```bash
# R167b (DRY=1 first: prints each arm's line and starts nothing)
RIG=<rig alias> WORK=<run folder on the rig> PRESET=release/flash-next-v4/settings.env \
  BIN=<v4-rc2 tree on the rig>/bin/llama-server \
  MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> WD_ENV=MIN_AVAIL_MB=60 \
  ARMS="v4r v4nk v4ne" ARM_DROP_v4nk="GGML_CUDA_KEEPALIVE GGML_CUDA_KEEPALIVE_EAGER" \
  ARM_DROP_v4ne=GGML_CUDA_KEEPALIVE_EAGER LOADS="warm:v4r r1:v4r n1:v4nk e1:v4ne r2:v4r n2:v4nk e2:v4ne" \
  bash docs/data/2026-11-16_r167_keepalive_depth/run/r167.sh
RIG=<rig alias> WORK=<run folder on the rig> STEP=wait bash docs/data/2026-11-16_r167_keepalive_depth/run/r167.sh
RIG=<rig alias> WORK=<run folder on the rig> STEP=collect OUT=<folder> \
  bash docs/data/2026-11-16_r167_keepalive_depth/run/r167.sh

# R167a (DRY=1 first), then STEP=wait and STEP=collect as above
RIG=<rig alias> WORK=<run folder on the rig> PRESET=release/ornith-v4/settings.env \
  BIN=<v4-rc2 tree on the rig>/bin/llama-server \
  MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> DEPTHS="0 4096 8192 16384 32768 48640" \
  ARMS="o4 o4nk o4ne" ARM_DROP_o4nk="GGML_CUDA_KEEPALIVE GGML_CUDA_KEEPALIVE_EAGER" \
  ARM_DROP_o4ne=GGML_CUDA_KEEPALIVE_EAGER LOADS="warm:o4 o1:o4 n1:o4nk e1:o4ne o2:o4 n2:o4nk e2:o4ne" \
  bash docs/data/2026-11-16_r167_keepalive_depth/run/r167.sh
```

The tables (the driver writes `table.md` and `classes.tsv` at the end of the run; here they were built on the rig
after its run, with the same result):

```bash
python3 -I bench/deep_lane/deep_table.py <folder> --control v4r --out r167b_table.md --tsv r167b_classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control v4r --class-on pass2 --out r167b_table_pass2.md \
  --tsv r167b_classes_pass2.tsv
python3 -I docs/data/2026-11-10_deep_lane/run/prompt_depth.py <folder> --control v4r --out r167b_prompt_depth.md \
  --tsv r167b_prompt_depth
# R167a: the same three with --control o4 and the r167a_ names
```

## Files

| File | Content |
|---|---|
| [r167b_table.md](r167b_table.md), [r167b_classes.tsv](r167b_classes.tsv) | R167b, `deep_table.py`'s tables, control `v4r`, a load's speed: decode by depth and over all depths, the step on each pass, prompts, the draft, the sampler pass, the checks |
| [r167b_table_pass2.md](r167b_table_pass2.md), [r167b_classes_pass2.tsv](r167b_classes_pass2.tsv) | the same, decode classed on pass 2 alone |
| [r167b_prompt_depth.md](r167b_prompt_depth.md), [r167b_prompt_depth_prompt.tsv](r167b_prompt_depth_prompt.tsv), [r167b_prompt_depth_generation.tsv](r167b_prompt_depth_generation.tsv), [r167b_prompt_depth_memory.tsv](r167b_prompt_depth_memory.tsv) | R167b, `prompt_depth.py`'s tables: prompt reading per depth and over all chunks, the host's memory and each card's per load, generation as a check |
| [r167b_loads.tsv](r167b_loads.tsv), [r167b_requests.csv](r167b_requests.csv) | R167b, per load its line and binary as it ran; per request its counts, times and reply md5s |
| [r167a_table.md](r167a_table.md), [r167a_classes.tsv](r167a_classes.tsv) | R167a, `deep_table.py`'s tables, control `o4`: decode by depth and over all depths, the step on pass 1, prompts, the sampler pass, the checks |
| [r167a_table_pass2.md](r167a_table_pass2.md), [r167a_classes_pass2.tsv](r167a_classes_pass2.tsv) | the same, decode classed on pass 2 alone |
| [r167a_prompt_depth.md](r167a_prompt_depth.md), [r167a_prompt_depth_prompt.tsv](r167a_prompt_depth_prompt.tsv), [r167a_prompt_depth_generation.tsv](r167a_prompt_depth_generation.tsv), [r167a_prompt_depth_memory.tsv](r167a_prompt_depth_memory.tsv) | R167a, `prompt_depth.py`'s tables: prompt reading per depth and over all chunks, the host's memory and each card's per load, generation with the step on both passes |
| [r167a_loads.tsv](r167a_loads.tsv), [r167a_requests.csv](r167a_requests.csv) | R167a, per load its line and binary as it ran; per request its counts, times and reply md5s |
| [run/r167.sh](run/r167.sh) | the run, every part: start, wait, status, collect; the preset and each arm's drops and adds from the environment |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2` (`pascal-p106-v4` `eb82916d3`; the rig build's `llama-server`
reports `eb312b7bc`, whose tree is `v4-rc2`'s). Model: Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0
([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open License 1.0 and
qwen-community-1.0), as retyped by [Q218](../2026-11-03_retype/README.md), with its MTP draft file (R167b);
Ornith-1.5-122B as Ornith-1.5-122B Pascal N2P GGUF (`valid1`), described with its licence in
[release/ornith-v3](../../../release/ornith-v3/README.md) (R167a); the prompts are
the deep lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and
`requests.csv` hold times, counts and md5s, no text. Measured on a Manli M-P106L9-N6G (nine P106-090).
