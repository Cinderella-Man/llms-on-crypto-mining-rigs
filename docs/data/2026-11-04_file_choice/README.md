# Which Flash-Next file the rig should run: quality, fit and speed together (2026-11-04)

Registry row Q811. The goal is the highest-quality Qwen3.8-Flash-Next file that runs on nine 6 GB P106-090 cards and a
4 GB host, well above the 5–8 tok/s of the first baseline. This folder puts three things side by side for the eight
files of the quality chain (FN2, [2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)):
- **quality:** KL divergence and same top token against Swift-1.5's Q8_0, measured on held-out agent sessions, wiki
  text and code;
- **fit:** what each file puts on the cards, from its GGUF headers, planned at 32k to 262k context with F16 and q8_0
  cache, without a draft, with the DFlash2 drafter and with the MTP draft;
- **speed:** an **estimate** of the GPU time a decoded token takes, from the bytes it reads per weight type and the
  speed each type's kernel reached on a P106.

Nothing ran on a GPU or a rig. Every speed number here is est., from the model described below.

## Result

| File | Agent KLD (same top token) | Wiki, code KLD | On the cards | Table | 64k | 128k | 262k | 64k + DFlash2 | est. ms/token (tok/s): timed / expected kernels | vs Q2_0 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| ISTA IQ3_S | 0.117 ± 0.002 (93.4 %) | 0.161, 0.032 | 51,945 MiB | shard 2 | no (−674) | no (−983) | no (−1,498) | no | 100.3 (10.0) / 98.6 (10.1) | +17.9 / +25.7 |
| Swift IQ3_S | 0.121 ± 0.003 (93.7 %) | 0.153, 0.032 | 52,066 MiB | shard 1, reshard | no (−689) | no (−998) | no (−1,513) | no | 102.5 (9.8) / 100.8 (9.9) | +20.0 / +27.8 |
| **Swift IQ3_XXS** | **0.176 ± 0.004 (92.3 %)** | 0.248, 0.054 | 44,710 MiB | shard 3 | ub 64: 155; `-ot` ub 256: 387 | `-ot` ub 128: 159 | no (−598) | `-ot` ub 256: 202 | 98.9 (10.1) / 91.9 (10.9) | +16.4 / +19.0 |
| Swift IQ3_XXS-hcq8 | not scored (agent run failed) | 0.245, — | 44,018 MiB | shard 3 | ub 128: 187 | `-ot` ub 128: 233 | no (−526) | ub 64: 201 | 114.4 (8.7) / 90.6 (11.0) | +32.0 / +17.6 |
| **Swift IQ2_XS** | **0.225 ± 0.005 (91.5 %)** | 0.351, 0.075 | 37,196 MiB | shard 1, reshard | ub 512: 569 | ub 256: 453 | `-ot` ub 128: 230 | ub 512: 505 | 87.2 (11.5) / 82.4 (12.1) | +4.8 / +9.5 |
| ISTA Coder "IQ1_M" | 0.274 ± 0.005 (90.7 %) | 0.846, 0.131 | 27,904 MiB | shard 2 | ub 512: 1,719 | ub 512: 1,124 | ub 256: 581 | ub 512: 1,634 | 99.9 (10.0) / 98.2 (10.2) | +17.4 / +25.2 |
| Swift Q2_0-hcq8 | 0.283 ± 0.005 (90.7 %) | 0.428, 0.112 | 35,038 MiB | shard 3 | ub 512: 822 | ub 512: 200 | ub 64: 298 | ub 512: 659 | 98.0 (10.2) / 71.6 (14.0) | +15.5 / −1.4 |
| Swift Q2_0 (the rig's file) | 0.285 ± 0.006 (90.7 %) | 0.428, 0.112 | 35,730 MiB | shard 3 | ub 512: 737 | ub 256: 664 | ub 64: 221 | ub 512: 573 | 82.4 (12.1) / 73.0 (13.7) | — |

- **KLD:** against Swift-1.5 Q8_0; the reference against itself at another micro-batch, the noise floor, is 0.029
  (agent), 0.017 (wiki), 0.004 (code). ISTA's files are quantized from the original model, so their KLD against
  Swift's reference also holds the fine-tune's difference.
- **Fit:** F16 cache, nine cards; the largest micro-batch (`-ub`) that leaves 150 MiB on every card, and the fullest
  card's free MiB. Whole layers unless marked `-ot` (routed-expert tensors moved to the next card, which turns off
  pipeline parallelism in prompts). `no (x)`: the best split at any micro-batch, with experts moved, leaves x MiB.
  Every cell for q8_0 cache and for the MTP draft is in [fit.md](fit.md).
- **Table:** the shard that holds the 28.8 GB n-gram table (`per_layer_token_embd`). The rig reads it from its SSD,
  which needs the table alone in its shard; "reshard" files have it in shard 1 with the weights and need
  `scripts/gguf_reshard.py split --isolate` first (C160).
- **Speed:** est. ms of GPU time per decoded token, summed over the cards, at short context without a draft. "Timed":
  every kernel at the speed measured on a P106 so far. "Expected": tonight's kernels that have not run on a GPU yet
  (IQ2_XXS, IQ2_XS, IQ3_S, the one-column Q3_K and the short-row Q8_0) at the speed of their nearest timed relative.

### What to run

**Swift IQ3_XXS is the highest-quality file that fits nine cards.** On agent text its KLD is 0.176 against Q2_0's
0.285 (1.6× lower, same top token 92.3 against 90.7 %), on code 0.054 against 0.112 (2.1× lower). It costs est.
+16.4 ms a token on the kernels timed so far (98.9 against 82.4 ms, 10.1 against 12.1 tok/s), +19.0 ms once tonight's
kernels run at their expected speed (91.9 against 73.0 ms, 10.9 against 13.7 tok/s): 17–21 % slower decoding for
1.6× lower KLD. At 64k it fits on whole layers only at `-ub 64`, with 155 MiB on the fullest card (`-ts 6,6,7,5,5,5,5,5,5`),
right at the 150 MiB margin. With 8 routed-expert tensors moved it leaves 551 MiB at `-ub 128` and 387 MiB at `-ub 256`
(`-ts 6,6,7,5,6,4,5,5,5 -ot blk\.5\.ffn_down_exps=CUDA1,blk\.18\.ffn_up_exps=CUDA3,blk\.18\.ffn_down_exps=CUDA3,blk\.23\.ffn_down_exps=CUDA4,blk\.29\.ffn_up_exps=CUDA5,blk\.29\.ffn_down_exps=CUDA5,blk\.38\.ffn_down_exps=CUDA7,blk\.43\.ffn_down_exps=CUDA8`),
and with DFlash2 202 MiB at `-ub 256` (9 moved). Its table is alone in shard 3 already. Which of the two 64k lines
reads prompts faster, a micro-batch of 64 with pipeline parallelism or 256 without it, is not known (Q510, Q777).

**Swift IQ2_XS is the fallback** if IQ3_XXS's margin or prompt speed disappoints: 1.27× lower agent KLD than Q2_0
(0.225, same top token 91.5 %) for est. +4.8 ms a token (+9.5 expected), with the room Q2_0 has: 64k on whole layers at
`-ub 512` with 569 MiB (`-ts 6,6,6,5,5,5,5,5,6`), with DFlash2 505 MiB, 128k at `-ub 256` with 453 MiB. It needs C160's
reshard before the rig can keep its table on the SSD.

**The rest:**
- **Neither IQ3_S file fits nine cards at any context.** They have the lowest KLD (0.117–0.121 on agent text), but
  even at 32k with experts moved the fullest card lacks 153 MiB; at 64k they need est. ~3.9 GB less on the cards.
  A spine retype gives ~0.6 GB, so only fewer experts get there (Q815, parked).
- **The hcq8 retypes** (hyper-connections in Q8_0, router in F16) score the same as their originals (Q2_0: 0.283
  against 0.285; IQ3_XXS on wiki: 0.245 against 0.248) with 692 MiB less on the cards, but run slower until a
  short-row Q8_0 kernel is timed: +15.5 ms a token on today's timed kernels, −1.4 ms expected. IQ3_XXS-hcq8 is the
  one that fits 64k on whole layers at `-ub 128` (187 MiB) and with DFlash2 at `-ub 64` (201 MiB). If C81's kernel
  reaches its expected speed and its agent KLD (Q814) matches IQ3_XXS's, it replaces IQ3_XXS.
- **ISTA's Coder "IQ1_M" holds no IQ1_M tensor.** The name is its file type; every tensor has the same type as in
  ISTA's IQ3_S file, with 256 experts instead of 512. So a token reads what IQ3_S reads (est. +17 ms over Q2_0), and its
  KLD is the cost of the pruning: 0.274 on agent text, but 0.846 on wiki text, twice Q2_0's. Not a candidate.
- **Swift IQ3_S differs from ISTA's IQ3_S only in the routers** (F32 against BF16, 120 MiB); every other tensor has the
  same type.

## What each file puts on the cards and reads per token

From the GGUF headers ([layout.tsv](layout.tsv), [token_bytes.tsv](token_bytes.tsv)). The cards hold everything but
the n-gram table and the token embeddings (260–322 MiB), which llama.cpp keeps in host memory. A decoded token reads
every dense tensor once and 10 of the 512 routed experts per layer (10 of 256 in the Coder):

| File | On the cards | A token reads | Routed experts (by type, MB) | Shared expert | Dense: attention, recurrent, hyper-connections, router, output |
|---|---|---|---|---|---|
| Swift Q2_0 | 35,730 MiB | 4,155 MB | 664: Q2_0 | 106 | 3,386 |
| Swift Q2_0-hcq8 | 35,038 MiB | 3,430 MB | 664: Q2_0 | 106 | 2,660 |
| Swift IQ2_XS | 37,196 MiB | 4,241 MB | 693: IQ2_S 357, Q2_0 down 221, IQ2_XXS 93, IQ1_M 22 | 135 | 3,413 |
| Swift IQ3_XXS | 44,710 MiB | 4,807 MB | 838: IQ3_S 183, IQ4_NL down 166, Q2_0 down 138, IQ2_S 105, IQ2_XS 95, IQ2_XXS 76, IQ3_XXS 75 | 121 | 3,848 |
| Swift IQ3_XXS-hcq8 | 44,018 MiB | 4,081 MB | as IQ3_XXS | 121 | 3,122 |
| Swift IQ3_S | 52,066 MiB | 5,285 MB | 982: IQ4_NL down 359, IQ3_XXS 213, IQ2_S 210, IQ3_S 141, Q2_0 down 42, IQ4_XS 17 | 146 | 4,157 |
| ISTA IQ3_S | 51,945 MiB | 5,159 MB | as Swift IQ3_S | 146 | 4,031 |
| ISTA Coder "IQ1_M" | 27,904 MiB | 5,096 MB | as IQ3_S (10 of 256) | 146 | 3,968 |

The dense tensors are 2.7–4.2 GB of every token, the routed experts 0.66–0.98 GB. The IQ3_S files' spine is the
heaviest: Q6_K attention and recurrent projections, 1.32 GB a token. The routed down projections are 640 weights wide,
which no 256-weight IQ type fits, so they are Q2_0 or IQ4_NL in every file.

Which shard holds the table: Q2_0, IQ3_XXS and both hcq8 files are the rig's resharded sets (table alone in shard 3);
ISTA's IQ3_S and Coder have it alone in shard 2 as published; Swift's IQ2_XS (341 other tensors) and IQ3_S (372) have
it in shard 1 and need the reshard.

## Fit on nine cards

[scripts/split_plan.py](../../../scripts/split_plan.py), imported by [fit.py](../../../bench/file_choice/fit.py),
finds the split that leaves the most free memory on the fullest card, on whole layers and with routed-expert tensors
moved (`--split-experts`), for every configuration in [fit.tsv](fit.tsv) (2,304 plans). Per-card costs are the ones
llama.cpp projected for Flash-Next on Miner Llama v2.0 ([Q213](../2026-11-03_flash_next_fit/README.md),
`run/plan_splits.sh` there): KV cache per attention layer (161 MiB at 64k in F16, 86 in q8_0), 3 MiB of recurrent
state per Gated DeltaNet layer, and compute buffers that grow with micro-batch × context (102–112 MiB per card at 64k
and `-ub 64`, 600–674 at `-ub 512`; 1,886–1,961 at 262k and `-ub 512`), with 6,001–6,007 MiB free per card. The
planner matched llama.cpp's own projection within 4–8 MiB per card on Q2_0 and IQ3_XXS (Q213); on this run it gives
221 MiB for Q2_0 at 262k and `-ub 64`, where Q213's probe gave 224.

The drafts, on the last card:
- **DFlash2:** 611 MiB of weights, 64 MiB of compute and its cache, 45 MiB at 64k (the fork's fit, Q740; scaled with
  the context elsewhere: est.), 720 MiB at 64k in all. Any draft also makes the main model keep a copy of the
  recurrent state per drafted token: 12.3 instead of 3 MiB per Gated DeltaNet layer at three drafted tokens, +335 MiB
  spread over the cards.
- **MTP:** Q213's measured cost, 2,175–4,462 MiB (64k at `-ub 64` to 262k at `-ub 512`) with its token embeddings in
  host memory; `mtp-shr` is the same with `LLAMA_DRAFT_SHARE=embd,head` (the target's head, 497 MiB less, Q635). The
  planner overestimates free memory with a draft by ~36 MiB against llama.cpp's projection (Q213).

At 64k, F16 cache (from [fit.md](fit.md); whole layers / experts moved):

| File | No draft | DFlash2 | MTP | MTP, shared head |
|---|---|---|---|---|
| Swift Q2_0 | ub 512: 737 / 1,034 | ub 512: 573 / 933 | ub 512: 557 / 707 | ub 512: 557 / 771 |
| Swift IQ2_XS | ub 512: 569 / 879 | ub 512: 505 / 762 | ub 512: 328 / 564 | ub 512: 328 / 584 |
| Swift IQ3_XXS-hcq8 | ub 128: 187 / ub 256: 459 | ub 64: 201 / ub 256: 287 | no (100) / ub 128: 291 | ub 64: 201 / ub 256: 175 |
| Swift IQ3_XXS | ub 64: 155 / ub 256: 387 | no (101) / ub 256: 202 | no (41) / ub 128: 219 | no (101) / ub 128: 268 |
| Swift, ISTA IQ3_S | no | no | no | no |

- **q8_0 cache buys no context** on these cards, as in Q213: compute buffers grow by about what the cache saves.
  The exception is 262k for Q2_0 and IQ2_XS at `-ub 64` (430 and 219 MiB in q8_0 against 221 and 10 in F16).
- **Longest context per file** (150 MiB on every card, any micro-batch): Q2_0, Q2_0-hcq8 and the Coder 262k on whole
  layers; IQ2_XS 262k with experts moved (or whole layers with q8_0 cache); IQ3_XXS and IQ3_XXS-hcq8 128k with experts
  moved; IQ3_S none.

## The speed model

est. ms per token = Σ over weight type and role (MB the token reads ÷ the GB/s of that type's kernel on a P106)
+ 33.48 ms. The constant is the GPU time [R13's profile](../2026-11-03_flash_next_baseline/README.md) of Swift Q2_0
spent outside weight products (attention, the recurrent update, norms, copies, routing), which is the same for every
file of this architecture. The cards run one after another for a single token, so the sum is the decode time when
nothing else waits; R13 measured 136 ms of GPU time and 7.8 tok/s.

The GB/s per type and role are in [bench/file_choice/efficiency.tsv](../../../bench/file_choice/efficiency.tsv), each
row with its source:

| Kernels | GB/s on a P106, in the model | Source |
|---|---|---|
| Q2_0 (experts and dense) | 95.2 (55 % of 173) | R16's bench: 105 GB/s over the four expert shapes (61 %), times ~0.9 for the model, the share round 1's IQ kernels kept in the model (0.83–0.89) |
| BF16 | 141.9 (82 %) | R16's bench: the hyper-connection projections at 157–159 GB/s (91 %), times ~0.9 |
| v2's Q4_K, Q5_K, Q6_K, IQ4_XS | 87–91; the 248,320-row output layer 144 | R13's profile |
| IQ2_S, IQ3_XXS gate/up (Pascal) | 42.6, 45.1 | round 2's bench (48.0, 52.6) times round 1's in-model share of its bench speed (0.89, 0.86) |
| IQ2_XXS, IQ2_XS, IQ3_S gate/up | timed 27.5–29.6 (llama.cpp's in the model); expected 42.6–45.1 | round 1's profile; C2's kernels not yet timed |
| IQ3_S dense | timed 25.5; expected 52.2 | round 2's bench × 0.86–0.87 |
| Q3_K | timed 33.3 (llama.cpp's); expected 89 | R13's profile; C4's kernel not yet timed |
| Q8_0 hyper-connections (hcq8) | timed 27.7 (llama.cpp's, 320-wide rows); expected 89 | R28's profile; C81's path not yet timed |
| F32, F16 routers | 83.5, 41.8 | R13's and R28's profiles |
| IQ4_NL routed down | 30 (est., 20–41) | not timed: llama.cpp's kernel at 640-wide rows, between its Q2_0 down's GB/s and its picoseconds per weight |
| small shared-expert products (Q4_0, Q5_0, IQ4_NL, Q8_0) | 23–36 | R13's profile |

Where IQ3_XXS's extra time against Q2_0 goes (timed / expected, ms): the routed gate/up products +11.9 / +7.6, the
routed down projections +4.7 (IQ4_NL's 166 MB take 5.5 ms at 30 GB/s, 4.0–8.3 over 20–41 GB/s), the shared expert +1.0 / +0.9, and the dense tensors
−1.1 / +5.8: on today's timed kernels Q2_0's Q3_K attention runs on llama.cpp's slow kernel, and once C4's Q3_K kernel
runs, Q2_0's spine becomes cheaper than IQ3_XXS's Q6_K one. A Pascal IQ4_NL expert kernel at Q2_0's speed (C162)
would take ~3.8 ms off IQ3_XXS and ~8.2 ms off the IQ3_S files.

What the model leaves out: context length (attention grows with it, the same for every file), the prompt phase,
drafts (S18's DFlash2 gave 1.15–1.24× at equal tokens on Q2_0), and launch costs of small tensors beyond what R13's
per-type figures hold. The Q2_0 numbers inherit R16's estimate for the speed-ladder build (84–90 ms with both kernels);
for every other file the model is a forecast until the rig measures it.

## What to measure first on the rig

1. **IQ3_XXS's lines (Q813):** `scripts/fit_probe.sh` on header-only copies for the two 64k lines above and the
   DFlash2 line (seconds each), so the 155 MiB margin is checked before a load; then a diagnostic profile of IQ3_XXS
   on `-ladder3` (`run/fn_line.sh` with `MODE=diag` of
   [2026-11-03_flash_next_baseline](../2026-11-03_flash_next_baseline/README.md), ~15 min) against
   [token_bytes.tsv](token_bytes.tsv), type by type.
2. **The agent check of Swift IQ3_XXS** (S13, Q227) on the line the probe and Q510 favour, against Swift Q2_0 on the
   same build.
3. **The untimed kernels on one P106 (Q812):** C2's IQ2_XXS, IQ2_XS and IQ3_S, C4's Q3_K, C81's short-row Q8_0, and
   llama.cpp's IQ4_NL routed down at 640 → 2,560 (no bench case yet); then `model.py` again with the measured GB/s.
4. **Prompt speed for IQ3_XXS's choice:** pipeline parallelism on against off (Q510) and the micro-batch schedule
   (Q777).
5. **Swift IQ2_XS** after the reshard (C160, Q710).

## Files

| File | What |
|---|---|
| [ranking.md](ranking.md) | the result table above, as `model.py` writes it |
| [fit.md](fit.md) | per file, context and cache type, for no draft, DFlash2, MTP and MTP with a shared head: the largest micro-batch that leaves 150 MiB on every card, on whole layers and with experts moved |
| [fit.tsv](fit.tsv) | every plan (2,304): file, context, cache type, micro-batch, draft, whole or moved, the fullest card's free MiB, `-ts`, how many tensors moved and the `-ot` list |
| [layout.tsv](layout.tsv) | per file: experts used, file type, shard sizes, the table's shard, MiB on the cards, the token embeddings' MiB, MB a token reads (routed, shared, dense) |
| [token_bytes.tsv](token_bytes.tsv) | per file, role and weight type: MB a token reads, the GB/s assumed (timed, expected) and the est. ms |
| [speed.tsv](speed.tsv) | per file: est. ms and tok/s (timed, expected), the difference to Q2_0, and the ms per role |
| `headers.tar.gz` | `dump.py`'s JSON of each file's headers: shape keys, shards, and every tensor's name, type, shape, bytes and shard (8 files, 40 KB) |

The tools are in [bench/file_choice/](../../../bench/file_choice/README.md): `dump.py`, `fit.py`, `model.py`,
`efficiency.tsv` and `run.sh`.

## Commands

On a machine that holds the files (headers only are read; under three minutes of one core in all):

```bash
# FILES: one line per file, NAME=SHARD[,SHARD...]; KLD: llama-perplexity's logs as NAME.<agent|wiki|code>.log plus
# q8_0-ub256.*.log for the noise floor (FN2's folder, docs/data/2026-11-03_flash_next_quality)
FILES=files.list KLD=<KLD log folder> OUT=<output folder> bash bench/file_choice/run.sh
```

`files.list` named, in this order: `swift-q2_0`, `swift-iq3_xxs` (the rig's 3-shard sets, Q435), `swift-iq2_xs`,
`swift-iq3_s`, `ista-iq3_s`, `ista-coder-iq1_m` (as published), `swift-q2_0-hcq8`, `swift-iq3_xxs-hcq8` (FN17's
retypes). The run used Python 3.14 at `nice -n 19` on the quant machine's CPU.

## Provenance

The model files: [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(Q2_0, IQ2_XS, IQ3_XXS, IQ3_S; Swift Open License 1.0 and qwen-community-1.0),
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(IQ3_S) and [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
(qwen-community-1.0), and FN17's retypes of the Swift files
([2026-11-03_retype](../2026-11-03_retype/README.md)); only their headers were read, and the tables hold tensor names,
types, shapes and sizes. KLD values: FN2's logs (llama-perplexity, Swift-1.5 Q8_0 reference). Kernel speeds: this
repository's measurements, linked per row of `efficiency.tsv`. Tables and scripts of this repository (MIT).
