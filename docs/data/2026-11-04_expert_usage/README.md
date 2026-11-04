# How Flash-Next uses its 512 experts on agent text, and what a pruned file would fit (2026-11-04)

Registry row Q232 ([quants/experiments.md](../../../quants/experiments.md)). Qwen3.8-Flash-Next has 48 layers, each with
512 routed experts and a shared expert; every token picks 10 routed experts per layer. This folder counts how often each
routed expert is picked on agent (coding) text, how concentrated that is, how much of the routing a file that keeps
only each layer's busiest N experts would lose, and what such a file would fit on nine 6 GB P106-090 cards.

The counts come from FN4's importance matrix: `llama-imatrix` on ukisai's Swift-1.5 Q8_0 over 200 chunks of 512 tokens
(102,400 tokens, so 1,024,000 routings per layer) of publishable agent sessions outside the held-out ones FN2 scores
KLD on ([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)). **This is one calibration text:
usage on other agent tasks, other languages and general text may differ.** Everything here ran on a CPU from the matrix,
GGUF headers and a few MB of weights; no GPU and no rig.

## Result

**Coding does not concentrate on a small subset of experts.** In every layer, half of the experts carry about 91 % of
the routings and the other half the remaining 9 %, but no small set does the work:

| Per layer (48 layers) | median | range |
|---|---|---|
| experts covering 50 % of routings | 54 | 28–125 |
| experts covering 80 % | 164 | 96–270 |
| **experts covering 90 %** | **242** | **168–341** |
| experts covering 95 % | 305 | 241–388 |
| experts covering 99 % | 406 | 363–451 |
| effective experts (e to the entropy of usage) | 230 | 148–385 |
| Gini of usage (0 even, 1 one expert) | 0.65 | 0.41–0.77 |

- Each token picks its own 10 experts, so a long text touches nearly all of them: only 68 of the 24,576 experts
  (0.28 %) saw no token, 43 of them in the last eight layers and 17 in the first eight. The busiest expert takes up to
  4.7 % of a layer's routings, 24× the mean share (layer 35, expert 9).
- The first eight layers spread widest (90 % at 278 experts on average), the last eight are the most concentrated (205).
- Ranking by activation energy (the sum over routed tokens of the squared SwiGLU activation that enters the expert's
  down projection) picks nearly the same experts as ranking by routings: the two top-256 sets share 238 experts on
  average (213 at least). Ranking by the energy per routing, a REAP-like average without the router weight, picks a
  different set (174 shared; its Spearman correlation with routings is 0.20–0.67 per layer).

**The tail costs little on this text.** Each layer keeping its N most-routed experts ([summary.tsv](summary.tsv),
[prune.tsv](prune.tsv)):

| Kept per layer | Routings kept, all layers (worst layer) | Activation energy kept (worst layer) |
|---|---|---|
| 496 | 99.98 % (99.93) | 99.99 % (99.95) |
| 480 | 99.91 % (99.76) | 99.96 % (99.86) |
| 472 | 99.85 % (99.66) | 99.94 % (99.80) |
| 464 | 99.78 % (99.46) | 99.91 % (99.72) |
| 448 | 99.61 % (98.89) | 99.84 % (99.50) |
| 416 | 99.08 % (97.20) | 99.63 % (98.72) |
| 384 | 98.25 % (94.68) | 99.28 % (97.46) |
| 320 | 95.46 % (87.38) | 97.96 % (93.67) |
| 288 | 93.38 % (82.82) | 96.88 % (90.91) |
| 256 | 90.76 % (77.70) | 95.44 % (86.91) |
| 128 | 72.84 % (51.03) | 82.97 % (64.06) |

A routed token whose expert is cut goes to the next expert on the router's list, so "routings lost" bounds the change,
not the quality cost; that needs a KLD run.

**What fits nine cards** ([prune_fit.md](prune_fit.md), [prune_bytes.tsv](prune_bytes.tsv); F16 cache, the largest
micro-batch that leaves 150 MiB on every card, whole layers or with routed-expert tensors moved by `-ot`):

| File (agent KLD unpruned) | Kept | On the cards | 64k | 64k + DFlash2 | 128k | Routings lost (worst layer) |
|---|---|---|---|---|---|---|
| Swift IQ3_S (0.121) | 512 | 52,066 MiB | no | no | no | — |
| Swift IQ3_S | 464 | 47,547 MiB | `-ot` ub 128: 153 | no | no | 0.22 % (0.54) |
| Swift IQ3_S | 448 | 46,040 MiB | `-ot` ub 256: 165 | `-ot` ub 128: 211 | `-ot` ub 64: 153 | 0.39 % (1.11) |
| Swift IQ3_S | 416 | 43,028 MiB | ub 256: 172; `-ot` ub 512: 185 | ub 128: 299 | `-ot` ub 128: 389 | 0.93 % (2.80) |
| Swift IQ3_S | 384 | 40,015 MiB | ub 512: 207 | ub 512: 150 | ub 128: 407 | 1.75 % (5.32) |
| ISTA IQ3_S (0.117) | 464 | 47,438 MiB | `-ot` ub 128: 167 | no | no | 0.22 % (0.54) |
| ISTA IQ3_S | 416 | 42,930 MiB | ub 256: 184 | ub 128: 311 | `-ot` ub 128: 400 | 0.93 % (2.80) |
| FN18, Q4_0 experts (not scored) | 320 | 44,981 MiB | ub 64: 175; `-ot` ub 256: 308 | `-ot` ub 256: 178 | `-ot` ub 64: 184 | 4.54 % (12.6) |
| FN18 | 288 | 40,916 MiB | ub 256: 451 | ub 256: 266 | ub 128: 212 | 6.62 % (17.2) |
| Swift IQ3_XXS (0.176), for reference | 512 | 44,710 MiB | ub 64: 155; `-ot` ub 256: 387 | `-ot` ub 256: 202 | `-ot` ub 128: 159 | — |
| Swift IQ3_XXS | 480 | 42,137 MiB | ub 256: 243 | ub 256: 187 | ub 64: 231 | 0.10 % (0.24) |

- **Both IQ3_S files fit 64k by cutting 48 experts per layer (9.4 %)**, which drops 0.22 % of this text's routings, and
  on whole layers by cutting 96 (18.8 %, 0.93 % of routings). That is the file the rig cannot hold today: agent KLD
  0.117–0.121 against IQ3_XXS's 0.176, est. +3.6 ms a token on today's timed kernels
  ([file choice](../2026-11-04_file_choice/README.md)), since a token still reads 10 experts of the same types.
- **FN18's file** (Q4_0 experts, Q8_0 dense tensors, from the Q8_0 with FN4's matrix; its KLD run has not finished)
  needs 192 of its 512 experts cut (37.5 %) to fit 64k at IQ3_XXS's size, which drops 4.5 % of routings (12.6 % in the
  worst layer). That is beyond the 25 % at which published pruning keeps coding scores (below).
- llama.cpp takes one expert count for all layers (ISTA's Coder file, 256 everywhere, loads), so every cut here is the
  same N in every layer. Covering 99 % of routings layer by layer would keep 79 % of all experts ([coverage.tsv](coverage.tsv)).

**ISTA's Coder kept a different half.** Its file keeps 256 of 512 experts per layer; they are byte for byte 256 of ISTA
IQ3_S's experts (router rows and down-projection data match for all 48 × 256, [kept.tsv](kept.tsv),
[kept_ids.tsv](kept_ids.tsv)). Swift-1.5's routers equal the original's (cosine 1.0000 for the same expert id in every
layer, 0.35 for neighbouring ids), so the ids mean the same experts in both models. On this agent text the Coder's 256
carry **66.8 % of the routings** (52.0–83.8 % per layer) and 73.1 % of the activation energy; the 256 most-routed carry
90.8 %, and the two sets share 161 experts per layer on average. FN2 scored the Coder against ISTA IQ3_S, whose tensor
types it shares, so the difference is the cut alone: agent KLD 0.274 against 0.117, wiki 0.846 against 0.161, code
0.131 against 0.032. ISTA chose its experts on code, agentic, vision and spatial calibration; a usage cut at 256 would
lose 9.2 % of agent routings instead of 33.2 %. Its effect on wiki text is not measured.

**Layers 24–39 are trained copies of layers 8–23** ([best.tsv](best.tsv), [pairs.tsv](pairs.tsv),
[weights.tsv](weights.tsv)). Expert e of layer L and of layer L+16 (L = 8–23) is used alike (usage rank correlation
0.82–0.94; every other pair of layers −0.12 to 0.12), has a similar router row (cosine 0.83–0.95, against 0.28–0.41 for
neighbouring experts and 0.04–0.14 between unrelated layers) and correlated weights (Pearson r 0.12–0.56 over the first
160 rows of the gate and down projections of experts 0, 100 and 372; other experts and other layers 0.00). The copies
have trained apart: they are not shared weights, and a delta between them would leave at least 68 % of the variance.
For pruning, a paired layer keeps much the same experts as its partner.

### Quality risk

- **FN2's Coder** (above): cutting half the experts, chosen on other calibration, costs 0.157 of agent KLD and 0.685 of
  wiki KLD over the same file unpruned; one user measured its factual recall at 33 % against 63 %
  ([flash_next_variants.md](../../research/notes/flash_next_variants.md)).
- **REAP** (arXiv 2510.13999): at 25 % of experts removed, generative coding drops 1.9 % on average; at 50 %, 6.9 %, and
  multiple-choice knowledge 13 %. Pruning by frequency alone collapses at 50 %; calibration must match the domain
  ([models_large_moe_and_compression.md](../../research/notes/models_large_moe_and_compression.md) section 5).
  Samsung's Qwen3-Coder-Next REAP (512 → 384 experts) averages 70.5 against 72.9.
- **MoEXBench** (arXiv 2608.21693): +27 % perplexity on general text at 20 % pruning.
- The IQ3_S cuts that fit 64k (48–96 experts, 9–19 %) sit below every published cut; FN18's 37.5 % sits above REAP's
  25 %. The gap to spend is IQ3_S's 0.055 of agent KLD below IQ3_XXS. Whether a 48-expert cut costs less than that is
  Q891's KLD run.

### What this does not show

- **One text.** 102,400 tokens of one set of agent sessions. Each expert averages 2,000 routings, so the order of the
  well-used experts is stable within this text, but experts with tens of routings rank by chance, and other tasks,
  languages and general text use other experts (the Coder's wiki KLD shows how much). Q892 counts a second text.
- **Counts, not router weights.** The matrix holds how often an expert is picked and the energy of its input, not the
  router's weight for it or the norm of its output, which REAP's criterion needs (Q892).
- **Fit, not quality.** No pruned file has been built or scored; the fit is the planner's on scaled headers (exact for
  a file cut by slicing the expert tensors and router rows), checked against llama.cpp's own projection within 4–8 MiB
  per card on unpruned files ([file choice](../2026-11-04_file_choice/README.md)).

## Files

| File | What |
|---|---|
| [layers.tsv](layers.tsv) | per layer: routings, routings per token (10.000), unused experts, experts covering 50/80/90/95/99/99.9 % of routings and of activation energy, entropy, effective experts, Gini, the busiest expert against the mean, the top five experts with their shares, the Spearman correlation of routings and energy per routing, the overlap of the top 256 by routings with the top 256 by energy and by energy per routing |
| [coverage.tsv](coverage.tsv) | those measures over the 48 layers: mean, median, min, max, and the sum over layers |
| `experts.tsv.gz` | per layer and expert (24,576 rows): routings, share, rank; activation energy, share, rank; energy per routing; the mean squared norm of the routed tokens' hidden state |
| [prune.tsv](prune.tsv) | per layer, kept N (64–496) and ranking (routings, energy, energy per routing): routings and energy kept |
| [summary.tsv](summary.tsv) | the same over all layers, with the worst layer |
| [prune_bytes.tsv](prune_bytes.tsv) | per file and kept N: MiB of routed experts and MiB on the cards |
| [prune_fit.md](prune_fit.md) | per file and kept N, F16 cache, no draft and DFlash2, 32k–262k: the largest micro-batch that leaves 150 MiB on every card (whole layers / experts moved) |
| [prune_fit.tsv](prune_fit.tsv) | every plan (3,168): file, kept N, context, micro-batch, draft, whole or moved, the fullest card's free MiB, `-ts`, the moved tensors |
| [kept.tsv](kept.tsv), [kept_ids.tsv](kept_ids.tsv) | ISTA's Coder: per layer the original ids it kept, how they were matched, the routings and energy they carry here against the top 256, the overlap, and the router cosine between Swift-1.5 and the original |
| [pairs.tsv](pairs.tsv), [best.tsv](best.tsv) | the usage rank correlation of every pair of layers, and each layer's best partner |
| [weights.tsv](weights.tsv) | for the 16 paired layers and four control pairs: router cosine (same and neighbouring expert) and the weight correlation of three experts' gate and down projections |
| [run/](run/) | `expert_usage.py`, `layer_pairs.py`, `coder_kept.py`, `prune_fit.py` (standard-library Python) and `run.sh` |

## Commands

On a machine that holds the matrix and the files (CPU only, under 20 minutes at `nice -n 19`; it reads the matrix's
per-expert tensors, the GGUF headers, the router rows and 4 KiB per expert of the ISTA files, and 160 rows of six
experts in 20 layer pairs of the Q8_0):

```bash
# shard lists comma-separated; FILES: one NAME=SHARD[,SHARD...] line per file to plan
IMATRIX=<imatrix-swift15-q8_0-agent-train.gguf> Q8=<Swift-1.5 Q8_0 shards> \
ORIG=<ISTA IQ3_S shards> PRUNED=<ISTA Coder shards> FILES=<files.list> OUT=<output folder> \
bash docs/data/2026-11-04_expert_usage/run/run.sh
```

`files.list` named `swift-iq3_xxs`, `swift-iq3_s`, `ista-iq3_s` (the files of [file choice](../2026-11-04_file_choice/README.md))
and `fn18-pq4exp` (FN18's five shards). `prune_fit.py` imports [bench/file_choice/fit.py](../../../bench/file_choice/fit.py)
and [scripts/split_plan.py](../../../scripts/split_plan.py); the others need [scripts/gguf_reshard.py](../../../scripts/gguf_reshard.py).
`KEEP` changes the grid of kept counts. The run used Python 3.14.

The matrix itself (FN4's step of the quality chain):

```bash
llama-imatrix -m <Swift-1.5 Q8_0 first shard> -f calib-agent-train.txt -o imatrix.gguf -c 512 -b 512 --chunks 200 -ngl 99 -fa on <split>
```

with `calib-agent-train.txt` from [2026-10-28_ornith35_kld/run/calib_text.py](../2026-10-28_ornith35_kld/README.md) and
the run script in [2026-11-03_flash_next_quality/run/](../2026-11-03_flash_next_quality/README.md) (`fn4.sh`).

## Provenance

The importance matrix was computed by Miner Llama's v3 base (`37e717ff6`) from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF) Q8_0
(Swift Open License 1.0 and qwen-community-1.0) over publishable agent sessions. The pruned-file plans read the headers of
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(IQ3_XXS as resharded for the rig, IQ3_S),
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(IQ3_S) and FN18's file; the Coder match read router rows and expert data of the IQ3_S and of
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
(qwen-community-1.0). The tables hold counts, sums, ids, sizes and correlations, no weights or text. KLD values: FN2's
logs. Tables and scripts of this repository (MIT).
