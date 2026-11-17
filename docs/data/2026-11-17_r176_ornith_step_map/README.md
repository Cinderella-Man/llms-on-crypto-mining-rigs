# Ornith-1.5-122B's v4.0 decode step taken apart (R176, Ornith's half of Q1510; with Q1544's row read; 2026-11-17)

What this measured: where a decode step of Ornith-1.5-122B Pascal N2P GGUF (`valid1`) goes on Miner Llama v4.0 with
the released preset, part by part and kernel by kernel, at 0 and 8,192 tokens of context. The preset decodes one
token a pass; passes of 2, 3 and 4 tokens come from Ornith's own MTP layer drafting 1, 2 or 3 tokens a step, which
prices any draft for this model. Before any load, one row of the file's token embedding was read: the row a published
drafter needs for its mask token. Nine P106-090 cards,
[release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env) plus the profilers' own variables.
Registry rows Q1510 (the map) and Q1544 (the drafter's mask row) in [quants/experiments.md](../../../quants/experiments.md).
The method is R121's for Flash-Next ([2026-11-17_r121_flash_step_map](../2026-11-17_r121_flash_step_map/README.md));
the plan it serves is [the Flint and Ornith review](../../research/notes/v5_review_flint_ornith_2026-11.md).

Each value is one load. Two values in a cell are the arm's two loads, in run order.

## Result in plain words

- **One token takes 53.72 and 53.73 ms at the start of a context, and 52.17 and 52.18 of it is the nine cards'
  work.** The host's own part is 0.31 ms. The hops between cards, with launch latency and gaps, take 1.22 and
  1.23 ms (est.). The largest parts of the cards' work are the routed experts' fused gate-and-up product
  (12.02 and 12.00 ms) and the recurrent layers' products (14.58 and 14.57).
- **At 8,192 tokens the step takes 64.41 and 64.42 ms, and the attention kernel alone is the growth**:
  `FLASH_ATTN_EXT` grows 10.22 and 10.23 ms of the step's 10.69 and 10.69. The mask built on the host adds 0.06 ms.
- **A second token in a pass nearly doubles the cards' work** (52.17 → 96.59 ms at depth 0). Ornith's dense products
  are Q5_0 and Q8_0, and at 2 tokens they cost 1.8 to 2.3 times one token: the read-once kernels Flash-Next's preset turns
  on for 2 to 4 tokens cover IQ4_XS, Q4_K and Q5_K, and the one for these types is off (new row Q1580). A third and a fourth token add 29.32 and 25.78 ms.
- **So Ornith's own MTP draft still does not pay at the start of a context, and is level at 8,192 tokens.** With two
  drafted tokens it writes a token in 55.74 ms against 53.69 and 53.71 without a draft, and in 63.59 against 64.33 and
  64.34 at 8,192 (this greedy reply: 2.560 and 2.612 tokens a step). It breaks even at 2.68 and 2.61 tokens a step.
- **Row 248077 of the token embedding, the published drafter's mask token, is all zeros** in this file, as the
  drafter's maker says of GGUF targets.

## Result

### Row 248077 of the token embedding (FN114, Q1544)

`token_embd.weight` is Q8_0, 3,072 × 248,320 ([embd_row.txt](embd_row.txt), [run/embd_row.py](run/embd_row.py): the
header and the named rows only, under a second on the rig, no model loaded):

| Row | Token | Bytes not zero | Values exactly zero | Largest value | Block scales that are zero |
|---|---|---:|---:|---:|---:|
| 248077 | `[PAD248077]` (the drafter's mask token) | 3,049 of 3,264 | **3,072 of 3,072** | 0 | **96 of 96** |
| 248076 | `<\|audio_pad\|>` | 3,161 of 3,264 | 7 of 3,072 | 2.498e-4 | 0 of 96 (the first eight 4.8e-7 to 1.8e-6) |
| 248078 | `[PAD248078]` | 3,052 of 3,264 | 3,072 of 3,072 | 0 | 96 of 96 |
| 1000 | `Ġwould` | 3,234 of 3,264 | 29 of 3,072 | 0.0703 | 0 of 96 |

The row's 8-bit values are not zero, but every block's scale is: a Q8_0 block keeps its scale (the block's largest
magnitude over 127) as a 16-bit float, and for a row whose values are about 1e-6 that scale is below the 16-bit
float's smallest step (6e-8), so it rounds to 0 (est., from the maker's stated scale). The maker's row cannot be put
back inside this Q8_0 tensor; it has to come from the engine or from a tensor stored in another type.

### The one-token step and where it goes

The preset's step, the repeated request (`p2`), ms ([map.md](map.md) tables 1 and 6):

| Part | Depth 0 | Depth 8,192 |
|---|---|---|
| **the step** (S37's split of the host profile, median) | **53.72, 53.73** | **64.41, 64.42** |
| the server's ms a token (gen_ms / 256, host profile and trace on) | 53.69, 53.71 | 64.33, 64.34 |
| the nine cards' GPU time | 52.17, 52.18 | 62.72, 62.71 |
| the cards' part less their GPU time: the eight hops, launch latency, gaps (est.) | 1.22, 1.23 | 1.31, 1.31 |
| the host before the pass's inputs are set (server turn, decode set-up) | 0.13, 0.13 | 0.19, 0.19 |
| the host after the token (bookkeeping, the next update's start) | 0.18, 0.18 | 0.19, 0.19 |

R167a measured this preset on the same build with no profiler at 53.50 and 53.45 ms a token at depth 0 and 63.98
and 63.89 at 8,192 ([2026-11-16_r167_keepalive_depth](../2026-11-16_r167_keepalive_depth/README.md)).

The nine cards' work by family of launch groups, ms a pass, the group profile ([map.md](map.md) table 5):

| Family | Depth 0 | Depth 8,192 |
|---|---|---|
| routed experts: gate, up and GLU (one fused launch a layer: N2P 6.21 and 6.20, N3P 5.81 and 5.81) | 12.02, 12.00 | 12.04, 12.04 |
| routed experts: down | 5.48, 5.46 | 5.48, 5.49 |
| recurrent layers' products (`attn_qkv`, `attn_gate`, `ssm_out`, `ssm_alpha`, `ssm_beta`; Q5_0) | 14.58, 14.57 | 14.42, 14.51 |
| attention layers' products (`attn_q`, `attn_output` Q5_0; `attn_k`, `attn_v` Q8_0) | 4.22, 4.22 | 4.23, 4.23 |
| `GATED_DELTA_NET` (36 layers) | 3.47, 3.46 | 3.50, 3.49 |
| the output head (Q5_0, 3,072 × 248,320) | 3.01, 3.01 | 3.01, 3.02 |
| shared expert (gate, up and GLU 1.73 and 1.74; down 1.15 and 1.16; its gate 0.15 and 0.14) | 3.03, 3.04 | 3.02, 3.08 |
| router (F32) and its top-k chain | 1.61, 1.61 | 1.60, 1.58 |
| `FLASH_ATTN_EXT` (12 attention layers) | 0.56, 0.56 | **10.79, 10.80** |
| conv window (`SSM_CONV`) | 0.24, 0.21 | 0.24, 0.20 |
| everything else (norms, adds, the conv window's `CONCAT` and state copies, the sampling chain) | 3.64, 3.71 | 3.74, 3.71 |
| **the pass** | **51.86, 51.87** | **62.07, 62.13** |

- **The growth with depth is the attention kernel.** From 0 to 8,192 tokens the pass grows 10.21 and 10.26 ms, of
  which `FLASH_ATTN_EXT` is 10.22 and 10.23 ([map.md](map.md) table 4a); every other launch group moves by 0.15 ms or
  less. The host's part before a pass, where the mask is built, grows 0.06 and 0.06 ms. At 8,192 cells the twelve
  layers' K and V (F16, 2 KV heads of 256) are 201 MB, which the kernel reads at about 19 GB/s (est.).
- **The recurrent layers' small operations:** the conv window takes `CONCAT conv_input` 0.48 and 0.47 ms,
  `SSM_CONV+SILU` 0.24 and 0.21 and the 36 conv-state copies 0.78 and 0.78 (1.50 and 1.46 together); the gate inputs
  `SIGMOID` (beta) 0.10 and 0.15 and `SOFTPLUS+MUL` (alpha) 0.28 and 0.31; the gated norm's `SILU+MUL` 0.09 and 0.09
  (verify/labels.tsv.gz).
- **The first request at a depth** writes 54.84 and 55.35 ms a token at depth 0 against its repeat's 53.69 and 53.71,
  with the same typical step (53.73 and 53.74 against 53.72 and 53.73: [map_p1.md](map_p1.md)), so the cost sits in a
  few passes, as R192 found on Flash-Next. At 8,192: 64.49 and 64.52 against 64.33 and 64.34.

### Passes of 2, 3 and 4 tokens

The step with Ornith's MTP layer drafting 1, 2 or 3 tokens, the repeated request, ms ([map.md](map.md) table 1;
one load per width; host profile and trace on):

| Width | Depth | The step | Verify pass: GPU time | Verify pass: the rest | The draft's span (its GPU time) | Tokens a step (this reply) | Server ms a token | Break-even tokens a step (est.) |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 0 | 53.72, 53.73 | 52.17, 52.18 | 1.53, 1.53 | — | 1 | 53.69, 53.71 | — |
| 2 | 0 | 108.38 | 96.59 | 3.84 | 7.98 (6.56) | 1.896 | 56.85 | 2.02 |
| 3 | 0 | 143.96 | 125.91 | 4.65 | 13.40 (11.59) | 2.560 | 55.74 | 2.68 |
| 4 | 0 | 176.00 | 151.69 | 5.48 | 18.64 (16.21) | 2.813 | 61.31 | 3.28 |
| 1 | 8,192 | 64.41, 64.42 | 62.72, 62.71 | 1.69, 1.70 | — | 1 | 64.33, 64.34 | — |
| 2 | 8,192 | 123.60 | 109.61 | 4.07 | 9.95 (8.40) | 1.896 | 64.66 | 1.92 |
| 3 | 8,192 | 167.91 | 145.72 | 4.99 | 17.23 (14.79) | 2.612 | 63.59 | 2.61 |
| 4 | 8,192 | 202.27 | 172.67 | 5.90 | 23.72 (20.41) | 3.048 | 65.41 | 3.14 |

Break-even is the step over the one-token step of the first load (est.). The draft's span is a catch-up pass over
the accepted tokens (3.19 ms of GPU time at four tokens, depth 0) and one pass a drafted token (4.18 to 4.46 ms), all
on the last card. **Without its own draft's cost**, a verify pass alone (GPU time and the rest) breaks even at 1.87,
2.43 and 2.93 tokens a step at depth 0, and at 1.76, 2.34 and 2.77 at 8,192 (est.): the price of a verify for any
draft of this model, the published drafter's included.

What a pass costs by family, ms, the group profile, one load per width ([map.md](map.md) table 5):

| Family | 1 token | 2 tokens | 3 tokens | 4 tokens | 1 token, 8,192 | 4 tokens, 8,192 |
|---|---|---:|---:|---:|---|---:|
| routed experts: gate, up and GLU | 12.02, 12.00 | 19.40 | 28.11 | 36.12 | 12.04, 12.04 | 37.09 |
| routed experts: down | 5.48, 5.46 | 11.04 | 15.51 | 19.51 | 5.48, 5.49 | 20.33 |
| recurrent layers' products | 14.58, 14.57 | 28.71 | 35.29 | 41.04 | 14.42, 14.51 | 41.24 |
| attention layers' products | 4.22, 4.22 | 8.68 | 10.73 | 12.42 | 4.23, 4.23 | 12.43 |
| the output head | 3.01, 3.01 | 5.55 | 7.02 | 8.48 | 3.01, 3.02 | 8.56 |
| shared expert | 3.03, 3.04 | 5.16 | 6.49 | 8.30 | 3.02, 3.08 | 8.44 |
| `GATED_DELTA_NET` | 3.47, 3.46 | 6.23 | 8.99 | 11.69 | 3.50, 3.49 | 11.73 |
| `FLASH_ATTN_EXT` | 0.56, 0.56 | 0.76 | 1.16 | 1.17 | 10.79, 10.80 | 20.27 |
| everything else (with the conv window, the router) | 5.49, 5.53 | 10.42 | 11.84 | 11.56 | 5.58, 5.49 | 11.46 |
| **the pass** | **51.86, 51.87** | **95.94** | **125.13** | **150.29** | **62.07, 62.13** | **171.55** |

- **The second token is the dear one: +44.08 ms a pass at depth 0** ([map.md](map.md) table 4b). Ornith's Q5_0 and
  Q8_0 dense products (recurrent and attention layers, shared expert, head) take 48.10 ms at 2 tokens against 24.84 at
  one: `ssm_out` 4.31 → 10.07 (its residual add no longer fused), `attn_qkv` 5.72 → 10.50, `attn_gate` 3.87 → 7.03, the
  head 3.01 → 5.55. The read-once kernels for 2 to 4 tokens that Flash-Next's preset turns on
  (`GGML_CUDA_MMVQ_PASCAL_DMT`, `_Q5K_MT`) cover IQ4_XS, Q4_K and Q5_K; for Q5_0 and Q8_0 a read-once kernel exists behind `GGML_CUDA_MMVQ_PASCAL_MT` (off; Q211) for the
  products the k-column kernels decline ([docs/switches.md](../../switches.md)). The routed
  experts grow 17.50 and 17.46 → 30.44 ms, since each token picks its own experts.
- **The conv window's `CONCAT` costs 4.01, 4.04 and 4.06 ms a pass at 2, 3 and 4 tokens** against 0.48 at one.
- **`GATED_DELTA_NET` adds 2.76, 2.76 and 2.70 ms a token** of the pass (6.23, 8.99, 11.69): the drafted tokens'
  snapshots of the recurrent state.
- **Width 4 at 8,192 tokens ran its checkpoints in host memory:** card 4 lacked 20.70 MiB for a checkpoint store when
  the 8,192-token prompt was read, and the server moved its checkpoints to the host from then on (both width-4 loads;
  the cards held 5,989 MiB on card 4 after loading, [memory.tsv](memory.tsv)). That slows a request's restore (4
  tokens read at 2.0 and 2.1 tok/s), not the steps above.

### What the map prices

Each row's ms are a token's (one-token step) unless said; "worth more" or "worth less" is against the estimate the row
had before this run.

| Row | What it changes | Before this run | Now |
|---|---|---|---|
| Q350, C865 (Q1559) | the routed experts' gate and up products run apart, not fused | est. 3.2 (6.0 %): Q188's kernel bench, fused 12.83 | **the fused launch takes 12.02 and 12.00** (N2P 6.21 and 6.20, N3P 5.81 and 5.81 a pass); Q188's ratio of two apart to fused (0.755 N2P, 0.746 N3P) puts the products apart at est. 9.03, so **est. 2.9 (5.4 %)** less a GLU launch a layer: about the same, a little less. At 2 to 4 tokens the fused launch takes 19.40, 28.11 and 36.12 a pass |
| C152 (Q702) | decode attention that reads the cache at card speed | est. 8.1 at 8,192 (12.7 %) | **worth more:** `FLASH_ATTN_EXT` is 10.79 and 10.80 at 8,192 (0.56 at depth 0), the whole growth of the step (10.22 of 10.69); its floor est. 1.3 to 1.5 (201 MB of K and V, and 3.2 GFLOP at the card's FP32 rate), so **up to est. 9.3 (14.4 %)**; at 4 tokens 20.27 a pass |
| R177 (Q1512) | the attention mask built on the card instead of the host | "says whether row 7's growth is the kernel or the mask" | **answered:** the kernel. The host's part before a pass grows 0.06 and 0.06 to 8,192 tokens: est. at most 0.2 |
| C640 (Q1125) | the Q8_0 cast of the data handed between cards | est. 0.7 (1.3 %), by bytes (C140's fit: the eight hops' 12,288 B est. 1.17) | **ceiling 1.22 and 1.23 at depth 0** (the cards' part beyond their GPU time: hops, launch latency, gaps), 1.31 and 1.31 at 8,192; est. 0.7 stands |
| C96 (Q633) | the gate inputs inside `GATED_DELTA_NET` | est. 2.5 to 4 with C97 (4.7 to 7.5 %) | **worth less:** `SIGMOID` 0.10 and 0.15 and `SOFTPLUS+MUL` 0.28 and 0.31: ceiling 0.38 and 0.46 |
| C97 (Q634) | the conv window and norm fusion | (with C96) | **worth less in one-token decode:** `CONCAT` 0.48 and 0.47, `SSM_CONV+SILU` 0.24 and 0.21, the 36 state copies 0.78 and 0.78 (1.50 and 1.46), the gated norm's `SILU+MUL` 0.09 and 0.09: est. up to 1.3 if one kernel remains. **Worth more with a draft:** `CONCAT` alone takes 4.01, 4.04 and 4.06 a pass at 2, 3 and 4 tokens |
| Q1281, Q1513 (C723, C849) | Ornith's own MTP draft again, with its catch-up cut and probabilistic verification | break-even est. 152.1 a step for est. 2.49 tokens (v3.0's step 165.1) | the step at 3 tokens takes **143.96 at depth 0 and 167.91 at 8,192**; break-even **2.68 and 2.61 tokens a step**, so est. 2.49 still loses (est. 7.6 % at depth 0, 4.7 % at 8,192). Measured: level at 8,192 on this reply (63.59 against 64.33 and 64.34 ms a token), a loss at depth 0 (55.74 against 53.69 and 53.71). The catch-up pass is 3.19 GPU ms of the span |
| FN114 (Q1544) | Ornith with the published drafter (15-token blocks) | "the break-even from Q1510's passes of 2 to 4 tokens" | **row 248077 is zeros** (above); a verify alone breaks even at 1.87, 2.43 and 2.93 tokens a step at 2, 3 and 4 tokens (1.76, 2.34, 2.77 at 8,192) before the drafter's own pass. Passes of 5 to 16 tokens are not measured (new row Q1581) |
| new: R197 (Q1580) | Ornith's 2- to 4-token passes with the read-once switches for Q5_0 and Q8_0 that exist (off) | none | the second token's dense products cost +23.26 a pass (24.84 → 48.10); the switches' share is what the run measures |

## What ran

- **Rig and build:** a nine-card P106-090 rig (Manli M-P106L9-N6G, the stock 4 GB host), nothing else running;
  Miner Llama v4.0 (`pascal-p106-v4` `eb82916d3`, the tag `miner-llama-v4.0`; the rig build's `llama-server` reports
  `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`, whose tree is v4.0's), md5 of `llama-server`
  `e7afd26db01f4c671549ef12c504ada3`.
- **Model:** Ornith-1.5-122B Pascal N2P GGUF (`valid1`: N2P and N3P routed experts, Q5_0 output layer),
  48,196,722,592 bytes, md5 `7985df46ec7bb3867e482eaf91aaef39`. 48 decode layers (36 Gated DeltaNet, 12 attention,
  every fourth) and one MTP layer (`blk.48`, with `nextn.eh_proj`); routed experts N2P in 24 decode layers and N3P in
  24, the shared expert and the attention products Q5_0 and Q8_0, the routers F32.
- **The driver:** [bench/deep_lane](../../../bench/deep_lane/README.md) through [run/r176.sh](run/r176.sh), one
  detached run. Per depth (0, 8,192) a load sends a prime (the prompt without its 64-token tail), decode pass 1 (the
  first request at the depth: 256 tokens at temperature 0, end of text ignored) and decode pass 2 (the same request
  again). The client waits 4 s before every request (`PAUSE=4`), so each profile cuts into one piece per request.
- **The arms, each its own load:**

  | Arm | Width | Line | Profiler (the variables added to the preset's) |
  |---|---|---|---|
  | `w1h1`, `w1h2` | 1 | the preset | `GGML_CUDA_PROFILE_HOST` (each card's host and GPU time per evaluation, nothing synchronized) and `LLAMA_STEP_TRACE` (the server's and the contexts' host marks per step) |
  | `w1g1`, `w1g2` | 1 | the preset | `GGML_CUDA_PROFILE` with `GGML_CUDA_PROFILE_MODE=group` (every launch group timed with events, the stream synchronized after every evaluation) |
  | `w2h1`, `w2g1` | 2 | the MTP line, `--spec-draft-n-max 1` | as `w1h`, as `w1g` |
  | `w3h1`, `w3g1` | 3 | the MTP line, `--spec-draft-n-max 2` | as `w1h`, as `w1g` |
  | `w4h1`, `w4g1` | 4 | the MTP line, `--spec-draft-n-max 3` | as `w1h`, as `w1g` |

  The MTP line is the preset with `--spec-type draft-mtp`, the split that fitted Ornith's MTP draft on v3.0
  (`-ts 6,6,6,6,6,5,5,5,5 -ub 256`: the last card also holds the draft's context;
  [2026-11-06_ornith_mtp_rig2](../2026-11-06_ornith_mtp_rig2/README.md), "The fit") and a context of 16,384 tokens
  (the deepest prompt here is 8,192 + 256), which leaves room for v4's checkpoints on the cards. A pass reads only
  the cache's used cells, so the context's size does not change a step at these depths. The server's draft
  probability floor is 0, so every step drafts the full count and verifies one token more.
  Order: `w1h1 w1g1 w1h2 w1g2 w2h1 w2g1 w3h1 w3g1 w4h1 w4g1`, the one-token map first. Before every load: no llama
  process and at most 100 MiB on any card; every load under the RAM watchdog.

## How it was read

On the operator's machine after the run, with standard-library Python 3 ([run/tables.sh](run/tables.sh)):

1. R121's [split_trace.py](../2026-11-17_r121_flash_step_map/run/split_trace.py) cuts each profile file into one file
   per generation request (a pause of more than 3 s cuts; a piece with at least 45 main passes is a generation
   request).
2. Per host-profile request, S37's
   [step_split.py](../2026-11-10_served_step_split/run/step_split.py): the step, the main pass's GPU time on the nine
   cards and the rest, each draft pass's lead, GPU time and turn, over the typical steps.
3. For the one-token loads, [run/host_split.py](run/host_split.py) splits the step from the step trace: the host's
   turn before the pass's inputs are set, the nine cards' part (inputs set to the token back on the host) and the
   host's part after it; with step_split.py's GPU time, the cards' part less their GPU time is the hops between cards,
   the launches' latency and the gaps (est.: medians of two files). For the drafted loads, R86's
   [align_t0.py](../2026-11-11_r86_step_trace/run/align_t0.py) and C405's
   [step_gap.py](../2026-11-11_c405_server_gap/run/step_gap.py) give the step stage by stage.
4. Per group-profile request, R49's [verify_parts.py](../2026-11-10_served_verify_parts/run/verify_parts.py) with
   R41's parts ([kpass_parts.py](../2026-11-04_kpass_profile/run/kpass_parts.py), 7.6 µs subtracted per launch
   group): the typical main passes by part, bucket and label. Ornith's draft passes are the evaluations with the MTP
   layer's `eh_proj` product (`--other-regex eh_proj`).
5. [run/step_map.py](run/step_map.py) (R121's, widened: any widths, every neighbouring pair of widths, the launch
   groups summed into families, and the host part) joins them per width, depth and request ([map.md](map.md): the
   repeated request; [map_p1.md](map_p1.md): the first).

## Commands

```bash
# the mask-token row, before any load (reads the header and the named rows only; under a second on the rig)
nice -n 19 python3 -I docs/data/2026-11-17_r176_ornith_step_map/run/embd_row.py \
  <models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf 248077 248076 248078 1000

# the run, from the repository root on a machine that runs only ssh and tar; the rig's paths in the environment
RIG=<rig alias> WORK=<run folder on the rig> BIN=<v4.0 build on the rig>/bin/llama-server \
  MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> DRY=1 STEP=start \
  bash docs/data/2026-11-17_r176_ornith_step_map/run/r176.sh
# then the same without DRY=1, then STEP=wait, then STEP=collect OUT=<folder>

# the tables, on any machine with Python 3: S this repository, OUT the run's results (requests.csv, hp-*.csv,
# st-*.jsonl, gp-*.csv), RES the tables
S=<this repository> OUT=<results> RES=<tables> bash docs/data/2026-11-17_r176_ornith_step_map/run/tables.sh
```

The run took 14:31:43 to 16:52:07 on the rig (2 h 20 min: each load reads the 48 GB file from a USB stick in 540
to 670 s); the tables took under two minutes. The five group-profile traces (322 MB together) are not published.
Their sizes and md5s:

| Trace | Bytes | md5 |
|---|---:|---|
| `gp-w1g1.csv` | 88,594,677 | `7a321fb29bb202e8cd45009649ed8d9b` |
| `gp-w1g2.csv` | 88,596,458 | `5d3ccb9b53ce5cfa95da4f2a7ac5d8b2` |
| `gp-w2g1.csv` | 58,234,071 | `df4302789c702958a0a6a20504e7ac09` |
| `gp-w3g1.csv` | 45,344,907 | `36ef8100adea87d9e8de3a55fd682843` |
| `gp-w4g1.csv` | 41,714,158 | `0b28f02f35e03555402a07d45eea75da` |

Every table here comes from them, or from the host profiles and step traces, through the scripts above.

## Files

| File | What it holds |
|---|---|
| [embd_row.txt](embd_row.txt) | `embd_row.py`'s output: rows 248077, 248076, 248078 and 1000 of the token embedding |
| [map.md](map.md), [map_p1.md](map_p1.md) | `step_map.py`: the step, the pass by part, bucket and family per width and depth, what grows with depth and width by label, and the one-token step's host part, each load by value, for the repeated and the first request |
| `map.tsv.gz`, `map_p1.tsv.gz` | every value of those tables in long form: table, width, depth, request, load, name, value |
| [requests.csv](requests.csv) | every request of every load: prompt and generation times, drafted and accepted tokens, the reply's md5 |
| [loads.tsv](loads.tsv) | every load's environment and server line as it ran (paths as `<OUT>`, `<MODELS>`, `<BIN>`) |
| [memory.tsv](memory.tsv) | each card's memory after loading and the host's lowest MemAvailable and most swap in use, per load |
| [table.md](table.md), [deep_classes.tsv](deep_classes.tsv) | the deep lane's own tables of the run (its classes against `w1h1` are not this folder's measure: the arms are profilers and widths, not changes) |
| `split/<load>.tsv`, `split/<load>-<request>.md` | `step_split.py`: every generation step's parts, and the tables per request |
| `host/<load>.md`, `host/<load>.tsv` | `host_split.py` for the one-token loads |
| `gap/gap-<load>.md`, `gap/steps-<load>.tsv`, `gap/align-<load>.md` | `step_gap.py`'s waterfall and `align_t0.py`'s clock offset for the drafted loads |
| `verify/parts.tsv`, `verify/accounting.tsv`, `verify/per_card.tsv`, `verify/cut.tsv`, `verify/labels.tsv.gz`, `verify/last_card_groups.tsv.gz`, `verify/passes.tsv.gz` | `verify_parts.py`'s tables for every group-profile load and request, joined with a load and a request column |
| [run/r176.sh](run/r176.sh) | the run: the arms on the deep lane, start, wait, status, collect |
| [run/embd_row.py](run/embd_row.py), [run/tables.sh](run/tables.sh), [run/host_split.py](run/host_split.py), [run/step_map.py](run/step_map.py) | the row read and the reading |

## Limits

- **One reply per width and depth, greedy, and one load per width above 1.** The tokens a step are this reply's;
  widths 1, 2 and 3 write the same reply at depth 0, width 4 another; at 8,192 widths 2 and 3 write one reply and
  widths 1 and 4 two others. Every load of a width wrote the same reply, so the profilers changed no token.
- **The MTP line differs from the preset** in its split, micro-batch and context (above). The cards are alike, so a
  part's sum over the nine cards does not depend on which card runs it; each card's share does.
- **The group profile synchronizes after every evaluation**, so its GPU times by launch group are this folder's kernel
  figures and its step times are not the preset's (69.22 and 69.21 ms a token at depth 0 against 53.69 and 53.71).
- **The hops' share** of the cards' part beyond their GPU time is an estimate from medians of two files.
- **The unfused gate and up products' price** scales Q188's kernel bench by the served fused time; C865's own run
  measures it.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), v4.0. Model: Ornith-1.5-122B by ornith-ai, quantized for this project with
Unsloth's importance matrix (MIT); the prompts are the deep lane's recorded agent text
([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables hold times, counts, kernel labels and md5s, no
text. Measured on a Manli M-P106L9-N6G (nine P106-090).
