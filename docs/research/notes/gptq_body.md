# Better rounding for Flint's body tensors: GPTQ, AWQ-style scaling and learned rounding in Q4_1, Q5_0, Q6_K and Q8_0 blocks (2026-10-24)

Scope: can better rounding of Flint's body tensors (routed experts, attention, shared expert, dense layers), written
into llama.cpp's simple block formats, buy enough quality at equal bytes that cheaper formats fit under the quality
bar? Since decode time follows bytes on the fork, any KLD saved this way turns into speed. The note covers:
- the published gains of GPTQ, AWQ, OmniQuant, AutoRound/SignRound and YAQA over round-to-nearest (RTN), especially
  against llama.cpp's importance-matrix (imatrix) RTN and on MoE experts;
- what per-expert GPTQ needs for Flint's 256 experts: tokens, Hessians, memory and time;
- which block formats GPTQ can write and which AWQ folds are exact in Flint;
- what it buys on K2h's recipe in bytes and tok/s;
- a pipeline for the 7-GPU quant machine, with the smallest decisive first experiment.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (papers, code, PRs, model cards, measurements in this repository) and
  secondary.
- Nothing was run on the rig or the quant machine. The per-expert token counts and input statistics in §2.2 were read
  from Unsloth's published Flint importance matrix (687 chunks × 512 tokens) on a laptop, in seconds.

Starting point, measured ([registry](../../../quants/experiments.md)):
- **Flint.** 49 layers: 2 dense and 47 MoE, each MoE layer with 256 routed experts of intermediate 512, top-6 sigmoid
  routing renormalized, and a shared expert of intermediate 1,024. Hidden 2,048; 32 query and 4 KV heads of 128;
  vocabulary 150,144.
- **Shapes and the input each matrix reads:**

  | Matrix | Shape (rows × input) | Input |
  |---|---|---|
  | Attention q / k / v | 4,096 / 512 / 512 × 2,048 | attention-norm output |
  | Attention output | 2,048 × 4,096 | attention result |
  | Routed gate / up, per expert | 512 × 2,048 | FFN-norm output, routed tokens only |
  | Routed down, per expert | 2,048 × 512 | that expert's SwiGLU output |
  | Shared gate / up | 1,024 × 2,048 | FFN-norm output, all tokens (the router reads the same input) |
  | Shared down | 2,048 × 1,024 | shared SwiGLU output |
  | Dense layers 0–1 | gate/up 6,144 × 2,048, down 2,048 × 6,144 | |

- **The bar:** within noise of UD-Q4_K_XL, KLD 0.120 against BF16 (in practice ≲ 0.125).
- **Lead candidate K2h:** KLD 0.109 at 2.139 GB per token; G3h 0.106 at 2.21; K1h 0.112 at 2.10.
- **All body tensors in these files are imatrix RTN** from llama-quantize with Unsloth's importance matrix. Our own
  importance matrix scores the same (Q12). The makers' int4 is plain RTN (Q74).
- **Q56/Q91:** GPTQ with searched block scales took a Q5_0 output head from +0.023 KLD (RTN without an imatrix)
  through +0.0073 (imatrix RTN, same calibration) to +0.0023. Against imatrix RTN that removes 68 % of the excess.
- **Pricing.** On the fork's kernels decode time follows bytes. Streaming rates on large shapes are Q8_0 150, Q5_0 138
  and Q4_1 145 GB/s, so a weight costs 7.08 / 4.98 / 4.31 ps.
  - In the microbenchmark Q4_0 takes Q4_1's time per weight (10.1 against 10.0 ms per Flint token on one card).
  - Near the bar the file-side frontier trades 0.015–0.025 KLD per ms saved
    ([unconventional ideas](unconventional_ideas.md)).

## Verdict

- **Better rounding is a lever for the always-read tensors, not for the routed experts' format.**
  - Bytes saved per weight:
    - Routed gate/up of layers 9–48 from Q5_0 to Q4_1 saves 0.0625 bytes per weight: 31.5 MB per token, 0.34 ms
      (+1.3 %).
    - Q8_0 to Q5_0 saves 0.375 bytes per weight: 0.0056 ms per MB, 0.04–0.16 ms for each 8-layer group.
  - What the expert move costs: imatrix RTN Q4_1 costs ~+0.022 KLD there (Q36's groups summed, sub-additive), 0.065
    per ms. GPTQ would have to remove ≥ 70 % of that to reach the frontier's rate, and published GPTQ removes far less
    than that against imatrix.
  - Q4_0 is no cheaper per weight than Q4_1 on these cards.
- **The payoff is concentrated where RTN Q5_0 hurts most.** Q77's expensive Q8_0 groups (attention k/v and q in layers
  0–24, attention output 0–24, shared expert 2–24, dense layers) cost 0.035–0.2 KLD per ms as RTN Q5_0.
  - Their inputs are also the anisotropic ones (§2.2): attention-norm output in layers 0–16 has median max-to-mean
    channel energy of 12–31×, shared-expert down inputs 18–96×, dense layers 29–45×.
  - The routed gate/up inputs are nearly flat in layers 2–45 (2.5–7.5×, participation ratio 0.92–0.97). There the
    imatrix already does what a diagonal weighting can, and GPTQ can gain only through correlations between channels.
- **Expected gain (est.), K2h's recipe.** The guide is the fraction r of imatrix RTN's KLD cost that GPTQ removes:
  - **The evidence brackets r widely:**
    - 0–13 % on dense Llama models against imatrix K-quants (gptq-gguf-toolkit);
    - 13 % on MoE experts at 2 bits against imatrix (NVIDIA ModelOpt PR #2840, measured as KL);
    - 68 % on Flint's own head (Q56).
  - **Mid case** (experts r 0.15, always-read r 0.4): at K2h's KLD 0.109, −76 MB per token (2.06 GB, +1.6 % tok/s); at
    KLD 0.117, −126 MB (2.01 GB, +2.7 %).
  - **Head-like case** (always-read r 0.65): −148 / −205 MB, +3.2 / +4.5 %.
  - **Literature case** (0.15 / 0.25): −50 / −101 MB, +1.1 / +2.2 %.
  - **For comparison:** RTN alone reaches KLD 0.117 at −50 MB (K1h's groups). So GPTQ adds +1–3.5 points at the bar
    and +1–3 points at K2h's quality.
  - With a Pascal Q6_K kernel (Q101), GPTQ-rounded Q6_K for K2h's remaining Q8_0 tensors adds up to est. −0.21 GB
    (~+4 %) at +0.004–0.009 KLD.
- **GPTQ is the method to try; AWQ and learned rounding are not first.**
  - AWQ written into Q4_1 lost to llama.cpp's own imatrix Q4_1 on Llama-2-7B (PPL 5.8952 against 5.8605).
  - AutoRound's tuned GGUF files sit at noise level against its own optimized RTN.
  - YAQA beats GPTQ by 24–36 % KL at INT4 g32, but costs 30–50 GPU-hours for an 8B dense model and has no MoE path.
  - All AWQ folds Flint needs are exact (§3.3), so AWQ stays a zero-byte add-on to test after GPTQ.
- **Smallest decisive first experiment (E1, ~3 GPU-hours, 1–2 days of scripting).**
  - Build G3 with GPTQ-rounded Q5_0 in three of Q77's expensive groups, one group per file: attention output 0–8,
    shared expert 9–16, attention k/v 9–16.
  - Score each against Q77's RTN scores of the same groups on the same GPU set, plus an offline H-weighted error proxy
    for every tensor class.
  - **Kill:** GPTQ recovers less than a quarter of RTN's KLD cost in all three groups. Then body GPTQ is worth
    < 0.3 ms and the expert work is not started.

## Ranked table

Gains and bytes are against K2h (2.139 GB per token, KLD 0.109, ~26.5 ms per token on six cards, est.). "r" is the
fraction of imatrix RTN's KLD cost that GPTQ removes; tok/s follows the fork's bytes-to-time rates above.

| # | Tensor class | Method | Expected KLD gain (est.) | Bytes it lets us drop (est.) | Effort | First experiment | Kill criterion |
|---|---|---|---|---|---|---|---|
| 1 | K2h's Q8_0 always-read groups in layers 0–32: attention q 9–24, output 0–32, k/v 0–32, shared expert 2–32, dense 0–1 (0.82 GB) | GPTQ into Q5_0: searched block scales, column order, H from the file's own inputs | Each group's Q77 cost falls by r ≈ 0.25–0.65; the expensive groups have the anisotropic inputs, so the upper half is plausible there | At K2h's KLD 50–150 MB per token (+1.1–3.2 %); at KLD 0.117, 100–205 MB (+2.2–4.5 %). Counted with row 2's headroom | Capture tool + `gptq_body.py` from `gptq_head.py`: 1–2 days; ~3 GPU-h | **E1:** G3 + GPTQ-Q5_0 for o0-8, sh9-16, kv9-16 against Q77's RTN scores (G3 0.1040; +0.0105, +0.0071, +0.0070) | Recovery < 1/4 of RTN's cost in all three groups |
| 2 | K2h's existing Q5_0 body tensors: routed down 2–45, gate/up 9–48, attention q 0–8 and 25–48, output 33–48, k/v 33–48, shared 33–45 | GPTQ at the same type (no byte change) | Base KLD −0.004 to −0.010: r × ~0.027 (experts ~0.011–0.016 of it, always-read ~0.011) | None directly; the headroom is what row 1 spends | Experts need per-expert H: 3–4 days, ~5–8 GPU-h | **E2:** offline proxy on 4 expert layers, then K2h with GPTQ-Q5_0 experts | Expert proxy error ratio > 0.85 |
| 3 | Routed gate/up 9–48 | GPTQ into Q4_1 | RTN +0.022 (Q36, sub-additive); GPTQ leaves (1 − r) of it | 31.5 MB, 0.34 ms (+1.3 %) | With E2 | E2 probe: G3 with gate/up 9–48 GPTQ-Q4_1 against RTN-Q4_1 | Recovery < 0.7: then dearer than the frontier |
| 4 | Routed down 2–45 | GPTQ into Q4_1 | RTN est. +0.009 (Q18 minus F1→F3) | 17.3 MB, 0.19 ms (+0.7 %) | With E2 | Same probe | Recovery < 0.55 |
| 5 | K2h's whole Q8_0 pool (0.93 GB, head excluded) | GPTQ into Q6_K (needs Q101's kernel) | RTN est. +0.012 (Q90: +0.014 for G3's pool); with GPTQ +0.004–0.009 | −0.21 GB, est. −1.1 ms (+4 %) at Q6_K ~140 GB/s | GPTQ writer for Q6_K sub-blocks of 16 with int8 scales, +2 days | After Q101's rig gate: G3 with GPTQ-Q6_K against Q90's RTN-Q6_K (0.118) | Q101 below 130 GB/s, or recovery < 1/4 |
| 6 | Residual readers (q/k/v; routed + shared gate/up + router), v → output, up → down | AWQ-style per-channel scaling folded into norm weights, v rows and up rows: exact, 0 bytes | Est. ≤ 10 % of RTN's cost on top of imatrix; may help the 8-bit activation blocks of layers 46–48 | 0 directly | 2–3 days (α search per layer) | After E1: as a pre-transform on E1's three groups | < 0.002 KLD on the three groups together |
| 7 | Always-read dense part | Learned rounding: YAQA (full-model KL Hessian), AutoRound | AutoRound at noise against imatrix; YAQA −24 to −36 % KL against GPTQ (dense 8B, g32) | Beyond row 1 if YAQA's margin holds | Weeks; 30–50 GPU-h per 8B in the paper | Only after E1 passes (breakthrough candidate 6) | E1 fails |
| — | Rejected | Q8_0 at 8 bits with GPTQ; experts in Q4_0; an all-Q4_1 body | Q8_0's error is below Flint's compute-path floor; Q4_0 decodes at Q4_1's time per weight here; all-Q4_1 (0.280) would need ~70 % of imatrix's excess removed, which is QAT (Q40) | — | — | — | — |

## 1. Evidence: what better rounding gains over RTN at 4–5 bits

### 1.1 Against plain RTN, group 128 (dense models)

WikiText-2 perplexity, INT4 group 128, RTN / GPTQ / AWQ / learned rounding, FP16 in brackets (P):

| Model | RTN | GPTQ | AWQ | Learned | Source |
|---|---|---|---|---|---|
| Llama-2-7B | 5.73 | 5.69 (act-order 5.63) | 5.60 | — (5.47) | AWQ Table 4 |
| Llama-2-7B | 5.72 | 5.61 | 5.62 | OmniQuant 5.58 | OmniQuant Table 1 |
| Llama-2-13B | 4.98 | 4.98 | 4.97 | OmniQuant 4.95, SignRound 4.96 (4.88) | OmniQuant, SignRound |
| Llama-2-70B | 3.46 | 3.42 | 3.41 | OmniQuant 3.40, SignRound 3.40 (3.32) | same |
| Mistral-7B | 5.42 | 5.37 | 5.37 | SignRound 5.34 (5.25) | SignRound |
| Llama-1-7B | 5.96 | 6.22 (act-order 5.83) | 5.78 | OmniQuant 5.77 (5.68) | AWQ, OmniQuant |

- At group 128, GPTQ removes ~0–50 % of RTN's gap to FP16 (median ~30 %), AWQ 10–55 %, learned rounding 20–68 %.
  Without groups the gains are larger (Llama-2-7B per channel: RTN 6.11, GPTQ 5.83, OmniQuant 5.74).
- **The only group-32 GPTQ table** is ZeroQuant-V2 on OPT (Table E.15, asymmetric W4): GPTQ removes 62–88 % of RTN's
  gap. OPT is outlier-heavy, so this overstates the gain for Llama-type models.
- **YAQA** (Table 6, INT4 with an FP16 absmax scale per 32 weights, Llama-3.1-8B-Instruct): KL 0.033 (LDLQ, which is
  GPTQ), 0.025 (YAQA-A), 0.021 (YAQA-B). No RTN row exists in any version, so it ranks rounders against each other,
  not against RTN.

### 1.2 Against imatrix RTN in GGUF blocks

The baseline that matters here. llama.cpp's imatrix is a diagonal Hessian with a per-block weighted scale search;
ikawrakow introduced it as dropping the off-diagonal terms of the GPTQ/AWQ-style objective (PR #4861).
- **The imatrix alone already does most of what AWQ does at group 32.**
  - Imatrix vs plain RTN (#4969): error relative to FP16 0.51–0.88× for Q4_0, 0.31–0.55× for Q4_1, 0.31–0.89× for
    Q5_0.
  - AWQ in GGUF vs plain RTN removes 13–51 %:
    - llama.cpp's AWQ path #4593: Llama-2-7B Q4_1 6.0656 → 5.9849;
    - AMD Quark's AWQ → Q4_1: 5.9994 → 5.8952.
  - Imatrix Q4_1 on the same model gives 5.8605, better than AWQ. The AWQ path was removed from llama.cpp in #5768 as
    unused.
- **GPTQ in K-quants is level with imatrix K-quants.** IST-DASLab's gptq-gguf-toolkit on Llama-3.1-8B-Instruct at
  ~4.5 bpw:
  - GPTQ: Wiki2 6.89, C4 11.35, FineWeb 9.36;
  - imatrix K-quant mix: 6.91 / 11.40 / 9.36;
  - F32: 6.73 / 11.01 / 9.14.
  - GPTQ closes 0–13 % of the imatrix gap.
- **Learned rounding:**
  - AutoRound's own GGUF tables (q4_k_s, average accuracy, optimized RTN vs tuned): Llama-3.1-8B 0.6228 / 0.6239;
    Qwen3-8B 0.6303 / 0.6320; Qwen3-30B-A3B 0.6762 / 0.6777, the tuned runs taking hours where RTN takes minutes.
  - Community KLD on Qwen3.8-27B: AutoRound Q4_K_M 0.01466 against an older imatrix Q4_K_M 0.01549; AutoRound Q4_0
    0.0266 against bartowski's imatrix Q4_0 0.0265 (different setups, S).
  - ikawrakow objects that tuning against block outputs on a calibration set overfits (ik discussion #657).
- **Better scale search alone helps Q4_0/Q5_0, not Q4_1.**
  - compilade's draft #12557 lowers imatrix KLD by 13–29 % for those two types: Qwen2.5-Coder-3B Q4_0
    0.0480 → 0.0343, Q5_0 0.0101 → 0.0088; Llama-3.1-8B-Instruct Q5_0 0.0123 → 0.0087.
  - ik_llama.cpp #295 matches it for Q4_0.
  - Q4_1 is untouched: #302 moves Llama-3-8B 7.5309 → 7.5277, because imatrix Q4_1's two-parameter fit is already at
    that search's optimum.
- **No llama.cpp PR adds GPTQ or OBQ error feedback to llama-quantize.** #11024 ("Support GPTQ") was closed as stale.
  The converters dequantize GPTQ checkpoints to float (#14810, #17069), which loses the rounding.

### 1.3 On MoE experts

- **NVIDIA ModelOpt PR #2840** (merged 2026-10-22) runs GPTQ on fused HF MoE experts, with per-expert Hessians from
  routed tokens, 1,024 × 512 calibration tokens and damping 0.3.
  - IQ2_XXS on the routed experts only, KL against BF16 on wikitext-2, as max-abs / imatrix / GPTQ:
    - Qwen3.8-Flash-Next 0.2971 / 0.2863 / 0.2492 (−13.0 % against imatrix, z = −10.8);
    - GLM-5.3-Flash 0.7402 / 0.7211 / 0.6298 (−12.7 %).
  - The only MoE expert result against an imatrix baseline found. It is at 2 bits, where rounding gains are usually
    larger than at 5.
- **MoEQuant** (W4 per channel, no groups, average of 7 tasks), FP / RTN / AWQ / GPTQ:
  - Qwen-MoE-14B 51.20 / 36.89 / 42.52 / 49.00;
  - Mixtral 56.80 / 47.18 / 48.64 / 53.42;
  - DeepSeek-MoE-16B 40.86 / 35.85 / 39.23 / 39.01.
  - Per-channel W4 is far from group-32 blocks with an imatrix. Its own additions, expert-balanced sampling and a
    route-weighted Hessian, add +0.86 and +0.49 points on DeepSeek-MoE-16B.
- **No RTN baseline in the large-scale runs.** IST-DASLab's MoE-Quant (GPTQ on DeepSeek-R1, 256 experts, g128) recovers
  100.3 % (OpenLLM v1) and 98.8 % (reasoning) with experts only, and 96.5 % with every layer quantized. Red Hat's GPTQ
  W4A16 cards for Qwen3-30B-A3B (98.2–98.5 %) and Qwen3-Next-80B-A3B (99.8–100.5 %) also report no RTN baseline.
- **Where bits pay in an MoE.** QuantMoE-Bench (GPTQ g128 on DeepSeek-MoE-16B at 2 bits) finds attention and shared
  experts gain the most per extra bit (average 49.4 → 56.8 → 61.6). Choosing experts by routing frequency is barely
  better than random. Q18 and Q77 found the same on Flint.

### 1.4 What this predicts for Flint (est.)

- **Literature prior against imatrix RTN at 4–5 bits:** r ≈ 0–0.15 on dense models, ~0.13 on MoE experts at 2 bits.
- **Flint's own head: r = 0.68.**
  - The head's input has one dimension at 285× the average mean square, and 99.4 % of its H-weighted energy lies in
    64 directions.
  - Strong correlations are exactly what GPTQ uses and the diagonal imatrix cannot.
- **Which side each class sits on.** The body inputs that resemble the head's are the anisotropic ones of §2.2: the
  attention-norm output in layers 0–16 and 46–48, the shared-expert SwiGLU output, the dense layers, and the FFN-norm
  output in 46–48. These are also Q77's most expensive groups.
  - The routed gate/up inputs of layers 2–45 look like the near-isotropic case where the literature prior applies.
- **Scenarios used in §4:** experts r 0.15 (0.4 optimistic), always-read tensors r 0.25 / 0.4 / 0.65.

## 2. MoE specifics: per-expert Hessians for 256 experts

### 2.1 What GPTQ needs per expert

- **One Hessian per expert serves gate and up.** H_e = Σ x xᵀ over the tokens routed to expert e, x the FFN-norm
  output (2,048). MoE-Quant ties gate and up the same way (`--tie_gptq_handles`).
- **Down needs a second, smaller Hessian:** 512 × 512 over that expert's SwiGLU outputs.
- **The shared expert's gate/up Hessian is the all-token Hessian of the same input.** It is a natural prior for rare
  experts.
- **Optional, route weighting.** Each expert's output enters the residual stream multiplied by its route weight (0.31
  to 0.10 by rank, Q51), so weighting each token by its route weight (MoEQuant: X·diag(c)·Xᵀ) or by its square (the
  output error's scale) focuses H on tokens where the expert matters. This is an ablation, not the default.

### 2.2 Tokens per expert and input structure, from Unsloth's importance matrix

Unsloth's imatrix counts the tokens routed to each expert (687 × 512 = 351,744 tokens; mean 351,744 × 6 / 256 = 8,244
per expert). Read for this note:

| Layer | Min | 1st pct | 5th pct | Median | 95th pct | Max | Experts < 2,048 tokens | Top quarter's share |
|---|---|---|---|---|---|---|---|---|
| 2 | 1,427 | 2,543 | 3,776 | 7,690 | 14,712 | 25,221 | 2 | 0.39 |
| 16 | 1,813 | 2,018 | 3,139 | 7,348 | 15,238 | 31,854 | 3 | 0.42 |
| 32 | 744 | 3,140 | 4,346 | 7,704 | 13,670 | 29,005 | 1 | 0.38 |
| 44 | 2,848 | 3,742 | 4,600 | 7,654 | 14,270 | 33,569 | 0 | 0.38 |
| 47 | 0 | 1,983 | 2,929 | 6,572 | 20,224 | 52,322 | 4 | 0.48 |
| 48 | 89 | 1,493 | 2,911 | 6,823 | 18,284 | 64,004 | 7 | 0.48 |
| All 47 | 0 | 2,451 | 4,643 (10th pct) | | | | 66 of 12,032 (734 < 4,096; 5 < 512; 1 with none) | |

- **Coverage.** Routing is moderately skewed (top quarter 36–48 %) and most skewed in layers 46–48.
  - With ~1 M calibration tokens the mean rises to ~23,400 and the 1st percentile to ~7,000 (est., scaling the counts).
    Nearly every expert's 2,048 × 2,048 Hessian is then well determined; the down Hessians (512) always are.
- **Published practice for rare experts:** natural routing with 0.5–2 M tokens (MoE-Quant: 512 × ≤ 4,096; ModelOpt:
  1,024 × 512).
  - Experts with no tokens fall back to RTN or a data-free grid (MoE-Quant, ModelOpt, QMoE), with stronger damping
    (0.1–0.3) for few-token experts.
  - Scale-QLoRA's DeepSeek-V4 run had 34 % of gate/up Hessians below rank and saw no visible damage with 1 % damping.
  - llm-compressor instead runs every expert on every token by default (`moe_calibrate_all_experts`), which gives each
    expert the all-token Hessian; no A/B of the two modes is published.
- **For Flint (est.):** 1 % damping, raised to 10 % when an expert has fewer than 2 × 2,048 tokens, and a blend toward
  the shared expert's all-token H (scaled to the expert's count) below 2,048 tokens. That covers the ~1 % of experts
  short of data. Whatever the rule, the few affected experts carry ≤ 0.5 % of the routed tokens.
- **Input structure** (diagonal of H from in_sum2/counts):
  - Columns: median over the band of max channel energy over mean, largest single layer in brackets, and participation
    ratio (Σe)² / (n·Σe²), where 1 means flat.
  - Routed-expert values are medians over experts.

| Input (tensors that read it) | Layers 0–1 | 2–8 | 9–16 | 17–24 | 25–45 | 46–48 |
|---|---|---|---|---|---|---|
| Attention-norm output (q, k, v) | 31 (48), PR 0.59 | 23 (65), 0.68 | 12 (25), 0.89 | 2.5, 0.97 | 2.1–2.3, 0.97–0.98 | 39 (127), 0.33 |
| Attention result (output) | 10.5, 0.29 | 5.4 (12), 0.64 | 6.4 (33), 0.71 | 4.1, 0.80 | 5.6–6.7 (66), 0.60–0.68 | 24 (132), 0.54 |
| FFN-norm output (routed and shared gate/up, router) | dense: 45, 0.41 | 6.5 (26), 0.92 | 7.5, 0.93 | 2.9, 0.96 | 2.5–4.2, 0.96–0.98 | 109–430 (658), 0.01–0.11 |
| Routed SwiGLU output (routed down) | dense: 29, 0.51 | 10.7 (23), 0.60 | 5.9, 0.75 | 5.3, 0.77 | 7.2–7.9, 0.66–0.70 | 14 (94), 0.45 |
| Shared SwiGLU output (shared down) | — | 8.8, 0.71 | 21 (240), 0.54 | 18, 0.44 | 18–96 (359), 0.07–0.36 | 68 (147), 0.05 |

- **Where the anisotropy is:** the attention inputs of layers 0–16 and 46–48, every shared-expert down input, the dense
  layers and layer 46–48's FFN input. Q77's sensitivity agrees: kv 0–16, shared 2–16, attention output 0–24 and the
  dense layers are its dearest groups per MB.
- **Where the inputs are flat:** routed gate/up in 2–45 and q/k/v in 17–45. Imatrix weighting has little to work with
  there, and so, probably, does GPTQ.
- The diagonal does not show correlations, so this is a proxy. E1's capture computes each H's eigenvalue spectrum and
  settles it.

### 2.3 Memory and time on the quant machine (est.)

| Hessian | Input dim | Size (FP32) | Per MoE layer |
|---|---|---|---|
| Routed gate+up, one per expert | 2,048 | 16.8 MB | 256 → 4.29 GB |
| Routed down, one per expert | 512 | 1.05 MB | 256 → 0.27 GB |
| Shared gate+up (= all-token FFN input) | 2,048 | 16.8 MB | 0.017 GB |
| Shared down | 1,024 | 4.2 MB | 0.004 GB |
| Attention q/k/v (one) | 2,048 | 16.8 MB | 0.017 GB |
| Attention output | 4,096 | 67 MB | 0.067 GB |
| Dense layers 0–1 (gate+up, down) | 2,048, 6,144 | 16.8 + 151 MB | 0.17 GB per dense layer |

- **Memory.** ~4.7 GB per MoE layer, ~220 GB for all 47 in FP32 (half if only one triangle is kept). They are built and
  used in passes of ~8 layers on the two V100s (32 GB each) or ~4 layers per 3090 and discarded after the solve.
  - Keeping them on the scratch disk would break its ≥ 50 GB free rule.
- **Accumulation.** ~53 M multiply-adds per token per MoE layer (6 experts × (2,048² + 512²) plus the dense
  Hessians): ~2.5 G per token for all layers, 5 PFLOP for 1 M tokens, minutes of FP32 GEMM on one card. The forward
  passes and moving activations dominate.
- **GPTQ solve.** Columns × rows² per matrix: 4.8 G multiply-adds per expert (gate and up 2.15 G each, down 0.54 G),
  1.24 T per layer, 58 T for all experts; ~3 T for all always-read tensors.
  - Batched over experts in torch on a GPU this is under an hour.
  - The CPU, row-parallel numpy of `gptq_head.py` would take hours for the experts.
- **Capture.** Per pass, one prefill of the calibration text through the GGUF with a callback.
  - Est. 10–30 min for 1 M tokens on a 3090 pair.
  - ~5 passes for the routed layers 9–48: 1–3 GPU-hours.

## 3. Format compatibility

### 3.1 Which blocks GPTQ can write

| Type | Block | GPTQ group and scale rule | Writer status |
|---|---|---|---|
| Q5_0 | 32 weights, FP16 d, value d·(q − 16) | Block = group; d set when GPTQ reaches the block, from the updated weights (ggml's max rule or its diag(H)-weighted search) | Exists (`gptq_head.py`, tested against gguf-py and llama-quantize) |
| Q4_0 | 32, FP16 d, d·(q − 8) | Same | Exists |
| Q4_1 | 32, FP16 d and m, d·q + m | Asymmetric group: d and m from the updated block (ggml's `make_qkx3_quants`-style weighted search), then sequential rounding | Packer to add (a few lines) |
| Q8_0 | 32, FP16 d, q in [−127, 127] | Possible; nothing to gain | Packer to add for completeness |
| Q6_K | 256 = 16 sub-blocks of 16, int8 sub-scales, FP16 d | Group = 16-weight sub-block, super-block d fixed first; gptq-gguf-toolkit does GPTQ on K-quant scales | To write; only with Q101's kernel |

- **The fit is natural.** GGUF blocks run along the input dimension, which is GPTQ's column order, so the solve sweeps
  columns block by block, as for the head.
- **Act-order breaks the scale timing.** It needs every block's scale fixed beforehand, and on the head it gave nothing
  (0.00314 in both orders).
- **Expert tensors are 3-D in GGUF**: `ffn_gate_exps` has ne = [2,048, 512, 256] and `ffn_down_exps` [512, 2,048,
  256]. Each expert slice is solved alone and written back in place, and `compose_gguf.py` swaps whole tensors into a
  recipe file.
- **Solve against the BF16 GGUF, not HF safetensors.** The quant machine has Flint's BF16 GGUF, so source weights and
  captured inputs share llama.cpp's tensor layout and no HF-to-GGUF mapping is needed.

### 3.2 Where the gain is, by format (est.)

- **Q8_0: nothing.**
  - Q8_0's relative weight error (~0.4–0.6 %) sits below Flint's compute-path floor: Unsloth's Q8_0 scores KLD 0.052
    against BF16, and one file on two GPU types differs by 0.065 (Q21).
  - In a mix, Q8_0 tensors are byte-identical to the Q8_0 reference's.
- **Q5_0 always-read tensors: the main case.**
  - Q77's RTN costs of 0.0014–0.0105 per 8-layer group are measurable.
  - Every group GPTQ makes cheap saves 0.375 bytes per weight.
- **Q5_0 routed experts: headroom only.**
  - Their RTN excess is small: down 2–45 ~0.011 (F1 → F3), gate/up 9–48 ≤ 0.005 (GQ8 against Q2; F3 → G3).
  - It is spread over flat inputs (gate/up), so GPTQ buys ≤ ~0.002–0.006 KLD there.
- **Q4_1:** only as a probe that makes the expert signal large enough to measure. As a target it saves 9 % of the
  expert bytes (row 3).
- **Attention q/output in Q5_0 in layers 25–48:** flat attention-norm inputs in 25–45 (participation ratio
  0.97–0.98). GPTQ gains are likely small there, except in the 14 global layers, where
  [long-context quality](long_context_quality.md) X2 already proposes GPTQ calibrated on 8k transcripts.

### 3.3 AWQ-style scaling: which folds are exact in Flint

AWQ multiplies weight column j by s_j and divides input channel j by s_j, folding the division into whatever produced
the input. In Flint's GGUF these folds are exact (up to float rounding of the folded values) and cost no bytes or
launches:

| Input | Fold the 1/s into | Readers that take the s | Constraint |
|---|---|---|---|
| Attention-norm output | `attn_norm` weight (F32) | q, k, v | One s per layer for all three |
| FFN-norm output | `ffn_norm` weight | All 256 routed gate/up, shared gate/up, **and the router** | One s per layer. The router's columns must take s too or routing changes: llm-compressor's AWQ added the router for Qwen3-MoE (#2451: PPL 11.65 → 11.28), and leaving it out was reported to flip 14–17 % of top-8 slots on Qwen3.6 (#3337, unverified) |
| Attention result | `attn_v` rows | attention output | s indexed by (KV head, dim), tied across the 8 query heads of a group; sinks and softmax are unaffected; the f16 KV cache stores the scaled v |
| SwiGLU output | `ffn_up` rows (routed per expert, shared, dense) | down | silu(gate)·up is linear in up, so one s per expert; gate's rows cannot take it |
| Final norm output | `output_norm` weight | head | Exact |

- **Row scalings leave the producer's own quantization unchanged.** A 32-block lies within one row, and RTN is
  scale-equivariant per block. The power-of-two up/down rebalance of [survey A](quant_speed_a_file_only.md) §3.9 is
  the uniform special case of the up → down fold.
- **What it can buy on top of the imatrix is small (est. ≤ 10 % of RTN's cost):**
  - the imatrix already weights salient channels per block;
  - the GGUF evidence above has AWQ losing to imatrix Q4_1 at group 32;
  - AWQ protects a salient weight only when the block's other 31 weights can afford a coarser step.
- **One side effect may help:** dividing the dominant channels of layers 46–48 (FFN-norm channel energy up to 658× the
  mean) shrinks the outliers inside llama.cpp's 8-bit activation blocks, the effect that made rotations hurt in Q13
  when it ran the other way.
- **Order:** test it after GPTQ, as a pre-transform on E1's groups (row 6).

## 4. Expected payoff on K2h's recipe

What K2h reads per token (est., from Q73/Q77's recipes and the shapes above; matches the measured 2.139 GB):

| Class | Q8_0 layers | Q5_0 layers | Q8_0 MB | Q5_0 MB |
|---|---|---|---|---|
| Attention q | 9–24 | 0–8, 25–48 | 142.6 | 190.3 |
| Attention output | 0–32 | 33–48 | 294.2 | 92.3 |
| Attention k + v | 0–32 | 33–48 | 73.5 | 23.1 |
| Shared expert | 2–32, 46–48 | 33–45 | 227.2 | 56.2 |
| Dense layers | 0–1 | — | 80.2 | — |
| Routed gate/up | 2–8 | 9–48 | 93.6 | 346.0 |
| Routed down | 46–48 | 2–45 | 20.0 | 190.3 |
| Output head | — | GPTQ | — | 211.4 |
| Router | F32 | | 98.6 | |
| **Total** | | | **931** (+99 router) | **1,110** |

Routed experts are 650 MB (30 %), the always-read tensors 1,180 MB (55 %).

**Cost of moving each remaining Q8_0 group to RTN Q5_0** (Q77, KLD per ms saved, 0.0056 ms per MB), plus the two
Q4_1 expert moves (0.0107 ms per MB, since Q5_0 is slow per byte):

| Group | MB | ms | RTN KLD | KLD per ms |
|---|---|---|---|---|
| sh25-32 / o25-32 | 18.9 / 25.2 | 0.11 / 0.14 | 0.0021 / 0.0030 | 0.020 / 0.021 |
| q17-24 / q9-16 | 25.2 each | 0.14 each | 0.0039 / 0.0050 | 0.028 / 0.035 |
| gate/up 2–8 | 33.0 | 0.19 | 0.0060 | 0.032 |
| kv25-32 / sh17-24 / dense 0–1 | 6.3 / 18.9 / 28.3 | 0.04 / 0.11 / 0.16 | 0.0014 / 0.0046 / 0.0072 | 0.040 / 0.043 / 0.045 |
| down 2–45 → Q4_1 | 17.3 | 0.19 | est. 0.009 | 0.048 |
| o17-24 / o9-16 / o0-8 | 25.2 / 25.2 / 28.3 | 0.14 / 0.14 / 0.16 | 0.0079 / 0.0090 / 0.0105 | 0.056 / 0.064 / 0.066 |
| gate/up 9–48 → Q4_1 | 31.5 | 0.34 | est. 0.022 | 0.065 |
| sh9-16 / sh2-8 | 16.5 each | 0.09 each | 0.0071 / 0.0073 | 0.077 / 0.079 |
| kv17-24 / kv0-8 / kv9-16 | 6.3 / 7.1 / 6.3 | 0.04 each | 0.0031 / 0.0064 / 0.0070 | 0.088 / 0.161 / 0.198 |

**Payoff model (est.)**:
- **Steps.**
  1. GPTQ first lowers K2h's KLD by r times the excess of its existing Q5_0 tensors: experts ~0.016, always-read
     ~0.011.
  2. Groups then move to GPTQ-Q5_0, cheapest per ms first, each at (1 − r) times its RTN cost, until the target KLD is
     reached.
- **Assumptions.** Costs are taken as additive. Round 5 found them slightly sub-additive (K2 predicted 0.108, measured
  0.1068), so the model is a little conservative.
- **Token time.** ~26.5 ms (est.) for K2h on six cards.

| Scenario | r experts / always-read | At K2h's KLD (0.109) | At KLD 0.117 |
|---|---|---|---|
| RTN only | 0 / 0 | — | −50 MB, −0.28 ms, +1.1 % (≈ K1h's groups) |
| Literature | 0.15 / 0.25 | −50 MB (2.09 GB), +1.1 % | −101 MB (2.04 GB), +2.2 % |
| Mid | 0.15 / 0.40 | −76 MB (2.06 GB), +1.6 % | −126 MB (2.01 GB), +2.7 % |
| Head-like | 0.15 / 0.65 | −148 MB (1.99 GB), +3.2 % | −205 MB (1.93 GB), +4.5 % |
| Optimistic | 0.40 / 0.65 | −181 MB (1.96 GB), +4.0 % | −248 MB (1.89 GB), +5.5 % |

- **What fills the budget.** In every scenario the groups chosen are always-read ones: shared expert 17–32, attention
  output 25–32 then 0–24, q 9–24, k/v 17–32, the dense layers. Routed gate/up 2–8 joins only at expert r 0.4.
- **What never gets in.** The Q4_1 move of routed gate/up 9–48 never enters; down 2–45 → Q4_1 enters only above r 0.6.
- **What the experts contribute:** raising expert r from 0.15 to 0.4 adds ~35–45 MB (~1 point). Experts are the
  smaller lever, through headroom.
- **In tok/s:** the mid case at KLD 0.117 takes K2h from ~37.7 to ~38.7 tok/s; the head-like case to ~39.4.
- **The Q6_K route (row 5) stacks with this.**
  - Once Q101's kernel passes ≥ 130 GB/s on the rig, the Q8_0 tensors left after the moves above go to GPTQ-Q6_K.
  - All of K2h's 0.93 GB pool in Q6_K is −0.21 GB, est. −1.1 ms, at est. +0.012 RTN (+0.004–0.009 with GPTQ).
  - A per-group choice between GPTQ-Q5_0 and GPTQ-Q6_K is a knapsack, which E1's measured r feeds.
- **For scale:** K2h still reads its router in F32 (99 MB). The F16 router costs nothing in quality (Q35) and saves
  49 MB per token, more than routed gate/up → Q4_1 would.

## 5. Plan on the quant machine

### 5.1 Pipeline

1. **Calibration text** (never wikitext-2 test, which KLD scores):
   - the head's set: wikitext-2 train, 500 × 512 tokens, and Flint's own transcripts from Q51;
   - plus Q41's BFCL transcripts (agent format), and 8k-token chunks of them for the global layers;
   - ~0.5 M tokens for the dense tensors, ~1 M for the experts.
2. **Capture** (code on the laptop, runs on the quant machine).
   - Generalize `head_dump` (or llama-imatrix's collector, which already sees every product's input and the expert ids)
     into a tool that streams chosen graph tensors per ubatch as F16 rows to a torch accumulator on a GPU.
   - Tensors to stream:
     - `attn_norm-N`;
     - the attention result before the output projection;
     - `ffn_norm-N`;
     - `ffn_moe_topk-N` and `ffn_moe_weights_norm-N`;
     - `ffn_moe_swiglu-N`;
     - the shared expert's SwiGLU output.
   - The accumulator builds H per tensor class, per expert for routed tensors, plus token counts.
   - Source file: the mix being improved (G3 for E1, since Q77's RTN references are G3-based; K2h afterwards). The
     inputs are then the deployed file's own activations, as in Q56.
3. **Solve.** `gptq_body.py`, grown from `gptq_head.py`.
   - Reads the BF16 GGUF tensor, solves per row block or per expert with torch on a GPU, and writes the recipe's type
     (Q5_0, Q4_1; later Q6_K) with ggml's scale rules.
   - Damping 1 %; rare experts as in §2.2.
   - A `proxy` mode reports the H-weighted relative output error, tr(ΔW H ΔWᵀ) / tr(W H Wᵀ), for the tensor as it
     stands in the mix (the real imatrix-RTN baseline) and for GPTQ. On the head this proxy predicted the full-model
     KLD differences (Q56).
   - Tests as in `test_gptq_head.py`: H = I reproduces RTN; Q4_1 packing against gguf-py; RTN byte-identical to
     llama-quantize with Unsloth's imatrix.
4. **Compose and score.** `compose_gguf.py` swaps the GPTQ tensors into the mix. KLD against BF16 runs on the GPU set
   that scored the reference (80 × 512 wikitext-2, rescale off).
   - Each composed file is ~31–33 GB, so delete it after scoring.

### 5.2 Experiments in order

- **E1 (decisive, ~3 GPU-hours).**
  - Capture the dense inputs of all 49 layers from G3 (~0.5 GPU-h).
  - Run the proxy for every always-read tensor and print each H's eigenvalue spectrum.
  - Build three files, each G3 with one group in GPTQ-Q5_0: o0-8, sh9-16, kv9-16.
  - Score them with a G3 rescore on Q77's GPU set (4 runs, ~2 GPU-h).
  - Recovery f = (KLD_RTN − KLD_GPTQ) / (KLD_RTN − KLD_G3), where KLD_RTN is Q77's RTN score for that group.
  - **Pass:** f ≥ 0.4 in at least two groups. **Kill:** f < 0.25 in all three, given a paired noise of ~0.002 on
    differences of 0.007–0.011.
- **E2 (experts, ~5–8 GPU-hours; only if E1 passes or the expert proxy on layers 9, 25, 41 and 47 is ≤ 0.8).**
  - Per-expert H for gate/up 9–48 and down 2–45, in passes of 8 layers.
  - (a) K2h with GPTQ-Q5_0 experts, which measures the headroom.
  - (b) G3 with gate/up 9–48 in GPTQ-Q4_1 against RTN-Q4_1 (donor tensors from the S1/S2 files). This probe has a
    ~0.02 RTN signal, so it measures expert r well above the noise.
  - Variant: route-weighted H.
- **E3 (the K2h-G mix, ~8–12 GPU-hours).**
  - Re-run §4's ranking with the measured r per class.
  - Compose 4–6 candidates around KLD 0.109 and 0.117 with GPTQ on every touched tensor.
  - Score them, then run the winner through Q106's long-context KLD, Q41's task lanes and the rig's tg128 against
    UD-Q4_K_XL, as for K2h.
- **E4 (after Q101's rig gate).** GPTQ-Q6_K for the remaining Q8_0 pool, against Q90's RTN-Q6_K.

Total to a K2h-G decision: est. 15–25 GPU-hours and about a week, excluding Q41's task runs.

### 5.3 Risks

- **Routing-flip noise** (~0.005 in whole-mix comparisons) hides gains below ~0.003. Hence the probes with large RTN
  costs (E1's groups, E2's Q4_1 probe) and paired scoring on one GPU set.
- **Calibration overfit** (ikawrakow's objection to learned rounding). GPTQ fits less than tuned rounding, and the
  head kept ~90 % of its gain on another domain. Keep wikitext test held out and check transcript KLD (Q106's M1).
- **Long context.** GPTQ calibrated on 512-token text could fit the global layers' short-range behaviour only.
  Calibrate them on 8k chunks ([long-context quality](long_context_quality.md) X2).
- **Upstream changes.** Inputs come from the file being improved; once many tensors change, re-capture once from the
  new mix before the final solve.
- **Activation rounding.** llama.cpp rounds activations to 8 bits per 32 values, which GPTQ does not model. On the
  head this changed logits by only 0.1–0.3 %.
- **Overflow.** GPTQ on up changes the down-projections' inputs slightly. The fork's FP32 activation sums (Q76) make
  that irrelevant for decode; prompt processing (MMQ) keeps Q5_0's scale-only layout.
- **Disk and memory:** ≥ 50 GB free on the scratch disk, Hessians in passes (§2.3), one composed file at a time.

## Already covered

| Topic | Where |
|---|---|
| GPTQ for the always-read tensors, YAQA as the follow-up, "attention output 0–32 in Q5_0" as a first test | [Breakthrough candidates](breakthrough_candidates.md) §6; this note adds the experts, the payoff model, the input statistics and a first experiment against Q77's existing RTN files |
| GPTQ head, screened head | Q56, Q91, Q92, [quants/gptq_head](../../../quants/gptq_head/README.md) |
| GPTQ for the global layers' Q5_0 attention, calibrated on long transcripts | [Long-context quality](long_context_quality.md) X2 |
| Rounding tools surveyed before Q56 (PR #12557, AutoRound, no GPTQ → Q4_x writer for MoE) | [Survey A](quant_speed_a_file_only.md) §3.1 |
| Quantization-aware recovery (QAT, Block-AP), AutoRound as its cheap baseline | [Survey C](quant_speed_c_retraining.md) §3.7–3.8, Q40 (parked) |
| 6-bit rung and its kernel | Q90, Q101, [Pascal forks and quants](pascal_forks_and_quants.md) |
| Coarser scale groups (GPTQ works with group-64 scales too) | Q97, [unconventional ideas](unconventional_ideas.md) §5 |
| Rotations | Q13, Q15 (no gain on Flint) |
| F16 router (−49 MB per token in K2h, free) | Q23, Q35 |
| Own importance matrix against Unsloth's | Q12 |
| Q5_K experts (same bytes as Q5_0, lower error, 13 % slower per weight) | [Pascal forks and quants](pascal_forks_and_quants.md) idea 8 |

## Not verified

- r for any Flint body tensor. The scenarios rest on the head (0.68), dense-model GGUF results (0–0.13) and one MoE
  result at 2 bits (0.13).
- That diagonal anisotropy predicts GPTQ's gain. The off-diagonal structure of the body Hessians is unknown until
  E1's capture.
- Additivity of Q77's group costs beyond round 5's check, and the RTN cost of the two Q4_1 expert moves (taken from
  Q18/Q36 sums).
- The Q5_0 excess of K2h's existing tensors (~0.027 in total), inferred from F1/F3/G3/GQ8 differences of 0.005–0.011
  that sit near the noise.
- Q6_K's speed on the rig (Q101's gate) and the cost of a GPTQ writer for its sub-blocks.
- Capture throughput with a per-ubatch callback, and the torch solve times. These are arithmetic, not measured.
- Unverified items from secondary or unreviewed sources: llm-compressor issue #3337 and PR #3338 (routing flips under
  AWQ without the router), the Qwen3.8-27B community KLD figures, the gptq-gguf-toolkit README's undated tables.

## Sources

Rounding methods (P):
- E. Frantar et al., "GPTQ", [arXiv 2210.17323](https://arxiv.org/abs/2210.17323).
- J. Lin et al., "AWQ", [arXiv 2306.00978](https://arxiv.org/abs/2306.00978), Table 4.
- W. Shao et al., "OmniQuant", [arXiv 2308.13137](https://arxiv.org/abs/2308.13137), Table 1.
- W. Cheng et al., "Optimize Weight Rounding via Signed Gradient Descent" (SignRound/AutoRound),
  [arXiv 2309.05516](https://arxiv.org/abs/2309.05516).
- Z. Yao et al., "ZeroQuant-V2", [arXiv 2303.08302](https://arxiv.org/abs/2303.08302), Table E.15 (group 32).
- A. Tseng, Z. Sun, C. De Sa, "Model-Preserving Adaptive Rounding" (YAQA),
  [arXiv 2505.22988](https://arxiv.org/abs/2505.22988), Table 6.
- A. Tseng et al., "QuIP#", [arXiv 2402.04396](https://arxiv.org/abs/2402.04396).

GGUF and llama.cpp (P):
- llama.cpp [#4593](https://github.com/ggml-org/llama.cpp/pull/4593) (AWQ, merged 2023-12-27; removed in
  [#5768](https://github.com/ggml-org/llama.cpp/pull/5768), 2024-02-28);
  [#4861](https://github.com/ggml-org/llama.cpp/pull/4861) and [#4969](https://github.com/ggml-org/llama.cpp/pull/4969)
  (importance matrix; imatrix against RTN for legacy types);
  [#12557](https://github.com/ggml-org/llama.cpp/pull/12557) (cumulative-search rounding, draft);
  [discussion #5063](https://github.com/ggml-org/llama.cpp/discussions/5063);
  [#11024](https://github.com/ggml-org/llama.cpp/issues/11024);
  [#14810](https://github.com/ggml-org/llama.cpp/pull/14810), [#17069](https://github.com/ggml-org/llama.cpp/pull/17069).
- ik_llama.cpp [#295](https://github.com/ikawrakow/ik_llama.cpp/pull/295), [#302](https://github.com/ikawrakow/ik_llama.cpp/pull/302),
  [discussion #657](https://github.com/ikawrakow/ik_llama.cpp/discussions/657).
- [IST-DASLab/gptq-gguf-toolkit](https://github.com/IST-DASLab/gptq-gguf-toolkit) README.
- AMD Quark, [GGUF export for llama.cpp](https://quark.docs.amd.com/latest/pytorch/export/gguf_llamacpp.html).
- Intel AutoRound, [docs/gguf_alg_ext_acc.md](https://github.com/intel/auto-round/blob/main/docs/gguf_alg_ext_acc.md),
  [docs/moe.md](https://github.com/intel/auto-round/blob/main/docs/moe.md).
- Unsloth, [Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) (S: method summary and KLD
  table).
- [tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF) (S).

MoE quantization (P):
- NVIDIA Model-Optimizer [PR #2840](https://github.com/NVIDIA/Model-Optimizer/pull/2840) (GPTQ on fused MoE experts,
  KL against imatrix, merged 2026-10-22).
- [IST-DASLab/MoE-Quant](https://github.com/IST-DASLab/MoE-Quant) (GPTQ on DeepSeek-V3/R1; zero-token experts →
  identity Hessian); [RedHatAI/DeepSeek-R1-quantized.w4a16](https://huggingface.co/RedHatAI/DeepSeek-R1-quantized.w4a16),
  [RedHatAI/Qwen3-30B-A3B-quantized.w4a16](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-quantized.w4a16).
- [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor): `moe_calibrate_all_experts` (#1596,
  #1760, #2827; #2826 for never-activated experts), AWQ mappings for Qwen3-MoE, router as a balance layer (#2451).
- X. Hu et al., "MoEQuant", [arXiv 2505.03804](https://arxiv.org/abs/2505.03804), Table 1 and ablation.
- P. Li et al., "Examining Post-Training Quantization for Mixture-of-Experts" (QuantMoE-Bench),
  [arXiv 2406.08155](https://arxiv.org/abs/2406.08155).
- E. Frantar, D. Alistarh, "QMoE", [arXiv 2310.16795](https://arxiv.org/abs/2310.16795).
- "EAQuant", [arXiv 2506.13329](https://arxiv.org/abs/2506.13329); Y. J. Kim et al., "MoQE",
  [arXiv 2310.02410](https://arxiv.org/abs/2310.02410); Scale-QLoRA, [arXiv 2609.04526](https://arxiv.org/abs/2609.04526),
  Appendix K.

This project:
- [Quant registry](../../../quants/experiments.md): Q12, Q13, Q18, Q21, Q35, Q36, Q41, Q45, Q51, Q56, Q73, Q74, Q76,
  Q77 (per-group table), Q90, Q91, Q97, Q101, Q106.
- [quants/gptq_head](../../../quants/gptq_head/README.md) (method, offline proxy, head results).
- [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md) (per-format time per Flint token).
- [token_cost.py](../../../scripts/token_cost.py), [compose_gguf.py](../../../scripts/compose_gguf.py).
- Notes: [breakthrough candidates](breakthrough_candidates.md), [unconventional ideas](unconventional_ideas.md),
  [long-context quality](long_context_quality.md), [Pascal forks and quants](pascal_forks_and_quants.md),
  [survey A](quant_speed_a_file_only.md), [survey C](quant_speed_c_retraining.md).
- Unsloth's Flint importance matrix (`imatrix_unsloth.gguf`, 687 × 512 tokens): per-expert counts and in_sum2 read for
  §2.2.
- llama.cpp fork source: `src/models/flint.cpp` (graph names `attn_norm`, `ffn_norm`, `ffn_shexp`);
  `src/llama-graph.cpp` (`ffn_moe_topk`, `ffn_moe_weights_norm`, `ffn_moe_swiglu`).
