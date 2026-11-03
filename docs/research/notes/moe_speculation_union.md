# Keeping speculative verification cheap on a fine-grained MoE: the union of experts, and what it means for token trees (2026-11-04)

Qwen3.8-Flash-Next is a hybrid mixture-of-experts model: 48 layers (Gated DeltaNet layers and 12 attention layers),
hyper-connections (a four-stream residual), 512 routed experts of which 10 run per token, and a shared expert. On the
rig (nine P106-090 cards: Pascal, compute 6.1, 5 SMs, ~173 GB/s, `dp4a`, no tensor cores; a layer split over PCIe 1.0 x1,
one card working at a time) decode is bandwidth-bound. A one-token step costs 136 ms, and each extra token of a k-token
verify pass adds 50–59 ms ([Q583](../../data/2026-11-03_verify_cost/README.md)). New small-batch kernels
([Q591](../../data/2026-11-04_mtok_experts/README.md)) read each *distinct* expert once across the k tokens, so the
cost of a pass now follows the **union of the experts** its tokens route to. This note answers four questions from
papers, model cards, repositories and pull requests (mostly 2024–2026): how that union grows with k and with tree
shape; which methods keep MoE verification cheap and whether they stay exact; how tree and branch speculation is done
on hybrid recurrent models; and what acceptance lengths coding and agent work reach. It then prices a token tree on this
rig and ranks what to try.

It builds on [idle cards](idle_cards.md), [pipelined speculation](pipelined_speculation.md),
[Flash-Next drafters](flash_next_drafters.md), [speculation plumbing](speculation_plumbing.md) and
[drafter acceptance](drafter_acceptance.md), and does not repeat them. What they settle: guesses can run as one-token
steps on the idle cards at no wall-time cost while at most S_eff − 1 are in flight; batched trees did not pay at Flint's
measured verify cost (c(3) = 2.2, c(4) = 3.6 tokens, Sequoia's dynamic program); DFlash2 and the MTP draft each gain
~1.15× today; a copy-gated chain needs per-shape CUDA graphs. Sources were read on 2026-11-03 and 04; nothing here was
run on the rig.

Conventions:
- "est." marks this note's own arithmetic or inference; the basis is given next to it.
- **τ** is accepted tokens per step, the bonus token included. **u(k)** is the number of distinct routed experts per
  layer among k tokens. **T(k)** is the time of a k-token pass, T1 = T(1).
- **r** is the marginal cost of one more verified token, as a fraction of T1; **S** is the speed-up of a speculation
  scheme over one-token decoding.
- Q-numbers are rows of the [registry](../../../quants/experiments.md).

## Summary

1. **At 512 experts and 10 per token the union is almost linear in k for any block the rig can afford.** Independent
   routing gives u = 19.8 / 38.9 / 48.1 / 74.7 / 138.6 at k = 2 / 4 / 5 / 8 / 16 (10 for one token), and 95 % of the
   experts are touched only after ~152 tokens (Flint ~127, DeepSeek-V3 ~95, Qwen3-30B ~47, Mixtral ~11). The regime
   where verification is nearly free, a block big enough that "every expert is read anyway", starts at ~150 tokens and
   would cost est. ~300 ms of expert reads alone on this rig.
2. **Neighbouring tokens share some experts, by a model-dependent amount.** Distinct experts as a share of independent
   routing, at 4–11 tokens: 46–81 % on the Qwen3 family and GPT-OSS, 70–80 % on a Cohere MoE, 74–87 % on Flint, ~103 %
   on DeepSeek-V3.1. Nothing is published for Qwen3-Next-class models (512 experts, shared expert), and Flash-Next's own
   trace is still planned (Q625). An indirect estimate from Q583's timings puts a real 4-token verify at est. 27–30
   distinct experts per layer, 68–77 % of independent routing.
3. **After the kernels the cost of an extra verified token is est. 8–12 % of a token, down from 37–43 %.** Half is the
   expert union (est. 4.6–5.7 ms per token at k ≤ 8), half is per-row fixed cost (est. 3–5.6 ms: logits row, recurrent
   state rows, small operations). A 5-token pass is est. 1.34–1.39× a one-token pass (1.46–1.51× if the per-row cost is
   today's), a 16-token pass 2.1–2.4×.
4. **A node pays only if its path-acceptance probability exceeds r·S, est. 0.15–0.25 on the rig after the kernels (0.4–0.5
   today).** That rule explains why three DFlash2 positions do not pay today (cumulative acceptance 0.65 / 0.42 / 0.30)
   and why four to six will after the kernels (est. 1.6–1.7× over a one-token baseline; 2.0–2.1× at three drafted
   tokens with the acceptance Q561 measured on agent requests).
5. **Published trees buy 0.01–0.03 accepted tokens per extra node, an order of magnitude below that threshold.** They
   pay on GPUs where a 64–512-node pass costs a few times a token (r < ~0.01). On this rig the first sibling branch is
   worth verifying only where the drafter's top-1 is unsure and its second choice holds ≥ 0.15–0.25 of the probability
   mass; depth on a high-probability path is worth much more. Wide trees are the wrong tool for the aim of 5–20 tokens
   per step; long chains over copied text are the right one (est. 4.5–7.8× inside a copy span).
6. **Exact methods choose which nodes to verify; inexact ones change which experts a node uses.** Cascade, EVICT,
   EcoSpec and CaDDTree are exact and report +5–25 % over fixed settings. MoE-Spec and AcceptMoE cap or share the
   experts of a verify pass, are not exact (−0.27 points of accuracy; −1.4 % acceptance) and report +10–30 % over
   EAGLE-3. With 512 experts per layer a published budget (12.5–50 % of the experts) binds only from k ≈ 7 to
   k ≈ 35 tokens.
7. **Tree verification on Gated DeltaNet layers is solved on newer GPUs and is not on this stack.** Five 2026 papers
   (TreeWY, Bole, SpecLA, Weaver, GDN Tree-Scan) verify a whole tree in one pass with WY or closed-form kernels and no
   per-node state; llama.cpp keeps one 113 MiB state plane per verified token, which fits 3–8 tokens and not 16–32 (est.
   1.8–3.6 GiB), and its only tree prototype excludes hybrid models. **Recording the inputs of the verified tokens and
   replaying the accepted path** (est. ~1 MiB per node) is the enabling piece for long chains and trees here.
8. **Acceptances of 5–20 per step occur in two places only:** function-level code with 16-token block drafters (τ 6–8,
   trees 9–11), and copy spans with retrieval drafts (5.5–10 in coding-agent pipelines). Learned drafters on agent
   sessions reach 3.7–4.2 (DSpark block 7: 3.70; MTP with 7 steps: 4.24), and DFlash2 as evaluated at block 4 is capped
   at 4. It takes
   a per-token match probability of ~0.85 / 0.93 / 0.94 / 0.95, sustained over chains of 8 / 16 / 32 / 64, for τ = 5 / 10 /
   15 / 20.
9. **Idle cards help branches by running them as one-token steps, not by drafting.** A hedge branch in the pipeline
   reads its own experts on an otherwise idle card and costs no union; a batched column costs r on the critical path.
   Drafting cannot hide behind the verify, because a block starts at the bonus token the last card samples. Both need
   Q593 (a wrong pipelined guess costs ~2.9 plain tokens on the rig, 1.6× slower decoding) understood first.
10. **Top ideas** (section 8): finish and time Q591's two kernel halves to k = 8 and trace Flash-Next's routing (Q625):
    1.15× → est. 1.6–2.1× for the existing drafters; long chains over copied text (Q617): est. +20–45 % on agent
    sessions; recorded-input state replay (new): enables chains of 16–32, est. −8–17 % of a 32-token pass.

## 1. The setting in numbers

Measured on the rig ([Q583](../../data/2026-11-03_verify_cost/README.md), Swift-1.5 GSQ-RCO Q2_0 on Miner Llama v3's
base; GPU ms summed over nine cards, llama-bench passes of k tokens):

| Part of the pass | k = 1 | k = 5 | × |
|---|---:|---:|---:|
| routed experts (llama.cpp's Q2_0 kernel) | 27.2 | 70.7 | 2.6 |
| BF16 hyper-connections | 44.6 | 123.0 | 2.8 |
| attention and recurrent projections (Q3_K, IQ4_XS, Q4_K) | 24.5 | 67.9 | 2.8 |
| recurrent state (delta net, convolution) | 4.3 | 6.2 | 1.4 |
| output layer (logits for the last row only) | 3.5 | 3.6 | 1.0 |
| other operations without weights | 26.3 | 30.8 | 1.2 |
| **GPU total** | **139.6** | **323.0** | **2.3** |
| wall time, profiler off | 136 | 360 | 2.65 |

- **The dense parts and the experts both scale because llama.cpp's kernels re-read weights per column.** A real
  4-token verify costs the experts only 45.1 ms (llama-bench's random token ids give 58.3): consecutive tokens of one
  answer touch fewer experts than random ids.
- **Experts per token are now cheap and the union is visible.** The one-token Q2_0 expert kernel takes 6.96 ms against
  llama.cpp's 25.2 ([R16](../../data/2026-11-03_kernel_timing/README.md)); its routed part is 6.06 ms for 10 experts, so
  est. 0.606 ms per distinct expert per layer (summed over layers). The multi-token expert kernel
  ([Q591](../../data/2026-11-04_mtok_experts/README.md)) and the k-column dense kernels
  ([C70](../../data/2026-11-04_kcol_dense/README.md)) are written, built and checked against references, not yet timed on
  a P106.
- **T1 after the fixes is est. 86–90 ms** (136 − ~20 for Q223's experts − ~30 for the hyper-connections at Q546's 11–14 ms
  estimate). The tables below use 90 ms; the ratios move little with it.
- **Sampling and draft costs sit on top of T(k):** a DFlash2 draft pass is est. 15–25 ms, the MTP draft's four passes
  30.9 ms, 18.5 of it the draft's own output layer.

## 2. How the union of experts grows

### 2.1 Independent routing is the ceiling

If each token picks K of E experts independently and uniformly, the expected distinct count is
u(k) = E·(1 − (1 − K/E)^k). [MoESD](https://arxiv.org/html/2505.19645v4) (NeurIPS 2025) derives it, checks it against
DeepSeek-V2-Lite and Qwen1.5-MoE, and gives the token count at which a pass touches a share τ of the experts,
⌈log_{1−K/E}(1−τ)⌉. For Flash-Next (E = 512, K = 10):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| u(k), distinct experts per layer | 10.0 | 19.8 | 29.4 | 38.9 | 48.1 | 57.1 | 74.7 | 107.9 | 138.6 | 239.6 | 367 | 471 |
| in one token's expert reads | 1.0 | 2.0 | 2.9 | 3.9 | 4.8 | 5.7 | 7.5 | 10.8 | 13.9 | 24.0 | 36.7 | 47.1 |

Tokens to touch 95 % of a layer's experts (MoESD's formula): Flash-Next 152, Flint (256 experts, top-6) 127,
DeepSeek-V3 (256, top-8) 95, Qwen3-30B/235B (128, top-8) 47, OLMoE (64, top-8) 23, Mixtral (8, top-2) 11. The finer
the experts, the longer a verify stays in the regime where every added token reads new weights. MoESD's headline,
speedups of up to 2.29× at moderate batch sizes (Qwen2-57B-A14B), comes from the other regime, where a pass already
touches nearly every expert because several requests share it. The rig serves one agent, so a pass is k tokens of one
stream.

### 2.2 What is measured

| Model (experts, top-k) | Block | Distinct experts per layer, measured | Independent routing | Measured / independent | Source and conditions |
|---|---|---|---|---|---|
| Flint (256, 6; sigmoid router) | 4 / 8 / 16 consecutive tokens | 20.1 / 35.6 / 59.7 | 23.2 / 44.2 / 80.8 | 0.87 / 0.81 / 0.74 | [Q51](../../data/2026-10-23_q51/README.md), generated text at T 1.0 |
| Qwen3-Coder-30B-A3B (128, 8) | 2 / 4 / 8 / 11 verified tokens | 12.1 / 18.2 / 26.1 / 30.2 | 15.5 / 29.1 / 51.6 / 65.1 | 0.78 / 0.63 / 0.51 / 0.46 | [arXiv 2609.22156](https://arxiv.org/abs/2609.22156) (2026): EAGLE-3 blocks, SpecBench, one A100 |
| Cohere MoE (128, 8) | 4 verified tokens | 20.4 | 29.1 (25.4 for four independent requests) | 0.70 (0.80) | [Cohere](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding): adjacent tokens share 0.38 of their picks against 0.12 expected |
| Qwen3-235B-A22B (128, 8) | 4 verified tokens | 23.7 | 29.1 | 0.81 | [EcoSpec](https://arxiv.org/html/2607.12696v1), EAGLE-3 |
| GPT-OSS-120B (128, 4) | 4 verified tokens | 11.6 | 15.3 | 0.76 | EcoSpec |
| DeepSeek-V3.1 (256, 8; shared expert) | 4 verified tokens | 31.4 | 30.5 | 1.03 | EcoSpec, MTP; it describes the routing as more balanced |
| OLMoE-1B-7B (64, 8) | tree of 31 / 127 nodes | 39 / 54 | 63.0 / 64.0 | 0.62 / 0.84 | [MoE-Spec](https://arxiv.org/html/2602.16052v1), EAGLE-3 trees |
| Qwen3-30B (128, 8) | adjacent pairs | – | – | reuse 2.0× chance | [arXiv 2608.18261](https://arxiv.org/abs/2608.18261) |

- **Locality is real and model-specific.** It is strongest on Qwen3-Coder-30B-A3B (0.63 of independent at 4 tokens),
  weaker on Qwen3-235B, GPT-OSS and Flint (0.76–0.87) and absent on DeepSeek-V3.1.
  A study of 20 MoE models ([Liang et al., ICLR 2026](https://arxiv.org/html/2505.16056v3)) finds shared experts hurt
  local routing consistency (segment-16 score: Qwen3 54, DeepSeek-V2-Lite 38, Qwen2 37 and Qwen1.5-MoE 31 with shared
  experts). Flash-Next has a shared expert and four times Qwen3-30B's expert count, so a Qwen3-Coder-class locality is
  the optimistic case. No measurement of a Qwen3-Next-class model was found; a preprint on Qwen3-Next offloading
  ([Research Square](https://www.researchsquare.com/article/rs-11268552)) studies substitution of missing experts, not
  overlap.
- **The right baseline skews toward popular experts.** Cohere's independent baseline for four requests (25.4) sits 13 %
  below uniform (29.1) because popular experts repeat. Comparing against uniform overstates locality; the
  shuffled-position baseline of `bench/expert_overlap` (the layer's own popularity without locality) is the same idea.
- **Verified blocks include rejected tokens, which probably correlate less than accepted text (est.; no paper separates
  them).** Flint's Q51 measured generated text (the best case); the Qwen3-Coder and EcoSpec rows measure real verify
  blocks.
- **Fine-grained layers run faster than their bytes predict at low batch.** Cohere's NeurIPS 2026 paper
  ([Why Speculative Decoding Works Better Than Predicted on Sparse MoE Models](https://neurips.cc/virtual/2026/poster/150811),
  10 models, 8–512 experts) attributes it to small per-expert weights leaving bandwidth idle, higher L2 hit rates and
  routing overlap, and fits unique-expert counts from overlap statistics within 4 % (a maximum-entropy model). On the rig the
  Q2_0 kernels reach 132 GB/s on gate/up and 82 GB/s on down products (R16), so extra experts are not entirely hidden.
- **Training-time fixes exist and need retraining:** a routing-consistency loss
  ([StickyMoE](https://arxiv.org/html/2607.08780): top-1 switch rate −59 % on a 22M model), router fine-tuning
  ([ReMoE](https://arxiv.org/html/2605.27081): +26–27 % overlap on DeepSeek-V2-Lite and Qwen1.5-MoE), expert-coactivation training
  ([Apple, arXiv 2609.22471](https://www.alphaxiv.org/abs/2609.22471): +21 % throughput on a 1B model). A pre-registered
  study found a perplexity cost beyond its 1 % gate at 137M parameters
  ([arXiv 2608.18261](https://arxiv.org/abs/2608.18261)). Out of scope for a released file.

### 2.3 Flash-Next: what is known

- **Not measured.** Q625 plans the trace (`bench/expert_overlap`'s route dump over held-out agent sessions, windows of
  1–8 and 16 tokens against 19.8 / 38.8 / 74.7 at 2 / 4 / 8).
- **An indirect estimate from Q583 (est.).** The stock expert kernel's time rises with distinct experts from k = 3 on
  (+1.35–1.5 ms per expert for random ids at k = 3–5; ~1.08 averaged from k = 1 to 4). Taking 45.1 ms for a real
  4-token verify, against 58.3 for random ids (u ≈ 38.9) and 27.2 at k = 1, gives u ≈ 27–30, 68–77 % of independent
  routing. Basis and caveats: the kernel's time is assumed linear in distinct experts, llama-bench's random ids are taken as
  independent, and the verify's tokens are one accepted token plus three drafts (some rejected). Treat as a bracket, not a
  measurement.
- **What the tables below assume:** a "uniform" column (independent routing) and a "local" column with Flint's measured
  ratios (0.93 / 0.87 / 0.81 / 0.74 at 2 / 4 / 8 / 16 tokens). The estimate above lies between them.

### 2.4 Trees: width and depth

- **A tree node is a column; its cost is its own routing.** Nodes share ancestors' attention and recurrent state, never
  expert reads: each node's token is new, so it routes anew. A tree of N nodes costs like a chain of N columns plus the
  tree mask. What differs is the correlation between nodes. Parent and child are adjacent tokens (the measured case).
  Siblings are different tokens in the same context; no published measurement exists. EcoSpec notes that a
  high-probability candidate can activate experts disjoint from the others.
- **Published tree sizes.** On OLMoE a 31-node EAGLE-3 tree touches 39 of 64 experts, a 127-node tree 54 (MoE-Spec).
  On Qwen3-30B-A3B, EVICT verifies 74.7 % fewer tokens than EAGLE-3's 32-node tree, activates 32.5 % fewer experts and
  cuts verify latency 26.6 % ([arXiv 2605.00342](https://arxiv.org/html/2605.00342v1)): the first nodes cost the most
  experts, and a 128-expert model saturates at tens of nodes (est.).
- **At 512 experts a tree does not saturate.** Even 64 nodes touch est. 367 of 512 experts per layer under independent
  routing, ~220 ms of expert reads alone on this rig (0.606 ms per distinct expert). The 256–512-node trees of GPU-rich
  papers (DDTree below) are out of reach.

## 3. What a node costs, and when it pays

### 3.1 The pass after the kernels (est.)

T(k) = T1 + 0.606 ms × (u(k) − 10) + (k − 1) × per-row cost. T1 = 90 ms; per-row cost 3.0 ms (est.: the logits row after
Q591's Q5_K k-column kernel ~0.4, recurrent state rows ~1.1–1.8 as measured in Q583's verify, small operations ~0.5–1);
"local" uses Flint's measured ratios (the k = 32 ratio, 0.67, extrapolates the trend).

| k | u(k): uniform / local | T(k) ms: uniform / local | × T1: uniform / local | Q583 today (× T1) |
|---|---|---|---|---|
| 1 | 10 / 10 | 90.0 | 1.00 | 1.00 |
| 2 | 19.8 / 18.4 | 98.9 / 98.1 | 1.10 / 1.09 | 1.37 |
| 3 | 29.4 / 26.5 | 107.8 / 106.0 | 1.20 / 1.18 | 1.80 |
| 4 | 38.9 / 33.8 | 116.5 / 113.4 | 1.29 / 1.26 | 2.25 |
| 5 | 48.1 / 40.9 | 125.1 / 120.7 | 1.39 / 1.34 | 2.65 |
| 6 | 57.1 / 47.4 | 133.6 / 127.6 | 1.48 / 1.42 | – |
| 8 | 74.7 / 60.1 | 150.2 / 141.3 | 1.67 / 1.57 | – |
| 12 | 107.9 / 83.1 | 182.3 / 167.3 | 2.03 / 1.86 | – |
| 16 | 138.6 / 102.6 | 212.9 / 191.1 | 2.37 / 2.12 | – |
| 32 | 239.6 / 160.5 | 322 / 274 | 3.58 / 3.05 | – |

- **Sensitivity.** With today's measured per-row cost (5.6 ms: logits row 2.3, state 1.8, other 1.5) the 5-token pass is
  1.46–1.51× and the 16-token pass 2.56–2.80×. If the dense products grow +10 % per extra token instead of staying flat
  (the assumption in [cheap MTP draft](cheap_mtp_draft.md)), add ~6 ms per row: r then reaches 0.15–0.17. C70's timing
  decides which row applies.
- **Q591's target** (a 5-token pass at most ~1.3× a one-token pass) is just below this estimate: 1.34–1.39× with
  3 ms per row.
- **The marginal cost r** between k = 4 and 8 is 0.078–0.094 (3 ms per row) to 0.106–0.122 (5.6 ms), against 0.37–0.43
  today (50–59 ms on 136).

### 3.2 The rule

For a draft whose nodes have path-acceptance probabilities p_i (the probability that a node's whole path is accepted),
E[τ] = 1 + Σ p_i. With cost T(k) = T1·(1 + (k − 1)·r), adding one node raises the speed-up only if it beats the current
average: **p > r · S**. On the rig after the kernels that is r ≈ 0.08–0.12 times S ≈ 1.8–2.0: p* ≈ 0.15–0.25. Today it is
0.37–0.43 × ~1.15: p* ≈ 0.4–0.5.

**A DFlash2 chain** (est.: cumulative acceptance 0.65 / 0.42 / 0.30 at positions 1–3, the figures of
[#30927's test](https://github.com/ggml-org/llama.cpp/pull/30927) (τ = 2.37); beyond, ×0.75 per position; draft pass 20 ms):

| Drafted tokens d (k = d + 1) | Cumulative acceptance of the last | τ | Step ms, local–uniform | S over a one-token baseline |
|---|---|---|---|---|
| 1 | 0.65 | 1.65 | 118–119 | 1.25 |
| 2 | 0.42 | 2.07 | 126–128 | 1.46–1.48 |
| 3 | 0.30 | 2.37 | 133–137 | 1.56–1.60 |
| 4 | 0.23 | 2.60 | 141–145 | 1.61–1.66 |
| 5 | 0.17 | 2.76 | 148–154 | 1.62–1.69 |
| 6 | 0.13 | 2.89 | 154–162 | 1.61–1.69 |
| 7 | 0.095 | 2.99 | 161–170 | 1.58–1.67 |

The curve flattens at four to six drafted tokens, where p falls to the threshold. At Q561's agent-request acceptance
(3.11 tokens per step at three drafted) the same step gives 2.05–2.10×. Today's measurement (1.15× at equal tokens,
break-even at ~2.65 tokens per step) is the same arithmetic at r = 0.4.

**A sibling branch** (est.): if the first draft is wrong 35 % of the time and its runner-up holds 0.12 of the mass, one
extra node adds 0.12 tokens, below p* ≈ 0.15–0.25: S 1.54–1.59 against 1.56–1.60 for the chain alone at d = 3, a tie at
best.
Adding the runner-up's two continuations (0.07, 0.04) makes it 1.44–1.52. A 0.12 runner-up is an assumption; the
drafter's top-2 mass on Flash-Next agent text is unmeasured.

**A copy chain** (est.): a 6-token match continues correctly 95–99 % of the time inside copied text (Q64, Flint's own
output). At 0.95 per token a pass of 16 columns (15 drafted) accepts 11.2 tokens at T(16) = 2.1–2.4 T1, 4.7–5.3× inside
the span; 32 columns accept 16.1 at 3.05–3.58 T1, 4.5–5.3×. At 0.98 per token, 32 columns accept 23.8, 6.6–7.8×. The
optimum is flat from 16 to 32 columns because p = 0.95^j falls to p* only at j ≈ 27–37.

### 3.3 What published trees buy per node

| Setting | Chain → tree | τ | Extra nodes | Δτ per extra node | Source |
|---|---|---|---|---|---|
| EAGLE-3 on Qwen3-8B, HumanEval / MBPP (T 0) | the paper's two EAGLE-3 settings, 16 and 60 tokens | 3.17 → 3.65 / 2.82 → 3.25 | 44 | 0.011 / 0.010 | [DFlash paper](https://arxiv.org/html/2602.06036v1) |
| MTP depth 3, Qwen3.5-35B-A3B (SpecBench) | 3 → 6 / 14 / 39 nodes | 3.24 → 3.30 / 3.38 / 3.58 | 3 / 11 / 36 | 0.020 / 0.013 / 0.009 | [TreeWY](https://arxiv.org/html/2608.20961) |
| Qwen3-32B with a 0.6B draft model (llama.cpp prototype) | 3 → 12 nodes | 2.46 → 2.72 | 9 | 0.029; no faster (22.4 ms for 12 nodes against 13.1 for one token) | [unsloth PR #224](https://github.com/unslothai/llama.cpp/pull/224) |
| DFlash + DDTree, Qwen3-Coder-30B-A3B, SWE-bench Lite | block-16 chain → tree, best budget of 16–1,024 | 3.61 → 5.71 | up to hundreds | est. ≤ 0.02 | [DDTree](https://arxiv.org/html/2604.12989v1) |
| EVICT truncation, Qwen3-30B-A3B (T 0) | 32-node tree → ~8 nodes | 3.65 → 3.09 | 24 removed | 0.023 | EVICT |

Per-node values of 0.01–0.03 clear the threshold only when r·S is below ~0.02, i.e. a pass of 64–512 nodes costing a
few times a token: DDTree measures on eight H200s with PyTorch attention and reports speed-ups peaking at 256–512 nodes.
At r ≈ 0.1 the optimal tree has a handful of nodes. The llama.cpp prototype shows the small-tree case: even at r ≈ 0.065
on a B200 (22.4 ms for 12 nodes against 13.1 for one token) a 12-node tree did not beat a tuned 3-node chain.

## 4. Methods that cut MoE verification cost

| Method | What it changes | Exact? | Reported | Fit for the rig |
|---|---|---|---|---|
| Multi-token expert kernel (reads each distinct expert once; Q591's R7; GPU serving kernels group tokens by expert) | how a pass reads weights | yes (bit-identical to k one-token passes in R7) | est. 70.7 → 26–29 ms at k = 5 | in hand; timing pending |
| [Cascade](https://arxiv.org/html/2506.20675v1) (MICRO 2025) | speculation length per iteration from measured utility = ETR / relative iteration time; tests K in short trials, falls back to K = 0 | yes | static K slows MoEs by up to 54 % (Mixtral, n-gram drafts); Cascade caps the loss at ~5 % and gains 7–14 % over the best static K | cheap host policy; its utility is section 3.2's quantity |
| [EVICT](https://arxiv.org/html/2605.00342v1) (2026) | verifies the prefix of the draft tree with the best E[accepted] / profiled cost C(k) | yes (claimed, sampling argument) | Qwen3-30B-A3B: −74.7 % verified tokens, −32.5 % experts, −26.6 % verify latency, τ at 85 % of EAGLE-3, +15–25 % over EAGLE-3 in SGLang | the profiled C(k) table is Q583's curve |
| [EcoSpec](https://arxiv.org/html/2607.12696v1) (2026) | tree nodes scored by P(path) / marginal new experts, from a small predictor of per-layer routing | yes (verification rule untouched) | experts 23.7 → 20.5 and 1.22 → 1.36× (Qwen3-235B, T 0); 11.6 → 10.6 and 1.14 → 1.31× (GPT-OSS-120B); DeepSeek-V3.1 31.4 → 31.2, 1.10 → 1.15×; predictor 4 ms, top-K overlap 80–93 % | needs a 48-layer routing predictor; est. +2–4 % here (below) |
| [CaDDTree](https://arxiv.org/abs/2606.01813) (2026) | tree budget per round that maximizes tokens per unit time; throughput is unimodal in the budget under a convex cost | yes | matches DDTree with an oracle budget (dense Qwen3-4B/8B) | the stopping rule for a cost-aware tree |
| [MoE-Spec](https://arxiv.org/html/2602.16052v1) (2026) | caps the experts of a whole verify tree at B (top-B by summed router probability); the others are truncated or substituted | **no** | +10–30 % over EAGLE-3; mean τ −1.4 %; Qwen3-30B-A3B ~95 % of quality at 2.0×, Mixtral ~78 % at the top speed | its budgets are 12.5–50 % of the experts: at 512 experts that is 64–256, which binds from k ≈ 7 to k ≈ 35 |
| [AcceptMoE](https://arxiv.org/abs/2608.02989) (2026) | one expert set per verify block, sized by tree position and offline commitment probabilities | **no** (−0.27 points mean accuracy over 12 model-task pairs) | +29 % over EAGLE-3 with all experts on GPU; 2.06× under offloading; host-to-device traffic −74–77 % | inexact |
| [Lynx](https://arxiv.org/pdf/2411.08982), [XShare](https://arxiv.org/html/2602.07265v1) | batch-aware expert selection; XShare extends it to speculation | no | Lynx up to 1.55× lower serving latency at small accuracy loss; XShare +14 % throughput with speculation | inexact |
| [SpecMoE](https://www.alphaxiv.org/abs/2604.10152.md), [DraftExpert](https://arxiv.org/abs/2607.24434) | the target's own experts (a resident subset) as the drafter | yes (target verifies) | up to 4.3× over on-demand offloading; 1.45× average on offloaded DeepSeek-V2-Lite and Moonlight | needs a second pass through a layer split; rejected earlier |

- **Exactness.** Choosing which nodes to verify leaves each verified node's distribution the target's own. Capping or
  sharing the experts a node uses makes it the capped model's, so accepted tokens follow the capped model. Using the
  capped model only to *draft* (SpecMoE, DraftExpert) stays exact. No method found verifies with capped experts and
  repairs the distribution. Q618 (skip low-weight routed experts on confident tokens) is the same kind of trade:
  inexact by design and gated on KLD.
- **How much expert steering can save here (est.).** EcoSpec's 13 % fewer experts bought +11 % speed on a stack where
  experts are a large share of a step (Cohere measures 30–55 % for its MoE). On the rig after the kernels the union at
  k = 5 is 15–18 % of the pass (19–23 ms of 121–125), so a 13 % cut is ~2–2.5 % of the pass: +2–4 %. The per-row costs
  are the other half of the marginal cost and expert-aware pruning does not touch them.
- **A cap that binds here would be steep.** For k = 8 (u 60–75 of 512) a budget of 40 experts cuts 33–47 % of the
  expert reads, est. 12–21 ms of 141–150, ~9–14 % of the pass, at a quality cost nobody has measured at top-10 of 512.
- **Production stacks run Qwen3-Next as a chain.** Qwen's card launches SGLang with top-k 1, 3 steps, 4 draft tokens and
  vLLM with 2 speculative tokens ([model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)).

## 5. Trees and forks on hybrid recurrent models

A Gated DeltaNet layer is a recurrence: a rejected token has already changed the state, so a verify needs per-token
states, or inputs from which to rebuild them. A tree adds one requirement: a node's state must start from its *parent's*
state, not from the previous token in the batch.

| Method | What it keeps per verified node | Fork and rollback | Reported | Trees? |
|---|---|---|---|---|
| vLLM and SGLang default; llama.cpp's `n_rs_seq` planes ([#28123](https://github.com/ggml-org/llama.cpp/pull/28123)); the fork's per-slot groups (Q550) | a full state per draft position | roll back by indexing the accepted state | 112.6 MiB per position on Flash-Next (9.4–15.9 MiB per card) | no: snapshots cannot be shared across branches ([TreeWY](https://arxiv.org/html/2608.20961)) |
| [Mamba in the Llama](https://arxiv.org/html/2408.15237) (2024) | one state saved before the draft; recompute the accepted prefix | restore and replay | 2.3–2.6× on an RTX 3090 (Mamba 2.8B), 1.58–1.88× hybrids | no, chains |
| [STree](https://arxiv.org/html/2505.14969v2) (NeurIPS 2025) | the tree packed into one sequence with an ancestor mask on the state-transition terms; one state | replay cached activations to the last accepted token | MambaInLlama-8B: τ 2.47–2.83 against 2.04–2.34, 1.74–1.98×; none on H100 | yes, Mamba-2's scalar decay only |
| uzu (Apple; [notes](uzu_lessons.md)) | per-node k, v, β, decay and conv window; verify from the committed state in WY form | commit = replay of the accepted path | up to 16 / 32 nodes; no per-token state copies | yes, GDN |
| [TreeWY](https://arxiv.org/html/2608.20961) (2026) | one pseudo-value matrix N × d_v per head, 128× smaller than a state at d = 128 | one triangular solve per tree; rebuild the accepted state at commit | 1.49× where memory binds, 0.985× geomean where it does not; trees "not yet a throughput win" (graph capture) | yes, GDN |
| [Bole](https://arxiv.org/abs/2608.01651) (2026, SGLang) | token-level factors, losslessly; reconstruct only the chosen state | closed-form tree verification kernel | 3.4–7.7× faster linear-attention verification, 82–99× less transient memory, up to 4.72× over autoregressive and 2.03× over the best tree baseline | yes, GDN |
| [SpecLA](https://arxiv.org/abs/2607.16673), [Weaver](https://arxiv.org/abs/2607.06763), [GDN Tree-Scan](https://arxiv.org/abs/2609.23900) (2026) | compact factors; rollback-free verification; path replay into the native state | topology-aware kernels | 1.70× on GDN-1.3B (H100); 4.08× and +16.3 % over DFlash (Weaver); coding-agent outcomes on a DGX Spark | yes |
| [ReplaySSM](https://tridao.me/blog/2026/replayssm/) (Dao AI Lab and NVIDIA, 2026) | a checkpoint plus a ring of the last steps' inputs | rollback = a pointer move | verify kernel 1.27–1.72× the decode kernel against 4.85× for snapshots (Qwen3.5-122B, 6 drafts); 3.0–3.3× more concurrent requests | chains only |
| llama.cpp tree prototype ([unsloth #224](https://github.com/unslothai/llama.cpp/pull/224)) | sequence ids and KV masks | none for recurrent state | hybrids (qwen35, qwen3next, nemotron-h) unsupported | trees on attention only |

- **Time is small, memory is the limit on this rig.** Per-position snapshots add ~1.8 ms per extra row in Q583's verify
  (+5.5 ms for three rows); a fork copies ~10–16 MiB per card (est. 0.1–0.2 ms). The 112.6 MiB per position set what
  fits: k planes cost est. 0.55 / 0.88 / 1.8 / 3.5 GiB over the cards at k = 5 / 8 / 16 / 32 (47–80 / 75–127 / 150–254 /
  300–509 MiB per card, depending on how many recurrent layers it holds), against 735–869 MiB free on the cards Q550
  checked (64k context, `-ub 512`). A forest of w chains from one committed state needs w groups; pipelined depth 3
  already takes 901 MiB.
- **Recording inputs shrinks a node by ~100× (est.).** TreeWY's ratio is 128× per head; at ~1 MiB per node the records
  of 32 nodes are ~32 MiB over the cards, and the per-row state cost falls with the writes (est. ≤ 0.3 ms). Commit
  replays the accepted path with the decode kernel's arithmetic, so states stay bit-identical (uzu's note); the price is
  one recurrence update per accepted token, about what a decode step already does.
- **A root-level fork of chains needs no tree kernel.** Chains that start from the same committed state are independent
  sequences: w × d columns, w state groups, the attention mask is block-diagonal. A tree with shared prefixes needs the
  tree-aware recurrence (STree, TreeWY, Bole) or unrolling (STree counts 8–32 states and 32–192 tokens for depth-4–6
  trees that pack into 15–63 tokens).
- **Exactness is a live hazard.** TreeWY's token streams are not bit-identical to the baseline (acceptance compared
  instead); ReplaySSM differs from sequential decoding at 1e-4–1e-3 with tensor cores; a vLLM report
  ([#63593](https://github.com/vllm-project/vllm/issues/63593), unconfirmed) shows MTP with forced rejections diverging
  on a Gated DeltaNet hybrid at K ≥ 5. The fork's identity checks (tokens, logits, state, probe) cover its own paths, and
  on Pascal a replay kernel can use the FP32 recurrence exactly.
- **None of these kernels targets compute 6.1 or llama.cpp.** The measurements are on Ampere-to-Blackwell GPUs (RTX
  3090, H100, H200, B200, B300, DGX Spark) and Apple silicon.

## 6. Acceptance lengths on coding and agent work

| Drafter and target | Workload | τ (bonus included) | Settings | Source |
|---|---|---|---|---|
| DFlash, block 16, Qwen3-8B | HumanEval / MBPP / LiveCodeBench | 6.50 / 5.95 / 7.27 (T 1: 5.52 / 5.21 / 6.69) | Transformers backend | [DFlash](https://arxiv.org/html/2602.06036v1) |
| EAGLE-3 (its 60-token setting), Qwen3-8B | the same | 3.65 / 3.25 / 3.03 | the same | DFlash |
| DFlash, Qwen3-Coder-30B-A3B | HumanEval / MBPP / LiveCodeBench | 8.09 / 7.23 / 6.42; 3.5× / 3.2× / 2.6× at concurrency 1 | SGLang, B200 | DFlash |
| DFlash and DDTree, Qwen3-Coder-30B-A3B | SWE-bench Lite | 3.61 → 5.71 (HumanEval 8.02 → 10.72) | budgets 16–1,024, 8 × H200 | [DDTree](https://arxiv.org/html/2604.12989v1) |
| DFlash2 and MTP, Flash-Next | GSM8K / HumanEval / MBPP / MT-Bench | DFlash2 3.26 / 2.94 / 3.06 / 2.83; MTP 3.08 / 2.52 / 2.65 / 2.50 | block 4, T 1.0, SGLang, GB300 | [Inco AI card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) |
| DFlash2, Flash-Next, on the rig | agent requests (Q561) | 2.56 at two drafted, 3.11 at three; MTP 3.07–3.22 | `-b 512`, T 1.0 | [Q561](../../data/2026-11-03_flash_next_dflash2/README.md) |
| DSpark (block 7) and MTP (7 steps), Flash-Next | 120 held-out coding-agent sessions (0–128K context) | DSpark 3.70 (≤ 32K: 4.10; > 32K: 3.46); MTP 4.24; chat, math and code prompts: 4.55–4.75 against 4.67 | greedy, SGLang, H100 (TP 2) | [DSpark card](https://huggingface.co/shadowpa0327/Qwen3.8-Flash-Next-DSpark) |
| MTP depth 3, Qwen3.5-35B-A3B | SpecBench | 3.24 chain (capped at 4) | vLLM fork, B200 | TreeWY |
| MTP, DeepSeek-V3 / V3.1 | various topics / seven benchmarks | second-token acceptance 85–90 % / τ 2.85 at 4 verified tokens | one MTP layer; EcoSpec: greedy, 8 × H200 | [DeepSeek-V3 report](https://arxiv.org/pdf/2412.19437), EcoSpec |
| EAGLE-3 on MoE targets | SpecBench-type | Qwen3-30B-A3B 3.65 (32-node tree), Qwen3-235B 3.57, Ling-flash-2.0 3.94; MoE-Spec's runs 2.9–3.2 | T 0, SGLang / PyTorch | EVICT, MoE-Spec |
| Retrieval and suffix drafts (Devstral, Gemma3-27B, Qwen3.6-27B) | SWE-bench Verified, TeamBench | accept length 5.5–6.0 at batch 1, up to 10.4 at batch 16 | draft caps 24–64; dense targets | [AgSpec](https://arxiv.org/html/2610.01108) |
| SuffixDecoding | SWE-bench, text-to-SQL | up to 5.3× over autoregressive, 2.8× over EAGLE-2/3; decoding time reduced 2.3–6.3× | EAGLE's draft models could not run past 2K tokens | [paper](https://arxiv.org/abs/2411.04975), [Snowflake](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/) |
| `ngram-copy`, Flint's own output | copied spans (≥ 8 tokens: 28 % of tokens; edits 53 %, rewrites 99 %) | a 6-token match continues 95–99 % inside a copy, 14–22 % in fresh text | T 1.0, 65,520 tokens | Q64, [bench/ngram_replay](../../../bench/ngram_replay/) |

- **Agent text is the hard case for learned drafters.** DFlash (block 16) on Qwen3-8B falls from τ 6–7 on function-level
  code to 3.6 on SWE-bench Lite (DDTree's table). DSpark was trained on coding-agent sessions and reaches 3.70 on held-out
  ones; MTP with seven steps reaches 4.24 there. Inco AI's card evaluates DFlash2 at block 4 (its serving stack's limit;
  llama.cpp allows up to 7, [Flash-Next drafters](flash_next_drafters.md)), so τ is at most 4 in that evaluation.
- **What 5–20 needs.** With a per-token match probability a sustained over a chain of L drafts, τ = (1 − a^(L+1)) / (1 − a):
  τ = 5 needs a ≈ 0.85 over 8, τ = 10 a ≈ 0.93 over 16, τ = 15 a ≈ 0.94 over 32, τ = 20 a ≈ 0.95 over 64 (est.). Learned
  drafters start at 0.65–0.78 for the first position and decay; only copied spans hold 0.95. Branching raises per-position
  acceptance a little (section 3.3) and cannot lift a chain from 0.7 to 0.95.
- **Quality of the evidence.** The Inco AI and DSpark numbers are model-card measurements at different temperatures and
  stacks; DSpark's are greedy, from one checkpoint (step 1,600). The DDTree, TreeWY, EVICT and AgSpec numbers are from
  preprints. AgentSpec ([arXiv 2608.24004](https://arxiv.org/pdf/2608.24004)) studies speculation for batches of agents;
  its PDF was not readable here and it is not used.

## 7. What the idle cards can do for branches

- **A branch in the pipeline pays no union.** A hedge branch (the runner-up as its own sequence, one-token steps through
  the fork's single-token kernels) reads its own 10 experts per layer on a card that would idle. While at most S_eff − 1
  tokens are in flight it costs no wall time ([pipelined speculation](pipelined_speculation.md)); a batched column costs
  r ≈ 0.1 T1 on the critical path. Each branch needs a state group (112.6 MiB, or ~1 MiB with recorded inputs). This is
  idle-cards idea 7 (top-κ chains as separate sequences), which no paper evaluates on a layer split.
- **It does not work yet.** Q551 measured pipelined speculation at depth 3 decoding 1.6× slower (a wrong guess +372 ms,
  ~2.9 plain tokens; the event model had assumed dropped guesses use idle cards). Where the time goes is Q593. Nothing in
  this section is worth building before it is understood.
- **Drafting on idle cards cannot hide behind the verify (est.).** A DFlash2 block starts at the last accepted token,
  the bonus token the last card samples at the end of the pass, and reads target features at layers 5 / 15 / 25 / 35 / 45.
  The block's context update can overlap the pass's last stages; its denoising pass over the mask tokens cannot. An MTP
  draft needs the final hidden state. Pre-drafting for the all-accepted case helps only with probability ~0.3 at four
  verified tokens (cumulative acceptance of the last position) times the draft's 13–20 % of a step: est. ≤ 4–6 %.
  Speculative speculation (SSD, SpecBranch; [idle cards](idle_cards.md) section 2.1) hedges the outcome with extra
  drafts and has the same dependency.
- **A second rig as a draft farm (Q616) parallelizes drafters, it does not hide them.** Several drafters on the same
  hidden states cost the wall time of the slowest plus ~1 ms of LAN (est. from Q616's 50 KB per step). Their agreement is a
  calibration signal for the chain length (section 8, idea 4), not a way to make the verify cheaper.
- **Branches from several drafters form a small tree, and section 3.2's rule applies to it.** A second drafter's
  candidate is worth a node only if its path probability exceeds 0.15–0.25, which needs two drafters that disagree where
  both are unsure; how often that happens on Flash-Next agent text is unmeasured.

## 8. Ideas for the rig

Ranked by expected gain on agentic speed for this setup. "Est. effect" is against the stated baseline; "disproved if"
names the measurement that would end the idea. Ideas marked **new** have no registry row yet.

| # | Idea | Est. effect | Cost | Disproved if | Registry |
|---|---|---|---|---|---|
| 1 | Time Q591's two halves (R7 experts, C70 dense) to k = 8 on a P106, trace Flash-Next's routing (Q625), and refit section 3.1's table with measured T(k), u(k) and per-row cost; extend the trace to sibling pairs and copy spans | DFlash2 and MTP from ~1.15× to est. 1.6–2.1× over the post-kernel baseline at 3–5 drafted tokens (section 3.2); fixes every other estimate here | kernels written; ~1 day of rig time, ~30 min on the quant machine for the trace | T(5) > 1.6 × T(1) with both halves in, or per-row cost > 9 ms (break-even then p* ≈ 0.3 and drafts stay ≤ 1.2×) | Q591, Q625 (the sibling and copy-span extension is **new**) |
| 2 | Long chains over copied text: copy-gated 16–32-token chains from a match or the answer store, with the multi-token expert kernel to 16 tokens | agent sessions +20–45 % on top of the drafter's gain (copy share 25–40 %, 4.5–7× inside a span, est.) | kernel beyond 8 tokens; chain controller; needs idea 3 | copy share of Flash-Next's agent output < 15 % (Q64-style replay on its own transcripts), or continuation < 85 % inside spans | Q617, Q624, Q562, Q555 |
| 3 | Record the inputs of verified tokens and replay the accepted path instead of one 113 MiB state plane per token (uzu, TreeWY and ReplaySSM designs on the fork's recurrence kernel; FP32 and bit-identical to decode) | enables k = 16–32 and forks (est. 3.5 GiB of planes at 32); −0.8–1.5 ms per row, −8–17 % of a 32-token pass, +3–5 % at k = 5 | replay kernel, state-group rework, exactness check; 1–2 weeks | per-row state cost after Q591 under 0.5 ms and k ≤ 8 enough, so VRAM never binds | **new** (Q550's groups are the baseline) |
| 4 | Choose the number of drafted tokens per step from the drafter's confidences and a measured T(k) table: draft only positions whose path probability exceeds r·S; verify shapes limited to ~4 (k = 1, 3, 5, 9, 17) for the graph cache; optionally the agreement of two drafters as confidence | est. +5–12 % over the best fixed length (Cascade +7–14 %, EVICT +15–25 % over fixed trees, Strata's policy +2–5 %) | host-side, ~150 lines; per-shape graphs ([speculation plumbing](speculation_plumbing.md)) | an offline replay of DFlash2's per-position probabilities on agent transcripts gains < 3 % | Q605 (fixed count), Q409 (Ornith); Flash-Next version **new** |
| 5 | Offline tree-value replay: DFlash2 top-k marginals per position on Flash-Next agent transcripts, best-first trees (DDTree) at budgets 3–16, Δτ per extra node against p* = r·S | decides Q614 before any tree mask or tree kernel is written; no rig | ~1 day on the quant machine | its expectation (no tree pays) fails if Δτ per extra node is ≥ 0.15 at a budget ≤ 8 | **new** (gates Q614) |
| 6 | Hedge branches as pipelined one-token steps, once Q593 explains the 372 ms per wrong guess | est. +5–10 % where the drafter's top-1 is below ~0.6; zero union cost | Q593 first; per-branch state groups (idea 3 shrinks them) | the wrong-guess cost stays above ~0.5 plain tokens, or runner-up mass < 0.1 | Q593, Q595, Q550; hedge variant **new** (idle-cards idea 7) |
| 7 | A small tree in the batch: width 2 at the first uncertain position, ≤ 6–8 nodes, a forest of chains with per-branch state groups | est. 0–5 %; negative if the runner-up holds < 0.15 | tree mask, w state groups or a tree-aware recurrence | idea 5 shows Δτ per node < 0.15 | Q614 |
| 8 | Idle-card drafting that overlaps the draft's context update with the verify's last stages | est. ≤ 4–6 % (dependency on the bonus token) | placement of the drafter on the last card; fork work | the drafter's context update is < 20 % of its pass | Q615 |
| 9 | Expert-aware node selection (EcoSpec): a per-layer routing predictor scores nodes by new experts | est. +2–4 % (experts ~15–18 % of a 5-token pass) | a 48-layer predictor, training data, ~4 ms per step | idea 1's per-row cost is ≥ 5.6 ms (experts are then under half of the marginal cost) or siblings share ≥ 0.9 of independent | **new**, parked: reopens if idea 1 shows experts ≥ 60 % of the marginal cost |
| 10 | Capped expert budgets in verification (MoE-Spec, AcceptMoE): a budget below u(k) at k ≥ 8 | est. 9–14 % of a k = 8 pass, inexact (−0.3 points accuracy at published budgets; unmeasured at top-10 of 512) | KLD run on FN1's bases, kernel support | KLD of the capped verify exceeds Q618's gate; or exactness is required | **new**, parked; related Q618 |
| 11 | A second rig as a draft farm with agreement-gated length | est. ≤ 5 %: drafters run in parallel after the verify, not under it | two rigs, LAN round trip ~1 ms | idea 4's replay shows agreement adds < 2 % | Q616 |

Ideas 1–3 form one chain: the kernels fix the price of a column, the recorded-input state makes long chains and forks fit,
and copied text supplies the 5–20 accepted tokens that trees on fresh text cannot; idea 3 ranks above idea 4 although its
own gain is smaller, because idea 2 needs it. Ideas 5–7 are the token-tree work of Q614: do 5 first, and build 7 only if
a node buys at least ~0.15–0.25 tokens at an uncertain position.
