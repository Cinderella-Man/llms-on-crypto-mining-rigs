# A cheap MTP draft for Ornith-1.5: what one draft reads, and when a miss costs nothing (2026-10-28)

Scope: Ornith-1.5's built-in MTP layer drafts tokens that the model accepts 85–91 % of the time (Q341). This note asks
whether the draft can be made so cheap, at a small loss of acceptance, that a wrong draft costs nearly nothing. It counts
the bytes and the P106 time of one draft for each way of shrinking it, collects what the 2023–2026 literature says about
cheap drafters, and builds a cost model for both ways of serving drafts on the rig. Nothing was run for it: the numbers
come from the data folders, the registry, the GGUF headers and the papers. It builds on
[Q341's data](../../data/2026-10-28_ornith_mtp_accept/README.md), [the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md),
[idle cards](idle_cards.md), [pipelined speculation](pipelined_speculation.md), [uzu lessons](uzu_lessons.md),
[MTP head research](mtp_head_research.md) and [the drafter page](../../drafter.md) ("What a guess may cost"). Conventions are
those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it. MB and GB are decimal.
- **A draft** is one MTP pass (the `blk.48` layer, fed the trunk's final hidden state of token k−1 and the embedding of
  token k) plus the output head and sampling. **ℓ** is one MTP pass without the head (the catch-up that writes the
  MTP layer's own KV entry for a committed token); **g** a draft's GPU time; **c_d** g plus the host's share.
- **P_j** is the probability that the j-th draft of a chain is accepted together with every draft before it;
  **E_k = Σ_{j≤k} P_j** the expected accepted drafts per step. Q366's first look gives E_1, E_2, E_3 = 0.86, 1.55, 2.06,
  so P_1..P_3 = 0.86, 0.69, 0.51 (effective ratios r_2 = 0.80, r_3 = 0.74; the per-position means 0.76 and 0.69 are
  lower because easy stretches accept every draft). P_4..P_7 = 0.36, 0.24, 0.15, 0.085 are extrapolated (r 0.70, 0.66,
  0.62, 0.58, est.).
- **f** is a variant's acceptance relative to the full draft at each chain position, so P_j becomes P_j·f^j (a
  shortlist that misses the token at one position ends the chain there).
- **Mode (a):** llama.cpp's batched draft-and-verify (`--spec-type draft-mtp`): k drafts one after another, then one
  verify step over k+1 tokens. **Mode (b):** Miner Llama's pipelined executor (`--pipe-depth`): each guess is a
  one-token step that enters card 1 while later cards finish earlier tokens; the last card samples and keeps a guess only
  if it equals the sample.
- **Baseline:** Q212's N2P/N3P file ("valid1", 48.2 GB) on nine P106-090 cards, 14.5 tok/s on the agent check, 69.0 ms
  per generated token (Q189). L = 69 ms is one token's pass through the nine cards; c ≈ 8.5 ms the slowest stage when the
  pipeline is full, s_1 ≈ 7.7 ms card 1's stage, s_d ≈ 8 ms the stage of the card that holds the MTP block and the head
  (the last), S_eff = L/c ≈ 8 (est., from L and Flint's S_eff of 7.2).
- "Points" are absolute percentage points of first-draft acceptance.

## Verdict

- **One draft of the rig file reads 0.63 GB, and five sixths of it is the output head:** valid1's MTP layer is already
  2–3-bit (N2P/N3P experts, Q5_0 dense; 105 MB per pass), its head is Q5_0 (524 MB). Est. 6.3 ms per draft on a P106:
  MTP pass 1.8 ms at agent context, head 3.5, sampling over 248,320 logits ~0.3, host ~0.7. The layer's acceptance in
  this file has not been measured: Q341 measured UD-Q2_K_XL's Q8_0 MTP layer and AD's Q4_K one.
- **Below ~4.5 bits a P106 matrix-vector product is issue-bound** (~3.8–4.2 ps per weight for Q4_0, N2P and N3P,
  Q188), so fewer bits buy memory, not time. The head gets cheaper only with fewer rows (a 16–32k frequency shortlist:
  0.33–0.65 ms) or a low-rank ranker over the whole vocabulary (~0.6 ms, est.). The MTP pass has a floor of ~1.2 ms (est.)
  set by its dense projections (0.5 ms), its attention over its own KV (0.6 ms at 6k tokens, ~3.9 ms at 40k) and small
  ops (0.35 ms). Its routed experts are 0.39 ms of it; fewer experts per draft buy 0.2–0.4 ms. What they do buy is card
  memory: the block's 256 experts are 0.87 of its 0.95 GB.
- **Mode (a): the verify sets the price, not the draft.** After Q211, a 3-token verify of valid1 costs est. 1.77 plain
  tokens (122 ms), because three consecutive tokens route to ~22 distinct experts per layer instead of 8. The best chain is
  k = 2: est. 53.5 ms per token, 18.7 tok/s (1.29×) with today's draft. A shortlist or low-rank head adds 2–4 %, a draft
  that costs nothing 10 %. A wrong draft cannot be made cheap by shrinking it: its cost is the extra verified token
  (~26 ms, est.). Exchange rate: 1 ms per draft ≈ 1.5 points of first-draft acceptance (est.). On the 35B, whose verify is
  dense-dominated and whose head is ~19 % of a token's bytes, the shortlist head is worth est. +10–15 %.
- **Mode (b): a miss already costs nothing; the draft's size decides how many guesses fit.** "A guess under one card's
  turn hides in idle time" is the latency condition for the first draft after a miss (c_d + hand-off ≤ s_1 ≈ 7.7 ms:
  today's 6.3 ms meets it). The binding condition is the drafting card's load per round: it runs its own stage for every
  pass, dropped guesses included, plus every draft and every catch-up:
  **(1 + D + M)·s_d + (D + M)·g + (1 + M)·ℓ ≤ L + M·c**, M = E_D. On a balanced split today's draft fits D = 3 (est.
  28.3 ms per token, 35 tok/s, 2.4×); a guess of ≤ 2 ms fits D = 5 (est. 26.3–26.6 ms, 37.6–38 tok/s, +6–7 %); D = 7
  fits no MTP draft with its catch-up, because the passes alone, dropped guesses included, fill 93 % of the last card's
  round.
- **The largest mode-(b) lever is not the draft but the dropped guesses' work on the drafting card.** If guesses
  behind a miss skip their last-card layers, head and sampling, today's full draft fits D = 6–7 (est. 24.1–24.4 ms,
  ~41 tok/s, 2.8–2.9×), level with a free draft. Past that point every saved millisecond is worth nothing and every lost
  point of acceptance costs: keep the full head.
- **Mode (b) beats mode (a) by est. 1.9–2.2× on the 122B** (35–41 against 18.7–19.4 tok/s): it reads each guess's experts
  once on idle cards instead of ~2–3× the experts serially in a wide verify. It needs R8 (the hybrid's recurrent-state
  rollback; its replay is a few-token Gated DeltaNet update) and an MTP guesser in the executor, not Q211's multi-token
  expert kernels. These are upper estimates: the round model reproduces Flint's measured 70 %-right oracle (2.09× against
  2.06–2.13×), but R8's own cost, the extrapolated P_4..P_7 and the served overheads that made Flint's head cost ~35 %
  more in the server than on an idle card (Q193) are not in them. Per task, est. 1.2× (mode a) and 1.8–2.0× (mode b) if
  generation is ~75 % of a task.
- **Shrink rows, not bits.** In order: measure valid1's own MTP acceptance (with the served couplings) and the per-draft
  profile; a 32k shortlist and a low-rank ranker for the draft head; an MTP format ladder on a fixed trunk; a lighter MTP
  pass (attention window, top-4) for long contexts; an expert subset for memory. Fifteen new rows (C1–C15 = registry Q397–Q411; the two parked ideas Q412–Q413) below.

## 1. What one draft reads

Shapes from the GGUF headers ([UD-Q2_K_XL](../../data/2026-10-27_ornith/header-Ornith-1.5-122B-UD-Q2_K_XL.txt),
[valid1](../../data/2026-10-28_ornith_valid1/header-Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.txt)): hidden 3,072, 32 query
heads and 2 KV heads of 256 (the query projection carries the output gate), 256 routed experts of width 1,024 with 8
used, one shared expert of width 1,024, vocabulary 248,320. Bytes per weight: Q8_0 1.0625, Q6_K 0.820, Q5_0 0.6875,
Q4_0/Q4_K 0.5625, N3P 0.406, N2P 0.336.

**The MTP layer, per draft** (MB read; est. from the shapes):

| Part | Shape | Weights | Q8_0 | Q5_0 | Q4_0 | valid1 |
|---|---|---|---|---|---|---|
| attention q + gate | 3072 × 16384 | 50.3 M | 53.5 | 34.6 | 28.3 | 34.6 (Q5_0) |
| attention k, v | 2 × 3072 × 512 | 3.1 M | 3.3 | 2.2 | 1.8 | 3.3 (Q8_0) |
| attention output | 8192 × 3072 | 25.2 M | 26.7 | 17.3 | 14.2 | 17.3 |
| `eh_proj` ([embedding; hidden] → hidden) | 6144 × 3072 | 18.9 M | 20.0 | 13.0 | 10.6 | 13.0 |
| shared expert | 3 × 3072 × 1024 | 9.4 M | 10.0 | 6.5 | 5.3 | 6.5 |
| router (F32) | 3072 × 256 | 0.8 M | 3.1 | 3.1 | 3.1 | 3.1 |
| routed experts, top-8 | 8 × 9.4 M | 75.5 M | 80.2 | 51.9 | 42.5 | 27.1 (N2P gate/up, N3P down) |
| … top-4 / top-2 / shared only | | 37.7 / 18.9 / 0 M | 40.1 / 20.1 / 0 | 26.0 / 13.0 / 0 | 21.2 / 10.6 / 0 | 13.6 / 6.8 / 0 |
| **layer, top-8** | | 182.5 M | **197** (UD) | 128 | **106** (AD, Q4_K) | **105** |
| its KV read (f16, 2 KiB per position) | | | 12 MB at 6k positions, 81 MB at 39.4k | | | |
| block held on the card | | | 2.68 GB (UD) | | 1.35 GB (AD) | 0.95 GB, experts 0.87 |

**The head, per draft:**

| Head | MB | est. ms on a P106 | Acceptance cost (est.) |
|---|---|---|---|
| Q8_0, 248,320 rows (AD) | 811 | 4.8 (169 GB/s, v2.0 profile) | reference |
| Q6_K (UD) | 626 | ~4.5 (K-quant kernel) | ~0 |
| Q5_0 (valid1; Q171: no KLD cost) | 524 | 3.5 (150 GB/s) | ~0 |
| Q4_0 | 429 | 3.0 (3.87 ps per weight, Q188) | ~0 |
| N2P (2.69 bits) | 256 | 3.2 (4.16 ps per weight): no faster than Q4_0 | unknown |
| 64k shortlist, Q8_0 rows | 214 | 1.3 | 0.5–1 points |
| 32k shortlist, Q8_0 rows | 107 | 0.65 | 1.5–2.5 points |
| 16k shortlist, Q8_0 rows | 54 | 0.33 | 3–4.5 points |
| 8k shortlist, Q8_0 rows | 27 | 0.17 | 6–9 points |
| low-rank ranker (rank 256, Q8_0, 67 MB) + exact Q5_0 rescoring of its top 2,048 rows (4 MB) | 71 | ~0.6 + top-k ~0.2 | 0.5–1.5 points |

Acceptance costs of the shortlists are scaled from FastMTP's MTP head (32k: −4 % of accepted drafts over a 3-draft
chain, 16k: −10 %) and Flint's agent-text coverage here (16,384 ids 95.6 %, 29,288 ids 97.7 %); the low-rank ranker's
from SpecVocab (section 2.2). None is measured on Ornith.

**Time per draft on one P106 (est., 6k positions of context):**

| File or variant | MTP pass | Head | Sampling | Host per draft step | Draft c_d |
|---|---|---|---|---|---|
| UD-Q2_K_XL (MTP Q8_0, head Q6_K) | 2.2 | 4.5 | 0.3 | 0.7 | ~7.7 |
| AD-IQ3_XXS-IQ2_S (MTP Q4_K, head Q8_0) | ~1.9 | 4.8 | 0.3 | 0.7 | ~7.7 |
| **valid1** (MTP N2P/N3P + Q5_0, head Q5_0) | **1.8** | **3.5** | 0.3 | 0.7 | **~6.3** |
| valid1, 32k shortlist head | 1.8 | 0.65 | < 0.1 | 0.7 | ~3.25 |
| valid1, 16k shortlist head | 1.8 | 0.33 | < 0.1 | 0.7 | ~2.9 |
| valid1, low-rank ranker head | 1.8 | 0.6 | ~0.2 | 0.7 | ~3.3 |
| "lite": 32k head, attention window 2,048, top-4 experts, the chain kept on the card | 1.2 | 0.65 | < 0.1 | ~0.1 | ~2.0 |

- valid1's MTP pass: dense products 103.8 M weights at Q5_0's 4.6 ps per weight (0.48 ms) plus k/v in Q8_0 (0.02);
  experts 50.3 M gate/up weights at N2P's fused 5.53 ps and 25.2 M down weights at N3P's 4.27 ps (0.39 ms, Q188); its
  attention reads its KV at ~21 GB/s, as the trunk's decode attention does (Q210: 0.6 ms at 6k, 3.9 ms at 39.4k);
  norms, rope, gates, router and combine ~0.35 ms (Ornith's "everything else" of 16.3 ms over 49 layers in the v2.0
  profile).
- The host share is llama.cpp's draft loop on the 2-core Celeron: one decode call per draft, the drafted id read back
  before the next step's embedding (est. from the ~1 ms host gap per plain token, [idle cards](idle_cards.md)). A chain
  kept on the card needs the draft tokens' embedding rows there: the token embedding stays in host memory today.
- **At long context the MTP pass is attention:** at 40k positions ~5.1 ms, three quarters of it reading the layer's
  own KV (est.). A window of the last 2,048 positions caps it at ~0.2 ms, and caps the layer's prompt-side pass and KV
  memory with it.
- **The 35B** (Qwen3.6-35B-A3B's shapes: hidden 2,048, 256 experts of width 512): MTP pass ~0.07 GB in Q8_0 (est.), head
  0.54 GB; a draft is ~0.61 GB against a token's 2.82 GB (P-Q5) to 3.20 GB (Q8_0), and the head ~89 % of it. A 32k
  shortlist (71 MB) cuts the draft to ~0.14 GB.

## 2. What is known

### 2.1 Quantized drafts and quantized MTP layers

| Work | What was quantized | Acceptance | Speed | Bearing here |
|---|---|---|---|---|
| ML-SpecQD ([2503.13565](https://arxiv.org/abs/2503.13565)) | the target itself cast to MXFP4 as its own draft (CPU, greedy, 8 drafts) | 71.2 % (Llama-2-7B, QA), ~91 % (Qwen2.5-Coder-7B, code) | 1.86× / 2.03×; 2.22× / 2.72× with a second, smaller draft | a 4-bit copy of the same weights stays close; here the trunk is already 2–3-bit |
| QSpec ([2410.11305](https://arxiv.org/abs/2410.11305)) | the same weights, W4A4 to draft, W4A16 to verify | not reported | up to 1.64× | the same principle with activations |
| HierSpec ([2505.22179](https://arxiv.org/abs/2505.22179)) | targets in W8A8/W4A16/W4A8; EAGLE-2's head kept in FP16 | target quantization barely moves the acceptance length; it cites GPTQ-quantized drafts losing acceptance substantially | tree verification erodes 4-bit gains (verify/decode 1.8 at tree size 60 for W4A16) | wide verify steps cost more on low-bit weights: why mode (a) stays narrow here |
| Nemotron 3 Super ([2604.12374](https://arxiv.org/abs/2604.12374)) | NVFP4 release; the MTP layers kept in BF16 "to preserve" MTP | — | — | practice: the MTP layer is the last thing quantized; Unsloth's UD-Q2_K_XL also keeps it in Q8_0 inside a 2–3-bit file |
| Gemma 4's MTP drafter in llama.cpp ([community write-up, 2026-05](https://point.free/blog/gemma-4-mtp/)) | a Q4_K_M against a Q8_0 drafter | Q8_0 31B pair: ~89 / 83 / 74 / 64 % at 1–4 drafts | the Q4_K_M drafter 13 % faster overall on a laptop GPU | a cheaper drafter wins where the draft is a large share of a step |
| Ornith-1.5 (Q341) | the MTP layer in Q8_0 (UD) and Q4_K (AD), each in its own trunk | 0.845 / 0.865 against 0.799 / 0.831 (agent / code) | — | 3.4–4.6 points, the layer's format and the trunk's mixed; valid1's 2–3-bit layer unmeasured |

### 2.2 A smaller draft vocabulary

| Work | Mechanism | Acceptance | Speed |
|---|---|---|---|
| FR-Spec ([2502.14856](https://arxiv.org/abs/2502.14856)) | EAGLE-2's draft head over the most frequent tokens of Llama-3-8B's 128k vocabulary; verification over all of it | acceptance length 3.89 → 3.77 / 3.57 / 3.32 / 3.06 at 64k / 32k / 16k / 8k (ShareGPT counts; SlimPajama counts 3.80 / 3.63 / 3.40 / 3.13); Qwen-2-7B similar or smaller drops | the head is 49 % of drafting time (62 % with softmax); 1.12× over EAGLE-2 |
| VocabTrim ([2506.22694](https://arxiv.org/abs/2506.22694)) | the subset taken from the target's own outputs on calibration text | — | up to +16 % speed-up (as summarized in [2603.05210](https://arxiv.org/abs/2603.05210)) |
| SpecVocab ([2602.13836](https://arxiv.org/abs/2602.13836)) | a vocabulary per step: a trained low-rank ranker over the whole vocabulary, exact logits for its top 2,048 | Qwen3-8B acceptance length 5.01 against 4.69 (FR-Spec 32k) and 4.84 (VocabTrim 32k) | up to +8.1 % over EAGLE-3 |
| FastMTP ([2509.18362](https://arxiv.org/abs/2509.18362)) | MiMo-7B's MTP head (152k vocabulary) over 8–64k frequent tokens, 3 chained drafts | MT-Bench 2.690 → 2.677 / 2.622 / 2.519 / 2.426 at 64k / 32k / 16k / 8k (accepted drafts −0.8 / −4.0 / −10 / −16 %); C-Eval 16k −0.026 | 32k: 56.3 → 63.4 tok/s; +12–16 % on the fine-tuned head |
| llama.cpp PR #29143 (registry Q43) | Qwen3.6's MTP with a 32k draft vocabulary | 82 % with 2 drafts | 1.04× on the fork: verify-bound |
| Flint's head here ([drafter.md](../../drafter.md)) | the first 4k / 8k / 16k / 29k draft ids | cover 84.2 / 90.3 / 95.6 / 97.7 % of agent-session tokens | the shortlisted output layer was 0.47 of a 2.93 ms guess |

A draft vocabulary is an MTP head's cheapest lever because the draft shares the model's 248,320-row head, and only rows
cut its time on Pascal. Coding-agent text is concentrated (Flint: 97.7 % in 29k ids), so 32k costs little; 8k costs a
lot everywhere.

### 2.3 Fewer experts in a draft

| Work | Mechanism | Acceptance | Bearing here |
|---|---|---|---|
| DraftExpert ([2607.24434](https://arxiv.org/abs/2607.24434)) | self-drafting MoE models (DeepSeek-V2-Lite, Moonlight-16B-A3B) with the shared expert plus the top-r routed experts in every layer | simple shared + top-r drafts "often below 50 %"; its distilled drafter 84–87 %; 1.45× | cutting every layer hurts; cutting the one MTP layer is milder (unmeasured) |
| MoE-Spec ([2602.16052](https://arxiv.org/abs/2602.16052)) | an expert budget for the verify step | lossy | not a draft change |
| Flint, Q281 / Q288 (parked) | top-5 or dynamic expert skipping in the target | lossy there | in a draft the output stays exact |

### 2.4 Chained MTP drafts

| Work | Setting | Acceptance by draft | Bearing here |
|---|---|---|---|
| DeepSeek-V3 ([2412.19437](https://arxiv.org/abs/2412.19437)) | one MTP module, the second token | 85–90 %, 1.8× tokens per second | the reference for one draft |
| FastMTP ([2509.18362](https://arxiv.org/abs/2509.18362)) | MiMo-7B's one MTP layer reused for drafts 1–3 | ~70 / 10 / ~0 %; after self-distilled fine-tuning 80 / 56 / 36 % | an MTP layer trained for one step collapses when chained |
| Nemotron 3 Super ([2604.12374](https://arxiv.org/abs/2604.12374)) | SPEED-Bench, 7 drafts | acceptance length DeepSeek-R1 2.70, Qwen3-Next 3.33 (coding 4.32), Nemotron 3 Super 3.45 (two MTP layers with shared weights) | Qwen3-Next's one layer chains well: Ornith's family |
| Ornith-1.5 (Q366, first look) | drafts 1–3 | 0.86 / 1.55 / 2.06 accepted per step (r_2 0.80, r_3 0.74) | the chain holds; P_4..P_7 extrapolated here sit between Qwen3-Next's average and its coding figure |
| Qwen3-Next-80B-A3B ([model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)) | recommended settings | vLLM 2 speculative tokens, SGLang 3 steps | on fast GPUs the verify grows too |
| Gemma 4 MTP (community, above) | 1–4 drafts | 35 / 44 / 49 / 49 tok/s | the gain stops at 3 |

### 2.5 Cost models for cheap drafts

- **Leviathan et al.** ([2211.17192](https://arxiv.org/abs/2211.17192)): with a draft costing c target steps and
  acceptance α, a step yields (1 − α^{γ+1})/(1 − α) tokens and the wall-time factor is
  (1 − α^{γ+1}) / ((1 − α)(γc + 1)); some γ pays whenever α > c. It assumes the verify step costs one plain step, which
  is the assumption Pascal breaks (Q19, Q166: a 2-token expert product 3–4× one token today).
- **Sequoia** ([2402.12374](https://arxiv.org/abs/2402.12374)) sizes draft trees against a measured verify-cost curve;
  at Flint's c(3) = 2.2 and c(4) = 3.6 trees cannot pay ([idle cards](idle_cards.md)).
- **PEARL** ([2408.11850](https://arxiv.org/abs/2408.11850)) overlaps drafting with verification (pre-verify and
  post-verify) to hide the draft: what mode (b) does with idle cards.
- **Pipelined speculation:** PPSD ([2509.19368](https://arxiv.org/abs/2509.19368)), SPD
  ([2605.30852](https://arxiv.org/abs/2605.30852)), PipeInfer ([2407.11798](https://arxiv.org/abs/2407.11798)); vLLM
  and SGLang (September 2026) speculate under pipeline parallelism with the MTP drafter on the last stage
  ("tail-drafting"), the same placement as here ([idle cards](idle_cards.md)).
- **Flint's own cost simulator** ([drafter.md](../../drafter.md#a-cheaper-guess-2026-10-27)): a guess on card 1 hides up
  to ~3 ms (card 1's stage ~3.4 ms, the last card's ~5.3 ms), then costs ~3–4 tok/s per ms; in the server a 2.93 ms
  guess cost ~4 ms plus ~0.75 ms per token of following (Q193, Q194).
- **Coupling:** Q341's acceptance is Σ min(p, q), what rejection sampling with probabilistic drafts accepts. Greedy
  drafts, llama.cpp's default (Q43's note), are accepted with E[p(argmax q)]; mode (b)'s sample-and-match with a
  position-keyed draw sits between, at least (1 − TV)/(1 + TV) for Gumbel coupling (Daliri et al.,
  [2408.07978](https://arxiv.org/abs/2408.07978)). The rig's stock line accepted 41–56 % of up to three AD drafts per step
  (2.2–2.7 tokens), below the ~2.9 its offline chain implies (est.: Q366's chain scaled to AD's 0.80–0.83 first
  draft), on text the model wrote itself: the served couplings cost points.

## 3. Mode (a): draft, then verify

**Time per token:** T_a(k) = (k·c_d + c_c + V(k+1)) / (1 + E_k), with c_c ≈ 2 ms the MTP catch-up over the accepted
tokens and V(m) the verify of m tokens. Leviathan's form is the case V(m) = V(1), c_c = 0, P_j = α^j.

**The verify after Q211** (valid1, est.): V(1) = 69 ms splits into dense products 25 ms (5.3 G weights at Q5_0), routed
experts 18 (2.4 G gate/up weights at N2P/N3P's fused 5.1–5.5 ps, 1.2 G down at 4.3–4.7), attention 7 at agent context,
Gated DeltaNet 4 and the rest 15 (norms, routers, small ops, hand-offs). Kernels that read each weight once leave the
dense part nearly flat (+10 % per extra token assumed), but m consecutive tokens route to u(m) distinct experts per
layer: 256·(1 − (1 − 8/256)^m) = 15.75 / 23.3 / 30.5 for independent routing, and 30 tokens reach ~60 % of the experts
(the Ornith benchmark), close to independent; u = 15 / 22 / 28.5 is used.

| m | Dense | Experts (u) | Attention (+50 %/token) | DeltaNet (+60 %) | Rest (+15 %) | V(m), ms | Plain tokens |
|---|---|---|---|---|---|---|---|
| 1 | 25 | 18 (8) | 7 | 4 | 15 | 69 | 1.00 |
| 2 | 27.5 | 33.8 (15) | 10.5 | 6.4 | 17.3 | 95.5 | 1.38 |
| 3 | 30 | 49.5 (22) | 14 | 8.8 | 19.5 | 121.8 | 1.77 |
| 4 | 32.5 | 64.1 (28.5) | 17.5 | 11.2 | 21.8 | 147.1 | 2.13 |

**Speed by draft variant** (est.; plain 69.0 ms, 14.5 tok/s):

| Draft | c_d, ms | f | k | E_k | ms per token | tok/s | Against plain | Against today's draft |
|---|---|---|---|---|---|---|---|---|
| today's (valid1) | 6.3 | 1 | 1 / 2 / 3 | 0.86 / 1.55 / 2.06 | 55.8 / **53.5** / 54.9 | 17.9 / **18.7** / 18.2 | 1.24 / **1.29** / 1.26× | — |
| 32k shortlist head | 3.25 | 0.98 | 2 | 1.51 | 52.0 | 19.2 | 1.33× | +2.9 % |
| 16k shortlist head | 2.9 | 0.96 | 2 | 1.46 | 52.7 | 19.0 | 1.31× | +1.6 % |
| low-rank ranker head | 3.3 | 0.99 | 2 | 1.53 | 51.6 | 19.4 | 1.34× | +3.7 % |
| lite | 2.05 | 0.96 | 2 | 1.46 | 52.0 | 19.2 | 1.33× | +2.9 % |
| free (c_d = 0) | 0 | 1 | 2 | 1.55 | 48.5 | 20.6 | 1.42× | +10 % |

- **k = 2 is the optimum for every draft**: a third draft adds 0.51 accepted tokens for ~26 ms more verify.
- **Break-even:** at k = 2 a variant pays when 2·Δc_d / 136 ms > ΔE_2 / 2.55, i.e. 1 ms saved per draft buys
  0.037 accepted drafts per step, ≈ 1.7 % relative acceptance per chain position, ≈ 1.5 points of first-draft
  acceptance (dE_2/df ≈ P_1 + 2P_2 = 2.24). The 32k head saves 3.05 ms against est. 1.5–2.5 points: it pays a little;
  16k saves 3.4 ms against 3–4.5 points: about even.
- **Today, before Q211,** v2.0 with the MTP head gained nothing on the agent check (Q158): a 2-token expert product at
  3–4× a token makes V(3) ≈ 2.5–3 tokens (est.).
- **A confidence gate** on the chain (stop drafting when the MTP's top probability falls below a threshold, as EAGLE-2's
  dynamic depth and SpecDec++ ([2405.19715](https://arxiv.org/abs/2405.19715)) do) saves verify width exactly where the
  third draft is unlikely; Q341's bins show acceptance tracking confidence (0.94–0.99 above 0.99, ~0.5 below 0.5):
  est. +2–5 % over a fixed k = 2 (C13).
- **The 35B** (bytes only, est.): drafts 2 × 0.61 GB against a 3-token verify of ~4.2 GB (dense ~2.1 GB nearly flat,
  experts ~0.7 GB × 2.75) make the drafts ~22 % of a step; a 32k head (0.14 GB per draft) gives est. 1.84 against
  2.14 GB per token at −2 points, **+10–15 %** after fixed costs. On the 35B the shortlist is worth doing in mode (a).

## 4. Mode (b): guesses on idle cards

**The round model.** A round runs from one miss to the next. The corrected token passes all cards (L); up to D guesses
follow one stage apart; the first M = E_D of them are right on average and come out one cycle c apart:
**T_b = max(L + M·c, W_d) / (1 + M)**, where the drafting card's work per round is
**W_d = (1 + D + M)·s_d + (D + M)·g + (1 + M)·ℓ**: its stage for every pass (the D dropped guesses run to the end, as the
executor drops rather than withdraws), one draft per guess submitted and one catch-up per committed token. On Flint's
measured pipeline (L 24.7 ms, c 3.4 ms) with guesses right 70 % of the time at depth 3, the first term gives 2.09×; the
oracle measured 2.06–2.13× at matched clocks ([idle cards](idle_cards.md) 1.3).

**Two conditions make a miss free.** (1) Latency: the first draft after a miss reaches card 1 before card 1 has
finished the corrected token, c_d + hand-off ≤ s_1 ≈ 7.7 ms. valid1's 6.3 ms draft meets it; UD's ~7.7 ms is at the
edge; the 35B's full-head draft (est. ~4.5–5 ms) fits a card's turn on five cards (~6.4 ms at Qwen3.6's 31 tok/s,
Q43) but not on nine (~3.5 ms, est.), and its shortlist draft (~1.3 ms) fits both. (2) Load: W_d ≤ L + M·c. The guess cost each depth allows on valid1
(ℓ = 1.8 ms, full acceptance, est.):

| Guesses in flight D | M | Largest g, balanced split | Largest g, dropped guesses skip the drafting card |
|---|---|---|---|
| 3 | 2.06 | 6.4 ms | 11 ms |
| 4 | 2.42 | 3.7 ms | 8.8 ms |
| 5 | 2.65 | 2.0 ms | 7.3 ms |
| 6 | 2.80 | 0.9 ms | 6.3 ms |
| 7 (= S_eff − 1) | 2.88 | none | 5.6 ms |

**Speed by variant** (valid1, balanced split, the best D for each, est.):

| Draft | g, ms | ℓ | f | D | M | W_d | L + M·c | ms per token | tok/s | Against plain |
|---|---|---|---|---|---|---|---|---|---|---|
| today's | 5.6 | 1.8 | 1 | **3** | 2.06 | 82.3 | 86.5 | **28.3** | **35.4** | 2.44× |
| today's, deeper | 5.6 | 1.8 | 1 | 5 | 2.65 | 118.7 | 91.5 | 32.5 | 30.8 | 2.12× |
| 32k shortlist head | 2.55 | 1.8 | 0.98 | 5 | 2.53 | 93.8 | 90.5 | 26.6 | 37.6 | 2.59× |
| 16k shortlist head | 2.2 | 1.8 | 0.96 | 5 | 2.41 | 89.7 | 89.5 | 26.3 | 38.0 | 2.62× |
| low-rank ranker head | 2.6 | 1.8 | 0.99 | 5 | 2.59 | 94.9 | 91.0 | 26.4 | 37.9 | 2.61× |
| lite | 1.95 | 1.2 | 0.96 | 5 | 2.41 | 85.8 | 89.5 | 26.3 | 38.1 | 2.63× |
| free | 0 | 0 | 1 | 7 | 2.88 | 87.1 | 93.5 | 24.1 | 41.5 | 2.86× |
| **today's, dropped guesses skip the drafting card** | 5.6 | 1.8 | 1 | 6 | 2.80 | 86.5 | 92.8 | **24.4** | **41.0** | 2.83× |

- **Where the optimum lies:** shrink the guess until W_d fits at the depth wanted, then stop. While W_d binds, a
  millisecond per guess is worth (D + M)/(1 + M) ≈ 1.7–2.1 ms per token at D = 3–5 (6–8 % of a token), as much as
  ~6 points of first-draft acceptance; once W_d fits, it is worth nothing, while each point of first-draft acceptance
  (carried through the chain) still moves T by ~1.5 % (est.).
- **How many drafts in flight:** D = 3 with today's draft on a balanced split; D = 5 for a guess of ≤ 2 ms; D = 6–7
  (S_eff − 1) only when dropped guesses skip the drafting card, and then with the full head, whose acceptance beats any
  shortlist once time is free. Deeper than S_eff − 1 never pays: the cards are full.
- **Three ways to make room on the drafting card:** a cheaper guess (−3 to −4 ms per guess, a file and a small engine
  change, +6–7 %); dropped guesses that skip it (−8 ms per dropped guess, the larger gain, +16 % over today's best,
  reopens Q283); a lighter drafting card (each layer moved off it ~1.35 ms and ~1 GB, memory permitting: valid1 + MTP
  block + R8's state leave little room).
- **With a shorter chain** (r_4..r_7 = 0.55, 0.45, 0.40, 0.35, Qwen3-Next's average acceptance length of 3.33 at 7
  drafts): today's D = 3 is unchanged at 35.4 tok/s, the cheap heads give ~37, a free draft ~39 (est.).
- **Clocks:** Ornith's cards sit at base clock half the time (Q154); filling the pipeline lifts them, which Flint's
  every-guess-wrong runs showed as +20–24 % with no right guess. Not counted above.
- **Coupling:** the executor keeps a guess only if it equals the last card's sample, so P_j is the coupled match
  probability, below Σ min(p, q): est. 1–4 points at each position (C1 measures it). Draw the draft with the same
  position-keyed uniform as the trunk's sampler, as Flint's head does.
- **Memory:** valid1 leaves the MTP block (0.95 GB), its KV (2 KiB per position) and R8's state to fit beside 46.4 GB of
  trunk on the cards; an expert subset of the MTP block (C7) frees 0.65–0.76 GB of it.
- **Re-anchoring:** a guess made from the newest trunk state is a first draft (0.86); one made from the chain is at
  depth j. When the drafting card has slack, rebuilding the chain's frontier from the newest trunk state through the
  guesses in flight (MTP passes without the head) keeps every new guess at depth ≤ D (C14).

## 5. What to measure first

Predicted effects are this note's estimates (sections 3 and 4), valid1 at agent context.

| # | Variant | Predicted acceptance change | Mode (a) | Mode (b) | Cost of the measurement |
|---|---|---|---|---|---|
| 1 | **valid1's MTP layer as it is** (C1, C2): Σ min(p, q), greedy and coupled acceptance, chained to 3; the per-draft profile on one P106 | 0 to −4 points against UD's Q8_0 layer (AD's Q4_K layer was 3.4–4.6 lower in another trunk) | sets every row below | sets every row below | Q341's tool rebuilt on the low-bit branch, two V100s, ~1 h; a profiled `draft-mtp` probe on the rig, ~20 min |
| 2 | **32k shortlist head** and **low-rank ranker** (C4, C5) | −1.5 to −2.5 and −0.5 to −1.5 points | +3 to +4 % | +6 to +7 % (D 3 → 5) | Ornith's token counts from data run 1's sessions, then Q341/Q366's tool with the head swapped; hours on the V100s |
| 3 | **MTP format ladder on the UD trunk** (C3): Q8_0 as shipped, valid1's block, dense Q8_0 + N2P/N3P experts, all Q4_0 | decides whether valid1's formats cost the 0–4 points; dense Q8_0 + N2P/N3P experts est. within 1 point of Q8_0 for +0.07 GB | ±1 % | +1 to +3 % if points come back for ≤ 0.3 ms | `compose_gguf.py` (minutes) and Q341's tool, ~2 h |
| 4 | **Lite MTP pass** (C8, C6): attention over the last 2,048 positions, top-4 experts | −1 to −3 points together | ~+1 % | ~+1 % at 6k; est. +8–12 % at 40k, where the pass's attention costs ~3.9 ms | Q341's tool on 8k–40k-token chunks |
| 5 | **Expert subset of the MTP block** (C7): the 64 experts its router picks most on agent text | −1 to −4 points | 0 | frees 0.65 GB for R8's state or a light drafting card | the router's counts from one run, then Q341's tool |

The engine work that the cost model ranks first is not a variant: R8 with the MTP chain as the executor's guesser (C10)
and dropped guesses that skip the drafting card (C11).

## 6. What Q211's kernels must support for these variants

- **Mode (a):** expert products for 2–4 tokens that read each weight once, with the down product and the weighted
  combine fused, in valid1's N2P/N3P and UD's IQ2_S/IQ3_XXS (then IQ2_XXS, IQ1_S, Q2_K, Q3_K); the dense Q5_0/Q8_0 products
  for 2–4 tokens at Ornith's shapes (3072 × 16384, 8192 × 3072, the 248,320-row head read once for all rows); Gated
  DeltaNet and attention for 2–4 tokens.
- **The MTP block's own types at 1–4 tokens** (the catch-up over accepted tokens): N2P gate/up and N3P down in valid1,
  Q8_0 in UD and the 35B, through the same expert kernel.
- **The shortlist head needs no new kernel:** permute the output layer's rows by frequency once, so the shortlist is a
  view of its first N rows, and map the drafted index back to a token id in the sampler (or keep a separate N-row tensor
  with its id map, 54–107 MB). Sampling over N logits exists.
- **The low-rank ranker:** a one-token product with 256-wide rows (248,320 × 256), a top-k over 248,320 approximate
  logits, then the exact rows gathered (`GET_ROWS` of 2,048 Q5_0 rows) and one small product.
- **Mode (b) needs none of the multi-token expert kernels:** every trunk step is a one-token step. It needs R8, whose
  replay of the accepted path is a few-token Gated DeltaNet update, and it profits from the MTP catch-up batched over
  2–4 committed tokens (the bullet above).

## 7. New registry rows

Labels C1–C15 stand for registry IDs assigned in [quants/experiments.md](../../../quants/experiments.md).

| Label | Row | Status |
|---|---|---|
| C1 | valid1's own MTP acceptance (N2P/N3P experts, Q5_0 dense): Q341's tool on the low-bit branch, agent and code text, drafts 1–3 (Q366's chain), reporting E[p(argmax q)] for greedy drafts and the position-keyed coupled match for pipelined sample-and-match beside Σ min(p, q). Predicted 0 to −4 points against UD's Q8_0 layer | planned |
| C2 | The per-draft cost of Ornith's MTP on one P106: `GGML_CUDA_PROFILE` of `draft-mtp` drafts and catch-up batches with valid1 at 4k, 16k and 40k positions (MTP pass, head, sampling, host gap per draft step); replaces this note's est. 1.8 + 3.8 + 0.7 ms | planned |
| C3 | An MTP format ladder on a fixed trunk: UD-Q2_K_XL's trunk with its MTP block in Q8_0 (as shipped), valid1's formats, dense Q8_0 + N2P/N3P experts and all Q4_0, composed byte for byte; acceptance per chain position. Separates the layer's format from the trunk's, which Q341's AD/UD gap mixes | planned |
| C4 | (into Q368) Draft-head shortlists from Ornith's own generated agent tokens: 8k, 16k, 32k and 64k rows in Q8_0, coverage first, then acceptance per chain position; a frequency-permuted output layer so the shortlist is a view of its first rows. Predicted 32k −1.5 to −2.5 points for 3.5 → 0.65 ms. The screened head is dropped from Q368 (no time gain below ~4.5 bits on Pascal) | planned |
| C5 | A low-rank ranker for the draft head: rank-256 SVD of the output layer (248,320 × 256, Q8_0, 67 MB) ranks the whole vocabulary, exact Q5_0 rows rescore its top 2,048 (SpecVocab without training first); recall of the sampled token offline, then acceptance. Est. ~0.8 ms at full coverage | planned |
| C6 | Fewer routed experts in the MTP pass (top-4, top-2, shared expert only, router renormalized): acceptance per chain position offline. Saves 0.2–0.4 ms per pass in valid1, so it pays only below ~0.5 points per 0.2 ms | planned |
| C7 | An expert subset for the MTP block: keep the 32–64 experts its router picks most on agent text and route among them; frees 0.65–0.76 GB of card memory for R8's state groups or a lighter drafting card; acceptance offline | planned |
| C8 | A windowed MTP attention (the last 1,024–4,096 positions) for drafts and catch-ups: caps the layer's KV reads (est. 3.9 ms per pass at 40k, 0.6 ms at 6k), its KV memory and its prompt-side pass; the output stays exact; acceptance offline at 8k–40k positions | planned |
| C9 | MTP KV without catch-up: keep the chained drafts' own KV entries at accepted positions instead of recomputing them from the trunk's states, saving one MTP pass (est. ~1.8 ms) per committed token; acceptance offline with Q366's chain | planned |
| C10 | The MTP chain as the pipelined executor's guesser (Q341's engine row, specified): drafts on the card holding the MTP block, a chain from the trunk state after each miss and one extension per right guess, D from the drafting card's load (1 + D + M)·s_d + (D + M)·g + (1 + M)·ℓ ≤ L + M·c: D = 3 with today's draft, 5 with a ≤ 2 ms guess; est. 35 → 38 tok/s on valid1. Needs R8, not Q211 | planned (after R8) |
| C11 | Dropped guesses skip the drafting card (reopens Q283): once the last card's sampler finds a miss, the guesses behind it run no layers, head or sampling there (a device flag read by early-exit kernels, or conditional graph nodes where compute 6.1 allows them), so an MTP guesser can keep 6–7 guesses in flight with the full head; est. 35 → ~41 tok/s on valid1 | planned |
| C12 | A light drafting card: move layers off the card holding the MTP block and the head until its load fits (~1.35 ms and ~1 GB per layer); a fit probe of valid1 + MTP block + R8's state at 32–49k positions first | planned |
| C13 | A confidence-gated chain length in mode (a): stop drafting when the MTP's top probability falls below a threshold, chosen offline from Q366's per-position confidence; est. +2–5 % over a fixed k = 2, as each extra verified token costs ~26 ms | planned |
| C14 | Re-anchored chains in mode (b): when the drafting card has slack, rebuild the chain's frontier from the newest trunk state through the guesses in flight (MTP passes without the head), so new guesses stay at chain depth ≤ D; offline acceptance first; est. +0.2–0.5 accepted guesses per round | planned |
| C15 | Ornith's expert overlap for 2–8 consecutive tokens (Q51 for Ornith): distinct experts per layer on agent text, which sets V(m) in mode (a) and Q211's payoff (independent routing: 15.75 / 23.3 / 30.5 experts at 2 / 3 / 4 tokens) | planned |
| — | Parked: a 2–3-bit full-vocabulary draft head or screen. A P106 product is issue-bound at ~3.8–4.2 ps per weight below ~4.5 bits (Q188), so it saves ≤ 15 % of the Q5_0 head's 3.5 ms. Reopens with a kernel under ~3 ps per weight | parked |
| — | Parked: a separately trained small draft head for Ornith (EAGLE-style). The built-in layer is accepted 0.85–0.91 against Flint's trained head's 0.25–0.37. Reopens if C1 finds valid1's layer far below Q341's | parked |

## 8. Not verified

- Every P106 time in section 1 is arithmetic from the kernel tables (Q188, Q166, the v2.0 profile, Q210), not a profile
  of the MTP graph (C2).
- P_4..P_7 are extrapolated from Q366's first look; Q366's own numbers are a first look, not the finished row.
- The verify model V(m) assumes Q211's kernels at the dense kernels' efficiency and nearly independent routing; C15 and
  Q211's own timings replace both.
- The acceptance costs of the shortlists, the ranker, fewer experts, the attention window and the expert subset come
  from other models' papers and Flint's coverage; none is measured on Ornith.
- Stage times (c, s_1, s_d) and S_eff are derived from L and Flint's pipeline, not traced on Ornith.
- The round model omits R8's own cost, the executor's per-token host work and its KV-step stalls (Q153), and the served
  overheads that made Flint's head cost ~35 % more in the server than on an idle card.

## Sources

Quantized drafts and MTP layers
- E. Georganas, D. Kalamkar, A. Kozlov, A. Heinecke, "ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts", [arXiv 2503.13565](https://arxiv.org/abs/2503.13565) ([html](https://arxiv.org/html/2503.13565v1))
- "QSpec: Speculative Decoding with Complementary Quantization Schemes", [arXiv 2410.11305](https://arxiv.org/abs/2410.11305)
- "Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design", [arXiv 2505.22179](https://arxiv.org/abs/2505.22179) ([html](https://arxiv.org/html/2505.22179v1))
- NVIDIA, "Nemotron 3 Super", [arXiv 2604.12374](https://arxiv.org/abs/2604.12374) ([html](https://arxiv.org/html/2604.12374v1): MTP layers kept in BF16, Table 2's acceptance lengths)
- C. E. Sørensen, "Quantizing Gemma 4's MTP drafters" (2026-05-29), [point.free](https://point.free/blog/gemma-4-mtp/) (community measurement)
- DeepSeek-AI, "DeepSeek-V3 Technical Report", [arXiv 2412.19437](https://arxiv.org/abs/2412.19437)
- Qwen, [Qwen3-Next-80B-A3B-Instruct model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) (recommended MTP settings)

Draft vocabulary
- W. Zhao et al., "FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling", [arXiv 2502.14856](https://arxiv.org/abs/2502.14856) ([html](https://arxiv.org/html/2502.14856v1), Table 1)
- R. Goel et al., "VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs", [arXiv 2506.22694](https://arxiv.org/abs/2506.22694); summarized in [arXiv 2603.05210](https://arxiv.org/abs/2603.05210)
- "Speculative Decoding with a Speculative Vocabulary" (SpecVocab), [arXiv 2602.13836](https://arxiv.org/abs/2602.13836) ([html](https://arxiv.org/html/2602.13836v1))
- "FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction", [arXiv 2509.18362](https://arxiv.org/abs/2509.18362) ([html](https://arxiv.org/html/2509.18362v1): Table 1, Table 2, section 3.4)

Fewer experts
- "DraftExpert: Expansion-Aware Self-Speculative Decoding for End-Device MoE Inference", [arXiv 2607.24434](https://arxiv.org/abs/2607.24434)
- "MoE-Spec: Expert Budgeting for Efficient Speculative Decoding", [arXiv 2602.16052](https://arxiv.org/abs/2602.16052)

Cost models, pipelines and coupling
- Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding", [arXiv 2211.17192](https://arxiv.org/abs/2211.17192)
- Z. Chen et al., "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding", [arXiv 2402.12374](https://arxiv.org/abs/2402.12374)
- T. Liu et al., "PEARL: Parallel Speculative Decoding with Adaptive Draft Length", [arXiv 2408.11850](https://arxiv.org/abs/2408.11850)
- K. Huang et al., "SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths", [arXiv 2405.19715](https://arxiv.org/abs/2405.19715)
- Y. Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees", [arXiv 2406.16858](https://arxiv.org/abs/2406.16858)
- PPSD, [arXiv 2509.19368](https://arxiv.org/abs/2509.19368); SPD, [arXiv 2605.30852](https://arxiv.org/abs/2605.30852); B. Butler et al., "PipeInfer", [arXiv 2407.11798](https://arxiv.org/abs/2407.11798)
- M. Daliri, C. Musco, A. T. Suresh, "Coupling without Communication and Drafter-Invariant Speculative Decoding", [arXiv 2408.07978](https://arxiv.org/abs/2408.07978)

This project
- [Q341's data](../../data/2026-10-28_ornith_mtp_accept/README.md) (acceptance by file, text and confidence),
  [the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md) (profile, MTP lines, attention at depth),
  [valid1's data](../../data/2026-10-28_ornith_valid1/README.md), [the 35B's data](../../data/2026-10-28_ornith35_kld/README.md),
  [the N2P/N3P kernels on a P106](../../data/2026-10-28_lowbit_rig/README.md) (Q188), [the IQ kernels](../../data/2026-10-27_iq_kernels/README.md)
  (Q166, Q210)
- [docs/drafter.md](../../drafter.md) ("What a guess may cost", "A cheaper guess"), [idle cards](idle_cards.md),
  [pipelined speculation](pipelined_speculation.md), [uzu lessons](uzu_lessons.md), [MTP head research](mtp_head_research.md)
- Registry ([quants/experiments.md](../../../quants/experiments.md)): Q19, Q43, Q51, Q154, Q158, Q162, Q166, Q171, Q188,
  Q189, Q193, Q194, Q210, Q211, Q212, Q281, Q283, Q288, Q341, Q366–Q369
