# Long-context and multi-turn quality of Flint quants: what 512-token KLD cannot see, cheaper measurements that can, and where the bytes should go (2026-10-24)

Scope: Q41's first multi-turn pair (lane E2) put G3 7.0 points below Unsloth's UD-Q4_K_XL (90 % interval −15.3 to
+1.3), although G3 is the closer file to BF16 by KLD (0.101 against 0.120). Every KLD in the
[registry](../../../quants/experiments.md) is measured on 512-token chunks of wikitext-2. This note asks whether that
yardstick misses what long, multi-turn agent use needs:
- what the 2024–2026 literature says about quantization at long context, in the attention path, in MoE routing and in
  agent tasks;
- which choices in G3, K2h and K1h are exposed;
- which cheap measurements on the 7-GPU quant machine would predict multi-turn task quality better than 512-token KLD,
  with costs and exact commands;
- which mix changes buy long-context quality per byte;
- a decision procedure for the release candidate.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (papers, code, model cards, measurements in this repository) and
  secondary.
- Nothing here was run on the rig or the quant machine. Section 2.3 re-analyses Q41's E2 pilot rows (per-attempt CSV).

Starting point, measured:
- **Flint:** 49 layers (2 dense, 47 MoE with 256 routed experts, top-6, and a shared expert), hidden 2,048, 32 query
  and 4 KV heads of 128, per-head RMSNorm on q and k, a learned sink per head, vocabulary 150,144; it always reasons.
  Its **14 global layers (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48)** attend to the whole context with no
  positional encoding. The other 35 layers attend to a sliding window of 513 keys with RoPE
  ([rec note](rec_engine.md)). The KV cache is f16 in every run discussed here.
- **KLD against BF16** (80 × 512 tokens of wikitext-2): G3 0.101 (0.104 on Q77's GPU set), G3h 0.106, K2h 0.109,
  K1h 0.112, UD-Q4_K_XL 0.120, Unsloth's Q8_0 0.052. llama-perplexity scores only the second half of each chunk
  (`first = n_ctx/2`, `tools/perplexity/perplexity.cpp:1772`), so these are positions 256–511.
- **Q41:**
  - E1 single-turn: G3 − UD −0.33 points, 90 % interval −1.24 to +0.58.
  - E2 multi-turn pilot (100 tasks, K = 1): UD 39 %, G3 32 %, gap mostly in miss_param (14 against 8 of 25).
  - Decision rule (2026-10-24): if E2 at K ≥ 2 keeps G3 more than 3 points below UD, the candidate moves up to the
    F1 class (KLD ~0.096).

## Verdict

- **Short-context KLD can mislead for agent use, but the first E2 pair does not show that it did.** It has three blind
  spots for long agent turns:
  - **Architecture.** In a 512-token chunk every sliding-window layer sees the whole chunk, so Flint's 14 global layers
    do nothing there that a sliding layer doesn't. Their long-range role, the only direct path for information more
    than 512 tokens back, is never scored.
  - **Format.** Wikitext has no chat template, special tokens, tool schemas (3–7.5k tokens at the start of every BFCL
    multi-turn prompt), tool outputs, reasoning or calls.
  - **Averaging.** A mean over all tokens is dominated by locally predictable ones. Perplexity correlates with
    long-context benchmarks at −0.18 to +0.27; restricted to the tokens that need the long context, at −0.90 to
    −0.96 (LongPPL).

  The first pair points away from long context:
  - G3's deficit is flat across context length (2, 2 and 3 tasks behind in the three length terciles of 33–34 tasks).
  - G3 leads on BFCL's long_context category (11 against 10 of 25).
  - The two files' verdicts differ on 19 % of tasks, less than UD's own two attempts (23.5 %); 13 against 6 discordant
    tasks give a sign-test p of 0.17.
  - The deficit sits in miss_param, a whether-to-call decision, the kind that low-bit quantization flips more often
    than which-tool decisions (arXiv 2608.06564).
  - In the literature, 4-bit long-context losses appear mostly at 64k–128k; at 8k every method tested matched BF16
    (Mekala et al., EMNLP 2025). BFCL multi-turn conversations run 4–33k tokens, median 8k.
- **Our mixes are exposed exactly where 512-token KLD cannot look.**
  - UD-Q4_K_XL keeps attention q, k, v and output at Q8_0 in 13 of the 14 global layers (Q5_K only in layer 45).
  - G3 has q at Q5_0 in seven global layers (25, 29, 33, 37, 41, 45, 48) and output at Q5_0 in five (33–48).
  - K2h adds Q5_0 k/v in five global layers and q in layers 0, 1 and 5; K1h also has output at Q5_0 from layer 25.
  - G3's KLD lead is bought in the routed and shared experts, which act on each position alone.
  - The decision log's fallback, the F1 class, has the same attention tensors as G3. It would not touch a long-context
    cause.
- **The two cheap measurements to run first** (est. 4–5 GPU-hours together for five files on one GPU set). Both use
  llama-perplexity with two small patches:
  1. **Transcript KLD:** KLD on Flint's own BFCL multi-turn conversations, teacher-forced in 8,192-token chunks against
     Unsloth's Q8_0. A per-token dump splits the score into reasoning, calls and answers and isolates tokens copied
     from more than 512 tokens back.
  2. **Wikitext at 8k:** KLD at `-c 8192` against `-c 512` on the same files and the same reference.

  KLD(G3)/KLD(UD) is 0.84 at 512 tokens. If that ratio stays ≲ 0.9 on agent transcripts at 8k, short-context KLD is
  not what misled. If it reaches ≥ 1, it is.
- **The top mix fix, if a long-context gap shows: all 14 global layers' attention at Q8_0 ("G3g").**
  - **For G3:** q in seven and output in five global layers, +37.7 MB per token (+1.7 % of G3h's 2.21 GB, est.
    +0.21 ms). That is half of what the F1-class fallback adds (+71 MB).
  - **For K2h and K1h:** the same plus their global k/v and early-global q, +51 and +57 MB.
  - **A zero-byte alternative:** GPTQ rounding of those twelve Q5_0 tensors, calibrated on long agent transcripts.
    GPTQ took the Q5_0 output head from +0.023 to +0.002 KLD (Q56).
  - **Not at risk:** the router (F32 or F16 in our files, BF16 in UD; a Q8_0 router would be a downgrade) and the f16
    KV cache.
- **Decision procedure: measure before acting on the E2 rule, and decide on enough tasks.**
  - The rule as written (G3 > 3 points behind at K = 2 on the 100 pilot tasks) fires about 20 % of the time when the
    files are equal: the standard error is 3.4–4.0 points (est.).
  - Decide on the 799-task lane at K ≥ 2 (standard error 1.2–1.4 points), or on the 200 miss_param tasks at K = 5
    (1.5 points).
  - Choose the fix that the KLD measurements point to.
- **A numerical check comes free with the transcript KLD.** G3, K2h and K1h run without the MoE down-projection
  rescale, on llama.cpp's FP16 activation-block sums (Q41's build).
  - That was shown NaN-free on 41,000 wikitext tokens.
  - Long agent prompts full of special tokens are a different activation regime, and the transcript run's NaN count
    settles it.

## Measurements, ranked

Ranked by information per GPU-hour for the release decision. GPU-hours are job-time estimates for scheduling on the
quant machine, not speed results.

| # | Measurement | What it predicts | Cost (est.) | How to run | Decision it informs |
|---|---|---|---|---|---|
| M0 | Re-analysis of the existing E2 rows by context length, category and failure type | whether the gap tracks context length | 0 (done, §2.3) | `per_attempt.csv`: `max_context`, `category`, `error_type` | No length trend: long context is not the leading explanation of the pilot gap |
| **M1** | **Transcript KLD:** BFCL multi-turn conversations from Q41's E2 runs (UD's and G3's, equal numbers), rendered with the chat template, truncated to 8,192 tokens, teacher-forced against Q8_0; per-token dump | KLD on the release domain: format, tool schemas and outputs, reasoning, positions 4,096–8,191 | Q8_0 base ~0.5 GPU-h; ~0.3–0.5 GPU-h per file; base file 29.5 GB for 24 conversations | Two patches (`--parse-special` for llama-perplexity, per-token dump), a render mode for `multi_turn.py`, `llama-perplexity -c 8192` (§3.1) | Ratio to UD ≤ 0.9: KLD ordering holds, keep the bytes; ≥ 1.0: test G3g |
| **M2** | **Wikitext KLD at `-c 8192`** (16 chunks, positions 4,096–8,191) against `-c 512` (80 chunks), both against Q8_0 | the effect of length alone, within one text domain | ~0.2–0.3 GPU-h per file; base 19.7 GB (+6.1 GB at 512) | stock llama-perplexity (§3.2) | Whether any file's KLD grows with context faster than UD's |
| M3 | **Long-range tokens:** M1/M2 dumps restricted to tokens copied from more than 512 tokens back, plus synthetic key–value retrieval text at `-c 32768` | retrieval through the 14 global layers, the long-context mechanism itself | dumps 0; synthetic ~0.3 GPU-h per file, base 19.7 GB | copy parse as in Q64; a text generator (§3.3) | Whether the global-layer tensors (G3g) are what to restore |
| M4 | E2 miss_param only: 200 tasks, K = 5, G3h against UD | the whether-to-call gap | ~6 server-hours per pair, 12–18 GPU-h | `multi_turn.py run` limited to miss_param (§3.4) | Whether the category gap is real (detectable: ~4.3 points) |
| M5 | E2 full lane: 799 tasks, K ≥ 2 | the multi-turn release gate | ~10 server-hours per pair at K = 2, 20–30 GPU-h | Q41 as planned (§3.5) | The E2 rule, decided at a standard error of 1.2–1.4 points |
| M6 | RULER multi-key needles and variable tracking at 32k and 64k, generated with reasoning | absolute long-context retrieval of the release file | ~2–4 GPU-h per file (2 tasks × 50 samples × 2 lengths) | RULER prompts through `/completion` (§3.6) | Release gate only; low sensitivity at ≤ 32k |
| M1b | Control: BFCL single-turn transcripts at `-c 1024` | agent format at short length | ~0.2 GPU-h per file; base 6.1 GB | as M1 | Separates format from length when M1 moves |

One add-on settles Q62's open half: an M2 run of G3h with `-ctk q8_0 -ctv q8_0` (q8_0 KV was checked only at 512
tokens).

## Mix fixes, ranked

Bytes per token use Q77's parameter counts: attention q 8.39 M per layer, output 8.39 M, k+v 2.1 M, shared expert
6.29 M. Q8_0 costs 1.0625 bytes per weight and Q5_0 0.6875, so each upgrade adds 0.375 bytes per weight. Time uses the
fork's streaming rates (Q8_0 150, Q5_0 138 GB/s). Near the bar the fork trades 0.015–0.025 KLD per ms
([unconventional ideas](unconventional_ideas.md)), so 0.2 ms is worth ~0.003–0.005 KLD of headroom (est.).

| # | Change | MB per token (est. time) | Expected effect | When |
|---|---|---|---|---|
| X1 | **G3g:** attention q in global layers 25, 29, 33, 37, 41, 45, 48 and output in 33, 37, 41, 45, 48 to Q8_0, so all 14 global layers have Q8_0 attention (k/v already are) | +37.7 (+0.21 ms) | At 512 tokens −0.001 to −0.003 KLD (est. from Q36/Q77's late-layer costs). At long context it removes all 5-bit tensors from the only long-range path, matching UD's allocation there | If M1/M2/M3 show G3's ratio to UD rising with context |
| X2 | GPTQ-rounded Q5_0 for X1's twelve tensors, calibrated on long agent transcripts | 0 | Most of Q5_0's excess error removed (est., from Q56: +0.023 → +0.002 on the head) | Instead of X1 if bytes matter; 1–2 days to capture each layer's inputs |
| X3 | K2h/K1h: k and v of the global layers 33–48 back to Q8_0 | +3.9 (+0.02 ms) | Restores the cached keys and values that every long-range lookup reads | Whenever K2h or K1h stays in play |
| X4 | K2h/K1h: X1 plus X3, plus q in global layers 0, 1, 5 (K1h: output in 25 and 29 as well) | +51.1 / +57.4 (+0.29 / +0.32 ms) | As X1 | As X1, for the lighter candidates |
| X5 | Importance matrix from 8k chunks of Flint's agent transcripts (`llama-imatrix --parse-special`), then re-quantize the Q5_0 tensors | 0 | Small (est.). Q12 found an own imatrix equal to Unsloth's, but only on 512-token wikitext. The global layers' attention-output inputs differ most between short and long contexts | If X1 helps and bytes matter |
| X6 | All attention q in 25–48 and output in 33–48 to Q8_0, sliding layers included | +125.8 (+0.70 ms) | Sliding layers see at most 513 keys at any context, so no long-context gain beyond X1 (est.) | Only if M3 contradicts that |
| X7 | Shared expert back to Q8_0 (K2h 33–45, K1h 25–45) | +30.7 / +49.5 (+0.17 / +0.28 ms) | Acts on each position alone, no long-context mechanism. Shared experts are sensitive in general (QuantMoE-Bench; Q45's X1) | By M1's per-file KLD only |
| X8 | F1 class: routed down-projections Q8_0 in 2–45, gate/up 2–8 back to Q5_0 | +71 (~0.5 ms) | −0.005 KLD at 512 tokens; acts on each position alone | Only if M1 tracks 512-token KLD and M5 still fails |
| X9 | Router or KV cache precision | 0 | Router is already F32/F16 (UD: BF16), nothing to gain. KV stays f16; q8_0 KV needs the M2 add-on first | — |

## 1. Literature

### 1.1 Long context: where quantization losses appear

- **Mekala et al., "Does quantization affect models' performance on long-context tasks?"** (arXiv 2505.20276, EMNLP
  2025).
  - **Setup:** Llama-3.1 8B/70B and Qwen-2.5 7B/32B/72B under FP8, GPTQ-int8, AWQ-int4, GPTQ-int4 and BNB-nf4, on
    Ruler, OneRuler (8k, 64k, 128k), NoCha (up to 128k) and long-form output tasks (9.7k examples).
  - **Average drop against BF16:** FP8 and GPTQ-int8 0.2–0.8 %, AWQ-int4 1.8 %, GPTQ-int4 2.7 %, BNB-nf4 6.9 %. On
    Ruler, GPTQ-int4 loses 21.2 % on Llama-3.1 8B and 10.3 % on Qwen-2.5 72B.
  - **By length:** "At the 8K context length, all methods perform similarly to BF16". 4-bit losses grow at 64k and
    128k, up to 59 % (BNB-nf4, Llama-3.1 70B, OneRuler); NoCha drops more steeply beyond ~75k.
  - **For Flint:** 4-bit weight losses are a 64k+ phenomenon in these models. The models are uniform 4-bit, whereas
    Flint's attention is mostly 8-bit.
- **Li et al., "Evaluating Quantized Large Language Models"** (arXiv 2402.18158, ICML 2024), section 7: LongEval (key–value
  retrieval up to 16k) and multi-document QA (up to 6k).
  - Weight-only W3 loses more on texts ≥ 4k than below; W4 stays within 2 % on long texts.
  - KV-cache quantization hurts more than weight quantization at the same width. KV8 is safe on long texts for most
    models, and KV4 only on short ones. Vicuna-7B on LongEval at 16k: FP16 57.8, W8KV8 56.4, W4KV8 59.6, W8KV4 37.0.
- **Fang et al., "What is Wrong with Perplexity for Long-context Language Modeling?"** (arXiv 2410.23771).
  - Perplexity averages over tokens that need no long context.
  - Key tokens are those whose log-probability with the full context exceeds that with a 4,096-token context by
    more than 2 (and whose long-context log-probability exceeds −2).
  - Perplexity correlates with LongBench, LongEval and RULER at −0.18, 0.24 and 0.27; perplexity on key tokens
    (LongPPL) at −0.96, −0.90 and −0.90.
  - Fewer than 10 % of tokens pass the first criterion on GovReport.
- **Needle tests saturate.**
  - RULER (Hsieh et al., arXiv 2404.06654, COLM 2024): models score "close to perfect" on the vanilla needle; only
    half of 17 long-context models hold up at 32k on harder variants (multiple needles, tracing, aggregation).
  - NoLiMa (Modarressi et al., arXiv 2502.05167, ICML 2025) removes literal overlap between needle and question: 11
    of 13 models fall below half of their short-context score at 32k (GPT-4o 99.3 → 69.7).
  - A plain single needle at ≤ 32k therefore separates two near-lossless quants of one model poorly.

### 1.2 The attention path

- **KV cache over weights.** Li et al. (above) find the KV cache the most length-sensitive part. Here the KV cache is
  f16 in Q41 and on the rig; q8_0 KV (Q62) was checked only at 512 tokens.
- **Outlier values in q and k.** Jin et al., "Massive Values in Self-Attention Modules are the Key to Contextual
  Knowledge Understanding" (arXiv 2502.01563, ICML 2025):
  - concentrated large values sit in q and k, not v;
  - disrupting them drops passkey retrieval from 100 % to 0–9 % and GSM8K from 76.9 to 4.0 on Llama-3-8B, while
    parametric-knowledge tasks fall far less (Cities 99.4 → 88.2);
  - quantizers that rescale channels (AWQ, SmoothQuant) keep contextual tasks, and GPTQ degrades them;
  - **the concentration is caused by RoPE** and is absent in models without it. In Flint the RoPE layers are the
    sliding ones, whose behaviour 512-token KLD does measure in full. Its global layers are NoPE, and q/k RMSNorm
    bounds every head's q and k.
- **Retrieval heads.**
  - Wu et al., "Retrieval Head Mechanistically Explains Long-Context Factuality" (arXiv 2404.15574): fewer than 5 %
    of heads do the copying from long contexts. Masking the top 50 (~5 %) drops every model's needle score below 50,
    and random heads barely matter.
  - DuoAttention (Xiao et al., arXiv 2410.10819): the remaining heads need only sinks and recent tokens. 25 % of
    heads kept full in an MHA model and 50 % in a GQA model hold LongBench (40.08 full against 40.21 for
    Llama-3-8B-1048K).
  - Flint builds that split into its architecture: the 14 global layers (28 % of layers) are where any retrieval head
    must live.
- **NoPE attention is sensitive to its sharpness at length.** Wang et al., "Length Generalization of Causal
  Transformers without Position Encoding" (arXiv 2404.12224):
  - NoPE failures beyond the training length "always co-occur with the distraction of attention distributions";
  - a per-head temperature fixes them, about 1.2× scale for 2× length and 1.8× for 8×.
  - Changes to the effective logit scale of a global head matter more the longer the context.
- **What makers keep in high precision.**
  - DeepSeek-V3 (arXiv 2412.19437, §3.3.1) keeps the embedding, output head, MoE gates, norms and attention operators
    in BF16/FP32 under FP8 training.
  - gpt-oss (arXiv 2508.10925), which alternates 128-token banded and dense attention with learned sinks like Flint,
    quantizes only its MoE weights (90+ % of parameters) to MXFP4. Its released config leaves attention, routers,
    embedding and head unconverted (`modules_to_not_convert`).
  - Red Hat's Gemma-4-26B-A4B NVFP4 keeps the head and router in original precision.

### 1.3 MoE routers and shared experts

- **Shared experts and early layers deserve more bits.** QuantMoE-Bench (Li et al., arXiv 2406.08155), GPTQ on
  DeepSeek-MoE-16B and Mixtral at 2–4 bits:
  - DeepSeek-MoE at 2 bits averages 49.44; with 4-bit attention 56.77 (2.06 bits), with 8-bit shared experts as well
    61.55 (2.12 bits);
  - 4 bits for the first four blocks beat 4 bits for the last four (Mixtral 51.12 against 45.54).
  - Our own sweeps agree: the shared expert must stay Q8_0 in layers 2–16 (Q77), and Q4_1 in 25–48 costs +0.019 KLD
    (Q45's X1).
- **Routing consistency matters most for long generation.** VSRAQ (arXiv 2606.05688) measures per-token, per-layer
  Jaccard overlap of quantized and full-precision top-k sets and aligns router logits during quantization.
  - On Solar-Open-100B at W4A16, generation tasks (MMLU-Pro, GPQA-Diamond, GSM8K with answer extraction) go 72.19
    (AutoRound) → 73.22 against 73.62 unquantized.
  - Likelihood multiple-choice tasks barely move (76.44 → 76.76).
  - It claims, without measuring it, that expert mismatches compound over long generations.
- **Flint's routing flips at the 6th expert from rounding alone** (Q21), which sets its KLD floor (0.052–0.065).
  Routing depends on the hidden state, not on distance, so no source found ties router error to context length.
- **Router precision itself is not the issue here.** DeepSeek-V3, Gemma-4 NVFP4 and every Flint file in this project
  keep the router unquantized.

### 1.4 Multi-turn and agent tasks

- **ACBench** (Dong et al., arXiv 2505.19433, ICML 2025): across 15 models, 4-bit GPTQ/AWQ keeps workflow generation and
  tool use within 1–3 % but loses 10–15 % on real-world application tasks.
- **Red Hat's Gemma-4-26B-A4B NVFP4** (model card): BFCL v4 single-turn 83.85 → 82.00, multi-turn 62.13 → 62.13,
  agentic 62.16 → 50.33. BFCL's multi-turn lane did not move where the agentic lane lost 11.8 points.
- **Jang et al., "Flat Score, Amplified Failures"** (arXiv 2607.27275), τ²-bench at 16, 8 and 4 bits, 456 episodes per
  cell:
  - no score change survives correction;
  - 4 bits amplify existing tool-name hallucinations up to 2.5× and create "essentially no new failures" (rank
    correlation of failure sets ≥ 0.94).
- **Wu, Dhiman and Koshiyama, "Which Decisions Low-Bit Quantization Breaks, and How to Predict Them"** (arXiv 2608.06564):
  - decision margins shrink roughly linearly under quantization (m′ ≈ c·m + b, c < 1);
  - whether to call a tool is "often more sensitive than which tool to call";
  - margins predict flip rates with a median error of 1.0 point;
  - GPTQ and GGUF flip far fewer whether-to-call decisions than plain rounding.
  - BFCL's miss_param category tests exactly that decision: the model should ask for a missing parameter instead of
    calling (BFCL v3 multi-turn blog).
- **Calibration under quantization.** Proskurina et al. (arXiv 2405.00632, NAACL 2024 Findings): 4-bit GPTQ lowers
  confidence in the true labels, and the loss falls "disproportionately" on samples where the full model was already
  unsure.
  - Dutta et al. (arXiv 2407.09141): answer flips at near-equal accuracy, and compressed models "significantly worse"
    on MT-Bench free-form generation.
- **Multi-turn is hard regardless of quantization.** Laban et al. (arXiv 2505.06120): every model tested loses 39 % on
  average when a task arrives in underspecified pieces over several turns, mostly as unreliability, through premature
  assumptions it does not recover from.
  - Quantization that thins decision margins acts on the same weak spot: when to ask instead of assuming.

### 1.5 Global against sliding-window layers

No primary source measures quantization sensitivity of global against sliding-window layers in an interleaved model;
searches found only per-token or per-layer mixed precision (HyQuant, SliderQuant, MXSens). The argument here is
therefore architectural:
- **Positions and keys:** a sliding layer computes the same function at 8k as at 512 tokens, over at most 513 keys,
  so 512-token KLD scores its quantization error fully.
- **Global layers:** they act on all keys, without positional decay to suppress distant distractors. Their role grows
  with the context and is invisible below 513 tokens.
- **Outliers:** the RoPE-induced outliers that make q/k quantization dangerous for contextual tasks (Jin et al.) would
  sit in Flint's sliding layers, which short-context KLD covers.

## 2. Our mixes in that light

### 2.1 What a 512-token chunk exercises

- **Positions:** llama-perplexity scores positions 256–511 of each chunk.
- **Attention spans:** with a window of 513 keys including the current token, every sliding layer attends to the
  full chunk, exactly as the global layers do. Flint is effectively an all-global model on this test.
- **Untested:** whether the global layers still find the right key among thousands, and how quantization noise in
  their q, k, v and output changes that. Also untested are the chat template, special tokens, tool schemas and the
  model's own reasoning.
- **Same for every per-layer sweep:** Q36, Q45, Q73, Q77 and Q90 all ranked layer groups on this test. Their finding
  that layers 25–48 tolerate lower precision covers the global layers only at short range.

### 2.2 Precision per class, global layers marked

| Tensor class (weights per layer) | G3 | K2h | K1h | UD-Q4_K_XL |
|---|---|---|---|---|
| Attention q (8.39 M) | Q8_0 0–24, Q5_0 25–48 | Q5_0 0–8 and 25–48, Q8_0 9–24 | as K2h | Q8_0; Q5_K in 19, 26, 35, 39, 43, 45, 46 |
| Attention k, v (2.1 M) | Q8_0 | Q8_0 0–32, Q5_0 33–48 | as K2h | Q8_0; Q5_K in the same 7 |
| Attention output (8.39 M) | Q8_0 0–32, Q5_0 33–48 | as G3 | Q8_0 0–24, Q5_0 25–48 | Q8_0; Q5_K in the same 7 |
| Shared expert (6.29 M) | Q8_0 | Q5_0 33–45 | Q5_0 25–45 | Q8_0 in 34 layers, Q5_K in 12, IQ4_XS in 1 |
| Routed gate/up | Q8_0 2–8, Q5_0 9–48 | as G3 | as G3 | IQ4_XS, Q5_K, Q4_K, IQ3_S |
| Routed down | Q5_0 2–45, Q8_0 46–48 | as G3 | as G3 | Q5_K 29, IQ4_XS 16, Q6_K 1, Q8_0 1 |
| Dense layers 0–1 | Q8_0 | Q8_0 | Q8_0 | Q8_0 |
| Router | F32 or F16 | as G3 | as G3 | BF16 |
| Output head | Q8_0 (G3h: GPTQ Q5_0) | GPTQ Q5_0 | GPTQ Q5_0 | Q8_0 |

Global layers whose attention is below 8 bits:

| File | q | k, v | Output | Extra MB to make all global attention Q8_0 |
|---|---|---|---|---|
| UD-Q4_K_XL | 45 (Q5_K) | 45 | 45 | — |
| G3 / G3h | 25, 29, 33, 37, 41, 45, 48 | — | 33, 37, 41, 45, 48 | 37.7 |
| K2h | 0, 1, 5, 25, 29, 33, 37, 41, 45, 48 | 33, 37, 41, 45, 48 | 33, 37, 41, 45, 48 | 51.1 |
| K1h | as K2h | as K2h | 25, 29, 33, 37, 41, 45, 48 | 57.4 |

UD's allocation is from its GGUF header ([K-quant note](kquant_decode_pascal.md)). G3, K2h and K1h are from Q73's and
Q77's recipes. The two strategies differ in kind:
- **UD-Q4_K_XL:** keeps nearly all attention at 8 bits and spends 4–5 bits on the experts.
- **G3:** spends 5–8 bits on the experts and the whole shared expert, and saves on attention q and output in the second
  half of the model.

On 512-token wikitext G3's choice wins by 0.019 KLD.

### 2.3 What the first E2 pair shows

Attempt 1 of both files on the same 100 pilot tasks (25 scenarios × 4 categories), sampled the makers' way. Terciles
are by the length UD's conversation reached.

| Context reached | Tasks | UD passes | G3 passes |
|---|---|---|---|
| 4.2–7.3k | 33 | 15 | 13 |
| 7.4–9.1k | 33 | 12 | 10 |
| 9.1–32.7k | 34 | 12 | 9 |

| Category | UD | G3 | Discordant tasks (UD only / G3 only) |
|---|---|---|---|
| base | 12 | 11 | 3 / 2 |
| miss_func | 3 | 2 | 2 / 1 |
| miss_param | 14 | 8 | 6 / 0 |
| long_context (median 10k tokens) | 10 | 11 | 2 / 3 |

- **Failure types (UD / G3):** state mismatch 22 / 17, empty turn 22 / 24, response mismatch 16 / 27.
  - Conversations with a malformed step: 11 / 14; with a step at the 4,096-token limit: 4 / 3.
  - G3's six miss_param losses are 3 state mismatches (a call where it should have asked, or a wrong one), 2 empty
    turns and 1 response mismatch.
- **Noise:** the files' verdicts differ on 19 tasks (13 against 6, two-sided sign test p 0.17). UD's own attempts 1
  and 2 differ on 23.5 % of 34 tasks.
  - miss_param's 6–0 alone has p 0.03, but it was picked out of four categories after the fact (×4: 0.12).
  - UD's 56 % on miss_param is above its 48 % on base, unusual for that category, which suggests UD drew high there.
- **Length:** nearly half of these conversations pass 8,192 tokens (100 of 234 attempt rows), 12 % pass 12,288 and
  under 1 % 32,768.
  - The first prompt, system text plus tool schemas, is 3.1–7.5k tokens (median 5.0k).

So far nothing ties the gap to context length. It is consistent with noise, possibly plus a thinner margin on
whether-to-call decisions.

### 2.4 How large a Q5_0 error in attention is at long context (est.)

- **Error size.** Group-64 4-bit round-to-nearest leaves a relative error of 0.086 on Flint's weights (Q74). Each
  extra bit halves it, so Q5_0 (group 32) is ~0.03–0.045 and Q8_0 ~0.004–0.006 (est.). The importance matrix lowers
  the error on the channels that matter.
- **Effect on a logit.** With RMSNorm on q and k per head, a weight error changes q's direction, not its length. A
  random direction error of relative size ε moves a logit s = q·k/√128 by about ε·s_max/√128.
  - Q5_0, ε = 0.04 and s_max 10–30 (an assumed range for QK-normalized heads): 0.035–0.11 nats per key.
  - Q8_0: 0.004–0.013.
- **Effect on retrieval.** A global head must give the target more weight than the summed mass of all distractors,
  and without positional decay that mass grows with the number of keys N. The target's margin shrinks by about
  ln(N₂/N₁) if distractor logits stay put: 2.8 nats from 512 to 8,192 keys, 4.2 nats to 32k (est.). A fixed
  perturbation then tips more near-tied lookups.
- **Against NoPE's own sensitivity.** NoPE needs a ~1.2× change of attention scale (~20 % of every logit) to handle
  twice the length (Wang et al.). That is two orders of magnitude above the Q5_0 perturbation.
- **Conclusion:** Q5_0 q and output in global layers are unlikely to break retrieval at BFCL's 8–33k (est.), and their
  effect grows with length, as Mekala et al. saw for 4-bit files at 64k+. The estimate assumes random errors.
  Quantization errors are not random where activations have outlier channels, which is why the measurement is still
  worth its few GPU-hours.

### 2.5 Risk ranking for long, multi-turn use

1. **Global k and v in Q5_0** (K2h, K1h: layers 33–48). Every cached key and value carries the error, and it enters
   both which token is found and what is read from it.
2. **Global q in Q5_0** (G3, K2h, K1h: seven layers in 25–48; K2h, K1h also 0, 1, 5). Direct perturbation of
   long-range selection, the part that grows with ln N.
3. **Global attention output in Q5_0** (G3, K2h: 33–48; K1h: 25–48). The error is relative and does not depend on
   position, but it rides on the retrieved content: 4 % noise on a copied argument value matters only when its margin
   is thin.
4. **Shared expert in Q5_0** (K2h 33–45, K1h 25–45). Acts on each position alone; risky for decisions in general
   (§1.3), not for length.
5. **Routed down-projections in Q5_0** (2–45, all candidates). Acts on each position alone and is scored fully by
   512-token KLD; UD's experts are lower still (IQ4_XS, Q5_K).
6. **The GPTQ Q5_0 head** (G3h, K2h, K1h). Acts on each position alone; calibrated on wikitext and Flint's own
   transcripts, it loses only ~0.001 on a domain it was not calibrated on (Q56). It thins every token's margin a
   little, including whether-to-call tokens.

Sliding-layer tensors (q in 26–28, 30–32, …, 46–47 and so on) carry no long-context-specific risk: 512-token KLD
already measures them as they act at any context.

## 3. Measurements on the quant machine

### 3.0 Ground rules: compare like with like

- **One reference per text.** Unsloth's Q8_0, on the 2 × V100 32 GB set (the only set that holds it with its buffers).
  BF16 (79.6 GB) does not fit the GPU sets once the long-context buffers are added.
- **One GPU set, one build, one set of flags for all candidates:** Q41's build, `-b 2048 -ub 512 -fa on -ngl 99`, f16
  KV. UD-Q4_K_XL runs with the rescale and ours with `LLAMA_NO_MOE_DOWN_RESCALE=1`, as in Q41.
  - The same file moves by KLD 0.065 between GPU types and by ~0.07 from summation order in one-token decoding
    (Q21, Q38). G3 scored 0.101 and 0.104 on two sets.
  - Batch shape changes the summation order too, so `-b`/`-ub` stay fixed.
- **Ratios and differences within one setting only.** An 8k KLD against Q8_0 and a 512-token KLD against BF16 differ
  in reference, positions and text. Never compare them in absolute terms. The 512-token run is repeated against Q8_0
  (6.1 GB base) so that every ratio uses the same reference.
- **Paired statistics by chunk.** Per chunk and file, take the mean KLD; take the difference per chunk; report its
  mean and standard error across chunks. Est. 0.002–0.004 for 16–24 chunks of 4,095 scored tokens, enough to resolve
  a change of 0.05–0.08 in the 0.84 ratio.
- **The Q8_0 reference favours Q8_0 tensors.** A candidate's Q8_0 tensors are byte-identical to the reference's, so they
  contribute no divergence. That flatters G3g against G3, and UD's Q8_0 attention against G3's Q5_0.
  - If a decision hinges on G3g, repeat that pair against BF16 on 4 chunks at 8k, with the routed experts on the CPU
    (`-ot`, 94 GB host RAM). A CPU or mixed run adds its own compute-path offset, equal for every candidate.

### 3.1 M1: transcript KLD

**Material.** The E2 raw outputs from Q41: each step's model text, with reasoning. Take equal numbers of conversations
from UD's and G3's runs, and from Q8_0's anchor runs once they exist, so that no candidate is scored only on its own
samples. Each conversation must reach at least 8,192 tokens: ~43 % do, so ~340 per file and attempt on the 799-task
lane (est.).

**Rendering.** Replay each conversation through BFCL's backends, as `multi_turn.py score` already does, to regenerate
the tool results. Then render the final step's prompt with the GGUF's chat template and append the final output. The
result is the exact token stream the model saw plus its last answer, with every earlier `<think>` block kept, as in the
runs. This needs a render mode for [multi_turn.py](../../../bench/bfcl_flint/multi_turn.py), ~60 lines.

**Chunks.**
- **Build:** tokenize with special tokens parsed, truncate each conversation to exactly 8,192 tokens and concatenate.
  Every conversation starts with a special token, so boundaries survive re-tokenization; the builder checks that the
  re-tokenized file splits exactly at multiples of 8,192.
- **Size:** 24 conversations give 24 chunks and 98,280 scored tokens. The base file holds 150,148 16-bit values per
  scored token (`perplexity.cpp:147`), 300 KB, so 29.5 GB.
- **Tool block:** scored positions 4,096–8,191 include the end of the tool block in conversations whose schemas pass
  4,096 tokens. The dump lets the analysis drop those positions.
- **BOS:** if the GGUF sets `add_bos`, llama-perplexity overwrites each chunk's first token with BOS
  (`perplexity.cpp:566-569`); the scored half is unaffected.

**Patches** (on the fork, small, no effect on other tools):
1. **`--parse-special` for llama-perplexity.** Today the flag is registered for llama-imatrix only (`common/arg.cpp:3227-3232`).
   llama-perplexity tokenizes without parsing special tokens (`perplexity.cpp:473`), so `<|assistant|>` or
   `<tool_call>` would become plain text pieces. The fix adds `LLAMA_EXAMPLE_PERPLEXITY` to the flag and passes
   `params.parse_special` to `common_tokenize`.
   - Only the base run tokenizes. Candidates read the token ids from the base file (`perplexity.cpp:1716`).
2. **Per-token dump in `kl_divergence()`.** `kld_values` and `p_diff_values` already hold one value per scored token
   (`perplexity.cpp:283-284, 1739-1740`).
   - The patch also keeps the same-top flag and both models' log-probability of the actual token from `log_softmax`
     (`:191-240`).
   - It writes them with the chunk and position when an environment variable names a file, ~30 lines.
   - In perplexity mode the same patch writes each token's log-probability, which M3's LongPPL selection needs.

**Commands.**

```bash
# reference, once (2 x V100), with the parse-special patch
llama-perplexity -m flint-Q8_0.gguf -f bfcl_e2_c8192.txt --parse-special -c 8192 --chunks 24 \
  -b 2048 -ub 512 -ngl 99 -fa on --kl-divergence-base q8.bfcl_e2.c8192x24.kld
# each candidate, all on the same GPU set; token ids come from the base file
LLAMA_KLD_DUMP=g3h.bfcl_e2.dump LLAMA_NO_MOE_DOWN_RESCALE=1 \
llama-perplexity -m flint-G3h.gguf -c 8192 -b 2048 -ub 512 -ngl 99 -fa on \
  --kl-divergence-base q8.bfcl_e2.c8192x24.kld --kl-divergence
```

**Memory.**
- **GPU:** Q8_0 (42.3 GB) fits the V100 pair with ~0.25 GB of global-layer KV at 8k and the usual compute buffers. Every
  candidate fits two 3090s or the V100s.
- **Host:** the logits buffer is reserved at n_ctx × n_vocab × 4 bytes (`perplexity.cpp:1743`), 4.9 GB at 8k; about
  half of it is touched.

**Outputs.**
- Mean KLD and same-top rate per file, and the ratio to UD.
- KLD by segment: system and tools, user, tool output, reasoning, call, answer.
- KLD by position bin, and on the first token after each `</think>` (the call/no-call token).
- The count of NaN or infinite values, which is the overflow check for files that run without the rescale.

**Cost (est.).** The base prefills 197k tokens, minutes of GPU time. Model load and the 30 GB base file (write once, read
per candidate) dominate: ~0.5 GPU-hours for the base, ~0.3–0.5 per candidate on a two-card set. Delete the base file
afterwards; it is cheaper to regenerate than to keep on the scratch disk, which must keep ≥ 50 GB free.

**M1b, the format control.** BFCL single-turn transcripts (E1 prompts plus answers) of at least 1,024 tokens,
truncated to 1,024, at `-c 1024`. 40 chunks give 20,440 scored tokens and a 6.1 GB base, ~0.2 GPU-hours per file
(est.). If M1 moves and M1b moves with it, the cause is format, not length.

### 3.2 M2: wikitext at 8k

```bash
llama-perplexity -m flint-Q8_0.gguf -f wiki.test.raw -c 8192 --chunks 16 -b 2048 -ub 512 -ngl 99 -fa on \
  --kl-divergence-base q8.wiki.c8192x16.kld
llama-perplexity -m flint-Q8_0.gguf -f wiki.test.raw -c 512 --chunks 80 -b 2048 -ub 512 -ngl 99 -fa on \
  --kl-divergence-base q8.wiki.c512x80.kld
# candidates against both bases, same flags as in 3.1
```

- **Size:** wikitext-2's test set is ~0.3 M tokens for a 150k vocabulary (est.), up to ~36 chunks of 8,192. 16 chunks
  give 65,520 scored tokens and a 19.7 GB base.
- **Cost:** ~0.2–0.3 GPU-hours per file at 8k and ~0.1 at 512 (est.).
- **Script:** [quality_kld.sh](../../../scripts/quality_kld.sh) hard-codes `-c 512`; a `CTX` variable would make
  this its own mode.
- **Limit:** wikitext articles run a few thousand tokens and chunks cross article boundaries, so M2 tests robustness
  to length more than retrieval. Its job is to separate a length effect from M1's format effect.
- **Add-on:** the q8_0 KV question of Q62 is one more candidate run (G3h with `-ctk q8_0 -ctv q8_0`) against the 8k
  base.

### 3.3 M3: long-range tokens and synthetic retrieval

- **Long-range copy tokens, free from the dumps.**
  - **Parse:** the greedy longest-previous-match copy parse of Q64 ([ngram_replay](../../../bench/ngram_replay/README.md))
    over each chunk's tokens (stored in the base file).
  - **Selection:** keep tokens inside matches of ≥ 8 tokens whose source starts more than 512 tokens back. In agent
    transcripts these are mostly argument values, file names and IDs from earlier tool outputs and turns: needles in
    natural text, reachable directly only through the global layers.
  - **Report:** KLD and same-top rate on them, per file.
- **LongPPL's key tokens with Flint's own short context.**
  - **Short pass:** one extra Q8_0 pass at `-c 1024` over windows that overlap by 512 tokens, so that every position
    is scored once with 512–1,023 tokens of context. That is about what the sliding layers alone can see.
  - **Selection:** keep tokens whose 8k log-probability exceeds that by more than 2 and is itself above −2 (LongPPL's
    thresholds).
  - **Cost:** ~0.2 GPU-hours (est.), with the dump patch in perplexity mode.
- **Synthetic key–value retrieval.** 32k-token documents:
  - **Layout:** the first half ~1,000 records ("The access code for room 4821 is 39457."), the second half queries
    with answers ("Room 4821: 39457").
  - **Score:** KLD and the reference's NLL on the answer tokens only.
  - **Size:** `-c 32768` with 4 chunks gives 65,532 scored tokens and a 19.7 GB base; the host logits buffer is
    ~10–20 GB per process.
  - **Cost:** ~0.3 GPU-hours per file (est.).
  - **At 64k:** run it with 2 chunks only when the machine is otherwise idle, because the host buffers double.

### 3.4 M4: the miss_param subset

- **What:** 200 miss_param tasks, K = 5, G3h against UD, with the harness and settings of Q41.
- **Power:** with E[p(1−p)] = 0.118 (UD's E2 pilot) and no scenario clustering inside one category, the paired
  standard error is √(2 × 0.118 / 5 / 200) = 1.5 points. That detects ~4.3 points with 80 % power (est.).
- **Cost:** ~3 server-hours per file at the harness's E2 estimate (12 server-hours for 799 tasks × 5), 12–18 GPU-hours
  for the pair.
- **When:** run it if miss_param is still ≥ 8 points behind at K = 2, or if M1 shows thinner margins at call/no-call
  tokens.

### 3.5 M5: the full E2 lane

| Tasks × attempts | Paired SE, plain / clustered by scenario (est.) | 80 % power to detect | P(point estimate < −3) if the files are equal |
|---|---|---|---|
| 100 × 2 (the pilot) | 3.4 / 4.0 points | ~10–11 points | 19–23 % |
| 799 × 2 | 1.2 / 1.4 | ~3.4–3.9 | ~1–2 % |
| 799 × 3 | 1.0 / 1.15 | ~2.8–3.2 | < 1 % |

- **Basis:** E[p(1−p)] = 0.118, ω² ≈ 0 (Q30, Q41), and the pilot's clustering factor of 1.16 (SE 5.03 against 4.32).
- **Non-inferiority:** Q41's 3-point margin needs N·K ≥ 12.4 × 0.118 / 0.03² ≈ 1,630 when the files are equal. That
  is about 799 × 2 without clustering and 799 × 3 with it, the K the task note already allows.
- **Cost:** K = 2 for one pair is ~10 server-hours (20–30 GPU-hours) at the harness's estimate.

### 3.6 M6: RULER at 32k and 64k

- **Why:** the rig serves Flint at up to 262,144 tokens of context, and 4-bit losses start at 64k–128k in Mekala et
  al. A release file therefore gets one absolute long-context check, beyond BFCL's 33k.
- **Tasks:** RULER's multi-key needle (`niah_multikey_2` or `_3`) and variable tracking (`vt`), 50 samples each at 32k
  and 64k.
  - Generate with reasoning (n_predict 4,096) through `/completion` with the chat template, and score exact match.
  - The slots need ≥ 70k tokens of context each (Flint's KV is 28 KiB per token, ~1.9 GB at 64k).
- **Cost:** ~2–4 GPU-hours per file (est.: 9.6 M prompt tokens at ~3,000 tok/s, plus reasoning).
- **Gate:** the release file within 10 points of UD on each task and length. At 50 samples the per-cell standard error
  is ~5–7 points, so this catches failures, not small differences.
- **Prior:** expected to saturate at 32k (RULER, NoLiMa on literal needles); 64k is where a difference could show.

### 3.7 Costs together (est.)

| Stage | Files | GPU-hours |
|---|---|---|
| M1 + M2 (+ M1b) | Q8_0 bases; UD, G3h, K2h, K1h, G3gh | 4–5 (+1) |
| M3 synthetic + LongPPL pass | same | 2 |
| M4 | UD, G3h | 12–18 |
| M5 at K = 2 | UD, the chosen file | 20–30 |
| M6 | the chosen file, UD | 4–8 |

The KLD stages fit inside one morning of one GPU set. They should run before M4 and M5, because they say which file
the task runs should compare.

## 4. Mix fixes

- **Byte arithmetic.**
  - Per layer, Q8_0 → Q5_0 saves 3.146 MB for attention q or output, 0.788 MB for k+v and 2.359 MB for the shared
    expert (Q77's counts × 0.375 bytes).
  - Time per upgraded q or output tensor is 8.914 MB / 150 GB/s − 5.768 MB / 138 GB/s = 0.0176 ms; per k+v pair
    0.0044 ms; per shared expert 0.0132 ms (est., large-shape streaming rates).
- **G3g (X1).**
  - **Contents:** twelve tensors, 37.7 MB per token, 0.21 ms (est.). G3gh reads ~2.249 GB per token against G3h's
    2.211 and UD-Q4_K_XL's 2.213.
  - **Build:** in a minute with [compose_gguf.py](../../../scripts/compose_gguf.py), taking the tensors from Unsloth's
    Q8_0. Q8_0 is plain rounding of BF16, so these are the tensors a full quantization would write:

    ```bash
    compose_gguf.py flint-G3h.gguf flint-G3gh.gguf \
      'blk\.(25|29|33|37|41|45|48)\.attn_q\.weight=flint-Q8_0.gguf' \
      'blk\.(33|37|41|45|48)\.attn_output\.weight=flint-Q8_0.gguf'
    ```

  - **K2h and K1h:** K2hg and K1hg add `blk\.(33|37|41|45|48)\.attn_(k|v)` and `blk\.(0|1|5)\.attn_q` (K1h also
    `blk\.(25|29)\.attn_output`).
- **Why G3g ranks above the F1-class fallback.** F1 has the same attention tensors as G3. It differs only in the routed
  experts (down-projections Q8_0 in 2–45, gate/up Q5_0 in 2–8), +71 MB for −0.005 KLD at 512 tokens.
  - If the multi-turn gap has a long-context cause, F1 spends twice G3g's bytes away from it.
  - If it has none, G3g costs ~0.2 ms that the measurements will show to be unnecessary. That is why G3g waits for M1
    to M3 instead of replacing G3h now.
- **GPTQ for the global-layer tensors (X2).**
  - **Tooling:** [quants/gptq_head](../../../quants/gptq_head/) rounds one matrix from captured inputs.
  - **For q and output:** the inputs are the attention-norm output (q) and the attention result (output) of each
    global layer, captured with `llama-eval-callback` or an extension of `head_dump` on 8k transcript chunks.
  - **Calibration:** the attention result at long context is a mixture over thousands of tokens, unlike at 512, so
    calibrate on the long transcripts.
  - **Expected:** Q56's head kept ~90 % of GPTQ's gain across text domains (est.).
- **Not proposed.**
  - A Q8_0 router: the router is unquantized in every file.
  - q8_0 KV for long contexts, until the M2 add-on clears it: KV quantization is the most length-sensitive kind in Li
    et al.
  - Sliding-layer upgrades for long-context reasons: §1.5.

## 5. Decision procedure for the release candidate

1. **Now, no GPU.** The pilot gap shows no length trend (§2.3). Keep G3h, K2h and K1h as they are while measuring; do
   not trigger the F1-class move on the 100-task reading alone.
2. **KLD stage (M1, M2, M3; ~6–7 GPU-hours, one GPU set).** Files: UD-Q4_K_XL, G3h, K2h, K1h, G3gh. For each, compute
   R = KLD(file)/KLD(UD) on wikitext at 512, wikitext at 8k, transcripts at 8k and long-range tokens, all against
   Q8_0.
   - **R(G3h) ≤ 0.90 on transcripts and long-range tokens, upper 95 % bound < 1.0:** short-context KLD did not
     mislead on this pair, and G3h's bytes stay. The E2 gap is noise or a decision-margin effect that a mean KLD
     averages away, and the task stage decides.
   - **R(G3h) ≥ 1.0 on transcripts or long-range tokens while ≤ 0.9 at 512:** short-context KLD misled.
     - If G3gh brings R back to ≤ 0.9, G3gh replaces G3h as the candidate (+38 MB), and X2 is the zero-byte
       follow-up.
     - If it does not, read the dump by segment and position before choosing. The F1 class is right only if the
       excess sits in tokens that the experts decide rather than the context.
   - **K2h and K1h** stay on the shortlist only with R ≤ 0.95 on transcripts and long-range tokens; otherwise their
     global-attention variants (+51/+57 MB) replace them.
   - **Any NaN** in a no-rescale run on transcripts blocks that configuration until it runs on FP32 activation sums
     (Q76) or with the rescale.
3. **Task stage.**
   - **Lane:** the E2 decision is taken on the full lane at K ≥ 2 for the file chosen in step 2 against UD, with Q41's
     3-point non-inferiority margin (lower 90 % bound above −3).
   - **The pilot reading:** the 100-task K = 2 result is an early warning only. Act on it before the full lane only
     if its point estimate is below −3 and step 2 found a mechanism.
   - **miss_param:** if it stays ≥ 8 points behind at K = 2, M4 (K = 5 on 200 tasks) decides whether the
     whether-to-call gap is real. A real gap with R ≤ 0.9 points at the head and the shared expert, whose errors thin
     every decision margin, rather than at long-context attention.
4. **Release gates.**
   - E1 and E2 non-inferior; the format gate (malformed + runaway, upper 90 % bound ≤ +0.5 points).
   - No NaN in any run with the release build and environment.
   - M6 within 10 points of UD at 32k and 64k.
   - The model card states the long-context checks next to the 512-token KLD.

What each stage needs, in samples:

| Check | Sample | Resolves (est.) |
|---|---|---|
| Transcript or wikitext KLD at 8k | 16–24 chunks × 4,095 scored tokens | paired KLD difference ±0.004–0.008 (95 %); ratio change ≥ 0.05–0.08 |
| Long-range tokens | whatever M1/M2 contain (est. 2–8 % of scored tokens, 1,500–8,000) | ratio change ≥ ~0.15 |
| E2 lane | 799 tasks × 2–3 attempts | 3-point margin; 80 % power for ~3–4 points |
| miss_param | 200 tasks × 5 | ~4.3 points |
| RULER | 50 samples × task × length | failures, not small gaps |

## Already covered

- **Noise floor and compute paths:** Q21 (0.052–0.088 from arithmetic alone), Q38's one-token summation-order noise
  (~0.07); the research page's noise-floor section.
- **Per-layer sensitivity at 512 tokens:** Q18, Q36, Q45, Q73, Q77, Q90. This note does not repeat their rankings; it
  marks which of their layers are global (§2.2).
- **Task protocol, KLD as a filter and not a ranking, multi-turn agentic damage:**
  [task-level quality checks](task_level_quality_checks.md) and Q41. This note adds the length and format dimension
  and the decision-margin literature (arXiv 2608.06564) for miss_param.
- **Sliding-window correctness in batched prefill at 1,024 tokens:** Q26.
- **q8_0 KV at 512 tokens:** Q62; the long-context half is the M2 add-on here.
- **The GPTQ head and its calibration across domains:** Q56, [gptq_head](../../../quants/gptq_head/).
- **A needle at 32k and 128k as an exactness test for the planned Pascal attention kernel:**
  [attention at long context](pascal_attention_long_context.md) §7. That tests the kernel; M6 tests the file.
- **Long-context speed and the 64k cliff:** [long-context cliff](long_context_cliff.md), Q54, Q79.
- **The lossy-room frame** (~1 ms of headroom, 0.015–0.025 KLD per ms): [unconventional ideas](unconventional_ideas.md).

## Not verified

- **Context dependence of KLD:** whether any Flint file's KLD gap to UD grows between 512 and 8k tokens, or on agent
  text. No published measurement of KLD against context length for weight quantization was found; M1 and M2 are the
  first.
- **The reference at 8k:** Q8_0's own distance from BF16 at 8k, and how much the Q8_0 reference flatters Q8_0 tensors
  (§3.0).
- **FP16 overflow on agent text:** whether llama.cpp's FP16 activation-block sums overflow on long agent prompts
  without the rescale. The channel RMS read from Unsloth's importance matrix (≤ 302 outside layers 46–48) is an average
  over its calibration text; single tokens can be far larger.
- **The miss_param gap:** whether it is real; 6–0 on 25 tasks at K = 1 is not evidence on its own.
- **The logit-noise estimate (§2.4):** it assumes random weight errors and logit scales of 10–30 for Flint's
  QK-normalized global heads.
- **Unsloth's calibration:** the context length of Unsloth's importance-matrix calibration (the docs give content and
  size, not chunk length).
- **Retrieval heads in Flint:** where they sit. The architecture confines them to the 14 global layers; which of those
  matter most is not known.
- **Tooling:** the render mode, the chunk builder, the dump patch and the parse-special patch are specified, not
  written.

## Sources

Papers and model cards (P), read 2026-10-24:
- Mekala, Atmakuru, Song, Karpinska, Iyyer, [Does quantization affect models' performance on long-context tasks?](https://arxiv.org/abs/2505.20276)
  (arXiv 2505.20276 v3, EMNLP 2025): average drops by method, Ruler/OneRuler/NoCha by length, the 8K statement, the 59 %
  case.
- Li et al., [Evaluating Quantized Large Language Models](https://arxiv.org/abs/2402.18158) (arXiv 2402.18158, ICML
  2024), section 7 and Table 7: weight, activation and KV-cache quantization on LongEval and multi-document QA.
- Fang et al., [What is Wrong with Perplexity for Long-context Language Modeling?](https://arxiv.org/abs/2410.23771)
  (arXiv 2410.23771 v5): key-token definition, thresholds, correlations.
- Hsieh et al., [RULER](https://arxiv.org/abs/2404.06654) (arXiv 2404.06654, COLM 2024); Modarressi et al.,
  [NoLiMa](https://arxiv.org/abs/2502.05167) (arXiv 2502.05167, ICML 2025).
- Jin et al., [Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding](https://arxiv.org/abs/2502.01563)
  (arXiv 2502.01563 v4, ICML 2025): Table 1, section 3.6, the RoPE origin.
- Wu, Wang, Xiao, Peng, Fu, [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574)
  (arXiv 2404.15574); Xiao et al., [DuoAttention](https://arxiv.org/abs/2410.10819) (arXiv 2410.10819): retrieval-head
  ratios, Tables 3–4.
- Wang et al., [Length Generalization of Causal Transformers without Position Encoding](https://arxiv.org/abs/2404.12224)
  (arXiv 2404.12224 v2): distraction, per-head temperature, scale rule of thumb.
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (arXiv 2412.19437), §3.3.1;
  OpenAI, [gpt-oss-120b & gpt-oss-20b model card](https://arxiv.org/abs/2508.10925) (arXiv 2508.10925) and
  [gpt-oss-120b `config.json`](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) (`quantization_config`);
  Red Hat, [gemma-4-26B-A4B-it-NVFP4](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-NVFP4) model card (BFCL v4
  single-turn, multi-turn, agentic; unquantized head and router).
- Li, Jin, Cheng, Chen, [QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts](https://arxiv.org/abs/2406.08155)
  (arXiv 2406.08155): shared-expert, attention and first-block allocations.
- [Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models](https://arxiv.org/abs/2606.05688)
  (arXiv 2606.05688 v1): routing Jaccard, generation against likelihood tasks.
- Dong et al., [Can Compressed LLMs Truly Act?](https://arxiv.org/abs/2505.19433) (arXiv 2505.19433, ICML 2025).
- Jang, Yang, Lim, Park, [Flat Score, Amplified Failures](https://arxiv.org/abs/2607.27275) (arXiv 2607.27275).
- Wu, Dhiman, Koshiyama, [Which Decisions Low-Bit Quantization Breaks, and How to Predict Them](https://arxiv.org/abs/2608.06564)
  (arXiv 2608.06564 v5).
- Laban, Hayashi, Zhou, Neville, [LLMs Get Lost In Multi-Turn Conversation](https://arxiv.org/abs/2505.06120) (arXiv
  2505.06120); Proskurina et al., [When Quantization Affects Confidence of Large Language Models?](https://arxiv.org/abs/2405.00632)
  (arXiv 2405.00632, NAACL 2024 Findings); Dutta et al., [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)
  (arXiv 2407.09141).
- Gorilla team, [BFCL v3: multi-turn and multi-step function calling](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html):
  category definitions (miss_param: ask instead of assuming), state- and response-based checks.

Secondary (S):
- Unsloth, [Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs): calibration content
  (agentic coding, chat, multilingual; > 1.5 M tokens for v2.0), Divergence-300@32.
- Searches for per-layer-type quantization studies of interleaved local/global models (HyQuant arXiv 2608.27875,
  SliderQuant arXiv 2603.25284, MXSens arXiv 2607.17733): none compares global with sliding-window layers.

Code (P): llama.cpp as in the fork's `pascal-p106-v2` at `032b130f4` (upstream base `d77256999`):
- `tools/perplexity/perplexity.cpp`: `first = n_ctx/2` (`:540`, `:1772`), base-file layout (`:147`, `:518-524`),
  tokenization without special tokens (`:473`), BOS overwrite (`:566-569`), token ids read from the base file (`:1716`),
  per-token KLD arrays (`:283-284`, `:1739-1740`), logits reservation (`:1743`), same-top count (`:238-239`);
- `common/arg.cpp:3227-3232`: `--parse-special`, registered for llama-imatrix only.

This repository:
- [quants/experiments.md](../../../quants/experiments.md): Q12, Q18, Q21, Q26, Q36, Q41, Q45, Q56, Q62, Q64, Q73, Q74,
  Q76, Q77, Q90, the 2026-10-24 decision-log rows.
- [bench/bfcl_flint](../../../bench/bfcl_flint/README.md) (multi-turn loop, process flags, E2 cost estimates) and Q41's
  E2 pilot rows (per-attempt CSV with `max_context`, `category`, `error_type`).
- Notes: [rec engine](rec_engine.md), [task-level quality checks](task_level_quality_checks.md),
  [unconventional ideas](unconventional_ideas.md), [attention at long context](pascal_attention_long_context.md),
  [long-context cliff](long_context_cliff.md), [K-quant decode](kquant_decode_pascal.md) (UD-Q4_K_XL's allocation),
  [quants/research.md](../../../quants/research.md) (rescale removal, noise floor), [gptq_head](../../../quants/gptq_head/README.md).
