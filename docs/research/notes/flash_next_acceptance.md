# Raising the drafters' acceptance for Qwen3.8-Flash-Next on coding-agent text (2026-11-04)

Speculative decoding is the multiplier left on Qwen3.8-Flash-Next on the P106 rig. A one-token step costs 136 ms
([Q215](../../data/2026-11-03_flash_next_baseline/README.md)); DFlash2 with two drafted tokens reaches 9.5 tok/s against
7.8 without a draft ([Q561](../../data/2026-11-03_flash_next_dflash2/README.md)); and the small-batch kernels
([Q591](../../data/2026-11-04_mtok_experts/README.md)) aim to make a 5-token pass cost ~1.3× a one-token pass, where it
costs 2.65× today. Tonight's analyses narrowed the other bets: wide token trees buy est. 0.01–0.03 accepted tokens per
extra node ([MoE speculation](moe_speculation_union.md)) and copy chains gain at most +5–6 % ([Q617](../../data/2026-11-04_copy_spans/README.md)).
What is left is the drafter's own acceptance, multiplied by cheaper verify passes and a draft length chosen per step
(Q687). This note asks five questions of papers, model cards, pull requests and issues (2025–2026 where they exist):

1. How a 2–3-bit target changes the acceptance of drafters fitted to the original model, and whether the target file
   should be chosen for acceptance as well as quality.
2. How the model's MTP head, block drafters (DFlash, DFlash2, DSpark) and EAGLE-3 compare on agentic and coding text:
   acceptance by position and cost per drafted token.
3. Which training-free changes raise acceptance, and which of them keep the output exact.
4. What cheap adaptation of a drafter to the target's own agent output costs and gains.
5. Where DFlash2's block width and MTP's depth should sit once a k-token pass is cheap.

It builds on [Flash-Next drafters](flash_next_drafters.md), [what is new](flash_next_news_2026-11.md),
[drafter acceptance](drafter_acceptance.md), [the cheap MTP draft](cheap_mtp_draft.md),
[MTP head research](mtp_head_research.md), [MoE speculation](moe_speculation_union.md) and
[docs/drafter.md](../../drafter.md), and does not repeat them. Sources were read on 2026-11-04. Nothing was run, built
or downloaded; the fork's behaviour is read from the source of its base (`37e717ff6`).

Conventions:
- "est." marks this note's own arithmetic or inference; the basis is given next to it.
- **τ** is tokens per step, the bonus token included. **P_j** is the probability that the j-th drafted token is accepted
  together with all earlier ones, so τ = 1 + ΣP_j; the conditional acceptance is a_j = P_j / P_(j−1). Published
  "weighted" per-position rates are P_j.
- **T(m)** is the time of an m-token verify pass (m = k + 1 for k drafted tokens), T1 = T(1); **r** is the cost of one
  more column as a fraction of T1; **D** is the draft time of a step; **S** = τ·T1 / (D + T(k+1)) is the speed-up over
  decoding without a draft.
- **Sample-and-match** is the fork's default verification: the draft is its argmax, the target samples its own token at
  each position and the draft survives while they agree. **Rejection** is `--spec-draft-sampling probabilistic`: the draft
  is sampled and a token is accepted with probability min(1, p/q).
- Effects are on generation tok/s. Generation was ~71 % of the nine common agent-check tasks' time with DFlash2
  (1,483 s of 2,088 s, Q561), so a per-task effect is ~0.7 of the figures here.
- Q-numbers are rows of the [registry](../../../quants/experiments.md).

## Summary

1. **The 2-bit file costs acceptance: est. 5–15 % of τ against an FP8-class target, and no source measures it on equal
   text.** With sampled verification a position accepts with probability Σ min(p, q) = 1 − TV(p, q); a drafter fitted to
   the original model loses at most TV(original, quantized) per position, which Pinsker bounds by √(KLD/2): 0.38 at the
   rig's 0.285 and 0.24 at IQ3_S's 0.12. The observed losses are far below that: FP8 verifiers lose nothing (EAGLE-3,
   first position 0.705 → 0.714), an MTP head at Q4_K_M loses 1.7 points against Q8_0, Unsloth's MTP speed-up falls from
   1.67× on UD-Q4_K_XL to 1.34× on UD-IQ1_S (a file of KL ~0.33), and the rig's DFlash2 sits at Inco AI's FP8 figures on
   one coding request (3.11 against 2.83–3.26). Section 2.
2. **Choose the file by KLD; acceptance follows it, so it is not a second objective.** At est. 2–4 % of τ per 0.1
   of KLD, a ≤ 54 GB file at KLD 0.12 instead of 0.285 would add est. 3.5–7 % (Section 2.4). A drafter refitted on the
   deployed file's own output recovers part: +5 % on an IQ3_S target. The rest is the quantized target's added randomness.
3. **Of the Flash-Next block drafters only DSpark documents its training target, and it is NVFP4**, with agent sessions
   that target wrote itself. Inco AI states the FP8 checkpoint it measured on, not the one it trained on.
4. **MTP and DFlash2 tie at three drafted tokens on the rig (τ 3.10 and 3.11); DFlash2 leads beyond.** On GPUs at three
   drafted tokens DFlash2 has +6–17 % τ over MTP (Flash-Next, FP8, temperature 1.0), and at seven drafted +9–20 % (27B).
   Its draft is one pass of est. 15–25 ms whatever the block; MTP pays est. 7–7.7 ms per drafted token. After the
   kernels that crossover sits at k = 3 (Section 6).
5. **DSpark does not beat MTP on agent text.** Its card puts it below the model's MTP head at 7 steps on held-out
   agent sessions (3.70 against 4.24, greedy), and a community DSpark for the 27B target sits 12–17 % below MTP-7 on
   code and math at temperature 1.0. Its edge is a cheaper pass (1.75 against 2.03 ms per token at 0–8K tokens of
   context), which turns to a loss at 96–128K (3.87 against 3.52 ms). Block drafters also lose more with context than MTP
   does (DSpark −44 % from ≤ 8K to 96–128K; MTP −20 %).
6. **Acceptance at positions 4–7 on agent text is unmeasured**, and the only 7-position measurement found (a community
   INT4 port) collapses after position 3 (0.31 → 0.14 → 0.10 → 0.02 → 0.00). With that tail the best draft is 3–4
   tokens at S ≈ 1.75; with the rig-anchored tail est. 5–6 tokens at S ≈ 2.2–2.4 (Section 6).
7. **The verification rule is the largest exact lever, and the fork's base already has it.** `--spec-draft-sampling
   probabilistic` gained +3.1 to +11.9 % accepted length for DFlash2 at temperature 1.0 and +2.6 to +8.0 % for MTP (merged
   llama.cpp PRs, six GPU/model pairs for DFlash2); `--spec-draft-p-min` must then be 0. Block verification (lossless) adds
   +5–8 % wall-clock over token-wise rejection in its paper and is not implemented anywhere in llama.cpp. Section 4.1.
8. **Relaxed acceptance is lossy and gives +8–10 % τ**; unions of two drafters, the target's previous-step top-k,
   drafter top-k and temperature, and drafting only inside tool-call JSON are each worth est. ≤ 3 % on exact output
   (Sections 4.2–4.5).
9. **Self-distillation on the deployed target's agent sessions has the largest published gains of any adaptation**:
   +38 % τ on held-out agent sessions for a block drafter that got agent data after chat data (DSpark), +49 % for an MTP
   head trained for one step only (FastMTP, not transferable: Qwen3.8's head is trained for several), +5 % for a
   DFlash2 refit on IQ3_S output. Est. +5–15 % τ for the rig, ~1–2 days of quant-machine time for 10–20 M tokens
   (Section 5).
10. **Q687's length rule is exact under sample-and-match, and needs a causal confidence under rejection.** A rule that
    keeps or drops a drafted token by that token's own probability biases sampled output once the draft is verified by
    rejection (DSpark's appendix gives a 0.85/0.15 against 0.70/0.30 example); the rule must use the distribution at
    the position (max probability, the selector's conditional rows), as `--spec-draft-p-min` does (Section 6.3).
11. **Ranked ideas** (Section 7): the kernels with DFlash2 at 5–6 drafted tokens (S 1.22 → est. 2.2–2.4); a DFlash2 refit
    on the target's own agent sessions (est. +5–15 % τ); Q687's per-step length (est. +5–12 %); rejection verification
    (est. +4–10 %).

## 1. The setting in numbers

Measured on the rig (Swift-1.5 GSQ-RCO Q2_0 on Miner Llama v3's base, sampler temperature 1.0, top-p 0.95, top-k 20,
greedy drafts with sample-and-match):

| Quantity | Value | Source |
|---|---|---|
| DFlash2, 2 drafted tokens, agent check | τ 2.56 (0.778 of drafted tokens accepted over 60 requests, 0.62–0.94 per request), step ~270 ms, 9.5 against 7.8 tok/s | [Q561](../../data/2026-11-03_flash_next_dflash2/README.md) |
| DFlash2, 3 drafted tokens, smoke requests | τ 3.11 (coding, temperature 1.0), 2.84 (greedy); 4 drafted tokens crossed the host's memory floor | same |
| Gyro MTP draft, 3 drafted tokens | τ 3.10 (0.701 of drafted tokens), step ~347 ms | [Q226](../../data/2026-11-03_flash_next_mtp/README.md) |
| T(m), m = 1…5 | 136, 185.8, 244.6, 305.4, 359.9 ms (r ≈ 0.36–0.43; 2.65× at five) | Q583 |
| After the kernels (est.) | T1 86–90 ms; T(5) ≈ 1.3 × T1, so r ≈ 0.075 (0.10 if the per-row cost stays at today's) | [MoE speculation](moe_speculation_union.md) |
| Draft cost (est.) | DFlash2 one pass 15–25 ms; MTP 7–7.7 ms per drafted token (four passes 30.9 ms, 18.5 ms of it the 248k-row output layer; Q635 est. −3 ms of the four) | Q583, Q635 |
| KLD against the Q8_0 reference on agent text | Q2_0 0.285, IQ3_S ~0.12; the floor from summation order alone is 0.023–0.025 | [quality chain](../../data/2026-11-03_flash_next_quality/README.md), Q217 |

Per-position acceptance on the rig, est. from two measurements: Q561's agent check (S18) gives P_1 + P_2 = 1.56 (τ 2.56) and the 3-drafted
smoke ΣP_j = 2.10–2.11, so a chain of P = 0.86 / 0.70 / 0.54 fits both. The tail is not measured; this note uses a
conditional acceptance of 0.72 beyond position 3 (est.: llama.cpp's #30927 test shows a_2 = 0.65 and a_3 = 0.71, so
conditional acceptance does not fall quickly), giving P_4…P_7 = 0.39 / 0.28 / 0.20 / 0.145 and τ(k) = 1.86 / 2.56 / 3.10
/ 3.49 / 3.77 / 3.97 / 4.12 for k = 1…7.

## 2. Target quantization and acceptance

### 2.1 What acceptance is under each verification rule

- **Sample-and-match** (the fork's default): a draft token x_d survives with probability p(x_d), where p is the target's
  distribution after its sampler chain. The best draft is the target's mode, and anything that moves or flattens the
  target's mode costs acceptance directly.
- **Rejection:** a sampled draft token survives with probability Σ min(p, q) = 1 − TV(p, q)
  ([Leviathan et al.](https://arxiv.org/abs/2211.17192), Lemma 3.3 and Corollary 3.6); a perfect q accepts everything.
  For constant per-token acceptance α the expected tokens per step are (1 − α^(γ+1)) / (1 − α), and the wall-time gain
  (1 − α^(γ+1)) / ((1 − α)(γc + 1)) with c the cost ratio of draft to target.
- Both are per position. A drafter fitted to the original distribution p_B and verified against the quantized p_Q loses
  at most TV(p_B, p_Q) per position (|p_Q(x) − p_B(x)| ≤ TV for any token), and Pinsker bounds the mean TV by √(KLD/2).
  The bound is loose here because a quantized model's KL sits in a thin tail of positions: most tokens of agent text are
  near-deterministic for both files.

### 2.2 Three ways the drafter and the verifier differ

| Channel | What happens | Can it be recovered? |
|---|---|---|
| Label shift | the drafter learned p_B; the verifier samples from p_Q | by fitting on p_Q's own output; the part of p_Q that is quantization noise stays |
| Input drift | the drafter reads hidden states of the quantized target (DFlash2: layer inputs 5/15/25/35/45), which differ from the original's | by fitting on the deployed file's hidden states |
| Base against fine-tune | the rig runs Swift-1.5, drafters are fitted to the base model | by the same refit; acceptance on the rig sits at published levels, so it is small |

Of the Flash-Next block drafters only [DSpark](https://huggingface.co/shadowpa0327/Qwen3.8-Flash-Next-DSpark) documents
its training target: NVFP4 (draft BF16), with stage-2 data regenerated by the target and stage-3 agent sessions that the
target wrote in thinking mode. [Inco AI's card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) states the
official FP8 checkpoint as its measurement target and gives no training data or target. A community port of a Qwen3.8-27B
drafter says outright that heavier quantization "perturbs the very hidden states it learned to read"
([Doopeworld](https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM)), and quantifies nothing.

### 2.3 What was measured

| Source | Target | Result |
|---|---|---|
| [Red Hat, Qwen3-32B EAGLE-3, vLLM](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3/discussions/2) | FP8-dynamic against BF16 | P_1…P_3 0.714 / 0.492 / 0.333 against 0.705 / 0.476 / 0.313: no loss |
| [Inco AI, Flash-Next](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) | official FP8, temperature 1.0, 3 drafted | τ 2.83–3.26 (HumanEval 2.94, MBPP 3.06); MTP 2.50–3.08 |
| Rig, Q2_0 | temperature 1.0 | τ 3.11 at 3 drafted on one coding request; 2.56 at 2 drafted over agent sessions |
| [Unsloth MTP heads](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md) | UD-Q4_K_XL, greedy, B200 | head BF16 / Q8_0 / Q4_K_M: acceptance 66.5 / 66.1 / 64.4 %; speed-up 1.67× on UD-Q4_K_XL, 1.34× on UD-IQ1_S |
| [Doopeworld, Qwen3.8-27B DSpark](https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM) | GPTQ-Int4, drafter fitted to FP8 | τ 2.45–2.79 at 7 drafted (probabilistic); the maker's 3.39 is on other workloads and sampling |
| [hotocooo DFlash2 v2](https://huggingface.co/hotocooo/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-DFlash) | IQ3_S | Inco AI's drafter refit 10,000 steps on 2.9 M rows from that target: τ 2.82 → 2.98 (training subset), 3.03 held out |
| [nazeshinjite](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF) | UD-Q5_K_XL | the drafter itself at Q4_0 against Q8_0: acceptance 86.4 against 85.4 %, so the drafter's own bits are free |
| [llama.cpp #25618](https://github.com/ggml-org/llama.cpp/issues/25618) | Q4_K_M, IQ2_XXS, Q8_0 vs BF16 | a k-column verify and a one-token decode sum in different orders; on quantized targets that flips near-ties (the runner-up at 0.0001–0.27 nats), BF16 targets stayed identical |

The Unsloth pair is the one controlled contrast of target quantization: the same shared Q8_0 head, greedy, one GPU, a
4-bit and a 1-bit target. Dividing the speed-ups (1.34 / 1.67 = 0.80) gives −20 % of τ at equal step costs, an upper
end, since the 1-bit file's step also differs. Unsloth's UD-IQ1_S has KL ~0.33 against the NVFP4 model
([zephel01's card](https://huggingface.co/zephel01/Qwen3.8-Flash-Next-12G32G-GGUF)), the rig's Q2_0 has 0.285 against its
Q8_0. The IQ3_S and Q4_K_XL class sits at the bottom of the range.

### 2.4 Reading for the rig

- **Est. cost of the 2-bit file: 5–15 % of τ** against an FP8-class target. Basis: a conditional acceptance 5 % lower
  (relative) at each position takes τ at 3 drafted tokens from 3.10 to 2.91 (−6 %), 10 % lower to 2.73 (−12 %); the
  Unsloth pair bounds it above at −20 %; the rig's own τ at published levels bounds it from below.
- **Slope: est. 2–4 % of τ per 0.1 of KLD** (6–12 % over 0.285, from the 5–10 % lower conditional acceptance above). Q2_0
  → a ≤ 54 GB file at KLD 0.12 would be est. +3.5–7 % τ. Choose by KLD (Q217, Q218, Q219, Q221); acceptance does not need a second objective. A file's mean KLD
  does not say where its errors sit, so the offline replay below decides close calls.
- **What a refit buys: up to the retrain's gain**, +5 % at IQ3_S, est. +5–10 % at Q2_0 (a noisier target leaves more to
  learn). It cannot beat the quantized target's own entropy.
- **The cheap measurement is a replay by file, on the quant machine:** the same agent transcripts through the Q2_0,
  IQ3_S and Q8_0 files with the same DFlash2 drafter and `LLAMA_SPEC_DUMP_FULL_BLOCK=1` (Q688's dump), acceptance by
  position per file, at the same verify width (#25618: different widths flip near-ties).

## 3. The MTP head, block drafters and EAGLE-3 on agentic and coding text

### 3.1 Accepted tokens per step

| Source | Target, sampler | Drafted | MTP | DFlash2 | DSpark | EAGLE-3 |
|---|---|---:|---|---|---|---|
| [Inco AI, Flash-Next](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (GB300, SGLang) | FP8, temperature 1.0, top-p 0.95, top-k 20 | 3 | 2.50–3.08 | 2.83–3.26 | – | – |
| [z-lab / Inco AI, Qwen3.8-27B](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (H200) | same sampler | 7 | 3.74–5.02 (HumanEval 3.91, MBPP 3.99) | 4.10–5.46 (4.39, 4.79) | 3.01–4.36, a community drafter | – |
| [DSpark card, Flash-Next](https://huggingface.co/shadowpa0327/Qwen3.8-Flash-Next-DSpark), held-out agent sessions | NVFP4, greedy | 7 | 4.24 (≤ 32K 4.41, > 32K 4.13) | – | 3.70 (≤ 32K 4.10, > 32K 3.46) | – |
| [DSpark paper](https://arxiv.org/abs/2607.05147), Qwen3-4B, chain drafting, code sets (MBPP, HumanEval) | BF16, temperature 1.0 | block | – | DFlash 4.40 (MBPP) – 4.74 (HumanEval) | 5.13–5.38 | 3.69–4.16 |
| Rig | Q2_0, temperature 1.0 | 2 / 3 | 3.10 at 3 | 2.56 at 2, 3.11 at 3 | – | – |
| [ik_llama.cpp #2761](https://github.com/ikawrakow/ik_llama.cpp/pull/2761) | Flash-Next, `n_max` 3 | 3 | – | accepted 88.6 % code, 73.1 % extraction, 39.5 % prose; speed +39 %, +28 %, −7 % | – | – |
| [nazeshinjite](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF) (M5 Max) | UD-Q5_K_XL, temperature 1.0 | 3 (MTP) / 5 | 89.7 % per drafted token | 87.4 %; speed against MTP: prose −1 %, code +14 %, rewrite +18 %, list +29 % | – | – |

- DFlash2's card describes a block-diffusion drafter that keeps the top candidates at each position, a selector that
  traces one coherent path through them, and two-tap dynamic convolutions against decay along the block. Its Flash-Next
  figures use block 4 only (3 drafted); the 27B card uses block 8.
- The DSpark card's numbers are greedy. DFlash's paper puts temperature 1.0 at −13 % of τ against greedy (Qwen3-4B,
  block 16: 6.54 → 5.69), so greedy τ overstates sampled τ by ~15 % (est.). No source compares EAGLE-3 with MTP on
  agentic text for any Qwen3.8 model.
- No EAGLE-3 drafter exists for Flash-Next. Elsewhere EAGLE-3 trails the block drafters in τ (the table) and costs more
  per drafted token, since it is autoregressive (3.4).

### 3.2 Acceptance by position (P_j unless marked)

| Source | P_1 | P_2 | P_3 | P_4 | P_5 | P_6 | P_7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rig (est., Section 1) | 0.86 | 0.70 | 0.54 | 0.39 | 0.28 | 0.20 | 0.145 |
| [llama.cpp #30927](https://github.com/ggml-org/llama.cpp/pull/30927)'s own DFlash2 test | 0.65 | 0.42 | 0.30 | | | | |
| Qwen3.8-27B DSpark, GPTQ-Int4 target, probabilistic ([Doopeworld](https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM)) | 0.76 | 0.45 | 0.31 | 0.14 | 0.10 | 0.02 | 0.00 |
| Qwen3-32B EAGLE-3, BF16 ([Red Hat](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3/discussions/2)) | 0.705 | 0.476 | 0.313 | | | | |
| MiMo-7B-RL's own MTP head applied recurrently, trained for one step (FastMTP's base) | 0.70 | 0.10 | ~0 | | | | |

- Position-1 conditional acceptance on Qwen3-4B at temperature 1.0 ([DSpark paper](https://arxiv.org/abs/2607.05147)):
  math EAGLE-3 0.81, DFlash 0.88, DSpark 0.93; chat 0.53 and 0.72 for EAGLE-3 and DFlash; on code DFlash falls from 0.87
  at the first position to 0.78 at the last of its block. Conditional acceptance can rise along a block (EAGLE-3 on chat,
  0.53 → 0.74) because only survivors count.
- The one 7-position measurement on a Qwen3.8 drafter collapses after position 3; the rig's own tail is unmeasured. The
  verify-step dump already holds what is needed: `LLAMA_SPEC_DUMP_FULL_BLOCK=1` decodes DFlash2's whole block while the
  draft stays at `n_max` (Q688).

### 3.3 Text type and context length

- **Context:** on held-out agent sessions DSpark falls from τ 5.64 at 0–8K tokens to 3.16 at 96–128K (−44 %), MTP with
  7 steps from 5.07 to 4.07 (−20 %) ([card](https://huggingface.co/shadowpa0327/Qwen3.8-Flash-Next-DSpark)). Inco AI's
  drafter reads a 2,048-token window of target features. The rig's agent sessions run to 65k tokens of context; the
  per-request acceptance of Q561 (0.62–0.94) has not been split by context.
- **Text:** per-token acceptance is far higher on code and structured text than on prose (ik_llama.cpp above;
  [Strata #5125](https://github.com/Niko1221/Strata/issues/5125) measured MTP 72.9 % against DFlash2 61.6 % on a P100
  rig, 6–8 % faster for MTP). On the rig's agent check the long reasoning requests (≥ 800 drafted tokens) accepted
  0.74–0.78 and the short tool-call turns mostly 0.85–0.95 ([per-request file](../../data/2026-11-03_flash_next_dflash2/agent/dflash2b/draft_acceptance.csv)).
  64 % of the output is reasoning, so the long requests carry the session's 0.778.

### 3.4 Cost per drafted token

| Drafter | Cost shape | On the rig (est.) | Published |
|---|---|---|---|
| MTP head | one pass per drafted token, each with the full-vocabulary output layer | 7.7 ms per drafted token; ~7 ms with the head shared (Q635); ~3.7 ms with a 32k-row shortlist (C71) | DSpark card, H100: MTP-7 2.03 ms per token against DSpark 1.75 at 0–8K context |
| DFlash / DFlash2 | one pass for the whole block, flat in the block size | 15–25 ms per step | [LMSYS](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/): a 5-layer DFlash drafting 4, 8 or 16 tokens has lower latency than 1-layer EAGLE-3 drafting 4 |
| DSpark | the parallel pass plus a rank-256 Markov head over the block | as DFlash2, +0.2–1.3 % of a round going from 4 to 16 tokens | [paper](https://arxiv.org/abs/2607.05147) (batch 128) |
| EAGLE-3 | autoregressive, one pass per drafted token | as MTP unless a reduced draft vocabulary is trained | LMSYS: τ equal to DFlash's on Qwen3-4B (4.2) at 2.1× against 3.3× speed-up |
| The verify column | the dominant cost today | ~87 ms per drafted position (Q561) | – |

### 3.5 Reading for the rig

- At three drafted tokens the two drafters tie in τ and in step time (Q561). Beyond three DFlash2's cost stays flat
  while MTP's grows by a pass per token: Section 6.
- **DSpark (Q568) is expected below MTP, not above DFlash2, on accepted length.** Its case is training on agent
  sessions, which the refit in Section 5 would give DFlash2 as well; only its cheaper pass could pay, and only at long
  blocks.
- A per-step length rule (Section 6) picks up the text-type and context effects without a content classifier: the
  drafter's confidence falls on prose and at long context.
- The first measurement is free: bucket the dump's `n_acc` by `pos` (context length) and by whether the step is in a tool
  call.

## 4. Training-free ways to raise acceptance

### 4.1 The verification rule and the draft's sampler

- **Under sample-and-match the draft should be its argmax**, which is what the fork does. A sampled draft token x ~ q
  survives with probability Σ q(x)p(x), the argmax with p(argmax q), and a drafter that is overconfident (the project's
  Flint head: mean top probability 0.542 against a hit rate of 0.454) still has its argmax right more often than a
  sample. Drafter temperature and top-k have nothing to tune in this mode except the confidence cut
  (`--spec-draft-p-min`; nazeshinjite's best was 0.75 against 0.3, 0.0 and 0.9).
- **Rejection raises acceptance where the target is uncertain.** At a position with p_top = 0.7 a calibrated sampled
  draft accepts with probability 1, the argmax with 0.7. llama.cpp merged it as `--spec-draft-sampling probabilistic`
  for the simple draft and MTP ([#27694](https://github.com/ggml-org/llama.cpp/pull/27694), 2026-10-02) and for DFlash2
  ([#30663](https://github.com/ggml-org/llama.cpp/pull/30663), 2026-10-23). The draft sampler takes the request's
  temperature and seed with top-k 10, and the residual is built over the target's candidates after its sampler chain
  (read in the fork's base: `spec_retune` in `common/speculative.cpp`,
  `common_sampler_sample_and_accept_n_rejection` in `common/sampling.cpp`; the server enables it for temperature > 0).
  Measured accepted-length gains:

  | Drafter, target | Hardware | Gain |
  |---|---|---|
  | DFlash2, Qwen3.8-27B, temperature 1.0 | RTX 5090 / DGX Spark / RTX Pro 6000 | +4.3 / +4.6 / +3.9 % |
  | DFlash2, Muse-Glimmer-30B, temperature 1.0 | the same | +11.9 / +3.1 / +7.1 % |
  | MTP, mean over temperatures 0.5–1.0 | RTX 5090 | Qwen3.6-27B +3.4 %, Qwen3.6-35B-A3B +3.3 %, Qwen3.5-9B +8.0 %, Ornith-1.5-9B +2.6 % |
  | simple draft, Llama-3.1-8B with a 3B draft, 0.5–1.0 | RTX 5090 | Q4_K_M +9.8 %, Q8_0 +7.4 %, IQ2_XS +5.9 % |

  Gains rise with temperature and are noise near 0.1–0.3. The rig samples at 1.0 on a noisier 2-bit target, so the
  upper half of the DFlash2 range is plausible (est. +4–10 % τ).
- **The confidence cut has to go.** Under rejection the draft's top probability does not predict acceptance (overlap
  of q and p does); [#31294](https://github.com/ggml-org/llama.cpp/pull/31294) measured, in a revision that skipped the
  cut for sampled drafts, +27 % tokens per pass for DFlash2 and +32 % for MTP at temperature 1.0 against `p_min` 0.75,
  and merged a warning instead. Run `--spec-draft-p-min 0` with probabilistic drafts.
- **Backend sampling is the risk.** The rig's agent check runs `-bs`; #27694's thread has a report of a range-check
  error with probabilistic drafting and backend sampling, unresolved on the page. Smoke it first.
- **Block verification** ([Sun et al.](https://arxiv.org/abs/2403.10444)) decides a drafted block jointly instead of
  token by token; it is lossless, optimal in expected tokens per iteration, never worse than token-wise verification,
  and gains 5–8 % wall-clock over it. The ingredients are those of the rejection function above (q and p at each
  position). Est. +3–6 % τ for DFlash2 on top of rejection: the published gain is for autoregressive chains at
  temperature > 0, and DFlash2's q is top-16-limited with a thin tail.
- **Shared-noise coupling is the other route to the same acceptance** (sample-and-match with draft and target drawing
  from position-keyed noise; [drafter.md section 5](../../drafter.md#5-coupling-with-the-sampler), Q96). Strata's maker
  found no lasting gain for its coupled MTP drafts (Q469, parked); llama.cpp's measured gains for rejection are the
  evidence Q469 waits for, on non-agent text.
- **Drafter top-k and temperature:** q at top-10 against the target's top-20 / top-p 0.95 chain loses the target mass
  beyond q's tenth token (est. ≤ 1–2 %), and the DFlash2 selector keeps 16 candidates. A per-drafter logit temperature
  fitted offline to maximize Σ min(p, q_τ) is a calibration, est. +1–3 % τ; drafters are overconfident, so τ > 1 is the
  likely direction under rejection (and a sharper draft under match).
- **A lower sampling temperature raises acceptance, and changes the sampler.** DFlash's paper: τ 5.69 at temperature 1
  against 6.54 greedy (Qwen3-4B); the model card recommends temperature 1.0 for thinking mode and gives coding no
  separate setting. Est. +4–8 % τ at 0.6–0.7; the gate is the agent check's pass rate, not acceptance.

### 4.2 Relaxed acceptance (lossy)

- **Typical acceptance** (Medusa): accept a candidate whose target probability exceeds min(ε, δ·exp(−entropy)); the
  authors state they do not insist on the target's distribution. Judge decoding
  ([arXiv 2501.19309](https://arxiv.org/abs/2501.19309)) trains a judge head on the target's embeddings to accept
  correct tokens that differ from the target's choice (up to 9× on 8B/405B); also lossy.
- **TensorRT-LLM's relaxed acceptance for MTP** ([blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html)):
  in the thinking phase only, accept a draft token if it is among the top 10 and within delta of the top
  probability. At MTP depth 3: acceptance 2.82 strict, 3.06 (delta 0.5) and 3.10 (delta 0.6); speed-up 1.08× and 1.09×.
  Accuracy moved both ways: MMLU-Pro 84.0 → 81.2, GPQA Diamond 71.0 → 69.2, LiveCodeBench 62.0 → 66.0 (315 problems),
  AIME 2024 68.0 → 74.0 (30 problems).
- **Reading:** +8–10 % τ for MTP, est. similar for DFlash2, in exchange for output that no longer follows the sampler.
  The project's gate is exact output (Q691 is parked on the same ground); if tried, thinking-phase only, with the agent
  check's pass rate and a KLD gate. Lossless relaxations are the ones in 4.1 and the multi-draft methods (Q731, Q614).

### 4.3 Two drafters

- **No source combines two learned drafters losslessly.** The published combinations choose between drafters per
  sequence or per step: Snowflake's [Arctic Inference](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic)
  runs suffix decoding first and falls back to the draft model when its score is below the speculation budget, and on
  SWE-Bench reaches 302 output tok/s against 286 (suffix only), 123 (draft model only) and 75.8 (no speculation);
  MetaSD ([arXiv 2604.05417](https://arxiv.org/abs/2604.05417)) and HedgeSpec
  ([arXiv 2510.20064](https://arxiv.org/abs/2510.20064)) pick drafters with bandits. A union of candidates is a token
  tree, which buys est. 0.01–0.03 accepted tokens per node against a threshold of 0.15–0.25 on the rig.
- **The rig has the copy + DFlash2 version already:** +5–6 % today, +13–15 % after the kernels (Q617, Q733).
- **MTP + DFlash2 agreement as confidence:** no primary source. Cost: a second draft (+7–8 ms of a ~140 ms step, 5 %), the
  MTP head's 1.9–2.7 GB on a card, and a second rollback state. Gain: a sharper length rule than DFlash2's own
  probabilities, est. ≤ 1–2 % net. Parked.

### 4.4 The target's previous-step top-k

Token Recycling ([arXiv 2408.08696](https://arxiv.org/abs/2408.08696), ACL 2025) keeps, for each token, the top-k
successors seen at earlier decoding steps (an adjacency matrix under 2 MB), builds a tree from it and updates it from
every verify step; it reports ~2× speed-up, 30 % above other training-free drafters. ToolSpec
([arXiv 2604.13519](https://arxiv.org/abs/2604.13519)) uses it as its general drafting method. On the rig the verify
already computes every row's logits and the dump records the target's top 8 per sampled position, so the hit rate can be
replayed offline at no cost; on agent text the copy lookups it competes with are capped at +5–6 % (Q617). Est. +1–3 %
(Q275).

### 4.5 Drafting only inside tool-call JSON or code

- **ToolSpec** drafts the schema's deterministic tokens with a finite-state machine and fills variable fields from
  retrieved earlier calls: 3.5–4.2× on four tool-calling benchmarks (PLD 1.8–2.1×, EAGLE-2/3 1.56–2.12×), lossless,
  no agentic-coding benchmark, no learned drafter in the mix.
- **On the rig** the tool-call bodies are 23 % of the generated tokens (Q594), 49 % of arguments are copied from the
  context (Q617), and a learned drafter already accepts short tool-call turns at 0.85–0.95. Appending the grammar's
  forced tokens (keys and punctuation where one token is allowed) is est. +1–2 %, not a new multiplier.
- **Content gating is what the length rule does:** ik_llama.cpp's −7 % on prose and nazeshinjite's −1 % disappear when
  the draft shrinks where confidence falls. Q687 covers it.

## 5. Cheap adaptation to the target's own agent output

| Work | Student | Data | Cost | Gain |
|---|---|---|---|---|
| [FastMTP](https://arxiv.org/abs/2509.18362) | MTP head of MiMo-7B-RL, one shared layer applied recurrently | 389.4 K samples the target wrote (temperature 0.6, top-k 20, top-p 0.95; 42 % general, 18 % math, 13 % code, 27 % Chinese) | < 1 day, one H20 server | τ at 3 drafted: 1.83 → 2.54 (fixed data) → 2.73 (self-distilled); first / second / third position 70 / 10 / 0 % → 80 / 56 / 36 %. The base head had been trained for one step |
| [DSpark](https://huggingface.co/shadowpa0327/Qwen3.8-Flash-Next-DSpark) | 0.66 B, 5 layers, block 7 | stage 3: 95,596 Open-SWE-Traces sessions the target wrote in thinking mode (2.9 B tokens, 1,800 steps of 16 × ≤ 32K), after chat stages | 4 × H100, target served beside the trainer; hours not stated | held-out agent sessions τ 2.69 (chat stage) → 3.70 (+38 %); chat, math, code 4.75 → 4.55 (−4 %) |
| [hotocooo](https://huggingface.co/hotocooo/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-DFlash) | Inco AI's DFlash2 | 2.9 M rows captured from an IQ3_S target | 10,000 steps | τ 2.82 → 2.98 (train), 3.03 (held out): +5–7 % |
| [DFlash](https://arxiv.org/abs/2602.06036) | 3–8-layer drafter | ~800 K samples regenerated by the target, 6 epochs, 512 anchors per sequence, cached target features | not reported | the recipe; τ falls 13 % at temperature 1 |
| [Domain draft models](https://arxiv.org/abs/2503.07807) | small drafters | historical queries, curated, synthetic | – | offline distillation beats online by 11–25 %; logit-level (white-box) beats text-only by 2–10 %; synthetic alignment data reaches 80–93 % of real queries |
| [LK losses](https://nebius.com/blog/posts/lk-losses) | EAGLE-3, Medusa, MLP speculator | – | – | acceptance-matched loss, under rejection sampling: +3.9 % (EAGLE-3, Llama-3.1-8B), +7.7–8.2 % (gpt-oss-120b, Qwen3-235B-A22B), +5.6 % on a DeepSeek-V3 MTP fine-tune |

**For the rig (est.):**
- **Student:** DFlash2 first: 0.6B, shares the target's embeddings and head, so a refit changes no memory. The MTP head
  second: FastMTP's +49 % came from a head trained for one step, and Qwen3.8's head is trained for several, so est. +3–8 %
  carries over, on a 1.9–2.7 GB head the 4 GB host only just holds.
- **Data:** sessions written by the rig's own file (Q2_0) on publishable prompts, agent-shaped (tool calls, a growing
  context, thinking), the labels and hidden states from the same file. That is DSpark's stage 3 and hotocooo's recipe, and
  consistent with [docs/drafter.md](../../drafter.md): the teacher is the file that verifies, a different generator costs
  est. 1–3 %, ≤ ~4 epochs per data size, ~6 % gained per doubling of data.
- **Objective follows the verification rule:** sample-and-match with greedy drafts rewards top-1 agreement with the
  target's samples, which is cross-entropy; rejection rewards overlap, which is the TV/LK family (+4–8 % in the
  table). Decide the verification mode (4.1) before fitting.
- **Cost (est.):** generation of 10–20 M tokens at 300–600 tok/s aggregate on the quant machine, 5–19 hours; a dump of the
  5 tapped layers × 2,560 × 2 bytes = 25.6 KB per token (12.8 KB at FP8), 0.13–0.5 TB for 10–20 M tokens at ~1–2 k tok/s
  (docs/drafter.md's measured dump rates), 1.5–6 hours; training 6·0.6 B·20 M ≈ 7 × 10^16 FLOPs, ~1 hour per epoch on one
  RTX 3090 at ~20 TFLOPs effective, ≤ 4 epochs. The whole loop is ~1–2 days of quant-machine time. The drafter, its checkpoints and its data card get published
  (project rule); the target's outputs fall under the Swift and Qwen community licences, and the prompt sets need their
  licences checked first (Open-SWE-Traces' is unchecked).
- **Gain: est. +5–15 % τ** at 2–3 drafted tokens (hotocooo +5–7 % on mixed data at IQ3_S; DSpark's +38 % shows what agent
  data does for a drafter that lacked it), more at positions 4–7 where the tail decides the draft length.

## 6. DFlash2's block width and MTP's depth when a k-token pass is cheap

### 6.1 The model

S(k) = τ(k)·T1 / (D + T(k+1)), T(m) = T1·(1 + r·(m − 1)), τ from Section 1's rig-anchored chain, D = 20 ms for DFlash2
(flat in k) and 7.7·k ms for MTP, T1 = 127 ms today (the in-session figure) and 90 ms after the kernels:

| k | τ | DFlash2 today | DFlash2, r 0.075 | DFlash2, r 0.10 | MTP, r 0.075 | MTP, r 0.10 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.86 | 1.15 | 1.43 | 1.41 | 1.60 | 1.57 |
| 2 | 2.56 | **1.23** | 1.87 | 1.80 | 1.94 | 1.87 |
| 3 | 3.10 | 1.21 | 2.14 | 2.04 | 2.09 | 1.99 |
| 4 | 3.49 | 1.17 | 2.29 | 2.15 | **2.13** | **2.00** |
| 5 | 3.77 | 1.10 | 2.36 | **2.19** | 2.09 | 1.96 |
| 6 | 3.97 | 1.03 | **2.37** | 2.18 | 2.02 | 1.88 |
| 7 | 4.12 | 0.96 | 2.36 | 2.14 | 1.94 | 1.79 |

Today's column reproduces the measured 1.22× at k = 2 and the flat top at 2–3; T(6)…T(8) extend Q583's +55 ms per
column. Both P_j and D are est. (Section 1).

### 6.2 What it says

- **Rule:** drafted position j pays when P_j > (r + m_j)·S, where m_j is the draft's marginal cost per position as a
  fraction of T1 (≈ 0 for DFlash2 inside its block, 0.04–0.09 for MTP). Today r·S ≈ 0.5, so only the first two
  positions pay (P_3 = 0.54 is break-even). After the kernels r·S ≈ 0.17–0.23 for DFlash2, so positions up to 5–6 pay
  (P_5 = 0.28, P_6 = 0.20) and 7 does not (0.145); MTP's threshold is 0.26–0.38, so it stops at 4.
- **DFlash2: draft 5–6 tokens (verify width 6–7), not the block's 7.** The surface is flat from 5 to 7 (within 2.5 %);
  D ± 10 ms moves S by ±7 % and not the best k. A drafter trained at block 16 would add nothing the rig can use:
  P_j beyond 6 is below 0.2. DFlash's own ablation says a drafter trained at a larger block runs well at smaller ones and
  not the reverse (block 16 trained / 8 tested: τ 5.09 on Math500 against 5.21 trained at 8), so Inco AI's block 8 is
  enough. GPU practice agrees: nazeshinjite's speed on Metal is flat from `n_max` 5 to 8 (74.5 → 76.2 tok/s).
- **MTP: depth 3–4**, and 4–5 (S ≈ 2.4) if each pass costs ~3.5–3.7 ms, which the shared, shortlisted head (Q564, Q635,
  C71) is aimed at. With the same P_j, MTP with a cheap head matches DFlash2 (2.40 against 2.37), so the choice between
  them comes down to which drafter's P_j is higher at positions 4–6: measure.
- **If the tail collapses like the INT4 port's** (0.76 / 0.45 / 0.31 / 0.14 / 0.10 / 0.02 / 0.00), the best k is 3–4 at
  S ≈ 1.75 (r 0.075), and a longer block buys nothing.
- **Verify widths for the graph cache** (Q687 uses 1 / 3 / 5 / 9 / 17): DFlash2's useful widths are 1, 3, 6 and 8 (k = 0,
  2, 5, 7). A set of 1 / 3 / 5 / 8 (k = 0, 2, 4, 7) loses ≤ 3 % against exact k (S 2.29 against 2.36 at k = 4 against 5).

### 6.3 A confidence that decides the length

- DSpark's scheduler ([paper](https://arxiv.org/abs/2607.05147)) is the closest published form of Q687: a confidence
  head c_k = σ(wᵀ[h_k; embedding of the previous drafted token]) trained on 1 − TV, calibrated per position by sequential
  temperature scaling (ECE 3–8 % raw, ~1 % calibrated; ROC-AUC 0.81–0.90), prefix survival a_j = Π c_i, positions
  admitted greedily while τ·SPS(B) rises, with the step rate SPS profiled at start-up. Moderate load gives 4–6 tokens a
  request against 2 for MTP-1, and +51–52 % aggregate throughput over MTP-1 in a DeepSeek-V4 serving system (the
  baseline also differs in drafter, so the scheduler's own share is not isolated).
- SpecDec++ ([arXiv 2405.19715](https://arxiv.org/abs/2405.19715)): a trained acceptance head and a stop threshold gave
  +7.2 % (Alpaca), +9.4 % (GSM8K), +11.1 % (HumanEval) over baseline speculative decoding on Llama-2-chat 7B/70B. With
  Q687's est. +5–12 % these are the same range.
- **Exactness.** DSpark's appendix shows that a scheduler that looks at later drafted tokens before deciding on an earlier
  one biases the output (its example gives 0.85 / 0.15 for a target of 0.70 / 0.30). The fork's rule (`LLAMA_SPEC_ADAPT`)
  keeps the longest prefix whose path probabilities, products of the drafter's probabilities of the drafted tokens, pay for
  the verify. Under sample-and-match this is exact whatever the rule reads: the target samples every emitted token. Under
  `--spec-draft-sampling probabilistic` it is exact only if the decision to keep a token does not depend on that token's
  own sampled value: use the max probability or the selector's conditional rows at the position (a function of the prefix),
  not the sampled token's probability. `--spec-draft-p-min` already works that way.
- **Calibration is the first thing to measure:** the selector's path probabilities against realized acceptance, by
  position and context bucket, on the dump (Q688). DFlash2's selector rows are the analogue of DSpark's c_k.

## 7. Ideas for the rig

Ranked by est. generation tok/s gained. B0 is today's DFlash2 line (9.5 tok/s); B1 is est. after Q591 and the dense
fixes: ~11 tok/s without a draft and ~24–26 with DFlash2 at 5–6 drafted tokens. Percentages are of τ (≈ of speed at a
fixed k) on B1, 1 % ≈ 0.25 tok/s. "No row" means the registry has none yet. Lossy or sampler-changing ideas are marked.

| # | What to do | Est. effect | Effort | Risk | First measurement | Registry |
|---:|---|---|---|---|---|---|
| 1 | Time Q591's kernel halves to k = 8, then let DFlash2 draft 5–6 tokens (verify width 6–7) instead of 2 | S 1.22× → est. 2.2–2.4×; 9.5 → est. 24–26 tok/s (the faster baseline included) | kernels written; ~1 day of rig time | the tail of P_j collapses (S then ~1.75×) or r stays at 0.10 (S 2.19×) | T(1…8) of both kernel halves on one P106; `LLAMA_SPEC_DUMP_FULL_BLOCK=1` dump for P_1…P_7 on agent transcripts (quant machine) | Q591, Q605, Q688 |
| 2 | Refit DFlash2 on sessions the rig's own Q2_0 writes, agent-shaped, labels and features from the same file | est. +5–15 % τ at 2–3 drafted, more at 4–7: +1.3–3.8 tok/s | ~1–2 days of quant-machine time, 4–6 days of work | licence of prompts and of Swift's outputs; overfit past ~4 epochs; the verification rule changes the best loss | held-out agent transcripts, acceptance by position, refit against original drafter at the same anchors (dump replay, no rig) | Q572 (parked; reopens when Q591's kernels are timed) |
| 3 | Q687's per-step length from a calibrated, causal confidence (selector rows) | est. +5–12 % over the best fixed length: +1.3–3.0 tok/s | built; rig run next | under rejection the rule must not read the sampled token's own probability (6.3) | selector path probability against realized acceptance by position and context bucket; ROC-AUC, ECE (target ≥ 0.8, ≤ 3 %) | Q687 (C134, Q760–Q764) |
| 4 | `--spec-draft-sampling probabilistic --spec-draft-p-min 0` on the S18 line | est. +4–10 % τ at temperature 1.0 (DFlash2 +3.1–11.9 % measured elsewhere): +1.0–2.5 tok/s | a flag; one smoke, one agent-check arm | `-bs` range-check report; unit check of exactness | smoke with `-bs` on the rig; offline Σ min(p, q) against p(x_d) on the dump's steps | Q731 (offline), Q469 (parked; its reopening evidence is llama.cpp's gains) |
| 5 | Relaxed acceptance in the thinking phase (top-10, delta 0.5–0.6) — **lossy, needs the owner's OK** | est. +6–10 % τ: +1.5–2.5 tok/s | ~1 day | output no longer follows the sampler; quality moves ±3 points on benchmarks | KLD of relaxed against strict output on Q216's bases; agent-check pass rate | no row (add as parked) |
| 6 | Sample the agent at 0.6–0.7 instead of 1.0 — **changes the sampler** | est. +4–8 % τ: +1.0–2.0 tok/s | a flag; agent-check arms | the card recommends 1.0 for thinking; repetition loops; pass rate unknown | agent-check pass rate at 1.0 / 0.7 / 0.6, acceptance per arm | no row |
| 7 | Block verification (lossless joint verification of the drafted block) on top of 4 | est. +3–6 % τ: +0.8–1.5 tok/s | 2–3 days: one function in `common/sampling.cpp`, a chi-square exactness test | low–medium; DFlash2's q is top-16-limited | replay on the dump: expected tokens per step, token-wise rejection against block verification, on the same drafted blocks | no row |
| 8 | Pick the file for acceptance too: replay the same transcripts through Q2_0, IQ3_S, Q8_0 with one drafter | est. +3–7 % τ if a ≤ 54 GB file at KLD ~0.12 exists; decides close calls between retypes | ~0.5 day on the quant machine | none | acceptance by position per file at one verify width (replay, quant machine) | Q217, Q218, Q219, Q221 for the files; acceptance replay: no row |
| 9 | Fit the MTP head (not DFlash2) on the same sessions; or requantize its experts (Q705) | est. +3–8 % τ on the MTP line only (0 on DFlash2); Q4_K_M costs 1.7 points, Q2_0 est. 3–6 points (−5–10 % τ) | 2–3 days | the head must fit the 4 GB host; requantizing loses acceptance | acceptance of the Gyro head at Q4_K_M, IQ4_NL and Q2_0 experts on the agent check's text | Q705; head fit: no row |
| 10 | Token Recycling candidates from the verify's own top-k as a second source | est. +1–3 %: +0.25–0.75 tok/s | 1 day offline, 2–3 in the engine | overlaps the copy lookups (Q617) | hit rate on the dump's target top-8 | Q275 |
| 11 | Fit a per-drafter logit temperature and top-k for rejection offline | est. +1–3 % τ | 0.5 day on the dump | none | Σ min(p, q_τ) over τ, top-k 5 / 10 / 16 | no row (belongs beside Q731) |
| 12 | Context-length effect: acceptance by context bucket from the dump; a wider tap window or a switch to MTP at long context only if the decay matches DSpark's −44 % | est. 0–4 % | 0.5 day | a second drafter costs host memory | `n_acc` by `pos` buckets of the existing and the new dumps | no row |
| 13 | Append the tool-call grammar's forced tokens to the draft | est. +1–2 % | 1–2 days | needs the lazy grammar's state in the draft path (Q594's neighbour) | share of tool-call tokens where the grammar allows one token | no row |
| 14 | DSpark converted for `draft-dspark` (Q568) | est. −5…+3 % against DFlash2: below MTP-7 on its own agent set, cheaper per pass | 1–2 days | the card needs a patched SGLang for block 7 | convert; per-position P_j on the dump next to DFlash2's | Q568 (lower it) |
| 15 | MTP + DFlash2 agreement or union | est. ≤ +1–2 % net of a 5 % step cost | 3+ days | two drafters resident on the 4 GB host | cross-tab of DFlash2 and MTP top-1 against the target's sample on the dump | no row (parked); Q687's optional agreement |

Ideas 1–4 compose: 1 sets the price of a column, 3 chooses how many columns each step buys, 4 raises the acceptance of
each column where the target is uncertain, and 2 raises what the deep positions give, which sets how far 1 and 3 can
go. Idea 2 should wait for 4's decision on the verification mode, since the loss it trains depends on it.

## Sources

- Speculative sampling and verification: [Leviathan et al.](https://arxiv.org/abs/2211.17192);
  [block verification](https://arxiv.org/abs/2403.10444); [Medusa's typical acceptance](https://arxiv.org/abs/2401.10774);
  [Judge decoding](https://arxiv.org/abs/2501.19309);
  [TensorRT-LLM relaxed acceptance](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html).
- Drafters: [DFlash](https://arxiv.org/abs/2602.06036);
  [DSpark](https://arxiv.org/abs/2607.05147); [FastMTP](https://arxiv.org/abs/2509.18362);
  [LK losses](https://nebius.com/blog/posts/lk-losses); [acceptance-aware training](https://arxiv.org/abs/2609.24150);
  [domain draft models](https://arxiv.org/abs/2503.07807); [SpecDec++](https://arxiv.org/abs/2405.19715);
  [LMSYS on DFlash and Spec V2](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/).
- Training-free drafting: [Token Recycling](https://arxiv.org/abs/2408.08696);
  [ToolSpec](https://arxiv.org/abs/2604.13519); [SuffixDecoding](https://neurips.cc/virtual/2025/poster/115453) and
  [Snowflake's hybrid](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic);
  [MetaSD](https://arxiv.org/abs/2604.05417); [HedgeSpec](https://arxiv.org/abs/2510.20064).
- Model cards: [Inco AI DFlash2, Flash-Next](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2);
  [Qwen3.8-27B DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2);
  [DSpark, Flash-Next](https://huggingface.co/shadowpa0327/Qwen3.8-Flash-Next-DSpark);
  [DSpark port, Qwen3.8-27B](https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM);
  [Unsloth MTP heads](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md);
  [nazeshinjite's GGUFs](https://huggingface.co/nazeshinjite/Qwen3.8-Flash-Next-DFlash2-GGUF);
  [hotocooo's refit](https://huggingface.co/hotocooo/Qwen3.8-Flash-Next-Abliterated-Uncensored-Turbo-Brilliance-50x-Reasoning-Instruct-GSQ-RCO-DFlash);
  [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next);
  [Red Hat's Qwen3-32B speculator thread](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3/discussions/2).
- llama.cpp and other engines: pull requests [#27694](https://github.com/ggml-org/llama.cpp/pull/27694),
  [#30663](https://github.com/ggml-org/llama.cpp/pull/30663), [#31294](https://github.com/ggml-org/llama.cpp/pull/31294),
  [#30927](https://github.com/ggml-org/llama.cpp/pull/30927); issue [#25618](https://github.com/ggml-org/llama.cpp/issues/25618);
  [ik_llama.cpp #2761](https://github.com/ikawrakow/ik_llama.cpp/pull/2761);
  [Strata #5125](https://github.com/Niko1221/Strata/issues/5125).
- The project's data: [Q561](../../data/2026-11-03_flash_next_dflash2/README.md),
  [Q226](../../data/2026-11-03_flash_next_mtp/README.md), [quality chain](../../data/2026-11-03_flash_next_quality/README.md),
  [copy spans](../../data/2026-11-04_copy_spans/README.md), [tree replay](../../../bench/tree_replay/README.md),
  [registry](../../../quants/experiments.md).
