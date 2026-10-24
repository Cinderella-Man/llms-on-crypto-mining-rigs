# Model branches: one Pascal branch or one branch per model, and what each speed-up needs for Qwen, gpt-oss, GLM, Ornith and dense models (2026-10-24)

Scope: the owner has decided that the project stays Pascal-specific (Q144) and asks whether the Pascal llama.cpp fork
should carry branches tuned to single models (a Flint branch, a Qwen branch, …) or one branch whose speed-ups apply to
every model. This note:
- takes each candidate model (Qwen3.6-35B-A3B and its family, Qwen3-30B-A3B, gpt-oss-20b/120b, GLM-4.5-Air and
  GLM-4.7-Flash, Ornith-1.5-122B, a dense control) and each of the project's speed-ups, and says per pair whether it
  applies unchanged, needs a change (what, and how much work) or cannot apply, with the expected gain on the rig;
- answers what fits on nine 6 GB cards with a 4 GB host, in which quant and at what context;
- compares three branch strategies against how llama.cpp itself handles per-architecture kernels, and recommends one
  with a concrete layout;
- lists the work that would make the next two or three models fast on the rig, and the rig measurements to run first.

It builds on [fork scope](fork_scope.md), which classifies every fork change as generic, Pascal, MoE or Flint with
file:line evidence; that classification is not repeated here.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it. Speeds without it are
  measurements from the [registry](../../../quants/experiments.md) (Qn) or the [overview](../../overview.md).
- Code citations are `branch:path:line` in the fork's worktrees on 2026-10-24: `v2` = `pascal-p106-v2` (`032b130f4`),
  `comb` = `pascal-p106-v2-combined` (`b3052a4f8`), `pers` = `pascal-p106-v2-persistent` (`2c3ba4a2c`), `mask2` =
  `pascal-p106-v2-gpumask2` (`6f943f792`), `pipe2` = `pascal-p106-v2-pipespec2` (`74f7c7182`), `q6k` =
  `pascal-p106-v2-q6k` (`03b92cf10`). `ggml-cuda/` stands for `ggml/src/ggml-cuda/`. Upstream code is read from these
  trees (upstream base `d77256999`).
- Type mixes marked "read" come from the GGUF headers of local copies, with
  [scripts/token_cost.py](../../../scripts/token_cost.py) (pure Python, header only): bytes read per generated token,
  experts counted at top-k, the embedding row left out. Architecture numbers come from the GGUF headers, llama.cpp's
  model code and the configs cited in [small MoE models](models_small_moe.md) and
  [large MoE models](models_large_moe_and_compression.md).
- Speed estimates use one model calibrated on Qwen3.6 Q4_0 (Q38: 22.79 tok/s kernels off, 29.04 on, 4 cards): time per
  token = Σ bytes per type ÷ in-model rate + fixed time. Rates: Pascal kernels 120–140 GB/s (Q4_0, Q8_0, Q5_K; the
  microbenchmark's 120–150), llama.cpp's own kernels with 2 warps 50–62 GB/s for Q4_K/Q6_K and 60–75 for MXFP4/IQ4 types
  ([K-quant decode](kquant_decode_pascal.md) §3), ~45 GB/s for IQ2/Q2_K (est.). Fixed time: ~14 ms for Qwen3.6 (what
  Q38 leaves after weight reading: 30 Gated DeltaNet layers add small kernels), ~12 ms for a 47–48-layer
  attention-only MoE (Flint's v2 token minus its weight time), ~3 ms for a 7 B dense model.
- Nothing ran on the rig or the quant machine for this note.

## Verdict

- **One Pascal branch, not one branch per model.**
  - Every speed-up that matters for other models is keyed to the GPU (compute 6.1) and the tensor type, not to the
    model. The one model-shaped kernel, the persistent MoE layer, already checks the shapes and falls back to the
    per-op path on anything else (`pers:ggml-cuda/moe-persistent.cu:961-998`), so a model never needs its own build to
    be safe.
  - llama.cpp itself keeps per-model code in one tree and picks it at run time from the model's shapes: 167
    architecture files in `src/models/`, the fused top-k router instantiated per expert count including
    `case 288: // StepFun 3.7` (`pers:ggml-cuda/topk-moe.cu:401, 475`), flash-attention tile kernels per head size
    including 576/512 for DeepSeek-style MLA (`pers:ggml-cuda/fattn-tile.cu:48-54`), MMVQ launch tables per GPU family
    (`pers:ggml-cuda/mmvq.cu:107-124`).
  - Per-model branches would multiply rebases on upstream's busiest files and the rig validation by the number of
    models, and make users pick a binary per model, for no runtime benefit.
- **Recommendation: option (c) with option (a)'s gating.** One code line (`pascal`); model-specific kernels live in it
  as shape-class template instantiations chosen from the tensors at run time, never by branch; per-model profiles
  (recommended file, cards, context, command line, switches, screen file, measured speed) are files in this
  repository (§4.3).
- **What transfers to Qwen-class MoE unchanged:**
  - CUDA graphs on 6.1 (Qwen3.6 Q4_0 18.5 → 19.6 tok/s on the rig), the 2-warp launch shape, and the Pascal kernels for
    Q4_0/Q4_1/Q5_0/Q8_0/Q5_K/IQ4_XS (Qwen3.6 Q4_0 +27 %, Q38);
  - PR #31047 for prompts (est. +20–30 % for 256 experts at top-8, as Flint's +22–29 %), the graph-reuse fix;
  - the multi-token kernels for MTP verify steps (Qwen3.6's MTP was held at 1.04× by the stock ones, Q43);
  - GPU sampling for tool requests (Q67), worth more on Qwen3.6 than on Flint: its 248,320-token logits row is 993 KB,
    est. 5–7 ms per tool-request token over PCIe 1.0 x1;
  - for attention-only models (Qwen3-30B-A3B, gpt-oss, GLM, dense Llama-class) also pipelined speculation (Flint
    1.30–1.55×, Q87) and the on-card mask.
- **What needs a change:**
  - **Formats, the largest lever.** The common downloads are Q4_K_M: 96 % of a qwen35moe Q4_K_M token is Q4_K and Q6_K
    (read), types that the v2 kernels don't cover at all. Even bartowski's Qwen3.6 Q4_0 reads 20 % of its bytes in a
    Q6_K head (read). Landing the built Q6_K kernel (Q101) gives est. +12–18 % on Qwen3.6 Q4_0; a new Pascal Q4_K kernel
    with it gives est. 1.5–1.7× on the Q4_K_M files of every model, dense ones included.
  - **Hybrid recurrent models** (the Qwen3.5/3.6, Ornith and Qwen3.8 line): the on-card mask needs the hybrid input wired
    (est. 0.5–1 day); pipelined speculation needs a ring of recurrent states, one per in-flight token (est. 3–5 days).
    Upstream's rollback snapshots only cover the tokens of one verify batch (`pipe2:src/llama-memory-recurrent.cpp:200-218`).
  - **gpt-oss:** an MXFP4 kernel (est. 3–5 days, est. +15–25 %) and a launch shape for rows of 2,880.
  - **The persistent MoE kernel:** one shape-class template row per model family (hidden size, experts, top-k, widths,
    router kind, shared-expert kind) plus four-type combinations; est. +3–5 % per model, worth doing only after Flint's
    rig timing (Q88).
- **What adds nothing for other models:**
  - the FP32 sums' speed gain: upstream rescales only Mistral 4, Mellum and Flint (`v2:src/llama-graph.cpp:2528`), and
    Qwen files decode correctly with FP16 block sums (Q38, Q82);
  - the norm + Q8_1 fusion: no measurable gain even on Flint (Q117);
  - Q4_1_64 and the Hadamard line: Flint files only.
- **Fit on 54 GB:**
  - Qwen3.6, Qwen3-30B-A3B, gpt-oss-20b and GLM-4.7-Flash fit at Q4 to Q8 with long context.
  - GLM-4.5-Air and Ornith-1.5-122B fit only at 2–3 bits, at est. 8–16 tok/s.
  - gpt-oss-120b (62.6–65.4 GB in every file) doesn't fit.
  - A dense 7–8 B fits on one card.
- **Top work for other models, in order:** a rig model matrix with no new code; land Q6_K; a Pascal Q4_K kernel; GPU
  sampling for tools; hybrid memory in the mask and the executor; MXFP4 with K-aware launch shapes; persistent-kernel
  shape classes (§5).

## 1. The models and their files

### 1.1 Architectures

| Model (llama.cpp arch) | Layers, hidden | Experts / top-k / router | Shared expert | Attention | Vocabulary | Source |
|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (`qwen35moe`); same shapes: Qwen3.5-35B-A3B, Ornith-1.5-35B-A3B, Qwen3.8-35B-A3B distillation, KAT-Coder, AgentWorld | 40 (+1 MTP layer), 2,048 | 256 × 512 / 8 / softmax, renormalized | 1 × 512, output scaled by sigmoid(w·x) | 10 full-attention layers (16 q / 2 KV heads, head 256) + 30 Gated DeltaNet layers | 248,320 | GGUF headers (read); `v2:src/models/qwen35moe.cpp:493-506, 510-535` |
| Qwen3-30B-A3B (`qwen3moe`); same: Coder-30B, Instruct/Thinking-2507, Tongyi-DeepResearch | 48, 2,048 | 128 × 768 / 8 / softmax, renormalized | none | all full GQA, 32 q / 4 KV, head 128 | 151,936 | `v2:src/models/qwen3moe.cpp:137-149`; config |
| gpt-oss-20b (`openai-moe`) | 24, 2,880 | 32 × 2,880 / 4 / softmax over the selected 4; expert biases; SwiGLU-OAI with clamp | none | 12 sliding-128 + 12 full, 64 q / 8 KV, head 64, attention sinks | 201,088 | `v2:src/models/openai-moe.cpp:132-142`; config |
| gpt-oss-120b | 36, 2,880 | 128 / 4 / as 20b | none | as 20b | 201,088 | config |
| GLM-4.5-Air (`glm4moe`) | 46 (+1 MTP), 4,096; layer 0 dense | 128 × 1,408 / 8 / sigmoid + selection bias, renormalized | 1 × 1,408 | full GQA, 96 q / 8 KV, head 128 | 151,552 | `v2:src/models/glm4-moe.cpp:384-394`; conversion `v2:conversion/glm.py:171-180` |
| GLM-4.7-Flash (`deepseek2`, MLA) | 47, 2,048; layer 0 dense | 64 × 1,536 / 4 / DeepSeek-style (gating function from the GGUF) | 1 × 1,536 | MLA, kv_lora 512, head sizes 576/512 | — | `v2:conversion/glm.py:249-250`; `v2:src/models/deepseek2.cpp:22-30` |
| Ornith-1.5-122B (`qwen35moe`) | 48, 3,072 | 256 × 1,024 / 8 / softmax, renormalized | 1 × 1,024, sigmoid-gated | 12 gated attention (32 q / 2 KV, head 256) + 36 Gated DeltaNet | 248,320 (est., Qwen3.5 tokenizer) | config in [large MoE models](models_large_moe_and_compression.md) |
| Dense controls: Llama-2-7B (`llama`, the llama.cpp CUDA scoreboard's model); Qwen3.8-27B (`qwen35`, hybrid) | 32, 4,096; 64, 5,120 | — | — | Llama: full MHA; Qwen3.8-27B: 16 full + 48 Gated DeltaNet | 32,000; 248,320 | GGUF headers (read) |

Flint, for comparison: 47 MoE layers, hidden 2,048, 256 × 512, top-6, sigmoid with renormalization, a plain shared
expert of 1,024, 35 of 49 layers with a 513-token window.

### 1.2 What the popular files are made of, and what the Pascal kernels cover

Share of the bytes read per generated token, by type. "v2" covers Q4_0, Q4_1, Q5_0, Q8_0, Q5_K and IQ4_XS
(`pers:ggml-cuda/mmvq-pascal.cu:650`); "q6k" adds Q6_K (`q6k:ggml-cuda/mmvq-pascal.cu:63`); "Q4_K" adds the Q4_K
kernel of the fork's branch `pascal-p106-v2-kquant` (Q149, [kernel design](pascal_mmvq_integration.md) section 18).

| File | GB per token | Types | v2 | + q6k | + Q4_K |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B Q4_0, bartowski (read) | 2.08 | Q4_0 55 %, Q6_K 23 % (head 20 %), Q8_0 14 %, F32 5 % (router), Q5_0 2 %, Q4_1 1 % | 72 % | 95 % | 95 % |
| qwen35moe-35B Q4_K_M: Ornith-1.5-35B and the Qwen3.8-35B-A3B distillation (both read) | 2.02 | Q4_K 57 %, Q6_K 39 %, F32 4.5 % | 0 % | 39 % | 96 % |
| Qwen3.6-35B-A3B Q4_K_M, ggml-org (read): routed experts Q4_K, every other matrix Q8_0 | 2.57 | Q8_0 58 %, Q4_K 22 %, Q6_K 16 % (head), F32 3.5 % | 58 % | 74 % | 96 % |
| Qwen3.6-35B-A3B UD-Q4_K_XL, Unsloth | — | not read | — | — | — |
| Qwen3-30B-A3B Q4_K_M (est., llama-quant's Q4_K_M rules on the config) | ~1.9 | Q4_K ~70 %, Q6_K ~27 % (head, half the expert downs, half of attn_v), F32 ~3 % | 0 % | ~27 % | ~97 % |
| Qwen3-30B-A3B Q4_0 (est.) | ~1.85 | Q4_0 ~83 %, Q6_K ~14 % (head), F32 ~3 % | ~83 % | ~97 % | ~97 % |
| gpt-oss-20b, any file (est.: 11.5–12.1 GB files keep MXFP4 experts; the F16 file is 1.68 GB larger, which matches attention, head and embedding in Q8_0) | ~2.57 | MXFP4 ~49 % (experts), Q8_0 ~51 % (attention 26 %, head 24 %) | ~51 % (rows of 2,880: see §2.3) | same | same |
| GLM-4.5-Air UD-IQ2_XXS (est.) | ~5.9–6.7 | IQ2_XXS expert gate/up; expert down IQ4_NL, because 1,408 is not a multiple of 256 and llama-quant falls back (`pers:src/llama-quant.cpp:409-430`); attention (5 B parameters per token) and shared expert types not read | low | | |
| Ornith-1.5-122B UD-IQ2_XXS / UD-Q2_K_XL (est.) | ~4.0–4.7 | IQ2/Q2_K/IQ3 experts (~1.2 GB); the non-expert ~3.3 GB (Gated DeltaNet, attention, shared expert, head) not read | partial | | |
| Llama-2-7B Q4_0 (read) | 3.75 | Q4_0 97 %, Q6_K 3 % | 97 % | 100 % | |
| Qwen3.8-27B Q4_0, Unsloth (read) | 15.33 | Q4_0 82.5 %, Q5_K 6.8 %, Q6_K 6.8 % (head), Q4_1 2.9 %, Q8_0 and F32 1 % | 93 % | 99 % | |

- **Q6_K is in every file.** llama-quant puts the output head in Q6_K for the Q4 file types, so even a "Q4_0" download
  reads a Q6_K head every token. For Qwen3.6 that head is 248,320 × 2,048: 0.417 GB, a fifth of the token.
- **The common download is the uncovered one.** A Q4_K_M token of the qwen35moe-35B family is 57 % Q4_K and 39 % Q6_K,
  so today's fork runs it almost entirely on llama.cpp's kernels (est. 19–23 tok/s against Q4_0's measured 29.0).
- **Shapes that don't fit the K-quant block of 256** push llama-quant to 32-block types: GLM-4.5-Air's expert down
  (1,408) goes to IQ4_NL, Q5_0 or Q8_0, gpt-oss's 2,880-wide tensors likewise (`pers:src/llama-quant.cpp:403-430`).

## 2. Per model, per trick

Gains are on the P106 rig against the fork as it is today (v2/combined) unless the row says otherwise. "Effort" is
fork work, excluding the rig run.

### 2.1 Qwen3.6-35B-A3B and the qwen35moe-35B family

Measured: Q4_0 stock 18.5 tok/s, with CUDA graphs 19.6, fork kernels off/on 22.79 → 29.04 (4 cards, Q38); 14.0 at 32k
and 8.1 at 64k of context (stock); MTP 1.04× on the fork (Q43); Unsloth's UD-Q4_K_XL has KLD 0.013 against Q8_0 (Q82).

| Trick | Applies? | Change needed | Gain on the rig (est.) | Effort (est.) |
|---|---|---|---|---|
| CUDA graphs on 6.1 | yes | none | +6 % (measured) | — |
| 2-warp launch shape | yes | none (a compute-6.1 table on `pers`, `9d5f1530e`) | ~+16 % (19.6 → 22.8, two sessions) | — |
| Pascal 1-token kernels (v2 types) | Q4_0 files yes; Q4_K_M no | none for Q4_0 | Q4_0 +27 % (measured); Q4_K_M 0 | — |
| Pascal Q6_K (Q101, built) | yes: the head in every file, 39 % of Q4_K_M | rig gate (≥ 130 GB/s), then merge | Q4_0 +12–18 % (head 8.5 → 3.7 ms); Q4_K_M +15–20 % | 0.5 day |
| Pascal Q4_K (new) | Q4_K_M, Q4_K_S, the Q4_K tensors of UD files | the Q5_K kernel without the fifth bit; 144-byte super-blocks are 16-byte aligned at K = 512 and 2,048 | Q4_K_M 1.5–1.7× together with Q6_K (est. 19–23 → 30–36 tok/s) | 2–3 days |
| FP32 activation sums, rescale switch | sums yes, switch no | none | 0: no rescale exists for `qwen35moe` | — |
| Shared-expert fusion fix (Q50) | likely: gate/up/GLU of the 512-wide shared expert | none | +0.5–1 % (Flint +0.7 %) | — |
| Persistent MoE kernel | no | a Qwen shape class: top-8, softmax, shared width 512 with its sigmoid gate (an extra dot product, sigmoid and scale before the add); combos (Q4_0, Q4_0, Q5_0), (Q4_0, Q4_0, Q8_0), (Q4_0, Q4_1, Q8_0) for the Q4_0 file (read); four-type combos for Q4_K_M, whose shared gate is Q4_K and shared down Q6_K in 21 of 40 layers (read) | +3–4 % (40 layers × ~30 µs of a ~34 ms token) | 4–6 days after the refactor of §2.7 |
| Norm writes Q8_1 (Q100) | yes (2,048) | none | ~0 (none measurable on Flint, Q117) | — |
| Screened head | engine yes | a Qwen3.6 screen file (248,320 × 2,048; a GPTQ Q4_0 screen is 286 MB) checked for exactness against the Q6_K head | +2–4 % (Flint measured +1.5–2.3 %) | 1 day on the quant machine |
| PR #31047 (prompts) | yes | none | +20–30 % prompt speed | — |
| On-card attention mask | no (hybrid memory keeps the host mask) | pass the device-mask flag for the hybrid cache's attention part (`mask2:src/llama-graph.cpp:4249` against `:3412`) and set its cell inputs (`:1345-1349`) | −0.3 ms per card with attention layers per token at 32k: ~3 % on 5 cards; ~10 % at 131k | 0.5–1 day |
| Pipelined speculation, rollback | no: a recurrent state can't drop a tail | a ring of D + 1 recurrent states per sequence (~60 MiB each, ≤ 540 MiB over the cards); the `s_copy` indirection that already picks each step's source state (`mask2:src/llama-graph.cpp:1361-1368`) is the hook | 1.3–1.4× on 6 cards (Flint: 1.34–1.41×) | 3–5 days |
| CPU-free loop (packed uploads Q120, device-fed token Q121) | after the rollback | none beyond it | streams of conversations +21 % at 8 (as Flint, Q120) | — |
| GPU sampling for tools (Q67) | yes | none | +15–22 % on tool requests (logits copy 993 KB: est. 4.7 ms, plus host sampling) | planned |
| MTP + multi-token kernels | yes | rerun Q43 on the combined build; the 32k draft vocabulary (Q43) cuts each draft's 0.417 GB head read | 1.15–1.3× (a P40 measured 1.36× at 2 drafts) | rig only |
| GPTQ head/body, F16 router | per file | F16 router: the F32 router is 86 MB per token (read); GPTQ below Q6_K needs a KLD check per model | F16 router +1–2 %; Unsloth's KLD 0.013 leaves little reason for body work | hours per file |

### 2.2 Qwen3-30B-A3B

Not run on the rig. A plain attention-only MoE: every multi-GPU piece applies as it is.

| Trick | Applies? | Change needed | Gain on the rig (est.) | Effort (est.) |
|---|---|---|---|---|
| CUDA graphs, 2-warp shape | yes | none | as Flint's (+14 % and +27 % on Flint; Qwen3.6 +6 % and ~+16 %) | — |
| Pascal 1-token kernels | Q4_0/Q8_0 files yes | none; expert downs have K = 768 (24 Q4_0 blocks: 4 lanes per row; 3 K-quant super-blocks: 1 lane per row, `pers:ggml-cuda/mmvq-pascal.cu:667-673`) | Q4_0 est. 33–37 tok/s on the fork against est. ~22 stock | — |
| Pascal Q6_K | yes (head; ~27 % of Q4_K_M) | rig gate, merge | Q4_0 +8–10 %; Q4_K_M +15–20 % | 0.5 day |
| Pascal Q4_K | Q4_K_M | as §2.1; check the 1-lane shape at K = 768 | Q4_K_M 1.5–1.75× with Q6_K (est. ~22 → 33–40 tok/s) | with §2.1 |
| FP32 sums, rescale switch | sums yes, switch no | none | 0 | — |
| Shared-expert fusion fix | only where ggml-alloc reuses an FFN input | none | ~0 | — |
| Persistent MoE kernel | no | a shape class: 128 × 768, top-8, softmax, no shared expert (so nothing hides the wait for the top-k, est. smaller gain) | +3–5 % | 2–3 days after §2.7 |
| Norm writes Q8_1 | yes (2,048) | none | ~0 | — |
| Screened head | engine yes | a screen file (151,936 × 2,048) | +1–3 % (head ~10 % of active parameters) | 1 day |
| PR #31047 | yes | none | +15–25 % prompts (128 experts, top-8) | — |
| On-card mask | yes (plain cache) | none | KV is 96 KiB per token, so long context matters more here: −3 ms per token at 32k on 9 cards (Q116's estimate) | — |
| Pipelined speculation | yes | none (correctness checks per model) | 1.3–1.5× on 6–9 cards (Flint measured 1.30–1.55×) | rig only |
| CPU-free loop | yes | none | as Flint's | — |
| GPU sampling for tools | yes | none | +12–16 % on tool requests (608 KB logits row, Flint-sized) | planned |
| GPTQ head/body, F16 router | per file | F16 router saves ~25 MB per token | +1 % | hours |

### 2.3 gpt-oss-20b and gpt-oss-120b

gpt-oss-120b does not fit (every file 62.6–65.4 GB); only a community REAP-58B prune (MXFP4, 32.8 GB) does, with
unknown quality. The rows below are gpt-oss-20b.

| Trick | Applies? | Change needed | Gain on the rig (est.) | Effort (est.) |
|---|---|---|---|---|
| CUDA graphs, 2-warp shape | yes | none | +5–15 % (24 layers, fewer launches than Flint) | — |
| Pascal 1-token kernels | Q8_0 half of the bytes | rows of 2,880 are 90 Q8_0 blocks: 45 two-block chunks give one lane per row and no 16-byte staging (`pers:ggml-cuda/mmvq-pascal.cu:667-673`, `mmvq-pascal-impl.cuh:352-353`); a shape chosen by K (one block per lane: 2 lanes per row) | unknown until a sweep; +10–25 % over stock if the Q8_0 parts reach 90–140 GB/s | 0.5–1 day + sweep |
| Pascal MXFP4 (new) | experts, half of the bytes | 17-byte blocks (odd alignment, staged with byte shifts as Q6_K's), a 16-entry table as IQ4_XS, E8M0 scales; expert biases and SwiGLU-OAI already in the kernel's epilogue ([kernel design](pascal_mmvq_integration.md) §6) | +15–25 % (MXFP4 from 60–75 to ~95–110 GB/s, as IQ4_XS) | 3–5 days |
| FP32 sums, rescale | — | none | 0 (MXFP4 and Q8_0 never read FP16 sums) | — |
| Shared-expert fusion fix | — | — | 0 | — |
| Persistent MoE kernel | no | softmax over the selected 4, router and expert biases, SwiGLU-OAI with clamp, no shared expert, MXFP4 device functions, H = 2,880 | +3–6 % (24 layers) | 1.5–2 weeks: not before MXFP4 decode exists |
| Norm writes Q8_1 | no | 2,880 is not a multiple of 512 (`comb:ggml-cuda/norm.cu:925`) | 0 | — |
| Screened head | engine yes (plain head product) | a screen file (201,088 × 2,880) | +3–5 % (head ~24 % of the bytes) | 1 day |
| PR #31047 | yes | none | 0–10 % (32 experts at top-4 fill tiles already) | — |
| On-card mask | yes (sliding-window pairs and plain cache) | none | small at its 131k maximum (24 KiB KV per token) | — |
| Pipelined speculation | yes | none; a sliding-window model needs `n_ubatch` ≥ slots (`pipe2:src/llama-context.cpp:1636-1642`) | 1.2–1.35× spread over 6 cards (fits on 3) | rig only |
| GPU sampling for tools | yes (where tool requests carry a grammar) | none | +12–15 % on tool requests (804 KB logits row) | planned |
| GPTQ head/body, F16 router | head only | GPTQ head below Q8_0 | +3–6 % if the KLD allows | 1 day |

### 2.4 GLM-4.5-Air and GLM-4.7-Flash

- **GLM-4.5-Air is a poor fit.** Its 5 B parameters of attention per token (96 query heads) dominate the bytes, the
  routed expert downs fall back to IQ4_NL, and only ~2-bit files fit: UD-IQ2_XXS (42.7 GB) up to ~32k of context with
  f16 KV (184 KiB per token) ([large MoE models](models_large_moe_and_compression.md) §3). Est. 7–9 tok/s on today's
  fork and 11–13 with Pascal Q4_K/Q6_K kernels, if its attention is K-quant (types not read).
- **GLM-4.7-Flash fits well.** 31 B total, ~3.6 B active, Q4_K_M 18.3 GB, Q8_0 31.8 GB, MLA KV 53 KiB per token.
  - A P40 runs its Q4_0 at 41.2 tok/s.
  - Flash attention at MLA head sizes is slow on Pascal by a maintainer's account, so it wants `-fa off`.
  - Est. 22–30 tok/s on the fork with a Q4_0 file.

| Trick | GLM-4.5-Air | GLM-4.7-Flash |
|---|---|---|
| CUDA graphs, 2-warp shape | yes; +5–10 % (byte-bound at ~6 GB per token) | yes |
| Pascal 1-token kernels | by type; IQ2_XXS and IQ4_NL not covered (IQ4_NL: est. 1 day from the IQ4_XS and Q4_0 code) | Q4_0/Q8_0 files yes; Q4_K_M needs Q4_K |
| Pascal Q6_K, Q4_K | its attention, if K-quant: est. 1.3–1.5× | as Qwen3-30B |
| FP32 sums, rescale | correctness only | correctness only |
| Shared-expert fusion fix | likely, +0.5 % | likely |
| Persistent MoE kernel | far: H = 4,096, selection bias, IQ2 device functions | a shape class: 64 × 1,536, top-4, shared 1,536, DeepSeek-style router (topk_moe already accepts bias and groups); est. +3–5 %, 3–4 days after §2.7 |
| Norm writes Q8_1 | yes (4,096), ~0 | yes (2,048), ~0 |
| Screened head | needs a screen; head ~8 % of the bytes | needs a screen |
| PR #31047 | yes | yes |
| On-card mask | yes | yes (MLA uses the plain cache) |
| Pipelined speculation | yes; 9 cards, so depth is available | yes |
| GPU sampling for tools | yes | yes |

### 2.5 Ornith-1.5-122B

The same architecture as Qwen3.5-122B-A10B: a hybrid like Qwen3.6, at hidden 3,072 and 1,024-wide experts.
- **Files that fit:** UD-IQ2_XXS (42.0 GB) and UD-Q2_K_XL (44.5 GB), at 32–64k of context (est., from Qwen3.5-122B's
  per-layer packing). UD-IQ3_XXS (49.5 GB) is marginal.
- **Speed:** est. ~4.0–4.7 GB per token (est., ~9 B active parameters, ~73 % of the bytes outside the routed experts),
  9–12 tok/s on today's fork, 13–16 with Pascal Q4_K/Q6_K kernels if its non-expert tensors are K-quants. For
  comparison, four P40s run Qwen3.5-122B UD-Q3_K_XL at 29.5 tok/s.

| Trick | Applies? | Change needed | Gain (est.) |
|---|---|---|---|
| CUDA graphs, 2-warp shape | yes | none | as Qwen3.6 |
| Pascal kernels | the Q5_K/Q8_0 non-expert tensors; not IQ2/Q2_K/IQ3 experts | Q4_K/Q6_K kernels for the rest of the non-expert tensors; an IQ2/Q2_K family (lookup grids) | 1.3–1.4× with Q4_K/Q6_K; an IQ2 kernel family is 1–2 weeks for est. +10–15 % more |
| FP32 sums, rescale | correctness only | — | 0 |
| Persistent MoE kernel | no | H = 3,072, 1,024-wide experts, IQ2 device functions | far |
| Norm writes Q8_1 | yes (3,072) | none | ~0 |
| On-card mask, pipelined speculation | no (hybrid) | as Qwen3.6 (§2.1); a recurrent snapshot is ~144 MiB (36 × 4 MiB), so a ring of 6–9 costs 0.9–1.3 GB on cards that a 2-bit file already fills | as Qwen3.6, if memory allows |
| Screened head | engine yes | a screen file (248,320 × 3,072) | +2–3 % |
| PR #31047, GPU sampling for tools | yes | none | as Qwen3.6 |

### 2.6 Dense control

Measured: Llama-2-7B on one card, stock: Q4_0 19.5 tok/s (73 GB/s), Q4_K_M 15.0. Qwen3.8-27B Q4_0 on four cards: 4.3
tok/s with CUDA graphs.

| Trick | Llama-class 7–8 B (attention-only) | Qwen3.8-27B (hybrid) |
|---|---|---|
| CUDA graphs on 6.1 | yes; +6–7 % on other Pascal cards (#27721) | yes (3.9 → 4.3 with Q4_0 and graphs; different files) |
| 2-warp shape | yes; +5.4 % on a 10-SM GTX 1060 (#25479), more on 5 SMs (unmeasured) | yes |
| Pascal 1-token kernels | Q4_0 97 % covered: est. 1.4–1.6× (19.5 → 27–31 tok/s) | Q4_0 93 % covered: est. 1.55–1.65× (4.3 → 6.7–7.1) |
| Pascal Q6_K, Q4_K | Q4_K_M est. 1.6–1.9× (15.0 → 24–28) | Q6_K head +5–8 % (→ 7.2–7.7) |
| FP32 sums, shared-expert fusion, persistent kernel, PR #31047 | — | — |
| Norm writes Q8_1 | yes (4,096), ~0 | no (5,120) |
| Screened head | small: 3 % of the bytes (32k vocabulary), ~7 % for 128k-vocabulary Llama 3 | 7 % of the bytes (Q6_K head) |
| On-card mask | yes | no (hybrid) |
| Pipelined speculation | only on a split of ≥ 2 cards; a 7 B fits one, so spreading it for depth: est. 1.1–1.3× | needs the recurrent ring; then est. 1.2–1.45× on 4–6 cards (each card's stage is long against the ~1 ms hand-off) |

The dense model is the cheapest clean A/B on the rig: one card, the llama.cpp CUDA scoreboard's own model, Q4_0 and
Q4_K_M, stock against the fork. It also shows the Pascal kernels' gain on wide dense matrices, which no Flint run does.

### 2.7 The persistent MoE kernel: any shape, or one instantiation per model?

What is fixed today, and why:
- **Shapes are compile-time:** hidden 2,048, 256 experts, top-6, widths 512 and 1,024
  (`pers:ggml-cuda/moe-persistent.cu:14-18`). The work list, counters and output groups are constant expressions of
  them (`:33-58`); the router logits sit in a per-lane register array of `NE / 32` entries (`:215`); the activations'
  Q8_1 copy is a shared array of `H / 32` blocks (`:565`); staging buffers follow each product's K.
- **The router is sigmoid with renormalization only.** The kernel's top-k is a copy of upstream's sigmoid path
  (`:215-279`), and the matcher refuses softmax, selection bias, expert groups and the softmax after top-k
  (`pers:ggml-cuda/ggml-cuda.cu:3561-3562`).
- **A plain shared expert is required** (`pers:ggml-cuda/ggml-cuda.cu:3621-3640`), with one type for its gate/up and
  its down (`pers:ggml-cuda/moe-persistent.cu:989`). The qwen35moe Q4_K_M files break that rule in 21 of 40 layers.
- **Twelve type combinations**, all from Flint files (`:842-853`).

Two ways to generalize it:

| | All shapes at run time | One template row per model family |
|---|---|---|
| How | work-list offsets as integers; the logit array sized for the largest expert count (512: 16 registers per lane); loops over K not unrolled; shared memory sized at launch | `struct moep_shape { H, NE, KU, F, FS; router kind; shared kind; }` as a template parameter. Router kinds: sigmoid + renormalization (Flint), softmax + renormalization (Qwen3, Qwen3.5/3.6), softmax over the selected (gpt-oss), sigmoid + bias (+ groups) (GLM, DeepSeek-style), all of which upstream's top-k already handles as runtime flags. Shared kinds: none, plain, sigmoid-gated (Qwen3.5/3.6) |
| Risk | the kernel runs at 128 registers with two blocks per SM; the K-quant variants already spill 16–76 bytes; with one block per SM the prototype ran 0.88–0.91× the per-op chain (Q88). More registers most likely erase the gain (est.) | none at run time: each row compiles to what Flint's does today |
| Cost | 1–2 weeks with tuning, outcome uncertain (est.) | refactor 2–3 days with Flint bit-identical (the self-check mode compares every stage); then 2–4 days per row with its `MOE_LAYER` tests (est.) |
| Build | one kernel per type combination | one kernel per row and type combination: today 12; est. +3 for Qwen3.6 Q4_0, +2 for qwen35moe Q4_K_M once Q4_K/Q6_K device functions exist, +3–4 for Qwen3-30B. A CMake list can build a subset, as upstream's `GGML_CUDA_FA_ALL_QUANTS` does for attention |
| Selection | the shapes read from the matched graph | the matcher reads shapes and kinds from the matched graph and looks the row up; no row → the per-op path, exactly as today |

- **Template rows are the clean way.** They follow upstream's own pattern: per-expert-count instantiations of the fused
  router, with named model counts like StepFun's 288 (`pers:ggml-cuda/topk-moe.cu:401`), and flash-attention instances
  per head size.
- **The gain is small outside Flint.** Flint saves ~1.3 ms per token over 47 layers (Q88, one card). That suggests
  est. +3–4 % for Qwen3.6 and +3–5 % for Qwen3-30B-A3B. The kernel comes after the format kernels in §5, and only
  once Flint's rig timing confirms the gain on real files.

### 2.8 FP32 sums and the rescale: who overflows?

- **Upstream rescales only three architectures.** The MoE down-projection rescale is built only for Mistral 4, Mellum
  and Flint (`v2:src/llama-graph.cpp:2528`). For every other model there is nothing to switch off, so the FP32 sums'
  +9.6 % (Q76) cannot repeat.
- **Qwen does not overflow in practice.** Qwen3.6 files decode correctly on llama.cpp's FP16-sum kernels: Q38 on the
  rig, and Q82's KLD on the quant machine. The FP32 sums stay as a free correctness net.
- **The one other candidate is Mistral Small 4** (119 B, 128 experts, top-4, MLA). Its UD-IQ3_XXS (42.8 GB) fits, and
  its rescale could go at one-token steps as Flint's does (est. +5–10 %).

## 3. What fits on the rig

Nine cards with ~6,068 MiB each, est. ~5.6 GB per card for weights after buffers; a 4 GB host, so `-ub 128` at
262,144 tokens unless the on-card mask is in the build ([release v2](release_v2.md) §4.1). KV per token is f16 for the
full-attention layers.

| Model | File (GB) | KV per token | Cards at 32k (est.) | Longest context on 9 cards (est.) | Notes |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | Q4_0 20.8; Q4_K_M 22.1–22.3; UD-Q4_K_XL 22.4; Q8_0 36.9 | 20 KiB + 60 MiB of recurrent state | 4–5 (Q4); 7–8 (Q8_0) | 262,144 for every file (Q8_0: ~42 GB with KV) | Q4_0 runs on 4–9 cards with 262,144 tokens (measured, overview) |
| Ornith-1.5-35B, Qwen3.8-35B-A3B distillation | Q4_K_M 21.7 | as Qwen3.6 | 4–5 | 262,144 | same shapes as Qwen3.6 |
| Qwen3-30B-A3B | Q4_K_M 18.6; UD-Q4_K_XL 17.7; Q8_0 32.5 | 96 KiB | 4 (Q4); 7 (Q8_0) | Q4: 262,144 (~44 GB with f16 KV); Q8_0: ~131k | the large KV makes the on-card mask and long-context work matter |
| gpt-oss-20b | 11.5–12.1 (every file) | 24 KiB + a bounded window | 3 | 131,072 (its maximum) | spread wider only for pipelining |
| gpt-oss-120b | 62.6–65.4 | — | — | does not fit | REAP-58B prune: 32.8 GB, quality unknown |
| GLM-4.7-Flash | Q4_K_M 18.3; Q8_0 31.8 | 53 KiB (MLA latent) | 4; 7 | Q4: ~200k | `-fa off` on Pascal |
| GLM-4.5-Air | UD-IQ2_XXS 42.7; UD-IQ2_M 44.4; UD-Q2_K_XL 47.4 | 184 KiB | 9 | ~32k (IQ2_XXS), ~8k (Q2_K_XL) | its 1.4 GiB MTP layer may occupy VRAM (untested) |
| Ornith-1.5-122B | UD-IQ2_XXS 42.0; UD-Q2_K_XL 44.5; UD-IQ3_XXS 49.5 | 24 KiB + 144 MiB of state | 9 | 32–64k (Q2); IQ3_XXS marginal | the only ~120 B option, at 2-bit quality |
| Llama-class 7–8 B | Q4_0 3.8–4.7; Q4_K_M 4.1–4.9 | 0.5 MiB (Llama-2), 128 KiB (Llama 3) | 1 | native maximum | the scoreboard model |
| Qwen3.8-27B | Q4_0 15.3 | 64 KiB + state | 4 (measured at 32k) | ~262k (est.) | hybrid |

- **Already in the project's hands** (no download): Qwen3.6-35B-A3B Q4_0 with its MTP file, Ornith-1.5-35B-A3B Q4_K_M,
  a Qwen3.8-35B-A3B distillation Q4_K_M, Qwen3.8-27B Q4_0 and Llama-2-7B Q4_0 (their headers were read for this
  note). By the owner's inventory, not read here, the quant machine holds Qwen3.6-35B-A3B Q4_K_M, Ornith-1.5-122B, the
  Qwen3.8 family and a GLM flash-class model.
- **To fetch for the matrix:** Qwen3-30B-A3B Q4_K_M (18.6 GB) and gpt-oss-20b (12.1 GB).

## 4. Branch strategy

### 4.1 How llama.cpp handles per-architecture code

- **Model graphs:** one file per architecture under `src/models/` (167 of them), all in one tree. The loader picks one
  from `general.architecture`. Model-specific numerics sit there too, e.g. the down rescale's list of three
  architectures (`v2:src/llama-graph.cpp:2528`).
- **Kernels:** templated on shapes and instantiated per value that some model needs, then selected at run time from
  tensor shapes:
  - the fused MoE router per expert count, powers of 2 up to 512 plus 288 ("StepFun 3.7") and 576
    (`pers:ggml-cuda/topk-moe.cu:401-409, 475`);
  - flash-attention tile kernels per head size, with generated instance files for 40 to 576/512
    (`ggml-cuda/template-instances/fattn-tile-instance-dkq576-dv512.cu` and siblings; `pers:ggml-cuda/fattn-tile.cu:48-54`);
  - MMQ/MMVQ per quantization type.
- **GPUs:** launch tables per GPU family (RDNA2–4, GCN, Turing, GB10, generic), selected at compile time by
  architecture and at run time by compute capability (`pers:ggml-cuda/mmvq.cu:107-124`). On `pers` the fork's 2-warp
  rule is already its own compute-6.1 table (`9d5f1530e`).
- **Model data:** per-model defaults travel in the GGUF, not in code. Qwen3.6's header carries `general.sampling.temp
  = 1.0`, `top_k = 20` and `top_p = 0.95` (read).
- **No per-model branches.** Upstream has none, and neither does any fork the project's notes cite; per-model work
  lands as model files, instantiations and table rows.

### 4.2 The three options

| | (a) One Pascal branch, model-specific kernels chosen at run time | (b) A branch per model | (c) One Pascal branch plus model profiles in this repository, no code branches |
|---|---|---|---|
| **What lives where** | everything in `pascal`; model-shaped kernels as template rows picked from tensor shapes | `pascal-flint`, `pascal-qwen36`, …, each the Pascal base plus that model's kernels and defaults | code as (a); per-model files (recommended GGUF, cards, context, command line, switches, screen file, measured speed) in this repository |
| **Rebasing on upstream** | once per update; the fork's kernels sit in their own files with one hook each (`mmvq-pascal.cu`, `moe-persistent.cu`) | N times per update, each through upstream's busiest files (`ggml-cuda.cu` 64, `llama-graph.cpp` 49, `llama-context.cpp` 52 upstream commits in three months); every Pascal fix cherry-picked N times, with drift when one is missed | as (a) |
| **User confusion** | one tarball; a model gets whatever kernels match its tensors | pick the branch per model; a rig serving two models at once on separate card sets (overview, finding 8) needs two builds; a model without a branch looks unsupported though the base would speed it up | one tarball, plus a page per model that says what to download and how to run it, measured or not |
| **Testing** | per release: one model matrix on the rig (est. 20–30 min per model); per shape row: `test-backend-ops` cases at that model's shapes, natively and through the compute_61 PTX | every branch revalidated after every rebase, with its own tarball | as (a), plus each profile's numbers re-measured when its row changes |
| **Precedent** | upstream llama.cpp (§4.1) | none found | GGUF metadata already carries per-model defaults; this repository's [release v2](release_v2.md) §4 is a Flint profile in all but name |
| **Failure mode** | build time and binary size grow with instantiations (a CMake list bounds them) | forks of the fork; fixes landing on one branch only | profiles go stale unless the release checklist re-measures them |

### 4.3 Recommendation and layout

**(c), using (a)'s mechanism for the few model-specific kernels:**

- **Fork branches:**
  - `pascal`: the default and release line from v2.0, tags `pascal-vX.Y`; `pascal-p106` and `pascal-p106-v2` stay
    frozen ([fork scope](fork_scope.md) §5.2).
  - Topic branches are named by mechanism: `pascal/q4k-mmvq`, `pascal/mxfp4-mmvq`, `pascal/moe-persistent-shapes`,
    `pascal/hybrid-rollback`, and `up/<topic>` for upstream candidates. Each lands on `pascal` after its rig A/B (the
    2026-10-24 rule).
  - A model never gets a long-lived branch. A topic branch may be motivated by one model, but it ships in `pascal`.
- **How model-specific code is gated:**
  - **Type kernels** (`mmvq-pascal.cu`) by device (compute 6.1), tensor type and row width, as today. The shape choice
    becomes a function of K as well (`mmvqp_choose(type, gate, K)`) for widths like 768, 1,408 and 2,880.
  - **The persistent MoE kernel** by a shape-class table (§2.7), one row per model family: Flint (2,048 / 256 / 6 /
    512 / 1,024, sigmoid, plain shared expert), qwen35moe-35B (2,048 / 256 / 8 / 512 / 512, softmax, sigmoid-gated
    shared expert), qwen3moe-30B (2,048 / 128 / 8 / 768 / —, softmax, none), later others.
    - Instances in generated files under `ggml-cuda/template-instances/`, as upstream does for attention.
    - A CMake list chooses which rows to build; `GGML_CUDA_PERSISTENT_MOE` switches it at run time.
    - Lookup by the matched graph's shapes and kinds. `general.architecture` is never consulted, so fine-tunes and
      distillations with the same shapes (Ornith-1.5-35B, the Qwen3.8 distillation, KAT-Coder) get the row for free.
  - **Hybrid support** in the mask and the executor by memory type (hybrid, recurrent), not by model.
  - **Graph-level per-model behaviour** stays where upstream keeps it (`src/models/*.cpp`, the rescale list).
- **Per-model files in this repository:** `models/<model>/`, one folder per profiled model:
  - `README.md`: the profile. Recommended files with links, sizes and checksums; which share of a token each file
    reads in which type, and what the fork covers; cards and context; the server command line; environment switches;
    measured speed and quality, with what is not measured.
  - `server.sh`: that command line, taking the model path.
  - Screen side files are release assets (or a model repository on Hugging Face), with the recipe in
    [quants/recipes](../../../quants/recipes/) and the tool in [quants/gptq_head](../../../quants/gptq_head/).
  - The README's model table links each profile and separates measured from unmeasured rows.
  - The Flint profile is [release v2](release_v2.md) §4 moved there.
- **Testing:**
  - each shape row adds `MOE_LAYER` cases at that model's shapes (as Flint's 17), run natively and through the
    compute_61 PTX on the quant machine;
  - each release runs the rig matrix: one warm llama-bench pair per profile, stock against `pascal`.

## 5. Work list for the next models

The top three other models are the qwen35moe-35B family (Qwen3.6, Qwen3.5, Ornith-1.5-35B, the Qwen3.8 distillation,
KAT-Coder: one set of shapes), the Qwen3-30B-A3B family and gpt-oss-20b; a dense 7–8 B is the control, and
GLM-4.7-Flash a fourth if wanted.

| # | Item | Models | Gain on the rig (est.) | Effort (est.) | Before it |
|---|---|---|---|---|---|
| 0 | Rig model matrix, no new code (§5.1) | all | sets every baseline below | half a rig day | — |
| 1 | Land the Pascal Q6_K kernel (Q101) | every Q4-type download's head; 27–39 % of Q4_K_M files | Qwen3.6 Q4_0 +12–18 %; Q4_K_M files +15–20 %; Qwen3-30B Q4_0 +8–10 % | 0.5 day after the microbenchmark gate | the gate (≥ 130 GB/s) |
| 2 | Pascal Q4_K kernel: 1–4 tokens, `MUL_MAT` and `MUL_MAT_ID`, gate/GLU fusion, routed + shared launch; from the Q5_K kernel | all Q4_K_M files, dense included | with item 1: qwen35moe Q4_K_M 1.5–1.7× (19–23 → 30–36 tok/s), Qwen3-30B Q4_K_M 1.5–1.75×, dense Q4_K_M 1.6–1.9× | 2–3 days + microbenchmark + tests | Q4_K in `bench/mmvq_pascal` first (gate ≥ 120 GB/s at K = 512 and 2,048) |
| 3 | GPU sampling for tool requests (Q67, planned) | all | Qwen3.6 +15–22 % on tool requests, others +12–16 % | 2–3 days (planned) | — |
| 4 | Hybrid memory: device mask for the hybrid cache, then a ring of recurrent states for the executor | qwen35moe family, Ornith-122B, Qwen3.8-27B | mask ~3 % at 32k and ~10 % at 131k; pipelined decoding 1.3–1.4× on 6 cards | 0.5–1 day + 3–5 days | — |
| 5 | Launch shapes by row width: K = 768, 1,408, 2,880 and other block counts that aren't 2^k | gpt-oss, Qwen3-30B downs, GLM | unknown until the sweep | 0.5–1 day + sweep | microbenchmark sweep at those widths |
| 6 | Pascal MXFP4 kernel | gpt-oss-20b (and MXFP4_MOE files of other models) | +15–25 % on gpt-oss-20b | 3–5 days | item 5 |
| 7 | Persistent kernel: shape-class refactor, then the qwen35moe-35B and qwen3moe-30B rows | qwen35moe family, Qwen3-30B | +3–5 % per model | 2–3 days + 2–4 days per row | Flint's rig timing (Q88); items 1–2 for the Q4_K_M combinations |
| 8 | Screens per model | Qwen3.6, gpt-oss, Qwen3-30B | +2–5 % on chat | 1 day each on the quant machine | Q125's method |
| 9 | IQ4_NL in the Pascal kernels | GLM-4.5-Air's downs, some UD files | small | 1 day | — |

- **Where this lands (est.):**
  - Qwen3.6 Q4_0: 29 → ~34 tok/s with item 1, then 1.15–1.4× more from MTP (item 0's rerun) or pipelined decoding
    (item 4).
  - The qwen35moe-35B Q4_K_M downloads: 19–23 → 30–36 tok/s with items 1–2.
  - Qwen3-30B-A3B Q4_K_M: ~22 → 33–40 tok/s with items 1–2. Pipelined decoding already applies there, 1.3–1.5× more.
  - gpt-oss-20b: est. 23–30 → 28–37 tok/s with items 5–6.
- **Order:** items 1–3 help every model, the common downloads most. Item 4 opens the whole Qwen3.5/3.6/Ornith/Qwen3.8
  line to the multi-GPU pieces. Items 5–7 are per-family.

### 5.1 Rig measurements to run first

All with a warm-up (pp512 before tg128 in one process), stock against the combined build in one session, the q6k
tarball as a third arm:

1. **Qwen3.6-35B-A3B Q4_0 on 5 cards:** stock, combined, q6k. Combined against q6k is the Q6_K kernel on the head of
   the most common Qwen download.
2. **A qwen35moe-35B Q4_K_M** (Ornith-1.5-35B or the Qwen3.8 distillation, already local) on 5 cards: today's K-quant
   baseline, which items 1–2 have to beat.
3. **Llama-2-7B Q4_0 and Q4_K_M on one card:** the dense control, comparable with the llama.cpp CUDA scoreboard.
4. **One profiled Qwen3.6 Q4_0 token** (`GGML_CUDA_PROFILE`, [cuda_profile_summary.py](../../../scripts/cuda_profile_summary.py))
   to split the ~14 ms that isn't weight reading into Gated DeltaNet ops, attention and launches. It decides whether
   a fusion on the DeltaNet side is worth more for Qwen than the persistent MoE kernel.
5. **Q43 again on the combined build:** Qwen3.6 MTP with the multi-token Pascal kernels in the verify steps.
6. **After the downloads:** Qwen3-30B-A3B Q4_K_M on 4–6 cards (and `llama-pipespec` on 6 cards: pipelined decoding
   applies to it unchanged) and gpt-oss-20b on 3 cards.

```bash
# M: model, CB: combined tarball's bin/, Q6: q6k tarball's bin/, ST: stock build's bin/; CUDA_VISIBLE_DEVICES sets cards
for B in $ST $CB $Q6; do $B/llama-bench -m $M -ngl 99 -fa on -p 512 -n 128 -r 3 -o jsonl; done
GGML_CUDA_PROFILE=qwen36.prof $CB/llama-bench -m $M -ngl 99 -fa on -p 0 -n 32 -r 1
```

## Not verified

- **Not run on the fork or the rig:** Qwen3-30B-A3B, gpt-oss, GLM, Ornith-1.5-122B and modern dense models.
  - Their gains are this note's arithmetic on the calibration above, with in-model rates from Flint's and Qwen3.6's
    measurements.
  - The fixed time per token (~14 ms for Qwen3.6, ~12 ms for attention-only MoE models) is inferred, not profiled.
- **Type mixes not read:** Unsloth's Qwen3.6 UD-Q4_K_XL, every gpt-oss file (inferred from file sizes), GLM-4.5-Air's
  and Ornith-1.5-122B's non-expert tensors, and Qwen3-30B-A3B's files (from llama-quant's rules). The quant machine's
  model inventory is the owner's, not read.
- **Kernels that don't exist yet:** the Q4_K and MXFP4 speeds (120–145 and 95–110 GB/s) are estimates from the Q5_K
  and IQ4_XS kernels' measured 120–128 and 94–104 GB/s. The one-lane shapes at K = 768 and 2,880 have no measurement.
- **Correctness on these models:**
  - whether gpt-oss's attention sinks run correctly on sm_61's flash-attention path;
  - whether the screen is exact against a Q6_K head (checked only against Q8_0 heads);
  - whether the shared-expert fusion fix fires in Qwen3.6's graph.
- **Executor and context:**
  - The recurrent-state ring is a design; upstream's per-batch snapshots (`n_rs_seq`) were read, not tried with the
    executor.
  - Context limits per card count assume ~5.6 GB of weights per card and f16 KV; the compute buffers per architecture
    on Pascal are unmeasured.
- **Not checked in a GGUF:** GLM-4.7-Flash's router (sigmoid with selection bias assumed, DeepSeek-V3 style).

## Sources

The fork, read-only (P):
- `pers:ggml-cuda/moe-persistent.cu` (shapes, work list, top-k, combinations, support checks), `pers:ggml-cuda/ggml-cuda.cu`
  (the MoE-layer matcher), `pers:ggml-cuda/mmvq-pascal.cu` and `mmvq-pascal-impl.cuh` (types, lane derivation, shape
  table), `pers:ggml-cuda/mmvq.cu` (per-GPU tables, the compute-6.1 table of `9d5f1530e`), `pers:ggml-cuda/topk-moe.cu`,
  `pers:ggml-cuda/fattn-tile.cu` and `template-instances/`, `q6k:ggml-cuda/mmvq-pascal.cu`, `comb:ggml-cuda/norm.cu`,
  `comb:src/llama-graph.cpp` (screened head), `mask2:src/llama-graph.cpp` (device and hybrid masks),
  `pipe2:src/llama-memory-recurrent.cpp`, `pipe2:src/llama-memory-hybrid.cpp`, `pipe2:include/llama.h` (`n_rs_seq`),
  `pipe2:src/llama-context.cpp` (`pipe_init`), `v2:src/llama-graph.cpp` (rescale), `v2:src/models/` (`qwen35moe`,
  `qwen3moe`, `openai-moe`, `glm4-moe`, `deepseek2`, `flint`), `pers:src/llama-quant.cpp` (type fallbacks),
  `v2:conversion/glm.py`.

This repository (P):
- [quants/experiments.md](../../../quants/experiments.md): Q19, Q38, Q43, Q50, Q55, Q67, Q76, Q78, Q82, Q87, Q88, Q92,
  Q100, Q101, Q108–Q110, Q116, Q117, Q120, Q121, Q125, Q144.
- [overview](../../overview.md), [hardware spec](../../mining_rig.md), [README](../../../README.md).
- Notes: [fork scope](fork_scope.md), [v2 synthesis](v2_synthesis.md), [kernel design](pascal_mmvq_integration.md)
  (§5, §6, §15–17), [K-quant decode](kquant_decode_pascal.md), [pipelined speculation](pipelined_speculation.md),
  [release v2](release_v2.md), [small MoE models](models_small_moe.md), [large MoE models](models_large_moe_and_compression.md),
  [HyperQwen lessons](hyperqwen_lessons.md), [dense 27B](dense_27b_speedups.md), [transfer to RTX 3090](transfer_rtx3090.md),
  [transfer to V100](transfer_v100.md), [microbenchmark](../../../bench/mmvq_pascal/README.md).
- GGUF headers read with [scripts/token_cost.py](../../../scripts/token_cost.py) and a header reader: bartowski's
  Qwen3.6-35B-A3B Q4_0, Ornith-1.5-35B-A3B Q4_K_M, the Qwen3.8-35B-A3B distillation Q4_K_M, Unsloth's Qwen3.8-27B Q4_0,
  TheBloke's Llama-2-7B Q4_0.

External (P), as cited in the notes above:
- llama.cpp [#27721](https://github.com/ggml-org/llama.cpp/pull/27721) (CUDA graphs on Pascal),
  [#25479](https://github.com/ggml-org/llama.cpp/pull/25479) (2 warps on Pascal),
  [#31047](https://github.com/ggml-org/llama.cpp/pull/31047) (MoE tiles for MMQ),
  [#19020](https://github.com/ggml-org/llama.cpp/issues/19020) (GLM-4.7-Flash on a P40),
  [#27980](https://github.com/ggml-org/llama.cpp/issues/27980) (Qwen3.5-122B on four P40s).
- Model configs and GGUF file listings on Hugging Face, as tabulated in [small MoE models](models_small_moe.md) and
  [large MoE models](models_large_moe_and_compression.md) (Qwen3-30B-A3B, Qwen3.6-35B-A3B, gpt-oss-20b/120b,
  GLM-4.5-Air, GLM-4.7-Flash, Qwen3.5-122B-A10B, Ornith-1.5-122B).
