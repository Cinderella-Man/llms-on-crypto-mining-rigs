# Faster generation for Flint (40B-A2.5B MoE) on 9 × P106-090 (sm_61, 5 SMs, PCIe 1.0 x1, no P2P), as of 2026-10-22

Scope: every technique that could raise the generation speed of Recursive's Flint on this rig with llama.cpp-family engines, both for one conversation and for several served at once. Covered: the model's lineage and what it implies for speculative decoding, the MoE activation rescaling that llama.cpp added for Flint, expert-count reduction, kernel-level levers on Pascal, quantization layouts, multi-GPU and serving settings, and other engines. All sources were read on 2026-10-22 unless a date is given. Conventions:
- "Claim to test" marks anything not confirmed by a primary source or not measured on this rig.
- "(my calc)" marks this note's own arithmetic, including byte counts read from GGUF headers with HTTP range requests (no tensor data downloaded).
- "P" and "S" in source lists mean primary (PR, issue, source code, model card, paper) and secondary (blog, community repo, aggregator).
- llama.cpp source links point to master as of 2026-10-22 (commit `970b51c`). The rig's build `d77256999` is seven commits behind it, none of which touch the code discussed here.

The rig's facts are in the [hardware spec](../../mining_rig.md); measured numbers are in [docs/benchmarks/](../../benchmarks/). Nothing in this note was run on the rig beyond what those benchmark pages report.

**Starting point** (measured, [Flint benchmark](../../benchmarks/2026-10-22-flint.md) and [speed-ups](../../benchmarks/2026-10-22-speedups.md), llama.cpp `d77256999`, flash attention on):
- Recursive's Q4_1 file (26.0 GB):
  - llama-bench tg128: 27.92 tok/s on 5 cards and 28.09 on 9 with CUDA graphs ([PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)); 24.56 without graphs. pp512 161–171 tok/s.
  - Generation with 8,192 / 32,768 tokens already in context: 25.7 / 19.5 tok/s.
  - llama-server, `-c 262144 -b 512 -ub 128`, 9 cards, real chat requests: 22.47 tok/s; 24.26 with backend sampling (`-bs`).
  - Two slots (`-np 2`): 21.2–23.6 tok/s per conversation, 35.4–42.4 combined.
- Unsloth UD-Q4_K_XL (25.1 GB): tg128 17.4 (5 cards) / 19.0 (9 cards); 15.8 tok/s served with `-bs`.
- Quality against Unsloth's Q8_0 (wikitext-2, 80 × 512 tokens): Q4_1 mean KLD 0.316, same top token 78.6 %, PPL +3.7 %; UD-Q4_K_XL 0.124, 87.6 %, +3.3 %.
- Qwen3.6-35B-A3B Q4_0 on the same rig: 19.6 tok/s with graphs; 27.7 tok/s served with graphs, MTP (3 drafts, 76 % accepted) and `-bs`.

Headline findings (each is sourced below):
1. **Flint is not derived from Qwen3.6-35B-A3B, so no Qwen-specific trick transfers.** Recursive says it was trained from scratch. Its tokenizer is a fresh 150,000-token BPE (149,744 merges over 256 byte tokens, plus 144 special tokens from id 150000) that shares no id space with Qwen3.6, GLM-4.7 or DeepSeek-V3. The architecture differs in every speed-relevant respect: 49 layers of sliding-window and NoPE full attention with sinks instead of 30 Gated DeltaNet + 10 attention layers, 6 of 256 experts instead of 8, no MTP layer.
   - What it does inherit is code, not weights: the Hugging Face config uses Qwen2/3-MoE field names, llama.cpp's converter subclasses its Qwen2-MoE converter, and the chat format follows GLM-4.x conventions.
   - The blog says Recursive's architecture search *started* from a 35B-total / 3B-active baseline configuration, which may be the origin of the Qwen3.6 suspicion.
2. **No drafter exists for Flint.** There is no MTP/NextN layer in the checkpoint, no EAGLE-3, DFlash or DSpark head, and no small same-tokenizer model. In llama.cpp that leaves only the n-gram drafters (`ngram-simple`, `ngram-map-k`, `ngram-mod`, `ngram-cache` with optional static lookup files). Synthetic acceptance (`--spec-synth-len`) can measure what a future drafter would be worth on this rig.
3. **On this rig, verifying a few extra tokens is nearly free for Flint.** Two slots decode in 1.0–1.13× the time of one (my calc from the two-slot run). Short n-gram drafts (up to 5 tokens, so the verify batch stays on the MMVQ path for Q4_x experts) therefore cost little when rejected, while long drafts on file-rewrite output can win several times over.
4. **The activation rescaling that llama.cpp applies to Flint costs ~470 kernel launches per token, about a quarter of all launches** (my estimate). It adds 11 graph ops per MoE layer, including a CUDA `POOL_1D` kernel that runs one thread per output and loops serially over 512 values.
   - There is no flag, environment variable or GGUF key to turn it off; the condition is hard-coded per architecture. The PR that added Flint says "Auto-scaling leads to slowdown of 10-30%".
   - Removing it is numerically exact when the down-projection weights use a format whose CUDA dot product ignores the Q8_1 block sum: Q8_0, Q6_K, IQ4_NL, IQ4_XS, MXFP4, NVFP4. With Q4_0, Q4_1, Q5_0 or Q5_1, removing it is safe only if activations stay below ~2,000, which nobody has measured for Flint. Expected gain +5–12 % (claim to test).
5. **Attention, not experts, dominates Flint's per-token bytes.** In the Q4_1 file a token reads 1.66 GB: attention 35 %, routed experts 33 %, output head 12 %, shared expert 11 %, router 6 % (my calc from the GGUF header). The usual MoE recipe (attention at Q8_0, experts at 4 bits) therefore costs Flint a lot of speed. UD-Q4_K_XL reads 2.21 GB per token, Unsloth's Q8_0 2.70 GB. The bits are cheapest when spent on routed experts, which make up 95 % of the file but a third of the bytes per token.
6. **About a third of the token time is fixed per-kernel overhead.** An estimated ~1,900 kernel launches per token (≈3,500 graph nodes) account for ~10–13 ms of the 35.8 ms token; CUDA graphs already removed 4.9 ms of CPU launch cost. vLLM's Blackwell path runs a Flint MoE layer in seven kernels, which shows the ceiling: ~36–45 tok/s if fixed costs vanished (my calc).
7. **Recursive's Q4_1 and NVFP4 files are quantization-recovered checkpoints, so KLD against a BF16-derived reference understates them.** A community comparison found the Q4_1 file "deliberately differs from BF16" (KLD 0.675 on code) yet scores the same on 300 tool-calling tasks as BF16 and every other 4-bit file. Quant choices for Flint need a task-level check, not only KLD.
8. **Expert-count reduction is a weak lever here.** Going from 6 to 4 experts saves only ~11 % of the bytes per token and no kernel launches. On a similar 256-expert model, cutting 8 → 4 experts with standard renormalization lost 4.65 MMLU points.
9. **At long context the KV cache of the 14 full-attention layers dominates.** It holds 28 KiB per token, ~0.94 GB read per generated token at 32k. On sm_61 a q8_0 KV cache moves attention from the GQA-aware TILE kernel to the VEC kernel, which re-reads K/V for each of the 8 query heads per KV head. So q8_0 is likely slower than f16 there until open [PR #30792](https://github.com/ggml-org/llama.cpp/pull/30792) or similar lands (claim to test).
10. **No other engine helps on this rig.** ik_llama.cpp merged Flint on 2026-10-22 without the rescaling, but needs AVX2 on the host. vLLM's and SGLang's Flint kernels target Blackwell and Hopper. Recursive's `rec` engine is MLX/Metal. Unsloth's `iq1-narrow` branch contains no Flint code at all.

### Lever summary

Expected effects are this note's estimates (claims to test) unless marked measured.

| Lever | Status (2026-10-22) | Expected effect on this rig | Effort |
|---|---|---|---|
| Remove the MoE down-projection rescale for Flint | No switch; one-line patch; upstream plan is to skip it on backends with BF16/F32 activation support (no CUDA PR) | +5–12 % single stream. Exact with Q8_0 / IQ4_NL / IQ4_XS / Q6_K / NVFP4 / MXFP4 down-projections; otherwise needs an activation-range check | Low (rebuild) |
| Parallel CUDA `POOL_1D` reduction | Metal equivalent merged 2026-10-20 ([#30862](https://github.com/ggml-org/llama.cpp/pull/30862)); CUDA kernel is one thread per output | +1–4 %, keeps the rescale's safety | Low–medium |
| n-gram self-speculation | Mainline | 2–5× on file-rewrite output, ~1.0× on chat and reasoning | Low |
| More parallel slots (`-np 3`–`4`) | Mainline | Combined throughput 1.5–2.5× one stream (2 slots measured 1.5–1.75×) | Low |
| Unsloth Q8_0 everywhere (42.3 GB, 9 cards) | Published | 22–26 tok/s (−5 to −20 %) at near-reference quality; rescale removable | Low (copy) |
| Custom layout: Q4_0 gate/up experts, IQ4_NL down, Q4_0 attention q/o, Q8_0 k/v, shared expert, head | `llama-quantize --tensor-type` | 26–30 tok/s alone, 28–33 with the rescale removed; quality to measure | Medium |
| Recursive NVFP4 GGUF (22.9 GB, quantization-recovered) | Published | 20–31 tok/s (NVFP4 decode efficiency on sm_61 unknown); rescale removable | Low–medium |
| Expert count 6 → 5 / 4 | `--override-kv flint.expert_used_count=int:N` | +2–8 %; quality loss expected | Low |
| MTP / EAGLE-3 / DFlash / draft model | None exist; no same-tokenizer model | n/a without training a head (days of GPU time) | Very high |
| Backend sampling `-bs` | Mainline | +8 % measured; disabled per request when a grammar (tool calls, JSON schema) is attached | Done |
| CUDA graphs on sm_61 | Open PR #27721 | +14 % measured | Done |
| Pascal MMVQ patches ([#25479](https://github.com/ggml-org/llama.cpp/pull/25479), [#30334](https://github.com/ggml-org/llama.cpp/issues/30334)) | Open | +0–5 % | Medium |
| Fused QKV tensor (GGUF surgery) | Flint's loader accepts `attn_qkv` | +2–4 % | Medium |
| KV cache type at long context | Mainline; native q8_0 TILE kernel open ([#30792](https://github.com/ggml-org/llama.cpp/pull/30792)) | q8_0 likely slower than f16 without #30792; with it, possibly +10–30 % at 32k+ | Low / medium |
| Context checkpoints in llama-server | Mainline | Avoids full re-reading of long agent prompts (minutes at ~160 tok/s) | Low |
| Card count, clock mode | Configuration | 5 vs 9 cards measured equal; PowerMizer 0–5 % | Low |
| `-sm tensor` | Flint not on the deny list | Loss expected (98 host-staged AllReduces per token) | — |
| ik_llama.cpp | Flint merged 2026-10-22 | Not usable on a host without AVX2 | — |

---

## 0. The model, its lineage, and where the 36 ms per token go

### Takeaway
Flint is a from-scratch model with its own tokenizer. Its architecture shares only generic parts with Qwen3.6-35B-A3B: hidden size 2048, 256 experts of width 512, a shared expert. Everything llama.cpp does for Qwen3.6 (MTP, Gated DeltaNet kernels, Qwen drafters) is irrelevant.

Per token, Flint reads ~1.66 GB of weights in the Q4_1 file. At 64–73 GB/s (73 GB/s is the single-card Q4_0 rate measured on this rig) that is 23–26 ms of the measured 35.8 ms. The remaining ~10–13 ms is fixed cost per kernel launch: about 1,900 launches per token, of which ~470 belong to the activation rescaling (section 2).

Two consequences:
- Removing launches is worth as much as removing bytes.
- Because attention projections are 42 % of the active body, the quant layout of attention matters more for speed than that of the experts.

### Cited Findings
**Model card, config and weights** ([RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), single commit "Initial release" 2026-10-21; P)
- "Flint was trained from scratch through pretraining, mid-training, supervised fine-tuning, and reinforcement learning."
- 39.8 B total parameters; "approximately 2.5B active per token, including the output head"; "The active body alone is approximately 2.2B parameters".
- [config.json](https://huggingface.co/RecursiveSI/flint/blob/main/config.json):
  - `FlintForCausalLM`; 49 layers, hidden 2048; 32 query heads, 4 KV heads, head_dim 128.
  - `layer_types` gives full attention on layers 0, 1, 5, 9, …, 45, 48 (14 layers) and a 513-token sliding window on the other 35.
  - `no_rope_layer_indices` lists exactly the 14 full-attention layers: RoPE (theta 10000) only on sliding layers.
  - `mlp_only_layers` [0, 1] (dense FFN 6144); 256 experts, `num_experts_per_tok` 6, `moe_intermediate_size` 512, `shared_expert_intermediate_size` 1024.
  - `router_score_function` "sigmoid", `norm_topk_prob` true, `softmax_type` "learnable" (attention sinks); vocab 150144, `bos_token_id` 150000; untied embeddings.
  - The schema uses Qwen2/3-MoE field names (`decoder_sparse_step`, `mlp_only_layers`, `norm_topk_prob`, `shared_expert_intermediate_size`, `router_aux_loss_coef`).
- [model.safetensors.index.json](https://huggingface.co/RecursiveSI/flint/blob/main/model.safetensors.index.json), 36,734 tensors (my check):
  - Layers 0–48 only; no `mtp`, `nextn`, `eagle` or draft tensors.
  - Router is `mlp.gate.weight` without a selection bias; no shared-expert gate; per-layer `self_attn.softmax_offset` (the sinks), `q_norm`, `k_norm`.
  - Total 79.57 GB in BF16.
- [tokenizer.json](https://huggingface.co/RecursiveSI/flint/blob/main/tokenizer.json) (my check):
  - Byte-level BPE with 150,000 tokens and 149,744 merges (256 byte tokens plus merges, the signature of a tokenizer trained from scratch). 144 added tokens from id 150000: `<|begin_of_text|>`, `<|end_of_text|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`, `<think>`, `<tool_call>`, `<arg_key>`, `<arg_value>`, reserved ids.
  - Its pre-tokenizer splits digits in groups of 1–3, then CJK runs, then applies a GPT-4o-style regex.
  - Against the tokenizers of Qwen3.6-35B-A3B, GLM-4.7 and DeepSeek-V3, 68.5 %, 66.0 % and 55.0 % of Flint's token strings also occur, but only 273, 263 and 8 tokens sit at the same id. That is essentially the byte tokens, so there is no id compatibility.
- [chat_template.jinja](https://huggingface.co/RecursiveSI/flint/blob/main/chat_template.jinja): GLM-4.x-style roles and tool calls (`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`). Its `replace("\x00", "")` needed llama.cpp [PR #30929](https://github.com/ggml-org/llama.cpp/pull/30929) "jinja: support hexadecimal string escapes" (merged 2026-10-21).

**Recursive's blog and quantized releases**
- [Introducing Recursive Flint](https://www.recursive.com/updates/introducing-recursive-flint) (2026-10-21; P):
  - The team "built it from scratch in under four months".
  - Architecture search: "Starting from a baseline with 35 billion total parameters and 3 billion active parameters", the search ended at 40B total / 2.5B active.
  - The lm_head was quantized to 4 bits, and 4-bit quality was recovered with "an offline distillation stage". The final NVFP4 model is "within 0.7 points of the 16-bit model".
  - Its speed chart (an image; decode at 1K context, values read off the chart) shows llama.cpp on an RTX Spark at ~96.6 tok/s for Flint against ~72.7 for Qwen3.6-35B-A3B (1.33×). The rig's ratio is 27.9 / 19.6 = 1.42×.
- [RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf) (2026-10-21; P):
  - The Q4_1 file stores "the MLX export's trained INT4 codes" with FP16 scales and offsets; embeddings and other floating-point tensors stay F32.
  - Recommended flags include `-b 256 -ub 128 --cache-ram 0` and "Leave `--swa-full` unset".
  - The MLX source export ([RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4), config `mlx_export`) is group-64 affine INT4 with `"scale_rule": "mse"`, INT8 embeddings.
- [RecursiveSI/flint-nvfp4-gguf](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf) (2026-10-21; P): 22.9 GB, "primarily intended for NVIDIA GPUs"; tested only on Blackwell parts.
- Recursive's SGLang kit ([sglang/README.md](https://huggingface.co/RecursiveSI/flint/blob/main/sglang/README.md), 2026-10-21; P) says speculative decoding is "not validated" and ships no draft model. It gives the KV cost as 28 KiB per token on the 14 full-attention layers plus 70 KiB per token on the 35 sliding layers for the last 513 tokens.

**llama.cpp support** ([PR #30928](https://github.com/ggml-org/llama.cpp/pull/30928), opened 2026-10-21 by Recursive, merged 2026-10-21 as `a37cc461`, first release b11803; P)
- Body: "Uses the adaptive expert-activation scaling from #30861 with a 1,024 threshold" and "Auto-scaling leads to slowdown of 10-30%", to be addressed in follow-up PRs. Validation used H200, B300, RTX PRO 6000 and M4 Pro; no Pascal.
- A maintainer approved it, calling the rescale a workaround "until we have a better fix".
- The converter `conversion/flint.py` subclasses the Qwen2-MoE converter. A new pre-tokenizer type "flint" was added.
- GGUF keys, read from the header of Recursive's Q4_1 file (my check):
  - `flint.expert_used_count = 6`, `flint.expert_gating_func = 2` (sigmoid), `flint.expert_weights_norm = true`, `flint.expert_weights_scale = 1.0`.
  - `flint.expert_shared_count = 2` (the 1024-wide shared expert counted as two of width 512), `flint.leading_dense_block_count = 2`.
  - `flint.attention.sliding_window = 513`, plus boolean arrays `flint.attention.sliding_window_pattern` and `flint.attention.rope_pattern`.
  - `tokenizer.ggml.add_bos_token = false`. No key controls the rescale.
- The graph ([src/models/flint.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/flint.cpp)) is a standard pre-norm block:
  - Separate Q/K/V matmuls, or one fused `attn_qkv` if the GGUF has it. Per-head Q/K RMSNorm, RoPE on sliding layers only, flash attention with sinks via the iSWA cache.
  - `build_moe_ffn` for the routed experts plus `build_ffn` for the shared expert, summed.
  - The graph records every layer input (`t_layer_inp`), the hook EAGLE-3/DFlash drafters use.

**On this rig** ([Flint benchmark](../../benchmarks/2026-10-22-flint.md), [small MoE overhead](../../benchmarks/2026-10-22-small-moe-overhead.md), [calibration](../../benchmarks/2026-10-22-single-card-calibration.md))
- Small MoE models cost ~10–11 µs per graph node per token without CUDA graphs, with the GPU 95–98 % busy.
- Single-card bandwidth efficiency on Llama-2-7B: Q8_0 54 %, Q4_0 42 %, Q4_K_M 35 % of 173 GB/s.
- Card count made no difference for Flint (27.92 on 5 cards, 28.09 on 9).

### Inferences
**Parameters read per token (my calc from the config):**

| Part | Parameters per token | Share of active |
|---|---|---|
| Attention q + o (49 × 2 × 2048 × 4096) | 822 M | 33 % |
| Attention k + v (49 × 2 × 2048 × 512) | 103 M | 4 % |
| Routed experts (47 × 6 × 3 × 2048 × 512) | 887 M | 35 % |
| Shared expert (47 × 3 × 2048 × 1024) | 296 M | 12 % |
| Dense FFN, layers 0–1 | 75 M | 3 % |
| Router (47 × 2048 × 256) | 25 M | 1 % |
| Output head (150,144 × 2048) | 308 M | 12 % |
| Token embedding (host, one row per token) | 308 M | — |

**GPU bytes read per token, by file (my calc from tensor types in each GGUF header):**

| File | Size | GPU bytes per token | Layout | Router / head |
|---|---|---|---|---|
| Recursive Q4_1 | 25.99 GB | 1.66 GB (attention 578 MB, experts 554, head 192, shared 185, router 99, dense 47) | Q4_1 everywhere; token embedding F32 (1.23 GB host page cache) | F32 / Q4_1 |
| Recursive NVFP4 | 22.86 GB | 1.45 GB | NVFP4 everywhere plus one F32 scale per tensor; embedding BF16 | BF16 / NVFP4 |
| bartowski Q4_0 (imatrix) | 22.83 GB | 1.78 GB | Q4_0 experts (5 of 47 down-projections Q4_1); attn_q Q6_K in 25 layers; k/v Q8_0; shared Q8_0/Q5_0 | F32 / Q6_K |
| bartowski Q4_1 (imatrix) | 25.18 GB | 1.88 GB | as Q4_0 but Q4_1 | F32 / Q6_K |
| bartowski IQ4_NL (imatrix) | 25.32 GB | 2.35 GB | IQ4_NL experts with Q6_K/Q5_K in ~11 layers; attention, shared, dense Q8_0 | F32 / Q8_0 |
| Unsloth UD-Q4_K_XL | 25.05 GB | 2.21 GB | experts Q5_K / IQ4_XS / Q4_K; attention Q8_0 (42 layers) or Q5_K | BF16 / Q8_0 |
| ggml-org Q4_K_M | 23.39 GB | 2.26 GB | experts pure Q4_K; everything else Q8_0 | F32 / Q8_0 |
| Unsloth Q8_0 | 42.30 GB | 2.70 GB | Q8_0 everywhere | BF16 / Q8_0 |

**Time budget of the current token** (Q4_1, 5 cards, CUDA graphs, 35.8 ms; my calc):
- Weights: 1.66 GB at 64–73 GB/s (the Q4_0 single-card rate, 73 GB/s, is the upper end) ≈ 23–26 ms.
- Fixed cost: the remaining ≈ 10–13 ms. A consistency check: the same split with UD-Q4_K_XL (2.21 GB, 52.6 ms on 9 cards) implies 52–56 GB/s, close to the 60 GB/s measured for Q4_K_M on one card.
- Graph size, estimated from the graph code:
  - ~73–75 nodes per MoE layer, ~3,500 per token. That is similar to Qwen3.6-35B-A3B's measured 3,727.
  - After CUDA fusion on sm_61, about 40 kernel launches per MoE layer. Attention takes ~14–16: fused norms, three Q8_1 quantize + matvec pairs for Q/K/V, the KV write, flash attention, output projection, residual.
  - The FFN takes ~24: norm, router matvec, fused top-k, fused gate+up matvec, 10 rescale kernels, down matvec, fused expert sum, shared-expert matvecs, fused adds.
  - In total ≈ 1,900 launches per token, so ≈ 5–7 µs per launch.
- CUDA graphs removed 40.7 − 35.8 = 4.9 ms, ≈ 2.5 µs per launch of CPU-side cost.
- Layer boundaries: 9 cards (8 hops) ran as fast as 5 (4 hops), so hops are not a visible cost.
- KV cache: 28 KiB per token of context on the 14 full-attention layers plus ~50 MB for the sliding-window cells. Negligible at short context; at 8k ≈ 0.23 GB, at 32k ≈ 0.94 GB per generated token.
- The 8k → 32k slowdown (38.9 → 51.3 ms for +0.70 GB of KV) implies ~57 GB/s for attention reads.

**Ceiling.** With zero fixed cost the Q4_1 file would run at 1.66 GB / 64–73 GB/s ≈ 36–45 tok/s. vLLM's low-latency path, seven kernels per MoE layer (section 7), shows that this kind of fusion is possible in principle.

### Gaps
- No measured graph node count or kernel count for Flint on the rig. A verbose run prints `graph nodes`; `nsys` would count kernels but needs host RAM.
- Bandwidth efficiency of Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 and NVFP4 on these cards is unmeasured; only Q4_0, Q4_K_M and Q8_0 were calibrated.
- Tokenizer origin is stated nowhere by Recursive. "Fresh BPE" is an inference from the merge count and the id comparison.

---

## 1. Speculative decoding

### Takeaway
- **Learned drafters: none exist.** Flint has no MTP layer, and no EAGLE-3, DFlash or DSpark head has been published. No small model shares its tokenizer, and llama.cpp refuses draft models whose vocabulary differs.
- **n-gram drafting is what remains,** and it suits Flint's job as a coding agent better than chat: rewriting a file, applying an edit with context, or repeating tool output produces long runs that `ngram-mod`/`ngram-simple` can draft.
- **Flint's batch cost on this rig is favourable.**
  - Two tokens cost 1.0–1.13× one token (my calc from the two-slot run). For Q4_x experts the MoE matmuls stay on the cheap MMVQ path up to 6 tokens per step.
  - So short drafts (≤ 5 tokens) are cheap when rejected.
  - Long drafts (`ngram-mod` defaults 48–64) need ~7–11 accepted tokens to pay for their verify step, which happens on file rewrites but not in reasoning text.
- **Sizing a future drafter.** Synthetic acceptance (`--spec-synth-len`) measures what a drafter of given quality would yield on this rig before anyone trains one.

### Cited Findings
- No drafters:
  - Checkpoint tensor list: no MTP/NextN tensors (section 0).
  - ggml-org's own GGUF repo has "no mmproj, MTP or DFlash sidecars" ([ggml-org/Flint-GGUF](https://huggingface.co/ggml-org/Flint-GGUF), 2026-10-22; P).
  - Requests for MTP, DFlash, EAGLE or a smaller sibling are unanswered in the model's Hugging Face discussions (#2 and #4, 2026-10-21).
  - A search of Hugging Face model names for Flint-related repos (2026-10-20 to 2026-10-22) found quants, MLX/EXL3/AWQ conversions and abliterations, but no drafter, pruned or distilled model.
- Draft-model compatibility: draft and target must match vocabulary type and BOS/EOS, and differ in size by at most 128 tokens; otherwise "draft model vocab type must match target model to use speculation" — `common_speculative_are_compatible` in [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp). There is no cross-vocabulary drafting in mainline.
- n-gram types and flags (all in [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp), defaults from [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)):
  - `--spec-type ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod|ngram-cache` (comma-chainable).
  - `ngram-mod`: n-match 24 / n-min 48 / n-max 64.
  - `ngram-simple` / `ngram-map-k`: size-n 12, size-m 48, min-hits 1.
  - `--spec-default` turns on `ngram-mod` 24/48/64.
  - `-lcs/--lookup-cache-static` and `-lcd/--lookup-cache-dynamic` load n-gram caches into llama-server.
- Earlier results with these methods: gpt-oss-120b (also a sliding-window + sinks model) translating code comments went from 181 to 446 tok/s ([#18471](https://github.com/ggml-org/llama.cpp/pull/18471)). A CRLF/LF mismatch moved acceptance from 0.11 to 0.83 ([#19164](https://github.com/ggml-org/llama.cpp/pull/19164), merged 2026-01-30).
- Another community GGUF runs Flint with `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 3 --spec-ngram-mod-n-max 128`, noting that the n-gram drafter "only helps on repeated text" ([Makdok/Flint-40B-A2.5B-24GB-128k-GGUF](https://huggingface.co/Makdok/Flint-40B-A2.5B-24GB-128k-GGUF), 2026-10-22; S).
- Synthetic acceptance ([PR #27711](https://github.com/ggml-org/llama.cpp/pull/27711), merged 2026-08-27; P):
  - `--spec-synth-len L` or `--spec-synth-rates p0,p1,…` replace draft/target matching with seeded random acceptance "for benchmarking"; output is not valid.
  - It needs an initialized drafter ([tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp)).
- Sequence rollback: speculative decoding needs partial `seq_rm` on the target context, which the iSWA cache supports (the server's `common_context_can_seq_rm` probe; [common/common.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/common.cpp)). Flint has no recurrent state, so none of the host-RAM state checkpoints that hurt hybrid models apply.
- MMVQ limits on sm_61 ([mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu), `get_mmvq_mmid_max_batch_pascal_older`, tuned in [#20905](https://github.com/ggml-org/llama.cpp/pull/20905)):
  - `MUL_MAT_ID` (the expert matmuls) uses MMVQ up to 6 tokens for Q4_0, Q4_1, Q5_0, Q5_1 and IQ4_NL; 5 for Q4_K, Q5_K and IQ4_XS; 4 for Q8_0, Q6_K, MXFP4 and NVFP4. Larger batches go to dp4a MMQ.
  - Dense matmuls use MMVQ up to 8 tokens.
- On this rig the dense Qwen3.8-27B's best MTP setting was 1–2 drafts because a 3-token step cost 1.71× a 1-token step. Qwen3.6-35B-A3B Q4_0's 2- and 4-token steps cost 1.49× and 2.15× ([speed-ups](../../benchmarks/2026-10-22-speedups.md)).

### Inferences
**Flint's verify cost (my calc).**
- Single slot with `-bs`: 41.2 ms per step.
- Two slots: 46.6 ms per step on the code prompt (21.19 / 21.72 tok/s) and 42.3 ms on the explanation prompt (23.64 / 23.63).
- That gives r(2) ≈ 1.03–1.13, against 1.49 for Qwen3.6 on the same rig measured with `-p 2` (different method, so compare with care).
- The routed-expert matvecs for 6 × 512-row experts appear latency-bound on 5 SMs: doubling their work adds ~1–5 ms, not the ~8 ms the extra bytes would take at full bandwidth.

**Short drafts.**
- Model: acceptance a, k drafted tokens, verify step r(k+1)·T1 with T1 ≈ 36–41 ms; n-gram drafting costs ~nothing on the GPU.
- A round yields 1 + (tokens accepted). At r(4) ≈ 1.2–1.4 (claim to test), drafting 3 tokens pays as soon as about one of them is accepted on average.
- Draft lengths up to 5 keep the verify batch at ≤ 6 tokens, on MMVQ for Q4_x experts. For Q8_0 or NVFP4 experts the limit is 3 drafts (4-token batch).

**Long drafts.**
- A 64-token verify runs as a prompt batch: at the measured 161–171 tok/s pp512 that is ~0.38–0.5 s, ≈ 10–13 T1.
- On a file rewrite, where whole drafts are accepted, a round could deliver ~60 tokens in ~0.5 s, i.e. 100+ tok/s for those segments against ~24 (claim to test).
- On reasoning text, `ngram-mod`'s 24-token match rarely triggers, so the cost there is close to zero; a mismatched long draft wastes ~0.4 s.

**Two drafting regimes worth testing:**
- Short: `--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 4`.
- Long: `--spec-type ngram-mod` (defaults), or chained with the short one: `--spec-type ngram-mod,ngram-simple`. The priority is fixed and n-gram types come first.

**Slots and drafts interact.** Two slots drafting 3 tokens each make an 8-token batch, past the MMVQ limit for expert matmuls.

**Static lookup.** A static n-gram cache built from the repository being edited (`llama-lookup-create` → `-lcs`) could raise acceptance for code that appears in neither prompt nor output. Untested anywhere for Flint.

**A learned drafter's ceiling.**
- An EAGLE-3 head for Flint would be one 2048-wide layer plus a trimmed head: ~0.1–0.2 GB and a few dozen nodes, so ~3–5 ms per draft token on this rig.
- Gain = synthetic-acceptance speed minus draft cost. Measuring the first part costs an hour (experiment 12); training the head costs days of GPU time on other hardware.

### Gaps
- No direct r(b) measurement for Flint (`llama-bench -p 1,2,…,8 -n 0`).
- No measured n-gram acceptance on Flint's agentic output (tool calls with file contents, edits, reasoning).
- Whether llama-server's SWA cache always keeps enough cells to roll back a long rejected draft without re-reading the prompt: the cache holds n_swa + n_ubatch cells, padded to 256.

---

## 2. The adaptive expert-activation scaling (PRs #30861 and #30928)

### Takeaway
**What it does.** Before the routed-expert down-projection, llama.cpp measures each expert-token's largest activation. It divides the activations by a power of two so the largest is ≤ 1024, runs the matmul, and multiplies the result back. The round trip is exact.

**Why it exists.** CUDA's integer matmul kernels quantize activations to Q8_1 blocks of 32 values that store the block sum in FP16. For weight formats whose dot product uses that sum (Q4_0, Q4_1, Q5_0, Q5_1), a sum above 65,504 becomes infinity and the token's output becomes NaN. Threshold 1024 × 32 values = 32,768 keeps the sums in range.

**What it costs.** Eleven graph ops per MoE layer:
- abs, a `POOL_1D` max, scale, log, scale, ceil, clamp, scale, exp, div, and a mul after the matmul.
- That is ≈ 470 extra kernel launches per token on CUDA, about a quarter of all launches (my estimate).
- The CUDA `POOL_1D` kernel uses one thread per output: for Flint's decode shape that is 6 threads, each looping over 512 values.

**Switching it off.** There is no flag, environment variable or GGUF key; the list of architectures is hard-coded.

**When removal is safe.** The scale factor is exactly 1 whenever activations are ≤ 1024, so dropping the scaling changes nothing for such tokens. For larger activations the outcome depends on the down-projection's weight format:
- Exact (no overflow possible): weight formats whose CUDA kernels never read the Q8_1 sum, in both the decode and prompt paths: Q8_0, Q6_K, IQ4_NL, IQ4_XS, MXFP4, NVFP4.
- Safe for decode only: Q4_K and Q5_K. Their decode kernels skip the sum, but their prompt kernels store it.
- Unsafe: with Q4_0, Q4_1, Q5_0 or Q5_1, any token whose block sums exceed FP16 produces NaN, and the NaN then sits in the KV cache.
- How large Flint's activations get has not been published.

**Upstream direction.** The plan is to mark the matmul as needing BF16/F32 activations and skip the rescale on backends that support that. No CUDA work toward this is under review, and Pascal has no BF16 hardware.

### Cited Findings
- [PR #30861](https://github.com/ggml-org/llama.cpp/pull/30861) "llama : scale moe down projection per expert token" (opened 2026-10-20, merged 2026-10-21; P):
  - It replaces a global-max rescale with a per-expert-token power-of-two factor from a `POOL_1D` max reduction, undone on the output.
  - It applied to Mistral 4 and Mellum. The maintainer tested it on CUDA ("all good").
  - Comment on 2026-10-21: some quantization implementations "use the Q8_1 to quantize the activations", and its F16 block sums "can still overflow"; the scaling would be made more conservative. Another contributor suggested BF16 for the Q8_1 scale.
- [PR #30928](https://github.com/ggml-org/llama.cpp/pull/30928) (merged 2026-10-21; P):
  - Adds `LLM_ARCH_FLINT` to the condition and changes the threshold from 1/32768 to 1/1024 with the code comment "we leave x32 headroom for the F16 sums of 32 values in Q8 activation blocks".
  - States the 10–30 % slowdown, measured on H200/B300/RTX PRO 6000/M4 Pro.
- Code ([src/llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp), `build_amax_src1_f16` and `build_moe_ffn`; P):
  - `if (arch == LLM_ARCH_MISTRAL4 || arch == LLM_ARCH_MELLUM || arch == LLM_ARCH_FLINT)` guards the rescale.
  - A TODO says it "should only be applied when the corresponding backend does not support" BF16 activation precision on the matmul, linking the plan comment in [#30656](https://github.com/ggml-org/llama.cpp/pull/30656#issuecomment-6269597922) (2026-10-19).
  - That plan: graph-level rescale as a stopgap, then auto-detect backends with src1 precision support and skip it there.
- CUDA kernels (P):
  - [vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh): `vec_dot_q4_0/q4_1/q5_0/q5_1_q8_1` take the full `ds` half2 (scale and sum). `vec_dot_q8_0`, `iq4_nl`, `iq4_xs`, `q6_K`, `mxfp4` and `nvfp4` read only `__low2float(ds)`, the scale.
  - [mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh) `mmq_get_q8_1_ds_layout`: Q4_0, Q4_1, Q5_1, Q4_K and Q5_K use the DS4 layout (scale + sum); Q5_0, Q8_0, Q6_K, IQ4_NL, IQ4_XS, MXFP4 and NVFP4 use D4 (scale only).
  - [pool1d.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/pool1d.cu): one thread per output element with a serial loop over the window, blocks of 256 threads.
- Measured behaviour of the overflow:
  - [Issue #30704](https://github.com/ggml-org/llama.cpp/issues/30704) (open, 2026-10-17): with activations of 1e5–3e5 on CUDA's MMVQ path, q8_0, mxfp4, iq4_nl and q4_K gave correct results while q4_0 and q5_0 gave NaN.
  - The same issue showed the earlier approach (F32 precision flag, CPU fallback) made Mellum 8× slower on CUDA.
- [Issue #30950](https://github.com/ggml-org/llama.cpp/issues/30950) (open, 2026-10-21): a second FP16 overflow in Q4_1/Q5_1 matvecs (offset × block sum multiplied in FP16) made gpt-oss emit stray "G" tokens.
  - That path is compiled only with `FAST_FP16_AVAILABLE`, which excludes compute capability 6.1 ([common.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh)), so sm_61 multiplies in FP32 and is not affected.
- The Metal backend's naive `POOL_1D` took 247 µs for a [2048, 8, 1] max on an M2 Ultra; a cooperative reduction cut it to 6.35 µs ([PR #30862](https://github.com/ggml-org/llama.cpp/pull/30862) "metal : optimize argmax and pool_1d", merged 2026-10-20; P). No CUDA counterpart exists.
- Root-cause fixes, both open:
  - [#22571](https://github.com/ggml-org/llama.cpp/pull/22571) "Swap out F16 for BF16 in Q8_1 activations" (since 2026-05-01): about 1 % slower on an RTX 3090; the author asked for pre-Ampere tests. A reviewer said it should be done for all backends at once.
  - [#30260](https://github.com/ggml-org/llama.cpp/pull/30260) "cuda : support required FP32 activations in MUL_MAT_ID" (since 2026-10-10): FP32 cuBLAS per expert with a stream sync, no reviews. That is an unattractive path for decode on Pascal.
- Related: [#30924](https://github.com/ggml-org/llama.cpp/pull/30924) (open, 2026-10-21) removes Metal's in-kernel rescale because the graph-level one makes it redundant.
- ik_llama.cpp's Flint port ([ik PR #2725](https://github.com/ikawrakow/ik_llama.cpp/pull/2725), merged 2026-10-22; P) has no such rescale. Its author had no NVIDIA GPU, so CUDA was not exercised.

### Inferences
**Cost on this rig (my calc).**
- 10 rescale kernels × 47 layers = 470 launches.
- At the ≈ 5–7 µs per launch implied by section 0 that is 2.3–3.3 ms. The serial `POOL_1D` (512 dependent-latency loads per thread, from L2) plausibly adds 0.2–1.4 ms more.
- Total ≈ 2.5–4.7 ms of 35.8 ms, so removing it gives +7–15 %; a conservative +5–12 % is the claim to test.
- The trailing `mul` likely fuses with the router-weight `mul`, since CUDA's multi-mul fusion takes same-layout operands.

**How to recover it, by safety:**
1. **Exact.** Down-projection weights in Q8_0, IQ4_NL, IQ4_XS, Q6_K, MXFP4 or NVFP4, plus a patch dropping `LLM_ARCH_FLINT` from the condition. This covers Unsloth's Q8_0 and Recursive's NVFP4 file as published, and any custom quant with `ffn_down_exps` in IQ4_NL (section 5).
2. **Keeps the safety.** Rewrite the CUDA `POOL_1D` kernel as a warp/block reduction (as Metal did), cutting the serial part; the other nine tiny kernels remain. Gain +1–4 %.
3. **Root fix for Pascal.** Store the Q8_1 sum scaled by 1/32 (exact, a power of two) in the CUDA quantize kernel and multiply back by 32 in the Q4_0/Q4_1/Q5_x dot products and the DS4 MMQ loads. That raises the overflow point 32× at zero cost and lets the rescale go even for Q4_x files. BF16 sums (#22571) would do the same with a small conversion cost. Both are local patches, 0.5–2 days of work.
4. **Unsafe without data.** Removing the rescale on the Q4_1 file. The factor is 1 for every expert-token whose activations stay ≤ 1024, so if Flint's activations never approach ~2,000 nothing changes. If they do, those tokens become NaN. Measure the activation range first (experiment 6).

**Quality.** Options 1–3 are numerically exact or better than today. Option 4 is either harmless or catastrophic per token, never a gradual loss.

**Speed measurements need no safety.** A speed A/B can use the Q4_1 file even with the unsafe patch, because llama-bench's throughput does not depend on the values.

### Gaps
- The distribution of `ffn_moe_down` input magnitudes in Flint is unpublished.
- No per-op timing of `POOL_1D`, `ABS`, `LOG`, `EXP`, `CEIL`, `CLAMP`, `SCALE` and `DIV` on a 5-SM card; `test-backend-ops perf` would give it, but the rig's build script omits tests.
- Whether the rescale's `mul` fuses on sm_61, and whether the gate/up MMVQ fusion is a net win on 6.1 (#30334 found ±5 % on a P40).

---

## 3. Expert-count reduction

### Takeaway
- **How.** `--override-kv flint.expert_used_count=int:N` changes Flint's experts per token; the key name was read from the GGUF.
- **Speed.** It saves little on this rig:
  - Routed experts are a third of the bytes per token, so 6 → 4 removes ~0.18 GB of 1.66 GB (11 %) and 6 → 5 ~0.09 GB (5.6 %), with no fewer kernel launches.
  - The two-slot data suggests the expert matvecs are partly latency-bound, so even that saving may not fully appear. Expected +2–8 % (claim to test).
- **Quality.** With the router renormalized over the kept experts (`norm_topk_prob`), the expert branch's strength shifts. The literature shows real losses at a one-third cut that only a modified normalization recovers, and that needs a code change. Wikitext perplexity understates the loss on code.

### Cited Findings
- GGUF key `flint.expert_used_count = 6`; `flint.expert_weights_scale = 1.0` is read as optional by Flint's loader ([src/models/flint.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/flint.cpp)). llama-bench has no `--override-kv` ([tools/llama-bench/llama-bench.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp)); llama-server, llama-cli and llama-perplexity do.
- Flint's graph passes the scalar `n_expert_used`. Per-layer `expert_used_count` arrays (loadable since [#25444](https://github.com/ggml-org/llama.cpp/pull/25444), merged 2026-09-03) would not take effect without code changes.
- No mainline flag for threshold-based or per-layer expert reduction exists. Related open PRs: [#18029](https://github.com/ggml-org/llama.cpp/pull/18029) `--moe-n-expert` (since 2025-12-14) and [#26631](https://github.com/ggml-org/llama.cpp/pull/26631) letting `MUL_MAT_ID` skip index −1 (since 2026-08-05; CUDA not reviewed). [#30593](https://github.com/ggml-org/llama.cpp/pull/30593) `--moe-cache-topk` was closed unmerged on 2026-10-15.
- ik_llama.cpp's Smart Expert Reduction `-ser Kmin,t` ([ik PR #239](https://github.com/ikawrakow/ik_llama.cpp/pull/239), 2025-03-02) has been a no-op since a 2025-10 rewrite of its expert selection. The project's guidance (2026-01-20) is to use `--override-kv` instead ([ik discussion #1099](https://github.com/ikawrakow/ik_llama.cpp/discussions/1099)).
- Published numbers:
  - Chen & Yao, [arXiv 2609.04575](https://arxiv.org/abs/2609.04575) (2026-09-04; P), on Qwen3.6-35B-A3B (256 experts, top-8, renormalized): 8 → 4 experts loses 4.65 MMLU points with standard renormalization, but 0.35 when normalizing by the probability mass of the top 16 while running 4. They also report that "perplexity and downstream accuracy can favor different" normalization settings.
  - DeepSeek-V3-family model at Q8_0, wikitext: PPL 3.3470 with 8 experts, 3.3952 with 6 (+1.4 %). Lowering `expert_weights_scale` from 2.5 to 2.35 brought it to 3.3809, on the reasoning that fewer summed expert outputs have a smaller norm ([comment on #11446](https://github.com/ggml-org/llama.cpp/pull/11446#issuecomment-2794491074), 2025-04-10; S).
  - Qwen3-30B-A3B static top-k (Oncescu et al., [arXiv 2511.02237](https://arxiv.org/abs/2511.02237), 2025-11-04; P): GPQA 60.2 / 56.2 / 54.3 and LiveCodeBench 62.1 / 63.2 / 58.2 at k = 8 / 5 / 4; AIME24 holds to k = 5, then falls to 75.8 at k = 4.
  - Yang et al., [arXiv 2505.03531](https://arxiv.org/abs/2505.03531) (2025-05-06; P): on sigmoid-routed DeepSeek-V3 a uniform 6 → 4 lowered the benchmark average from 72.9 to 70.6, and sigmoid weights "tend to polarize toward 0 or 1".
  - A controlled comparison of dynamic expert-skipping methods found plain static top-k hard to beat ([arXiv 2610.28369](https://arxiv.org/abs/2610.28369), 2026-10-20; P).
  - Kimi/GLM-5.2 code tests: top-1 token agreement 89.6–91.2 % keeping 6 of 8 experts and 81.9–83.3 % keeping 4. The thread concludes that wikitext perplexity makes dropping experts look nearly free while code text shows the cost ([ik discussion #2106](https://github.com/ikawrakow/ik_llama.cpp/discussions/2106), 2026-07; S).
- No expert-count experiment on Flint has been published.

### Inferences
- **Expected on this rig (my calc).** 6 → 5 experts: −92 MB per token, about +2–4 %; 6 → 4: −185 MB, +4–8 %. The ratio of quality lost to speed gained is worse than for any quant change in section 5.
- **Cheap approximation of the top-k₂ normalization.** Run k₁ experts with a constant scale below 1 standing in for "mass of top-4 ÷ mass of top-6": `--override-kv flint.expert_used_count=int:4 --override-kv flint.expert_weights_scale=float:0.8`. Sweep 0.7–0.95. The exact method needs a `build_moe_ffn` patch: select top-6, sum their weights, keep the first 4.
- **Quality evaluation.** Use code text and a task check, not wikitext alone.

### Gaps
- No Flint data at any k. No measurement of how much of the expert matvec time scales with the number of experts on 5 SMs.

---

## 4. Kernel-level levers on Pascal

### Takeaway
- **Flint's ops already run on fast sm_61 paths:**
  - Flash attention with sinks: the TILE kernel for f16 KV, VEC for quantized KV.
  - A fused top-k router for 256 sigmoid experts with weight normalization.
  - Fused gate+up+SwiGLU matvecs for routed and shared experts (the MMVQ fusion is enabled for 6.1).
  - Fused RMSNorm+RoPE(+KV write), chained adds.
- **Not available to Flint:** the shared-expert + routed-expert fusion (it requires equal widths; Flint's are 1024 and 512).
- **Remaining levers, roughly in order of size:**
  - Remove or speed up the rescale (section 2).
  - Cut launches by fusing Q/K/V into one tensor.
  - Small Pascal MMVQ patches.
  - Graph chunking (≤ 2 %).
  - At long context, a quantized KV cache through the TILE kernel.
- **No Flint-specific performance PR** was open or merged in llama.cpp after #30928 as of 2026-10-22.

### Cited Findings
**Attention**
- Kernel choice ([fattn.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn.cu) `ggml_cuda_get_best_fattn_kernel`; P): without tensor cores and with head size 128, decode (1 query) uses:
  - the TILE kernel for f16 K/V when the GQA optimization applies (ratio 8 here);
  - the VEC kernel for quantized K/V.
  - Both apply sinks ([fattn-tile.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-tile.cuh), [fattn-vec.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-vec.cuh)). sm_61 has no fast FP16, so the math is FP32.
- The sliding-window cache holds `n_swa·(unified ? n_seq : 1) + n_ubatch` cells padded to 256 ([src/llama-kv-cache-iswa.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache-iswa.cpp)). That is 768 cells at `-ub 128`, about 50 MB for all 35 sliding layers.
- `--swa-full` makes the sliding layers keep the whole context ([common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)); Recursive says to leave it unset.
- [PR #30792](https://github.com/ggml-org/llama.cpp/pull/30792) (open, 2026-10-19; P): the VEC kernel "does not reuse K/V across the GQA heads", so quantized KV "ends up slower than an f16 one at long context". The PR routes q8_0 K/V into the GQA-aware TILE kernel.
  - On a gfx1151 at 63k context: q8_0 KV 31.2 → 50.5 tok/s, against 43.4 for f16.
  - Its new loader asserts FP16 accumulation, so whether it covers sm_61's FP32 TILE path needs checking.
- On the rig, Qwen3.6's q8_0 KV cost 1–2 tok/s against f16 ([Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)).
- Community KLD for Flint KV types: q8_0 KV +6 % KLD; q4_0 KV "+67 %", with the advice not to use it ([Makdok card](https://huggingface.co/Makdok/Flint-40B-A2.5B-24GB-128k-GGUF), 2026-10-22; S).

**MoE matmuls and fusions** ([ggml-cuda.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu); P)
- `ggml_cuda_should_fuse_mul_mat_vec_q` disables MMVQ fusion only for `cc <= GGML_CUDA_CC_PASCAL` (600), so 6.1 keeps it ([issue #30334](https://github.com/ggml-org/llama.cpp/issues/30334), open since 2026-10-11, measured fusion off as ±5 % depending on quant on a P40).
- `ggml_cuda_match_shared_expert` fuses routed and shared gate/up only when the shared expert has the same width as a routed expert. Flint's (1024 vs 512) does not qualify.
- Top-k MoE fusion covers sigmoid gating, an optional selection bias, and weight normalization, for power-of-two expert counts up to 512 ([topk-moe.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/topk-moe.cu)).
- Chains of up to 8 adds or muls fuse into one kernel.
- The small-K MMVQ variant is used for Q4_x on Pascal; it is excluded only for IQ3_S, Q2_K and Q3_K on older NVIDIA ([mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)). It matters here because the down-projection's K is only 512.
- Each MMVQ call quantizes its input to Q8_1 with a separate kernel, so three separate Q/K/V matmuls launch three quantize kernels.
- Flint's loader accepts a fused `blk.N.attn_qkv.weight` if present (`create_tensor_qkv`, [src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)). No published Flint GGUF has one. ik_llama.cpp fuses QKV at load time with `-mqkv` ([ik PR #2725](https://github.com/ikawrakow/ik_llama.cpp/pull/2725)).

**Launch overhead**
- CUDA graphs: [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) (open since 2026-08-25) changes the gate from Volta to DP4A; +14 % for Flint on the rig.
- [PR #30804](https://github.com/ggml-org/llama.cpp/pull/30804) "CUDA: chunk graphs to amortize graph launch" (open, 2026-10-19): "1-2% TG speed-up"; the author saw no improvement multi-GPU.
- [PR #30662](https://github.com/ggml-org/llama.cpp/pull/30662) (open, 2026-10-16) stops the graph cache from evicting a graph that is about to be used.
- [PR #25479](https://github.com/ggml-org/llama.cpp/pull/25479) (open since 2026-07-09): 2 warps instead of 4 for single-token MMVQ on Pascal DP4A; +2.5 to +5.4 % decode on a GTX 1060 (geomean +3.1 %).
- vLLM's Flint work cut a decode step from 1,096 kernels to 346. For up to 8 requests its low-latency path runs each MoE layer as "seven kernels", with Q/K norm, RoPE, routing and the expert combine fused into the neighbouring matvec kernels ([vLLM blog](https://vllm.ai/blog/2026-10-21-flint), 2026-10-21; S). Blackwell only.

**Tensor split.** Flint is not on the `-sm tensor` deny list ([src/llama-arch.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp) `llm_arch_supports_sm_tensor`), and #30861 taught the meta backend about `POOL_1D`. On Pascal the AllReduce falls back to NCCL or a host-staged butterfly ([dense_27b_speedups.md](dense_27b_speedups.md), section 2).

### Inferences
- **Fused QKV (my calc).** Concatenating `attn_q`, `attn_k` and `attn_v` rows into one `attn_qkv` saves 2 matvecs and 2 quantize launches per layer, ≈ 196 launches per token, ≈ 1–1.4 ms (+3–4 %).
  - In Recursive's Q4_1 file all three are Q4_1, so the merge is a byte-level row concatenation (each row is an independent run of blocks) with no requantization.
  - Files with Q8_0 K/V and 4-bit Q need either a common type or no merge.
- **Graph chunking (#30804).** Unlikely to matter across 5–9 cards, where each card's graph is already a chunk.
- **Pascal MMVQ patches.** #25479 targets exactly the single-token MMVQ that dominates Flint, so expect +2–4 %. Turning the gate/up fusion off for 6.1 (#30334) is a toss-up to measure.
- **Shared expert as routed experts.** Appending the shared expert to the routed tensor as two extra 512-wide experts with fixed weight 1 would let one `MUL_MAT_ID` cover both: ≈ 4–6 launches fewer per layer, ≈ +4–6 %. That needs GGUF surgery plus a graph change, about 1–2 days.
- **Long context (my calc).** At 32k the full-attention KV read (0.94 GB) is 57 % of the weight bytes.
  - On sm_61 a q8_0 cache halves the bytes but loses GQA reuse, since VEC reads each KV head once per query head (8×, partly from L2). So expect f16 ≥ q8_0 today.
  - With a q8_0 TILE path (#30792, if it applies to sm_61): ~19.5 → ~22–24 tok/s at 32k and larger gains at 64–128k (claim to test).
  - Never use `--swa-full` here: the sliding layers would then read up to the whole context.
- **`-sm tensor`.** 2 AllReduces per layer × 49 = 98 per token. At ~0.1–0.3 ms each through host memory that is 10–30 ms, against a 36 ms token: a loss.

### Gaps
- No kernel-level profile of Flint on Pascal (launch count, per-kernel times).
- Whether #30792 compiles for and helps sm_61.
- No Pascal data for #30804 or #30662.

---

## 5. Quantization for speed with quality

### Takeaway
**Two facts make Flint unusual:**
- Its routed experts are 95 % of the file but only a third of the bytes per token.
- Its official 4-bit files are quantization-recovered checkpoints whose distance from BF16 is "by design".

**The speed-optimal direction** is therefore the opposite of the usual MoE recipe: keep attention, shared expert and head lean (they are read every token), and spend spare bits on routed experts (cheap per token). Use only simple block formats (Q4_0, Q4_1, Q5_0, Q8_0, IQ4_NL) that the 5-SM cards unpack quickly.

**Rescale-safe experts.** Putting `ffn_down_exps` in IQ4_NL or Q8_0 makes removing the activation rescale exact (section 2).

**Measuring quality.** KLD against a BF16-derived reference is fine for ranking BF16-derived quants against each other. Comparing them with Recursive's Q4_1 or NVFP4 needs a task check (tool calling, coding), because those files were deliberately moved away from BF16 to recover task quality.

### Cited Findings
- **Byte tables:** section 0 (GGUF headers).
- **Community comparison against BF16** ([terelabs/flint-GGUF](https://huggingface.co/terelabs/flint-GGUF) README and `comparison.csv`, 2026-10-21; S). KLD on code text (ctx 2048, 40 chunks), plus 300 BFCL v3 tool-calling tasks with greedy decoding:
  - Recursive Q4_1: KLD 0.675, top-1 80.4 %, BFCL 81.0 %. Annotated as quantization-aware-trained INT4 that "deliberately differs from BF16", so KLD and top-1 measure that distance rather than quality.
  - UD-Q4_K_XL: 0.389 / 86.5 % / 80.0 %. Their imatrix Q4_K_M: 0.440 / 85.8 % / 78.3 %. lmstudio Q4_K_M (no imatrix): 0.668 / 82.1 % / 80.3 %. Their IQ4_XS: 0.473 / 85.3 % / 78.7 %. BF16 itself: 79.7 %.
  - "BFCL accuracy is statistically indistinguishable from BF16 for every file" (±4.5 points at 95 %).
  - Their recipe keeps attention, dense FFN, shared expert, embedding and head at Q8_0 and compresses only routed experts.
- **Another community comparison** (wikitext, against BF16; [romanerm/flint-GGUF](https://huggingface.co/romanerm/flint-GGUF), 2026-10-22; S): KLD Q4_K_M 0.259, IQ4_NL 0.329, IQ4_XS 0.344.
- **Imatrix files:**
  - Unsloth's `imatrix_unsloth.gguf_file` (225 MB, 687 chunks, 531 entries), used for its UD and Q8_0 files.
  - bartowski's `RecursiveSI_flint-imatrix.gguf` (451 MB, 501 chunks) and calibration text.
  - terelabs' `imatrix.gguf` (451 MB; ~890k tokens of tool-calling dialogues, code and prose) — repos above, plus [bartowski/RecursiveSI_flint-GGUF](https://huggingface.co/bartowski/RecursiveSI_flint-GGUF) (2026-10-22).
  - bartowski also publishes imatrix Q4_0 (22.83 GB) and Q4_1 (25.18 GB).
- **`llama-quantize` behaviour** ([src/llama-quant.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp), [tools/quantize/quantize.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp); P):
  - `--tensor-type NAME=TYPE` patterns are lower-cased and matched with `std::regex_search` against full tensor names, first match wins. Manual overrides bypass the mixture logic.
  - `--output-tensor-type` and `--token-embedding-type` set the head and embedding.
  - `ffn_gate_inp` (the router) is never quantized and keeps its source type: BF16 in Unsloth's files, F32 in Recursive's and bartowski's.
  - `--allow-requantize` permits starting from Q8_0.
  - The repo's [scripts/quantize_docker.sh](../../../scripts/quantize_docker.sh) wraps this.
- **Flint tensor names** in llama.cpp GGUFs (from the Q4_1 header):
  - `blk.N.attn_{q,k,v,output}.weight`, `blk.N.attn_{q,k}_norm.weight`, `blk.N.attn_sinks.weight`.
  - `blk.{0,1}.ffn_{gate,up,down}.weight` (dense).
  - `blk.N.ffn_gate_inp.weight`, `blk.N.ffn_{gate,up,down}_exps.weight` (3-D, 256 experts), `blk.N.ffn_{gate,up,down}_shexp.weight`.
  - `output.weight`, `token_embd.weight`.
- **MXFP4 and NVFP4 on sm_61:** both have dp4a + lookup-table MMVQ and MMQ paths in mainline ([vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh), [mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh)); there is no hardware FP4.
  - NVFP4 decodes an FP8 (E4M3) scale every 16 values; MXFP4 a power-of-two scale every 32.
  - A community MXFP4_MOE Flint file exists ([prithivMLmods/flint-MXFP4_MOE-40B-A2.5B](https://huggingface.co/prithivMLmods/flint-MXFP4_MOE-40B-A2.5B), 2026-10-22; S).
  - Flint's quantization-aware targets were group-64 INT4 (MLX) and NVFP4, not MXFP4.

### Inferences
**Bytes per token for candidate layouts** (my calc; F = file size on GPUs):

| Layout | Bytes per token | F | Rescale removable exactly | Expected speed alone / with rescale removed (claim to test) |
|---|---|---|---|---|
| Recursive Q4_1 (today) | 1.66 GB | 24.8 GB | No | 27.9 measured / — |
| A: gate/up experts Q4_0, down IQ4_NL, attn q/o Q4_0, k/v Q8_0, shared Q8_0, head Q8_0, dense Q8_0 | 1.84 GB | 22.6 GB | Yes | 26–30 / 28–33 |
| B: as A but attention all Q8_0 (the usual recipe) | 2.25 GB | 23.0 GB | Yes | 22–26 / 24–28 |
| C: experts Q4_1, attn q/o Q5_0, k/v, shared, head Q8_0 | 2.00 GB | 25.1 GB | No | 24–28 / — |
| D: gate/up experts Q5_0, down Q8_0, rest Q8_0 | 2.47 GB | 32.5 GB | Yes | 22–26 / 23–28 |
| E: Unsloth Q8_0 | 2.70 GB | 42.0 GB | Yes | 22–26 / 23–28 |
| Recursive NVFP4 | 1.45 GB | 22.2 GB | Yes | 20–31 / 22–34 |

Model used: t = 10–13 ms fixed + bytes ÷ η, with η ≈ 64–73 GB/s for Q4_x, 85–93 for Q8_0 (single-card calibration), 52–56 for K/IQ mixes, and 40–65 for NVFP4 (unmeasured; per-16 FP8 scale decode is heavier than Q4_0).

What the table suggests:
- **Q8_0 is the quality-first option.** It is predicted only ~5–20 % slower than Q4_1, because Q8_0 is the most efficient format per byte on these cards and attention dominates either way. It needs 8–9 cards and leaves ~1 GB per card for KV and buffers (~9 GB ≈ 300k tokens of f16 KV across 9 cards).
- **Layout A is the speed-first BF16-derived option.** Whether 4-bit attention q/o costs quality is the open question: Recursive's own 4-bit files quantize attention to 4 bits and recover it by distillation; a plain imatrix quant does not.
- **bartowski's Q4_0 gains nothing over the official file** (1.78 vs 1.66 GB per token, predicted 25–28.5 tok/s): its Q6_K head and Q6_K `attn_q` add bytes in a K-quant format.
- **The router:** going from F32 to BF16 saves 49 MB per token (−3 %). It happens automatically when quantizing from Unsloth's BF16/Q8_0 files, but cannot be done to Recursive's file with `llama-quantize`.
- **The F32 embedding of the official file** costs 1.23 GB of host page cache; a requantized Q8_0 embedding is 0.33 GB. That matters for the 4 GB host's headroom (slots, checkpoints), not for speed.
- **MXFP4 for BF16-derived weights.** Its power-of-two block scale is coarser than Q4_0's FP16 scale, so expect higher error at ~6 % fewer bytes (claim to test). Low priority.

**Exact commands for layout A,** from Unsloth's Q8_0 with Unsloth's imatrix (quantizing from BF16 avoids the small double-quantization error). Patterns are tried in order:

```bash
llama-quantize --allow-requantize --imatrix imatrix_unsloth.gguf_file \
  --tensor-type 'ffn_down_exps=iq4_nl' \
  --tensor-type 'ffn_(gate|up)_exps=q4_0' \
  --tensor-type '_shexp=q8_0' \
  --tensor-type 'blk\.[01]\.ffn_(gate|up|down)\.weight=q8_0' \
  --tensor-type 'attn_(k|v)\.weight=q8_0' \
  --tensor-type 'attn_(q|output)\.weight=q4_0' \
  --output-tensor-type q8_0 --token-embedding-type q8_0 \
  flint-Q8_0.gguf flint-A.gguf Q4_0
```

Variants:
- Layout C: swap the expert lines for `'ffn_(gate|up|down)_exps=q4_1'` and use `'attn_(q|output)\.weight=q5_0'`.
- Layout B: `'attn_(q|k|v|output)\.weight=q8_0'`.
- Through the repo script: `EXTRA="--tensor-type … --output-tensor-type q8_0" scripts/quantize_docker.sh <src> <models> flint-Q8_0.gguf flint-A.gguf Q4_0 imatrix_unsloth.gguf_file`.

### Gaps
- No Q4_0, Q4_1, Q5_0 or IQ4_NL imatrix KLD for Flint against a common reference.
- No task-level comparison of Recursive's Q4_1 and NVFP4 against BF16-derived quants beyond BFCL (n = 300).
- No efficiency measurement of NVFP4 or IQ4_NL on 5-SM cards.

---

## 6. Multi-GPU layout and serving

### Takeaway
- **Card count and micro-batch:**
  - Card count does not matter for Flint (5 = 9 measured), so use the cards for context and slots.
  - `-ub`/`-b` do not affect decode speed; keep `-ub 128` for long contexts on the 4 GB host.
- **Batching is the cheapest throughput lever.** Two slots measured 1.5–1.75× combined; three or four slots should stay on the MMVQ path (≤ 6 tokens per step for Q4_x experts) and keep scaling. Use this for agents that run sub-tasks in parallel.
- **Why Qwen3.6's two slots did badly (my reading).** It was measured on the older build without `-bs`, so each extra slot probably paid ~11 ms of CPU sampling over 248k logits, which matches its +12.5 ms per step. Flint's run had `-bs`, a smaller vocabulary and CUDA graphs. Re-running Qwen3.6 with `-bs` would test this (claim to test).
- **Backend sampling** (+8 %) silently switches off for any request with a grammar. Agent clients send tools, and tool-call grammars count.
- **Context checkpoints** decide whether a long agent conversation is re-read from scratch at ~160 tok/s.

### Cited Findings
- **Measured on the rig** ([Flint benchmark](../../benchmarks/2026-10-22-flint.md)):
  - 5 vs 9 cards: 27.92 vs 28.09 tok/s.
  - Two slots with `-bs`: 21.19 / 21.72 (code) and 23.64 / 23.63 (explanation) tok/s.
  - 262,144 context on 9 cards uses 33.9 of 54 GB.
- **Qwen3.6's two slots:** 15.5 → ~13 tok/s each, on build `85d5703a3` without graphs and without `-bs` ([Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)). On that rig llama-server was 16–18 % slower than llama-bench, attributed to CPU sampling.
- **Backend sampling** is disabled when a grammar or reasoning budget is present ("backend sampling is not compatible with grammar, disabling", [common/sampling.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp)) and when `n_probs > 0` ([tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp)).
- **Tool calls:** Flint's tool calls follow the `<tool_call>…<arg_key>…<arg_value>` template, which llama.cpp's automatic template parser handles ([common/chat.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp)).
  - When a request defines tools with `tool_choice` auto, the parser builds a lazy grammar that triggers on the tool-call marker ([common/chat-auto-parser-generator.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/chat-auto-parser-generator.cpp)).
  - Creating that grammar sampler (and the reasoning-budget sampler that a lazy grammar enables) is enough to switch backend sampling off for the whole request ([common/sampling.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp)).
- **Checkpoints and memory:**
  - `-ctxcp/--ctx-checkpoints` (default 32 per slot), `-cms` minimum spacing, `--cache-ram` ([common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)).
  - The server forces full prompt re-processing when SWA cache data are missing ([server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp)). Fixes to the SWA checkpoint logic merged on 2026-10-15 ([#30580](https://github.com/ggml-org/llama.cpp/pull/30580)) and 2026-10-19 ([#30666](https://github.com/ggml-org/llama.cpp/pull/30666)) are in the rig's build.
  - The repo's server scripts pass `--cache-ram 0 --ctx-checkpoints 0` ([setup guide](../../setup.md#running-llama-server-on-a-4-gb-host)).
- **Slot sharing:**
  - `-kvu/--kv-unified` shares one KV buffer across slots.
  - Open [#30970](https://github.com/ggml-org/llama.cpp/pull/30970) (2026-10-22) would share prompt-prefix cells between slots.
- **Clocks:** the power limit is fixed between 60 and 75 W. Clock-mode experiments are described in [dense_27b_speedups.md](dense_27b_speedups.md), section 5.

### Inferences
- **Slots (my calc).** The routed-expert matvecs read ~0.55 GB per token for 6 experts. With s slots, distinct experts per layer grow sublinearly and the rest of the step is shared, so combined throughput should keep rising to s = 4–6. Per-slot speed falls slowly: perhaps ~18–21 tok/s each at s = 4 (claim to test).
  - Beyond 6 tokens per step, expert matmuls move to MMQ.
  - Each full 262k context costs ~7 GiB of f16 KV. Two such slots already use 3.9–5.5 GB per card, so three or four slots need a smaller context each (e.g. `-np 4 -c 524288`, 131,072 per slot), or `-kvu` with a shared budget.
- **Checkpoints for Flint (my calc).** A checkpoint holds only the sliding-window state: 35 layers × ~641–768 cells × 2 KiB ≈ 45–55 MB. That is small next to a hybrid model's ~150 MB recurrent state.
  - With `--ctx-checkpoints 0`, any mismatch between the cached tokens and the next request (re-rendered reasoning, retokenized output, a client trimming history) forces re-reading the whole prompt: ~3 min for a 30k-token agent context at ~160 tok/s.
  - 4 checkpoints cost ~200 MB of host RAM. Re-enabling them is the biggest end-to-end latency lever for agent use, though it does not change tok/s.
- **Backend sampling and tools.** When a request carries tools, expect the served speed to drop back toward the plain 22.5 tok/s (−8 %); check the log for the warning. A sampler path that applies the (lazy) grammar on the GPU does not exist in llama.cpp today.
- **Server overhead.** Even with `-bs`, served decode (41.2 ms) is 5.4 ms slower per token than llama-bench (35.8 ms); the cause is unidentified. Candidates on the 2-core host: the HTTP/streaming thread, detokenization, server-side sampler bookkeeping. Measure with llama-cli/llama-completion at the same flags.
- **Power mode.** Each card computes only ~1/5 to 1/9 of every token. Holding P0 clocks with PowerMizer (runtime, under the running Xorg) might shave a few percent (claim to test).

### Gaps
- No measurement of 3+ slots for Flint, of Qwen3.6's two slots with `-bs`, or of the server-vs-bench gap's cause.
- No measurement of served speed for tool-carrying requests (lazy grammar, CPU sampling) from the agent clients Recursive ships configs for (OpenCode, pi).

---

## 7. Other engines and forks

### Takeaway
None is usable or faster on this rig:
- **ik_llama.cpp** supports Flint and skips the rescale, but needs AVX2 on the host, and its CUDA graphs are disabled below Ampere.
- **vLLM and SGLang** have Flint support aimed at Blackwell and Hopper; Pascal is irrelevant.
- **Ollama** wraps llama.cpp; **MLX** and Recursive's **`rec`** are Apple-only.
- **Unsloth's `iq1-narrow` branch** adds three sub-IQ1 quant types and contains no Flint code.

Ideas worth porting into mainline for this rig:
- A cheaper or removable rescale (section 2).
- Load-time QKV fusion (ik's `-mqkv`).
- The vLLM-style fusion of norm, RoPE, routing and expert combine into neighbouring matvecs.
- `rec`'s multi-token resident decode with GPU sampling, which is Metal-specific.

### Cited Findings
- **ik_llama.cpp** ([ik PR #2725](https://github.com/ikawrakow/ik_llama.cpp/pull/2725), merged 2026-10-22; P):
  - Ported from mainline #30928 with RoPE pattern, sinks, two dense layers plus 47 MoE layers, sigmoid gating, shared expert, and the fast paths `-muge`, `-mqkv`, `-rtr`, split-mode graph and `--swa-compress`.
  - Only CPU numbers were posted; CUDA was not exercised; no rescale.
  - The README says only Turing+ CUDA and AVX2/NEON CPUs are fully supported. An AVX1 build PR ([ik PR #2138](https://github.com/ikawrakow/ik_llama.cpp/pull/2138)) has been open since 2026-07-15 with NaN reports.
  - Its Pascal status page ([docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md), 2026-10-14) says CUDA graphs are not used below Ampere.
  - The rig's Celeron has no AVX at all ([hardware spec](../../mining_rig.md)).
- **Unsloth:**
  - The guide ([unsloth.ai/docs/models/flint](https://unsloth.ai/docs/models/flint), updated 2026-10-22; S) recommends UD-Q4_K_XL (or UD-Q3_K_XL), temperature 1.0 / top-p 0.95 / top-k 20, and building from `git clone --branch iq1-narrow https://github.com/unslothai/llama.cpp`.
  - That branch ([tree](https://github.com/unslothai/llama.cpp/tree/iq1-narrow), head dated 2026-10-14, 7 commits ahead of a 2026-10-09 upstream base) adds IQ1_XS/IQ1_XXS/IQ1_XXXS (1.19–1.44 bpw) with CPU and CUDA kernels.
  - It has no Flint architecture; Unsloth's Flint port lived in a separate PR ([unslothai/llama.cpp #314](https://github.com/unslothai/llama.cpp/pull/314), 2026-10-21) that is superseded by upstream.
  - No Unsloth Flint GGUF uses the new types; no Q4_0/Q4_1 files are published there.
- **vLLM:** [PR #62938](https://github.com/vllm-project/vllm/pull/62938) (open, 2026-10-21). "The Flint kernels target a single Blackwell GPU (TP=1, BF16 KV cache)" ([vLLM blog](https://vllm.ai/blog/2026-10-21-flint)). SGLang: [PR #45624](https://github.com/sgl-project/sglang/pull/45624) (open, 2026-10-21) for H200/B200/B300. Neither runs on compute capability 6.1.
- **Ollama:** Flint support first arrived for macOS via MLX in v0.40.8; the llama.cpp-based runner picked up b11803 in a 2026-10-22 pre-release ([issue #19186](https://github.com/ollama/ollama/issues/19186) and release notes). It gives no speed advantage over llama.cpp itself.
- **`rec`** ([recursive-oss/rec](https://github.com/recursive-oss/rec), created 2026-10-21; P):
  - Built on MLX 0.32 with custom Metal kernels: a persistent decode megakernel, "expert teams" across the 6 routed experts and the shared expert, up to 8 tokens per GPU dispatch with GPU sampling.
  - No speculative decoding (`num_draft_tokens=0`) and no expert reduction.
  - The README claims 1.4–2.6× llama.cpp's throughput on an M4 Pro. Apple M4 Pro / M5 Pro / M5 Max only.
- **MLX:** [mlx-lm PR #1998](https://github.com/ml-explore/mlx-lm/pull/1998) is open (2026-10-21).

### Gaps
- No Flint CUDA run of ik_llama.cpp anywhere, so its numerical behaviour without the rescale is unknown.

---

## Experiment ideas

Ranked by expected gain per hour of effort. Each expected outcome is a claim to test, not a prediction to rely on.

**Conventions for every experiment:**
- Record the build commit and patches, file checksum, cards used, context size and flags.
- Use llama-bench for raw speed. Use [scripts/server_bench.sh](../../../scripts/server_bench.sh) for served speed (four fixed chat prompts at temperature 0, 256 tokens) and [scripts/parallel_bench.sh](../../../scripts/parallel_bench.sh) for slots.
- Measure quality off the rig with [scripts/quality_kld.sh](../../../scripts/quality_kld.sh), plus a task check (a fixed set of coding and tool-calling prompts compared with the Q8_0 reference) whenever Recursive's quantization-recovered files are involved.
- Log `nvidia-smi` clocks, power and throttle reasons.
- Keep `-ub 128` at long context and watch `MemAvailable` on the 4 GB host.

1. **Rescale removal: speed A/B and per-op cost.**
   - Hypothesis: the 470 rescale launches per token cost 2.5–4.7 ms; removing them gives +5–12 % single-stream decode.
   - Setup:
     - Build the CUDA-graphs variant twice: stock, and with `|| arch == LLM_ARCH_FLINT` deleted from the condition in `build_moe_ffn` (src/llama-graph.cpp).
     - Speed: `llama-bench -ngl 99 -fa on -p 0 -n 128 -r 5` on 5 cards with the Q4_1 file, alternating builds. Speed is valid even though the patch is numerically unsafe for Q4_1.
     - Also build `test-backend-ops` (`-DLLAMA_BUILD_TESTS=ON`) and run `test-backend-ops perf -b CUDA0 -o POOL_1D`, then `ABS,LOG,EXP,CEIL,CLAMP,SCALE,DIV,MUL` on one card.
   - Measure: tg128 delta; µs per op; their sum × 47 compared with the delta.
   - Expected: +5–12 %; `POOL_1D` the slowest single op.
   - Effort: 1–2 h (build on the PC).
   - Risk: none for the measurement. Do not serve the patched build with Q4_x down-projections until experiment 6 says it is safe.

2. **n-gram self-speculation on agent-style output.**
   - Hypothesis: short drafts gain on code and lose nothing on chat; long drafts gain 2–5× on file rewrites.
   - Setup:
     - Prompt sets: (a) the four standard chat prompts; (b) "return this 2–4k-token file with one function renamed"; (c) a recorded agent turn with a tool call that writes a file.
     - Configurations on llama-server with `-bs`:
       - (i) none;
       - (ii) `--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 4`;
       - (iii) `--spec-type ngram-mod` (defaults 24/48/64);
       - (iv) `--spec-type ngram-mod,ngram-simple` with (ii)'s sizes.
   - Measure: served tok/s per prompt set, drafted and accepted counts from the log.
   - Expected: (ii) +0–20 % on (b)/(c), ±2 % on (a); (iii) 2–5× on (b), ~1.0× on (a).
   - Effort: 1–2 h.
   - Risk: low.

3. **More slots.**
   - Hypothesis: combined throughput keeps rising to 4 slots while each stays above ~18 tok/s.
   - Setup:
     - `llama-server … -bs -np {2,3,4} -c {262144,393216,524288}` (131,072 tokens per slot), and `-np 4 -kvu -c 524288`.
     - Send 2/3/4 concurrent requests with parallel_bench.sh, extended to N clients.
   - Measure: per-request and combined tok/s, per-card VRAM, host RAM.
   - Expected: 4 slots ≈ 65–85 tok/s combined.
   - Effort: 1 h.
   - Risk: low; watch KV memory per card.

4. **Batch-cost curve and graph size (instrumentation).**
   - Hypothesis: for Flint a b-token step costs ≈ 1.0–1.15× (b = 2), 1.1–1.4× (b = 4) and 1.3–1.7× (b = 6) of one token, with a jump at b = 7 (MMQ).
   - Setup:
     - `llama-bench -ngl 99 -fa on -p 1,2,3,4,5,6,7,8,16,64 -n 0 -r 3` on the Q4_1 and Q8_0 files.
     - One verbose run per file to read `graph nodes`.
   - Measure: r(b), node count.
   - Expected: as hypothesised; ~3,500 nodes.
   - Effort: 0.5 h per file.
   - Risk: none.
   - Use: sets draft lengths for experiments 2 and 12, and the slot count for 3.

5. **Unsloth Q8_0 on 9 cards, with and without the rescale.**
   - Hypothesis: Q8_0 runs 22–26 tok/s (23–28 with experiment 1's patch, which is exact for Q8_0) at near-reference quality.
   - Setup:
     - Copy `flint-Q8_0.gguf` (42.3 GB) to the rig's model drive.
     - `llama-bench -ngl 99 -fa on -p 512 -n 128 -r 3` on 9 cards, and the server benchmark with `-bs` at a context that fits (try 262,144; fall back to 131,072).
   - Measure: tg128, served tok/s, per-card VRAM.
   - Expected: −5 to −20 % against Q4_1; the rescale removal recovers about half of that.
   - Effort: 2–3 h (copy dominates).
   - Risk: low; tight VRAM on the head card.

6. **Measure Flint's down-projection activation range** (off the rig).
   - Hypothesis: the largest `ffn_moe_down` input magnitudes stay below 2,048 on code, chat and tool-call text. If so, removing the rescale is a no-op even for Q4_x files.
   - Setup:
     - CPU build on the PC with a callback name added in `build_amax_src1_f16` (`cb(amax, "ffn_moe_amax", il)`).
     - Run `llama-eval-callback` over ~100k tokens of mixed text with the Q8_0 file; collect the maxima per layer.
     - Then run llama-perplexity on the rig (GPU, 10 chunks, `-ub 128`) for Q4_1 with experiment 1's patched build against the stock build.
   - Measure: max and 99.99th percentile per layer; GPU PPL equality.
   - Expected: unknown. Either "safe with margin" (then serve the patched build) or a list of layers that exceed (then use experiment 8 or a rescale-safe layout).
   - Effort: 2–3 h.
   - Risk: none.

7. **Single-card format calibration.**
   - Hypothesis: Q4_1 and IQ4_NL decode at ≈ Q4_0's 42 % of bandwidth; Q5_0/Q5_1 at 36–40 %; NVFP4 and MXFP4 at 30–40 %.
   - Setup: quantize Llama-2-7B (as in the [calibration](../../benchmarks/2026-10-22-single-card-calibration.md)) to Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 and NVFP4; run `llama-bench -p 0 -n 128 -r 3` on one card.
   - Measure: tok/s × GPU bytes per token.
   - Expected: as hypothesised.
   - Effort: 1–2 h.
   - Risk: none.
   - Use: picks the layout for experiment 9 and predicts experiment 10.

8. **Parallel `POOL_1D` reduction for CUDA.**
   - Hypothesis: a warp/block reduction (one block per output row, as in Metal's #30862) recovers 1–4 % while keeping the rescale.
   - Setup: patch `pool1d.cu` with a reduction path when the window is ≥ 64; check with `test-backend-ops -o POOL_1D`; repeat experiment 1's benchmark.
   - Measure: µs per op, tg128.
   - Expected: +1–4 %.
   - Effort: 2–3 h.
   - Risk: low.

9. **Custom layout A (and C) from Unsloth's Q8_0.**
   - Hypothesis: layout A decodes at 26–30 tok/s on the stock build and 28–33 with the rescale removed (exact, since its down-projection is IQ4_NL). Its KLD against Q8_0 is well below the official Q4_1's.
   - Setup:
     - Quantize with the commands in section 5 on the PC; run experiment 4's curve, llama-bench and the server benchmark on 5 cards.
     - Quality: quality_kld.sh against Q8_0, plus the task check against Q8_0 and the official Q4_1.
   - Measure: tg128, served tok/s, KLD, task results.
   - Expected: as hypothesised. If 4-bit attention costs too much on the task check, layout C is the fallback.
   - Effort: 4–6 h.
   - Risk: low (quality to be measured).

10. **Recursive's NVFP4 file.**
    - Hypothesis: it runs at 20–31 tok/s (22–34 with the rescale removed, which is exact for NVFP4), with the best task quality of the 4-bit options.
    - Setup: download `flint-nvfp4.gguf` (22.9 GB); llama-bench and the server benchmark on 5 cards, stock and patched builds; task check.
    - Measure: tg128, served tok/s, task results.
    - Expected: as hypothesised; the decisive unknown is NVFP4 matvec efficiency on 5 SMs (experiment 7).
    - Effort: 2–3 h.
    - Risk: low.

11. **Context checkpoints for multi-turn agent use.**
    - Hypothesis: with `--ctx-checkpoints 4`, turn 2+ of an agent conversation reuses the cache; with 0 it is sometimes re-read in full.
    - Setup: replay a 5-turn conversation that grows to ~30k tokens through llama-server with `--ctx-checkpoints 0` and with `4`, using the client's real request format (tools, reasoning content).
    - Measure: per-turn time to first token, prompt tokens processed per turn (log), host RAM.
    - Expected: minutes saved per turn when a mismatch occurs; ~200 MB extra host RAM.
    - Effort: 1–2 h.
    - Risk: low.

12. **Synthetic acceptance: what a drafter would be worth.**
    - Hypothesis: with Flint's flat batch cost, a drafter reaching a mean of 3 tokens per round (2 of 3 drafts accepted) would raise served speed to ~45–60 tok/s before its own cost.
    - Setup: `llama-server … -bs --spec-type ngram-simple --spec-ngram-simple-size-n 1 --spec-ngram-simple-size-m {2,3,5} --spec-synth-len L`, with L from 1.8 up to m + 1 (the allowed maximum). The 1-token n-gram key makes a draft available at almost every step; output is invalid by design, so time only.
    - Measure: served tok/s against the plain baseline.
    - Expected: as hypothesised.
    - Effort: 1 h.
    - Risk: none.
    - Use: decides whether training an EAGLE-3 or DFlash head for Flint is worth days of GPU time.

13. **Pascal MMVQ patches.**
    - Hypothesis: 2-warp single-token MMVQ ([#25479](https://github.com/ggml-org/llama.cpp/pull/25479)) adds +2–4 %; disabling gate/up MMVQ fusion for 6.1 ([#30334](https://github.com/ggml-org/llama.cpp/issues/30334)) is ±3 %.
    - Setup: apply #25479's diff; separately change `cc <= GGML_CUDA_CC_PASCAL` to `cc < GGML_CUDA_CC_VOLTA` in `ggml_cuda_should_fuse_mul_mat_vec_q`; A/B against stock on the Q4_1 file.
    - Measure: tg128.
    - Expected: as hypothesised.
    - Effort: 2–3 h.
    - Risk: low.

14. **Fused QKV GGUF.**
    - Hypothesis: merging `attn_q`/`attn_k`/`attn_v` of the Q4_1 file into `attn_qkv` saves ~196 launches per token, +3–4 %.
    - Setup: a gguf-py script (on the PC) that writes a new file with `blk.N.attn_qkv.weight` = byte concatenation of the Q, K and V rows (all Q4_1) and drops the three originals; load-check and llama-bench.
    - Measure: tg128, identical greedy output on fixed prompts.
    - Expected: +3–4 %.
    - Effort: 3–4 h.
    - Risk: low; file-format mistakes show up as load errors or garbage, not subtle damage.

15. **Expert count 5 and 4.**
    - Hypothesis: +2–4 % at 5 experts and +4–8 % at 4, with a measurable loss that a weight-scale override partly recovers.
    - Setup:
      - Speed via llama-server with `--override-kv flint.expert_used_count=int:{5,4}`, or llama-bench on a copy edited with gguf-py's `gguf_set_metadata.py`.
      - Quality via quality_kld.sh with `CAND_ARGS="--override-kv flint.expert_used_count=int:4 --override-kv flint.expert_weights_scale=float:{1.0,0.9,0.8}"` on code text, plus the task check.
    - Measure: tok/s, KLD, task results.
    - Expected: as hypothesised; not worth deploying unless the loss is negligible on the task check.
    - Effort: 2–3 h.
    - Risk: none.

16. **Long-context KV type.**
    - Hypothesis: on sm_61, q8_0 KV is slower than f16 at 32k+ (VEC kernel without GQA reuse); with #30792 it becomes faster.
    - Setup: `llama-bench -ngl 99 -fa on -p 0 -n 64 -d 8192,32768,65536 -ctk f16,q8_0 -ctv f16,q8_0` on 9 cards, stock and with #30792 applied (check that it builds for sm_61).
    - Measure: tg at each depth, VRAM.
    - Expected: as hypothesised.
    - Effort: 2–3 h.
    - Risk: low.

17. **Server overhead and clock mode.**
    - Hypothesis: part of the 5.4 ms per token between served and benchmarked decode is recoverable; holding P0 clocks adds 0–5 %.
    - Setup:
      - llama-completion vs llama-server at identical flags and prompts.
      - `nvidia-settings -a '[gpu:N]/GPUPowerMizerMode=1'` for the cards in use: a runtime setting under the running Xorg that reverts on restart.
    - Measure: per-token time, clocks.
    - Expected: as hypothesised.
    - Effort: 1–2 h.
    - Risk: low; higher idle power.

18. **Qwen3.6 two-slot rerun (diagnostic).**
    - Hypothesis: with graphs and `-bs`, Qwen3.6's two slots lose far less than 15.5 → 13 tok/s, confirming that CPU sampling, not MoE structure, caused the earlier result.
    - Setup: repeat the Qwen baselines' two-slot run on the CUDA-graphs build with `-bs`.
    - Measure: per-slot tok/s.
    - Expected: ≥ 14.5 tok/s each.
    - Effort: 1 h.
    - Risk: none.

19. **Q8_1 sum scaling for Pascal (local patch).**
    - Hypothesis: storing the Q8_1 block sum ÷ 32 (exact) removes the overflow for Q4_0/Q4_1/Q5_x at no cost, so the rescale can go for every file, including the official Q4_1.
    - Setup: patch the CUDA Q8_1 quantize kernels and the Q4_0/Q4_1/Q5_0/Q5_1 MMVQ dot products and DS4 MMQ loads; validate with `test-backend-ops -o MUL_MAT,MUL_MAT_ID` and with #30704's large-activation cases; then experiment 1's benchmark with the rescale removed.
    - Measure: correctness, tg128.
    - Expected: experiment 1's gain, safely, on the Q4_1 file.
    - Effort: 1–2 days.
    - Risk: medium (kernel bugs show as wrong output; test thoroughly).

20. **Bigger fusion project.**
    - Hypothesis: treating the shared expert as two always-on routed experts, plus fusing Q/K norm, RoPE and the router into neighbouring matvecs (vLLM's design), cuts launches per MoE layer from ~40 to ~15. Speed approaches 33–40 tok/s.
    - Setup: graph and kernel work in a fork; measure per change.
    - Expected: highly uncertain.
    - Effort: 1–3 weeks.
    - Risk: high.

21. **A learned drafter for Flint** (only if experiment 12 shows ≥ 1.5×).
    - Hypothesis: an EAGLE-3 head trained on Flint's own outputs (llama.cpp converts heads with `convert_hf_to_gguf.py --target-model-dir` and runs `--spec-type draft-eagle3`) reaches acceptance ~0.6–0.7 at 2–3 drafts.
    - Setup: train on other hardware; convert; test on the rig.
    - Expected: 1.3–1.8× served speed on chat-like output (claim to test).
    - Effort: days plus GPU rental.
    - Risk: high.

Not recommended for testing on this rig, with reasons in the sections above: `-sm tensor` (98 host-staged AllReduces per token), ik_llama.cpp (needs AVX2), vLLM/SGLang (Blackwell/Hopper kernels), `--swa-full` (the sliding layers would read the whole context), q4_0 KV (large measured quality loss).
