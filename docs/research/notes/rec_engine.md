# rec, Recursive's engine for Flint: how it runs the model and what carries over to the P106 rig (2026-10-23)

Scope: [rec](https://github.com/recursive-oss/rec) is the inference engine Recursive, the makers of Flint, publish for
running Flint locally (Apache-2.0, created 2026-10-21, v0.1.0 at `e53663b`; read at commit `0dcf43c`, 2026-10-22). This
note covers what rec is, how it runs Flint (attention, MoE, numerics, weight format), whether it does any speculative
decoding, how it serves requests, the speed it claims, and what could carry over to Flint on nine P106-090 cards with this
project's llama.cpp fork (branch `pascal-p106`). It also draws on Flint's Hugging Face model cards and configs and on
docs.recursive.com. Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference.
- "Claim to test" marks anything not measured on this rig.
- "P" and "S" in source lists mean primary (code, docs, model cards) and secondary.
- `rec_mlx/…` stands for `engine/src/rec_mlx/…` in the rec repository; all line numbers refer to `0dcf43c`.

Starting point on the rig, measured: in a five-turn chat on nine cards at 262,144 tokens of context Flint generates
28–32 tok/s with the fork and 14–16 tok/s with stock llama.cpp ([Q58](../../../quants/experiments.md)); S4-L0 decodes
37.4 tok/s in llama-bench tg128 on six cards (26.8 ms per token).

## Verdict

- **rec runs only on Apple Silicon Macs and cannot run on the rig.** It is a Python engine on MLX whose decode path is a
  hand-written Metal kernel; a Go program installs and launches it. It requires macOS on arm64, an M4 Pro, M5 Pro or M5
  Max GPU and at least 48 GiB of unified memory, keeps the whole model on one GPU, and contains no CUDA, PyTorch, Triton,
  vLLM, SGLang or llama.cpp code.
- **Its speed comes from fusing a whole token into one GPU dispatch.** One persistent kernel runs the embedding, all 49
  layers and the output head, with spin barriers on device atomics between phases, and loops up to 8 tokens per dispatch
  with sampling on the GPU. Its decode attention reads each key and value once for all 8 query heads of a KV head and
  splits the context into 32–1,024 partitions.
  - Claimed result on an M4 Pro: 1.4–2.6× llama.cpp's decode speed, from 131.2 against 88.0 tok/s at 1K of context to
    22.6 against 8.6 at 252K.
  - From the chart, rec reads the KV cache at ~200 GB/s at every depth, the same rate at which it reads the weights.
    llama.cpp falls from ~160 to ~50 GB/s as the context grows (est.).
- **rec has no speculative decoding of any kind, and Flint has no MTP head.**
  - No drafter, MTP, EAGLE, Medusa, n-gram or lookahead code exists; rec sets mlx-lm's draft-token count to 0.
  - The INT4 checkpoint loads strictly into a model class with no MTP module, and the config has no next-n keys.
  - The makers' SGLang kit calls speculative decoding "not validated".
  - Speculation for Flint stays this project's own work: an EAGLE-3 head (Q20) or n-gram drafting (Q53).
- **The 4-bit weights rec loads are the makers' own target format, which explains the group-64 layout of their Q4_1
  GGUF.**
  - The MLX export (`RecursiveSI/flint-int4`) is group-64 affine INT4 with BF16 scale and bias for every projection,
    expert and the output head, INT8 embeddings, and `"scale_rule": "mse"`.
  - rec names it a quantization-aware-trained model (`QATModel`), and the GGUF card says its Q4_1 holds "the MLX
    export's trained INT4 codes".
  - Repacking 64-weight groups into Q4_1's 32-weight blocks duplicates every scale and offset: 5.0 bits per weight
    instead of 4.5 (est.).
- **rec never stores anything in FP16 and never rescales activations.** Hidden states, activations, KV and logits are
  BF16, and every accumulation, including the activation sums of the 4-bit dot products, is FP32. Flint's down-projection
  overflow is therefore an artefact of llama.cpp's FP16 activation block sums. FP32 sums (part of Q38) remove it at the
  source.
- **The makers' serving settings differ from this project's runs in four ways:**
  - min-p 0 (rec rejects any other value; the makers' llama.cpp command passes `--min-p 0`), while llama-server's default
    of 0.05 applied in Q58;
  - `reasoning_content` kept in the history (model card, SGLang kit, GGUF card), while `server_bench.sh PROMPTS=chat`
    resends answers without their reasoning;
  - greedy decoding "not validated", while Q30's BFCL runs were greedy;
  - every turn grammar-constrained, while llama-server constrains only after a tool-call trigger.
- **What carries over:**
  - the attention kernel's layout, as a worked design for Q59/Q60;
  - the makers' sampling and history settings, for benchmarks and task checks;
  - a grammar mask applied on the GPU and computed on the host during the next forward, for tool requests (Q55);
  - FP32 activation sums (Q38) and the shared expert folded into the routed-expert launches (Q50).

  A persistent per-card kernel is the long-term version of rec's main idea; it would take weeks.

## 1. What rec is

**Components** (repository root; P):

| Part | Language | What it does | Where |
|---|---|---|---|
| `cli/` | Go 1.25 | The `rec` command: installs and updates itself; downloads the model with resumable ranges and SHA-256 checks; refuses unsupported platforms; starts, supervises and stops the server through a control socket; issues the local API key; launches pinned pi and OpenCode harnesses. It contains no inference code | `cli/platform.go:20-43`, `cli/manifest.go:83-85`, `cli/download.go:33-123`, `cli/serve.go:85-127, 213-361`, `cli/engine.go:9-15` |
| `engine/` (`rec_mlx`) | Python, Metal Shading Language in Python strings | The engine: model (`model.py`), loader (`quantized.py`, `checkpoint.py`), decode kernels (`kernels/`), KV cache and prompt reuse (`cache.py`, `prefix.py`), sampling, grammar, HTTP server (`server.py`, `scheduler.py`, `anthropic.py`) | `engine/src/rec_mlx/` |
| `engine/native/grammar_native.cpp` | C++ (nanobind) | Two MLX primitives: xgrammar's token mask on a CPU stream, and its hand-off to the GPU | `grammar_native.cpp:44-164` |
| `packaging/` | Python, JSON | A relocatable CPython 3.12 runtime with the engine wheel; the pinned model manifest | `build-mlx-runtime.py`, `flint-int4-mlx.json` |
| `cli/clients/` | TypeScript | pi extension and OpenCode patches | `pi/flint-pi.ts`, `pi/pi-transport.mjs` |

**Compute backends.**
- **Prefill** runs the MLX model op by op. The experts go through mlx-lm's `SwitchGLU`, attention through
  `mx.fast.scaled_dot_product_attention` with sinks. Prompts are processed in chunks of 2,048 tokens (M4 Pro) or 4,096
  (M5 Pro, M5 Max) (`quantized.py:122-136`, `model.py:191-212`, `kernels/device.py:25-40`).
- **Decode** runs rec's own Metal kernel, `kernels/megakernel.py`. Source transforms in `kernels/*.py` specialise it per
  chip and checkpoint, and it is compiled through `mx.fast.metal_kernel`.

**Dependencies** (`engine/pyproject.toml:1-15`): Python ≥ 3.12, `mlx==0.32.0`, `mlx-lm==0.31.3`, `transformers==5.16.1`
(tokenizer only), `xgrammar==0.2.1`; build: hatchling, `nanobind==2.13.0`, clang++. The model manifest validator accepts
only engine `mlx`, OS `darwin`, arch `arm64`, backend `metal` (`cli/manifest.go:83-85`).

**Platform and memory.**
- **Supported hardware:** Apple Silicon only. Both the Go preflight (`cli/platform.go:20-43`, `cli/engine.go:17-24`) and
  the engine (`rec_mlx/engine.py:57-61`) check for it.
  - GPUs: M4 Pro, M5 Pro and M5 Max (`applegpu_g16s`, `applegpu_g17s`; `kernels/device.py:15-40`,
    `kernels/megakernel.py:991`).
  - Memory: at least 48 GiB (`min_ram_bytes` 51,539,607,552, `packaging/flint-int4-mlx.json:12`).
  - The docs list NVIDIA "RTX Spark" support as "coming soon".
- **Weights:** 22.6 GB in five safetensors shards.
- **KV cache:** BF16. The 14 full-attention layers need 14 × 4 KV heads × 128 × 2 (K, V) × 2 B = 28 KiB per token, i.e.
  7.0 GiB at 262,144 tokens (est.). It is allocated in up to four blocks of 65,536 tokens (`cache.py:48-118`). The 35
  sliding layers hold 513 tokens each, 35 MiB in all (est.).
- **Parallelism:** one GPU and one generation at a time. Up to 9 requests are admitted (one runs, the others queue);
  further ones get HTTP 429 (`server.py:32, 106-107`). There is no tensor, pipeline or expert parallelism and no multi-device code. The README calls rec
  "optimized for single-session usage".

## 2. How rec runs Flint

### The model

| Part | Flint, as rec implements it | Where |
|---|---|---|
| Layers | 49: layers 0–1 dense (SwiGLU, intermediate 6,144), 2–48 MoE | config `mlp_only_layers`; `model.py:257-271` |
| Attention | 32 query heads, 4 KV heads, head dim 128; per-head RMSNorm on Q and K before RoPE; a learned per-head sink (`softmax_offset`) | `model.py:142-213`; `megakernel.py:248-266` |
| Layer types | 14 full-attention layers (0, 1, 5, 9, …, 41, 45, 48) with no positional encoding; 35 sliding-window layers of 513 keys including the current token, with RoPE θ = 10,000 rotating dimension pairs (d, d + 64) | config `no_rope_layer_indices`; `quantized.py:60`; `megakernel.py:254-262, 868`; `checkpoint.py:24-25` |
| Router | Sigmoid of FP32 logits; top-6 picked on sigmoid + `expert_bias`, which is filled with zeros when the checkpoint has none (Flint's has none); route weights = the chosen sigmoid scores / (their sum + 1e-20); no scaling factor | `model.py:227-254, 341-344` |
| Experts | 256 routed, intermediate 512; a shared SwiGLU expert, intermediate 1,024, always on, added with weight 1 (no gate) | `model.py:237-254` |
| Head | Vocabulary 150,144, untied, 4-bit | config; `kernels/mixed_bits.py:25` |

### Decode: one persistent kernel per token

- **Weight layout.** At load, every tensor is packed into 13 byte banks of four layers each, with a 64-word descriptor
  per layer (`megakernel.py:1012-1109`).
- **Launch.** `flint_decode` launches 36 threadgroups of 256 threads (M4 Pro, M5 Pro; 76 on M5 Max) that stay resident
  for the whole token: embedding, 49 layers, final norm and output head (`megakernel.py:823-975, 1238-1239`;
  `kernels/device.py:4-40`).
- **Synchronisation.** Phases are separated by device-wide spin barriers on atomics (`sync_all`,
  `megakernel.py:137-159`), or by narrower team barriers: one team per KV head for Q/K/V and attention
  (`kv_teams.py:81-150`), one team per expert for gate/up → down (`megakernel.py:800-820`). A barrier that does not
  complete within 10⁷ spins aborts the token ("GPU synchronization failed", `megakernel.py:1264-1268`): the kernel
  depends on all threadgroups being resident at once.
- **Phases of a layer:**
  1. RMSNorm;
  2. Q, K and V, one group per head, including the head norm, RoPE and the KV write (`megakernel.py:872-888`);
  3. attention partials and merge (892-903);
  4. output projection (905);
  5. residual and norm (907-910);
  6. router and top-6 (912-914);
  7. expert gate/up, a team hand-off, expert down (916-923);
  8. weighted sum into the residual (945-960).
- **Up to 8 tokens per dispatch.** The 8-step loop wraps the whole body (`kernels/resident_block.py:62-88`), and EOS is
  checked inside the kernel (`kernels/sampling.py:26-38`).
  - At temperature 1 with top-p 1 and top-k 0, the kernel samples by Gumbel-max with noise generated beforehand
    (`sampling.py:11-14, 42-44`); at temperature 0 it takes the argmax.
  - Every other setting, including the default 1.0 / 0.95 / 20 and any grammar, runs 8 single-token dispatches inside one
    compiled MLX graph, still sampling on the GPU (`kernels/masked_sampling.py:30-146`).
  - The host receives tokens 8 at a time (`engine.py:273-294`).

### Attention

Decode attention (`megakernel.py:619-693, 868-903, 1133-1146`):
- **Work unit.** One SIMD group (32 lanes) per KV head, context partition and group of query heads. Below 1,024 tokens a
  group has 4 query heads; above, all 8 of the KV head (`query_heads=4`, `long_heads=8`, `long_context=1024`).
  - Each lane owns 4 of the 128 dimensions, so each key and value is read once for every query head that shares it, and
    the query stays in registers.
- **Scores.** Each head's dot product is reduced across the 32 lanes with `simd_sum`, once per key and head. An online
  softmax keeps each head's running maximum and denominator. The next key and value are loaded before the current one is
  used (647-660).
- **Partitions.** Sliding layers and short global layers use 32 partitions. Global layers use 64 at 1,024 tokens, 128 up
  to 8K, 256 up to 32K, 512 up to 64K and 1,024 beyond (890-891). A separate merge phase combines the partials with
  log-sum-exp weights (277-318).
- **Sinks.** Partition 0 starts with the sink as its running maximum and a denominator of 1. That adds exp(sink) to the
  softmax denominator with a zero value vector (631-632).
- **Rounding.** Beyond 1,024 tokens, global-layer partials are rounded to BF16, as MLX's own two-pass kernel does (378,
  890).
- **KV layout.** BF16, keys then values, 4 heads × capacity × 128 per layer. Sliding layers write a 513-slot ring
  (`cache.py:52-118`; `megakernel.py:868-869`).
- **Prefill** uses MLX's fused attention with sinks (`model.py:191-212`).

### MoE routing and expert kernels

- **Router.**
  - The FP32 router weights are converted to BF16 only if the conversion is exact; otherwise loading fails
    (`megakernel.py:85-92`). Flint's router weights are BF16 values stored as FP32.
  - The dot product with the normalised BF16 input is FP32, the logit is rounded to BF16, then the sigmoid is applied
    (`megakernel.py:406-419`; `kernels/mixed_bits.py:32-38`).
- **Top-6.** Each expert team computes only the rank it serves, so no barrier separates selection from expert work
  (`kernels/rank_routing.py:11-47`).
- **Expert teams.** With 32 MLP workers, each routed expert gets 4 threadgroups and the shared expert 8. Gate and up are
  computed together and SiLU(gate)·up is written as BF16. A team-local hand-off follows, then the down projections of all
  seven experts (`megakernel.py:746-820`; `kernels/expert_teams.py`).
- **Combine.** The six weighted routed outputs are summed, the shared expert added, then the residual (945-960). The QAT
  transform rounds each weighted product and each partial sum to BF16 (`mixed_bits.py:39-54`: "QAT casts routing weights
  and each weighted expert product to BF16"; "MLX's strided six-expert sum rounds to BF16 after every addition").
- **3-bit routed experts** are supported (`qat_dot3`, `mixed_bits.py:3-21, 55-109`; `quantized.py:22`). The published
  checkpoint uses 4 bits (`routed_expert_bits: 4`).

### Numerics

- **Precision.** Storage is BF16 throughout (hidden state, activations, KV, logits) and accumulation is FP32. No FP16
  value appears in the decode kernel, and nothing scales activations before the down projection.
- **The 4-bit dot product** (`megakernel.py:171-196`):
  - activations are converted to FP32 and pre-divided by 1, 16, 256 and 4,096, so the masked nibbles of a 16-bit word
    multiply without shifts;
  - the affine group term is scale·Σ(q·x) + bias·Σx, with the activation sum Σx kept in FP32;
  - the down projection and the 3-bit path convert integers to floats exactly through the `0x4b000000` exponent trick
    (`megakernel.py:560-567`; `mixed_bits.py:13-17`).
- **Exactness.** `engine/tests/validate_resident.py:236-289` requires the fused kernel's per-layer hidden states, logits,
  sampled tokens, RNG state and KV cache to equal MLX's op-by-op model bit for bit.
  - The routed SiLU uses fast `exp` only after a startup guard shows identical results for all 65,536 BF16 gate values
    (`kernels/activation_optimization.py:18-122`).
  - The kernel is compiled with `math_mode: safe` (`megakernel.py:1221`).
- **Logits** come from the 4-bit head with FP32 accumulation and are stored as BF16 (`megakernel.py:972, 1256-1257`). The
  makers' SGLang kit instead runs the head in FP32 (`--enable-fp32-lm-head`).

### Weights and quantization

- **What rec loads.** Only `RecursiveSI/flint-int4` (revision `3b93f90`), MLX safetensors, 22.6 GB
  (`quantized.py:16-79`; `checkpoint.py:7-30`).
  - Projections, experts and the output head: group-64 affine INT4, with a BF16 scale and bias per 64 weights.
  - Embeddings: INT8. Norms, sinks and the router stay floating point.
  - The config's `mlx_export` block: `bits 4`, `embedding_bits 8`, `routed_expert_bits 4`, `scale_rule "mse"`.
- **How the weights were made.** rec names the checkpoint a quantization-aware-trained model (`QATModel`, `QATSession`,
  `quantized.py:16, 98`). The INT4 model card does not say how it was produced. The GGUF card says "Q4_1 stores the MLX
  export's trained INT4 codes with FP16 scales and offsets".
- **Group 64 as Q4_1 (est.).** Group-64 affine with BF16 scale and bias costs 4 + 32/64 = 4.5 bits per weight. Q4_1
  stores an FP16 scale and minimum per 32 weights (5.0 bits per weight). Each group of 64 therefore becomes two Q4_1
  blocks with the same scale and offset, which is the layout found in the makers' GGUF. The repack is exact wherever the
  BF16 scale and bias are representable in FP16.
- **No other formats or tools.** rec has no GGUF loader and no quantization or conversion tool;
  `packaging/mlx-model-manifest.py` only pins file hashes.
- **An unused cache class.** `model.py:40-105` defines a quantized KV cache (8-bit keys, 4-bit values, group 64; the
  config says 8/8). The engine never builds it: it uses BF16 caches (`quantized.py:81-92`), and the manifest must say
  `kv_format: bf16` (`checkpoint.py:7-8`).

### Context and prompt reuse

- **Context limit.** 262,144 tokens (`quantized.py:17`). `rec` launches the engine with the manifest's 262,144
  (`cli/engine.go:12`; `flint-int4-mlx.json:10`); the Python CLI alone defaults to 8,192 (`rec_mlx/cli.py:32`).
- **Prompt reuse.** One prompt cache.
  - Global-layer KV is append-only shared storage. The sliding layers are checkpointed at the start of each prefill
    chunk, and the last two checkpoints are kept (`prefix.py:47-63`).
  - A new prompt reuses its common prefix if a checkpoint lies at or before the point where it diverges; otherwise it is
    processed from token 0 (`prefix.py:82-118`).
- **Idle-time prefill.** After a reply of at least one prefill chunk, rec prefills prompt plus reply while idle, and a new
  request interrupts it between layers (`engine.py:170-222`; `scheduler.py:129-136`). It never reuses KV written during
  decode: "its arithmetic differs from batched prefill and changes probabilities" (`prefix.py:74`).

## 3. Speculative decoding

- **The search.** The full tree (code, tests, docs, packaging, workflows) was searched for `draft`, `specul`, `mtp`,
  `nextn`/`next_n`, `eagle`, `medusa`, `ngram`/`n-gram`, `lookahead`, `verify` and `accept`. The only speculation-related
  hits are mlx-lm server fields that rec sets to nothing: `num_draft_tokens=0` (`scheduler.py:50`) and the handler's
  draft-model fields (`server.py:215-216`). There is no drafter, verify step, tree attention or lookup decoding.
- **The 8-token packets are not drafts.** Each of the 8 steps is a full forward pass of one token on the token sampled
  before it (`resident_block.py:81`; `sampling.py:26`). Only host round trips are saved.
- **Flint has no MTP or next-n weights.**
  - The Hugging Face config has no MTP, next-n or draft keys.
  - The INT4 checkpoint loads with `strict=True` into a model class that has no MTP module (`quantized.py:37-44`;
    `model.py:274-345`), and a strict load fails on any unknown tensor.
  - The BF16 repository holds no other weights: the 15 shards and their index, config files, tokenizer, template,
    README, licence, assets and the SGLang kit.
- **The makers' GPU serving path.** The SGLang kit's README says speculative decoding is "not validated" and ships no
  draft model. Its serve scripts pass no speculative flags.
- **Consequence.** rec offers no acceptance figures, draft lengths or verify kernels to reuse. The open options remain
  this project's own: an EAGLE-3 head trained for Flint (Q20) and copy-gated n-gram drafting (Q53), both bounded by the
  verify cost of Flint's experts on the rig ([HyperQwen lessons](hyperqwen_lessons.md),
  [speculation plumbing](speculation_plumbing.md)).

## 4. Serving

**API** (`server.py:25-341`, built on mlx-lm's `APIHandler`):
- **Endpoints:** OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, `/v1/models` and `/health`, on loopback only.
  - A bearer token is required (`rec auth token`).
  - Port 11500 when started by `rec` (`cli/cli.go:95`), 8080 from the Python CLI.
- **Restrictions** (`server.py:227-274`):
  - the model must be `flint`;
  - `min_p` and `frequency_penalty` must be 0 and `repetition_penalty` 1;
  - `tool_choice` may only be `auto` or `none`, and `parallel_tool_calls: false` is refused;
  - logprobs work only without streaming;
  - custom stop strings are refused while the Flint grammar is on.
- **Tool-call IDs** are derived from the seed, the sampling settings and the prompt, so a replayed request returns an
  identical message (`chat.py:97-128`).

**Chat template.**
- **rec's own copy.** rec ships its own copy of Flint's template (`rec_mlx/chat_template.jinja`, SHA-256 `44fd1b0f…`,
  `engine/tests/test_chat.py:57-62`) and uses it instead of the checkpoint's (`chat.py:16`).
  - The checkpoint's template does its input sanitising in a macro: it strips NUL bytes and `<think>` tags and escapes
    role tokens in non-assistant text. rec does the same in Python (`chat.py:20-53`).
- **Layout** (`chat_template.jinja:1-54`):
  - turns: `<|system|>…`, `<|developer|>…`, `<|user|>…`;
  - assistant turns: `<|assistant|>`, then an optional `<think>reasoning</think>`, the content, then any tool calls;
  - tool results: `<|observation|><tool_response>…</tool_response>`;
  - the tool list goes in its own `<|system|>` block before the first non-system message;
  - the generation prompt is `<|assistant|>` alone: the model opens `<think>` itself.

**Reasoning.**
- **Always on.** "Flint's reasoning is always enabled" (docs, generation settings). There is no switch: on the Anthropic
  API, `thinking: disabled` or `display: omitted` only hides the reasoning (`anthropic.py:339-351`).
- **History.** A replayed assistant message's `reasoning_content` (or `reasoning`) is rendered back as
  `<think>…</think>` (`server.py:283-286`; template line 33). The model card, the SGLang README and the GGUF card all ask
  clients to keep `reasoning_content` in the history.
- **Budget.** At least 32,768 output tokens for hard problems (model card). On GPQA-Diamond the reasoning length had a
  median of 4.3K tokens, p95 33K and p99 46K (SGLang README). The GGUF card's client configurations allow 65,536.

**Tool calls and grammar.**
- **Format.** `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>…</tool_call>`, the GLM-4.7 layout.
  - String values are raw and other types are JSON; the parser types each value by the tool's schema (template lines
    15-16, 38-42; `chat.py:73-94`).
  - After its calls the model ends the turn with `<|observation|>` (`grammar_builder.py:63-71`).
- **Grammar on every request.** rec builds an xgrammar structural tag for each request by default (`rec_mlx/cli.py:39-42`;
  `server.py:255, 288-294`; `grammar_builder.py:37-82`):
  1. an optional `<think>…</think>`;
  2. text that may not contain any protocol marker;
  3. either the end of the turn (optionally `<|user|>`) or, when the request has tools, one or more calls followed by
     `<|observation|>`.

  Stop tokens are `<|end_of_text|>`, `<|user|>` and `<|observation|>`. Tools marked `strict: false` have their argument
  schemas enforced only with `--enforce-tool-schemas`. The SGLang kit likewise refuses to start without its Flint
  grammar backend.
- **Mask path.** After each sampled token, xgrammar fills a 150,144-bit mask (4,692 words, 18.8 KB) on a CPU stream while
  the GPU already runs the next forward. A one-thread GPU marker orders the hand-off, and the GPU masks the logits and
  samples (`grammar.py:45-97`; `masked_sampling.py:97-127`; `grammar_native.cpp:44-164`). The logits never leave the GPU.

**Sampling defaults.**
- **Values:** temperature 1.0, top-p 0.95, top-k 20, presence penalty 0, seed 0 (`engine.py:22-29`;
  `rec_mlx/cli.py:57-71`). Flint's `generation_config.json` and the docs give the same values.
- **Client values are dropped.** rec's pi transport deletes temperature, top-p and top-k from client requests, so these
  defaults always apply (`cli/clients/pi/pi-transport.mjs:13-14`).
- **No min-p.** rec rejects any other value (`server.py:234-236`), and the makers' llama.cpp command on the GGUF card
  passes `--temp 1 --top-p 0.95 --top-k 20 --min-p 0`.
- **Greedy.** The SGLang README calls greedy decoding "not validated". The model card's evaluations sample at 1.0 / 0.95 /
  20 and report average pass@1 over at least five attempts.

## 5. Performance claims

The README's only figure is a chart: decode speed on an Apple M4 Pro with 48 GB, 4-bit weights, prompts of 1K–252K
tokens, 1,024 output tokens, five runs per context, prefill excluded (`.github/assets/throughput.svg`; values from its
description and plotted points). The chart gives no llama.cpp version, flags, file or KV type.

| Context | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K | 252K |
|---|---|---|---|---|---|---|---|---|---|
| rec, tok/s | 131.2 | 128.3 | 123.0 | 113.7 | 99.0 | 82.4 | 61.7 | 39.1 | 22.6 |
| llama.cpp, tok/s | 88.0 | 85.9 | 83.7 | 79.5 | 71.4 | 58.9 | 42.9 | 24.2 | 8.6 |
| Ratio | 1.49 | 1.49 | 1.47 | 1.43 | 1.39 | 1.40 | 1.44 | 1.62 | 2.63 |

**What the curves imply (est.):**
- **Weights.** In this format a Flint token reads ~1.45 GB of weights: 2.49 B 4-bit parameters at 4.5 bits plus a 49 MB
  BF16 router. At 1K, rec's 7.6 ms per token is ~190 GB/s; llama.cpp's 11.4 ms is ~128 GB/s.
- **KV cache.** The slopes give the attention read rate at 28 KiB per context token:
  - rec: ~200 GB/s at every depth (202 from 1K to 32K, 195 from 128K to 252K);
  - llama.cpp: 162, 148, 104 and 49 GB/s over the same four steps.
- **Where the gain comes from.** At short context it is the fixed cost per token, one dispatch against hundreds. At long
  context it is attention.

The repository contains no prefill figures, quality evaluations or benchmark scripts; its tests check protocol, parity and
packaging. The docs give one informal figure: pi gives "roughly 1.5× faster workflows than OpenCode".

## 6. Carry-over to the rig and the fork

### rec on the rig

Not possible:
- **Platform.** rec runs only on macOS arm64 with an M4 Pro, M5 Pro or M5 Max (`cli/manifest.go:83-85`;
  `cli/platform.go:20-43`; `engine.py:57-61`; `kernels/device.py:15-40`).
- **Kernels.** Its decode path is Metal Shading Language source compiled by `mx.fast.metal_kernel`, and it has no CUDA
  backend.
- **Memory.** It requires 48 GiB of unified memory and puts the 22.6 GB model, the KV cache and the scratch buffers on one
  GPU. The rig has 6 GB per card, 4 GB of host RAM and no peer-to-peer.
- **One device.** The persistent kernel's barriers assume one device holding every layer. A version for nine cards would
  be a new engine, not a port (est.).

### Carry-over table

| Idea (rec's evidence) | Where it goes here | Expected value / effort |
|---|---|---|
| **GQA-aware split-KV decode attention:** a lane owns 4 dimensions for all 8 query heads of a KV head; online softmax per partition; the sink as partition 0's initial state; next K/V prefetched into registers; partitions scaled with depth (32–1,024) and a separate merge (`megakernel.py:619-693, 890-903`) | The Q59 microbenchmark's lane-owns-4-dimensions candidate, then the Q60 port ([long-context attention](pascal_attention_long_context.md)). Take the sink handling, the prefetch and the depth-scaled partition count. rec reduces each head's score across all 32 lanes for every key (8 shuffle reductions per key); on sm_61 the transpose-reduce in the Q59 design costs fewer shuffles (est.) | rec holds ~200 GB/s of KV at every depth on its GPU, the same as its weight reads (est.). Q60's prediction for S4-L0 stands: 32k ~25 → ~30 tok/s, 262k ~7.5 → ~12.5–14 (claim to test). Effort as planned: 2–3 days for Q59 and 3–4 days for Q60 |
| **The makers' sampling and history settings:** min-p 0, top-k 20, top-p 0.95, temperature 1.0; `reasoning_content` sent back in the history | `scripts/server_bench.sh` (`PROMPTS=chat` drops reasoning; it falls back to top-k 40 when the GGUF has no default), the run commands in the README, Q41 | Benchmarks and task checks run in the configuration the makers validated. A follow-up turn then diverges from the cache only after the previous reply (est.). Hours |
| **Task checks the makers' way:** sampled at 1.0 / 0.95 / 20, average pass@1 over ≥ 5 attempts, tool calls under the Flint grammar, reasoning kept | Q41 protocol ([task-level checks](task_level_quality_checks.md)); Q30 was greedy | Comparable with the model card; consistent with the finding that greedy verdicts behave like random draws. More GPU-hours on the quant machine |
| **Grammar mask applied on the GPU:** 18.8 KB per token, computed on the host while the next forward runs; logits never leave the GPU (`masked_sampling.py:97-127`; `grammar_native.cpp:138-164`) | Design (c) of [tool-request sampling](tool_request_sampling.md) (Q55, Q66, Q67). rec and the SGLang kit use mask-first semantics, unlike llama-server's sample-then-check | Keeps tool requests on the `-bs` path, avoiding Q55's estimated 9–11 % loss. On the Celeron the mask (est. 0.1–0.15 ms with llguidance) and its upload (~0.1 ms) would overlap the 27–32 ms forward (est.). Several days in the fork |
| **FP32 activation sums, no rescale:** BF16 activations with FP32 sums, nothing in FP16 (`megakernel.py:171-196, 778-798`) | Q38's FP32-sum item, which removes `LLAMA_NO_MOE_DOWN_RESCALE`'s reason to exist for every format, including UD-Q4_K_XL | Measured rescale-off gains: +5 % (Q8_0), +8 % (Q4_1), +11 % (Q5_0). Part of Q38 |
| **The shared expert as a seventh expert in the routed launches:** gate/up of 6 routed experts plus the shared one in one phase, then all 7 down projections in the next (`megakernel.py:746-798`) | Q50, which only fuses the shared expert with itself (gate, up, SWIGLU), could put it into the routed `MUL_MAT_ID` gate/up and down launches | Q50's saving (47 launches, activation quantizations and SWIGLU kernels per token) plus the shared down projection's own 47 launches (est. a few % per token; claim to test). 1–3 days |
| **A whole-token persistent kernel:** one dispatch per token, barriers on device atomics, team barriers where only a subset depends on a result (`megakernel.py:137-159, 800-820`; `kv_teams.py`) | A long-term fork project: one persistent kernel per card per token for that card's layers. Pascal supports grid-wide synchronisation of co-resident blocks (cooperative launch, compute 6.0+) | Targets the fixed per-launch cost, estimated at 10–13 ms of the pre-fork 35.8 ms token ([Flint speed-ups](flint_speedups.md)). Its share of the fork's token is not measured yet: Q49's host-only timeline is still open. Weeks of work; every format and the attention kernel would be rewritten |

### What does not carry over

- **rec itself and its Metal kernels** (above).
- **8-token packets with on-GPU sampling.** In a nine-card layer split the host moves the hidden state between cards on
  every token, and llama-server's `-bs` already samples on the GPU (est.).
- **Re-prefilling a reply instead of reusing its decode KV.** Prompt processing runs at 80–85 tok/s on the rig (Q58), so
  reusing the decode KV, as llama-server does, is far cheaper there.
- **BF16 storage.** Pascal has no BF16 arithmetic. FP32 sums give the same range for the overflow-prone values.
- **A group-64 affine format.** The makers' codes would need ~10 % fewer bytes than in Q4_1 (est.). It would take a new
  ggml type and new kernels, and the makers' 4-bit tensors already lose against this project's own quants inside its
  files (Q44).
- **3-bit routed experts.** The code supports them, but no 3-bit checkpoint is published.

### Facts that bear on this project's assumptions

- **No MTP head:** confirmed by the strict load, the config and the SGLang kit (section 3).
- **Group-64 layout:** explained. The makers' Q4_1 repacks the MLX export's group-64 INT4 codes, which rec's code treats
  as quantization-aware-trained (section 2).
- **Down-projection overflow:** the makers' own engine has no rescale and no FP16 storage. The overflow comes from FP16
  block sums in llama.cpp, not from the model.
- **Sampling:** the makers use min-p 0. llama-server's default min-p 0.05 was in force in Q58, and in the replay of
  [tool-request sampling](tool_request_sampling.md).
- **History:** the makers keep reasoning in the history. The chat benchmark resends visible answers only.
- **Greedy:** "not validated" in the makers' SGLang kit. Q30's BFCL runs were greedy.
- **Constrained output:** rec constrains every turn, and its free text may not contain protocol markers. llama-server
  constrains only after a tool-call trigger.
- **Tool arguments:** the makers' parser types argument values by the tool's schema (strings stay raw even when they
  look like JSON). A tool-call harness for Flint has to do the same.

## Not verified

- Whether `flint-int4` came from quantization-aware training. The evidence is the code's names and comments and the GGUF
  card's "trained INT4 codes"; no model card says so.
- Whether the makers' Q4_1 GGUF holds exactly `flint-int4`'s codes, scales and offsets. Comparing one expert tensor of
  each would settle it in about an hour.
- Which llama.cpp build, file, flags and KV type produced the README chart's llama.cpp curve.
- The KV and weight read rates of section 5. They are derived from the chart and assume 28 KiB per context token and the
  same KV size for llama.cpp.
- Whether llama.cpp's Flint tool-call parser types arguments by schema as rec does.
- Whether rec's attention layout reaches Q59's gate (f16 ≥ 120 GB/s) on a P106 (claim to test).
- What min-p 0.05 against 0 does to Flint's task scores (claim to test).

## Sources

rec at commit `0dcf43c765195c44d72e3c59c6849a228ebe638b` (P), under
`https://github.com/recursive-oss/rec/blob/0dcf43c765195c44d72e3c59c6849a228ebe638b/`:
- `README.md`; `.github/assets/throughput.svg` (chart values); `engine/pyproject.toml`; `packaging/flint-int4-mlx.json`,
  `packaging/build-mlx-runtime.py`, `packaging/mlx-model-manifest.py`.
- `cli/platform.go`, `cli/engine.go`, `cli/manifest.go`, `cli/download.go`, `cli/serve.go`, `cli/cli.go`,
  `cli/clients/pi/pi-transport.mjs`.
- `engine/src/rec_mlx/`: `model.py`, `quantized.py`, `checkpoint.py`, `cache.py`, `prefix.py`, `engine.py`, `cli.py`,
  `server.py`, `scheduler.py`, `anthropic.py`, `chat.py`, `chat_template.jinja`, `grammar.py`, `grammar_builder.py`.
- `engine/src/rec_mlx/kernels/`: `megakernel.py`, `mixed_bits.py`, `rank_routing.py`, `kv_teams.py`, `expert_teams.py`,
  `routing_attention.py`, `resident_block.py`, `resident_optimizations.py`, `activation_optimization.py`, `sampling.py`,
  `masked_sampling.py`, `device.py`, `__init__.py`.
- `engine/native/grammar_native.cpp`; `engine/tests/validate_resident.py`, `engine/tests/test_chat.py`.

Hugging Face (P, fetched 2026-10-23):
- [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint): model card, `config.json`, file list,
  [`sglang/README.md`](https://huggingface.co/RecursiveSI/flint/blob/main/sglang/README.md),
  [`sglang/serve_b300.sh`](https://huggingface.co/RecursiveSI/flint/blob/main/sglang/serve_b300.sh).
- [RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) at revision `3b93f90`: model card,
  `config.json` (`mlx_export`), `flint-int4.json`, `generation_config.json`, `chat_template.jinja`.
- [RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf): model card (how the Q4_1 is
  built, recommended llama.cpp command and sampling).

docs.recursive.com (P, fetched 2026-10-23): [rec](https://docs.recursive.com/edge/rec.md),
[llama.cpp](https://docs.recursive.com/edge/llama-cpp.md), [NVIDIA recipe](https://docs.recursive.com/local/nvidia),
[generation settings](https://docs.recursive.com/guides/generation-settings),
[releases](https://docs.recursive.com/resources/releases.md).

This repository:
- [quants/experiments.md](../../../quants/experiments.md): Q20, Q30, Q38, Q41, Q44, Q49, Q50, Q53, Q55, Q58, Q59, Q60,
  Q66, Q67.
- [pascal_attention_long_context.md](pascal_attention_long_context.md), [tool_request_sampling.md](tool_request_sampling.md),
  [task_level_quality_checks.md](task_level_quality_checks.md), [speculation_plumbing.md](speculation_plumbing.md),
  [hyperqwen_lessons.md](hyperqwen_lessons.md), [flint_speedups.md](flint_speedups.md).
- [scripts/server_bench.sh](../../../scripts/server_bench.sh); [Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md).
