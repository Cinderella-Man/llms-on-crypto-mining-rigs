# Why Flash-Next reads prompts at ~60 tok/s on nine P106 cards, and what would make it several times faster (2026-11-04)

Scope: prompt processing (prefill) of Qwen3.8-Flash-Next (`qwen4exp`: 48 layers, 36 Gated DeltaNet and 12 attention layers
with a sparse-attention indexer, 512 routed experts with 10 active plus a shared expert, four hyper-connection streams,
a 28.8 GB n-gram table read from the SSD) as the Swift-1.5 GSQ-RCO Q2_0 file, on nine P106-090 cards (compute 6.1, 5 SMs,
`dp4a`, no tensor cores, PCIe 1.0 x1, no peer access) behind a 2-core Celeron with 4 GB of RAM, with Miner Llama v3's base
(layer split, `-b 2048 -ub 512`, four scheduler copies). The note starts from what the rig measured on the agent check,
fits the time of a prompt to a few primitives, splits a prompt token's card time into its parts and checks each part
against the literature and the upstream source, explains DFlash2's `-b 512` loss, prices a second rig, and ranks ideas.
It builds on [prefill on Pascal](prefill_pascal.md) (written for the dense and MoE models), [card hops](card_hops_x1.md),
[Flash-Next versions](flash_next_variants.md), [Strata](strata.md) and [what is new](flash_next_news_2026-11.md), and does
not repeat them. Sources were read on 2026-11-04: llama.cpp source and pull requests, the flash-linear-attention (FLA)
code, the Gated DeltaNet papers, vLLM, SGLang and the serving papers named below. Nothing was run, built or downloaded.

Conventions:
- "(est.)" marks this note's own arithmetic or inference; the basis is next to it. Numbers without it are measured on the
  rig or quoted from the linked source.
- **Micro-batch**: the `-ub` unit (≤ 512 tokens) that crosses the nine cards. **Call**: one `llama_decode`, up to `-b`
  tokens, split into micro-batches. **Pass**: a micro-batch that runs alone, with nothing in front of or behind it.
  **Drain**: the pipeline waiting for every card, as a state save forces. **Period** (p): the time between two
  consecutive micro-batches leaving the last card.
- **Card time** is the GPU time of one prompt token summed over the nine cards; it is what a pass pays per token.
- **N** is the number of new prompt tokens of a request (the part after the cached prefix), the `prompt_n` of the server's
  timings; T(N) is that request's prompt time.
- "Agent check" is [Q218's run](../../data/2026-11-03_retype_agent/README.md) of the retyped Swift Q2_0 (12 tasks, 62
  requests; per-request rows in [requests.csv](../../data/2026-11-03_retype_agent/requests.csv) and
  [agent/turns.csv](../../data/2026-11-03_retype_agent/agent/turns.csv)). "Baseline" is
  [Q215's](../../data/2026-11-03_flash_next_baseline/README.md) single requests. Sums, medians and fits below are hand
  arithmetic on those rows.

## Summary

1. **The ~60 tok/s is mostly pipeline fill and drain, not a per-token speed.** Over the agent check's 62 requests
   (37,776 new prompt tokens in 586 s, 64.5 tok/s), a prompt of N ≤ 460 tokens takes about 2.4 s + 14 ms × N (30 tokens
   2.6 s, 450 tokens 8.4 s, 54 tok/s); 1.0–1.7k tokens take 16–19 s (78 tok/s); 2.0–3.3k tokens take 18–26 s
   (107–129 tok/s); the baseline's 7.2k and 32.7k prompts run at 139 and 149 tok/s. Short turns dominate the session: 38 of
   the 62 requests have 460 tokens or fewer. Prefill is 23 % of the session's wall time (586 of 2,556 s) and generation
   73 %, so a 2.5× faster prefill is worth ~14 % of a session, not 2.5×.
2. **A prompt is cut into passes that the cards cannot overlap.** llama.cpp's `decode()` returns without waiting for the
   cards, so micro-batches of consecutive calls keep flowing. What drains the pipeline is the server's recurrent-state
   checkpoint (112.6 MiB, save 0.67 s, restore 0.63 s), taken 516 and 4 tokens before the prompt's end. Every prompt
   above 516 tokens therefore ends with a 512-token pass that runs alone (est. 7.6 s), two saves and a 4-token pass:
   ≈ 9 s whatever the prompt's length. The first part pipelines, with a period of 0.4–1.0 s per 512 tokens for the first
   few micro-batches (one stage is ~0.8 s) and 2.0–3.3 s once a call holds 6–60 micro-batches. The fit of section 1.2
   reproduces the 10 requests of 568–935 tokens to within 1.6 s.
3. **Card time is ~13.8 ms per prompt token, and the parts are not what the dense-model notes assumed.** The staged
   card-to-card hops are 3.3 ms of it (24 %, the one part with a measured rate); the matrix products (experts,
   projections, shared expert, hyper-connections) are est. 3.6–6.4 ms; the Gated DeltaNet scan on CUDA cores est.
   0.15–0.6; attention and indexer ≤ 0.7 at ≤ 7k context; element-wise ops 0.5–1.5; the n-gram rows 0.1–0.7. The parts
   sum to est. 7.8–13.2 ms. No prompt-phase profile of Flash-Next exists yet (Q157, Q264).
4. **The Gated DeltaNet scan is not a cost on compute 6.1.** The per-token CUDA kernel is 384 blocks of 128 threads with
   the state in registers; a prompt token costs it 170 MFLOP (0.55 ms of card time if issue-bound, est.), about 1.4 % of
   the token's 12 GFLOP of matrix products. FLA's chunked form (chunk 64, WY representation) does ~1.8× more
   multiply-adds per token than the recurrence (est.) and pays back only through tensor cores, which is why upstream's
   chunked kernel is enabled on two architectures (sm_86, sm_89) and gains +1 to +21 % of whole prompts there.
5. **MoE at ~10 tokens per expert is compute-bound with 1.4–1.5× tile padding, not bandwidth-bound.** At `-ub 512` a
   micro-batch touches all 512 experts of a layer once (66 MB of expert bytes per prompt token, 0.5 ms), against est.
   1.65 ms of `dp4a` work with padding. The fork carries upstream #31047's tile choice already; #31352's 128-row tile
   skips the K-quants that hold most of the dense weights, so it is worth est. 1–4 %.
6. **The cards are idle 60–75 % of the time in a long prompt (est.), cause unknown.** Period and stage give 25–40 %
   utilization at 6–60 micro-batches. Suspects: the micro-batches really in flight, synchronous input copies queued behind
   compute, contention on the two switches' uplinks, the host relaying staged copies. One per-card timeline settles it.
7. **DFlash2's `-b 512` costs 21 % of prompt speed because every 512-token call becomes a drain** (the drafter takes the
   target's hidden states after each call, est.). The model of section 1 predicts −19 % for the agent check's requests.
   `-b 1024 -ub 256`, keeping the taps on the device (upstream #31506, merged 2026-11-03) or giving the drafter only the
   prompt's last tokens restores the overlap.
8. **A second rig does not shorten an agent turn.** A hand-off of a 3k-token turn moves the recurrent state (112.6 MiB)
   plus ~90 MiB of KV: est. 1.9 s at the 941 Mbit/s link, on top of a prefill that is as slow on the second rig. With two
   agents, two independent rigs serve more turns than one prefill rig and one decode rig at today's prefill:decode ratio
   (est. 3.1 against 2.5 turns per minute).
9. **What would make it several times faster is pipeline structure first, kernels second.** Taking the checkpoint without a
   drain and running the whole prompt as one call (est. 1.4–1.7× on the agent check's prompt time), and a taper of small
   micro-batches at the start and end (est. 1.3–1.6× alone), together est. 2.0–3.1× on prompt time, 3k-token turns from
   ~24 s to ~13 s and 450-token turns from 8.4 s to ~3.7 s. The card-time fixes (F16 hops, tile tables, K-quant dense
   tensors, prefetch) add est. 1.15–1.25× on top. "Ideas for the rig" at the end ranks them.

## 1. What the rig measures

### 1.1 The agent check's prompts, by turn and by size

The 62 requests of the retyped Swift Q2_0 agent check (`-b 2048 -ub 512`, F16 cache, one slot, 12 bug-fixing tasks). Each
task starts with ~1,549 tokens of system prompt and tool schemas already in the slot, so turn 1 is the task text.

| Turn | Requests | New tokens, median (range) | Prompt time, median (range), s | Share of prompt time | Share of tokens |
|---|---|---|---|---|---|
| 1: task text | 12 | 789 (403–2,780) | 15.0 (8.7–38.0, the cold first request of the run) | 32 % | 28 % |
| 2: files read | 12 | 1,550 (598–3,312) | 17.4 (10.9–25.7) | 36 % | 52 % |
| 3: second tool result | 12 | 399 (32–1,577) | 8.2 (2.6–18.5) | 18 % | 15 % |
| 4 and later: small results | 26 | 33 (28–351) | 2.65 (2.4–8.2) | 14 % | 5 % |

By size:

| New tokens | Requests | Prompt time, median (range), s | Prompt tok/s at the median |
|---|---|---|---|
| 28–36 | 18 | 2.6 (2.4–3.0) | 12 |
| 70–175 | 8 | 4.2 (3.6–5.0) | 25 |
| 257–470 | 12 | 8.2 (6.2–9.0) | 47 |
| 568–935 | 10 | 13.7 (10.9–16.4) | 51 |
| 1,027–1,689 | 9 | 17.4 (16.1–18.5) | 78 |
| 1,984–3,312 (warm) | 4 | 19.6 (18.1–25.7) | 110 |
| 2,780, cold (nothing cached) | 1 | 38.0 | 73 |

- **The 28–36-token requests are 29 % of the requests and 8 % of the prompt time**, at 2.4–3.0 s each. A pass of 30
  tokens costs est. 0.9 s (0.5 s + 14 ms per token, section 1.2); the checkpoint save (0.67 s), a restore (0.63 s) and
  the 4-token tail (est. 0.25 s) make up the rest (section 1.3).
- **Depth does not show up.** The 450-token requests take 8.1–8.7 s at 1.5k, 3.4k and 4.4k tokens of context. Attention
  and indexer cost less than the noise up to ~6.8k tokens.
- **Prompt time per request averages 9.4 s; the 4 warm requests of 2–3.3k tokens take 18–26 s.**

### 1.2 A prompt's time as passes and a pipeline

Fits (hand arithmetic; ±1.6 s, the model runs 0–1.6 s low on the 568–935 bin):

- **One pass of m ≤ 512 tokens** takes est. 0.5 s + 13.8 ms × m: the 13.8 ms is the slope of the 30–460-token requests
  (30 tokens 2.6 s, 105 tokens 4.2 s, 382 tokens 8.2 s, 450 tokens 8.4 s, intercept 2.4 s with the checkpoint work). A
  full 512-token pass is est. 7.6 s, nine stages of ~0.84 s.
- **A prompt of N ≤ 516 tokens** is that one pass plus the end-of-prompt save (0.67 s), the 4-token tail (est. 0.25 s) and
  a restore (0.63 s): T ≈ 2.4 s + 14 ms × N.
- **A prompt of N > 516 tokens** is a first part of N − 516 tokens, then the last 512 tokens as their own pass, then the
  tail. The server saves a checkpoint where the first part ends and another 4 tokens before the prompt's end (the baseline
  logs show them at 6,709 and 7,221 of 7,225 tokens), and each save waits for every card. The tail is
  7.6 + 2 × 0.67 + 0.25 ≈ 9.2 s. The first part is one pass if it holds ≤ 512 tokens, else a pipelined call of
  n micro-batches costing 7.6 s + (n − 1) × p.

Check, first part of one micro-batch or less (tail 9.2 s, first part 0.5 s + 13.8 ms per token):

| N | Measured, s | Model, s |
|---|---|---|
| 598 | 10.9 | 10.8 |
| 693 (median of the 568–935 bin) | 13.7 | 12.1 |
| 935 | 15.3 | 15.5 |
| 1,027 | 16.1 | 16.8 |

The period p of the pipelined part, from the requests above 1,027 tokens (T − 16.8 s per extra micro-batch beyond the
first):

| N | First part, tokens | Measured T, s | T − 16.8 s | Extra micro-batches | p, s per 512 tokens |
|---|---|---|---|---|---|
| 1,358 | 842 | 16.5 | −0.3 | 0.6 | ≈ 0 (noise) |
| 1,689 | 1,173 | 17.3 | 0.5 | 1.3 | 0.4 |
| 1,984 | 1,468 | 18.1 | 1.3 | 1.9 | 0.7 |
| 2,006 | 1,490 | 18.7 | 1.9 | 1.9 | 1.0 |
| 2,307 | 1,791 | 20.5 | 3.7 | 2.5 | 1.5 |
| 3,312 | 2,796 | 25.7 | 8.9 | 4.5 | 2.0 |
| 7,183 (baseline) | 6,667 | 51.6 | 34.8 | 12.0 | 2.9 |
| 32,674 (baseline) | 32,158 | 220 | 203 | 61.8 | 3.3 |

- **The pipeline runs near its ideal for the first two or three micro-batches** (p ≈ 0.4–1.0 s against a stage of
  ~0.84 s) and slows to 2.0–3.3 s as a call grows. Context depth is not the cause below ~7k tokens (section 1.1). The
  baseline's 149 tok/s at 32k (6.4 ms per token) is the 3.3 s end of this curve.
- **Anatomy of a 3,000-token prompt (est.):** first part 2,484 tokens = 7.6 s for the first micro-batch + 3.85 × 2.0 s =
  15.3 s; tail 9.2 s; total 24.5 s (interpolated measurement 24.1 s). The tail is 38 % of it.
- **A 450-token prompt** is 6.7 s of pass, 1.3 s of save and restore, 0.25 s of tail and ~0.2 s of other work: 8.4 s.
  Nine cards work one after another and each is busy one ninth of the time.

### 1.3 Why the tail exists

- **`decode()` does not wait.** The function ends without synchronizing; the source comment says the wait for the
  computation to finish is "automatically done when obtaining the model output"
  ([llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp)). Calls that follow each
  other without a state read or a logits read keep the pipeline full. A reused graph still waits, but under pipeline
  parallelism the fork rebuilds instead (Q109).
- **A checkpoint save does wait**: it copies the recurrent state of every card to the host (112.6 MiB, 0.67 s at the x1
  rate), so the nine cards drain first. The server takes one 4 + `-ub` tokens before a prompt's end and one 4 tokens
  before it, so that a client whose re-rendered history differs in the last tokens can resume from the second
  ([Q263](../../../quants/experiments.md) measured the same for Flint: a 19-token turn cost 0.63–0.68 s of prompt time).
- **The logits read is the third and unavoidable drain**, once per prompt.

## 2. Where a prompt token's time goes (question 1)

### 2.1 The budget

Card time per prompt token, nine cards, `-ub 512`, from the model's shapes (the profile shapes in
[Q215's decode profile](../../data/2026-11-03_flash_next_baseline/decode_profile_v3.txt)) and the rates in the linked
notes. Matrix products per prompt token: routed experts 2.36 GMAC, attention and Gated DeltaNet projections 2.7 GMAC
(`attn_qkv` 2,560 × 10,240, `attn_gate` 2,560 × 6,144, `ssm_out` 6,144 × 2,560 in 36 layers; q, k, v, o in 12), shared
expert 0.24, hyper-connections 0.63, router 0.06: ~6 GMAC. At the ~2 T multiply-adds per second per card that Flint's MMQ
fitted ([prefill_pascal.md](prefill_pascal.md) 1.3) that is 3 ms of card time before padding.

| Part | Basis | Est. ms per prompt token | Share of 13.8 |
|---|---|---|---|
| Staged hops of the four-stream residual, 8 boundaries | 40,960 B × (1/213 + 1/189 MB/s) = 0.41 ms per boundary ([card hops](card_hops_x1.md) 1.2) | 3.3 | 24 % |
| Routed experts, Q2_0 MMQ | 2.36 GMAC × 1.4–1.5 padding at 10 tokens per expert (section 2.3); bytes floor 0.5 | 1.8–2.6 | 13–19 % |
| Attention and Gated DeltaNet projections (Q3_K, IQ4_XS, Q4_K, Q5_K) | 2.7 GMAC; K-quant MMQ slower than the 2 T fit | 1.4–2.7 | 10–20 % |
| Shared expert (Q2_0) | 0.24 GMAC | 0.12–0.2 | 1 % |
| Hyper-connection products | 0.63 GMAC; BF16 or Q8_0 made no measurable prompt difference (Q218: 59.6 → 60.7 tok/s) | 0.3–0.9 | 2–6 % |
| Gated DeltaNet scan and conv (36 layers) | section 2.2 | 0.15–0.6 | 1–4 % |
| Attention (12 layers), indexer, masks | no depth effect up to ~6.8k (section 1.1) | ≤ 0.7 | ≤ 5 % |
| Element-wise work on the streams, norms, RoPE, top-10-of-512 routing, row gathers | ~15 passes per layer over 21 MB per 512 tokens at ~170 GB/s | 0.5–1.5 | 4–11 % |
| N-gram rows | 1.4–3.3 SSD reads per prompt token × 152 µs, plus 27 ms of x1 per micro-batch (section 2.6) | 0.1–0.7 | 1–5 % |
| **Sum of the estimates** | | **7.8–13.2** | |
| **Measured slope of a pass** | | **13.8** | 100 % |

- **Only the hops are measured-rate arithmetic.** The rest rests on rates fitted to Flint and on shapes; the gap to
  13.8 ms sits at the high end of the dense, expert and element-wise rows. A `GGML_CUDA_PROFILE` summary of one 512-token
  micro-batch (`scripts/cuda_profile_summary.py --tokens 512 --by op`) replaces this table (Q157, Q264).
- **The hops are the largest single part of a pass** because nothing overlaps them in a pass: 1.5 s of an 8.4-s
  450-token prompt. In a full pipeline they sit on the source card's compute stream and add to its stage
  ([card hops](card_hops_x1.md) 4.5).
- **Hyper-connections are not a prompt cost.** The decode profile's 43 ms of BF16 products per token are matrix-vector
  work; at 512 tokens they are 0.63 GMAC of matrix-matrix work.

### 2.2 The Gated DeltaNet scan without tensor cores

- **What runs.** llama.cpp's CUDA kernel for `GATED_DELTA_NET`
  ([gated_delta_net.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/gated_delta_net.cu)) keeps
  a state column in registers (eight lanes per column, sixteen rows per lane, four columns per warp), loops over the
  micro-batch's tokens inside one block, and per token does `Sᵀk`, the delta update and `Sᵀq`. For 48 value heads of 128
  it launches 384 blocks of 128 threads. The state is read and written once per micro-batch (3.1 MiB per layer).
  Tensor-core builds try an MMA chunked kernel first.
- **Its cost on the P106 (est.).** Per token and layer: 48 heads × 3 × 128² = 2.36 M multiply-adds (4.7 MFLOP); over 36
  layers 170 MFLOP. At ~300 warp-instructions per token per warp, the 1,536 warps of a layer cost 460k warp-instructions
  per token; on 20 schedulers (5 SMs × 4) at 1.5 GHz that is 15 µs per layer and token, 0.55 ms per token over 36
  layers. 1,536 warps over 20 schedulers is more than the 16 each can keep resident, so latency is hidden and the kernel
  is issue-bound. Range 0.15–0.6 ms of 13.8.
- **Chunked form on CUDA cores.** FLA's `chunk_gated_delta_rule` runs a gate cumsum, a fused "K·Kᵀ, triangular solve,
  w and u" step (the WY representation of the product of Householder-like updates), the inter-chunk state recurrence and
  an output step; the chunk is 16, 32 or 64 tokens, default 64
  ([chunk.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)). The
  papers give the cost as O(L·C·d + L·d²) against the recurrence's O(L·d²)
  ([DeltaNet](https://arxiv.org/abs/2406.06484), [Gated DeltaNet](https://arxiv.org/abs/2412.06464), the
  [author's walk-through](https://sustcsonglin.github.io/blog/2024/deltanet-2)). At C = 64, d = 128 the per-token terms
  are K·Kᵀ 8k, triangular solve 1.4k, `T·K` and `T·V` 16k, `W·S`, `Kᵀ·U` and `Q·S` 16k each, intra-chunk output 16k:
  ~90k multiply-adds against the recurrence's 49k (est.). On CUDA cores the chunked form therefore costs more, and its
  matrix products gain nothing without tensor cores.
- **What others found.** Upstream's chunked kernel (PR #29353, merged 2026-10-14) uses MMA, is enabled only on sm_86 and
  sm_89 and only when the context verifies one token; it gained +0.9 to +21 % of whole prompts
  ([PR](https://github.com/ggml-org/llama.cpp/pull/29353)). A proof of concept with 64-token chunks reports the stock
  kernel as issue-bound at one warp per scheduler on an RTX 5090
  ([discussion #28514](https://github.com/ggml-org/llama.cpp/discussions/28514)). NVIDIA's own GDN plugin requires SM80
  and K = V = 128 ([TensorRT-Edge-LLM](https://nvidia.github.io/TensorRT-Edge-LLM/0.7.0/cpp_api/plugins/gatedDeltaNetPlugin.html)).
  No chunked implementation for GPUs without tensor cores was found.
- **Verdict.** A chunked GDN on SIMT is not a lever (≤ 3 %, Q462 stays parked); reopen it if the profile shows the scan at
  ≥ 10 % of a micro-batch. The in-place state and the fused conv and norm (Q437, Q438) are decode work, est. −5 to −7 ms
  per decoded token; a prompt pays the state gather once per micro-batch, not per token.

### 2.3 MoE at ~10 tokens per expert

Independent routing of 10 of 512 experts per token (est.; Flint's measured routing was ~80 % as distinct). Expert bytes:
34.0 GB of Q2_0 over 48 layers. Expert compute: 2.36 GMAC per token at 2 T multiply-adds per second per card, times tile
padding with 8-column steps.

| `-ub` | Experts touched per layer | Tokens per touched expert | Expert bytes per prompt token | Bytes time at 130 GB/s, ms | `dp4a` time with padding, ms | Expert time per token, est. ms |
|---|---|---|---|---|---|---|
| 64 | 367 | 1.7 | 381 MB | 2.9 | 5.4 (×4.6) | 5.4 |
| 128 | 471 | 2.7 | 244 MB | 1.9 | 3.4 (×2.9) | 3.4 |
| 256 | 509 | 5.0 | 132 MB | 1.0 | 2.2 (×1.9) | 2.2 |
| 512 | 512 | 10 | 66 MB | 0.5 | 1.65 (×1.4) | 1.65 |
| 1,024 | 512 | 20 | 33 MB | 0.26 | 1.4 (×1.2) | 1.4 |

- **At `-ub 512` the experts are compute-bound.** A micro-batch reads every expert once, 66 MB per prompt token, which is a
  third of the `dp4a` time. Expert batching across micro-batches is not possible in a layer-split pipeline: two
  micro-batches are never at the same layer at the same time. A larger `-ub` is the only way to more tokens per expert.
- **A larger micro-batch buys little.** `-ub` 1,024 saves est. 0.25 ms per token (2 %) and doubles the fill: on Flint
  pp2048 fell from 612 to 415 tok/s and pp8192 from 714 to 692
  ([Q109's grid](../../data/2026-10-24_rig_prefill/README.md)). VRAM caps it as well: compute buffers grow by 13–16 bytes
  per micro-batch token per context token ([fit probe](../../data/2026-11-03_flash_next_fit/README.md)), so Q2_0 fits 64k
  at `-ub` 512 (735 MiB free on the fullest card), 128k at 256 (664), 262k at 64 (224); `-ub` 1,024 at 64k would need
  another ~0.47 GiB per card (est.).
- **Smaller micro-batches cost experts.** At `-ub` 128 the experts cost est. +1.75 ms per token (+13 % of a pass) and at
  64 +3.75 ms. That is the price of the taper in section 5, paid only on the first and last few hundred tokens.
- **Token sorting is already in.** The ids path sorts token-expert pairs by expert and quantizes each token once, and
  upstream #31047 sizes the tile columns by tokens per expert and launches only tiles with work; the fork carries both
  (Q108). P40 results of the PR: 1.40× at ubatch 64 and 1.02× at 2,048 for a 128-expert model
  ([PR](https://github.com/ggml-org/llama.cpp/pull/31047)); an independent Flash-Next pp2048 report in its thread gives
  "~6.1" without units. Grouped-GEMM prior art confirms the structure: MegaBlocks avoids padding with block-sparse
  products ([arXiv 2211.15841](https://arxiv.org/abs/2211.15841)), and vLLM's fused-MoE path pads each expert's rows to
  the kernel's tile height with `moe_align_block_size`, the same padding as here
  ([docs](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/fused_moe/)).
- **#31352 reaches little of this model.** It raises the tile height from 64 to 128 in the two J = 64 rows of 11 types
  (+8 to +18 % pp512 per type on a GTX 1080 Ti, Qwen3.6-35B-A3B Q2_0 +13 % at `-ub` 128; K-quants lose 19–37 %)
  ([PR](https://github.com/ggml-org/llama.cpp/pull/31352)). Expert tiles of 10 tokens use the J ≤ 24 rows (est.); the
  J = 64 rows serve the dense products, and the Swift file's biggest dense tensors (`attn_qkv`, `attn_gate`, `attn_q`)
  are Q3_K. Est. +1–4 % (Q285).
- **FP32 cuBLAS does not help.** MMQ is ~2.5× faster than SGEMM on this card at any batch and the non-MMQ expert path
  ran at 0.6× on Flint ([prefill_pascal.md](prefill_pascal.md) 2.3, Q28).

### 2.4 Hyper-connections at batch

Four streams of 2,560 F32 values per token: a residual is 40,960 B, and the projections (320 × 10,240 up, 10,240 × 320
down, 10,240 × 4 inject) are 0.63 GMAC per token. llama.cpp runs the BF16 projections with a software-converting
matrix-vector kernel for 1–8 tokens and BF16 cuBLAS for 9–128 ([variants note](flash_next_variants.md) 4); the path at
512 tokens is unmeasured on Pascal. The retyped Q8_0 file runs MMQ. The two files read prompts at 59.6 and 60.7 tok/s on
the same eight tasks, which bounds the difference at ~2 % of a prompt (est. 0.3–0.9 ms per token for either). The
15-ms-per-token decode gain of the retype does not carry over to prompts.

### 2.5 The sparse-attention indexer during prefill

- **What the graph does.** `build_qsa_sel` scores pooled key blocks with one fused `ggml_lightning_indexer` node, takes
  `ggml_top_k` over pools (a budget of 2,048 cells) and turns the selection into an `[n_kv, n_tokens]` mask; the code has
  no branch that skips the selection when the context is under the budget
  ([qwen4exp.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen4exp.cpp)). Pascal has no sparse
  gather (it needs Turing), so the 12 attention layers read the whole cache whatever the mask selects.
- **Measured cost.** None visible up to ~6.8k tokens of context (section 1.1). At 32k the period grows from 2.9 s (7k) to
  3.3 s (32k) per 512 tokens: +14 % for 4.5× the context, which bounds attention, indexer and mask together at ~15–20 %
  of a micro-batch there (est.). Q230's gather kernel and Q702's head pairing are decode-at-depth work and prompt work
  beyond ~16k only.
- **Memory.** The `[n_kv, n_tokens]` mask is the 13–16 bytes per (token × context) in the compute buffers (upstream #30815
  targets it); at 64k and `-ub` 512 it is ~0.47 GiB per card (est.).

### 2.6 N-gram rows

- **Per prompt token** the host hashes 16 row numbers, the CPU gathers 16 × ~90 B from the memory-mapped table (rows may be
  prefetched in the input step), and 10 KB goes to card 1 over x1: 5 MB and 27 ms per micro-batch, 0.4 % of a pass
  (Q447, raw rows to the card at 1.4 KB per token, saves at most that; its 1–2 % of prefill includes the gather).
- **The SSD reads are the larger part.** The baseline's long prompts made 8–13 reads of 4 KiB per prompt token (17,814
  for 1,382 tokens, 270,264 for 32,674), against 1.4–3.3 per token that the replay of agent text predicted with ~2 GiB of
  page cache (Q228); the baseline prompts were fresh text with a cold table. At 152 µs per read at queue depth 1 and the
  agent text's 1.4–3.3 reads, a prompt token pays 0.2–0.5 ms before card 1 can start (est. 1–4 % of a pass). In a
  pipelined call the host gathers the next micro-batch while the cards compute, so only a pass and the first micro-batch
  pay for it.
- **Prefetch is a small lever.** llama.cpp announces a micro-batch's rows in its input step. Announcing all of a prompt's
  rows when the request starts (the server knows the tokens) hides the reads behind the first micro-batches: est. 0.1–0.4
  s on a 450-token prompt (Q523).

## 3. Why the cards idle in a long prompt

Utilization of a card in a full pipeline is stage / period: 0.84 / 2.0 = 42 % at 6 micro-batches, 0.84 / 3.3 = 25 % at
63 (est.), the same range as the 65–89 % idle share that Q157 records for Flint. Suspects, with what each predicts:

- **Micro-batches in flight.** The pipeline holds up to `GGML_SCHED_MAX_COPIES` = 4 micro-batches. With nine stages that
  caps utilization at 4/9 = 44 %, and a period of (loaded latency) / 4. Flint's 1-copy build ran 3.9–4.2× slower than 4
  copies, and an 8-copy build changed nothing (765 against 766 tok/s, Q157), which a pure copy cap cannot explain.
- **Synchronous input copies.** The scheduler copies each card's inputs with a copy and a stream synchronize; the copy
  queues behind that card's earlier micro-batches, so the host cannot run ahead of the slowest card. The card-built mask
  (Q110, Q116) removed the biggest input; token ids and the n-gram rows still arrive this way on cards 0 and 1.
- **Link contention.** In a full pipeline up to four hops run at once through two switches whose uplinks carry 360–480 MB/s
  for four cards (90–120 MB/s each): a 21-MB hop takes 0.35–0.47 s instead of 0.2 s ([card hops](card_hops_x1.md) 4.5).
  If this were the cause, halving the hop bytes (F16) would shorten the period by about the same share.
- **The host relaying staged copies** (H1 of the card-hops note): with two cores and several hops at once, a driver thread's
  wake-up queues behind the submit thread. `-t 1` and thread pinning would move the period.
- **Other:** thermal clocks over long runs, host memory pressure (MemAvailable ~230 MiB, swap in use), cold table pages.
  None is excluded by the data.

The discriminating measurements are in the ideas (I3): a per-card busy timeline in a 4,096-token prompt, the 8-copy build
on Flash-Next, an F16-hop A/B, `-t 1`.

**The per-card timeline** ([2026-11-10_prompt_timeline](../../data/2026-11-10_prompt_timeline/README.md), R42): on
the v3.0 release line a fresh 4,097-token prompt keeps 1.42 of the nine cards busy on average (card 1 the busiest at
18.78 % of the prompt's 29,983.0 ms, card 8 at 6.43 %). The host waits for the cards after each of the prompt's four
calls (45.9 % of the time: after each call the MTP draft's catch-up needs the call's rows, and the 512-token call
between the two checkpoints runs on one card at a time), blocks inside each call's third queued micro-batch on cards
1 to 7 (29.5 %, est. a full CUDA launch queue: no prompt micro-batch runs as a CUDA graph), and spends 9.8 % on the
draft's catch-up. The main model logs no compute-buffer reallocation.

## 4. Methods for CUDA-core-only GPUs (question 2)

| Method | Fit to compute 6.1 and this model | Est. effect on Flash-Next prompts | Verdict |
|---|---|---|---|
| Larger micro-batch | experts −0.25 ms per token at 1,024; fill and drain double; VRAM +0.47 GiB per card at 64k | negative below ~8k (Flint pp2048 −32 %) | no |
| Smaller and tapered micro-batches | shorter fill and tail; experts +1.75 ms per token at 128, +3.75 at 64 | −3 to −8 s per prompt (section 5) | yes (I2) |
| Sort tokens by expert, tile by tokens per expert | in the ids path; #31047 carried (Q108) | done | — |
| dp4a MMQ tile 128 (#31352) | dense Q2_0, IQ4_XS, Q8_0 rows only; K-quants lose | +1–4 % | cherry-pick (I7) |
| FP32 cuBLAS for dense parts | SGEMM 0.8 T against MMQ 2 T multiply-adds per second | < 1× | no |
| Chunked GDN on SIMT | 1.8× more multiply-adds; the scan is ≤ 4 % | ≤ 3 % | parked (I10) |
| Prefetch n-gram rows before a micro-batch needs them | rows announced at the input step today | +1–4 % of a pass | small (I6) |
| Fewer bytes per hop (F16, Q8_0) and hops off the compute stream | hops are 24 % of a pass | +8–12 % of a pass | yes (I4) |
| A prompt cache that keeps more state | checkpoints cost drains, not bytes | see below | yes (I1) |
| Weight-stationary prompt experts (Strata's `STRATA_MMQ_WS`, Q446) | experts 13–19 % of a pass | ≤ 3 % | parked (I11) |

**Which state the cache keeps.** The hit rate is already 96.6–96.9 % from the second turn, and
[agent/turns.csv](../../data/2026-11-03_retype_agent/agent/turns.csv) shows the live state continuing through the
generated reply (cached tokens at turn t + 1 equal cached + new + generated at turn t), so replies are not re-read. What
costs time is taking and restoring checkpoints. SGLang keeps recurrent states for hybrid models in a radix tree
(`MambaRadixCache`): a state is forked into the tree after chunked prefill and after decoding, a match returns the
deepest node that has a state, eviction is LRU
([SGLang hybrid models](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)); the fork is a copy
inside GPU memory. The equivalent here is Q440 (checkpoints in numbered on-card stores) plus Q448 and Q548 (spread, so an
edit deep in the history resumes near it): est. 12.5 MiB per card per checkpoint instead of 112.6 MiB through the host,
and a copy that can be ordered on each card's stream so the host does not wait.

## 5. Where the several-fold gain is

The structure of section 1 gives two targets.

- **No drain: the prompt as one stream of calls (I1).** Write the checkpoint state with a copy that is ordered on each
  card's stream (an on-card store, Q440), so the host never waits, and issue the prompt's calls back to back: the last 512
  tokens and the 4-token tail are then micro-batches of the same pipelined stream. The fixed 9.2 s tail becomes the last
  micro-batch (p ≈ 2 s) plus ~0.3 s of copies. The logits read stays. Est. T(3,000) 24.1 → 17.6 s; T(450) 8.4 → 7.0 s;
  T(30) 2.6 → 1.2 s; the agent check's prompt time 586 → 372 s (1.6×; range 1.4–1.7×).
- **Taper (I2).** Fill and tail are pass latencies, and a pass costs 0.5 s + 13.8 ms per token, so micro-batches of 64–128
  tokens at the start and end of a prompt (and for all of a short one) shorten them: est. 1.4 s for a 64-token micro-batch
  to cross the cards, 2.3 s for 128, against 7.6 s for 512. A 450-token prompt as 64-token micro-batches: 1.4 s + 6 × 0.3
  s ≈ 3.2 s plus checkpoint work; a 3,000-token prompt: fill 1.4 s instead of 7.6 s, tail 3.8 s instead of 7.6 s. Est.
  alone −7.6 s on a 3k-token prompt (1.5× on the check's prompt time), −4.7 s on a 450-token one. It needs a server
  change (the prompt loop's call sizes), not a kernel or scheduler change: a call of fewer tokens than `-ub` is one
  smaller micro-batch, calls do not drain, and the compute buffers were reserved for 512. Q363 names it.
- **Both (I1 + I2).** One stream of calls, tapered at both ends: est. T(3,000) 13.4 s (1.8×), T(1,358) 4.4 s, T(450)
  3.7 s, T(30) 1.2 s; the check's prompt time 586 → 190–290 s (2.0–3.1×), prompt speed 64 → 130–200 tok/s, the session's
  wall time −12 to −15 %.
- **Kernels second.** Hops (I4) −1.65 ms per token, tiles and K-quant dense tensors (I7, I8) −0.3 to −0.8, prefetch (I6)
  −0.1 to −0.4: est. 13.8 → 11.0–11.8 ms, 1.2–1.25× on a pass. The steady state (I3) is the largest unknown: if a
  63-micro-batch call reached a period of 1.0–1.5 s, a 32k prompt would read at ~300–470 tok/s instead of 149.

Honest scale: prefill is 23 % of the agent check's wall time. The ideas above make prompts 2–3× faster and sessions
1.13–1.18×; only decode (73 %) moves sessions further, and a faster decode (DFlash2, MTP) raises prefill's share.

## 6. DFlash2's `-b 512` (question 3)

- **What the line does.** The DFlash2 arm used `-b 512 -ub 512` and prompts at 47.1 against 59.4 tok/s on the common
  tasks (MTP at `-b 2048`: 56.3, −5 %) ([data](../../data/2026-11-03_flash_next_dflash2/README.md)). With `-b` equal to
  `-ub` every call is one micro-batch.
- **Why `-b` is 512.** The drafter's layer-input buffer holds 5 taps × 2,560 × `-b` floats in pinned host memory: 100 MiB
  at 2,048, 25 MiB at 512, on a host with ~100–230 MiB to spare; the `-b 2048` arm tripped the host-memory watchdog at its
  first 2,776-token prompt.
- **Why it costs 21 % (est.).** The target's calls do not wait (section 1.3), but the drafter consumes the taps after
  each call, which makes each call a drain. Replace the pipelined first part of every prompt by n separate passes of ~7 s
  where today a call of n micro-batches costs 7.6 s + (n − 1) × p: each of the 13 requests of 1,086 tokens or more gains
  5 s per micro-batch of its first part beyond the first (n = 2: 5 requests, n = 3: 5, n = 4: 1, n = 5: 1, n = 6: 1;
  +135 s). The agent check's 586 s become 721 s, 52 tok/s (−19 %) against the measured −21 %. The second turn's first
  token (task files, ~1,500 new tokens) took 25.3 s against 18.4 and 20.4 s, +5 to +7 s; the model says +5 s for n = 2.
- **What other engines do.** vLLM schedules one token budget per step (`max_num_batched_tokens`), prefill chunks and
  decode tokens share it, chunked prefill is on by default, and its EAGLE proposer is built to take the target's hidden
  states ([optimization docs](https://docs.vllm.ai/en/latest/configuration/optimization/),
  [eagle.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/eagle.py)). SGLang runs the NGRAM and
  DFLASH algorithms without mixed chunked prefill, so prefill goes in prefill-only batches whose size is the server's
  chunk, not the draft block, and DFLASH keeps only a sliding window of the draft's KV
  ([docs](https://docs.sglang.io/advanced_features/speculative_decoding.html)). The sources do not say how the drafter's own
  prompt pass is batched; the shared idea is that the prompt batch size is a target setting and speculation starts when
  decoding does.
- **Options for the rig**, cheapest first: (1) `-b 1024 -ub 256`: four micro-batches per call, taps 50 MiB, one drain per
  1,024 tokens; (2) upstream #31506 (merged 2026-11-03), which keeps the target's features on the device (+5 to +98 %
  DFlash prompt speed on other GPUs; the rig's build predates it), then `-b 2048` if the taps fit; (3) the drafter ingests
  only the last W prompt tokens' taps, as SGLang's DFLASH window does, so prompt calls need no sync and the buffer is W
  tokens (acceptance after a long prompt unmeasured); (4) the main model's 260 MiB of token embeddings on a card (Q563)
  frees the host room for (1) or (2).

## 7. Disaggregation (question 4)

- **What moves.** The recurrent state is 112.6 MiB (the checkpoint, 12.5 MiB per card) and the attention plus indexer
  cache ~30 KiB per token in F16 ([variants note](flash_next_variants.md) 1): a 3k-token turn moves ~90 MiB of KV, a 6.5k
  context 190 MiB. At the rigs' measured 941 Mbit/s each way
  ([Q511](../../data/2026-11-03_rig_network/README.md)), ~110 MB/s payload (est.): the state takes 1.0 s, KV for a 3k-token
  turn 0.8 s, 1.9 s together; F16 states (not tested for exactness) halve it. A 4 GB host cannot stage 200 MiB (MemAvailable
  ~230 MiB), so a hand-off streams per card in ~12-MiB pieces (card → host 213 MB/s → NIC → host → card 189 MB/s).
- **What the engines do.** Prefill-decode disaggregation is DistServe's and Splitwise's design: prompt and token phases on
  separate machines, the KV cache moved between them, 7.4× more requests within latency targets or 1.4× throughput at 20 %
  lower cost ([DistServe](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677)). SGLang
  extends its protocol to hybrid models with a separate state channel; a Mamba or Gated DeltaNet state goes as one
  contiguous block after prefill, not streamed like paged KV
  ([SGLang](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)). Their gains are throughput
  at many concurrent requests on fast fabrics.
- **Why it does not shorten a turn here.** The agent's next step needs the model's answer, which needs the whole prompt:
  turn time = prefill + transfer + decode, and the second rig is as slow at prefill as the first. 3k-token turn: 24 s
  prefill + 1.9 s transfer.
- **With two agents.** A turn is est. 24 s of prefill and 14 s of decode (130 tokens at 9 tok/s). Two independent rigs
  finish 2 turns per 38 s (3.1 per minute). One prefill rig and one decode rig are bound by the prefill rig: 24 s per turn
  (2.5 per minute). The split wins only when prefill is shorter than decode, i.e. after I1 and I2 (13 s against 14 s).
- **Two rigs for one model (Q505)** is a fit decision, not a speed one. An 18-stage pipeline keeps the sum of the stage
  times, so a pass's latency is unchanged and adds one 21-MB hop over the LAN (0.19 s per 512 tokens), and the cards are
  not what limits the steady state. The point of Q505 stays the bigger file (IQ3_S, 47–55 GB).
- **A cheaper second-rig use:** a replica for another agent session, or a saved-state target (section 4's checkpoint store
  copied over the LAN, 1 s per 113 MiB) for resuming a session on the other rig.

## 8. Where this note differs from the earlier ones

- [card_hops_x1.md](card_hops_x1.md) 4.5 takes a stage of 2.0–3.4 s per 512-token micro-batch and a full-pipeline speed of
  150–260 tok/s from the 32k point. The agent check's passes give a stage of ~0.84 s and a single-micro-batch latency of
  ~7.6 s; the pipeline's period is 2–3.3 s only for long calls, so cards are busy 25–42 % there, and the hop is 24 % of a
  pass, not 6–19 % of a stage. Its P4 (keep `-ub` 512 and four copies) holds for the steady state and not for the fill
  and the tail.
- [prefill_pascal.md](prefill_pascal.md) estimated a fixed ~2.2 s per request for Flint; for Flash-Next the fixed part
  is 2.4 s for a 30-token turn and ~9 s for any prompt above 516 tokens, because the state save drains nine cards and the
  last 512 tokens run alone.
- [flash_next_variants.md](flash_next_variants.md) and Q228 put the table's prompt reads at 1.4–3.3 per token; fresh text
  with a cold table cost 8–13.

## 9. Gaps

- No profile of a Flash-Next prompt micro-batch exists; section 2.1 is arithmetic until Q157/Q264 run.
- The v3 server's checkpoint positions (516 and 4 before the end) are read from the baseline's logs; the prompt-time model
  of section 1.2 assumes each save drains the pipeline for ~0.67 s and the last 512 tokens run as one pass.
- Why the period grows from ~0.8 s to 3.3 s with the call's length (section 3).
- Q3_K and IQ4_XS MMQ speed on `dp4a`, and the BF16 hyper-connection path above 128 tokens, are unmeasured.
- The GDN kernel's time per 512-token micro-batch on a P106 (est. 0.6 ms per token at most).
- Whether DFlash2's drafter makes each call a drain (the data only show the speed), and how it performs with a prompt
  window.
- Whether a stream-ordered on-card checkpoint (Q440) keeps the pipeline full; the branch writes it, no P106 run exists.
- Whether back-to-back server calls of fewer than `-ub` tokens keep the pipeline full (the premise of I2); llama-server's
  loop between calls has not been read for a wait.
- The size of the Q3_K tensors in the Swift file (est. 0.8 GB from the layer shapes, some layers IQ4_XS).
- State transfer between two rigs: only the link's iperf figure is measured.

## Sources

llama.cpp and the fork (P)
- [`gated_delta_net.cu`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/gated_delta_net.cu),
  [`llama-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp),
  [`qwen4exp.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen4exp.cpp), read 2026-11-04
- [PR #29353](https://github.com/ggml-org/llama.cpp/pull/29353) GDN chunked kernel,
  [PR #31047](https://github.com/ggml-org/llama.cpp/pull/31047) MoE layout for MMQ,
  [PR #31352](https://github.com/ggml-org/llama.cpp/pull/31352) Pascal dp4a tile,
  [PR #31506](https://github.com/ggml-org/llama.cpp/pull/31506) DFlash features on the device,
  [PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105) DFlash,
  [discussion #28514](https://github.com/ggml-org/llama.cpp/discussions/28514) chunked GDN proof of concept

Linear attention, MoE and pipelines (P)
- S. Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length",
  [arXiv 2406.06484](https://arxiv.org/abs/2406.06484); S. Yang, J. Kautz, A. Hatamizadeh, "Gated Delta Networks",
  ICLR 2025, [arXiv 2412.06464](https://arxiv.org/abs/2412.06464); [DeltaNet Explained, part II](https://sustcsonglin.github.io/blog/2024/deltanet-2)
- [FLA `chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)
- [TensorRT-Edge-LLM GDN plugin](https://nvidia.github.io/TensorRT-Edge-LLM/0.7.0/cpp_api/plugins/gatedDeltaNetPlugin.html)
- T. Gale et al., "MegaBlocks", [arXiv 2211.15841](https://arxiv.org/abs/2211.15841);
  [vLLM fused MoE](https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/fused_moe/fused_moe/)
- Y. Huang et al., "GPipe", [arXiv 1811.06965](https://arxiv.org/abs/1811.06965) (batch splitting across devices)

Serving, speculation and disaggregation (P unless noted)
- Y. Zhong et al., "DistServe", OSDI 2024, [arXiv 2401.09670](https://arxiv.org/abs/2401.09670); P. Patel et al.,
  "Splitwise", [arXiv 2311.18677](https://arxiv.org/abs/2311.18677)
- [SGLang: hybrid models](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/),
  [SGLang speculative decoding docs](https://docs.sglang.io/advanced_features/speculative_decoding.html)
- [vLLM optimization docs](https://docs.vllm.ai/en/latest/configuration/optimization/),
  [vLLM `eagle.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/eagle.py) (S: constructor only)

This project
- [Agent check on the retyped file](../../data/2026-11-03_retype_agent/README.md) (requests, turns), [baseline](../../data/2026-11-03_flash_next_baseline/README.md),
  [DFlash2](../../data/2026-11-03_flash_next_dflash2/README.md), [fit probe](../../data/2026-11-03_flash_next_fit/README.md),
  [prompt grid on Flint](../../data/2026-10-24_rig_prefill/README.md), [rig network](../../data/2026-11-03_rig_network/README.md)
- [Registry](../../../quants/experiments.md): Q28, Q89, Q108–Q110, Q116, Q157, Q215, Q218, Q226, Q228, Q230, Q231, Q263,
  Q264, Q284, Q285, Q362, Q363, Q437, Q438, Q440, Q446, Q447, Q448, Q462, Q505, Q510, Q523, Q548, Q561, Q563, Q609, Q702

## Ideas for the rig

Ranked by est. effect on prompt tok/s over the agent check's mix (64.5 tok/s today); "s" is est. seconds saved on a 3k-token
turn (24 s today). Effects overlap: I1 and I2 both shorten the last pass, together 2.0–3.1×. First measurement for I1–I3,
~1 h on the idle rig (a model load is 5.4 min per setting): `llama-bench -p 32,64,128,256,512,1024,2048,4096 -n 0 -b 2048
-ub 512` (and `-ub 128`, `256`) gives the pass latency and the period without any checkpoint; the server with a fresh
2,048-token prompt, `--ctx-checkpoints 0` against the default, gives the cost of the drains.

| # (rank by tok/s; by s) | Idea | Est. prompt tok/s | Est. s saved, 3k-token turn | Effort | Risk | First measurement | Id |
|---|---|---|---|---|---|---|---|
| I1 (1; 3) | Checkpoints without a drain: write the state with a stream-ordered on-card copy at the micro-batch that ends at the checkpoint, issue the prompt's calls back to back so the last 512 tokens and the 4-token tail stay in the pipelined stream, read the logits once | 64 → 90–110 (1.4–1.7×); a 30-token turn 2.6 → 1.2 s | 5–7 | 3–5 days on top of Q440's stores | medium: exactness of restored states, `--ctx-checkpoints` semantics, host memory | server, 2,048-token fresh prompt, `--ctx-checkpoints 0` against 4: the difference is the drains (predicted ≥ 5 s); then the on-card store's timing | Q440, Q362, Q263 (the stream-ordered, in-call form: no row yet) |
| I2 (2; 2) | Taper: micro-batches of 64–128 tokens for the first and last few hundred tokens of a prompt and for whole prompts under ~600 tokens, 512 in between, issued as calls of fewer tokens than `-ub` | 64 → 85–100 alone (1.3–1.6×) | 7–8 | 1–2 days (server prompt loop) | low: experts +1.75 ms per token at 128; more graph builds on the Celeron | `llama-bench -ub 128/256/512` grid and a server run at `-ub` 128: T for 450, 1,000, 2,000, 3,000 tokens | Q363 (the taper schedule: no row yet) |
| I5 (3; 1) | DFlash2 line: restore the overlap with `-b 1024 -ub 256`, then #31506 with the taps on the device, then a prompt window for the drafter | 47 → 55–59 on that line (+17–25 %); 0 on the plain and MTP lines | 5–15 on the DFlash2 line | flag test; 1–3 days for the window | memory (taps 50 MiB); acceptance after a windowed prompt | the DFlash2 agent line at `-b 1024 -ub 256`, MemAvailable logged | Q561, Q563, Q609 (the window: no row yet) |
| I3 (4; 4) | Find why the cards idle 60–75 % in a long call: 8-copy build on Flash-Next, a per-card busy timeline, F16-hop A/B, `-t 1` | 5–10 % on the mix; 32k prompts 149 → 300–470 if solved (40 % chance, est.) | 2–5 | 2 days to measure; 1–2 weeks if engine work | unknown cause | per-card timeline of a 4,096-token prompt (`GGML_CUDA_PROFILE_HOST`); pp4096 at 4 against 8 copies | Q157, Q510, Q284 |
| I4 (5; 5) | F16 (or Q8_0) residual on the hops, then hops on a copy stream and chunked staging so the two legs overlap | +8–12 % of a pass (−1.65 ms of 13.8), 5–10 % on the mix | 1–2 | 1–2 days plus the KLD gate | KLD from the F16 range | dump the eight boundaries on the quant machine: range and round-trip KLD; then pp2048 with the cast on and off | Q231, Q284; chunked staging: card-hops P3 (no row) |
| I8 (6; 6) | Move the Q3_K attention and Gated DeltaNet tensors (`attn_qkv`, `attn_gate`, `attn_q`, est. ~0.8 GB) to a type whose MMQ tile is fast (IQ4_XS, Q4_0 class) | 2–6 % | 0.5–1.5 | a retype run (as Q218) | +0.2–0.3 GB per file; KLD; decode speed of the new type unmeasured | `llama-bench` pp512 on one P106 for Q3_K, IQ4_XS, Q4_0, Q8_0 at 2,560 × 10,240 | none |
| I7 (7; 7) | Take #31352's 22 table rows and retune per type and `J` for 5 SMs | 1–4 % (not the K-quants) | 0.3–1 | 0.5 day | pp64 −3 % on a 28-SM card; K-quants excluded | `llama-bench` pp64, pp128, pp512 per type before and after | Q285 |
| I6 (8; 8) | Announce all of a prompt's n-gram rows when the request starts (readahead in a thread) | 1–2 % on the mix, 2–4 % on 100–1,000-token turns | 0.1–0.4 on a 450-token turn, ~0 at 3k | 1 day | low | the same prompt twice with `cache_prompt` off, cold against warm table pages | Q523, Q447 |
| I10 (9; 9) | GDN on CUDA cores: Strata's column-pipelined recurrence or a chunked form | ≤ 3 % (parked) | ≤ 0.5 | 1–2 weeks | not worth it | profile share of `GATED_DELTA_NET` in a 512-token micro-batch; reopen at ≥ 10 % | Q462 |
| I11 (10; 10) | Weight-stationary prompt experts | ≤ 3 % (parked) | ≤ 0.5 | 2–3 weeks | not worth it | profile share of `MUL_MAT_ID`; reopen at ≥ 25 % of a micro-batch | Q446 |
| I9 (11; 11) | Saved prefix states (slot save and restore, or a hybrid prefix tree like SGLang's `MambaRadixCache`) for session starts | 0 on the agent check (the 1,549-token prefix is in the slot); a 10–15k-token system prompt: 70–110 s → ~2 s once per session | 0 | slot API: no code; automatic restore 1–2 days | low | save and restore at 5k and 15k tokens, host memory logged | none for Flash-Next (prefill note E8, Q548) |
| I12 (12; 12) | Second rig as a prefill server | 0, and −1.9 s of transfer per turn; two independent replicas serve 3.1 turns per minute against 2.5 | −1.9 | 1–2 weeks | memory on a 4 GB host | only if prefill drops below decode (after I1 and I2): copy one rig's checkpoint to the other over the LAN and time it | Q505 (related) |

The cheapest lever is outside the engine: a harness that does not send a tool output's last 1,000 tokens saves 2–4 s of a
prompt (the period of two micro-batches at 1.0–2.0 s each).
