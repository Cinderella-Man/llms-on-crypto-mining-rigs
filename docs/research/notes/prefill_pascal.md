# Prompt processing on the P106 rig: where the time goes, how to make it several times faster, and how much of it agents can skip (2026-10-24)

Scope: prompt processing (prefill) of Flint (40B-A2.5B MoE) on nine P106-090 cards with this project's llama.cpp fork, for
agent workloads in which tool outputs and long histories dominate. Round 5 ([multi-conversation serving](multi_conversation_serving.md))
found that at two new prompt tokens per generated token prefill sets the total: est. 56–72 generated tok/s on nine cards.
This note splits a prompt token's cost into its parts at 2k, 8k, 32k and 131k of context from the fork's code and the
measured numbers, reads the batched matrix kernel (MMQ) and the attention kernel that compute 6.1 runs, models how
llama.cpp pipelines a prompt across the cards under 6 GB per card and 4 GB of host RAM, measures how much of an agent
session's prompt tokens a cache could serve, and lists prior art and the first rig experiments. Conventions are those of
the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (paper, code, measurements in a repository) and secondary.
- "Laptop", "quant machine" and "rig" are the three places work runs: code and correctness on a laptop GPU, quality and
  training on the quant machine (never speed), speed on the rig only.
- `file:line` refers to the fork's branch `pascal-p106-v2` at `032b130f4`. CUDA files are in `ggml/src/ggml-cuda/`. The
  fork changed only the one-token kernels, so MMQ, flash attention and the scheduler are upstream `d77256999`.
- **Card time** is the GPU (and blocking host) time one prompt token costs, summed over the cards: what a full pipeline
  pays per token. **Wall time** is the time per prompt token that one request sees. With the cards working one after
  another they are equal; when several micro-batches overlap across the cards, wall time falls and card time does not.
- "ub" is the micro-batch (`-ub`), "copies" the scheduler's pipeline copies (`GGML_SCHED_MAX_COPIES`, default 4).

Starting point, measured ([registry](../../../quants/experiments.md), [Flint benchmark](../../benchmarks/2026-10-22-flint.md),
[Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md)):
- **llama-bench pp512** (`-ub 512`, the 512-token prompt is one micro-batch, so nothing overlaps): S4-L0 199 tok/s on six
  cards (5.0 ms per token), UD-Q4_K_XL 193–199, imatrix Q5_0 198 / 185 on five / nine cards, Unsloth's Q8_0 (42 GB) 185 on
  nine, the makers' Q4_1 171 / 161. The format and the file size barely matter, and four more cards cost 0.33 ms per
  token (0.08 ms per extra hop).
- **No MMQ is slower:** the makers-fast file (Q4_1_64, which has no MMQ kernel) read prompts at 103 against 170 tok/s
  (Q28).
- **Real chat** (Q58: nine cards, `-c 262144 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0`): 80–85 tok/s over the
  conversation. Per turn on the fork with S4-L0: 69 new tokens in 2.6 s, 1,076 in 9.8 s, 452 in 5.4 s, 435 in 5.2 s, 380
  in 5.3 s: ~7.1 ms per token plus ~2.2 s per request (est., straight-line fit). Stock builds without `-bs` or
  checkpoints took 0.45–1.2 s for 19–78-token prompts.
- **Deep fills** (Q79: llama-bench, nine cards, `-b 2048 -ub 128`): 0 → 32k in ~234 s (~7.1 ms per token on average),
  32k → 64k in ~436 s (~13.3 ms per token): +0.19 ms per prompt token for every 1k tokens of context, ~4 ms per token at
  depth 0 by extrapolation (est., two points).
- Flint: 49 layers, 35 sliding-window (513 tokens), 14 full-attention without positional encoding, 32 query and 4 KV heads
  of 128, hidden 2,048, 256 routed experts (top 6, 512 wide) plus a 1,024-wide shared expert. A token computes ~2.2 G
  multiply-adds in matrix products (0.89 G of them routed experts).

## Verdict

- **Most of a short prompt's time is wasted expert arithmetic.** On compute 6.1 llama.cpp sizes MMQ's expert tiles for the
  whole micro-batch (64 token columns) although each expert sees ~3 tokens at `-ub 128` and ~12 at `-ub 512`, so 95 % of
  the expert `dp4a` work at `-ub 128` multiplies padding (`mmq.cu:189-225`; only RDNA3/4 size by tokens per expert,
  `:197-201`). That is 7–9 ms of a ~10–12 ms prompt token at `-ub 128` (est.), and it explains why pp512 ignores the format,
  why 20–80-token prompts took up to a second, and most of what `-ub 128` costs. Upstream PR #31047 (open since
  2026-10-23) sizes the tiles by tokens per expert and launches only tiles with work: on a P40 (compute 6.1, same kernels)
  1.40× prompt speed at ubatch 64 and 1.22× at 128 on a 128-expert model. Est. on the rig: expert time 7–9 → 1.1–2.0 ms
  per token, prompts 1.6–2.0× faster at 8k.
- **The pipeline across the cards holds at most two micro-batches at `-ub 128`.** When two consecutive micro-batches have
  the same padded KV length, which at `-ub 128` is every other one, llama.cpp reuses the graph and first waits for every
  card to finish (`src/llama-context.cpp:1537-1545`). At `-ub` ≥ 256 the KV length changes every micro-batch and nothing
  waits. Rebuilding prompt graphs instead of waiting (one line), or `-ub 256`, lets up to four micro-batches overlap:
  est. 1.4–1.8× wall time at ≤ 8k. A scheduler that keeps a prompt's micro-batches flowing (TurboPrefill, PR #24219)
  measured 4.5× over llama.cpp's pipeline on 10 × P104-100 at PCIe Gen1 x1.
- **The attention mask costs 18 bytes of PCIe per prompt token per context token on nine cards.** The host fills an
  n_kv × ub f16 mask and copies it synchronously to every card, one card after another (`ggml-backend.cpp:1829-1837`):
  3.2 ms per prompt token at 32k and 12.5 ms at 131k of host time (est.). Its four pipeline copies in pinned host RAM and
  on every card are also what forces `-ub 128` at 262k. A mask expanded on each card from the cells' positions (4 bytes
  per cell) removes both: est. 1.3–1.5× at 32k, 1.5–2× at 131k, and `-ub 512` with four copies at any context.
- **At depth, attention arithmetic dominates.** The 14 full-attention layers cost 229 MFLOP per prompt token per 1k of
  context; the FP32 TILE kernel at an assumed ~1 TFLOPS takes ~7 ms per token at 32k and ~30 ms at 131k (est.). Once the
  three fixes above land, attention is ~65–70 % of card time at 32k and ~90 % at 131k, and a Pascal prefill attention
  kernel becomes the next lever.
- **Together (est., nine cards):** card time per prompt token today ~10–12 / 12–14 / 20–22 / 52–54 ms at 2k / 8k / 32k /
  131k, after the three fixes ~4–5 / 5–6 / 11–12 / 34 ms; wall time 4.4 / 5.6 / 10 / ~29 ms in llama-bench's fill today
  (7.1 ms at ≤ 2k in llama-server), 1.2–2.0 / 1.6–2.5 / 3.3–4.7 / 9–14 ms after. At round 5's two prompt tokens per
  generated token the staggered total rises from est. 56–72 to 130–190 generated tok/s.
- **Caching already serves ~90 % of an agent session's prompt tokens; the rest is still the bottleneck.** In 200
  multi-turn tool sessions from Q41 (BFCL v4, 2,364 requests) each request re-sends 39 prompt tokens per generated token.
  llama-server's prefix reuse with context checkpoints leaves 10.4 % to process (4.1 per generated token), half of it the
  first request's ~5,000-token system and tool prompt. Restoring a saved state for that shared prompt and reusing the
  generated turn exactly would leave 2.4 % (0.9 per generated token, est.). Coding agents with 1–3k-token tool outputs keep
  1–4 fresh prompt tokens per generated token however good the cache (est.).
- **Each request pays ~2 s before its first token, about as much as its new tokens in a tool loop.** With `-bs` every
  request attaches a sampler, which forces a full scheduler re-reserve (three worst-case graph builds,
  `src/llama-context.cpp:1407-1459`, `:731-885`); two context checkpoints each wait for the pipeline to drain and copy
  ~45 MiB to the host (`tools/server/server-context.cpp:4380-4455`); and the prompt's last 4 tokens run as their own
  micro-batch. Est. −0.5–1.5 s per request from fixing these.
- **cuBLAS does not help.** Compute 6.1 already uses MMQ for every quantized product (`mmq.cu:422-424`).
  `GGML_CUDA_FORCE_MMQ` and `GGML_CUDA_FORCE_CUBLAS` are build options, not environment variables; the first changes
  nothing here, the second routes expert products through a per-expert path that waits on the host twice per call
  (`ggml-cuda.cu:2110-2160`), the path behind Q28's 103 against 170 tok/s.
- **First rig experiment, no code, ~1.5 h:** `llama-bench -p 512,2048,8192 -n 0 -b 8192 -ub 128,256,512` on the
  4-, 2- and 1-copy builds and with `LLAMA_GRAPH_REUSE_DISABLE=1`, then one profiled micro-batch per `-ub`. It tests both
  mechanisms: four copies at `-ub 128` should run no faster than two copies (the reuse wait), `LLAMA_GRAPH_REUSE_DISABLE=1`
  should lift them 1.4–1.8×, and pp8192 at `-ub 512` should run ≥ 2× pp512 (overlap). Then the PR #31047 cherry-pick.

## Ranked changes

Gains are prompt speed (wall time) on nine cards against today's server setting (`-ub 128`, four copies), for a prompt
whose context ends at 8k or 32k, each change on top of the rows above it (est.).

| # | Change | Expected prefill gain at 8k / 32k (est.) | Effort | First experiment | Kill criterion |
|---|---|---|---|---|---|
| 1 | **MMQ expert tiles sized by tokens per expert, only tiles with work launched**: cherry-pick upstream PR #31047 (`mmq.cu`, `mmq.cuh`, `mmid.cu`) | 1.6–2.0× / 1.3–1.5×; card time −45–60 % at ≤ 8k; tiny prompts 2–4× | ½–1 day: cherry-pick, `test-backend-ops -o MUL_MAT_ID`, rig A/B | E3: the E1 grid on `pascal-p106-v2-prefill` against `pascal-p106-v2`, 1-copy builds first (no overlap effects) | pp2048 at `-ub 128` < 1.15× faster, or MUL_MAT_ID's profiled share not at least halved |
| 2 | **Keep the micro-batch pipeline full**: no graph reuse (rebuild instead of a full wait) for prompt micro-batches under pipeline parallelism (`src/llama-context.cpp:1537`); `-ub 256` where host RAM allows | 1.4–1.8× / 1.1–1.3× (host mask copies cap it at depth until row 3) | Hours (one condition); `-ub 256` is a flag | E1: `LLAMA_GRAPH_REUSE_DISABLE=1` at `-ub 128`, and `-ub 256`/`512` with 4 copies | pp8192 at `-ub 512` with 4 copies < 1.5× pp512 (overlap does not happen) |
| 3 | **Mask built on each card** from cell and token positions (a small op that expands 4 B per cell into the f16 mask), later inside the attention kernels | 1.05–1.15× / 1.3–1.5×; 1.5–2× at 131k; `-ub 512` × 4 copies at 262k | 2–3 days (graph op); 3–5 days in-kernel (breakthrough candidate 3) | E2's host timeline at 32k: gap between card splits | Mask copies < 10 % of wall time at 32k |
| 4 | **Per-request fixed cost**: reserve once with a backend sampler attached, one checkpoint per prompt with an asynchronous copy, no separate 4-token tail | −0.5–1.5 s per request (est.): 1.3–1.6× on BFCL-like tool turns of ~200 new tokens | 1–2 days (server, llama-context) | E6: llama-server `-lv 4`, ± `-bs`, `--ctx-checkpoints 0/4` | Per-request intercept < 0.5 s without `-bs` and checkpoints |
| 5 | **Avoid prefill** (section 4): saved state for the shared system and tool prompt, a checkpoint at the end of each generated turn, no history pruning in the harness | Prompt tokens to process −50–75 % in agent sessions; a 5–15k-token first request 1–3 min → 1–3 s | Slot API: no code; auto-restore and end-of-turn checkpoint 1–2 days | E7: offline cache replay; E8: slot save/restore timing | Re-render check shows exact matches and harnesses never resend from scratch |
| 6 | **Pascal prefill attention**: TILE retile with register blocking (K/V in shared memory once per 4 tokens is shared-memory bound), skip all-masked 64-key tiles of the sliding-window ring, optional int8 Q·Kᵀ on `dp4a` | 1.1× / 1.3–1.5×; 1.5–1.8× at 131k | 1–2 weeks; int8 Q·Kᵀ needs a KLD check at 32k | `test-backend-ops perf -o FLASH_ATTN_EXT` at prompt shapes (nb 128/512, kv 768/8,192/32,768) | TILE already ≥ 70 % of FP32 peak at nb 128 |
| 7 | **Hops on a copy stream**: the staged copy to the next card runs on the source card's compute stream (`ggml-cuda.cu:2687`), so it delays that card's next micro-batch | 1.05–1.1× / 1.03× | 1–2 days (CUDA backend) | E2 profile: hop share of the source card's stage | Hop < 5 % of a stage |
| 8 | **MMQ table retuned for 5 SMs** (row tile `I` per type, `J`, stream-k) for Q8_0, Q5_0, Q5_K, IQ4_XS, Q6_K | 1.03–1.08× after row 1 | 1 day + 2–3 h rig | `test-backend-ops perf -o MUL_MAT,MUL_MAT_ID` at Flint's prompt shapes, I 64 vs 128 | < 3 % of MMQ time |
| 9 | **Prompt chunks as pipeline work in the streams mode** (Q104) | No single-prompt gain; staggered total at 2 prompt tokens per generated token est. 56–72 → 130–190 tok/s with rows 1–3 | 3–5 days (planned in Q104) | Simulator extension with this note's card times | Chunk overhead > 30 % of card time at 64-token chunks |
| 10 | cuBLAS FP32 (`GGML_CUDA_FORCE_CUBLAS` build) | < 1× (Q28: 0.6×) | Build only | Control in E3 if wanted | — |

## 1. Where a prompt token's time goes (question 1)

### 1.1 What a prompt micro-batch runs on compute 6.1

- **Matrix products: MMQ for every quantized weight.** `ggml_cuda_should_use_mmq` returns true on NVIDIA GPUs without FP16
  tensor cores at any batch size (`mmq.cu:422-424`). Dense products tile 64 rows × up to 64 tokens. Expert products
  (`MUL_MAT_ID`, `ggml-cuda.cu:2066-2108`) first sort the token-expert pairs by expert and quantize each token once
  (`mmq.cu:276-353`), then launch a grid of (row tiles × ub/J token tiles × 256 experts); blocks whose expert has fewer
  tokens than their tile start exit at once (`mmq.cuh:1024-1026`).
- **The tile is computed whole.** The `dp4a` loop runs over all J columns of a tile whatever the expert's token count
  (`mmq.cuh:908-939`); only the write-back stops at the last real column. J is picked to minimize the number of tiles
  over the micro-batch's token count (`mmq.cu:189-225`), so J = 64 for any micro-batch of 57 tokens or more.
- **Attention: TILE flash attention, FP32.** Prompt micro-batches run `flash_attn_tile<128,128,4,8>`: 32 query rows (4
  tokens × the 8 query heads of a KV head), 256 threads, 64 keys per iteration, K and V converted to FP32 in shared memory
  (`fattn-tile.cuh:89-156`, `:1149-1235`). It computes every cell up to n_kv: the full-attention layers' context and the
  sliding-window layers' whole ring of `pad256(513 + ub)` cells (768 at `-ub 128`, 1,280 at `-ub 512`), masked or not.
- **Inputs: the mask is a host input.** `build_attn_inp_kq_mask` makes an f16 n_kv × ub tensor that the host fills for
  each micro-batch (`src/llama-graph.cpp:30-47`). The scheduler keeps a copy of every graph input per pipeline copy and per
  card and copies user inputs synchronously before each card's split (`ggml-backend.cpp:1823-1840`).
- **Hops:** 8 KB per token (F32 hidden state) to the next card, staged through host memory by the driver (no P2P), issued
  on the source card's stream (`ggml-cuda.cu:2687`).
- **Launches:** a prompt graph changes with the KV length every one or two micro-batches, so its ~1,500–2,000 kernels
  over the cards (est.) mostly launch one by one, ~10–20 ms of Celeron time per micro-batch, small next to the micro-batch
  itself.

### 1.2 Experts per micro-batch

Routing taken as independent across tokens (Q51 measured Flint at ~80 % of random distinctness, with a layer's top quarter
of experts taking 44 % of reads, so real counts are a little more skewed). Expert bytes at ~2.3 MB per expert (G3/S4-L0
gate, up and down; est.), 47 MoE layers.

| ub | Experts touched per layer | Tokens per touched expert | J today | Padding today | J with PR #31047 | Padding with #31047 (incl. second tiles) | Expert bytes per prompt token |
|---|---|---|---|---|---|---|---|
| 32 | 136 | 1.4 | 32 | 23× | 8 | 5.7× | 460 MB |
| 64 | 200 | 1.9 | 64 | 33× | 8 | 4.2× | 340 MB |
| 128 | 244 | 3.2 | 64 | 20× | 8 | 2.5× | 210 MB |
| 256 | 255 | 6.0 | 64 | 11× | 8 | 1.55× | 125 MB |
| 512 | 256 | 12 | 64 | 5.3× | 16 | 1.47× | 60 MB |
| 1,024 | 256 | 24 | 64 | 2.7× | 24 | ~1.45× | 40 MB |

Two facts follow:
- **Today the padding sets the expert cost**, and it is worst for 33–64-token prompts, which agents send all the time.
- **After the fix the bytes set it:** a micro-batch reads nearly every expert once, so the expert bytes per prompt token
  fall as 1/ub. At ~120–140 GB/s of MMQ streaming that is 1.5–1.7 ms per token at `-ub 128`, 0.9–1.0 at 256 and ~0.45 at
  512 (est.). Larger micro-batches stay worth having, for a different reason than today.

### 1.3 Cost model and its calibration

Per prompt token, nine cards (est.):
- **Matrix products** at an MMQ rate of ~2.0 T int8 multiply-adds per second per card (46 % of the `dp4a` peak at
  ~1.7 GHz; fitted below): routed experts 0.89 G × padding, dense products (attention projections, shared expert, dense
  layers) 1.3 G, no padding at ub ≥ 64.
- **Attention** at ~1 TFLOPS for TILE: 229 MFLOP per 1k of context on the full-attention layers, 35 × 32 × 128 × 4 ×
  cells FLOP on the sliding-window layers (0.44 ms at 768 cells, 0.73 at 1,280).
- **Mask copies**: n_kv × 2 B to each of nine cards at ~190 MB/s, one after another, plus the sliding-window mask.
- **Hops**: 8 × 0.08 ms (measured: five against nine cards in pp512).
- **Other kernels and host**: ~0.5 ms.

Checks against the measured points:

| Measurement | Model | Measured |
|---|---|---|
| pp512, six cards, `-ub 512` (one micro-batch, no overlap) | experts 2.4 + dense 0.7 + sliding-window 0.7 + hops 0.4 + other 0.5 ≈ 4.7 ms | 5.0 ms |
| 21 / 38 / 78-token prompts, stock, no `-bs` (J 24 / 40 / 40, 100 / 147 / 225 experts touched) | expert work 0.18 / 0.43 / 0.66 s + ~0.25 s | 0.45 / 0.80 / 1.21 s |
| pp512 of Q8_0 (42 GB) against UD-Q4_K_XL (25 GB), nine cards | compute-bound: bytes don't matter | 185 / 186 tok/s |
| Qwen3.6-35B-A3B pp512, four cards (8 of 256 experts, 16 per expert, 4× padding) | ~3.5–4 ms | 4.0–4.3 ms (231–247 tok/s) |
| llama-bench fill slope with depth (≤ 2 micro-batches in flight) | attention 0.23/2 + masks 0.10 ≈ 0.21 ms per 1k | 0.19 ms per 1k |

The ~2.0 T multiply-adds per second is a fit to these points, not a measurement; E2's profile measures it.

### 1.4 Today, by context length

Nine cards, `-ub 128`, four copies (est.; ms per prompt token):

| Component | 2k | 8k | 32k | 131k |
|---|---|---|---|---|
| Routed experts (MMQ, J = 64) | 7–9 | 7–9 | 7–9 | 7–9 |
| Dense products | 0.7 | 0.7 | 0.7 | 0.7 |
| Attention, sliding-window layers | 0.45 | 0.45 | 0.45 | 0.45 |
| Attention, full layers | 0.46 | 1.8 | 7.3 | 30 |
| Mask copies (host, one card after another) | 0.27 | 0.85 | 3.2 | 12.5 |
| Hops | 0.7 | 0.7 | 0.7 | 0.7 |
| Other kernels, mask fill, launches | 0.5 | 0.5 | 0.6 | 1.0 |
| **Card time** | **10–12** | **12–14** | **20–22** | **52–54** |
| Experts / attention / masks share | 72 / 8 / 2 % | 64 / 16 / 6 % | 38 / 35 / 15 % | 15 / 57 / 23 % |
| Wall, llama-bench fill (`-b 2048`; measured slope) | 4.4 | 5.6 | 10.2 | ~29 (extrapolated) |
| Wall, llama-server (`-b 512`, Q58) | 7.1 | — | — | — |

- **Up to ~16k the experts dominate; from ~30k attention and masks do.** The hops are a steady ~0.7 ms.
- **The host** spends ~0.2–0.3 ms per token on mask fill, graph build and launches at short context (est.), and the
  synchronous mask copies make it the bottleneck from ~32k once the pipeline overlaps.
- **Six cards** have three fewer hops (−0.25 ms) and a third less mask traffic, but no bar-quality file holds long context
  there (Q54, Q105).

### 1.5 The fixed part of each request

The ~2.2 s intercept of the chat turns, decomposed from the server code (est.):
- **Backend sampling re-reserves the scheduler.** `llama_set_sampler` at slot launch and at release
  (`tools/server/server-context.cpp:2033-2036`, `:447`) set `sched_need_reserve` (`src/llama-context.cpp:1440`, `:1459`),
  so the request's first decode runs `sched_reserve`: three worst-case graph builds and allocator passes over nine
  backends (`:731-885`). At start-up the same call took ~0.5 s on the Celeron, including first allocations. Est.
  0.1–0.5 s per request.
- **Two checkpoints.** The server stops the prompt 4 + ub and 4 tokens before its end and saves a checkpoint before each
  of those decodes (`server-context.cpp:4380-4398`, `:4444-4455`). Each waits for every queued micro-batch and copies the
  ~640 live sliding-window cells (~45 MiB) to the host, ~0.2–0.3 s each. Restoring the previous turn's checkpoint adds
  another ~0.25 s.
- **Small tails.** The last 4 tokens run on the stock multi-token path (pp4 117 ms, Q19), and every odd-sized micro-batch
  rebuilds its graph.
- **Pipeline fill and drain.** Each request's first and last micro-batches run without overlap: ~0.5–0.8 s at `-ub 128`
  today.

For an agent turn of ~200 new tokens this fixed part is as large as the per-token part (BFCL-like sessions: ~12 requests).

## 2. MMQ on compute 6.1 (question 2)

### 2.1 Tiles and occupancy on 5 SMs

- **The Pascal table** (`mmq-config-pascal-dp4a.cuh:1-273`, used for compute 6.1 at `mmq.cuh:259`): every type 256 threads
  (8 warps), occupancy 2, row tile I = 64, token tile J = 8…64, 256 weights of K per iteration, no stream-k. A Q8_0 tile at
  J = 64 takes ~28 KB of shared memory (19 KB weights, 9 KB activations), so two blocks per SM: 10 resident blocks on a
  P106-090.
- **Waves.** Attention q (4,096 rows) at `-ub 128` is 64 × 2 = 128 blocks, 12.8 waves; k and v (512 rows) are 16 blocks, so
  the second wave runs 6 of 10 slots. Small products lose ~20–40 % to wave tails (est.), a minor cost next to the padding.
- **Expert grids** today: gate/up 8 row tiles × 2 token tiles × 256 experts = 4,096 blocks per call at `-ub 128`, half of
  which exit at once.

### 2.2 `dp4a` limits

- **The inner loop is shared-memory bound.** For each 8 `dp4a` the Q8_0-layout loop issues ~4 broadcast activation loads
  and ~1 weight load from shared memory, plus scales (`mmq-vec-dot.cuh:110-140`). At 1 shared-memory transaction per clock
  per SM against 4 warp-instructions issued, that caps the loop near 40–60 % of the `dp4a` peak (est.), consistent with the
  ~46 % fitted in section 1.3.
- **No throttle is known.** Prefill uses `dp4a` at about a tenth of its nominal rate ([Pascal forks](pascal_forks_and_quants.md),
  section 6); Q103's one-hour calibration settles whether the P106-090 has a mining-SKU limit.
- **Useful work today is far lower:** at `-ub 128` the experts do ~5 % useful `dp4a` work, so the card's useful rate on them
  is ~0.1 T multiply-adds per second.

### 2.3 cuBLAS FP32 against MMQ

- **Dense products:** SGEMM on a P106-090 reaches ~1.5–1.7 TFLOPS (est., 70–80 % of 2.2), 0.75–0.85 T multiply-adds per
  second, plus a dequantization pass. MMQ's ~2.0 T is ~2.5× faster at any ubatch.
- **Expert products:** the non-MMQ path copies the routing ids to the host, sorts there, copies back (two stream waits),
  then runs one product per used expert (`ggml-cuda.cu:2110-2160`): ~240 launches per call and no overlap across cards.
  Q28 measured it: 103 against 170 tok/s.
- **So** `GGML_CUDA_FORCE_CUBLAS=ON` is a control at most. `GGML_CUDA_FORCE_MMQ=ON` changes nothing on these cards
  (`mmq.cu:418-424`).

### 2.4 MoE tiles: upstream PR #31047

- **What it does:** the routing helper (`mmid.cu`) precomputes the list of (expert, token tile) pairs that have work, the
  MMQ grid runs over that list (`ntj = ceil(pairs/J) + experts` tiles instead of ub/J × experts), and J is chosen against
  tokens per expert for every GPU (the RDNA3/4 rule of `mmq.cu:197-201`). The author attributes 2–4 % to the layout and
  the rest to smaller tiles.
- **P40 measurements** (gemma4 26B-A4B Q4_0, pp8192): ub 32 1.29×, 64 1.40×, 128 1.22×, 256 1.11×, 512 1.04×; a 3B MoE
  1.17× at ub 64; dense models unchanged.
- **Why the rig should gain more.** A P106-090 has a sixth of a P40's `dp4a` rate at half its bandwidth, so wasted
  arithmetic weighs ~3× more per byte. Flint's 256 experts and top 6 put it at ~3 tokens per expert at `-ub 128`, where the
  P40 curve peaks. Est. expert time 7–9 → 1.1–2.0 ms per token at `-ub 128`, 2.4 → ~0.8 at `-ub 512`.
- **Exactness:** each output still sums over K in the same order inside one thread, so results should be bit-identical;
  `test-backend-ops` and a 4 × 512-token perplexity on the rig confirm it.
- **Fit with the fork:** the fork never touched MMQ, `mmid.cu` or the config tables, and the PR targets master of the same
  week; both commits cherry-pick without conflicts ("E3 and E4 built", section 6).

### 2.5 Tile retuning for Pascal

- **cycle-five/pascal-mmq-autorunner** (P40, Llama-2-7B Q4_0): changing Q4_0's I from 64 to 128 gave pp512 1,009.5 →
  1,174.2 tok/s (+16.3 %) with identical perplexity.
- **Issue #30235** (Tesla P4, compute 6.1): Q8_0 at I = 128, +12 % pp512 on BERT encoders; applying I = 128 to every type
  cost −24 % (Q4_K_M) and −52 % (Q6_K), and I = 160 −46 %. The best tile depends on the type and the card.
- **madsteph74/pascallama.cpp:** I = 96 everywhere, no A/B published.
- **On 5 SMs** a larger I halves the blocks per product, which helps large products (attention q/o, the shared expert)
  and hurts small ones (k, v, expert tiles), so a per-type, per-J search on one P106 is needed. After row 1, dense products
  are ~0.7 of ~5 ms per token, so retuning is worth est. 3–8 %.

### 2.6 Smaller MMQ changes

- **Gate and up in one MMQ launch** with the SwiGLU epilogue (the one-token kernels already fuse them): one activation pass
  and one launch less per layer, est. +1–3 %.
- **FP32 activation sums in MMQ's DS4 layout** (Q4_0, Q4_1, Q5_K on compute 6.1, `mmq.cuh:56-100`), as the one-token
  kernels have (Q76): drops the down-projection rescale for prompts too, ~470 launches per micro-batch, est. +0.5–2 %.
- **Stream-k** is off for Pascal; with 10 resident blocks it would remove the wave tails of small products (est. +2–5 % of
  dense time). Unmeasured.

## 3. Pipelining a prompt across nine cards (question 3)

### 3.1 How llama.cpp overlaps micro-batches

- **Mechanism.** With layer split, all layers on GPUs and no tensor overrides, pipeline parallelism is on
  (`src/llama-context.cpp:527-555`). The scheduler keeps `n_copies` sets of inputs (`ggml-backend.cpp:1973`) and rotates
  them per allocated graph (`:2092-2108`); before reusing a copy on a card it waits for that card's event
  (`:1829-1840`), so up to `n_copies` micro-batches can be in flight.
- **The graph-reuse wait.** `process_ubatch` reuses the previous graph when its shapes match and then calls
  `ggml_backend_sched_synchronize`, a wait for all cards (`src/llama-context.cpp:1537-1545`), because a reused graph does
  not rotate its input copy. The KV length is padded to 256 (`src/llama-kv-cache.cpp:1285-1299`) and is part of the
  mask's shape (`src/llama-graph.cpp:49-66`, `:729-751`), so:
  - at `-ub 128` every second micro-batch reuses the graph and drains the pipeline: at most 2 in flight, est. ~1.6–1.8×
    over no overlap on nine cards;
  - at `-ub` ≥ 256 every micro-batch changes the KV length, so nothing is reused mid-prompt and up to `n_copies` overlap.
- **The host walks the cards in order.** Each card's mask copy blocks the host for its transfer time before the next card's
  split is issued, so at depth the host, not the cards, sets the rate.
- **Hops sit on the source card's stream**, which delays that card's next micro-batch by the transfer (~5–10 % of a
  stage at `-ub 128`, est.).
- **No test so far covers this.** pp512 with `-ub 512` is one micro-batch and never overlaps; the server's prompts and
  llama-bench's deep fills run at `-ub 128`. Every prompt number measured on the rig so far ran without, or with at most
  two-way, overlap.

### 3.2 Memory for micro-batch × copies

Host (pinned) and per-card mask bytes are n_ctx × ub × 2 B × copies: 256 MiB at 262k / `-ub 128` / 4 copies, which fits
the 267 MiB of pinned buffers measured for Flint and the 285 MiB compute buffer per card measured for Qwen3.6 at that
setting. With the desktop running the host has ~2.6 GiB free; llama-server at 262k already
holds ~1.2 GB of mapped embeddings plus checkpoints (~45–70 MiB each), and a ~1 GiB pinned buffer brought the machine
down once ([Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)). The budget used here is ≤ 256–384 MiB pinned
(est.).

| Context | ub × copies with ≤ 256 MiB of mask | Also fits at ≤ 512 MiB |
|---|---|---|
| 32k | 512 × 4 (128 MiB), 1,024 × 4 (256 MiB) | 2,048 × 4 |
| 131k | 256 × 4, 512 × 2, 1,024 × 1 | 512 × 4 |
| 262k | 128 × 4 (today), 256 × 2, 512 × 1 | 256 × 4, 512 × 2 |

On the cards the same mask bytes sit in each compute buffer, plus activations growing with ub (est. ~150 KB per token) and
the last card's logits buffer for ub rows (150,144 × 4 B per row: 73 MiB at 128, 294 MiB at 512, reserved even though a
prompt needs one row; `n_outputs_max` is not exposed on the command line). Nine cards with a G3h-class file have ~1 GB per
card spare at 262k (Q58), enough for any row of the table; the last card may need a `-ts` shift at ub 512. The
sliding-window cache and each checkpoint grow with ub: 768 cells at ub < 256, 1,024 at 256–511, 1,280 at 512
(`src/llama-kv-cache-iswa.cpp:73`).

### 3.3 Which micro-batch × copies

Wall time per prompt token at ~8k on nine cards (est.; overlap factor 1.6–1.8 with two in flight, 2.5–3.3 with four):

| Setting | Today's MMQ | With PR #31047 |
|---|---|---|
| ub 128 × 4 copies (today; reuse wait → 2 in flight) | ~8 | ~4 |
| ub 128 × 4, no reuse wait (row 2) | ~5 | ~2.4 |
| ub 256 × 2 | ~5.5 | ~3.6 |
| ub 256 × 4 (≤ 131k) | ~3.3 | ~2.2 |
| ub 512 × 1 (the 1-copy build) | ~7.6 | ~6.1 |
| ub 512 × 2 | ~4.2 | ~3.4 |
| ub 512 × 4 (≤ 64k, or any context with row 3) | ~2.5 | ~2.0 |

- **Copies beat micro-batch size once the padding is gone.** The 1-copy build is the worst choice except against today's
  setting, so Q89's 1-copy idea is a memory fallback, not a speed-up.
- **Recommended now (no code, pending E1):**
  - ≤ 64k: `-ub 512 -b 2048`;
  - 131k: `-ub 256`;
  - 262k: `-ub 256` on the 2-copy build.
- **After rows 1–3:** `-ub 512` with four copies at every context.

### 3.4 Prompt chunks between other conversations' tokens

- **Card time is what the streams mode pays.** Round 5 priced a prompt token at 5–7 ms of pipeline time, the wall time of
  a prefill that blocks every card today. Entered as pipeline work items (Q104's streams mode), a prompt token costs its
  card time divided over the cards: ~12 ms / 9 × 1.2 (imbalance) ≈ 1.6 ms per token today and ~0.7 ms after rows 1–3
  (est.). At two prompt tokens per generated token the nine-card staggered total becomes 1 / (2 × 1.6 + 3.8 ms) ≈ 140
  tok/s with today's kernels and ≈ 190 tok/s after (est.), against 56–72 with blocking prefills.
- **Chunks are not free here.** On a GPU where a decode batch is bandwidth-bound, added prompt tokens ride along at almost
  no cost; that is the premise of Sarathi-Serve's chunked prefills. On these cards a prompt token adds its full card time,
  and small chunks re-read the experts more often: card time per prompt token after row 1 ~7 ms at 32-token chunks, ~5
  at 64, ~4 at 128 (est., at ~2k of context; section 1.2).
- **Latency for the others.** A chunk occupies each card for its tokens × card time / 9: ~25–30 ms per card for 32 tokens
  after rows 1–3 (est.), about one decode token's whole pass. With one chunk in flight each card spends ~11 % of its time
  on the chunk, and other conversations slow by est. 30–60 % while a prefill runs.
- **Policy:**
  - run a prompt alone through the full pipeline (ub 256–512, four copies) when nothing else is generating or when more
    than ~2k new tokens wait and the user is watching;
  - otherwise feed 64-token chunks at low priority, one in flight;
  - keep the decode tokens ahead of the chunks, as round 5's policy already does for guesses.

## 4. Avoiding prefill (question 4)

### 4.1 What llama-server reuses today

- **The slot's own prefix.** `cache_prompt` reuses the longest common token prefix with the slot's previous request. For
  Flint the 35 sliding-window layers keep only ~513 + ub cells, so reuse past the live window needs a context checkpoint
  at or before the divergence (`server-context.cpp:4056-4160`). Checkpoints are made 4 + ub and 4 tokens before each
  prompt's end and at the start of the last user message (`:4370-4398`); none during or after generation.
  `--ctx-checkpoints` defaults to 32 per slot (`common/common.h:639`), ~45–70 MiB each; 2–4 fit the host.
- **A full match through the generated tokens needs no checkpoint**: the live sliding-window cells are exactly the window
  at the end of generation.
- **Not available for this model as served:** `--cache-reuse` (KV shifting) is refused unless the sliding-window cache is
  full size (`src/llama-kv-cache-iswa.cpp:251-255`, `server-context.cpp:1396-1406`), and `--swa-full` keeps full KV for
  the sliding-window layers (+70 KiB per token, and TILE then computes every cell of them: est. +1.9 ms per prompt token
  at 8k), worthwhile only below ~8k.
- **Host prompt cache:** `--cache-ram` copies whole states (28 KiB per token) into host RAM and must stay 0 on this host
  ([long-context cliff](long_context_cliff.md)).
- **Slot files:** `--slot-save-path` with `POST /slots/<id>?action=save|restore` writes and reads a slot's state with its
  checkpoints appended (`server-context.cpp:2770-2860`), tensor by tensor through one temporary buffer
  (`src/llama-context.cpp:3028-3075`), so even a 131k state needs only ~128 MiB of host RAM at a time.
- **No sharing between slots:** with one KV stream per slot, nothing crosses between conversations except through files;
  the unified cache would share cells but makes every decode step read every conversation's KV (round 5).

### 4.2 How agent harnesses resend history

- **Every request carries the whole conversation**, so the server only has to recognize the prefix:
  - the system prompt and tool schemas first;
  - then the messages;
  - each tool result appended as a new message.
- **Rewrites in the middle break the prefix.** Anything a harness changes inside the history invalidates the cache from
  the first changed token: OpenCode's prune step replaces tool outputs older than ~40k tokens once that saves ≥ 20k
  (constants from a secondary description); compaction replaces the history with a summary; dynamic lines (date, working
  directory, git status) inside the system prompt change it between sessions. At today's 85 tok/s re-reading 40k tokens
  costs ~8 min, so pruning should be off for a local server with 262k of context.
- **Reasoning.** Flint's template renders each assistant message's `reasoning_content` back into the prompt when the client
  sends it. Most OpenAI-compatible clients drop it; then the prefix ends where the previous answer began, and the server
  restores the checkpoint 4 tokens before the previous prompt's end and re-reads the visible answer, the tool call and the
  new tool output.
- **Tool calls are re-rendered**, not replayed: the server parses the generated `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`
  into JSON, the client sends JSON back, and the template writes the XML again with `tojson` for non-string values. Any
  difference from the generated tokens (spacing, number formatting, key order) moves the divergence back to the previous
  prompt's end, so with reasoning kept, the whole reasoning of the last turn is read again (Flint reasons 600–7,900 tokens
  per reply).

### 4.3 How much a cache could serve

Q41's E2 pilot logs record, for 200 BFCL v4 multi-turn sessions (2,364 requests, UD-Q4_K_XL, reasoning kept in the
history), every request's prompt length and generated tokens. Replaying them under four policies (counted per request
against the previous request of the same session; first request processed in full unless stated):

| Policy | Prompt tokens processed | Share of re-sent tokens | Per generated token |
|---|---|---|---|
| No cache (re-send everything) | 18.2 M | 100 % | 39.1 |
| llama-server today (prefix to 4 tokens before the previous prompt's end) | 1.90 M | 10.4 % | 4.1 |
| + generated turn reused exactly (only tool output and new user text) | 1.46 M | 8.0 % | 3.1 |
| + first request's system and tool prompt restored from a saved state | 0.43 M | 2.4 % | 0.9 |

- **The first request dominates what is left:** 5,033 tokens median, 54 % of today's residual. Each later request adds a
  median 153 tokens (mean 398), of which a median 24 (mean 199) is new text; the rest is the model's own previous turn.
- **Coding agents differ in tool output.** A coding session (est.: a 10–15k-token system and tool prompt, 1–3k-token tool
  outputs, 300–800 generated tokens per step, 30 steps) re-sends ~0.9 M tokens, of which a cache with a saved system prompt
  serves ~94–97 %; the fresh 3–6 % is still 1–4 prompt tokens per generated token, which is why prefill speed (sections
  1–3) still sets the total.
- **So caching serves est. 90–97 % of an agent session's prompt tokens today or with small changes**, and the remaining
  fresh tokens make up the prefill load to optimize.

### 4.4 Changes

- **Saved state for the shared prompt.** The first request of every session re-reads the system prompt and tool schemas:
  5k tokens take ~60 s at today's ~85 tok/s, 15k ~3 min. A slot file of the same prefix is 28 KiB per token (140–420 MB)
  and loads in ~1–3 s (SSD ~485 MB/s, ~190 MB/s per card, est.). No code: the harness or a small proxy calls the slot API at
  session start. A fork option could restore automatically when a new prompt starts with a saved prefix.
- **A checkpoint at the end of each generated turn, and at `</think>`.** It makes a re-render difference in the tool call
  cost only the tool call, not the reasoning (one ~45 MiB copy, ~0.25 s). One day in the server.
- **Per-request fixed cost** (row 4): reserve once at start-up with a backend sampler attached, so attaching and detaching
  identical chains does not re-reserve; keep one end-of-prompt checkpoint and copy it asynchronously into a pinned buffer;
  merge the 4-token tail into the last micro-batch when no checkpoint is needed at that point.
- **Long sessions belong on the SSD.** At 131k a lost cache costs a full re-read: est. 30–45 min today, 10–15 min after
  rows 1–3, against ~20–40 s to save or restore the 3.7 GB state (one card's link at a time, ~190–213 MB/s). Saving a
  slot after each turn of a long session, or before switching sessions on one slot, is the cheapest long-context speed-up
  there is.
- **Harness settings:** keep the system prompt static (dynamic lines at its end), disable history pruning, send reasoning
  back only if the template reproduces it exactly (E7 checks), one slot per agent.

## 5. Prior art (question 5)

| Work | What it does | Result | What carries over |
|---|---|---|---|
| llama.cpp PR #31047 (open, 2026-10-23) | Linearized MoE tile list, J by tokens per expert for all GPUs | P40 gemma4 26B-A4B: 1.40× at ub 64, 1.22× at 128, 1.04× at 512; dense unchanged | Row 1 |
| TurboPrefill, llama.cpp PR #24219 (closed RFC, 2026-06) | Accumulates a prompt's output-free micro-batches and dispatches them stage by stage through the layer split | 10 × P104-100: 77 → 345 tok/s (GPT-OSS-120B, 16k), 12 × P104-100 Llama-3-70B 37 → 199; 1.6–3.0× on modern cards | Evidence that llama.cpp's pipeline leaves most overlap unused on Gen1 x1 rigs; row 2 |
| llama.cpp PR #6017 (merged 2024-03) | Pipeline parallelism: input copies, events, overlapping micro-batches | — | The mechanism of section 3.1 |
| llama.cpp issue #28056 (closed) | Why inputs must not be overwritten while a previous micro-batch reads them; sync before `set_inputs` on reuse | No prefill cost measured on one GPU | The origin of the reuse wait |
| llama.cpp PR #28874 (open) | Copies user inputs before cross-device inputs so the host does not wait at each split | — | Q102; matters with one copy |
| llama.cpp PR #29963 (open) | Uploads user inputs once per device, asynchronously | Compute buffers 9,133 → 2,988 MiB per GPU at 262k; MI50 prefill +28 % at 8k, +155 % at 32k (with its other commits) | The cheaper half of row 3 (one copy, async), still 2 B per cell and token on PCIe |
| Issue #30235 (open) | Pascal DP4A Q8_0 I = 128 | Tesla P4 +12 % pp512; other types −24…−52 % | Row 8: per-type tuning |
| cycle-five/pascal-mmq-autorunner | Search over MMQ table rows on a P40 | Q4_0 I 64 → 128: pp512 +16.3 % | Row 8 |
| ik_llama.cpp PR #2142, #2580 | FP32 TILE prefill retile (half2 tiles converted at use, scores in registers); head-256 prefill to the FP32 tile kernel | P100 prefill +4–9 %; +95 % (f16) / +148 % (q8_0) at 16k | Row 6 |
| Sarathi-Serve (OSDI 2024) | Chunked prefills, stall-free scheduling, uniform micro-batches against pipeline bubbles | 2.6× (Mistral-7B, A100), 3.7× (Yi-34B, 2 × A100), 5.6× (Falcon-180B, pipeline parallel) serving capacity | Section 3.4: the schedule carries over, the free piggybacking does not |
| DeepSpeed-FastGen (Dynamic SplitFuse) | Splits prompts and fuses them with decode tokens into constant-size batches | — | Same premise as Sarathi |
| SGLang RadixAttention | Prefix tree of KV blocks shared across requests | — | Section 4: one slot per agent covers it here |
| CachedAttention / AttentionStore (USENIX ATC 2024) | Multi-turn KV caches saved to cheaper memory and storage, layer-wise preloading, scheduler-aware eviction | TTFT −87 %, prefill throughput 7.8×, cost −70 % on multi-turn conversations | Slot files and checkpoints (section 4.4) |
| SageAttention (ICLR 2025) | INT8 Q·Kᵀ with smoothed K, higher-precision P·V | 2.1× over FlashAttention-2 on modern GPUs | Row 6's `dp4a` variant, with a KLD check |

No llama.cpp change that builds the attention mask on the GPU was found (search, 2026-10-24).

## 6. First experiments (question 6)

`BIN` is a build's `bin/`, `MODEL` the file (S4-L0 for continuity, then G3h/K2h), nine cards with the usual split,
`LLAMA_NO_MOE_DOWN_RESCALE=1` (UD-Q4_K_XL `=tg`). llama-bench sizes its context to the prompt, so pinned memory stays small;
deep points use llama-batched-bench, which keeps no state copy (Q79).

**E1. Micro-batch × copies × reuse (rig, no code, ~1.5 h, mostly model loads).** On the existing `pascal-p106-v2` builds
with 4, 2 and 1 copies (Q89's `-sched2`, `-sched1`):

```bash
for B in v2 v2-sched2 v2-sched1; do
  $B/bin/llama-bench -m $MODEL -ngl 99 -fa on -p 512,2048,8192 -n 0 -b 8192 -ub 128,256,512 -r 2 -o jsonl
done
LLAMA_GRAPH_REUSE_DISABLE=1 v2/bin/llama-bench -m $MODEL -ngl 99 -fa on -p 2048,8192 -n 0 -b 8192 -ub 128 -r 2 -o jsonl
```

Predictions (est., pp8192, tok/s): 1 copy 80–95 / 115–135 / 160–175 at ub 128 / 256 / 512; 2 copies 125–170 /
190–240 / 255–310; 4 copies 125–170 (same as 2 copies: the reuse wait) / 295–440 / 400–570; 4 copies with
`LLAMA_GRAPH_REUSE_DISABLE=1` at ub 128 200–310. Kills: 4 copies at ub 512 below 1.5× pp512 (no overlap); 1 copy at
ub 512 below 1.6× ub 128 (the padding model is wrong).

**E2. Where one micro-batch goes (rig, ~30 min).** `GGML_CUDA_PROFILE=/tmp/pp.csv $BIN/llama-bench -m $MODEL -ngl 99 -fa on
-p 512 -n 0 -b 512 -ub 128 -r 1` (then `-ub 512`), `scripts/cuda_profile_summary.py /tmp/pp.csv --tokens 4 --by op`, and the
same with `GGML_CUDA_PROFILE_HOST` at `-p 8192`, and with llama-batched-bench `-c 33024 -npp 32768 -ntg 1 -npl 1` for the
host gaps at depth. Predictions:
`MUL_MAT_ID` ~70–75 % of GPU time at ub 128, ~40–50 % at ub 512; `FLASH_ATTN_EXT` ~4 % at depth 0; at 32k the host spends
~0.4 s per 128-token micro-batch between card splits (mask copies). The profiler waits after every evaluation, so it
measures shares, not overlap.

**E3. PR #31047 (laptop or quant machine 1 day, rig 1 h).** Branch `pascal-p106-v2-prefill` off `pascal-p106-v2` with
the two commits (built: see below); `test-backend-ops -o MUL_MAT_ID` and `-o MUL_MAT` natively and through the
compute_61 PTX; a bit-identity check of the expert products against the base branch; rig: E1's grid on the 1-copy build
first (no overlap effects), then the 4-copy build; a 4 × 512-token perplexity with `-ub 128` against the base. Expected
(est.): 1-copy pp8192 at ub 128 80–95 → 160–200 tok/s, at ub 512 160–175 → 200–240; tiny prompts 2–4× faster.

**E4. Reuse fix (fork, hours).** In `process_ubatch` reuse a graph under pipeline parallelism only when no
micro-batch is in flight (built: see below): prompt graphs after the first rebuild, ~3–5 ms of host time each; decode
keeps reuse and its CUDA graphs. Same rows as E1's `LLAMA_GRAPH_REUSE_DISABLE=1` run; tg128 must not move.

**E3 and E4 built (Q108, Q109, fork branch `pascal-p106-v2-prefill`, commit `82bbb3aa4`, off `pascal-p106-v2`).**
- **Row 1 (Q108).** PR #31047's two commits, "CUDA: linearize MoE data layout for MMQ/MMF" and "Apply MMQ tile size
  optimization more generally", cherry-pick without conflicts: neither the fork nor the 16 upstream commits between the
  fork's base and the PR's base touch `mmq.cu`, `mmq.cuh`, `mmf.cu`, `mmf.cuh`, `mmid.cu` or `mmid.cuh`.
- **Row 2 (Q109): a graph is reused only when nothing is in flight.** `llama_context` sets a flag after each
  `graph_compute` and clears it in `synchronize()`; under pipeline parallelism `process_ubatch` reuses the previous
  graph only while the flag is clear and rebuilds it otherwise.
  - A rebuilt graph goes through `ggml_backend_sched_alloc_graph`, which moves to the scheduler's next input copy, so
    micro-batches overlap up to `n_copies` deep. That is the path every micro-batch at `-ub` ≥ 256 already takes, so it
    adds no new synchronization case.
  - When the flag is clear the reuse wait (`ggml_backend_sched_synchronize`) has nothing left to wait for. That is the
    case of token generation and speculative verify steps: one micro-batch per decode call, after the caller has read
    the previous logits. They keep their reused graphs and CUDA graphs.
  - Results do not change: the same kernels run on the same inputs, only the host stops waiting.
  - **Not E4's `ubatch.n_tokens > 1` rule:** that rule would also rebuild the 2–5-token verify steps of speculative
    decoding and the multi-token kernels (Q78, Q87), which run after a synchronize and gain nothing from a rebuild.
  - **Not reuse with per-copy graphs:** keeping `n_copies` reusable graphs, each bound to its own input copy, would
    also save the rebuild's host time (~3–5 ms per micro-batch, est.). It needs per-copy graph results in
    `llama_context` and a way to pin the scheduler's input copy. Worth doing only if E2 shows host time setting the
    rate at `-ub 128` once the wait is gone.
  - Any other path that synchronizes the scheduler leaves the flag set, which costs at most one rebuild.
- **Tests** (a 7-GPU quant machine, RTX 3090s; correctness only):
  - `test-backend-ops`, native sm_86 code (MMQ with tensor cores, MMF for F16/BF16): `MUL_MAT_ID` for Q8_0, Q4_0,
    Q4_1, Q5_0, Q5_K, IQ4_XS, Q6_K, Q4_K, F16, BF16 and F32 at every batch size (608 cases, among them 92 new ones at
    9–512 tokens with Flint's routing of 256 experts, top 6), `MUL_MAT_ID_FUSION` (13) and `MUL_MAT` for the same types
    (1,600) all pass.
  - The same with a build that has only compute_61 PTX: the driver compiles the Pascal code and ggml's host code, which
    then sees 6.1 as the highest compiled architecture, takes the Pascal decisions too (`dp4a` MMQ with its tile sizes,
    no MMF, no stream-k). `MUL_MAT` and `MUL_MAT_ID_FUSION` pass, and so does `MUL_MAT_ID` for every type but Q8_0,
    whose run stops at its first 5-token case with a launch error, in the base built the same way as well: the host
    reads compute 8.6 and sends Q8_0 batches of up to 8 tokens to the few-token MoE kernel (`get_mmvq_mmid_max_batch`
    in `mmvq.cu`), whose Pascal build allows 4 (`get_mmvq_mmid_max_batch_for_device`). On a compute 6.1 card both read
    the same table.
  - G3 split by layer over two RTX 3090s, llama-perplexity over 8 × 512 tokens at `-ub 128` and `-ub 512`: the branch
    reproduces the base's perplexity in every chunk and writes byte-identical reference-logit files. Compared with the
    base it gives the same KLD statistics as the base compared with itself (mean 0, maximum 6 × 10⁻⁵ from the rounding
    of the stored logits, the same top token everywhere), so the logits are identical. With
    `LLAMA_GRAPH_REUSE_DISABLE=1` they are identical as well.
- **Measuring it on the rig.** On nine cards. `V2` and `PF` are the `bin/` of the `pascal-p106-v2` and
  `pascal-p106-v2-prefill` tarballs, `V2S1` and `PFS1` their 1-copy builds (`…-sched1`), `MODEL` S4-L0 for continuity
  (then G3h/K2h), `WIKI` wikitext-2's `wiki.test.raw`:

  ```bash
  export LLAMA_NO_MOE_DOWN_RESCALE=1
  # 1. same results: 4 x 512 tokens at -ub 128 (expect KLD 0 and 100 % same top token)
  PPL="-m $MODEL -ngl 99 -fa on -f $WIKI -c 512 --chunks 4 -b 512 -ub 128 --kl-divergence-base /tmp/pf.kld"
  $V2/llama-perplexity $PPL && $PF/llama-perplexity $PPL --kl-divergence
  rm /tmp/pf.kld
  # 2. the prompt grid, 1-copy builds first, then 4 copies (E1's rows for V2 and V2S1)
  for B in $V2S1 $PFS1 $V2 $PF; do
    $B/llama-bench -m $MODEL -ngl 99 -fa on -p 512,2048,8192 -n 0 -b 8192 -ub 128,256,512 -r 2 -o jsonl
  done
  # 3. row 2 alone: V2 without any reuse at -ub 128
  LLAMA_GRAPH_REUSE_DISABLE=1 $V2/llama-bench -m $MODEL -ngl 99 -fa on -p 2048,8192 -n 0 -b 8192 -ub 128 -r 2 -o jsonl
  # 4. generation must not move
  for B in $V2 $PF; do $B/llama-bench -m $MODEL -ngl 99 -fa on -p 0 -n 128 -r 3 -o jsonl; done
  ```

  Reading it:
  - At `-ub` 256 and 512 nothing is reused mid-prompt, so `PF` against `V2` (and `PFS1` against `V2S1`) is row 1 alone.
    pp512 at `-ub 512` is one micro-batch, so it shows row 1 without any overlap.
  - At `-ub 128` step 3 against `V2` is row 2 alone, and `PF` against step 3 is row 1. On the 1-copy builds row 2
    replaces a full wait every second micro-batch by each card's wait for its own previous micro-batch.
  - Expected (est., pp8192, sections 3.3 and 6): `V2` at `-ub 128` 125–170 tok/s, step 3 200–310, `PF` 1.5–2× step 3;
    `PF` at `-ub 512` 1.2–1.4× `V2`; tg128 within noise.

**E5. Mask on the card (fork, 2–3 days, then rig).** The mask input becomes per-cell positions and sequence ids (4–8 B per
cell, per copy) plus the micro-batch's positions; a `KQ_MASK` op (CPU reference and CUDA kernel) expands them on each card
into the f16 tensor, which the allocator then holds once per card instead of once per copy. Correctness: masks bit-identical
to the host's for causal, sliding-window and several streams; then llama-batched-bench at `-c 32768/131072`,
`-npp 32768` / `131072`, `-ub 256/512`, four copies, against the base.

**E5 built (fork branch `pascal-p106-v2-gpumask2`, commit `6a842d091`; the prompt path first as Q110, decode as Q116).**
- **The op.** `GGML_OP_KQ_MASK` (CPU reference and CUDA kernel) takes F32 cell pairs `[2, n_kv, n_stream]` (the cell's
  position; bit s set when the cell holds sequence s, none for an empty cell; both exact in F32 below 2^24), with any
  strides in the cell and stream dimensions so that it reads a window of a larger array, and I32 token pairs
  `[2, n_tps, n_stream]` (position; sequence id 0–23). It writes the f16 or f32 mask by the host's rule: 0 where the
  token and the cell share a sequence, the cell is not later than the token and lies inside the sliding window
  (standard, chunked or symmetric), −∞ elsewhere.
- **Cell copies on the cards.** Each KV cache keeps these pairs for all its cells on every card that holds its layers
  (8 bytes per cell, 1 MiB at 131k, plus one scratch row). A step scatters into each copy, with `SET_ROWS`, the cells it
  writes and the cells that `seq_rm` or a `seq_cp` within a stream changed since the previous step. That list has a
  fixed 32 entries, the unused ones pointing at the scratch row, so speculative decoding's removal of rejected draft
  tokens after every step keeps the graph as it is. More changes than that, any other change to the cells (clear,
  `seq_keep`, `seq_add`, `seq_div`, a context shift, a stream copy, a state load), and any step whose cells did not
  reach every copy (a step on the host mask, a graph that leaves out a card's layers) mark the copies stale; the next
  step then uploads all cells. The purge inside a step's own slot update (the sliding-window cache dropping a
  sequence's older cells) leaves the copies valid: those cells lie outside the window of every later token of that
  sequence, so their old values mask them just as well.
- **The graph.** For the sliding-window cache pair (Flint's) and the plain cache, the attention input is one F32 tensor
  per step: the tokens' positions and sequences, then each cache's update (rows and values), all integers below 2^24
  and so exact; only while the copies are stale does a full upload come with it. Every graph input reaches a card as a
  synchronous copy, so the input is staged once per card (a copy node and a cast of its integers, which the scheduler
  callback places on the card with the masks) and everything else reads views of it there: a card takes one copy per
  step, of about 0.5 KB for one token, where the host mask takes two (one per cache) of 2 bytes per cell and token.
  The first layer on each card updates that card's copy (the copy lives in the card's KV buffer, so the scheduler runs
  the update there), and every layer gets its own mask node from the updated copy, placed on the layer's card as the
  norms are. In a prompt micro-batch the mask also sits once in the card's compute buffer instead of once per pipeline
  copy. All of these nodes keep their addresses from step to step, so llama.cpp reuses the graph and CUDA graphs keep
  replaying it; the graph is rebuilt where it was before (when n_kv grows by 256) and once after a full upload.
- **When.** Every micro-batch with causal attention, no ALiBi or M-RoPE, on offloaded caches with their own cells and at
  most 24 sequences (`LLAMA_GPU_KQ_MASK=<min tokens>`, default 1; 0 off). The rest keeps the host mask, as do the
  caches inside the hybrid memory (Qwen3.6's), whose input code reads the mask tensor itself.
  `LLAMA_GPU_KQ_MASK_CHECK=1` synchronizes before each step's inputs, reads every card's copy back, applies the step's
  inputs to it, expands the mask by the op's rule and aborts if a single value differs from the host mask.
- **Checked, prompt path (Q110, RTX 3090, the op natively and through the compute_61 PTX):** the op exact against the CPU
  in 20 cases; G3 on two cards, flash attention on: prefill with one sequence (`-c 1024 -ub 256`), four sequences in
  four streams and in one unified stream, one-token and four-sequence decode steps. All 1,606 masks the check expanded
  were identical to the host's, and each run's perplexity equals the host-mask run's to every printed digit, with the
  same top token everywhere and KLD at the base file's storage floor.
- **Checked, cell copies and decode (Q116), on the CPU backend:** random tiny models of llama (dense and MoE), qwen3,
  flint, gemma2, gemma3, cohere2 and gpt-oss through `test-llama-archs` (prompt, decode, mixed batches, a state round
  trip): all pass, and all 143 masks the check expanded from the copies were identical to the host's, 65 of them right
  after a full upload. On three of those models `llama-lookup` (drafts of up to 8 tokens, the rejected tail removed
  after every step) and `llama-parallel` (a system prompt copied to four clients' sequences, finished clients removed;
  four streams, and one unified stream) give the host-mask runs' output token for token, with the same number of graph
  reuses.
- **Expected on the rig (est., from section 1.4's table, nine cards):** prompt processing at `-ub 128`: mask copies
  3.2 → ~0.1 ms per prompt token at 32k and 12.5 → ~0.4 ms at 131k; card time 20–22 → 17–19 ms and 52–54 → 40–42 ms;
  more on the wall clock where the host's synchronous copies set the pace, and `-ub 512` with four copies at 262k
  without 1 GiB of pinned host mask. The expansion itself writes the mask once per layer on its card (~55 µs per
  full-attention layer at 32k and `-ub 128`, est.). Decode at depth: the host mask copies go, ~3 ms per token at 32k and
  ~12 ms at 131k, plus one synchronous copy per card at every depth. Rig run: llama-batched-bench at `-npp 32768` and `131072` with `LLAMA_GPU_KQ_MASK=0` and the default
  (prompt and generation columns), a CUDA graph count over a short decode, and the KLD check of one-token decode.

**E6. Fixed cost per request (rig, ~30 min).** llama-server with `-lv 4` (timestamps for `sched_reserve` and checkpoint
traces), a two-turn chat with `server_bench.sh`, four runs: ± `-bs` × `--ctx-checkpoints 0/4`. Expected: `-bs` adds one
re-reserve per request (0.1–0.5 s), each checkpoint 0.2–0.3 s.

**E7. Cache replay of agent transcripts (laptop, offline, minutes).**
- **The method of section 4.3:** for each request take the previous request's prompt plus generated tokens and count what
  the live cache, a checkpoint and a saved system prompt would serve.
- **Re-render check:** render each logged history with Flint's GGUF template (`reasoning_content`, parsed `tool_calls`),
  tokenize it, and find the first token that differs from prompt + generated tokens. That decides whether the end-of-turn
  checkpoint (section 4.4) is needed.
- **Then real harnesses:** the same replay on a few recorded OpenCode or pi sessions, which have larger tool outputs than
  BFCL.

**E8. Slot save and restore (rig, ~20 min).** Save after a 5k-, 15k- and 32k-token prompt, restart the server, restore, and
time both with host memory logged; then send the same prompt plus one turn and check that only the new tokens are read.

## What this changes in the plan

- **Priority 7 (prompt processing) moves up next to the pipelined executor:** for agents it sets the total, and its top
  three changes are a cherry-pick, a one-line condition and a 2–3-day op, all exact.
- **Q89's grid gains the reuse arm and the PR #31047 build;** the 1-copy build stays a memory fallback.
- **Q80 should report pp2048 and pp8192 at `-ub` 256/512**, never pp512 alone: pp512 is one micro-batch and hides the
  overlap.
- **Round 5's prefill price falls:** prompt chunks in the streams mode cost card time (est. ~1.6 ms of pipeline time per
  prompt token today, ~0.7 after rows 1–3), not 5–7 ms.
- **Agent serving (priority 8) gains the cache items:** saved system-prompt states, end-of-turn checkpoints, and the
  per-request re-reserve and checkpoint costs.
- **Breakthrough candidate 3 (on-card mask) matters for prefill throughput at depth, not only for memory:** the
  synchronous per-card copies are 15 % of card time at 32k and 23 % at 131k today (est.).

## Already covered

| Direction | Where it is |
|---|---|
| Scheduler copies and `-ub` at long context, the 1- and 2-copy builds | Q89; [breakthrough candidates](breakthrough_candidates.md), candidate 3 |
| Mask computed on the card | Breakthrough candidate 3; Q65 (mask copies of speculative steps); this note adds the per-token PCIe cost of prompts |
| Prompt sweeps, `GGML_CUDA_FORCE_MMQ`/`CUBLAS`, issue #30235 | Q80; [survey tracker](../../experiments.md) E1.6, E2.3, E2.4 |
| PR #28874 (input copy order) | Q102; [Pascal forks](pascal_forks_and_quants.md) |
| MMQ autotuner, `mmq_y` 96, PR #31047 and TurboPrefill named | [Pascal forks](pascal_forks_and_quants.md), "For prompt processing" and section 1 |
| Prompt chunks as pipeline work, streams mode, round 5's prefill price | Q104; [multi-conversation serving](multi_conversation_serving.md) |
| Multi-token kernels for 2–4-token tails | Q78 |
| Decode attention at depth, q8_0 KV scratch, KV-sharded attention | Q59, Q60, Q61, Q95; [attention note](pascal_attention_long_context.md) |
| Host prompt cache and llama-bench's state copy on 4 GB | Q79; [long-context cliff](long_context_cliff.md) |
| Headless mode, second SO-DIMM | Q83, Q86 |
| Instruction-throughput calibration | Q103 |

## Not verified

- MMQ's ~2.0 T multiply-adds per second per card is fitted to pp512, tiny prompts and Qwen3.6, not measured; the padding
  share rests on it.
- The overlap factors (1.6–1.8 with two micro-batches, 2.5–3.3 with four) are a model; no prompt run on the rig has
  overlapped more than two micro-batches.
- That the reuse wait is what limits overlap at `-ub 128`; host waits on events or copies could limit it too (E1 tells).
- TILE's prompt throughput (~1 TFLOPS assumed) and its shared-memory limit at 32 rows.
- Mask copies at ~190 MB/s one card after another; copies behind a shared switch may be slower.
- PR #31047's gain on 5 SMs, and bit-identity of its results on a compute 6.1 card itself (the checks under "E3 and
  E4 built" ran on RTX 3090s; step 1 of its rig commands is the check).
- The decomposition of the ~2.2 s per request (re-reserve, checkpoints, tails, fill and drain).
- Expert bytes per token assume independent routing and ~2.3 MB per expert; MMQ streaming at 120–140 GB/s is assumed.
- TurboPrefill's mechanism and settings on the P104 rigs; its page gives speeds, not configurations.
- Harness behaviour: OpenCode's prune constants come from a secondary description; system-prompt and tool-output sizes of
  coding agents are estimates; whether Flint's template reproduces generated tool calls and reasoning token for token.
- The BFCL replay counts tokens from logged prompt lengths, assuming each request's history extends the previous one
  exactly.
- Slot file restore times on the rig; the ~1 GB headroom per card at 262k for larger `-ub`.

## Sources

llama.cpp and the fork (P)
- Fork `pascal-p106-v2` at `032b130f4`:
  - `ggml/src/ggml-cuda/mmq.cu` (J choice `:189-225`, RDNA rule `:197-201`, expert path `:276-353`, `ggml_cuda_should_use_mmq`
    `:355-424`);
  - `mmq.cuh` (config `:164-213`, tile loop `:873-943`, expert early exit `:1024-1026`, grid `:1413-1437`);
  - `mmq-config-pascal-dp4a.cuh` (`:1-273`), `mmq-vec-dot.cuh` (`:110-140`), `mmid.cu`;
  - `ggml-cuda.cu` (`MUL_MAT_ID` dispatch `:2066-2108`, host-sorted fallback `:2110-2160`, `cpy_tensor_async` `:2649-2712`);
  - `fattn-tile.cuh` (FP32 table `:89-156`, column switch `:1149-1235`);
  - `ggml/src/ggml-backend.cpp` (`GGML_SCHED_MAX_COPIES` `:911-912`, input copies `:1823-1864`, splits `:1867-1945`,
    `n_copies` `:1973`, `alloc_graph` `:2092-2108`, `synchronize` `:2133-2145`);
  - `src/llama-context.cpp` (pipeline conditions `:527-555`, `sched_reserve` `:731-885`, `set_sampler` `:1407-1459`, reuse
    and wait `:1523-1591`, file I/O `:3028-3075`); `src/llama-graph.cpp` (`:30-66`, `:729-751`); `src/llama-graph.h`
    (`:839-869`); `src/llama-kv-cache.cpp` (`:1285-1299`); `src/llama-kv-cache-iswa.cpp` (`:73`, `:251-255`);
  - `tools/server/server-context.cpp` (release `:447`, cache reuse `:1396-1406`, sampler `:2033-2036`, checkpoints
    `:2668-2731`, slot files `:2770-2860`, prefix and checkpoint restore `:4056-4160`, prompt splitting and checkpoints
    `:4370-4455`); `common/common.h` (`:639-642`); `common/arg.cpp` (`--ctx-checkpoints`, `--cache-ram`, `--swa-full`,
    `--slot-save-path`).
- [PR #31047](https://github.com/ggml-org/llama.cpp/pull/31047) "CUDA: linearize MoE data layout for MMQ/MMF" and its diff
- [PR #24219](https://github.com/ggml-org/llama.cpp/pull/24219) "Intra-Prompt Pipeline Scheduling for Multi-GPU Prefill";
  [TurboPrefill](https://github.com/sergey-automation/TurboPrefill)
- [PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017), [issue #28056](https://github.com/ggml-org/llama.cpp/issues/28056),
  [PR #28874](https://github.com/ggml-org/llama.cpp/pull/28874), [PR #29963](https://github.com/ggml-org/llama.cpp/pull/29963),
  [issue #30235](https://github.com/ggml-org/llama.cpp/issues/30235)
- [cycle-five/pascal-mmq-autorunner](https://github.com/cycle-five/pascal-mmq-autorunner),
  [madsteph74/pascallama.cpp](https://github.com/madsteph74/pascallama.cpp)
- ik_llama.cpp [PR #2142](https://github.com/ikawrakow/ik_llama.cpp/pull/2142), [PR #2580](https://github.com/ikawrakow/ik_llama.cpp/pull/2580)

Serving and caching (P unless marked)
- A. Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve", OSDI 2024,
  [arXiv 2403.02310](https://arxiv.org/abs/2403.02310)
- C. Holmes et al., "DeepSpeed-FastGen", [arXiv 2401.08671](https://arxiv.org/abs/2401.08671)
- L. Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (RadixAttention),
  [arXiv 2312.07104](https://arxiv.org/abs/2312.07104)
- B. Gao et al., "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention", USENIX
  ATC 2024, [arXiv 2403.19708](https://arxiv.org/abs/2403.19708)
- J. Zhang et al., "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration", ICLR 2025,
  [arXiv 2410.02367](https://arxiv.org/abs/2410.02367)
- OpenCode context compaction (prune and compaction phases), [DeepWiki description](https://deepwiki.com/vbgate/learn-opencode/7.3-context-compaction-system) (S)
- Gemma 4 26B-A4B's 128 experts with top-8 routing (the P40 model of PR #31047), [vLLM recipe](https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it) (S)

This project
- [Quant registry](../../../quants/experiments.md) (Q19, Q28, Q41, Q51, Q54, Q58, Q76, Q78, Q79, Q80, Q89, Q102, Q103,
  Q104, Q105), [Flint benchmark](../../benchmarks/2026-10-22-flint.md), [Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md),
  [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md), [hardware spec](../../mining_rig.md),
  [multi-conversation serving](multi_conversation_serving.md), [breakthrough candidates](breakthrough_candidates.md),
  [Pascal forks and quants](pascal_forks_and_quants.md), [attention note](pascal_attention_long_context.md),
  [long-context cliff](long_context_cliff.md), [kernel integration note](pascal_mmvq_integration.md) (section 14),
  [formats and kernels survey](quant_speed_b_formats_kernels.md), `scripts/cuda_profile_summary.py`, `scripts/server_bench.sh`
- Q41's E2 pilot logs (200 BFCL v4 multi-turn sessions, per-request prompt and generated token counts), replayed for
  section 4.3
