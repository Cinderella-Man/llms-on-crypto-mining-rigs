# The host in the decode loop: where a 2-core Celeron's time goes per token, and how far it can leave the loop (2026-10-24)

Scope: what the rig's host (a Celeron 3865U, 2 cores at 1.8 GHz, 4 GB of RAM) does for every generated token when
Flint runs as a layer split over six or nine P106-090 cards with CUDA graphs on compute 6.1; which of that work sits on
the critical path and which overlaps the cards; how much of it can be moved onto the cards (GPU sampling, the sampled
token fed to the first card without a host round trip, card-to-card hand-off without host synchronization, CUDA graphs
over the whole token, device-side token loops); what limits several conversations on two cores (Q104); and what
llama-server adds per token. It builds on [multi-conversation serving](multi_conversation_serving.md) (round 5, whose
host table this note takes apart), [pipelined speculation](pipelined_speculation.md), [speculation plumbing](speculation_plumbing.md),
[sampling for tool requests](tool_request_sampling.md), [prompt processing](prefill_pascal.md) and the persistent-kernel
sketch in [breakthrough candidates](breakthrough_candidates.md). Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (code, docs, papers, measurements in a repository) and secondary.
- `file:line` refers to the fork's branch `pascal-p106-v2` at `032b130f4`; "pipespec" marks `pascal-p106-v2-pipespec` at
  `a217f42c8` (the executor of Q87/Q104). CUDA files are in `ggml/src/ggml-cuda/`.
- "Serial" host work delays the next card's start; "overlapped" host work runs while a card computes. N is the number
  of conversations generating at once.

Starting point, measured ([registry](../../../quants/experiments.md), [benchmarks](../../benchmarks/)):
- S4-L0 on six cards, fork build: 37.4 tok/s in llama-bench tg128 (26.8 ms per token, of which ~2.8 ms is the logits
  copy llama-bench requests); real chat on nine cards with backend sampling (`-bs`) 31.9 tok/s, ~31.3 ms per token
  (Q58), at a depth that grows to ~8–20k over the conversation.
- With CUDA graphs the host launches all six cards' graphs within ~1.9 ms of a token's start while each card computes
  4–8 ms (Q49). In the host-profiled trace of that run the host spends 0.12–0.17 ms inside each card's `graph_compute`
  and enters the next card's ~0.32 ms later (this note's reading of the trace).
- CUDA graphs on compute 6.1 removed 4.9 ms per token before the fork (makers' Q4_1, five cards, 24.56 → 27.92 tok/s);
  `-bs` saved 3.3 ms per token in the server (44.5 → 41.2 ms, nine cards) ([upstream patches](upstream_pascal_patches.md),
  [speed-ups](../../benchmarks/2026-10-22-speedups.md)).
- Small MoE models on one card lose 4–11 % when both CPU cores are busy, 0–6 % with one
  ([small-MoE overhead](../../benchmarks/2026-10-22-small-moe-overhead.md)).
- Round 5's host model: ~0.25–0.35 ms per card per token plus synchronous mask copies; one core submits ~240–305 tokens/s
  at 8k on nine cards, against a card ceiling of ~262 tok/s with a light last card.
- Nsight Systems and nvprof refuse the P106-090 ([experiments tracker](../../experiments.md) E2.2), and CUDA timing events
  change the timing on these cards (12 events per token slowed S4-L0 from 37.4 to 23.4 tok/s, Q49). Neither can give the
  host's share of a token here.

## Verdict

- **Within a token the host is off the critical path; between tokens it is not.** Once card 1's graph is launched the host
  stays ahead of the cards: it needs ~0.25–0.35 ms per card and each card computes 3–6 ms. What is serial is the gap from
  the last card's sample to card 1's next launch: reading the token, the server's per-token work, `llama_decode`'s
  bookkeeping, the mask fill, a one-row embedding lookup **on the CPU** (llama.cpp keeps the input layer on the host,
  `src/llama-model.cpp:1689-1690`) and card 1's synchronous input copies. Est. 0.3–1.5 ms per token at ≤ 8k
  (Q87's simulator assumes 1.5 ms), plus 0.4–0.8 ms of host-staged hops on the GPU side (Q81 unmeasured): **est.
  2–7 % of a 31 ms token identified**.
- **The largest term is unexplained: llama-server ran ~8–10 ms per token slower than llama-bench before the fork.** Flint
  on nine cards: llama-bench 35.6 ms per token including its 2.8 ms logits copy, llama-server with `-bs` 41.2 ms, so ~8 ms
  that neither sampling nor the copy accounts for; Qwen3.6 on nine cards: 53–54 against 64.5 ms, of which host sampling
  explains est. 1–2 ms. If even part of that gap persists on the fork, where a served token takes ~31 ms, the server path
  is **est. 10–30 % of a served token**, more than everything above. No measurement so far separates it.
- **Both cores are kept busy by waiting.** With nine CUDA contexts and two logical CPUs, CUDA's default schedule makes the
  waiting thread yield-spin ("If C > P, then CUDA will yield…"), so the main thread occupies a core for the whole token.
  The fork builds with OpenMP, and the token's one CPU graph (the embedding row) runs as a 2-thread OpenMP region
  (default `-t 2`); afterwards libgomp's worker spins 300,000 iterations before sleeping, est. ~20 ms of the second core
  per token at Skylake-family `pause` latency. The HTTP thread, the streaming parser, Xorg and GNOME get what is left.
  `-t 1` (or `OMP_WAIT_POLICY=passive`) removes the spinner at no cost: the CPU computes one 8 KB row per token.
- **Streaming re-parses the whole reply for every token.** `common_chat_session::feed` appends the new text and runs
  `common_chat_parse` over everything generated so far (`common/chat.cpp:1753-1760`), on the HTTP thread for each SSE
  chunk: O(n) per token, O(n²) per reply. Est. 0.1–0.15 ms per token at 760 tokens on the Celeron (0.024 ms measured on a
  desktop core) and ~1–1.5 ms at 8k. Every speed this project has published used non-streaming requests
  (`scripts/server_bench.sh`); agent clients stream.
- **At N = 8 the host, not the cards, sets the rate.** Per token the executor's submit costs est. 3.3–4.2 ms at 8k
  against a 3.8 ms card cycle (est. 90–110 % of it), and ~45–60 synchronous host-to-card copies per token (est.) are
  most of the per-card cost: every graph input (positions, masks, KV indices, the embedding row) goes through
  `ggml_backend_tensor_copy` → `cudaMemcpyAsync` + `cudaStreamSynchronize` (`ggml-backend.cpp:1829-1840`,
  `ggml-cuda.cu:846-851`), in pipespec too. One asynchronous, packed upload per card per token (later a memcpy node
  inside the card's CUDA graph), with the decode mask built on the card (Q116), brings the submit cost to est.
  1.0–1.8 ms per token: 550–1,000 submissions per second on one core, so the cards' ceiling binds again at every context.
- **The sampled token can reach card 1 without the host.** In the executor, the next token's slot can be enqueued before
  the current one finishes, with its token id arriving as a 4-byte card-to-card copy from the sampler's output and the
  embedding lookup moved onto card 1; the host only reads the ids afterwards for detokenization and stop checks, and
  drops the enqueued token on a stop or grammar rejection (the executor's drop path). Est. +2–5 % single stream; it hides
  per-conversation latency at N > 1 but does not raise the ceiling. SGLang's overlap scheduler ("future tokens", 1.1×)
  and vLLM's Model Runner V2 (GPU-side input preparation, 56 % on a 0.6B model) are the same idea on fast hosts; rec
  returns 8 tokens per dispatch.
- **A device-side token loop across the cards is not possible with CUDA graph conditional nodes**: a conditional body's
  nodes "must all reside on a single device". Per-card WHILE loops chained through mailboxes in mapped host memory are
  possible in principle (sm_61 support is undocumented; a 10-minute sample run settles it), but after the two changes above
  they would save est. ≤ 1 % of a single stream for weeks of work. Without P2P a card cannot write into another card's
  memory; both can only share pinned, mapped host memory, which is what the driver's staged copy already uses.
- **Blocking synchronization would hurt today** (each of the ~50 synchronous copies per token would pay an OS wake-up;
  on a DGX Spark the opposite change, spin instead of the default, gave +19–46 % generation, llama.cpp PR #16585). It
  becomes useful only for threads that wait without submitting, after the copies are asynchronous.
- **First measurement (rig, ~30 min, no code):** the same file, build and cards in llama-bench tg128, llama-server `-bs`
  non-streaming and streaming, with per-thread CPU (`pidstat -t` or `top -H`), major page faults per token
  (`/proc/<pid>/stat`), the nine cards' NVML utilization summed every 100–200 ms (single stream: the share of wall time with
  a kernel running somewhere), and `-t 1` against `-t 2`. Then, if `perf` is on the rig, uprobes on `libcudart.so.12`'s
  synchronize and launch calls, which split every token into host-busy and GPU-wait time without CUDA events.

## Ranked changes

Gains are against today's fork build (`pascal-p106-v2`), nine cards, ≤ 8k of context, `-bs`; "N conversations" means
the executor's streams mode (Q104) at N = 8 unless stated.

| # | Change | Single stream (est.) | N conversations (est.) | Effort | First experiment (rig, ≤ 30 min) | Kill criterion |
|---|---|---|---|---|---|---|
| 1 | **Find and cut the server-path gap**: `-t 1` (no OpenMP spinner), explicit `-np`, the submit thread and the server/HTTP threads on separate cores, `--no-mmap` if page faults show (the embedding table is read on the CPU every token) | 0–25 % (up to the ~8–10 ms pre-fork gap; `-t 1` alone est. 0–5 %) | frees 0.3–0.8 of a core | hours (flags) to 2 days (thread placement in the executor) | M0 (section 5.1) | llama-server within 1 ms per token of llama-bench minus the logits copy, and no thread but the main one above 30 % of a core |
| 2 | **One asynchronous, packed input upload per card per token** from a per-slot pinned staging buffer, on the card's compute stream (then captured as a memcpy node in the card's CUDA graph); with Q116's on-card decode mask | +0.5–1 % (only card 1's copies are serial today) | host per token 3.3–4.2 → 1.0–1.8 ms at 8k, 5.6–6.5 → 1.1–1.9 ms at 32k; one core submits 550–1,000 tok/s, so N = 8 reaches the cards' 235–256 (~210 at 32k) | 3–5 days (`ggml-backend.cpp` input path with `caller_sync` slots, `llama-context` staging; upstream drafts #30685, #29963) | M1 on pipespec streams N = 4: count and time of `cudaStreamSynchronize` calls per token | synchronous copies < 0.5 ms of host time per token |
| 3 | **Device-fed next token**: the sampler's id copied card-to-card into card 1's token input, the embedding lookup on card 1, the next token enqueued before the current one finishes (an executor slot whose token is the previous slot's sample) | +2–5 % (the sample→launch gap and the CPU embedding split) | lower time between tokens below saturation; no ceiling change | 4–6 days on pipespec (input-layer placement flag ½ day) | M1: time from the sample's synchronize return to the next token's first graph launch | gap < 0.5 ms per token |
| 4 | **Incremental streaming parse** (parse only when a delimiter token can change the message; plain reasoning and content as appended deltas) and **no whole-prompt detokenize per request** (`server-context.cpp:2376`) | 0–3 % on long streamed replies (CPU contention only) | 3–15 % of a core | 1–2 days | the same 4k-token reply streamed and not streamed, HTTP thread CPU and client-side time between chunks | < 1 % difference and < 5 % of a core |
| 5 | **Synchronization policy**: keep spin/yield for the submitting thread; blocking sync (PR #26487) only for collector threads, after #2 | blocking now est. −2 to −10 %; 0 after #2 | +0.2–0.5 core after #2 | hours (an `LD_PRELOAD` shim that sets the device flags, or an env switch in the fork) | Spin / default (yield) / BlockingSync, tok/s and CPU per thread | BlockingSync within 1 % and ≥ 30 % of a core saved → adopt; otherwise keep the default |
| 6 | Hand-off through mapped pinned mailboxes (the producer card writes its 8 KB activation and a flag into pinned host memory, the consumer reads it zero-copy) instead of staged peer copies and events | +0.3–1 % | small | 1–2 weeks | Q81's hop benchmark with a mailbox variant | staged hop ≤ 0.1 ms, or the mailbox not ≥ 30 % faster |
| 7 | Device-side token loop: per-card CUDA graph WHILE loops (or pre-launched per-token graphs) chained through host-memory mailboxes, positions and cells advanced on the card | ≤ 1 % over #2 + #3 | matters only above ~600 submissions per second | 3–6 weeks | NVIDIA's `graphConditionalNodes` sample and a stream-memory-ops attribute query on one P106 (10 min) | WHILE nodes fail on sm_61, or #2 already below 1 ms per token |
| 8 | Two submit threads (cards 1–4 and 5–9) | 0 | 1.3–1.7× submit capacity, at the server's expense | 1–2 weeks (per-thread split ranges in the scheduler) | after #2: host time per token at N = 8 | host below 0.8 × the card cycle after #2 |
| — | `cudaLaunchHostFunc` chains, nine host threads (one per card), one CUDA graph over all nine cards | rejected (sections 2.5, 3.3) | | | | |

Tool requests keep their own lever, GPU sampling with a host grammar check (Q67, +9–11 % est.), which this note does not
repeat.

## 1. The host's per-token budget today (question 1)

### 1.1 What the host does for one decode token

llama-server with `-bs`, one conversation, nine cards, CUDA graphs captured and reused. The steps, in order, with the
fork's code and an estimate on the Celeron (est. unless measured):

| # | Step | Code | Cost (est.) | Serial / overlapped | Basis |
|---|---|---|---|---|---|
| 1 | Wake from the wait for the last card (CUDA's default schedule yield-spins here) | `ggml-cuda.cu:2706-2711`; `ggml-backend.cpp:2133-2137` loops over all ten backends | 5–30 µs; up to a scheduler slice if another thread holds the core | serial | CUDA runtime docs; the other synchronize calls hit finished streams |
| 2 | Read the sampled id (four small async copies already queued), accept it in the sampler | `src/llama-context.cpp:1168-1183`; `server-context.cpp:4740-4748` | 0.02–0.05 ms | serial | accept 0.002–0.004 ms on a desktop core × 4–6 ([tool sampling](tool_request_sampling.md) §2.2) |
| 3 | Detokenize the token, check stop strings, queue a partial result for the HTTP thread | `server-context.cpp:4764`, `:2088-2130`, `:2299-2342` | 0.02–0.1 ms | serial | short string work; one mutex and condition variable |
| 4 | Server loop: task queue, slot iteration, batch build | `server-queue.cpp:278-330`; `server-context.cpp:3431`, `:3700-3725` | 0.02–0.1 ms | serial | |
| 5 | `llama_decode` front: batch validation, ubatch, KV cell, graph-reuse test | `src/llama-context.cpp:1523-1547` | 0.05–0.2 ms | serial | |
| 6 | `set_inputs`: positions, KV indices, the f16 masks filled cell by cell | `src/llama-context.cpp:1581`; `src/llama-graph.cpp:30-47` | 0.05–0.1 ms at 8k, 0.2–0.4 ms at 32k | serial | 5–10 ns per cell (est.) |
| 7 | CPU split: one embedding row by `get_rows` from the host-resident table, as a 2-thread OpenMP region | `src/llama-model.cpp:1689-1690`; `ggml/src/ggml-cpu/ggml-cpu.c:3433-3453` | 0.01–0.1 ms; a page fault on the mapped table adds 0.1–20+ ms | serial | the table is mapped from the model file (1.17 GB in F32 for the makers' Q4_1) |
| 8 | Card 1's inputs: the 8 KB embedding row, positions, masks, KV indices, each a synchronous host-to-card copy | `ggml-backend.cpp:1829-1840` → `ggml-cuda.cu:846-851` | 0.1–0.3 ms | serial | ~15–30 µs per small synchronous copy; a 16 KB mask at 8k ≈ 0.09 ms at ~190 MB/s |
| 9 | Card 1's graph launch (reused graph: same split uid, no property check) | `ggml-cuda.cu:4929`, fast path `:2769-2775` | 0.02–0.05 ms | serial | Q49: 0.12–0.17 ms inside `graph_compute` per card with the profiler's own two events |
| 10 | Cards 2–9: the same input copies, the graph launch, the hop enqueued on the source stream with an event the destination waits on | `ggml-backend.cpp:1865-1945`; `ggml-cuda.cu:2687-2700` | ~0.25–0.35 ms per card, 2–2.8 ms in all | overlapped while the host stays ahead | Q49 (~0.32 ms between card entries) |
| 11 | Hops: 8 staged copies of 8 KB through host memory | `cudaMemcpyPeerAsync`, no P2P | 0.05–0.1 ms each, 0.4–0.8 ms | serial on the GPU side, no host time | two DMA legs at ~200 MB/s plus latency (est.; Q81 open) |
| 12 | Last card: output head, the GPU sampler chain (top-k by radix select over 150,144 logits, then top-p, min-p, temperature, draw on ≤ 20 survivors) | `top-k.cu:240-340`; `src/llama-sampler.cpp:1483-1519` | 0.1–0.5 ms GPU | serial (GPU) | `top-k.cu` uses CUB's DeviceTopK only from CCCL 3.4.3, newer than CUDA 12.9's, so one row takes the radix path |
| 13 | Wait for the last card | | the rest of the token | host idle but spinning | |
| 14 | HTTP thread (streaming only): update the parsed message, JSON for the chunk, SSE write | `server-queue.cpp:551-575`; `server-task.cpp:159-170`, `:991-996`; `common/chat.cpp:1753-1760` | 0.1–0.3 ms plus the re-parse: 0.1–0.15 ms at 760 tokens, ~1–1.5 ms at 8k | overlapped, on the other core, contends | 0.024 ms per token at 760 tokens on a desktop core (tool sampling note) × 4–6, linear in the reply's length |
| 15 | libgomp's worker after step 7: 300,000 spins before it sleeps | GCC `GOMP_SPINCOUNT` default | ~20 ms of the second core per token | overlapped, contends | 300,000 × ~140 cycles (`pause` on Skylake-family cores, Intel's optimization manual) at 1.8 GHz (est.) |

**Totals (est.).** Serial host work (steps 1–9) 0.3–1.2 ms at ≤ 8k and 0.6–1.5 ms at 32k; Q87's simulator uses 1.5 ms.
Hops add 0.4–0.8 ms. Steps 10, 14 and 15 overlap the cards as long as a core is free for them. Together the identified
part is 0.7–2.3 ms of a ~31 ms served token, **2–7 %**. On tool requests (host sampling) the logits copy adds 2.82 ms on
the last card's stream and the host chain 0.4–0.7 ms (Q55).

### 1.2 What is measured

| Measurement | Value | What it says about the host |
|---|---|---|
| Host launch span, six cards, CUDA graphs (Q49) | all graphs launched within ~1.9 ms of a token's start; each card computes 4–8 ms | steps 8–10 overlap the cards; the host is not the limit inside a token |
| CUDA graphs on compute 6.1, before the fork | 24.56 → 27.92 tok/s (makers' Q4_1, five cards): −4.9 ms per token | eager launches cost ~5 ms per token; with graphs the launch cost is the ~0.02–0.05 ms per card of step 9 |
| `-bs`, before the fork, nine cards | 44.5 → 41.2 ms per token | host sampling path = copy + host chain ≈ 3.3 ms |
| llama-bench against llama-server, Flint, nine cards, before the fork | tg128 28.09 tok/s (35.6 ms, with the 2.8 ms copy) against 24.26 tok/s with `-bs` (41.2 ms) | **~8 ms per token in the server path that sampling does not explain** (est.: 41.2 − (35.6 − 2.8)) |
| The same for Qwen3.6, nine cards, 32k ([Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)) | 18.5–18.9 against 15.5 tok/s (53–54 against 64.5 ms), server without `-bs` | ~11 ms, of which host sampling over 248k logits explains est. 1–2 ms |
| CPU contention on one card (small-MoE benchmark) | both cores busy: −4 to −11 %; `-t 1` = `-t 2` in llama-bench | a busy core costs little while nothing else needs it |
| Host-profiled trace of S4-L0, six cards (Q38/Q49's run) | median 30.8 ms per token against a 26.9 ms GPU chain; 14 of the last 40 tokens took 46–58 ms, **at the same token positions in both profiled runs** (kernels on and off) | something tied to the token sequence (llama-bench feeds `std::rand() % n_vocab`, `llama-bench.cpp:2205-2219`) stalls tokens by 15–27 ms under the profiler; an unprofiled tg128 run's three repetitions took 3.60–3.61 s each |

The two pre-fork comparisons differ from llama-bench in more than the server: `-np` auto (four slots, unified KV), a
262,144-token cache, short chat prompts at temperature 0. None of these changes a one-token step's arithmetic at short
depth (the mask and attention read only the used cells), so the gap most likely sits in host scheduling: the server's
threads competing with two spinning waiters (steps 1 and 15), or page faults on the mapped embedding table under the
4 GB host's memory pressure (step 7). The repeated stall positions in the profiled trace fit token-dependent faults,
but the profiler's own file writes could also explain them; M0's fault counter decides.

### 1.3 What is missing

- An unperturbed host timeline: CUDA events distort the token (Q49) and Nsight Systems refuses the card. Host clocks
  and uprobes on the CUDA runtime's calls are the remaining tools (section 5.1).
- The server-against-bench gap on the fork build, and the same request streamed.
- Page faults per token, per-thread CPU during decode, and the effect of `-t 1`.
- The hop's real cost (Q81) and the GPU sampler chain's time on a P106 (tool sampling experiment 5).
- The host sampling chain on the Celeron itself (Q68).

## 2. Taking the host out (question 2)

### 2.1 GPU-side sampling

- **Upstream.** Backend sampling (PR #17004, merged 2026-01-04) runs the sampler chain in the graph; follow-ups made it
  work with several outputs per sequence (#25532), drafters (#24655, #26958), static graphs across ubatches (#30223). Open:
  #25919 (reasoning budget with `-bs`), #30918 (fused penalties), #29576 (smaller host output buffers), #30196 (one
  synchronize for all outputs: +6.9 % with DFlash on an RTX PRO 6000). Grammar requests still switch it off for the whole
  request ([tool sampling](tool_request_sampling.md) §1); nothing upstream changes that.
- **The executor.** pipespec samples on the last card per sequence: greedy (`--pipe-backend-greedy`) or the
  temperature chain with each draw keyed by (seed, position) through `llama_sampler_set_pos` (`include/llama.h:1509`),
  so a token submitted before its predecessor is sampled draws the right noise. Only the 4-byte id comes back.
- **On a P106.** The chain's cost is the radix top-k over 150,144 floats (a few passes over 600 KB, est. 0.1–0.3 ms) plus
  small kernels on ≤ 20 survivors, all inside the last card's CUDA graph; no host launch cost.
- **Gain left for chat:** none beyond `-bs` (already on). For tool requests: Q67. For speculation: rows per position.

### 2.2 Feeding the sampled token straight into the next step

**Today.** The id goes to the host (step 2), back through the server (steps 3–5), and the embedding row is looked up on
the CPU from the host-resident table (step 7) and copied to card 1 (step 8). Every next token waits for that round trip.

**Design: a device-fed slot in the executor.**
1. **Input layer on card 1.** A flag (environment variable or `llama_model_params` field) next to
   `src/llama-model.cpp:1689-1690` puts `token_embd` and its `get_rows` on the first GPU. The token tensor becomes a
   4-byte input of card 1's split. VRAM: the table's size on card 1 (~0.33 GB in Q8_0, 1.17 GB in F32; F32 tables should
   be converted). `-ot token_embd.weight=CUDA0` tests the placement without code, but any tensor override switches
   pipeline parallelism off (`src/llama-context.cpp:527-532`), so it is a decode-only test.
2. **Token from the device.** When slot s+1 is submitted with "token from slot s", pipespec queues a
   `cpy_tensor_async` from slot s's sampled-id tensor on the last card to slot s+1's token input on card 1, with the
   usual event (the same path as a hop, `ggml-cuda.cu:2687-2700`), and skips the host upload of the token.
3. **Everything else is known in advance.** Positions, KV cells and the masks of a one-token step do not depend on the
   token's value, so the host prepares slot s+1's inputs while slot s runs, in the slot's own buffers (the executor's
   slots already own them).
4. **The host reads behind.** It collects slot s's id when its event fires, detokenizes, checks stop strings and the
   grammar, and if the token ends the reply or is rejected it drops the already-enqueued slot s+1 and removes its cell
   (`llama_memory_seq_rm`), exactly as for a wrong guess.

**Exactness.** The same graph and kernels run for the same token; the embedding row is copied (F32) or dequantized by
`get_rows` on the card instead of the CPU. For quantized tables the GPU's fused multiply-add can round differently from
the CPU build (FMA off), so the check is the executor's identity test, with KLD as the fallback criterion if the last bit
of the embedding moves.

**Gain (est.).** The serial gap of section 1.1 (0.3–1.5 ms) shrinks to the 4-byte hop (~0.05–0.1 ms): +2–5 % on a
31 ms token. It also removes the CPU split, the 8 KB upload and any page fault on the mapped table. At N ≥ S_eff the
cards are busy anyway, so the ceiling does not move.

**Prior art.**
- SGLang's overlap scheduler runs one batch ahead and resolves the next batch's inputs from "future tokens" on the GPU:
  1.1× throughput over its previous version (Llama-3.2-3B, 4,096 input and 2,048 output tokens).
- vLLM's Model Runner V2 keeps request state on the device and prepares inputs with GPU kernels that consume the
  previous step's sampling results; outputs reach the CPU on a separate stream: 25K against 16K output tok/s (+56 %) for
  Qwen3-0.6B on a GB200, 6.3 % lower time per token for GLM-4.7 with MTP on four GB200s.
- TensorRT-LLM's overlap scheduler accepts one extra decoding step for the same overlap.
- rec samples on the GPU and loops 8 tokens per dispatch, returning them 8 at a time ([rec note](rec_engine.md) §2).

### 2.3 Device-side "next token" loops: CUDA graph conditional nodes

- **What exists.** IF and WHILE nodes arrived with CUDA 12.3–12.4, IF/ELSE and SWITCH with 12.8. A kernel sets the
  condition with `cudaGraphSetConditional()`; the body runs again while it is non-zero. Device graph launch (graphs
  instantiated with `cudaGraphInstantiateFlagDeviceLaunch`, launched from a kernel) exists since CUDA 12.0.
- **The rule that decides it here:** a conditional body's nodes, like a device-launched graph's, "must all reside on a
  single device", and may contain only kernel, empty, memcpy, memset, child-graph and conditional nodes (no host nodes).
  A loop over the whole token, nine cards, cannot be one WHILE node.
- **Compute 6.1.** NVIDIA's documentation states no compute-capability requirement; NeMo's support check tests only a
  driver ≥ 12.3 (the rig's 580 driver reports 13.0, and CUDA 12.9 has the API). Whether instantiation succeeds on sm_61
  is untested: NVIDIA's `graphConditionalNodes` CUDA sample on one P106 answers it in minutes.
- **What a rig version would be.** One WHILE graph per card: wait for the upstream mailbox (a kernel polling a flag in
  pinned, mapped host memory), run the card's layers, write the activation and a flag into the downstream mailbox (a memcpy
  node into mapped host memory is allowed), advance position and cell counters on the card; the last card samples and
  writes the id into card 1's mailbox and into a ring the host reads. The decode mask must be built on the card (Q116);
  the n_kv shape changes every 256 tokens, so a loop runs at most 256 tokens before the host rebuilds it.
- **Costs and risks.** Polling kernels hold a block and send PCIe reads while they wait (sm_61 has no `__nanosleep`, so
  the backoff is a clock loop); a card's poll traffic shares the switch uplink with the others' hops; stop strings and
  grammars are host decisions, so the loop runs ahead and the host must cancel it through a flag and roll back cells;
  none of it fits ggml's graph model.
- **Gain (est.).** Over changes #2 and #3 the loop removes the remaining per-token graph launches (~9 × 20–50 µs) from
  the host: ≤ 1 % single stream, and the host's submit cost at N conversations, which #2 already brings to ~1–2 ms per
  token. Weeks of work for that.
- **Prior art.** Galvez et al. (Interspeech 2024) moved RNN-T greedy decoding into a WHILE loop: 2.5× end to end for a
  1.1B model on a GPU that sat idle ~80 % of the time; NeMo measured 3.1× (600M) and 2.7× (1.1B) on an A100. The gains
  come from a host that could not keep up with tiny kernels, the situation CUDA graphs already fixed here.

### 2.4 Spinning and blocking on two cores

- **What runs today.** No llama.cpp code sets device flags on compute 6.1 (only compute 12.1 gets spinning,
  `ggml-cuda.cu:404-411`, PR #16585). The default, `cudaDeviceScheduleAuto`, yields "if C > P", C being the process's
  active contexts and P the logical CPUs: nine contexts against two CPUs, so every wait is a yield loop that keeps a core
  busy but lets other threads run on it.
- **Wait points per token.** One long wait for the sampled token, ~45–60 short synchronizations inside the user-input
  copies (steps 8 and 10; est. from 5–7 graph inputs per card), ten short ones in `ggml_backend_sched_synchronize`; in pipespec one event wait per slot.
- **Blocking (`cudaDeviceScheduleBlockingSync`, PR #26487 as `GGML_CUDA_BLOCKING_SYNC`, open):** frees the core but every
  wait pays an interrupt and a thread wake-up, est. 20–80 µs on the Celeron; ~50 per token would add 1–4 ms of host time,
  most of it hidden behind the cards except card 1's copies and the final wait: est. −2 to −10 %. The PR reports
  "essentially the same" throughput on a Quadro P5000 (one card, a 9B model), without numbers. The opposite change, spin
  instead of the default, gave +19 % and +46 % generation on a DGX Spark (PR #16585), where extra synchronize calls each
  paid a wake-up.
- **Spin (`cudaDeviceScheduleSpin`)** never yields: lowest latency, but on two cores it starves the server thread and the
  desktop. The default yield loop is the right choice for the submitting thread.
- **Persistent host threads per card:** nine spinning threads on two cores would time-slice against each other and
  against the server; threads waiting in blocking mode pay the wake-up per hop. Rejected. The right shape is one
  submitting thread (spin or yield, pinned to one core) and one thread for the server and collection on the other.
- **The OpenMP spinner** (step 15) is a separate waste with the same effect, removed by `-t 1` or
  `OMP_WAIT_POLICY=passive`. ggml sets `KMP_BLOCKTIME=200` (`ggml-cpu.c:3924-3930`), which only LLVM/Intel OpenMP reads;
  the build links GCC's libgomp, and `--poll` applies only to ggml's own thread pool, which an OpenMP build does not use.

### 2.5 Host functions, mapped pinned memory and stream-ordered hand-off

- **`cudaLaunchHostFunc`.** The function runs after the stream's earlier work and blocks later work; it "must not make any
  CUDA API calls", and host functions of independent streams "may be serialized". It could only prepare host data that a
  later stream-ordered copy reads (for example next-token inputs); its dispatch is a driver thread's wake-up, comparable
  to the main thread's. With the inputs of a one-token step known in advance (2.2, point 3) there is nothing left for it
  to do. Rejected.
- **Card-to-card writes without P2P.** These cards have no P2P ([hardware spec](../../mining_rig.md)), so no CUDA mapping
  lets a kernel on one card store into another card's memory. Both cards can map the same pinned host buffer
  (`cudaHostAllocPortable | cudaHostAllocMapped`, or `cudaHostRegister`): the producer's stores are posted writes over its
  x1 link, the consumer's loads are PCIe reads (~1–2 µs per round trip, est.), and a flag written after
  `__threadfence_system()` orders them. That is the mailbox of 2.3 and of the persistent-kernel sketch; it moves the same
  8 KB over the same two links as the driver's staged copy and saves only the copy engines' setup and the event
  (est. 10–20 µs per leg, 0.1–0.3 ms per token). Worth building only if Q81 finds ≥ 0.2 ms per hop.
- **Stream-ordered hand-off already exists.** llama.cpp orders hops with `cudaEventRecord` on the source stream and
  `cudaStreamWaitEvent` on the destination (`ggml-cuda.cu:2697-2700`); the documentation says an event "may be from a
  different device" and that the wait is "performed efficiently on the device when applicable". No host thread waits per
  hop. Stream memory operations (`cuStreamWaitValue32`/`WriteValue32` on a value in host memory) would replace the event
  with a semaphore in host memory; their V2 versions are on by default since the CUDA 11.7 documentation, and support on
  compute 6.1 is a device-attribute query. They add nothing over events for hops; they matter only for a device loop.
- **One CUDA graph over all nine cards.** CUDA graphs may span several GPUs, so one graph per token (input uploads as
  memcpy nodes from pinned staging, nine cards' kernels, hop copies, the sampler's id back) would cut the host to one
  launch per token. ggml captures one graph per split and backend; a cross-backend capture is a scheduler redesign
  (maintainers on PR #31004 point to a scheduler per graph), and whether staged peer memcpy nodes capture without
  P2P is unverified. Change #2 gets most of the same host saving card by card. Rejected for now.

## 3. Several conversations (question 3)

### 3.1 What limits submissions per second

Per token in streams mode (pipespec `pipe_submit`, `src/llama-context.cpp:1747-1890` on pipespec), nine cards (est.):

| Part | ≤ 1k | 8k | 32k | Basis |
|---|---|---|---|---|
| Batch, KV cell, reuse test, `set_inputs` | 0.2–0.4 | 0.25–0.45 | 0.4–0.7 | section 1.1 steps 5–6 |
| CPU embedding split and its upload | 0.05–0.15 | 0.05–0.15 | 0.05–0.15 | step 7 |
| Per card: 5–7 synchronous input copies, graph launch, hop enqueue (× 9) | 1.8–2.5 | 1.8–2.5 | 1.8–2.5 | step 10 |
| Mask copies inside those (n_kv × 2 B to each card) | ~0.1 | 0.8 | 3.1 | ~190 MB/s per link, one card after another |
| Output copy, event, collection, sampler readback | 0.05–0.1 | 0.05–0.1 | 0.05–0.1 | |
| Server work per token (streaming adds the re-parse on the other core) | ~0.25 | ~0.25 | ~0.25 | round 5 |
| **Total on the submitting core** | **2.5–3.4** | **3.3–4.2** | **5.6–6.5** | round 5's table |

At 8k one core submits 240–305 tokens per second against the cards' ~262 (light last card) — the host is ~90–110 % of
the card cycle — and at 32k only 155–180. The second core is not free: the OpenMP spinner (with `-t 2`), the HTTP thread
with its O(n) re-parse for every streamed token, Xorg and GNOME share it.

### 3.2 What raises the ceiling

Host time per token on the submitting core after each change, nine cards (est.):

| | ≤ 1k | 8k | 32k | 131k | Submissions per second at 8k |
|---|---|---|---|---|---|
| Today | 2.5–3.4 | 3.3–4.2 | 5.6–6.5 | 14.9–15.8 | 240–305 |
| + decode mask on the card (Q116) | 2.4–3.3 | 2.5–3.4 | 2.6–3.5 | 2.8–3.7 | 295–400 |
| + one async packed upload per card (#2) | 1.0–1.7 | 1.0–1.8 | 1.1–1.9 | 1.2–2.0 | 550–1,000 |
| + device-fed tokens (#3) | 0.9–1.6 | 0.9–1.7 | 1.0–1.8 | 1.1–1.9 | 590–1,100 |

With Q116 and #2 the cards bind again at every context: 235–256 tok/s at N = 8 (round 5's staggered estimate), ~210 at
32k (attention on the first card), and groups of two above S_eff (337–403 at N = 16) fit as well. The per-card cost
after #2 is the graph launch, one async copy and the hop's event (est. 40–80 µs); the rest is llama-side bookkeeping.

- **Server work moves off the submitting core.** Round 5 already asked for a submit thread separate from the server; with
  `-t 1` and the incremental parse (#4) the second core carries the server for all conversations (est. 0.25 ms per token
  plus 0.1–0.6 ms of re-parse per streamed token today, 3–16 % of a core at 262 tok/s).
- **One host thread per group of cards** (#8) splits a token's nine launches over two threads; it needs a hand-off per
  token between the threads (the downstream thread may wait on an event only after the upstream one has recorded it)
  and a scheduler that submits a range of splits. On two cores it takes the server's core. Only if #2 leaves the host
  above ~0.8 of the card cycle.
- **Batched submissions** of several conversations to one card do not cut launches (one graph per token per card); only
  merging two conversations into one ubatch does (round 5's groups of two, which halve launches per token above S_eff).
- **Per-conversation graph rebinding** is already gone in the streams mode (slots bound to conversations). What remains:
  a rebuild when a conversation's n_kv crosses a 256-cell boundary (`src/llama-kv-cache.cpp:1285-1299`), est. 5–15 ms
  of host plus one uncaptured step, about once per second at 262 tok/s over eight conversations (~1 % of host time);
  padding decode slots' n_kv to 1,024 cells cuts that by 4× for a few masked cells more per attention call. In
  llama-server `-np N`, every change of the active slot set changes the merged batch and costs a rebuild and a recapture
  (10–20 ms, [speculation plumbing](speculation_plumbing.md) §1.6); the streams mode avoids it.

## 4. The server path (question 4)

### 4.1 Per token

| Item | Where | Cost on the Celeron (est.) | Notes |
|---|---|---|---|
| Sampling readback and accept | `server-context.cpp:4740-4748` | 0.02–0.05 ms | host chain 0.4–0.7 ms more without `-bs` |
| Detokenize one token, stop strings | `:4764`, `:2088-2130` | 0.02–0.1 ms | `validate_utf8` checks only the last 4 bytes (`server-common.cpp:1022`) |
| Partial result into the queue | `:2299-2342` | ~0.01 ms | streaming only |
| HTTP thread: re-parse of the whole reply, diff, JSON, SSE write | `server-queue.cpp:569`, `server-task.cpp:159-170`, `common/chat.cpp:1753-1760` | 0.1–0.3 ms + 0.1–0.15 ms at 760 tokens, ~1–1.5 ms at 8k | O(n²) per reply; Flint thinks 600–7,900 tokens per reply (Q58) |
| Timing print | `server-context.cpp:702-720` | ~0 | every 3 s after 100 tokens |

### 4.2 Per request (agent clients)

| Item | Where | Cost (est.) | Notes |
|---|---|---|---|
| Chat template render and tokenization of the re-sent history (10–50k tokens) | HTTP thread | 0.1–0.5 s on the Celeron | runs beside the decode loop on the other core; a special-token partition bug once made this quadratic (issue #12724, fixed by PR #12706: 4.66 → 0.24 s for 500 turns) |
| Prefix match against the slot | `server_tokens::get_common_prefix` | < 1 ms | O(n) token compare |
| Sampler attach re-reserve with `-bs`, checkpoints, 4-token tail | [prefill note](prefill_pascal.md) §1.5 | ~2 s per request in all | covered there (Q111) |
| Whole-prompt detokenize for the final response, on the main loop | `server-context.cpp:2376` | 5–20 ms per 50k-token prompt | done for every request; only `/completion`'s `prompt` field uses it |

### 4.3 Settings on this host

| Setting | Effect here | Recommendation |
|---|---|---|
| `-t` / `--threads` | default = physical cores = 2 (`common/common.cpp:294-300`); the CPU only looks up one embedding row per token, as a 2-thread OpenMP region whose worker then spins ~20 ms (est.) | `-t 1 -tb 1` (pending M0) |
| `--poll` | used only by ggml's own thread pool; this OpenMP build ignores it | `OMP_WAIT_POLICY=passive` if `-t` stays 2 |
| `--threads-http` | default max(`-np` + 4, cores − 1) = 5 at `-np 1` (`server-http.cpp:150-153`); idle threads cost nothing per token | default |
| `--no-mmap` | the CPU-resident embedding table stays in anonymous memory instead of evictable page cache; also Q84's load path | if M0 shows major faults during decode; only for tables ≤ ~0.4 GB on the 4 GB host |
| `-np` | auto gives four slots and a unified cache (round 5) | explicit `-np 1` (or N for the executor) |
| `--cache-ram 0`, `--ctx-checkpoints 1–4` | host RAM on 4 GB (Q79, Q111) | as in the setup guide |
| `cudaDeviceScheduleAuto` (yield) | section 2.4 | keep; no blocking sync |

### 4.4 Known llama.cpp items

- PR #30685 (draft, 2026-10-16): asynchronous input copies in the scheduler, returned to draft after a Vulkan slowdown; a
  reviewer notes that asynchronous copies "do use GPU time" where CPU copies into host-visible memory don't.
- PR #29963 (open): graph inputs uploaded once per device, asynchronously on the split's stream (compute buffers 9,133 →
  2,988 MiB per GPU at 262k).
- PR #30881 (draft) and #30986 (open): a batched tensor copy API and CUDA gathers into mapped host memory; motivated by
  one copy and one stream wait per range.
- PR #29796 (draft): skips redundant `cudaStreamSynchronize` calls on one device: +16.9 % without speculation on Windows
  (RTX PRO 6000); maintainers want the fix in the scheduler instead.
- PR #30196 (open): one synchronize for all of a batch's outputs (+6.9 % with DFlash).
- PR #11867 (open since 2025): overlaps CUDA graph building with execution; 2.8–3.6 % generation on fast CPUs, "more
  significant" on slow ones per the author.
- PR #28058 (open): makes the synchronize before `set_inputs` unconditional (fixes cross-request contamination on
  integrated GPUs); after a sampled token the context is already idle, so plain decoding pays only the calls, but it
  rules out overlapping `llama_decode` calls.
- PR #26487 (open): `GGML_CUDA_BLOCKING_SYNC` (section 2.4). PR #16585 (merged): spin on compute 12.1.
- Issue #21042 (closed): the server's default `--threads` slowed one setup by 50 % against llama-cli; issue #25121
  (closed): llama-bench against llama-server differences traced to different defaults.

## 5. Plan (question 5)

### 5.1 The first measurement: the host's real share of a token

**M0 (rig, ~30 min, no code).** One model (S4-L0 for continuity, then the release candidate), the `pascal-p106-v2` build,
nine cards, short context; nothing else on the cards. `$B` and `$M` as in the other notes' commands.

```bash
B=/opt/llama-builds/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin; M=/mnt/models2/flint/flint-S4L0.gguf
E="LLAMA_NO_MOE_DOWN_RESCALE=1"
ssh rig 'mkdir -p /tmp/host'
# 1. reference: llama-bench tg128 (its token includes the 2.8 ms logits copy), -t 1 and -t 2
ssh rig "$E $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 3 -t 1,2 -o md"
# 2. llama-server -bs, -np 1, short context; once with -t 2 (today), then restarted with -t 1
ssh rig "$E setsid nohup $B/llama-server -m $M -ngl 99 -fa on -c 16384 -np 1 -bs --cache-ram 0 -t 2 \
    --port 8089 > /tmp/host/server.log 2>&1 < /dev/null &"
# 3. around each client run of ~1,000 tokens (non-streaming, then streaming, same prompt and seed):
#    minor and major faults before and after (fields 10 and 12), threads, card utilization, swap
ssh rig 'P=$(pgrep -f llama-server); awk "{print \$10, \$12}" /proc/$P/stat > /tmp/host/faults_before.txt; \
    top -H -b -d 1 -n 40 -p $P > /tmp/host/threads.txt & \
    timeout 40 nvidia-smi --query-gpu=timestamp,index,utilization.gpu --format=csv,noheader -lms 200 > /tmp/host/util.csv & \
    vmstat 1 40 > /tmp/host/vmstat.txt &'
# ... client run ...
ssh rig 'awk "{print \$10, \$12}" /proc/$(pgrep -f llama-server)/stat > /tmp/host/faults_after.txt'
```

The client is a few lines of Python on the rig (urllib, `stream` false and true, `max_tokens` 1,000, temperature 1,
fixed seed), recording the server's `timings.predicted_per_second` and, when streaming, the time between SSE chunks.
Read:
- **Server gap** = server ms per token − (llama-bench ms per token − 2.82 ms). Above ~1 ms the server path costs real
  time on the fork too.
- **Per-thread CPU** (`top -H`): the main thread near 100 % is the yield-spin; a second thread near 70–80 % with `-t 2`
  is the OpenMP worker; the HTTP thread's share grows with the streamed reply's length.
- **Major faults** (field 12 of `/proc/<pid>/stat`, before and after) per generated token: any at all means the mapped
  embedding table is being re-read from the stick during decode.
- **GPU-busy share**: NVML's utilization is the share of each sample period with a kernel running; summed over the nine
  cards per timestamp it is the share of wall time with a kernel running somewhere (single stream: one card at a time).
  1 − that sum is the host, hops and inter-kernel gaps together.
- `-t 1` against `-t 2`, streamed against not streamed: the size of rows 1 and 4 of the ranking.

**M1 (rig, ~30 min, needs `perf` with uprobes; installing it is the owner's call).** The binaries link
`libcudart.so.12` dynamically, so its entry points can be probed without rebuilding or CUDA events:

```bash
L=$(ssh rig "ldd $B/llama-server | awk '/libcudart/{print \$3}'")
for f in cudaStreamSynchronize cudaEventSynchronize cudaGraphLaunch cudaMemcpyAsync; do
  ssh rig "perf probe -x $L $f && perf probe -x $L ${f}%return"
done
ssh rig 'perf record -e "probe_libcudart*:*" -p $(pgrep -f llama-server) -o /tmp/host/up.data -- sleep 10 && \
    perf script -i /tmp/host/up.data > /tmp/host/up.txt; perf probe -d "probe_libcudart*:*"'
# afterwards: copy /tmp/host off the rig, stop the server, rm -rf /tmp/host
```

Per token this gives: the long synchronize (the wait for the sampled token) and the time spent in it; the number of
synchronous copies and their summed time; the interval from the long synchronize's return to the next first
`cudaGraphLaunch` (the serial gap of change #3); and the host time between card launches (change #2). A probe costs a few
µs on the Celeron (est.), ~100 hits per token. Without `perf`, the same timeline needs a host-clock ring buffer in the
fork (entry and exit of `llama_decode`, `set_inputs`, each split's launch, the synchronize, the sampler, `process_token`),
about a day, built on the quant machine.

### 5.2 Experiments per change

| Change | Experiment | Expected (est.) | Decides |
|---|---|---|---|
| #1 settings | M0's `-t 1`/`-t 2`, streamed/not, faults; then `taskset` the server's HTTP threads to core 1 | 0–10 % | which flags go into the setup guide and `server_bench.sh`; whether `--no-mmap` belongs there |
| #2 async packed uploads | M1 on pipespec streams (`rig_streams.sh`, N = 4, 8k): synchronous copies per token and their time | ≥ 1 ms per token of synchronous copies | build #2 before the llama-server integration of the streams mode |
| #3 device-fed token | M1's serial gap on plain decoding and on pipespec `d0` | 0.4–1.5 ms per token | build #3 in pipespec |
| #3 placement alone | `-ot 'token_embd\.weight=CUDA0'` on llama-bench tg128 and llama-server (decode only: the override disables pipeline parallelism) | +0.1–0.5 ms per token; no faults | whether the input layer on card 1 is worth its VRAM before #3 |
| #4 parse | M0's streamed run with a 4k-token reply; HTTP thread CPU over the reply | 0.5–1.5 ms per token of HTTP CPU late in the reply | build #4 |
| #5 sync policy | an `LD_PRELOAD` library (built on the quant machine) whose constructor calls `cudaSetDeviceFlags` for each device before llama.cpp starts: Spin, default, BlockingSync | BlockingSync −2 to −10 % today | keep the default; revisit after #2 |
| #7 device loop | NVIDIA's `graphConditionalNodes` sample built for sm_61 on the quant machine, run on one P106; query the stream-memory-ops attributes | instantiates or not | whether the loop is possible at all |

Kill criteria are those of the ranking table.

## What this changes in the plan

- **Before any more single-stream engine work is credited on served speed, M0 runs once.** All published speeds are
  non-streaming; llama-bench and llama-server differed by ~8–10 ms per token before the fork, more than most planned
  kernel changes save.
- **The streams mode of Q87/Q104 needs asynchronous inputs (#2) and Q116 before its llama-server integration**: without
  them one core cannot feed eight conversations at 8k, and at 32k not even six.
- **The executor gains a device-fed slot (#3)**, the single-stream use of the same machinery as guesses: the next token
  is always "guessed" right because it is the sample itself.
- **Setup guide candidates:** `-t 1`, explicit `-np`, and (if M0 shows faults) `--no-mmap` with a quantized embedding
  table; `server_bench.sh` should also measure a streamed run.
- **Device-side loops and zero-copy mailboxes stay ideas**, kept for after #2 and #3 and Q81.
- Suggested registry rows (not added here): M0/M1 as one host-share measurement; #2, #3, #4 as engine items under
  priority 2; the sm_61 conditional-node capability test.

## Already covered

| Direction | Where it is |
|---|---|
| Host sampling cost for tool requests, GPU sampling with a host grammar check, llguidance masks | [tool sampling](tool_request_sampling.md); Q55, Q66–Q69 |
| Host sampling chain timed on the Celeron | Q68, [bench/tool_sampling](../../../bench/tool_sampling/) |
| CUDA graphs on compute 6.1 (PR #27721), launch overhead per node | [upstream patches](upstream_pascal_patches.md); Q34; [small-MoE overhead](../../benchmarks/2026-10-22-small-moe-overhead.md) |
| Graph rebuild, recapture and per-shape graphs for verify steps | Q52; [speculation plumbing](speculation_plumbing.md) |
| Host launch span within a token, event-profiler distortion | Q49; [kernel note](pascal_mmvq_integration.md) §10 |
| Round 5's host submit rate and the executor's streams mode | Q104; [multi-conversation serving](multi_conversation_serving.md) §1.4, §3.3 |
| Executor slots, `caller_sync`, drop instead of withdraw, races | Q87, Q93; [pipelined speculation](pipelined_speculation.md) |
| Mask built on the card for prompts and decode | Q110, Q116; [prefill note](prefill_pascal.md) |
| Graph-reuse wait under pipeline parallelism, scheduler copies, input copy order (PR #28874) | Q109, Q89, Q102 |
| Persistent kernel per card and its host-free end state (mailboxes in mapped memory) | Q88; [breakthrough candidates](breakthrough_candidates.md) §2 |
| Hop cost between cards | Q81 |
| Per-request fixed cost (re-reserve, checkpoints), prompt caching for agents | Q111; [prefill note](prefill_pascal.md) §1.5, §4 |
| Headless mode, second SO-DIMM, `--no-mmap` load time | Q83, Q86, Q84 |
| Lighter last card, split output head | [multi-conversation serving](multi_conversation_serving.md) §2.3; [unconventional ideas](unconventional_ideas.md) idea 6 |
| Tensor and expert parallelism over x1 links | [multi-GPU physics](multigpu_pascal_performance.md) Q3–Q4 |
| rec's 8-token dispatches and GPU grammar masks | [rec note](rec_engine.md) |

## Not verified

- Every per-step cost in sections 1.1 and 3.1 is an estimate from the code and from Q49's profiled trace; none was timed
  without the profiler on the rig.
- That the pre-fork server gap (~8–10 ms per token) persists on the fork, and where it sits.
- That libgomp's spin lasts ~20 ms per token on this CPU (the `pause` latency of Kaby Lake is taken from Intel's
  Skylake figure), and that the worker thread spins at all at the rig's token rate.
- Page faults on the mapped embedding table during decode, and whether the profiled trace's repeated stalls come from
  them or from the profiler.
- The re-parse cost at 8k tokens (scaled linearly from one desktop measurement at 760 tokens).
- The synchronous copy count per card (5–7 inputs per split, from the graph's input list) and their ~15–30 µs each.
- Wake-up latency of blocking synchronization on the Celeron.
- CUDA graph conditional nodes, device graph launch and stream memory operations on compute 6.1.
- Whether a staged peer copy needs any driver host thread, and whether such copies can be captured into a multi-device
  graph without P2P.
- The gains of changes #2 and #3, which assume the host stays ahead of the cards once the serial gap is gone.

## Sources

CUDA and system documentation (P)
- CUDA Runtime API, device management: `cudaSetDeviceFlags` (`cudaDeviceScheduleAuto`, `Spin`, `Yield`,
  `BlockingSync`), <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html>
- CUDA Runtime API, execution control: `cudaLaunchHostFunc`,
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html>; stream management: `cudaStreamWaitEvent`
  with events from other devices, <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html>
- CUDA Programming Guide, CUDA Graphs: conditional nodes (single device, allowed node types), device graph launch,
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
- CUDA Driver API, stream memory operations, <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html>;
  V1/V2 enablement in the 11.7 documentation, <https://docs.nvidia.com/cuda/archive/11.7.0/cuda-driver-api/group__CUDA__MEMOP.html>
- CUDA graph constraints (conditional nodes since CUDA 12.3), <https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html>
- D. Fontaine, "Dynamic Control Flow in CUDA Graphs with Conditional Nodes", NVIDIA Technical Blog (updated 2025-02-03),
  <https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes>
- "CUDA 10" (graphs spanning several GPUs), OLCF briefing, <https://www.olcf.ornl.gov/wp-content/uploads/2019/03/CUDA10_olcf_concall_march.pdf>
- NVML API, `nvmlUtilization_t` (share of time with a kernel running), <https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html>
- GCC libgomp manual: `GOMP_SPINCOUNT` (300,000 by default), `OMP_WAIT_POLICY`,
  <https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html>, <https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html>
- Intel 64 and IA-32 Architectures Optimization Reference Manual, "Pause Latency in Skylake Microarchitecture" (as quoted
  on LKML, <https://lkml.iu.edu/hypermail/linux/kernel/1704.0/04504.html>)

Device-side loops and overlapped scheduling (P)
- D. Galvez, V. Bataev, H. Xu, T. Kaldewey, "Speed of Light Exact Greedy Decoding for RNN-T Speech Recognition Models on
  GPU", Interspeech 2024, [arXiv 2406.03791](https://arxiv.org/abs/2406.03791); NeMo
  [PR #8191](https://github.com/NVIDIA/NeMo/pull/8191); NeMo's conditional-node support check (`cuda_python_utils.py`)
- LMSYS, "SGLang v0.4" (zero-overhead batch scheduler, future tokens), <https://www.lmsys.org/blog/2024-12-04-sglang-v0-4>
- vLLM, "Model Runner V2", <https://vllm.ai/blog/2026-03-24-mrv2>
- TensorRT-LLM, overlap scheduler, <https://nvidia.github.io/TensorRT-LLM/1.2.0/torch/features/overlap_scheduler.html>

llama.cpp pull requests and issues (P): [#11867](https://github.com/ggml-org/llama.cpp/pull/11867),
[#12706](https://github.com/ggml-org/llama.cpp/pull/12706) / [issue #12724](https://github.com/ggml-org/llama.cpp/issues/12724),
[#16585](https://github.com/ggml-org/llama.cpp/pull/16585), [#17004](https://github.com/ggml-org/llama.cpp/pull/17004),
[#25919](https://github.com/ggml-org/llama.cpp/pull/25919), [#26487](https://github.com/ggml-org/llama.cpp/pull/26487),
[#28058](https://github.com/ggml-org/llama.cpp/pull/28058), [#29576](https://github.com/ggml-org/llama.cpp/pull/29576),
[#29796](https://github.com/ggml-org/llama.cpp/pull/29796), [#29963](https://github.com/ggml-org/llama.cpp/pull/29963),
[#30196](https://github.com/ggml-org/llama.cpp/pull/30196), [#30223](https://github.com/ggml-org/llama.cpp/pull/30223),
[#30685](https://github.com/ggml-org/llama.cpp/pull/30685), [#30881](https://github.com/ggml-org/llama.cpp/pull/30881),
[#30918](https://github.com/ggml-org/llama.cpp/pull/30918), [#30986](https://github.com/ggml-org/llama.cpp/pull/30986),
[#31004](https://github.com/ggml-org/llama.cpp/pull/31004); issues [#21042](https://github.com/ggml-org/llama.cpp/issues/21042),
[#25121](https://github.com/ggml-org/llama.cpp/issues/25121)

The fork (P)
- Branch `pascal-p106-v2` at `032b130f4`: `ggml/src/ggml-cuda/ggml-cuda.cu` (device flags `:404-411`, synchronous
  `set_tensor` `:846-851`, peer copy and events `:2649-2704`, synchronize `:2706-2711`, uid fast path `:2769-2775`, graph
  launch `:4929`, host profiler `:4980-5008`, warm-up and capture `:5047-5090`), `top-k.cu` (`:1-60`, `:240-340`);
  `ggml/src/ggml-backend.cpp` (`copy_input` `:1823-1863`, `compute_splits` `:1865-1945`, `synchronize` `:2133`);
  `ggml/src/ggml-cpu/ggml-cpu.c` (poll loop `:3219-3230`, OpenMP path `:3412-3453`, `KMP_BLOCKTIME` `:3919-3930`);
  `src/llama-model.cpp:1688-1690`; `src/llama-context.cpp` (pipeline-parallel conditions `:526-555`, `synchronize`
  `:887-892`, sampled token `:1168-1183`, `process_ubatch` `:1523-1591`, `graph_compute` `:2748-2775`);
  `src/llama-graph.cpp` (`:30-47`); `src/llama-sampler.cpp` (`:1483-1519`); `src/llama-kv-cache.cpp` (`:1285-1299`);
  `common/common.h` (`:77`, `:635`), `common/common.cpp` (`:291-300`, `:1760-1823`), `common/chat.cpp` (`:1753-1770`);
  `tools/server/server-context.cpp` (`:702-720`, `:2088-2130`, `:2299-2342`, `:2354-2410`, `:3431`, `:3700-3725`,
  `:4740-4780`), `server-queue.cpp` (`:278-330`, `:551-575`), `server-task.cpp` (`:159-170`, `:991-996`),
  `server-http.cpp` (`:150-153`), `server-common.cpp` (`:1022`); `tools/llama-bench/llama-bench.cpp` (`:2198-2222`)
- Branch `pascal-p106-v2-pipespec` at `a217f42c8`: `src/llama-context.cpp` (`pipe_submit` `:1747-1890`),
  `ggml/src/ggml-backend.cpp` (`caller_sync` in `copy_input` `:1835-1864`), `examples/pipespec/pipespec.cpp` (`:40-42`),
  `include/llama.h:1509`

This project
- [Quant registry](../../../quants/experiments.md) (Q38, Q49, Q55, Q58, Q67, Q68, Q81, Q84, Q87, Q104, Q110, Q111, Q116),
  [experiments tracker](../../experiments.md) (E2.2), [hardware spec](../../mining_rig.md),
  [Flint benchmark](../../benchmarks/2026-10-22-flint.md), [speed-ups](../../benchmarks/2026-10-22-speedups.md),
  [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md), [Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md),
  [small-MoE overhead](../../benchmarks/2026-10-22-small-moe-overhead.md), `scripts/cuda_profile_summary.py`,
  `scripts/server_bench.sh`, Q38's host-profiled trace of S4-L0 on six cards (summarized in section 1.2), and the notes
  linked in the scope
