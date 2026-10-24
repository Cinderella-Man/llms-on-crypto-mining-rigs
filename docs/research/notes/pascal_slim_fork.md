# A slim Pascal-only fork: what generic llama.cpp code costs a P106 token, and what to remove (2026-10-24)

Scope: the fork will not be merged upstream, so it may be Pascal-specific (Q144, Q146). This note audits the fork's code
for everything that costs time per generated token on compute 6.1 and could be removed, specialised or hard-wired:
runtime branches for other architectures, feature checks, generality Flint on nine P106-090 cards doesn't need, and
choices tuned for newer GPUs that slow Pascal down. It follows one token end to end (`llama_decode`, `llama_context`, the
graph, `ggml_backend_sched`, the CUDA backend, sampling, llama-server), then the device side, CUDA graphs, the scheduler,
the build, and the server. Each item gets a cost per token, whether it runs every token, the Pascal-only change and its
risk. It builds on [host in the decode loop](host_decode_loop.md), [the Pascal kernel design](pascal_mmvq_integration.md),
[shared-expert launches](shared_expert_launches.md), [fork scope](fork_scope.md), [prefill](prefill_pascal.md),
[breakthrough candidates](breakthrough_candidates.md), [v2 synthesis](v2_synthesis.md) and
[attention at long context](pascal_attention_long_context.md).

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (code, NVIDIA documents, measurements in this repository) and secondary.
- Code citations are `branch:path:line`. `pers` = `pascal-p106-v2-persistent` at `ad3e84f27` (the combined build, the
  persistent MoE kernel on by default on compute 6.1, the compute-6.1 gating of Q146); `pipe2` =
  `pascal-p106-v2-pipespec2` (`74f7c7182`); `mask2` = `pascal-p106-v2-gpumask2` (`6f943f792`). `ggml-cuda/` stands for
  `ggml/src/ggml-cuda/`. The upstream base is `d77256999`.
- "Serial" host work lies between the last card's sample and card 1's first kernel; "overlapped" host work runs while a
  card computes. "Kernel" is one launch; "launch group" is what the fork's event profiler times as one unit.
- Nothing ran on the rig or the quant machine for this note, and nothing was compiled. On a laptop: the fork's code was
  read, two existing rig recordings were re-read with a few lines of Python (Q38's host-profiled S4-L0 trace on six
  cards; the per-token traces of Q120's `bench/pipespec/rig_hostloop.sh` run, K2h-r16 on nine cards; raw logs stay
  outside the repository), and `objdump` listed the sections of a built v2 tarball.

Starting point, measured:
- **A token on nine cards** is ~25–29 ms (K2h-r16 39.3 tok/s, 25.4 ms, in llama-bench; UD-Q4_K_XL served
  31.4–34.0 tok/s, Q118, Q55). The persistent MoE kernel adds +5.6 % on nine cards (K2h-r16) and +6.0 % (UD-Q4_K_XL)
  on the rig (fork commit `ad3e84f27`).
- **Host time inside a token** (Q38's host profile, S4-L0, six cards, Pascal kernels): 0.11–0.18 ms (median 0.15) inside
  each card's `graph_compute`, 0.13–0.24 ms (median 0.18) between one card's exit and the next card's entry (the
  scheduler's input copies), all six cards launched within ~1.8 ms against a 26.9 ms profiled GPU chain.
- **Host time removed without single-stream gain** (Q120's rig run, nine cards): packed asynchronous uploads cut the
  executor's host submit time from 2.91 to 1.65 ms per token for one conversation at an unchanged rate (38.5 against
  37.8 tok/s, one run each) and gave +21 % at eight conversations.
- **The input layer on card 1** (`LLAMA_INPUT_ON_GPU=1`, same run): plain `llama_decode` 33.69 → 34.18 tok/s, −0.43 ms
  per token (one run; the two configurations without it read 33.67 and 33.69).
- **An empty CUDA-graph node costs 1.85 µs on one P106** ([bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md)).
- **NVIDIA:** a repeat `cudaGraphLaunch` costs ~2 µs + 200 ns per node before Ampere, where launch "still takes O(n)
  time" (Xeon Silver 4208 at 2.1 GHz, RTX 3060, CUDA 11.8 and 12.6); Ampere and newer launch straight-line graphs in
  near-constant time.

## Verdict

- **Host-side generality is not where a P106 token's time goes, because CUDA graphs already took it out of the token.**
  With a graph replayed, the CUDA backend runs none of its per-node code: the fusion matchers, `ggml_cuda_info()` and
  compute-capability lookups, `GGML_CUDA_CC_*` branches, type switches, `ggml_cuda_should_use_*` choices, pool
  allocations, occupancy queries and PDL checks execute only while a graph is captured (`pers:ggml-cuda/ggml-cuda.cu:4972`,
  `:5336-5375`): at warm-up and once per 256-cell step of the KV length. Per token and card what remains is a ~350-node
  compatibility loop, four hash-map lookups, a uid compare and `cudaGraphLaunch`; in the scheduler, one event record, the
  hop and 5–8 synchronous input copies per card. That is ~250 driver calls per token on nine cards, and ~75–90 % of the
  host time overlaps the cards' work (Q38, Q49, Q120).
- **Only the serial gap counts for one conversation, and it is small once the embedding lookup leaves the CPU.** Plain
  decoding with the input layer on card 1 runs as fast as the device-fed executor's steady state (28.3 ms per token in
  both, est. from Q120's traces with their prompt time removed), so what is left between a sample and card 1's next
  kernel is est. ≤ 0.3 ms.
- **Realistic saving per token, single stream, nine cards, short context:**
  - (a) build flags: **~0** (≤ 0.05 ms). Release is already `-O3`, sm_61-only device code and static ggml; the rest
    changes binary size, build time and load time.
  - (b) small specialisations: **0.2–0.4 ms of new code** (one stream synchronize per token instead of 40, a node order
    that lets an existing upstream fusion fire, capture without an eager warm-up token, the server's thread handshake,
    the one-token bookkeeping), plus **0.45–0.65 ms from three pieces already built on other branches** (the input layer
    on card 1, measured 0.43 ms; the decode mask on the card, Q116; packed uploads, Q120). For anyone who runs without
    `LLAMA_NO_MOE_DOWN_RESCALE=tg`, a Pascal default for it is worth 2.6 ms (Q76) and switches the persistent kernel on.
  - (c) larger rewrites: **1.2–2.6 ms, all of it on the device**: a Pascal decode attention kernel (Q60; est. 0.5–1.3 ms
    at short context, where TILE attention costs ~35–38 µs per layer), fewer attention-block launches (QKV in one launch,
    one epilogue for norms, RoPE and both cache writes; est. 0.6–1.2 ms), and a two-kernel sampler (0.05–0.1 ms). A
    Pascal-only scheduler or fixed decode graph is worth ≤ 0.3 ms for one conversation; its value is in the executor.
  - **Host-side slimming in total is under 1 ms: ~0.65–1.05 ms including the built pieces (2.5–4 % of a token), 0.2–0.4 ms
    of it new code (~1 %).** The fixed cost that remains is ~600–670 kernels per token and the TILE attention kernel.
- **The largest single items found:**
  1. the rescale default (B1): 2.6 ms for default command lines, 0 for the project's own;
  2. the input layer on card 1 (B2): 0.43 ms measured, built, not on `pers`;
  3. TILE attention's fixed cost (C1): ~1.8 ms per token at 256 cells, est. 0.5–1.3 ms to gain;
  4. attention-block launches (C2): ~245 removable kernels;
  5. 39 redundant `cudaStreamSynchronize` calls per served token on the serial path (B3);
  6. Flint's `build_attn` node order, which defeats upstream's norm+RoPE+cache-write fusion on 35 layers (B6);
  7. a rebuild and eager evaluation at every 256-cell KV step, 18 slots × +9.7 ms in the executor (B9, C4).
- **Several timing effects are larger than every host item here.** The same binary's plain runs vary 31–38 tok/s with
  prompt and clock state (Q120's references), and warm against cold runs differ by ~5 ms per token (v2 synthesis, Q141).
  Items below ~0.3 ms (1 %) cannot be timed in llama-bench; they are verified by call and launch counts and shipped in
  bundles.
- **First change:** cherry-pick `LLAMA_INPUT_ON_GPU` (`pipe2:src/llama-model.cpp:1699-1705`) onto `pers` and make it the
  default on compute 6.1. It is measured (−0.43 ms), takes the CPU backend, its OpenMP region, an 8 KB synchronous copy
  and the memory-mapped embedding table's page faults out of every token, and is bit-identical for this model (Q121's
  check). Then B3 + B6 + B9 + B10 + B11 as one "decode hygiene" commit with an off switch.

## Ranked removal table

Single stream, nine cards, ≤ 8k of context, against `pers` with `LLAMA_NO_MOE_DOWN_RESCALE=tg` unless the row says
otherwise. "Order" is the suggested build order. Verification for every row: logits byte-identical
(`llama-perplexity -ub 1 --kl-divergence-base` and greedy text, as in Q117), then timing in three alternating tg-only
rounds after a warm-up (`llama-bench -p 0 -n 256 -r 5`), or a call or launch count where the effect is below noise.

| # | Item | Est. µs per token | Effort | Risk | How to verify | Order |
|---|---|---|---|---|---|---|
| **(a)** | **Build flags** (total ≈ 0–50 µs) | | | | | |
| A1 | `CMAKE_CUDA_ARCHITECTURES=61-real` for rig tarballs (test builds keep the compute_61 PTX for laptop and 3090 checks) | 0; fatbin 17 MB → est. 9–11 MB | minutes | none | `cuobjdump --list-ptx` empty; tg unchanged | 6 |
| A2 | CUDA kernels instantiated only for the types rig files use (F16, BF16, F32, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S) | 0; build time and size est. −30–50 % | 1–2 days | low: an untrimmed type must fail at load | load every release file; test-backend-ops for the kept types | 7 |
| A3 | `GGML_LTO=ON`, `GGML_STATIC=ON` (static cudart, no PLT) | 0–20 serial | hours | low | host instruction count per token with `perf stat` | 7 |
| A4 | `GGML_OPENMP=OFF` or `-t 1` | 0–50 while a CPU split exists; 0 after B2 | minutes | none | Q118's `-t 1/2` rows | — |
| A5 | `GGML_CUDA_COMPRESSION_MODE=none`, `FORCE_MMQ`, `FORCE_CUBLAS`, `NO_PEER_COPY`, `SCHED_MAX_COPIES` | 0 for decode (`FORCE_CUBLAS` and `NO_PEER_COPY` harmful) | — | — | — | — |
| **(b)** | **Small specialisations** (new code 200–400; built pieces 450–650) | | | | | |
| B1 | Rescale off at one-token steps by default on compute 6.1 when every down-projection type reads no FP16 sum | 2,580 (Q76) and the persistent kernel on, for default command lines only; 0 with `=tg` set | ½–1 day | low: a type check at graph build | KLD as Q76; the persistent kernel's first-use log line | 1 |
| B2 | Input layer on card 1 by default (`pipe2`'s `LLAMA_INPUT_ON_GPU`) | **430 measured** (one run) | hours, plus 0.25–0.33 GB on card 1 in the split | low (bit-identical in Q121) | two alternating plain-decode rounds; major faults per token (Q118) | 1 |
| B3 | One stream synchronize per token: skip when nothing is in flight, sync only the output backend | 40–120 serial | hours | low | uprobe count of `cudaStreamSynchronize` per token: 40 → 1–10 | 2 |
| B4 | Decode mask on the card (Q116, `mask2`) | 50–100 serial at ≤ 8k, ~300+ at 32k | built; merge | medium (merge with `pers`) | Q116's checks | 3 |
| B5 | One packed asynchronous upload per card for one-token `llama_decode` steps (Q120's mode 2 outside the executor, only for a reused graph after a synchronize) | 80–150 serial, ~1,000 host overlapped | 1–2 days | medium: input buffer lifetime | Q120's identity runs; uprobe count of synchronous copies | 3 |
| B6 | `cpy_k` right after `k_cur` in Flint's attention so upstream's `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS` fusion fires | 35 kernels: 90–140 GPU | hours | low: node order only | profile shows 35 fused groups per token | 2 |
| B7 | llama-server at `-np 1`: no worker-thread handshake per decode | 10–50 serial; more when the second core is busy | 1 day | medium: server threading | host trace around `yield_to_queue` | 5 |
| B8 | One-token fast path in `llama_decode` bookkeeping (batch allocator, KV dry run) | 20–40 serial | 2 days | medium: memory-module invariants | KV state hash after 1,000 tokens equal; uprobe-timed `llama_decode` | 5 |
| B9 | Capture at once when only KV-length-shaped tensors changed (no eager warm-up evaluation) | 20–40 average (plain); executor: part of 300–700 | hours | low | per-token trace across a 256-cell step | 2 |
| B10 | No `GET_ROWS` of `out_ids` on one-token steps | 5–8 GPU, one host copy | hours | low | profile: 2 groups fewer | 2 |
| B11 | Hoists: the CUDA-graph compatibility loop behind the uid fast path; the CPU threadpool lookup by name in `graph_compute` | ~5 serial, ~25 overlapped | trivial | none | — | 2 |
| **(c)** | **Larger rewrites** (1,200–2,600 on the device) | | | | | |
| C1 | Pascal decode attention: one GQA-aware launch per layer instead of TILE + combine (Q59/Q60), masked-tile skip | 500–1,300 at short context; the 0.7–0.8 ms per 256 cells slope below the window to est. ~0.2 | 3–5 days (planned) | medium | test-backend-ops `FLASH_ATTN_EXT`; profile per call | 4 |
| C2 | Attention-block launches: QKV in one Pascal launch (−98), one epilogue for Q/K norm, RoPE and both cache writes (−147, B6 included), attention output written as Q8_1 (−49) | 600–1,200 | 1–2 weeks | medium | per-stage bit identity as in Q88; launch count | 5 |
| C3 | GPU sampler in two kernels (radix top-k's 11 launches plus the chain) | 50–100 | 2–3 days | low–medium | seeded draws identical; profile | 6 |
| C4 | A fixed one-token graph per card (device-side KV length, no rebuild or recapture at KV steps) and device-fed tokens in llama-server | ≤ 300 single stream; executor 300–700 average | 2–3 weeks | medium–high | traces without step stretches | 8 |
| C5 | One persistent kernel per layer (attention + MoE), breakthrough candidate 2's next step | est. 1,000–2,000 beyond C2 | weeks | medium | as Q88 | later |

## 1. The per-token host path

### 1.1 What runs for one token

llama-server with backend sampling (`-bs`), one slot, nine cards, graph reused and replayed; code on `pers`. Costs on
the Celeron are estimates unless marked; ~1–3 µs per driver call on an idle stream or event (est.: ~0.5–1 µs on a 4+ GHz
desktop core, scaled to 1.8 GHz without turbo).

| # | Step | Code | Calls per token | Cost (est.) | Serial? |
|---|---|---|---|---|---|
| 1 | Wait for the token, then three more synchronize sweeps over all ten backends: after decode, in `common_sampler_sample`, in `llama_get_sampled_token_ith` | `server-context.cpp:4584-4589`; `common/sampling.cpp:607`, `:624`; `src/llama-context.cpp:4720-4723`, `:916-922`; `ggml-backend.cpp:2143-2154` | 30 `cudaStreamSynchronize` (one blocks) | 30–90 µs beyond the wait | serial |
| 2 | Server worker handshake: `yield_to_queue` wakes the queue worker before the decode and waits for it to go idle after | `server-queue.cpp:222-260`, `:163-206` | 2 thread wake-ups | 10–50 µs; a scheduler slice if the other core is busy | serial |
| 3 | Sample readback, accept, detokenize, stop strings, partial result | `server-context.cpp:4768-4797`, `:2098-2224` | — | 20–100 µs (Q118: server within 0.3–2.3 ms of llama-bench) | serial |
| 4 | `llama_decode` front: output count vector of `n_seq_max` entries, batch allocator (`std::set` and `std::map` per token, 256-sequence arrays cleared), KV `prepare` as a dry run with copy and restore for both caches, `output_reserve` | `src/llama-context.cpp:2257`, `:2282`, `:2328`, `:2371`; `src/llama-batch.cpp:257`, `:285`, `:832-862`; `src/llama-kv-cache.cpp:786-850`; `src/llama-kv-cache-iswa.cpp:185-190` | 1 | 20–50 µs | serial |
| 5 | `process_ubatch`: KV apply, `can_reuse` over ~10 inputs, a fourth synchronize sweep (reused graph under pipeline parallelism) | `src/llama-context.cpp:1587-1597`; `src/llama-graph.cpp:1637` | 10 `cudaStreamSynchronize` | 10–30 µs | serial |
| 6 | `set_inputs`: positions, four index tensors, two f16 masks filled cell by cell (two `unordered_map`s and a vector per mask) | `src/llama-context.cpp:1633`; `src/llama-kv-cache.cpp:1779-1821`, `:1619` | 1 | 10–20 µs at ≤ 1k cells, 50–100 µs at 8k | serial |
| 7 | `graph_compute`: the CPU threadpool function looked up by name, thread counts set | `src/llama-context.cpp:3134-3150` (`:3142`) | 1 | ~1 µs | serial |
| 8 | CPU split: one embedding row by `GET_ROWS` from the memory-mapped table, as an OpenMP region | `src/llama-model.cpp:1689-1691`; `ggml-cpu/ggml-cpu.cpp:170-191`; `ggml-cpu/ggml-cpu.c:3437` | 1 | 5–50 µs; a page fault 0.1–20 ms (0.23 faults per token on new text, Q118) | serial |
| 9 | Card 1's inputs: the 8 KB row, positions, four index tensors, two masks, each `cudaMemcpyAsync` + `cudaStreamSynchronize` | `ggml-backend.cpp:1827-1873` (`:1838-1842`); `ggml-cuda/ggml-cuda.cu:853-859` | ~8 | 150–220 µs (~20–28 µs each, Q38/Q120) | serial |
| 10 | Card 1's `graph_compute`: compatibility loop, map lookups, uid check, `cudaGraphLaunch` of ~70 kernel nodes | `ggml-cuda.cu:5313-5380`, `:2737-2768`, `:2774-2789`, `:5211`; `common.cuh:1475-1497` | 1 | 3–6 µs + 25–35 µs launch (O(n) before Ampere) | serial |
| 11 | Cards 2–9: event wait, hop (`cudaMemcpyPeerAsync` + event record + stream wait), 5–7 synchronous input copies, `graph_compute`, event record | `ggml-backend.cpp:1875-1957`; `ggml-cuda.cu:2659-2716` | ×8 | ~0.25–0.35 ms per card (Q38: 0.13–0.24 ms between cards, 0.11–0.18 inside) | overlapped |
| 12 | Output copies queued: the sampled id (4 B) and up to three more rows | `src/llama-context.cpp:2560-2563`, `:2170-2200` | 1–4 | ~3–10 µs | overlapped |

**Totals (est.).** Serial host work today: 0.3–0.8 ms at ≤ 8k, of which steps 8–9 (the CPU split and card 1's copies) are
0.2–0.5 ms and the generic bookkeeping (steps 1, 2, 4–7, 10) 0.1–0.3 ms. Overlapped host work: ~2–2.8 ms per token on
nine cards. Q120's rig run bounds the serial part from the other side: with the input layer on card 1, plain decoding
equals the device-fed executor's steady state, so the serial gap left is ≤ ~0.3 ms (est.: plain 34.18 tok/s is 28.4 ms
per token once its 22-token prompt is removed, against the fed trace's 28.3 ms between steps; one run each).

### 1.2 What does not run per token any more

These are the checks the audit was asked to find. With CUDA graphs on compute 6.1 (fork commit `fedac87b2`, PR #27721;
exactly 6.1 since `9d5f1530e`, `pers:ggml-cuda/ggml-cuda.cu:5301`), each runs only during warm-up and capture:
- the per-node dispatch, `ggml_cuda_try_fuse` with its dozens of pattern matchers per node, the persistent-MoE matcher and
  `ggml_cuda_compute_forward` (`ggml-cuda.cu:4972` skips the whole loop when a captured graph is replayed;
  `:4000`, `:3687`);
- `ggml_cuda_info()` and `GGML_CUDA_CC_*` lookups in the op code, `ggml_is_contiguous` and type switches,
  `ggml_cuda_should_use_mmvq/mmq`, the Pascal kernels' shape choice (`mmvqp_choose`), the MMVQ table probe of Q146;
- pool allocations (their addresses are baked into the graph), `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and the
  parallel-block search of every flash-attention call (`fattn-common.cuh:1137`, `:1200`), the PDL capability cache
  behind a mutex (`common.cuh:1683-1736`, `:1748-1763`).

A Pascal-only rewrite of these would change nothing per token. They matter for prompt micro-batches, which run eagerly,
but there the host's ~10–20 ms per micro-batch is ~1 % of the micro-batch ([prefill note](prefill_pascal.md) §1.1).

### 1.3 Items

- **B3, synchronize once.** A served token calls `ggml_backend_sched_synchronize` four times (step 1 and step 5), each
  sweeping the nine CUDA streams and the CPU backend, so 39 of the 40 `cudaStreamSynchronize` calls find an idle stream.
  `llama_context` already tracks `gf_in_flight` (`src/llama-context.cpp:922`, `:1587`) but `synchronize()` does not
  consult it. Change: return early when nothing was submitted since the last synchronize (every asynchronous submit
  sets the flag), and in decode synchronize only the backend of the last split, whose completion implies every earlier
  card's through the hop events. Est. 40–120 µs serial. Prior art: llama.cpp PR #29796 (draft) skips redundant
  synchronizes on one device, +16.9 % on Windows without speculation; maintainers want it in the scheduler.
- **B7, the server handshake.** `yield_to_queue` (`server-queue.cpp:222-260`) wakes the queue worker so that HTTP tasks
  are handled during a decode, then waits for that worker to notice the yield has ended and go idle
  (`condition_tasks.wait`, `:249-251`): two futex wake-ups on the serial path, est. 5–20 µs each on an idle core, more if
  the second core runs the OpenMP spinner (until B2), the HTTP thread's streaming re-parse or the desktop. At `-np 1` the
  main loop can poll the task queue between tokens instead.
- **B8, one-token bookkeeping.** `llama_batch_allocr::init` keeps 256-sequence structures (`LLAMA_MAX_SEQ`,
  `src/llama-cparams.h:8`) and allocates a `std::set` entry and a `std::map` node per token (`src/llama-batch.cpp:257`,
  `:285`); `prepare` places the ubatch in both caches, copies the touched cells, restores them, and `apply` places it
  again (`src/llama-kv-cache.cpp:786-850`, `:1135`); each mask fill builds two `unordered_map`s (`:1619`). A fast path for
  one token of one sequence whose cell is free: est. 20–40 µs.
- **B11, hoists.** `ggml_cuda_graph_check_compability` walks all ~350 nodes of the card's graph every token
  (`ggml-cuda.cu:2737-2768`, called at `:5336`), although a matching uid (`:2781`) proves the graph unchanged; and
  `graph_compute` looks up `ggml_backend_cpu_set_threadpool` by name every call (`src/llama-context.cpp:3142`).
- **Kept as they are:** `cuda_graph()`'s `ggml_time_us` and hash lookup (`common.cuh:1475-1497`, < 1 µs per card),
  `cudaGetDevice`/`cudaSetDevice` in `ggml_cuda_set_device` (`ggml-cuda.cu:126-138`), the scheduler's two input loops per
  split (`ggml-backend.cpp:1898`, `:1903`). Together < 10 µs per token, overlapped.

## 2. Device-side generality

### 2.1 Kernels per token on `pers`

The v1 event profile of S4-L0 has 1,020 launch groups per token; Q50 removed 96, Q100 43 groups (287 kernels), Q88 329
groups (423 kernels), which leaves ~550 groups, ~600–670 kernels per token (est.). One MoE layer at one token:

| Kernel | Count | v1 profile, µs per call (5–8 µs of it event overhead) | Code |
|---|---|---|---|
| Attention norm with the residual add, writing Q8_1 (Q100) | 1 | 13.2 (norm alone) | `comb:ggml-cuda/norm.cu:909-926` |
| Q, K, V products (Pascal kernels, reading the norm's Q8_1 copy) | 3 | 54.7 / 22.4 / 20.3 | `mmvq-pascal.cu` |
| Q norm (+ RoPE), K norm (+ RoPE) | 2 | 14.2 with RoPE (70 per token), 13.2 without | `ggml-cuda.cu:4762-4770` |
| `SET_ROWS` of K and of V | 2 | 9.5 | `set-rows.cu` |
| Flash attention: TILE `<128,128,1,8>` and the combine kernel | 2 | 43.4 for both | `fattn.cu:704-718`; `fattn-common.cuh:924`, `:1221` |
| Attention output: quantization, product with the residual add | 2 | 81.6 / 58.3 | `mmvq.cu` |
| The MoE layer (Q88) | 1 | — (171–174 µs on the rig A/B) | `moe-persistent.cu` |
| **Total** | **13** | | |

At ~1.85 µs per graph node plus a small kernel's ramp and tail on 5 SMs, each removed kernel saves ~2.5–4 µs (est.).

### 2.2 Attention on compute 6.1

- **Selection.** Without tensor cores, f16 K/V and Flint's GQA ratio of 8 make `ggml_cuda_get_best_fattn_kernel` return
  TILE for one token (`fattn.cu:560`, `:704-718`). Upstream's own comment: "On NVIDIA (Pascal and older) the GQA
  optimizations seem to be detrimental in some cases" (`fattn-tile.cuh:1249`). TILE stores K/V tiles as FP32 in shared
  memory, because `FAST_FP16_AVAILABLE` is undefined for 6.1 (`common.cuh:274`).
- **Fixed cost.** At 256 cells (tg32, S4-L0) a layer's attention takes 43.4 µs in the event profile, ~35–38 µs without
  the event overhead, for 0.5 MB of K/V: ~1.8 ms per token, ~15× what the bytes need at 130 GB/s. With 4 KV heads the grid
  is 4 × `parallel_blocks` blocks on 5 SMs, followed by the combine launch.
- **Slope.** Q120's executor traces show the steady token time rising 0.67–0.84 ms per 256 cells of context (27.8 → 28.6
  → 29.3 ms) below ~768 tokens, where all 49 layers' KV lengths grow (the sliding-window ring is not full yet): ~15 µs per
  layer per 256 cells, ~35 GB/s (est.). The parallel-block search (`fattn-common.cuh:1192-1224`) picks a 2-wave grid
  once 512 cells allow 20 blocks, which fits the step (est., not profiled). Deep contexts reach ~130 GB/s (Q54's
  slope), so the loss is fixed cost and shape, not bandwidth.
- **C1** is the planned Pascal decode kernel ([attention note](pascal_attention_long_context.md) §6, Q59/Q60): one
  launch, K/V read once per KV head, split-KV merged in the last block, masked tiles skipped. That note puts its
  short-context gain at ~2 %; the profile above allows up to ~5 % (est. 0.5–1.3 ms). Writing the output as Q8_1 for the
  attention output product removes one more kernel per layer (part of C2).

### 2.3 Fusions that don't fire, and launches to merge

- **B6, node order.** Upstream fuses `RMS_NORM+MUL+ROPE+VIEW+SET_ROWS` into one kernel (#26767,
  `ggml-cuda.cu:4762`), but only when the five nodes are adjacent. Flint's `build_attn` expands `q_cur`, `k_cur` and
  `v_cur` first and the cache writes afterwards (`src/llama-graph.cpp:3691-3723`, "so that they are not reordered, …
  the number of splits in the graph is reduced"), so K's RoPE is followed by V's product and every token runs 98 separate
  `SET_ROWS` (the v1 profile). On a full GPU layer split, card boundaries fall between layers, so expanding `cpy_k`
  right after `k_cur` changes no split and fuses K's cache write on the 35 RoPE layers: −35 kernels, est. 90–140 µs.
- **C2, the attention block.**
  - Q, K and V read the same Q8_1 row: one Pascal launch with three weight slices, each with its own type and row count
    (as shared-expert design A, [shared-expert note](shared_expert_launches.md)), also fills the 5 SMs that K's and V's
    512 rows leave half idle: −98 kernels.
  - One epilogue kernel for the Q norm and RoPE, the K norm, RoPE and cache write, and V's cache write (14 layers without
    RoPE included): −147 kernels, B6's 35 among them.
  - The attention output written as Q8_1 by the attention kernel: −49.
  - Together est. 0.6–1.2 ms. Flint's shapes (head 128, 32 query and 4 KV heads, sinks) may be fixed at compile time,
    as the persistent MoE kernel fixes its own.
- **B10.** The last layer gathers `out_ids` rows for `cur` and `inpSA` (`src/models/flint.cpp:127`) although a one-token
  step has one output; upstream keeps the gather so that prompt graphs keep their topology
  (`src/llama-graph.cpp:3000-3007`, PR #14275), which a one-token graph doesn't need: two kernels and one input copy.

### 2.4 Choices already fixed for sm_61

- Device code is sm_61 only (`scripts/build_llamacpp_docker.sh`: `CMAKE_CUDA_ARCHITECTURES=61`), so `__CUDA_ARCH__`
  guards already compile out the tensor-core paths (MMA flash attention and MMF become stubs), PDL device calls
  (`common.cuh:134-143`) and `__restrict__` exceptions; the MMVQ device table is chosen at compile time
  (`MMVQ_PARAMETERS_PASCAL_DP4A`, Q146).
- Host-side choices (`fast_fp16_available(cc)`, `turing_mma_available(cc)`, the MMVQ host table) run at capture only
  (section 1.2).
- **What is left is register pressure in Pascal code itself:** the persistent kernel's eight K-quant variants spill
  16–76 bytes ([kernel design](pascal_mmvq_integration.md) §17), and UD-Q4_K_XL's five layers with Q4_K, IQ3_S or Q6_K
  run llama.cpp's generic `mul_mat_vec_q`. That is Pascal kernel coverage, not removable generality.

### 2.5 Sampling on the card

With `-bs` and top-k 20, CUDA 12.9's CCCL is older than the 3.4.3 that `DeviceTopK` needs (`top-k.cu:29`), so one row
takes the radix path: an init, four histogram and select pairs over the 150,144 logits, a reset and a gather, 11 launches
(`top-k.cu:212-251`), then the top-p, min-p, temperature and draw kernels. Est. 60–120 µs; one pass with per-block top-20
lists and one merge-and-draw kernel would take est. 10–20 µs (C3). With the screened head (Q92) the top-k runs over
1,024 candidates and the gain shrinks.

## 3. CUDA graphs on 6.1

- **Per token:** the compatibility loop (B11), the uid fast path (`ggml-cuda.cu:2781`), `cudaGraphLaunch` (`:5211`).
  Before Ampere a launch costs ~2 µs + 200 ns per node on a 2.1 GHz Xeon (NVIDIA), est. 3–4 µs + 0.3–0.4 µs per node on
  the Celeron: ~0.2–0.3 ms per token over ~600 kernel nodes, card 1's 25–35 µs of it serial. Every kernel C2 removes
  saves this too.
- **At every 256-cell step of the KV length** (`src/llama-kv-cache.cpp:1300-1312`, minimum pad 256 at `:1305`) the mask's
  shape changes, so the llama graph is rebuilt (graph build, scheduler split, allocation), the CUDA backend sees changed
  node properties and runs that token **eagerly** (`ggml-cuda.cu:5340-5360`: "need at least 2 calls with no property
  change"), and the next token captures and updates the executable graph.
  - **Measured in the executor** (Q120's traces, K2h-r16, nine cards): after each step 18 consecutive tokens take
    +9.7 ms (28.3 → 38.0 ms), one per executor slot, each slot rebuilding and running eagerly once; with fed tokens
    +4.1 ms each, partly hidden. That is 175 ms (74 ms fed) per step, 0.3–0.7 ms per token on average over 512 tokens.
  - **Plain decoding and llama-server** have one graph per card, so a step costs one such token plus a capture:
    est. 10–20 ms per 256 tokens, 0.04–0.08 ms per token.
- **B9:** for one-token graphs, capture at once when only KV-length-shaped tensors changed. The eager evaluation goes;
  the rebuild stays. The rebuild's share of the +9.7 ms is unknown; a host trace of one step splits it.
- **One graph per bucket** does not help one conversation: its KV length only grows, so each bucket is visited once.
- **C4, a fixed decode graph:** the full KV length in the graph and the real length on the device, so nothing changes
  at a step. Prerequisites: C1's kernel skipping masked tiles, and Q116's mask on the card. With today's TILE kernel,
  padding the KV length further costs ~15 µs per layer per 256 cells (section 2.2), so larger buckets are a loss until C1.

## 4. The scheduler and multi-device copies

- **Per token, graph reused** (`ggml-backend.cpp:2128-2141`): no split, no allocation (`:2103-2120` runs only on a
  rebuild). `compute_splits` (`:1875-1957`) walks ten splits: CPU (the embedding row), then each card with its hop and
  5–8 user inputs. Each user input is an event synchronize plus a synchronous copy (`:1836-1842`), each hop a peer copy on
  the source stream with an event the destination waits on (`ggml-cuda.cu:2697-2709`), each split an event record
  (`ggml-backend.cpp:1950`).
- **Generic branches** that a CUDA-only, layer-split-only scheduler would drop: the host-weight copy callback, CPU and
  non-CUDA copy fallbacks, multi-backend buffer-type checks, `callback_eval`. Each is a branch or an indirect call per
  input, < 5 µs per token in all (est.).
- **What costs time is the synchronous copies:** ~55 per token, ~1.26 ms of host time on nine cards (Q120: 2.91 → 1.65 ms
  submit time with packed uploads). Off the critical path except card 1's ~8.
- **For one conversation such a scheduler is worth what B2, B4 and B5 give** (≤ 0.5 ms with them, ≤ 0.3 ms after them).
  For several conversations Q120 already showed the next limit is not host submit time (13 ms of waits per token at
  N = 8).
- **Hops.** Eight staged peer copies of 8 KB through host memory, est. 0.05–0.1 ms each on the GPU side (Q81
  unmeasured). Whether the driver needs a host thread for a staged copy without P2P is unknown; if it does, each hop
  wakes a thread on the Celeron.

## 5. Build and binary

| Setting | Today (`scripts/build_llamacpp_docker.sh`) | Per token | Load time and size |
|---|---|---|---|
| Optimisation | `CMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`) | already | — |
| CUDA architectures | `61` = sm_61 machine code + compute_61 PTX (143 + 143 entries) | 0: the driver loads the machine code and never JIT-compiles the PTX | `61-real`: fatbin 17 MB → est. 9–11 MB of a 62 MB `llama-server` |
| Quant types | all ~30 types' MMQ, MMVQ, get_rows, conversion and FA cases (`GGML_CUDA_FA_ALL_QUANTS=OFF` already) | 0: lazy module loading (default since CUDA 12.2) loads kernels on first use | trimmed list: est. −30–50 % size and build time |
| Fatbin compression | `GGML_CUDA_COMPRESSION_MODE=size` (default) | 0 | decompression per module at first use (ms) |
| LTO, `-fno-plt`, static cudart | off; ggml/llama static, cudart and cuBLAS dynamic | est. 0–20 µs serial: ~250 runtime calls per token through the PLT, and some host inlining | static cudart ~+1 MB |
| CPU backend | SSE4.2, no AVX/FMA/F16C (the Celeron has none), OpenMP on, CPU repack on | the CPU computes one row per token; OpenMP's fork and libgomp's spin cost the second core (est. ~20 ms of it per token at `-t 2`) | — |
| `GGML_CUDA_FORCE_MMQ` / `FORCE_CUBLAS` | off | `FORCE_MMQ` changes nothing on 6.1; `FORCE_CUBLAS` slows MoE prompts (Q28: 103 against 170 tok/s) | — |
| `GGML_CUDA_NO_PEER_COPY` | off | would turn every hop into two synchronous host copies | — |
| `GGML_SCHED_MAX_COPIES` | 4 | 0 for decode (a reused graph keeps one copy) | prompt memory ([prefill note](prefill_pascal.md) §3.2) |
| cuBLAS | linked (prompt-sized F16/F32 products) | 0 for decode | 0.85 GB of shared libraries (`libcublasLt` 749 MB), paged in only where used |

The binary's host code (`.text` 25 MB, `.rodata` 8 MB) is touched in a few hundred KB per token; LTO and PGO might trim
the serial host path by ~10 % (est. 10–40 µs), below what llama-bench can resolve.

## 6. The server and sampling path

- **Q118 measured the server within 0.3–2.3 ms of llama-bench minus its logits copy**, and first-pass page faults on the
  memory-mapped embedding table explain 1–2 ms of that (v2 synthesis §1.1). B2 removes those faults, so after it the server
  path is est. 0.1–0.3 ms per token.
- **Serial and removable:** the redundant synchronize sweeps (B3, 40–120 µs) and the worker handshake (B7, 10–50 µs).
- **Serial and kept:** sampling readback and accept (`server-context.cpp:4768-4773`), detokenizing
  (`common_token_to_piece`, `:4790`), stop strings over the unsent tail (`process_token`, `:2098-2224`), est. 20–100 µs.
- **On the other core:** JSON, SSE and the streaming parser, whose re-parse of the whole reply per token grows with the
  reply ([host in the decode loop](host_decode_loop.md) #4; est. 0.1–0.15 ms at 760 tokens, ~1–1.5 ms at 8k). It
  contends with the decode thread only while the second core is short of time.
- **Logging** compiles to a level check per call (`common/log.h:112-127`), and the timing print runs every 3 s: nothing
  to remove.

## 7. Order and verification

1. **B2 + B1:** the input layer on card 1 and the rescale default on compute 6.1. Each has an off switch; verify by
   identical logits and two alternating tg-only rounds (B2 is ≥ 1 %, so llama-bench sees it).
2. **The hygiene bundle B3, B6, B9, B10, B11** (one switch for the bundle, one per item for A/B):
   - uprobes on `libcudart.so.12` count `cudaStreamSynchronize` per token (40 → ≤ 10);
   - `GGML_CUDA_PROFILE` counts launch groups per token (−37);
   - a per-token host trace across a 256-cell step shows the eager token gone;
   - one alternating timing round for the bundle (est. 0.15–0.3 ms, ~1 %).
3. **Merge the built pieces B4 and B5** onto `pers`, with Q116's and Q120's identity checks.
4. **C1 (Q60), then C2:** kernel work with per-stage bit identity as in Q88, then rig timing.
5. **B7 and B8** only if a host trace after steps 1–3 still shows ≥ 0.1 ms serial in the server or in `llama_decode`.
6. **C3, A1, A2 and A3** at release time; **C4** with the executor's llama-server integration.

## 8. Built: B2, B3, B5 and B6 (Q150)

One commit on the unified v2 build (fork `pascal-p106-v2-next`, `2d6363023`). Each piece has an off switch, and all are
on by default where every CUDA device has compute capability 6.1: the CUDA backend reports a `COMPUTE_6_1` feature, which
llama.cpp reads.
- **B2, the input layer on card 1:** `LLAMA_INPUT_ON_GPU` defaults to on there; `0` turns it off, and elsewhere it stays
  off unless set.
- **B5, packed uploads in plain decoding** (`LLAMA_INPUT_UPLOAD`, 2 or 0): an evaluation that `process_ubatch` submits
  with nothing in flight runs the scheduler with `caller_sync` and input upload mode 2, so each split's host inputs go up
  in one asynchronous copy from pinned staging memory instead of an event wait and a synchronous copy per input.
  `caller_sync` is set before the graph is allocated (the packs are laid out then) and cleared after the submission, so
  the K-shift and training keep the waits. In mode 2 the scheduler copies an input that has no pack synchronously:
  `llama_decode`'s next micro-batch may overwrite a host input before an asynchronous copy from it has run, which is also
  why plain decoding does not offer mode 1.
- **B3, synchronize once:** `synchronize()` returns at once when nothing was submitted since the last one (every
  submission, the K-shift included, marks the context in flight), and a reused graph skips its synchronize sweep when
  nothing is in flight, so a served token sweeps the backends once instead of four times. The sweep still covers every
  backend: syncing only the last split's backend would rest on the layer split's chain of hop events.
  `LLAMA_INPUT_UPLOAD=0` restores the four sweeps.
- **B6, the node order:** `build_attn` with a sliding-window cache expands Q, V, then K, as the plain-cache `build_attn`
  does, so K's `RMS_NORM, MUL, ROPE, VIEW, SET_ROWS` are adjacent and upstream's fused kernel writes K into the cache on
  Flint's 35 RoPE layers.

Checks on two V100s through the compute-6.1 PTX (Pascal kernels and the persistent MoE kernel forced, the switches
forced on), K2h-r16 in a two-card layer split:
- the logits of 2 × 128 tokens are byte-identical to the build before, with each switch alone and with all of them, at
  1 and at 32 tokens per micro-batch; with only the node order changed, so B6's fused cache write is exact;
- `llama-pipespec` plain decoding (one `llama_decode` per token after a synchronize, so packed uploads at every token):
  160 tokens and their per-token logit hashes identical; [check_identity.sh](../../../bench/pipespec/check_identity.sh)
  with everything on: 14 pipelined runs (oracle, copy, continuation and n-gram guesses at depths 1 and 3) identical to
  plain decoding in tokens, logits, KV state and probe logits;
- greedy `llama-completion`, 2 × 128 tokens: identical.

## What this changes in the plan

- **The host is not the next single-stream lever after B2:** ≤ 0.3 ms of serial host time per token remains, against
  ~1.8 ms of TILE attention at 256 cells and ~600 kernels. Q60's attention kernel moves up: it pays at short context, not
  only at depth.
- **The executor pays a rebuild and eager evaluation per slot at every 256-cell step** (0.3–0.7 ms per token on average
  in Q120's traces). Q121's measured single-stream gain of fed tokens (1.4 ms per token) is mostly the d0 path's own
  costs: ~0.4 ms per token of these stretches (175 against 74 ms per step) and ~0.4–0.5 ms of steady-state serial work
  (medians 28.3 against 27.8 ms, 29.0 against 28.6 ms), not a gap that plain `llama_decode` has. Plain decoding with the
  input layer on card 1 already matches the fed steady state.
- **`LLAMA_INPUT_ON_GPU` and Q116's decode mask belong on the release branch**, not only in the executor and the mask
  branch.
- Suggested registry rows (not added here): B2 + B1 as one engine row; the hygiene bundle; the executor's per-slot step
  cost as a C7 item.

## Already covered

| Direction | Where it is |
|---|---|
| Per-node host dispatch out of the token: CUDA graphs on compute 6.1 (−4.9 ms per token before the fork) | fork `fedac87b2`, PR #27721; [upstream patches](upstream_pascal_patches.md) |
| Pascal-only choices gated to compute 6.1 (MMVQ table, the CUDA-graph gate): host branches at capture only | Q146, fork `9d5f1530e` |
| Shared-expert gate/up/GLU fusion: −3 kernels per MoE layer | Q50, [shared-expert note](shared_expert_launches.md) |
| The norm writes Q8_1 and absorbs the residual add: −244 quantizations, −43 adds per token | Q100 |
| The MoE down rescale off at one token (~470 graph ops per token), FP32 activation sums | Q76 |
| One launch per MoE layer: −423 kernels per token, on by default on compute 6.1 | Q88, fork `ad3e84f27` |
| Packed asynchronous uploads, the embedding lookup on card 1, device-fed tokens (executor) | Q120, Q121, [host in the decode loop](host_decode_loop.md) §6 |
| Decode mask built on each card | Q116 (`mask2`) |
| Pascal decode attention kernel, TILE's shared-memory limit | Q54, Q59, Q60, [attention at long context](pascal_attention_long_context.md) |
| Server flags (`-t 1`, `--mlock`), streaming re-parse, page faults | Q118, Q119, [host in the decode loop](host_decode_loop.md) |
| Cold clocks and PowerMizer | Q141 (no gain), [power and tuning](power_and_tuning.md) |
| MMQ, cuBLAS and graph reuse for prompts | Q108, Q109, [prefill note](prefill_pascal.md) |

## Not verified

- Every per-call host cost on the Celeron: `cudaStreamSynchronize` on an idle stream (1–3 µs), a futex wake-up (5–20
  µs), `cudaGraphLaunch`'s per-node cost (NVIDIA's figure is for a 2.1 GHz Xeon), the bookkeeping of steps 4–6.
- That plain decoding's serial gap is ≤ 0.3 ms: inferred from one plain run and one fed trace with prompt time removed
  by a ratio.
- B2's −0.43 ms: one run per configuration.
- The real time of small kernels: the event profile adds 5–8 µs per group, so the 2.5–4 µs saved per removed kernel is
  inferred, as is TILE's ~35–38 µs at 256 cells.
- That the attention slope below the window comes from TILE's parallel-block choice; that the executor's +9.7 ms per
  slot splits between rebuild and eager evaluation in any particular way.
- Q150's per-token synchronize and copy counts, its behaviour across a context shift and with partial offload, and its
  time per token on the rig.
- Whether a staged peer copy without P2P needs a driver host thread (Q81).
- C2's and C3's savings; the size and build-time effect of A1 and A2.

## Sources

NVIDIA (P)
- "Constant Time Launch for Straight-Line CUDA Graphs and Other Performance Enhancements", NVIDIA Technical Blog:
  pre-Ampere launch O(n), ~2 µs + 200 ns per node (CUDA 11.8 and 12.6, Xeon Silver 4208, RTX 3060),
  <https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements>
- CUDA Runtime API, stream management (`cudaStreamSynchronize`, `cudaStreamWaitEvent`),
  <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html>
- CUDA C++ Programming Guide, lazy loading (`CUDA_MODULE_LOADING`) and CUDA graphs,
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>

llama.cpp (P): PR [#26767](https://github.com/ggml-org/llama.cpp/pull/26767) (fused norm, RoPE and cache write),
[#27721](https://github.com/ggml-org/llama.cpp/pull/27721) (CUDA graphs on Pascal),
[#14275](https://github.com/ggml-org/llama.cpp/pull/14275) (why `out_ids` stays in the graph),
[#29796](https://github.com/ggml-org/llama.cpp/pull/29796) (redundant synchronizes),
[#29963](https://github.com/ggml-org/llama.cpp/pull/29963) and [#30685](https://github.com/ggml-org/llama.cpp/pull/30685)
(asynchronous graph inputs), [#31047](https://github.com/ggml-org/llama.cpp/pull/31047) (MMQ MoE tiles).

The fork (P), `pers` at `ad3e84f27` unless marked
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_set_device` `:126-138`, synchronous `set_tensor` `:853-859`, peer copy and
  events `:2659-2716`, compatibility loop `:2737-2768`, uid fast path `:2774-2789`, persistent matcher `:3687`,
  `ggml_cuda_try_fuse` `:4000`, the five-op fusion `:4762`, capture-only node loop `:4939`, `:4972`, graph launch `:5211`,
  graph gate `:5301`, `graph_compute` `:5313-5380` (warm-up `:5340-5360`, capture `:5375`), `GGML_CUDA_GRAPH_OPT`
  `:5535`; `common.cuh` (`FAST_FP16_AVAILABLE` `:274`, `is_enabled` `:1298`, `cuda_graph()` `:1475-1497`, PDL
  `:1683-1763`); `fattn.cu:560`, `:704-718`; `fattn-tile.cuh:1249-1253`; `fattn-common.cuh:924`, `:1137`, `:1192-1224`;
  `top-k.cu:29`, `:212-251`.
- `ggml/src/ggml-backend.cpp`: `copy_input` `:1827-1873`, `compute_splits` `:1875-1957`, `alloc_graph` `:2103-2120`,
  `graph_compute_async` `:2128-2141`, `synchronize` `:2143-2154`; `ggml/src/ggml-cpu/ggml-cpu.cpp:170-191`,
  `ggml-cpu.c:3437`.
- `src/llama-context.cpp` (`synchronize` `:916-922`, `process_ubatch` `:1567-1650`, `decode` `:2221-2600`,
  `graph_compute` `:3134-3160`, `llama_get_sampled_token_ith` `:4720-4723`); `src/llama-batch.cpp` (`:28`, `:257`, `:285`,
  `:832-862`); `src/llama-kv-cache.cpp` (`:786-850`, `:1135`, `:1300-1312`, `:1619`, `:1779-1821`);
  `src/llama-kv-cache-iswa.cpp:185-190`; `src/llama-graph.cpp` (rescale switch `:1829-1842`, `can_reuse` `:1637`,
  `build_inp_out_ids` `:3000-3020`, `build_attn` for the sliding-window cache `:3658-3730`); `src/llama-model.cpp:1689-1691`;
  `src/models/flint.cpp:87-127`; `common/sampling.cpp:607`, `:624`; `tools/server/server-context.cpp` (`:2098-2224`,
  `:4584-4589`, `:4768-4797`); `tools/server/server-queue.cpp` (`:163-206`, `:222-260`).
- `pipe2:src/llama-model.cpp:1699-1705` (`LLAMA_INPUT_ON_GPU`), `pipe2:ggml/src/ggml-backend.cpp:981-992`, `:1879`
  (packed uploads); `mask2` (Q116).
- Commits `fedac87b2` (graphs on 6.1), `9d5f1530e` (compute-6.1 gating), `ad3e84f27` (persistent kernel default,
  rig +5.6–8.8 %).

This project (P)
- [Quant registry](../../../quants/experiments.md): Q28, Q38, Q49, Q50, Q54, Q55, Q59, Q60, Q76, Q81, Q88, Q100, Q108,
  Q109, Q116–Q121, Q141, Q144, Q146.
- Rig recordings re-read for this note: Q38's `GGML_CUDA_PROFILE_HOST` trace and event profile of S4-L0 on six cards
  ([`scripts/cuda_profile_summary.py`](../../../scripts/cuda_profile_summary.py)); Q120's executor run on nine cards
  ([`bench/pipespec/rig_hostloop.sh`](../../../bench/pipespec/rig_hostloop.sh), per-token traces and summary).
- [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md) (1.85 µs per empty graph node on a P106);
  [`scripts/build_llamacpp_docker.sh`](../../../scripts/build_llamacpp_docker.sh); the notes linked in the scope.
