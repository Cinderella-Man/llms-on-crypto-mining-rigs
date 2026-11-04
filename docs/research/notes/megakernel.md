# One persistent kernel per card for Flash-Next on the P106: the bound, the design and a staged plan (2026-11-04)

A megakernel runs all of a token's work on a GPU inside one resident launch: each operation becomes a stage, and a
device-side schedule with barriers or counters between dependent stages replaces the kernel boundaries. This note
asks what that is worth for Qwen3.8-Flash-Next (Swift-1.5 GSQ-RCO Q2_0) on nine P106-090 cards (compute 6.1, 5 SMs,
PCIe 1.0 x1, no peer access), from the traces already recorded: registry row Q670, its census
[docs/data/2026-11-04_megakernel_census](../../data/2026-11-04_megakernel_census/README.md), and the microbenchmark that
settles the two numbers the census cannot, [bench/megakernel_p106](../../../bench/megakernel_p106/README.md).

## Verdict

- **The ceiling is ~7 ms per token, about 6 % of a token.** A Flash-Next token launches ~2,520 launch groups
  (~2,900 kernels with the activation quantizations; ~3,200 in R28's retyped file) over the nine cards. A small kernel
  costs 2.1–2.4 µs as a CUDA-graph node on a P106 (Q88's rig bench; an empty node 1.84 µs). Removing every launch at
  no cost saves 6.7 ms of R13's token (7.4 ms of R28's); R28's token is ~121 ms.
- **A barrier per stage takes most of that back.** A P100 needs 1.8–2.3 µs per `grid.sync()` at one block per SM and
  2.1–3.5 µs at two (Zhang et al. 2020), about what a P106 graph node costs. With a 1 µs barrier on every stage the
  saving is ~3.8 ms per token, with 2 µs ~0.9 ms. Only stages that need no grid-wide barrier pay (dataflow counters,
  as Hazy Research and Q88 use, or chains of elementwise stages inside one block), and ordinary kernel fusion removes
  the same launches with no barrier at all.
- **R13's ~33 ms of "operations without weights" is mostly not launch cost.** Of the 30.9 ms the census counts per
  token (1,658 groups): ~17–23 ms over all groups is the event profiler's own cost (est. 7–9 µs per group, the profiled
  near-zero groups minus the same kernels in a graph); the 393 groups that move a state, a cache or attention take
  16.4 ms profiled (the recurrent-state gathers and copies, the conv window, the Gated DeltaNet update, flash
  attention), which Q436–Q438's fusions attack; the 1,265 near-zero groups would cost ~2.9 ms unprofiled. In Q583's
  k-token passes the weightless time is 33.8 ms per pass plus 2.1 ms per token: fixed per pass, as both launch cost and
  state traffic are.
- **The hops between cards may cost more than the whole ceiling.** Flash-Next's residual (4 streams × 2,560 FP32,
  40 KB) crosses eight card boundaries per token; at the rig's measured 213 MB/s down and 189 MB/s up that is ≥ 0.4 ms
  per hop, ~3.4 ms per token for the bytes alone, and the profiled hops take 1.7–2.2 ms each. A per-card kernel does
  not touch them.
- **Stage 1 (a recurrent layer's chain of weightless ops as one persistent launch): stop as a build.** After Q437 and
  Q438 a recurrent layer keeps ~25 small launches; at 0.3–1.3 µs net per launch over 36 layers that is 0.3–1.2 ms per
  token. Go for the microbenchmark only (one P106 and a second card for the hop, ~10 minutes). Stage 1 is built if the
  bench's
  `census6` mix shows a persistent stage at ≤ 50 % of a graph stage (≥ ~1.2 µs saved per stage, ≥ ~3.5 ms per token
  over the whole token), or if Q680 finds that decode does not replay CUDA graphs at all (then the graphs come first).

## 1. The census: what the recorded traces say

Sources: R13's `GGML_CUDA_PROFILE` trace of v3 on the unmodified Swift Q2_0 file (94 decoded tokens at 65–7,200 tokens
of context), R28's of the retyped file (hyper-connections in Q8_0), Q583's llama-bench trace of k-token passes, Q88's
rig bench for the graph costs. Tables, scripts and the method: the
[census data folder](../../data/2026-11-04_megakernel_census/README.md).

| Per token, nine cards (medians) | R13 | R28 |
|---|---:|---:|
| Launch groups / est. kernels | 2,520 / 2,898 | 2,520 / 3,188 |
| Groups without weights / their profiled ms | 1,658 / 30.9 | 1,658 / 31.0 |
| GPU ms, profiled | 129.1 | 113.3 |
| Hops between cards, host before card 0, wall (profiled, ms) | 15.6, 11.5, 155.4 | 14.3, 11.1, 139.8 |
| Groups of < 10 / 10–20 / 20–50 / ≥ 50 µs | 619 / 1,075 / 215 / 612 | 626 / 1,058 / 319 / 516 |
| Ceiling of a per-card persistent kernel (ms) | 6.7 | 7.4 |

- **Per card**: 250–340 groups per token, of which 163–234 without weights; 54–113 under 10 µs.
- **The profiler records durations only.** It places a CUDA event between every two launch groups, so a group's
  duration runs from the end of the group before it to its own end: its launch gap, the event's cost and its kernels
  together. Within a card's evaluation nothing separates a gap from a kernel; between evaluations the host clock at
  each card's synchronize gives the window each card had. Start and end times per kernel need either `%globaltimer`
  stamps written by the kernels themselves (as `bench/persistent_pascal`'s traced run does), a CUPTI activity trace
  (NVIDIA's profilers refuse the CMP cards, and CUPTI is the layer under them), or a calibration of the event cost on
  the same card, which is what the microbenchmark's `graph+events` variant measures.
- **The profiler's own cost.** Elementwise groups with no real work (scales, gates, adds, norms of 2,560 values) take
  9.2–10.2 µs at the median in the trace; the same kinds of kernel take 2.1–2.4 µs in a graph without events on the
  same card model. That gap (est. 7–9 µs per group) is why R13's profile sums to 136 ms while unprofiled tokens take
  112–120 ms at comparable context, and why the earlier persistent-kernel estimate for Flint (+20–35 %,
  [breakthrough candidates](breakthrough_candidates.md) §2) became +5.6–8.8 % on the rig (Q88).
- **Whether decode replays graphs at all** is not in the trace. The fork enables CUDA graphs on compute 6.1, but a
  graph is re-captured or run eagerly whenever a node's pointers or shapes change; if the scheduler's rotating input
  copies changed them every token, decode would run eagerly and each launch would cost the Celeron's launch time
  instead. A short line with `GGML_CUDA_GRAPH_DEBUG=1` counts the reuses (Q680).

## 2. Prior art

- **Hazy Research's Llama-1B megakernel** ([post](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles),
  [code](https://github.com/HazyResearch/Megakernels)): the whole Llama-3.2-1B forward pass as seven instruction types
  run by an interpreter on every SM from instruction lists built ahead of time; no grid barriers, only counters in
  global memory that each instruction waits on and increments; shared memory in 16 KiB pages handed to the next
  instruction so its weight loads start early. Under 1 ms per forward pass on an H100 at 78 % of its bandwidth
  (~2.5× vLLM, > 1.5× SGLang), under 680 µs on a B200. Their numbers for a boundary: a dummy launch 2.1 µs on a stream
  and 1.3 µs in a CUDA graph on an H100; their model puts ~5 µs of stall (stragglers, launch, load latency) at each
  kernel boundary. The megakernel's own overhead on a B200 is ~250 of 600 µs in activation loads, stores and consistency checks. Their
  [70B tensor-parallel version](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main) adds a global work
  queue (−14 % without it) and pipelining that shrinks the gap between matrix stages from 10.2 to 3.4 µs.
- **Mirage Persistent Kernel** ([arXiv 2512.22219](https://arxiv.org/abs/2512.22219),
  [code](https://github.com/mirage-project/mirage)): a compiler lowers the model to an SM-level graph of tasks and
  events (an event only where a producer's output tile overlaps a consumer's input tile); one persistent kernel with a
  worker per SM and scheduler warps on four dedicated SMs; cross-task pipelining preloads the next task's data.
  1.0–1.7× against SGLang and vLLM on A100, H100 and B200 (Qwen3-8B on an A100: 14.5 → 12.5 ms per token against a
  ~10 ms bound).
- **Grid-barrier cost** ([Zhang, Wahib et al., IPDPS 2020](https://arxiv.org/abs/2004.05371)): `grid.sync()` on a
  P100 (56 SMs) takes 1.77–2.26 µs at one block per SM, 2.06–3.52 at two, 3.45–4.90 at four, 6.5–8.4 at eight; a V100
  is slightly faster. The cost grows with blocks per SM, barely with threads per block. Their reduction with grid sync
  matched the kernel-boundary version within 1 %. No GP106 or 5-SM figure exists; the bench measures one.
- **Cooperative launch on compute 6.x** ([CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html),
  [runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html)): from compute 6.0, when
  `cudaDevAttrCooperativeLaunch` is set; every block resident (grid ≤ occupancy × SMs, else
  `cudaErrorCooperativeLaunchTooLarge`); Linux without MPS. The 9.0 and 10.2 guides required `-rdc=true` for
  `grid.sync()`; the 11.x and 12.x guides do not, and the bench builds without it on CUDA 12.9. CUDA 11.0 added
  cooperative kernels to graphs, including stream capture
  ([release notes](https://docs.nvidia.com/cuda/archive/11.0/cuda-toolkit-release-notes/index.html)). CUDA 12.9 is the
  last toolkit for Pascal, the R580 driver the last branch.
- **CUDA graphs**: they remove the host's per-launch cost and shrink, but do not remove, the GPU-side gap between
  dependent kernels. NVIDIA's [CUDA graphs post](https://developer.nvidia.com/blog/cuda-graphs/) measured 20 kernels of
  2.9 µs on a V100 at 9.6 µs per kernel with a synchronize after each, 3.8 µs overlapped and 3.4 µs as one graph.
  Pre-Ampere GPUs launch a graph in time linear in its nodes
  ([constant-time launch post](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)).
  Programmatic dependent launch, the newer way to overlap dependent kernels, needs compute 9.0. Upstream llama.cpp
  keeps graphs off below Volta since [PR #6766](https://github.com/ggml-org/llama.cpp/pull/6766) (a P40 lost 4 % in
  2024; [PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749) enabled Volta and Turing in 2026, the old gate
  never having been re-measured); this fork enables them on compute 6.1.
- **Persistent threads** ([Gupta, Stuart, Owens 2012](https://escholarship.org/uc/item/3j76d3td)): 2–2.5× over many
  launches when each step has little work, nothing when each block works long; a grid-wide sync can cost more than a
  launch as resident blocks grow. **Persistent RNNs** ([Diamos et al. 2016](https://proceedings.mlr.press/v48/diamos16.html)):
  ~30× at small batch on a Titan X, from keeping the recurrent weights in registers across time steps plus a barrier
  ~10× cheaper than a launch; it needs the weights to fit on chip, which a 2.5 GB-per-card model does not.
- **In this project**: Q88's persistent MoE layer for Flint (one launch instead of 13, counters instead of barriers,
  two blocks per SM needed) gained 198.5 → 171.3 µs per layer on a P106, ~2.3 µs per removed launch, and +5.6–8.8 % per
  token in the fork; with one block per SM it was slower than the 13 launches. Flint's makers run a whole token as one
  persistent kernel on Apple GPUs ([rec](rec_engine.md)).

## 3. What a per-card megakernel would hold for Flash-Next

A card holds 5–6 of the 48 layers (36 Gated DeltaNet, 12 attention). Per layer and token, from the trace's labels:

- **Hyper-connections** (four residual streams of 2,560): before and after each half of the layer, the mixer reads
  the streams (`DSV4_HC_PRE`), the BF16 or Q8_0 down projection 10,240 → 320 and up projection 320 → 10,240, two
  10,240 × 4 injects, and the post-mix (`SCALE+SIGMOID+SCALE+DSV4_HC_POST`).
- **A recurrent layer**: norm, Q3_K/IQ4_XS `attn_qkv` (2,560 → 10,240) and `attn_gate`, alpha/beta (2,560 × 48 BF16),
  the conv window over the cached history (Q438 fuses conv, SiLU, L2 and the slide), the Gated DeltaNet update of a
  ~3 MiB state read and written in place (Q437), the gated output norm and `ssm_out`.
- **An attention layer**: norm, `attn_q` (2,560 → 12,288) and K/V, RoPE, the cache writes, the sparse indexer
  (`indexer.q_proj`, `LIGHTNING_INDEXER`, `TOP_K`, the mask), flash attention over the selected cells, the output.
- **The experts**: the F32 router (2,560 × 512), top-k with renormalization, the routed Q2_0 gate/up + SwiGLU and down,
  the shared expert and its sigmoid gate, the weighted sum, the residual.

Each of these exists as a fork kernel and would become a device function behind a stage table: the Pascal
matrix-vector kernels (`mmvq-pascal`, Q223's Q2_0, Q546's BF16, C70's k-column kernels), the Gated DeltaNet and conv
kernels, `fattn-pascal`, the hyper-connection kernels, norms and top-k. The constraints:

- **Registers.** One kernel's register budget is the largest of its device functions'. Q88's MoE layer needed 128
  registers and ran 2 blocks of 8 warps per SM, below the 20 warps the product kernels use; with one block per SM it
  lost to the separate launches. The weight-streaming stages set the floor: ≥ 2 blocks × 8 warps per SM.
- **Barriers or counters.** A grid barrier per stage costs about what it saves (section 2), so the schedule has to be
  dataflow: a counter per producer tile, consumers waiting only on the tiles they read (Hazy, MPK, Q88), and redundant
  small reductions (each block recomputing a 2,560-value norm from L2) instead of a barrier where that is cheaper.
- **What the boundaries gave for free**: ordering of global writes across blocks (`__threadfence` and L2-only reads of
  what another block wrote), the allocator's buffer reuse (a stage may not overwrite a buffer a slower block still
  reads), and CUDA graphs' per-size capture (the stage table becomes the thing that varies per batch shape).
- **Prefetch.** The gain beyond launches is overlap: streaming the next product's weights while the small stages run.
  A GP106's L2 is small next to a 6.5 MB hyper-connection projection, so the overlap is limited to the head of each
  product.

## 4. The hops between cards bound it

On a layer split the nine cards run one after another for each token: the token's time is the sum of the cards' times
and the eight hops. A hop moves the residual (40 KB) from card i to card i + 1 through the host: there is no peer
access, so the driver or ggml stages it through pinned host memory. The rig's links carry ~213 MB/s down and ~189 MB/s
up ([hardware](../../mining_rig.md)): ≥ 0.4 ms per hop for the bytes, ~3.4 ms per token. The profiled hops are
1.7–2.2 ms each, with the profiler's synchronize and CSV write inside; the unprofiled hop is not measured, and the
bench's `--hop` mode measures it three ways (two copies through the host, a driver-staged peer copy, a mailbox in
mapped memory).

- A per-card megakernel leaves the hops as they are. What shortens them is taking the host out: card i's last stage
  writes the residual into mapped pinned memory with a sequence flag, card i + 1's first stage polls it (Q682). That
  saves the host's synchronize-copy-launch latency, not the bytes.
- The bytes stay: the residual is FP32 and the output has to stay exact, so it cannot shrink.
- The end state is a token-persistent pipeline: each card's kernel stays resident across tokens, polling its mailbox,
  with sampling on the last card and the token id mailed back to card 0 (Q684, parked behind stage 1 and Q682).

## 5. With k-token verify passes and the recurrent states

- **Fixed per pass, shared by k tokens.** In Q583's passes the weightless time is 33.8 ms per pass + 2.1 ms per token
  (k = 1–5; 37.4 + 1.2 for k = 2–5). Launch costs are per pass too, so a megakernel's saving per token falls as 1/k: at
  DFlash2's ~3 tokens per step, ~1–2 ms per token.
- **Shapes per k.** A verify pass's width changes per step, so the stage table needs every width the drafter uses, as
  per-size graphs do (Q701), or the stages take k as a runtime value (C70's k-column kernels already do).
- **Recurrent states.** Q437 reads and writes the Gated DeltaNet state in place from its cache row, which a stage can
  do as well; a verify pass keeps one snapshot per drafted token for rollback (+5.5 ms at k = 4 in Q583), and the host
  decides the rollback after sampling, so a pass's megakernel ends at the pass. Recorded-input replay (Q686) and a
  snapshot ring (R8) change what the stages write, not where the kernel ends.
- **Pipelined speculation** (several tokens in flight across the cards) keeps one pass per card at a time, so a
  per-card kernel composes with it; a token-persistent pipeline (Q684) would have to carry the pipe slots.

## 6. A staged plan

| Stage | What | Gate to start it | Est. gain per token |
|---|---|---|---|
| 0 | `bench/megakernel_p106` on one P106 and the hop between two cards (~10 min); Q680's graph check (~10 min with the load) | now | measures: graph µs per stage, barrier µs, the profiler's cost per group, the hop |
| 0b | The same launches removed without a persistent kernel: fusions of adjacent small ops (Q683), Q437/Q438 on the rig | now (code) | ~2.3 µs per removed launch |
| 1 | One recurrent layer's weightless chain (after Q437/Q438) as one persistent launch in the fork, behind a switch | stage 0: persistent ≤ 50 % of a graph stage on `census6`; or Q680 finds no graph replay | 0.3–1.2 ms |
| 2 | A whole layer, products included, with Q88's counters | stage 1 within 20 % of its estimate | est. 1–2 ms more |
| 3 | All of a card's layers per token | stage 2 | up to the ~7 ms ceiling, less the barriers |
| 4 | Token-persistent pipeline with mailboxes (Q684) | stage 3 and Q682 | the host's share of the hops |

Every stage keeps the fork's arithmetic (same device functions, same reduction order), so its output can be checked
bit for bit against the launches, as Q88's was.

## 7. Ideas this note registers

- **Q680** Does decode replay CUDA graphs on the rig? `GGML_CUDA_GRAPH_DEBUG=1` on one short line counts reuses against
  warm-up resets per card and token (C141).
- **Q681** A lighter kernel profiler: events per layer or per sampled group instead of between every group, with the
  bench's measured event cost subtracted, so small ops stop looking 4× their size (C142); built 2026-11-04 as
  `GGML_CUDA_PROFILE_MODE=layer`, `sample=<N>` and `graph` ([data](../../data/2026-11-04_light_profiler/README.md)),
  the rig check next.
- **Q682** Host-mapped mailbox hops between cards, if the bench's hop shows the staged hop well above its bytes (C143).
- **Q683** The census as a fusion worklist: the single-op groups that remain (`SCALE`, `SILU`, `ADD`, `RMS_NORM+SCALE`,
  the activation quantizations) merged into neighbours, ~2.3 µs per removed launch (C144).
- **Q684** Parked: the token-persistent pipeline (section 4), behind stage 3 and Q682.
