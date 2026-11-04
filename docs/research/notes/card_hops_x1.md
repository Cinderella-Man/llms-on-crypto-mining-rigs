# Cheaper card-to-card hops on PCIe 1.0 x1 without peer-to-peer: bytes, the driver path, the host and prompts (2026-11-04)

Scope: what one hop between two P106-090 cards costs in a layer split (nine cards, two ASM1184e switches, every card on a
PCIe 1.0 x1 link, no peer access, a 2-core Celeron host, driver 580, CUDA 12.9), which part of it is bytes and which is
latency, and what can make it cheaper for decode and for prefill. It starts from the finding of
[the megakernel note](megakernel.md) that the hops may outweigh every launch-overhead idea, and builds on
[multi-GPU performance](multigpu_pascal_performance.md) (Q2, Q5), [the host in the decode loop](host_decode_loop.md)
(2.4, 2.5), [idle cards](idle_cards.md) (1.1), [prompt processing](prefill_pascal.md) (3.1) and
[distributed inference](distributed_inference.md). Registry rows: Q81, Q231, Q284, Q505, Q680–Q684 in
[quants/experiments.md](../../../quants/experiments.md).

Conventions:
- "(est.)" marks this note's own arithmetic or inference; the basis is next to it. Numbers without it are measured or
  quoted from the linked source.
- `file:line` refers to the fork at `37e717ff6` (the v2.0 engine). CUDA files are in `ggml/src/ggml-cuda/`.
- A **hop** is everything between the last kernel of card i's layers and the first kernel of card i + 1's layers: the
  copy of the residual (Flash-Next: four streams × 2,560 FP32 = 40,960 B per token), the synchronization between the two
  cards and the gap before the next graph starts. A **boundary** is the place between two cards; a token crosses eight.

## Summary

- **The bytes alone take 0.41 ms per hop and 3.3 ms per token.** 40,960 B at the rig's measured 213 MB/s down and
  189 MB/s up is 192 + 217 µs. Both rates are at the PCIe protocol limit of a gen1 x1 link (est., section 1.2), so only
  fewer bytes, fewer boundaries or overlap shorten that part.
- **An unprofiled Flash-Next hop is est. 0.5–0.9 ms: 4–7 ms of a ~121 ms token (3–6 %), not the 14–18 ms the profiled
  hops suggest.** `GGML_CUDA_PROFILE` synchronizes the stream after every evaluation and writes a CSV line per launch
  group, and the census defines a hop as the host window minus the card's GPU time, so the profiled 1.7–2.2 ms contains
  the profiler's own work, the synchronous input copies and the graph launch. The in-situ evidence without a profiler is
  Flint's token: nine cards against six differ by 1.2 ms over three boundaries, 0.4 ms per boundary for an 8 KB hop whose
  bytes take 0.08 ms. A fixed part of ≥ 0.3 ms per hop is 10× what fast hosts show for a driver-staged copy (18–33 µs).
  Even at 4–7 ms the hops are as large as the whole launch ceiling of the persistent-kernel track (≤ 7 ms) and larger than
  the fusions of Q683 (0.5–1.5 ms).
- **The fixed part lives in the driver's staged copy, which NVIDIA does not document, and the leading explanation is host
  involvement.** A staged peer copy runs in chunks, and an answer on NVIDIA's developer forum says the CPU has to step in
  between the legs; with two cores held by a yield-spinning submit thread and libgomp's spinner, a driver thread's wake-up could
  cost the 0.3 ms (hypothesis H1, est.). A hop built only from copy-engine operations ordered by events, or from a mailbox
  in mapped host memory, needs no such thread. C140's `--hop` run decides it if it is read as in section 6.
- **Fewer bytes is the one lever that is certain.** F16 or BF16 halves the hop (−1.6 ms per token), Q8_0 blocks quarter it
  (−2.4 ms); both exist as CUDA `CPY` kernels in the fork, and upstream's own host-staged all-reduce already puts BF16 on
  the wire by default (Q231 is the KLD gate). A verify pass of k tokens carries k × 40 KB per boundary, so bytes matter
  more with speculation: 13 ms of a 4-token pass.
- **A hop without the host thread is not new code.** Upstream's internal all-reduce (PR #22299) is a mailbox in mapped
  pinned memory: stores, `__threadfence_system()`, a volatile flag, a polling peer. It runs on Volta and later for two
  GPUs; a fork ports it to Pascal boards without peer access with a plain spin and measured +2–3 % decode in tensor
  split (and +20 % prompt speed from a separate chunked exchange). Q682 starts from that code; it saves the driver's
  fixed part, not the bytes.
- **Nothing else bypasses host memory on Pascal GeForce.** GDRCopy and GPUDirect RDMA are for Tesla and RTX workstation
  parts (GeForce is not listed), the open kernel modules and every published peer-to-peer patch need Turing or later,
  and the P106 runs the proprietary 580 branch. Stream memory operations are the only item that might work: the V2 calls
  need no driver option since CUDA 11.7 (the V1 option `NVreg_EnableStreamMemOPs` would be the owner's decision), but
  support on a GP106 is untested and they add nothing over events until events prove slow.
- **A CUDA graph over several cards gains nothing here.** The hops are already asynchronous (the host launches all splits within a few ms and waits
  once); a graph memcpy node runs on the graph's own device, and no source shows a staged
  peer copy inside a graph.
- **Prefill is not link-bound yet, but the hop sits on the source card's compute stream.** The links cap prompt speed at
  est. 2.5k tok/s with F32 hops (5k with F16) against 60–150 measured, but each staged copy holds the source's stream for
  both legs, 0.2–0.5 s per 512-token micro-batch against a stage of est. 2–3.4 s: est. +6–20 % prompt speed from a copy
  stream (Q284), and F16 or Q8_0 shrinks what is left.
- **The host side is small per hop and unmeasured per wake-up.** A decoded token has no per-hop host synchronization;
  eight sequential syncs happen only under the profiler, with pipeline parallelism off, or with the inputs unpacked on
  card 1. The missing number is the wake-up latency of a yield, spin and blocking wait on this Celeron with 0, 1 and 2
  busy threads; section 6 adds it to the bench.
- **Measure a hop with the host clock and a chain, not with events.** CUDA timing events change the token on these cards
  (Q49: 12 events per token cost 14 ms), `cudaEventElapsedTime` fails across devices, and `%globaltimer` ticks at 1 µs on
  pre-Hopper GPUs and has a per-GPU epoch.

## 1. What a hop is on the rig

### 1.1 The path in the fork

For a decoded token the scheduler walks the nine splits in order (`ggml-backend.cpp:2045-2140`). Per boundary, for the
residual (not a user input):
1. A wait on the destination for its previous use of this input copy: `cudaStreamWaitEvent` (`:1905`), no host block.
   Events exist only when pipeline parallelism is on (`GGML_SCHED_MAX_COPIES`, default 4, `ggml/CMakeLists.txt:188`); the
   condition is layer split, every layer offloaded, no tensor override and no MoE cache (`src/llama-context.cpp:567-595`).
   Without it the scheduler calls `ggml_backend_synchronize` on the destination first (`:1907`): the host waits for an
   idle stream, a few µs at decode.
2. `ggml_backend_cuda_cpy_tensor_async` (`ggml-cuda.cu:2663-2718`): `cudaMemcpyPeerAsync` on the **source card's compute
   stream** (`:2701`), a `cudaEventRecord` of a timing-free event on the source stream (`:2711`), a `cudaStreamWaitEvent`
   of the destination stream on it (`:2714`). There is no check for peer access: without it the driver stages the copy.
3. The split's host inputs (positions, masks, the embedding row on card 0) as synchronous copies, one `cudaMemcpyAsync`
   and one synchronize each, unless packed (Q120; off in the release defaults).
4. The graph launch (a CUDA graph replay on compute 6.1), then a record of the split's event (`:2133-2135`).

That is six runtime calls per boundary plus the input copies. The host spends 0.12–0.17 ms in each card's
`graph_compute` and enters the next card's ~0.32 ms later (Q49's host trace, per [host_decode_loop.md](host_decode_loop.md));
six cards are enqueued in ~1.9 ms (est. ~3 ms for nine) and the host then waits once for the last card. The copy and
its events are captured in no graph. `GGML_SCHED_DEBUG=2` prints each split's inputs with their sizes; a boundary that carries more
than one tensor pays one copy and one event pair for each (not yet checked for Flash-Next).

Prompt micro-batches use the same path: up to four micro-batches in flight, each hop on the source card's compute stream.

### 1.2 The bytes

| Payload per token and boundary | Bytes | Down (213 MB/s) | Up (189 MB/s) | One hop | Eight hops |
|---|---:|---:|---:|---:|---:|
| Four streams, F32 | 40,960 | 192 µs | 217 µs | 0.41 ms | 3.27 ms |
| F16 or BF16 | 20,480 | 96 µs | 108 µs | 0.21 ms | 1.64 ms |
| Q8_0 blocks (320 × 34 B) | 10,880 | 51 µs | 58 µs | 0.11 ms | 0.87 ms |

"Down" is card to host (the rig's 213 MB/s D2H), "up" host to card. For a pass of k tokens multiply by k: a 4-token
verify costs 1.64 ms per boundary, 13.1 ms per pass, in F32. For the small sizes of the question (est., rate plus
~5–10 µs of launch per leg): 4 KB 19 + 22 µs, 16 KB 77 + 87 µs, 64 KB 308 + 347 µs; below ~16 KB the fixed costs
dominate, above it the bytes do.

- **The link runs at its protocol limit (est.).** In the TLP model of [Neugebauer et al.](https://www.cl.cam.ac.uk/research/srg/netos/projects/pcie-bench/neugebauer2018understanding.pdf)
  a memory write adds a 24 B header per payload and a completion adds 20 B. A gen1 lane carries 250 MB/s before
  overhead; 128 B writes give 211 MB/s (measured 213) and 64 B completions give 191 MB/s (measured 189). The match
  suggests 128 B payloads and 64 B read completions on this root complex, and that no efficiency is left to win.
- **The two legs use different links, so they can overlap.** The staged copy's first leg loads card i's link and the
  second loads card i + 1's. Two chunks overlap them (see section 3.1): 40 KB takes 0.31 ms instead of 0.41, four chunks
  0.26 ms before the per-chunk event cost (est.).
- **Behind a switch the uplink is shared.** One switch's PCIe 2.0 x1 uplink carries 360–480 MB/s for four cards (measured
  90–120 MB/s each, host to card, with nine cards transferring, [hardware](../../mining_rig.md); card to host was
  not measured). At decode one hop is active at a time
  and the uplink does not matter; with several tokens in flight (Q87, Q550) or a full prompt pipeline it does (section 4.5).
- **Card 4 sits on its own root port** and measures ~200 MB/s; it has no uplink to share.

### 1.3 The fixed part

Measured latencies of a staged peer copy and of the pieces, none on a gen1 x1 link:

| Measurement | Setup | Value | Source |
|---|---|---|---|
| Small staged peer copy, P2P disabled | 2 × RTX 3090, PCIe 4.0 x16, `p2pBandwidthLatencyTest` | 18.4 and 32.3 µs (1.0–1.2 µs with P2P) | [ik_llama.cpp #1380](https://github.com/ikawrakow/ik_llama.cpp/discussions/1380) |
| 4 KB `cudaMemcpyPeer` (driver-staged) | 2 × P100, one on a PCIe 3.0 x4 chipset link, driver 580 | ~33 µs | [Kmic-68/llama.cpp #12](https://github.com/Kmic-68/llama.cpp/pull/12) |
| Launch overhead per staging step | NVIDIA forum, non-P2P peer copies | ~5 µs per step; 100–200 µs above 5 MB | [forum](https://forums.developer.nvidia.com/t/cudamemcpypeerasync-launch-overhead/268437) |
| Null kernel launch | NVIDIA forum | ≥ 5 µs | [forum](https://forums.developer.nvidia.com/t/host-device-latencies/71000) |
| 128 B DMA read round trip | PCIe 3.0 x8 NIC, Xeon | ~1,000 ns, ~900 ns of it PCIe | Neugebauer et al., Fig. 2 |
| CPU read of GPU memory over BAR1 | V100, GDRCopy | 0.9 µs for 1 B, 8 µs for 4 KB; `cudaMemcpy` overhead 6–7 µs | [GDRCopy README](https://github.com/NVIDIA/gdrcopy) |
| Six-card against nine-card token, Flint, unprofiled, per boundary at 8 KB | this rig | 0.4 ms (the difference of L, 24.6–24.8 against 23.4–23.6 ms over 3 boundaries); 0.55–0.65 ms by the fit in the same table | [idle_cards.md](idle_cards.md) 1.1 |

Fast hosts show 20–35 µs per staged small copy. The Celeron's host-side cost is 3–5× that (est.), 60–150 µs, and a gen1
link adds a few µs per PCIe round trip (a 24 B request and completion take ~0.1 µs each on the wire at 250 MB/s, est.).
That sums to 0.06–0.15 ms of fixed cost. Flint's six-versus-nine-card difference says 0.3–0.5 ms. The gap is the
unexplained part, and three hypotheses cover it:

- **H1: a driver thread relays the staged copy and competes for the two cores.** A forum answer on NVIDIA's developer
  site says staged transfers proceed in chunks that overlap the legs, that the CPU therefore has to intervene, and that
  this explains the host blocking another user saw ([forum](https://forums.developer.nvidia.com/t/queueing-device-to-device-peer-memcpy-stalls-concurrent-copy-operations/295894)).
  On this rig the host is not blocked (Q49), but a relay thread would be woken by the first leg's completion and queue
  behind a yield-spinning submit thread and libgomp's spinner (~20 ms per token with `-t 2`, est., Q119). Prediction:
  the staged hop gets slower with busy cores; `-t 1` or `OMP_WAIT_POLICY=passive` and thread pinning make it faster; an
  event-ordered D2H + H2D and a mailbox do not change.
- **H2: GPU-side semaphores in host memory carry the hand-off** (a release on card i, an acquire poll on card i + 1).
  Then the fixed part is a few tens of µs and nothing but bytes is left. Prediction: no dependence on host load.
- **H3: staged copies serialize through one driver resource.** One forum report has a host-staged copy queued behind
  pending work stall a concurrent copy; with several tokens in flight the hops would queue. Q120's eight conversations
  reached 66 tok/s against an est. 235–256 with 13 ms of waits per token that nothing else explains. Prediction: the
  latency of one hop grows with the number of pairs copying at once (Q81's "9 pairs at once").

### 1.4 Why the profiled hop overstates

The fork's `GGML_CUDA_PROFILE` records an event between every two launch groups and, per its own comment, synchronizes
the stream after every evaluation. The census takes a card's hop as the host clock at its synchronize, minus the previous
card's, minus its GPU time (`run/census.py`). That window holds, in series: the previous card's CSV lines (250–340),
the synchronous input copies, the peer copy (≥ 0.41 ms), the graph launch and the synchronize's wake-up. An unprofiled
token overlaps the host work with the cards. The profiled number is an upper bound; 1.7–2.2 ms minus est. 0.2–0.3 ms of
CSV, 0.2–0.35 ms of input copies and 0.2–0.3 ms of launch and wake-up leaves ~1 ms, still above the 0.5–0.9 ms of the
in-situ estimate, which is what C140's `--hop` will settle.

## 2. Small-transfer latency: pinned, pageable, peer, events, syncs

- **Pinned, 4–64 KB (est.):** bytes / rate + ~5–10 µs per leg, as in section 1.2. Allocate with `cudaHostAlloc` and the
  portable flag; under unified addressing a pinned block is mapped and portable for every context anyway. Pinned
  `cudaMemcpyAsync` is fully asynchronous.
- **Pageable:** [NVIDIA's synchronization rules](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html): a
  device-to-pageable copy returns only when finished, a pageable-to-device copy is staged first, and the driver may
  synchronize with the stream to stage. For a hop it adds a CPU `memcpy` (40 KB at a few GB/s on one DDR4 channel, ~10–20
  µs, est.) and puts the host in the loop. Hops use pinned buffers only.
- **`cudaMemcpyPeerAsync` without peer access:** [NVIDIA's guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
  says the copy goes through host memory and is slower, and that an asynchronous inter-device copy may overlap work in
  other streams. A forum answer describes two transfers (D2H, then H2D), possibly in chunks of a ~2 MB double buffer,
  each step ~5 µs of launch. Upstream tried it for the large path of its all-reduce (2 × RTX 5090, PCIe only) and found it slower than explicit
  chunked D2H and H2D with events, and reverted ([PR #22299](https://github.com/ggml-org/llama.cpp/pull/22299)).
  llama.cpp's maintainers closed a proposal to avoid it on non-P2P boards because the fallback works
  ([PR #21846](https://github.com/ggml-org/llama.cpp/pull/21846)); the reporter had claimed garbled output through
  event ordering on two non-P2P cards, and the maintainers disputed the premise.
- **A cross-device event wait:** `cudaStreamWaitEvent` works across devices and is the documented way to order two cards
  (same guide). No measured latency exists in any source found. The mechanism is not documented either; the likely one
  is a semaphore in host memory (est.), which would cost a posted write plus a poll read, ~2–5 µs of link time plus the
  poll interval. The bench's `event-only` variant (section 6) measures it.
- **Host synchronization:** `cudaStreamSynchronize` on an idle stream costs a few µs. On a busy stream the wake-up is
  the cost: ≥ 5 µs for a yield-spinning wait on a free core, est. 20–80 µs for a blocking wait (an interrupt and a thread
  wake-up) on this Celeron ([host_decode_loop.md](host_decode_loop.md)); no measurement exists on a 2-core host with
  busy threads. Measured elsewhere: blocking sync cost 3.5 % of decode on a laptop RTX 4070 (30.3 / 32.2 → 29.2 / 31.2
  tok/s, [PR #26487](https://github.com/ggml-org/llama.cpp/pull/26487)); spin instead of yield gave +19 % and +46 % on a
  DGX Spark where extra synchronizes each paid a wake-up ([PR #16585](https://github.com/ggml-org/llama.cpp/pull/16585)).

## 3. Taking the host thread out of the hop

### 3.1 Kernels writing to mapped host memory, a peer polling a flag (Q682)

**It exists upstream.** [PR #22299](https://github.com/ggml-org/llama.cpp/pull/22299) (merged 2026-05-10, fork
`allreduce.cu`) is the mechanism, for two GPUs in tensor split:
- Buffers from `cudaHostAlloc` with the portable and mapped flags, written and read only by kernels
  (`allreduce.cu:268-298`); no write-combined flag.
- Phase 1: every thread stores 16 B vectors into the host buffer (BF16 on the wire for F32 data by default, `:435`,
  `:765-772`); `__threadfence_system()`; thread 0 stores a monotonically increasing token into its own flag, one 64 B
  line per block and rank, with a volatile store and a second `__threadfence_system()` (`:151-162`). A missing fence
  caused about one hang in fifteen runs on a Blackwell board before it was added.
- Phase 2: thread 0 polls the peer's flag until it equals the token, `__nanosleep(100)` between reads (`:166-170`);
  then `__threadfence_system()` and the data are read as vectors.
- No atomics: they would need `hostNativeAtomicSupported`, absent on PCIe consumer GPUs (comment at `:57-59`); one writer
  and one reader per flag.
- Limits: exactly two devices; `__nanosleep` is sm_70, so the pipeline is not built below Volta (`:408-418`). A two-slot
  ring plus events keeps a slot from being overwritten before its reader is done (`:225-235`).

**It has run on Pascal.** The fork [Kmic-68/llama.cpp #12](https://github.com/Kmic-68/llama.cpp/pull/12) allows it on
compute 6.x when `cudaDeviceCanAccessPeer` says no, with the comment "pre-Volta has no __nanosleep: spin on the mapped
host flag". Two P100s with one on a PCIe 3.0 x4 chipset link, tensor split: tg256 +2–3 %, and with a chunked
prompt-time exchange pp512 339.8 → 406.3 tok/s (+20 %), logits within KLD 4e-6. For layer split the pattern is simpler:
one producer, one consumer, a tail kernel on card i and a head kernel on card i + 1.

**Facts for a Pascal version:**
- **Ordering.** `__threadfence_system()` orders a thread's writes for host threads and peer devices
  ([CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)); the
  scoped acquire and release loads and `fence.sc` belong to the sm_70 memory model, so on sm_61 the idiom is volatile
  plus the system fence, as upstream does. PCIe posted writes from one device stay in order, so the data cannot pass the
  flag; the consumer reads the data only after the flag read returns.
- **Atomicity.** Naturally aligned 1–16 B loads and stores to mapped host memory are single accesses as seen by other
  devices; atomic functions on mapped memory are not atomic across the host or other GPUs
  ([CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)). A 4 B flag
  needs nothing more; a 16 B store can carry 12 B of data and a flag.
- **No caching.** The GPU does not cache mapped host memory ([best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)),
  so every poll is a PCIe read and every data byte is read once over the link.
- **Poll latency (est.).** A 128 B read round trip is ~0.9 µs of PCIe on a gen3 x8 NIC; on a gen1 lane behind a switch a
  24 B request and a 24 B completion add ~0.3 µs of wire time over the two links, plus a switch crossing each way:
  est. 1.5–2.5 µs per poll, so a flag set by card i is seen ~2–4 µs later. A polling card sends ~12 MB/s each way, ~6 %
  of its link, and four polling cards load a switch uplink by ~10 %; a `clock64()` backoff of a few µs halves that.
- **Write-combined or not.** The runtime reference says write-combined memory crosses PCIe faster on some systems but
  is slow for CPU reads ([cudaHostAlloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)); it
  recommends it for buffers the CPU writes and the device reads. The CPU never touches the mailbox, so the flag is
  harmless; whether it lowers a device's read latency is unknown and is an A/B in the bench.
- **Flag in the line.** NCCL's LL protocol stores 4 B of data with a 4 B flag in one 8 B access so no fence is needed;
  LL128 does 120 B with 8 B of flag but is disabled where PCIe cannot guarantee 128 B atomic stores
  ([Demystifying NCCL](https://arxiv.org/abs/2507.04786)). For a link-bound 40 KB hop a 50 % payload doubles the bytes
  (+0.4 ms); the fence and flag (one extra round trip, est. 2–5 µs) is the better trade.
- **Hang risk.** A consumer kernel that waits for a mailbox must time out and report (the bench gives up after 2^22 polls);
  a card whose polling kernel starts while the previous token is still computing burns only its own idle SMs and link
  reads.

**What it saves.** The mailbox moves the same 40 KB over the same two links; with the producer's last kernel writing to
host memory and the consumer's first kernel reading from it, the two legs and the event are one handshake. It removes
the driver's fixed part and the cross-device event (est. 0.06–0.5 ms per hop, 0.5–4 ms per token), and with per-chunk
flags it lets the consumer start on the first chunk while the producer writes the second (about max(leg) + one chunk,
0.26 ms against 0.41 for 40 KB). It does not remove the bytes: the F16 and Q8_0 savings of section 4 apply to it too.

### 3.2 Stream memory operations

`cuStreamWriteValue32` and `cuStreamWaitValue32` enqueue a store and a wait on a 32-bit location; for mapped host memory
the device pointer comes from `cuMemHostGetDevicePointer`, and the calls reject managed memory. The CUDA 11.7 and 11.8
reference says there are two versions: the legacy one is off by default and needs `modprobe nvidia
NVreg_EnableStreamMemOPs=1` on Linux, with support queried through `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`; the V2
functions "are available by default on all platforms" ([CUDA 11.8](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-driver-api/group__CUDA__MEMOP.html)).
The 12.9 reference no longer mentions the option, the 12.x headers map the plain names to the V2 calls, and a forum
thread could not find a V1 symbol there ([forum](https://forums.developer.nvidia.com/t/is-cuda-driver-api-custreamwritevalue32-v2-deprecated-from-cuda-12-0/318380)).
An older report of `CUDA_ERROR_NOT_SUPPORTED` was a Quadro M6000 on CUDA 10 without the option
([forum](https://forums.developer.nvidia.com/t/custreamwaitvalue32-returns-cuda-error-not-supported/79861)).
- **Pascal GeForce.** No document gives a compute-capability or product-class condition; the page says to query. Whether
  a GP106 under driver 580 accepts the V2 calls is unknown. A 20-line program answers it: the three attributes, then a
  write on card A and a wait on card B through one mapped word.
- **Driver option.** Only the V1 path needs `NVreg_EnableStreamMemOPs`, and changing a module parameter needs a module
  reload on a machine that is never rebooted: the owner's decision, and only if the V2 calls fail.
- **Worth.** They would replace the event pair of a hop by a host-memory semaphore with the data copies as ordinary
  pinned D2H and H2D. The ordering is invisible to CUDA and can deadlock if misused (the reference warns). They add
  nothing over events unless the bench's `event-only` timing is high; then they are the host-independent alternative to
  a mailbox for the synchronization alone.

### 3.3 CUDA graphs over several cards

- Documented: a graph memcpy node copies "from the device on which the graph resides"; conditional bodies and device graphs
  must be single-device; an event-wait node may name an event from another device or context
  ([graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
  [graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)).
- Not documented: kernel nodes on different devices in one graph. A forum answer says explicit construction works and a
  second poster doubted it; no example was posted ([forum](https://forums.developer.nvidia.com/t/synchronization-of-cuda-graphs-launched-on-single-node-multi-gpus-setup/340725)).
  No source shows a staged peer copy as a graph node without P2P.
- Worth: none for hops. The hop is not in a graph and costs the host ~0.1 ms of issue time that overlaps the cards; a
  graph would remove that host time and leave the GPU side unchanged. Parked.

### 3.4 What the alternatives need, so their absence on Pascal is clear

| Mechanism | Needs | P106-090 |
|---|---|---|
| GDRCopy (CPU maps GPU memory through BAR1) | a Data Center or RTX GPU (formerly Tesla or Quadro), Kepler or newer; GeForce is not listed ([README](https://github.com/NVIDIA/gdrcopy)) | not supported; and it moves CPU↔GPU data, not GPU↔GPU |
| GPUDirect RDMA, `nvidia_p2p_get_pages` | "available on both Tesla and Quadro GPUs" ([docs](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)) | not supported |
| `PeerMappingOverride` registry key | exposes a third-party device's BAR to a GPU for GPUDirect Async and RDMA; root only by default ([libgdsync](https://github.com/gpudirect/libgdsync)) | not a GPU-to-GPU switch |
| Open kernel modules | Turing or later ([README](https://github.com/NVIDIA/open-gpu-kernel-modules)) | Pascal is excluded |
| Published P2P patches for GeForce | the open modules (Turing+), resizable BAR with a BAR1 aperture as large as VRAM, Above-4G decoding; IOMMU in passthrough ([tinygrad fork](https://github.com/tinygrad/open-gpu-kernel-modules) and its descendants) | excluded twice: Pascal is not Turing, and Pascal GeForce has no resizable BAR ([hardware_tuning.md](hardware_tuning.md)) |
| NVLink, NCCL P2P | NVLink bridges or peer access | none; NCCL's shared-memory transport (stages through host memory) is the fallback ([NCCL docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)) |
| P2P through the PCH or a switch | a chipset the driver lists and a topology it accepts; Linux bare metal also needs the IOMMU off ([CUDA guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)) | the driver reports no peer access for any pair |

`nvidia-smi topo -p2p r` says why per pair: OK, CNS (chipset not supported), GNS (GPU not supported), TNS (topology not
supported) or NS ([forum legend](https://forums.developer.nvidia.com/t/what-is-the-meaning-of-cns-chipset-not-supported-error-in-nvidia-smi/68055)).
It is read-only and answers whether a pair behind one ASM1184e (the PIX pairs) is refused by the GPU or the topology.

## 4. Fewer bytes

### 4.1 F16, BF16 and Q8_0 hops (Q231)

- **Size.** Section 1.2: F16 −0.20 ms per hop, Q8_0 −0.30 ms; −1.6 and −2.4 ms per token, and k times that per pass.
- **Precedent.** Upstream's host-staged all-reduce puts BF16 on the wire for F32 data by default
  (`GGML_CUDA_AR_BF16_THRESHOLD`, default 1); its perplexity over 32 chunks matched NCCL (3.4044 against 3.4043,
  [PR #22299](https://github.com/ggml-org/llama.cpp/pull/22299)). That sums two partial results once per all-reduce;
  a layer split rounds the residual at eight boundaries, so the error is a different kind and needs its own KLD (Q231).
- **Error model (est.).** F16 rounds to 2^-11 of each value, BF16 to 2^-9 and a Q8_0 block to 2^-8 of its largest
  element. The worst-case error of eight roundings, added in quadrature, is ~0.14 % (F16), ~0.55 % (BF16) and ~1.1 % of a
  block's maximum (Q8_0), on a residual that already carries Q2_0 weight noise. The risks are outlier channels (Q8_0's block maximum hides the
  small channels beside a large one) and F16's range of 65,504, which massive activations in deep models can exceed;
  BF16 has F32's range, and F16 can take a per-boundary scale.
- **Implementation with existing ops.** The fork has CUDA `CPY` kernels F32→F16, F32→BF16, BF16→F32, F32→Q8_0 and
  Q8_0→F32 (`cpy.cu:250-262`, `:488`, `:577`). The graph builder knows each layer's device (`dev_layer`), so it can add a
  cast to the narrow type after the last layer of a card and back to F32 at the start of the next; the scheduler then
  copies the narrow tensor. Two extra launches per boundary (~5 µs). 2,560 is a multiple of 32, so Q8_0 rows are whole.
  The hyper-connection kernels keep reading F32.
- **First measurement.** On the quant machine, dump the residual at the eight boundaries for a few hundred tokens: the
  largest absolute value per stream (the F16 range check), the round-trip error per format, then KLD over the usual
  chunks with the round trip inserted. Gate: within noise of the F32 hop.
- **Two rigs (Q505).** A crossing between rigs carries the same 40 KB at gigabit's 117 MB/s: 0.35 ms of bytes (est.)
  plus the RPC round trip; F16 and Q8_0 save 0.18 and 0.26 ms there too.

### 4.2 Can the streams be summed or compressed before a hop?

- **Exact: no.** Flash-Next's four streams are mixed per layer by a learned, input-dependent combination
  (`DSV4_HC_PRE`, the mixer read before each half-layer) and updated by a post-mix; the next card's first operation reads
  all four to compute its mixing weights and keeps all four for its own post-mix. If the post-mix is doubly stochastic,
  as in DeepSeek's manifold-constrained hyper-connections (est.; this model's mixer was not read for this note), it
  preserves the sum of the streams, but the sum cannot recover them, and the weights depend on each stream.
- **Boundary inside a layer: no gain.** Cutting after the pre-mix sends the single mixed vector (10 KB) and the four
  streams (40 KB) the post-mix still needs.
- **Lossy: possible, unmeasured, and small.** The streams start as copies of the token embedding and may stay
  correlated; stream 0 in Q8_0 and streams 1–3 as 4-bit deltas to it would cost ~7 KB against Q8_0's 10.9 KB, ≤ 0.04 ms
  per hop (0.3 ms per token), because below ~10 KB the fixed costs dominate. It needs the correlation structure of the
  real streams, which the dump of 4.1 provides; parked.

### 4.3 Fewer boundaries

A layer split reads every weight once per token whatever the card count, so a token on fewer cards saves one hop per
removed boundary: est. 0.5–0.9 ms each. Flash-Next's Swift Q2_0 needs seven cards at 64k and eight at 128k (Q213) and
the IQ3_XXS file all nine, so for the best-quality file there is nothing to remove, and for Q2_0 at ≤ 64k nine to seven
saves est. 1.1–1.8 ms. Pipelined speculation (Q87, Q550) wants every card.

### 4.4 Passes of k tokens

Verify passes and drafts carry k tokens per hop: the bytes scale with k, the fixed part does not. At Q583's 4-token pass
the bytes are 13.1 ms of a 241.5 ms pass in F32, 6.5 in F16, 3.5 in Q8_0, so F16 saves 6.5 ms per pass (est. 2.2 ms per
accepted token at ~3 per step) and Q8_0 9.8 ms. The mailbox and manual-staging savings (fixed costs) shrink by 1/k.

### 4.5 Prefill: the pipeline and the link

Each 512-token micro-batch carries 21.0 MB per boundary (512 × 40,960 B). Alone, a hop takes 98 ms down and 111 ms up
(0.21 s); when every card of a switch transfers at once, each gets 90–120 MB/s and the pair takes 0.35–0.47 s (est. from
the measured nine-card rates).
- **The ceiling.** A switch's uplink carries 4 × 40,960 B = 164 KB per prompt token per direction; at 360–480 MB/s that
  is est. 2.2–2.9k tok/s with F32 hops, 4.4–5.8k with F16. Flash-Next's prompt speed is 59–72 tok/s at 1.4k and 128–149 at
  32k (Q215), so the links are not what limits prompts today.
- **What the hop does to a stage.** The staged copy is one operation on the source card's compute stream
  (`ggml-cuda.cu:2701`), so the next micro-batch cannot start on that card until both legs are done: a stage is compute
  plus hop. From the measured speeds and the pipeline efficiency (3 micro-batches over 9 stages at 1.4k, 64 over 9 at
  32k) a full pipeline runs at est. 150–260 tok/s, a stage of 2.0–3.4 s per micro-batch; the hop is est. 6–19 % of
  that. For Flint's 8 KB the same arithmetic gave 5–10 % ([prefill_pascal.md](prefill_pascal.md), row 7).
- **A copy stream fixes the overlap and needs a stable source buffer.** Issued on a separate stream behind an event, the
  copy runs while the next micro-batch computes, but the allocator reuses the compute-buffer region of the residual for
  the next graph; the last op must write into a persistent per-copy buffer (a 21 MB device-to-device copy at 173 GB/s is
  0.12 ms) or the next graph must wait for the D2H. Q284 carries the idea.
- **`-ub` and the scheduler copies.** The link does not bind, so `-ub` is set by compute efficiency: Q109 measured pp8192
  at 565 / 636 / 714 / 692 tok/s for `-ub` 128 / 256 / 512 / 1024 on Flint, so 512 stays. `GGML_SCHED_MAX_COPIES` (the name
  in the fork; PR #6017 called it `LLAMA_SCHED_MAX_COPIES`) stays at 4: one copy cost 3.9–4.2× (Q89); eight would cost compute
  buffer memory the cards do not have at long context. The link would bind at a `-ub` where hop time exceeds stage time,
  i.e. only after prompt compute gets 3–5× faster; then the hop is a third or more of a stage and the copy stream and F16 matter
  most.
- **Card order.** Alternating cards between the two switches does not help: in a full pipeline every card transfers once
  per stage whatever the order, so each switch carries the same four hops.

## 5. The host side

- **What runs per boundary:** section 1.1. The host walks the nine splits once per token (0.32 ms per card) while the cards
  compute 3–6 ms each, so it stays ahead (Q49) and nothing waits for the host inside the token.
- **Syncs per token:** one final wait for the last card, ~45–60 short synchronizations in the unpacked input copies
  (mostly card 1's; Q120), ten in `ggml_backend_sched_synchronize`. Eight sequential syncs happen in three cases:
  under `GGML_CUDA_PROFILE` (a stream sync after every evaluation); with pipeline parallelism off (any `-ot` override or
  the MoE cache), where the scheduler waits for the destination's stream before each hop copy (idle at decode, so cheap,
  but it serializes prompt micro-batches); and in the bench's `host-staged` column.
- **Spin, yield, block.** CUDA's default heuristic makes the waiting thread yield when contexts exceed logical CPUs: nine
  contexts on two CPUs yield-spin, so the submit thread occupies a core for the whole token. `cudaDeviceScheduleSpin`
  never yields and starves the second core's work; `BlockingSync` frees the core and pays an interrupt wake-up per wait.
  At one long wait per token the wake-up is irrelevant (≤ 0.1 ms of 121); the policy matters only if a driver thread
  needs the core (H1). Keep the default for the submit thread, as [host_decode_loop.md](host_decode_loop.md) concludes,
  until the wake-up bench shows otherwise.
- **Placement (to test under H1).** The submit thread alone on core 0; the HTTP threads, the driver's threads, and the
  rest on core 1 (`taskset -p` per thread from `top -H`); `-t 1` or `OMP_WAIT_POLICY=passive` so libgomp's spinner is
  gone (Q119); the NVIDIA interrupts' affinity on core 1 (est. 0–2 %, [hardware_tuning.md](hardware_tuning.md)).
  If H1 is false none of this moves the hop.

## 6. Measuring a hop without a profiler

**What not to use.** CUDA timing events change the token on these cards (Q49: 12 events per token took S4-L0 from 37.4
to 23.4 tok/s); `cudaEventElapsedTime` fails for two events of different devices (the multi-GPU guide above), so a hop
cannot be timed with events at all; `GGML_CUDA_PROFILE` synchronizes after every evaluation (section 1.4);
`GGML_CUDA_PROFILE_HOST` adds two events per evaluation with no sync, which shows launch cadence and not the hop;
`%globaltimer` has a per-GPU epoch (est.) and updates at 1 MHz before Hopper ([forum](https://forums.developer.nvidia.com/t/question-about-nsight-tools-timestamping-mechanism-and-clock-sources/323887)),
fine for a 0.4 ms hop and not for a 2 µs kernel.

### 6.1 Reading C140's `--hop` (`bench/megakernel_p106`)

`--hop A,B` prints, per size (10, 40, 160 KB) and per way, the host-clock median [p10, p90] of one isolated hop: two
pinned copies with a host sync on each (`host-staged`); one `cudaMemcpyPeerAsync` + event + a one-warp kernel on B
(`peer-copy`, ggml's path); a mailbox polled by a kernel on B (`mailbox`). Every number contains the host's launches
and one `cudaStreamSynchronize` wake-up (est. 15–30 µs on the Celeron); subtract that for a GPU-side hop.

Fit each column against bytes from the 10 and 40 KB points (the 160 KB point may show overlapped legs): slope in ns per
byte and intercept in µs.

| Reading | Meaning | Do |
|---|---|---|
| `peer-copy` slope ≈ 10 ns/B (1/213 + 1/189 MB/s = 9.98), intercept ≤ 0.1 ms | bytes-bound; H2 | F16/Q8_0 (4.1), fewer cards; Q682 not worth it |
| slope ≈ 10, intercept 0.2–0.5 ms | fixed-cost bound; H1 or H3 | manual staging first (below); Q682 if it does not remove it |
| slope ≥ 15 ns/B | an extra copy or serialization in the staged path | read the 160 KB point; manual staging |
| slope 5–8 ns/B at 160 KB | the driver overlaps the legs | the chunked idea gains less than est. |
| `mailbox` − `peer-copy` intercept ≥ 0.2 ms (the old gate of [host_decode_loop.md](host_decode_loop.md)) | Q682 pays | build it from the upstream kernel |
| mailbox timeouts ≠ 0 | the mailbox column is invalid | rerun |

The hop of a token is then 8 × (`peer-copy` at 40 KB − 15–30 µs); compare it with 4–7 ms.

### 6.2 What to add to the bench

1. **`manual`:** a pinned D2H on A's stream, an event, a wait on B's stream, a pinned H2D on B's stream; with 1, 2 and 4
   chunks. It is upstream's copy-engine pattern and the test of H1: if it sits near bytes + 30 µs while `peer-copy`
   sits 0.3 ms higher, the driver's staging is the cost.
2. **`event-only`:** a one-warp kernel on A, an event, a wait on B, a one-warp kernel: the cross-device event cost.
3. **Ping-pong mailbox:** two flags, A→B→A for 10^5 round trips in kernels, read with `clock64()` on A: one-way flag
   latency at 4 B, 4 KB and 40 KB without the host clock. Poll back-off variants (none, a ~2 µs `clock64()` loop).
4. **Host load:** 0, 1 and 2 busy threads on the two cores (spinners pinned with `pthread_setaffinity_np`) around every
   column. H1 predicts `peer-copy` rises and `manual` and `mailbox` do not.
5. **Concurrency:** 1, 2 and 4 disjoint pairs at once, (0,1), (2,3), (5,6), (7,8): H3 predicts per-hop latency grows with
   the pairs; the mailbox should not.
6. **Pairs:** (0,1) same switch, (3,4) switch to direct, (4,5) direct to switch, (3,5) across: one run each.
7. **Sizes:** 4,096, 10,880 (Q8_0), 20,480 (F16), 40,960, 163,840.
8. **Chain:** k = 2…9 cards in a ring on one host clock (each card adds 1 to the buffer so the chain checks the data);
   time versus k gives the hop with the launch and wake-up as the intercept.
9. **`cudaHostAllocWriteCombined`** on the mailbox against plain pinned: poll latency and the 40 KB time.
10. **Sync latency:** a trivial kernel launched and waited on with `Auto`, `Spin` and `BlockingSync`, 0/1/2 busy threads, p50
    and p99 (the missing wake-up number of section 2).

### 6.3 In situ

- **Token time against the number of cards** with one file that fits the smallest count, same context, same build:
  decode and prompt tok/s at 5…9 cards (`-ts` and `CUDA_VISIBLE_DEVICES`). The decode slope per added boundary is the
  hop plus its scheduler share, the number that matters; the prompt slope is 0.41 ms per token per boundary if hops do
  not overlap and ~0 if they do. Flint's 0.08 ms per extra hop in a one-micro-batch pp512 (8 KB × 2 legs) is exactly the
  bytes.
- **`GGML_SCHED_DEBUG=2`** on one short decode: the inputs of every split, to count the tensors that cross a boundary.
- **`scripts/cudacount.c`** (`LD_PRELOAD`; it has no validated counts yet, so check it against a known call count
  first): calls per token of `cudaMemcpyPeerAsync`, event records, stream waits and syncs.
- **Host-load A/B on the real server:** the placement of section 5 with H1 as the prediction; decode tok/s on one agent
  line.

## 7. Where this note differs from the earlier ones

- [megakernel.md](megakernel.md) §4 and the census put the profiled hops at 1.7–2.2 ms and read them as the hop. They
  include the profiler's synchronize, CSV write and serialized host work (1.4); the unprofiled hop is est. 0.5–0.9 ms.
- megakernel.md §4 says the mailbox saves "the host's synchronize-copy-launch latency" and that the bytes cannot shrink
  because the output has to stay exact. An unprofiled token has no per-hop host synchronize (1.1), so the mailbox saves
  the driver's staged-copy overhead; and the bytes can shrink to F16, BF16 or Q8_0 if the KLD check allows (4.1).
- [multigpu_pascal_performance.md](multigpu_pascal_performance.md) Q2 estimates ~0.1 ms per hop for an 8 KB residual. The
  unprofiled Flint token puts it near 0.4 ms (1.3); its Q5 gap on whether host-staged copies overlap compute is closed:
  they sit on the source card's compute stream and do not (4.5). The scheduler's copies option is `GGML_SCHED_MAX_COPIES`.
- [host_decode_loop.md](host_decode_loop.md) §2.5 calls a graph over all nine cards possible; the documented single-device
  rules and the absence of any example make it unverified, and it gains nothing for hops (3.3). Its gate for the mailbox
  (≥ 0.2 ms per hop) is plausibly met by Flint's in-situ 0.3–0.5 ms of fixed cost; C140 decides.

## 8. Gaps

- The unprofiled fixed cost of a staged hop on this rig, and which of H1, H2 and H3 explains it. Nothing in any source
  describes the driver's non-P2P path beyond staging through host memory in chunks.
- The cross-device event latency and the wake-up latency of the three wait modes on a 2-core host under load.
- Whether `cuStreamWaitValue32_v2` runs on a GP106 under driver 580, and whether `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`
  is still the right query.
- The latency of a read from mapped host memory behind an ASM1184e: the switch's datasheet gives no latency, and no
  gen1 x1 figure was found; est. 1.5–2.5 µs per poll.
- Whether a Flash-Next boundary carries more than the residual (`GGML_SCHED_DEBUG`).
- The residual's range and its correlation across streams at the eight boundaries (the dump of 4.1).
- Whether `cudaHostAllocWriteCombined` lowers a device's read latency.

## Ideas for the rig

Decode, ranked by est. ms saved per decoded token (one hop est. 0.5–0.9 ms, 8 per token, ~121 ms per token). The savings
overlap: D1 and D3 are alternatives; D2 is incremental over D4 (−0.3 to −1.0 ms more once D4 exists).

| # | Idea | Est. effect per token | Effort | Risk | First measurement | Id |
|---|---|---|---|---|---|---|
| D1 | Q8_0 blocks for the residual hop: cast after a card's last layer, back to F32 at the next card's first (existing `CPY` kernels) | −2.4 ms (−9.8 ms per 4-token pass) | 1–2 days | quality: outlier channels, block maximum; KLD gate | dump the eight boundaries on the quant machine; round-trip error and KLD against F16 | new |
| D2 | Mailbox hop: card i's tail kernel writes mapped pinned memory and a flag, card i + 1's head kernel polls it, from upstream's `allreduce.cu` with a plain spin on sm_61, with F16 or Q8_0 on the wire | −2.0 ms alone (0.2–4); −0.3 to −1.0 over D4 | 1–2 weeks | medium: hangs, polling load, graph interplay | C140 `--hop`: `mailbox` − `peer-copy` intercept ≥ 0.2 ms, and D4 not already capturing it | Q682 |
| D3 | F16 (or BF16 if the range check fails) for the residual hop | −1.6 ms (−6.5 ms per 4-token pass) | 1–2 days | quality: KLD; F16 range | the same dump: largest absolute value per stream, KLD of the round trip | Q231 |
| D4 | Manual staged hop: pinned D2H on the source stream, an event, an H2D on the destination, in 2 chunks, replacing `cudaMemcpyPeerAsync` for non-P2P pairs in `cpy_tensor_async` | −1.5 ms (0.6–4: leg overlap ~0.6–0.8, the driver's fixed part 0–3) | 1–2 days | per-copy staging buffers against pipeline reuse; low | C140 with the `manual` variant: intercept against `peer-copy`, host load 0/1/2 | new (cf. Q81) |
| D5 | Nine to seven cards for Swift Q2_0 at ≤ 64k context, one conversation | −1.1 to −1.8 ms | a flag | loses pipelined speculation's cards (Q550); IQ3_XXS needs all nine | tok/s against card count 5…9 on one file (6.3) | Q81 |
| D6 | Host placement: submit thread alone on a core, everything else and the NVIDIA interrupts on the other, `-t 1` | −1.0 ms (0 if H1 is false) | an hour | none | C140 `peer-copy` under 0/1/2 busy threads; then decode tok/s on one agent line | Q118, Q119 |
| D7 | Stream memory operations (`cuStreamWriteValue32` / `WaitValue32` on a mapped word) as the hop's semaphore | ≤ −0.4 ms | 2–3 days | support on GP106 unknown; V1 option is the owner's decision | the 20-line attribute and cross-card test; `event-only` timing | new |
| D8 | Coding streams 1–3 as 4-bit deltas to stream 0 (stream 0 in Q8_0) | parked: ≤ −0.3 ms beyond Q8_0 | 1 week | quality; depends on the streams' correlation | the correlation of the four streams at the eight boundaries (the D1 dump) | new (parked) |
| D9 | One CUDA graph over all nine cards | 0 (hops are asynchronous already) | weeks | staged memcpy nodes unverified; scheduler redesign | none until a hop proves host-bound | new (parked) |

Prompt speed, ranked by est. effect on prompt tok/s (Flash-Next 59–72 at 1.4k, 128–149 at 32k, agent check 61–64; stage
est. 2.0–3.4 s per 512-token micro-batch, hop 0.2–0.47 s on the source card's compute stream):

| # | Idea | Est. effect | Effort | Risk | First measurement | Id |
|---|---|---|---|---|---|---|
| P1 | Hops on a copy stream behind an event, from a persistent per-copy source buffer | +6–20 % in a full pipeline (less on 3-micro-batch prompts) | 1–2 days | buffer lifetime against the allocator; event ordering | the bench: a 0.5 s kernel on A's stream and a staged copy on a second stream: does the copy overlap? then pp8192 at `-ub 512` on Q2_0 | Q284 |
| P2 | F16 or Q8_0 residual hop (D3, D1) for prompts | +3–10 % (F16) or +5–14 % (Q8_0) while hops sit on the stream (hop 0.2–0.47 → 0.1–0.23 / 0.06–0.13 s); ~0 once P1 hides them, until prompt compute is 3–5× faster | with D1 and D3 | as D1 and D3 | pp8192 with the cast on and off | Q231 |
| P3 | Manual chunked staging for micro-batches: D2H chunks overlap the legs (D4 for prompts) | +2–5 % while hops sit on the stream | with D4 | low | C140 `manual` at 160 KB (and a 21 MB size) | new |
| P4 | `-ub` 512 and four scheduler copies kept; no change until prompt compute is 3–5× faster | 0 | none | none | the hop-to-stage ratio from P1's run | Q89, Q109 |

Order to run: D6 and the placement A/B cost an hour each; C140's extended `--hop` (6.2: `manual`, `event-only`, host
load, concurrency, pairs) is ~1 h on the idle rig and decides D2, D4, D6 and H1–H3 together; the dump and KLD (D1, D3,
D8) are quant-machine work that needs no rig.
