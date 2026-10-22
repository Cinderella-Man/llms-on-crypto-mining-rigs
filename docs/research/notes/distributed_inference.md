# Distributed and heterogeneous LLM inference: pairing the 9 × P106-090 rig with a LAN workstation (as of 2026-10-22)

Scope: systems that split one model across machines, or treat remote GPUs as backends. Each is checked
against the rig's limits (Celeron 3865U with 2 cores and no AVX, 4 GB RAM of which 3.7 GiB is usable,
~26 GB free disk, gigabit Realtek NIC, nine Pascal sm_61 GPUs on PCIe 1.0 x1, no P2P). Rig facts come from
the [hardware spec](../../mining_rig.md) or were measured on the rig on 2026-10-22. The other machine is a
separate, more powerful PC on the same LAN (a "workstation") with a modern NVIDIA GPU; where an argument
depends on its GPU, an 8 GB card is assumed. Every web claim has a URL; the date given is the source's own
date when known. Computations are mine and are labelled as inferences. Anything not measured on this hardware is a
"claim to test".

---

## 1. llama.cpp RPC backend (rpc-server): state, protocol, weight upload, cache, multi-device, latency, benchmarks, limitations

### Takeaway
As of October 2026 the llama.cpp RPC backend is still labelled a fragile, insecure proof of concept, but it
has matured a lot since 2025:
- One `rpc-server` can expose several GPUs, and moves tensors between them on the server (Oct 2025).
- Unchanged graphs are recomputed by ID instead of re-sent (May 2026).
- Commands run asynchronously, so pipeline stages can overlap (Aug 2026).
- Weights stream from the client over TCP into each remote GPU, so the rig needs no model file and no
  page cache for weights. It still needs host RAM equal to the largest single tensor, plus disk space for
  the optional `-c` cache.

Measured results so far consistently show RPC adding capacity, not decode speed.

### Cited Findings

**Status, protocol, flags**
- The README calls RPC a proof of concept: "the functionality is fragile and insecure". It says "Never run
  the RPC server on an open network or in a sensitive environment!" The main host connects over TCP to one
  or more `ggml-rpc-server` instances, and the client is built with `-DGGML_RPC=ON` and run with
  `--rpc host:port[,host:port...]`. —
  [llama.cpp tools/rpc/README.md (fetched 2026-10-22)](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- Current `rpc-server` options: `-t/--threads` (CPU device only), `-d/--device dev1,dev2`, `-H/--host`
  (default `127.0.0.1`), `-p/--port` (default 50052) and `-c/--cache`. With no `-d`, it serves all non-CPU
  devices. If the host is not `127.0.0.1` it prints a boxed warning, but still starts. —
  [rpc-server.cpp, master (fetched 2026-10-22)](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/rpc-server.cpp)
- Wire protocol: every message has a 64-bit size prefix.
  - The command set on master includes ALLOC_BUFFER, SET_TENSOR, SET_TENSOR_HASH, GET_TENSOR, COPY_TENSOR,
    GRAPH_COMPUTE, GRAPH_RECOMPUTE, DEVICE_COUNT, SET/GET_TENSOR_2D, and COMM_INIT / COMM_ALLREDUCE /
    COMM_FREE.
  - Each endpoint has one client-side dispatcher thread that drains a FIFO queue, with `send_async` for
    graph compute and tensor set/get.
  - `rpc_tensor` stores strides as uint32, and a TODO notes strides over 4 GiB would be truncated.

  — [ggml-rpc.cpp, master (fetched 2026-10-22)](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp)
- Protocol version: RPC_PROTO went from 5.1 to 6.0 with the async/event work. A 6.0 client refuses a 5.1
  server, and only major.minor is compared. —
  [unslothai/llama.cpp PR #181 (2026-09-04)](https://github.com/unslothai/llama.cpp/pull/181)
- A 2026 benchmark repo says builds "must match exactly, since the RPC protocol is not versioned". That
  conflicts with the RPC_PROTO handshake above. Both agree that the same commit on both ends is the safe
  choice. — [kjaiswal/llama-cpp-distributed-benchmarks (2026)](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)
- A user reports that differing builds "look like network failures". —
  [michaelspost.com (2026-08-22)](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- Heterogeneous builds work in practice: Metal client with a CUDA server
  ([kjaiswal](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)), and Ada and Blackwell built
  from one image with `CMAKE_CUDA_ARCHITECTURES="89;120"`
  ([michaelspost.com, 2026-08-22](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)).
- RDMA transport (RoCEv2 via libibverbs, or Thunderbolt 5 on macOS 26.2+) is negotiated at handshake, with
  TCP as the fallback. `GGML_RPC_NO_RDMA=1` forces TCP. `GGML_RPC_DEBUG` levels 0–3 go up to per-command
  timings and byte counts; the client also needs `-lv 5`. —
  [README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)

**How weights get to the remote GPUs (the disk and RAM question)**
- By default llama.cpp spreads weights and KV cache across all local and remote devices in proportion to
  available memory; `--tensor-split` overrides this. —
  [README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- Only the host needs model files. —
  [kjaiswal (2026)](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)
- Weights cross the network once at load, and only to a cold worker. —
  [michaelspost.com (2026-08-22)](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- On the server, SET_TENSOR receives the whole payload into a host `std::vector<uint8_t>` sized to the full
  message, then calls `ggml_backend_tensor_set`. No chunking is visible at that layer. —
  [ggml-rpc.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp)
- How the `-c` cache works:
  - It was added by "rpc : send hash when tensor data is above some fixed threshold" (PR #12496, merged
    2025-03-28), which closed the long-requested "RPC offloading using a local model copy" (issue #10095).
    Caching whole GGUFs on the server was tried and dropped as slow.
  - The threshold (`HASH_THRESHOLD`) is 10 MiB, and only weight buffers qualify.
  - Hash: 64-bit FNV-1a.
  - Cache file: `cache_dir/<16-hex-digit hash>`, read entirely into memory with `std::ifstream` (not
    mmapped).
  - On a miss, the client sends the data with a cache flag and the server saves it.
  - Default directory: `$HOME/.cache/llama.cpp/rpc`, or `$LLAMA_CACHE/rpc`, or `$XDG_CACHE_HOME`.

  — [PR #12496](https://github.com/ggml-org/llama.cpp/pull/12496);
  [issue #10095](https://github.com/ggml-org/llama.cpp/issues/10095);
  [ggml-rpc.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp);
  [rpc-server.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/rpc-server.cpp)
- `rpc-server` tracks every allocated buffer and frees them all when the client disconnects. A 2026
  discussion asks for cache files to be tagged with the model name, since the cache directory is a flat
  pile of hashes. — [Discussion #19044](https://github.com/ggml-org/llama.cpp/discussions/19044)
- In a 2024 thread, model loading over RPC ran at about 1.9–2.1 Gb/s on 2.5 GbE, i.e. near line rate. —
  [Discussion #9136](https://github.com/ggml-org/llama.cpp/discussions/9136)

**One server exposing all GPUs vs one server per GPU**
- PR #16276, "rpc : add support for multiple devices" (rgerganov, merged 2025-10-04):
  - One endpoint exposes several devices; device IDs were added to commands, plus a DEVICE_COUNT command.
  - A new COPY_TENSOR copies between devices on the same server. Before this, device 0 → device 1 data went
    through the main host.
  - Author's gigabit benchmarks (client with no GPU; remote RTX 5090 + GTX 1660; gemma-3-4b Q4_0): two
    separate endpoints went from pp512 1039.8 / tg128 33.18 to 1267.0 / 33.68. Large pp gains, small tg
    gains.
  - slaren noted the remaining latency comes from every command blocking on a server reply, and suggested
    async operations.

  — [PR #16276](https://github.com/ggml-org/llama.cpp/pull/16276)
- The README now shows one server exposing all accelerators by default, with `CUDA_VISIBLE_DEVICES=0` or
  `--device CUDA0` to restrict it. — [README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- Older guides (e.g., the Seeed Jetson tutorial) used one rpc-server per device on separate ports. —
  [Seeed wiki](https://wiki.seeedstudio.com/ai_robotics_distributed_llama_cpp_rpc_jetson/)
- PR #26500 (merged 2026-08-30) fixed failures with two or more `ggml-rpc-server` instances (`[create_node]
  invalid data ptr`, issue #28047) by sending buffer pointers only to the server that owns them. —
  [unsloth PR #181](https://github.com/unslothai/llama.cpp/pull/181)

**2025–2026 performance work on round trips**
- PR #22701, "rpc : use graph uid instead of graph cache" (rgerganov, merged 2026-05-05): the client
  stores the last graph uid; if unchanged, it sends GRAPH_RECOMPUTE with only device and uid, not the
  serialized graph. — [PR #22701](https://github.com/ggml-org/llama.cpp/pull/22701)
- On master, graphs with uid ≠ 0 are cached per device, up to 1024 entries, and `free_buffer` invalidates
  them. — [ggml-rpc.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp)
- An open PR, #29287, proposes a multi-graph cache behind `--graph-cache-mib`. Its merge status is
  unknown. — [PR #29287](https://github.com/ggml-org/llama.cpp/pull/29287)
- PR #18626, "rpc : implement event and async backend APIs" (merged 2026-08-26):
  - Adds a per-endpoint dispatcher with a background I/O thread and advertises `caps.async` and
    `caps.events`, so `ggml_backend_sched` can overlap work across RPC devices.
  - Measured on 2 × DGX Spark over ConnectX-7 (Qwen3.8-27B): pp 1.72–1.89× vs one node, tg 0.93–0.96×
    (no tg gain).
  - Open review notes flag a possible early sync barrier and an event-wrapper leak.

  — [unsloth PR #181 (2026-09-04)](https://github.com/unslothai/llama.cpp/pull/181)

**Measured benchmarks (none on Pascal and none over gigabit with a large MoE)**
- 10 GbE direct link, Mac M2 Ultra (client) + DGX Spark (rpc-server). Iperf 9.41 Gbps, about 0.17 ms
  network RTT per token.
  - Qwen2.5-72B Q4_K_M: tg 11.1 → 5.9 t/s with RPC, pp 28.2 → 29.5.
  - Qwen2.5-7B: tg 91.8 → 52.7, pp 76 → 318.
  - MoE numbers (Qwen3-235B-A22B, 5–8 t/s) are projections, not measurements.

  — [kjaiswal (2026)](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)
- Plain 1 GbE, 0.29 ms RTT (RTX 5060 Ti 16 GB host + RTX 4060 Ti 16 GB worker; `-dev RPC0,CUDA0 -ts 6,4
  -sm layer --flash-attn 1`):
  - Qwen3.8-27B dense Q4_K_M: 16.9 t/s without MTP, about 23–35 t/s with MTP.
  - Qwen3.5-35B-A3B MoE Q4_K_M: 51.7 t/s, or 92–115 t/s with MTP. Prefill about 600 t/s (27B).
  - The author judges the 1 GbE link not to be the bottleneck, at about 10 KB of activations per token.

  — [michaelspost.com (2026-08-22)](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- GLM-4.6 IQ4_XS, local multi-GPU box plus one RTX 5090 over RPC (10/40 Gb NIC):
  - Local only: 762.7 pp / 26.4 tg.
  - RPC device first in order: about 240 pp / 16 tg.
  - RPC device in the middle (`CUDA0,RPC0,CUDA1,…`): 680 pp / 22.5 tg.
  - The ~9 Gb link limited by PCIe x1 3.0: 159 pp / 12.7 tg.
  - rgerganov: "We are always putting RPC devices first in the device chain" to avoid copying logits over
    the network.

  — [Discussion #16625 (2025-10-17/18)](https://github.com/ggml-org/llama.cpp/discussions/16625)
- GLM-4.5-Air, all on GPU: about 17 t/s on 3 local GPUs with 11 layers' experts on CPU, vs about 53 t/s
  with RPC across 4 GPUs. GLM-4.6 with heavy `--n-cpu-moe`: 5 t/s local vs 2 t/s with RPC. The poster
  concludes RPC works when everything fits in VRAM and is poor when combined with CPU offload. Earlier
  posters (2024–25) point to sequential layer processing, not bandwidth, as the limit; 2024 inference
  traffic was only 30–50 Mb/s. — [Discussion #9136 (2024-08 to 2025-10)](https://github.com/ggml-org/llama.cpp/discussions/9136)
- jukofyork, three machines with two GPUs each, reported after #16276 that GLM-4.6 "works much better" over
  RPC, and that draft models help more because small batches are cheaper to send. —
  [PR #16276 discussion](https://github.com/ggml-org/llama.cpp/pull/16276)
- A third-party tool (SOLLOL) lists llama.cpp RPC as "5x slower than local, version-sensitive, manual
  setup" in its tests. This is a vendor claim with an unclear setup. — [SOLLOL on PyPI](https://pypi.org/project/sollol/)

**Mixing local and remote GPUs, and placing tensors**
- `-dev RPC0,CUDA0 -ts 6,4` mixes a remote and a local GPU. —
  [michaelspost.com](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- Explicit `--device` ordering was accepted in #16625, even though rgerganov said in the same thread that
  the order "cannot currently be overridden". The thread contradicts itself. —
  [Discussion #16625](https://github.com/ggml-org/llama.cpp/discussions/16625)
- `-ot/--override-tensor <regex>=<buffer type>` targets RPC buffers named `RPC[host:port]`, e.g.
  `-ot 'blk.1*=RPC[localhost:50052]'` (rgerganov, 2025-08-02). It failed in llama-bench at the time, but
  worked in llama-cli and llama-server.
  - Pinning attention to a remote desktop GPU over 1 Gbit "works but performs poorly": the link saturated
    during prompt processing (2025-08-05).
  - A later user says it "does work fine" (2025-11-18).

  — [Discussion #15020](https://github.com/ggml-org/llama.cpp/discussions/15020)
- Device names follow the global-counter scheme `RPC0`, `RPC1`, … (e.g. `RPC0: <host>:50052`). —
  [PR #16276](https://github.com/ggml-org/llama.cpp/pull/16276)

**Known limitations and bugs**
- The RPC server leaks CUDA graphs during inference until OOM. Workaround: `GGML_CUDA_DISABLE_GRAPHS=1` on
  the server. — [issue #20315](https://github.com/ggml-org/llama.cpp/issues/20315)
- `--split-mode tensor` causes CUDA lockups, so `-sm layer` stays pinned. Missing `-ngl 99` silently spills
  layers to CPU (10.4 vs 16.9 t/s). The RPC socket opens before CUDA is ready. —
  [michaelspost.com](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- Flash attention over RPC worked in that setup (`--flash-attn 1`, Ada and Blackwell workers). —
  [michaelspost.com](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- `ik_llama.cpp` lags upstream RPC: a Nov 2025 request asks it to port the upstream RPC changes. —
  [PR #16276 page, linked ik_llama.cpp #978](https://github.com/ggml-org/llama.cpp/pull/16276)

### Inferences
- **Disk.** Without `-c`, the rig needs no disk space for weights. With `-c`, it writes every weight tensor
  over 10 MiB, which is effectively its whole share of the model. A 45 GB model therefore needs about 45 GB
  of cache, more than the ~26 GB free on the rig's SSD. `LLAMA_CACHE` would have to point at a USB 3.0 drive or similar.
- **RAM.** The rig needs host RAM equal to the largest single tensor, because SET_TENSOR, and a cache hit
  via ifstream, buffer the whole tensor. Estimates from the configs below:

  | Model | Largest expert tensor |
  |---|---|
  | Qwen3-30B-A3B | 113 MB (Q4_K) to 165 MB (Q6_K) |
  | Qwen3-Next-80B-A3B | 302–440 MB |
  | GLM-4.5-Air | 415–606 MB |
  | Qwen3-235B-A22B | about 453 MB (Q4_K) |

  That fits in the rig's 3.7 GiB of usable RAM even with GNOME running (~1.1 GiB used at idle). Claim to
  test: watch RSS while loading.
- **Weak CPU.** In this topology, tokenization, sampling, scheduling, graph building and the token-embedding
  lookup run on the workstation. The rig's Celeron only receives TCP, deserializes commands and launches
  CUDA kernels, though kernel launch overhead stays on the rig's CPU as in standalone. Claim to test:
  whether rig CPU usage during decode drops compared with standalone.
- **One server vs nine.** One `rpc-server` exposing CUDA0–CUDA8 (after #16276) keeps GPU-to-GPU hops on the
  rig via COPY_TENSOR. Nine separate servers make the hidden state cross the network twice at every
  boundary, through the workstation. The one-server layout should be strictly better for decode (see §3).
- **Logits.** If the last device is an RPC device, about 0.6–0.8 MB of logits per token crosses the network
  (§3). Keeping the output layer on a local client device (the workstation GPU, or its CPU) avoids that. This is presumably
  why RPC devices are placed first by default.
- **No benchmark covers this setup.** None of the found numbers cover Pascal workers, a gigabit link with a
  45+ GB MoE, or nine GPUs behind one RPC endpoint. All of it must be measured.

### Gaps
- I couldn't find whether `rpc-server` reports remote op support to the client: whether the client's
  `supports_op` reflects Pascal limits, e.g. which FA kernels or quant types sm_61 supports.
- Issue #20315 matters for the rig only if CUDA graphs are active: current llama.cpp disables them at
  runtime on pre-Volta GPUs such as the P106-090 (claim C4 in the
  [report](../moe-inference-on-the-p106-rig.md)), so it applies only if graphs are re-enabled, e.g. with the
  PR #27721 patch.
- I couldn't find how often GRAPH_RECOMPUTE hits during decode, i.e. whether the graph stays unchanged
  token to token or is re-serialized when the KV size changes. No serialized-graph sizes were found.
- No source documents how many synchronous round trips per token remain after #18626 within one server.
- Whether model loading overlaps network receive with PCIe upload on the server is not documented. The code
  suggests it doesn't, per tensor.
- The COMM_ALLREDUCE commands suggest tensor-parallel support over RPC, but I found no documentation or
  benchmark.

---

## 2. Other systems for multi-machine / heterogeneous inference (status, backends, Pascal, parallelism, network, MoE)

### Takeaway
No surveyed system besides llama.cpp RPC (and things that wrap it, like LocalAI and GPUStack's llama-box)
combines CUDA on Pascal, GGUF quantization, MoE support, and workers that don't need a model file:
- **prima.cpp** is the most relevant research system, but lists no MoE models and expects a model copy on
  each device.
- **distributed-llama** has experimental Vulkan and Qwen3-30B-A3B, but caps nodes at the model's KV-head
  count (4 for Qwen3-30B-A3B).
- **exo, Parallax, vLLM, SGLang** are effectively excluded by MLX-CUDA or vLLM compute-capability floors.
- **Petals** is stale, and **Cake** lacks GGUF-style quantization.

### Cited Findings

| System | Status (as found) | GPU backends | Pascal sm_61? | Parallelism | Network / model distribution | MoE |
|---|---|---|---|---|---|---|
| **llama.cpp RPC** | PoC label; active development through Aug 2026 ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md), [unsloth #181](https://github.com/unslothai/llama.cpp/pull/181)) | Any ggml backend on the server (CUDA, Metal, Vulkan, CPU) | Yes; llama.cpp runs on the rig ([single-card calibration](../../benchmarks/2026-10-22-single-card-calibration.md)) | Layer split, `-ot` per-tensor placement; allreduce commands present ([ggml-rpc.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp)) | TCP or RDMA; client streams weights; optional `-c` cache | Yes, any GGUF MoE ([#16625](https://github.com/ggml-org/llama.cpp/discussions/16625), [michaelspost](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)) |
| **prima.cpp** | ICLR 2026 paper ([ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e991e5587c1daa49bbf9a818b3f02f9a-Abstract-Conference.html)); repo moved to Gitee, no releases, Docker tag 1.0.2-cuda ([Gitee](https://gitee.com/zonghang-li/prima.cpp)); GitHub URL now 404 | CUDA, Metal, CPU; Vulkan and ROCm "not yet" ([arXiv v2](https://arxiv.org/html/2504.08791v2), [Gitee](https://gitee.com/zonghang-li/prima.cpp)) | Not discussed; tested on 2080 Ti and 3070 ([arXiv v2](https://arxiv.org/html/2504.08791v2)) | Pipelined-ring: ring of devices, several rounds per token, disk prefetch overlapped with compute; Halda scheduler | ZeroMQ; ports 9000/10000; `--world/--rank/--master/--next`; each device appears to need a local model copy (inferred from README) ([Gitee](https://gitee.com/zonghang-li/prima.cpp)) | None listed (Llama, Qwen2.5, QwQ, R1 distills) ([Gitee](https://gitee.com/zonghang-li/prima.cpp)) |
| **distributed-llama (b4rtaz)** | Last news 2025-09-16; last named releases v0.13.0 (2025-03-23) ([GitHub](https://github.com/b4rtaz/distributed-llama)) | CPU (ARM, x86-64 AVX2 optimized); Vulkan experimental since v0.13.0 (`DLLAMA_VULKAN=1`) ([GitHub](https://github.com/b4rtaz/distributed-llama), [README via jsDelivr](https://cdn.jsdelivr.net/gh/b4rtaz/distributed-llama@main/README.md)) | Untested; Vulkan on Pascal is a claim to test | Tensor parallel; 2^n nodes; max nodes = number of KV heads | Root holds the model; Ethernet sync; only q40 + q80 buffer (or f32) | Qwen3-30B-A3B (CPU 2025-09-05, Vulkan 2025-09-16) ([GitHub](https://github.com/b4rtaz/distributed-llama)) |
| **exo** | 48k stars, no releases listed; Linux "currently runs on CPU, GPU support under development" ([GitHub](https://github.com/exo-explore/exo)); Linux CUDA tracked in #913 ([issue #861](https://github.com/exo-explore/exo/issues/861)) | MLX (Apple GPU); `mlx-cuda12/13` extras on Linux ([GitHub](https://github.com/exo-explore/exo)) | No: MLX CUDA needs CC ≥ 7.0 or SM ≥ 7.5, sources conflict ([mintlify MLX mirror](https://www.mintlify.com/ml-explore/mlx/installation)) | Topology-aware tensor and pipeline sharding; RDMA over TB5 | Zenoh discovery; `--no-worker` orchestration-only node; port 52415 | Large MoEs (DeepSeek v3.1, Kimi K2) on Mac Studios ([GitHub](https://github.com/exo-explore/exo)) |
| **Gradient Parallax** | v0.0.1 Oct 2025; OpenClaw integration Feb 2026 ([GitHub](https://github.com/GradientHQ/parallax)) | SGLang and vLLM (GPU), MLX LM (Mac) | Effectively no: vLLM docs require CC ≥ 7.0 (older docs) or 7.5 (current) ([vLLM docs](https://docs.vllm.ai/getting_started/installation/gpu.html), [v0.10.0](https://docs.vllm.ai/en/v0.10.0/getting_started/installation/gpu.html)) | Pipeline-parallel sharding, dynamic routing | Lattica P2P; nodes may differ in location | Many large MoEs (DeepSeek V3.2, GLM 4.7–5.2, gpt-oss-120b, Qwen3.6-35B-A3B…) |
| **Petals** | 522 commits, no releases listed; no 2025–26 activity found ([GitHub](https://github.com/bigscience-workshop/petals)) | PyTorch CUDA (install cmd uses cuda 11.7), AMD, Apple | Unknown | Pipeline chain of servers each holding blocks | Public or private swarm, port 31330 | Mixtral 8x22B listed |
| **Cake (evilsocket)** | "experimental … changed very quickly"; FAIR licence ([GitHub](https://github.com/evilsocket/cake)) | CUDA, Metal, Vulkan, Accelerate, CPU | CC requirement not stated | Transformer blocks sharded by VRAM/compute | Master streams weights to workers (zstd + CRC32, cached on worker); mDNS; `--cluster-key` | Not mentioned; no general quantization described |
| **LocalAI P2P** | Docs call it experimental / tech preview ([localai.io](https://localai.io/features/distribute/)) | Whatever llama.cpp supports | Same as llama.cpp, inferred | Worker mode = llama.cpp RPC sharding (built on llama.cpp PR #6829); federated mode routes whole requests to one node | Token-joined P2P (mDNS on LAN or DHT); or `LLAMACPP_GRPC_SERVERS` list | Via llama.cpp |
| **GPUStack** | v2.2.x (mirror updated 2026-06-18) has an "Allow Distributed Inference Across Workers" option ([SourceForge mirror](https://sourceforge.net/projects/gpustack.mirror/files/v2.2.0/)) | vLLM, SGLang; llama-box (llama.cpp-based) offered RPC mode ([sourcepulse](https://www.sourcepulse.org/projects/1840028), [PyPI](https://pypi.org/project/gpustack)) | vLLM/SGLang no; llama-box presumably yes | Cluster scheduler and RPC | Server and workers | Via backends |
| **Ollama** | No built-in multi-machine split of one model, per guides ([dev.to](https://dev.to/runcai/ollama-distributed-inference-what-is-possible-what-is-not-and-when-to-move-beyond-local-serving-2ajl)) | n/a | n/a | Multiple instances only scale concurrency | Ollama GGUFs may not work with llama.cpp RPC ([kjaiswal via search](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)) | n/a |
| **LM Studio LM Link** | Preview since v0.4.6 (2026-02-27), with Tailscale ([LM Studio docs](https://lmstudio.ai/docs/lmlink), [Tailscale blog](https://tailscale.com/blog/lm-link-remote-llm-access)) | n/a | n/a | Remote access to a model running entirely on another machine, not a split | E2E encrypted Tailscale mesh | n/a |
| **vLLM / SGLang multi-node** | Mainstream | CUDA etc. | No (CC ≥ 7.0/7.5); unofficial `vllm-pascal` image exists ([Docker Hub](https://hub.docker.com/r/johnnykings/vllm-pascal)) | TP/PP/EP over Ray | Needs fast interconnect | Yes |

Additional cited details:
- **prima.cpp mechanics.**
  - Each layer window sends one FP32 vector of the embedding size e (4e bytes per hop), with about L/W
    hops per token.
  - mmap with OS page cache, plus prefetch to resolve the "prefetch-release conflict".
  - Testbed: Wi-Fi at 320–610 Mbps and 3–7 ms latency.
  - Results: 70B Q4K at 674 ms/token, vs llama.cpp single device at 10,120 ms. exo and dllama ran out of
    memory on 70B; 8B: exo 263 ms, dllama 459 ms, prima 15 ms. 26 t/s for 32B with speculative decoding.
  - No comparison against llama.cpp RPC.

  — [arXiv 2504.08791v2](https://arxiv.org/html/2504.08791v2);
  [arXiv listing](https://arxiv.org/abs/2504.08791)
- **prima.cpp build and placement.** Output layer runs on CPU unless `--keep-out-in-cuda` is set. Build
  needs ZeroMQ ≥ 4.3.2, HiGHS ≥ 1.9.0 (rank 0) and fio. Multiple GPUs per device are not addressed. —
  [Gitee README](https://gitee.com/zonghang-li/prima.cpp)
- **Tinygrad in exo.** A community issue reports tinygrad was removed from upstream exo in the v1 prep,
  and an "exo-cuda" fork claims V100/M40 operation. Third-party claim, not verified; the issue page
  returned 404 when fetched. — [search result summary of exo issue #1039](https://github.com/exo-explore/exo/issues/1039)
- **mlx.distributed and CUDA.** As of Aug 2025, mlx.distributed was not integrated with mlx-cuda. —
  [exo issue #861](https://github.com/exo-explore/exo/issues/861)
- **Tensor parallelism in llama.cpp itself.** "ggml: backend-agnostic tensor parallelism (experimental)",
  PR #19378, merged 2026-04-09, enabled with `--split-mode tensor`. Reports: tg nearly doubled on a 3-GPU
  rig; pp slows; a ROCm MoE crash. — [ai-muninn blog](https://ai-muninn.com/en/blog/qwen38-dual-2080ti-tensor-parallel);
  [issue #22307](https://github.com/ggml-org/llama.cpp/issues/22307);
  [piefed thread](https://piefed.jeena.net/c/localllama@sh.itjust.works/p/372925/llama-cpp-don-t-sleep-on-split-mode-tensor)
- **Research pointers (titles only, not read).** Tessera (kernel-granularity disaggregation across
  heterogeneous GPUs, [arXiv 2604.10180](https://arxiv.org/pdf/2604.10180)) and HeteroMosaic (edge
  heterogeneous execution, [arXiv 2607.12839](https://arxiv.org/pdf/2607.12839)). llm-d is
  Kubernetes/vLLM-centric and not aimed at home rigs ([Red Hat, 2025-11-21](https://developers.redhat.com/articles/2025/11/21/introduction-distributed-inference-llm-d)).

### Inferences
- **distributed-llama.** The KV-head cap means Qwen3-30B-A3B (4 KV heads per its
  [config](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)) can use at most 4 nodes, so at
  most 4 of the 9 GPUs. Its tensor parallelism syncs several times per layer, so latency per token grows
  with layers × RTT (see §3). Its x86 path is AVX2-optimized, and the rig's Celeron has no AVX; with
  Vulkan the CPU does less, but this is untested.
- **prima.cpp.** It targets several hosts, each with its own disk and RAM. Its mmap-and-prefetch design
  assumes the model file is on each device, which collides with the rig's ~26 GB of free disk. Its benefit
  (overlap of disk streaming) matters when models exceed memory, not when everything fits in the rig's
  54 GB plus a workstation GPU's 8 GB of VRAM.
  With two hosts (workstation + rig) it would be a 2-node ring. Multi-GPU per rank, i.e. using nine GPUs
  in one rig process, is undocumented.
- **Cake.** Workers don't need model files, which matches the rig's disk limit. But without
  GGUF/K-quant-style quantization, weights would likely be F16/BF16. That is half the capacity per GB of
  Q8, and Pascal's 1/64 FP16 rate is a problem. Unverified.
- **exo, Parallax, vLLM, SGLang** cannot use sm_61 through official builds. They are only relevant as
  context, or if the workstation alone serves.
- **LocalAI and GPUStack** do not add new parallelism over llama.cpp RPC. They add discovery, encryption
  (P2P mode) and management. LocalAI's P2P tunnel may help with the no-auth problem of raw rpc-server
  (unverified; see §5).

### Gaps
- I couldn't confirm whether prima.cpp's current Gitee version supports Qwen3 MoE, multi-GPU per rank, or
  loading without a local model file.
- I couldn't find any distributed-llama Vulkan result on NVIDIA, nor whether Vulkan works on Pascal with
  driver 580. No dllama 2026 release info was found.
- I couldn't confirm exo's current Linux NVIDIA status beyond "CPU for now", or whether tinygrad was
  removed (that source was 404).
- GPUStack v2's current llama-box/RPC status is unknown.
- Petals: no maintenance status found. Pascal support of current PyTorch CUDA wheels was not checked.
- No measured comparison of these systems against llama.cpp RPC on the same hardware was found, other
  than prima.cpp's paper (which skips RPC).

---

## 3. Network cost model (layer/pipeline split vs expert-parallel) over gigabit

### Takeaway
For layer (pipeline) splits, per-token payloads are tiny: 8–28 KiB per boundary for hidden sizes
2048–7168. Over gigabit, round-trip latency dominates, not bandwidth.
- With one or two network crossings per token, the overhead is about 0.3–4 ms. That is roughly 2–25 % of a
  15–20 ms P106 token.
- Shipping full logits over the network costs another 5–7 ms per token.
- Any scheme crossing the network every layer costs 25–250 ms per token at 0.2–2 ms RTT, and loses to
  running on the rig alone. That covers expert parallelism, attention-on-workstation with experts-on-rig,
  and tensor parallelism.
- Break-even needs per-crossing latency well below the per-layer compute time (about 0.3 ms per layer here).

### Cited Findings
- Only the hidden state crosses the network after each layer (ggerganov, 2024-08-23). —
  [Discussion #9136](https://github.com/ggml-org/llama.cpp/discussions/9136)
- prima.cpp: 4e bytes (FP32) per hop. — [arXiv v2](https://arxiv.org/html/2504.08791v2)
- A practitioner measures about 10 KB of activations per token on 1 GbE with 0.29 ms RTT. —
  [michaelspost.com](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- About 0.17 ms round trip per token on 10 GbE; iperf 9.41 of 10 Gbps, i.e. about 94 % TCP efficiency. —
  [kjaiswal](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)
- Model shapes from Hugging Face configs:

  | Model | hidden | layers | experts / active | vocab | KV heads |
  |---|---|---|---|---|---|
  | [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json) | 2048 | 48 | 128 / 8 (moe_intermediate 768) | 151,936 | 4 |
  | [Qwen3-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json) | 2048 | 48 | 512 / 10 (+shared) | 151,936 | n/a |
  | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json) | 2880 | 36 | 128 / 4 | 201,088 | 8 |
  | [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json) | 4096 | 46 | 128 / 8 (+1 shared, first layer dense) | 151,552 | 8 |
  | [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json) | 4096 | 94 | 128 / 8 | 151,936 | 4 |

  Qwen3-Next uses full attention every 4th layer.
- Logits stay off the network only if an RPC device is not last in the chain. —
  [Discussion #16625](https://github.com/ggml-org/llama.cpp/discussions/16625)
- Per-command blocking was the main RPC latency source in Oct 2025 (slaren). —
  [PR #16276](https://github.com/ggml-org/llama.cpp/pull/16276)
- Async commands landed in Aug 2026, but gave no tg gain in the one published two-node measurement. —
  [unsloth PR #181](https://github.com/unslothai/llama.cpp/pull/181)

### Inferences (my calculations)
Assumptions:
- Gigabit TCP goodput is 117 MB/s (941 Mbit/s, the same ~94 % efficiency as the 10 GbE iperf above).
- Activations are FP32.
- A "crossing" is synchronous: RTT + payload/bandwidth.
- Rig decode compute for Qwen3-30B-A3B is about 3.0 B weights read per token (from the config: experts
  8×3×2048×768 + attention ≈ 57 M per layer × 48 layers + 311 M output head). At Q4_K_M (~4.85 bpw) that
  is about 1.84 GB per token, so ≈ 10.7 ms at 173 GB/s peak, or about 15–18 ms at 60–70 % efficiency.
  Layer split is sequential, so a single card's bandwidth applies.

**Per-boundary decode payload and cost over gigabit**

| hidden | payload / token | serialization | crossing @0.2 ms RTT | crossing @2 ms RTT | pp ubatch 512: payload / time |
|---|---|---|---|---|---|
| 2048 (Qwen3-30B-A3B, Qwen3-Next) | 8 KiB | 0.07 ms | 0.27 ms | 2.07 ms | 4 MiB / 36 ms |
| 2880 (gpt-oss) | 11.25 KiB | 0.10 ms | 0.30 ms | 2.10 ms | 5.6 MiB / 50 ms |
| 4096 (GLM-4.5-Air, Qwen3-235B) | 16 KiB | 0.14 ms | 0.34 ms | 2.14 ms | 8 MiB / 72 ms |
| 5120 (hypothetical) | 20 KiB | 0.18 ms | 0.38 ms | 2.18 ms | 10 MiB / 90 ms |
| 7168 (hypothetical) | 28 KiB | 0.25 ms | 0.45 ms | 2.25 ms | 14 MiB / 126 ms |

**Logits, if the output layer sits on a remote device:** 151,936 × 4 B = 594 KiB ≈ 5.2 ms per token, or
201,088 × 4 B = 786 KiB ≈ 6.9 ms (gpt-oss). For comparison, standalone on the rig the same logits cross
PCIe 1.0 x1 (~190 MB/s) in about 3.2–4.2 ms.

**Per-token network overhead by topology (decode, Qwen3-30B-A3B shapes)**
- **T1: one rpc-server, all 9 GPUs, output layer on the client** (workstation GPU or CPU). About 1 crossing in and 1
  out, ≈ 0.3–4 ms plus command overhead. That is about 2–25 % of a 15–18 ms token.
- **T2: as T1, output layer on a rig GPU.** Adds 5.2 ms, so about 5.5–9 ms (30–50 % overhead).
- **T3: nine rpc-servers, one per GPU.** The hidden state goes through the client at each of 8 boundaries
  (get + set). That is 8–16 extra crossings, about 2.2–4.3 ms at 0.2 ms RTT, or 17–33 ms at 2 ms RTT.
- **T4: attention, dense and router on the workstation GPU, routed experts on rig GPUs (`-ot exps=RPC…`).** Two
  crossings per MoE layer: 48 × 2 × (0.27–2.07 ms) ≈ 26–199 ms per token, i.e. 2–13× the whole compute
  time. For prefill, 96 crossings × 36 ms ≈ 3.4 s per 512-token ubatch, which caps pp at about 150 t/s
  from the network alone. That matches the observed 1 GbE saturation in
  [Discussion #15020](https://github.com/ggml-org/llama.cpp/discussions/15020).
- **T5: expert parallelism across hosts** (all-to-all, dispatch + combine per layer). Payload per phase is
  up to k·h·4 B = 64 KiB (Qwen3-30B-A3B, k=8), i.e. 0.56 ms + RTT.
  - Qwen3-30B-A3B: 2 phases × 48 layers ≈ 73–246 ms per token.
  - GLM-4.5-Air (k=8, h=4096): about 122–287 ms per token.
  - Using bf16 instead of FP32 halves the payload but not the RTT term.
- **Tensor parallelism across hosts** (dllama style, at least 2 syncs per layer): latency-only floor of
  48 × 2 × RTT ≈ 19–192 ms per token.

**Break-even**
- Per-layer compute on one P106 for Qwen3-30B-A3B is ≈ 15–18 ms / 48 ≈ 0.3–0.4 ms. A per-layer network
  scheme only breaks even if a full crossing (RTT + payload) is well under that, i.e. RTT ≪ 0.1 ms. That is
  RDMA or Thunderbolt territory, not gigabit TCP over a Realtek NIC (or a WireGuard-based mesh VPN).
- Whole-model layer splits (T1) break even easily on bandwidth. The question is only whether the
  constant 0.3–4 ms is worth paying for the capacity or CPU offload gained.
- The 0.2 ms vs 2 ms RTT spread is the biggest swing factor. Measured RTT on plain 1 GbE was 0.29 ms
  ([michaelspost](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)), while ~2 ms was
  measured on the rig over a WireGuard-based mesh VPN. Using the plain LAN path instead of a VPN overlay
  could matter as much as anything else.
- Speculative decoding or MTP amortizes per-forward-pass round trips over several accepted tokens, which is
  why draft models help more over RPC
  ([PR #16276](https://github.com/ggml-org/llama.cpp/pull/16276),
  [michaelspost: MTP about 2.1× on 27B](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)).

**On the rig's internal hops, for comparison.** No P2P, PCIe 1.0 x1 ≈ 190 MB/s. An 8 KiB hidden state per
GPU-to-GPU hop through host memory takes about 43 µs plus sync, so 8 hops ≈ 0.35 ms plus launch and sync
costs. For prefill, 4 MiB per hop ≈ 22 ms, so 8 hops ≈ 176 ms per 512-token ubatch, a cap of about
2,900 t/s, well above what P106s can compute. Internal layer-split hops are cheap compared with a network
round trip at 2 ms RTT.

### Gaps
- No measured per-token RPC command count or bytes for llama.cpp after #18626/#22701. `GGML_RPC_DEBUG=3`
  on this setup would give it.
- The real Celeron software overhead per RPC command (TCP stack, deserialization, sync_all_backends) is
  unknown, and could add a fixed cost per command beyond the wire RTT.
- No measured P106-090 decode numbers are cited here. The 15–18 ms figure is a bandwidth-based estimate.

---

## 4. Scenario analysis

### Takeaway
- **(a) Workstation as RPC client and host of the model file.** Removes the rig's disk, page-cache and most CPU limits.
  Costs a 6–10 minute load over gigabit per (re)load of a 45 GB model, a constant per-token network tax,
  and requires the workstation to be on during inference.
- **(b) Rig standalone.** Has no network tax, but is bounded by ~26 GB of free disk, 3.7 GiB of usable RAM
  and Celeron-side work.
- **(c) Workstation GPU + rig GPUs together.** With an 8 GB workstation GPU, gives ~62 GB of VRAM. It is
  viable only as a whole-layer split (2 crossings per token) or with speculative decoding.
  Attention-on-workstation-GPU / experts-on-rig is predicted to be much slower.
- **(d) Loading from a network share onto the standalone rig.** Keeps runtime fully local, with the same
  gigabit load cost as (a). Use `--no-mmap` if the share must be disconnectable after load.

### Cited Findings
- Only the host needs model files; RPC devices receive weights over the network. —
  [kjaiswal](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks);
  [michaelspost](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- Server-side whole-tensor buffering and the 10 MiB cache threshold. —
  [ggml-rpc.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp)
- The `-c` cache lets restarts skip the transfer. —
  [michaelspost](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- Remote and local devices can be mixed with `-dev RPC0,CUDA0 -ts …`; device order affects pp and tg a lot
  (RPC first ≈ 16 tg vs middle ≈ 22.5 tg in one GLM-4.6 test). —
  [michaelspost](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html);
  [Discussion #16625](https://github.com/ggml-org/llama.cpp/discussions/16625)
- `-ot` to RPC buffer types works in llama-cli and llama-server. Attention on the remote GPU saturated 1 GbE
  in prefill. — [Discussion #15020](https://github.com/ggml-org/llama.cpp/discussions/15020)
- RPC is good when everything fits in VRAM, and poor with CPU offload. —
  [Discussion #9136](https://github.com/ggml-org/llama.cpp/discussions/9136)
- Decode gets slower when splitting a model that fits locally (-47 % for 72B on 10 GbE). —
  [kjaiswal](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)

### Inferences
**(a) Workstation = client holding the GGUF; rig = one `rpc-server` exposing CUDA0–CUDA8**
- Limits removed:
  - Rig disk: none needed without `-c`.
  - Rig RAM for mmap/page cache: none, apart from per-tensor staging of about 0.1–0.6 GB peak (§1).
  - Rig CPU for tokenization, sampling over a ~152k vocab, scheduling and embedding lookup: moved to the
    workstation CPU.
  - The token-embedding table stays in workstation RAM.
- Limits that remain:
  - CUDA kernel launches and synchronization still run on the Celeron.
  - The KV cache lives on rig GPUs.
  - On the tested desktop install, Xorg's contexts still occupy some VRAM on all nine GPUs (measured on the
    rig).
- Load time for 45 GB:

  | Path | Rate | Time |
  |---|---|---|
  | Pure gigabit | 117 MB/s | ≈ 6.4 min |
  | Network receive and PCIe 1.0 x1 upload serialized per tensor | ≈ 72 MB/s | ≈ 10.4 min |
  | Later loads from a `-c` cache on a USB 3.0 SSD (~400 MB/s read, serialized with ~190 MB/s PCIe) | ≈ 129 MB/s | ≈ 5.8 min |
  | PCIe-bound floor | 190 MB/s | ≈ 3.9 min |

  The internal SSD cannot hold a full 45 GB cache (~26 GB free).
- Per-token tax: T1 ≈ 0.3–4 ms if the output layer is on the workstation; add about 5 ms if it is on the
  rig (§3).
- Ops costs: the workstation must stay up and on the same llama.cpp commit; the rig must expose an
  unauthenticated port (§5).
- This is the topology most likely to run models that do not fit the rig's disk, e.g. a 45–50 GB
  Qwen3-Next-80B-A3B Q4 quant.

**(b) Rig standalone**
- No network cost.
- Model size on the internal disk is limited to about 24 GB (e.g. Qwen3-30B-A3B Q4_K_M at about 18 GB is
  fine) unless an external USB 3.0 disk is used.
- mmap loading with 4 GB RAM streams through the page cache. It works in principle, but the CPU-resident
  tensors (token embeddings) stay mmapped.
- Celeron-side work (sampling a ~152k vocab, graph building) and logits over PCIe 1.0 x1 (≈ 3 ms per token)
  add per-token overhead. The size of that overhead is a claim to test.
- The single-host details are covered in the [multi-GPU note](multigpu_pascal_performance.md) and the
  [ops note](ops_and_benchmarking.md). This note only supplies the comparison.

**(c) Workstation GPU (8 GB assumed) + rig GPUs (54 GB)**
- **c1: whole-layer split.** RPC devices first (rig layers), then the final layers and output head on the
  workstation GPU. Two crossings per token, and logits stay on the workstation. This adds about 8 GB of
  capacity (62 GB total minus Xorg/compute buffers), enough to push some 55–60 GB quants into VRAM; the fit
  is a claim to test. The layers on the workstation GPU run faster than on a P106.
- **c2: attention/dense/shared expert on the workstation GPU, routed experts on the rig.** Per-layer crossings are
  predicted to cost 26–199 ms per token in decode and to cap prefill at about 150 t/s (§3). This is
  expected to lose to (b) on any model that fits the rig alone. It is still worth one measurement to
  validate the cost model.
- **c3: speculative decoding.** A small draft model, or MTP heads, on the workstation GPU or rig, with the
  target split as in c1. The per-pass network tax is amortized over accepted tokens.
- Any (c) variant makes the workstation a permanent participant. That conflicts with a goal of running the
  rig standalone, but suits experiments.

**(d) Loading from a network share (NFS/SMB/NBD/SSHFS) onto the standalone rig**
- Load time: like (a), at about 6.4–7.5 min for 45 GB at gigabit line rate.
- After load, inference is fully local with zero per-token network cost.
- With mmap, CPU-resident tensors such as token embeddings remain backed by the remote file. If the page
  cache evicts them (likely with 4 GB RAM), they are re-read over the network, so the share must stay
  mounted. `--no-mmap` should copy them into anonymous RAM so the share can go away after load. Claim to
  test.
- NBD/iSCSI gives a block device with the kernel's own caching. NFS is simplest. SSHFS adds encryption on
  a 2-core Celeron and may not reach line rate (claim to test).
- Copying once to a USB 3.0 SSD is the alternative: faster reloads, and it does not need the workstation at
  all.
- (d) fits best if the goal is to use the workstation only as a model store and build box.

### Gaps
- No measurement of llama.cpp memory behaviour with full GPU offload on a 4 GB host is cited here: page
  cache, `--no-mmap` staging-buffer sizes, which tensors stay CPU-side.
- No source on the Celeron 3865U's TCP throughput or WireGuard throughput. The Celeron does have AES-NI
  (measured on the rig).
- How llama.cpp places `output.weight` when only RPC devices plus CPU are listed was not confirmed.

---

## 5. Security and operations notes

### Takeaway
`rpc-server` has no authentication or encryption, and has had memory-safety bugs exploitable for remote
code execution (RCE) in 2026. Treat its port as equivalent to remote code execution for anyone who can
reach it:
- Bind it to a specific LAN or VPN address, never `0.0.0.0`.
- Firewall it to the workstation's IP.
- Run it as an unprivileged user, not root.
- Or keep it on loopback behind an SSH tunnel, if the Celeron can handle the encryption cost.

### Cited Findings
- README: "fragile and insecure … Never run the RPC server on an open network". —
  [README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- `rpc-server` warns when bound to anything other than `127.0.0.1`, but still starts. —
  [rpc-server.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/rpc-server.cpp)
- PR #20908, "RPC GRAPH_COMPUTE Arbitrary RCE patch" (merged 2026-03-23, a GHSA fix), fixed a
  `deserialize_tensor()` validation bypass. Later issues report:
  - an unauthenticated remote DoS (#25288);
  - a remote controlled indirect call via a deserialized function pointer (#25289);
  - wrong affected-version ranges in advisories (#30786, open).

  — [PR #20908](https://github.com/ggml-org/llama.cpp/pull/20908)
- A further RPC issue: PAD_REFLECT_1D can write past the destination tensor in release builds. —
  [issue #29914](https://github.com/ggml-org/llama.cpp/issues/29914)
- A practitioner's pattern:
  - `-H 127.0.0.1` on the worker plus an `ssh -L` tunnel.
  - Note that Docker `-p` port mappings bypass host INPUT firewall rules (so `--network host`).
  - `llama-server` also has no auth.
  - Use `ssh -O exit` for teardown.

  — [michaelspost.com](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)
- LocalAI's P2P mode joins workers by token over libp2p (mDNS on LAN, or DHT). —
  [LocalAI docs](https://localai.io/features/distribute/)
- LM Studio LM Link uses an end-to-end encrypted Tailscale mesh, but only for whole-model remote access. —
  [LM Studio docs](https://lmstudio.ai/docs/lmlink)

### Inferences
- If the rig is administered over SSH as root, an `rpc-server` started that way would expose a root RCE
  surface. Running it as the regular desktop user, or a dedicated user with access only to `/dev/nvidia*`,
  under systemd sandboxing is the obvious mitigation.
- Binding to the address of a WireGuard-based mesh VPN gives encryption and ACLs for free. But the ~2 ms
  RTT measured over such a VPN on the rig, and possible WireGuard CPU cost on the Celeron during a 45 GB
  load, cut against it. Binding to the plain LAN IP with an nftables allow-rule for the workstation's IP is
  faster but unencrypted.
- RPC_PROTO lockstep means both ends need rebuilding together on every update. Mismatches can look like
  network errors.
- Starting a long-running listening service on the rig changes its state and opens an attack surface, so
  it warrants a deliberate decision rather than a routine test run.

### Gaps
- I couldn't confirm whether the fixes for #25288 and #25289 have been merged, or their CVE/GHSA IDs.
- No source on the encryption throughput of WireGuard, SSH or SSHFS on a Celeron 3865U.

---

## Experiment ideas

### Takeaway
These are options, not decisions. They are ordered roughly from cheap and low-risk to expensive. Every
number in §1–§4 that wasn't measured on this hardware is listed under "Claims to test". As of 2026-10-22,
none of these experiments had been run.

### Claims to test
1. One `rpc-server` can expose all nine P106-090s, and the workstation client sees RPC0–RPC8 with sane
   free-memory numbers (Xorg contexts subtract some).
2. Rig RAM peak during RPC load ≈ largest tensor (≈ 0.1–0.6 GB). Loading never needs the GGUF or page
   cache on the rig.
3. A cold 45 GB load over plain gigabit takes 6.4–10.4 min, depending on whether network receive and
   PCIe 1.0 x1 upload overlap.
4. Plain LAN RTT is ≈ 0.2–0.3 ms vs ≈ 2 ms over a WireGuard-based mesh VPN, and VPN throughput on the
   Celeron may fall below line rate.
5. Decode tax for topology T1 (one server, output on the client) is ≤ ~25 % vs standalone on the same
   model. With output on the rig, an extra ~5 ms per token.
6. Nine separate rpc-servers (T3) decode measurably slower than one server (T1).
7. Attention-on-workstation-GPU / experts-on-rig (T4) is several times slower in decode, and pp is capped at about
   150 t/s by 1 GbE.
8. `-ot …=RPC[...]` / RPC device ordering works on the current build. Which form is current, `RPC[host:port]`
   or `RPCn`, is unknown.
9. Flash attention, the quant types used and CUDA graphs behave the same through RPC on Pascal as locally.
   Issue #20315 (`GGML_CUDA_DISABLE_GRAPHS=1`) may or may not apply.
10. Speculative decoding / MTP gives a larger relative speedup over RPC than standalone.
11. `--no-mmap` loading from NFS lets the share be unmounted after load. With mmap, the token embeddings
    keep generating NFS reads.
12. Distributed-llama Vulkan runs on Pascal with driver 580. If it does, it is limited to 4 nodes for
    Qwen3-30B-A3B.

### Experiments

**E1: network baseline.** Low effort, no risk.
- Topology: workstation ↔ rig over the plain LAN address and, if one is used, over the VPN overlay.
- Method: `ping`, `iperf3` in both directions, single and parallel streams; record rig CPU (`mpstat`)
  during iperf.
- Measure: RTT p50/p99, Mbit/s, Celeron CPU %.
- Expected: about 940 Mbit/s on LAN; the VPN possibly lower with high CPU; RTT about 0.2–0.4 ms on LAN.
- This drives every number in §3 and feeds claims 3–4.

**E2: matched RPC build and device enumeration.** Low effort. Risk: version mismatch looks like a network
fault.
- Method: build one llama.cpp commit with `-DGGML_RPC=ON`: the workstation with its own toolkit for its
  GPU (e.g. CUDA 13.x for sm_120); the rig with CUDA 12.9, `CMAKE_CUDA_ARCHITECTURES=61`. A CUDA 13 toolkit
  can't build for Pascal, so the rig binary can be built on the workstation in the
  `nvidia/cuda:12.9.2-devel-ubuntu24.04` container and copied over.
- Run `rpc-server -H <LAN-IP> -p 50052` as a non-root user on the rig, firewalled to the workstation IP.
  On the client, run `llama-cli --rpc <rig>:50052 --list-devices`.
- Measure: device list, reported free VRAM, RPC_PROTO in the handshake.
- Expected: RPC0–RPC8 visible (claim 1).
- Running a listener on the rig changes its state and warrants a deliberate decision.

**E3: load path and memory.** Medium effort, low risk.
- Model: Qwen3-30B-A3B Q4_K_M (about 18 GB, also fits the rig disk), then a ~45 GB MoE quant such as a
  Qwen3-Next-80B-A3B or GLM-4.5-Air low-bit quant.
- Method: load over RPC with `-ngl 99`, without `-c`, then with `-c` (`LLAMA_CACHE` on a USB 3.0 SSD on the
  rig).
- Measure: wall-clock load time, NIC throughput, rig RSS peak (`/proc/<pid>/status` VmHWM), PCIe upload
  rate, disk written.
- Expected: claims 2–3. With `-c`, a fast second load and cache size ≈ the model.

**E4: standalone vs RPC on the same model.** Medium effort, low risk.
- Model: Qwen3-30B-A3B Q4_K_M, which fits both ways.
- Arms:
  - A: rig standalone `llama-bench` (pp512, tg128).
  - B: workstation client, `-dev RPC0..RPC8` only, output on rig.
  - C: same as B with the output layer on the workstation (its GPU or CPU, via device order or `-ot
    output=…`).
- Add `GGML_RPC_DEBUG=3`, `-lv 5` to count commands and bytes per token.
- Measure: t/s, ms per token, bytes per token, rig CPU %.
- Expected: B slower than A by a few ms per token; C between A and B. Tests claims 5 and 9.

**E5: one server vs nine servers.** Low effort once E4 works.
- Method: start nine `rpc-server` instances (`--device CUDAi`, ports 50052–50060) vs one server; repeat E4-B.
- Expected: nine servers slower in tg because of per-boundary bounces through the client (claim 6). Also
  confirms the #26500 multi-server fix.

**E6: workstation GPU + rig whole-layer split (c1).** Medium effort.
- Model: one that fits in 54 GB, for an apples-to-apples comparison with E4, plus one that needs 55–60 GB
  (fit to be checked; e.g. a larger quant of the same family).
- Method: `-dev RPC0,…,RPC8,CUDA0 -ts …` with the last layers and output on the workstation GPU. `-sm layer`,
  `-ngl 99`.
- Measure: t/s, VRAM per device, crossings per token.
- Expected: small tax vs rig-only on the fitting model; gains capacity.

**E7: attention-on-workstation-GPU / experts-on-rig (c2), as a cost-model check.** Low effort, expected
negative.
- Method: put all non-expert tensors on CUDA0 (the workstation GPU) and `-ot "ffn_.*_exps=RPC[...]"` spread across rig
  devices.
- Measure: tg, pp, NIC utilization, crossings per token.
- Expected: tg far below E4-A; pp NIC-bound at about 150 t/s (claim 7). If the result is much better than
  predicted, the cost model in §3 needs revising.

**E8: speculative decoding / MTP over RPC (c3).** Medium effort.
- Method: target split as in E4-B or E6; a draft model (small same-family dense, or a model with built-in
  MTP) on the workstation GPU or a rig GPU.
- Measure: acceptance rate and tg, against the same configuration on standalone.
- Expected: larger relative gain over RPC (claim 10).

**E9: standalone rig with a network-share model store (d).** Medium effort. Low risk if mounted read-only.
- Method: export the model directory from the workstation over NFS (and optionally NBD or SSHFS); load on
  the rig with mmap vs `--no-mmap`. After load, watch `nfsstat`/`iftop` during generation, then try
  unmounting.
- Measure: load time, RAM, runtime NFS traffic.
- Expected: about 6.5–7.5 min for 45 GB; with mmap, some runtime reads; with `--no-mmap`, none (claim 11).
- Compare with a one-time copy to a USB 3.0 SSD on the rig.

**E10: rig-internal alternatives the RPC path touches.** Higher risk, experimental.
- Option: try `--split-mode tensor` on the rig standalone, and through RPC if the COMM_ALLREDUCE path is
  live. Expect allreduce over PCIe 1.0 x1 without P2P to be costly. Upstream reports lockups and MoE
  crashes.
- Option: distributed-llama with `DLLAMA_VULKAN=1` on up to 4 GPUs for Qwen3-30B-A3B (claim 12).
- Option: prima.cpp as a 2-node ring (workstation + rig). It needs the model file on both machines and
  CUDA builds on both. Low expected value.

**E11: security hardening check.** Low effort.
- Method: systemd unit for rpc-server as a non-root user, bound to the LAN or VPN address, with an
  nftables allow-list for the workstation only.
- Verify from a third host that port 50052 is unreachable. Compare against the SSH tunnel (`-H 127.0.0.1`
  + `ssh -L`) for load throughput, which also measures the Celeron's encryption cost.

### Open questions
- Is the workstation acceptable as a permanent participant in some modes, e.g. a session-scoped "big model
  mode", while the rig stays standalone for daily use? This decides whether (a)/(c) or (d) is worth
  investing in.
- After #18626/#22701, how many synchronous round trips per token remain with one server and nine devices?
  Does GRAPH_RECOMPUTE hit on most decode steps?
- Does the Celeron sustain gigabit TCP plus CUDA uploads at once, and does a WireGuard VPN overlay
  throttle it?
- Can the output head be placed on the workstation's CPU (not only its GPU) in an RPC-only device list,
  and is ~5 ms of workstation-CPU matmul over a 152k vocab better than 5 ms of network?
- Is there any maintained system besides llama.cpp RPC that does expert-aware placement over a LAN for GGUF
  MoE models on Pascal? None was found.
- Are there any measured gigabit results for a 40–60 GB MoE over llama.cpp RPC? None were found; the
  closest are 1 GbE with a 35B-A3B
  ([michaelspost](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html)) and 10 GbE with a
  dense 72B ([kjaiswal](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)).
