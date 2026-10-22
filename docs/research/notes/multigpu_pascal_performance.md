# Performance physics of MoE inference on 9 × P106-090 over PCIe 1.0 x1 (no P2P), as of 2026-10-22

Scope: where the bottlenecks are for llama.cpp MoE decode and prefill on the rig (9 × P106-090, sm_61, 5 SMs, ~173 GB/s measured, PCIe 1.0 x1, two ASM1184e switches, Celeron 3865U with 2 cores, 4 GB RAM); what parallelism schemes can help; tuning knobs; real measurements from similar rigs. Rig facts come from the [hardware spec](../../mining_rig.md) and are treated as fixed. "Claim to test" marks anything not verified by a source or measurement. Calculations marked "(my calc)" are this note's own arithmetic, not from a source. Nothing in this note was measured on the rig.

Headline findings (each is sourced below):
1. **In current llama.cpp master, CUDA graphs are switched off at runtime on Pascal.** The gate is `cc < GGML_CUDA_CC_VOLTA`, so sm_61 gets `disable_due_to_gpu_arch = true` whatever the build flags say. An open one-line PR (#27721, Aug 2026) enables them for sm_61. On a GTX 1080 it gave +8% to +78% MoE tg, and +25% on a P102-100. So "llama.cpp with CUDA graphs enabled" on this rig almost certainly means graphs are not actually running.
2. **Batch-1 layer split is capped by one GPU's bandwidth.** Pascal GDDR5 cards reach roughly 50–66% of theoretical bandwidth in llama.cpp tg. A single P106-100 does ~30 t/s on Llama-2-7B Q4_0, which is ~117 GB/s, or 61% of 192 GB/s.
3. **The 5-SM P106-090 has to pull about twice the bandwidth per SM** of any Pascal card with published llama.cpp numbers. Whether MMVQ can do that is unverified, so this is a top claim to test.
4. **On a 2-core 1.8 GHz CPU with no graphs, launch and CPU overhead for a ~48-layer MoE (~1.5–2k graph nodes/token) is likely comparable to the ~11–17 ms of GPU time for a 3B-active model.** This is a claim to test, but the GTX 1080 and P102-100 results show Pascal MoE decode is launch-bound even on desktop CPUs.
5. **Tensor and expert parallelism need about 2 host-staged synchronisations per layer.** On PCIe 1.0 x1 each costs about ≥0.1 ms (claim to test), and the CPU launches N× more kernels. Both are expected to lose to layer split at batch 1 unless graphs are enabled and the model is large/dense-ish. The evidence is P100/V100 tensor-split results, PR #19378 results on MoE models, and ThunderEP/CoMoE.
6. **Logits come back to the host every token.** For a 152k vocab that is ~608 KB over a ~213 MB/s link, about 2.9 ms per token (my calc). The merged backend-sampling feature (PR #17004) can avoid this.
7. **llama.cpp loads tensors in one sequential loop, with no per-GPU threads.** Load time is therefore about model size divided by one x1 link (~190 MB/s), not by the 715 MB/s aggregate.

---

## Q1. Batch-1 decode with layer split: single-GPU-bandwidth ceiling, MMVQ efficiency on Pascal, 5-SM behaviour

### Takeaway
With `--split-mode layer`, GPUs run their layer slices one after another. Batch-1 tg is therefore at most single-GPU effective bandwidth divided by active bytes per token. Pascal GDDR5 cards reach ~50–66% of theoretical bandwidth in llama.cpp tg. A P106-090 will probably land at ~60–115 GB/s effective, against 173 GB/s measured peak. The low end applies if 5 SMs cannot keep enough loads in flight (claim to test).

### Cited Findings
- llama.cpp's default `layer` mode is pipeline parallelism: "Each GPU holds a contiguous slice of layers" and the KV cache for each layer lives on that layer's GPU. Pipeline parallelism "minimizes data transfers between GPUs but requires many tokens to scale well." — [llama.cpp docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- Layer split adds capacity but little single-user speed, because one card computes at a time — [canitrun.dev guide (via search summary)](https://canitrun.dev/guides/multi-gpu-setups/); [vramglass multi-GPU guide (zh-tw)](https://vramglass.com/zh-tw/learn/multi-gpu-local-llm-guide)
- Batch-1 decode is memory-bandwidth bound. On RDNA4 a user reports decode "100% memory bandwidth bound", and dense models there reach ~79–83% of peak bandwidth — [llama.cpp discussion #21043](https://github.com/ggml-org/llama.cpp/discussions/21043)
- In llama.cpp's CUDA scoreboard (Llama-2-7B Q4_0, 3.56 GiB file, llama-bench -ngl 99), tg128 results — [llama.cpp discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013):
  - P106-100: 30.40 (FA off) / 30.64 (FA on) t/s; a separate Aug 7 2025 comment (build 5688) gives 29.42 / 29.94.
  - GTX 1060 6GB: 27.79 / 28.18 (also 27.74 / 28.08 on build 6123, Aug 11 2025).
  - Tesla P4: 33.29 / 33.12. Tesla P40: 54.74 / 53.73. Tesla P100: 58.35 / 61.99.
  - GTX 1070 Ti: 37.82 / 37.87. GTX 1080 Ti: 62.49 / 61.38. Titan Xp: 76.08 / 73.84.
  - Quadro P2000: 23.63 / 23.76. Quadro P1000: 13.99 / 13.65.
  - GTX 1050 Ti: 19.06 / 20.27 (Jun 24 2026). GTX 1050: 15.23 / 16.48 (Jul 17 2026; variant not stated).
- Fusion work in the CUDA backend (Dec 2025) took Qwen3-30B-A3B Q4_K_M tg from 246.96 to 352.06 t/s on an RTX 5090. The post's napkin math puts gpt-oss-20b on a 5090 at "~417 t/s (~80%)" of a "~520 t/s" speed of light — [am17an blog, "Optimizing Token Generation in llama.cpp's CUDA Backend"](https://am17an.bearblog.dev/new-post/)
- On 3 × CMP 50HX, locking the core clock at 1700 MHz changed decode by −0.8% and at 2100 MHz by +1.1%. The author concludes decode is memory-bound on those (Turing, 20 GB) cards — [FASTCHIP/cmp50hx-llama-research](https://github.com/FASTCHIP/cmp50hx-llama-research)
- llama.cpp's CUDA backend defines `GGML_CUDA_CC_DP4A 610` as "minimum compute capability for __dp4a". sm_61 is therefore the minimum target for the int8 dot-product paths (MMVQ/MMQ) — [ggml-cuda/common.cuh (master, read 2026-10-22)](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/common.cuh)
- FP16 avoidance is built in: `fast_fp16_hardware_available()` returns false for `cc == 610`, `fast_fp16_available()` excludes builds whose highest compiled arch is 610, and `FAST_FP16_AVAILABLE` is not defined when `__CUDA_ARCH__ == 610` — [ggml-cuda/common.cuh](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/common.cuh)
- MMVQ (the batch-1 quantized mat-vec path) is chosen when `ne2 <= MMVQ_MAX_BATCH_SIZE`. A `mul_mat_id` (MoE) fallback path "requires stream synchronization" — [ggml-cuda.cu (master)](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu)
- Qwen3-30B-A3B config: hidden_size 2048, 48 layers, 32 Q heads / 4 KV heads, head_dim 128, 128 experts with 8 active, moe_intermediate_size 768, vocab 151,936, untied embeddings — [Qwen3-30B-A3B config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)

### Inferences
- **Effective bandwidth in llama.cpp tg (my calc).** Bytes per token ≈ the 3.82 GB Q4_0 file; KV at tg128 is negligible. Spec bandwidths are NVIDIA spec-sheet values, not re-fetched.

  | Card | SMs | Spec BW | tg128 (FA on) | Effective BW | % of spec | GB/s per SM |
  |---|---|---|---|---|---|---|
  | P106-100 | 10 | 192 GB/s | 30.6 | ~117 GB/s | 61% | 11.7 |
  | GTX 1060 6GB | 10 | 192 | 28.2 | ~108 | 56% | 10.8 |
  | Tesla P4 | 20 | 192 | 33.3 | ~127 | 66% | 6.4 |
  | Tesla P40 | 30 | 346 | 54.7 | ~209 | 60% | 7.0 |
  | GTX 1070 Ti | 19 | 256 | 37.9 | ~145 | 57% | 7.6 |
  | GTX 1080 Ti | 28 | 484 | 62.5 | ~239 | 49% | 8.5 |
  | Titan Xp | 30 | 548 | 76.1 | ~291 | 53% | 9.7 |
  | Quadro P2000 | 8 | 140 | 23.8 | ~91 | 65% | 11.3 |
  | Quadro P1000 | 5 | 82 | 14.0 | ~53 | 65% | 10.7 |
  | GTX 1050 Ti | 6 | 112 | 20.3 | ~77 | 69% | 12.9 |
  | Tesla P100 (HBM2) | 56 | 732 | 62.0 | ~237 | 32% | 4.2 |

- **Pascal GDDR5 cards land at ~50–69% of spec bandwidth in llama.cpp tg.** On the same dense Q4_0 test that would put the P106-090 near ~110–120 GB/s, *if* it behaves like a P106-100.
- **The 5-SM problem (claim to test).** At 61% of 192 GB/s across only 5 SMs, each SM needs ~23 GB/s. No Pascal card in the table exceeds ~13 GB/s/SM. The 5-SM Quadro P1000 does reach 65% of its own (low) 82 GB/s.
  - ALU is unlikely to be the limit for Q4_0 MMVQ. My estimate is ~0.5 lane-ops per weight against ~218 G lane-ops/s per SM at 1.7 GHz.
  - The more likely limit is memory-level parallelism. By Little's law, ~192 GB/s × ~0.5–0.6 µs means ~100–115 KB in flight, or ~20–23 KB per SM.
  - The 173 GB/s measured on the rig ([hardware spec](../../mining_rig.md)) suggests a streaming kernel can do it. MMVQ with its block and warp configuration may not.
  - Plausible range for P106-090 tg on Llama-2-7B Q4_0: ~16–31 t/s (60–118 GB/s). This is the first thing to measure (Experiment E1).
- **K-quants and IQ-quants do more integer work per weight than Q4_0** (unpacking, scales, lookup tables). On a 5-SM card they are more likely to become issue-bound than on a 10-SM P106-100. Comparing tg per GB of weights across Q4_0 / Q4_K / IQ4_XS / Q8_0 on one P106-090 would show whether the card is bandwidth- or issue-limited (claim to test).
- **Small per-expert matmuls (my calc).** In Qwen3-30B-A3B one expert matrix is 2048 × 768 ≈ 1.57 M weights ≈ 0.9 MB at ~4.5 bpw. The gate+up `mul_mat_id` for 8 experts is ~14 MB (~120 µs at 115 GB/s) and down is ~7 MB (~60 µs). These kernels are long enough to fill 5 SMs.
- **Most nodes per layer are tiny** (norms, rope, router softmax/top-k, adds, KV copies): microseconds of GPU work each. For those, launch overhead rather than SM count decides throughput (see Q2). A 5-SM GPU fills with fewer blocks, so small kernels waste less of it than they would on a big GPU. The cost is fixed per-launch latency, not idle SMs.
- **FP16 is not a concern** for the main decode path as long as weights are quantized: MMVQ/MMQ run on dp4a/int, and llama.cpp marks cc 610 as lacking fast FP16. F16/BF16 *weight* tensors in a GGUF would fall back to slower paths. Checking the GGUF tensor types is worthwhile (claim to test).
- **KV reads add to bytes per token at depth (my calc).** Qwen3-30B-A3B with F16 KV reads 4 KV heads × 128 × 2 (K,V) × 2 B × 48 layers ≈ 96 KB per context position per token. That is ~0.4 GB/token at 4k context and ~3.1 GB/token at 32k, which exceeds the ~1.9 GB of weights. q8_0 KV roughly halves it, but is incompatible with `-sm tensor`.

### Gaps
- No published llama.cpp benchmark of a P106-090 (5 SMs) was found. All per-SM reasoning above is extrapolated from other cards.
- No source gives the fraction of peak bandwidth reached by the MoE `mul_mat_id` MMVQ path on Pascal specifically. The GTX 1080 MoE numbers in Q2 are launch-bound, so they do not isolate kernel efficiency.
- Whether llama.cpp currently keeps the input embedding (`tok_embd`) on the CPU and uploads an 8 KB hidden vector per token was not verified in source (claim to test).

---

## Q2. Per-token overheads: host-staged hops, synchronisation, kernel launch with/without CUDA graphs, kernel count, 9 GPUs vs fewer

### Takeaway
The PCIe hop cost for layer split is small: an 8 KB F32 activation per GPU boundary, likely ~0.05–0.2 ms per hop host-staged on PCIe 1.0 x1 (claim to test). Two other costs are larger.
- **CPU kernel launches.** CUDA graphs are disabled on Pascal in llama.cpp master, so a ~48-layer MoE issues ~1.5–2k individual launches per token from a 1.8 GHz 2-core CPU.
- **Logits.** The 152k-vocab logits come back over x1 each token, about 2.9 ms.

Using the minimum number of GPUs saves only ~0.1–0.2 ms per removed hop. Its larger benefits are fixed VRAM overhead and freeing GPUs for a second instance.

### Cited Findings
- **CUDA graphs on Pascal are disabled at runtime.** In `ggml_cuda_graph_set_enabled`: `if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) { graph->disable_due_to_gpu_arch = true;` — [ggml-cuda.cu (master, read 2026-10-22)](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu)
- Graphs were originally gated to Ampere+. PR #25749 (merged 2026-07-16) extended them to Volta/Turing; "Pascal (CC 6.x) and older remain excluded".
  - The maintainer said the gate existed because the ggml-side graph logic was more primitive at the time, and older GPUs may have been hurt; nobody had re-checked since.
  - Graphs add ~128 bytes per node per graph, "about 180 kB for a 1.5k-node graph".
  - On 8 × V100, `-sm layer` tg moved "mostly within noise" (Qwen3.5 122B-A10B Q8_0: 41.94 → 47.68 t/s; Gemma 4 26B-A4B: 62.46 → 64.99). `-sm tensor` roughly doubled to tripled (122B-A10B: 17.47 → 58.25).
  — [llama.cpp PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749)
- **Open PR #27721 "CUDA graphs on Pascal (sm_61)"** (opened 2026-08-25, still open): a one-line change moving the gate from `GGML_CUDA_CC_VOLTA` to `GGML_CUDA_CC_DP4A` (610). Results — [llama.cpp PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721):
  - GTX 1080, llama-bench tg64 at depth 0 / depth 4096, graphs off → on:
    - granite-3.1-3b-a800m Q4_K_M: 81.2 → 124.5 / 76.5 → 97.4.
    - Ling-3.0-tiny Q6_K: 52.7 → 79.8 / 51.7 → 76.4.
    - granite-4.0-h-tiny Q4_K_M: 57.4 → 79.5 / 54.2 → 77.4.
    - OLMoE-1B-7B Q4_K_M: 150.5 → 169.5 / 115.8 → 124.6.
    - LFM2.5-8B-A1B Q4_K_M: 116.2 → 128.5 / 112.2 → 125.6.
    - Dense Qwen-based 9B Q5_K_M: 28.5 → 30.1 / 27.9 → 29.5.
  - GTX 1080 prefill was flat (pp512 1935 → 1909) and perplexity was identical.
  - P102-100 (sm_61, Ubuntu, driver 580.173.02, CUDA 12.9): Ling-3.0-tiny Q8_0 tg128 70.80 → 88.42 (+25%); pp512 flat.
  - P100 (sm_60, not changed by the PR): graphs regressed single-card and `-sm layer`. On 4 × P100, Qwen3.6 35B-A3B Q8_0 `-sm tensor` went 39.58 → 70.62 t/s with graphs, against 61.44 for `-sm layer`.
  - The PR cites #6766's original P40 measurement: a dense model with ~700 graph nodes regressed 53.84 → 51.67 t/s with graphs.
- The original CUDA-graphs PR (#6766) added the commit "Disable CUDA graphs for old GPU arch and with env var" and the `GGML_CUDA_DISABLE_GRAPHS` env var — [PR #6766 commit list](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/6766/commits). NVIDIA's write-up reports up to 1.2× for Llama 7B on H100 and says graphs were then restricted to batch size 1 — [NVIDIA blog, Aug 2024](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs)
- Generic launch cost (search-result summary of NVIDIA's CUDA Graphs intro, V100 with a fast host): a 2.9 µs kernel cost 9.6 µs per step with a synchronise after every launch, 3.8 µs with overlapped launches, and 3.4 µs as a captured graph — [NVIDIA "Getting started with CUDA Graphs" (via search snippet)](https://developer.nvidia.com/blog/cuda-graphs)
- In the CUDA backend, the `mul_mat_id` fallback path "synchronizes the stream, so we cannot use CUDA graphs". Graph capture needs a warmup: "at least 2 calls with no property change" — [ggml-cuda.cu](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu)
- Kernel fusion is disabled for `cc <= GGML_CUDA_CC_PASCAL` ("fusion is not universally faster on Pascal") — [ggml-cuda.cu](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu). However, `GGML_CUDA_CC_PASCAL` is defined as 600 — [common.cuh](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/common.cuh) — so the gate appears to exclude only sm_60 and older, and sm_61 may still get fusion (claim to test). Concurrent QKV streams (`GGML_CUDA_GRAPH_OPT=1`) "only work on a single GPU for now" — [am17an blog](https://am17an.bearblog.dev/new-post/)
- **Cross-GPU copies:** `ggml_backend_cuda_cpy_tensor_async` calls `cudaMemcpyPeerAsync` between different physical GPUs. It returns false only if compiled with `GGML_CUDA_NO_PEER_COPY`, and there is no runtime check of peer access. Completion is ordered with `cudaEventRecord` on the source stream and `cudaStreamWaitEvent` on the destination stream, not a host sync — [ggml-cuda.cu](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu)
- Peer access is opt-in via `GGML_CUDA_P2P`, needs driver support "usually restricted to workstation/datacenter GPUs", and "may cause crashes or corrupted outputs" on some boards, for example with IOMMU — [multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- When peer access is unavailable, the CUDA driver "stages the transfer via CPU memory" and the copy "will transparently fall back to a normal copy through the host" — [NVIDIA SC11 multi-GPU slides](https://www.nvidia.com/docs/io/116711/sc11-multi-gpu.pdf); [NVIDIA GPUDirect/UVA webinar](https://developer.download.nvidia.com/CUDA/training/cuda_webinars_GPUDirect_uva.pdf)
- Staged-copy latency (old hardware, 2013): 8.2 µs P2P versus 16.8–17.4 µs staged. "A single cudaMemcpy adds roughly 10 µs" — [APEnet+ paper, arXiv:1307.8276](https://arxiv.org/pdf/1307.8276). A forum report says host-staged peer copies queued behind pending work stalled concurrent copies — [NVIDIA forum](https://forums.developer.nvidia.com/t/queueing-device-to-device-peer-memcpy-stalls-concurrent-copy-operations/295894)
- **Measured hop cost on x1 links with fast GPUs and a fast CPU** (2026-07-31, llama.cpp build 10109, CUDA 12.9, no P2P) — [Agent Workflow Lab (note.com)](https://note.com/agentworkflowlab/n/na17f86500825):
  - Gemma-4 26B-A4B Q8_0 split over two RTX 3090s gave tg 94.91 t/s on Gen2 x2+x1 and 95.75 on x1+x1, so link width made no measurable difference to layer-split tg.
  - A 12B model on a single card ran ~10% slower on the Gen2 x1 card than on the x2 card. The author notes these were different physical cards.
- PR #6017 (pipeline parallelism, 8 × A100 PCIe): Llama 7B Q4_0 tg128 ≈ 145 t/s on 1 GPU versus ≈ 135 t/s on 8 GPUs — [PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017)
- **Vulkan counter-example.** Qwen2.5-1.5B Q4_K_M decode on P100 went from 419.7 t/s on 1 GPU to 150.4 (2 GPUs, layer) and 133.3 (4 GPUs, layer), with V100 similar. The vignette says Vulkan transfers go through host RAM at "roughly 1 GB/s" — [llamaR multi-GPU vignette](https://cran.nics.utk.edu/cran/web/packages/llamaR/vignettes/multi-gpu.html)
- **Backend sampling (PR #17004, merged).** Sampling can run inside the GPU graph, so "only the sampled token needs to be transferred", or only filtered logits. It adds `--backend-sampling` and `--backend-dist` flags — [llama.cpp PR #17004](https://github.com/ggml-org/llama.cpp/pull/17004)
- llama.cpp's own tuning doc says that if tg is very slow, try `-t 1`; a big gain means the CPU is oversubscribed — [token_generation_performance_tips.md (mirror)](https://git.peisongxiao.com/CS348Project/llama.cpp/src/branch/master/docs/development/token_generation_performance_tips.md)

### Inferences
- **Hop cost (my calc, claim to test).** The activation per boundary is hidden × 4 B: 8 KB for Qwen3-30B-A3B, ~16 KB for hidden 4096.
  - Host staging means two legs: D2H at ~213 MB/s (~38 µs) and H2D at ~189 MB/s (~43 µs), plus fixed driver/PCIe latency of ~10 µs per leg (APEnet-era estimate) and event waits.
  - Estimate: ~0.1 ms per hop (range ~0.06–0.2 ms). 9-GPU layer split has 8 hops (~0.5–1.6 ms/token), 4-GPU has 3 hops (~0.2–0.6 ms).
  - Only one hop is active at a time at batch 1, so the shared ASM1184e upstream (PCIe 2.0 x1) does not matter here.
- **Cross-check from PR #6017:** 1/135 − 1/145 s ≈ 0.51 ms for 7 extra hops ≈ 73 µs per hop on A100 PCIe with a fast host. The note.com result shows hops are negligible on Gen2 x1 at ~10 ms/token.
- **Cross-check from the Vulkan vignette:** ~4.3 ms extra per token for one hop. Hop costs are dominated by backend synchronisation design, not by link bandwidth. Our CUDA path uses events, not host syncs, but host-staged peer copies may serialise (claim to test, E4).
- **Logits D2H (my calc).** 151,936 × 4 B ≈ 608 KB per token at ~213 MB/s ≈ 2.9 ms per token. That is roughly 15–25% of a 3B-active token time. CPU sampling over 152k floats on a no-AVX 1.8 GHz Celeron adds more (unknown, claim to test). `--backend-sampling` with greedy or top-k on GPU should remove most of both (E6).
- **Graph node count for a 48-layer MoE (my estimate, claim to test).**
  - Attention ≈ 12–16 nodes per layer: norm, mul, Q/K/V matmuls, Q/K norms, 2 × rope, 2 × KV copy, FA, O matmul, residual add.
  - MoE ≈ 18–25 nodes per layer: norm, mul, router matmul, softmax, top-k/argsort, get_rows, weight normalisation, gate/up `mul_mat_id`, GLU, down `mul_mat_id`, weighting, ~k−1 expert-sum adds, residual add.
  - Total ≈ 30–40 per layer × 48 ≈ 1,500–1,900 nodes, plus the output layer. This is consistent with the "1.5k-node graph" example in PR #25749 and with ~700 nodes for a dense 7B in PR #6766.
- **Launch-bound calibration from PR #27721 (my calc).**
  - granite-3.1-3b-a800m (32 layers, ~0.5 GB/token at Q4) has a GTX 1080 bandwidth ceiling of several hundred t/s. It ran 81 t/s without graphs, i.e. 12.3 ms/token for ~1.2–1.3k nodes, which is ≈ 9–10 µs of host-side cost per node on the author's Windows desktop.
  - With graphs it reached 124 t/s (8 ms), still not bandwidth-bound.
  - The Celeron 3865U (1.8 GHz, no turbo, 2 cores shared with GNOME/Xorg) will probably need at least as long per node. Linux launch overhead is generally lower than Windows WDDM, which partly offsets this.
  - Rough range: ~6–12 µs per node → ~9–23 ms of CPU time per token for a 48-layer MoE. That is comparable to or larger than the ~11–17 ms of GPU time for a 3B-active Q4 model. The 3B-active case is therefore likely CPU/launch-bound unless graphs are enabled (claim to test, E2/E3).
- **CPU work and GPU work overlap only partly.** The CPU enqueues ahead while the GPU executes. Per-token graph build, the sampling sync and the logits copy are serial bubbles.
  - Model: T_token ≈ max(T_gpu, T_launch) + T_graph_build + T_logits + T_sample + (G−1)·T_hop.
- **9 GPUs vs the minimum.** Layer split's bandwidth ceiling does not change with GPU count.
  - Fewer GPUs saves ~0.1–0.2 ms per removed hop, one ggml scheduler split per removed GPU, and the fixed per-GPU VRAM (CUDA context, compute buffer, and the existing Xorg context on every GPU).
  - Fewer GPUs also leaves GPUs free for a second instance.
  - More GPUs buys room for KV cache, longer context or a higher-bit quant.
  - For a model that fits on 4 GPUs, the expected tg gain from using 4 instead of 9 is small (~2–8%). The throughput gain from running a second instance on the freed GPUs could be large, but it competes for the same 2 CPU cores.

### Gaps
- No measured cudaMemcpyPeerAsync latency for small (KB-scale) host-staged copies on PCIe 1.0 x1 with CUDA 12.x was found. The ~10 µs/leg figure is from 2013 hardware. E4 should measure it.
- No measurement of llama.cpp per-node CPU cost on a low-end mobile Celeron was found.
- Whether host-staged `cudaMemcpyPeerAsync` serialises the source and destination streams, which would stall the CPU-ahead pipeline, is unverified.

---

## Q3. Tensor parallelism (row / `-sm tensor`) over x1 without P2P

### Takeaway
`-sm row` is deprecated. `-sm tensor` (merged 2026-04-09, experimental) splits weights and KV and all-reduces partial results about twice per layer. On this rig each all-reduce is host-staged over PCIe 1.0 x1, and four GPUs share each switch's PCIe 2.0 x1 upstream. The CPU must also launch kernels on every GPU, and without CUDA graphs on Pascal that cost is multiplied by N. The evidence points to a clear loss for MoE models at batch 1. The only plausible wins are dense or large-active models at deep context, and only with graphs enabled.

### Cited Findings
- `-sm tensor` was merged 2026-04-09 as experimental. It wraps several backends in a "meta" backend, exchanges partial results with `shfl_tensor_async` followed by `GGML_ADD`, and later gained NCCL/RCCL support and "BF16 for allreduce". Requirements: FA on, KV not quantized, `-fit` unsupported. Known issues include a CUDA-graph VRAM leak — [PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378)
- PR #19378 author's guidance: tensor parallelism "works comparatively better for slow GPUs with fast interconnect" and "comparatively worse for fast GPUs with slow interconnect running small, sparse models". "The key factor is how long the base runtime is vs. the synchronization overhead." — [PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378)
- PR #19378 benchmarks (PCIe 4.0 x16, RTX 4090) — [PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378):
  - 2 GPUs, tg128 tensor/layer: llama 8B F16 1.60×; gemma4 31B Q8_0 1.53×; **gemma4 26B-A4B Q4_0 0.70×; qwen35moe 35B-A3B Q4_0 0.69×**.
  - 4 GPUs: llama 70B Q4_K_M tg 21.25 → 48.31 (2.27×); llama 8B pp512 0.41×.
  - MoE prefill can win: gpt-oss-120B pp512 1.42×; glm4moe 106B-A12B pp512 at depth 65,536 2.14×.
- Docs: tensor mode is "much more bottlenecked by the GPU interconnect speed". If multi-GPU is slower than single-GPU, try layer. With NCCL, cross-GPU reductions use it; without NCCL, a warning says performance "will be suboptimal". Several MoE architectures (e.g. DeepSeek2, Grok, OLMoE, Jamba) are listed as not implemented — [multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- `-sm row` is "Deprecated" with "comparatively poor performance" — [multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- NCCL uses the SHM transport (host memory) when P2P is unavailable — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/2.8.3/env.html). On dual RTX 5090 (PHB, no NVLink), patched P2P versus SHM moved end-to-end vLLM TP=2 by about +2.7% mean (−1.9% to +4.7%) — [vLLM forum](https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870)
- **Pascal/Volta TP evidence:**
  - 4 × P100 Qwen3.6 35B-A3B Q8_0 tg: `-sm tensor` 39.58 without graphs versus 61.44 with `-sm layer`; tensor reached 70.62 with graphs force-enabled — [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)
  - 8 × V100: `-sm tensor` tg for Qwen3.5 122B-A10B Q8_0 went 17.47 → 58.25 t/s once graphs were enabled — [PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749)
  - P100 patch set: 2 × P100 with `-sm tensor` gained ~15% pp and ~10% tg; "upstream llama.cpp's internal AllReduce only handles exactly two devices" (search summary of the repo README; not re-verified) — [key-sh/p100-llama-cpp](https://github.com/key-sh/p100-llama-cpp); [qr0n/p100-llama-cpp](https://github.com/qr0n/p100-llama-cpp)
  - With NCCL enabled the AllReduce path changes, and ggml forces peer access on when built with NCCL (`#if defined(GGML_USE_NCCL)` → `use_peer_access = true` for VMM pools) — [ggml-cuda.cu](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu). How this behaves on GeForce cards without P2P is unverified.
- **3 × CMP 50HX** (Gen2 x8, P2P working at 2.47 GiB/s; Qwen3.8-27B dense Q4_K_XL; Sep 2026) — [FASTCHIP/cmp50hx-llama-research](https://github.com/FASTCHIP/cmp50hx-llama-research):
  - tensor versus layer decode: +56% short prompt, +60% at 32k, +88–97% at 192–250k.
  - Prefill was roughly the same.
  - Tensor lost with ≥2 concurrent clients (−8.4% at 2 clients, −33.5% at 5), so production stayed on layer split.
- ik_llama.cpp "graph" split (Feb 2026, 4 × 3090): GPT-OSS-120B tg 163.7 (ik graph) versus 122.9 (mainline tensor); pp 5,225 versus 728. The author called mainline's implementation "a gimmick not ready for prime time" (competing fork; treat with caution). RPC devices cannot be used for graph parallel — [ik_llama.cpp discussion #1247](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247)
- R vignette (Vulkan): 4-way row split on P100 gave 130.0 t/s versus 419.7 on a single GPU, for a 1.5B model — [llamaR vignette](https://cran.nics.utk.edu/cran/web/packages/llamaR/vignettes/multi-gpu.html)

### Inferences
- **Traffic per token (my calc).** About 2 all-reduces per layer (after attention O and after the FFN/MoE down projection).
  - Each moves hidden × 4 B (8 KB F32, or 4 KB with BF16 all-reduce) per GPU, in and out.
  - 48 layers → ~96 all-reduces per token.
  - Host-staged, each needs at least a D2H leg, a host or ring reduction, and an H2D leg: ≥ ~2 × (10 µs + 4–8 KB / ~0.1–0.2 GB/s) ≈ 0.08–0.2 ms with uncontended links.
  - With 4 GPUs behind one ASM1184e (PCIe 2.0 x1 upstream, measured ~90–120 MB/s per GPU when all are busy, ~715 MB/s total), 4 × 8 KB through a ~400 MB/s upstream adds ~80 µs per direction.
  - Communication ≈ 96 × 0.1–0.3 ms ≈ 10–30 ms per token, fully exposed at batch 1.
- **Launch multiplication.** Each GPU runs its own copy of every layer's kernels, so the single CPU thread must enqueue ~N × 1.5–2k launches per token. Without graphs, on the Celeron, that is ~N × 9–23 ms (N=4: ~36–90 ms). The P100 (fast host, x16) result already shows this cost: 39.6 t/s for tensor versus 61.4 for layer.
- **What tensor saves.** It cuts GPU memory time to ~1/N: ~11–17 ms → ~3–4 ms for 3B-active on 4 GPUs. Communication of ≥10 ms plus launches of ≥36 ms (no graphs) dominates.
  - Prediction for MoE 3B-active: **tensor ≈ 0.3–0.7× layer** without graphs, and roughly break-even at best with graphs (claim to test, E7).
  - For a 10B-active model on 9 GPUs, communication (~96–130 all-reduces, partly through shared switches) and 9× launches make a loss likely too.
- **Possible wins (claim to test):** dense or large-active models at deep context with graphs on (as in the V100, P100 and CMP 50HX data), and prefill of big MoE at very deep context (PR #19378's glm4moe result). Neither matches the first target (3B-active, single user, short context).
- Tensor mode forbids quantized KV, so KV costs more VRAM. Pascal has no native BF16 math, so BF16 all-reduce conversion is software, which is cheap for 8 KB vectors.

### Gaps
- No `-sm tensor` result over x1 risers or without P2P on CUDA was found.
- It is unclear how the meta backend moves data when P2P is unavailable and NCCL is not installed: `cudaMemcpyPeerAsync` host staging, or NCCL SHM.
- No measured all-reduce latency for KB-sized messages over PCIe 1.0 x1 was found.

---

## Q4. Expert parallelism (EP) for batch-1 decode

### Takeaway
In principle EP lets several GPUs read different experts at once, beating the single-GPU ceiling for the expert part of each layer. At batch 1 it needs a broadcast plus a gather per MoE layer over host-staged x1 links, and multi-GPU launches from a 2-core CPU. Random routing also gives poor balance: 8 experts over 9 GPUs typically puts 2–3 on the busiest GPU.
- 3B-active (Qwen3-30B-A3B): rough break-even or a loss.
- ~10B-active with large experts and few layers: a possible modest win.
- llama.cpp has no EP mode. The closest proxy is `-sm tensor` on supported MoE architectures.
- The research systems (ThunderEP, CoMoE) target batch ≥6 on x16 links.

### Cited Findings
- **ThunderEP** (arXiv 2609.40093, 2026-09-30) — [ThunderEP, arXiv:2609.40093](https://arxiv.org/html/2609.40093):
  - Setup: 6 × RTX 4090 (PCIe 4.0 x16) and 6 × RTX 5090 (PCIe 5.0 x16), two GPUs per host bridge. "Neither system supports GPUDirect P2P", so all traffic is staged through host memory.
  - Host-staged bandwidth: 56.5 GB/s for one 5090, falling to 35.5 GB/s per GPU when two share a bridge.
  - "Decode transfers only a few kilobytes per GPU". In decode, communication and expert compute are not overlapped because "each GPU processes too few tokens to form an effective pipeline". Flag polling is a large share of small-transfer time.
  - Small-message (1 KB–1 MB) dispatch was 2.30×/2.65× faster than NCCL on 4090/5090.
  - Qwen3-30B-A3B decode at batch 24: exposed MoE communication −50.1%, layer latency −15.2%. Decode throughput gains over vLLM averaged 1.12–1.17×.
  - Batch-1 decode was not reported (batches 6–96).
- **CoMoE** (arXiv 2610.09424, Oct 2026) targets consumer GPUs with "limited PCIe bandwidth and host-mediated communication with no peer-to-peer (P2P) support". It uses host-backed token multicast for dispatch and host-staging aggregation for combine, and claims up to 1.46× throughput (RTX 5090; abstract only) — [CoMoE, arXiv:2610.09424](https://arxiv.org/abs/2610.09424)
- A CPU-GPU local MoE system uses "SmallEP" expert parallelism for prefill on PCIe-only nodes, with 2 × RTX 5090 (abstract only) — [arXiv:2606.10493](https://arxiv.org/pdf/2606.10493)
- No llama.cpp PR or discussion implementing expert parallelism was found. Searches returned only general guides and TensorRT-LLM docs (TP, EP, and hybrid `moe_tp_size × moe_ep_size`) — [TensorRT-LLM EP docs](https://nvidia.github.io/TensorRT-LLM/0.21.0/advanced/expert-parallelism.html)
- `-sm tensor` MoE support initially covered only Qwen3-MoE and GPT-OSS. It was added 2026-02-08 — [ik_llama.cpp discussion #1247](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247)
- "Strata", a different engine, ran a 125B-total / 6B-active MoE (Q2_0) fully in VRAM across 3 × CMP 50HX at 80–82 tok/s decode on short prompts. The engine is unfamiliar and the result unverified — [FASTCHIP/cmp50hx-llama-research](https://github.com/FASTCHIP/cmp50hx-llama-research)

### Inferences
- **Batch-1 EP budget for Qwen3-30B-A3B (my calc).**
  - Expert bytes per layer ≈ 8 × 3 × 0.9 MB ≈ 21–23 MB, ~0.19 ms at 115 GB/s on one GPU. Per token ≈ 1.1 GB, ~9.6 ms.
  - Ideal 8-way spread: ~0.025 ms per layer. With random routing over 9 GPUs, the busiest GPU typically has 2–3 experts → ~0.06–0.07 ms. Saving ≈ 0.12–0.13 ms per layer ≈ 6 ms per token.
  - Cost per layer: H2D broadcast of 8 KB to up to 8 GPUs (two groups of four sharing a ~400 MB/s upstream, ~80 µs per group plus fixed latency), then D2H gather of up to 8 × 8 KB, plus event and host synchronisation, plus kernel launches on several GPUs from 2 cores. About 0.15–0.3 ms per layer → **7–14 ms added per token against ~6 ms saved**. Net: likely a loss (claim to test, E8).
- **For a ~10B-active model with bigger experts** (hidden ~4096; expert bytes per layer maybe 50–80 MB at ~4 bpw; my assumption, check the specific model), the saving per layer grows to ~0.3–0.5 ms while communication grows only to ~0.2–0.35 ms (16 KB vectors). That could net ~10–25% on the expert part. Fewer layers help, and switch contention hurts (claim to test).
- **EP conflicts with llama.cpp's tensor layout.** All experts of a layer live in one 3D tensor (`ffn_*_exps`), so `-ot` override-tensor rules place whole tensors and cannot spread one layer's experts over GPUs. Real EP would need code changes (my reading; verify).
- **`-sm tensor` as an EP proxy.** If the meta backend splits expert tensors across GPUs for Qwen3-MoE, every GPU reads 1/N of every active expert: perfect balance, with all-reduce traffic similar to EP's broadcast plus gather. Testing it is the cheapest way to probe EP physics, with all the TP caveats in Q3.
- A 2-core CPU hosting the reduction and polling flags (as in ThunderEP/CoMoE) would also compete with kernel launching.

### Gaps
- No EP benchmark on PCIe Gen1/Gen2 x1 links, or at batch 1, was found anywhere.
- Which dimension `-sm tensor` splits MoE expert tensors along is not documented in the sources read.

---

## Q5. Prompt processing (pp): compute, dp4a MMQ, pipeline parallelism scaling, effect of x1

### Takeaway
pp is compute-bound and runs on dp4a int8 MMQ. A P106-100 does ~407–438 t/s pp512 on 7B Q4_0. The P106-090 has half the SMs, so expect roughly half per card (claim to test). Pipeline parallelism (multiple ubatches in flight) gave ~5.3× on 8 GPUs in the original PR. On this rig the activation hop per ubatch (~4 MB for 512 × 2048 F32) costs ~20 ms per hop at ~190 MB/s, small next to per-stage compute on slow GPUs. Scaling is limited by pipeline fill, `n_batch`/`n_ubatch` and the scheduler's copy count.

### Cited Findings
- pp512 (Llama-2-7B Q4_0, FA off/on) — [discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013):
  - P106-100: 406.9 / 438.5. GTX 1060 6GB: 416.9 / 446.2.
  - P40: 1007 / 1080. P100: 761 / 787. 1080 Ti: 1084 / 1138. 1070 Ti: 714 / 791.
  - Quadro P1000 (5 SMs): 183 / 174. Quadro P2000: 309 / 312. GTX 1050 Ti: 267 / 268.
- P106-100 on Vulkan: ~184 t/s pp, ~29.8 t/s tg — [discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- PR #6017 (pipeline parallelism, merged 2024-03-13; 8 × A100 PCIe, Llama 7B F16) — [PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017); [commit f30ea47](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/f30ea47a87ed4446ad55adb265755dc9102956a2):
  - pp8192 ≈ 4,527 t/s (1 GPU, -ub 4096), 8,588 (2), 15,100 (4), 23,821 (8; -ub 256). Q4_0 pp8192 on 8 GPUs: 4,227 (master) → 16,992 (PR).
  - `n_ubatch` (default 512) sets the per-computation batch. The number of scheduler input copies (`LLAMA_SCHED_MAX_COPIES`, default raised from 2 to 4) sets how much work can queue without syncing, at a memory cost.
  - Only CUDA was implemented.
- PCIe link effects on pp:
  - Gemma-4 26B-A4B over two 3090s: x1+x1 pp was 9.3–13.2% lower than x2+x1 in most conditions (+1.4% in one). Replacing a Gen2 x1 3090 with a Gen3 x16 4090 raised pp 603 → 824 and cut TTFT 175 → 109 ms. 118B-A8B over 4 GPUs reached pp 2,227 at depth 0 — [note.com](https://note.com/agentworkflowlab/n/na17f86500825)
  - A P104-100 article reports prefill rising 9 → 38 t/s after a link was fixed from x1 to x4. That setup offloads experts to CPU, so prefill streams weights over PCIe; this does not apply to an all-in-VRAM rig — [CSDN/Toutiao article (search summary; page fetch failed)](https://blog.csdn.net/laot007/article/details/161087262)
- On 3 × CMP 50HX, `-ub 1024` gave ~+3% prefill. Changing the MMQ→cuBLAS threshold gave ~8–13.8% faster prefill at 32k — [FASTCHIP repo](https://github.com/FASTCHIP/cmp50hx-llama-research)
- CMP 40HX (sm_75) runs Llama-2 7B Q4_0 at only ~144 t/s pp but ~80 t/s tg. The issue requests dp4a emulation via dp2a, implying a weak dp4a path on that mining card — [llama.cpp issue #24616](https://github.com/ggml-org/llama.cpp/issues/24616)

### Inferences
- **Compute estimate (my calc).** 7B dense ≈ 13.5 G ops/token (2 × 6.74 B), so P106-100 at ~430 t/s sustains ~5.8 T int8 ops/s. Its dp4a peak is ~17 T ops/s (1280 lanes × 4 MAC × 2 ops × ~1.7 GHz), so it runs at ~33% of peak.
  - P106-090 dp4a peak is ~7.8 T ops/s at 1531 MHz (4 × the rig's 1.96 TFLOPS FP32), ~9.8 T at 1911 MHz.
  - At the same ~33% efficiency (~2.6–3.2 T ops/s) it would do roughly 190–240 t/s pp512 on 7B Q4_0, about half a P106-100.
  - For a 3B-active MoE (~6 GFLOP/token), the dense-equivalent ceiling is ~400–500 t/s per card. MoE MMQ is less efficient: each of 128 experts sees only ~512×8/128 ≈ 32 tokens per ubatch. Realistic single-card-equivalent pp is maybe ~150–300 t/s (claim to test).
- **Pipeline pp over x1 (my calc).** Per hop per ubatch the copy is ubatch × hidden × 4 B = 512 × 2048 × 4 ≈ 4 MB, ~20 ms at ~200 MB/s.
  - Stage compute for 12 of 48 layers at ~200 t/s per card ≈ 0.6 s per 512-token ubatch, so transfers are ~3–5% of stage time if they overlap.
  - Host-staged peer copies may not overlap with compute (claim to test).
  - x1 links are not the main pp limiter on this rig. The faster 3090s above saw ~10% pp loss on Gen2 x1.
- **Pipeline fill (my calc).** With 9 stages and 16 ubatches (an 8k prompt at -ub 512), efficiency ≈ 16/(16+8) ≈ 67%. With the default `n_batch` 2048 (4 ubatches) and 4 scheduler copies, at most ~4 ubatches are in flight, so >4 stages cannot all be busy.
  - Raising `-b`, lowering `-ub` (e.g. 128–256) and building with more scheduler copies are the knobs (claim to test, E9). Smaller ubatches hurt MMQ efficiency on MoE.
- Effective pp across 4 GPUs for a 3B-active model is perhaps ~2–3.5× one card's pp, i.e. ~300–900 t/s total (very rough, claim to test).
- **Power limit matters more for pp than tg** (see Q8). pp is where 60 W versus 75 W shows up.

### Gaps
- No published pp numbers for MoE models on Pascal mining cards in multi-GPU pipeline mode.
- The current name and default of the copies option (`GGML_SCHED_MAX_COPIES` versus `LLAMA_SCHED_MAX_COPIES`) was not confirmed in current source.

---

## Q6. Model load time, 4 GB host RAM, mmap vs no-mmap, pinned staging

### Takeaway
llama.cpp's loader walks tensors in one sequential loop with no per-GPU threads. With `--no-mmap`/`--load-mode none` it uses 4 pinned 1 MB staging buffers with async uploads; mmap mode copies from the page cache. Either way, loading this rig should be bound by one x1 link at a time (~190 MB/s):
- ~18.7 GB (Qwen3-30B-A3B Q4_K_M) → ~1.6 min
- ~44 GB → ~4 min
- ~54 GB → ~4.8 min

That is much longer than the ~80 s parallel estimate (my calc). Parallel per-GPU loading would need a code change.

### Cited Findings
- `load_all_data` — [src/llama-model-loader.cpp (master, read 2026-10-22)](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model-loader.cpp):
  - "4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives" (`n_buffers = 4`; 1 MiB unless direct I/O alignment applies).
  - "When not using mmaped io use async uploads from pinned memory to GPU memory".
  - Async uploads are disabled when `use_mmap || check_tensors`.
  - A single `upload_backend` is chosen from the device of `bufs[0]`, and tensors are processed in one sequential `for` loop; no per-GPU threads were found.
  - Without mmap, tensors in non-host buffers can also be "staged through a temporary buffer sized like the tensor", loaded "biggest-first".
  - Load modes: `LLAMA_LOAD_MODE_MMAP`, `MMAP_MLOCK`, `AUTO`, `DIRECT_IO`. A warning suggests "consider using --load-mode none for better performance".
- Pinned host buffers use `cudaMallocHost`, fall back to CPU memory on failure, and can be disabled with `GGML_CUDA_NO_PINNED` — [ggml-cuda.cu](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu)
- Narrow links slow loading: server-ready time was 99.39 s with a Gen2 x1 3090 against 39.14 s after swapping in a Gen3 x16 4090, for the same 26B model with files pre-cached — [note.com](https://note.com/agentworkflowlab/n/na17f86500825)
- llama.cpp's default is mmap unless a device does not support it, and disabling mmap makes loading slower (search summary of Jan's llama.cpp docs) — [jan.ai llama.cpp docs](https://jan.ai/integrations/llamacpp)

### Inferences
- **Load time ≈ model bytes / ~0.19 GB/s while the loop is sequential (my calc).** Disk (SATA SSD, a few hundred MB/s) and single-channel DDR4 memcpy are faster than the link, so the link dominates. Parallel loading across 9 GPUs (aggregate ~715 MB/s measured) would cut 44 GB to ~60–65 s. That needs a patch or 9 concurrent loaders (option, claim to test).
- **4 GB RAM.** mmap streams from the SSD and the page cache cannot hold the model, so a second load re-reads from disk. mlock is impossible. `--no-mmap` with 4 × 1 MB pinned buffers keeps RAM use tiny and is likely the most predictable path. The "temporary buffer sized like the tensor" path could allocate ~100+ MB per expert tensor, which is fine but adds RAM pressure next to 9 CUDA contexts, GNOME and Xorg (claim to test).
- **Disk capacity is a hard prerequisite.** The 64 GB SSD had ~26 GB free (51 % full) on 2026-10-22, so local copies are limited to ~24 GB with a safety margin, one model at a time: an ~18.7 GB 30B-A3B Q4 GGUF fits, but Q8_0 files and larger stretch models do not. Network storage for those would add its own throughput limit (e.g. ~110 MB/s on 1 GbE) and would then dominate the load time. This is outside this note's scope but bounds every experiment.

### Gaps
- No direct measurement of llama.cpp load time across many GPUs on x1 links was found.
- Whether the async pinned path serves all devices or only `bufs[0]`'s device was not fully resolved from the code excerpt.

---

## Q7. Concurrency and throughput: multiple instances vs one, parallel slots, MoE batching

### Takeaway
Aggregate throughput can scale far beyond batch-1. On a P102-100 a dense small model went from 78 t/s at B=1 to 493 t/s at B=64. Replicas beat splits when the model fits. On this rig the limits are:
- the 2 CPU cores, which every instance's launch thread shares;
- MoE expert-union growth, which weakens batching gains;
- `-sm tensor`, which stops scaling with more clients.

### Cited Findings
- P102-100 (sm_61), `llama-batched-bench`, Phi-4-mini Q4_K_M, S_TG at TG=128: B=1 78.46, B=2 125.2, B=4 158.4, B=8 185.1, B=16 408.0, B=32 478.0, B=64 493.1, B=128 427.0 t/s (Dec 14–15 2025) — [llama.cpp discussion #18030](https://github.com/ggml-org/llama.cpp/discussions/18030)
- Replicas versus splits (Qwen2.5-1.5B Q4_K_M, 4 × P100): DP=4 replicas reached 975 t/s aggregate against 419.7 single and 130–133 for 4-way splits. TP=2 × DP=2 gave 306 — [llamaR vignette](https://cran.nics.utk.edu/cran/web/packages/llamaR/vignettes/multi-gpu.html)
- On 3 × CMP 50HX, layer split with `--parallel 5` kept scaling (54.42 aggregate t/s at 5 clients), while tensor plateaued around 36 t/s and lost at ≥2 clients — [FASTCHIP repo](https://github.com/FASTCHIP/cmp50hx-llama-research)
- llama-server's default of 4 parallel slots divides context and VRAM. One user reports a clear single-user speedup after setting `--parallel 1` (forum anecdote) — [V2EX thread](https://fast.v2ex.com/t/1208365)
- Pipeline parallelism "requires many tokens to scale well" and favours batch throughput — [multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)

### Inferences
- **MoE batching (my calc).** The expected fraction of experts touched by B independent tokens in one layer is 1 − (1 − k/E)^B. For Qwen3-30B-A3B (k=8, E=128): B=1 6.3%, B=2 12.1%, B=4 22.8%, B=8 40.3%, B=16 64.4%.
  - Expert bytes per step grow ~3.6× from B=1 to B=4, so aggregate tg at B=4 is maybe ~1.5–2.5× B=1, versus ~3–4× for dense.
  - Attention, router and output-head bytes are shared and amortise well.
- **CPU limit.** Each llama.cpp instance has one main thread issuing launches (without graphs on Pascal), and there are 2 cores, also hosting GNOME/Xorg. Two instances (e.g. 4 + 4 GPUs, or 4 + 5) is probably the practical maximum, and each will be slower than when alone if it is launch-bound (claim to test, E11).
  - Batched decode in one instance amortises launches across sequences (same kernel count per step), so slots are more CPU-efficient than extra instances on this host.
- **Graphs and batching.** The `mul_mat_id` fallback path synchronises the stream. If batched MoE decode hits it, each MoE layer costs a host round trip, which is expensive with a slow CPU (claim to test).
- **Layout options (my calc).** A ~18.7 GB 30B-A3B Q4 needs ~4 GPUs, so 2 instances on 8 GPUs plus 1 spare GPU for a small model is one layout. One instance on all 9 GPUs with more slots and longer context is another.

### Gaps
- No MoE batched-bench data on Pascal was found.
- No data on several llama.cpp instances sharing a weak 2-core host was found.

---

## Q8. Hardware and driver tuning knobs (power, clocks, persistence, memory OC, P-states, ASPM)

### Takeaway
- Decode is memory-bound, so core clocks and power limits mostly affect pp. This is with the caveat that a 5-SM card may be partly issue-bound.
- Memory overclocking is the only knob that raises the decode ceiling. Mining profiles show ~+12–18% Ethash, a bandwidth proxy, from offsets around +400 (Windows units) / +800 (Linux units) on GP106. It carries real stability and silent-corruption risk.
- NVML application/locked memory clocks are generally unsupported on Pascal/Turing GeForce. CUDA forces P2 on GeForce. Whether the P106-090's P2 memory clock is lower than P0 must be checked.
- No source on ASPM's effect on GPU transfer latency was found.

### Cited Findings
- Core clock barely affects decode: on CMP 50HX, 1700 MHz gave −0.8% and 2100 MHz +1.1% — [FASTCHIP repo](https://github.com/FASTCHIP/cmp50hx-llama-research)
- Power limits: one blog swept power limits on dual 3090s in llama.cpp, measuring joules per token — [Jacques Mattheij](https://jacquesmattheij.com/llama-energy-efficiency/). An anecdote says a 4090 kept tg at ~40% power — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1tayu5t/snapshots/2026-05-15T23%3A40%3A01.25497Z). No Pascal power-sweep data was found.
- **P2 state.** On Pascal GeForce, starting a CUDA app puts the card in P2 with lower memory clocks ("from 11GHz to 9GHz for the 1080 and from 11GHz to 10GHz for the 1080Ti"). There is no NVreg option to disable P2 forcing for CUDA — [NVIDIA forum "Remove P2 forced state"](https://forums.developer.nvidia.com/t/remove-p2-forced-state-from-drivers/241998). The Windows-only "CUDA - Force P2 State" profile setting in NVIDIA Profile Inspector restores clocks — [HotHardware](https://www.hothardware.com/news/geforce-memory-clocks-gimped-with-discord-open)
- **Clock locking.** A GTX 1070 user got "unsupported" from nvidia-smi application-clock commands, citing NVML support for "Maxwell or newer" GeForce — [NVIDIA forum](https://forums.developer.nvidia.com/t/gtx-1070-nvidia-smi-p-states-and-application-clocks-support/51107). `nvmlDeviceSetApplicationsClocks` and `nvidia-smi lmc` return NVML_ERROR_NOT_SUPPORTED on 2080 Ti / 2060 Super, while "those APIs work on Ampere cards". HiveOS-type mining tools reportedly lock Pascal/Turing memory clocks at fixed points — [NVIDIA forum, Feb 2023](https://forums.developer.nvidia.com/t/lock-memory-clock-in-pascal-and-turing-cards/244093). On Maxwell, `nvidia-smi -ac 3600,1544` put a card in P3 — [Einstein@Home forum](https://einsteinathome.org/content/low-memory-clock-maxwell2-cards-960970980-probably-titan-x)
- **Overclocking tools on Linux:**
  - `nvidia-settings` with Coolbits; headless use of `GPUGraphicsClockOffsetAllPerformanceLevels` reported working on Pascal — [Folding@home forum](https://forum.foldingathome.org/viewtopic.php?p=306691)
  - `nvidia_oc` (Rust, NVML) has `set --mem-offset` in MHz, X11/Wayland, root, and a suggested systemd unit — [docs.rs nvidia_oc](https://docs.rs/nvidia_oc)
  - NVML memory offsets "secretly use the 2x DDR clock speeds", so Windows values roughly double on Linux. Newer `nvmlDeviceSetClockOffsets` (NVML v555) per-P-state behaviour was buggy for some users — [NVIDIA forum](https://forums.developer.nvidia.com/t/nvmldevicegetminmaxclockofpstate-nvmldevicesetclockoffsets-issues/318332)
- **Mining memory-OC profiles (community-submitted, unverified):**
  - P106-100: memory offset +300 (Linux) / +150 (Windows) at 90% power (~10.4 MH/s, 89 W); +600 / +300 at 100% — [Kryptex P106-100 low](https://www.kryptex.com/tr/overclocking/nvidia-p106-100-low-overclock); [Kryptex P106-100 medium](https://kryptex.com/overclocking/nvidia-p106-100-medium-overclock)
  - A "high" P106-100 profile lists +1000 (Linux) / +500 (Windows) at 86% power with a ~24 MH/s claim, which looks implausible next to the other profiles — [Kryptex P106-100 high](https://kryptex.com/cs/overclocking/nvidia-p106-100-high-overclock-2)
  - GTX 1060 6GB Samsung: 18.68 → 22.0 MH/s (+18%) with +400 (Windows) / +800 (Linux) memory, +50 core, 85 W — [Kryptex GTX 1060 Samsung](https://www.kryptex.com/overclocking/nvidia-gtx-1060-samsung-medium-overclock)
  - ASUS GTX 1060 6GB Micron: 20.03 → 22.35 MH/s (+12%) at +400. +440 raised it to 22.54 but produced artifacts — [Kryptex ASUS GTX 1060](https://www.kryptex.com/overclocking/asus-dual-geforce-gtx-1060-oc-6gb)
- No measured llama.cpp tg gain from VRAM overclocking was found on any card. Sources only argue that tg should scale with bandwidth — [llama.cpp discussion #21043](https://github.com/ggml-org/llama.cpp/discussions/21043)
- NCCL's P2P level defaults to PHB-level topologies where supported. Without P2P, SHM via host is used — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/2.8.3/env.html)

### Inferences
- **Memory OC is the only way to raise the decode ceiling (claim to test).** Ethash is close to pure memory bandwidth, so +12–18% Ethash suggests ~+10–15% effective bandwidth on GP106 GDDR5 at the edge of stability. tg would gain at most that.
  - Risks: GDDR5 instability can cause silent numerical corruption, driver Xid errors or hangs. A hang may need a reboot, and on the tested install, which uses passphrase-only LUKS disk encryption, every reboot needs someone at the rig's keyboard.
  - Corruption shows up as degraded perplexity or KL divergence before any crash, so validation must be numerical, not just "no crash".
- **Power limit (60–75 W).** For memory-bound tg, a lower limit probably costs little. For pp (dp4a-bound), throughput should fall roughly with core clock under the cap. If the 5-SM card turns out issue-bound in MMVQ, core clock and power limit will also matter for tg.
- **P2 memory clock on mining cards.** The 173 GB/s measured on the rig (90% of 192 GB/s theoretical) is a typical streaming efficiency. If that was measured under CUDA, P2 is probably not clamping memory on this card, but that must be confirmed by reading clocks during a llama.cpp run (read-only, E13).
- **Locked clocks.** `-lgc`/`-lmc` are likely unsupported on GP106 GeForce, extrapolating from the reports above (claim to test with read-only `nvidia-smi -q -d SUPPORTED_CLOCKS`). Offsets via Coolbits work under Xorg, which already runs on this rig.
- **Persistence mode** matters little while Xorg holds a context on all GPUs; the driver stays initialised.
- **ASPM (claim to test).** L0s/L1 exit latencies (µs-scale) on idle x1 links could add to every small transfer, i.e. each hop, the logits copy and all-reduces. Disabling ASPM via kernel policy is a reversible experiment, but runtime policy changes can destabilise some platforms.

### Gaps
- No Pascal power-limit versus tg/pp sweep was found.
- No VRAM-OC versus tg measurement was found on any card.
- No P106-090-specific OC data was found (VBIOS 86.06.5B.00.0C, memory vendor unknown).
- No source on ASPM's effect on CUDA small-transfer latency was found.
- The NVML documentation for `nvmlDeviceSetGpuLockedClocks` architecture support was not fetched.

---

## Q9. Real-world measurements from similar rigs (pp/tg, model, quant, GPU count, flags, date, source)

### Takeaway
Many single-card Pascal and mining-card numbers exist. True multi-GPU Pascal MoE numbers are scarce, and none use PCIe 1.0 x1 risers with nine cards. The closest analogues are the 4 × P100 and 8 × V100 tensor/layer results, the 3 × CMP 50HX study, and the 3090s on Gen2 x1. No P106-090 LLM benchmark was found.

### Cited Findings

| Hardware (count, link) | Model / quant | Mode / flags | pp t/s | tg t/s | Date | Source |
|---|---|---|---|---|---|---|
| P106-100 ×1 | Llama-2 7B Q4_0 | CUDA, FA off/on | 407 / 438 | 30.4 / 30.6 | scoreboard | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| P106-100 ×1 | Llama-2 7B Q4_0 | build 5688, FA off/on | 426 / 437 | 29.4 / 29.9 | 2025-08-07 | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| P106-100 ×1 | (Vulkan bench model) | Vulkan | ~184 | ~29.8 | ~2025 | [#10879](https://github.com/ggml-org/llama.cpp/discussions/10879) |
| GTX 1060 6GB ×1 | Llama-2 7B Q4_0 | FA off/on | 414 / 444 | 27.7 / 28.1 | 2025-08-11 | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| Quadro P1000 ×1 (5 SMs) | Llama-2 7B Q4_0 | FA off/on | 183 / 174 | 14.0 / 13.7 | scoreboard | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| GTX 1050 Ti ×1 | Llama-2 7B Q4_0 | FA off/on | 267 / 268 | 19.1 / 20.3 | 2026-06-24 | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| Tesla P40 ×1 | Llama-2 7B Q4_0 | FA off/on | 1007 / 1080 | 54.7 / 53.7 | scoreboard | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| Tesla P100 ×1 | Llama-2 7B Q4_0 | FA off/on | 761 / 787 | 58.4 / 62.0 | scoreboard | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| GTX 1080 Ti ×1 | Llama-2 7B Q4_0 | FA off/on | 1084 / 1138 | 62.5 / 61.4 | scoreboard | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013) |
| CMP 170HX ×1 (PCIe 1.1 x4 native) | Llama-2 7B Q4_0 | FA off/on | 2699 / 3016 | 141 / 156 | 2026-07-30 | [#15013](https://github.com/ggml-org/llama.cpp/discussions/15013); link: [arXiv:2505.03782](https://arxiv.org/pdf/2505.03782) |
| CMP 40HX ×1 | Llama-2 7B Q4_0 | CUDA | ~144 | ~80 | 2026 | [#24616](https://github.com/ggml-org/llama.cpp/issues/24616) |
| GTX 1080 ×1 | granite-3.1-3b-a800m Q4_K_M (MoE) | graphs off → on (PR #27721) | – | 81.2 → 124.5 (d0) | 2026-08-26 | [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) |
| GTX 1080 ×1 | OLMoE-1B-7B Q4_K_M | graphs off → on | – | 150.5 → 169.5 | 2026-08-26 | [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) |
| GTX 1080 ×1 | llama-bench run in PR description (model not restated in the PR summary; likely Ling-3.0-tiny Q6_K) | graphs off → on | 1935 → 1909 (pp512) | 41.5 → 73.9 (tg64) | 2026-08-25 | [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) |
| P102-100 ×1 | Ling-3.0-tiny Q8_0 (MoE) | graphs off → on, CUDA 12.9, drv 580 | 2457 → 2455 | 70.8 → 88.4 | 2026-08 | [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) |
| P102-100 ×1 | Phi-4-mini Q4_K_M | batched-bench B=1 → 64 | – | 78.5 → 493 (aggregate) | 2025-12-15 | [#18030](https://github.com/ggml-org/llama.cpp/discussions/18030) |
| P104-100 ×1 (+CPU expert offload) | Qwen3.6 35B-A3B / 28B-A3B (MoE) | offload, `-ctk q8_0` | ~51 | ~20 | 2026 | [CSDN (search summary)](https://blog.csdn.net/laot007/article/details/161087262) |
| Tesla P40 ×1 | Qwen3-Coder-30B-A3B Q4_K_M | LocalScore | 288 | 29.3 | n/a (unverified snippet) | [LocalScore P40](https://www.localscore.ai/accelerator/1729) |
| P100 ×4 (sm_60) | Qwen3.6 35B-A3B Q8_0 | layer / tensor (no graphs) / tensor (graphs) | – | 61.4 / 39.6 / 70.6 | 2026-08 | [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) |
| P100 ×4 (Vulkan) | Qwen2.5-1.5B Q4_K_M | 1 GPU / 2 layer / 4 layer / 4 replicas | – | 420 / 150 / 133 / 975 aggr. | n/a | [llamaR vignette](https://cran.nics.utk.edu/cran/web/packages/llamaR/vignettes/multi-gpu.html) |
| V100 ×8 (SXM2) | Qwen3.5 122B-A10B Q8_0 | layer graphs off → on | – | 41.9 → 47.7 | 2026-07 | [PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749) |
| V100 ×8 | Qwen3.5 122B-A10B Q8_0 | tensor graphs off → on | (pp slow) | 17.5 → 58.3 | 2026-07 | [PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749) |
| CMP 50HX ×3 (Gen2 x8, P2P 2.47 GiB/s) | Qwen3.8-27B UD-Q4_K_XL (dense) | layer / tensor | ~500 (p32k) | 25.2 / 39.3 (short) | 2026-09-22 | [FASTCHIP](https://github.com/FASTCHIP/cmp50hx-llama-research) |
| CMP 50HX ×3 | Qwen3.8-Flash-Next Q2_0 (125B, 6B active) | "Strata" engine | 2,100–2,400 | 80–82 (short), 62 @100k | 2026-07+ | [FASTCHIP](https://github.com/FASTCHIP/cmp50hx-llama-research) |
| RTX 3090 ×2 (Gen2 x1 + x1, no P2P) | Gemma-4 26B-A4B Q8_0 | layer, FA, b2048/ub512 | (−9–13% vs x2+x1) | 95.75 | 2026-07-31 | [note.com](https://note.com/agentworkflowlab/n/na17f86500825) |
| 3×3090 (Gen2 x2/x1/x1) + 4090 (Gen3 x16) | Laguna-S-2.1 118B-A8B UD-Q4_K_S | layer, 4 GPUs | 2,227 (d0) | 88.6 (d0) / 85.8 (d8192) | 2026-07-31 | [note.com](https://note.com/agentworkflowlab/n/na17f86500825) |
| RTX 4090 ×2 (PCIe 4.0 x16) | qwen35moe 35B-A3B Q4_0 | layer / tensor | – | 205.7 / 141.4 | 2026-04 | [PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378) |
| A100 ×1 vs ×8 (PCIe) | Llama 7B Q4_0 | layer (pipeline PR) | (pp scales ~4–5×) | 145 vs 135 | 2024-03 | [PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017) |

### Inferences
- **For a 3B-active MoE on Pascal**, the single-card numbers bracket expectations. A P40 (~2× P106 bandwidth) does ~29 t/s on Qwen3-Coder-30B-A3B in an unverified LocalScore entry. A GTX 1080 does 50–170 t/s on small MoEs, launch-bound.
- **Layer-split tg barely changes with link width** (3090 x1 data), so the PCIe 1.0 x1 risers matter mainly for TP/EP, loading and logits, not for layer-split tg.
- **Every multi-GPU result that beats layer split at batch 1** (V100 ×8, P100 ×4 with graphs, CMP 50HX ×3) uses x8–x16 links, fast hosts and, except CMP, CUDA graphs. Nothing in the table resembles this rig's x1 links plus 2-core host combination.

### Gaps
- No published multi-GPU numbers for P104-100, P106-100 or P106-090 rigs, and no 6+ GPU x1-riser LLM rigs, were found in English or Chinese sources searched (CSDN, Zhihu, Toutiao, V2EX). Bilibili videos were not reachable as text.
- No GTX 1070/1080 multi-GPU llama.cpp MoE numbers were found.
- CMP 30HX/90HX LLM numbers were not found.

---

## Q10. Bottleneck model (formulas + rough numbers for 3B-active and 10B-active MoE at ~Q4)

### Takeaway
At batch 1 with layer split, per-token time is roughly max(GPU bandwidth time, CPU launch time) plus serial extras (logits, hops, graph build, sampling).
- **3B-active:** CPU/launch and GPU time are of similar size, so CUDA graphs and backend sampling are the big software levers.
- **10B-active:** GPU-bandwidth-bound, so memory bandwidth (MMVQ efficiency, memory OC) is the lever.

All numbers below are my estimates from the cited inputs, and all are claims to test.

### Inferences
**Formulas (layer split, batch 1, G GPUs used)**
- Active bytes per token: `A = Σ(active weight params × bpw/8) + KV_bytes(ctx)`.
  - For Qwen3-30B-A3B: attention ≈ 0.906 B params, routed experts 8 × 3 × 2048 × 768 × 48 ≈ 1.81 B, output head 2048 × 151,936 ≈ 0.31 B (Q6_K in Q4_K_M).
  - So `A ≈ 2.73 B × 4.9/8 + 0.31 B × 6.56/8 ≈ 1.67 + 0.25 ≈ 1.9 GB`.
  - KV ≈ 96 KB × ctx (F16).
- `T_gpu = A / BW_eff`, with BW_eff ∈ {173 GB/s peak, ~115 GB/s if like P106-100 (61% of 192), ~80 GB/s pessimistic 5-SM case}.
- `T_launch = N_nodes × t_node`, N_nodes ≈ 30–40 × n_layers, t_node ≈ 6–12 µs on this CPU without graphs (≈1–3 µs with graphs, guess).
- `T_hop ≈ 2 × (t_fixed ≈ 10 µs + S_act / ~0.2 GB/s)`, S_act = hidden × 4 B → ~0.1 ms at hidden 2048.
- `T_logits ≈ vocab × 4 B / 0.213 GB/s` → ~2.9 ms at 152k vocab (≈0 with `--backend-sampling` greedy/top-k).
- `T_token ≈ max(T_gpu, T_launch) + T_build + T_logits + T_sample + (G−1)·T_hop`.
- TP (N GPUs): `T ≈ T_gpu/N + 2·n_layers·T_allreduce + N·T_launch` (no graphs), with T_allreduce ≈ 0.1–0.3 ms.
- EP: `T ≈ T_attn+shared + T_experts × (max_load/k) + n_moe_layers·(T_bcast + T_gather)`.

**3B-active (Qwen3-30B-A3B, Q4_K_M ≈ 18.7 GB file → 4 GPUs; ctx ≈ 0–4k)**

| Term | Estimate |
|---|---|
| A | ~1.9 GB (+0.4 GB at 4k ctx F16 KV) |
| T_gpu | 11 ms (173 GB/s) / 17 ms (115) / 24 ms (80) |
| T_launch (no graphs) | ~1,700 nodes × 6–12 µs ≈ 10–20 ms |
| T_logits | ~2.9 ms |
| Hops | 3 × 0.1 ≈ 0.3 ms (4 GPUs); 8 × 0.1 ≈ 0.8 ms (9 GPUs) |
| T_build + T_sample | unknown, guess 1–4 ms on the Celeron |
| **Total** | **~20–30 ms → ~33–50 t/s** (upper bound ~60–75 t/s if graphs + backend sampling make it bandwidth-bound at 115–173 GB/s) |

- 4 vs 9 GPUs: ~0.5 ms difference (~2–3%).
- TP on 4 GPUs: T_gpu/4 ≈ 4 ms + ~96 × 0.1–0.3 ms ≈ 10–29 ms + 4 × 10–20 ms launches ≈ 54–113 ms → ~9–18 t/s without graphs. With graphs ≈ 14–35 ms → at best on par with layer split.
- EP over 9 GPUs: saves ~6 ms of expert time, costs ~7–14 ms → ≤ break-even.

**10B-active (~100B total; ~3.5 bpw needed to fit ~44 GB in 9 × ~5.5 GB, so "Q4" active bytes are an upper bound)**

| Term | Estimate |
|---|---|
| A | ~4.4 GB at 3.5 bpw; ~6.0 GB at ~4.8 bpw (Q4_K_M-like; that model size would only fit at ~80B total) |
| T_gpu | 25–35 ms (173) / 38–52 ms (115) / 55–75 ms (80) |
| T_launch | ~48–64 layers × 30–40 nodes ≈ 1.5–2.6k nodes × 6–12 µs ≈ 9–31 ms, mostly hidden under T_gpu |
| T_logits | ~2.9 ms (assuming ~150k vocab) |
| Hops | 8 × 0.1–0.2 ≈ 0.8–1.6 ms (hidden 4096 → 16 KB) |
| **Total** | **~42–58 ms (3.5 bpw, 115 GB/s) → ~17–24 t/s**; ~13–18 t/s at Q4-like active bytes; ~25–35 t/s if BW_eff ≈ 173 GB/s |

- Memory OC of +10% would gain ≈ +8–9% here, because the GPU term dominates.
- EP might help this class modestly: bigger experts, so compute saved per sync is larger (claim to test).
- TP over 9 GPUs: ~96–130 all-reduces × 0.15–0.4 ms (switch contention) ≈ 15–50 ms + 9× launches → expected loss.

**Prefill (both models).** Compute-bound at ~200–230 t/s pp512 per P106-090 for 7B-dense-equivalent work. For a 3B-active MoE roughly ~150–300 t/s per card-equivalent, with ~2–3.5× pipeline scaling across 4 GPUs given enough ubatches in flight. Activation hops (~4 MB per ubatch per hop, ~20 ms) are a small fraction of stage time.

**Load.** ~model bytes / 0.19 GB/s (sequential loader): ~100 s (18.7 GB), ~230 s (44 GB).

### Gaps
- The biggest uncertainties are t_node on this CPU, BW_eff of MMVQ on 5 SMs, and real host-staged hop latency. Each changes the 3B-active estimate by ±30–50%.

---

## Q11. Experiment ideas (options, not decisions)

### Takeaway
These are ordered roughly from cheapest and most informative to riskiest. E1–E6 calibrate the bottleneck model with no hardware risk. E14 (memory OC) and E15 (ASPM) carry real risk and should run only after a deliberate decision. Changing clocks or power, stopping GDM, or editing xorg.conf all change the rig's state, unlike the read-only inspection the other experiments need.

### Inferences
- **E1. Single-GPU calibration (no risk).**
  - What: `llama-bench -sm none -mg N -ngl 99 -fa 1` on one P106-090 with Llama-2-7B Q4_0 (direct comparison with P106-100's ~30.6 tg / ~438 pp), then the same model in Q4_K_M, IQ4_XS and Q8_0, plus a small MoE that fits in 6 GB (OLMoE-1B-7B Q4_K_M or granite-3.1-3b-a800m, as used in PR #27721).
  - Measure: tg/pp; effective GB/s = file bytes × tg; repeat on 2–3 different GPUs.
  - Expected: if tg ≈ 28–31 t/s on 7B Q4_0, the 5-SM concern is moot. If ≈ 16–22 t/s, MMVQ is issue- or MLP-limited, and quant choice and core clock matter more.
- **E2. Is decode CPU-bound? (no risk; the optional GDM stop is low risk).**
  - What: run E1's MoE with `-t 1` and `-t 2`; with a CPU hog on core 1 versus none; optionally with GNOME/GDM stopped (a change to the rig's state; does not reboot).
  - Measure: tg, `nvidia-smi dmon -s u` GPU utilisation, `perf top` on the llama thread.
  - Expected: if tg moves with CPU contention and GPU utilisation is well below 100%, decode is launch-bound.
- **E3. CUDA graphs on sm_61 (software only).**
  - What: build llama.cpp with PR #27721's one-line gate change (`GGML_CUDA_CC_VOLTA` → `GGML_CUDA_CC_DP4A`). A/B with `GGML_CUDA_DISABLE_GRAPHS=1`, on a single GPU, then layer split over 2/4/9 GPUs, with dense and MoE models.
  - Measure: tg at depth 0 and 4096, perplexity or KL-divergence identity, VRAM delta.
  - Expected: +10–50% tg for MoE (GTX 1080 / P102-100 precedent), possibly more on this CPU. Watch the known graph VRAM leak in `-sm tensor`.
- **E4. Host-staged hop and copy microbenchmark (no risk).**
  - What: small CUDA program built with `/usr/local/cuda-12.9/bin/nvcc -arch=sm_61`. Times `cudaMemcpyPeerAsync` (no peer access) for 4 KB / 8 KB / 16 KB / 608 KB / 4 MB between pairs (same switch PIX vs cross-switch PHB vs GPU4), and explicit pinned D2H + H2D. Repeat with 1, 4 and 9 pairs concurrently.
  - Measure: event-timed and wall-clock latency, per-size bandwidth, whether a peer copy blocks other streams.
  - Expected: ~50–200 µs for 8 KB, ~3 ms for 608 KB; contention behind the ASM1184e when concurrent.
- **E5. Layer-split GPU-count sweep (no risk).**
  - What: one model that fits on 4 GPUs, run on 4, 5, 6 and 9 GPUs (`CUDA_VISIBLE_DEVICES`/`-ts`), keeping the same GPUs per switch where possible.
  - Measure: tg; the slope per added GPU ≈ T_hop plus scheduler overhead.
  - Expected: ~0.1–0.3 ms per added GPU. Confirms or refutes "use the minimum GPUs".
- **E6. Logits and sampling cost (no risk).**
  - What: `--backend-sampling` (PR #17004) versus default; greedy versus the default sampler chain.
  - Measure: tg.
  - Expected: −2 to −4 ms per token for 152k vocab, i.e. +10–20% tg on 3B-active if E2 shows it is not purely launch-bound.
- **E7. `-sm tensor` vs `-sm layer` (no risk).**
  - What: Qwen3-MoE (supported) on 2 GPUs on the same switch, 2 across switches, and 4; with and without NCCL installed; with and without the E3 graph patch. FA on, F16 KV.
  - Measure: tg/pp, CPU utilisation.
  - Expected: tensor ≈ 0.3–0.7× layer without graphs; closer with graphs. The cross-switch pair may beat the same-switch pair because the switch upstream is not shared.
- **E8. EP feasibility microbenchmark (no risk).**
  - What: extend E4 to "broadcast 8 KB to k GPUs + gather k × 8 KB" with host staging, k = 2…8, across switch layouts. Compare with the per-layer `mul_mat_id` kernel time from `nsys` (or `GGML_CUDA` op timing) on one GPU.
  - Measure: per-layer sync cost against expert compute saved.
  - Expected: sync ≥ savings for 3B-active. Gives a go/no-go number before any EP coding.
- **E9. Prefill pipeline scaling (no risk).**
  - What: `llama-bench -p 512,2048,8192` with `-ub 128/256/512` and `-b 2048/4096/8192` on 1 (small model), 2, 4 and 9 GPUs. Optionally rebuild with more scheduler copies (`GGML_SCHED_MAX_COPIES=8`; verify the option name).
  - Measure: pp scaling efficiency, PCIe RX/TX via `nvidia-smi dmon -s t`.
  - Expected: 2–3.5× on 4 GPUs for long prompts; worse with default `-b`.
- **E10. Load path (no risk; needs the model file on local disk, ≤~24 GB, or on network storage).**
  - What: time model load with `--load-mode mmap` versus `none`/`--no-mmap`, with caches dropped, watching per-GPU PCIe throughput.
  - Measure: total seconds, which GPU is active when.
  - Expected: sequential, ~190 MB/s aggregate. Quantifies the value of a parallel-load patch (~3–4× faster for big models).
- **E11. Throughput layouts (no risk).**
  - What: (a) one instance on 9 GPUs with `-np 1/2/4/8` (`llama-batched-bench` and llama-server); (b) two instances on 4 + 4 GPUs, each `taskset` to one core; (c) (b) plus a small model on GPU 8.
  - Measure: aggregate and per-request t/s, CPU %.
  - Expected: slots scale better per CPU core than instances. MoE gains are sublinear (expert union 23% at B=4, 40% at B=8 for 128/8).
- **E12. Power-limit sweep (low risk; within the 60–75 W allowed range; changes the rig's state).**
  - What: `nvidia-smi -pl 60/65/70/75` on one GPU, running E1.
  - Measure: tg, pp, W, clocks, throttle reasons.
  - Expected: tg nearly flat, pp drops at 60 W, unless the 5-SM card is issue-bound.
- **E13. P-state and clock inspection (read-only).**
  - What: `nvidia-smi -q -d PERFORMANCE,CLOCK,SUPPORTED_CLOCKS,POWER` during a llama.cpp run; check P-state, memory clock versus max, and clocks_throttle_reasons; `nvidia-smi -q -d SUPPORTED_CLOCKS` to see whether `-lgc`/`-ac` are possible.
  - Expected: confirms whether CUDA P2 lowers the memory clock on this mining VBIOS.
- **E14. Memory overclock — RISKY TO STABILITY AND DATA INTEGRITY. Run only after a deliberate decision.**
  - What: enable Coolbits (xorg.conf edit, a persistent config change), then on ONE GPU raise `GPUMemoryTransferRateOffset` in +100 steps (Linux units ≈ 2× Windows), up to about the mining-profile range (+600–800 Linux). At each step run a bandwidth test, `llama-bench` tg, and a perplexity or KL-divergence check against the stock baseline. Stop at the first tg regression, numerical drift, artifact or Xid in `dmesg`.
  - Expected: ≤ +10–15% tg if bandwidth-bound.
  - Risks: silent corruption; driver hang needing a reboot, which on the tested install (passphrase-only LUKS) needs someone at the keyboard to type the passphrase; long-term memory wear on aged mining cards.
  - Alternative: `nvidia_oc --mem-offset` (NVML) has the same risks.
- **E15. PCIe ASPM — MODERATE SYSTEM RISK. Run only after a deliberate decision.**
  - What: read-only first: `lspci -vvv` LnkCap/LnkCtl ASPM state for the GPUs and the ASM1184e ports, and `/sys/module/pcie_aspm/parameters/policy`. If ASPM is active, an option is to set policy `performance` at runtime (reversible) and repeat E4/E5.
  - Expected: lower small-transfer latency if L1 is engaged.
  - Risk: some platforms misbehave when ASPM changes at runtime. The boot-time `pcie_aspm=off` route needs a reboot (on the tested install, with the LUKS passphrase typed at the keyboard).
- **E16. KV type and depth (no risk).**
  - What: F16 versus q8_0 KV at depth 0/4k/16k with `-sm layer`.
  - Expected: tg falls with depth as KV bytes approach weight bytes (96 KB per position per token for Qwen3-30B-A3B F16). q8_0 KV recovers part of that, but is incompatible with `-sm tensor`.

### Gaps
- PR #27721 is unmerged, so E3 requires a local patch. Re-check its status before building.
- Disk space (~26 GB free on the 64 GB SSD on 2026-10-22) limits E5–E11 to one model of up to ~24 GB at a time; larger files, such as Q8_0 of 30B-class models, need network storage.
