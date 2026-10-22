# llama.cpp-family inference engines for MoE on 9 x P106-090 (Pascal sm_61, PCIe 1.0 x1, no P2P), as of 2026-10-22

Scope: mainline llama.cpp (ggml-org), ik_llama.cpp, Pascal patch forks, wrappers. All sources were read on 2026-10-22 unless a date is given. "Claim to test" marks anything not confirmed by a primary source or not measured on sm_61. The rig's facts (5 SMs per card, ~173 GB/s, FP16 at 1/64 rate, dp4a, 4 GB host RAM with 3.7 GiB usable, Celeron 3865U with no AVX, PCIe 1.0 x1, no P2P) are in the [hardware spec](../../mining_rig.md) and are not re-cited here.

Version anchors for mainline:
- The tested rig's build is commit 85d5703a3 from 2026-09-04. That predates v0.5.0 (2026-09-23) and v0.6.0 (2026-10-05, nightly b11429, bundles ggml v0.26.0). The newest nightly on 2026-10-22 is b11815. Sources: [v0.6.0 release](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0), [AICoder news on v0.6.0, secondary](https://aicoder.com/news/news-20261006-llamacpp-v0-6-0-glm-5-3-flash-batch-ext-metal), [releases page](https://github.com/ggml-org/llama.cpp/releases).
- Mainline now uses version labels (0.x.y) alongside the bNNNNN build numbers. Issues quote strings like "0.3.0-dev, build 10793" ([#30235](https://github.com/ggml-org/llama.cpp/issues/30235)) and "0.6.0-dev (build 33, commit 1623d8c)" ([#30334](https://github.com/ggml-org/llama.cpp/issues/30334)).

---

## 1. Mainline llama.cpp: is Pascal/sm_61 still supported, and which CUDA code paths does it take?

### Takeaway
Mainline still compiles and runs on sm_61 with a CUDA 12.x toolkit. No source announces a removal, but Pascal is not a CI-tested target either: Pascal bugs sit unanswered as "bug-unconfirmed", and the practical threat is the toolchain (CUDA 13 cannot target Pascal), not the code. On sm_61, matmuls take MMVQ at batch 1–8 and dp4a MMQ above that, and FlashAttention takes FP32 vec/tile kernels because sm_61 has no fast FP16. CUDA graphs are probably inactive on Pascal (pre-Ampere check), which matters a lot with a weak host CPU (claim to test).

### Cited Findings
**Toolchain and support status**
- CUDA 13.0 removed offline compilation for sm_61 (and Maxwell/Volta), so Pascal needs CUDA 12.x. The P40 report used CUDA 12.9.2 — [llama.cpp discussion #25978, 2026-07-21](https://github.com/ggml-org/llama.cpp/discussions/25978); also [ik_llama.cpp docs/pascal-volta.md, 2026-10-13](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md). The ik doc calls CUDA 12.9 plus the 580 driver branch the last pairing that supports these GPUs.
- CUDA 12.9 fails to compile on glibc 2.41+ (`sinpi`/`cospi`/`rsqrt` noexcept conflict) and needs a header patch. No 12.x release fixes it — [#25978](https://github.com/ggml-org/llama.cpp/discussions/25978). (Ubuntu 24.04 ships glibc 2.39, so the rig itself is unaffected. A separate, more powerful PC on the same LAN (a "workstation") running a newer distribution, or a Docker base image with glibc 2.41+, may hit it.)
- In a Docker build, the CUDA backend must link the `libcuda` stub, or linking fails with `undefined reference to 'cuMemCreate'` — [#25978](https://github.com/ggml-org/llama.cpp/discussions/25978).
- Mainline's CMake adds Pascal (`61-virtual`) to the default architecture list only when the toolkit is older than CUDA 13. Release b10472 (2026-08-17) shipped Windows CUDA 12.4 and 13.x builds but no Linux CUDA asset — [localaimaster, 2026-09-20 (inferred from CMake, not a binary check)](https://localaimaster.com/blog/pascal-gpus-after-cuda-13). By 2026-10-22, every nightly ships Linux CUDA 12.8 and 13.4 builds and Windows CUDA 12.4 and 13.4 builds — [releases page](https://github.com/ggml-org/llama.cpp/releases). Whether the Linux CUDA 12.8 asset contains sm_61 SASS or PTX is a claim to test.
- No official llama.cpp statement on Pascal support in 2026 was found. Several recent Pascal bugs are open with no maintainer reply:
  - [#30334 (2026-10-11)](https://github.com/ggml-org/llama.cpp/issues/30334): MMVQ fusion check misses cc 6.1.
  - [#30235 (2026-10-09)](https://github.com/ggml-org/llama.cpp/issues/30235): Pascal MMQ tile config.
  - [#25593 (2026-07-12)](https://github.com/ggml-org/llama.cpp/issues/25593): sm_60 FP16 accuracy. It was marked stale on 2026-09-26 and un-staled on 2026-10-09.
- Downstream packagers are dropping Pascal from prebuilt binaries:
  - LM Studio 0.4.21's CUDA backend reportedly no longer supports sm_61 (open issue, no staff reply) — [lmstudio-bug-tracker #2316, 2026-08-22](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2316).
  - Docker Model Runner's bundled llama.cpp excludes Pascal — [docker/model-runner #929](https://github.com/docker/model-runner/issues/929).

**Matmul paths on sm_61**
- The ik_llama.cpp Pascal document (a fork, but its CUDA backend descends from mainline's) gives these rules:
  - n = 1–8 uses MMVQ.
  - n > 8 uses MMQ on sm_61 and sm_70, which have dp4a (cc ≥ 610).
  - sm_60 (P100) has no dp4a, so it dequantizes to f32 and runs cuBLAS SGEMM. `GGML_CUDA_FORCE_MMQ` does not affect sm_60.
  - F16 weights on Pascal are converted to f32 for SGEMM. Types outside the MMQ/MMVQ sets fall back to cuBLAS.
  - Source: [ik docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md).
- Mainline build doc definitions — [docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md):
  - `GGML_CUDA_FORCE_MMQ` (default false) "forces custom quantized matmul kernels instead of FP16 cuBLAS, even without int8 tensor core support". It lowers VRAM use but slows large batches.
  - `GGML_CUDA_FORCE_CUBLAS` (default false) forces FP16 cuBLAS for quantized models. It may overflow and uses more memory.
- The P40 user built with `-DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_F16=OFF`. The reasoning: dp4a INT8 is about 47 TOPS on a P40, against about 183 GFLOPS FP16 at 1/64 rate — [#25978](https://github.com/ggml-org/llama.cpp/discussions/25978). The 8×P40 user's system_info line shows `ARCHS = 610 | FORCE_MMQ = 1 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | FA_ALL_QUANTS = 1` — [discussion #19248, 2026-02-01](https://github.com/ggml-org/llama.cpp/discussions/19248).
- Mainline has a per-architecture MMQ config file, `ggml/src/ggml-cuda/mmq-config-pascal-dp4a.cuh`:
  - It holds entries for 22 quant types, all using tile width I=64.
  - Raising I to 128 for Q8_0 only gave +12.1% pp512 on a Tesla P4 (sm_61). Applying I=128 to all types regressed Q4_K_M by 24.4% and Q6_K by 51.7%. I=160 regresses, and I ≥ 192 fails to compile (Pascal shared-memory limit). Token generation was unchanged (−0.3%).
  - The issue is open with no reply.
  - Source: [#30235, 2026-10-09; re-checked on b11476](https://github.com/ggml-org/llama.cpp/issues/30235).
- A third-party patch says the MMQ code "was restructured between June and October 2026". Upstream's Pascal tiles were 8/16/32/64 rows, and the patch sets all 242 rows to a 96-row tile. Source: [madsteph74/pascallama.cpp](https://github.com/madsteph74/pascallama.cpp). An Oct-2026 downstream PR notes the MMQ API changed (`ggml_cuda_mmq_get_config` gained `prec_src1`) — [PolyStrata PR #19](https://github.com/VecSzn/PolyStrata/pull/19).
- MMVQ fusion bug on cc 6.1:
  - `ggml_cuda_should_fuse_mul_mat_vec_q` excludes Pascal with `cc <= GGML_CUDA_CC_PASCAL` (600). P40/P106 (610) are therefore not excluded, and fusion stays on.
  - Effect on a P40 with fusion off: tg128 Mixtral Q2_K −5.5%, Mixtral Q3_K_M +5.3%, Llama Q4_0 +4.7%, Llama Q4_K_M −1.2%.
  - Fusion off plus forced small-K for Q2_K MUL_MAT_ID gave +12% tg on Mixtral Q2_K.
  - First bad commit: f77c13b (PR #16715). Status: open, unconfirmed.
  - Source: [#30334, 2026-10-11](https://github.com/ggml-org/llama.cpp/issues/30334).
- v0.6.0 (2026-10-05) CUDA changes relevant to MoE — [v0.6.0 release](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0):
  - Fused shared experts into MMVQ (#29184).
  - Fixed an MMQ memory fault when expert count far exceeds ubatch size (#29941).
  - Added a model-driven W4A4 NVFP4/MXFP4 path.
  - Changed CUDA FA scheduling and tuned the FP16 tile FA configs (#26289).
  - None of the changes is Pascal-specific.
- `GGML_CUDA_MMQ_PREC` (`auto`/`q8`/`q4`) overrides activation precision for NVFP4/MXFP4 matmuls — [build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md). Its effect on sm_61 MXFP4 (gpt-oss) is not documented.

**FlashAttention on sm_61**
- `fast_fp16_available` is false on sm_61 and true on sm_60 and sm_70. sm_61 "always uses the f32 kernels". Kernels by head size (ik fork, inherited from mainline's tile/vec kernels):
  - 64 and 128: vec_f32 for n_q ≤ 8, tile_f32 for n_q > 8.
  - 256: vec_f32, then tile_f32.
  - Other head sizes fall to the CPU in ik.
  - Source: [ik docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md).
- In mainline, sm_61 is excluded from the fast-FP16 path. A proposed fix copies that exclusion to sm_60, where FP32 math was silently done in FP16 — [#25593](https://github.com/ggml-org/llama.cpp/issues/25593). The cc 610 exclusion means `GGML_CUDA_F16`-style FP16 paths should not engage on P106 anyway. Whether the `GGML_CUDA_F16` CMake option still exists in Oct 2026 is unclear: build.md no longer lists it.
- Mainline PR #26289 tunes the FP16 tile FA configs for head sizes 40–112 (+14.5% mean kernel speedup on P100). That is the fp16 path, which sm_61 does not take — [PR #26289](https://github.com/ggml-org/llama.cpp/pull/26289).
- Which FA K/V type pairs get compiled is now set by `GGML_CUDA_FA_QUANTS`, with default `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`. Uncompiled combinations fall back to f16-f16 with a warning. `GGML_CUDA_FA_ALL_QUANTS` is a deprecated alias for `=all` — [build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
- `-fa` defaults to `auto`. It is required for quantized V cache and for `-sm tensor` — [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md).

**CUDA graphs on Pascal**
- The original CUDA graphs PR (#6766) disabled graphs for GPUs older than Ampere (`cc < CC_AMPERE` sets `disable_due_to_gpu_arch`). A maintainer said the reason was performance: one Turing user found graphs "a bit slower" — [PR #6766 via semanticdiff](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/6766/overview), [NVIDIA blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs).
- ik_llama.cpp's 2026-10-13 code still has "CUDA graphs are not used below Ampere (`cc < CC_AMPERE` in `ggml-cuda.cu`)" — [ik docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md).
- Mainline PR #26802 (merged 2026-08-11) narrowed a guard that disabled CUDA graphs for some MoE models — [TurinTech write-up](https://www.turintech.ai/blog/inside-llama-cpp-pr-26802). It does not touch the architecture check.

**Quant types with fast Pascal kernels**
- The Pascal dp4a MMQ config has entries for 22 types — [#30235](https://github.com/ggml-org/llama.cpp/issues/30235). Which 22 is not listed in the issue.
- On a Tesla P4, Q8_0 benefits from wider tiles while Q4_K/Q6_K regress — [#30235](https://github.com/ggml-org/llama.cpp/issues/30235).
- On P40, Q2_K and Q3_K_M MoE decode depend on the MMVQ fusion and small-K settings — [#30334](https://github.com/ggml-org/llama.cpp/issues/30334).

### Inferences
- `GGML_CUDA_FORCE_MMQ=ON` is probably a no-op on sm_61: there are no FP16 tensor cores, so MMQ is already chosen. `GGML_CUDA_FORCE_CUBLAS=ON` would probably route batched matmuls to f32 SGEMM after dequantization (slower, more VRAM). Both are claims to test.
- If CUDA graphs are disabled pre-Ampere in mainline too, the rig's `GGML_CUDA_GRAPHS=ON` does nothing. Each decode step then pays one CPU-side launch per kernel on a 1.8 GHz 2-core Celeron. For a ~48-layer MoE that is on the order of 1,000+ launches per token (estimate). Host-side launch and sync overhead could cap batch-1 tg well below the bandwidth limit. Claim to test (check the debug log or nsys).
- Pascal kernel tuning in mainline is drifting: tables restructured mid-2026 and the fusion gates are inconsistent (≤600 versus <700). Pinning a known-good commit and benchmarking before each bump looks prudent.
- With only 5 SMs per P106-090, MMQ tile/occupancy choices tuned on 30–60-SM cards may be badly off. Small-batch pp (ub ≤ 64) could be dominated by launch and tail effects. Claim to test.

### Gaps
- No primary statement from ggml-org maintainers on Pascal's status (supported, best-effort, or planned removal) was found for 2025–2026.
- The exact mainline list of quant types with MMQ/MMVQ on dp4a (IQ1_S…IQ4_XS, MXFP4) and their relative speeds on sm_61 was not found. MXFP4 on Pascal (gpt-oss) is not documented.
- Whether mainline (not ik) still has the `cc < Ampere` CUDA-graphs gate in Oct 2026 was not verified from source.
- Whether `GGML_CUDA_F16` and `GGML_CUDA_FA` are still CMake options in Oct 2026: build.md omits them.

---

## 2. Multi-GPU: split modes, pipeline parallelism, inter-GPU copies without P2P, CUDA graphs, NCCL

### Takeaway
Mainline has three useful modes:
- `none`.
- `layer` (default; pipeline parallel; tolerates slow links).
- `tensor` (experimental tensor parallel via a "meta" backend with AllReduce; needs FA and unquantized KV; not compatible with `--fit`; bottlenecked by interconnect).

`row` is deprecated. Layer split does not speed up batch-1 decode; pipeline parallelism helps prompt processing only. P2P is now opt-in (`GGML_CUDA_P2P`), so on this rig cross-GPU copies stage through host RAM by default. NCCL matters only for `-sm tensor`. No source tests tensor parallelism over PCIe x1 with nine GPUs.

### Cited Findings
- Mainline split modes — [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md):
  - `none`: one GPU, chosen with `-mg`.
  - `layer` (default): pipeline parallel; contiguous layer blocks, with each layer's KV on its own GPU; "can tolerate slow interconnect speeds".
  - `row`: deprecated, "avoid in new deployments".
  - `tensor`: experimental; splits weights and KV via a "meta device".
  - The doc's summary line: "Pipeline-parallel maximizes batch throughput; tensor-parallel minimizes latency." Tensor mode is "more bottlenecked by GPU interconnect speed".
  - If multi-GPU is slower than single-GPU, the doc suggests checking that NCCL is in use, trying `layer`, or adding lanes.
- Flag semantics — [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md):
  - `-ts/--tensor-split`: proportions in `--device` order. Without it, `layer`/`row` split by free memory and `tensor` splits evenly.
  - `-mg/--main-gpu`: matters only for `none`.
  - `-ngl`: `auto` (default), `all`, or N.
  - `-dev/--device`: device selection. `--list-devices` shows names.
  - `-fit/--fit`: `on` by default. Not supported with `tensor`, where `--ctx-size` may have to be set manually.
- `-sm tensor` requirements and limits — [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md):
  - `-fa off` is a hard error.
  - Quantized KV errors out; f32/f16/bf16 only.
  - Unsupported architectures fail with an error. MoE/hybrid architectures listed as unsupported: Grok, MPT, OLMoE, DeepSeek2, GLM-DSA, Nemotron-H(-MoE), Granite-Hybrid, LFM2-MoE, Minimax-M2, Mistral4, Kimi-Linear, Jamba, Falcon-H1, plus Mamba/Mamba2 and others.
  - OOM advice: reduce `--ctx-size`, then `--parallel`, then `-ngl`.
- History of the tensor-parallel PR (#19378, JohannesGaessler; backend-agnostic, AllReduce):
  - ik's discussion says it landed around 2026-02-07, initially "only works for 2 equal-VRAM GPUs" and dense models. GPT-OSS and Qwen3-MoE support followed 2026-02-08, and NCCL integration by 2026-02-20 — [ik_llama.cpp discussion #1247](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247).
  - Other sources date the merge to about April 2026 — [Marvin-42 insights](https://insights.marvin-42.com/articles/reddit-welcomes-llamacpp-tensor-parallelism-with-an-experimental-warning-label). The dates conflict.
  - v0.6.0 re-enabled `-sm tensor` for Qwen4Exp (#28569), fixed tensor split for fused QKV with uneven K/V heads (#29294), and changed how inactive AllReduce shards are cleared (#29793) — [v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0).
- Tensor-mode AllReduce without NCCL on Pascal — [key-sh/p100-llama-cpp, re-checked 2026-09-18](https://github.com/key-sh/p100-llama-cpp):
  - The internal AllReduce rejects pre-Volta GPUs (it uses `__nanosleep`). Pascal falls back to a generic path, "f32, through host RAM, plus a separate add kernel".
  - On 2×P100 that fallback moved 20.5 GB over PCIe in a 2048-token prefill (23.1% of GPU time).
  - Upstream's internal AllReduce also logs `internal AllReduce init failed (n_devices != 2?)` with more than two devices.
  - The fork's patch 0003 enables the internal AllReduce below Volta, and 0006 extends it beyond two devices (correctness-tested only, on virtual devices).
- pascallama.cpp reports that on Pascal without NCCL, tensor split "falls back to an internal AllReduce" and works on two GPUs. `--split-mode tensor` and the MoE cache are mutually exclusive (upstream errors) — [madsteph74/pascallama.cpp](https://github.com/madsteph74/pascallama.cpp).
- NCCL — [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md):
  - Chosen at build time with `-DGGML_CUDA_NCCL=ON` (default).
  - Not bundled with CUDA.
  - "Used automatically for cross-GPU reductions in `tensor` mode." It has no role in `layer` mode.
  - ikawrakow: "AllReduce on 2 GPUs is a totally trivial operation", so NCCL matters beyond 2 GPUs — [ik discussion #1247](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247).
- P2P:
  - `GGML_CUDA_P2P` is opt-in ("setting it to any value enables it"). It needs driver support, usually workstation/datacenter GPUs, and may corrupt output with IOMMU on some boards — [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md), [build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
  - On a dual-socket P100 box, setting `GGML_CUDA_P2P` caused a 10.7× pp regression (357 → 33 t/s) even though `cudaDeviceCanAccessPeer` returned true — [key-sh/p100-llama-cpp](https://github.com/key-sh/p100-llama-cpp).
  - The current docs no longer mention `GGML_CUDA_NO_PEER_COPY`.
  - `GGML_CUDA_PEER_MAX_BATCH_SIZE` still appears in system_info strings (128) — [#19248](https://github.com/ggml-org/llama.cpp/discussions/19248). Setting it to 0 had no effect on a non-P2P layer-split corruption bug — [#20052](https://github.com/ggml-org/llama.cpp/issues/20052).
- Pipeline parallelism (layer mode):
  - PR #6017 added async `llama_decode`, multiple scheduler input copies, and the `n_batch` (default 4096 logical) versus `n_ubatch` (default 512 physical) distinction — [PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017).
  - `GGML_SCHED_MAX_COPIES` defaults to 4. A proof-of-concept notes that llama-server syncs after every output-producing decode, which defeats overlap — [ggeorgovassilis/llama.cpp-experiments #3](https://github.com/ggeorgovassilis/llama.cpp-experiments/issues/3).
- Pipeline parallelism helps PP only — [discussion #20252, 2026-03-08..10](https://github.com/ggml-org/llama.cpp/discussions/20252):
  - ggerganov: "If you configure it correctly, the PP performance scales nearly linear with the number of devices, even for single request."
  - 4×A40: pp8192 1462 → 2546 → 3530 t/s for 1/2/3 devices; tg128 flat at ~31.7.
  - 4×P40 + 2080 Ti (Qwen3 14B Q8_0): pp8192 ~292 → ~333 t/s; tg128 ~15 in every configuration.
  - A user building with `-DGGML_SCHED_MAX_COPIES=8` still saw "sched copies = 4" in the log.
  - Pipeline parallelism is silently disabled if compute-buffer allocation fails, and also when RPC devices are present or with partial CPU offload.
  - `CUDA_SCALE_LAUNCH_QUEUES=4x` made no difference for that user.
- `CUDA_SCALE_LAUNCH_QUEUES=4x` enlarges the CUDA command buffer, "mainly for multi-GPU pipeline parallelism" — [build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
- In layer mode, MoE expert tensors stay on their layer's GPU: neither variant offloads MoE tensors to a second GPU specially — [discussion #22183](https://github.com/ggml-org/llama.cpp/discussions/22183).
- Risers can corrupt output. A non-P2P layer-split "garbage at ctx > 2048" bug on 2×3090 was traced to a marginal Gen3 riser causing Xid 79 bus drops. Replacing the riser fixed it, while `GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_CUDA_PEER_MAX_BATCH_SIZE=0`, `FORCE_MMQ`, KV type and FA changes all had no effect — [#20052, 2026-03](https://github.com/ggml-org/llama.cpp/issues/20052).
- PCIe x1 with modern GPUs: on a 4-GPU mixed rig (Gen2 x2/x1/x1, Gen3 x16), a 118B-A8B MoE reached about 88 tok/s tg, and single-GPU tg on the x1 cards was about 10% slower than on x2. The author warns this does not prove x1 is harmless — [note.com/agentworkflowlab](https://note.com/agentworkflowlab/n/na17f86500825?hl=en).

### Inferences
These are back-of-envelope estimates for this rig. They assume f32 activations, host-staged copies, and a hidden size of ~2048 (Qwen3-30B-A3B class).
- **Layer split, decode:** 8 GPU boundaries × 8 KB activation × (D2H + H2D) ≈ 130 KB per token. That is negligible bandwidth at 190 MB/s per link, so the cost is mostly latency: 8 boundary syncs per token plus host staging.
- **Layer split, prefill:** at ub=512, each boundary carries about 4 MB each way, roughly 40 ms per boundary at ~190 MB/s, or about 0.3 s per 512-token ubatch if serialized. Pipeline overlap could hide part of that. This is unlikely to dominate given the expected low compute-bound pp of 5-SM cards.
- **Tensor split, decode:** two AllReduces per layer × ~48 layers ≈ 96 host-staged reductions per token across 9 GPUs. Estimated about 1.5 MB per GPU per token, or about 14 MB aggregate through the ~715 MB/s host side, which is ~20 ms per token (≤ ~50 t/s ceiling) before any per-sync latency. Tensor mode on this rig is therefore not an obvious win for decode and likely a big loss for prefill. key-sh measured 20.5 GB of traffic in a 2048-token prefill on 2×P100 with the f32 host fallback. Claim to test; NCCL with bf16 would roughly halve the bytes.
- **NCCL:** without P2P it would use host shared-memory transport. NCCL buffers on nine 6 GB cards cost VRAM, and NCCL's support for sm_61 in current releases was not verified. Claim to test.
- **Batch-1 decode in layer mode** is strictly sequential across GPUs, so tg ≈ (one GPU's bandwidth-limited time per layer × layers) + per-boundary overhead + CPU launch overhead. More GPUs add capacity, not speed.

### Gaps
- No measurements were found of `-sm tensor` (mainline) or `-sm graph` (ik) with more than 4 GPUs, on PCIe x1, or without P2P.
- Whether mainline tensor mode on Pascal currently uses NCCL successfully with 9 devices (NCCL architecture support, VRAM overhead) is untested.
- The exact current behaviour of cross-device copies without P2P (cudaMemcpyPeerAsync staging versus explicit host buffers) and their per-copy latency on this host.

---

## 3. MoE features, tensor placement, auto-fit, expert count, MTP/speculative decoding, KV cache

### Takeaway
Mainline offers the pieces needed for MoE on this rig:
- MUL_MAT_ID kernels in MMQ/MMVQ, with shared-expert fusion as of v0.6.0.
- `-ot` regex placement.
- `--fit` auto-placement, on by default, with `llama-fit-params` to print the result.
- MTP (merged 2026-05-16, `--spec-type draft-mtp`).
- A family of draft and draftless speculative types (draft-simple, eagle3, dflash, dspark, ngram-*).
- KV quantization, which needs FA and is excluded from `-sm tensor`.

The MoE expert-cache work targets host-RAM offload, so it is irrelevant here (4 GB RAM).

### Cited Findings
- `-ot/--override-tensor` places tensors by regex, e.g. `\.ffn_.*_exps\.=CPU`. ik warns that up/gate experts should not be on separate GPUs ("might cause a bit of a deadlock") — [ik docs/parameters.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md). The same syntax is used in mainline, e.g. `-ot ".ffn_(up|down)_exps.=CPU"` in [#19248](https://github.com/ggml-org/llama.cpp/discussions/19248).
- `--fit` and `llama-fit-params`:
  - `--fit` (default on) auto-fits unset arguments to device memory. It is not supported with `-sm tensor` — [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md).
  - `llama-fit-params` prints the CLI args it would choose. If `-ngl`, `-ts` or `-ot` is set explicitly, fit does not change that allocation — [discussion #18049](https://github.com/ggml-org/llama.cpp/discussions/18049).
  - A user report says fit fills GPU0 first and gives MoE no special treatment when everything fits — [search summary of #18049 / #22183](https://github.com/ggml-org/llama.cpp/discussions/22183).
- `--cpu-moe` / `--n-cpu-moe N` keep expert weights in host RAM — [ik parameters](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md). Not usable on a 4 GB host.
- MoE expert caches for host-offloaded experts:
  - PR #29887 was reportedly squashed into master on 2026-10-07 (commit d6cf9ac, build b11480). It doubled 30B-A3B decode on a T4 (22.6 → 43.2 t/s) — [DEV write-up](https://dev.to/sampop3/llamacpps-moe-cache-nearly-doubled-30b-decode-speed-on-a-free-t4-15np). The merge status conflicts with the PR page.
  - Open PRs: #30594 (`--moe-cache-mib`, `--moe-cache-profile`, `--moe-cache-hybrid`) — [PR #30594](https://github.com/ggml-org/llama.cpp/pull/30594) — and #27861 (`--moe-expert-cache N`) — [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861).
  - Cache hits apply only to ubatches of ≤32 tokens. The MTP draft never gets the cache, and the cache is mutually exclusive with `-sm tensor` — [pascallama.cpp](https://github.com/madsteph74/pascallama.cpp).
- MTP:
  - Mainline PR #22673 (am17an) merged 2026-05-16. Tested on Qwen3.6-27B and 35B-A3B. Flags: `--spec-type draft-mtp --spec-draft-n-max N` — [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), [Unsloth MTP guide](https://unsloth.ai/docs/models/mtp).
  - Qwen3.6 MTP heads ship in separate `-MTP-GGUF` files. Unsloth recommends `--spec-draft-n-max 2` — [Unsloth](https://unsloth.ai/docs/models/mtp).
  - RTX 3090: 38 → 65 t/s — [DataCamp](https://www.datacamp.com/tutorial/multi-token-prediction-llama-cpp).
  - v0.6.0 added MTP to Qwen4Exp and probabilistic sampling for drafts/MTP (#27694) — [v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0).
- MTP on Pascal:
  - Single P40, Qwen3.6-35B-A3B Q3_K_XL: 37.9 t/s baseline. With MTP: depth 2 51.6 t/s (best), depth 3 48.3, depth 4 41.9, depth 6 34.5 — [#25978](https://github.com/ggml-org/llama.cpp/discussions/25978).
  - P40 + P6000, Qwen3.6-35B-A3B Q8_0 + MTP: 70.8 t/s at 79.3% acceptance — [pascallama.cpp](https://github.com/madsteph74/pascallama.cpp).
- MTP plus `-sm tensor` on 2×P100 hit `ggml-backend-meta.cpp:2026: GGML_ASSERT(bcj.nodes[i]) failed` during the draft step. The reporter says "llama.cpp 0.6" solved it — [shinbunbun/llama-cpp-p100-patches #10, 2026-10-15](https://github.com/shinbunbun/llama-cpp-p100-patches/issues/10).
- Speculative types in mainline `--spec-type` — [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md):
  - `none`, `draft-simple` (`-md`, `--spec-draft-n-max` default 3, `--spec-draft-n-min`, `-ngld`, `-devd`), `draft-eagle3`, `draft-dflash`, `draft-dspark` (Qwen3 backbone only), `draft-mtp`, `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, and `ngram-mod` (~16 MB constant memory, shared across slots).
  - The doc warns that "MoE models need long drafts" for ngram-mod.
  - Tensor split mode does not support backend sampling.
  - v0.6.0 fixed n-gram drafts being rejected at temperature > 0 (#29924) — [v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0).
- KV cache — [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md):
  - `-ctk/-ctv` default f16.
  - FA is required for quantized V.
  - `-sm tensor` forbids quantized KV.
  - The compiled FA K/V pairs default to q4_0/q4_0, q8_0/q8_0, f16/f16, bf16/bf16 — [build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
  - The P40 MTP run held 50.5–51.5 t/s from 4K (f16 KV) to 131K (q8_0 KV) context — [#25978](https://github.com/ggml-org/llama.cpp/discussions/25978).
  - On P100, f16 KV instead of q8_0 gave +24% tg on a 75K-token request (Qwen3.8-27B) — [key-sh](https://github.com/key-sh/p100-llama-cpp).
- Expert-count reduction: ik has `-ser K,f` (e.g. `-ser 1,6` uses 6 experts) — [ik parameters](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md). In mainline the usual route is `--override-kv <arch>.expert_used_count=int:N`. That is not documented in the sources read: claim to test.

### Inferences
- On this rig `-ot` is the tool for packing whole layers, or splitting a layer's attention and experts, across the nine 6 GB pools. Examples: put the output/embedding-heavy layers on a GPU with fewer blocks, or keep `ffn_up_exps` and `ffn_gate_exps` of a layer on the same GPU (ik warning).
- MTP is the most promising single-user decode accelerator with evidence on Pascal (+36% on one P40). On a nine-GPU layer split, each verify step still walks all nine GPUs. Extra draft tokens activate more distinct experts per step, so MoE gains may be smaller than on dense models. Claim to test.
- KV quantization mostly buys context capacity. On a bandwidth-starved rig the f16 versus q8_0 choice is a measurable trade (P100: f16 faster at long context). Quantized KV also rules out `-sm tensor`.
- An n-gram or draftless method (ngram-mod, ngram-map-k) costs no VRAM. That suits code/agentic repeats but not open-ended chat.

### Gaps
- No source documents `--override-kv ...expert_used_count` quality and speed effects on current Qwen3-MoE/gpt-oss in mainline.
- SWA cache behaviour (`--swa-full`) and context shift for gpt-oss/Gemma-class SWA models were not researched in Oct-2026 docs.
- No Pascal multi-GPU measurements of draft-model, EAGLE3 or DFlash speculative decoding were found.

---

## 4. RPC backend and Vulkan backend as alternatives to CUDA

### Takeaway
RPC is a proof-of-concept, insecure transport that ships activations between hosts. It is useful only if the workstation were to join as a device, and pipeline parallelism is disabled when RPC devices are present. Vulkan on Pascal matches CUDA on tg but roughly halves pp on a P106-100 in the community scoreboard. It is a fallback, for example if a CUDA 12 toolchain becomes impractical.

### Cited Findings
- RPC — [tools/rpc/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md):
  - `rpc-server` exposes local devices (all accelerators by default). One server can expose several GPUs (CUDA0, CUDA1), selected with `--device`.
  - `-c` caches large tensors locally (`$HOME/.cache/llama.cpp/rpc`, `LLAMA_CACHE`) to avoid re-sending weights.
  - The client passes `--rpc host:port,...`.
  - RDMA is supported (RoCEv2, Thunderbolt), with `GGML_RPC_NO_RDMA` to force TCP.
  - "Fragile and insecure… Never run the RPC server on an open network". The example log shows "RPC server v3.0.0".
- Pipeline parallelism is disabled when RPC devices are active — [discussion #20252](https://github.com/ggml-org/llama.cpp/discussions/20252). ik's `-sm graph` cannot use RPC-connected devices — [ik discussion #1247](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247).
- A 2026 two-desktop RPC write-up found 1 GbE was not the bottleneck: activations move per token, and weights move once at load — [michaelspost.com, 2026-08-22](https://michaelspost.com/posts/2026-08-22-two-gpu-llama-cpp-rpc.html). An older thread found RPC across machines about 2× slower than local — [discussion #9136](https://github.com/ggml-org/llama.cpp/discussions/9136).
- Vulkan scoreboard, Llama 2 7B Q4_0, pp512 / tg128 (no FA, then FA) — [discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879):

  | GPU | no FA | FA |
  |---|---|---|
  | P106-100 | 183.78 / 29.77 | 183.40 / 30.79 |
  | P104-100 | 311.90 / 46.18 | 325.30 / 48.64 |
  | P102-100 | 510.62 / 62.83 | 574.56 / 66.48 |
  | GTX 1080 Ti | 585.54 / 67.75 | — |
  | Tesla P40 | 488.06 / 59.36 | — |

  The CUDA scoreboard has P106-100 at 406.94 / 30.40 without FA and 438.49 / 30.64 with FA — [discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013). So Vulkan gives about the same tg and ~45% of the pp on a P106-100.
- On DGX Spark (Mar 2026), Vulkan pp was always lower than CUDA while tg was about equal — [NVIDIA forum](https://forums.developer.nvidia.com/t/vulkan-as-alternative-backend-for-llama-cpp/363516).
- v0.6.0 Vulkan changes: sparse FA for quantized K/V (#29639), a shared-memory OOB fix (#29988), and a revert of Vulkan MoE-aware mat_mul_id tile selection (#29936) — [v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0).
- Ollama enables Vulkan by default when installed (`OLLAMA_VULKAN=0`, `GGML_VK_VISIBLE_DEVICES`) — [Ollama docs](https://docs.ollama.com/gpu).

### Inferences
- Vulkan does not need CUDA 12.x, so it is immune to NVIDIA's toolchain deprecation. It could be the long-term fallback if the 580 driver or CUDA 12.9 stack becomes unmaintainable. Its multi-GPU split, MoE performance and launch overhead on nine Pascal cards are unknown.
- RPC could let the workstation's GPU join as a tenth device: weights are cached with `-c`, and per-token activations are small. It disables pipeline parallelism and adds network latency to every token. Untested.

### Gaps
- No Vulkan multi-GPU or MoE numbers on Pascal were found. No Vulkan `-sm tensor` data on NVIDIA Pascal.
- No measurements of RPC with 9 local GPUs (for example, one rpc-server per GPU) versus native CUDA multi-GPU.

---

## 5. ik_llama.cpp (ikawrakow), Oct 2026

### Takeaway
ik_llama.cpp is very active. Its headline features are:
- `-sm graph` tensor/graph parallelism, which it says is "extremely effective for dense and MoE".
- Fused MoE (default on).
- SOTA IQ_K/IQ_KS/KT quants.
- MTP.
- Rich `-cuda` tuning.

Pascal was officially "in progress" until a contributor (mb8565) landed a run of Pascal PRs in Sep–Oct 2026 plus a Pascal/Volta status doc (2026-10-14). Two red flags for this rig:
- The README requires AVX2 for the CPU side and lists AVX-only CPUs as unsupported. The Celeron 3865U has no AVX at all.
- Some FA/KV combinations fall back to CPU FA on Pascal.

### Cited Findings
**Status and supported hardware**
- README: CUDA "Turing or newer" with "Pascal and Volta support in progress" (changed on 2026-10-14 to "recent progress on bringing Pascal and Volta up to speed", linking docs/pascal-volta.md). CPU needs "AVX2 or better, or ARM_NEON". The not-supported list includes "ROCm, Vulkan, Metal, old NVIDIA GPUs, and AVX-only CPUs". No releases are published; Docker images (CPU, CUDA 12) are on ghcr.io — [ik_llama.cpp README](https://github.com/ikawrakow/ik_llama.cpp), [PR #2670](https://github.com/ikawrakow/ik_llama.cpp/pull/2670).
- Pascal/Volta doc (code at 2026-10-13) — [docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md):
  - sm_61 has dp4a (MMQ) but not fast FP16, so it always uses f32 FA kernels.
  - CUDA graphs are not used below Ampere.
  - FA head sizes 64/128/256 run on GPU (vec_f32 / tile_f32). 512 runs via the new tile kernel (GQA multiple of 4, no sinks). 192/128 and 576/512 (MLA) run on GPU only for n_q ≤ 8. Many other sizes run on CPU.
  - Quantized KV on Pascal decode: head 64 supports f16 only. Head 128 supports f16, q8_0, q4_0, iq4_nl and some mixed pairs. Head 256 supports f16 and q8_0.
  - Observed placement: gpt-oss-20b with q8_0 KV does "prompt GPU, decode CPU". Llama-3.1-8B with q5_0 or bf16 KV aborts. `-ctk q8_KV` crashes (SIGSEGV).
  - Recommended build: `-DCMAKE_CUDA_ARCHITECTURES="60;61;70"`, CUDA 12.x. sm_61 placement was derived on a P100 shimmed to report cc 6.1; a real P40 ran only FA kernel tests.
- Pascal PRs:
  - #2509 (merged 2026-09-23): IQ4_K MMVQ via `__byte_perm`. tg128 P40 31.75 → 67.88 t/s (+113%), P100 34.48 → 60.26. P100 Qwen3-30B-A3B mix 34.2–37.4 → 40.8–43.4 t/s. Perplexity identical — [PR #2509](https://github.com/ikawrakow/ik_llama.cpp/pull/2509).
  - #2510: opt-in `-cuda fa-fabsum=1`, an fp32-accumulating variant of the fp16 tile FA for sm_60/62. sm_61 is unaffected since it uses tile_f32 — [PR #2510](https://github.com/ikawrakow/ik_llama.cpp/pull/2510).
  - #2647 (merged 2026-10-12): head-size-512 FA on Pascal GPU. P40 Gemma-4-12B pp8192 67 → 386 t/s, tg64 at depth 8192 10.3 → 25.9 — [PR #2647](https://github.com/ikawrakow/ik_llama.cpp/pull/2647).

**Features — [docs/parameters.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md)**
- Split mode:
  - `-sm none|layer|graph` (default none). `graph` "splits tensors and the compute graph… extremely effective for dense and MoE".
  - `-smgs` forces graph-mode scheduling.
  - `-smf16`/`-smf32` set the inter-GPU exchange type.
  - `-grt q8_0|bf16|f16|f32` sets the graph reduce type, to cut inter-GPU bytes.
  - `--max-gpu N` caps GPUs per layer in graph mode, for when "all GPUs performs worse than using two".
  - `-ts`, `-dev`, `-mg`.
- Graph handling and offload:
  - `-gr/--no-gr` graph reuse (default on).
  - `-sas` async scheduler.
  - `--fit`, `--fit-margin` (default 1024 MiB), `--gpu-fit-margin`.
  - `-ot` (up/gate experts on separate GPUs may deadlock).
  - `-ooae` (offload only active experts; default on).
  - `--offload-policy`.
- MoE:
  - `-fmoe` (fused up/gate MoE; default on).
  - `-muge` (merge up/gate experts).
  - `-ser` (custom active-expert count).
  - `-mqkv` (merge QKV; disables mmap).
  - `-rtr` (run-time repack; CPU-side).
  - `-no-fug`, `-no-mmad`.
- Attention and KV:
  - `-mla 3` (default) for DeepSeek-style MLA.
  - `-amb` (attention max batch, default 256 MB).
  - `-khad`/`-vhad` Hadamard on the KV cache.
  - `-cuda` tweaks fusion, the GPU offload threshold, the MMQ-ID threshold, `fa-offset`, and `fa-fabsum`.
- Speculative: `--spec-type mtp|dflash|dspark|…`, `-mtprot`.
- Environment: `GGML_CUDA_NO_PINNED`, `GGML_CUDA_NO_PINNED_WEIGHTS`. NCCL is disabled only at build time (`-DGGML_NCCL=OFF`).
- Graph-parallel architectures include LLAMA, QWEN3MOE, GLM4_MOE, DEEPSEEK2, GLM_DSA, QWEN35MOE and GEMMA4. P2P: the doc only advises checking `nvidia-smi topo -p2p r` and ReBAR.

**Known issues and comparisons**
- Known issues — [README](https://github.com/ikawrakow/ik_llama.cpp):
  - Gibberish with `-sm graph` combined with partial offload, `--cpu-moe` or tensor overrides; the workaround is `-cuda graphs=0`.
  - Avoid `-rtr` with hybrid MoE.
  - Unsloth `_XL` models with f16 tensors "likely to fail".
- Benchmarks versus mainline tensor mode, 4×3090, 2026-02-20 — [ik discussion #1247](https://github.com/ikawrakow/ik_llama.cpp/discussions/1247):

  | Model | ik `-sm graph`, pp / tg (t/s) | mainline `-sm tensor`, pp / tg (t/s) |
  |---|---|---|
  | GPT-OSS-120B-MXFP4 | 5225 / 163.7 | 727.7 / 122.9 |
  | LLaMA-3-70B Q4_0 | 1803 / 49.6 | 287.9 / 48.0 |

  The same thread reports mainline tensor mode on 2×3090 hitting OOM in sweeps (Qwen3-30B-A3B at 32K).
- ikawrakow, 2×3090: `layer` 51 t/s versus `graph` 72 t/s. A user with 2×5070 Ti on Windows saw graph drop tg from ~40 to ~13 t/s; it was resolved by moving to Linux ("Windows-related", P2P suspected) — [ik issue #1730, 2026-05](https://github.com/ikawrakow/ik_llama.cpp/issues/1730).
- A third-party blog claims 3–4× multi-GPU gains with graph mode. This is unverified and secondary — [Medium](https://medium.com/@jagusztinl/llama-cpp-performance-breakthrough-for-multi-gpu-setups-04c83a66feb2).

### Inferences
- The AVX2 requirement is a potential blocker. The CPU backend is always compiled and linked, and the IQK CPU kernels assume AVX2. Building with `GGML_NATIVE=ON` on the Celeron may fail to compile, and a binary built elsewhere may SIGILL. Claim to test first, because it gates every other ik experiment.
- Every ik FA case that falls back to CPU is fatal for speed on a 2-core Celeron. Example: gpt-oss (head size 64) with q8_0 KV decodes on CPU in ik on Pascal. KV type choice must follow ik's Pascal table. Run with `GGML_SCHED_DEBUG=1` to check where each node lands.
- IQ4_K now has a fast Pascal MMVQ (P40 tg doubled). No Pascal numbers exist for IQ2_K/IQ3_K/IQ5_K/IQ_KS/IQ_KT. ikawrakow notes that MMQ for block-size-16 quants is slower than for block-size-32 ([PR #2509](https://github.com/ikawrakow/ik_llama.cpp/pull/2509)). Trellis (KT) quants are compute-heavy and are a claim to test on 5-SM cards.
- For this rig, `-sm graph`'s gains depend on inter-GPU AllReduce latency over host-staged x1 links, which is likely the opposite of the 3090/NVLink-class setups where it shines. `--max-gpu 2` (TP within pairs) and `-grt q8_0` exist precisely to trade compute parallelism for fewer bytes.

### Gaps
- No ik_llama.cpp numbers on sm_61 multi-GPU, on PCIe x1, or with more than 4 GPUs were found.
- Whether ik's CUDA build compiles and runs at all on a non-AVX host (for example with GGML_NATIVE=OFF and no AVX flags) is not documented.
- IQ_KS/IQ_KT MMQ/MMVQ performance on dp4a Pascal is undocumented.

---

## 6. Other forks and wrappers

### Takeaway
Wrappers mostly ship prebuilt binaries, and Pascal coverage there is eroding. LM Studio and Docker Model Runner reportedly dropped sm_61. Ollama and koboldcpp still ship CUDA 12 builds covering 6.1. Pascal-specific community patch sets exist (pascallama.cpp for sm_61 MMQ tiles; key-sh and shinbunbun P100 patches for AllReduce, MMVQ and FA). For a nine-GPU interactive setup, llama-server plus llama-swap, built from source for sm_61, is the low-friction baseline.

### Cited Findings
- Ollama:
  - Docs: compute capability ≥ 5.0. CC 5.0–6.2 require driver ≥ 570. The 6.1 row lists GTX 1050–1080 Ti, TITAN X/Xp, Quadro P-series and Tesla P40/P4. Selection via `CUDA_VISIBLE_DEVICES` (UUIDs recommended). Vulkan is on by default when installed — [docs.ollama.com/gpu](https://docs.ollama.com/gpu).
  - Ollama v0.32.14 (2026-08-15) ships a CUDA v12 runner including compute 6.1 on Linux, alongside v13 — [localaimaster](https://localaimaster.com/blog/pascal-gpus-after-cuda-13).
  - Ollama pins a patched llama.cpp revision — [computingforgeeks](https://computingforgeeks.com/ollama-vs-vllm-vs-llama-cpp/).
- LM Studio 0.4.21: the reporter says the bundled CUDA backend no longer supports sm_61. Issue open, no staff response — [#2316](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2316).
- koboldcpp:
  - Main Linux/Windows builds are CUDA 12 plus Vulkan. An "oldpc" build uses CUDA 11.4 for CC 3.5/5.0/6.1/7.5 — [koboldcpp.net releases doc](https://koboldcpp.net/docs/install/releases/).
  - v1.122/1.122.1 (2026-09-26) notes include "Revert P40 increase VRAM usage issue" — [freedom.tech release post](https://freedom.tech/posts/2026-09-26-koboldcpp-1-122-1/), [SourceForge v1.122.1](https://sourceforge.net/projects/koboldcpp/files/v1.122.1/).
- llama-swap: a Go proxy in front of llama-server that loads and swaps models per request (YAML model→command map, TTL unload, groups). Repo last pushed 2026-08-16, ~5.4k stars — [ecosyste.ms summary](https://summary.ecosyste.ms/projects/378903), [glukhov.org guide](https://www.glukhov.org/llm-hosting/llama-swap/).
- Pascal patch forks:
  - madsteph74/pascallama.cpp: sm_61 MMQ 96-row tiles plus `#pragma unroll`, pinned to commit a0e6dc738 (Oct 2026). Build flags `GGML_CUDA_FORCE_MMQ=ON`, `CMAKE_CUDA_ARCHITECTURES=61`. No isolated before/after numbers — [repo](https://github.com/madsteph74/pascallama.cpp).
  - key-sh/p100-llama-cpp (base b10660): only patches 0001 and 0005 apply to sm_61 ("small" gain), and 0003/0004 share code with sm_61 but are untested there. The fork also fixes a P40 FA zero-occupancy launch assert (0012) — [repo](https://github.com/key-sh/p100-llama-cpp).
  - shinbunbun/llama-cpp-p100-patches and sbendhiba fork (P100, tensor mode + MTP) — [issue #10](https://github.com/shinbunbun/llama-cpp-p100-patches/issues/10).
  - Forks with sm_60 FP16-accuracy fixes: TheTom/llama-cpp-turboquant, spiritbuun/buun-llama-cpp, Kmic-68 — [#25593](https://github.com/ggml-org/llama.cpp/issues/25593).
  - schmared/tesla-p40-wsl2-cuda-llamacpp (P40, CUDA 12.9, sm_61) — [repo](https://github.com/schmared/tesla-p40-wsl2-cuda-llamacpp).
- Docker Model Runner excludes Pascal from its bundled llama.cpp CUDA build; the suggested fix is adding arch 61 — [docker/model-runner #929](https://github.com/docker/model-runner/issues/929).

### Inferences
- Wrappers add little for a single-user nine-GPU rig and risk shipping binaries without sm_61. Ollama's scheduler also hides `-ot`/`-ts` control. Building mainline (and optionally ik) from source, fronted by llama-swap for model switching, keeps every knob available.
- pascallama's tile change and #30235's Q8_0 I=128 change are cheap sm_61 MMQ experiments that mainly affect prompt processing, not decode.

### Gaps
- No 2026 information was found for GPUStack/llama-box, LocalAI, Jan, llamafile or croco.cpp (Nexesenex) regarding Pascal or multi-GPU MoE. They were not researched in depth.
- Ollama's "new engine" (ollamarunner) multi-GPU behaviour on Pascal: no source found.

---

## 7. Real measurements on Pascal (pp/tg, hardware, flags, versions)

### Takeaway
Single-card scoreboards give a solid baseline: a P106-100 does ~30 t/s tg and ~400–440 t/s pp512 on Llama-2-7B Q4_0 under CUDA. A P106-090 has half the SMs, so expect similar tg and roughly half the pp (inference). The only nine-P106 MoE data point is a self-reported ">30 t/s" on a Qwen3.6-35B-A3B-class model (Oct 2026). Multi-GPU Pascal results consistently show layer split adding capacity, not tg.

### Cited Findings
- CUDA scoreboard, Llama 2 7B Q4_0, `-ngl 99`, single GPU (`-sm none`), pp512 / tg128 in t/s — [discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013):

  | GPU | no FA | FA | Build / date |
  |---|---|---|---|
  | P106-100 | 406.94 / 30.40 | 438.49 / 30.64 | 5fd160b |
  | P106-100 (rerun) | 425.73 / 29.42 | 436.90 / 29.94 | b5688, 2025-08-07 |
  | GTX 1060 | 416.85 / 27.79 | 446.19 / 28.18 | — |
  | Tesla P40 | 1007.42 / 54.74 | 1079.66 / 53.73 | — |
  | Tesla P4 | 514.53 / 33.29 | — | — |
  | GTX 1070 Ti | 714.44 / 37.82 | — | — |
  | GTX 1080 Ti | 1084.41 / 62.49 | 1138.14 / 61.38 | — |
  | Titan Xp | 1154.96 / 76.08 | — | — |
  | Tesla P100 (no dp4a) | 760.80 / 58.35 | — | — |
  | GTX 1050 Ti | 266.70 / 19.06 | — | 2026-06-24 |

- Vulkan scoreboard (same model): P106-100 183.78 / 29.77, P104-100 311.90 / 46.18, P102-100 510.62 / 62.83 — [discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879).
- Nine-P106 rig (Reddit r/LocalLLaMA, covered 2026-10-10):
  - 9× P106 6 GB = 54 GB VRAM, bought for ~$35. A "Qwen3.6 35B-A3B-class model" runs at ">30 tokens per second", self-reported.
  - The article says PCIe 3.0 x1. No software, flags, quant or pp numbers are given — [Guru3d](https://www.guru3d.com/story/retired-ninegpu-mining-rig-bought-for-provides-54gb-vram-for-local-llm-inference/), [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/reddit-user-pays-usd35-for-a-crypto-mining-rig-left-in-a-garage-and-gets-54gb-of-vram-at-65-cents-per-gb-the-first-test-for-2017-era-gpus-a-35b-parameter-ai-model-running-locally-at-over-30-tokens-per-second).
  - The configuration matches this rig (9 × P106, 54 GB). The link measured on this rig is PCIe 1.0 x1, which conflicts with the article's "3.0 x1".
- Single P40, Qwen3.6-35B-A3B Q3_K_XL (CUDA 12.9.2, `CMAKE_CUDA_ARCHITECTURES=61`, `FORCE_MMQ=ON`, `F16=OFF`) — [#25978, 2026-07-21](https://github.com/ggml-org/llama.cpp/discussions/25978):
  - tg 37.9 t/s, or 51.6 with MTP n-max 2.
  - pp512 982 t/s. Prefill ~790 t/s at 6K, 422–609 at 25K, ~216 at 78K.
  - VRAM 21.7–22.8 GB.
- P40 + P6000 (tensor split), Qwen3.6-35B-A3B Q8_0 + MTP: 70.8 t/s decode. Single P40 with MTP: ~75 t/s ("earlier Oct figure") — [pascallama.cpp](https://github.com/madsteph74/pascallama.cpp).
- P40, MoE Mixtral (Q2_K / Q3_K_M) tg128: 32.4 / 25.8 t/s stock; Llama Q4_0 52.3, Q4_K_M 48.9 (build 0.6.0-dev 1623d8c, CUDA 12.4, driver 550) — [#30334](https://github.com/ggml-org/llama.cpp/issues/30334).
- 4×P40 + RTX 2080 Ti, Qwen3 14B Q8_0, layer split: pp8192 ~292 (1 device) → ~333 (2–3 devices); tg128 ~15 in every configuration — [discussion #20252](https://github.com/ggml-org/llama.cpp/discussions/20252).
- 8×P40, Kimi-K2.5 UD-Q2_K_XL with experts in host RAM: 5.5–5.9 t/s tg. Build b7840, `ARCHS=610`, `FORCE_MMQ`, `-fa on`, `--fit on`. The poster says "over 20 TPS on most smaller MOE models" without figures — [discussion #19248](https://github.com/ggml-org/llama.cpp/discussions/19248).
- 2×P100 (no dp4a; dual socket; CUDA 12.4; driver 580), Llama-3.1-8B Q4_K_M — [key-sh](https://github.com/key-sh/p100-llama-cpp):
  - 1 card tg128 39.26 → 79.57 t/s with patches.
  - 2 cards `-sm tensor` tg 56.80 → 70.00.
  - Qwen3.8-27B 2 cards `-sm tensor`: pp8192 323.6 → 378.7, tg128 22.3 → 24.5.
- P40 IQ4_K (ik_llama.cpp), Qwen3.5-4B: tg128 31.75 → 67.88 t/s. P100 Qwen3-30B-A3B mix tg128 34.2–37.4 → 40.8–43.4 t/s — [ik PR #2509](https://github.com/ikawrakow/ik_llama.cpp/pull/2509).
- P40 Gemma-4-12B (ik, head-512 FA on GPU): pp8192 67 → 386 t/s; tg64 at depth 8192 10.3 → 25.9 — [ik PR #2647](https://github.com/ikawrakow/ik_llama.cpp/pull/2647).
- Searches found no gpt-oss-20b or GLM-4.5-Air numbers on any Pascal card, and no P104-100/P102-100 MoE numbers. A benchmark site lists a P100 row for gpt-oss-20b with no value — [search summary](https://llm-benchmark.de/benchmark/run-20260821-144810-b25e86?lang=en).

### Inferences
- **Decode ceiling for a ~3B-active MoE at ~4.5 bpw:** about 1.7–2 GB of weights per token. In layer split, at most one 173 GB/s card is active at a time, so the pure-bandwidth ceiling is ~85–100 t/s. Realistic Pascal efficiency (the P106-100 reaches ~30 t/s on a 3.6 GB 7B, i.e. ~110 GB/s effective) suggests ~50–60 t/s before multi-GPU and CPU-launch overheads. The reported ">30 t/s" on nine P106s implies those overheads roughly halve throughput. Estimate.
- **P106-090 versus P106-100:** 5 SMs versus 10, and ~173 versus ~192 GB/s. Expect tg ~85–90% and pp ~45–55% of the P106-100 scoreboard (about 27 t/s and ~200–220 t/s pp512 on Llama-2-7B Q4_0). This is a cheap single-card calibration experiment.

### Gaps
- No public multi-GPU MoE benchmark exists on P104/P106/P102 mining cards with stated flags and versions, apart from the uncorroborated Guru3d/Tom's ">30 t/s".
- No gpt-oss-20b/120b or GLM-4.5-Air numbers on any Pascal card were found.
- No CMP-card (Pascal-based CMP is rare; the CMP 170HX is GA100) llama.cpp data.

---

## 8. Tuning knobs: flags, env vars and build options that plausibly matter on this rig

### Takeaway
The most consequential knobs are:
- Build: `CMAKE_CUDA_ARCHITECTURES=61`, CUDA 12.x, and `GGML_NATIVE` handling when building on the workstation.
- Split mode and placement: `-sm`, `-ts`, `-ot`, `--fit`.
- FA plus KV type: must stay on GPU kernels for sm_61.
- `-ub`/`-b` for prefill.
- Speculative decoding: MTP or ngram.
- The fusion and MMQ-config details that move ±5–12% on Pascal.

Several build flags in the tested rig's build config (`GGML_CUDA_GRAPHS`, `GGML_CUDA_NCCL`, `FORCE_MMQ`) are probably inert for layer-split decoding on Pascal.

### Cited Findings and knob table
Every "expected effect" entry is an inference unless a source is cited.

**Build options (mainline)**

| Knob | What it does | Expected effect on this rig |
|---|---|---|
| `CMAKE_CUDA_ARCHITECTURES=61` | Compile only for sm_61 ([build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)) | Required when building on the workstation. `GGML_NATIVE=ON` targets the build machine's GPU and CPU. It also shortens the build on a 2-core host. |
| `GGML_NATIVE` (default ON) | Build for the build host's hardware ([build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)) | ON is fine when building on the rig. When building on the workstation, set OFF, or the CPU backend gets AVX2/AVX-512 and SIGILLs on the no-AVX Celeron (claim to test; see [FreeBSD SIGILL from -march=native](https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=294215)). `GGML_CPU_ALL_VARIANTS` + `GGML_BACKEND_DL` select a CPU variant at runtime (documented for ARM/PPC; x86 baseline variant is a claim to test). |
| CUDA toolkit 12.x (12.9) | sm_61 offline compile removed in 13.0 ([#25978](https://github.com/ggml-org/llama.cpp/discussions/25978)) | Workstation: a Docker CUDA 12.9 image; watch the glibc ≥ 2.41 header issue and the libcuda stub. |
| `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS` | Force MMQ / force FP16 cuBLAS ([build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)) | Likely a no-op / a loss on sm_61. |
| `GGML_CUDA_FA_QUANTS` (`GGML_CUDA_FA_ALL_QUANTS` deprecated) | Which FA K/V pairs are compiled ([build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)) | The default subset covers q8_0/q8_0 and q4_0/q4_0. `all` adds mixed pairs (e.g. q8_0 K + q4_0 V) at longer compile time. |
| `GGML_CUDA_GRAPHS` | CUDA graphs | Probably inert on pre-Ampere ([PR #6766](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/6766/overview), [ik doc](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md)). Claim to test. |
| `GGML_CUDA_NCCL` (default ON) | NCCL for `-sm tensor` reductions ([multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)) | Irrelevant in layer mode. May cost VRAM at init if used. |
| `GGML_SCHED_MAX_COPIES` (default 4) | Pipeline-parallel input copies ([PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017)) | PP only. Each copy costs compute-buffer VRAM on 6 GB cards. One user saw the setting not take effect ([#20252](https://github.com/ggml-org/llama.cpp/discussions/20252)). |
| `GGML_CUDA_CCCL_VERSION` | Pin CCCL; CUB DeviceTopK needs ≥ 3.4.3 ([build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)) | May matter when building with CUDA 12.9's bundled CCCL (claim to test). |
| `GGML_RPC=ON` | RPC backend | Only for the workstation-as-device experiment. |
| `GGML_VULKAN=ON` | Vulkan backend | Fallback backend, for A/B testing. |

**Runtime flags (mainline)**

| Knob | What it does | Expected effect on this rig |
|---|---|---|
| `-sm layer` / `tensor` / `none`; `-ts`; `-dev`; `-mg` | Split mode and proportions ([multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)) | Layer is the safe baseline. Tensor is experimental: FA plus f16 KV required, no `--fit`, interconnect-bound. |
| `--fit on/off`, `llama-fit-params` | Auto placement; prints args ([#18049](https://github.com/ggml-org/llama.cpp/discussions/18049)) | Gives a starting `-ts`/`-ot`. Override when the last GPU holds the output head. |
| `-ot regex=CUDAn` | Per-tensor placement | Balance 9 × ~6 GB. Keep a layer's up/gate experts together. |
| `-ngl all` | Full offload | Mandatory (4 GB host). |
| `-fa on` | FlashAttention | Needed for quantized KV. On sm_61 it uses f32 vec/tile kernels. On the P106-100 it gives +3–8% pp and ±0 tg ([#15013](https://github.com/ggml-org/llama.cpp/discussions/15013)). |
| `-ctk` / `-ctv` f16, q8_0, q4_0 | KV type | Capacity versus speed. f16 was faster at long context on P100 ([key-sh](https://github.com/key-sh/p100-llama-cpp)). |
| `-b` / `-ub` | Logical / physical batch ([PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017)) | Larger `-ub` raises pp but grows compute buffers per GPU. Pipeline overlap needs `-b > -ub` (claim to test). |
| `-np/--parallel`, `--kv-unified` | Server slots | Throughput mode later. |
| `--spec-type draft-mtp --spec-draft-n-max 2` | MTP ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)) | +36% tg on P40 for Qwen3.6-35B-A3B ([#25978](https://github.com/ggml-org/llama.cpp/discussions/25978)). Needs MTP GGUFs. |
| `--spec-type ngram-mod / ngram-map-k / ngram-simple` | Draftless speculation ([speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)) | Zero VRAM. Helps repetitive output. |
| `-md … --spec-type draft-simple -ngld -devd` | Draft model | Costs VRAM; can be placed on a specific GPU with `-devd`. |
| `--override-kv <arch>.expert_used_count=int:N` | Fewer active experts | Faster decode, lower quality (claim to test). |
| `--no-mmap` / `--mlock` | Loading behaviour | With 4 GB RAM and models larger than RAM, mmap streaming is likely needed; test load time and stability. |

**Environment variables**

| Variable | What it does | Expected effect on this rig |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | Device subset | Run 1/2/3/9-GPU scaling tests. |
| `GGML_CUDA_P2P` | Opt-in P2P | Leave unset: no P2P on this hardware, and it can corrupt output or regress ([key-sh](https://github.com/key-sh/p100-llama-cpp)). |
| `CUDA_SCALE_LAUNCH_QUEUES=4x` | Bigger launch queue for pipeline parallelism ([build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)) | Possibly relevant with 9 GPUs and a slow host. No effect in one report ([#20252](https://github.com/ggml-org/llama.cpp/discussions/20252)). |
| `GGML_CUDA_DISABLE_GRAPHS=1` | Disable CUDA graphs ([#20052](https://github.com/ggml-org/llama.cpp/issues/20052)) | Diagnostic only. |
| `GGML_SCHED_DEBUG=1` | Print each node's backend ([ik doc](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md)) | Verify nothing (FA, MoE ops) falls back to the Celeron. |
| `GGML_CUDA_NO_PINNED`, `GGML_CUDA_NO_PINNED_WEIGHTS` | Disable pinned host memory ([ik parameters](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md)) | May help on a 4 GB host if pinned allocations fail (claim to test in mainline). |
| `GGML_CUDA_MMQ_PREC` | MXFP4/NVFP4 activation precision ([build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)) | gpt-oss experiment. |

**ik_llama.cpp-specific knobs** ([parameters.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md)): `-sm graph`, `--max-gpu N`, `-grt q8_0`, `-smf32`, `-smgs`, `-fmoe` (default on), `-muge`, `-ser`, `-gr`, `-sas`, `-amb`, `-mla`, `-cuda` (fusion, offload thresholds, MMQ-ID threshold), `-khad/-vhad`, `--spec-type mtp`, and `-mtprot`.

**Source-level tweaks** (patch experiments):
- MMQ tile I for Q8_0 = 128 in `mmq-config-pascal-dp4a.cuh` ([#30235](https://github.com/ggml-org/llama.cpp/issues/30235)).
- The 96-row tiles ([pascallama.cpp](https://github.com/madsteph74/pascallama.cpp)).
- Correcting the MMVQ fusion gate to `< 700` for cc 610 ([#30334](https://github.com/ggml-org/llama.cpp/issues/30334)).
- key-sh 0001 (q4_K dp4a sum reuse, +4%) and 0003/0006 (Pascal internal AllReduce, more than 2 devices) ([key-sh](https://github.com/key-sh/p100-llama-cpp)).

### Inferences
- Because decode on this rig is likely bound by a mix of per-GPU bandwidth and host launch/sync latency, knobs that cut kernel count help more than raw-kernel tweaks. Candidates: fusion, fewer GPUs per token path (fewer, fuller GPUs if the model fits), and MTP (amortising launches over multiple tokens).
- Using fewer GPUs for a model that fits (for example 6 instead of 9) cuts boundaries and launches per token. This may beat spreading evenly over nine. Claim to test.

### Gaps
- The exact current names and defaults of some mainline env vars (`GGML_CUDA_NO_PINNED`, `GGML_CUDA_DISABLE_FUSION`) were not confirmed in Oct-2026 mainline docs.
- A mainline backend (GPU) sampling flag exists (referenced in speculative.md), but its name and default were not confirmed. CPU sampling over 150k–200k vocabularies on a no-AVX Celeron could cost ms per token.

---

## Experiment ideas (options, not decisions)

### Takeaway
The experiments run roughly in order of information per hour: cheap single-card calibration and placement checks first, then multi-GPU scaling, then speculative decoding, then forks and backends. All use `llama-bench` or `llama-sweep-bench`-style sweeps with pp512/tg128 at several depths and `-r 3+`, while recording build commit, CUDA version, driver and full flags. Effort and risk ratings are rough.

### Experiments
1. **Single-card calibration against the P106-100 scoreboard.**
   - Engine: mainline at current master, built with `CMAKE_CUDA_ARCHITECTURES=61`.
   - Run: `llama-bench -m llama-2-7b.Q4_0.gguf -ngl 99 -sm none -mg N -fa 0,1` on one card.
   - Measure: pp512, tg128, `nvidia-smi` clocks and link state.
   - Expected: tg ~26–29 t/s and pp ~200–230 t/s, versus the P106-100's 30.4 / 407 ([#15013](https://github.com/ggml-org/llama.cpp/discussions/15013)).
   - Effort: low. Risk: none.
   - Variants: repeat on all 9 cards to find weak cards or risers; Xid errors in `dmesg` are a riser warning per [#20052](https://github.com/ggml-org/llama.cpp/issues/20052).
2. **Verify code paths and fallbacks.**
   - Run: `GGML_SCHED_DEBUG=1` (or verbose logs) with the target MoE (Qwen3-30B-A3B / Qwen3.6-35B-A3B, gpt-oss-20b) across `-ctk/-ctv` f16, q8_0, q4_0.
   - Measure: any node on CPU, especially FA and MUL_MAT_ID; whether the log reports CUDA graphs disabled for the GPU architecture.
   - Expected: everything on GPU for head-size 64/128 models with f16 or q8_0 KV. Graphs disabled.
   - Effort: low.
3. **Build-flag A/B on sm_61:** FORCE_MMQ on/off, FORCE_CUBLAS on, `GGML_CUDA_FA_QUANTS=all` versus default.
   - Measure: pp512, tg128, and VRAM on one card and on the 9-GPU layer split.
   - Expected: FORCE_MMQ no change; FORCE_CUBLAS worse pp and more VRAM.
   - Effort: medium (rebuilds; build on the workstation with `GGML_NATIVE=OFF`).
   - Risk: a workstation build must not include AVX in the CPU backend.
4. **GPU-count scaling in layer mode.**
   - Model: one that fits in 3–4 cards (e.g. gpt-oss-20b MXFP4 or a ~12 GB Qwen3-30B-A3B quant). Run on 3, 4, 6 and 9 GPUs via `CUDA_VISIBLE_DEVICES` with an auto or even `-ts`.
   - Measure: tg128 and pp512 at depths 0/4K/16K.
   - Expected: tg falls slightly as GPUs are added (more boundaries and launches); pp roughly flat or slightly up with pipeline parallelism.
   - Effort: low. This quantifies the per-boundary cost on PCIe 1.0 x1 and the CPU-launch bottleneck.
5. **Placement tuning.**
   - Compare `--fit on` against `llama-fit-params` output against a hand-made `-ts` / `-ot`. Hand-made options: fewer layers on the GPU holding the output head; experts and attention of a layer co-located.
   - Measure: tg/pp, max context before OOM, per-GPU VRAM headroom (Xorg context included).
   - Expected: small tg change; better context capacity.
   - Effort: medium.
6. **Batch sizes for prefill.**
   - Sweep `-ub` 64/128/256/512/1024 and `-b` 512–4096, plus `CUDA_SCALE_LAUNCH_QUEUES=4x`, on the 9-GPU layer split.
   - Measure: pp at 2K/8K prompts; compute-buffer VRAM; whether "pipeline parallelism enabled" appears in the log.
   - Expected: pp rises with ub until VRAM-bound. Host-staged activation copies (~4 MB per boundary at ub 512) become visible.
   - Effort: low.
7. **MTP speculative decoding.**
   - Model: Qwen3.6-35B-A3B MTP GGUF (Unsloth) in a quant that fits ~50 GB with KV.
   - Run: `--spec-type draft-mtp --spec-draft-n-max 1/2/3`.
   - Measure: tg, acceptance rate, VRAM cost of the MTP heads.
   - Expected: +20–40% tg if acceptance is ~75–80% (P40: +36% at n-max 2, [#25978](https://github.com/ggml-org/llama.cpp/discussions/25978)). Gains may shrink with 9-GPU boundary latency.
   - Effort: low–medium.
8. **Draftless speculation.**
   - Run: `ngram-mod`, `ngram-map-k` and `ngram-simple` on code-editing and summarisation prompts.
   - Measure: tg and acceptance.
   - Expected: gains only on repetitive output. The doc says MoE needs long drafts.
   - Effort: low.
9. **KV type versus speed and context.**
   - Run: f16, q8_0, q4_0 K/V (FA on) at depths 0/8K/32K.
   - Measure: tg, pp, max context.
   - Expected: quantized KV extends context. tg may drop at depth (P100 saw f16 +24% at 75K).
   - Effort: low.
10. **Pascal MMQ/MMVQ source patches.**
    - Apply separately: (a) the #30334 fusion gate fix (`cc < 700`) for MMVQ fusion; (b) Q8_0 `I=128` in `mmq-config-pascal-dp4a.cuh`; (c) pascallama 96-row tiles.
    - Measure: tg/pp per quant type (Q4_K_M, Q8_0, Q2_K/Q3_K MoE) with interleaved A/B runs.
    - Expected: ±5–12% depending on quant type ([#30334](https://github.com/ggml-org/llama.cpp/issues/30334), [#30235](https://github.com/ggml-org/llama.cpp/issues/30235)).
    - Effort: medium. Risk: patches drift with upstream.
11. **Mainline `-sm tensor` on this rig.**
    - Run: 2, 4 and 9 GPUs; FA on; f16 KV; `--fit off` with explicit `-c`; NCCL build versus not. Model: Qwen3-MoE or gpt-oss (supported architectures).
    - Measure: tg, pp, VRAM (NCCL buffers), correctness (perplexity versus layer mode), whether the internal AllReduce refuses Pascal or more than 2 devices.
    - Expected: likely slower than layer mode for pp, and possibly for tg, given ~14 MB/token of host-staged reduction traffic (estimate). This tests whether latency, not bandwidth, dominates.
    - Effort: medium. Risk: experimental code, asserts ([shinbunbun #10](https://github.com/shinbunbun/llama-cpp-p100-patches/issues/10)).
12. **ik_llama.cpp feasibility gate.**
    - Run: build on the rig (`GGML_NATIVE=ON`, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61`), or on the workstation with native off and no AVX.
    - Measure: does it compile and run without SIGILL on the no-AVX Celeron?
    - Expected: uncertain. The README requires AVX2.
    - Effort: low–medium. Risk: may simply not work.
13. **ik_llama.cpp performance, only if 12 passes.**
    - Run: `-sm layer` versus `-sm graph` with `--max-gpu 2/3`, `-grt q8_0`, `-smf16`/`-smf32`, `-fmoe` on/off, `-ser 6` (Qwen3 default 8), and IQ4_K or IQ4_KS quants versus mainline Q4_K_M at equal size.
    - Measure: tg, pp, perplexity/KLD.
    - Expected: IQ4_K MMVQ could beat Q4_K on Pascal tg ([PR #2509](https://github.com/ikawrakow/ik_llama.cpp/pull/2509)). Graph mode benefit over x1 is unknown.
    - Effort: medium–high.
14. **Vulkan backend A/B.**
    - Run: mainline built with `GGML_VULKAN=ON`, single card and 9-GPU layer split.
    - Measure: tg/pp versus CUDA.
    - Expected: equal tg and roughly half the pp per the scoreboard ([#10879](https://github.com/ggml-org/llama.cpp/discussions/10879)). Multi-GPU and MoE are unknown.
    - Effort: medium. Value: hedge against the CUDA 12 end of life.
15. **Expert-count reduction.**
    - Run: `--override-kv qwen3moe.expert_used_count=int:6` (mainline) or `-ser 6,1` (ik).
    - Measure: tg gain versus perplexity/KLD and task quality.
    - Expected: tg up roughly in proportion to fewer expert reads; quality drops (claim to test).
    - Effort: low.
16. **Host-side bottleneck profiling.**
    - Run: `nsys` (or `perf top` on the host) during batch-1 decode on 9 GPUs.
    - Measure: GPU idle gaps between kernels, CPU utilisation of the 2 cores, time in sampling/tokeniser.
    - Expected: visible launch-bound gaps, given no CUDA graphs and a 1.8 GHz CPU. Sampling over large vocabularies may show up.
    - Effort: medium.
17. **Workstation as RPC device (optional).**
    - Run: rpc-server with `-c` on the workstation GPU added to the rig's devices.
    - Measure: tg/pp, load time.
    - Expected: capacity gain at a latency cost; disables pipeline parallelism.
    - Effort: medium. Risk: insecure; LAN only.
18. **Prebuilt binary check.**
    - Run: test whether the official Linux CUDA 12.8 nightly runs on sm_61 and on the no-AVX CPU.
    - Expected: unknown. It could avoid building on the 2-core host entirely.
    - Effort: low.

### Open questions
- Does mainline still disable CUDA graphs below Ampere? If so, how large is CPU launch overhead on this host relative to GPU time per token?
- Is Pascal formally supported, best-effort, or headed for removal in mainline? No maintainer statement found.
- Does `-sm tensor` (mainline) or `-sm graph` (ik) ever win over layer split with 9 GPUs on host-staged PCIe 1.0 x1, for decode or for prefill? Is NCCL usable on sm_61 here, and at what VRAM cost?
- Does ik_llama.cpp work at all on a CPU without AVX/AVX2?
- Which quant family (K-quants, IQ4_XS/NL, MXFP4, ik IQ4_K/IQ4_KS/KT) gives the best tg per GB on dp4a sm_61 with 5 SMs?
- How much of the reported ">30 t/s" nine-P106 result (Guru3d/Tom's) is reproducible, and with which flags?
- Is the Guru3d "PCIe 3.0 x1" description or the measured PCIe 1.0 x1 link the right one for these cards? Can the link be raised in BIOS, and does it matter for layer-split decode?
- What is the per-copy latency of host-staged cross-GPU copies on this platform? That is the per-boundary decode cost in layer mode.
- MTP versus ngram versus draft-model on a nine-GPU layer split: does extra MoE expert diversity in verify batches erase the gains?
- Can the official prebuilt Linux CUDA 12.8 binaries (with CPU variants) run on sm_61 and the no-AVX CPU, removing the need to build?
