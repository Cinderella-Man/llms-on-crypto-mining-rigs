# Non-llama.cpp LLM inference engines for MoE on the 9 × P106-090 rig (as of 2026-10-22)

Scope: every inference engine outside the llama.cpp family, checked against this rig: 9 × P106-090
(sm_61, 6 GB, 5 SMs, ~173 GB/s, FP16 at 1/64 of FP32, dp4a, no tensor cores, no bf16), PCIe 1.0 x1 per
GPU with no P2P, a Celeron 3865U (SSE4.2 and AES-NI, no AVX/AVX2/FMA), 4 GB RAM (3.7 GiB usable), ~26 GB
free disk, driver 580, CUDA 12.9 max (12.9.2 installed). Target: MoE models only. Engines that are
llama.cpp/ggml derivatives (PowerInfer, HOBBIT, Lemonade, Cortex, beellama.cpp, llamAmpere) are noted
only where the engine names under investigation or MoE questions bring them up. Strata and Maya bundle
ggml code but are separate engines, so they are covered.
Hardware facts for the rig come from the [hardware spec](../../mining_rig.md).

---

## Q1. What are "maya", "strata" and "llama. infer"?

### Takeaway
"Maya", "Strata" and "llama. infer" are three engine names checked here by name. "Maya" and "Strata" almost certainly mean two MoE-focused local engines that were widely
discussed in September–October 2026: **mw00/project-maya** and **Niko1221/Strata**. Maya's notices
credit Strata code. "llama. infer" has no exact match. The closest candidates are **ninfer**
(Neroued/ninfer, part of the same 2026 wave and credited by Strata), **PowerInfer**, and older or
smaller projects (InferLLM, Inferflow, mann-uofg/infer). Confidence in any of them is low to medium.

### Cited Findings
**Maya: mw00/project-maya (confidence high, ~85 %)**
- Its description: "run GLM-5.3-Flash (321B MoE) and DeepSeek-V4.1-Flash-1017 (748B MoE) privately on your own GPU(s), from one card to 16". It ships an engine, server, dashboard, OpenAI/Anthropic API and quant tools, under MIT. About 1.4k stars, 455 commits. Benchmark tables cite versions v1.0.6–v1.1.17 — [GitHub mw00/project-maya](https://github.com/mw00/project-maya) (fetched 2026-10-22); a ~1.2k-star C++ listing — [Trendshift](https://trendshift.io/repositories/289302).
- Hardware: "NVIDIA, compute capability 7.0 or newer (V100, RTX 20 and newer)"; "Pascal (Tesla P40 / P100, GTX 10) is experimental, not validated yet"; CUDA 13 path "not for the V100 or Pascal"; "32 GB runs it"; it "pins most of the free RAM (all but about 6 GB) for its expert tier"; it needs a fast NVMe; Maya-S quant 96.5 GB, DeepSeek Maya-S 340 GB — [project-maya README](https://github.com/mw00/project-maya).
- Third-party notices: "Strata, ggml/llama.cpp, exllamav3, beellama.cpp" — [project-maya README](https://github.com/mw00/project-maya).
- Other "Maya" repos are not engines: nahidalam/maya is a multilingual multimodal *model* — [GitHub](https://github.com/nahidalam/maya); a 435M "Maya" yes/no safety classifier — [Medium](https://medium.com/@visrow/llm-guards-vs-maya-a-lightweight-435m-yes-no-safety-gate-for-ai-agents-7e9fd5f23142).

**Strata: Niko1221/Strata (confidence high, ~90 %)**
- An MIT C/C++ engine "that runs Qwen3.8-Flash-Next, a 125-billion-parameter model, on consumer PCs", splitting the model across GPU/RAM/SSD. It also runs Qwen3.6-35B-A3B. v0.2.2, engine 0.1.47; 30.2k stars — [GitHub Niko1221/Strata](https://github.com/Niko1221/Strata) (fetched 2026-10-22).
- A Japanese overview (2026-10-06) lists Strata with a link to the fork **dhoard/strata**. That fork has 2 stars and the same v0.2.2 — [npaka, note.com](https://note.com/npaka/n/n974bea5ca338?hl=en); [GitHub dhoard/strata](https://github.com/dhoard/strata).
- stratallm.org calls itself "Unofficial Community Documentation"; its speed figures are inconsistent — [stratallm.org](https://stratallm.org/).
- Strata says it is "Built with parts of llama.cpp / ggml (MIT)" and credits ideas from "Splash, ninfer, and HyperQwen" — [HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md).
- A 2026-10-19 Japanese post calls it "all the rage", and its author had not run it — [Tech千一夜, note.com](https://note.com/techsenichiya/n/n8f8016e5c1ff?hl=en).

**"llama. infer": candidates (no exact match found)**
- **ninfer** (Neroued/ninfer). Upstream README: "from-scratch C++/CUDA engine for a single NVIDIA GeForce RTX 5090", covering Qwen3.5 dense/MoE and Qwen3.6-35B-A3B. It needs CUDA with sm_120a and has "no multi-GPU or distributed serving". Many forks exist — [search results incl. Neroued/ninfer](https://github.com/Neroued/ninfer); [tgui9660 fork, sm_120a](https://github.com/tgui9660/ninfer). One article describes it as "Scratch-written C++/CUDA, no offloading, no multi-GPU, aimed at one RTX 5090, with community forks for the 3090" — [carteakey.dev, 2026-09-30/10-05](https://carteakey.dev/blog/local-inference/the-rise-of-overfit-inference-engines/). It has 2,404 stars and was updated 2026-09-24 — [contextstudios.ai](https://www.contextstudios.ai/de/lokale-ki/neroued--ninfer-qwen36-35b-a3b-5090). *Confidence ~30 %*: same buzz cycle as Maya/Strata, and the name ends in "infer".
- **PowerInfer** (SJTU-IPADS) is a well-known "-Infer" engine. It credits llama.cpp's ggml and runtime — [GitHub PowerInfer](https://github.com/SJTU-IPADS/PowerInfer). *Confidence ~20 %*.
- **InferLLM** (MegEngine) is "a lightweight LLM model inference framework" that "mainly references and borrows from the llama.cpp project". Last news item 2023-08-16; 752 stars; no MoE — [GitHub MegEngine/InferLLM](https://github.com/MegEngine/InferLLM). *Confidence ~10 %*.
- **Inferflow** is a configurable inference engine (2024 paper) — [arXiv 2401.08294](https://arxiv.org/pdf/2401.08294). *~5 %*.
- **mann-uofg/infer** is a C++/CUDA engine for Llama-3.2-1B only — [GitHub](https://github.com/mann-uofg/infer). **Nithin1729S/llama-infer-cpp** is an educational Llama 2 runtime — [GitHub](https://github.com/Nithin1729S/llama-infer-cpp). *~5 % each*.
- No project named "llama.infer", "inferd", "infer.cpp" or "llamainfer" turned up in searches (2026-10-22).
- Others from the same 2026 "overfit engine" wave: DwarfStar/ds4, Splash, llamAmpere (a llama.cpp fork for SM86), gufo — [carteakey.dev](https://carteakey.dev/blog/local-inference/the-rise-of-overfit-inference-engines/). Also TensorFold, Magnitude and Halogen — [npaka](https://note.com/npaka/n/n974bea5ca338?hl=en).

### Inferences
- The names most likely come from a thread or article comparing the 2026 single-model engines (Strata, ninfer, ds4, Maya, Splash). "llama. infer" may be "llama[.cpp], [n]infer" run together, i.e. two separate names.
- These engines are tuned for one model each on modern RTX cards. Pascal is at best "experimental" in them (see Q3).

### Gaps
- No primary source spelled "llama.infer". Which project the name refers to (ninfer, PowerInfer or something else) remains open.

---

## Q2. Toolchain constraints that decide almost every Python engine (PyTorch, Triton, FlashInfer, NumPy, CPU)

### Takeaway
Official PyTorch builds that run on sm_61 are the **CUDA 12.6 wheels up to and including PyTorch 2.14**.
cu128/cu129 wheels from 2.8 start at sm_75, and **2.15 removes CUDA 12.6 entirely**. Community sm_61
builds exist; one fork published a torch 2.13 cu129 sm_61 wheel. Triton (and so most Triton kernels),
FlashInfer (sm75+), FlashAttention (sm80+), Marlin and recent fused-MoE kernels do not officially
support Pascal. NumPy ≥2.4 needs x86-64-v2 (SSE4.2), which the Celeron has. Whether PyTorch wheels run
on a CPU without AVX is not settled.

### Cited Findings
- PyTorch 2.8 cu128/cu129 builds ship "7.5; 8.0; 8.6; 9.0; 10.0; 12.0". **CUDA 12.6 builds keep "5.0; 6.0; 7.0; 7.5; 8.0; 8.6; 9.0"**. A maintainer: "Pytorch is supporting older GPUs in 12.6 builds" (opened 2025-07-03, closed 2026-01-16) — [pytorch#157517](https://github.com/pytorch/pytorch/issues/157517); [dev-discuss](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128).
- The PyTorch 2.15 RFC (opened 2026-07-17, updated 2026-09-23) removes CUDA 12.6. Pre-Turing users "should remain on PyTorch 2.14 (CUDA 12.6) or earlier", and 2.14 is "the final release where ANY CUDA 12.x wheels are published" — [pytorch#190385](https://github.com/pytorch/pytorch/issues/190385); [dev-discuss notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432).
- In practice: "torch 2.8.0 works fine on Pascal when it is the cu126 build". cu128 wheels fail on sm_61, and the failure is silent until a kernel runs — [Whisper-WebUI#653](https://github.com/jhj0517/Whisper-WebUI/issues/653).
- Conflicting source: one table says sm_61 appears only in PyTorch 1.0–1.7.1. That counts explicit sm_61 targets and ignores the sm_60 SASS in cu126 builds — [Qualiteg, updated 2026-09-23](https://journal.qualiteg.com/pytorch_and_supported_gpu_version/). (Explanation: sm_60 cubins run on sm_61 under CUDA's same-major binary compatibility; this is consistent with the cu126 user reports above.)
- A community **torch 2.13.0 wheel built for CUDA 12.9 / sm_61 / CPython 3.12** is published with zxmaster/vllm-pascal-sm61. Without it the build "compiles torch from source, which takes hours" — [GitHub zxmaster/vllm-pascal-sm61](https://github.com/zxmaster/vllm-pascal-sm61).
- **Triton**: triton-windows lists "GTX 10xx (Pascal architecture, sm61) and older" as not supported — [triton-windows README](https://github.com/woct0rdho/triton-windows). A 2022 PyTorch PR said "triton is still buggy with Pascal devices" — [pytorch PR 90020](https://github.com/pytorch/pytorch/pull/90020). vLLM v0.7.2 on a P40 failed with "Triton only supports devices of CUDA Capability >= 7.0" — [vllm#13202](https://github.com/vllm-project/vllm/issues/13202). A Pascal vLLM fork nevertheless runs the `TRITON_ATTN` backend on P40s — [Mikec78660/vLLM-Pascal](https://github.com/Mikec78660/vLLM-Pascal). Another fork replaces "Triton kernels that use sm_70+ atomics" — [zxmaster](https://github.com/zxmaster/vllm-pascal-sm61). Net: Triton works partially on sm_61 with patches.
- **FlashInfer**: oldest supported architecture is SM 7.5 (Turing) — [FlashInfer README](https://cdn.jsdelivr.net/gh/flashinfer-ai/flashinfer@main/README.md); SGLang docs: FlashInfer "only supports sm75 and above" — [SGLang install](https://docs.sglang.io/get_started/install.html).
- **bitsandbytes**: NF4/FP4 needs "6.0+ Pascal (GTX 10X0 series, P100) or newer". CUDA 11.8–12.6 Linux wheels include sm60, 12.8–12.9 wheels start at sm70. "a Turing generation device or newer is recommended" — [bitsandbytes install docs](https://huggingface.co/docs/bitsandbytes/installation).
- **NumPy**: since 2.4 the default x86-64 baseline is X86_V2 (SSE4.2 etc.), and NumPy "will fail to load" on CPUs missing a baseline feature. AVX2/AVX512 are dispatched at runtime — [NumPy CPU/SIMD docs](https://numpy.org/devdocs/building/cpu_simd.html); [Checkmk werk 19914](https://checkmk.com/werk/19914).
- **PyTorch on non-AVX CPUs**: "Illegal instruction" reports on old CPUs are traced to oneDNN code or to mis-detected AVX2 kernels. Older wheels or `-mno-avx` source builds were workarounds, but one user still crashed after disabling AVX — [PyTorch forum 196950](https://discuss.pytorch.org/t/illegal-instruction-core-dumped-when-importing-pytorch/196950); [PyTorch forum: build without AVX](https://discuss.pytorch.org/t/how-to-build-pytorch-without-avx/158408). No official current statement of PyTorch's CPU baseline was found.
- **FP16 on sm_61**: "sm_61 fp16 ALU is 1/64 rate" — [exllamav3#512](https://github.com/turboderp-org/exllamav3/issues/512). The P40 llama.cpp builds route matmuls through INT8/dp4a instead of FP16 — [llama.cpp discussion 25978](https://github.com/ggml-org/llama.cpp/discussions/25978). A Pascal vLLM fork says "fp16 is the only working dtype on CC 6.x" and dequantizes FP8 checkpoints to FP16 at load — [Mikec78660/vLLM-Pascal](https://github.com/Mikec78660/vLLM-Pascal).

### Inferences
- Any PyTorch engine must be pinned to torch ≤2.14 cu126, or to a self-built or community sm_61 wheel (e.g. torch 2.13 cu129 sm_61). Engines that require newer torch, or CUDA 13 (SGLang now, FreeToken, Sonar/Aphrodite cu130), are out.
- FP16 *storage* with FP32 *math* (as cuBLAS GEMMs with FP32 compute do) should run at FP32 speed. The P40 vLLM-fork numbers (Qwen3.5-9B FP16, TP=2, 23 tok/s) look bandwidth-bound, which fits this. Kernels that do native half/half2 arithmetic (many GPTQ/EXL2/AWQ GEMV kernels) would crawl. This is an inference; see claims to test.
- The Celeron's x86-64-v2 support is enough for NumPy ≥2.4. PyTorch wheels are *probably* fine thanks to runtime ISA dispatch, but oneDNN paths have crashed on old CPUs, so this must be tested on the rig.
- Every multi-process Python engine (vLLM/SGLang workers, one per GPU) imports torch and creates a CUDA context per process. On 4 GB of host RAM, 9 such processes are likely to OOM. No source measured this; see claims to test.

### Gaps
- No official list of the exact SASS/PTX targets inside the 2.14 cu126 wheel was retrieved. The 12.6 arch list above comes from the 2.8-era issue.
- No source measured per-worker host RAM for vLLM/SGLang on Pascal.

---

## Q3. Engine-by-engine survey (established engines)

### Takeaway
No mainstream production engine (vLLM, SGLang, Aphrodite/Sonar, TensorRT-LLM, LMDeploy, DeepSpeed-MII,
TGI, ExLlamaV3) officially supports sm_61 in 2026. Their floors are sm_70 to sm_80, and some now need
CUDA 13. The paths with **some real Pascal + MoE evidence** are the **community vLLM-Pascal forks**,
**fastllm/ftllm** (older versions or patched builds), **MLC-LLM** (Vulkan, or CUDA with workarounds),
**mistral.rs/candle** (open pre-Volta build PRs), and plain **HF transformers + accelerate on torch ≤2.14
cu126**. CTranslate2 runs on Pascal but supports no MoE models. KTransformers and PowerInfer need CPU
compute with AVX2-class SIMD and lots of RAM.

### Cited Findings

**vLLM (upstream)**
- Requires "compute capability 7.5 or higher". Binaries "compiled with CUDA 12.9" — [vLLM GPU install docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) (fetched 2026-10-22).
- Upstream never merged Pascal support. PR #4409 noted PyTorch was only shipping sm_60 — [vllm PR 4409](https://github.com/vllm-project/vllm/pull/4409).

**vLLM Pascal forks**
- **zxmaster/vllm-pascal-sm61** is a patch layer, not a fork. It runs upstream **v0.31.0** with torch 2.13 / CUDA 12.9 / sm_61 and supports GGUF and TurboQuant KV. "experimental, tuned for a single 4 GB GTX 1050 Mobile". No multi-GPU or MoE mentioned — [GitHub](https://github.com/zxmaster/vllm-pascal-sm61) (© 2026, 8 commits).
- **Mikec78660/vLLM-Pascal** is a sibling of 1Cat-vLLM (V100). Torch 2.10 built from source with cu124, archs 6.0/6.1, NCCL included. TRITON_ATTN on Pascal; AWQ int4 through a custom split-K GEMV; W8A16 via Exllama WNA16. TP=4 on P40s with NCCL ("P2P is enabled between the P40s", `NCCL_P2P_DISABLE=0`). Numbers: Qwen3.5-9B FP8→FP16 TP=2 P40 23.35 tok/s single-stream; Qwen3.8-27B AWQ TP=2 13.2 tok/s (vs llama.cpp 13.6–13.7); Qwen3.8-27B INT8 TP=4 16.67 tok/s. Its MoE claims (Qwen3.5 35B) have **no supporting numbers**, and Qwen3.8-flash-next is "Pending". Test LXC: 24 GiB RAM + 8 GiB swap. Building torch from source needs a ~27 GB box. Cold boot 5–15 min. "Never pass --enforce-eager". "the x1 link does not limit tok/s" (their x1 context). Last dated note 2026-09-11 — [GitHub](https://github.com/Mikec78660/vLLM-Pascal).
- **Kotali-2019/vllm-pascal**: "dedicated sm_61 build, based on v0.28.0", with `pascal-patches/` and `pascal-patches-triton/`. README gives no tested hardware or perf — [GitHub](https://github.com/Kotali-2019/vllm-pascal).
- **uaysk/vllm-pascal**: P40, torch 2.5.1 + CUDA 12.1, `TORCH_CUDA_ARCH_LIST="6.1"` — [GitHub](https://github.com/uaysk/vllm-pascal) (search snippet).

**SGLang**
- "SGLang requires CUDA 13". "SGLang 0.5.19 is the last release with a CUDA 12 lane (PyTorch 2.13)". FlashInfer is sm75+; the fallback is `--attention-backend triton --sampling-backend pytorch` — [SGLang install docs](https://docs.sglang.io/get_started/install.html).

**Aphrodite Engine (now "Sonar", dphnAI)**
- Install docs: "CUDA wheels support Linux x86-64 and NVIDIA GPUs with compute capability 8.0 or newer". Package name remains `aphrodite-engine`; example uses `cu130` — [Sonar install](https://sonar.dphn.ai/getting-started/installation/); [GitHub aphrodite-engine](https://github.com/aphrodite-engine/aphrodite-engine). It lists tensor/pipeline/data/expert parallelism.

**TensorRT-LLM**
- Oldest supported architecture is Ampere (SM80, SM86). Unlisted architectures get "community support" only (page updated 2025-09-15) — [TRT-LLM support matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html).

**LMDeploy (TurboMind / PyTorch engine)**
- Supported GPUs listed: Volta sm70, Turing sm75, Ampere sm80/86, Ada sm89. No Pascal — [LMDeploy installation](https://lmdeploy.readthedocs.io/en/latest/get_started/installation.html).

**ExLlamaV2 / ExLlamaV3 / TabbyAPI**
- ExLlamaV2: "This project is archived for now", development moved to V3. Multi-GPU is `--gpu_split auto` (layer split). Paged attention needs FlashAttention 2.5.7+. No compute-capability statement — [GitHub exllamav2](https://github.com/turboderp-org/exllamav2).
- ExLlamaV3: "the extension doesn't currently compile for 6.1 at all", because the sm_75 lane uses `mma.m16n8k8`/`ldmatrix`. Anything below sm_80 maps to `CC_OLD`. Its floor is torch ≥2.6, CUDA ≥12.4. An **open proposal (2026-10-14)** adds a dp4a "legacy lane" for sm_61 as a secondary card in layer split. No maintainer reply yet — [exllamav3#512](https://github.com/turboderp-org/exllamav3/issues/512).
- Anecdote: P40 FP16 is poor for exllamav2, while P100 FP16 suits it — [lemmy.toldi.eu](https://lemmy.toldi.eu/post/419265/1928653).

**MLC-LLM (TVM)**
- NVIDIA support on Linux/Windows is "Vulkan, CUDA". No releases or 2025–26 news on the README — [GitHub mlc-llm](https://github.com/mlc-ai/mlc-llm).
- A 2023 1080 Ti build hit `AssertionError: sm61 not supported yet`. Maintainers said to use Vulkan, remove the assert, or pass `--no-cutlass-attn --no-cutlass-norm`. A note that "the latest CUDA runtime likely drops sm_61" — [mlc-llm#802](https://github.com/mlc-ai/mlc-llm/issues/802).
- **MoE on Pascal**: on 4 × P100 (CUDA 12.4), Qwen3-30B-A3B-q4f16_1 with `tensor_parallel_shards=4` failed with `cudaErrorSymbolNotFound` in `inclusive_scan`, while dense Qwen3-32B worked. The reporter said "It's a TVM bug. I got a solution" (fix not described; closed 2025-07-05) — [mlc-llm#3231](https://github.com/mlc-ai/mlc-llm/issues/3231).
- Other Pascal issues: "Pascal cards lost cuBLAS GEMM support" (#1549, 2024), P40 question (#2100) — [mlc-llm issue search](https://github.com/mlc-ai/mlc-llm/issues?q=pascal+OR+sm_61+OR+%22GTX+10%22+OR+P40).

**mistral.rs / Hugging Face candle (Rust)**
- mistral.rs v0.9.5: CUDA with FlashAttention V2/V3, Metal. GGUF 2–8 bit, GPTQ, AWQ, HQQ, FP8, BNB, ISQ. Multi-GPU docs cover "NCCL TP, P2P layer mapping, multi-node, and ring". "MoE optimizations" in v0.8.2. No compute-capability statement — [GitHub mistral.rs](https://github.com/EricLBuehler/mistral.rs).
- Issue "mistralrs cli doesn't work with GTX1070 in podman" is open (2026-03-07). PR "Add opt-in BF16, FP8 and FP4 compatibility paths for older NVIDIA GPUs" is a draft (updated 2026-10-11) — [mistral.rs issue search](https://github.com/EricLBuehler/mistral.rs/issues?q=pascal+OR+%22compute+capability%22+OR+sm_61).
- candle: PR "candle-kernels: build cleanly for Pascal (sm_60) and other pre-Volta GPUs" open (2026-04-25). "Build fails on Maxwell GPU due to __dp4a undefined" open. A bf16 WMMA pre-Ampere fix merged 2026-02-15 — [candle issue search](https://github.com/huggingface/candle/issues?q=pascal+OR+sm_61+OR+%22compute+cap%22).

**CTranslate2**
- CPU minimum "x86-64 processors supporting at least SSE 4.1" with runtime ISA selection. GPU "Compute Capability greater or equal to 3.5" — [CT2 hardware support](https://opennmt.net/CTranslate2/hardware_support.html). Latest 4.8.1 (2026-07-03) — [CERN PyPI mirror](https://simple-repository.app.cern.ch/project/ctranslate2).
- **No MoE**: Qwen 3 "Currently MoE models variants are not supported". Gemma 4 MoE variants not supported. No Mixtral/DeepSeek/OLMoE/GPT-OSS — [CT2 Transformers guide](https://opennmt.net/CTranslate2/guides/transformers.html).

**PowerInfer / PowerInfer-2 / SmallThinker**
- "Now we only support models with ReLU/ReGLU/Squared ReLU activation function". It needs its own "PowerInfer GGUF" with predictor weights. Multi-GPU is an unchecked TODO. Tested on x86-64 with AVX2. Credits ggml/llama.cpp runtime. SmallThinker-21BA3B/4BA0.6B released 2025-07-27 "with a framework for on-device inference" — [GitHub PowerInfer](https://github.com/SJTU-IPADS/PowerInfer).

**KTransformers**
- "Intel AMX and AVX512/AVX2 optimized kernels"; a later update: "Support AVX2-only CPU backend". "Heterogeneous expert placement (hot experts on GPU, cold experts on CPU)". DeepSeek-R1/V3 example on 24 GB VRAM + 382 GB DRAM. Latest dated update 2026-10-15 (v0.7.0) — [GitHub ktransformers](https://github.com/kvcache-ai/ktransformers).

**fastllm / ftllm**
- Supports "张量并行推理稠密模型和混合模式推理MOE模型" (TP for dense, hybrid mode for MoE). `--moe_device` places experts on cuda/cpu/numa/disk. `--tp auto` uses all GPUs, and "支持张量并行、奇数卡数量" (odd GPU counts). The CUDA build needs NCCL. Formats include FP16/BF16/FP8/NVFP4/MXFP4/INT4/K-Quant, AWQ, some GGUF. SM70 kernels ported from TurboMind, with "SM70 及以下始终保持原路径" (SM70 and below keep the original path). `-DCUDA_ARCH` build flag — [GitHub fastllm](https://github.com/ztxz16/fastllm).
- **Pascal evidence**: on a P102-100 (Pascal sm_61 mining card) a user built with `-DCUDA_ARCH=61 -DCUDA_NO_TENSOR_CORE=ON` after patching out FlashInfer. CUDA mode was "not fully stable", and "moe大点的还是会崩" (larger MoE still crashes). Contributor TylunasLi: "v0.1.5.1 … 可以在sm6.1上使用" (works on sm6.1) and "v0.1.6.0 后，项目增加了flashinfer依赖，没有再做sm6.1的适配" (after 0.1.6.0 FlashInfer was added and sm6.1 was not maintained) (2026-05-19…22) — [fastllm#679](https://github.com/ztxz16/fastllm/issues/679).
- On a Tesla P40, ftllm 0.1.7.0 ran Qwen3-Next-80B-A3B with `--device cuda --moe_device cpu`. The first prompt worked and the second hung in the WebUI. Closed 2026-06-11 — [fastllm#696](https://github.com/ztxz16/fastllm/issues/696).
- Merged PRs optimise "旧GPU在float16计算模式下的性能" (old-GPU float16 performance) (2025-06-26) and older-arch builds (2024) — [fastllm PR list](https://github.com/ztxz16/fastllm/issues?q=P40+OR+P100+OR+1080+OR+pascal+OR+sm_61).

**zeux/calm**
- Supports Mixtral 8x7B/8x22B, DBRX, MiniCPM-MoE with fp8 (e5m2)/gf4 weights. Single GPU only ("maximum single-GPU single-batch"). Tested on RTX 4090/A100/H100. **Archived 2025-05-29** — [GitHub calm](https://github.com/zeux/calm).

**tinygrad**
- Backends: CPU, OpenCL, METAL, CUDA, AMD, NV, QCOM, WEBGPU. No GPU-generation statement — [GitHub tinygrad](https://github.com/tinygrad/tinygrad). PR "cuda: fail fast on GPUs without 64-bit stream memory ops" closed 2026-10-07 — [tinygrad PR search](https://github.com/tinygrad/tinygrad/issues?q=pascal+OR+sm_61+OR+%22GTX+10%22+OR+P40).

**DeepSpeed-MII / FastGen**
- DeepSpeed-Kernels: "compute capabilities 8.0+ (Ampere+)" — [deepspeed-kernels](https://github.com/microsoft/deepspeed-kernels); [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-mii).

**Hugging Face TGI**
- "text-generation-inference is now in maintenance mode". The repo "was archived by the owner on Mar 21, 2026" — [GitHub TGI](https://github.com/huggingface/text-generation-inference).

**HF transformers + accelerate / bitsandbytes / HQQ / GPTQ**
- bitsandbytes NF4/FP4 is 6.0+ with sm60 in the ≤12.6 wheels (see Q2) — [bnb docs](https://huggingface.co/docs/bitsandbytes/installation).
- HQQ has pure-PyTorch and custom dequant CUDA backends "suitable for older GPUs". Fast fused 4-bit kernels (torchao, Marlin) are separate — [transformers HQQ docs](https://huggingface.co/docs/transformers/v5.14.0/quantization/hqq.md).
- No source found for GPTQModel/AutoRound on sm_61.

**AirLLM**
- "only ever keeps one layer on the GPU at a time". The model is split and saved layer-wise to disk. Flash-Next needs "~360GB of checkpoint disk" and "a 64GB machine is enough" for its n-gram table. Supports Mixtral, Qwen3-MoE, DeepSeek V2/V3/R1, Kimi K3 per-expert streaming. No multi-GPU. Latest entry 2026/09 — [GitHub airllm](https://github.com/lyogavin/airllm).

**MNN (Alibaba)**
- MNN-LLM supports Qwen/Baichuan/GLM/LLaMA. Backends: CPU (x86 SSE4.1 rated "A", AVX2/AVX512 "S"), OpenCL, Vulkan, Metal, CUDA. Multi-device = "CPU and GPU". No MoE mentioned. MNN 3.6.1 on 2026-07-22 — [GitHub MNN](https://github.com/alibaba/MNN).

**PaddlePaddle FastDeploy**
- Pre-built image "only supports SM80/90 GPUs"; SM86/89 need a package reinstall. Paddle 3.2.0 cu126 — [GitHub FastDeploy](https://github.com/PaddlePaddle/FastDeploy) (via search snippet of install docs).

**OpenVINO**
- The NVIDIA plugin lives in openvino_contrib, is build-from-source only, was validated on a Quadro RTX 4000 with CUDA 11.8, and its clone example uses branch 2022.3.0 — [openvino_contrib nvidia_plugin](https://github.com/openvinotoolkit/openvino_contrib/blob/master/modules/nvidia_plugin/README.md). Mainline OpenVINO targets Intel CPU/GPU/NPU.

**ZML**
- Platforms CUDA/ROCm/oneAPI/TPU/Neuron/CPU. The LLM example "currently supports" Llama 3.1/3.2, Qwen 3.5, LFM 2.5. No MoE. No compute-capability statement — [GitHub zml](https://github.com/zml/zml).

**Burn / CubeCL**
- CubeCL compiles kernels to CUDA PTX, HIP and WGPU (Vulkan/Metal/DX12) — [ferrotorch-cubecl docs](https://docs.rs/ferrotorch-cubecl). No LLM/MoE serving evidence or compute-capability floor found.

**Nexa SDK, Lemonade, Cortex**
- Nexa SDK: NexaML is marketed as a kernel-level engine with GGUF/MLX/.nexa. Older `nexaai` packaging built on ggml (`-DGGML_CUDA=ON`, `-DGGML_VULKAN=on`) — [docs.nexa.ai](https://docs.nexa.ai/); [PyPI nexaai](https://pypi.org/project/nexaai).
- Lemonade: GGUF runs on the llama.cpp backend (CPU, Vulkan, ROCm). ONNX runs on OnnxRuntime GenAI (NPU/iGPU). FastFlowLM targets the AMD XDNA2 NPU — [Lemonade FAQ](https://lemonade-server.ai/docs/guide/faq/); [server spec](https://lemonade-server.ai/docs/server/server_spec).
- Cortex (Jan): "archived by the owner on Jul 4, 2025"; llama.cpp default engine — [GitHub cortex.cpp](https://github.com/janhq/cortex.cpp).

**Chinese-lab engines**
- Chitu (Qingcheng Jizhi/Tsinghua): FP8/FP4 on Ascend, Muxi and "non-H-series NVIDIA GPUs"; pure-GPU/CPU/hybrid; A800 benchmarks — [SCMP 2025-03](https://www.scmp.com/tech/tech-war/article/3302516/chip-war-chinese-start-aims-break-nvidias-grip-ai-new-model-framework). No Pascal evidence.

### Inferences
- **Most plausible non-llama.cpp CUDA paths** (all experimental): (a) fastllm, pinned to v0.1.5.1 or patched, using TP and `--moe_device cuda`; (b) a vLLM-Pascal fork (zxmaster's v0.31 patch set with the published torch 2.13 sm_61 wheel, or Mikec's 1Cat lineage) in PP/TP mode; (c) MLC-LLM, compiled on a separate, more powerful PC on the same LAN (a "workstation"; TVM cross-compiles), using CUDA with cutlass off or Vulkan; (d) mistral.rs with candle pre-Volta patches; (e) HF transformers + accelerate `device_map` as a slow correctness baseline.
- **Host RAM decides between these**: fastllm, mistral.rs and MLC are single-process native runtimes with low host overhead. vLLM spawns one Python worker per GPU. On 4 GB that is a likely blocker beyond 1–2 GPUs.
- A single 6 GB card cannot hold any interesting MoE at 4-bit. Qwen3-30B-A3B is ~17–19 GB at 4-bit (rough estimate), so it spans ≥4 cards. A multi-GPU mode that works without P2P is mandatory.
- Batch-1 decode in layer/pipeline split is bounded by one card's bandwidth at a time, roughly 173 GB/s ÷ active bytes per token (my arithmetic). For a ~3B-active MoE at ~4.5 bpw (~1.7–1.9 GB), the ceiling is ~90–100 tok/s. For a ~12B-active MoE it is ~25 tok/s. Only TP or EP can exceed that, and both pay per-layer synchronisation over x1 links (Q5).
- fastllm's ported TurboMind SM70 kernels and its "SM70 and below keep the original path" switch suggest an old-GPU path still exists in current code. FlashInfer is now the build blocker for sm_61.

### Gaps
- vLLM-Pascal forks: no evidence that fused MoE (Triton `fused_moe`) runs on sm_61, and no MoE numbers anywhere.
- MLC-LLM: whether tensor parallelism works on the Vulkan backend (the docs page fetched did not say). Whether the #3231 TVM fix landed upstream.
- mistral.rs: what the "ring" multi-GPU backend is (possibly non-NCCL), and whether candle's CPU kernels assume AVX.
- tinygrad: Pascal status; whether the 64-bit stream-mem-op check excludes P106.
- No information on LightLLM, KsanaLLM (Tencent), RTP-LLM (Alibaba), xLLM (JD), InfiniLM, or ONNX Runtime GenAI MoE (Phi-3.5-MoE) on Pascal. Not researched in depth; all are datacenter-oriented.

---

## Q4. New 2025–2026 engines (single-model and MoE-specialised)

### Takeaway
The 2026 wave (Strata, Maya, ninfer, colibri, DwarfStar/ds4, FreeToken, Splash, Magnitude, TensorFold)
targets RTX-class cards and usually assumes ≥32 GB host RAM plus NVMe. **Strata** (an experimental sm_61
engine with P40/GTX 10 community reports, a no-P2P layer split, x1 slots OK) and **colibri** (Vulkan
fallback, a pre-Ampere CUDA tier for DeepSeek V4) are the only ones with any Pascal path. Both, as
shipped, are blocked here by host RAM, disk, CPU expert compute, or GPU-count limits.

### Cited Findings
**Strata**
- Pascal: "Pascal sm_61 (P40, P4, GTX 10 series): Also CUDA 12 engine". Setup keeps "a second CUDA 12.9 engine built with `-DSTRATA_EXPERIMENTAL_SM60=ON`", forced with `--cuda 12`. "12 GB or more is recommended". The maintainers test only a Tesla P100. P40 IQ3_S decode 37–38 tok/s (engine 0.1.43.1, 2026-10-12) — [OLDER_GPUS.md](https://github.com/Niko1221/Strata/blob/main/docs/OLDER_GPUS.md).
- v0.1.40.4 (2026-10-08) fixed a Pascal regression (P40 decode 36.4→17.7 tok/s) found by community bisection; "Maintainer Niko does not have Pascal hardware" — [Linux Compatible](https://www.linuxcompatible.org/story/strata-v01404-restores-decode-speed-on-pascal-gpus-after-regression/).
- Multi-GPU: "pipeline (layer) parallelism, not tensor parallelism". "No NVLink or peer-to-peer access is needed; cards on x4 or x1 slots work". Auto placement covers up to four cards (since 0.1.40). "Cards with 8 GB or less start Off". The Limits section says "Every card needs compute capability 7.5", while Pascal is opt-in via `--gpus`. 2 × P40 reported 19.6→34.8 tok/s — [MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md).
- Memory model: "RAM: all 24,576 experts, pinned". "The CPU computes the experts that are not on the GPU in place", with "AVX-512 / AVX2 kernels" from ggml — [HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md).
- Sizes: Qwen3.8 Q2_0 needs 37.6 GB RAM+VRAM, IQ2_XS 39.2 GB, IQ3_XXS 47.0 GB, IQ3_S 54.8 GB (downloads 66–76 GB). Qwen3.6-35B-A3B UD-Q4_K_XL is 22.9 GB. The 29 GB n-gram shard "stays on the SSD" (not stated to be optional). Low-RAM mode "keeps in RAM only what the graphics card does not hold". Rule: "RAM is at least the experts + about 10 GB" — [MODELS.md](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md).

**Project Maya**: CC ≥7.0, Pascal "experimental, not validated", 32 GB RAM, NVMe, smallest quant 94.7 GB — [project-maya](https://github.com/mw00/project-maya).

**ninfer**: sm_120a, a single RTX 5090, no multi-GPU — [Neroued/ninfer](https://github.com/Neroued/ninfer); [carteakey.dev](https://carteakey.dev/blog/local-inference/the-rise-of-overfit-inference-engines/).

**colibri (JustVugg/colibri)**
- "pure C, zero deps, experts streamed from disk". Backends CPU / Vulkan ("any GPU with a Vulkan 1.2 driver") / CUDA / Metal. "DeepSeek V4's CUDA tier also builds for Pascal and Turing (`CUDA_ARCH=portable-pre-ampere NO_TC=1`)". If CUDA can't target the card, setup falls back to Vulkan. "You need a computer with 8 GB of RAM at the very least". "The dense part (attention, shared experts, embeddings) stays in RAM", but "A GPU … holds the hottest experts and the dense layers". "The GPU computes the experts it holds while the CPU computes the rest". All-resident mode `CUDA_EXPERT_GB=auto PIN_GB=all`. 6 × RTX 5090 "every expert resident". Models: GLM-5.2/5.3, Kimi K3, DeepSeek V4 Flash, MiMo-V2.6, Qwen3.8-Flash-Next, Qwen3.6, OLMoE. int4 per-row/gs64, int8. 43.3k stars, v2.0.0. No SIMD requirement stated — [GitHub colibri](https://github.com/JustVugg/colibri); [README](https://github.com/JustVugg/colibri/blob/main/README.md).

**DwarfStar / ds4 (antirez)**
- Runs DeepSeek V4/V4.1 Flash and PRO, GLM 5.x, Qwen3.8 Flash Next, using project-produced GGUFs only. CUDA "can run DeepSeek v4 Flash on Ada Lovelace cards"; DGX Spark is the main target. Qwen Q2 uses 41.73 GiB of weights "plus 95.37 GiB of BF16 n-grams read from disk". "does not link against GGML" but adapts pieces — [GitHub ds4](https://github.com/antirez/ds4).

**FreeToken (FlashML)**
- "native support for NVIDIA RTX 30, RTX 40, and RTX 50" — [InfoQ 2026-08](https://infoq.com/news/2026/08/freetoken-local-inference). The CLI needs driver r580+, **CUDA 13**, Python 3.10+ (third-party review) — [Wavect review](https://wavect.io/blog/freetoken-ai-inference-engine-review/).

**Splash**: Apple Silicon only (Qwen3.6-35B-A3B, Qwen3.8-27B) — [LM Studio blog](https://lmstudio.ai/blog/splash-engine).
**Magnitude**: Apache-2.0 desktop app, "Any Apple Silicon, NVIDIA, or AMD GPU", tunes kernels per chip. No MoE or Pascal details — [magnitude.dev](https://magnitude.dev/).
**TensorFold**: decode-focused, speculative decoding and model-specific kernels, Apple Silicon/NVIDIA — [npaka](https://note.com/npaka/n/n974bea5ca338?hl=en) (repo github.com/ashhart/TensorFold not fetched).

### Inferences
- Strata on this rig, as shipped, hits several blockers. RAM is 4 GB against "experts + ~10 GB". The CPU computes uncached experts with AVX2/AVX-512 kernels the Celeron lacks. Disk has 26 GB free against 66–76 GB downloads plus a 29 GB n-gram shard. 6 GB cards "start Off". Auto placement handles ≤4 cards. Even so, 54 GB of VRAM is more than IQ2_XS's 39.2 GB, so a 9-way layer split with low-RAM mode and every expert in VRAM is *conceivable*. That would make it the only engine here with a named large MoE plus a Pascal kernel path, but it is unsupported territory.
- colibri: the stated 8 GB RAM floor and the "dense part stays in RAM" default both conflict with 4 GB. Its Vulkan 1.2 fallback and pre-Ampere CUDA tier make it the second candidate for a "does it start" test.
- Upgrading the rig's RAM (one free SO-DIMM slot, up to 32 GB per the [hardware spec](../../mining_rig.md)) would turn several "blocked by RAM" verdicts (Strata, colibri, Maya's 32 GB) into "testable". The no-AVX CPU and 6 GB cards would still be problems.

### Gaps
- Whether Strata's n-gram table can be skipped. Whether more than 4 cards and ≤8 GB cards work when forced with `--gpus`.
- colibri's CPU SIMD requirements, and whether it runs below 8 GB RAM when everything is VRAM-resident.
- Magnitude and TensorFold backends and Pascal status are unknown.

---

## Q5. MoE-specific offloading/caching systems, and expert parallelism over PCIe x1 without P2P

### Takeaway
The MoE offloading systems (MoE-Infinity, Fiddler, HOBBIT, ProMoE, Pre-gated MoE, MoBiLE, plus
KTransformers/Strata/colibri-style tiering) assume host RAM or CPU compute to hold or run cold experts.
That is exactly what this rig lacks, so their main idea does not apply. What applies is the reverse:
spread *all* experts across nine small GPUs and use EP/TP so several cards work on one token. vLLM's EP
runs on consumer GPUs without P2P via host-staged transfers, and a Sept-2026 paper (ThunderEP) optimises
exactly that. But EP gains are shown at batch ≥6, and small batches are said to favour TP. No engine
with a working sm_61 path has demonstrated EP.

### Cited Findings
- **MoE-Infinity**: offloads experts to host memory/SSD with activation-aware caching and prefetch. "The from-source build targets compute capabilities sm_80/sm_90 by default". "Single-server multi-GPU inference is supported, with expert parameters distributed round-robin across visible GPUs". Expert parallelism is on the roadmap. Suggests ≥16 GB VRAM — [GitHub MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity).
- **HOBBIT** is "built on llama.cpp" and swaps cache-miss experts for low-precision versions (up to 9.93× vs other offloaders) — [arXiv 2411.01433](https://arxiv.org/pdf/2411.01433). **ProMoE** uses proactive GPU-side expert caching (2.07× decode vs offloading baselines) — [arXiv 2410.22134](https://arxiv.org/html/2410.22134v3). **Fiddler** relies on offline expert-popularity profiling; **Pre-gated MoE** trains extra gates to prefetch next-layer experts — both summarised in [MoBiLE, arXiv 2510.12357](https://arxiv.org/pdf/2510.12357).
- **ThunderEP** (SNU, arXiv 2609.40093, 2026-09-30): EP dispatch/combine for "PCIe-Connected Consumer GPUs". "Every transfer must instead be staged through a bounce buffer in CPU memory". Tested systems: "Neither system supports GPUDirect P2P access between any pair of GPUs". Integrated into vLLM 0.27.1 (vs SGLang 0.5.18). Decode batches 6–96 only, with no overlap in decode "because each GPU processes too few tokens". 1.12–1.17× average decode throughput over vLLM. No code URL in the paper — [arXiv HTML](https://arxiv.org/html/2609.40093).
- vLLM EP on consumer hardware: "You'll have to replicate the non-expert layers per device, but EP runs fine". One user reported ~8–12 tok/s decode at 1M context over 10 GbE (2026-08-17) — [vLLM forum](https://discuss.vllm.ai/t/expert-parallelism-all-to-all-communication-without-nvlink-and-deepep/2421/5).
- "TP decodes small batches with lower latency and EP sustains throughput with large batches" (8×H200) — [Moebius, arXiv 2606.26607](https://arxiv.org/html/2606.26607v1).
- Engines that advertise EP: vLLM (upstream), Aphrodite/Sonar ("Tensor, pipeline, data, and expert parallelism"), SGLang, TensorRT-LLM — [Aphrodite README](https://github.com/aphrodite-engine/aphrodite-engine). All are sm_70+/sm_75+/sm_80+ (Q3).
- Layer-split engines without P2P that work over x1: Strata ("cards on x4 or x1 slots work"; inter-card traffic is "a few hundred KB through pinned RAM" per verify window) — [MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md). Maya splits DeepSeek layers across GPUs and uses TP over 4 or 8 GPUs for `maya-ds4` — [project-maya](https://github.com/mw00/project-maya).
- fastllm TP supports odd GPU counts — [fastllm](https://github.com/ztxz16/fastllm). The Mikec vLLM-Pascal fork's TP uses NCCL ("ncclAllGather … over the shared-memory transport on this box") — [vLLM-Pascal](https://github.com/Mikec78660/vLLM-Pascal).

### Inferences
- With no CPU offload and 4 GB RAM, the useful "MoE-aware" feature here is **placement across GPUs**: experts spread across nine cards, non-expert layers replicated or pinned to one card. MoE-Infinity's round-robin multi-GPU expert placement is conceptually the closest. Its offload cache would be pointless if everything fits, and it targets sm_80+.
- Back-of-envelope (my arithmetic, to be measured): batch-1 EP needs, per MoE layer, one scatter of the hidden state (~4–8 KB at fp16 for hidden 2–4k) to the GPUs owning the selected experts, plus a gather. At ~190 MB/s per GPU that is ~20–40 µs of transfer plus launch/sync latency per hop. About 48 layers × 2 hops ≈ 2–6 ms/token of overhead, which is small against a ~10–20 ms/token bandwidth-bound decode. TP needs 2 all-reduces per layer. A 9-rank ring over host memory is 16 steps per all-reduce and could cost tens of ms per token. These numbers decide whether EP/TP beats layer split on this rig, and they are unmeasured.
- EP in vLLM replicates non-expert layers per GPU (per the forum). On 6 GB cards that is affordable for ~3B-active models but eats KV-cache room.
- No engine combines a working sm_61 path + MoE + EP today. The closest stack would be a vLLM-Pascal fork with EP enabled (`--enable-expert-parallel` on vLLM ≥0.2x), which needs the Triton fused-MoE path to work on sm_61 (unknown).

### Gaps
- No measured batch-1 EP latency on PCIe without P2P anywhere (ThunderEP starts at batch 6).
- No public ThunderEP code found.
- Unknown whether colibri's multi-GPU mode computes different GPUs' experts in parallel within a layer.

---

## Q6. Vulkan/OpenCL backends on Pascal

### Takeaway
Non-llama.cpp engines with Vulkan/OpenCL that could reach a P106: **MLC-LLM** (Vulkan; maintainers
pointed sm_61 users there), **colibri** (Vulkan 1.2), **MNN** (Vulkan/OpenCL/CUDA), **tinygrad**
(OpenCL/WebGPU), **Burn/CubeCL** (wgpu) and **Nexa** (Vulkan via ggml in older builds). The only Pascal
Vulkan-vs-CUDA data point found (llama.cpp on a P40) shows decode ~10 % slower and prefill ~10× slower
on Vulkan. Multi-GPU over Vulkan is not documented for MLC or colibri.

### Cited Findings
- MLC-LLM NVIDIA support is "Vulkan, CUDA" — [mlc-llm README](https://github.com/mlc-ai/mlc-llm). For a 1080 Ti: "you can follow the instruction with vulkan for this device" (tqchen, 2023-10-24) — [mlc-llm#802](https://github.com/mlc-ai/mlc-llm/issues/802).
- colibri: Vulkan works on "any GPU with a Vulkan 1.2 driver" and keeps "the dense layers on the card" on discrete GPUs. A V100 user saw 17–19 tok/s Qwen3.6 via the fallback — [colibri README](https://github.com/JustVugg/colibri/blob/main/README.md).
- MNN backends: OpenCL, Vulkan, Metal, CUDA — [MNN](https://github.com/alibaba/MNN). tinygrad: OpenCL, WEBGPU, CUDA, NV — [tinygrad](https://github.com/tinygrad/tinygrad). CubeCL: CUDA PTX, HIP, WGPU — [ferrotorch-cubecl](https://docs.rs/ferrotorch-cubecl).
- P40 llama.cpp: Vulkan TG 5.1 tok/s vs CUDA 5.7 tok/s (Llama-3.3-70B IQ4_NL); prompt processing "about 10 times slower with Vulkan" (2026-02-15) — [FreeBSD forums](https://forums.freebsd.org/threads/report-on-running-llm-inference-on-freebsd.101694/latest).
- On newer RTX hardware NVIDIA reports the llama.cpp Vulkan back-end becoming competitive with CUDA — [Phoronix](https://www.phoronix.com/news/NVIDIA-Vulkan-AI-ML-Success). An architecture-coverage caveat: Qwen3.5 MoE DeltaNet layers lacked a Vulkan implementation in llama.cpp, while Qwen3-30B-A3B worked — [llama.cpp#19957](https://github.com/ggml-org/llama.cpp/issues/19957).

### Inferences
- On Pascal, Vulkan is a way to *run at all* when CUDA builds refuse sm_61. Decode (memory-bound) can come close to CUDA, but prefill suffers because there are no tensor cores, no cooperative matrix, and FP16 is slow.
- For MoE across 9 GPUs, Vulkan engines would need their own multi-device support. None of the non-llama.cpp ones document it. So Vulkan here is mainly a single-GPU path for small MoEs (e.g. OLMoE-1B-7B or SmallThinker-4B-A0.6B class at 4-bit).

### Gaps
- No `vulkaninfo` data for P106-090 on driver 580: Vulkan version, `shaderFloat16`, `VK_KHR_shader_integer_dot_product` (dp4a exposure).
- No MLC-LLM Vulkan MoE benchmark on Pascal.

---

## Q7. Synthesis: feasibility table, claims to test, experiment ideas

### Takeaway
Most engines can be ruled out on documented architecture floors. A short list stays plausible but
unproven: fastllm, vLLM-Pascal forks, MLC-LLM, mistral.rs/candle, HF transformers + accelerate, and,
as hacks, Strata and colibri. In each case, host RAM (4 GB), the no-AVX CPU, and multi-GPU without P2P
are the deciding unknowns. These are options, not a recommendation.

### Feasibility table

Legend: ✔ = documented, ~ = partial/community/patched, ✘ = documented unsupported, ? = unknown.

| Engine | Pascal sm_61? | MoE? | Multi-GPU w/o P2P? | Host-RAM / no-AVX blockers | Verdict |
|---|---|---|---|---|---|
| vLLM upstream | ✘ CC ≥7.5 ([docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)) | ✔ | ✔ TP/PP/EP via NCCL (host-staged) | 1 Python worker per GPU; likely OOM at 4 GB | **Infeasible** (arch floor) |
| vLLM-Pascal forks (zxmaster v0.31, Mikec/1Cat, Kotali v0.28, uaysk) | ~ community, torch 2.10–2.13 sm_61 builds | ~ claimed, unproven (no MoE numbers) | ~ TP via NCCL shown on P40 *with* P2P | Multi-process RAM; torch on no-AVX untested | **Plausible-but-hard** (research) |
| SGLang | ✘ needs CUDA 13; FlashInfer sm75+ | ✔ | ✔ | multi-process | **Infeasible** |
| Aphrodite / Sonar | ✘ CC ≥8.0 wheels | ✔ | ✔ (incl. EP) | multi-process | **Infeasible** |
| TensorRT-LLM | ✘ Ampere+ | ✔ | ✔ | — | **Infeasible** |
| LMDeploy (TurboMind) | ✘ sm70+ listed | ✔ | ✔ | — | **Infeasible** |
| DeepSpeed-MII/FastGen | ✘ CC ≥8.0 | ✔ | ✔ | — | **Infeasible** |
| HF TGI | ? (archived 2026-03-21) | ✔ | ✔ | — | **Infeasible / dead** |
| ExLlamaV2 / TabbyAPI | ? (archived; FP16-heavy kernels) | ✔ Mixtral-class | ~ layer split | low | **Unlikely** (FP16 1/64 rate) |
| ExLlamaV3 / TabbyAPI | ✘ doesn't compile for 6.1 (#512 proposal open) | ✔ | ~ layer split / TP | low | **Infeasible now; watch #512** |
| MLC-LLM | ~ Vulkan route; CUDA needs cutlass off; Qwen3-MoE TP failed on P100 (#3231) | ✔ (Qwen3-30B-A3B compiled) | ? TP via NCCL (CUDA); Vulkan multi-GPU unknown | single process; compile on workstation | **Plausible-experimental** |
| mistral.rs / candle | ~ pre-Volta build PR open; GTX1070 issue open | ✔ (GGUF/ISQ, "MoE optimizations") | ? NCCL TP / layer mapping / "ring" | Rust binary, low RAM; CPU SIMD ? | **Plausible-with-patches** |
| CTranslate2 | ✔ CC ≥3.5, SSE4.1 | ✘ no MoE | ✔ TP | fine | **Infeasible for MoE** |
| PowerInfer / SmallThinker fw | ? (ggml) | ✘ ReLU-sparse only | ✘ TODO | AVX2-tested CPU path | **Infeasible** |
| KTransformers | ? | ✔ | ✔ | CPU experts need AVX2+/large RAM | **Infeasible** |
| fastllm / ftllm | ~ v0.1.5.1 on sm6.1; later needs FlashInfer removal; P102/P40 reports | ✔ (`--moe_device cuda`) | ~ TP incl. odd counts via NCCL | native; RAM/CPU SIMD ? | **Plausible-but-fragile** |
| zeux/calm | ? | ✔ Mixtral/DBRX | ✘ single GPU | — | **Infeasible** (archived, 6 GB) |
| tinygrad | ? | ? (no MoE app documented) | ? | ? | **Unknown / research toy** |
| HF transformers + accelerate (+bnb/HQQ) | ✔ on torch ≤2.14 cu126 | ✔ | ✔ `device_map` (sequential) | single process; torch no-AVX untested | **Feasible baseline, slow** |
| AirLLM | ~ (PyTorch) | ✔ | ✘ | disk streaming over x1/SATA | **Infeasible in practice** |
| MNN | ? (CUDA/Vulkan/OpenCL) | ? | ✘ not documented | SSE4.1 OK | **Unlikely** |
| FastDeploy | ✘ SM80/90 images | ✔ | ✔ | — | **Infeasible** |
| OpenVINO | ✘ (NVIDIA plugin stale, contrib) | ~ | ✘ | — | **Infeasible** |
| ZML | ? | ✘ not listed | ? sharding | — | **Unlikely** |
| Burn / CubeCL | ? | ? | ? | — | **Unknown** |
| Nexa SDK | ? (ggml heritage) | ? | ? | — | **Unknown / llama.cpp-adjacent** |
| Lemonade | via llama.cpp (Vulkan) | via llama.cpp | via llama.cpp | — | **= llama.cpp** |
| Cortex | via llama.cpp | — | — | — | **Dead (archived 2025-07-04)** |
| Project Maya | ~ "experimental, not validated"; CC ≥7.0 | ✔ GLM-5.3-Flash / DS-V4.1-Flash | ✔ up to 16 (layer split, TP pairs) | 32 GB RAM, ≥94.7 GB model, NVMe | **Infeasible** |
| Strata | ~ experimental CUDA 12.9 sm_60/61 engine, P40/GTX10 reports | ✔ Qwen3.8-Flash-Next 125B, Qwen3.6-35B-A3B | ✔ layer split, no P2P, x1 OK; auto ≤4 cards; ≤8 GB cards off | experts pinned in RAM (+10 GB) or low-RAM mode; CPU experts AVX2/512; 29 GB n-gram shard | **Blocked as shipped; hack-test only** |
| ninfer | ✘ sm_120a | ✔ Qwen3.6-35B-A3B | ✘ | — | **Infeasible** |
| colibri | ~ DS-V4 CUDA tier pre-Ampere; Vulkan 1.2 fallback | ✔ many | ~ multi-GPU resident experts (details ?) | ≥8 GB RAM stated; dense in RAM by default; CPU SIMD ? | **Blocked as shipped; hack-test only** |
| DwarfStar ds4 | ? (Ada named) | ✔ | ✔ (8×L40S) | 95 GiB n-grams on disk | **Infeasible** |
| FreeToken | ✘ CUDA 13, RTX 30+ | ✔ | ? | — | **Infeasible** |
| Splash | ✘ Apple only | ✔ | — | — | **Infeasible** |
| Magnitude / TensorFold | ? | ? | ? | ? | **Unknown** |
| MoE-Infinity | ~ (PyTorch; default sm_80/90 build) | ✔ | ✔ round-robin experts | offload design assumes RAM/SSD | **Unlikely** |
| Fiddler / HOBBIT / ProMoE / Pre-gated MoE | research code | ✔ | ✘ mostly single GPU | CPU/host-RAM offload | **Not applicable** |

### Claims to test (all unverified for this rig)
1. PyTorch 2.14 cu126 (and the community torch 2.13 cu129 sm_61 wheel) imports and runs matmuls on the no-AVX Celeron 3865U without "Illegal instruction" (oneDNN risk).
2. Host RSS of one `import torch; torch.cuda.init()` process, and of a process holding contexts on all 9 GPUs. This decides multi-process engines (vLLM) vs single-process ones.
3. NCCL all-reduce/all-gather latency for 4–64 KB messages across 2/4/8/9 GPUs with no P2P (SHM transport via host). This bounds TP/EP decode overhead.
4. Native FP16 kernels crawl on GP106 while FP16-storage/FP32-math cuBLAS GEMMs run at FP32 speed (decides which quant kernels are usable).
5. fastllm v0.1.5.1 (and current with FlashInfer patched out) runs a MoE with `--moe_device cuda --tp N` on sm_61 without the crashes reported in #679.
6. Triton fused-MoE and TRITON_ATTN kernels in the vLLM-Pascal forks work on sm_61 for MoE models.
7. The MLC-LLM #3231 TVM scan bug is fixed upstream; MoE compiles and runs with `tensor_parallel_shards>1` on sm_61 (CUDA, cutlass off).
8. Strata's experimental sm_60/61 engine accepts 6 GB cards and >4 GPUs when forced (`--gpus`), and low-RAM mode works with ~4 GB host RAM when all experts are VRAM-resident. The n-gram shard is optional.
9. colibri runs below its stated 8 GB RAM floor when dense layers and experts are GPU-resident (Vulkan or pre-Ampere CUDA).
10. Vulkan on P106 (driver 580) exposes integer dot product, and prefill is ~10× slower than CUDA as the P40 anecdote suggests.
11. Batch-1 EP over x1 without P2P costs only a few ms/token (my estimate).

### Experiment ideas (options, not decisions)

**Common build recipe (workstation → rig)**
- Build in Docker on the workstation from `nvidia/cuda:12.9.2-devel-ubuntu24.04`, so glibc, Python 3.12 and CUDA match the rig. Set `TORCH_CUDA_ARCH_LIST="6.1"` / `CMAKE_CUDA_ARCHITECTURES=61`.
- **Compile all host code for `-march=x86-64-v2` (or `-mno-avx -mno-avx2 -mno-fma`), never `-march=native`.** Code built natively on a modern workstation CPU would emit AVX2 that crashes on the Celeron. For Rust use `RUSTFLAGS="-C target-cpu=x86-64-v2"`.
- Ship wheels or binaries with rsync. Keep models on the workstation and serve them over NFS on 1 GbE (~110 MB/s), since the rig has ~26 GB free and GPU loading is PCIe-bound anyway (~715 MB/s aggregate, ~80 s to fill all 9 cards per the [hardware spec](../../mining_rig.md)). Run everything read-only and as non-root where possible.

**E0. Platform probes (low effort, low risk; prerequisite for all others)**
- Install torch 2.14 cu126 in a venv. Run claims 1, 2 and 4: a matmul microbench of fp16 vs fp32 with `torch.backends.cuda.matmul` defaults, plus RSS logging.
- Build `nccl-tests` (CUDA 12.9, sm_61, x86-64-v2) on the workstation, copy over, and run `all_reduce_perf`/`all_gather_perf` for 4 B–1 MB across 2/4/8/9 GPUs (claim 3).
- Write a tiny CUDA host-staged scatter/gather microbench (pinned buffers, events) to bound EP latency (claim 11).
- Run `vulkaninfo` (claim 10).
- Measure: latency in µs per collective, RSS, illegal-instruction yes/no.

**E1. HF transformers + accelerate baseline (low–medium effort, low risk)**
- Model: start with OLMoE-1B-7B in 4-bit (~4 GB by my estimate, one card; fp16 would be ~14 GB), then Qwen3-30B-A3B in bnb NF4 or HQQ 4-bit with `device_map="auto"` across 4–5 GPUs. Pin torch ≤2.14 cu126 and bnb with sm60 in the wheel.
- Measure decode tok/s, prefill tok/s, per-GPU VRAM, host RSS, load time. This gives a correctness and quality reference for the faster engines. Expect slow decode (Python per-expert loops, sequential GPUs).

**E2. fastllm / ftllm (medium effort, medium risk)**
- Build two variants in Docker with NCCL: tag **v0.1.5.1** (contributor-reported sm6.1-OK) and current HEAD with FlashInfer conditionally compiled out (per #679), both with `-DUSE_CUDA=ON -DCUDA_ARCH=61 -DCUDA_NO_TENSOR_CORE=ON`.
- Model: Qwen3-30B-A3B INT4 (AWQ or fastllm export, made on the workstation), then Qwen3-Next-80B-A3B INT4 if it fits ~50 GB.
- Run `--device cuda --moe_device cuda --tp 0,1,2,3`, then `--tp auto` (9 GPUs).
- Measure decode tok/s vs TP degree (1/2/4/8/9). The TP-scaling curve shows whether x1/no-P2P all-reduce kills it. Also measure stability over multi-turn chats, host RSS and CPU load on the 2 cores.

**E3. vLLM-Pascal fork (high effort, high risk)**
- Use zxmaster's v0.31 patch set with its published torch 2.13 cu129 sm_61 wheel (avoids a multi-hour torch build), built in Docker on the workstation.
- Start single-GPU with a small MoE GGUF/AWQ (e.g. OLMoE-1B-7B or a ≤5 GB MoE). Then try `--pipeline-parallel-size 4` and `--tensor-parallel-size 2/4` with Qwen3-30B-A3B AWQ/GGUF. Optionally `--enable-expert-parallel`.
- Measure whether fused MoE runs (claim 6), host RSS per worker (expect the 4 GB RAM wall at ≥2–3 workers), decode tok/s, cold-start time.
- Alternative lineage: Mikec78660/vLLM-Pascal (torch 2.10 from source; needs a ~27 GB build box, which the workstation can provide if it has that much memory).

**E4. MLC-LLM (medium–high effort, medium risk)**
- On the workstation, `mlc_llm convert_weight` + `compile` Qwen3-30B-A3B with `q4f32_1` (avoids FP16 math; check it exists for this arch) or `q4f16_1`. Target CUDA sm_61 with cutlass/FlashInfer disabled and `tensor_parallel_shards=4`, plus a Vulkan single-GPU build of a small MoE.
- Copy the `.so` model libs and weights. Watch for the #3231 `inclusive_scan` failure.
- Measure decode/prefill tok/s for CUDA TP=4 vs Vulkan TP=1 (small model), and host RSS.

**E5. mistral.rs (medium–high effort, medium risk)**
- Build mistral.rs against a candle checkout carrying the pre-Volta PR (#3494) with CUDA 12.9 `CUDA_COMPUTE_CAP=61` and `target-cpu=x86-64-v2`. No FlashAttention features.
- Model: Qwen3-30B-A3B GGUF Q4_K_M with device-layer mapping across 4–5 GPUs; try the NCCL TP feature if it builds.
- Measure decode tok/s, whether dp4a-based GGUF kernels compile on sm_61, and RSS.

**E6. Strata "does it start" hack (high effort, high risk)**
- Use the experimental CUDA 12.9 engine (`-DSTRATA_EXPERIMENTAL_SM60=ON`, `--cuda 12`, `--gpus 0-8`, `--low-ram on`). Model: Qwen3.6-35B-A3B UD-Q4_K_XL (22.9 GB) first, then Qwen3.8 IQ2_XS (39.2 GB), served over NFS. Check whether the n-gram shard can be omitted.
- Measure whether it accepts 6 GB cards and 9 cards, whether CPU expert kernels hit AVX (crash), decode tok/s, RSS.
- Build on the workstation with host code at x86-64-v2. Strata's own CPU kernels may need AVX2 regardless.

**E7. colibri Vulkan/CUDA probe (medium effort, medium risk)**
- Build on the workstation (gcc/OpenMP, x86-64-v2) with the Vulkan backend and the `CUDA_ARCH=portable-pre-ampere NO_TC=1` tier.
- Model: OLMoE (smallest listed; 8 GB RAM stated), then Qwen3.6, with dense layers and experts on GPU (`CUDA_EXPERT_GB=auto`).
- Measure whether it runs at 4 GB RAM (swap off vs on), Vulkan vs CUDA decode, multi-GPU behaviour.

**E8. Optional hardware option, not a software test**
- Re-run the RAM-blocked candidates (Strata, colibri, vLLM multi-worker) after filling the free SO-DIMM slot (up to 32 GB per the [hardware spec](../../mining_rig.md)). That separates "blocked by 4 GB RAM" from "blocked by Pascal/CPU".

### Gaps
- No engine in this survey has published MoE results on any 6 GB Pascal card. Every positive verdict above rests on extrapolation from P40/P100/P102 reports.
- Per-engine CPU SIMD requirements (AVX vs SSE4.2) are undocumented for fastllm, colibri, mistral.rs/candle, MLC runtime and Strata's non-expert paths. Only CTranslate2 (SSE4.1) and MNN (SSE4.1 rated "A") state them.
- Release dates for several repos (Maya, Strata v0.2.2, MLC, tinygrad, ZML) were not visible on fetched pages.
