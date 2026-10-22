# The "infer" engine (a name mentioned alongside llama.cpp, Maya and Strata): identification and feasibility on the 9 × P106-090 rig (as of 2026-10-22)

Scope: work out which engine the name "infer" refers to, then assess it for 9 × P106-090. The cards are
GP106 / sm_61, 6 GB each, PCIe 1.0 x1, with no P2P. The host is a no-AVX Celeron with 4 GB RAM (3.7 GiB
usable), on CUDA 12.9.2 / driver 580 ([hardware spec](../../mining_rig.md)). Target: MoE only. All GitHub facts (stars, forks, push dates, file contents) were read
through the GitHub API/`gh` on 2026-10-22 unless stated otherwise. Maya, Strata and the older ninfer
summary are in [`engines_other.md`](engines_other.md) and are not repeated here, except where this file adds
detail.

---

## Q1. Which project is "infer"?

### Takeaway
Nothing is literally called "infer" in the 2026 local-engine wave. The best match is **ninfer / NInfer**
(Neroued/ninfer), confidence about 60 %. It is the only "-infer" engine that 2026 threads and roundups
keep putting next to Strata (and Maya). Strata credits it, and it has a large fork ecosystem. The
runner-up is **DeepInfer** (deepseek-ai/DeepInfer), confidence about 20 %. It went public on 2026-10-20
with huge buzz and targets DeepSeek-V4.1-Flash-1017, the same model as Maya, but it is a datacenter
engine. Everything else is 5 % or less. Whichever is meant, the rig verdict is the same (see Q3), so the
identification changes nothing for this rig.

### Cited Findings
**ninfer / NInfer (Neroued/ninfer), about 60 %**
- On HN, "ninfer" is benchmarked directly against Strata. One user ran "ninfer-3090-qwen3.8-27b … vs strata-qwen3.8-flash-next-iq3_xxs" in the "Run Qwen 3.8 Flash Next (125B) on consumer hardware" thread (2026-10-04) — [HN 49958120](https://news.ycombinator.com/item?id=49958120). Another comment in the same thread linked Neroued/ninfer — [HN 49953881](https://news.ycombinator.com/item?id=49953881).
- Other 2026 HN mentions: launch posts "NInfer: 271 tok/s (543 with speculative) for Qwen3.6-35B-A3B on one RTX 5090" (2026-07-23) — [HN 49020083](https://news.ycombinator.com/item?id=49020083) and "Ninfer: High-performance single-GPU inference" (2026-07-28) — [HN 49081189](https://news.ycombinator.com/item?id=49081189). A comment says "There is also a Ninfer fork for the v100 but it requires a custom format" (2026-09-30) — [HN 49911612](https://news.ycombinator.com/item?id=49911612). A "patched ninfer inference engine" was used for Qwen3.8 at 255 tok/s (2026-10-13) — [HN 50080273](https://news.ycombinator.com/item?id=50080273).
- Strata's HOW_IT_WORKS says Strata was built with ggml and takes "Ideas from Splash, ninfer and HyperQwen". Neither Strata's README nor Maya's README mentions any other "-infer" project (fetched 2026-10-22) — [Strata HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md); [project-maya README](https://github.com/mw00/project-maya).
- Roundups: "The Rise of Overfit Inference Engines" (2026-09-30, updated 2026-10-05) profiles Strata, ninfer, DwarfStar/ds4, Splash, llamAmpere and gufo. Only ninfer has a name ending in "infer". Maya is not mentioned — [carteakey.dev](https://carteakey.dev/blog/local-inference/the-rise-of-overfit-inference-engines/). Startup Fortune (2026-10-04) names the same six and writes both "ninfer" and "NInfer" — [Startup Fortune](https://startupfortune.com/a-wave-of-narrow-ai-inference-engines-is-beating-vllm-and-llamacpp-at-their-own-game/).
- The awesome-local-inference-recipes engine index lists llama.cpp (53 recipes), Strata (47), Project Maya (4), ninfer (3), Infernix (2) and sparkinfer (1). There is no engine called plain "infer" — [raytone-lab/awesome-local-inference-recipes](https://github.com/raytone-lab/awesome-local-inference-recipes) (README, 2026-10-22).
- Repo facts: 3,134 stars and 633 forks. Created 2026-06-26, last push 2026-10-22, Apache-2.0, C++. About 1,274 commits, 1,243 of them by Neroued. **No tags or GitHub releases** — [GitHub API: Neroued/ninfer](https://github.com/Neroued/ninfer). The earlier figure of 2,404 stars on 2026-09-24 — [contextstudios.ai](https://www.contextstudios.ai/de/lokale-ki/neroued--ninfer-qwen36-35b-a3b-5090) — shows it is still growing.

**DeepInfer (deepseek-ai/DeepInfer), about 20 %**
- Repo created 2026-10-16, publicly announced 2026-10-20. 19,038 stars and 1,989 forks by 2026-10-22, Apache-2.0, Python — [GitHub deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer). The HN launch post had 2,320 points on 2026-10-20 — [HN 50163277](https://news.ycombinator.com/item?id=50163277). DeepSeek announced it as "our in-house inference framework, now open source" — [HN 50163812 → X](https://news.ycombinator.com/item?id=50163812).
- Its only supported model is DeepSeek-V4.1-Flash, default checkpoint `DeepSeek-V4.1-Flash-1017` — [DeepInfer README](https://github.com/deepseek-ai/DeepInfer). That is the same checkpoint Maya targets ("DeepSeek-V4.1-Flash-1017 (748B MoE)") — [project-maya](https://github.com/mw00/project-maya). Anyone reading about local DeepSeek-V4.1-Flash in the week of 2026-10-20 would plausibly have seen both names.

**Other "-infer" names checked (each 5 % or less)**
- **PowerInfer** has moved to Tiiny-AI/PowerInfer. 9,845 stars, last push 2026-05-11 — [GitHub Tiiny-AI/PowerInfer](https://github.com/Tiiny-AI/PowerInfer). It only supports ReLU-sparse models and lists multi-GPU as a TODO (see `engines_other.md`). About 5 %.
- **Infernix** (Wallawalla47/Infernix): "C++/CUDA single-GPU inference engine for Qwen3.8 (incl. Qwen3.8-Flash-Next with offloaded experts) on the RTX 5090; grew from NInfer". 132 stars, created 2026-10-08, Windows releases dated 2026-10-09 to 2026-10-18 — [GitHub Infernix](https://github.com/Wallawalla47/Infernix). About 3 %.
- **sparkinfer** (gittensor-ai-lab/sparkinfer): "Fastest MoE/LLM inference runtime for consumer and edge Blackwell GPUs". 88 stars, created 2026-06-22 — [GitHub sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer). About 3 %.
- **mann-uofg/infer** is the only repo literally named "infer" in this space: "LLM inference engine in C++/CUDA for Llama-3.2-1B". 1 star, created 2026-10-12, no MoE — [GitHub mann-uofg/infer](https://github.com/mann-uofg/infer). About 2 %. Two Zig hobby repos, edwarble/infer and diogok/infer-zero, have 0 stars — [GitHub search](https://github.com/search?q=infer+in%3Aname&type=repositories&s=stars).
- A GitHub name search for "infer" (sorted by stars, created since 2025-06, plus C++/CUDA/Rust/C/Python/Go/Zig repos pushed in 2026) found no other local LLM engine called "infer". Most hits were unrelated: facebook/infer (static analyzer), dotnet/infer (Infer.NET), KuiperInfer (educational), OmniInfer (edge), dInfer (diffusion LMs), omni-infer (Ascend) — [GitHub search, 2026-10-22](https://github.com/search?q=infer+in%3Aname&type=repositories&s=stars). The engines note rules out InferLLM, Inferflow and llama-infer-cpp as stale or educational ([`engines_other.md`](engines_other.md)).
- NVIDIA/flare-local-inference (created 2026-10-20, 576 stars) is "a local inference server for NVIDIA SM120/SM121 GPUs" and is not called "infer" — [GitHub NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference).

### Inferences
- The name appeared as "llama. infer" and then as "llama.cpp AND infer". A remembered or dictated "ninfer" ("n-infer") could easily come out as "and infer". This is a guess, but it fits better than any other candidate. ninfer is also the only "-infer" engine that is discussed **with Strata** in the same threads and articles, as a list naming both implies.
- DeepInfer is the main alternative because it is this week's biggest inference news and overlaps with Maya's model. But it is a production cluster engine, not something people list among "run it at home" engines.
- Since every candidate is infeasible on sm_61 (Q3), the verdict can be stated as "almost certainly ninfer (or DeepInfer); infeasible either way" without settling the identification first.

### Gaps
- Reddit and X searches were not run directly (no API access). Only HN (Algolia API), GitHub and web roundups were checked. A Reddit thread spelling it "infer" could exist.
- No primary source spells any engine as bare "infer". The ~60/20 split is a judgement call, not something measured.

---

## Q2. What ninfer is: maker, dates, stack, models, quant formats, requirements, performance

### Takeaway
Upstream NInfer is a one-person (Neroued), from-scratch C++/CUDA engine for **one RTX 5090 only**. The
build accepts only `sm_120a`. It serves a closed list of Qwen checkpoints: dense 27B, plus the MoEs
Qwen3.6-35B-A3B and Qwen3.8-Flash-Next 125B. It uses its own `.ninfer` artifacts (groupwise-int Q4/Q5,
NVFP4, FP8) and has no multi-GPU. Community forks have grown it a lot. The most relevant is
**iamwavecut/ninfer-all**, which runs on Volta (sm_70) through Blackwell. It adds GGUF block formats,
Gemma 4 26B-A4B, Qwen3.8-Flash-Next with host or disk experts, and **layer-pipeline stages over up to
eight GPUs without P2P**. Even ninfer-all explicitly has **no Pascal build**.

### Cited Findings
**Upstream (Neroued/ninfer)**
- Self-description: "a from-scratch C++/CUDA inference engine for a single NVIDIA GeForce RTX 5090". It covers "Qwen3.5 Dense/MoE and Qwen4Exp (Qwen3.8-Flash-Next)", has OpenAI/Anthropic-compatible HTTP APIs, and uses "one GPU, one resident model, and one to eight execution lanes" — [ninfer README](https://github.com/Neroued/ninfer) (2026-10-22).
- Official artifacts: Qwen3.6-27B (groupwise-int, NVFP4), Qwen3.8-27B (groupwise-int, NVFP4) and **Qwen3.6-35B-A3B (groupwise-int)**. Users can convert their own weights. The engine "requires v3 artifacts" — [ninfer README](https://github.com/Neroued/ninfer). The Qwen3.6-35B-A3B artifact `qwen3_6_35b_a3b.ninfer` is 21.23 GiB — [HF neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer).
- MoE: "The 35B-A3B target additionally supports … CPU/GPU MoE expert offload with CUDA Graph decode under a GPU memory budget". "Qwen3.8-Flash-Next uses CPU/GPU Main experts, disk-backed NF4 PLE and segmented CUDA Graph" — [ninfer README](https://github.com/Neroued/ninfer). The budget is set with `NINFER_VRAM_BUDGET_MIB`. With CPU expert offload the default prefill chunk is 4096 — [ninfer docs/cli.md](https://github.com/Neroued/ninfer/blob/main/docs/cli.md).
- KV formats: "BF16, INT8, FP8, NVFP4, and K8V4". Speculative decoding: MTP (1–5 drafts), DFlash and DFlash2 — [ninfer README](https://github.com/Neroued/ninfer).
- Build requirements: "x86-64 Linux, an NVIDIA GeForce RTX 5090, a CUDA toolkit supporting `sm_120a`, CMake 3.28 or newer, a C++20 host compiler, Ninja, pkg-config, FFmpeg development libraries…, hwloc >= 2.5, liburing >= 2.5, and libcurl >= 7.85". "CUDA 13.4 is the validated development toolkit". "The build rejects CUDA architectures other than `sm_120a`" — [ninfer README](https://github.com/Neroued/ninfer). CMakeLists: `if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a") message(FATAL_ERROR "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a…")` — [ninfer CMakeLists.txt](https://github.com/Neroued/ninfer/blob/main/CMakeLists.txt).
- CPU expert kernels: the dispatch header defines `enum class Isa { Avx2, Avx512, Avx512Bf16 };` with no scalar or SSE variant. Kernel files are `src/ops/moe/kernels/cpu/{groupwise_int,nvfp4}/avx2.cpp` plus AVX-512 versions — [ninfer dispatch.h](https://github.com/Neroued/ninfer/blob/main/src/ops/moe/routed_swiglu/cpu/dispatch.h); [GitHub code search, repo:Neroued/ninfer avx2](https://github.com/Neroued/ninfer).
- Host RAM: the example server profile uses "the default shared pinned Host budget: 8 GiB plus eight model StateImages" for retained state and KV — [ninfer README](https://github.com/Neroued/ninfer).
- Multi-GPU: "no priority/QoS, multi-GPU, or distributed serving" — [ninfer README](https://github.com/Neroued/ninfer). Issue #313, "Two-GPU tensor parallelism for 16 GB boards — would you consider it", is still open (2026-09-24) — [ninfer #313](https://github.com/Neroued/ninfer/issues/313). Issue #391 (2026-10-10, open): "Ninfer process crashes on the second GPU in multi-ninfer, multi-GPU setup" — [ninfer #391](https://github.com/Neroued/ninfer/issues/391).
- Headline numbers (RTX 5090): Qwen3.6-35B-A3B groupwise-int decodes at 642.5 tok/s at C=1 with MTP3 (68.6 % acceptance) and 1,380.7 tok/s at C=8. A 7,680-token prefill runs at 17,705 tok/s — [ninfer README](https://github.com/Neroued/ninfer). *Claims to test; maintainer-measured on one card type.*
- Pascal/P40/1080/sm_61 issues: none. Searching the repo's issues for "pascal", "sm_61", "P40" and "1080" returned no Pascal-related results (2026-10-22) — [GitHub issue search](https://github.com/Neroued/ninfer/issues).

**Fork ecosystem (relevant subset)**
- **iamwavecut/ninfer-all**: 144 stars, created 2026-08-20, pushed 2026-10-22, VERSION `0.12.0-rtx3090`. It describes itself as "One line of NInfer for the Blackwell, Ada Lovelace, Ampere, Turing and Volta NVIDIA GPUs… GGUF block formats, ternary Bonsai 2, device route profiles, MTP and DFlash2" — [GitHub ninfer-all](https://github.com/iamwavecut/ninfer-all).
  - Models: the Qwen3.6/3.8 27B family, Qwen3.6-35B-A3B, **Qwen3.8-Flash-Next** and **Gemma 4 26B-A4B**. It serves the OpenAI, Anthropic and llama.cpp APIs "from one GPU, or from up to eight as pipeline stages on Linux" — [ninfer-all README](https://github.com/iamwavecut/ninfer-all).
  - Supported GPUs: "NVIDIA GPUs from Volta (compute capability 7.0) to consumer Blackwell (12.0), from 8 GB of memory". The architecture table says "Pascal and older | 6.x and older | none | … | no build". "Configure accepts no other target, so Pascal and older, Hopper and data-center Blackwell have no build" — [ninfer-all README](https://github.com/iamwavecut/ninfer-all); [ninfer-all docs/gpus.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/gpus.md). CMake error text: "NInfer supports CMAKE_CUDA_ARCHITECTURES of 70, 75, 80, 86, 89 and 120a" — [ninfer-all CMakeLists.txt](https://github.com/iamwavecut/ninfer-all/blob/master/CMakeLists.txt).
  - Toolkits: sm_70 needs CUDA 12.8 or 12.9 ("CUDA 13 no longer compiles `sm_70`"). sm_75/80/86/89 need 12.8 or newer. 120a needs 13.1 or newer — [ninfer-all docs/build-linux.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/build-linux.md).
  - Artifacts: Gemma 4 26B-A4B it QAT, "GGUF Q4_0 blocks", 14.75 GiB. Qwen3.6-35B-A3B NVFP4, 20.39 GiB, "`sm_120a` GPUs only". Qwen3.8-Flash-Next GSQ-RCO Q2_0 38.49 GiB, IQ3_S 54.50 GiB, Coder IQ1_M 31.02 GiB. "The Bonsai, GSQ-RCO, Flash-Next and Gemma 4 artifacts use formats that only this engine reads" — [ninfer-all README](https://github.com/iamwavecut/ninfer-all); [HF WaveCut Gemma 4 artifact](https://huggingface.co/WaveCut/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-NInfer-v3).
  - Flash-Next needs a separate n-gram table (26.92 GiB): "without a table the engine refuses to start" — [ninfer-all docs/qwen3-8-flash-next.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/qwen3-8-flash-next.md). It can also run "with its experts … on disk (`--expert-residency disk`, under 1 GB of RAM)" — [ninfer-all README](https://github.com/iamwavecut/ninfer-all). *Claim to test.*
  - Flash-Next Q2_0 decode: 136 tok/s on two RTX 5090s with all experts on GPU, 90 tok/s on two RTX 3090 Ti. 49–74 tok/s with experts in host memory, 17–35 tok/s with cold disk experts — [ninfer-all README](https://github.com/iamwavecut/ninfer-all). *Claims to test.*
  - CPU share of host and disk experts: "CPU threads compute a share of each decode call's cache misses" (`--expert-dma-share auto`) — [ninfer-all README](https://github.com/iamwavecut/ninfer-all). The CPU expert files are `gguf_expert_cpu_avx2.cpp` and `gguf_expert_cpu_avx512.cpp`, which use the F16C intrinsic `_cvtsh_ss` — [ninfer-all #44](https://github.com/iamwavecut/ninfer-all/issues/44) (2026-10-11).
  - Docker image `ghcr.io/iamwavecut/ninfer-all` is built on CUDA 13.4 / Ubuntu 26.04 and "carries the `sm_75`, `sm_86` and `sm_120a` builds". It needs driver 580+ and runs with `--ulimit memlock=-1` — [ninfer-all README](https://github.com/iamwavecut/ninfer-all).
- **geoffwatts/ninfer-v100** (246 stars, pushed 2026-09-14): a Tesla V100 port built for `sm_70` with CUDA 12.8. "On Volta, NVFP4 is decoded and executed in software using tuned FP16 tensor-core and SIMT kernels". Claims Qwen3.8-27B NVFP4 at 218.98 tok/s (synthetic near-100 % acceptance corpus) and Qwen3.6-35B-A3B DFlash at 139.58 tok/s on a V100-PCIe-32GB — [GitHub ninfer-v100](https://github.com/geoffwatts/ninfer-v100). The recipes list cites "Qwen3.8-27B · ninfer NVFP4 · MTP 4 — 2× Tesla V100 16GB → 180 tok/s" from Reddit — [awesome-local-inference-recipes](https://github.com/raytone-lab/awesome-local-inference-recipes). *Claims to test.*
- **Turing ports**: ttimasdf/ninfer-turing-sm75 ("ported to NVIDIA Turing (sm_75 / Tesla T4 / 2080Ti)", 2026-10-16) and mr-september/ninfer-2080ti-22g (81 stars, 2026-10-22) — [GitHub ninfer-turing-sm75](https://github.com/ttimasdf/ninfer-turing-sm75); [GitHub ninfer-2080ti-22g](https://github.com/mr-september/ninfer-2080ti-22g).
- **Don-Chad/ninfer-3090** (464 stars, 2026-10-16): Qwen3.8-27B on one RTX 3090 — [GitHub ninfer-3090](https://github.com/Don-Chad/ninfer-3090). ninfer-all continues "through the NInfer-3090 line" — [ninfer-all README](https://github.com/iamwavecut/ninfer-all).
- **ValerioDolci/capy-flash** (formerly ninfer-tp2; v0.7.0, 2026-10-22): Qwen3.8-Flash-Next NVFP4 on two 16 GB RTX 50 cards. It does "Tensor parallel over PCIe on two GPUs **without peer-to-peer** (pinned-host mailbox, FP32 partial sums)", and the CPU computes missing experts with "NVFP4 kernels in AVX-512 VNNI or AVX2". Requirements: "Linux, CUDA 13.1, two sm_120 GPUs … a CPU with AVX-512 (F/BW/VL, VNNI) or AVX2 + FMA + F16C", and "~75 GiB of free RAM, 135 GB of disk" (the 64 GB edition needs ~56 GiB free RAM) — [GitHub capy-flash](https://github.com/ValerioDolci/capy-flash).
- **star1872/ninfer-offload**: routed experts in CPU RAM. Requires "NVIDIA RTX 40-series (sm_89) or RTX 5090 (sm_120a). RTX 30-series and older are not supported", plus "x86-64 CPU with AVX2, FMA, F16C and BMI2". For Flash-Next, "about 80 GB of RAM … and 110 GB of disk" — [GitHub ninfer-offload](https://github.com/star1872/ninfer-offload).
- **k-i-k-s/ninfer-vulkan**: the description says "on broader GPUs thanks to Vulkan", but the README is unchanged from upstream (sm_120a), the latest commits (2026-09-16/18) are ordinary CUDA op tuning, and a code search for "vulkan" in the repo returned 0 hits — [GitHub ninfer-vulkan](https://github.com/k-i-k-s/ninfer-vulkan). *Unverified: forks may not be indexed by code search, but there is no evidence of a working Vulkan backend.*
- **JCraigWasTaken/ninfer-gfx906**: an AMD MI50/MI60 port, "closed (2026-09-04)". "On this hardware it is slower than llama.cpp", reaching "0.52-0.66x of llama.cpp's speed". "NInfer's big gains need matrix cores… gfx906 has no matrix cores, so a 4-token check here costs 2.3x a plain token. Deep drafts such as DFlash cannot pay off, and FP4/FP8 weights have no hardware path" — [GitHub ninfer-gfx906](https://github.com/JCraigWasTaken/ninfer-gfx906).

### Inferences
- In practice "ninfer" today means a family. Upstream is RTX 5090 only. ninfer-all is the broadest fork (Volta+, GGUF, Gemma 4 MoE, pipeline). capy-flash, ninfer-offload and Infernix are MoE/offload variants for RTX 40/50 with AVX2+ CPUs and 56–80 GB of RAM.
- For MoE on this rig, the only ninfer-family path that is not ruled out on RAM/CPU grounds is **ninfer-all with every expert resident on GPUs across pipeline stages**, i.e. Gemma 4 26B-A4B (14.75 GiB) or Qwen3.6-35B-A3B groupwise-int (21.23 GiB). That path is ruled out on GPU architecture instead (Q3).

### Gaps
- Upstream has no tagged releases, so "first release" can only be dated by repo creation (2026-06-26) and the first HN post (2026-07-23).
- No Pascal, P40 or GTX 10 report for ninfer or any fork was found (repo/issue searches for "ninfer pascal / p40 / 1080 / sm_61" returned nothing, 2026-10-22).
- Whether the upstream or ninfer-all binary checks CPUID at startup and refuses to run without AVX2, even when no experts are on the CPU, was not verified in code.

---

## Q3. Pascal sm_61 feasibility (and DeepInfer's)

### Takeaway
**Infeasible as shipped, and a port means a new kernel tier, not a build flag.** Upstream compiles only
for sm_120a. ninfer-all's oldest tier (sm_70) still runs its dense projections and all attention on
**FP16 tensor cores** (`mma.m8n8k4`), with cuBLAS FP16 GEMMs for wide calls and emulated
`ldmatrix`/`cp.async`. GP106 has no tensor cores, FP16 at 1/64 of FP32, and no bf16. A Pascal tier would
have to replace every tensor-core route with FP32-SIMT or dp4a code, fit kernels into Pascal's smaller
shared memory, and cope with 5-SM cards. Even then, the evidence from a no-matrix-core port (gfx906)
says most of ninfer's speed advantage, which comes from cheap speculative verification on tensor cores,
would disappear. **DeepInfer is infeasible by a much wider margin**: SM100 or Ascend 950 only, CUDA 13,
datacenter-scale EP.

### Cited Findings
- Upstream gate: "The build rejects CUDA architectures other than `sm_120a`" — [ninfer README](https://github.com/Neroued/ninfer); FATAL_ERROR in [CMakeLists.txt](https://github.com/Neroued/ninfer/blob/main/CMakeLists.txt).
- ninfer-all gate: only 70/75/80/86/89/120a. "Pascal and older … no build" — [ninfer-all README](https://github.com/iamwavecut/ninfer-all); [docs/gpus.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/gpus.md).
- What the oldest (Volta) tier still relies on: "Volta's tensor cores multiply FP16 only (`mma.m8n8k4`): there are no BF16 or integer tensor cores, no `cp.async` and no `ldmatrix`". A `70` build "runs its own routes where they decide the speed, and the kernels written for Ampere everywhere else, on emulated warp primitives (`src/ops/common/sm70_emulation.cuh`)". Specifics — [ninfer-all docs/build-linux.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/build-linux.md):
  - "Dense projections … up to 64 columns: a lane decodes the weights of its row straight into FP16 tensor-core fragments".
  - "From 65 columns … the weights are dequantized to FP16 … and multiplied by cuBLAS with FP32 accumulation".
  - "Attention of every KV format runs on Volta's tensor cores with FP32 accumulation".
  - "GGUF projections keep NInfer's DP4A vector kernel up to eight columns and ggml's DP4A matrix kernel above (as llama.cpp does on Volta)".
  - "T2 projections at one to eight columns … take an integer route… `dp4a`".
- The Turing tier needed kernels reshaped because "a block has 64 KiB of shared memory" — [ninfer-all docs/build-linux.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/build-linux.md).
- Low SM counts: "On a card with fewer SMs than the tuned route tables assume (the A4000 has 48), a GDN gating schedule whose cooperative grid would not be resident falls through to the first later route that fits the device; its speed there has not been tuned" — [ninfer-all pipeline-stages.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/maintainer/pipeline-stages.md). Unknown cards get "calibrated once at first start" — [ninfer-all README](https://github.com/iamwavecut/ninfer-all).
- No-matrix-core evidence: the gfx906 port ran at "0.52-0.66x of llama.cpp's speed while reading about half the bytes per token". "Most of NInfer's lead comes from speculative decoding: checking 4-8 drafted tokens costs about 1.5x one plain token, because the tensor cores absorb the extra math" on an RTX 5090, against 2.3x for a 4-token check on gfx906 — [ninfer-gfx906](https://github.com/JCraigWasTaken/ninfer-gfx906).
- **DeepInfer** hardware: "NVIDIA SM100 GPUs; Huawei Ascend 950 NPUs". Prerequisites: "GCC >= 13.0, Python 3.12, CUDA 13 (NVIDIA only)". Kernels: FlashMLA, DeepGEMM, DeepEP, TileKernels, TileLang. Parallelism: "Attention DP / balanced CP, MoE EP + EPLB". Benchmarks were "Measured on 4 SM100 GPUs for prefill and 64 for decode — each within one scale-up domain" — [DeepInfer README](https://github.com/deepseek-ai/DeepInfer). Examples use `--ep 8` and `--ep 2 --dp 8` — [DeepInfer examples/README.md](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/README.md). One HN commenter says "DS4.1F has 300GB of tensor weights" (2026-10-20) — [HN 50163172](https://news.ycombinator.com/item?id=50163172). The only older-GPU derivative found, HalfVulpes/Deepinfer-SM80 (73 stars, created 2026-10-20), targets "Ampere (`SM80` / `SM86`)… `mma.sync.aligned.m16n8k16` Tensor Cores" on vLLM V1, benchmarked on 6–8 × A100 — [GitHub Deepinfer-SM80](https://github.com/HalfVulpes/Deepinfer-SM80).

### Inferences
- What breaks first, in order, if someone forced `-DCMAKE_CUDA_ARCHITECTURES=61` on ninfer-all (CUDA 12.9 still supports sm_61):
  1. The CMake gate (trivial to remove).
  2. Device-code compile failures wherever `mma.sync` / `wmma` / `ldmatrix` / `cp.async` / bf16 intrinsics are used without a Volta-style fallback. The Volta emulation header covers `ldmatrix`/`cp.async`, but **no tier covers "no tensor cores at all"**.
  3. Kernels sized for 64 KiB+ shared memory per block. Pascal allows 48 KiB per block (general CUDA knowledge, not taken from ninfer docs).
  4. Cooperative-grid and route tables tuned for 48+ SMs on 5-SM cards.
  5. Warp-synchronous code written for Volta+ independent thread scheduling may misbehave on pre-Volta warps. This is general CUDA porting knowledge, not verified in ninfer code.
- The parts that could carry over are the GGUF dp4a paths (ggml's MMQ/vector kernels already run on sm_61 in llama.cpp), the host-side scheduler, the HTTP server and the pipeline-stage transport. That suggests a minimal Pascal tier: **GGUF Q4_0 Gemma 4 26B-A4B, dp4a projections, a new FP32-SIMT attention kernel, a BF16→FP32 or INT8 KV cache, no DFlash, cuBLAS SGEMM for prefill.** That is a multi-week kernel project. The likely end state is the one the gfx906 author reached: roughly llama.cpp speed at best, because decode is bandwidth-bound and the speculative-decoding advantage needs matrix cores.
- The rig's FP16 rate (1/64) rules out simply "running the Volta routes on SIMT FP16". Every FP16 tensor-core route would have to become FP32 math or dp4a.
- DeepInfer: no realistic path. The rig is three or more GPU generations short (SM100 vs sm_61), cannot run CUDA 13, and its 54 GB of VRAM is about a sixth of DeepSeek-V4.1-Flash's ~300 GB of weights (the HN figure above).

### Gaps
- The number of tensor-core call sites in ninfer-all (that is, the size of a Pascal port) was not counted. A compile attempt on a separate, more powerful PC on the same LAN (a "workstation"; experiment E2) would measure it.
- Whether CUDA 12.9's `cuda_bf16.h` software paths cover every bf16 operation the host/device code uses on sm_61 was not checked.

---

## Q4. Multi-GPU, host RAM, CPU and disk fit against this rig (if a Pascal tier existed)

### Takeaway
Apart from GPU architecture, ninfer-all's design fits the rig unusually well. Layer-pipeline stages
exchange only a small residual through **pinned host memory without P2P**, so PCIe 1.0 x1 is not the
bottleneck for single-stream decode. Every expert can stay GPU-resident, so the AVX2-only CPU expert
kernels need never run, and models are 15–21 GiB. The remaining rig-specific limits: **eight stages,
not nine**; the default pinned host KV and state budgets (around 8 GiB) must be turned off on a 4 GB
host; Flash-Next needs 65 GiB of model plus n-gram table, which does not fit the 26 GB of free disk; and
every upstream MoE offload mode assumes AVX2 and tens of GB of RAM.

### Cited Findings
- Pipeline design: "`--devices A,B,...` splits the model's layers into one pipeline stage per GPU". "It is a memory feature… a single stream is about as fast as one GPU, minus the boundary hops". "It is not tensor parallelism" — [ninfer-all pipeline-stages.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/maintainer/pipeline-stages.md).
- No-P2P transport: "the stages exchange their boundary rows through pinned host memory (`StageLink`), as Linux does between cards without peer access" — [pipeline-stages.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/maintainer/pipeline-stages.md). Measured "one 20 KB residual crossing per stage boundary per token is about 25 us against a ~21 ms step" on PCIe 3.0 x16 (27B dense, 3090 pair). Split decode was 48.5 vs 46.9 tok/s on one card — same source. All real-pair tests were "none with peer access" — same source.
- Stage limit: "up to eight GPUs on Linux". Cards may differ if the build runs on each — [ninfer-all README](https://github.com/iamwavecut/ninfer-all).
- MoE across stages: "Qwen3.8-Flash-Next runs its experts over two RTX 3090 Ti, two RTX 4090 and two RTX 5090" — [pipeline-stages.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/maintainer/pipeline-stages.md).
- DFlash taps cost "30 KiB per column, so 240 KiB for an eight-column DFlash2 verify and 30 MiB for the default 1,024-token prefill chunk" on the 27B 30/34 split — [pipeline-stages.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/maintainer/pipeline-stages.md).
- Host-memory defaults: upstream's example uses "8 GiB plus eight model StateImages" of pinned Host budget — [ninfer README](https://github.com/Neroued/ninfer). Flags `--host-kv-mib` and `--host-state-slots` appear in fork READMEs (e.g. `--host-state-slots 8 --host-kv-mib 8192`) — [ninfer-vulkan README](https://github.com/k-i-k-s/ninfer-vulkan); [capy-flash](https://github.com/ValerioDolci/capy-flash). The Docker image runs with `--ulimit memlock=-1` — [ninfer-all README](https://github.com/iamwavecut/ninfer-all).
- CPU expert kernels are AVX2 / AVX-512 only, and they use F16C (`_cvtsh_ss`) — [ninfer dispatch.h](https://github.com/Neroued/ninfer/blob/main/src/ops/moe/routed_swiglu/cpu/dispatch.h); [ninfer-all #44](https://github.com/iamwavecut/ninfer-all/issues/44). Offload forks state AVX2+FMA+F16C(+BMI2) requirements and 56–80 GB of RAM — [capy-flash](https://github.com/ValerioDolci/capy-flash); [ninfer-offload](https://github.com/star1872/ninfer-offload).
- Model sizes: Gemma 4 26B-A4B Q4_0 artifact 14.75 GiB; Qwen3.6-35B-A3B groupwise-int 21.23 GiB; Flash-Next Q2_0 38.49 GiB + n-gram table 26.92 GiB, and the engine "refuses to start" without the table — [ninfer-all README](https://github.com/iamwavecut/ninfer-all); [HF neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer); [docs/qwen3-8-flash-next.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/qwen3-8-flash-next.md).

### Inferences
- PCIe x1 cost (my arithmetic, using the rig's measured ~190 MB/s, [hardware spec](../../mining_rig.md)). A 20 KB residual hop takes about 0.1 ms plus latency, so 7 hops add roughly 1 ms per token. That is negligible against the tens of ms per token Pascal SIMT kernels would need. A 1,024-token prefill chunk moves about 20 MiB per hop, about 0.1 s each, which caps link-bound prefill near 1,000 tok/s. That is above anything the P106 cards would compute.
- VRAM fit (my arithmetic, before Xorg's per-GPU context and KV). Gemma 4 26B-A4B at 14.75 GiB needs about 3–4 cards. Qwen3.6-35B-A3B at 21.23 GiB needs about 4–5 cards. Both are within the 8-stage limit, leaving 4+ cards idle, or available for a second instance.
- Decode ceiling (my arithmetic). About 4B active parameters at Q4_0 (~4.5 bits) is about 2.25 GB read per token. At 173 GB/s per card, executed one stage at a time, the bound is about 75 tok/s. Realistic Pascal SIMT efficiency (50–70 %) and attention overhead suggest 25–45 tok/s as an upper estimate for a *hypothetical* port. llama.cpp on the same cards faces the same bound.
- Host RAM: with all experts on GPU and host KV/state tiers set to 0, peak host RAM would mostly be load-time buffering. Whether artifacts load by streaming (mmap) or need large pinned staging buffers is untested. That matters with 4 GB of RAM.
- A RAM upgrade (one free slot, up to 32 GB) would not change ninfer's verdict, because the blocker is the GPU architecture. It would only matter for the AVX2-dependent offload modes, and the CPU blocks those anyway.

### Gaps
- Whether ninfer-all allows more than 8 stages if a constant is changed, and whether a 9th card could hold only the embedding or head.
- Peak host RAM and pinned-memory use at load for a 15–21 GiB artifact were not documented.

---

## Q5. Could it be adapted? What would break first? (summary)

### Takeaway
Options, from cheapest to most expensive:
- **(a)** Use nothing from ninfer on the rig. Treat it as a reference design. Its pipeline-over-pinned-host and per-card route calibration resemble llama.cpp's layer split.
- **(b)** Run ninfer-all on the **workstation** (a modern tensor-core GPU such as sm_120, a CUDA 13 toolkit and tens of GB of RAM) to learn the engine and get comparison numbers.
- **(c)** Attempt a Pascal tier in a ninfer-all fork, limited to one GGUF MoE (Gemma 4 26B-A4B Q4_0) with dp4a projections and SIMT attention. Weeks of kernel work, and the gfx906 precedent says it would probably end up no faster than llama.cpp.

DeepInfer has no adaptation path.

### Cited Findings
- The gfx906 author's conclusion, from the only no-matrix-core port: "profiling shows no path to beating it [llama.cpp], so I have stopped work" — [ninfer-gfx906](https://github.com/JCraigWasTaken/ninfer-gfx906).
- ninfer-all already uses "ggml's DP4A matrix kernel" for GGUF on Volta, "as llama.cpp does" — [ninfer-all docs/build-linux.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/build-linux.md).
- Multi-architecture builds: "`CMAKE_CUDA_ARCHITECTURES` may list several architectures… each card runs the code of the newest listed architecture it can run" — [ninfer-all docs/build-linux.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/build-linux.md).

### Inferences
- Order of breakage for a forced sm_61 build: CMake gate → tensor-core device code (compile) → shared-memory sizing → 5-SM cooperative grids and route tables → runtime CPUID/AVX2 checks (if any) → pinned-host defaults on 4 GB RAM → the 8-stage cap (a 9th card goes unused).
- The CPU is not on the critical path *only if* every expert is GPU-resident. Every ninfer MoE offload mode (host or disk experts with CPU compute) is dead on this rig, with or without a Pascal port, because of AVX2/F16C and RAM.

### Gaps
- No one has published an attempt at sm_61 or sm_60 for ninfer. Effort estimates here are extrapolated from the Volta and gfx906 port write-ups.

---

## Q6. Experiment ideas (options, not decisions)

### Takeaway
None of these runs ninfer on the rig as-is. The cheap options settle the identification and measure how
far ninfer is from Pascal without touching the rig. The expensive option is a real port, with a known
risk of ending at "no faster than llama.cpp".

### Cited Findings
- ninfer-all builds sm_70 with CUDA 12.8/12.9, the same toolkit branch the rig uses, and 120a with CUDA ≥13.1, which suits a workstation with a current CUDA 13 toolkit — [ninfer-all docs/build-linux.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/build-linux.md).
- `--devices 0,0` "exercises the whole stage path on a single card" — [pipeline-stages.md](https://github.com/iamwavecut/ninfer-all/blob/master/docs/maintainer/pipeline-stages.md).
- Docker image `ghcr.io/iamwavecut/ninfer-all` (CUDA 13.4, sm_75/86/120a, 1.6 GB compressed) — [ninfer-all README](https://github.com/iamwavecut/ninfer-all).

### Inferences
**E1. Confirm the identification (zero effort, zero risk)**
- Settle which project the name refers to: ninfer (Neroued's single-GPU Qwen engine, many forks) or DeepInfer (DeepSeek's engine released 2026-10-20). The rig verdict does not depend on the answer.

**E2. "How far is Pascal?" compile probe on the workstation (low effort, no rig risk)**
- Setup: clone `iamwavecut/ninfer-all` in a `nvidia/cuda:12.9.2-devel-ubuntu24.04` container on the workstation (GCC 13, CMake ≥3.28, FFmpeg/libcurl dev). Remove the arch gate locally. Configure with `-DCMAKE_CUDA_ARCHITECTURES=61 -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF` and build with `-k 0`.
- Measure: the number of failing translation units, and the PTX instructions or intrinsics behind them (`mma`, `ldmatrix`, `cp.async`, bf16, shared-memory overflows). Grep for tensor-core call sites. Compare against the Volta tier's own routes, which are the template for a Pascal tier.
- Output: a concrete size estimate for option (c). No model download needed.

**E3. Learn the engine on the workstation GPU (low–moderate effort, no rig risk)**
- Setup: `docker run --gpus all --ulimit memlock=-1 ghcr.io/iamwavecut/ninfer-all serve …` on the workstation GPU (for example an 8 GB sm_120 card), or a native 120a build with a CUDA ≥13.1 toolkit.
- Model: Gemma 4 26B-A4B Q4_0 artifact (14.75 GiB; an 8 GB card means host experts, which a workstation with an AVX2 CPU and tens of GB of RAM allows). Or Qwen3.6-35B-A3B groupwise-int (21.23 GiB).
- Measure: decode and prefill tok/s, host RAM and pinned memory at load (`/proc/<pid>/status` VmLck/VmRSS), behaviour with `--host-kv-mib 0 --host-state-slots 0`, and `--devices 0,0` stage overhead. Compare with llama.cpp on the same GGUF (Gemma 4 QAT Q4_0 is a byte-for-byte import, per the README) on the same card.
- Purpose: shows whether ninfer's MoE path on GGUF is worth porting at all. If it does not clearly beat llama.cpp on an sm_120 card with tensor cores, it will not on Pascal.

**E4. Minimal Pascal tier prototype (high effort, weeks; high risk of "no gain")**
- Scope: a fork of ninfer-all with a `61` tier modelled on the `70` tier. One model (Gemma 4 26B-A4B Q4_0 GGUF, all experts GPU-resident). dp4a/MMQ projections (ggml's kernels already run on sm_61). A new FP32-SIMT attention kernel. INT8 or FP32 KV. cuBLAS SGEMM (FP32) for prefill. No DFlash; MTP optional. Kernels capped at 48 KiB shared memory. Route calibration for 5-SM cards.
- Build: on the workstation in a CUDA 12.9 container, with host code at generic x86-64 or `-march=x86-64-v2` (the Celeron has SSE4.2 only: no AVX, so no F16C). Check with `objdump -d | grep -c ymm` that no AVX instructions are in the host binary. Copy the binary, `libcudart`/`libcublas` 12.9 and libs to the rig. Serve the artifact from the workstation over NFS (the rig has ~26 GB of free disk).
- Run: `--devices 0,1,2,3` (3–4 stages for 14.75 GiB), `--host-kv-mib 0 --host-state-slots 0`, small `--max-context` (e.g. 8k).
- Measure: decode tok/s at 1 stream, prefill tok/s, StageLink hop time on x1, peak host RSS (must stay under ~3 GB), and stability. Compare with llama.cpp `-sm layer` on the same cards and GGUF. Success bar: beating llama.cpp at decode. The gfx906 precedent suggests it will not.

**E5. Mine ideas instead of porting (low effort)**
- Read ninfer-all's `pipeline-stages.md`, its device-route calibration (`docs/device-profiles.md`) and its n-gram copy drafting (`docs/ngram.md`) for techniques to try inside llama.cpp, which already runs on sm_61. Examples: exact memory-balanced stage solver, per-card route calibration, n-gram copy drafting. No build needed.

**E6. DeepInfer (only if the name refers to it): reading only**
- Nothing to run (SM100/Ascend 950, CUDA 13, EP across 4–64 GPUs). At most, read its DSpark/confidence-scheduled verification and EPLB code for ideas. No experiment on the rig is possible.

### Gaps
- E2 and E3 results would replace most of the extrapolation in Q3–Q5. Until they are run, every rig-specific speed figure above is arithmetic, not a measurement.
