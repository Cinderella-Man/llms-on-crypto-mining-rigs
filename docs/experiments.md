# Experiment tracker

The [research survey](research/moe-inference-on-the-p106-rig.md) defines 81 experiments in five phases (E0–E4). This
page tracks each one: **done**, **partial** (some of it measured, the rest still open), **open** (still worth doing for
the two focus models), or **n/a** (made irrelevant by a result, or outside the
[focus](#focus-from-here-two-moe-tracks-tuned-deep) set on 2026-10-22: two MoE models, Flint and Qwen3.6-35B-A3B, tuned
deep). Results live in [benchmarks/](benchmarks/) and, for all work since 2026-10-22, in the
[quant registry](../quants/experiments.md#registry) (Q1–Q79); experiments added after the survey are listed at the end.

**Status on 2026-10-23: 20 done, 10 partial, 6 open, 45 n/a, plus 8 added experiments.**

## Phase 0: infrastructure and calibration

| ID | Experiment | Status | Result |
|---|---|---|---|
| E0.1 | Read-only rig inventory | done | [scripts/rig_inventory.sh](../scripts/rig_inventory.sh); facts in [mining_rig.md](mining_rig.md) |
| E0.2 | Network baseline | n/a | Copies over SSH reach 35–90 MB/s; models move on USB sticks (E0.5), and NFS and RPC are outside the focus |
| E0.3 | Build factory (no-AVX, sm_61) | done | [scripts/build_llamacpp_docker.sh](../scripts/build_llamacpp_docker.sh), also for the fork's branches; Vulkan: [scripts/build_llamacpp_vulkan_docker.sh](../scripts/build_llamacpp_vulkan_docker.sh) |
| E0.4 | Binary parity, workstation vs rig build | n/a | All comparisons after E1.2 run container builds against each other |
| E0.5 | Model library and transfer methods | done | Library on a separate PC; USB 3 sticks moved by hand or filled over the network ([setup](setup.md#storing-models)) |
| E0.6 | Load path: storage source × load mode | partial | Load times on 9 cards: 199 s for 25.1 GB, 438 s for 32.5 GB, 568 s for 42.3 GB ([Flint chat](benchmarks/2026-10-23-flint-chat.md)); NFS and load modes untested |
| E0.7 | Results log and canary set | done | Every experiment since 2026-10-22 has a row in the [quant registry](../quants/experiments.md#registry) with its reference, build and result; raw logs are kept per experiment outside the repository |
| E0.8 | Single-card calibration | done | [single-card calibration](benchmarks/2026-10-22-single-card-calibration.md) |
| E0.9 | Launch-overhead probes with tiny MoEs | done | [small MoE overhead](benchmarks/2026-10-22-small-moe-overhead.md) |
| E0.10 | Code-path audit | done | CUDA graphs disabled on Pascal upstream; f16 KV runs the TILE flash-attention kernel, q8_0 KV the VEC kernel plus Hadamard rotations ([attention note](research/notes/pascal_attention_long_context.md), [Q54](../quants/experiments.md)); kernel load paths read from the SASS ([Q48](../quants/experiments.md)) |
| E0.11 | PCIe, host-staging and NCCL microbenchmarks | open | |
| E0.12 | Thermal steady state and power-log validity | partial | Power, clocks and temperatures logged in every run; generation speed holds through 7–23-minute chats ([Flint chat](benchmarks/2026-10-23-flint-chat.md)); no 30-minute soak or tokens-per-joule figure |

## Phase 1: first MoE baselines

| ID | Experiment | Status | Result |
|---|---|---|---|
| E1.1 | First MoE control | done | Qwen3.6-35B-A3B in place of Qwen3-30B-A3B: [Qwen baselines](benchmarks/2026-10-22-qwen-baselines.md) |
| E1.2 | Build delta | done | Old vs new llama.cpp: +7 % prompt processing ([Qwen baselines](benchmarks/2026-10-22-qwen-baselines.md#newer-llamacpp-and-cuda-graphs)) |
| E1.3 | GPU-count and placement sweep | done | Qwen3.6 on 4/5/9 cards, Flint on 5/6/9, Qwen3.8-27B on 4/5/9; a memory-balanced `-ts` fits Flint Q8_0 at 262k on 9 cards ([Q34](../quants/experiments.md)) |
| E1.4 | Qwen3.6-35B-A3B in two quants | partial | Q4_0: 19.6 tok/s with CUDA graphs, 29.0 on the fork ([Q38](../quants/experiments.md)); Q4_0 and two mixes scored against Q8_0 ([Q29, Q31–Q33](../quants/experiments.md)); UD-Q4_K_XL and UD-IQ3_XXS not measured |
| E1.5 | KV type × depth sweep | partial | f16 KV to 64k: Qwen3.6 14.0 tok/s at 32k and 8.1 at 64k; Flint on the fork, 9 cards, 23.3 at 32k and 3.6 at 64k, a cliff under investigation ([Q54, Q79](../quants/experiments.md)); q8_0 KV ~30 % slower than f16 at 8k with no measurable quality cost ([Q54, Q62](../quants/experiments.md)); q4_0 KV not measured |
| E1.6 | Prefill batch and pipeline tuning | partial | `-ub 128` needed for 262k context on a 4 GB host; real-chat prompts read at 80–85 tok/s on 9 cards ([Flint chat](benchmarks/2026-10-23-flint-chat.md)); no `-ub`/`-b` sweep |
| E1.7 | Logits and sampling cost | done | Backend sampling: +18 % Qwen3.6, +8 % Flint, +3 % dense 27B ([speed-ups](benchmarks/2026-10-22-speedups.md)); tool requests switch it off and sample on the host ([Q55](../quants/experiments.md)) |
| E1.8 | Nemotron 3.5 Lightning | n/a | Outside the focus (two MoE models) |
| E1.9 | Gemma 4 26B-A4B | n/a | Outside the focus (two MoE models) |
| E1.10 | GLM-4.7-Flash | n/a | Outside the focus (two MoE models) |
| E1.11 | gpt-oss-20b MXFP4 | n/a | Outside the focus (two MoE models) |
| E1.12 | LFM2.5-24B-A2B | n/a | Outside the focus (two MoE models) |
| E1.13 | Second-tier model sweep | n/a | Outside the focus (two MoE models) |
| E1.14 | Quality baseline per shortlisted model | partial | Flint: every quant against BF16, the compute-path noise floor and 300 BFCL tool-calling tasks ([Q8, Q21, Q30](../quants/experiments.md)); Qwen3.6: KLD against Q8_0 ([Q29, Q31–Q33](../quants/experiments.md)); larger task checks planned ([Q41, Q70, Q71](../quants/experiments.md)) |
| E1.15 | Wall-clock usefulness on real prompts | partial | Flint, a five-message chat with reasoning: 600–7,900 tokens per reply, 7–10 minutes per chat on the fork ([Flint chat](benchmarks/2026-10-23-flint-chat.md)); two thirds of Flint's output is reasoning ([Q64](../quants/experiments.md)); answers not judged, Qwen3.6 not run |

## Phase 2: tuning, patches and forks

| ID | Experiment | Status | Result |
|---|---|---|---|
| E2.1 | CUDA graphs on sm_61 | done | +7 % Qwen3.6, +14 % Flint, +16 % small MoE |
| E2.2 | Host-side profiling | partial | Nsight Systems and nvprof refuse the P106-090; the fork's own profilers time kernels and host launches per token ([Q38, Q49](../quants/experiments.md)): the host launches every card's graph within ~1.9 ms of a token's start. No host-only timeline without CUDA events yet |
| E2.3 | Pascal kernel patches | partial | In the project's fork: 2-warp launch shape, Pascal matrix-vector kernels for Q4_0, Q4_1, Q5_0, Q8_0, Q5_K and IQ4_XS, shared-expert fusion, FP32 activation sums ([Q34, Q38, Q46, Q50, Q76](../quants/experiments.md)); 2× stock llama.cpp in real chat ([Flint chat](benchmarks/2026-10-23-flint-chat.md)); multi-token kernels in progress ([Q78](../quants/experiments.md)). Upstream prefill patches (#30235, 96-row MMQ tiles) not built |
| E2.4 | Build-flag variants | partial | `-Xptxas -dlcm=ca` has little to change: 93 % of the matrix-vector loads and all MMQ and flash-attention loads already use the read-only cache ([Q48](../quants/experiments.md)); `GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_FORCE_CUBLAS` and `GGML_CUDA_FA_QUANTS=all` not built |
| E2.5 | Quant ladder and format comparison | done | Llama-2-7B Q4_0/Q4_K_M/Q8_0; Flint Q4_1/Q5_0/UD-Q4_K_XL/Q8_0, fitted by a decode-cost model ([Q17](../quants/experiments.md)); matrix-vector speed per format on one card ([Q25, Q46](../quants/experiments.md)) |
| E2.6 | MTP speculative decoding | done | Qwen3.6 +44 % on the CUDA-graphs build, 1.04× on the fork, whose verify steps run on llama.cpp's stock multi-token kernels ([Q43](../quants/experiments.md)); dense 27B ~1.7× with 1–2 drafts |
| E2.7 | Draftless n-gram and draft-model speculation | done | Flint: n-gram drafters replayed on its own output give ~1.03× at the rig's verify cost ([Q64](../quants/experiments.md)); even perfect drafts give at most ~1.3×, so no drafter is trained ([Q19, Q20](../quants/experiments.md)); parked until multi-token Pascal kernels exist ([Q78](../quants/experiments.md)) |
| E2.8 | Expert-count override sweep | done | Flint 6→5/4 experts: +4/+9 % speed, large quality loss ([Q6](../quants/experiments.md)) |
| E2.9 | `-sm tensor` versus layer split | open | |
| E2.10 | Throughput layouts | done | Two Qwen3.6 copies 31–35 tok/s; Flint two slots 35–42 tok/s |
| E2.11 | ik_llama.cpp feasibility gate | n/a | ik_llama.cpp needs AVX2 on the host, and its types are no faster on sm_61 ([research](../quants/research.md#the-levers-ranked)) |
| E2.12 | ik_llama.cpp performance | n/a | See E2.11 |
| E2.13 | Vulkan backend A/B | open | Build script and test plan in the [Vulkan guide](vulkan_on_the_rig.md); not run on the rig (part of [Q34](../quants/experiments.md)) |
| E2.14 | Official prebuilt nightly on sm_61 | n/a | The speed-ups here need the project's fork: 2× stock llama.cpp in real chat ([Flint chat](benchmarks/2026-10-23-flint-chat.md)) |
| E2.15 | Headless mode and persistence mode | open | |
| E2.16 | Power-limit sweep | n/a | 75 W is the cards' maximum, so a sweep can only lower it; energy use is outside the speed focus |
| E2.17 | Memory overclock (one card) | open | Needs owner approval (risk of hangs) |
| E2.18 | PCIe ASPM policy | n/a | ASPM is already disabled on every link |
| E2.19 | Second SO-DIMM before/after | open | Hardware change |

## Phase 3: stretch models, compression, the workstation as a partner

| ID | Experiment | Status | Result |
|---|---|---|---|
| E3.1 | Nemotron 3.5 Bolt 74B-A9B "24GB" | n/a | Outside the focus (two MoE models) |
| E3.2 | Qwen3-Coder-Next tiers | n/a | Outside the focus (two MoE models) |
| E3.3 | Qwen3.5-122B-A10B at ~2.5–3 bpw | n/a | Outside the focus (two MoE models) |
| E3.4 | Prune versus quantise at equal size | n/a | Outside the focus; expert pruning reads the same bytes per token ([research](../quants/research.md#the-levers-ranked)) |
| E3.5 | Mistral Small 4 with `-fa 0` | n/a | Outside the focus (two MoE models) |
| E3.6 | Laguna-S-2.1 | n/a | Outside the focus (two MoE models) |
| E3.7 | Further stretch sweep | n/a | Outside the focus (two MoE models) |
| E3.8 | Qwen3.5-122B ik IQ2_KL | n/a | Outside the focus (two MoE models) |
| E3.9 | Qwen3.5-122B with MTP | n/a | Outside the focus (two MoE models) |
| E3.10 | Qwen3.8-Flash-Next GSQ-RCO-Coder | n/a | Outside the focus (two MoE models) |
| E3.11 | Custom MoE-aware requant | done | Flint: speed-aware mixes from BF16 at UD-Q4_K_XL quality ([Q27, Q36, Q45, Q73](../quants/experiments.md)); S4-L0 31.9 against UD-Q4_K_XL's 28.0 tok/s in real chat on the fork ([Flint chat](benchmarks/2026-10-23-flint-chat.md)); Qwen3.6 mixes A and B scored ([Q31, Q33](../quants/experiments.md)), not yet timed on the rig |
| E3.12 | DIY REAP/REAM on the workstation | n/a | Expert pruning saves memory, not bytes per token, and both focus models fit ([research](../quants/research.md#the-levers-ranked)) |
| E3.13 | Layer pruning data point | n/a | Below the quality bar without retraining, and `--prune-layers` files don't load ([file-only levers](research/notes/quant_speed_a_file_only.md)) |
| E3.14 | RPC build, enumeration and hardening | n/a | RPC adds capacity, not speed, and both focus models fit on the rig |
| E3.15 | RPC load path and rig memory | n/a | See E3.14 |
| E3.16 | Standalone versus RPC on the same model | n/a | See E3.14 |
| E3.17 | One rpc-server versus nine | n/a | See E3.14 |
| E3.18 | Workstation GPU + rig whole-layer split | n/a | See E3.14 |
| E3.19 | Attention on the workstation GPU, experts on the rig | n/a | See E3.14 |
| E3.20 | Speculative decoding over RPC | n/a | See E3.14 |

## Phase 4: alternative engines and long shots

| ID | Experiment | Status | Result |
|---|---|---|---|
| E4.1 | Python platform probes | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.2 | HF transformers + accelerate correctness baseline | n/a | Flint's BF16 weights are the quality reference ([Q8](../quants/experiments.md)) |
| E4.3 | fastllm | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.4 | vLLM-Pascal fork | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.5 | MLC-LLM | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.6 | mistral.rs + candle pre-Volta patch | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.7 | Strata "does it start" | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.8 | colibri probe | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.9 | distributed-llama Vulkan | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.10 | ninfer-all compile probe at sm_61 | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.11 | ninfer-all on the workstation GPU | n/a | Outside the focus: engine work goes into llama.cpp |
| E4.12 | Minimal Pascal tier for ninfer-all | n/a | Not worth weeks of porting while llama.cpp keeps improving here |
| E4.13 | Idea-mining reads | done | Research notes on dense-model and Flint speed-ups, HyperQwen, Project Maya ([Q57](../quants/experiments.md)), Recursive's rec engine and tqLLM ([notes](research/notes/README.md)) |
| E4.14 | prima.cpp two-node ring | n/a | See E3.14 |
| E4.15 | Watch list | done | Checked 2026-10-23: [upstream status of the Pascal changes](research/notes/upstream_pascal_patches.md) |

## Added experiments

| Experiment | Result |
|---|---|
| Qwen3.8-27B dense baseline and speed-ups (Q4_0, MTP draft count, draft threshold) | 3.9 → 7.5 tok/s ([speed-ups](benchmarks/2026-10-22-speedups.md)) |
| Maximum context at 262,144 tokens on a 4 GB host | Fits with `-ub 128`; default settings crash the host ([Qwen baselines](benchmarks/2026-10-22-qwen-baselines.md#maximum-context-and-how-to-avoid-crashing-a-4-gb-host)) |
| llama-server host-RAM defaults (`--cache-ram`, `--ctx-checkpoints`) | Can exhaust 4 GB; safe settings in [setup](setup.md#running-llama-server-on-a-4-gb-host) |
| Multi-token verify cost on 5-SM cards | 2/3/4/8 tokens per step cost 1.33/1.71/1.74/2.69× one token for the dense 27B |
| Flint baseline | 28 tok/s, full 262k context ([Flint](benchmarks/2026-10-22-flint.md)) |
| Flint quantization comparison on a separate PC | [scripts/quality_kld.sh](../scripts/quality_kld.sh), [scripts/quantize_docker.sh](../scripts/quantize_docker.sh) |
| MoE down-projection rescale removal (Flint) | +8–11 % speed, but NaN output with Q4_1: unsafe for Q4_x/Q5_x down-projections; Q8_0 +5 % ([Flint](benchmarks/2026-10-22-flint.md#the-moe-down-projection-rescale)) |
| Two copies vs one copy with two slots | See E2.10 |

## Focus from here: two MoE tracks, tuned deep

Rather than trying every model, the next work goes deep on two MoE families, adapting both the model files and
llama.cpp to these cards.

**Track 1: Flint** (28 tokens/s; [Flint](benchmarks/2026-10-22-flint.md),
[research](research/notes/flint_speedups.md))

1. Score the two rescale-safe quants already made (imatrix Q4_1 with IQ4_NL or Q8_0 expert down-projections) and
   measure their speed on the rig with the rescale switched off; IQ4_NL may decode slowly on 5-SM cards
   ([tqLLM note](research/notes/tqllm.md)).
2. Confirm on the GPUs that Q8_0 without the rescale gives the same perplexity as with it.
3. Build the makers' distillation-tuned Q4_1 with only the expert down-projections converted to Q8_0, so their tuning
   survives and the rescale can be skipped safely.
4. Test per-tensor format choices that matter for speed: attention is 35 % of the bytes read per token, the output head
   12 % (a Q4_0 or Q8_0 head instead of Q6_K).
5. A cheaper rescale in llama.cpp itself (one fused kernel instead of ~11 operations, or 32-bit block sums), which
   would help every Flint user.
6. N-gram drafting for edit-heavy agent output, and 3–4 parallel slots.

**Track 2: Qwen3.6-35B-A3B and its derivatives** (27.7 tokens/s on real requests with multi-token prediction;
Ornith-1.5-35B-A3B and the Qwen3.8-35B-A3B distillation share the architecture)

1. Quality check of the Q4_0 file against Q8_0, and imatrix quants in fast formats that keep the MTP layer.
2. A rotated q4_0 KV cache to give a four-card copy the full 262,144-token context, and its interaction with MTP.
3. Baselines of Ornith-1.5 and the Qwen3.8 distillation with the same speed-ups.

**Software for both tracks**

1. Profile one generated token per model with Nsight Systems to find where the time goes.
2. Build and measure open llama.cpp Pascal patches: #30334 (MMVQ fusion check for compute capability 6.1), #30235
   (Pascal MMQ tile configuration), #20831 (dynamic MMVQ warp count for narrow matrices).
3. Fuse operations where the profile shows per-operation overhead dominating.
