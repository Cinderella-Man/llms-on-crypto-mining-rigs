# Experiment tracker

The [research survey](research/moe-inference-on-the-p106-rig.md) defines 81 experiments in five phases (E0–E4). This
page tracks each one: **done**, **partial** (some of it measured, the rest still open), **open**, or **n/a** (made
irrelevant by an earlier result). Results live in [benchmarks/](benchmarks/); experiments added after the survey are
listed at the end.

**Status on 2026-10-22: 14 done, 12 partial, 53 open, 2 n/a, plus 8 added experiments.**

## Phase 0: infrastructure and calibration

| ID | Experiment | Status | Result |
|---|---|---|---|
| E0.1 | Read-only rig inventory | done | [scripts/rig_inventory.sh](../scripts/rig_inventory.sh); facts in [mining_rig.md](mining_rig.md) |
| E0.2 | Network baseline | partial | Copies over SSH reach 35–90 MB/s; no iperf3 run yet |
| E0.3 | Build factory (no-AVX, sm_61) | done | [scripts/build_llamacpp_docker.sh](../scripts/build_llamacpp_docker.sh) |
| E0.4 | Binary parity, workstation vs rig build | open | |
| E0.5 | Model library and transfer methods | done | Library on a separate PC; USB 3 sticks moved by hand or filled over the network ([setup](setup.md#storing-models)) |
| E0.6 | Load path: storage source × load mode | partial | Load times from USB measured (~2.5–3.5 min for 21–26 GB); NFS and load modes untested |
| E0.7 | Results log and canary set | partial | Raw results kept per experiment; no common results database yet |
| E0.8 | Single-card calibration | done | [single-card calibration](benchmarks/2026-10-22-single-card-calibration.md) |
| E0.9 | Launch-overhead probes with tiny MoEs | done | [small MoE overhead](benchmarks/2026-10-22-small-moe-overhead.md) |
| E0.10 | Code-path audit | partial | CUDA graphs confirmed disabled on Pascal; KV-type code paths not audited |
| E0.11 | PCIe, host-staging and NCCL microbenchmarks | open | |
| E0.12 | Thermal steady state and power-log validity | partial | Power, clocks and temperatures logged in every run; no 30-minute soak |

## Phase 1: first MoE baselines

| ID | Experiment | Status | Result |
|---|---|---|---|
| E1.1 | First MoE control | done | Qwen3.6-35B-A3B in place of Qwen3-30B-A3B: [Qwen baselines](benchmarks/2026-10-22-qwen-baselines.md) |
| E1.2 | Build delta | done | Old vs new llama.cpp: +7 % prompt processing ([Qwen baselines](benchmarks/2026-10-22-qwen-baselines.md#newer-llamacpp-and-cuda-graphs)) |
| E1.3 | GPU-count and placement sweep | done | Qwen3.6 on 4/5/9 cards, Flint on 5/9, Qwen3.8-27B on 4/5/9 |
| E1.4 | Qwen3.6-35B-A3B in two quants | partial | Q4_0 only |
| E1.5 | KV type × depth sweep | partial | Depth up to 64k with f16; q8_0 KV only in the parallel-serving test |
| E1.6 | Prefill batch and pipeline tuning | partial | `-ub 128` needed for 262k context on a 4 GB host |
| E1.7 | Logits and sampling cost | done | Backend sampling: +18 % Qwen3.6, +8 % Flint, +3 % dense 27B ([speed-ups](benchmarks/2026-10-22-speedups.md)) |
| E1.8 | Nemotron 3.5 Lightning | open | |
| E1.9 | Gemma 4 26B-A4B | open | |
| E1.10 | GLM-4.7-Flash | open | |
| E1.11 | gpt-oss-20b MXFP4 | open | |
| E1.12 | LFM2.5-24B-A2B | open | |
| E1.13 | Second-tier model sweep | open | |
| E1.14 | Quality baseline per shortlisted model | partial | Flint quants scored against Q8_0 ([Flint](benchmarks/2026-10-22-flint.md#quantization-speed-against-quality)) |
| E1.15 | Wall-clock usefulness on real prompts | open | |

## Phase 2: tuning, patches and forks

| ID | Experiment | Status | Result |
|---|---|---|---|
| E2.1 | CUDA graphs on sm_61 | done | +7 % Qwen3.6, +14 % Flint, +16 % small MoE |
| E2.2 | Host-side profiling | open | |
| E2.3 | Pascal kernel patches | partial | CUDA-graphs patch adopted; MoE down-projection rescale switch built and tested (see added experiments); open Pascal PRs not yet built |
| E2.4 | Build-flag variants | open | |
| E2.5 | Quant ladder and format comparison | done | Llama-2-7B Q4_0/Q4_K_M/Q8_0; Flint Q4_1/Q5_0/UD-Q4_K_XL/Q4_0 |
| E2.6 | MTP speculative decoding | done | Qwen3.6 +44 %; dense 27B ~1.7× with 1–2 drafts |
| E2.7 | Draftless n-gram and draft-model speculation | open | |
| E2.8 | Expert-count override sweep | done | Flint 6→5/4 experts: +4/+9 % speed, large quality loss |
| E2.9 | `-sm tensor` versus layer split | open | |
| E2.10 | Throughput layouts | done | Two Qwen3.6 copies 31–35 tok/s; Flint two slots 35–42 tok/s |
| E2.11 | ik_llama.cpp feasibility gate | open | |
| E2.12 | ik_llama.cpp performance | open | |
| E2.13 | Vulkan backend A/B | open | |
| E2.14 | Official prebuilt nightly on sm_61 | open | |
| E2.15 | Headless mode and persistence mode | open | |
| E2.16 | Power-limit sweep | open | |
| E2.17 | Memory overclock (one card) | open | Needs owner approval (risk of hangs) |
| E2.18 | PCIe ASPM policy | n/a | ASPM is already disabled on every link |
| E2.19 | Second SO-DIMM before/after | open | Hardware change |

## Phase 3: stretch models, compression, the workstation as a partner

| ID | Experiment | Status | Result |
|---|---|---|---|
| E3.1 | Nemotron 3.5 Bolt 74B-A9B "24GB" | open | |
| E3.2 | Qwen3-Coder-Next tiers | open | |
| E3.3 | Qwen3.5-122B-A10B at ~2.5–3 bpw | open | |
| E3.4 | Prune versus quantise at equal size | open | |
| E3.5 | Mistral Small 4 with `-fa 0` | open | |
| E3.6 | Laguna-S-2.1 | open | |
| E3.7 | Further stretch sweep | open | |
| E3.8 | Qwen3.5-122B ik IQ2_KL | open | |
| E3.9 | Qwen3.5-122B with MTP | open | |
| E3.10 | Qwen3.8-Flash-Next GSQ-RCO-Coder | open | |
| E3.11 | Custom MoE-aware requant | partial | Flint imatrix Q5_0/Q4_1/Q4_0 made and scored; two rescale-safe mixes (Q8_0 or IQ4_NL down-projections) made, being scored |
| E3.12 | DIY REAP/REAM on the workstation | open | |
| E3.13 | Layer pruning data point | open | |
| E3.14 | RPC build, enumeration and hardening | open | |
| E3.15 | RPC load path and rig memory | open | |
| E3.16 | Standalone versus RPC on the same model | open | |
| E3.17 | One rpc-server versus nine | open | |
| E3.18 | Workstation GPU + rig whole-layer split | open | |
| E3.19 | Attention on the workstation GPU, experts on the rig | open | |
| E3.20 | Speculative decoding over RPC | open | |

## Phase 4: alternative engines and long shots

| ID | Experiment | Status | Result |
|---|---|---|---|
| E4.1 | Python platform probes | open | |
| E4.2 | HF transformers + accelerate correctness baseline | open | |
| E4.3 | fastllm | open | |
| E4.4 | vLLM-Pascal fork | open | |
| E4.5 | MLC-LLM | open | |
| E4.6 | mistral.rs + candle pre-Volta patch | open | |
| E4.7 | Strata "does it start" | open | |
| E4.8 | colibri probe | open | |
| E4.9 | distributed-llama Vulkan | open | |
| E4.10 | ninfer-all compile probe at sm_61 | open | |
| E4.11 | ninfer-all on the workstation GPU | open | |
| E4.12 | Minimal Pascal tier for ninfer-all | n/a | Not worth weeks of porting while llama.cpp keeps improving here |
| E4.13 | Idea-mining reads | partial | Research notes on dense-model and Flint speed-ups |
| E4.14 | prima.cpp two-node ring | open | |
| E4.15 | Watch list | open | |

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
