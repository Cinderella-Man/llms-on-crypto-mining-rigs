# Quant-speed research: starting facts (2026-10-22)

The facts and quality bar that the four quant-speed notes ([file-only levers](quant_speed_a_file_only.md), [formats and kernels](quant_speed_b_formats_kernels.md), [retraining](quant_speed_c_retraining.md), [community reports](quant_speed_d_community.md)) started from; they call it "the brief". The synthesis is [quants/research.md](../../../quants/research.md).

## The question

What can be done **to a model** (its file, its weight format, or by retraining parts of it), possibly together with
matching llama.cpp kernel changes, to make **single-stream decode** faster on the rig described below, preferably with
quality **within noise** of our best current quant (see "Quality bar")? Collect everything: papers, llama.cpp / ik_llama.cpp
PRs, issues and discussions, community quants and model cards, forks, blog posts, Reddit and forum threads (including
Chinese and Russian mining-card communities: P106, P104, P102, P40 owners). A one-line Reddit claim is worth recording
if it is marked as such.

## The rig (speed is only ever measured here)

- 9 × NVIDIA P106-090 6 GB (cut-down GP106, Pascal, compute capability 6.1): **5 SMs**, 640 CUDA cores, 1.5 MB L2,
  GDDR5 192-bit, ~173 GB/s measured (192 rated). INT8 `dp4a` at 4× the FP32 rate; FP16 at 1/64 of FP32 (unusable);
  no tensor cores. Power limit 75 W pulls the core clock to ~1700–1760 MHz under decode.
- Each card on a PCIe 1.0 x1 link (~190 MB/s), no peer-to-peer; host is a 2-core Celeron 3865U **without AVX**, 4 GB RAM.
- Engine: llama.cpp CUDA backend (CUDA 12.9, driver 580, the last for Pascal), commit `d77256999` (2026-10-22) plus the
  CUDA-graphs-on-Pascal PR #27721. Models are split by layer across 5–9 cards; card count doesn't change speed.
- Measured single-card decode efficiency (effective bandwidth ÷ 173 GB/s, Llama-2-7B): Q4_0 42 %, Q4_K_M 35 %, Q8_0 54 %.
  With only 5 SMs the card cannot unpack weights (or keep enough loads in flight) to saturate memory. The same model runs
  ~1.5× faster from Q4_0/Q4_1/Q5_0/Q8_0 ("fast formats") than from K-quant / I-quant files of the same size.

## The model: Flint (RecursiveSI/flint, released 2026-10-20)

- MoE, 39.8 B total, ~2.5 B active per token (2.2 B body + 0.3 B head). Trained from scratch, own 150,144-token vocab,
  untied embedding/head. 49 layers, hidden 2048; 32 query heads / 4 KV heads × 128; layers 0–1 dense FFN (6144);
  47 MoE layers with 256 experts (width 512), top-6 sigmoid routing with normalization, plus a shared expert (width 1024).
  14 full-attention layers (no RoPE) + 35 sliding-window (513) layers, attention sinks, per-head q/k RMSNorm. No MTP head,
  no drafter, no same-tokenizer small model exists.
- Makers' files: Q4_1 GGUF of their **quantization-recovered INT4** (MLX export: group-64 affine INT4, distillation-tuned;
  the GGUF re-expresses it as group-32 Q4_1) and an NVFP4 GGUF. Their Q4_1 is "deliberately different" from BF16 yet
  matched BF16 on 300 BFCL tool-calling tasks (community test).
- Per token, the makers' Q4_1 GGUF reads 1.66 GB of weights: attention 35 % (q/o projections dominate), routed experts 33 %,
  output head 12 %, shared expert 11 %, router 6 % (router is F32 in that file), dense 3 %.
- Token time 35.8 ms (5 cards, CUDA graphs): ~23–26 ms reading weights, ~10–13 ms fixed per-kernel cost from ~1,900 kernel
  launches per token (~5–7 µs each).
- llama.cpp's Flint support **rescales the MoE down-projection input** (~11 extra graph ops per MoE layer, ~470 launches per
  token; removing it gives +5–11 % speed) because CUDA's Q8_1 activation blocks store their sums in FP16, which overflow
  for Q4_0/Q4_1/Q5_0/Q5_1 down-projections with Flint's large activations (without it Q4_1 gives NaN). Q8_0, IQ4_NL,
  IQ4_XS, Q6_K, MXFP4, NVFP4 don't use those sums.

Secondary model, for "does this lever carry over?": **Qwen3.6-35B-A3B** (40 layers: 30 Gated DeltaNet + 10 full attention,
256 experts top-8 + shared expert, hidden 2048, has an MTP layer that llama.cpp already uses for drafting).

## What was already known

- [flint_speedups.md](flint_speedups.md) (esp. sections 0, 2, 4, 5:
  byte tables, the rescale, kernel levers incl. fused QKV and shared-expert merging, candidate mixes A–E)
- [quants/README.md](../../../quants/README.md) and [quants/guide.md](../../../quants/guide.md) (results, method)
- [Flint benchmark](../../benchmarks/2026-10-22-flint.md),
  [single-card calibration](../../benchmarks/2026-10-22-single-card-calibration.md)
- Other notes in `docs/research/notes/` (tqllm.md covers TurboQuant; dense_27b_speedups.md section 3 covers formats on
  5-SM cards; models_large_moe_and_compression.md covers expert pruning/merging).

Results so far (same top token vs Q8_0 / mean KLD / real-chat tok/s): Unsloth UD-Q4_K_XL 87.6 % / 0.124 / 15.8;
our **UD-mirror v1** (UD-Q4_K_XL's per-tensor bit budget re-expressed in Q4_1/Q5_0/Q8_0) 87.4 % / 0.129 / speed pending;
imatrix Q5_0 85.0 % / 0.177 / 18.9; makers' Q4_1 78.6 % / 0.316 / 24.3; imatrix Q4_0 71.4 % (broken, PPL +45 %).

## Resources for experiments (not for speed measurements)

- A quant machine: 3 × RTX 3090 24 GB, 2 × V100 32 GB (no BF16), 2 × RTX 3080 10/12 GB — 158 GB VRAM, PCIe x1–x8,
  no P2P; 94 GB RAM; Ryzen 9 5900X (24 threads). Flint BF16 (80 GB) is available there.
- A workstation: 8 GB laptop GPU (RTX 5070), 60 GB RAM, 32 threads.
- No paid cloud GPUs.

## Quality bar

Preferred: within noise of UD-mirror v1 (≈ ≥ 87 % same top token, mean KLD ≤ 0.14 on 80 × 512 tokens of wikitext-2).
A lossy lever counts only if it is ≥ 15 % faster and stays ≥ the makers' Q4_1 (78.6 %, plus a task check).
