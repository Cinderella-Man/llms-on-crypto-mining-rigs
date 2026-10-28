# Sub-1-bit quantization, and how small Qwen3.8-Flash-Next can get for the P106 rig (2026-10-28)

Qwen3.8-Flash-Next's best-rated quant, ISTA-DASLab's GSQ-RCO IQ3_S, is 54.8 GB on the cards, and nine 6 GB cards
hold about 45–47 GB of weights ([which version to run](flash_next_variants.md)). This note asks:
- how ISTA-DASLab made its GSQ-RCO files, and whether the method can be reproduced for smaller or Pascal-friendly files;
- what sub-1-bit methods reach, Samsung's NanoQuant first, and on which models;
- whether any of them, or any other method, gets Flash-Next onto the rig at quality close to ISTA's IQ3_S;
- how each kind of format decodes on a P106 (compute 6.1: dp4a, FP16 at 1/64 rate, no tensor cores).

Sources were read on 2026-10-28: papers, code repositories through the GitHub API, model cards, Hugging Face file
trees and discussions. Nothing was run. The sizes and formats of the local Flash-Next files come from the header
census in [docs/data/2026-10-27_flash_next_variants/](../../data/2026-10-27_flash_next_variants/README.md).

Conventions, as in the other notes:
- **(my calc)**: arithmetic in this note; **(est.)**: an estimate, not yet measured.
- "Bits" per weight count every stored byte of a tensor, scales included. "Effective" bits divide by the parameters
  of the original model, so pruned weights count as zero bits.
- Decode costs are in ps per weight and slot-equivalents, as in the [low-bit note](pascal_lowbit_format.md): the
  fork's kernels fit `time per weight = max(bytes per weight / 150 GB/s, slot-equivalents per weight / 0.56 T per s)`.
- "P" and "S" in the source list mean primary (paper, code, model card) and secondary (forum, blog).

## Summary

1. **The rig leaves 2.7–2.85 bits per routed-expert weight (my calc).** Flash-Next has 120.8 B routed-expert weights
   (15.1 GB per bit) and about 4.1 GB of other tensors on the cards. With ~47 GB of weights and no MTP draft, the experts
   get 42.9 GB (2.84 bits); with the draft, ~40.8 GB (2.70 bits). ISTA's IQ3_S spends 3.33 bits, so it needs 15–19 %
   fewer expert bytes at equal quality to fit.
2. **GSQ-RCO is two ISTA methods put together, and the pipeline that made these GGUFs is not public.**
   - **GSQ (Gumbel-Softmax Quantization)** learns which value of a fixed grid each weight takes, plus the scales. It
     starts from GPTQ and fits layer outputs and then block outputs.
   - **RCO (Riemannian Constrained Optimization)** picks one format per tensor under an exact byte budget by
     minimizing the end-to-end KL divergence to the full model.
   - For Flash-Next, every tensor was quantized by GSQ in every candidate GGUF format, RCO chose among them (352
     tensors, one format per fused expert tensor), and the results were stitched into standard GGUF files.
   - The public GSQ code writes integer grids for vLLM, and issues asking for the GGUF/IQ path have no answer; the
     calibration text for the non-Coder builds is not disclosed.
3. **The closest file to ISTA's IQ3_S that fits is ISTA's own IQ3_XXS (47.0 GB).** It scores 92.57 against BF16's
   93.12 on ISTA's three-task mean (99.4 %), and IQ3_S scores 93.26. Swift-1.5's builds with the same allocations
   measure KLD 0.240 (IQ3_XXS) and 0.145 (IQ3_S) against their own BF16. IQ3_XXS's agentic behaviour is untested. It
   fits at about 128k of q8_0 cache with no MTP draft.
4. **Sub-1-bit methods do not reach this quality on any model.**
   - The best 1-bit post-training results raise perplexity 1.6–4× on dense 7–70B models (NanoQuant: Llama-2-7B
     5.47 → 8.99–10.34, Llama-2-70B 3.32 → 6.52), and the zero-shot mean falls from ~71 to ~46–49 on 8B models.
   - No sub-1-bit factorization (NanoQuant, LittleBit, DBF) has been run on an MoE model. The MoE methods closest to it
     lose 8–30 % of task accuracy at 1.3–1.6 bits (BitsMoE, ProQuant, MoBiE).
   - NanoQuant's released code caps the rank at min(m, n), so Flash-Next's 640 × 2560 experts top out at 1.28 bits.
5. **The smallest good Flash-Next files are 1.7–1.9 bits, and none runs on the rig.**
   - ISTA's Coder keeps half of the experts (1.89 effective bits) and halves factual recall.
   - SyzygyResearch's Mach-2 (1.70 bits) is a fine-tune in trellis codes that needs its own fork and ~105 GB of host
     RAM.
   - EXL3 at 1.66–2.18 bits is trellis-coded for tensor cores.
6. **On Pascal the binary factorization is the cheapest low-bit decode, but that is a small prize here (est.).** Sign
   bits decode with masks and dp4a at ~0.5 slot-equivalents per stored bit: ~1 ps per weight at 1 bit and ~2.9 ps at
   2.8 bits. That compares with 3–6 for Q2_0, 6.4–9.2 for the v3 IQ kernels and 20.5 for llama.cpp's IQ3_XXS kernel.
   A token reads only 2.36 G expert weights, though, so with today's best kernels the experts take 7–22 ms of a
   ~70–100 ms token. Quality decides, and at ≤ 1 bit it fails.
7. **The largest waste in ISTA's files is the 640-wide down projection, not the bit width.**
   - `ffn_down_exps` rows are 640 long, so no 256-weight-block format fits them. The only choices are Q2_0
     (2.25 bits) and IQ4_NL (4.5).
   - RCO uses IQ4_NL in 39 of 48 layers in IQ3_S, and only 18 in IQ3_XXS: the cut from IQ3_S to IQ3_XXS takes 5.0 GB
     off the down projections and only 2.4 GB off gate/up.
   - A ~3.4-bit format with 128-weight blocks (an N3P variant from [route 2](pascal_lowbit_format.md)) would let the
     down projections sit at about 3.4 bits. That gives est. 0.56× of IQ3_XXS's down-projection error for +0.36 GB,
     much of the way to IQ3_S's 0.35× (my calc, section 6), and it decodes faster on Pascal than IQ4_NL.
8. **Recommendation:** no sub-1-bit work for Flash-Next. In order:
   1. Score ISTA's IQ3_XXS as the quality anchor (fit probe, KLD, agent check).
   2. Compose a 44–47 GB file from the published GSQ builds, allocated per 8-layer group on agent text as was done for
      Ornith (Q167). The four Swift-1.5 builds are already on the quality machine.
   3. Build N3P-128 for the down projections and requantize IQ3_S's IQ4_NL down tensors into it.
   4. Test a binary factorization at 2–3 bits on real expert matrices with the CPU proxy. It is the one idea from the
      sub-1-bit papers with a Pascal upside.

   Running GSQ ourselves into Pascal formats has the highest ceiling, and it is a project of days on the quality
   machine plus code ISTA has not released.

## 1. What the rig can hold, in bits per expert weight

Flash-Next's routed experts: 48 layers × 512 experts × (gate 640 × 2560 + up 640 × 2560 + down 2560 × 640) =
120.8 B weights; gate and up are 80.5 B (10.07 GB per bit), down 40.3 B (5.03 GB per bit) (my calc). A token uses 10
experts per layer: 2.36 G expert weights. The n-gram table (28.8 GB in IQ4_NL) stays on disk in every case
([flash_next_variants.md](flash_next_variants.md), section 5).

| Case | Weights on the cards (GB) | Other tensors | Routed experts | Bits per expert weight |
|---|---|---|---|---|
| ISTA IQ3_S (published) | 54.8 | 4.5 | 50.3 | 3.33 |
| ISTA IQ3_XXS (published) | 47.0 | ~4.1 | ~42.9 | 2.84 |
| **Rig, no MTP draft, ~128k of q8_0 cache** | ≤ ~47 | 4.1 | ≤ ~42.9 | **≤ 2.84** |
| **Rig, with the MTP draft (~2.7 GB), hyper-connections in Q8_0** | ≤ ~44.3 | 3.5 | ≤ ~40.8 | **≤ 2.70** |
| ISTA IQ2_XS (published) | 39.2 | ~3.8 | ~35.4 | 2.35 |
| ISTA Q2_0 (published) | 37.6 | ~3.6 | 34.0 | 2.25 |
| All experts at 1 bit (hypothetical) | ~19 | 4.1 | 15.1 | 1.0 |

The ceilings follow the Ornith-1.5-122B precedent (a 47.6 GB file with 122,880 tokens of F16 cache,
[benchmark](../../benchmarks/2026-10-27-ornith-122b.md#fitting-it-on-the-cards)) and need the
[fit probe](../../../scripts/fit_probe.sh) on Flash-Next's own files. Hyper-connections in Q8_0 save 0.59 GB of
the 1.26 GB in BF16 (my calc). Fitting ISTA's IQ3_S means removing 7.4–9.5 GB from its experts.

## 2. How ISTA-DASLab made the GSQ-RCO files

### 2.1 GSQ: Gumbel-Softmax Quantization

From the paper ([arXiv 2604.18556](https://arxiv.org/abs/2604.18556), v2 2026-05-15):
- **The grid is fixed:** symmetric, uniform, one scale per 128 weights; at 2 bits {−2, −1, 0, 1}. GSQ learns, per
  weight, logits over grid values (all four at 2 bits; a shift of −2…+2 around the GPTQ index at 3 bits and more), and
  the per-group scales. Training samples the choice with Gumbel-softmax at a temperature annealed from 2 to 0.05
  (Lion optimizer, 20 epochs).
- **The objective is reconstruction:** layer outputs first, then block outputs (Q/K, then V/O, then the MLP), each
  block frozen when done; for 2-bit Llama a final scale-only end-to-end pass. A Hessian enters only through the GPTQ
  start.
- **Calibration:** 4,096 sequences × 4,096 tokens (16.8M tokens). The text is FineWeb-Edu for Llama, OpenThoughts for
  Kimi, and OpenThoughts, OpenPlatypus and FineWeb-Edu for Qwen.
- **Cost on 8 × H200:** an 8B Llama 10 h, a 70B Llama 68 h, Kimi-K2.5's experts 24 h. The logits take 2–5× the memory of
  the weights being learned, so MoE layers are trained expert by expert.
- **GGUF:** GSQ can start from a published GGUF (Unsloth's), relearn the codes and write them back into the same
  K-quant format. Qwen3-8B Q2_K: task mean 50.03 → 56.28; Qwen3-4B Q2_K: 35.01 → 48.34. IQ formats and importance
  matrices do not appear in the paper.
- **Results against other methods** (Llama-3.1-8B, zero-shot mean, BF16 73.71):

  | Method | Bits | Mean |
  |---|---|---|
  | GSQ | 2.13 | 68.55 |
  | QTIP | 2.00 | 69.88 |
  | EfficientQAT | 2.25 | 63.79 |
  | GPTQ | 2.25 | 37.53 |
  | GSQ, ternary | 1.71 | 63.44 |
  | GSQ | 3.13 | 72.32 |

  On Llama-70B (BF16 78.99), GSQ with RCO's per-layer widths reaches 77.40 at 2.37 bits.

### 2.2 RCO: Riemannian Constrained Optimization

From [arXiv 2605.00649](https://arxiv.org/abs/2605.00649) (v3 2026-09-27):
- **The problem:** one of K options for each of N groups (formats per tensor, or experts kept or dropped) under an
  exact cost budget. Under a softmax relaxation the budget is a smooth manifold in logit space, and each step projects
  the gradient onto it. Sampled choices are made exactly feasible by a multiple-choice knapsack.
- **The loss is the end-to-end KL to the full model,** with candidate tensors swapped in.
- **Results:** Qwen3-8B at 2.25 bits, FineWeb-Edu perplexity 20.6 (FP16 10.96) in 62 minutes, against 21.4 for
  EvoPress in 11–14 hours.
- **Expert pruning:** at 50 % of experts removed, a non-uniform choice keeps 97 % of HumanEval against 55 % for a
  uniform one. That is the Coder release.

### 2.3 The Flash-Next pipeline, from the model card

- **Steps:** "Each weight tensor is quantized at every candidate GGUF quantization type with GSQ". The search assigns
  one type to each of 352 tensors: 304 dense tensors and 48 fused routed-expert tensors, "per layer (not per expert,
  which GGUF cannot express)". The results are stitched into one GGUF.
- **Builds:** whole-file targets of 2.40 (Q2_0), 2.50 (IQ2_XS), 3.00 (IQ3_XXS) and 3.50 bits (IQ3_S), counted over
  the ~125 B parameters outside the n-gram table.
- **The n-gram table:** IQ4_NL at a fixed 4.5 bits, outside the search. It is the second shard of each build, and
  the IQ3_S and IQ3_XXS builds list the same SHA-256 for it, so a second build costs only its first shard.
- **Kept unquantized:** the router (BF16), every hyper-connection projection, the DeltaNet α/β projections, the
  indexer projections and the n-gram key/value projections (BF16), and norms and small parameters (F32).
- **Not stated:** calibration text, token counts, compute hours and KLD. The importance matrix is not published
  ([discussion #39](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/39) is
  unanswered), though the sibling Qwen3.8-27B repository publishes one (1,000 chunks of 4,096 tokens).
- **Coder calibration** (maintainer, [Coder #22](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/discussions/22)):
  sci-code, GAIA, the_cauldron and OpenThoughts3, with no counts.
- **The allocations are published:** `tensor-allocation/*.rco-allocation.txt` lists the format of every tensor in
  each build, and the Coder's file also lists the kept experts by their original index.

What RCO chose for the routed experts (census and allocation files; my calc for the bits):

| Build | Gate/up formats (layers) | Gate/up bits | Down formats (layers) | Down bits | Experts (GB) |
|---|---|---|---|---|---|
| IQ3_S | IQ2_S 20, IQ3_XXS 17, IQ3_S 10, IQ4_XS 1 | 2.96 | IQ4_NL 39, Q2_0 9 | 4.08 | 50.3 |
| IQ3_XXS | IQ3_S 13, IQ2_S 10, IQ2_XS 10, IQ2_XXS 9, IQ3_XXS 6 | 2.72 | IQ4_NL 18, Q2_0 30 | 3.09 | 42.9 |
| IQ2_XS | IQ2_S 34, IQ2_XXS 11, IQ1_M 3 | 2.40 | Q2_0 48 | 2.25 | 35.5 |
| Q2_0 | Q2_0 48 | 2.25 | Q2_0 48 | 2.25 | 34.0 |

The down projections carry more bits than gate/up in every build above Q2_0, and between IQ3_S and IQ3_XXS RCO took
twice as many bytes from them (5.0 GB) as from gate/up (2.4 GB). With only 2.25 and 4.5 bits to choose from, each
step is a whole layer moving 2.25 bits.

### 2.4 What can be reproduced

- **[IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ)** (Apache-2.0, last commit 2026-07-05) supports Llama,
  Qwen3/3.5/3.6 dense and MoE, Gemma-4 and Kimi K2/K2.5. Its quantizers are integer, ternary, 1-bit and 2:4. It writes
  compressed-tensors checkpoints for vLLM, and its examples need 4 × H200 for Qwen3-30B-A3B. It has no Flash-Next
  adapter and no GGUF or IQ output.
- **Issues [#4](https://github.com/IST-DASLab/GSQ/issues/4) and [#9](https://github.com/IST-DASLab/GSQ/issues/9)**
  ask for the GGUF path. In June the reply was that a new codebase would come "very soon"; there has been none
  since.
- **[IST-DASLab/RCO](https://github.com/IST-DASLab/RCO)** (Apache-2.0) does GPTQ mixed precision and expert pruning,
  with no GGUF support.
- **Third-party builds:**
  - **ukisai's Swift-1.5 GSQ-RCO** reuses ISTA's allocations with its own GSQ passes and importance matrix; its code
    is not named.
  - **[pfeifferj's reproduction](https://huggingface.co/pfeifferj/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)** uses ISTA's public
    repositories with other candidates:
    - Formats: gate/up in Q2_0, Q2_K, Q3_K or Q4_K; down in Q2_0 or Q4_0; every other matrix Q8_0.
    - Size: 47.94 GB at 3.065 bits.
    - Quality: KL 0.111 against BF16 on 4,088 tokens, perplexity +1.7 %.
    - Effort: GSQ ran 8 steps on 64 sequences of 1,024 tokens, and no RCO step improved on its start.
    - **Every format in it is a simple block format with a Pascal kernel or a planned one;** none is an IQ grid.
  - **[Egalitaristen/AutoGSQ-RCO](https://github.com/Egalitaristen/AutoGSQ-RCO)** is an early GGUF packer, tested
    on Qwen3-4B only.

So the cheap way to reproduce GSQ-RCO is ukisai's: ISTA's allocations, our own importance matrix and `llama-quantize
--tensor-type`. It keeps RCO's allocation and gives up GSQ's rounding, which is where most of the gain at 2–3 bits
is.

## 3. Sub-1-bit quantization: what exists

### 3.1 How a format stores fewer bits than weights

| Mechanism | Methods | How | Training |
|---|---|---|---|
| Low-rank binary factorization | NanoQuant, LittleBit, LittleBit-2, DBF | W ≈ s₁ ⊙ (U Vᵀ) ⊙ s₂ᵀ with U, V in {±1}; r(n + m) sign bits for an n × m matrix | post-training (NanoQuant, DBF) or QAT (LittleBit) |
| Binarization with structured sparsity or codebooks | STBLLM (N:M), BTC-LLM (binary codebook), PTQ1.61, ARB-LLM, BiLLM | sign bits with masks, salient channels at higher precision | post-training |
| Entropy-coded ternary | QMoE | ternary values, 88.6 % zeros, a 2¹⁶-entry dictionary code | post-training |
| Cross-expert sharing | MoBE, MoE-SVD, Sub-MoE, D²-MoE, BitsMoE, MoBiE | a basis shared by the experts of a layer, plus small per-expert factors or deltas | post-training |
| Removing experts | REAP, RCO (ISTA's Coder), ProQuant | dropped experts cost 0 bits | post-training |

Several papers count only sign bits and indices: NanoQuant's all-in count puts STBLLM's "0.80 bits" at 4.13, and
BTC-LLM does not count its scales. Below, bits are all-in where a source gives them.

### 3.2 NanoQuant

Chong, Kim, Kim, Choi, ICML 2026 ([arXiv 2602.06694](https://arxiv.org/abs/2602.06694) v3;
[SamsungLabs/NanoQuant](https://github.com/SamsungLabs/NanoQuant), Apache-2.0):
- **Format:** W ≈ s₁ ⊙ (U Vᵀ) ⊙ s₂ᵀ, U ∈ {±1}^(n × r), V ∈ {±1}^(m × r), s₁ per output and s₂ per input channel; the
  code adds an optional per-rank scale. Storage r(n + m) + 16(n + m) bits.
- **Fitting, per block:**
  1. Diagonal K-FAC preconditioners from activation and gradient statistics.
  2. Tune the block's still-unquantized weights to absorb upstream error.
  3. ADMM on the preconditioned matrix: 400 steps of a Cholesky solve, a projection to signs times a rank-1
     magnitude, and a dual update.
  4. Balance the magnitudes, then refine the continuous latent factors and scales through a sign straight-through
     estimator on block outputs.

  Finally, only the scales are tuned on the KL to the full model's logits.
- **Calibration and cost:** 128 × 2,048 WikiText-2 tokens by default, up to 2.1M tokens. Calibrating on C4 instead
  worsens WikiText-2 perplexity sharply (Llama-2-7B 10.34 → 17.99), so the headline numbers are in-domain. One H100:
  Llama-2-7B 1.7–2.5 h, Llama-2-70B ~13 h.
- **Rank:** one rank per target width for the whole model; per-layer allocation is future work. The code sets
  r = mn·b/(m + n) − 16, rounded down to a multiple of 32 and **clamped to min(m, n)**.
- **Quality against other methods** (Llama-2-7B, WikiText-2 perplexity and 5-task zero-shot mean; FP16 5.47; Table 8):

  | Bits | Method | Perplexity | Zero-shot |
  |---|---|---|---|
  | 2.02 | QTIP | 6.29 | 62.05 |
  | 2.02 | AQLM + PV-tuning | 5.84 | 61.35 |
  | 2.00 | NanoQuant | 6.52 | 59.09 |
  | 1.58 | AQLM + PV-tuning | 7.32 | 55.22 |
  | 1.50 | NanoQuant | 7.08 | 55.81 |
  | 1.02 | AQLM + PV-tuning | 8.28 | 50.21 |
  | 1.00 | NanoQuant | 8.99 | 47.81 |

  It wins only at 1.5 bits and below; at 2 bits the codebook and trellis methods are better.
- **At 1 bit and below:**
  - Llama-2-70B 3.32 → 6.52; 0.55 bits 9.82.
  - Zero-shot means (6 tasks): Llama-3-8B 71.44 → 45.95, Qwen3-8B 71.22 → 48.94.
  - Ratio of 1-bit to FP16 perplexity by size: Llama-2 1.78–1.96×, Llama-3 2.44× (8B) and 3.96× (70B), Qwen3 2.18×
    (0.6B) to 1.71× (14B), Gemma-3-27B 5.6×. Larger models do not reliably lose less.
  - No coding, reasoning or MMLU-type results; no MoE model.
- **Code (HEAD `a9e0a4308`, 2026-07-03, three commits):**
  - Model types `llama`, `mistral`, `mixtral`, `mobilellm`, `qwen3`, `gemma*` and `opt`, factorizing only modules
    named `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj`. No expert handling, and nothing for Qwen-Next
    types.
  - Output is its own `.pt` format.
  - Decode kernel: half2/bf16x2 with 16-bit accumulation. Prefill kernel: derived from Marlin, needs sm_80.
- **Third-party:**
  - [arelath/llama.cpp](https://github.com/arelath/llama.cpp) (branch `nanoquants`) adds
    `GGML_OP_NANOQUANT_LINEAR` with FP32-accumulating CUDA kernels. It has no `MUL_MAT_ID`, so no MoE. Its Qwen3-8B GGUF
    goes from WikiText-2 perplexity 23.11 to 40.60 on its own short windows.
  - A replication on Llama-3.2-1B got 7 % higher perplexity than the paper, and found that most of the refinement's
    gain comes from flipping ~1 % of the signs.
  - A 1.13-bit drafter keeps 91 % of the full-precision acceptance length.

### 3.3 Other sub-1-bit and ~1-bit methods

| Method | Kind | Best result against FP16 | MoE | Code |
|---|---|---|---|---|
| LittleBit (NeurIPS 2025) | QAT, binary latent factors + residual factorization | Llama-2-7B 9.08 at 1.04 bits (196M training tokens) | no | [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit), CC BY-NC 4.0 |
| LittleBit-2 (ICML 2026) | LittleBit + a rotation toward the binary hypercube | Llama-2-7B 8.27 at 1.0, 9.65 at 0.55, 14.70 at 0.1; Llama-3-8B 11.53 at 1.0 | no | same, `--use_itq` |
| DBF (TMLR 2026) | two binary factors with scales, middle size not capped; ADMM, block fine-tuning, PV-tuning | Llama-2-7B (context 4,096, FP16 5.12) 6.09 at 2 bits (QuIP# 6.19, QTIP 5.86), 7.01 at 1.5 (AQLM + PV 7.32), 8.76 at 1.0 | no | [usamec/double_binary](https://github.com/usamec/double_binary) |
| BTC-LLM (ACL 2026) | ARB binarizer + learned transform + binary codebook | Qwen3-8B 9.72 → 11.60 at "1.11", 13.12 at "0.8" (scales not counted) | no | yes |
| STBLLM (ICLR 2025) | N:M structured binarization | Qwen3-8B 12.50 at "0.80"; 109 at "0.55" | no | yes |
| PTQ1.61, ARB-LLM, BiLLM | binarization with salient channels | Llama-2-7B 12.70 at 1.61; 11.80 at 2.51 all-in; 19.87 at 2.88 all-in | no | yes |
| QMoE (MLSys 2024) | entropy-coded ternary | SwitchTransformer-c2048 at 0.807 bits, C4 loss 1.18 → 1.26 | Switch only | yes |

### 3.4 MoE-specific results at 1–2 bits

| Method | Model | Bits | Result |
|---|---|---|---|
| BitsMoE ([arXiv 2606.00079](https://arxiv.org/abs/2606.00079)): 8-bit shared SVD basis, ILP bit allocation to expert factors | Qwen3-Next-80B-A3B, 7-task mean | 2.0 / 1.8 / 1.6 | 70.70 / 69.23 / 67.58 against 73.42 (GPTQ 2-bit 51.31) |
| ProQuant ([arXiv 2610.30779](https://arxiv.org/abs/2610.30779)): each expert pruned or kept at 2 or 3 bits | Qwen3-30B-A3B, six tasks incl. GSM8K, GPQA-D, BFCL | 2.0 / 1.5 | 74.49 / 67.30 against 80.94; 1-bit scalar experts did worse than pruning them (68.46 against 74.49) |
| ProQuant | Qwen3.5-35B-A3B | 1.5 | 60.85 against 87.32 |
| MoBiE (withdrawn after v4 for derivation errors): shared backbone + binary expert blocks | Qwen3-30B-A3B | ~1.34 | WikiText-2 8.70 → 12.80; zero-shot 70.10 → 57.58 |
| MoBE, MoE-SVD, Sub-MoE, D²-MoE: shared bases or deltas, no quantization | Qwen3-30B to Kimi-K2 | 15–30 % fewer parameters | 1–12 points lost; never combined with low bits |

### 3.5 Flash-Next files below 2 bits

| File | Bits | Method | Quality | Rig |
|---|---|---|---|---|
| ISTA GSQ-RCO-Coder | 1.89 effective (3.5 stored, 256 of 512 experts) | RCO expert pruning | SWE-bench Verified 79.2 against 82.8; one user measured factual recall 63 % → 33 % | fits (29.6 GB) |
| Unsloth UD-IQ1_S / UD-IQ1_M | ~2.5–2.8 over the experts (my calc), the down projections at 4.5 | imatrix | Unsloth's own set: KLD 0.396 / 0.315, UD-Q2_K_XL 0.225, Q8_0 0.027 | fit (43.7 / 45.7 GB); ranked below GSQ-RCO IQ2_XS in [flash_next_variants.md](flash_next_variants.md) |
| SyzygyResearch Mach-2-Additive-Medium | 1.70 | fine-tune with an undisclosed method, trellis codes with a bit rate per expert and a shared low-rank residual for "demoted" experts | maker: 93–98 % retention on Terminal-Bench 2.1, DeepSWE, NL2Repo | its own fork, tested on one A100; BF16 n-gram table in ~105 GB of host RAM |
| turboderp EXL3 | 1.66 / 2.18 / 2.68 | trellis | GPQA-D 86.4 / 89.4 / 91.4 against 91.7 | tensor-core kernels |
| agentionai Gyro-S/M/L | ~2.1–3.3 | trellis | maker: KLD 0.401 / 0.291 / 0.146 | their fork; est. 2–3.5× slower per weight than Q2_0 |

No Flash-Next file below 1.66 bits per expert weight exists, and no sub-1-bit Flash-Next file of any kind was found.

## 4. Applied to Flash-Next's experts

- **The rank cap (my calc, from the code's rule).** For a 640 × 2560 expert, bits = (r + 16) / 512: r = 256 gives
  0.53 bits, r = 480 gives 0.97, and the cap r = 640 gives 1.28. NanoQuant as released cannot give these experts 1.5–2
  bits. The cap is the code's, not the representation's: DBF lets the middle size exceed min(m, n) and reports its
  best results at 1.5–2.3 bits.
- **The shape is not against it.** At a given width, r / min(m, n) = bits / (1 + min/max): 0.8 × bits for 640 × 2560,
  0.73 × bits for Llama-2-7B's MLP. Whether 640-wide fine-grained experts have spectra heavy-tailed enough to compress
  is open: no source measures it.
- **All experts at ~1 bit** puts the model in ~19 GB, on four cards. By the dense-model evidence (section 3.2) that is
  a different model: perplexity 1.7–4×, zero-shot means down by a third. Flash-Next's community reports that Q2-class
  files already degrade in agent work.
- **A 1-bit tier for cold experts.** Putting the 116 least important experts per layer at 1 bit instead of 3.33 would
  free ~8 GB (my calc). ProQuant found 1-bit scalar experts worse than dropping them. Binary factorization could do
  better, but no result exists. Hot sets also change with the task: in Qwen3-30B the ten hottest experts of a layer
  are disjoint on WikiText, GSM8K and HumanEval. GGUF holds one format per fused expert tensor, so this needs split
  tensors and a second `MUL_MAT_ID` in the engine.
- **Running NanoQuant on Flash-Next:**
  - It needs a model adapter: the code does not know `qwen4exp`, its hyper-connections, its n-gram table or the
    DeltaNet layers.
  - It needs an expert-aware restructuring: ~74,000 linears against Llama-2-70B's 560, each tuned for 8 epochs.
  - The calibration is thin per expert: ~5,100 tokens per expert per layer at the default 262k tokens, unevenly
    spread.
  - The cost is est. days to weeks on the quality machine, for a quality class that section 3 already rules out.

## 5. On Pascal

Decode cost per routed-expert weight on a P106, and the experts' share of one token (2.36 G weights):

| Format family | Bits | ps per weight | Experts per token | Basis |
|---|---|---|---|---|
| Q4_0 (the fork's kernel) | 4.5 | 4.2 | 9.9 ms | measured |
| IQ2_S / IQ3_XXS, v3 Pascal kernels | 2.56 / 3.06 | 6.4–9.2 | 15–22 ms | measured |
| IQ3_XXS, llama.cpp's kernel | 3.06 | 20.5 | 48 ms | measured |
| IQ3_S, IQ2_XS, IQ2_XXS, IQ4_NL (no Pascal kernel yet) | 2.06–4.5 | est. 6–21 | — | [low-bit note](pascal_lowbit_format.md) |
| Q2_0 | 2.25 | est. 3–6 | 7–14 ms | generic kernel, no lookups |
| N2P / N3P (route 2) | 2.69 / 3.25 | est. 3.7 / 4.0 | ~9 ms | byte-permute decode in registers |
| **Binary factorization** | 1.0 / 2.0 / 2.8 | **est. ~1.0 / ~2.1 / ~2.9** | 2.4 / 5 / 6.8 ms | per 32 sign bits 8 LOP + 8 dp4a (`w & (0x01010101 << j)` against transposed int8 activations), ~0.5–0.6 slot-equivalents per stored bit; two stages per matrix |
| Trellis (QTIP, EXL3, Gyro, Mach-2) | 1.7–2.7 | est. 9–28 | 21–66 ms | a 32-bit hash multiply per weight (3 XMAD on sm_61) |
| Vector codebooks (AQLM, VPTQ) | ~2 | issue-bound, 5–9 slot-equivalents per byte | — | shared-memory lookups |

What this means:
- **Below ~3.5 bits the card is issue-bound per weight**, so most low-bit formats cost about as much time as Q4_0 or
  more. Binary factorization is the exception: its work scales with stored bits, not with weights. It is the only
  family that is faster per weight than Q4_0 while being smaller.
- **The experts are not where a Flash-Next token's time goes.** The dense tensors (3.4–4.8 GB per token, ~23–32 ms)
  and the operations without weights (~30–40 ms) dominate
  ([flash_next_variants.md](flash_next_variants.md), section 6). The whole gain from the fastest expert format over
  the v3 IQ kernels is ~10–15 ms of a ~70–100 ms token.
- **The binary decode fits sm_61 as a kernel:** both scales sit on the matrix borders. s₂ folds into the activations
  before int8 quantization and s₁ into the output; the rank-r intermediate is requantized to int8 between the two
  stages. In Miner Llama it would be a new tensor layout (two sign planes and three scale vectors per expert) and a
  two-stage `MUL_MAT_ID` kernel. arelath's FP32 kernels are a starting point for the dense case. Est. 1–2 weeks with
  tests.
- **Every published sub-1-bit kernel needs replacing on this card.** NanoQuant's decode accumulates in half2 (1/64
  rate), and its prefill needs sm_80. LittleBit ships none.

## 6. Routes to IQ3_S-like quality in 44–47 GB, ranked

1. **ISTA's IQ3_XXS as it is (47.0 GB).**
   - Quality: 99.4 % of BF16's task mean.
   - Fit: est. ~128k of q8_0 cache, no MTP draft. The hyper-connection rewrite to Q8_0 (lever 2 in
     [flash_next_variants.md](flash_next_variants.md)) frees 0.59 GB.
   - Speed: the v3 kernels cover its IQ2_S and IQ3_XXS layers; IQ3_S, IQ2_XS, IQ2_XXS and IQ4_NL need Pascal kernels.
   - Needs: a 47.0 GB download (its first shard; the table shard is the IQ3_S build's). Swift's IQ3_XXS is local.
2. **A composed file from the GSQ builds, allocated on our texts.**
   - Every GSQ build of one base comes from the same rounding, so tensors can be mixed across builds per 8-layer
     group, scored alone against a Q8_0 reference and composed at 44.3 and 47 GB. This is
     [Q167](../../data/2026-10-27_ornith_alloc/README.md)'s method for Ornith.
   - RCO's allocation optimized KL on ISTA's undisclosed calibration text. An allocation on held-out agent sessions
     targets the rig's actual work.
   - Caveat: GSQ calibrated each block on the outputs of its own build's earlier blocks, so mixing adds a small
     mismatch, which the composed file's KLD measures.
   - Swift's four builds are on the quality machine; ISTA's three smaller ones need ~124 GB of first shards.
3. **A 640-friendly 3-bit format for the down projections (N3P-128).**
   - An N3P variant with 128-weight blocks: 640 = 5 × 128. Half of N3P's 104-byte block plus a second FP16 scale is
     53 bytes per 128 weights, 3.31 bits, more with alignment padding; 3.375 bits is used below.
   - **The source:** requantize IQ3_S's 39 IQ4_NL down tensors (a 4.5-bit source, so its own error is small against a
     3.4-bit target), keep its 9 Q2_0 layers, and pair them with IQ3_XXS's gate/up. That is a ~46.8 GB file with the
     hyper-connections in Q8_0 (my calc).
   - **The down-projection error,** with squared error ∝ 4^(−bits) and equal layer sensitivities (my calc):

     | Down projections | Summed error (relative) | GB |
     |---|---|---|
     | IQ3_XXS's (18 × 4.5 + 30 × 2.25 bits) | 1.36 | 15.57 |
     | N3P-128 mix (39 × 3.375 + 9 × 2.25) | 0.76 | 15.93 |
     | IQ3_S's (39 × 4.5 + 9 × 2.25) | 0.47 | 20.53 |

     That is est. 0.56× of IQ3_XXS's error for +0.36 GB, and 0.35× for IQ3_S's, using 4.6 GB fewer than IQ3_S. RCO put
     IQ3_S's Q2_0 layers where they hurt least, which favours the mix.
   - Our N3P rounding is imatrix-based, not GSQ's; on Ornith's matrices N3-class levels matched IQ3_S at equal size.
   - Speed: the 39 layers read 0.64 G weights per token, so against IQ4_NL (est. 6–10 ps per weight) N3P-128 saves
     est. 1–4 ms per token.
4. **pfeifferj's GSQ-RCO reproduction (47.94 GB)** in Q2_0/Q2_K/Q3_K/Q4_K/Q4_0/Q8_0.
   - The same method, and simple block formats: Q4_0 and Q8_0 have the fork's kernels, Q2_K/Q3_K Pascal kernels are
     est. 3.4/4.9 ps per weight, and Q2_0 needs tuning.
   - Its quality evidence is thin (4,088 tokens of KL, one MMLU-Pro run where it beats BF16).
   - Worth one KLD run against IQ3_XXS: if it is level, it is the more Pascal-friendly file of the two.
5. **GSQ-style learned rounding into Pascal formats (N2P, N3P, N3P-128, Q2_0) from BF16.**
   - The highest ceiling: GSQ's gain over imatrix rounding is largest at 2–3 bits (Qwen3-8B Q2_K, task mean +6.3).
   - Needs: the 354 GB BF16 checkpoint; a Flash-Next adapter; the format projection ISTA has not released; per-expert
     training.
   - Cost: est. 2–5 days of the three RTX 3090s, scaling GSQ's 24 h on 8 × H200 for Kimi-K2.5's 1.0 T expert weights
     to Flash-Next's 0.12 T. The V100s have no BF16.
6. **Binary factorization at 2–3 bits (DBF-style, rank uncapped).** The one sub-1-bit idea with a Pascal upside (est.
   ~2.9 ps per weight at 2.8 bits), and the one with the least evidence: no MoE result, and DBF's fitting cost on
   Llama-2-7B alone is 6–8 h on an RTX 4090 plus 24 h on 4 × A100. Worth a CPU-proxy gate (E5), not more for now.
7. **Not pursued:**
   - sub-1-bit factorization of every expert (section 4);
   - a 1-bit cold-expert tier (no evidence it beats pruning, and hot sets change with the task);
   - expert pruning (knowledge loss in the Coder);
   - trellis and vector-codebook files (too slow on sm_61, or tensor-core only);
   - shared-basis MoE methods (15–30 % fewer parameters at 1–12 points, no quantized variant, no GGUF).

## 7. Experiments that settle it

| Step | Where | Cost | What it settles |
|---|---|---|---|
| E1. [Fit probe](../../../scripts/fit_probe.sh) of ISTA IQ3_XXS at 64k/128k/262k, q8_0 cache, with and without the MTP draft | rig | minutes | the real weight ceilings in section 1 |
| E2. KLD against a Q8_0 reference on wikitext-2, agent sessions and code: ISTA IQ3_XXS, Swift IQ3_XXS/IQ2_XS/Q2_0, pfeifferj 3.5-bit, ISTA IQ3_S as the bound (section 9 of [flash_next_variants.md](flash_next_variants.md)) | quality machine | a Q8_0 reference (~134 GB across seven GPUs), then hours per file | where each file sits under the project's own method |
| E3. Allocation per 8-layer group across Swift's four builds (Q167's `alloc.py`), composed at 44.3 and 47 GB | quality machine, V100s | est. 6–10 h | route 2's gain over RCO's own allocation |
| E4. N3P-128 in `bench/mmvq_pascal` on one P106, its CPU reference and quantizer; the requantized-down file of route 3 | code; rig (one card, minutes); quality machine | days of code, hours of scoring | route 3: decode ps per weight and the KLD against IQ3_XXS and IQ3_S |
| E5. CPU proxy: NanoQuant's ADMM and DBF at 1.0, 1.5, 2.0 and 2.7 bits, rank uncapped, against IQ2_S/IQ3_XXS/N2P/N3P at equal bits, on Ornith's 84 expert matrices and Flash-Next experts from one BF16 shard | quality machine (CPU or one GPU) | hours, one ~5 GB download | whether binary factorization is competitive per bit on fine-grained experts. Gate: within ~10 % of N2P/N3P's importance-weighted error at equal bits (the proxy leaves out NanoQuant's refinement, so it understates the method) |
| E6. Agent check of the two best files from E2–E4, with the n-gram table on an SSD | rig | a session per file | the choice |
| E7. GSQ-style rounding into Pascal formats (route 5) | quality machine | est. 2–5 days of three RTX 3090s, after the code | only if E3 and E4 leave a gap to IQ3_S that the agent check shows matters |

## Sources

GSQ-RCO
- ISTA-DASLab, [Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF): card,
  `tensor-allocation/` ([tree](https://huggingface.co/api/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/main/tensor-allocation)),
  discussions [#5](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/5),
  [#26](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/26),
  [#39](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/39),
  [#44](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/44),
  [#45](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/discussions/45) (P; discussions S)
- ISTA-DASLab, [Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF),
  discussions [#19](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/discussions/19),
  [#22](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/discussions/22) (P; discussions S)
- ISTA-DASLab, [Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF),
  [DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
  (IQ1_M at 1.75 bits keeps 98.7 % of its task mean) (P)
- Dadgarnia et al., "GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling",
  [arXiv 2604.18556](https://arxiv.org/abs/2604.18556) v2 (P); [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ) at
  `03fc164`, issues [#4](https://github.com/IST-DASLab/GSQ/issues/4), [#9](https://github.com/IST-DASLab/GSQ/issues/9) (P)
- Helcig, Alistarh, "Model Compression with Exact Budget Constraints via Riemannian Manifolds",
  [arXiv 2605.00649](https://arxiv.org/abs/2605.00649) v3 (P); [IST-DASLab/RCO](https://github.com/IST-DASLab/RCO) (P)
- [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  [pfeifferj/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
  [Egalitaristen/AutoGSQ-RCO](https://github.com/Egalitaristen/AutoGSQ-RCO) (P, makers' own numbers)

Sub-1-bit methods
- Chong, Kim, Kim, Choi, "NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models", ICML 2026,
  [arXiv 2602.06694](https://arxiv.org/abs/2602.06694) v3, Tables 2, 3, 4, 7, 8, 10 (P);
  [SamsungLabs/NanoQuant](https://github.com/SamsungLabs/NanoQuant) at `a9e0a4308`: `main.py`,
  `src/nanoquant/utils/utils.py` (`calculate_ranks`, model types), `kernel/cuda/` (P)
- [arelath/llama.cpp](https://github.com/arelath/llama.cpp) branch `nanoquants` and
  [arelath/Qwen3-8B-nanoquant-GGUF](https://huggingface.co/arelath/Qwen3-8B-nanoquant-GGUF) (S);
  [PseudoHunt/nanoquant-deletion-tests](https://github.com/PseudoHunt/nanoquant-deletion-tests),
  [PseudoHunt/flexdraft-1bit-drafter](https://github.com/PseudoHunt/flexdraft-1bit-drafter) (S)
- Lee, Kim, You, Kim, "LittleBit", NeurIPS 2025, [arXiv 2506.13771](https://arxiv.org/abs/2506.13771); Lee, Kim,
  "LittleBit-2", ICML 2026, [arXiv 2603.00042](https://arxiv.org/abs/2603.00042);
  [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) (P)
- Boža, Macko, "Double Binary Factorization", TMLR 2026, [arXiv 2505.11076](https://arxiv.org/abs/2505.11076) v3;
  [usamec/double_binary](https://github.com/usamec/double_binary) (P)
- BTC-LLM [arXiv 2506.12040](https://arxiv.org/abs/2506.12040); STBLLM [pprp/STBLLM](https://github.com/pprp/STBLLM);
  PTQ1.61 [arXiv 2502.13179](https://arxiv.org/abs/2502.13179); ARB-LLM
  [ZHITENGLI/ARB-LLM](https://github.com/ZHITENGLI/ARB-LLM); BiLLM
  [Aaronhuang-778/BiLLM](https://github.com/Aaronhuang-778/BiLLM); QMoE
  [arXiv 2310.16795](https://arxiv.org/abs/2310.16795) (P)

MoE compression
- BitsMoE [arXiv 2606.00079](https://arxiv.org/abs/2606.00079); ProQuant [arXiv 2610.30779](https://arxiv.org/abs/2610.30779);
  MoBiE [arXiv 2604.06798](https://arxiv.org/abs/2604.06798) (withdrawn); MoBE [arXiv 2508.05257](https://arxiv.org/abs/2508.05257);
  D²-MoE [arXiv 2502.17298](https://arxiv.org/abs/2502.17298); Sub-MoE [arXiv 2506.23266](https://arxiv.org/abs/2506.23266);
  DynaExq [arXiv 2511.15015](https://arxiv.org/abs/2511.15015) (routing skew across tasks); REAP
  [arXiv 2510.13999](https://arxiv.org/abs/2510.13999) (P)
- [wuliaott/moe-expert-grading](https://github.com/wuliaott/moe-expert-grading) (per-expert types in a llama.cpp fork,
  CPU and Vulkan, unreviewed) (S); [ik_llama.cpp #2422](https://github.com/ikawrakow/ik_llama.cpp/issues/2422)
  (partial super-blocks for 640-wide rows) (P)

Other Flash-Next files
- [Unsloth Qwen3.8-Flash-Next docs](https://unsloth.ai/docs/models/qwen3.8-next) (KLD table, reference set not
  stated); [SyzygyResearch/Mach-2-Additive-Medium](https://huggingface.co/SyzygyResearch/Mach-2-Additive-Medium) and
  [its fork's docs](https://github.com/SyzygyResearch/llama.cpp-mach1/blob/master/docs/mach1.md);
  [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3);
  [agentionai Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) (P, makers' own numbers)

This project
- [flash_next_variants.md](flash_next_variants.md) and its
  [census](../../data/2026-10-27_flash_next_variants/README.md); [pascal_lowbit_format.md](pascal_lowbit_format.md);
  [pascal_forks_and_quants.md](pascal_forks_and_quants.md), section 4; [slq.md](slq.md);
  [Q167's allocation data](../../data/2026-10-27_ornith_alloc/README.md)
