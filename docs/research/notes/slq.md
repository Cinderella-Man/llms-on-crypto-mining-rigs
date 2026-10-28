# SLQ (statistically-lossless quantization): what it is and what Miner Llama can use from it (2026-10-27)

SLQ is a quantization method from IST Austria's DAS Lab and ETH Zurich (Helcig, Kurtic, Alistarh; COLM 2026), with code
at [IST-DASLab/SLQ](https://github.com/IST-DASLab/SLQ). Models made with it appeared on Hugging Face in October 2026.
This note asks whether Miner Llama can use it on Pascal cards (compute 6.1, dp4a, no tensor cores): the models as they
are, the models converted to GGUF, or ideas taken into this project's own quantizer and formats.

Read-only study:
- SLQ's repository at commit `eb9f1f294c99940029465cc970862ba36951d0cc` (2026-10-08), read through the GitHub API as
  text: all 41 text files.
- The paper, [arXiv 2605.02404v2](https://arxiv.org/abs/2605.02404) (2026-08-09).
- The model cards and file lists of every SLQ model on Hugging Face.
- The safetensors headers and `config.json` of the three SLQ checkpoints already on the quality machine. No weights were
  read.

Nothing from SLQ was cloned, installed or run.

Conventions:
- **(measured)**: counted from files, with the data in
  [docs/data/2026-10-27_slq](../../data/2026-10-27_slq/README.md); no model was run for this note.
- **(my calc)**: arithmetic in this note.
- **(est.)**: an estimate, not yet measured.
- "bpw" counts every stored byte, scales included.
- EAR, the authors' fidelity metric, is defined in section 1.2.

## Summary

1. **SLQ is a bit-width allocator, not a new format.**
   - Its building block is plain GPTQ to asymmetric integer grids of 2–8 bits, with one scale and one zero point per
     128 weights of a row.
   - It quantizes every layer at every width, measures how much each layer's width costs the model's output
     distribution (Shapley values over random orders), and picks per-layer widths by integer linear programming.
   - A binary search then finds the smallest average width that meets a fidelity target: "task-lossless" (benchmark
     scores ≥ 99 % of BF16, reached at 3.3–4.7 bits per parameter) or "distribution-lossless" (EAR ≥ 0.99, the overlap
     of the top-10 next-token distributions, reached at 5.0–6.6 bits per parameter).
   - Fidelity is measured on the calibration text the search itself used.
2. **Models: none from the authors, none of this project's models.**
   - IST-DASLab has published no SLQ checkpoint; a request to do so has been open since May
     ([issue #1](https://github.com/IST-DASLab/SLQ/issues/1)).
   - The SLQ models that exist are The Kaitchup's Qwen3.8-27B, at EAR targets 98 %, 98.5 % and 99 %: three
     vLLM/Humming checkpoints (2026-10-21) and, since today, three GGUFs.
   - A few small MLX models (Apple only) exist too.
   - Nothing exists for Flint, Ornith-1.5 or Qwen3.6.
   - SLQ's code cannot run on Flint (transformers has no Flint model) and would skip Ornith's routed experts and
     DeltaNet projections (section 2.2).
3. **On Pascal (measured, my calc):**
   - The vLLM checkpoints cannot run: their Humming kernels need compute 7.5 or newer.
   - The Kaitchup GGUFs run as they are: 99 % of EAR98's bytes are formats with Miner Llama Pascal kernels (Q5_K, Q6_K,
     Q4_K, IQ4_XS, Q8_0). EAR98.5 and EAR99 keep the output head in BF16.
   - SLQ's integer grid converts exactly only into Q4_1 (layers of up to 4 bits) and Q5_1 or Q8_0 (5–7 bits). For
     EAR98 that costs 27–53 % more bytes than the Kaitchup GGUF at the same EAR.
4. **Quality per byte (measured):**
   - At the same measured EAR, Kaitchup's GGUFs, built from Unsloth's K-quant tensors, spend 3 %, 9 % and 17 % fewer
     bytes on the decoder than the integer-grid checkpoints (EAR98, 98.5, 99).
   - SLQ's formats are not better per byte than llama.cpp's.
   - EAR@10 on calibration text is not this project's quality test (full-vocabulary KLD on wikitext-2, within noise of a
     reference file). Comparing the two needs a run (section 5, E2).
5. **Verdict: an idea worth borrowing.** The format is not applicable and the existing models are outside this
   project. The allocator is worth taking: measured per-group costs, a multiple-choice knapsack, a binary search on the
   budget. It fits Ornith-1.5-122B exactly:
   - The routed experts' bits are free in time on the P106 (issue-bound at 3.5–4 ps per weight whatever the width,
     [pascal_lowbit_format.md](pascal_lowbit_format.md)).
   - The cards hold 45.4 GB.
   - So the question is only where the bytes go.

   Proposed experiment E1: allocate N2P/N3P and Q5_0/Q8_0 per 8-layer group of Ornith-122B. It needs no downloads,
   one extra CPU quant and est. 5–8 hours of the two V100s.

## 1. What SLQ is

### 1.1 The method, from the code

| Stage | Entry point | What it does |
|---|---|---|
| Quantize | `slq_quantize.py`, `src/quant/fast_obq.py`, `scripts/run_gptq.sh` | GPTQ per linear layer: one Hessian 2XᵀX/n per layer (1 % damping), then every width in {2, …, 8} from that Hessian. Asymmetric integer grid, min–max scale and zero per row and 128 input columns, computed at the start of each group from the error-updated weights; the scale is rounded to BF16, the zero is an integer; no act-order. Later blocks are calibrated on the outputs of the 4-bit block. Default calibration: 1,024 × 8,192 tokens of FineWeb-Edu `sample-10BT` |
| Layer store | same | per layer and width a dequantized BF16 copy (`<bits>.pth`, read by the search) and the codes, scales and zeros (`<bits>_qparams.pt`) |
| Sensitivity | `slq_build_database.py` | on 512 × 2,048 FineWeb-Edu tokens: (a) "linear" costs: Gaussian noise at 15 levels per layer group, cost = α × relative weight error; (b) multi-bitwidth Shapley values: for each width b < 8, 50 random orders, starting from all-8-bit and moving groups to b one at a time, each group credited with the change it causes in NLL, top-10 KL and EAR. Gate+up and q+k+v share a width (vLLM fuses them) |
| Allocation | `slq_search_dl.py` | an ILP (OR-Tools SCIP): one width per group, parameter-weighted mean ≤ budget, minimum summed cost; binary search on the budget (0.01-bit steps) until the measured EAR meets the target ("distribution-lossless"), or until a KL predicted from one uniform 4-bit anchor meets a recovery target ("task-lossless", the paper's Algorithm 2, with a check that rejects the result if measured and predicted KL drift apart by more than 2×) |
| Checkpoint | `slq_build_checkpoint.py` | the fake-quantized model in BF16 (full BF16 size) plus a Humming config that vLLM uses to pack the weights on load, and a `serve.sh` |

The paper ([arXiv 2605.02404](https://arxiv.org/abs/2605.02404)) adds a "γ² law": a symmetric grid's noise variance is
γ² times an asymmetric grid's, with γ = 2·max(|L|, |U|)/(U − L) for weights in [L, U], so near-lossless targets need
asymmetric grids. It also offers [RCO](https://github.com/IST-DASLab/RCO) as a slower, more exact replacement for the
ILP.

### 1.2 The claim and how it is measured

- **EAR** (expected acceptance rate), `src/metrics/evaluation.py`: at each position, both the BF16 model's and the
  quantized model's probabilities are renormalized over BF16's 10 most likely tokens. EAR is then the mean of
  Σ min(p, q). It is one minus the total-variation distance on that support: the chance that a token sampled from the
  quantized model would be accepted by the BF16 model in speculative sampling. The KL the search uses is the same top-10
  renormalized KL.
- **Distribution-lossless** = EAR ≥ 0.99 (0.985–0.99 per model in the paper). The paper says EAR ≥ 0.99 "generally
  implies" KL far below 0.01.
- **Task-lossless** = mean zero-shot benchmark recovery ≥ 99 % of BF16. The paper's noise-injection sweep over seven
  models finds accuracy flat while the top-10 KL stays below about 0.01.
- **The fidelity is in-sample.** The search, `eval/run_ear.py` and `eval/run_kl.py` all score EAR and KL on the same
  seeded FineWeb-Edu calibration set (`train=True`). The Kaitchup model cards say the same of their own EAR figures.

### 1.3 Bits per weight

- Table 7 of the paper counts asymmetric g128 as nominal + 0.156 bpw (20 bits per 128 weights, which I read as a
  16-bit scale and a 4-bit zero) and symmetric g128 as nominal + 0.125.
- The search reports the parameter-weighted mean over the layers it quantized (`compute_avg_bitwidth`). The embeddings,
  the output head and any layer left in BF16 are outside it; the paper's Table 1 footnote says the same of its GPTQ-Int4
  row ("bits averaged over subset").

| Model (paper) | Task-lossless bpp | Distribution-lossless bpp | vLLM speed vs BF16 (L40S, task- / distribution-lossless) |
|---|---|---|---|
| Qwen3-8B | 4.72 | 5.70 (EAR 0.99, top-10 KL 0.0023) | 1.94× / 1.74× |
| Qwen3-32B | 3.59 | 4.98 | — / 2.94× (1 GPU against BF16 on 2) |
| Qwen3.5-27B | 3.30 | 5.19 | — |
| Llama-3.3-70B-Instruct | 3.59 | 6.55 | 3.68× / 2.23× (2 GPUs against 4) |
| Qwen3.5-35B-A3B (MoE, Table 6) | 3.73 | 5.75 | — |

### 1.4 Cost of quantizing a model

- **GPU time:** the Shapley stage takes ~2.5 hours per width for an 8B model on one A6000, ~15 hours for the six widths
  below 8. The orders are independent, so it divides across GPUs. The paper gives no GPTQ time.
- **Disk (my calc):** the layer store keeps, per width, a BF16 copy (2 bytes per weight) and uint8 codes (1 byte), about
  21 bytes per weight for seven widths:

  | Model | Layer store |
  |---|---|
  | 8B | ~170 GB |
  | Flint (40B) | ~840 GB |
  | Ornith-1.5-35B | ~735 GB |
  | Ornith-1.5-122B | ~2.6 TB |

- **VRAM:** the sensitivity and search stages run the BF16 model on the GPUs (`load_model`, `device_map="balanced"`).
  Ornith-122B in BF16 is 244 GB, more than the quality machine's 158 GB.

## 2. Which models exist

### 2.1 Published SLQ models (Hugging Face, 2026-10-27)

| Repository | Base | Form | Size | Licence | Notes |
|---|---|---|---|---|---|
| [kaitchup/Qwen3.8-27B-SLQ-EAR98](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR98), [-EAR98.5](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR98.5), [-EAR99](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR99) | Qwen3.8-27B (dense hybrid, `qwen35`) | vLLM + Humming, packed uint4–8 | 23.7 / 25.6 / 30.5 GB | Apache-2.0 | "a customized version of SLQ": candidates from local GPTQ (INT2–8), an AutoRound quant, ISTA-DASLab's 3-bit GSQ and BF16, chosen per projection "including selected attention heads"; measured packed EAR 98.0008 / 98.5027 / 99.0025 %; on the quality machine already |
| [kaitchup/Qwen3.8-27B-SLQ-GGUF](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-GGUF) (created today) | same | GGUF, three files | 18.9 / 20.7 / 23.7 GB | Apache-2.0 | the same kind of search over GGUF tensors: per matrix, a choice among Unsloth's UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, Q4_0, Q8_0, BF16 and ISTA-DASLab's GSQ IQ3_S (never chosen); EAR@10 98.0004 / 98.5002 / 99.00003 %; MTP tensors kept (351 MB); validated on a GH200 |
| hermitdave/…-MLX-slq-… ([example](https://huggingface.co/hermitdave/Qwen3-1.7B-MLX-slq-4.49bpw)) | Qwen3-1.7B, LFM2.5-2.6B, Spark-X2.6-4B, K2-Horizon-7B | MLX | small (1.4 GiB for the 1.7B) | per base | Apple only |

- IST-DASLab's own recent releases are GSQ/RCO GGUFs and NVFP4 models, made by different methods.
- The SLQ search behind both Kaitchup releases is not public. The cards credit SLQ's paper and code but call it
  customized.

**Close to this project's models?** No. Flint (40B-A2.5B MoE), Ornith-1.5-122B/35B (hybrid MoE) and Qwen3.6 have no SLQ
file. Qwen3.8-27B is the dense model the rig ran on 2026-10-22
([speed-ups](../../benchmarks/2026-10-22-speedups.md), [dense_27b_speedups.md](dense_27b_speedups.md)), not one of
v3's models.

### 2.2 Could SLQ's code quantize Flint or Ornith on the quality machine?

Not as published.
- **Flint:** SLQ loads models through `transformers.AutoModelForCausalLM`, which has no Flint model
  ([drafter_data_generation.md](drafter_data_generation.md)).
- **Ornith-1.5:**
  - SLQ quantizes only `nn.Linear` modules whose names match `(q|k|v|o|gate|up|down)_proj$` (`scripts/run_gptq.sh`;
    `FastOBQ` asserts `nn.Linear`).
  - In transformers 5.8.1, the version SLQ pins, Qwen3.5-MoE keeps the routed experts as two 3-D parameters
    (`Qwen3_5MoeExperts.gate_up_proj`, `down_proj`,
    [modeling_qwen3_5_moe.py](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)).
  - It names the Gated DeltaNet projections `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` and `out_proj`.
  - A run would therefore quantize only the attention layers (12 in the 122B) and the shared experts, a few percent
    of the weights.
    The paper's MoE result (Table 6) needs code the repository does not contain.
- **The 122B:** a port would also need ~2.6 TB of layer store and a BF16 model larger than the machine's VRAM
  (section 1.4).
- **The result** would be a vLLM checkpoint that Pascal cannot run (section 3), which would then have to be converted
  at a byte cost (section 3.2).

The allocator alone is small (an ILP over a cost table) and is easier to rebuild over GGUF files (section 5).

## 3. How SLQ models run today, and on Pascal

### 3.1 Engines

- SLQ's checkpoints run in vLLM with [Humming](https://github.com/inclusionAI/humming) kernels. Humming supports
  "NVIDIA GPUs from SM75+", and BF16 activations need SM80+. vLLM itself needs compute 7.0+.
- Kaitchup tested on an RTX PRO 6000 Blackwell.
- The `fake` format (BF16 on the integer grid) runs anywhere transformers runs, at BF16 size and speed.
- None of this runs on compute 6.1.

### 3.2 Mapping SLQ's grid onto GGUF

An SLQ weight is w = s·(q − z), with q an integer of b bits, s a BF16 scale and z the zero point per row and 128 input
columns. GGUF blocks also run along the input dimension (32 weights), so each SLQ group covers four GGUF blocks:

| SLQ layer | Exact GGUF target | bpw | Pascal kernel in Miner Llama | Notes |
|---|---|---|---|---|
| 2–4 bits, any zero | Q4_1 (d = s, m = −s·z) | 5.0 | yes (fast format) | exact up to FP16 rounding of s and s·z |
| 5 bits | Q5_1 | 6.0 | no (generic kernel) | exact; or Q8_0 (8.5 bpw) for an integer zero |
| 6–7 bits, integer zero | Q8_0 (|q − z| ≤ 127) | 8.5 | yes | exact while z lies in the code range, as packed zeros do |
| 6–7 bits, float zero | none | — | — | Q8_0 holds no offset; lossy |
| 8 bits | none in 8.5 bpw | — | — | |q − z| reaches 255; Q8_0 is lossy, F16 exact at 16 bpw |

- Q4_0 and Q5_0 fit only when z is exactly 8 or 16.
- K-quants re-round the scales to 6 bits, so they are lossy too.
- The fake-quantized BF16 checkpoints hold the same values, but the widths per layer and the group grid would have to
  be read back from the config.

Applied to Kaitchup's EAR98 (my calc, from `vllm_summary.tsv`):
- Its projections are 6.6 % 4-bit, 56.1 % 5-bit, 34.0 % 6-bit and 3.3 % 7-bit by weight.
- An exact conversion is 6.87 bpw with Q5_1 for the 5-bit layers (20.9 GB of projections), or 8.27 bpw with fast
  formats only (25.2 GB).
- Kaitchup's own GGUF at the same EAR spends 16.4 GB: +27 % and +53 %.
- 94 of its 494 packed projections are 6- or 7-bit with float zero points (`vllm_modules.tsv`), which no fast format
  holds exactly.

**Converting SLQ checkpoints to GGUF is not worth it:** larger than a GGUF searched directly, and at ≥ 4 bits the
P106's decode time follows bytes.

### 3.3 The Kaitchup GGUF on Pascal (measured from its precision maps)

| File | Q5_K | Q6_K | Q4_K | IQ4_XS | Q8_0 | BF16 | IQ4_NL + Q3_K | Tensor bytes |
|---|---|---|---|---|---|---|---|---|
| EAR98 | 55.0 % | 15.4 % | 10.5 % | 9.7 % | 8.6 % | — | 0.9 % | 18.50 GB |
| EAR98.5 | 48.1 % | 24.5 % | 7.4 % | 5.5 % | 1.2 % | 12.5 % (head) | 0.8 % | 20.33 GB |
| EAR99 | 21.6 % | 50.7 % | 0.3 % | 0.6 % | 15.9 % | 10.9 % (head) | — | 23.32 GB |

- **Kernel coverage:** Miner Llama's Pascal matrix-vector kernels cover Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0, 99.1 % of
  EAR98's bytes. On a P106, Q5_K runs at 120–128 GB/s in [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md),
  near Q5_0's 138 GB/s.
- **BF16 head:** EAR98.5 and EAR99 read a 2.54 GB BF16 output head every token, a type with no Pascal kernel. EAR98
  (Q8_0 head) is the one to try.
- **Bytes per token (my calc):** EAR98 reads 17.8 GB per token (decoder + head), at least 9 % more than the 16.35 GB
  Q4_0 file the rig ran at 4.3 tok/s plain and 7.5 tok/s with MTP before Miner Llama's kernels.
- **MTP:** the file keeps its MTP tensors, so MTP decoding works.
- **Not measured** on the rig.

## 4. Quality against size, and against this project's bar

### 4.1 Is SLQ's integer grid better per byte than llama.cpp's formats? (measured)

Decoder projections only, each pair at the same EAR@10 target and measured by the same maker on the same 512 × 2,048
calibration tokens ([data](../../data/2026-10-27_slq/README.md)):

| Target | GGUF (Unsloth's K-quant tensors) | vLLM (SLQ integer grid, g128) | GGUF / vLLM bytes |
|---|---|---|---|
| EAR98 | 16.44 GB, 5.40 bpw | 16.87 GB, 5.34-bit mean (5.54 bpw stored) | 0.974 |
| EAR98.5 | 17.07 GB, 5.61 bpw | 18.75 GB, 5.98-bit mean (6.16 bpw) | 0.911 |
| EAR99 | 19.74 GB, 6.48 bpw | 23.66 GB, 7.59-bit mean (7.77 bpw) | 0.834 |

The GGUFs also quantize the embedding (Q4_K/Q6_K) and, in EAR98, the output head (Q8_0); the vLLM files keep both in
BF16. So the GGUFs reach the same EAR with fewer decoder bytes despite a lossier head.

Two notes on the comparison:
- The searches differed in their candidates, so this compares two whole pipelines, not two grids in isolation.
- The gap widens toward EAR 0.99. There llama.cpp's finer scales help: Q6_K has a scale per 16 weights, Q5_K a scale
  and minimum per 32, against SLQ's one scale and zero per 128.

### 4.2 Where EAR sits against KLD

The two tests differ in four ways:

| | SLQ's test | This project's test |
|---|---|---|
| Metric | top-10 EAR | full-vocabulary KLD and same top token |
| Text | FineWeb-Edu calibration text, in-sample | 80 × 512 tokens of wikitext-2 |
| Reference | BF16 | BF16 |
| Bar | absolute target | relative: within noise of a reference file |

Known points on Qwen3.8-27B:
- **Wikitext KLD**, bartowski's files against BF16 ([dense_27b_speedups.md](dense_27b_speedups.md)):

  | File | Size | KLD | Same top token |
  |---|---|---|---|
  | Q8_0 | 29.1 GB | 0.0009 | 98.75 % |
  | Q6_K | 23.9 GB | 0.0030 | 97.4 % |
  | Q5_K_M | 20.9 GB | 0.0053 | 96.9 % |
  | Q4_K_M | 17.4 GB | 0.0139 | 95.0 % |

- **EAR@10 on Kaitchup's calibration text** (Kaitchup's cards):

  | File | Size | EAR@10 |
  |---|---|---|
  | Unsloth UD-Q4_K_XL | 17.6 GB | 97.44 % |
  | Qwen's FP8, weights only | 30.9 GB | 98.10 % |
  | SLQ GGUF EAR98 | 18.5 GB | 98.0 % |
  | SLQ GGUF EAR99 | 23.3 GB | 99.0 % |

- **The missing comparison:** Kaitchup's table has no EAR for Unsloth's UD-Q5_K_XL (20.88 GB), the same size as
  EAR98.5 (20.70 GB) and one of its donors. So the cards do not show that the search beats Unsloth's own recipe at
  equal bytes.

What this suggests (not a measurement):
- Distribution-lossless at EAR 0.99 is a Q6_K-to-Q8_0-class fidelity on this model.
- Task-lossless is looser than this project's bar. SLQ's own uniform 4-bit baseline already keeps 97.6–99.8 % of
  benchmark scores, and this project found that task checks of a few hundred tasks cannot separate quants at all
  ([task_level_quality_checks.md](task_level_quality_checks.md)).

### 4.3 What it means for Flint and for Ornith on 54 GB

- **Flint** is far more sensitive than Qwen3.8-27B:
  - Unsloth's UD-Q4_K_XL scores KLD 0.120 and 87.7 % same top token against BF16, and the bar is within noise of that
    (KLD standard error ~0.004 in [flint-mixes.tsv](../../../quants/recipes/flint-mixes.tsv)).
  - Mixes at the bar already put bytes where Q36–Q77's per-group sweeps found them cheapest.
  - What SLQ adds is a second target: a mix within noise of Q8_0 (v2.0's headline file) at fewer bytes. On the P106,
    decode time follows bytes for every format of 4 bits and up, so such a file would be faster than Q8_0.
- **Ornith-1.5-122B** must hold ~2.73–2.87 bits per routed-expert weight to fit 45.4 GB on the cards
  ([pascal_lowbit_format.md](pascal_lowbit_format.md)).
  - That is below every SLQ result (lowest: 3.30 bpp dense, 3.73 MoE).
  - SLQ's 2- and 3-bit grids are uniform with one scale per 128 weights. The low-bit proxy measured uniform levels
    11–24 % worse than non-uniform ones at equal size, even with finer scales.
  - So SLQ's formats do not suit Ornith. Its allocation problem does: minimum damage at a fixed byte budget, which is
    exactly the bind the 122B is in.

## 5. Verdict and next steps

**Verdict: an idea worth borrowing.**

| Part of SLQ | Verdict | Why |
|---|---|---|
| The Kaitchup GGUFs | usable as is on Pascal (EAR98) | standard GGUF, 99 % in formats with Pascal kernels; but Qwen3.8-27B is not a v3 model |
| The vLLM/Humming checkpoints | not applicable | Humming needs compute 7.5+ |
| Conversion to GGUF | not worth it | exact only into Q4_1/Q5_1/Q8_0, +27–53 % bytes over a GGUF searched directly (EAR98) |
| SLQ's integer grids | not applicable | not better per byte than K-quants near 0.99 (section 4.1); uniform g128 grids are weak at Ornith's 2–3 bits |
| SLQ's code | not to run | transformers-only, no Flint model, skips Ornith's experts and DeltaNet projections, 2.6 TB layer store for the 122B |
| The allocator | **borrow** | per-group costs, a multiple-choice knapsack and a binary search on the budget replace hand-composed rounds of mixes, and fit Ornith's byte-bound problem |

### What to borrow, ranked

1. **Mix allocation by integer programming over measured costs, Ornith first.**
   - **Costs:** for each layer group and each format choice, the KLD change when that group alone moves to the other
     format. This is SLQ's "linear" estimate and what Q36 and Q77 measured by hand for Flint.
   - **Prices:** bytes, and decode milliseconds from [scripts/token_cost.py](../../../scripts/token_cost.py) for the
     dense tensors.
   - **Solver:** a multiple-choice knapsack (a few lines of dynamic programming; no ILP library needed at this size)
     gives the least predicted KLD at 45.4 GB on the cards, or the fewest bytes at a KLD target.
   - **Why it fits Ornith:** routed-expert time per weight hardly depends on N2P versus N3P, so the problem is purely
     where the 45.4 GB go.
2. **Shapley sampling only if the costs do not add up.**
   - Near Flint's bar they did: Q77 predicted K2 at 0.108 and measured 0.1068.
   - Ornith at ~2.9 bits is farther from its reference, so check additivity with one composed file before spending
     forward passes on random orders.
3. **A near-Q8_0 Flint file:** the same allocator with Flint's Q8_0 as the reference and Q5_0/Q6_K/Q8_0 as the choices.
   The gain depends on how much of Q8_0's KLD margin the cheaper groups can give up. Now E3, registry Q168 (section 8).
4. **Smaller ideas, not ranked:**
   - **EAR as a reported metric** next to KLD. It is easy to explain ("the BF16 model would accept 99 % of this file's
     tokens"), and it can be computed from the KLD base files without new runs of the model.
   - **The γ² law.** In GGUF terms, Q5_1 against Q5_0 in Q77's expensive groups. Miner Llama has no Pascal Q5_1
     kernel, and Q5_1 costs 0.5 bpw more, so test it only if the near-Q8_0 file (item 3) is pursued.

**Not borrowed:**
- SLQ's code, formats and checkpoints.
- Its default calibration text (FineWeb-Edu, ODC-By): this project's KLD bases and Unsloth's importance matrices
  already exist, and any new calibration source needs the publishable-data check first.

### E1: Ornith-1.5-122B allocation across 8-layer groups (recommended; registry Q167)

Builds on the low-bit work ([pascal_lowbit_format.md](pascal_lowbit_format.md)): queue row L3's file c, which uses N2P
for expert gate/up, N3P for down and Q5_0 for the dense tensors, at ~45.5 GB.

1. **Donor file d** (CPU, 1.5–3 h, like file c): the same recipe with the expert formats swapped (gate/up N3P, down
   N2P). With c, d and L3's optional file b (dense Q8_0), any per-group mix of the three choices can be composed.
2. **Costs:**
   - Groups: 6 groups of 8 layers × 3 classes (expert gate/up N2P → N3P, ≈ +0.9 GB per group; down N3P → N2P,
     ≈ −0.45 GB; dense Q5_0 → Q8_0, ≈ +0.3 GB).
   - That makes 18 files, each composed from c with one group switched (`scripts/compose_gguf.py`, run with a
     gguf-py that knows N2P/N3P, as it already needs one that knows Q4_1_64; see the coordination note below).
   - Each file is scored for KLD against O1's wikitext and agent-session bases on the two V100s.
   - Est. 10–15 min per file for both bases, plus composing (~45 GB written per file on hdd2, overlapping): est. 5–8 h.
3. **Solve** the knapsack at 45.4 GB on the cards (CPU, seconds). Compose the chosen mix and score it.
4. **Accept or iterate:**
   - Accept the prediction if the measured KLD is within 2 standard errors of it; otherwise sample a few random orders
     of group switches on the classes that interact (the Shapley step).
5. **Go/no-go:**
   - If the chosen mix beats file c by more than noise on both bases, run the agent check on the rig against file c.
     It is the same speed class, so this is a quality gate, not a speed race.
   - Otherwise keep the uniform N2P/N3P mix.

**Cost:** about 3 h of CPU, 5–8 h of the two V100s, ~45 GB of scratch at a time. No downloads.

**Waits on:**
- O1's two bases (running tonight) and file c's KLD.
- The V100s.
- N2P/N3P support in `compose_gguf.py`/gguf-py.

**Coordination:** the low-bit session owns the formats and file c; E1 changes only the mix across layers. It reuses
that session's files and tools and adds one quant (d).

### E2: does an EAR-driven GGUF search beat Unsloth's recipe on this project's test? (optional, needs downloads; registry Q180)

**Question:** KLD on wikitext-2 80 × 512 and on the agent-session text, against Unsloth's Qwen3.8-27B Q8_0 as the
base, for:
- Kaitchup EAR98 (18.9 GB) against UD-Q4_K_XL (17.6 GB);
- Kaitchup EAR98.5 (20.7 GB) against UD-Q5_K_XL (20.9 GB);
- the Q4_K_M already on the machine.

**What it answers:** whether SLQ-style search over donor tensors adds anything over Unsloth's own mixes near
near-lossless levels, on this project's test rather than on in-sample EAR.

**Cost:** ~1–1.5 h on two GPUs.

**Downloads (Apache-2.0, for the owner to approve):**

| File | Size |
|---|---|
| `kaitchup/Qwen3.8-27B-SLQ-GGUF`: EAR98 | 18.9 GB |
| `kaitchup/Qwen3.8-27B-SLQ-GGUF`: EAR98.5 | 20.7 GB |
| `unsloth/Qwen3.8-27B-GGUF`: Q8_0 | 29.0 GB |
| `unsloth/Qwen3.8-27B-GGUF`: UD-Q4_K_XL | 17.6 GB |
| `unsloth/Qwen3.8-27B-GGUF`: UD-Q5_K_XL | 20.9 GB |
| **Total** | **~107 GB** |

**Run it only if** Qwen3.8-27B joins the model list, or the owner wants outside evidence before E1. E1 answers the
question that matters for v3 directly.

## 6. Security observations

- **No text aimed at AI agents** anywhere I read: the SLQ repository (README, scripts, code, CITATION, issue #1), the
  paper's abstract and HTML, and the Kaitchup and hermitdave model cards. The READMEs carry no hidden HTML comments and
  no zero-width or bidirectional-control characters.
- **No install hooks or CI:**
  - The repository has no `setup.py`, `pyproject.toml`, `.github/` workflows, submodules or binaries; only 41 text
    files and 4 PNGs, in 3 commits.
  - `requirements.txt` pins its packages (torch 2.11.0, transformers 5.8.1, OR-Tools); the evaluation extras pin
    lm_eval 0.4.11 and vllm 0.21.0.
  - Humming is installed from source separately.
- **Network use:**
  - Only Hugging Face, through `datasets` (FineWeb-Edu, C4 at a pinned revision, wikitext, GSM8K, OpenThoughts) and
    `transformers`/`huggingface_hub` (models; `snapshot_download` in `eval/run_humming_speedup.py`).
  - No telemetry.
  - `scripts/push_to_hf.sh` uploads a checkpoint, only when run by hand with a token.
- **Defaults that run foreign code if the pipeline is ever used:**
  - `trust_remote_code=True` on every model, config and tokenizer load (`src/models/loading.py`, `slq_quantize.py`,
    `eval/run_humming_speedup.py`) executes Python shipped inside a model repository.
  - `torch.load(..., weights_only=False)` in `src/quant/qparams.py` unpickles; it is safe only on bundles you made
    yourself.
  - `_fix_tokenizer_config` rewrites a cached `tokenizer_config.json` in the Hugging Face cache.
  - `eval/run_lmeval.py` starts `lm_eval` through `subprocess`.
- **Model repositories:**
  - The Kaitchup repositories hold weights (safetensors or GGUF), configs, tokenizer files, SVG figures and a README
    with newsletter and donation links. There is no Python code, so `trust_remote_code` is not needed for them.
  - The SVGs I parsed contain no `<script>`, event handlers or external references.
  - The vLLM checkpoints' `serve.sh` pattern (`vllm serve … --quantization humming`) needs a non-stock vLLM build.
- **Not read:** the Humming repository's `AGENTS.md` and its code. The hermitdave repositories beyond one card.

## 7. Not verified

- **The paper's numbers** (bits, recovery, throughput, the Shapley cost) are the authors' own; I read them from the
  paper, not from a rerun.
- **Kaitchup's EAR figures** and the search behind them: the search is not public, and the EARs are in-sample.
- **Whether the paper's MoE run quantized Qwen3.5-35B-A3B's routed experts and DeltaNet projections**, and with what
  code: the published code would not.
- **The GGUF-against-vLLM byte comparison** (section 4.1) holds the EAR target fixed but not the search: different
  candidate pools, different runtimes for the EAR measurement.
- **The speed of the Kaitchup GGUF on the rig**: it was not run there.
- **E1's cost and gain**: estimates. Flint's Q77 showed per-byte costs spanning two orders of magnitude between groups,
  but Ornith's spread is unknown until step 2.

## 8. Where this stopped (2026-10-28)

Nothing of this work runs anywhere. Every result is in its registry row and data folder.

### Done

- **The research** (sections 1–7, Q179). SLQ is an allocator; its integer grids are not better per byte than
  K-quants; no SLQ model of this project's models exists.
- **E3, Q168: a Flint file near Q8_0.**
  - p065 reads 11.0 % fewer bytes per token than Unsloth's Q8_0, and its KLD is within noise of Q8_0 on wikitext and
    on agent text. Its same top token is 0.4–0.6 points lower, at the edge of noise.
  - It decodes **6.3 % faster than Q8_0 on the rig** (paired replay, 0.941 of Q8_0's ms per generated token,
    0.930–0.980).
  - The live agent check passed 12 / 12 at 47.1 tok/s.
  - p061 (−7.8 % bytes) is within noise on every measure.
- **E1, Q167: Ornith-1.5-122B's N2P/N3P mix per 8-layer group.** At file c's bytes, valid0 beats c and is level with
  UD-Q2_K_XL; valid1 (+0.91 GB) beats UD on all three texts. Ornith's late layers carry the cost, the reverse of
  Flint.

### Half-done

- **p065 is not released.** Its recipe is in [the Q168 data](../../data/2026-10-27_flint_near_q8/README.md)
  (`ITEMS="p065=q4,q5,kv4,kv5,o4,o5,sh4,sh5,gu5,head"`). The file (40,431,236,544 B, md5
  `8387b2e585a7b9cc10f40cb6b4531694`) exists only on the rig's model drive. Shipping it beside Q8_0 is the owner's
  decision.
- **valid1 went on to the rig through the low-bit work.**
  - With Q171's Q5_0 output layer it became Q212's file: 48.20 GB, better than UD-Q2_K_XL on all three texts.
  - On the rig (Q189, which covers Q251) it passed 12 / 12 at 14.5 tok/s against UD-Q2_K_XL's 11.3 with the IQ
    kernels, 1.28× decode.
  - valid0 exists only as a recipe: rebuilt from Q164's files c and d in ~3 min (`run/run.sh` in
    [the Q167 data](../../data/2026-10-27_ornith_alloc/README.md), `ITEMS="valid0=gu3,gu4,dn0,dn1,dn2,dn3"`).

### Next, in order

1. **Release decision on p065** (owner). If yes, publish it beside Flint-Pascal-Q5 with a model card that states the
   KLD, the rig ratio and the recipe. Then rerun the release agent check with p065 as the release file, to give the
   headline tok/s.
2. **Q248: a Q6_K rung for Flint.**
   - Make a Q6_K donor from BF16 (`llama-quantize --pure` with Unsloth's importance matrix, CPU).
   - Add `q6_*` switches to `near_q8.py` beside the Q5_0 ones: one more option per group, so the dynamic program
     takes three choices.
   - Score the early groups (gate/up 9–16, down 2–16, q 0–8) that Q5_0 hurt.
   - Kill criterion: no group moves to Q6_K for less than half of its Q5_0 cost.
3. **Q249: Ornith per layer in layers 16–47.** The same scripts with one-layer groups in that range: 2 × 32 switches,
   one KLD run each, est. ~10 h on two V100s. Kill criterion: if the best per-layer mix beats valid0 by less than one
   standard error at equal bytes, stop.
4. **Q250: an Ornith mix at UD's size** needs new downgrade switches (gate/up below N2P in layers 0–15, or dense in a
   4-bit format). Each new format is first a quantize of a donor file.
5. **Smaller:**
   - Q431: p065 with the Q8_0 head, to see whether the head explains its same-top-token drop.
   - Q432: the routers' share of agent-text KLD.
   - Q246: EAR beside KLD, from the existing base files.
   - Q180: needs ~107 GB of downloads approved.

### Open questions

- Why p065's same top token falls 0.4–0.6 points while its KLD does not move: is the Q5_0 head (+0.0023 KLD alone)
  the cause? Score p065 with the Q8_0 head (`p065h=q4,q5,kv4,kv5,o4,o5,sh4,sh5,gu5`).
- How the router type moves agent-text KLD by 0.227 at 8k (Q8_0's BF16 against F16, both near-exact): routing flips
  that compound with context, as Q21 found. A KLD at 8k against the same router type would separate the files from
  the routers.

### All items

| Item | Registry | State | Next step | Scripts and data |
|---|---|---|---|---|
| This note: SLQ read from source and paper, the published models' formats and bytes | Q179 | done | — | [2026-10-27_slq](../../data/2026-10-27_slq/README.md) |
| E3: a Flint file within noise of Q8_0 at fewer bytes | Q168 | done (rig 2026-10-28): p065 (−11.0 % bytes per token, KLD within noise of Q8_0 on wikitext and agent text) decodes 6.3 % faster than Q8_0 on the rig (paired replay 0.941, 0.930–0.980; sessions 0.963), live agent check 12 / 12 at 47.1 tok/s; p061 is the stricter fallback | a release decision (p065 as a "near-Q8_0" file beside Q8_0); a Q6_K rung (Q248) | [data](../../data/2026-10-27_flint_near_q8/README.md) (`near_q8.py`, `score.sh`) |
| E1: Ornith-1.5-122B's N2P/N3P mix per 8-layer group | Q167 | done (2026-10-28 07:16): at file c's bytes, valid0 beats c by 6.7 / 3.9 / 10 % KLD (wiki / agent / code) and is level with UD-Q2_K_XL at +3.05 GB; +0.91 GB (valid1) beats UD on all three; the late layers carry the cost. valid1 with a Q5_0 head (Q212) ran on the rig as Q189: 12 / 12, 1.28× UD's decode | finer groups (per layer) in the late half (Q249) | [data](../../data/2026-10-27_ornith_alloc/README.md) (`alloc.py`, `score.sh`) |
| E2: The Kaitchup's GGUFs against Unsloth's on this project's test | Q180 | planned, optional | the owner's approval of ~107 GB of downloads (section 5) | — |
| The other ideas from this note and from E1/E3 | Q245–Q251, Q431, Q432 | planned or parked (Q251 done as Q189) | Shapley costs (parked), EAR beside KLD, Q5_1 (parked), a Q6_K rung for Flint, Ornith per layer, an Ornith mix at UD's size, p065 with the Q8_0 head, the routers' share of agent-text KLD | the registry rows |

How to carry on:
- Each experiment's `run.sh` drives the quality machine over SSH and takes every host and path from the environment
  (its header lists them).
- Its state lives in `WORK` on that machine: `run/` (the scripts as pushed), `logs/` (every compose and
  llama-perplexity log) and the TSVs. Composed files are deleted once scored.
- A stopped run resumes by starting `score` again; items whose log holds a result are skipped. `collect` rebuilds the
  tables from the logs.
- Results land in the experiment's `docs/data/` folder with a README and a row in the data index, and in the registry
  row.

## Sources

- **SLQ code:** https://github.com/IST-DASLab/SLQ at `eb9f1f294c99940029465cc970862ba36951d0cc` (files cited by path
  above; `https://github.com/IST-DASLab/SLQ/blob/eb9f1f294c99940029465cc970862ba36951d0cc/<path>`); issue
  https://github.com/IST-DASLab/SLQ/issues/1.
- **Paper:** Helcig, Kurtic, Alistarh, "Statistically-Lossless Quantization of Large Language Models", COLM 2026,
  https://arxiv.org/abs/2605.02404 (v2, CC BY 4.0), https://openreview.net/forum?id=Xq1atP3a6S.
- **Humming:** https://github.com/inclusionAI/humming.
- **RCO:** https://github.com/IST-DASLab/RCO, https://arxiv.org/abs/2605.00649.
- **transformers 5.8.1:**
  https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
  and `.../qwen3_5/modeling_qwen3_5.py`.
- **Models:**
  - https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-GGUF
  - https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR98
  - https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR98.5
  - https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR99
  - https://huggingface.co/hermitdave/Qwen3-1.7B-MLX-slq-4.49bpw
  - https://huggingface.co/unsloth/Qwen3.8-27B-GGUF (file sizes)
- **This repository:**
  - Notes: [pascal_lowbit_format.md](pascal_lowbit_format.md), [gptq_body.md](gptq_body.md),
    [kquant_decode_pascal.md](kquant_decode_pascal.md), [dense_27b_speedups.md](dense_27b_speedups.md),
    [task_level_quality_checks.md](task_level_quality_checks.md).
  - Quants: [quants/experiments.md](../../../quants/experiments.md) (Q36, Q45, Q73, Q77),
    [flint-mixes.tsv](../../../quants/recipes/flint-mixes.tsv).
  - Benchmarks and data: [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md),
    [the data for this note](../../data/2026-10-27_slq/README.md).
