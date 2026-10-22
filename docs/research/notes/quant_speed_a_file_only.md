# Survey A: file-only levers for Flint decode speed on 9 × P106-090 (as of 2026-10-23)

**Scope.** Changes that can be expressed as a GGUF file which stock llama.cpp runs, or which needs only a small,
non-kernel llama.cpp patch. Engine reference: llama.cpp `d77256999` plus the CUDA-graphs patch #27721, read from the
local checkout a local llama.cpp checkout.

**Conventions.**
- **P / S.** P is a primary source (paper, PR, issue, source code, model card). S is a secondary source (blog, forum, Reddit).
- **"(my calc)"** marks this note's own arithmetic.
- **"(my measurement)"** marks numbers computed for this survey on the workstation (60 GB RAM, 32 threads). The inputs were:
  - the local Flint files: Unsloth Q8_0 and its imatrix (687 chunks), the makers' Q4_1, and our imatrix Q5_0, Q4_1 and UD-mirror v1;
  - a 12-chunk wikitext imatrix run with activation sums;
  - Flint's tokenizer, applied to 6.8 M tokens of English (wikitext-2 train) and code (llama.cpp's own C++, CUDA and Python sources).
- **"Claim to test"** marks anything not confirmed by a source or a measurement.
- **Nothing was run on the rig.** No repository file other than this one was changed.
- **Web research** was done by three helper agents. I re-checked the key PR and issue numbers and states with `gh`: #12557, #15550, #26584, #29600, #27976, #13037, #25187, #12718, #14891, #21652, transformers #49824, and ik #295 and #896.

---

## 1. Executive summary

1. **The ~470-launch rescale can be made unnecessary at file level, exactly.**
   - Why: overflow-scale activations occur only in the routed down-projections of layers 46–48 (section 3.0, item 3).
   - Fix (a): multiply each expert's `ffn_down_exps` block scales by 2^k and divide `ffn_up_exps` block scales by 2^k in those layers. SwiGLU is linear in the up branch, so this is exact, Q4-neutral and costs zero bytes. It edits only FP16 `d`/`m` fields, so it works on the makers' Q4_1 and on UD-mirror without requantization.
   - Fix (b): store those three tensors as Q8_0 (+8 MB/token).
   - Then turn off the rescale with the existing `LLAMA_NO_MOE_DOWN_RESCALE` patch.
   - Measured worth: +8 % (Q4_1), +11 % (Q5_0), +5 % (Q8_0).
   - Needs a NaN-free validation on the rig (experiment A1). Not relevant to Qwen3.6, which has no rescale.
2. **Q4_0's failure on Flint is not a missing-offset problem.**
   - Its error is the normal 1.4× of Q4_1 everywhere except the down-projections (and layer 48's attention) with dominant input channels, where it rises to 1.7–2.7×.
   - Hypothesis (claim to test, experiment A2): imatrix Q4_0 with layers 46–48 (and the shared-expert down of layer 12) at Q8_0 stops being "broken", but stays ~1.4× Q4_1's error.
   - Even a perfect Q4_0 saves only 10 % of Q4_1's bytes. A whole-model Q4_1 → Q4_0 swap is worth ≈ 2 ms (+6 %), short of the 15 % bar for a lossy lever.
3. **Better rounding cannot make Q4_1 match Q5_0**: Q5_0's weighted error is ~3× lower, and the best training-free rounders gain 10–30 %. The candidates:
   - compilade's draft llama.cpp PR #12557 cuts imatrix Q4_0 KLD by 28 % and Q5_0 by 13 % on dense models, and does nothing for Q4_1;
   - AutoRound writes true Q4_0/Q4_1/Q5_0 blocks and supports Qwen3.5/3.6-MoE. It probably supports Flint via transformers main (claim to test). Its 4-bit gains are at noise level in Intel's own tables, and a community AutoRound Q4_0 of Qwen3.8-27B (KLD 0.0266) equals bartowski's imatrix Q4_0 (0.0265);
   - no tool writes GPTQ, AWQ or HQQ results into Q4_0/Q4_1 for MoE models.
4. **No mix tool optimizes decode time; on this rig that objective differs from file size.**
   - Per MoE layer, attention and the six routed experts read the same 18.9 M params per token. Q5_0 costs 1.46× Q4_1 per parameter and Q8_0 1.70×, so Q5_0 is an overpriced middle step.
   - Unsloth, llama.cpp PR #15550 (`--target-bpw`, open) and Thireus' tool suite all optimize file size, which makes experts look 30× dearer than they are per token.
   - Recipe: per-group ΔKLD sensitivity × a per-format µs cost table, solved as a knapsack (experiment A3).
   - Expected (claim to test): UD-mirror quality at 38–41 ms (24–26 tok/s) instead of ~45 ms.
5. **Fused QKV is pure file surgery for Flint.**
   - The loader accepts `blk.N.attn_qkv.weight` through `create_tensor_qkv`. When q/k/v share a type (true in the makers' file and in every UD-mirror layer), the rows concatenate byte-for-byte.
   - Each MMVQ call quantizes its input separately, so the merge removes 2 matvecs + 2 quantizes per layer: 196 launches ≈ 1.0–1.4 ms (+3–4 %), exact.
6. **A merged `ffn_gate_up_exps` tensor is not a lever.**
   - Flint's loader doesn't accept it.
   - On sm_61, CUDA already fuses separate gate + up + SwiGLU into one MMVQ kernel. The merged path adds a separate GLU launch.
   - The same holds for Qwen3.6 (qwen35moe accepts the merged tensor).
7. **Shared expert as two always-on routed experts:** not expressible in a file, because the router normalization can't be bypassed. A small graph patch would save ~94–188 launches (+1.5–3 %).
8. **Router precision.**
   - F32 → F16/BF16 in the makers' file: −50 MB/token, ≈ 0.6 ms (+1.8 %), exact for practical purposes; gguf-py surgery only.
   - BF16 → Q8_0 saves 0.3 ms of bytes but adds a Q8_1 quantize launch per layer (+0.24–0.33 ms). It also quantizes a router input that has a 2,150-RMS channel in layer 46. Net ≈ 0, with routing risk: skip.
9. **Output head.**
   - It is 4.2 ms of a UD-mirror token at Q8_0. Q4_1 saves 1.75 ms (≈ 4 %); Q5_0 saves only 0.6 ms because of its slow format.
   - Vocabulary trimming is weak. Only ~15 k of 150 k tokens are non-Latin script (≈ 0.4 ms at Q8_0). A frequency-trimmed 50–65 k head misses 1.9–6.5 % of held-out English/code tokens (my measurement), and any trimming needs a patch.
10. **Rotations (QuaRot/SpinQuant R1, R2) fold into a Flint GGUF with no runtime op, but are low priority here.**
    - Flint's weights are already Gaussian in-block, and weight-only g32 rotation gains are small in the literature.
    - R1 cannot touch the down-projection input, where both the overflow and the worst Q4_0 penalty live.
    - Mainline's metadata-driven online Hadamard (`prism.hadamard.*`, #29600) needs a one-line arch allowlist patch for Flint and adds launches. It is no better than Q8_0 for the three affected layers.
11. **Structural cuts are not expressible in a Flint GGUF without code changes, and cost too much quality at a 15 % speed level.** The one exception is uniform query-head pruning, since `n_head` is global.
    - `--prune-layers` produces files that don't load (#27976).
    - Expert pruning (new: RAZOR, HOPE, OMP-MoE, OptEP) shrinks the file but not the bytes per token, so it gives no single-stream speed here.

---

## 2. Lever table

Times are per token: "−x ms" saves time against the stated baseline; "+y %" is the speed gain. Baselines are the makers' Q4_1 (35.7 ms with the rescale) or UD-mirror v1 (≈ 45 ms predicted).

| Lever | Class | Mechanism | Expected effect on single-stream decode (arithmetic) | Quality effect | Effort | Evidence | Carries over to Qwen3.6? |
|---|---|---|---|---|---|---|---|
| Overflow-proof file + rescale off | A (file + existing env patch) | 2^k rebalance of up/down expert scales in layers 46–48, or Q8_0 down there | −470 launches: measured −2.4 to −2.8 ms → **+8 % (Q4_1), +11 % (Q5_0)**; Q8_0 option costs +0.11 ms | None (exact) if no block sum exceeds 65,504 | 0.5–1 day | Rig speed measured; activation map measured off-rig; peaks not yet measured | No (qwen35moe has no rescale) |
| Fused QKV | A (file) | `attn_qkv` = row concat of q/k/v | −196 launches × 5–7 µs = **−1.0 to −1.4 ms (+3–4 %)** | None (bit-identical) | 0.5 day | Loader code read; launch count inferred | Partly: only 10 attention layers (≈ −40 launches) |
| Speed-aware mix | A (file) | Choose per-group format to minimise ΔKLD under a µs budget | UD-mirror 45 ms → 38–41 ms (my target) at equal KLD: **+10–18 %** | Equal by construction (must verify) | 2–4 days incl. sensitivity scan | Cost model fitted; sensitivity not measured | Yes; also keep `_shexp` and `_exps` the same type there so CUDA's routed+shared fusion stays on (claim to test) |
| Router F32 → F16/BF16 (makers' file) | A (file) | 99 → 49 MB/token | 24.6 M params × 25.7 µs/M = **−0.63 ms (+1.8 %)** | ≈ None | 1 h | Arithmetic only | Already BF16 in Unsloth files |
| Router → Q8_0 | A + patch (#26584) | 49 → 26 MB/token | −0.30 ms bytes, +47 quantize launches (+0.24–0.33 ms) → **≈ 0** | Routing-flip risk (massive input channel in layers 46/48) | 1 day | ik #896 found Q8_0 = F32 on Qwen3-30B (PPL) | Same ≈ 0 |
| Head Q8_0 → Q4_1 | A | 327 → 192 MB | 307.5 M × (13.7 − 8.0) µs/M = **−1.75 ms (≈ +4 %)** | Unknown (simulated weighted error 0.54 %); measure | 2 h | Arithmetic; no isolated head KLD data found | Bigger: 248k vocab (head ≈ 508 M params, −2.9 ms) |
| Head Q8_0 → Q5_0 | A | 327 → 206 MB, slow format | −0.6 ms (≈ +1.4 %) | Smaller loss | 2 h | Arithmetic | Yes |
| Vocabulary trim (script-based, −15 k rows) | A + patch (index-mapped head) | Fewer head rows | −31 M params → −0.43 ms at Q8_0 (≈ 1 %) | ≈ None for Latin-script users | 1–2 days | My vocab classification | Larger vocab, so larger gain |
| Vocabulary trim (frequency, 65 k) | A + patch | Head −57 % | −2.4 ms at Q8_0 (≈ 5 %) | **Held-out miss 1.9 % (en) / 5.5 % (code)**: model forced off-distribution | 1–2 days | My coverage measurement | Same risk |
| Q4_0 replaces Q4_1 (after protecting layers 46–48) | A | −10 % bytes on those tensors | Whole-model Q4_1 bytes 1.557 → 1.401 GB: −2.0 ms (+6 %) if η(Q4_0) = η(Q4_1) | ~1.4× Q4_1's weighted error; expected KLD between Q4_1's 0.28 and 0.4 (claim to test) | 1 day | My simulation; Q4_0 KLD measured broken | Qwen3.6 Q4_0 already works |
| Better rounding: PR #12557 | A (quantizer patch only) | Cumulative-search weighted rounding for Q4_0/Q5_0/IQ4 | 0 directly; enables cheaper formats | −28 % KLD (Q4_0), −13 % (Q5_0) on dense 3–8B; Q4_1 unchanged | 0.5 day | PR data (dense models) | Yes |
| Better rounding: AutoRound | A | SignRound tuned against block outputs → exact Q4_0/Q4_1/Q5_0 blocks | 0 directly | ~noise at 4 bits in Intel's tables; Qwen3.8-27B Q4_0 = imatrix Q4_0 | 1–2 days (Flint BF16 80 GB vs 94 GB RAM) | P docs + community card | Yes: ready-made AutoRound Q4_0/Q4_1/Q5_0 of Qwen3.6 exist |
| Rotation R1/R2 folded offline | A | Orthogonal residual / per-head V-O rotations | 0 speed | Small ± at g32 (OptRot/QuaRot tables at per-channel or g256; MR-GPTQ: INT4 RTN 77.12 → 77.16) | 2–3 days | Literature only | Yes; Prism path already allowed for qwen35moe |
| Online Hadamard on down inputs (Prism) | A + 1-line patch | Metadata-declared WHT before `ffn_down_exps` | +2 launches/layer → +0.5–0.65 ms; could replace the rescale (−2.6 ms) only if the block sums stay safe | Helps Q4 down-projections with dominant channels | 2–3 days | Code read; safety argument disputed (section 3) | Allowed already |
| Shared expert as 2 routed experts | A + graph patch | One `MUL_MAT_ID` with 8 slots | −94 to −188 launches → −0.5 to −1.3 ms (+1.5–3 %) | Exact | 2 days | Arithmetic | Has the CUDA routed+shared fusion already |
| Fused `ffn_gate_up_exps` | — | Merged tensor | **≥ 0 ms loss** (GLU no longer fused) | — | — | CUDA fusion code read | Same |
| Query-head pruning 32 → 28 | A (file) | Drop one q head per KV group, edit `head_count`, sinks | q/o −12.5 %: −103 M params → −1.4 ms (Q8_0) / −0.8 ms (Q4_1) | Likely beyond the bar without healing (CFSP: GQA head pruning degrades) | 1 day | Literature thin | Only 10 attention layers |
| Attention-block drop (8 of 49) | A + patch | Skip attention in chosen layers | 8 × (0.26 ms bytes + 14 launches) ≈ −2.7 to −2.9 ms (+6–7 %) | Dense-model data: Mistral-7B −1.7 MMLU for 8/32 dropped; no MoE/SWA data | 1–2 days | Literature (dense) | Fewer attention layers |
| Layer drop | A + gguf fix | Remove whole blocks | ≈ −0.8 ms per layer: 15 % needs ~7 layers | SLEB: −10 % layers → PPL +27 %; Qwen family collapses near 20 % (Gromov) | 1 day | Literature | Similar |
| Dynamic top-k with k₂ normalization | A + graph patch | Run 4–5 experts, normalise over the top 6 | Measured on rig: 6 → 5 +4 %, 6 → 4 +9 % (served) | Literature: −0.35 MMLU instead of −4.65 (softmax router) | 1 day | P paper; sigmoid untested | Yes (paper is on Qwen3.6) |
| Expert pruning/merging | A | Fewer experts in file | **0** (bytes/token unchanged) | — | — | Arithmetic | Same |

---

## 3. Sections per lever

### 3.0 Cost model and measurements shared by all levers

#### Decode-time model calibrated on Flint itself (my calc)

There are three rig measurements of whole Flint files (brief; `docs/benchmarks/2026-10-22-flint.md`). Their GPU bytes
per token come from my header read of the local files; routed-expert tensors are counted at 6/256 and the embedding is
excluded.

| File (cards) | GPU bytes / token | tg128 with rescale | tg128 without rescale |
|---|---|---|---|
| Makers' Q4_1 (5) | 1.656 GB: Q4_1 1.557 + F32 router 0.099 | 28.05 tok/s = 35.65 ms | 30.31 = 32.99 ms |
| imatrix Q5_0 (5) | 1.881 GB: Q5_0/Q5_1 1.505 + Q8_0 head 0.327 + BF16 router 0.049 | 21.33–22.3 = 44.8–46.9 ms | 23.76 = 42.09 ms |
| Unsloth Q8_0 (9) | 2.696 GB: Q8_0 2.646 + BF16 router 0.049 | 20.49 = 48.80 ms | 21.57 = 46.36 ms |

Fitting `t = F + Σ bytes/η_format` to the runs without the rescale, assuming Q4_1, Q8_0 and the float router share one η:
- **η(Q4_1) ≈ η(Q8_0) ≈ 78 GB/s and F ≈ 11.7 ms.** From η = (2.696 − 1.656) GB / (46.36 − 32.99) ms.
- **η(Q5_0) ≈ 59 GB/s.** Q5_0 is about 25 % less efficient per byte, because of the fifth-bit unpacking.
- **The rescale costs 2.4–2.8 ms in every file.** That is ≈ 470 launches × 5–6 µs, consistent with the brief.

**Cost per million parameters read per token** (this is the "per-format cost" in the brief's objective; my calc):

| Format | Bytes per weight | µs per M params | Relative to Q4_1 |
|---|---|---|---|
| Q4_0 | 0.5625 | ~7.2 (η assumed equal to Q4_1; unmeasured on Flint) | 0.9 |
| Q4_1 | 0.625 | 8.0 | 1.0 |
| Q5_0 | 0.6875 | 11.7 | 1.46 |
| Q8_0 | 1.0625 | 13.7 | 1.70 |
| BF16 / F16 | 2 | 25.7 | 3.2 |
| F32 | 4 | 51.4 | 6.4 |

**Parameters read per token, per MoE layer** (my calc from the config):

| Part | Params | Q4_1 time | Q8_0 time |
|---|---|---|---|
| Attention q, o, k, v | 18.87 M | 151 µs | 259 µs |
| Six routed experts (3 × 6 × 1.049 M) | **18.87 M** | 151 µs | 259 µs |
| Shared expert | 6.29 M | 50 µs | 86 µs |
| Router | 0.52 M | (F32: 27 µs; BF16: 13 µs) | — |

The head is 307.5 M params (2.46 ms at Q4_1, 4.21 ms at Q8_0). The two dense layers are 75.5 M.

**Prediction for UD-mirror v1** (my calc, claim to test):
- Its bytes per token: 2.259 GB, of which Q8_0 1.490, Q5_0 0.424, Q4_1 0.295, BF16 router 0.049.
- Predicted time: 23.6 ms (Q8_0 + Q4_1 + router) + 7.2 ms (Q5_0) + 14.3 ms fixed ≈ **45 ms, ≈ 22.2 tok/s tg128**.
- So it would sit at the imatrix Q5_0's speed with UD-Q4_K_XL's quality.

Two caveats:
- The fit rests on three files and one shared η. A per-format micro-benchmark on the rig would replace it (experiment A3).
- It ignores that the routed-expert matvecs are partly latency-bound (flint_speedups.md section 1).

#### Weight and activation facts I measured

**1. Flint's weights have no block offsets and are Gaussian inside every block** (my measurement).
- Sample: every tensor kind in layers 0, 3, 12, 30, 45, 46, 47 and 48 (8 sampled experts per expert tensor), plus the head and the embedding.
- In 32-weight blocks, |block mean| / block std is 0.144–0.150 for every tensor. That is exactly what i.i.d. Gaussian samples give (0.148 in a synthetic control).
- No block is one-sided. Row means are 0.01–0.035 std.
- Mean in-block sample kurtosis is 2.82–3.23 (the Gaussian expectation for n = 32 is 2.82).
- **"Q4_0 breaks Flint because it has no offset" is therefore not a property of the weights.**

**2. Simulated imatrix-weighted rounding error.**
- Method: a numpy re-implementation of `ggml-quants.c`: `make_qx_quants` for Q4_0/Q5_0 and `make_qkx3_quants` for Q4_1, with the `qw·sqrt(σ²+x²)` weights and Unsloth's imatrix. The error metric is Σ imatrix·Δw² / Σ imatrix·w².
- **Most tensors look like Gaussian noise.** Relative errors: Q4_0 0.0062–0.0070, Q4_1 0.0043–0.0050, Q5_0 0.0015–0.0017. The ratio Q4_0/Q4_1 is **1.37–1.45**, the same as for synthetic Gaussian weights (1.37).
- **Exceptions, all down-projections (and layer 48's attention) whose inputs have a few dominant channels:**

  | Tensor | Q4_0 / Q4_1 | Q4_0, Q4_1, Q5_0 error |
  |---|---|---|
  | `blk.48.ffn_down_shexp` | **2.69** | 0.029, 0.011, 0.0095 |
  | `blk.48.attn_output` | **2.45** | — |
  | `blk.12.ffn_down_shexp` | 2.20 | — |
  | `blk.47.ffn_down_shexp` | 2.07 | — |
  | `blk.48.attn_v` / `attn_q` | 1.81 / 1.68 | — |
  | `blk.46.ffn_down_shexp` | 1.70 | — |
  | `blk.48.ffn_down_exps` | 1.66 | — |

- **Synthetic control:** giving one channel 10⁴× the imatrix weight raises the ratio from 1.37 to 2.1 (Gaussian weights) or 3.0 (Laplace weights).
- **Mechanism:** with a dominant input channel, Q4_1's second parameter (the offset) lets the weighted fit hit the dominant weight exactly; Q4_0's single scale cannot.

**3. Flint's down-projection inputs have huge activations in the last three layers, and only there** (my measurement, from Unsloth's imatrix `in_sum2/counts`).

| Layer | Routed `ffn_down_exps` input: overall RMS / largest channel RMS | Shared-expert down input: overall / largest channel RMS |
|---|---|---|
| 2–11 | 0.04–0.14 / ≤ 1.8 | — |
| 12 | 0.38 / 73 | — |
| 18 | 0.57 / 60 | — |
| 30, 37, 45 | 1.3–3.4 / 174–302 | 45: 8.2 / 80 |
| **46** | **631 / 25,600** | 35 / 202 |
| **47** | **13 / 1,080** | — |
| **48** | **124 / 48,400** | 73 / 607 |

- The normed residual feeding gate/up and the router also has a dominant channel in layers 46 and 48: channel RMS 2,150 and 1,210, against an overall 24 and 15.
- A Q8_1 block sum overflows FP16 when it exceeds 65,504, and a single channel of 10⁵ is enough.
- **Inference:** the NaNs that the rescale prevents come from layers 46–48. Every other layer has ≥ 200× headroom on RMS. Peaks were not measured; experiment A1 measures them.

**4. The activations are not common-mode** (my measurement).
- Index: (Σ_block E[x])² / Σ_block E[x²], from a 12-chunk imatrix run with activation sums. A value of 0 means no common mode; 32 means fully common.
- Result: 0.05–0.5 for most inputs and 0.6–0.95 for the down inputs of layers 46–48. Single dominant channels, not shared-sign blocks, drive both the overflow and the Q4_0 penalty.

**5. The makers' Q4_1 is not RTN of BF16** (my measurement).
- Its weighted distance from the Q8_0 weights is 0.0073–0.0080 per tensor: more than imatrix RTN Q4_1 (0.0043–0.0050), and about the same as plain Q4_0 RTN.
- In layer 48 it reaches 0.050 on `ffn_down_shexp`.
- This confirms the codes moved during distillation. Mixing its tensors with BF16-derived ones (e.g. "makers' Q4_1 with Q8_0 down-projections") replaces trained weights with untrained ones, so it needs a task check.


### 3.1 Better rounding into Q4_0 / Q4_1 / Q5_0 / Q8_0

**Takeaway.**
- **Rounding is not the bottleneck for Flint.** Its weights are Gaussian in-block and llama.cpp's imatrix quantizer already gets the Gaussian-optimal ratios.
- **Q4_0 is broken by a few tensors whose inputs have dominant channels** (section 3.0); protecting those is a mix decision, not a rounding one.
- **The best training-free rounders reduce 4-bit error by 10–30 %.** That neither closes Q4_1 → Q5_0 (3×) nor makes Q4_0 a 15 % speed lever.

**How llama.cpp rounds legacy types today** (P, source: `ggml/src/ggml-quants.c` at `d77256999`)
- **Q4_0 with imatrix** (`quantize_row_q4_0_impl`):
  - weights are `qw[j]·sqrt(σ²_row + x_j²)`;
  - `make_qx_quants(32, nmax=8, rmse_type=1)` tries 19 inverse scales around −8/max (±0.9 steps), each followed by a weighted least-squares refit of the single scale;
  - rounding is independent per weight: no error feedback, so correlations between inputs are invisible.
- **Q5_0** is the same with nmax = 16.
- **Q4_1 / Q5_1** use `make_qkx3_quants`: 37 steps with a joint least-squares fit of scale and min, with the min clamped to ≤ 0.
- **Without an imatrix**, the `_ref` functions are plain round-to-nearest.
- **The mixture logic protects the first n_layer/8 down-projections.** When the target type is Q4_0 or Q5_0 with an imatrix, it moves them to Q4_1 or Q5_1 (`src/llama-quant.cpp` ~line 688, "Guard against craziness in the first few ffn_down layers"). For Flint that protects layers 0–5, while the damage is in layers 12 and 46–48.

**Cited findings**
- imatrix support for legacy types ([llama.cpp PR #4969](https://github.com/ggml-org/llama.cpp/pull/4969), merged 2024-01-16; P): the imatrix lowered quantization error to 0.51–0.88× for Q4_0 but 0.31–0.55× for Q4_1. Q4_1's two parameters exploit importance weights better, consistent with section 3.0's mechanism.
- compilade's cumulative-search rounding ([PR #12557](https://github.com/ggml-org/llama.cpp/pull/12557), open draft, created 2025-03-25, last updated 2025-08-18; P, checked with `gh`):
  - imatrix KLD 0.0480 → 0.0343 for Q4_0 and 0.0101 → 0.0088 for Q5_0;
  - Q4_1 is not touched (offset types listed as unsolved).
- ik_llama.cpp's scale search ([ik PR #295](https://github.com/ikawrakow/ik_llama.cpp/pull/295), merged 2025-03-29; P): pure Q4_0 PPL on Llama-3-8B 7.7039 → 7.6132, on DeepSeek-V2-Lite 6.9684 → 6.9286. [ik PR #302](https://github.com/ikawrakow/ik_llama.cpp/pull/302) (2025-04-01) barely moves Q4_1 (7.5309 → 7.5277).
- Intel AutoRound ([github.com/intel/auto-round](https://github.com/intel/auto-round), v0.16.1 released 2026-10-16; P, from helper-agent reading of `schemes.py`, `export_to_gguf/`):
  - Exports `gguf:q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0` and K-quants.
  - With `iters > 0`, SignRound learns per-weight up/down rounding and min/max scale factors against block-output MSE, then packs the exact llama.cpp block layout. It verifies the packed tensor by dequantizing a sample.
  - `iters=0` ("opt-RTN") is an imatrix-weighted scale search like llama.cpp's.
  - The `gguf:q4_0` preset puts the head in Q6_K, a slow format here; override it.
  - MoE: unfuses transformers' 3-D experts per expert, packs each and stacks them. Has Qwen3-MoE, Qwen3.5-MoE and Qwen3-Next converters and an env switch to route calibration through all experts.
  - Compute: `docs/moe.md` (2026-09-11) gives Qwen3.6-35B-A3B on an A100 at 28 GB VRAM, 30 GB RAM, ~60 min. `low_gpu_mem_usage` trades ~20 GB of VRAM for 30–100 % more time.
  - At 4 bits its own GGUF table shows noise-level gains (Qwen3-30B-A3B q4_k_s 0.6762 opt-RTN vs 0.6777 tuned).
- AutoRound's author recommends `iters=0` for GGUF, and ikawrakow objects that tuning against block outputs overfits the calibration set ([ik discussion #657](https://github.com/ikawrakow/ik_llama.cpp/discussions/657), 2025-07-30; P).
- Flint in transformers ([huggingface/transformers PR #49824](https://github.com/huggingface/transformers/pull/49824), merged 2026-10-21; P, checked):
  - not in release v5.20.2, so AutoRound needs transformers from git;
  - the experts use the fused `@use_experts_implementation` path, so AutoRound's generic unfusing should apply (claim to test);
  - attention with sinks runs in eager mode during calibration (slower).
- Community AutoRound Q4_0 of Qwen3.8-27B (1000 iters, 512 code-heavy samples): KLD 0.0266, PPL 7.067 vs BF16 6.950, at 14.64 GB ([tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF), card updated 2026-09-20; S).
  - bartowski's imatrix Q4_0 of the same model: KLD 0.02652 ([bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF); cited in `dense_27b_speedups.md` section 3).
  - Different text and reference runs, so this is indicative only.
- Ready-made AutoRound (`--enable_alg_ext`) Q4_0, Q4_1 and Q5_0 files of the secondary model exist, without quality numbers ([sphaela/Qwen3.6-35B-A3B-AutoRound-GGUF](https://huggingface.co/sphaela/Qwen3.6-35B-A3B-AutoRound-GGUF), created 2026-04-18, updated 2026-07-01; P, file list checked).
- GPTQ / AWQ / others into legacy blocks (P, helper-agent reading):
  - llama.cpp's converters ([#14810](https://github.com/ggml-org/llama.cpp/pull/14810), merged 2025-10-23; [#17069](https://github.com/ggml-org/llama.cpp/pull/17069), merged 2025-11-07) dequantize GPTQ and compressed-tensors to float, so `llama-quantize` re-rounds and the method's rounding is lost.
  - [IST-DASLab/gptq-gguf-toolkit](https://github.com/IST-DASLab/gptq-gguf-toolkit) (2025-09) does GPTQ straight to K-quants. It has a lossless repack of symmetric no-act-order 4-bit GPTQ into Q4_0 (it assumes g128 scales). Dense Llama-style models only.
  - AMD Quark's `export_gguf` ([docs](https://quark.docs.amd.com/latest/pytorch/export/gguf_llamacpp.html), Quark 0.13, 2026-09-28) writes AWQ results into Q4_0/Q4_1/Q8_0 blocks: Llama-2-7B Q4_1 PPL 5.9994 (llama.cpp) → 5.8952 (Quark AWQ), float 5.7964. Llama only; no MoE.
  - llama.cpp's old AWQ path ([#4593](https://github.com/ggml-org/llama.cpp/pull/4593), merged 2023-12-27, removed 2024) improved Q4_0/Q4_1 PPL by 0.02–0.08 on 7B models without an imatrix.
  - Not found: any HQQ, OmniQuant, SpQR, EfficientQAT or GPTQ-v2 path into GGUF Q4_0/Q4_1, any MoE-capable GPTQ → Q4_x writer, or any llama.cpp PR adding error feedback (GPTQ/OBQ) to `llama-quantize`.
- Precedent for grid mismatch: the known way Q4_0 fails is a QAT grid mismatch: Kimi int4 [−7, 7] ([ik PR #1677](https://github.com/ikawrakow/ik_llama.cpp/pull/1677), `--symmetric-q40`, merged 2026-04-23) and Gemma QAT Q4_0 ([#24603](https://github.com/ggml-org/llama.cpp/pull/24603), closed 2026-06-14). Flint's BF16 release is not on a grid (section 3.0, item 5), so that explanation does not apply here.
- Flint's makers' INT4 export is affine group-64 with "a free per-group bias", which AWQ's integer zero-points cannot express ([mattbucci/Flint-AWQ](https://huggingface.co/mattbucci/Flint-AWQ), 2026-10-22; S). It maps onto Q4_1 exactly, which is why the makers ship Q4_1.

**Inferences**
- **Q4_0 is a weak speed lever even if fixed (my calc).** Q4_0 saves 0.0625 bytes per weight against Q4_1. On the makers' layout that is 156 MB per token, ≈ 2.0 ms (+6 %), assuming Q4_0 decodes as efficiently per byte as Q4_1. It probably doesn't: Q4_0 blocks are only 2-byte aligned (`get_int_b2`), Q4_1's are 4-byte aligned.
- **The quality path:** the error ratio of 1.4 means Q4_0 belongs only where a tensor group is insensitive. That is an allocation question (section 3.3).
- **Q4_1 vs Q5_0.** Simulated imatrix errors are 0.0043–0.0050 for Q4_1 and 0.0015–0.0017 for Q5_0, close to 3×. The KLD gap is 0.280 vs 0.177 measured. A 10–30 % rounding gain moves Q4_1 to roughly 0.23–0.26 (claim to test), not to 0.177.
- **What rounding could still buy.** Error-feedback rounding sees input correlations and actual block outputs, which the diagonal imatrix cannot. That matters most for the dominant-channel tensors in layers 12 and 46–48. Those cost ~3 MB per layer to keep at Q8_0, so tuning them is not worth a day of AutoRound work.

**Gaps**
- No Flint data for any tuned rounding (AutoRound, GPTQ, #12557).
- No KLD comparison of the sphaela AutoRound files against imatrix files for Qwen3.6.
- The activation peaks (as opposed to the RMS) per layer are unknown.

**Experiment ideas**
- **A2. Q4_0 bisection (workstation or quant-machine CPU, ~3 h; needs ~23 GB free disk per file).**
  - From the Unsloth Q8_0 with its imatrix, make imatrix Q4_0 variants with the head at Q8_0:
    - (i) `--tensor-type 'blk\.(46|47|48)\.=q8_0'`;
    - (ii) the same plus `'blk\.12\.ffn_down_shexp=q8_0'`;
    - (iii) only `blk.48.(attn_output|ffn_down_shexp|ffn_down_exps)` at Q8_0.
  - Also make imatrix Q4_1 with (i), to see what protecting the tail buys Q4_1 (now 0.280).
  - Run `scripts/quality_kld.sh` against the existing Q8_0 base (80 × 512); ~12–15 min each.
  - Expected: (i) brings Q4_0 from KLD 0.566 to roughly 0.30–0.40 and PPL from +45 % to +6–9 %; Q4_1 + (i) improves on 0.280 by a few percent. Claims to test.
  - Disk: the workstation had 49 GB free at writing and other jobs were running, so I did not run this myself.
- **A8. PR #12557 cherry-pick (quant machine, ~2 h).** Apply it to the quantize build only (no runtime change); make imatrix Q4_0 + (i) and imatrix Q5_0; KLD each. Expected: −10 to −25 % KLD.
- **A7. AutoRound carry-over check on Qwen3.6 (quant machine, ~2 h + download).** KLD of the sphaela AutoRound Q4_0/Q4_1/Q5_0 against bartowski/Unsloth imatrix files of the same type, against a Q8_0 reference. Only if AutoRound wins by ≥ 15 % KLD, try `auto-round --format gguf:q4_1 --iters 200` on Flint:
  - transformers from git;
  - `LLAMA_CPP_ROOT` pointing at a checkout with #30928;
  - `low_gpu_mem_usage`, on one 24 GB 3090;
  - 80 GB BF16 against 94 GB RAM is tight, so load per block or use V100s with CPU offload.

### 3.2 Rotations and incoherence processing folded into the weights

**Takeaway.**
- **What folds offline.** R1 (the residual-stream rotation, with all RMSNorm gammas folded into the readers and the final norm into the head) and R2 (per-head V/O) fold into a Flint GGUF with no runtime op. Nothing in Flint's architecture blocks them:
  - q/k-norm acts after the projections;
  - sinks are per-head scalars;
  - the sigmoid router is just another residual reader with gamma and R1 folded in, so its logits are unchanged;
  - 4-KV GQA and the untied head are fine.
- **They are unlikely to pay off on Flint, for four reasons:**
  - its weights are already Gaussian in-block;
  - weight-only rotation at group size 32 gives little;
  - R1 leaves the SwiGLU output that feeds the down-projections unchanged, and that is exactly where Flint's overflow and its worst Q4_0 penalty are;
  - after gamma folding, the reader inputs are bounded by √2048 ≈ 45, so R1 is not needed for overflow on those.
- **Only an online Hadamard (R4) touches the down inputs.**

**Cited findings**
- **QuaRot** ([arXiv 2404.00456](https://arxiv.org/abs/2404.00456), 2024-04; P): fuses the residual rotation and per-head V/O Hadamards into the weights, but needs online Hadamards before `down_proj` and `out_proj`. Weight-only ablation (per-column, no groups): RTN PPL on Llama-2 7B 6.99 → 6.76.
- **SpinQuant** ([arXiv 2405.16406](https://arxiv.org/abs/2405.16406); P): R1/R2 are absorbed offline; R3/R4 "cannot be absorbed".
- **OptRot** ([arXiv 2512.24124](https://arxiv.org/abs/2512.24124), 2025-12-30; P): the cleanest offline-only, weight-only data.
  - 4-bit GPTQ g256 with no R3/R4, Llama-3.1-8B: 7.47 unrotated → 6.81 (QuaRot) / 6.71 (OptRot).
  - 4-bit RTN: 8.23 → 7.57 / 7.40.
  - It works on Qwen3 with q/k-norm (GPTQ Qwen3-8B 11.62 → 10.12).
  - These are g256 or per-channel results; g32 shrinks the gain.
- **MR-GPTQ** ([arXiv 2509.23202](https://arxiv.org/abs/2509.23202), ICLR 2026; P): weight-only INT4 RTN 77.12 → 77.16 average accuracy with a Hadamard on Llama-3.1-8B-Instruct. Theory: small groups neutralize rotation.
- **ikawrakow** on rotating weights: "zero success" with Hadamard transforms for weight quantization ([ik PR #1033](https://github.com/ikawrakow/ik_llama.cpp/pull/1033), 2025-12-03; P).
- **AMD signed-symmetric study** ([arXiv 2607.08779](https://arxiv.org/abs/2607.08779), 2026-06-12; P):
  - the only rotated-GGUF study found, with Hadamards fused offline because llama.cpp's graph is fixed;
  - Llama-3-8B after rotation: Q4_0 6.7 vs Q4_1 6.6 PPL (BF16 6.5);
  - no unrotated Q4_1 control, and their "Q4_0" baseline uses a positive absmax scale, unlike ggml's signed `max/−8`.
- **Tooling:** llm-compressor `SpinQuantConverter` ([PR #3262](https://github.com/vllm-project/llm-compressor/pull/3262), merged 2026-10-13; P).
  - Model-free R1/R2 on safetensors; folds the norms; default regexes include MoE routers and 3-D fused experts.
  - Errors on unmapped hidden-sized tensors; casts back to BF16, which adds a rounding step.
  - Flint's HF tensor names would need a custom mapping (claim to test).
- **Hadamard in mainline:**
  - [#21038](https://github.com/ggml-org/llama.cpp/pull/21038) (2026-04-01) rotates Q/K/V for quantized KV caches.
  - [#23615](https://github.com/ggml-org/llama.cpp/pull/23615) (2026-05-25) is the CUDA fast WHT.
  - [#29600](https://github.com/ggml-org/llama.cpp/pull/29600) (merged 2026-10-11, checked) lets GGUF metadata `prism.hadamard.*` declare weights stored as W·D·H; the graph then applies the signs D (a `ggml_mul`) and H (a `mul_mat` with `GGML_HINT_SRC0_IS_HADAMARD`, CUDA `fwht.cu`) to their inputs.
  - Allowed kinds include `ffn_down_exps`, `ffn_down_shexp`, `attn_output` and the head.
  - The arch switch admits only llama, qwen3, qwen3moe, qwen35, qwen35moe and qwen3next (local source `src/llama-model.cpp` ~2128); Flint throws.
- **Massive activations and down inputs:**
  - The super weight lives in an early `down_proj`, whose input product creates a large activation ([arXiv 2411.07191](https://arxiv.org/abs/2411.07191); P).
  - GLU activation spikes concentrate in the FFN of early and late layers ([arXiv 2405.14428](https://arxiv.org/abs/2405.14428); P). This matches Flint's layers 46–48.
  - On Llama-2-7B a Hadamard took the `down_proj` input maximum from "over 1,000" to about 15 ([PrefixQuant, arXiv 2410.05265](https://arxiv.org/abs/2410.05265); P).
- **Sum-overflow nuance** (helper-agent numeric check, my re-derivation):
  - With a Sylvester Hadamard of size B ≥ 32 aligned to the 32-blocks, each block sum of Hx equals 32/√B times a ±-sum over only the input channels whose index is ≡ 0 (mod 32). Random signs don't change that.
  - So a dominant channel either vanishes from every block sum or shows up 1.41× (B = 512) in all of them, depending on its index.
  - An offline permutation of the expert intermediate dimension would fix this: permute up/gate rows and down columns, which is exact.
  - The lead's note (`lead_hadamard.md`) estimates a 4–6× shrink of worst-case sums. These two views disagree, so verify on real activations before trusting either.

**Inferences**
- **Does residual rotation help or worsen the overflow?** Neither (my calc).
  - The overflowing tensor is the routed down-projection's input, silu(g)·u, which R1 leaves unchanged.
  - The readers of the residual (q/k/v, router, gate/up) see RMSNorm output. Once gamma is folded into the weights, every element of that input is bounded by √2048 ≈ 45, so their block sums cannot overflow with or without R1.
  - R1 does spread the residual's dominant channel (gamma-scaled RMS 2,150 in layer 46) over all 2048 inputs of the readers. That removes the imatrix concentration there; the effect on weight-only error is unclear, an experiment question (A9).
- For Flint the online Hadamard competes with two cheaper options: Q8_0 for the three affected down tensors (+8 MB per token), or the 2^k rebalance (section 3.9).
- Its extra value is better Q4 quality for down-projections with dominant channels. A Q8_0 down-projection in those layers already captures that.
- Cost: D and H are two launches per folded weight. For `ffn_down_exps` plus `ffn_down_shexp` that is ~188 launches ≈ 1 ms, which eats a third of the rescale saving (my calc).

**Gaps**
- No measurement of rotated Flint in any format.
- Whether Prism's `ggml_mul` sign step fuses on CUDA.

**Experiment idea (low priority) — A9.**
- Fold R1 + R2 with `SpinQuantConverter` and a Flint name mapping on the quant machine (CPU, ~1 h).
- Convert, then imatrix Q4_0 and Q4_1 with layers 46–48 at Q8_0. KLD against the unrotated twins from A2.
- Worth it only if A2 leaves Q4_0 short of Q4_1 by more than 20 % KLD.

### 3.3 Speed-aware mix search

**Takeaway.**
- **Every existing allocator minimises error at a file size; nothing minimises decode time.** On this rig the two objectives diverge in three ways:
  1. Per MoE layer, routed experts and attention each read 18.9 M params per token, although experts are 95 % of the file.
  2. Format efficiency is not monotone in size: Q4_1 ≈ Q8_0 ≈ 78 GB/s, Q5_0 ≈ 59 GB/s. Q5_0 costs 1.46× Q4_1 per parameter and buys less than Q8_0 (1.70×).
  3. Launch counts depend on tensor layout (fusion, rescale), not on bits.
- **UD-mirror v1 is a size-optimal allocation re-expressed in fast formats.** By construction it is not on the speed-quality frontier.

**Cited findings**
- **Unsloth Dynamic 2.0/3.0** ([docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs), [3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs); P, undated): the per-layer selection algorithm is not published. KLD and top-1 agreement are used for evaluation, not as a stated allocation rule. No speed term.
- **EAddario's `--target-bpw` / `--target-size`** ([llama.cpp PR #15550](https://github.com/ggml-org/llama.cpp/pull/15550), open since 2025-08-24, updated 2026-10-22; P, read):
  - per-tensor imatrix-weighted error estimated on sampled rows, a Pareto front per tensor, then Lagrangian allocation to a size budget;
  - legacy types are in the candidate pool and `--target-exclude` can bar K/I-quants;
  - it requires activation statistics in the imatrix ([#14891](https://github.com/ggml-org/llama.cpp/pull/14891));
  - the objective is size.
- **Thireus GGUF-Tool-Suite** ([github.com/Thireus/GGUF-Tool-Suite](https://github.com/Thireus/GGUF-Tool-Suite); P): measures per-tensor KLD sensitivity by degrading one tensor at a time, then fits a mix to RAM/VRAM limits from a chosen type list (q4_0/q4_1/q5_0/q8_0 allowed). Size objective.
- **Layer-wise statistics in mainline:** `llama-imatrix --show-statistics` ([#12718](https://github.com/ggml-org/llama.cpp/pull/12718), merged 2025-07-22; P) prints per-tensor activation statistics. With [#14891](https://github.com/ggml-org/llama.cpp/pull/14891) (merged 2026-10-04 per the helper agent) the imatrix also stores activation sums. These rank tensors by activation energy, not by KLD.
- **ik_llama.cpp `--fudge-factors`** ([ik PR #2361](https://github.com/ikawrakow/ik_llama.cpp/pull/2361), 2026-08-27; P): used to reproduce an Unsloth recipe. RAMP ([arXiv 2603.17891](https://arxiv.org/abs/2603.17891); P) optimises PPL under a bit budget.
- **Not found:** any allocator, paper or community script whose objective is decode latency, time per token, or a per-format kernel cost.
- **The rig's own data** (`quants/README.md`): imatrix Q5_0 (1.881 GB/token) and UD-mirror (2.259 GB/token) are predicted to run at the same speed (section 3.0). UD-mirror is far better in quality (87.4 % / 0.129 vs 85.0 % / 0.177). Spending bytes as Q8_0 instead of Q5_0 is nearly free on these cards.

**Inferences (my calc)**

Candidate moves from UD-mirror v1 (≈ 45 ms, with the rescale):

| Change | Time | Quality |
|---|---|---|
| Down-projections of layers 45–48 to Q8_0, rescale off (section 3.9) | −2.4 ms | ≥ |
| Fused QKV (section 3.4) | −1.0 to −1.4 ms | = |
| All Q5_0 → Q8_0 (0.424 GB → 0.655 GB) | +1.2 ms | ↑ |
| All Q5_0 → Q4_1 (→ 0.385 GB) | −2.3 ms | ↓ |
| Head Q8_0 → Q4_1 | −1.75 ms | ↓ (unknown) |
| Attention q/o Q8_0 → Q4_1 in the 25 least sensitive layers | 25 × 16.8 M × 5.7 µs/M = −2.4 ms | ↓ (unknown) |
| Routed experts Q4_1 → Q8_0 in the 10 most sensitive layers | 10 × 18.87 M × 5.7 µs/M = +1.1 ms | ↑ |

- The first two rows together take UD-mirror to ≈ 41 ms (24.4 tok/s) at unchanged or better quality.
- Trading the remaining rows by measured sensitivity is where 38–40 ms might come from (claim to test).
- For comparison, the makers' Q4_1 with sections 3.9 + 3.4 + the router fix would run ≈ 33.0 − 1.2 − 0.6 ≈ 31 ms (≈ 32 tok/s) at its own task-recovered quality.

**What a speed-aware allocator needs**
1. **A per-format cost table:**
   - µs per call for `MUL_MAT` with Flint's shapes (2048→4096, 4096→2048, 2048→512, 2048→1024, 1024→2048, 2048→150,144);
   - `MUL_MAT_ID` (2048→512 and 512→2048 over 256 experts, 6 used) at batch 1 for Q4_0, Q4_1, Q5_0, Q8_0, F16/BF16;
   - measured with `test-backend-ops perf` on one P106 (Flint shapes would need adding to `tests/test-backend-ops.cpp`), plus per-launch overhead under CUDA graphs.
2. **Per-group sensitivities ΔKLD_g(format).** Groups = tensor kind × layer band, against the Q8_0 reference, assuming additivity.
3. **A multiple-choice knapsack:** minimise Σ ΔKLD subject to Σ µs ≤ T and file size ≤ VRAM.
   - Constraints: q/k/v share a type per layer (to keep the QKV fusion); down-projections in overflow layers must be rescale-safe.
   - Simplest implementation: PR #15550's Lagrangian with its size term replaced by `bytes × (6/256 for _exps) / η(type)`.

**Gaps**
- No per-format µs table on the P106.
- No sensitivity scan for Flint.
- UD-mirror's speed is not measured yet; it is the next rig run and calibrates this model.

**Experiment ideas**
- **A3a. Cost table on the rig (1–2 h).**
  - Build `test-backend-ops` (`-DLLAMA_BUILD_TESTS=ON`) with the Flint shapes added.
  - `test-backend-ops perf -o MUL_MAT,MUL_MAT_ID -b CUDA0` per type.
  - Cross-check against the three whole-file points in section 3.0.
- **A3b. Sensitivity scan (quant machine, overnight, ~100 runs).** A "fake-quant in Q8_0" scheme avoids disk churn:
  - Keep one working copy of the Q8_0 file (42 GB).
  - For each (group, format), overwrite that group's Q8_0 blocks in place with Q8_0 re-encodings of the group quantized to the format (gguf-py plus numpy reimplementations of the quantizers, like this survey's). Restore afterwards.
  - Q8_0 re-encoding adds < 5 % to a Q4 error (claim to test against one real file).
  - Run `llama-perplexity --kl-divergence` on the GPUs (42 GB fits on 3090s + V100s) against the existing base, at 2–4 min per run.
  - Groups: {attn_qkv, attn_o, exps_gate_up, exps_down, shexp, dense} × layer bands {0–1, 2–11, 12, 13–29, 30–45, 46–48}, plus head; formats {Q4_0, Q4_1, Q5_0}. About 6 × 6 × 3 + 3 ≈ 111 runs, ≈ 6–8 h.
- **A3c. Solve and validate.**
  - Build the top 2–3 mixes with real `llama-quantize` from BF16.
  - Run full KLD, the task check, and rig tg128 and served speed.
  - Success means ≥ 87 % same top token and KLD ≤ 0.14 at ≤ 40 ms tg128.

### 3.4 GGUF surgery that cuts kernel launches

**Takeaway.**
- **Fused QKV is the one fusion that works from the file alone** for Flint and saves launches.
- **Merged gate/up expert tensors don't help on sm_61**, because the separate tensors already fuse.
- **The shared expert cannot become routed experts without a graph patch.**

**Which architectures accept which fused tensor names** (P, local source `d77256999`):
- **`blk.N.attn_qkv.weight`** (via `create_tensor_qkv`, falls back to separate q/k/v) is accepted by 111 of the 167 model files in `src/models/`, including flint, qwen2moe, qwen3, qwen3moe, qwen35, qwen35moe, qwen3next, qwen4exp, llama, llama4, deepseek, glm4/glm4-moe, gemma2/3/3n, granite/granite-moe, minimax-m2, step35, laguna, openai-moe, olmoe, phimoe, bailingmoe/bailingmoe3, mistral3, mellum.
  - Some archs use only `ATTN_QKV` directly: bloom, falcon, gpt2, mpt, qwen, starcoder, gptneox, dbrx, plamo2/3, internlm2, bailingmoe2, gemma4.
- **`blk.N.ffn_gate_up_exps.weight`** (`create_tensor_gate_up_exps` or explicit) is accepted only by deepseek2, deepseek2ocr, cohere2moe, gemma4, granite-swa, hy-v3, qwen35moe, qwen3next and qwen4exp. deepseek32, glm-dsa and dots3note pass it in their graphs too.
  - **Flint does not:** its loader creates `ffn_gate_exps`/`ffn_up_exps` as required tensors and passes `nullptr` for the merged one.
  - **qwen3moe does not either.**
- **Dense `ffn_gate_up`:** no such tensor name exists in `llama-arch.cpp`.
- **Shared-expert merge:** no tensor name exists. CUDA can fuse the shared and routed gate/up into one kernel (`ggml_cuda_match_shared_expert`, `ggml-cuda.cu` ~1924), but only when the shared and routed weights have the same width (`ne[1]`) and the same type. Flint's widths are 1024 and 512, so the fusion never fires.
- **Optional per-tensor `.scale` tensors** (`ffn_*_exps.scale` of shape [n_expert], `attn_*.scale` [1]) are loaded for every arch (`src/llama-model.cpp` ~1726–1772). They are a file-level hook for output scales, not input scales.
- **Flint also loads an optional router selection bias** `blk.N.exp_probs_b.bias`.

**Cited findings**
- **CUDA fusion patterns** (`ggml-cuda.cu`, P): `{MUL_MAT_ID, MUL_MAT_ID, GLU}` and `{MUL_MAT, MUL_MAT, GLU}` fuse separate gate and up matmuls with the SwiGLU into one MMVQ launch.
  - The merged path (`build_moe_ffn`, `llama-graph.cpp` ~2372) emits one `MUL_MAT_ID` followed by two views and a separate GLU.
  - Each MMVQ call quantizes its own input (`mmvq.cu` ~1552), so q, k and v each pay a quantize launch.
- **ik_llama.cpp fuses QKV at load time** with `-mqkv` ([ik PR #2725](https://github.com/ikawrakow/ik_llama.cpp/pull/2725), merged 2026-10-22; P). That is the same idea, done at load instead of in the file.

**Inferences (my calc)**
- **Fused QKV saves −196 launches**: 2 matvecs + 2 quantizes × 49 layers, ≈ 1.0–1.4 ms (+3–4 %).
  - Bytes are unchanged. Output is bit-identical if the row concatenation is exact; MMVQ rows are independent.
  - Requirements:
    - q, k and v of a layer must share a type. In the makers' file, and in UD-mirror (Q8_0 or Q5_0 per layer), they do.
    - imatrix-based requantization after the merge would need an imatrix entry for `attn_qkv`, a copy of `attn_q`'s since the input is the same. Merging after quantization avoids that.
- **Merged gate/up exps:** a separate GLU launch per MoE layer means +47 launches. Not useful.
- **Shared expert as two extra routed experts (graph patch):**
  - Split the 1024-wide shared expert into two 512-wide halves. They are exact: SwiGLU is elementwise, so the down projection sums over halves.
  - Store them as experts 256 and 257 in `ffn_{gate,up,down}_exps` (258 experts).
  - In `build_moe_ffn`, concatenate two fixed ids and weight 1.0 after the top-k normalization.
  - Saves the shared path's quantize, fused gate/up, quantize, down and add (~5 launches) for ~1–2 concat ops: −3 to −4 per layer × 47 = −141 to −188 launches. Some of that comes back because `MUL_MAT_ID` now processes 8 slots instead of 6. Net ≈ −0.5 to −1.3 ms.
  - The rescale would then also cover the shared expert; with section 3.9 that is moot.

**Experiment ideas**
- **A4. Fused-QKV file (PC, ~3 h).**
  - A gguf-py script reads each layer's `attn_q/k/v`, checks that the types match, writes `attn_qkv` with rows [q; k; v] (shape 2048 × 5120), drops the originals and copies everything else.
  - On the rig: llama-bench tg128 (5 cards, graphs) A/B against the source file; `llama-completion` greedy, 256 tokens on 4 prompts, which should be identical.
  - Expected +3–4 %.
- **A10 (later). Shared-expert-as-routed patch** in a fork, if A3 finds launch overhead still above ~10 ms after A1 + A4.

### 3.5 Output head

**Takeaway.**
- **The head is the largest single tensor read per token:** 307.5 M params, 12 % of the makers' bytes, 4.2 ms at Q8_0.
- **Its format matters more than trimming:**
  - Q8_0 → Q4_1 saves 1.75 ms;
  - Q8_0 → Q5_0 only 0.6 ms;
  - Q6_K (the llama.cpp default for legacy mixes) is a slow format on these cards.
- **Static vocabulary trimming is either negligible or risky for an agent model** (measurement below), and it needs a patch either way.

**Cited findings**
- **llama.cpp defaults** (`src/llama-quant.cpp`, P): head type is Q6_K for legacy and K-quant mixes unless `--output-tensor-type` is given. The k-quants PR [#1684](https://github.com/ggml-org/llama.cpp/pull/1684) (2023-06) found a 6-bit head improved Q4_0 PPL by ~0.03 on LLaMA-7B.
- **No isolated KLD data** for a Q4_x vs Q8_0 head on a modern model was found.
- **The makers' head is 4-bit**, recovered by distillation (makers' blog, cited in `flint_speedups.md`).
- **Trimming literature:**
  - Language-heuristic trimming gives up to 25 % faster generation on small models, but only 5–10 % on larger ones, and drifts from the full-vocabulary output ([arXiv 2311.09709](https://arxiv.org/abs/2311.09709), 2023-11 / 2024-04; P).
  - Frequency trims up to 40 % were near-lossless on 0.6–1B models on a phone, quality falling beyond 60 % ([SqueezeBits blog](https://blog.squeezebits.com/vocabulary-trimming-methods), 2025-08-04; S).
  - Deployed trimming is for drafters: the FR-Spec-style `d2t` trim in [llama.cpp issue #25187](https://github.com/ggml-org/llama.cpp/issues/25187) (open since 2026-07-01, checked). There, 32 k rows were too few for Swedish plus code (acceptance 0.87 → 0.72), and 65 k kept it.
  - [vLLM PR #59740](https://github.com/vllm-project/vllm/pull/59740) (open, 2026-10-02): a static 32 k list lost 5.7 % mean acceptance and 33 % on multilingual text; per-token SVD-selected rows fixed it.
- **Code:** Flint's loader sizes `output` to the full `n_vocab`. `--logit-bias` acts after the head is computed. The only row-subset head machinery is the drafters' `d2t` scatter in `eagle3.cpp`/`dflash.cpp`.

**My measurements (2026-10-23)**
- **Flint's 150,144 token strings by script:**
  - 129,062 ASCII-only (86.0 %);
  - 9,300 CJK/kana/hangul;
  - 3,631 extended Latin;
  - 3,512 Cyrillic;
  - 1,229 Arabic;
  - ~1,200 other symbols;
  - 1,009 partial-UTF-8 or special;
  - Greek, Indic, Thai and Hebrew ~1,200 combined.
  - Dropping all non-Latin scripts removes ≈ 15.2 k rows (10 %).
- **Coverage, from 2.38 M tokens of wikitext-2 train plus 4.39 M tokens of C++/CUDA/Python:**
  - 71,663 distinct ids;
  - in-sample, the top 65,536 cover 99.91 %;
  - held out (set built on one half of each corpus, tested on the other half), a 65,536-row head misses **1.85 % of English and 5.5 % of code tokens**; a 49,152-row head misses 2.7 % / 6.5 %.

**Inferences (my calc)**
- **Script trimming:** −15.2 k rows ≈ −31 M params ≈ −0.43 ms at Q8_0 (≈ 1 %). Not worth a patch.
- **Frequency trimming to 65 k:** −2.4 ms at Q8_0, but 1 in 18–54 tokens of new text would be unreachable, forcing non-canonical tokenizations. That is unacceptable for code and tool calls.
- **A better route is a dynamic shortlist** (low-rank scorer plus exact rows, as in vLLM #59740). That needs new runtime code, so it is class B.
- **Low-rank or clustered heads applied post hoc** have no tensor slot in Flint's graph and lose accuracy at useful ranks. Not found as a training-free GGUF method.

**Experiment idea — A6 (quant machine, ~1 h for quality, ~30 min on the rig).**
- Take UD-mirror v1 and the makers' Q4_1 (where swapping its 4-bit head for BF16-derived Q8_0 is also worth testing: 192 → 327 MB, +1.7 ms).
- Make variants with `--output-tensor-type q4_1` and `q5_0`.
- Measure KLD and the task check, then rig tg128.
- Accept Q4_1 if Δ same-top-token ≤ 0.3 points and ΔKLD ≤ 0.005 (claim to test).

### 3.6 Attention (35 % of the makers' bytes; 41 % of UD-mirror's)

**Takeaway.**
- **Format first.** The cheapest attention lever is its format (section 3.3); attention q/o at Q8_0 costs 5.3 ms more than at Q4_1 over the whole model.
- **What a file can express:** with Flint's graph, only uniform query-head pruning. Dropping attention blocks or low-rank projections needs code changes, and the literature's training-free losses at a useful speed level are large.

**Cited findings**
- **Attention drop on dense models** ([He et al., arXiv 2406.15786](https://arxiv.org/abs/2406.15786), v6 2024-10; P):
  - Llama-2-70B with half its attention layers dropped: 48.4 % faster, −2.4 % performance;
  - Mistral-7B with 8 of 32 dropped: MMLU 64.2 → 62.5, GSM8K 37.8 → 35.4.
- **On MoE models** ([arXiv 2406.02500](https://arxiv.org/abs/2406.02500); P): only layer/block drops were studied.
  - Mixtral, 8 of 32 MoE layers dropped: 71.5 → 67.0.
  - DeepSeek-MoE-16B (with shared experts), 4 of 28 MoE layers dropped: 61.8 → 55.3.
- **FinerCut** ([arXiv 2405.18218](https://arxiv.org/abs/2405.18218); P): Llama3-8B with 25 % of sublayers removed kept 88–90 %.
- **Query-head pruning under GQA:** CFSP ([COLING 2025](https://aclanthology.org/2025.coling-main.626.pdf); P) avoids head pruning because it degrades GQA models. MP-G ([arXiv 2609.26177](https://arxiv.org/abs/2609.26177); P) has only OPT-6.7B data: PPL 18.46 at 12.5 % of heads pruned.
- **Training-free low-rank:** SVD-LLM V2 at 20 % compression gives LLaMA-3-8B wikitext PPL 6.14 → 8.01, and ASVD 17.55 ([arXiv 2503.12340](https://arxiv.org/abs/2503.12340); P). At an equal 2.8 GB, 3-bit GPTQ (16.28) beats SVD-LLM V2 (119).
- **Not found:** any attention-drop or SVD study on a sliding-window + sinks model, an MoE with 4 KV heads, or Flint.

**Inferences (my calc)**
- **Uniform query-head pruning is file-expressible.**
  - `flint.attention.head_count` is global and llama.cpp maps query head h to KV head ⌊h / (n_head/n_head_kv)⌋.
  - Dropping one query head per KV group (32 → 28) needs:
    - the matching 128-row slices of `attn_q`;
    - the 128-column slices of `attn_output`;
    - the entries of `attn_sinks`.
  - Saves 103 M params per token: −1.4 ms at Q8_0, −0.8 ms at Q4_1 (2–4 %). Too small for a lossy lever.
- **Dropping 8 attention blocks** saves ≈ 2.7–2.9 ms (+6–7 %): 8 × 18.87 M params at Q8_0 plus ~14 launches each. It needs a per-layer skip list in `flint.cpp`. Also lossy and below 15 %.
- **The full-attention (NoPE) layers carry all long-range information**, since the sliding window is 513 tokens. Drop only sliding layers, and measure long-context tasks.

**Experiment idea (low priority) — A11.**
- A CPU build with a `FLINT_SKIP_ATTN=l1,l2,...` env switch in `flint.cpp` (attention output = 0 for listed layers).
- Greedy-select layers by ΔKLD, one at a time, on 20 chunks: about 35 sliding layers × 4 min ≈ 2.5 h on the quant machine's GPUs.
- Pursue only if 8 dropped layers stay under ΔKLD 0.02.

### 3.7 Layer pruning, skipping and MoE-specific skipping

**Takeaway.**
- **Nothing here reaches +15 % speed at the quality bar.**
- **Layer pruning:** ≈ +2 % per layer, with a sharp collapse risk.
- **Expert pruning:** gives no single-stream speed at all on this rig, because bytes per token don't change.
- **Static expert reduction (6 → 4):** already measured, +9 % served at KLD 0.316 → 0.555.
- **New in Sept–Oct 2026:** k₂-normalized top-k reduction (training-free, Qwen3.6) and several expert-pruning papers.

**Cited findings**
- **Dense layer pruning without healing:**
  - SLEB ([arXiv 2402.09025](https://arxiv.org/abs/2402.09025); P): removing 10 % of blocks takes LLaMA-2-7B wikitext PPL from 5.47 to 6.95, at 1.10–1.13× speed.
  - Gromov et al. ([arXiv 2403.17887](https://arxiv.org/abs/2403.17887); P): the Qwen family collapses at about 20 % of layers removed.
  - ShortGPT ([arXiv 2403.03853](https://arxiv.org/abs/2403.03853); P): Llama2-7B −27 % of layers kept MMLU but XSum fell from 19.4 to 0.7.
- **`llama-quantize --prune-layers`** ([PR #13037](https://github.com/ggml-org/llama.cpp/pull/13037), merged 2025-06-22, checked):
  - It rewrites `block_count` but not the per-layer arrays, so the output fails to load ([issue #27976](https://github.com/ggml-org/llama.cpp/issues/27976), closed as stale 2026-10-13, checked).
  - The fix PR #28597 was closed unmerged on 2026-10-20 (helper agent).
  - Flint has per-layer arrays (`sliding_window_pattern`, `rope_pattern`, possibly `expert_feed_forward_length`), so they would need fixing with gguf-py.
- **New, k₂ normalization** ([Chen & Yao, arXiv 2609.04575](https://arxiv.org/abs/2609.04575), 2026-09-04; P): run k₁ experts, normalise by the top-k₂ mass. Qwen3.6-35B-A3B 8 → 4 experts costs −0.35 MMLU instead of −4.65. Softmax routers only.
- **ACE** ([arXiv 2609.05228](https://arxiv.org/abs/2609.05228); P) is calibration-free slot skipping but needs runtime logic.
- **New expert pruning:**
  - RAZOR ([2609.30465](https://arxiv.org/abs/2609.30465), 2026-09-24): beats REAP by 2.1–5.6 points on Qwen3.6-35B-A3B at 25–50 %.
  - HOPE ([2609.18916](https://arxiv.org/abs/2609.18916)): second-order.
  - OMP-MoE ([2609.31631](https://arxiv.org/abs/2609.31631)): Qwen3-30B-A3B at 50 % keeps 93.3 %.
  - OptEP ([2610.27123](https://arxiv.org/abs/2610.27123), 2026-10-19).
  - All P, via the helper agent.
- **Measured on the rig** (`docs/benchmarks/2026-10-22-flint.md`): 6 → 5 experts +4 % served (KLD 0.379), 6 → 4 +9 % (KLD 0.555).
- **Not found:** any shared-expert-skipping measurement, any whole-layer pruning of a Qwen3-MoE or Qwen3.6 model, or any Flint data.

**Inferences (my calc)**
- **One MoE layer** costs ≈ 44.6 M params per token (0.55 ms at UD-mirror's mix) plus ~40 launches (0.2–0.28 ms), so ≈ 0.8 ms (≈ 2 %). The +15 % bar needs ~7 layers (14 %), where the literature shows large losses.
- **Skipping the shared expert in all layers** saves ≈ 5.3 ms (12 %), almost certainly catastrophically. Per-layer skipping needs a patch and a sensitivity scan; expected gain per layer is 0.11 ms.
- **For Flint's sigmoid router with weight normalization,** the k₂ trick is: keep the top k₁ experts, with weights s_i / Σ_{top-k₂} s_j. A ~10-line change in `build_moe_ffn`: take the top-6, sum their weights, then view the first k₁. Worth a 1-hour CPU KLD test, because it could make 6 → 5 nearly free. The speed gain stays ≤ 9 %, so it only stacks.
- **Expert caching** (`--moe-cache-mib`) serves experts kept in host RAM. With every expert resident in VRAM, as here, it has nothing to do.

**Experiment idea (low priority) — A12.**
- Patch `build_moe_ffn` with an env switch `MOE_K1=5` (select top-6, normalise over all 6, keep the first 5 slots).
- CPU KLD with the Q8_0 file: k₁ ∈ {5, 4} against the standard override (`expert_used_count=5/4`), ≈ 15 min per run.
- If 6 → 5 costs < 0.01 KLD, add it to the stack: +4 % served.

### 3.8 Router precision and embedding format

**Takeaway.**
- **Router:** go from F32 to F16 or BF16 (−0.6 ms in the makers' file). Do not go to Q8_0: it adds a quantize launch per layer that cancels the byte saving, and it quantizes an input with a 1,000–2,000-RMS channel in layers 46–48.
- **Embedding:** F32 → Q8_0 saves host RAM (1.23 → 0.33 GB page cache), not time.

**Cited findings**
- **Mainline** (`src/llama-quant.cpp`, P): skips `ffn_gate_inp` before any `--tensor-type` override is consulted. The converter forces routers to F32.
  - [PR #26584](https://github.com/ggml-org/llama.cpp/pull/26584) (draft, opened 2026-08-04 by ggerganov, no activity since; checked) would allow quantizing it.
- **ik_llama.cpp `--ffn-gate-inp-type`** ([PR #896](https://github.com/ikawrakow/ik_llama.cpp/pull/896), merged 2025-11-05; checked): Q8_0 and F32 routers were indistinguishable by PPL on Qwen3-30B-A3B. Decode +1 % on an RTX 4080, up to +6 % on CPU.
- **Not found:** any routing-flip measurement for BF16/F16/Q8_0 router weights, or any data on sigmoid routers.
- **CUDA paths (code read):**
  - BF16/F16/F32 weights go through the float matvec without quantizing the activation.
  - Quantized weights go through MMVQ, preceded by a Q8_1 quantize of the input. That costs one more launch per router call.

**Inferences (my calc)**
- **F32 → F16 for the makers' Q4_1:** 99 → 49 MB, −0.63 ms, +1.8 %. F16 has 3 more mantissa bits than BF16; router weights are small, so range is not an issue (check max |w| < 65,504).
- **BF16 → Q8_0:** −0.30 ms of bytes and +47 launches (+0.24–0.33 ms), so net ≈ 0.

**Experiment idea — A5 (PC 15 min + rig 30 min).**
- gguf-py: rewrite `blk.*.ffn_gate_inp.weight` of the makers' file as F16. Check that the max |w| is far below 65,504.
- Rig: llama-bench A/B, and `llama-perplexity` 10 chunks: PPL equal to ±0.01.
- Do the same F32 → Q8_0 for `token_embd` if host RAM matters (llama-quantize can't requantize from Q4_1 sources cleanly, so do this with gguf-py too).

### 3.9 Q4_0's failure and the FP16-sum overflow: fixes at file level

**Takeaway.** The two problems share a cause: a handful of dominant activation channels in the down-projection inputs of layers 46–48 (and to a lesser degree 12, 18, 30, 37, 45).
- **Overflow.** A per-expert power-of-two rebalance between `ffn_up_exps` and `ffn_down_exps` shrinks every down-projection input by 2^k. It changes no output (SwiGLU is linear in the up branch), changes no quantized code (block-relative formats), and costs no bytes or launches.
  - With the existing env-var patch removing the 470-launch rescale, that is the brief's "scale weights so activations shrink, with a compensating scale folded elsewhere", done exactly.
  - Alternative: Q8_0 down-projections in those layers (+8.3 MB/token from Q4_1, +0.11 ms).
- **Q4_0.** The same layers' down-projections (plus layer 48's attention) need ≥ Q4_1, preferably Q8_0. Rescaling does not help quality, since the problem is relative error, not magnitude.

**Cited findings**
- **Code** (`src/llama-graph.cpp` ~1670 and ~2510, P):
  - `build_amax_src1_f16`: abs, `POOL_1D` max, scale 1/1024, log, scale, ceil, clamp [0,127], scale, exp, div, and a mul on the output.
  - The guard is `arch == MISTRAL4 || MELLUM || FLINT`.
  - The local checkout already carries the `LLAMA_NO_MOE_DOWN_RESCALE` env patch.
- **CUDA's `quantize_q8_1`** (`ggml-cuda/quantize.cu` line ~100) stores `make_half2(d, sum)` where `sum` is the float sum of the 32 inputs. So overflow needs |Σ block| > 65,504, not |x| > 2,048. The 1,024 threshold assumes 32 equal same-sign values.
- **MMQ** uses the scale-and-sum layout (DS4) for Q4_0, Q4_1, Q5_1, Q4_K and Q5_K, and scale only (D4) for Q5_0 and Q8_0 (`mmq.cuh` per `flint_speedups.md`). So prompt processing of Q5_0 down-projections cannot overflow, but MMVQ decode can.
- **Rig measurements** (`docs/benchmarks/2026-10-22-flint.md`): rescale off gave Q4_1 +8 % with NaN on every chunk, Q5_0 +11 %, Q8_0 +5 %.
- **Precedents:**
  - bartowski's [PR #21652](https://github.com/ggml-org/llama.cpp/pull/21652) "Prevent the sum of the dequantized activation in q8_1 from overflowing" (open since 2026-04-09, last updated 2026-10-10; P, read):
    - Mistral-Small-4 gave NaN PPL only when later `ffn_down` layers were Q4_1: the same late-layer pattern as Flint;
    - it clamps the Q8_1 sum (lossy) in the reference, AVX2, AVX1 and CUDA code; PPL 5.5535 with clamping vs 5.5725 when recomputing in F32;
    - in its discussion a convert-time fix was suggested (2026-04-13) and dismissed. The runtime scaling that became #30861 was chosen instead (helper agent).
  - AWQ ([arXiv 2306.00978](https://arxiv.org/abs/2306.00978)) and llm-compressor map `up_proj → down_proj` scaling for Qwen3-MoE: per-channel scaling for quantization, not uniform scaling for range.
  - Nobody was found using uniform up/down rebalancing to avoid FP16 overflow.

**Inferences (my calc)**
- **Exactness.** For a power of two s = 2^k:
  - quant(W_up/s) = quant(W_up)/s, and quant(s·W_down) = s·quant(W_down), for Q4_0, Q4_1, Q5_0 and Q8_0: only the FP16 `d` (and `m`) change, by an exact exponent shift;
  - for expert down-projections d ~ 10⁻³, so s ≤ 2⁸ keeps it normal. For up-projections d ~ 10⁻², so s ≤ 2⁸ gives ≥ 4·10⁻⁵, near FP16's normal minimum 6.1·10⁻⁵. Prefer k ≤ 6 or accept subnormal `d` at ~0.1 % relative precision;
  - Q8_1 activation precision is scale-invariant (`d = amax/127`), as long as amax/127/2^k stays above FP16 subnormal range. In layers 46–48 typical activations are 10²–10⁴, so that holds.
- **Choice of k.** Layer 46 has a channel at RMS 25,600 and layer 48 one at 48,400. With k = 6, a single channel can reach ~4.2 M before a block sum overflows; k = 8 gives 16.8 M. For layers 30, 37 and 45 (channel RMS 174–302), k = 4 is ample.
- **Speed.** Expected to equal the measured rescale-off numbers: +8 % Q4_1, +11 % Q5_0. Combined with fused QKV (+3–4 %) and F16 router (+1.8 %), the makers' Q4_1 would go from 28.0 to ≈ 32–33 tok/s tg128 (claim to test).
- **Q4_0 / Q5_0 down-projections** need the same fix. Q5_0 is safe in MMQ (D4) but not in MMVQ.
- **The shared-expert down-projection is never rescaled.** It works today in the makers' Q4_1 (PPL 19.29 with the rescale, which doesn't touch it), so its block sums evidently stay below 65,504. Its largest channel RMS is 607 in layer 48.

**Gaps**
- Peak (not RMS) activations per layer: the imatrix only stores Σx².
- Whether any layer outside 46–48 ever crosses 65,504 on code, tool-call or long-context text.

**Experiment ideas**
- **A1. Overflow map and rebalanced file (PC ~2 h + rig ~1 h). Top candidate.**
  1. **Peaks.** CPU build on the PC with a `cb(amax, "ffn_moe_amax", il)` callback added in `build_amax_src1_f16`. Run with `LLAMA_NO_MOE_DOWN_RESCALE` unset, so the amax node exists. A `llama-eval-callback`-style tool logs per-layer max and 99.99th percentile over ~200 k tokens: wikitext, code, a recorded agent session with tool calls, and a 32 k-token document. CPU has no FP16 sums, so the raw values are safe to collect.
  2. **Rebalance.** A gguf-py script: for each listed layer, multiply the FP16 `d` (and `m` for Q4_1) of every block of `ffn_down_exps` by 2^k and divide those of `ffn_up_exps` by 2^k. Block layout: Q4_0 18 B with `d` at offset 0; Q4_1 20 B with `d` at 0 and `m` at 2; Q5_0 22 B with `d` at 0; Q8_0 34 B with `d` at 0. Write a new file. Inputs: the makers' Q4_1 and UD-mirror v1.
  3. **Verify off-rig.** CPU KLD of rebalanced vs original: expect ≈ 0 (bit-level FP16 differences only).
  4. **Rig.** `LLAMA_NO_MOE_DOWN_RESCALE=1 llama-perplexity` (GPU, 20 chunks wikitext plus a code file): PPL equal to the rescaled stock build within ±0.01, and no NaN. Then llama-bench tg128 and server_bench.
  - Alternative to step 2 for BF16-derived mixes: requantize with `--tensor-type 'blk\.(46|47|48)\.ffn_down_exps=q8_0'`, and use only the env patch.
  - Expected: +8 % (Q4_1) / +11 % (UD-mirror with Q5_0 down-projections), exact.
- **A2** (section 3.1) settles the Q4_0 half.

---

## 4. Top candidates for a deep dive (ranked)

1. **A1: rescale-free files via 2^k up/down rebalancing (or Q8_0 down-projections in layers 46–48).**
   - Exact, zero bytes, works on existing files without requantizing, and the speed is already measured: +8 % (Q4_1) / +11 % (Q5_0).
   - The new fact that makes it safe: the overflow-scale activations are confined to layers 46–48.
   - About a day including rig validation.
2. **A3: speed-aware mix search** (cost table + sensitivity scan + knapsack).
   - The only lever that can deliver UD-mirror quality at substantially better speed (target 38–41 ms vs ~45 ms; claim to test).
   - It also decides whether Q4_0, Q5_0 and a 4-bit head belong in the mix at all.
   - 2–4 days, mostly unattended GPU time on the quant machine.
3. **A4: fused-QKV file.** Exact, +3–4 %, half a day, stacks with everything.
4. **A2: Q4_0 bisection.** Cheap (≈ 3 h CPU) and tests this survey's mechanism: dominant channels, not missing offsets. Its result feeds A3's format menu and whether rounding work (A8, A7) is worth doing.
5. **A5: F16 router for the makers' file.** One hour, +1.8 %, exact. Bundle with A1 and A4 into a "makers' Q4_1, rig edition" (predicted ≈ 32–33 tok/s tg128 vs 28.0; claim to test).
6. **A6: head format.** Two KLD runs plus a rig run: up to −1.75 ms if Q4_1 holds quality.
7. **Rounding upgrades (A8: PR #12557; A7: AutoRound, starting with Qwen3.6's ready-made files).** Only if A3 shows Q4_0 or Q5_0 tensors on the frontier. Expected −10 to −25 % KLD for those formats; nothing for Q4_1.
8. **Not recommended:**
   - merged `ffn_gate_up_exps` (no gain on sm_61);
   - router Q8_0 (net ≈ 0, routing risk);
   - vocabulary trimming (≤ 1 % safe, 5 % unsafe);
   - expert pruning (no bytes per token saved);
   - layer, attention or head pruning (below the 15 % bar at acceptable loss);
   - rotations (small, and they miss the down-projection inputs).

**Carry-over to Qwen3.6-35B-A3B, summarised:**
- A1 is irrelevant (no rescale).
- A3 applies fully. Its larger head (248 k vocab) makes A6 worth ~2.9 ms.
- A4 applies to its 10 attention layers only.
- Its shared expert has the routed width (`moe_intermediate_size` = `shared_expert_intermediate_size` = 512; vocab 248,320; config read 2026-10-23), so CUDA's routed+shared gate/up fusion can fire. Keep `_shexp` and `_exps` in the same type in any mix (claim to test; the condition is `weight->type == shared_weight->type` in `ggml_cuda_match_shared_expert`).
- Ready-made AutoRound Q4_0/Q4_1/Q5_0 files exist for it (A7).
- Its dominant-channel map is unknown; the same imatrix scan (`in_sum2/counts` per layer) takes minutes on any published imatrix.
