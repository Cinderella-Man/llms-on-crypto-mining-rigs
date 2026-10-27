# Q74: are Flint's INT4 weights a repack of flint-int4, and are they rounding or QAT? (2026-10-23)

The per-tensor tables and scripts behind registry row Q74 ([experiments.md](../../../quants/experiments.md)) and the
weights section of the [rec engine note](../../research/notes/rec_engine.md). The makers' Q4_1 GGUF is an exact repack
of the MLX export `flint-int4`, and its INT4 codes are an MSE-scaled round-to-nearest quantization of the published BF16
weights, so KLD against BF16 judges the makers' files fairly. The check reads 68.2 M weights of three files without
downloading them: HTTP range reads of the two safetensors repos (about 180 MB in total) and the GGUF read in place
through its header offsets. It runs on any CPU with numpy.

| File | What |
|---|---|
| `run/st_header.py` | fetches a remote safetensors header with HTTP range requests and saves it as JSON |
| `run/fetch.py` | downloads slices (along axis 0) of remote safetensors tensors with range requests, logs them in `slices.json` |
| `run/gguf_index.py` | the GGUF header and tensor table as JSON (pure Python) |
| `run/common.py` | loaders shared by the checks: MLX 4-bit unpacking, Q4_1 blocks from the GGUF, BF16 slices |
| `run/q1_exact.py` | question 1: codes, scales and offsets of the GGUF against the MLX export, dequantized weights compared |
| `run/q2_rounding.py` | question 2: the INT4 codes against round-to-nearest of the BF16 weights at the export's own scale and bias, with min-max and MSE-searched grids for comparison |
| `run/q2_affine.py` | follow-up: the smallest weight perturbation per group that makes the codes round-to-nearest under any scale and bias |
| `run/probe.py` | where the mismatches sit: excess past the midpoint, residual histograms, inversion examples |
| `run/floats.py` | the unquantized tensors (norms, router, sinks) of both repos compared bit for bit |

## Tensors compared (68.2 M weights)

| Group | Tensors | Weights |
|---|---|---|
| Routed experts | `ffn_gate_exps` and `ffn_down_exps`: layers 3 and 45 (experts 0–3), layer 24 (experts 0–3 and 252–255) | 33.6 M |
| Attention output | `attn_output` (`o_proj`) of layers 3, 24 and 45 | 25.2 M |
| Attention V | `attn_v` of layer 24 | 1.0 M |
| Output head | `output` (`lm_head`), rows 20000–22047 and 100000–102047 | 8.4 M |
| Embeddings | `token_embd`, rows 20000–20063 | — |

## 1. Is the makers' Q4_1 GGUF an exact repack of flint-int4? Yes

| Check | Result |
|---|---|
| Codes | 100.0000 % identical in every tensor compared |
| Scales and offsets | Q4_1 `d` = fp16(scale) and `m` = fp16(bias) exactly, in 100 % of blocks |
| BF16 → FP16 conversion | Lossless: every scale is in the FP16 normal range |
| Block layout | Each group of 64 is stored as two Q4_1 blocks with identical `d` and `m` |
| Dequantized weights | GGUF and MLX agree bit for bit (max difference 0) |
| Negative control | GGUF expert 0 against MLX expert 1: 8 % of codes agree |
| `token_embd` | The GGUF stores it as F32, equal to the MLX INT8 dequantization (s·q + b in f32) in 100 % of values |

## 2. Are the INT4 codes round-to-nearest of the BF16 weights? Yes, up to about 0.01 of a step

The test rounds each BF16 weight to the nearest point of the export's own grid (s·q + b) and compares that code with
the stored INT4 code. A step is one grid spacing, s.

| Tensors | Codes off RTN | Groups with a rank inversion | INT4 error vs BF16 (rel. Frobenius) | RTN error at the same s, b | Min-max RTN error |
|---|---|---|---|---|---|
| Experts (gate, down) | 1.06–1.81 % | 2.6–8.4 % | 0.0856–0.0868 | 0.0855–0.0864 | 0.0892–0.0901 |
| `o_proj` | 1.01–1.65 % | 3.0–6.7 % | 0.0861–0.0882 | 0.0859–0.0877 | 0.0897–0.0914 |
| `v_proj` | 2.23 % | 10.4 % | 0.0866 | 0.0861 | 0.0897 |
| `lm_head` | 0.08–0.09 % (ties) | 0 | 0.0914–0.0941 | same | 0.0953–0.0980 |
| `embed_tokens` (INT8) | 0.22 % | — | — | — | — |

What the numbers show:

- **Mismatches are tiny.**
  - Every mismatch is exactly one code step.
  - A mismatched BF16 value lies only just past the rounding midpoint: by a median of 0.015 step, with 99 % of
    mismatches within 0.07–0.09 step.
  - The |w_bf16 − w_int4| distribution matches RTN: median 0.239 step and 99th percentile 0.52–0.55 (RTN gives
    0.25 and 0.495).
  - Errors above one step occur for 0.03 % of weights, in every tensor including `lm_head`. These are outliers clipped
    by the MSE-shrunk range.
  - The per-group mean error has a standard deviation of 0.034 step, the same as RTN noise (0.036): there is no
    systematic drift.
- **The scales are MSE-fitted to these BF16 weights.**
  - A least-squares refit of the scale for the stored codes gives a median ratio of 1.0005.
  - The INT4 error is 4 % below min-max RTN, and a 31-point MSE search (`q2_rounding.py`) lands within 0.7 % of it.
  - The rounding residuals are depleted near ±0.5 step (about 8.6 % against 10 % per bin). This is the selection
    effect of an MSE scale search: the search in `q2_affine.py` reproduces it, and min-max is flat. So the weights
    don't cluster on the grid.
- **The unquantized tensors are bit-identical in both repos:** layer-24 norms, q/k norms, sinks and router. The only
  exception is the final `model.norm`, where 2 of 2048 values differ (0.0313 against 0.0311, and 0.0068 against
  0.0065).
- **Not fully RTN-exact.** In 2.6–10 % of groups the codes reverse the order of two BF16 weights. No scale, bias or
  rounding arithmetic can do that.
  - With the best per-group scale and bias, 75–91 % of groups (100 % of `lm_head`) are exactly RTN.
  - The rest need the weights perturbed by a median of 0.007–0.012 step (99th percentile 0.016–0.047). That is about
    0.3 % of the weights' RMS, which is 3.4 steps.
  - The perturbation grows with depth (mismatch 1.1 % at L3, 1.5 % at L45) and clusters in particular rows and columns.
    In L24 `o_proj` the top columns all belong to one attention head.

### Verdict

flint-int4 is an MSE-scaled round-to-nearest quantization of almost exactly the published BF16 weights.

- The weights that were quantized differ from the BF16 release by about 1 % of a step, with a training-like
  row/column structure. Possible sources are a very light fine-tune or QAT pass on the linear weights only, with norms
  and router frozen, or an FP32 master that differs from the BF16 file.
- That difference is roughly 30 × smaller than the INT4 rounding noise (0.29 step RMS). It moves 1–2 % of codes by one
  step.
- **It is not GPTQ or AWQ.** Those would show 10–40 % of codes off RTN, many errors between 0.5 and 1 step, and
  per-channel scale ratios. None of these appear.
- **It is not QAT that moved the codes.** Such codes would sit many steps away from BF16, and the INT4 error would
  exceed RTN's. The INT4 error is within 0.5 % of RTN's.

Consequence: **KLD against BF16 is a fair test of these weights.** They don't deviate from BF16 by design beyond
rounding.

The files can't show whether training upstream of both files was quantization-aware. If it was, the BF16 file would be
that run's latent weights; the residuals show no clustering at grid points or midpoints that would hint at it.
Confidence is high for question 1 and for "RTN, not GPTQ, AWQ or code-moving QAT"; the source of the ~0.01-step
perturbation is a guess.

## How it was made

Made on 2026-10-23 on a CPU (numpy) from three files:

- **MLX INT4:** [RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) at commit
  `3b93f9053b530488405ae9fca51d4900d6ce21f0`. Weights are stored as `U8 [.., out, in/2]`, experts stacked as
  `switch_mlp`, with `.scales` and `.biases` per group of 64.
- **BF16:** [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) at commit
  `2a247be0c0aec895ccb0ac7ca92229ec6df37288`, one tensor per expert.
- **GGUF:** the makers' `flint-int4-q4_1.gguf` from
  [RecursiveSI/flint-int4-gguf](https://huggingface.co/RecursiveSI/flint-int4-gguf), read in place.

```bash
cd docs/data/2026-10-23_qat_check
# headers of the safetensors files that hold the compared tensors (see each repo's model.safetensors.index.json)
python -I run/st_header.py https://huggingface.co/RecursiveSI/flint-int4/resolve/<commit>/<shard>.safetensors data/int4/<shard>.json
python -I run/st_header.py https://huggingface.co/RecursiveSI/flint/resolve/<commit>/<shard>.safetensors data/bf16/<shard>.json
# slices: <tensor> <i0> <i1> triples, the names and ranges the check scripts load (e.g. the first four experts of
# layer 3's gate projection with their scales and biases; per-expert tensors and lm_head rows from the BF16 repo)
python -I run/fetch.py <shard url> data/int4/<shard>.json data/int4/slices model.layers.3.mlp.switch_mlp.gate_proj.weight 0 4 ...
python -I run/fetch.py <shard url> data/bf16/<shard>.json data/bf16/slices model.layers.3.mlp.experts.0.gate_proj.weight 0 -1 ...
python -I run/gguf_index.py flint-int4-q4_1.gguf data/gguf/index.json
export FLINT_INT4_GGUF=$PWD/flint-int4-q4_1.gguf
python -I run/q1_exact.py; python -I run/q2_rounding.py; python -I run/q2_affine.py; python -I run/probe.py; python -I run/floats.py
```

The scripts name every slice they load (`PAIRS` in `q1_exact.py`, the label lists in the others); a missing slice
fails with its key. Flint is Apache-2.0; the scripts are this project's (MIT).
