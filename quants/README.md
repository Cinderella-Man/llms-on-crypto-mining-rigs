# Quants for old GPUs

A subproject of [LLMs on crypto-mining rigs](../README.md): making and measuring model files (quantizations, "quants")
that run fast on Pascal GPUs with few streaming multiprocessors, such as the P106-090 mining cards, without giving up
more quality than necessary.

## Why this needs its own work

A quant stores a model's weights in fewer bits, packed in small blocks. Quantizers such as Unsloth or bartowski choose
the packing format tensor by tensor to get the best quality per gigabyte, assuming a modern GPU or a Mac that unpacks
any format almost for free. The P106-090 is different: with only 5 streaming multiprocessors, unpacking costs real
time, and the elaborate formats ("K-quants" and "I-quants") are much slower to unpack than simple ones (Q4_0, Q4_1,
Q5_0, Q8_0). On the rig, the same model runs **~1.5 times faster** from a simple-format file than from a K-quant file of
the same size ([Flint](../docs/benchmarks/2026-10-22-flint.md#quantization-speed-against-quality)). Nobody publishes
quants for that trade-off, so this subproject makes them.

## Method

1. **Measure quality on an ordinary PC**: perplexity, KL divergence and top-token agreement against a high-precision
   reference, on a fixed text ([guide](guide.md#measuring-quality-on-an-ordinary-pc)).
2. **Measure speed on the rig**: raw generation and real chat requests, same build, flags and card count
   ([guide](guide.md#measuring-speed-on-the-rig)).
3. **Make candidates**: one format for everything, mixes per tensor kind, or a good community quant's layer-by-layer
   allocation re-expressed in fast formats ([guide](guide.md#making-your-own-quant)).
4. **Check task quality** for the shortlist before anything is shared ([release checklist](releasing.md)).

## Results so far: Flint

Against [Flint](https://huggingface.co/RecursiveSI/flint)'s original BF16 weights (39.8 B parameters, 2.5 B per token):
80 × 512 tokens of wikitext-2, all files scored with the same CUDA build on the same quant machine. Speed is llama.cpp
`d77256999` + CUDA graphs on 9 × P106-090: real chat requests and `llama-bench` tg128 where measured, otherwise the
[cost model](../scripts/token_cost.py)'s prediction for tg128 (marked "pred."). "Rescale-free" files run with
`LLAMA_NO_MOE_DOWN_RESCALE=1` ([why](research.md#removing-the-moe-down-projection-rescale)). Recipes are in
[recipes/](recipes/) and every run in [experiments.md](experiments.md).

| File | Made by | Size | Same top token | Mean KLD | Perplexity vs BF16 | Real chat | tg128 |
|---|---|---|---|---|---|---|---|
| Q8_0 | Unsloth | 42.3 GB | 93.3 % | 0.052 | +3.3 % | — | 20.5 (9 cards) |
| UD-Q4_K_XL | Unsloth | 25.1 GB | 87.7 % | 0.120 | +3.9 % | 15.8 | 17.4–19.0 |
| **S4** (speed-aware mix, rescale-free) | here | 32.6 GB | **88.7 %** | **0.120** | **+3.4 %** | to measure | 23.0 pred. |
| **UD-mirror v1, rescale-free** | here | 27.4 GB | 87.5 % | 0.127 | +4.0 % | to measure | 23.2 pred. |
| UD-mirror v2, rescale-free (attention q at Q4_1) | here | 27.2 GB | 87.1 % | 0.142 | +3.6 % | to measure | 24.4 pred. |
| S3 (speed-aware mix, rescale-free) | here | 31.0 GB | 87.1 % | 0.149 | +4.7 % | to measure | 24.3 pred. |
| S2 (speed-aware mix, rescale-free) | here | 25.8 GB | 85.7 % | 0.178 | +5.4 % | to measure | 25.2 pred. |
| Q5_0 + imatrix | here | 27.6 GB | 84.5 % | 0.185 | +5.7 % | 18.9 | 22.3–22.8 |
| S1 (speed-aware mix, rescale-free) | here | 25.6 GB | 83.6 % | 0.219 | +4.6 % | to measure | 26.7 pred. |
| Q4_1 + imatrix | here | 25.0 GB | 81.0 % | 0.280 | +9.5 % | to measure | 27.1 pred. |
| Q4_0 + imatrix, seven dominant-channel tensors at Q8_0, rescale-free | here | 23.1 GB | 80.2 % | 0.315 | +3.8 % | to measure | to measure |
| Q4_1 | Flint's makers | 26.0 GB | 78.4 % | 0.322 | +4.5 % | **24.3** | **27.9–28.1** |
| **Makers' Q4_1, rescale-free, F16 router** | here | 26.3 GB | 78.5 % | 0.322 | +4.3 % | to measure | 30.5 pred. |
| Makers' Q4_1 as group-64 (`Q4_1_64`, same weights) | here | 23.5 GB | 78.7 % | 0.314 | +4.4 % | to measure | needs the [patch](../patches/q4_1_64-type.diff) |
| **Makers-fast**: all of the above plus fused Q/K/V, rescale-free | here | 23.9 GB | 78.9 % | 0.315 | +3.9 % | to measure | 30.4+ pred.; needs the patch |
| Q4_0 + imatrix | here | 22.6 GB | 71.7 % | 0.565 | +45 % | — (unusable) | — |

The makers' Q4_1 was tuned by distillation after quantizing, so its distance from BF16 overstates its real-task loss.
A tool-calling check confirms it: on 300 BFCL v3 tasks ([bench/bfcl_flint](../bench/bfcl_flint/)) BF16 calls 80.7 %
correctly, the makers' Q4_1 83.0 %, makers-fast 83.3 %, UD-mirror v1 rescale-free 83.0 %, S4 82.0 % and S1 78.3 %,
all within the ±4–5-point uncertainty of 300 tasks. Two Flint files whose KLDs differ by less
than a few hundredths are not distinguishable: the same file computed on two different GPUs already differs by KLD 0.065
([noise floor](research.md#measuring-quality-on-flint-the-noise-floor)).

What we have learned:
- **These cards are limited by instructions, not memory bandwidth**, so simple formats are fast and K-quants are not:
  Q4_1 and Q8_0 decode at ~0.4 of the 173 GB/s peak on Flint, Q5_0 at 0.30, K-quants and I-quants at 0.19
  ([research](research.md#where-a-tokens-time-goes)).
- **Where the bits go matters more than the format.** The cheapest quality is attention k/v and the shared expert at
  Q8_0; attention q at Q8_0 buys almost nothing; the output head must stay Q8_0 unless it was trained at 4 bits.
  The speed-aware mixes S1–S4 and the mirrors all beat a single fast format at the same speed.
- **Flint's rescale step can go**: only the expert down-projections of layers 46–48 overflow, so Q8_0 there (+8 MB per
  token) lets `LLAMA_NO_MOE_DOWN_RESCALE=1` remove ~470 kernel launches per token (+8 % measured) with identical output.
- **Q4_0 fails on Flint through seven dominant-channel tensors**, not through its missing offset; keep those at Q8_0.
- **Hadamard rotations don't help Flint's 4-bit files** in llama.cpp (they help only plain Q4_0), because its 8-bit
  activation blocks absorb the spread-out outlier channels.
- **The makers' Q4_1 is group-64 data** stored as group-32: the same weights fit in 23.5 GB with a group-64 type.

## Results so far: Qwen3.6-35B-A3B

Against bartowski's Q8_0 (80 × 512 tokens of wikitext-2, same quant machine); speed is the cost model's tg128 prediction
except where measured.

| File | Size | Same top token | Mean KLD | tg128 |
|---|---|---|---|---|
| Q4_0 (bartowski; head and four attention q tensors Q6_K) | 20.8 GB | 92.0 % | 0.036 | 19.6 measured, 20.8 pred. |
| Same file with every Q6_K tensor as Q8_0 | 21.0 GB | 92.0 % | 0.037 | 23.6 pred. |
| Mix B: imatrix Q4_0, `ssm_out`, shared expert, full attention and head Q8_0 | 20.7 GB | 92.8 % | 0.029 | 22.5 pred. |
| Mix A: mix B plus Q8_0 linear-attention qkv and gate | 21.1 GB | **94.1 %** | **0.020** | 20.2 pred. |

The K-quant leftovers in a "Q4_0" file are worth +13 % at no quality cost; the MTP draft file carries the same Q6_K head
([experiments.md](experiments.md), Q29 and Q31–Q33). Qwen3.6's cheapest quality is `ssm_out`, the shared expert and its 10
full-attention layers at Q8_0 (Q32); like Flint's, its head must stay at 8 bits.

## Next steps

What is running now, what comes next and in which order, with the decision gates: [experiments.md](experiments.md#plan).
Every experiment done so far, with its result, is in the same file.

## Tools

| Script | Runs on | What it does |
|---|---|---|
| [quality_kld.sh](../scripts/quality_kld.sh) | any PC with Docker | Perplexity and KL divergence of candidates against a reference |
| [quantize_docker.sh](../scripts/quantize_docker.sh) | any PC with Docker | Builds a quant with llama.cpp's `llama-quantize`, optionally with an importance matrix and per-tensor formats |
| [mirror_recipe.py](../scripts/mirror_recipe.py) | any PC | Re-expresses an existing quant's per-tensor formats in other formats, as `llama-quantize` options |
| [gguf_mix.py](../scripts/gguf_mix.py) | any PC | Shows which format each kind of tensor uses in a GGUF file |
| [bench.sh](../scripts/bench.sh), [server_bench.sh](../scripts/server_bench.sh) | the rig | Speed: raw generation and real chat requests |

Recipes for the quants made here are in [recipes/](recipes/). Nothing has been published yet; the
[release checklist](releasing.md) says what a file needs first, including that it must say clearly which hardware it is
for.
