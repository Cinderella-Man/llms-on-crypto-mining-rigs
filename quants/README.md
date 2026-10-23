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
`d77256999` on the P106-090 rig, `llama-bench` tg128 on six cards unless noted: "stock" is upstream llama.cpp, the
next column adds CUDA graphs on Pascal, "2-warp" adds [2 warps per block at one token](../patches/mmvq-pascal-2warps.diff)
and "2-warp wide rows" uses 2 warps only for rows of ≥ 2048 values ([patch](../patches/mmvq-pascal-2warps-widek.diff)). Where a file was not measured on the rig
yet, the [cost model](../scripts/token_cost.py)'s prediction for the standard build is given (marked "pred.").
"Rescale-free" files run with `LLAMA_NO_MOE_DOWN_RESCALE=1`
([why](research.md#removing-the-moe-down-projection-rescale)). Recipes are in [recipes/](recipes/) and every run in
[experiments.md](experiments.md).

**The bar is Unsloth's UD-Q4_K_XL.** It is what people download, and it agrees with BF16 on 87.7 % of next tokens. A
file that is faster but ~9 points worse in token agreement, such as the makers' own Q4_1, is not a substitute, so the
work here aims at UD-Q4_K_XL quality or better, faster than UD-Q4_K_XL on the same build.

| File | Made by | Size | Same top token | Mean KLD | Perplexity vs BF16 | tg128, stock llama.cpp | tg128 | tg128, 2-warp | tg128, 2-warp wide rows |
|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | Unsloth | 42.3 GB | 93.3 % | 0.052 | +3.3 % | — | 20.5 (9 cards) | — | — |
| **UD-Q4_K_XL** (the bar) | Unsloth | 25.1 GB | 87.7 % | 0.120 | +3.9 % | 17.6 | 19.5 | **24.8** | 24.6 |
| **S4-L0** (S4, attention output Q4_1 in layers 33–48, F16 router; rescale-free) | here | 32.5 GB | **88.7 %** | 0.123 | **+3.1 %** | — | 23.1 | 26.3 | **28.0** |
| **S5** (S4-L0 + attention q Q8_0 in layers 0–24, F16 router; rescale-free) | here | 32.6 GB | **89.1 %** | **0.105** | +3.5 % | — | 23.0 pred. | — | — |
| S4 (speed-aware mix, rescale-free) | here | 32.6 GB | 88.7 % | 0.120 | +3.4 % | — | 23.0 pred. | — | — |
| UD-mirror v1, rescale-free | here | 27.4 GB | 87.5 % | 0.127 | +4.0 % | — | 23.2 pred. | — | — |

Real chat requests at the full 262,144-token context on all nine cards (`-b 512 -ub 128 -bs`, four prompts of 256
tokens, [server_bench.sh](../scripts/server_bench.sh)), same wide-rows build: **S4-L0 24.7 tok/s, UD-Q4_K_XL 20.4**
(UD-Q4_K_XL on the CUDA-graphs build without the launch-shape patch: 15.8).

Below the bar, kept for reference:

| File | Made by | Size | Same top token | Mean KLD | Perplexity vs BF16 | tg128 | tg128, 2-warp |
|---|---|---|---|---|---|---|---|
| S4-L1 (L0 + Q4_1 expert down-projections in layers 25–45) | here | 30.0 GB | 87.7 % | 0.132 | +3.5 % | 24.1 pred. | — |
| UD-mirror v2, rescale-free (attention q at Q4_1) | here | 27.2 GB | 87.1 % | 0.142 | +3.6 % | 24.4 pred. | — |
| S3, S2, S1 (speed-aware mixes, rescale-free) | here | 31.0 / 25.8 / 25.6 GB | 87.1 / 85.7 / 83.6 % | 0.149 / 0.178 / 0.219 | +4.7 / +5.4 / +4.6 % | 24.3 / 25.2 / 26.7 pred. | — |
| Q5_0 + imatrix | here | 27.6 GB | 84.5 % | 0.185 | +5.7 % | 22.3–22.8 (5–9 cards) | — |
| Q4_1 + imatrix | here | 25.0 GB | 81.0 % | 0.280 | +9.5 % | 27.1 pred. | — |
| Q4_0 + imatrix, seven dominant-channel tensors at Q8_0, rescale-free | here | 23.1 GB | 80.2 % | 0.315 | +3.8 % | — | — |
| Q4_1 | Flint's makers | 26.0 GB | 78.4 % | 0.322 | +4.5 % | 28.1 (5 cards) | 31.8 (5 cards) |
| Makers-fast: makers' Q4_1 rescale-free, F16 router, group-64 (`Q4_1_64`), fused Q/K/V | here | 23.9 GB | 78.9 % | 0.315 | +3.9 % | 30.9 (5 cards) | 35.5 (5 cards) |
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
  ([research](research.md#where-a-tokens-time-goes)). llama.cpp's matrix-vector kernel reaches 75 GB/s of the card's
  179 GB/s; Pascal-tuned kernels reach 132–142 GB/s ([benchmark](../bench/mmvq_pascal/#results-on-the-p106-090)).
- **The launch shape alone is worth 13–27 %**: 2 warps per block instead of 4 at one token takes the makers' Q4_1 from
  28.1 to 31.8 tok/s and UD-Q4_K_XL from 19.5 to 24.8. Any comparison between files has to use the same build.
- **Where the bits go matters more than the format, and layers 0–24 matter most.** Lowering attention output, expert
  down-projections or expert gate/up to Q4_1 costs 0.007–0.018 KLD per 8-layer group in layers 0–24 and 0.001–0.006 in
  layers 25–48; attention q at Q8_0 in layers 0–16 alone takes KLD from 0.123 to 0.111. The output head must stay Q8_0
  unless it was trained at 4 bits.
- **Q5_0 is a poor deal on these cards**: it costs 1.4× Q4_1's decode time per weight, close to Q8_0's 1.6×, for much
  less precision than Q8_0.
- **Flint's rescale step can go**: only the expert down-projections of layers 46–48 overflow, so Q8_0 there (+8 MB per
  token) lets `LLAMA_NO_MOE_DOWN_RESCALE=1` remove ~470 kernel launches per token (+8 % measured) with identical output.
- **The makers' tuned tensors only work in their own file**: their experts, head or attention moved into a file made
  from BF16 make it worse.
- **Q4_0 fails on Flint through seven dominant-channel tensors**, not through its missing offset; keep those at Q8_0.
- **Hadamard rotations don't help Flint's 4-bit files** in llama.cpp (they help only plain Q4_0), because its 8-bit
  activation blocks absorb the spread-out outlier channels.

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
| [compose_gguf.py](../scripts/compose_gguf.py) | any PC | Builds a GGUF from the tensors of several quants of the same model (per-layer searches without re-quantizing) |
| [retype_tensors.py](../scripts/retype_tensors.py) | any PC | Converts selected tensors of a GGUF to another format and copies the rest byte for byte (e.g. an F16 router) |
| [token_cost.py](../scripts/token_cost.py) | any PC | Bytes read per token by format and a predicted decode speed on the rig, from the GGUF header |
| [bench.sh](../scripts/bench.sh), [server_bench.sh](../scripts/server_bench.sh) | the rig | Speed: raw generation and real chat requests |

Recipes for the quants made here are in [recipes/](recipes/). Nothing has been published yet; the
[release checklist](releasing.md) says what a file needs first, including that it must say clearly which hardware it is
for.
