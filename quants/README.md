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

Against Unsloth's Q8_0 of [Flint](https://huggingface.co/RecursiveSI/flint) (39.8 B parameters, 2.5 B per token);
speed with llama.cpp `d77256999` + CUDA graphs on 9 × P106-090.

| File | Made by | Size | Same top token | Mean KL divergence | Real chat speed | Generation (tg128) |
|---|---|---|---|---|---|---|
| UD-Q4_K_XL | Unsloth | 25.1 GB | **87.6 %** | **0.124** | 15.8 tok/s | 17.4–19.0 tok/s |
| Q5_0 + imatrix | here | 27.6 GB | 85.0 % | 0.177 | 18.9 tok/s | 22.3–22.8 tok/s |
| Q4_1 + imatrix, expert down-projections Q8_0 | here | 30.6 GB | 81.5 % | 0.257 | to measure | to measure |
| Q4_1 + imatrix | here | 25.0 GB | 80.9 % | 0.280 | to measure | to measure |
| Q4_1 + imatrix, expert down-projections IQ4_NL | here | 24.3 GB | 80.5 % | 0.281 | to measure | to measure |
| Q4_1 | Flint's makers | 26.0 GB | 78.6 % | 0.316 | **24.3 tok/s** | **27.9–28.1 tok/s** |
| Q4_0 + imatrix | here | 22.6 GB | 71.4 % | 0.566 | — (unusable) | — |
| **UD-mirror v1** (UD-Q4_K_XL's allocation in fast formats) | here | 27.0 GB | **87.4 %** | **0.129** | to measure | to measure |

The makers' Q4_1 was tuned by distillation after quantizing, so its distance from Q8_0 overstates its real-task loss
(see [the Flint research notes](../docs/research/notes/flint_speedups.md)).

**UD-mirror v1 keeps Unsloth's quality in formats these cards decode quickly**: 87.4 % top-token agreement and mean
KL divergence 0.129, against 87.6 % and 0.124 for UD-Q4_K_XL (perplexity +3.8 % vs +3.3 %; the differences are within
about two standard errors). Its speed on the rig is the next measurement.

What we have learned:
- **Simple formats are fast here; K-quants are not.** Q4_1 and Q5_0 beat UD-Q4_K_XL on speed by 20–50 %.
- **Unsloth's quality comes from where it spends bits**, not from K-quant formats as such: attention and the shared
  expert at Q8_0 in most layers, and the experts at 4–5.5 bits chosen layer by layer
  ([scripts/gguf_mix.py](../scripts/gguf_mix.py) shows this for any file).
- **Q4_0 breaks Flint** (no per-block offset); Q4_1 is fine.
- **Flint's 4- and 5-bit expert down-projections need llama.cpp's rescale step**, which costs 5–11 % speed; formats
  such as Q8_0 and IQ4_NL don't
  ([Flint](../docs/benchmarks/2026-10-22-flint.md#the-moe-down-projection-rescale), [patch](../patches/)).

## Next steps

1. Measure UD-mirror v1's speed on the rig ([recipe](recipes/flint-udmirror-v1.txt); quality already matches UD-Q4_K_XL).
2. Variants of the mirror: attention at Q5_0 instead of Q8_0 (attention is ~4 % of the file but ~35 % of what each token
   reads); expert down-projections at Q8_0 with the rescale step off.
3. The makers' distillation-tuned Q4_1 with only the expert down-projections widened to Q8_0.
4. Task-based quality checks for the best two or three files.
5. The same treatment for the Qwen3.6-35B-A3B family.

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
