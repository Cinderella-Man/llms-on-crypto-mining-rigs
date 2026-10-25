---
license: apache-2.0
base_model: RecursiveSI/flint
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - flint
  - pascal
  - mining-rig
---

# Flint K2h-r16 (GGUF): a Flint file for Pascal GPUs and mining rigs

A quantization of [Flint](https://huggingface.co/RecursiveSI/flint), a 39.8 B-parameter mixture-of-experts model that
uses 2.5 B parameters per token, built for old NVIDIA Pascal GPUs: the P106 and P104 mining cards, the GTX 10 series and
the Tesla P4/P40. It uses only Q5_0, Q8_0 and 16/32-bit tensors, the formats that the
[Pascal engine](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs) decodes fastest, and reaches lower error
than Unsloth's UD-Q4_K_XL (KL divergence 0.105 against 0.120) while reading fewer bytes per token.

This project is not affiliated with Flint's makers or with Unsloth.

| File | Size | SHA-256 |
|---|---|---|
| `flint-K2h-r16.gguf` | 29.4 GB | RIG-TBD (after the release header is written; [CHECKSUMS](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/CHECKSUMS)) |

## Who it is for

- **Pascal GPUs, above all mining rigs** with several 6–8 GB cards on PCIe x1 links, such as the Manli rig with nine
  P106-090 cards it was measured on. It needs ~30 GB of GPU memory plus the context: five or six 6 GB cards for short
  conversations, all nine for the full 262,144-token context.
- **Other GPUs and Macs** load it too (standard GGUF types, llama.cpp release b11803 or newer), but there the Pascal
  speed-up doesn't apply, and a K-quant file of the same quality is smaller: use Unsloth's UD-Q4_K_XL instead.

## Run it

With the Pascal engine (release archives RIG-TBD), nine cards, the full context:

```bash
llama-server -m flint-K2h-r16.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 3 --host 127.0.0.1 --port 8080
```

- `--pipe-depth 3` sends guessed tokens through the idle cards of the layer split; the output stays the same. Add
  `--pipe-guess staircase --pipe-draft flint-staircase-drafter-v2.0.gguf` for the trained guesser
  ([its card](../flint-staircase-drafter/README.md)).
- The engine drops llama.cpp's safety rescale of Flint's expert layers by itself on Pascal cards; don't set
  `LLAMA_NO_MOE_DOWN_RESCALE` to anything but `tg` or `0` for this file.
- What every flag does, and the six-card and agent variants:
  [the Flint page](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/models/flint/README.md).

With stock llama.cpp (b11803 or newer), the same line without `--pipe-depth`. The file's header carries the makers'
sampling (temperature 1.0, top-p 0.95, top-k 20, min-p 0), so no sampling flags are needed.

## Speed

Generated tokens per second on nine P106-090 cards (each on a PCIe 1.0 x1 link, a 2-core Celeron host):

| | Stock llama.cpp | Pascal engine v2 |
|---|---|---|
| Generation (`llama-bench`, 128 tokens after a 512-token prompt) | RIG-TBD | RIG-TBD |
| Served chat, one conversation | RIG-TBD | RIG-TBD |
| Served chat with pipelined speculation (`--pipe-depth 3`) | — | RIG-TBD |
| Served chat with the drafter | — | RIG-TBD |

On the same engine it is ~4–5 % faster than UD-Q4_K_XL (41.8 against 39.9 tok/s generation on a v2 build before the
release), because it reads 2.091 GB per token against 2.213.

## Quality

Against Flint's original BF16 weights, 80 × 512 tokens of wikitext-2 test:

| File | Size | Same top token | Mean KL divergence | Perplexity change |
|---|---|---|---|---|
| **K2h-r16** | 29.4 GB | **88.8 %** | **0.105** | **+2.7 %** |
| Unsloth UD-Q4_K_XL | 25.1 GB | 87.7 % | 0.120 | +3.9 % |
| Unsloth Q8_0 | 42.3 GB | 93.3 % | 0.052 | +3.3 % |

Task checks with the makers' sampling (T 1.0, top-p 0.95, top-k 20, min-p 0, reasoning kept):
- **Single-turn tool calling** (BFCL v4, 3,477 tasks): +0.14 points against UD-Q4_K_XL, 90 % interval −0.87 to +1.16:
  non-inferior at a 1.5-point margin. Measured on the same tensors with a 32-bit router, which leaves the KL divergence
  within its noise.
- **Multi-turn tool calling** (BFCL v4, 458 tasks), on G3, the parent mix: +0.87 points, 90 % interval −3.23 to +4.98.
- **8k-token agent conversations:** the tokens Flint writes match BF16 as closely as with UD-Q4_K_XL (KL ratio 0.999,
  the same tensors with a 32-bit router).
- Reasoning lengths of the parent mix are level with UD-Q4_K_XL's.

## How it was made

From Flint's BF16 weights with llama.cpp `d77256999`: llama-quantize with Unsloth's importance matrix, base Q5_0, Q8_0
where the per-group measurements showed it pays (attention q in layers 9–24, attention output in 0–32, k and v in 0–32,
the shared expert outside 33–45, the dense FFNs, expert gate/up in layers 2–8 and expert downs in 46–48); a
GPTQ-rounded Q5_0 output head (+0.002 KL divergence where round-to-nearest costs +0.023); the 47 routers in F16. The
complete recipe, the tensor-type map, per-tensor hashes and a script that rebuilds the file and checks it tensor by
tensor:
[reproduce/model.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/model.md).

## Licence and credits

- **Flint** by RecursiveSI is released under Apache-2.0, and so is this file: [LICENSE](LICENSE). Changes from the
  original: the weights are quantized to Q5_0 and Q8_0 (the output head GPTQ-rounded to Q5_0, the routers converted to
  F16) and stored as GGUF. "Flint" and "Recursive" are used only to describe the file's origin.
- **Unsloth's importance matrix** (`imatrix_unsloth.gguf` from
  [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF), Apache-2.0) guided the quantization; the file's
  `quantize.imatrix.*` fields record it.
- **The GGUF header** says `general.license = apache-2.0`, names Flint by RecursiveSI as the base model, and carries
  the makers' sampling in `general.sampling.*`. OWNER-TBD: `general.name` and `general.quantized_by`.
