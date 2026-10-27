# SLQ: the formats and bit widths of the published SLQ models (2026-10-27)

The data behind [slq.md](../../research/notes/slq.md) sections 3 and 4: what the only published SLQ models (The
Kaitchup's Qwen3.8-27B, in a vLLM/Humming version and a GGUF version) store, tensor by tensor, and how many bytes each
spends on the decoder at the same measured EAR target. Only file headers and the GGUF repository's precision maps were
read; no model was run, no weights were read and no speed or quality was measured here.

| File | What |
|---|---|
| `gguf_tensors.tsv` | every tensor of the three GGUF files (498 each): GGML type, donor file, payload bytes, shape; from the repository's precision maps |
| `gguf_formats.tsv` | per GGUF file and GGML type: tensors, bytes, share of the file's tensor bytes |
| `gguf_summary.tsv` | per GGUF file: tensor bytes, the embedding's and output head's type and bytes, and the rest (the 496 decoder projections) |
| `vllm_classes.tsv` | per vLLM checkpoint: bytes by tensor class (decoder, embedding, head, MTP, vision, other) |
| `vllm_modules.tsv` | every decoder projection of the three vLLM checkpoints: shape, stored bit width (packed bytes × 8 / weights), zero-point kind (`int` packed, `float` in BF16, `bf16` for a projection kept unquantized), bytes of weight + scales + zero points |
| `vllm_summary.tsv` | per vLLM checkpoint: packed and BF16 projections, parameter-weighted mean bit width, projection bytes, stored bits per weight with scales and zero points, share of projection weights per bit width |
| `run/run.sh` | the runs: fetch the precision maps, tally them, tally the safetensors headers, copy the tables back |
| `run/slq_bytes.py` | the tallies (standard library only) |

## Results in one table

Decoder projections only (the 496 matrices of the 64 blocks; embeddings, output head, MTP and vision excluded), at the
EAR@10 each file was built to and measured at by its maker on the same 512 × 2,048-token calibration set:

| Target | GGUF EAR@10 | GGUF decoder bytes | GGUF bits per weight | vLLM EAR@10 | vLLM decoder bytes | vLLM mean bit width (stored) | GGUF / vLLM bytes |
|---|---|---|---|---|---|---|---|
| EAR98 | 98.0004 % | 16.437 GB | 5.40 | 98.0008 % | 16.870 GB | 5.34 (5.54) | 0.974 |
| EAR98.5 | 98.5002 % | 17.073 GB | 5.61 | 98.5027 % | 18.747 GB | 5.98 (6.16) | 0.911 |
| EAR99 | 99.00003 % | 19.739 GB | 6.48 | 99.0025 % | 23.656 GB | 7.59 (7.77) | 0.834 |

GGUF bits per weight divide the decoder bytes by the 24.351 G projection weights. The vLLM files keep the embedding and
output head in BF16 (2.543 GB each); the GGUF files store the embedding in Q4_K (EAR98, EAR98.5) or Q6_K (EAR99) and the
head in Q8_0 (EAR98) or BF16.

## How it was made

- **GGUF:** [kaitchup/Qwen3.8-27B-SLQ-GGUF](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-GGUF) at revision
  `8528ce42479c946842d7b5868dfb592fb7d4a52b` (Apache-2.0): `assets/precision-ear98.svg`, `-ear98.5.svg`, `-ear99.svg`.
  Each map's tooltips give every tensor's GGML type, donor file and payload bytes; the GGUF files themselves (63 GB) were
  not downloaded.
- **vLLM:** [kaitchup/Qwen3.8-27B-SLQ-EAR98](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR98) at
  `1165bcbb96ef55e11689388e44691ceec46016fd`,
  [-EAR98.5](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR98.5) at `f11e58ea47a3e985517973356c60ae941d988f5b` and
  [-EAR99](https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-EAR99) at `7cf51bcf8341ff8bd2d904f03aef9310ddee6564`
  (Apache-2.0), copies already on the quality machine. Only the safetensors headers and `config.json` were read.
  A packed projection stores its weight as `I32` words, a BF16 `weight_scale` per row and 128 input columns, and a zero
  point per row and group (`I32`, packed at the layer's bit width, or BF16 for a float zero point); its bit width is the
  packed bytes × 8 divided by rows × columns, with the columns taken from the scale's group count × 128.
- The EAR figures are the maker's, from the model cards.
- On a separate machine, in Docker (`python:3.12-slim`, no network, low CPU priority).

Commands (with `QM` the quality machine's SSH alias, `WORK` a work directory there, and `CKPTS` the three checkpoint
directories there):

```bash
run/run.sh gguf
CKPTS="EAR98=<dir> EAR98.5=<dir> EAR99=<dir>" run/run.sh vllm
run/run.sh fetch
```
