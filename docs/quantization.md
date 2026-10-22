# Comparing and making quantizations

A language model is a huge table of numbers (weights). The original files store each weight in 16 bits, which for a
40-billion-parameter model means 80 GB, too much for this rig. A **quantization** ("quant") stores the same weights
with fewer bits, typically 4–8, by grouping weights into small blocks and keeping a shared scale per block. The file
shrinks two to four times and loses a little precision. GGUF files on Hugging Face are such quants, made by the model's
authors or by community quantizers such as Unsloth or bartowski.

This page explains how to tell how much a quant has lost, how to make your own, and why that matters on these cards.
Everything here except the speed test runs on any ordinary PC.

## Why the choice of quant matters more on this rig

Quants come in **formats**. Simple ones (Q4_0, Q4_1, Q5_0, Q8_0) store each block with one or two numbers; K-quants
and I-quants (Q4_K_M, IQ4_XS and the "dynamic" mixes such as Unsloth's UD files) use more elaborate block structures
that squeeze out more quality per gigabyte. On a modern GPU unpacking any of them is nearly free, so the smallest file
of a given quality wins. On the P106-090, with only 5 streaming multiprocessors, unpacking costs real time: the same
model generates text **~1.5 times faster** from a simple-format file than from a K-quant file of the same size
([single-card calibration](benchmarks/2026-10-22-single-card-calibration.md),
[Flint](benchmarks/2026-10-22-flint.md#quantization-speed-against-quality)). The best file for this rig is often not
the one that is best elsewhere, so it pays to measure, and sometimes to make your own.

## Measuring quality on an ordinary PC

Quality depends only on the weights, not on the hardware that runs them, so it can be measured on any PC with enough
RAM for the reference model. [scripts/quality_kld.sh](../scripts/quality_kld.sh) does it with llama.cpp's
`llama-perplexity` on the CPU, inside Docker:

1. A **reference** file reads a fixed text, by default 80 chunks of 512 tokens (about 41,000 tokens) of the
   wikitext-2 test set, the standard benchmark text of Wikipedia articles. At every position it predicts a probability
   for each possible next token. Those predictions are saved once (the file is large: ~6 GB for a model with a
   150,000-token vocabulary).
2. Each **candidate** quant reads the same text, and its predictions are compared with the reference's at every
   position.

Use the best version that fits in RAM as the reference: the original 16-bit weights if possible, otherwise a Q8_0
file, which is nearly identical to the original.

What the numbers mean:

| Number | What it says | How to read it |
|---|---|---|
| **Same top token** | How often the candidate's most likely next token is the reference's | The most intuitive one: 88 % means it would pick a different next word 12 % of the time |
| **Mean KL divergence** | How far the candidate's whole probability distribution drifts from the reference's, averaged over all positions | 0 means identical; lower is better. As a rough guide: below ~0.05 very close, ~0.1–0.2 noticeable, above ~0.3 large |
| **99th-percentile KL divergence** | The drift at the worst 1 % of positions | Catches quants that are fine on average but go badly wrong now and then |
| **Perplexity ratio** | How surprised the candidate is by the real text, relative to the reference (1.00 = same) | The classic number, but it can hide differences: two quants with similar perplexity can drift very differently |

Example from Flint, all against its Q8_0 file:

| Quant | Same top token | Mean KL divergence | Perplexity |
|---|---|---|---|
| Unsloth UD-Q4_K_XL | 87.6 % | 0.124 | +3.3 % |
| Q5_0 with importance matrix (made here) | 85.0 % | 0.177 | +4.6 % |
| Q4_1 from the model's makers | 78.6 % | 0.316 | +3.7 % |
| Q4_0 with importance matrix (made here) | 71.4 % | 0.566 | +45 % |

Two cautions. First, these numbers measure *distance from the reference*, not how well the model does its job: some
makers fine-tune their own quant after quantizing ("distillation"), which moves it away from the original weights on
purpose. Flint's makers did this, and their Q4_1 was reported to match the full model on tool-calling tasks despite a
large KL divergence. Check a shortlist with real tasks too. Second, the text matters: wikitext is English prose; a
coding model may drift differently on code.

```bash
# reference + candidates (paths relative to the models folder; the work folder needs wikitext-2-raw/wiki.test.raw)
scripts/quality_kld.sh <llama.cpp source> <models folder> <work folder> flint-Q8_0.gguf flint-Q4_1.gguf my-quant.gguf
# the same file with a runtime change, e.g. fewer experts per token
CAND_ARGS="--override-kv flint.expert_used_count=int:4" TAG=_e4 scripts/quality_kld.sh … flint-Q4_1.gguf
```

Each candidate takes 10–15 minutes on a 16-core desktop CPU for a 40B MoE model. The reference predictions are saved
and reused.

## Measuring speed on the rig

Speed does depend on the hardware, so it has to be measured on the rig itself, with the same build and flags you will
use: [scripts/bench.sh](../scripts/bench.sh) for raw generation speed and
[scripts/server_bench.sh](../scripts/server_bench.sh) for real chat requests through `llama-server`. Compare quants
on the same number of cards and the same context length.

## Making your own quant

[scripts/quantize_docker.sh](../scripts/quantize_docker.sh) runs llama.cpp's `llama-quantize` on a PC with Docker. It
needs a high-precision source file (16-bit original, or Q8_0 if that is all you have room for), and optionally an
**importance matrix**: a small file recording which weights mattered most while the model read some calibration
text, so the quantizer can spend its precision where it counts. Many quantizers publish theirs (for Flint, Unsloth's
`imatrix_unsloth.gguf_file`).

```bash
# one format for everything, with an importance matrix, output head kept at Q8_0 (the script's default)
scripts/quantize_docker.sh <llama.cpp source> <models folder> flint-Q8_0.gguf flint-imat-Q5_0.gguf Q5_0 imatrix.gguf
# a mix: Q4_1 overall, but the experts' down-projections at Q8_0
EXTRA="--output-tensor-type q8_0 --tensor-type ffn_down_exps=q8_0" \
  scripts/quantize_docker.sh <llama.cpp source> <models folder> flint-Q8_0.gguf flint-q4_1-downq8.gguf Q4_1 imatrix.gguf
```

To see how an existing quant is built, [scripts/gguf_mix.py](../scripts/gguf_mix.py) lists which format each kind of
tensor uses and its share of the file, reading only the header. For Flint, Unsloth's UD-Q4_K_XL keeps attention and
the shared expert at Q8_0 in most layers and gives the experts Q5_K or IQ4_XS layer by layer; that allocation, more
than the format of any single tensor, is what keeps it close to the reference.

`--tensor-type` takes a pattern matched against tensor names (`ffn_down_exps`, `attn_q`, `ffn_gate_shexp`, …), so
different parts of the model can get different formats. Quantizing a 40B model takes 3–8 minutes on a desktop CPU and
needs free disk space for the output (25–30 GB here).

Then measure it like any other quant: quality on the PC, speed on the rig.

## Traps found so far

- **Q4_0 can break a model.** Q4_0 blocks have a scale but no offset. For Flint that pushed perplexity up 45 %; Q4_1,
  which has an offset, was fine. Always measure.
- **Some formats need llama.cpp workarounds.** llama.cpp protects Flint's Q4_x/Q5_x expert down-projections from a
  16-bit overflow with an extra rescaling step that costs speed; with that step removed, a Q4_1 file produced
  nonsense (NaN). Formats such as Q8_0 and IQ4_NL don't need the step
  ([Flint](benchmarks/2026-10-22-flint.md#the-moe-down-projection-rescale)).
- **Requantizing from Q8_0 adds a small error** on top of the new format's own. It's fine for experiments; use the
  original weights for anything you publish.
- **A rig-tuned quant is not better everywhere.** On modern GPUs and Macs, a larger simple-format file is only slower.
  See the [release checklist](releasing-quants.md) before sharing one.
