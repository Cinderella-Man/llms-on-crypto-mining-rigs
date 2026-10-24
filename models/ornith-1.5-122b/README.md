# Ornith-1.5-122B on Pascal mining rigs

[![Ornith-1.5-122B](https://img.shields.io/badge/Ornith--1.5--122B-coming%20soon-718096)](#whats-next)

[Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B) is a fine-tune of Qwen3.5-122B-A10B aimed at
terminal coding agents: a 122 B-parameter mixture-of-experts model that uses about 10 B parameters per token. Like
Qwen3.6 it is a hybrid, with 36 linear-attention (Gated DeltaNet) layers and 12 full-attention layers. Its makers
release it under the MIT licence; this project is not affiliated with them.

**Status: coming soon.** It has not run on the reference rig yet. It is the project's ~120 B option: it fits the rig's
54 GB of GPU memory only at 2–3 bits per weight.

## What is known

- **Files that fit** (Unsloth's [GGUF repository](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF)): UD-IQ2_XXS
  (42.0 GB) and UD-Q2_K_XL (44.5 GB), with est. 32–64k tokens of context; UD-IQ3_XXS (49.5 GB) is marginal. All weights
  must sit on the cards: the 4 GB host cannot hold experts.
- **Expected speed:** est. 9–12 tok/s on today's engine, 13–16 with Pascal Q4_K and Q6_K kernels. Each token reads est.
  4.0–4.7 GB, and the 2–3-bit expert formats fall back to llama.cpp's own kernels.
- **What carries over unchanged:** CUDA graphs on Pascal, the launch shape, the Pascal kernels for its Q8_0 and Q5_K
  tensors, the prompt-processing changes and GPU sampling for tool requests.
- **What needs work:** kernels for the IQ2/Q2_K/IQ3 types and for Q4_K/Q6_K; hybrid-model support for the on-card mask
  and pipelined speculation, whose saved recurrent states (~144 MiB each) compete for memory that a 2-bit file already
  fills.

Details: [model branches](../../docs/research/notes/model_branches.md), section 2.5;
[large MoE models](../../docs/research/notes/models_large_moe_and_compression.md).

## What's next

1. **A first rig run with no new code**, using the smallest file that fits
   ([Q158](../../quants/experiments.md#registry)).
2. Then the kernels its files need, in the order that run shows.
