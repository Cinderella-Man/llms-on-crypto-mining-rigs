# Ornith-1.5-35B on Pascal mining rigs

[![Ornith-1.5-35B](https://img.shields.io/badge/Ornith--1.5--35B-coming%20soon-718096)](#whats-next)

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) is the smaller model of the Ornith-1.5 line,
built on the architecture of Qwen3.6-35B-A3B: a mixture-of-experts model with 256 experts of width 512, 8 of them plus a
shared expert used per token (about 3 B parameters), and 40 layers, of which 30 are linear-attention (Gated DeltaNet)
layers and 10 full attention. Its makers release it under the MIT licence; this project is not affiliated with them.

**Status: coming soon.** It is the nearer of the two Ornith targets, ahead of
[Ornith-1.5-122B](../ornith-1.5-122b/README.md), and has not run on the reference rig yet.

## What is known

- **Same shapes as Qwen3.6-35B-A3B** (read from its GGUF header), so everything on the
  [Qwen3.6 page](../qwen3.6-35b-a3b/README.md) about what fits and what the engine does applies here too: the full
  262,144-token context fits on four or five cards, and two copies fit on nine.
- **Its Q4_K_M file** (21.7 GB) reads 2.02 GB per generated token: 57 % in Q4_K, 39 % in Q6_K and 4.5 % in F32. None of
  those types has a Pascal kernel in the v2 engine yet, so today the file runs on llama.cpp's own kernels, with the
  engine's CUDA graphs and launch shape.
- **Expected speed:** est. 19–23 tok/s on today's engine, 30–36 with the Pascal Q6_K kernel (built, waiting for its rig
  gate) and a Pascal Q4_K kernel (planned), which together cover 96 % of its bytes.

Details: [model branches](../../docs/research/notes/model_branches.md), sections 1.2, 2.1 and 3.

## What's next

1. **A first rig run with no new code**: the Q4_K_M file on five cards, stock llama.cpp against the v2 engine
   ([Q159](../../quants/experiments.md#registry)).
2. **The Pascal Q6_K and Q4_K kernels** ([Q101, Q149](../../quants/experiments.md#registry)), shared with Qwen3.6.
3. **Hybrid-model support** for the on-card attention mask and pipelined speculation, also shared with Qwen3.6.
