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

## Readiness (2026-10-24)

What the model is, whether llama.cpp runs it, which files exist and what they should do on the rig, before any rig time
is spent. Nothing here has run on the rig; speeds are estimates (est.) built on the measured Qwen3.6-35B-A3B numbers,
because the two models have the same shapes.

### Architecture

From the makers' `config.json` and the Q4_K_M file's GGUF header:

| | |
|---|---|
| Family | Mixture of experts, a hybrid of linear and full attention; Hugging Face class `Qwen3_5MoeForConditionalGeneration`, llama.cpp architecture `qwen35moe` |
| Parameters | 35.5 B in total (the BF16 GGUF is 71.1 GB), about 3 B active per token |
| Layers | 40, in groups of three Gated DeltaNet layers and one full-attention layer (10 full attention, 30 linear), plus one MTP layer: the GGUF has 41 blocks and `nextn_predict_layers = 1` |
| Hidden size, vocabulary | 2,048; 248,320 |
| Experts | 256 of width 512, 8 per token, plus one shared expert of width 512 |
| Full attention | 16 query and 2 key/value heads, head size 256; the KV cache is 20 KiB per token |
| Linear attention | 16 key and 32 value heads of size 128, convolution kernel 4: a fixed state of about 60 MiB per sequence |
| Context | 262,144 tokens natively; the makers' YaRN setting (factor 4, about 1 M tokens) is applied statically and costs quality on ordinary prompts, so it stays off |
| Other | Reasons in a `<think>` block by default; the makers recommend temperature 0.6, top-p 0.95, top-k 20 for general use. A vision tower ships as a separate 0.9 GB `mmproj` file that text use doesn't need. MIT licence |

### Engine support

- **Supported in upstream llama.cpp, and so in the fork.** The `qwen35moe` architecture arrived with "models : support
  qwen3.5 series" (#19468, 2026-02-11). Since then upstream added MTP for the family (#24025, #26296, #30097) and
  EAGLE-3 speculation for Qwen3.5/3.6 (#24593); the makers' own GGUF card serves the model with a plain `llama-server`.
- **The server handles the hybrid memory:** context checkpoints of the recurrent state (`--ctx-checkpoints`) for prompt
  reuse, and rollback of speculative drafts up to the number of states the memory keeps. Nothing is missing for plain
  decoding.
- **The fork's multi-card work for a hybrid model**, shared with Qwen3.6, is built and waits for its rig timing
  ([Q162](../../quants/experiments.md#registry), fork branch `pascal-p106-v2-hybrid`): the on-card attention mask for
  the hybrid cache, and recurrent state groups that let pipelined speculation drop wrong guesses
  ([model branches](../../docs/research/notes/model_branches.md), section 2.1). MTP gains little on the rig today
  (1.04× on Qwen3.6, [Q43](../../quants/experiments.md#registry)), so it isn't part of the first run.

### Files

All sizes in GB (10⁹ bytes). Every Q4- and Q5-class file keeps the 248,320 × 2,048 output head in Q6_K (0.42 GB, a fifth
of a token's bytes), so no popular file is made only of types the Pascal kernels cover.

| Source | Files | Pascal kernels today |
|---|---|---|
| Makers (`Ornith-1.5-35B-A3B-GGUF`) | Q4_K_M 21.7, Q5_K_M 25.3, Q6_K 29.2, Q8_0 37.8, BF16 71.1 | Q4_K_M: none of its bytes (57 % Q4_K, 39 % Q6_K, 4.5 % F32, read). Q8_0: est. all but the F32 router |
| bartowski, with imatrix | Q4_0 20.4, Q4_1 22.4, IQ4_XS 19.3, IQ4_NL 20.3, Q4_K_S 21.1, Q4_K_M 21.9, Q5_K_S 24.6, Q5_K_M 25.5, Q6_K 30.5, Q8_0 37.8, down to IQ2_XXS 10.3; the imatrix is 0.19 | Q4_0: est. 72 % of its bytes (95 % with the built Q6_K kernel), if its mix matches bartowski's Qwen3.6 Q4_0 (read). IQ4_XS and Q5_K_S: bulk covered, type mix not read |
| Community, Unsloth-style dynamic mixes | UD-IQ4_XS 17.7, UD-Q4_K_S 20.9, UD-Q4_K_XL 22.4, UD-Q5_K_XL 26.6, UD-Q6_K_XL 31.8, UD-Q8_K_XL 38.5 | Type mixes not read. Unsloth itself publishes no GGUF repository for the 1.5-35B (it has the 1.0-35B and the 1.5-122B) |
| Other community files | APEX mixes (13.5–25.3), abliterated and MTP-grafted variants | Not read |

**Fit:** on nine cards (54 GB, about 5.6 GB of weights per card) every file up to Q6_K fits with the full 262,144-token
context (est.: file + 5.4 GB of KV and state); 8–32k tokens cost 0.2–0.7 GB. The Q4-class files (19–22 GB) need four or
five cards, Q5_K_M five or six, Q6_K six or seven, Q8_0 eight. Fewer cards are faster, so the first runs use five.

### Estimated rig speed

Decode, single stream, five cards, est. from the speed model in
[model branches](../../docs/research/notes/model_branches.md) (bytes per token and the measured Qwen3.6 rates):

| File | Bytes per token | Today's engine | With the Pascal Q6_K and Q4_K kernels |
|---|---|---|---|
| Makers' Q4_K_M | 2.02 GB | 19–23 tok/s on llama.cpp's own kernels (the K-quant efficiency fitted on Flint's UD-Q4_K_XL gives as little as 14) | 30–36 |
| bartowski Q4_0 | est. 2.1 GB | **29–31**: Qwen3.6's Q4_0, with the same shapes and quantizer rules, measured 29.0 on four cards and 31.1 on five | 33–36 with the Q6_K kernel alone |
| Q8_0 | est. 3.3 GB | 16–20, the lossless reference rather than a target | |

- **Recommended file for the rig's first run: bartowski's Q4_0 (20.4 GB)**, 5 cards, plus the makers' Q4_K_M as the
  K-quant baseline arm. Q4_0 is the fast file on today's engine; it is also the lower-quality one (on Qwen3.6, KLD 0.036
  against 0.013 for Unsloth's UD-Q4_K_XL), so it serves as the speed baseline, not the end product.
- **The end product** is a file at UD-Q4_K_XL quality that the Pascal kernels cover: the Q4_K kernel
  ([Q149](../../quants/experiments.md#registry)) makes the 22.4 GB dynamic mixes fast, and a mix of our own can be made
  from the makers' BF16 with the bartowski imatrix, as for Qwen3.6 (mix A and B).

### Files we already hold

- **Our model library:** the makers' Q4_K_M (21,713,463,040 bytes, header read), no other Ornith-1.5-35B file.
- **The quant machine:** no Ornith-1.5-35B file. It holds Ornith-1.5-122B (UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL;
  [Q158](../../quants/experiments.md#registry)) and Qwen3.6-35B-A3B files (Unsloth UD-Q4_K_XL, our mixes, bartowski's
  imatrix), none of which stands in for this model.
- **To fetch, onto the quant machine's bulk disk:** bartowski's Q4_0 and imatrix (20.6 GB) for the first
  run; the makers' Q8_0 (37.8 GB) only when the KLD base for a quality comparison is needed. Read the Q4_0's header first
  (`scripts/gguf_mix.py`) to confirm its type mix before rig time goes to it.

## What's next

1. **A first rig run with no new code**: bartowski's Q4_0 and the makers' Q4_K_M on five cards, stock llama.cpp against
   the v2 engine ([Q159](../../quants/experiments.md#registry)).
2. **The Pascal Q6_K and Q4_K kernels** ([Q101, Q149](../../quants/experiments.md#registry)), shared with Qwen3.6.
3. **Hybrid-model support** for the on-card attention mask and pipelined speculation, also shared with Qwen3.6: built,
   rig timing next ([Q162](../../quants/experiments.md#registry)).
