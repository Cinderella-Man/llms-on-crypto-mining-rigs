# Qwen3.6-35B-A3B on Pascal mining rigs

[![Qwen3.6-35B-A3B](https://img.shields.io/badge/Qwen3.6--35B--A3B-coming%20soon-718096)](#whats-next)

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) is a 35.95 B-parameter mixture-of-experts model that
uses about 3 B parameters per token. It is a hybrid: 30 of its 40 layers are linear-attention (Gated DeltaNet) layers
that keep a fixed-size state, and 10 are full attention. It reads up to 262,144 tokens of context and carries a
multi-token-prediction (MTP) layer for drafting. Qwen releases it under Apache-2.0; this project is not affiliated with
Qwen.

**Status: coming soon.** It has run on the reference rig with stock llama.cpp and with the v1 engine's kernels; the v2
engine, a recommended file and settings follow after the rig's model matrix.

## Measured so far

On the [reference rig](../../docs/mining_rig.md) (P106-090 cards, PCIe 1.0 x1, 4 GB host), tokens per second (tok/s):

| Build and file | Generation | Source |
|---|---|---|
| Stock llama.cpp, bartowski's Q4_0, 4–9 cards | 18.5 (14.0 at 32k tokens of context, 8.1 at 64k) | [Qwen baselines](../../docs/benchmarks/2026-10-22-qwen-baselines.md) |
| Stock llama.cpp with CUDA graphs on Pascal | 19.6 | [speed-ups](../../docs/benchmarks/2026-10-22-speedups.md) |
| The same, real chat requests with MTP (2 drafted tokens) and GPU sampling | 27.7 | [speed-ups](../../docs/benchmarks/2026-10-22-speedups.md) |
| v1 engine, Q4_0, 4 cards, Pascal kernels off → on | 22.8 → **29.0** (+27 %) | [Q38](../../quants/experiments.md#registry) |
| v1 engine, Q4_0, 5 cards, plain → MTP with 2 drafted tokens | 31.1 → 32.4 (1.04×) | [Q43](../../quants/experiments.md#registry) |

- **The full context fits on four or five cards:** 262,144 tokens take 27 GB of the rig's 54 GB, so two copies can run
  side by side and serve two conversations at ~16 tok/s each ([overview](../../docs/overview.md), finding 8).
- **Prompt processing:** 215–247 tok/s on stock llama.cpp.
- **MTP gains little on the v1 engine** because the fork's one-token kernels make plain decoding fast while MTP's
  verify steps still run on llama.cpp's stock multi-token kernels.

## Quality of the common files

Against the Q8_0 file, 80 × 512 tokens of wikitext-2 ([Q82](../../quants/experiments.md#registry)):

| File | Size | Same top token | Mean KL divergence | Perplexity change |
|---|---|---|---|---|
| Unsloth UD-Q4_K_XL | 22.4 GB | 95.4 % | 0.013 | +0.4 % |
| bartowski Q4_0 | 20.8 GB | 92.0 % | 0.036 | |
| Our mix A ([recipe](../../quants/recipes/qwen36-mix-a.txt)) | | 94.1 % | 0.020 | |
| Our mix B ([recipe](../../quants/recipes/qwen36-mix-b.txt)) | | 92.8 % | 0.029 | |

Unsloth's UD-Q4_K_XL is the quality bar for this model. Its speed on the rig is not measured yet: its Q4_K and Q6_K
tensors run on llama.cpp's own kernels until the Pascal kernels below land.

## What's next

1. **A rig model matrix** of Qwen3.6, Qwen3-30B-A3B and a dense control on the v2 engine against stock llama.cpp, with no
   new code ([Q148](../../quants/experiments.md#registry)).
2. **A Pascal Q6_K kernel** (built, waiting for its rig gate): every Q4 file keeps its 248,320-token output head in Q6_K;
   est. +12–18 % on the Q4_0 file ([Q101](../../quants/experiments.md#registry)).
3. **A Pascal Q4_K kernel** for Q4_K_M and UD files, est. 1.5–1.7× on Q4_K_M together with Q6_K
   ([Q149](../../quants/experiments.md#registry)).
4. **Hybrid-model support** for the on-card attention mask and for pipelined speculation, which keeps a group of
   recurrent states per pipe slot so that wrong guesses can be dropped: built, output identical to plain decoding in the
   CPU and GPU checks, rig timing next ([Q162](../../quants/experiments.md#registry)).
5. **GPU sampling for tool requests**, worth more here than on Flint: Qwen3.6's 993 KB score row per token would no
   longer cross the PCIe 1.0 link.

Per-trick details and estimates: [model branches](../../docs/research/notes/model_branches.md), section 2.1.
