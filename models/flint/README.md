# Flint on Pascal mining rigs

[![Flint](https://img.shields.io/badge/Flint-v2%20measured-2b6cb0)](#speed)

[Flint](https://huggingface.co/RecursiveSI/flint) is a 39.8 B-parameter mixture-of-experts model that uses 2.5 B
parameters per token, keeps most of its attention local, and reads up to 262,144 tokens of context. It reasons before
every answer, so replies run to 600–7,900 tokens. It fits on nine 6 GB cards with its full context and is this
project's flagship: every engine change was developed and measured on it. Flint is released under Apache-2.0 by its
makers; this project is not affiliated with them.

Speeds on this page were measured on the [reference rig](../../docs/mining_rig.md): nine P106-090 cards, each on a PCIe
1.0 x1 link, and a 2-core Celeron with 4 GB of RAM. Quality was scored on a separate machine.

## Recommended setup

| | |
|---|---|
| Engine | the v2 Pascal engine ([how to get it](../../README.md#run-it)) |
| Model file | Unsloth's [UD-Q4_K_XL](https://huggingface.co/unsloth/flint-GGUF) (25.1 GB), the file most people already have |
| Cards | all nine for the full 262,144-token context; six run one short conversation about as fast and leave three cards free |
| Sampling | the makers' settings: temperature 1.0, top-p 0.95, top-k 20, min-p 0 |

Unsloth's [Q8_0](https://huggingface.co/unsloth/flint-GGUF) (42.3 GB) is the near-lossless alternative: on the v1
engine it ran as fast as UD-Q4_K_XL ([below](#how-it-got-here)); on v2 it is not measured yet. **K2h-r16**, this
project's own Flint file, is ~4 % faster than UD-Q4_K_XL at better measured fidelity; it is not published yet.

## Command lines

Chat with the full context on nine cards, served on the rig's loopback address:

```bash
LLAMA_NO_MOE_DOWN_RESCALE=tg llama.cpp/build/bin/llama-server -m flint-UD-Q4_K_XL.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 3 --host 127.0.0.1 --port 8080
```

- **`LLAMA_NO_MOE_DOWN_RESCALE=tg`** switches off a safety rescale that llama.cpp applies to Flint's expert layers. The
  Pascal kernels sum in 32-bit floats, so one-token steps don't need it, and without it each expert layer runs as one
  persistent kernel. Use it only with this engine on Pascal cards.
- **`--pipe-depth 3`** sends up to three guessed tokens through the idle cards while the later cards finish the current
  one ([pipelined speculation](#pipelined-speculation)). The output is the same as without it. It applies to requests
  sampled entirely on the GPU (`-bs`, no grammar, no repetition penalties); every other request decodes as usual.
- **`-bs`** samples on the GPU instead of copying 600 KB of scores per token over the slow link.
- **`-b 512 -ub 128`, `--cache-ram 0` and four checkpoints** keep the 4 GB host from running out of memory at this
  context. llama.cpp's default micro-batch of 512 pins ~1 GiB of host memory at 262,144 tokens.
- **Agents** (pi, OpenCode): the same line. Every tool request carries a grammar, which makes llama-server sample it on
  the CPU, so tool requests decode 12–16 % slower than chat and skip pipelined speculation. Set the client's context
  window equal to `-c`.
- **Six cards:** add `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` and lower `-c` (UD-Q4_K_XL fits est. ~128k tokens there).
- **Q8_0** at the full context needs its layers split by memory: add `-ts 6,6,5,5,6,5,5,6,6`.

Open the built-in web UI through an SSH tunnel from your PC: `ssh -N -L 8080:127.0.0.1:8080 <rig>`, then
`http://127.0.0.1:8080`.

## Speed

Generation speed in tokens per second (tok/s) on nine cards. `llama-bench` numbers are 128 generated tokens after a
warm-up, because idle cards sit at a lower clock.

| Engine | UD-Q4_K_XL | K2h-r16 (ours) |
|---|---|---|
| Stock llama.cpp (`d77256999`) | 16.5 | |
| v1 engine (released 2026-10-23) | 32.6 | |
| v2: + FP32 activation sums, shared-expert fusion | 36.5 | |
| v2: + prompt-processing changes | 37.5 | 39.3 |
| **v2: + persistent MoE kernel** | **39.8** | **41.5** |
| v2 with pipelined speculation | | **41–47** (by prompt, below) |

- The first three rows come from one session, the next two from another
  ([Q117, Q88, Q125](../../quants/experiments.md#registry)).
- On six cards the persistent kernel takes UD-Q4_K_XL from 35.8 to 38.2 tok/s and K2h-r16 from 37.4 to 40.7.
- **Served** (llama-server, `-bs`, one conversation, before the persistent kernel): chat 31.4–34.0 tok/s, requests
  with tools 29.3–29.6 ([Q118, Q55](../../quants/experiments.md#registry)).

### Pipelined speculation

On a layer split only one card works at a time. Pipelined speculation feeds the idle cards guessed tokens: a guess for
token t+1 enters the first card while the later cards still finish token t, and each guess is checked by sampling the
real token, so the output is plain decoding's output. The guesses come from the conversation itself: the token that
followed the longest earlier match of the current text, or else the most frequent continuation of the last few tokens.

K2h-r16, 512 generated tokens per prompt, the fork's `llama-pipespec` tool, tok/s
([Q87](../../quants/experiments.md#registry)):

| Prompt | 9 cards, plain | 9 cards, pipelined (depth) | 6 cards, plain | 6 cards, pipelined (depth) |
|---|---|---|---|---|
| Chat | 31.0 | **42.7** (5) | 32.6 | **44.8** (3) |
| Code | 31.1 | **44.2** (5) | 34.5 | **46.2** (3) |
| Edit a file | 26.4 | **40.9** (2) | 28.4 | **39.9** (3) |
| Rewrite a text | 35.9 | **46.7** (6) | 34.1 | **47.3** (4) |

- **Part of the gain is the clock.** In plain decoding each card idles most of every token and drops to its 1,354 MHz
  base clock; with every guess wrong, nine cards still run 1.24× faster. At matched clocks the guesses give 1.10–1.3×
  ([idle cards](../../docs/research/notes/idle_cards.md)).
- **A trained guesser would pay much more.** With 70 % of guesses right the same pipeline runs 82–84 tok/s.
- These runs used a build without the persistent kernel; the v2 engine has both.

### Prompt processing

UD-Q4_K_XL on nine cards, tok/s for prompts of 512 / 2,048 / 8,192 tokens: stock llama.cpp 185 / 493 / 593, v1
187 / 494 / 593, v2 228 / 613 / 723. The gain comes from upstream llama.cpp's PR #31047, which sizes the MoE tiles by
tokens per expert ([Q117](../../quants/experiments.md#registry)).

## Quality

Against Flint's original 16-bit weights, 80 × 512 tokens of wikitext-2:

| File | Size | Same top token | Mean KL divergence | Perplexity change |
|---|---|---|---|---|
| Unsloth Q8_0 | 42.3 GB | 93.3 % | 0.052 | +3.3 % |
| Unsloth UD-Q4_K_XL | 25.1 GB | 87.7 % | 0.120 | +3.9 % |
| K2h-r16 (ours) | not published | 88.8 % | 0.105 | +2.7 % |

- **K2h-r16** is a mix of Q5_0 and Q8_0 tensors, formats the Pascal kernels decode fast, made from the original weights
  with Unsloth's importance matrix, with a GPTQ-rounded Q5_0 output head and a 16-bit router. The same tensors with a
  32-bit router score level with UD-Q4_K_XL on 3,477 single-turn tool-calling tasks (BFCL v4): +0.14 points, 90 %
  interval −0.87 to +1.16 ([Q41](../../quants/experiments.md#registry)). How the files were found:
  [quants/](../../quants/README.md).
- **The engine doesn't change the answers.** Its kernels change only the order of additions: outputs match stock
  llama.cpp within rounding, the v2 additions leave the output unchanged (logits byte-identical in every check, and the
  same perplexity to every printed digit with the FP32 sums), and pipelined speculation gives the same tokens as plain
  decoding.

## How it got here

**v1** (2026-10-23) doubled Flint's speed over stock llama.cpp. The
[five-turn chat benchmark](../../docs/benchmarks/2026-10-23-flint-chat.md) ran one conversation on nine cards with the
full context, same settings and seeds on both engines:

| Model file | Stock llama.cpp | v1 engine | Speed-up |
|---|---|---|---|
| Unsloth UD-Q4_K_XL | 14.20 tok/s | 27.97 tok/s | 1.97× |
| Unsloth Q8_0 | 14.78 tok/s | 29.55 tok/s | 2.00× |
| Miner S4-L0 (ours, the v1 mix) | 16.06 tok/s | 31.86 tok/s | 1.98× |

v1's steps, `llama-bench` on UD-Q4_K_XL, six cards:

| Change | Generation |
|---|---|
| Stock llama.cpp (`d77256999`) | 17.6 tok/s |
| CUDA graphs on Pascal (llama.cpp switches them off for every GPU before Volta; [upstream PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)) | 19.5 tok/s |
| 2 warps per block for one-token products on wide rows, 4 for narrow ones | 24.5 tok/s |
| Pascal matrix-vector kernels for Q8_0, Q4_0, Q4_1, Q5_0, Q5_K and IQ4_XS ([design](../../docs/research/notes/pascal_mmvq_integration.md)) | 34.1 tok/s |

**v2** added, on Flint:
- **FP32 activation sums** in the Pascal kernels, so the expert rescale can go for every file: +9.6 %
  ([note](../../docs/research/notes/v2_synthesis.md)).
- **The shared expert fused** with its activation: +0.7 %.
- **PR #31047** for prompts: +22–29 % prompt processing.
- **The persistent MoE kernel:** each expert layer at one token runs as one kernel instead of 13 launches: +5.6–8.8 %
  per token. It covers Flint's shapes and falls back to the normal path for any other model.
- **Pipelined speculation** in llama-server (`--pipe-depth`).

## Not measured yet

- Served speed with `--pipe-depth`; the tok/s above for pipelined speculation come from `llama-pipespec`.
- Generation deep into a long context on v2. The v1 engine generated 23.3 tok/s at 32,768 tokens of context with
  Miner S4-L0 ([Q54](../../quants/experiments.md#registry)).
- Coding-agent tasks end to end, and runs longer than an hour.

## More

- [Benchmarks](../../docs/benchmarks/): [Flint, first measurements](../../docs/benchmarks/2026-10-22-flint.md) (files,
  quality against speed), [the five-turn chat](../../docs/benchmarks/2026-10-23-flint-chat.md).
- [quants/](../../quants/README.md): making Flint files for these cards; [recipes](../../quants/recipes/); every
  experiment in the [registry](../../quants/experiments.md).
- Research notes: [v2 synthesis](../../docs/research/notes/v2_synthesis.md),
  [pipelined speculation](../../docs/research/notes/pipelined_speculation.md),
  [release settings](../../docs/research/notes/release_v2.md), [all notes](../../docs/research/notes/README.md).
