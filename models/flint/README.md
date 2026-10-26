# Flint on Pascal mining rigs

[![Flint](https://img.shields.io/badge/Flint-v2%20release-2b6cb0)](#speed)

[Flint](https://huggingface.co/RecursiveSI/flint) is a 39.8 B-parameter mixture-of-experts model that uses 2.5 B
parameters per token, keeps most of its attention local, and reads up to 262,144 tokens of context. It reasons before
every answer, so replies run to 600–7,900 tokens. It fits on nine 6 GB cards with its full context and is this
project's flagship: every engine change was developed and measured on it. Flint is released under Apache-2.0 by its
makers; this project is not affiliated with them.

Speeds on this page were measured on the [reference rig](../../docs/mining_rig.md): nine P106-090 cards, each on a PCIe
1.0 x1 link, and a 2-core Celeron with 4 GB of RAM. Quality was scored on a separate machine.

**The headline: Flint in near-lossless 8-bit, on all nine cards, at the full 262,144-token context.** Unsloth's Q8_0
(42.3 GB) picks the original model's next token 93.3 % of the time, against 87.7 % for the usual 4-bit download, and
on these cards it is also the faster file: in a five-turn chat at the full context the v1 engine generated 29.55 tok/s
with Q8_0 against 27.97 with UD-Q4_K_XL, and stock llama.cpp 14.78 (v2: RIG-TBD). The Pascal kernels decode Q8_0's
plain 8-bit blocks with few instructions per byte, at ~150 GB/s of a card's ~179, so reading 2.70 GB per token took no
longer than reading UD-Q4_K_XL's 2.21 GB.

## Recommended setup

| | |
|---|---|
| Engine | the v2 Pascal engine, commit `2d53179aa`: the release archives RIG-TBD (tag and link), or a build from source ([how](../../README.md#run-it)) | <!-- only:rc4 -->
| Engine | the v2 Pascal engine, commit `7e6ec762f`: the release archives RIG-TBD (tag and link), or a build from source ([how](../../README.md#run-it)) | <!-- only:rc3 -->
| Engine | the v2 Pascal engine, commit `594d20e08`: the release archives RIG-TBD (tag and link), or a build from source ([how](../../README.md#run-it)) | <!-- only:rc1 -->
| Model file, nine cards | Unsloth's [Q8_0](https://huggingface.co/unsloth/flint-GGUF) (42.3 GB): near-lossless, the full context fits |
| Model file, six cards | [Flint-Pascal-Q5](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF) (29.4 GB; recipe K2h-r16), this project's own file, better than UD-Q4_K_XL in quality and faster on the same engine |
| Cards | all nine for Q8_0 at the full 262,144-token context; six run Flint-Pascal-Q5 or UD-Q4_K_XL for one short conversation about as fast as nine, and leave three cards free |
| Sampling | the makers' settings: temperature 1.0, top-p 0.95, top-k 20, min-p 0 |

Unsloth's [UD-Q4_K_XL](https://huggingface.co/unsloth/flint-GGUF) (25.1 GB), the file most people already have, runs
too; Flint-Pascal-Q5 is ~4 % faster than it at better measured fidelity.

## Command lines

Chat with the full context on nine cards, served on the rig's loopback address, with the release archives unpacked
in `llama-p106-v2.0/` (RIG-TBD: the folder follows the release's tag; a build from source has the same binary in
`llama.cpp/build/bin/`):

```bash
llama-p106-v2.0/llama-server -m flint-Q8_0.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4 --host 127.0.0.1 --port 8080
```

- **`-ts 6,6,5,5,6,5,5,6,6`** splits Flint's layers by memory instead of by count, so that the 42.3 GB of weights and
  the full-context cache fit (llama.cpp's even split puts two full-attention layers on one card, ~6.3 GB). For
  Flint-Pascal-Q5 or UD-Q4_K_XL drop it and give the file's name instead.
- **`--pipe-depth 4`** sends up to four guessed tokens through the idle cards while the later cards finish the current
  one ([pipelined speculation](#pipelined-speculation)). The output is the same as without it. It applies to requests
  sampled on the GPU (`-bs`) without repetition penalties; tool requests run pipelined until the tool call starts.
- **The drafter:** add `--pipe-guess staircase --pipe-draft <the nine-card file>` (RIG-TBD: its name) for the trained guesser on nine cards ([below](#the-staircase-drafter)); RIG-TBD: whether the first card has room for it next to Q8_0 at the full context. <!-- only:drafter -->
- **`-bs`** samples on the GPU instead of copying 600 KB of scores per token over the slow link.
- **`-b 512 -ub 128`, `--cache-ram 0` and four checkpoints** keep the 4 GB host from running out of memory at this
  context. llama.cpp's default micro-batch of 512 pins ~1 GiB of host memory at 262,144 tokens. RIG-TBD: `-b 2048 -ub
  512` if the quick-start check shows it fits (faster prefill).
- **The engine drops llama.cpp's safety rescale of Flint's expert layers by itself** on Pascal cards: its kernels sum in
  32-bit floats, so one-token steps don't need it, and without it each expert layer runs as one persistent kernel.
  `LLAMA_NO_MOE_DOWN_RESCALE=0` keeps the rescale; any value other than `tg` and `0` is unsafe for this file.
- **Agents** (pi, OpenCode): the same line. Tool requests sample on the GPU too, with each token checked against the
  tool grammar on the host (RIG-TBD: their speed against chat). Set the client's context window equal to `-c`.
- **Agents, prompt cache:** add `--prefix-cache <dir>` so every conversation restores the shared system and tool prompt from a file instead of reading it again (the files outlive the server), and `--checkpoint-gen` so a client's re-rendered reply re-reads only from the last `</think>` or `<tool_call>`. RIG-TBD: the time to first token it saves. <!-- only:q163-q111 -->
- **Six cards:** Flint-Pascal-Q5 or UD-Q4_K_XL with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`, without `-ts`, and a lower `-c`
  (UD-Q4_K_XL fits est. ~128k tokens there). Q8_0 needs eight or nine cards.

Open the built-in web UI through an SSH tunnel from your PC: `ssh -N -L 8080:127.0.0.1:8080 <rig>`, then
`http://127.0.0.1:8080`.

## Speed

Generation speed in tokens per second (tok/s) on nine cards. `llama-bench` numbers are 128 generated tokens after a
warm-up, because idle cards sit at a lower clock.

### Q8_0 at the full context

| Nine cards, 262,144 tokens of context | Stock llama.cpp | v1 engine | **v2 release** |
|---|---|---|---|
| Five-turn chat, **Unsloth Q8_0** (`-ts 6,6,5,5,6,5,5,6,6`) | 14.78 | 29.55 | **RIG-TBD** |
| Five-turn chat, Unsloth UD-Q4_K_XL | 14.20 | 27.97 | RIG-TBD |
| Served, Q8_0, plain decoding | — | — | **RIG-TBD** |
| Served, Q8_0, `--pipe-depth 3` (guesses from the copy lookup and the request's continuation counts) | — | — | **RIG-TBD** |
| Served, Q8_0, the store, then the drafter | — | — | **RIG-TBD** | <!-- only:drafter -->
| `llama-bench`, Q8_0 (512-token prompt, then 128 generated) | RIG-TBD | 35.9 | **RIG-TBD** |

- The chat rows are the [five-turn chat benchmark](../../docs/benchmarks/2026-10-23-flint-chat.md): one conversation,
  the same settings and seeds on both engines ([Q58](../../quants/experiments.md#registry)).
- The `llama-bench` v1 number is v1's kernels just before its release: 26.9 tok/s with them off
  ([Q38](../../quants/experiments.md#registry)).

### The 4-bit files

| Engine | UD-Q4_K_XL | Flint-Pascal-Q5 (ours) |
|---|---|---|
| Stock llama.cpp (`d77256999`) | 16.5 | |
| v1 engine (released 2026-10-23) | 32.6 | |
| v2: + FP32 activation sums, shared-expert fusion | 36.5 | |
| v2: + prefill changes | 37.5 | 39.3 |
| v2: + persistent MoE kernel | 39.8 | 41.5 |
| v2: + pipelined speculation in llama-server (branch `pascal-p106-v2-next`) | 39.9 | 41.8 |
| **v2 release** | **RIG-TBD** | **RIG-TBD** |

- The first three rows come from one session, the next two from another, the `v2-next` row from a third
  ([Q117, Q88, Q125, Q160](../../quants/experiments.md#registry)).
- On six cards the persistent kernel takes UD-Q4_K_XL from 35.8 to 38.2 tok/s and Flint-Pascal-Q5 from 37.4 to 40.7.

### Served

llama-server, one conversation, GPU sampling (`-bs`), median per request, tok/s
([Q160](../../quants/experiments.md#registry)):

| Build | UD-Q4_K_XL | Flint-Pascal-Q5 (ours) |
|---|---|---|
| v2-next, plain decoding | 36.6 | 39.2 |
| **v2-next, pipelined speculation (`--pipe-depth 3`)** | **47.3** | **48.5** |
| v2 release, plain decoding, the chat command line above | RIG-TBD | RIG-TBD |
| **v2 release, `--pipe-depth 3`, the chat command line above** | **RIG-TBD** | **RIG-TBD** |
| v2 release, a request with tools, `--pipe-depth 3` | RIG-TBD | RIG-TBD |

- The v2-next rows ran at 8,192 tokens of context: chat, rewrite and edit prompts, each at temperature 0 and with two
  seeds at the makers' sampling, plus one streamed request. Pipelined speculation gave 1.04–1.18× on chat, 1.04–1.56×
  on edits and 1.44–2.79× on rewrites, and all ten pipelined answers were identical to the server without it.
- Before the release build, requests that carry tools sampled on the CPU and decoded 12–16 % slower than chat
  (29.3–29.6 tok/s on an earlier v2 build, [Q55](../../quants/experiments.md#registry)); the release samples them on the
  GPU.

### Pipelined speculation

On a layer split only one card works at a time. Pipelined speculation feeds the idle cards guessed tokens: a guess for
token t+1 enters the first card while the later cards still finish token t, and each guess is checked by sampling the
real token, so the output is plain decoding's output. The guesses come from the conversation itself: the token that
followed the longest earlier match of the current text, or else the most frequent continuation of the last few tokens.

Nine cards, 256 generated tokens per prompt at temperature 0, the fork's `llama-pipespec` tool on the `v2-next` build,
tok/s, plain decoding → three / four guesses in flight ([Q160](../../quants/experiments.md#registry)):

| Prompt | UD-Q4_K_XL plain | depth 3 | depth 4 | Flint-Pascal-Q5 plain | depth 3 | depth 4 |
|---|---|---|---|---|---|---|
| Chat | 32.2 | 40.4 | 42.5 | 34.3 | 41.8 | 44.4 |
| Code | 35.2 | 41.3 | 43.4 | 34.7 | 44.2 | 47.0 |
| Edit a file | 31.8 | 44.2 | 44.6 | 36.3 | 45.3 | 46.2 |
| Rewrite a text | 35.9 | **60.7** | **63.1** | 39.4 | 51.7 | 52.2 |

- Every output was identical to plain decoding.
- **Part of the gain is the clock.** In plain decoding each card idles most of every token and drops to its 1,354 MHz
  base clock; with every guess wrong, nine cards still run 1.24× faster. At matched clocks the guesses give 1.10–1.3×
  ([idle cards](../../docs/research/notes/idle_cards.md)).
- **A trained guesser would pay much more.** With 70 % of guesses right the pipeline runs 82–84 tok/s
  ([Q87](../../quants/experiments.md#registry)).

<!-- begin:drafter -->
### The staircase drafter

The release ships a trained guesser for pipelined speculation: a small transformer that predicts Flint's next token
from the hidden states Flint has already computed for the tokens in flight. The newest token has passed only the first
card, older ones are further along: a staircase of depths. The drafter sits behind the copy lookup and runs on the
first card in its own CUDA stream; the last card still samples every token, so the output is plain decoding's. It
learned from Flint Q8_0's hidden states and works with any Flint file. How it is built and trained:
[docs/drafter.md](../../docs/drafter.md).

- **Files** (~0.27 GB each, from the release assets or
  [Hugging Face](https://huggingface.co/CinderellaMan/Flint-Staircase-Drafter); RIG-TBD: their names): one for nine
  cards, trained on the nine-card splits, and one for six cards and any other split of 3–12 cards, trained on all of
  them. Both are trained only on Flint's answers to prompts whose answers may be published
  ([the sources](../../release/flint-v2.1/reproduce/drafter.md#1-prompts)).
- **Turn it on:** add `--pipe-guess staircase --pipe-draft <file>` to the chat command line above, which already has
  `--pipe-depth 3`. The engine builds the drafter's window of hidden states from the run's own layer split.

Served, `--pipe-depth 3`, tok/s, the median of eight requests (chat, code and edit answers of 512 tokens and a
coding-agent tool request of 1,536, each at temperature 0 and at the makers' sampling):

| | No guessing | Copy lookup and continuations | With the store | **With the drafter** |
|---|---|---|---|---|
| **Nine cards, Q8_0 at 262,144 tokens**, `-ts 6,6,5,5,6,5,5,6,6`, nine-card file | RIG-TBD | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Nine cards, Flint-Pascal-Q5, nine-card file | RIG-TBD | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Six cards, Flint-Pascal-Q5, general file | RIG-TBD | RIG-TBD | RIG-TBD | **RIG-TBD** |

- RIG-TBD: the share of the drafter's guesses that are right, and its accuracy on held-out text.
- **One guess costs 5.3–7.3 ms on the first card** (measured with the first heads of the same size), which also runs
  Flint's first layers; the design estimated 1.6–2.6 ms. That cost, more than the guessing quality, limits the gain.
- The first heads, trained partly on prompts whose answers may not be published, are not shipped; their measurements
  are in [docs/drafter.md](../../docs/drafter.md).
- Every answer must be identical to plain decoding in every configuration.
<!-- end:drafter -->
**Coming later (v2.1):** a trained guesser, the staircase drafter, that predicts Flint's next token from the hidden states of the tokens in flight, and a store of Flint's earlier answers ([how the drafter works](../../docs/drafter.md)). <!-- only:nodrafter -->

### Prefill

Prefill is the speed at which the engine reads the prompt. UD-Q4_K_XL on nine cards, tok/s at 512 / 2,048 / 8,192
tokens: stock llama.cpp 185 / 493 / 593, v1 187 / 494 / 593, v2 228 / 613 / 723; so prefill at 8k tokens is 723 tok/s
against stock's 593. The gain comes from upstream llama.cpp's PR #31047, which sizes the MoE tiles by tokens per expert
([Q117](../../quants/experiments.md#registry)).

Prefill of long prompts with the release's prefill-attention kernel, Flint-Pascal-Q5 on nine cards: RIG-TBD tok/s at 8,192 tokens and RIG-TBD at 32,768, against RIG-TBD and RIG-TBD with it switched off (`GGML_CUDA_FATTN_TILE_PASCAL=0`). <!-- only:q163-q111 -->

### With hardware tuning

A memory clock offset of RIG-TBD MT/s on every card adds RIG-TBD % to v2's generation and RIG-TBD % to served chat on
UD-Q4_K_XL, with the output unchanged. The full matrix of stock llama.cpp, v1 and v2 with and without it is in the
[README's hardware-tuning section](../../README.md#hardware-tuning).

## Quality

Against Flint's original 16-bit weights, 80 × 512 tokens of wikitext-2:

| File | Size | Same top token | Mean KL divergence | Perplexity change |
|---|---|---|---|---|
| Unsloth Q8_0 | 42.3 GB | 93.3 % | 0.052 | +3.3 % |
| Unsloth UD-Q4_K_XL | 25.1 GB | 87.7 % | 0.120 | +3.9 % |
| Flint-Pascal-Q5 (ours) | 29.4 GB | 88.8 % | 0.105 | +2.7 % |

- **Flint-Pascal-Q5** is a mix of Q5_0 and Q8_0 tensors, formats the Pascal kernels decode fast, made from the original weights
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
| Unsloth Q8_0 | 14.78 tok/s | 29.55 tok/s | 2.00× |
| Unsloth UD-Q4_K_XL | 14.20 tok/s | 27.97 tok/s | 1.97× |
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
- **PR #31047**: prefill +22–29 %.
- **The persistent MoE kernel:** each expert layer at one token runs as one kernel instead of 13 launches: +5.6–8.8 %
  per token. It covers Flint's shapes and falls back to the normal path for any other model.
- **Pipelined speculation** in llama-server (`--pipe-depth`, with `--pipe-floor`).
- **A Pascal decode-attention kernel** for Flint's attention shape: RIG-TBD.
- **Tool requests sampled on the GPU**, the grammar checked on the host, pipelined until the tool call: RIG-TBD.
- **The attention mask built on each card**, for prefill and decoding: decode at 64k tokens of context +12 %
  ([Q116](../../quants/experiments.md#registry)).
- **Pascal kernels for Q4_K and Q6_K**, for other models' common downloads; est. ≤ 1 % on UD-Q4_K_XL.
- **A Pascal prefill-attention kernel** for long prompts, which also reads a Q8_0 cache directly: RIG-TBD. <!-- only:q163-q111 -->
- **Prompt caching for agents** (`--prefix-cache`, whole-ring checkpoints, `--checkpoint-gen`): RIG-TBD. <!-- only:q163-q111 -->
- **The staircase drafter**, a trained guesser for pipelined speculation, with its side files: RIG-TBD. <!-- only:drafter -->

## Not measured yet

- Served speed with `--pipe-depth` at the full 262,144-token context (the measured server ran at 8,192), and the release
  build's numbers (RIG-TBD: delete this item once today's run is in).
- Generation deep into a long context on v2. The v1 engine generated 23.3 tok/s at 32,768 tokens of context with
  Miner S4-L0 ([Q54](../../quants/experiments.md#registry)).
- Coding-agent tasks end to end, and runs longer than an hour.

## More

- [release/flint-v2](../../release/flint-v2/README.md): everything about this release in one folder, with the recipes
  that rebuild the engine and Flint-Pascal-Q5 from scratch and every number behind the release.
- [Benchmarks](../../docs/benchmarks/): [Flint, first measurements](../../docs/benchmarks/2026-10-22-flint.md) (files,
  quality against speed), [the five-turn chat](../../docs/benchmarks/2026-10-23-flint-chat.md).
- [quants/](../../quants/README.md): making Flint files for these cards; [recipes](../../quants/recipes/); every
  experiment in the [registry](../../quants/experiments.md).
- Research notes: [v2 synthesis](../../docs/research/notes/v2_synthesis.md),
  [pipelined speculation](../../docs/research/notes/pipelined_speculation.md),
  [release settings](../../docs/research/notes/release_v2.md), [all notes](../../docs/research/notes/README.md).
