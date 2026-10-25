---
license: apache-2.0
base_model: RecursiveSI/flint
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - flint
  - speculative-decoding
  - pascal
  - mining-rig
---

# Flint staircase drafter: a trained guesser that makes Flint faster on multi-GPU rigs

**What this is.** A small add-on file (about 0.27 GB, in two versions for different card counts) for running [Flint](https://huggingface.co/RecursiveSI/flint) with
the [Pascal engine](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs) on several graphics cards. It guesses
Flint's next words, and the cards check every guess. Right guesses come almost for free; wrong ones are thrown away. The
answers are exactly the ones Flint gives without it, only faster.

**Why it helps on a rig.** When a model is split across several cards, each card holds a slice of its layers, and for
every word only one card works at a time while the others wait. The engine fills the waiting cards with guessed next
words ("pipelined speculation"). The better the guesses, the more of that idle time turns into speed. This file is a
guesser trained on Flint's own answers. It reads what Flint has already worked out for the words in flight, which the
simple guessers (copying from earlier text) can't do.

**How much faster.** Nine P106-090 mining cards, the median of chat, code, edit and tool requests: RIG-TBD tokens per
second with the drafter, against RIG-TBD with the simpler guessers alone and RIG-TBD without guessing.

## Use it

You need the Pascal engine v2 and a Flint model file: Unsloth's near-lossless
[Q8_0](https://huggingface.co/unsloth/flint-GGUF) on nine cards (the drafter learned from Q8_0's internal states), or
[Flint K2h-r16](https://huggingface.co/CinderellaMan/Flint-K2h-r16-GGUF) on six cards or more. Add three options to the
usual server command:

```bash
llama-p106-v2.0/llama-server -m flint-K2h-r16.gguf -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 3 --pipe-guess staircase --pipe-draft <nine-card file> --host 127.0.0.1 --port 8080
```

- `--pipe-depth 3` keeps three guesses in flight, `--pipe-guess staircase` uses the drafter, `--pipe-draft` names the
  file: the nine-card file on nine cards, the general file on any other count (below).
- For Q8_0, give `-m flint-Q8_0.gguf` and add `-ts 6,6,5,5,6,5,5,6,6`, which splits its layers by memory so that the
  full context fits (RIG-TBD: whether the drafter fits next to it at the full context).
- It helps one conversation at a time. Requests with tools are sped up until the tool call starts.
- `--pipe-store <file>` adds a store of Flint's earlier answers, consulted before the drafter (OWNER-TBD: whether the
  store file ships; RIG-TBD: what it adds).
- The full command-line guide: [the Flint page](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/models/flint/README.md).

## Card counts

| File | Use it on | Trained on |
|---|---|---|
| RIG-TBD (the nine-card file) | nine cards | the nine-card splits |
| RIG-TBD (the general file) | six cards, and any other count from 3 to 12 | every split of 3–12 cards |

The engine works out from your split which of Flint's layers each card holds and feeds the drafter the matching
inputs, so you don't configure anything.

## What we measured

Tokens per second, the median of eight requests (chat, code, edit and a coding-agent tool request, each at temperature
0 and at the makers' sampling):

| | No guessing | Copy lookup and continuations | With the store | **With the drafter** |
|---|---|---|---|---|
| **Nine cards, Unsloth Q8_0 at 262,144 tokens** (`-ts 6,6,5,5,6,5,5,6,6`), nine-card file | RIG-TBD | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Nine cards, K2h-r16, nine-card file | RIG-TBD | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Six cards, K2h-r16, general file | RIG-TBD | RIG-TBD | RIG-TBD | **RIG-TBD** |

RIG-TBD: the share of the drafter's guesses that were right, the cost of one guess, and its accuracy on 38 held-out
conversations. Every answer with the drafter must be identical to the answer without it. Every run with its numbers:
[docs/drafter.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/drafter.md).

## How it was made, and how to make it yourself

Flint wrote answers to code tasks built from open-source files and to prompts from public datasets whose answers may
be published; we recorded what Flint computes inside for each token and trained the drafter, from scratch, to predict
the next token from it (RIG-TBD: the number of answers and answer tokens the release heads saw).

| Prompts from | Licence | Prompts for |
|---|---|---|
| The Python standard library and llama.cpp's own source files | PSF License 2.0; llama.cpp MIT, with some files under Apache-2.0 or Apache-2.0 with LLVM exception | whole-file edits and coding-agent turns |
| Packages installed in the Python image, only those allowed below | MIT, BSD-2/3-Clause, Apache-2.0, PSF | whole-file edits and coding-agent turns |
| open-r1/OpenR1-Math-220k (`default` subset): crawled competition problems, without the DeepSeek-R1 solutions | Apache-2.0 | reasoning |
| OpenAssistant/oasst2: written by volunteers; root prompts only, `synthetic == False`, reviewed | Apache-2.0 | chat |
| Code-generation prompts | RIG-TBD: a source may be added after a research note now running | code |

Excluded: ShareGPT, UltraChat and Magicoder, and NuminaMath-CoT (about 45 % of its problems written by GPT-4),
NousResearch/hermes-function-calling-v1, Team-ACE/ToolACE and glaiveai/glaive-function-calling-v2 (written by language
models they do not name). Prompts that share text with the Q51 test set are left out. RIG-TBD: which of the clean
sources the release heads were trained on.

<details>
<summary>Site-packages, per package</summary>

Agent and whole-file edit prompts built from source files of packages installed in the Python image. Licence as the
package's METADATA states it (versions as installed when checked, 2026-10-25); pip's vendored packages and pyarrow's
vendored headers judged one by one. Allowed: MIT, BSD-2/3-Clause, Apache-2.0, PSF.

| Package | Licence | Included | Answers | Answer tokens |
|---|---|---|---|---|
| pyarrow 26.0.0 | Apache-2.0 | yes | 2833 | 2.803 M |
| huggingface_hub 2.3.0 | Apache-2.0 | yes | 837 | 0.867 M |
| pip 25.0.1 | MIT | yes | 688 | 0.751 M |
| pip/_vendor/rich (vendored in pip 25.0.1) | MIT | yes | 405 | 0.442 M |
| tqdm 4.70.1 | MPL-2.0 AND MIT | no | 172 | 0.212 M |
| pip/_vendor/pygments (vendored in pip 25.0.1) | BSD-2-Clause | yes | 164 | 0.175 M |
| fsspec 2026.9.0 | BSD-3-Clause | yes | 147 | 0.170 M |
| httpcore2 2.16.0 | BSD-3-Clause | yes | 155 | 0.160 M |
| anyio 4.15.1 | MIT | yes | 161 | 0.160 M |
| pip/_vendor/urllib3 (vendored in pip 25.0.1) | MIT | yes | 122 | 0.137 M |
| httpx2 2.16.0 | BSD-3-Clause | yes | 110 | 0.117 M |
| pip/_vendor/requests (vendored in pip 25.0.1) | Apache-2.0 | yes | 98 | 0.109 M |
| PyYAML 6.0.3 | MIT | yes | 96 | 0.100 M |
| pip/_vendor/cachecontrol (vendored in pip 25.0.1) | Apache-2.0 | yes | 86 | 0.092 M |
| pyarrow/include/arrow/vendored/double-conversion | BSD-3-Clause | yes | 76 | 0.075 M |
| pip/_vendor/packaging (vendored in pip 25.0.1) | Apache-2.0 OR BSD-2-Clause | yes | 65 | 0.072 M |
| packaging 26.3 | Apache-2.0 OR BSD-2-Clause | yes | 46 | 0.059 M |
| h11 0.16.0 | MIT | yes | 50 | 0.055 M |
| idna 3.20 | BSD-3-Clause | yes | 42 | 0.044 M |
| pip/_vendor/truststore (vendored in pip 25.0.1) | MIT | yes | 34 | 0.040 M |
| pip/_vendor/resolvelib (vendored in pip 25.0.1) | ISC | no | 41 | 0.040 M |
| click 8.5.0 | BSD-3-Clause | yes | 31 | 0.034 M |
| filelock 4.2.4 | MIT | yes | 32 | 0.034 M |
| pip/_vendor/msgpack (vendored in pip 25.0.1) | Apache-2.0 | yes | 33 | 0.034 M |
| pip/_vendor/idna (vendored in pip 25.0.1) | BSD-3-Clause | yes | 28 | 0.031 M |
| truststore 0.10.5 | MIT | yes | 31 | 0.030 M |
| pyarrow/include/arrow/vendored/datetime | MIT | yes | 22 | 0.024 M |
| pip/_vendor/platformdirs (vendored in pip 25.0.1) | MIT | yes | 21 | 0.021 M |
| pip/_vendor/distlib (vendored in pip 25.0.1) | PSF-2.0 | yes | 18 | 0.015 M |
| pyarrow/include/arrow/vendored/portable-snippets | CC0-1.0 | no | 11 | 0.014 M |
| pip/_vendor/__init__.py (vendored in pip 25.0.1) | MIT (pip's own) | yes | 10 | 0.014 M |
| pip/_vendor/certifi (vendored in pip 25.0.1) | MPL-2.0 | no | 12 | 0.013 M |
| pip/_vendor/distro (vendored in pip 25.0.1) | Apache-2.0 | yes | 13 | 0.013 M |
| pyarrow/include/arrow/vendored/datetime.h | Apache-2.0 (Arrow's own) | yes | 9 | 0.010 M |
| pip/_vendor/tomli (vendored in pip 25.0.1) | MIT | yes | 10 | 0.009 M |
| pyarrow/include/arrow/vendored/strptime.h | unclear (musl/NetBSD origin) | no | 8 | 0.008 M |
| pip/_vendor/pyproject_hooks (vendored in pip 25.0.1) | MIT | yes | 8 | 0.007 M |

Included: 6481 answers, 6.70 M answer tokens; excluded: 244 answers, 0.29 M.

</details>

The whole recipe, from the prompts to the exported file, with every command, the compute time and the data sizes:
[the drafter rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2.1/reproduce/drafter.md).
The design and every training run so far:
[docs/drafter.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/drafter.md). Everything
about the drafter's release in one folder:
[release/flint-v2.1](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/release/flint-v2.1).

## Licence and credits

- Apache-2.0 ([LICENSE](LICENSE)). The drafter is trained on Flint's outputs and internal states (Flint is Apache-2.0,
  by RecursiveSI); the prompts behind those outputs come from public datasets, listed with their licences in
  [the drafter rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2.1/reproduce/drafter.md#1-prompts).
- Made by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X) for
  [LLMs on crypto-mining rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs). Not affiliated with
  RecursiveSI.

| File | Size | SHA-256 |
|---|---|---|
| RIG-TBD (the nine-card file) | RIG-TBD (~0.27 GB) | see [CHECKSUMS](CHECKSUMS) |
| RIG-TBD (the general file) | RIG-TBD (~0.27 GB) | see [CHECKSUMS](CHECKSUMS) |

Check a download with `sha256sum -c CHECKSUMS`; the same checksums are in the
[GitHub repository](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2.1/huggingface/flint-staircase-drafter/CHECKSUMS).
