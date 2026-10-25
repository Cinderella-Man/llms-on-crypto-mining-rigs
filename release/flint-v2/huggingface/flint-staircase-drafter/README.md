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
conversations. Every answer with the drafter must be identical to the answer without it. Every number with its command:
[results.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/results.md).

## How it was made, and how to make it yourself

Flint wrote answers to prompts from public datasets whose answers may be published, and to code tasks built from
open-source files; we recorded what Flint computes inside for each token and trained the drafter to predict the next
token from it (RIG-TBD: the number of prompts and answer tokens). The whole recipe, from the prompts to the exported
file, with every command, the compute time and the data sizes:
[the drafter rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md).
The design and every training run so far:
[docs/drafter.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/drafter.md). The whole
release in one folder: [release/flint-v2](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/release/flint-v2).

## Licence and credits

- Apache-2.0 ([LICENSE](LICENSE)). The drafter is trained on Flint's outputs and internal states (Flint is Apache-2.0,
  by RecursiveSI); the prompts behind those outputs come from public datasets, listed with their licences in
  [the drafter rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md#1-prompts).
- Made by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X) for
  [LLMs on crypto-mining rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs). Not affiliated with
  RecursiveSI.

| File | Size | SHA-256 |
|---|---|---|
| RIG-TBD (the nine-card file) | RIG-TBD (~0.27 GB) | see [CHECKSUMS](CHECKSUMS) |
| RIG-TBD (the general file) | RIG-TBD (~0.27 GB) | see [CHECKSUMS](CHECKSUMS) |

Check a download with `sha256sum -c CHECKSUMS`; the same checksums are in the
[GitHub repository](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/CHECKSUMS).
