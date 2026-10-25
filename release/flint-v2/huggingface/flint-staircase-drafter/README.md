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
guesser trained on millions of Flint's own answers. It reads what Flint has already worked out for the words in flight,
which the simple guessers (copying from earlier text) can't do.

**How much faster.** Nine P106-090 mining cards, Flint K2h-r16, the median of chat, code, edit and tool requests:
50.4 tokens per second with the drafter, against 48.6 with the simpler guessers alone and 33.4 without guessing. The
drafter's own share is the step from 48.6 to 50.4 (+3.7 %, up to +8.6 % on chat and code); most of the 1.5× comes from
pipelined speculation with the copy lookup and the store.

## Use it

You need the Pascal engine v2 and a Flint model file: Unsloth's near-lossless
[Q8_0](https://huggingface.co/unsloth/flint-GGUF) on nine cards (the drafter learned from Q8_0's internal states), or
[Flint K2h-r16](https://huggingface.co/CinderellaMan/Flint-K2h-r16-GGUF), which the numbers below were measured with, on
six cards or more. Add three options to the usual server command:

```bash
llama-p106-v2.0/llama-server -m flint-K2h-r16.gguf -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 3 --pipe-guess staircase --pipe-draft flint-staircase-drafter-9cards-v2.0.gguf --pipe-store <store file> --host 127.0.0.1 --port 8080
```

- `--pipe-depth 3` keeps three guesses in flight, `--pipe-guess staircase` uses the drafter, `--pipe-draft` names the
  file: the nine-card file on nine cards, the general file on any other count (below).
- For Q8_0, give `-m flint-Q8_0.gguf` and add `-ts 6,6,5,5,6,5,5,6,6`, which splits its layers by memory so that the
  full context fits (RIG-TBD: whether the drafter fits next to it at the full context).
- It helps one conversation at a time. Requests with tools are sped up until the tool call starts.
- Use it with `--pipe-store`: the store of Flint's earlier answers (OWNER-TBD: shipped next to the drafter files) is
  consulted before the drafter. Without the store the drafter alone reaches 44.4 tokens per second on nine cards,
  less than the store alone (48.6); together they reach 50.4.
- The full command-line guide: [the Flint page](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/models/flint/README.md).

## Card counts

| File | Use it on | Trained on |
|---|---|---|
| `flint-staircase-drafter-9cards-v2.0.gguf` | nine cards | both nine-card splits (even, and with a light last card) |
| `flint-staircase-drafter-v2.0.gguf` | six cards, and any other count from 3 to 12 | every split of 3–12 cards, a quarter of its examples each on six and nine cards |

The engine works out from your split which of Flint's layers each card holds and feeds the drafter the matching
inputs, so you don't configure anything. On nine cards the nine-card file guesses right a little more often than the
general one.

## What we measured

Tokens per second, Flint K2h-r16 at 8,192 tokens of context unless the row says otherwise, the median of eight
requests (chat, code, edit and a coding-agent tool request, each at temperature 0 and at the makers' sampling):

| | No guessing | Copy lookup and store | **Store, then the drafter** |
|---|---|---|---|
| **Nine cards, Unsloth Q8_0 at 262,144 tokens** (`-ts 6,6,5,5,6,5,5,6,6`), nine-card file | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Nine cards (`-ts 6,6,6,6,6,6,6,5,2`), nine-card file | 33.4 | 48.6 | **50.4** |
| Nine cards, even split, nine-card file | 33.4 | 48.6 | **50.3** |
| Six cards, general file | 34.3 | 49.0 | **49.5** |

- On nine cards 53.8 % of the drafter's guesses were right; on six cards it only breaks even with the store.
- One guess costs 5.3–7.3 ms on the first card, more than the design's 1.6–2.6 ms: that cost, not the guessing
  quality, now limits the gain.

On 38 held-out conversations, the drafter's guess equals Flint's next token 54 % of the time on nine cards (the
nine-card file) and 58 % on six cards (the general file); on text that can't be copied from earlier in the
conversation, 48 % and 52 %.

Every answer with the drafter was identical to the answer without it. Every number with its command:
[results.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/results.md).

## How it was made, and how to make it yourself

Flint wrote 74.7 million tokens of answers to 76,431 prompts from public datasets and code tasks; we recorded what
Flint computes inside for each token and trained the drafter to predict the next token from it. The whole recipe, from
the prompts to the exported file, with every command, the compute time and the data sizes:
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
| `flint-staircase-drafter-9cards-v2.0.gguf` | RIG-TBD (~0.27 GB) | see [CHECKSUMS](CHECKSUMS) |
| `flint-staircase-drafter-v2.0.gguf` | RIG-TBD (~0.27 GB) | see [CHECKSUMS](CHECKSUMS) |

Check a download with `sha256sum -c CHECKSUMS`; the same checksums are in the
[GitHub repository](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/CHECKSUMS).
