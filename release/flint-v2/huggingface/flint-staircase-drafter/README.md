---
license: other
license_name: OWNER-TBD
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

# Flint staircase drafter: guessed tokens for pipelined speculation on multi-GPU layer splits

A small trained head (two transformer layers, 40,000 draft ids) that guesses [Flint](https://huggingface.co/RecursiveSI/flint)'s
next token from the hidden states Flint has already computed for the tokens in flight on a layer split. It is the
guesser of the [Pascal engine](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs)'s pipelined speculation:
while the later cards finish token t, guessed tokens t+1, t+2, … already run on the earlier cards, and the last card
keeps a guess only if it equals the token it samples. The output is exactly plain decoding's; good guesses make it
faster.

This project is not affiliated with Flint's makers.

| File | Size | SHA-256 |
|---|---|---|
| `flint-staircase-drafter-v2.0.gguf` | RIG-TBD (~0.29 GB) | RIG-TBD |

## How it works

On a split over several cards only one card works at a time. When the drafter guesses, the newest token in flight
has passed only the first card, the one before it the second or third, older ones are complete: a staircase of
depths. The drafter reads exactly those hidden states (each token's state at the depth it has reached), plus the
committed tokens' final states, and predicts the next token. It runs on the first card in its own CUDA stream; a
hidden state that arrives late makes a guess late, never wrong. The copy lookup (the token that followed the longest
earlier match of the text) guesses first inside copied text, and the store of Flint's earlier answers next.

## Card counts

RIG-TBD: the shipped head. It is either depth-general (trained on every split of 3 to 12 cards, a quarter of its
training each on six and nine cards) or a head for six and nine cards. The engine builds the window of hidden states
from the run's own layer split and maps each depth to the nearest trained layer at or below it, so it works on any
split of 3–12 cards; the nine-card split with a light last card (`-ts 6,6,6,6,6,6,6,5,2`) is the one it was picked on.

## Enable it

It needs the Pascal engine v2 (llama-server), release archives RIG-TBD:

```bash
llama-server -m flint-K2h-r16.gguf -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 3 --pipe-guess staircase --pipe-draft flint-staircase-drafter-v2.0.gguf --pipe-store <store file> --host 127.0.0.1 --port 8080
```

- It loads with any Flint GGUF (K2h-r16, Unsloth's UD-Q4_K_XL or Q8_0). It was trained on Flint Q8_0's hidden states;
  the numbers below are with K2h-r16.
- `--pipe-store` (RIG-TBD: whether the store file ships with the release) adds the store of Flint's earlier answers in
  front of the drafter.
- One conversation at a time: the drafter guesses for the request that generates alone, with GPU sampling (`-bs`) and
  without repetition penalties; tool requests run pipelined until the tool call.
- `-lv 4` logs the split and the window the drafter uses. `--pipe-draft-dev N` moves it to another card;
  `LLAMA_INPUT_ON_GPU=0` is needed when that card holds no layers.

## Measured

Acceptance (the share of guesses that equal Flint's sample at the makers' sampling, coupled), on 38 held-out
transcripts:

| Cards | Fresh text (outside copied spans) | Overall |
|---|---|---|
| 9 (6/6/6/6/6/6/6/5/2) | RIG-TBD | RIG-TBD |
| 9 (even split) | RIG-TBD | RIG-TBD |
| 6 | RIG-TBD | RIG-TBD |

Served speed, tokens per second, median per request (chat, code, edit and tool prompts at the makers' sampling),
K2h-r16 on P106-090 cards:

| | No guessing | Copy lookup and store | **With the drafter** |
|---|---|---|---|
| Nine cards | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Six cards | RIG-TBD | RIG-TBD | **RIG-TBD** |

One guess costs RIG-TBD ms on the first card. Every answer was identical to plain decoding (RIG-TBD).

## How it was made

Flint's own answers (74.66 M answer tokens at the makers' sampling) to 76,431 prompts from public datasets
(ShareGPT, UltraChat, Magicoder, Hermes function calling, NuminaMath, OpenR1-Math, ToolACE, glaive-function-calling-v2)
and source-file edit and agent tasks; hidden states and top-20 targets dumped from Flint Q8_0 at 26 layers; trained in
PyTorch with a warm-start chain; exported as Q8_0 with an F16 output head. Every command, the data sizes and the checks:
[reproduce/drafter.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md).

## Licence

OWNER-TBD. The head is trained on Flint's outputs and hidden states (Flint is Apache-2.0); the prompts behind those
outputs come from datasets with their own licences.
