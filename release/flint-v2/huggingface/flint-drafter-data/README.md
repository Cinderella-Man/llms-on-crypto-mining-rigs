---
license: other
license_name: OWNER-TBD
task_categories:
  - text-generation
tags:
  - flint
  - speculative-decoding
  - synthetic
---

# Flint drafter data: what the staircase drafter was trained on

The answers Flint wrote to train the [staircase drafter](https://huggingface.co/CinderellaMan/Flint-Staircase-Drafter),
described exactly enough to regenerate them, and (OWNER-TBD) the answers themselves. Everything else the drafter needs
(the hidden-state dumps, 1.3 TB) is recomputed from these answers with one command per file; see
[the drafter rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md).

## What is here

| File | What | Size |
|---|---|---|
| `manifest.json` | one entry per answer file: the writer (model file and SHA-256, engine and commit, server flags), the prompt file, the sampling, the number of answers and answer tokens, the file's SHA-256 | small |
| `prompts/*.jsonl` | OWNER-TBD: the four prompt files (76,431 prompts) | ~0.3 GB |
| `answers/gen-*.jsonl` | OWNER-TBD: Flint's answers with exact token ids (71,711 answers, 74.66 M answer tokens) | ~0.6 GB |

The prompt files are rebuilt with `prompts.py` and `tool_prompts.py` from public datasets (commands in the guide), so
publishing them is optional. The answers can be regenerated with the same writers and seeds, but sampled text from
GPUs differs between machines in the details, so only the published answers reproduce the drafter's exact training
set.

## Where the answers come from

| Writer | Engine | Answers | Answer tokens |
|---|---|---|---|
| Unsloth's Flint Q8_0 | llama.cpp (llama-server) | 7,148 | 8.61 M |
| G3, this project's mix | llama.cpp | 14,615 | 9.58 M |
| Unsloth's UD-Q4_K_XL | llama.cpp | 7,763 | 3.73 M |
| The makers' NVFP4 GGUF | vLLM | 42,185 | 52.74 M |

Sampling for every writer: temperature 1.0, top-k 20, top-p 0.95, min-p 0, reasoning kept, at most 2,048 answer
tokens, seed 20000 + the prompt's number.

## Licence

OWNER-TBD. The answers are Flint's outputs (Flint is Apache-2.0, by RecursiveSI). The prompts come from ShareGPT,
UltraChat, Magicoder, Hermes function calling, NuminaMath, OpenR1-Math, ToolACE and glaive-function-calling-v2 (each
under its own licence) and from source files of the Python standard library and llama.cpp; the answer files contain the
prompts' token ids.

Made by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X) for
[LLMs on crypto-mining rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs).
