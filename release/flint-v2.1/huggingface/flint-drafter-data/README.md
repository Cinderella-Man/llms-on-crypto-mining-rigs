---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - flint
  - speculative-decoding
  - synthetic
---

<!-- Draft for a later release, not part of v2.0: publishing the data is parked (2026-10-25). -->

# Flint drafter data: what the staircase drafter was trained on

The prompts, Flint's answers with their exact token ids, and the trained heads with their training checkpoints behind
the [staircase drafter](https://huggingface.co/CinderellaMan/Flint-Staircase-Drafter), a small model that guesses
Flint's next token so that a rig of old graphics cards runs Flint faster. Every prompt comes from a source whose
answers may be published. With this data and Unsloth's Flint Q8_0 anyone can redo the drafter's training, or continue
it from our checkpoints. Every step, command and data size is in
[the drafter rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2.1/reproduce/drafter.md).

## What is here

| Path | What | Size |
|---|---|---|
| `prompts/prompts*.strict.jsonl` | the three prompt files with the clean sources only and without Q51's text: 19,084 prompts, each with its id, category and source (RIG-TBD: the OpenAssistant prompts) | RIG-TBD |
| `answers/gen-*.jsonl` | Flint's answers to them with the prompt's and the answer's token ids: 17,745 answers, 17.91 M answer tokens | RIG-TBD |
| `manifest.json` | one entry per answer file: the writer (model file and SHA-256, engine and commit, server flags), the prompt file, the sampling, the number of answers and answer tokens, the file's SHA-256 | small |
| `vocab.npy` | the drafter's vocabulary: the 40,000 token ids it can guess | small |
| `heads/<run>/head.pt` | each run's best weights in PyTorch format; `export_gguf.py` turns one into the GGUF the engine loads | ~0.7–0.8 GB each |
| `heads/<run>/ckpt.pt` | each run's last training checkpoint: weights, optimizer and scaler state, step and elapsed time, for `train.py --resume` | ~2.1–2.4 GB each |
| `heads/<run>/config.json`, `log.jsonl` | the run's settings and its training log | small |

The runs: RIG-TBD, the rebuilt heads' chain from scratch to the two release heads
([training](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2.1/reproduce/drafter.md#5-training)).

## Sources and licences

Only sources whose answers may be published, with the licences their pages state (checked 2026-10-25). Every prompt
keeps its source in the `source` field.

| Source | Licence | Prompts for | Prompts |
|---|---|---|---|
| Python standard library and llama.cpp's own source files (not its `vendor/` and build folders) | Python Software Foundation License 2.0; llama.cpp MIT, with some files under Apache-2.0 or Apache-2.0 with LLVM exception | whole-file edits and coding-agent turns | 10,034 |
| Packages installed in the Python image (`site-packages/`), only those allowed below | MIT, BSD-2/3-Clause, Apache-2.0, PSF | whole-file edits and coding-agent turns | 6,307 |
| [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) (`default` subset): competition problems crawled from the web; the problems only, not the DeepSeek-R1 solutions | Apache-2.0 | reasoning | 2,743 |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2): written by volunteers; root prompts only, `synthetic == False`, reviewed | Apache-2.0 | chat | RIG-TBD |
| Code-generation prompts | RIG-TBD: a source may be added after a research note now running | code | RIG-TBD |

Prompts that share a 12-word span with a Q51 test prompt are left out of every source, so the drafter's Q51 scores
stay honest.

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

These counts are before the Q51 check.

</details>

**Excluded:** prompts from AI-MO/NuminaMath-CoT (about 45 % of its problems were written by GPT-4) and from
NousResearch/hermes-function-calling-v1, Team-ACE/ToolACE and glaiveai/glaive-function-calling-v2 (written by language
models they do not name), by the owner's ruling of 2026-10-25.

**Not used:** prompts from ShareGPT, UltraChat 200k and Magicoder (Evol-Instruct and OSS-Instruct). ShareGPT collects
conversations people shared from ChatGPT, and the terms for reusing them are unclear; LMSYS, who collected them to
train Vicuna, wrote "We will not release the ShareGPT dataset" ([FastChat](https://github.com/lm-sys/FastChat)).
UltraChat and Magicoder carry MIT and Apache-2.0 licences but were generated with OpenAI models. The project's first
drafter heads were trained partly on answers to those prompts; they are not published. Code prompts from llama.cpp's
`vendor/` folder (third-party code under its own licences), its build folders (generated files) and the site-packages
marked "no" above are left out too.

## Where the answers come from

| Writer | Engine | Answers | Answer tokens |
|---|---|---|---|
| Unsloth's Flint Q8_0 | llama.cpp (llama-server) | RIG-TBD | RIG-TBD |
| G3, this project's mix | llama.cpp | RIG-TBD | RIG-TBD |
| Unsloth's UD-Q4_K_XL | llama.cpp | RIG-TBD | RIG-TBD |
| The makers' NVFP4 GGUF | vLLM | RIG-TBD | RIG-TBD |

Sampling for every writer: temperature 1.0, top-k 20, top-p 0.95, min-p 0, reasoning kept, at most 2,048 answer
tokens, seed 20000 + the prompt's id number. Answers sampled on GPUs differ between machines in the details, so only
these files reproduce the drafter's exact training text; the writers and seeds above regenerate an equivalent set.

## What is not here, and how to recompute it

- **The hidden-state dumps** (~58 GB per million answer tokens): each answer replayed through Flint Q8_0 with
  `llama-staircase-dump`, which records 26 layers' inputs and the top-20 next-token probabilities per position
  ([step 3 of the guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2.1/reproduce/drafter.md#3-the-dump)).
- **The packed training sets**, written by the same command next to the dumps.
- **The test set** the drafter is scored on, 38 held-out Flint conversations, is in the GitHub repository
  ([reproduce/data/q51](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/release/flint-v2/reproduce/data/q51)).

## Licence

Apache-2.0 ([LICENSE](LICENSE)) for Flint's answers (Flint is Apache-2.0, by RecursiveSI), the heads, the checkpoints,
the vocabulary and the manifest. The prompts, and the prompt token ids inside the answer files, keep their sources'
licences listed above.

Made by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X) for
[LLMs on crypto-mining rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs). SHA-256 checksums of every
file: [CHECKSUMS](CHECKSUMS).
