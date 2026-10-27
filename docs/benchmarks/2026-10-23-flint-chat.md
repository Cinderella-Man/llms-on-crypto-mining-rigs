# Flint in a real chat: stock llama.cpp against this project

*Measured 2026-10-23.*

One conversation of five user messages, run six times: three Flint files, each on unmodified upstream llama.cpp and on
this project's [llama.cpp fork](https://github.com/Cinderella-Man/llama.cpp/tree/pascal-p106). Everything else is the
same: the rig, all nine cards, the full 262,144-token context, the server flags, the sampling and the seeds.

## Setup

**Engines.**
- Stock: llama.cpp `d77256999` (2026-10-22), no patches, built for compute capability 6.1 with
  [scripts/build_llamacpp_docker.sh](../../scripts/build_llamacpp_docker.sh).
- This project: the fork's `pascal-p106` branch at `eeaa59660`, the same upstream commit plus CUDA graphs on Pascal,
  the wide-rows launch shape, the switch for Flint's MoE down-projection rescale, and the Pascal matrix-vector kernels
  for Q4_0, Q4_1, Q5_0, Q8_0, Q5_K and IQ4_XS ([design](../research/notes/pascal_mmvq_integration.md)).

**Model files.** Quality is measured against Flint's original 16-bit weights on 80 × 512 tokens of wikitext-2
([quants/README.md](../../quants/README.md)).

| File | Size | Same next token as the original | Mean KL divergence from the original |
|---|---|---|---|
| Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/flint-GGUF) | 25.1 GB | 87.7 % | 0.120 |
| Unsloth [Q8_0](https://huggingface.co/unsloth/flint-GGUF) | 42.3 GB | 93.3 % | 0.052 |
| Miner S4-L0 (Q4), this project ([recipe](../../quants/recipes/flint-speed-s4-l0.txt)) | 32.5 GB | 88.7 % | 0.123 |

**Server.** `llama-server -ngl 99 -fa on -c 262144 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 -np 1`, on
cards 0–8, with `-ts 6,6,5,5,6,5,5,6,6` for Q8_0 so that its full-context cache fits. The fork runs Q8_0 and Miner S4-L0
with `LLAMA_NO_MOE_DOWN_RESCALE=1`; stock llama.cpp has no such switch and always rescales.

- `-ub 128` and `--cache-ram 0` keep the 4 GB host from running out of memory at this context.
- `-bs` samples on the GPU, so the 150,144 scores of each token never cross the slow PCIe link.
- `--ctx-checkpoints 4`: 35 of Flint's 49 layers attend only to the last 513 tokens. llama-server can reuse such a
  model's cached conversation only from a checkpoint of those layers (~55 MB of host memory each); without one, every
  turn would process the whole history again.

**The conversation.** Five messages about running an LLM on this kind of rig, the later ones referring to earlier
answers:
1. "I have an old crypto-mining rig: nine NVIDIA P106 cards with 6 GB each, all on PCIe x1 risers, a 2-core Celeron
   and 4 GB of RAM. I'd like to run a local LLM on it for chatting and some coding help. What should I check or set up
   first?"
2. "Write a bash script that prints each GPU's memory use, temperature and power draw every 5 seconds, and prints a
   warning line when a card goes above 80 °C."
3. "Why do the x1 risers matter so much when loading a model, but so little when generating text once it is loaded?"
4. "Turn the checklist from your first answer into a short table with three columns: item, why it matters, how to
   check it."
5. "Thanks! Summarize the most important points of our conversation in three bullet points."

Each request carries the whole history the way a chat client sends it: the user's messages and the model's visible
answers, without its reasoning. Sampling follows Flint's GGUF defaults (temperature 1.0, top-p 0.95, top-k 20) with seeds
1001–1005, and a reply may run to 65,536 tokens. Flint reasons before every answer; the longest reply took 7,909
tokens and every reply ended on its own. The fork's UD-Q4_K_XL run allowed 8,192 tokens per reply, which none of its
replies reached. A lower cap does not work for Flint: in an extra run of the fork's UD-Q4_K_XL with 3,072 tokens
per reply, the first three replies stopped at the cap while Flint was still reasoning, with no answer at all, so
turns 2–4 started from a history of 107–163 tokens.

**Fairness.** Stock llama.cpp gets every setting the fork gets. Its own switches offer nothing more on these cards: it
disables CUDA graphs in code for every GPU older than Volta, has no switch for Flint's rescale, and its other CUDA
options (`GGML_CUDA_PDL`, `GGML_CUDA_ALLREDUCE`, unified or pinned memory) concern other GPUs or setups.

## Results

| Model file | Engine | Load | Generated (answer) | Generation | New prompt tokens | Waiting for first tokens | Whole chat |
|---|---|---|---|---|---|---|---|
| Unsloth UD-Q4_K_XL | Stock llama.cpp | 199 s | 12,678 (2,551) | **14.20 tok/s** | 2,571 at 80 tok/s | 32.2 s | 926 s |
| Unsloth UD-Q4_K_XL | This project | 199 s | 12,308 (2,846) | **27.97 tok/s** | 2,867 at 83 tok/s | 34.5 s | 475 s |
| Unsloth Q8_0 | Stock llama.cpp | 568 s | 19,846 (2,257) | **14.78 tok/s** | 2,280 at 80 tok/s | 28.5 s | 1,372 s |
| Unsloth Q8_0 | This project | 565 s | 16,953 (2,239) | **29.55 tok/s** | 2,292 at 82 tok/s | 27.9 s | 603 s |
| Miner S4-L0 (Q4), ours | Stock llama.cpp | 438 s | 20,354 (2,271) | **16.06 tok/s** | 2,299 at 84 tok/s | 27.3 s | 1,296 s |
| Miner S4-L0 (Q4), ours | This project | 435 s | 12,185 (2,341) | **31.86 tok/s** | 2,412 at 85 tok/s | 28.4 s | 412 s |

**This project generates twice as fast with every file:** 1.97× on UD-Q4_K_XL, 2.00× on Q8_0, 1.98× on Miner S4-L0.

- **The time to the first token doesn't change.** Reading the new part of the prompt (80–85 tokens/s here) runs on
  llama.cpp's batched kernels, which the fork leaves as they are.
- **Compare speeds, not conversation times.** With sampling at temperature 1.0, Flint thinks for different lengths in
  each run: Miner S4-L0 wrote 20,354 tokens on stock llama.cpp and 12,185 on the fork. The visible answers are about the
  same length in every run (2,200–2,900 tokens).
- **With the fork, Miner S4-L0 is the fastest file at Unsloth's 4-bit quality** (+14 % over UD-Q4_K_XL), and Q8_0, the
  closest to the original, runs between the two.
- Generation speed hardly changes over the five turns, because the history stays short (at most 2,900 tokens of
  visible answers and messages).

## Every turn

**Unsloth UD-Q4_K_XL, stock llama.cpp**

| Turn | History | New prompt tokens | First token after | Generated (answer) | Generation |
|---|---|---|---|---|---|
| 1 | 69 | 69 at 27 tok/s | 2.6 s | 5,614 (987) | 14.17 tok/s |
| 2 | 1,094 | 1,030 at 101 tok/s | 10.2 s | 2,413 (586) | 14.25 tok/s |
| 3 | 1,708 | 619 at 87 tok/s | 7.1 s | 1,577 (476) | 14.25 tok/s |
| 4 | 2,212 | 509 at 74 tok/s | 6.9 s | 2,355 (322) | 14.14 tok/s |
| 5 | 2,551 | 344 at 64 tok/s | 5.4 s | 719 (180) | 14.23 tok/s |

**Unsloth UD-Q4_K_XL, this project**

| Turn | History | New prompt tokens | First token after | Generated (answer) | Generation |
|---|---|---|---|---|---|
| 1 | 69 | 69 at 26 tok/s | 2.7 s | 5,025 (1,172) | 27.78 tok/s |
| 2 | 1,279 | 1,215 at 117 tok/s | 10.4 s | 3,800 (686) | 27.70 tok/s |
| 3 | 1,993 | 719 at 83 tok/s | 8.6 s | 1,053 (542) | 28.12 tok/s |
| 4 | 2,563 | 575 at 81 tok/s | 7.1 s | 866 (267) | 28.35 tok/s |
| 5 | 2,847 | 289 at 51 tok/s | 5.6 s | 1,564 (179) | 28.90 tok/s |

**Unsloth Q8_0, stock llama.cpp**

| Turn | History | New prompt tokens | First token after | Generated (answer) | Generation |
|---|---|---|---|---|---|
| 1 | 69 | 69 at 25 tok/s | 2.7 s | 7,909 (1,030) | 14.72 tok/s |
| 2 | 1,137 | 1,073 at 109 tok/s | 9.9 s | 3,756 (326) | 14.90 tok/s |
| 3 | 1,491 | 359 at 69 tok/s | 5.2 s | 1,780 (344) | 14.96 tok/s |
| 4 | 1,863 | 377 at 73 tok/s | 5.2 s | 5,596 (380) | 14.68 tok/s |
| 5 | 2,260 | 402 at 73 tok/s | 5.5 s | 805 (177) | 15.01 tok/s |

**Unsloth Q8_0, this project**

| Turn | History | New prompt tokens | First token after | Generated (answer) | Generation |
|---|---|---|---|---|---|
| 1 | 69 | 69 at 26 tok/s | 2.7 s | 7,784 (1,030) | 29.23 tok/s |
| 2 | 1,137 | 1,073 at 110 tok/s | 9.7 s | 1,811 (423) | 30.62 tok/s |
| 3 | 1,588 | 456 at 87 tok/s | 5.3 s | 1,449 (325) | 30.40 tok/s |
| 4 | 1,941 | 358 at 70 tok/s | 5.1 s | 5,216 (314) | 29.29 tok/s |
| 5 | 2,272 | 336 at 66 tok/s | 5.1 s | 693 (147) | 30.51 tok/s |

**Miner S4-L0 (Q4), ours, stock llama.cpp**

| Turn | History | New prompt tokens | First token after | Generated (answer) | Generation |
|---|---|---|---|---|---|
| 1 | 69 | 69 at 25 tok/s | 2.7 s | 7,664 (960) | 16.02 tok/s |
| 2 | 1,067 | 1,003 at 117 tok/s | 8.6 s | 6,088 (415) | 16.02 tok/s |
| 3 | 1,510 | 448 at 83 tok/s | 5.4 s | 1,455 (334) | 16.34 tok/s |
| 4 | 1,872 | 367 at 71 tok/s | 5.2 s | 4,518 (390) | 16.03 tok/s |
| 5 | 2,279 | 412 at 77 tok/s | 5.3 s | 629 (172) | 16.38 tok/s |

**Miner S4-L0 (Q4), ours, this project**

| Turn | History | New prompt tokens | First token after | Generated (answer) | Generation |
|---|---|---|---|---|---|
| 1 | 69 | 69 at 26 tok/s | 2.6 s | 4,711 (1,033) | 31.73 tok/s |
| 2 | 1,140 | 1,076 at 110 tok/s | 9.8 s | 2,507 (419) | 32.04 tok/s |
| 3 | 1,587 | 452 at 84 tok/s | 5.4 s | 1,548 (402) | 32.17 tok/s |
| 4 | 2,017 | 435 at 83 tok/s | 5.2 s | 2,419 (358) | 31.63 tok/s |
| 5 | 2,392 | 380 at 72 tok/s | 5.3 s | 1,000 (129) | 31.96 tok/s |

## Reproducing

[scripts/server_bench.sh](../../scripts/server_bench.sh) with `PROMPTS=chat` runs this conversation and writes the
per-turn numbers and the transcript (`chat.json`):

```bash
ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=/path/to/llama.cpp/build/bin MODEL=/path/to/flint-S4L0.gguf TAG=chat PROMPTS=chat MAX_TOKENS=65536 SERVER_ARGS='-c 262144 -b 512 -ub 128 -bs --ctx-checkpoints 4' bash -s" < scripts/server_bench.sh
```

Drop `LLAMA_NO_MOE_DOWN_RESCALE=1` for UD-Q4_K_XL, and add `-ts 6,6,5,5,6,5,5,6,6` to `SERVER_ARGS` for Q8_0.
Every run's per-turn table and transcript, and the exact commands, are in
[docs/data/2026-10-23_chat](../data/2026-10-23_chat/README.md).
