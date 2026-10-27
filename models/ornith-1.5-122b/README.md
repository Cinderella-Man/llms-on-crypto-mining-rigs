# Ornith-1.5-122B on Pascal mining rigs

[![Ornith-1.5-122B](https://img.shields.io/badge/Ornith--1.5--122B-first%20rig%20run-718096)](#speed)

[Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B) is a fine-tune of Qwen3.5-122B-A10B aimed at
terminal coding agents: a 122 B-parameter mixture-of-experts model that uses about 10 B parameters per token. Like
Qwen3.6 it is a hybrid, with 36 linear-attention (Gated DeltaNet) layers and 12 full-attention layers. Its makers
release it under the MIT licence; this project is not affiliated with them.

**Status: it runs, at 2–3 bits.** It is the project's ~120 B option and fits the rig's 54 GB of GPU memory only in
2–3-bit files. With Unsloth's UD-Q2_K_XL file, a coding agent gets 9.7 tok/s out of it on Miner Llama v2.0 against
6.2 on stock llama.cpp, and v2.0 finished all twelve tasks of the agent check; Miner Llama's Ornith work (v3) starts
from these measurements.

## Recommended setup

| | |
|---|---|
| Engine | [Miner Llama v2.0](https://github.com/Cinderella-Man/llama.cpp/releases/tag/miner-llama-v2.0) (tag `miner-llama-v2.0`) |
| Model file | Unsloth's [Ornith-1.5-122B-UD-Q2_K_XL](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF) (44.5 GB): faster on the rig and closer to the makers' Q8_0 than the other rig-sized file |
| File check | 44,533,952,768 bytes, SHA-256 `93fb32cfb1c31b34df426b4a41d961bdaffcddbbd217d850262136cf3689ba66` (repository revision `745bd742473a`); [scripts/hf_get.sh](../../scripts/hf_get.sh) downloads it pinned and checked |
| Cards | all nine, split by bytes with `-ts 8,6,5,5,5,4,5,6,6` (llama.cpp's default split overfills card 5) |
| Context | 49,152 tokens with `-ub 512` |
| Sampling | the makers' settings: temperature 0.6, top-p 0.95, top-k 20, min-p 0 |

```bash
llama-p106-v2.0/llama-server -m Ornith-1.5-122B-UD-Q2_K_XL.gguf -ngl 99 -fa on -c 49152 -np 1 -b 2048 -ub 512 -ts 8,6,5,5,5,4,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

- **Don't use `--pipe-depth`** with this model on v2.0: pipelined speculation needs to step the recurrent state back
  after a wrong guess, which v2.0 cannot do, and the request fails.
- **The MTP head** (`--spec-type draft-mtp --spec-draft-n-max 3`, measured with the AD file below at `-ub 256`) helps
  stock llama.cpp's generation by 22 % but gives v2.0 nothing.
- Loading takes about 5.5 minutes from a USB 3 stick.
- **AtomicChat's [AD-IQ3_XXS-IQ2_S](https://huggingface.co/AtomicChat/Ornith-1.5-122B-GGUF)** (47.6 GB;
  47,643,074,624 bytes, SHA-256 `f6285e9a98a7cebcbafdbd4cc89d05f47999e7fce3cf9be74a32b7da6aace4ea`, revision
  `6c2d63ef5523`) runs with `-ts 5,6,5,6,5,5,6,5,7`, and up to 122,880 tokens of context with `-ub 128`.

## Speed

The [agent check](../../bench/agent_check/README.md) (the pi coding agent fixing twelve small bugs, 35 minutes, nine
cards, `-c 49152`), 2026-10-27:

| File, engine | Tasks done (passed) | Minutes per task, median | Generation, tok/s | First token per turn, median |
|---|---|---|---|---|
| UD-Q2_K_XL, stock llama.cpp `d77256999` | 9 (8) | 3.5 | 6.2 | 9.8 s |
| **UD-Q2_K_XL, Miner Llama v2.0** | **12 (12)** | **2.1** | **9.7** | **7.6 s** |
| AD-IQ3_XXS-IQ2_S, stock llama.cpp | 7 (6) | 3.8 | 5.4 | 10.2 s |
| AD-IQ3_XXS-IQ2_S, stock llama.cpp + MTP head | 9 (9) | 3.1 | 6.6 | 11.7 s |
| AD-IQ3_XXS-IQ2_S, Miner Llama v2.0 | 9 (7) | 2.9 | 8.8 | 8.7 s |
| AD-IQ3_XXS-IQ2_S, Miner Llama v2.0 + MTP head | 9 (8) | 2.5 | 7.9 | 11.4 s |

Per task, v2.0 is 1.67× as fast as stock with UD-Q2_K_XL on the eight tasks both passed (1.4× with the AD file), and
v2.0 with the MTP head ties v2.0 without it.

Details, the MTP head, memory and where the time goes: [the benchmark](../../docs/benchmarks/2026-10-27-ornith-122b.md).
Per-task and per-turn data, memory projections and GPU energy:
[docs/data/2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md).

## Quality

Against the makers' Q8_0 (the BF16 model does not fit the quality machine), by KL divergence and how often the top
token matches ([docs/data/2026-10-27_ornith_kld](../../docs/data/2026-10-27_ornith_kld/README.md)):

| File | wikitext | Agent sessions | Exercism code |
|---|---|---|---|
| **UD-Q2_K_XL** | **0.184 / 83.5 %** | **0.054 / 94.9 %** | **0.071 / 96.3 %** |
| AD-IQ3_XXS-IQ2_S | 0.233 / 81.4 % | 0.066 / 94.3 % | 0.081 / 96.2 % |

**Not measured yet:** how Ornith compares with Flint on coding tasks beyond these twelve.

## What's next (v3)

1. Pascal kernels for the 2–3-bit expert formats: the routed experts take 43 % of a token and run at a sixth of the
   card's bandwidth. The first ones, for IQ2_S and IQ3_XXS, take the UD file's expert products from 31.6 to 17.3 ms
   of GPU time per token ([data](../../docs/data/2026-10-27_iq_kernels/README.md)); their agent check is next.
2. Hybrid-model support for pipelined speculation and the card-built mask
   ([Q162](../../quants/experiments.md#registry)).
3. A Pascal-shaped Ornith file with smaller dense tensors, to free memory and bandwidth.
4. [Ornith-1.5-35B](../ornith-1.5-35b/README.md), which fits four or five cards with its full context.
