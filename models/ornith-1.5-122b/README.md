# Ornith-1.5-122B on Pascal mining rigs

[![Ornith-1.5-122B](https://img.shields.io/badge/Ornith--1.5--122B-first%20rig%20run-718096)](#speed)

[Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B) is a fine-tune of Qwen3.5-122B-A10B aimed at
terminal coding agents: a 122 B-parameter mixture-of-experts model that uses about 10 B parameters per token. Like
Qwen3.6 it is a hybrid, with 36 linear-attention (Gated DeltaNet) layers and 12 full-attention layers. Its makers
release it under the MIT licence; this project is not affiliated with them.

**Status: it runs, at 2–3 bits.** It is the project's ~120 B option and fits the rig's 54 GB of GPU memory only in
2–3-bit files. A coding agent gets 8.8 tok/s out of it on Miner Llama v2.0 against 5.4 on stock llama.cpp; Miner
Llama's Ornith work (v3) starts from these measurements.

## Recommended setup

| | |
|---|---|
| Engine | [Miner Llama v2.0](https://github.com/Cinderella-Man/llama.cpp/releases/tag/miner-llama-v2.0) (tag `miner-llama-v2.0`) |
| Model file | AtomicChat's [Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S](https://huggingface.co/AtomicChat/Ornith-1.5-122B-GGUF) (47.6 GB) |
| File check | 47,643,074,624 bytes, SHA-256 `f6285e9a98a7cebcbafdbd4cc89d05f47999e7fce3cf9be74a32b7da6aace4ea` (repository revision `6c2d63ef5523`); [scripts/hf_get.sh](../../scripts/hf_get.sh) downloads it pinned and checked |
| Cards | all nine, split with `-ts 5,6,5,6,5,5,6,5,7` (the default split runs out of memory) |
| Context | 49,152 tokens with `-ub 512`; up to 122,880 with `-ub 128` (prompts read slower) |
| Sampling | the makers' settings: temperature 0.6, top-p 0.95, top-k 20, min-p 0 |

```bash
llama-p106-v2.0/llama-server -m Ornith-1.5-122B-AD-IQ3_XXS-IQ2_S.gguf -ngl 99 -fa on -c 49152 -np 1 -b 2048 -ub 512 -ts 5,6,5,6,5,5,6,5,7 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

- **Don't use `--pipe-depth`** with this model on v2.0: pipelined speculation needs to step the recurrent state back
  after a wrong guess, which v2.0 cannot do, and the request fails.
- **The MTP head** (`--spec-type draft-mtp --spec-draft-n-max 3`, with `-ub 256 -ts 6,6,5,6,6,5,6,5,5`) helps stock
  llama.cpp's generation by 22 % but gives v2.0 nothing over the line above.
- Loading takes about 6 minutes from a USB 3 stick.

## Speed

The [agent check](../../bench/agent_check/README.md) (the pi coding agent fixing twelve small bugs, 35 minutes, nine
cards, `-c 49152`), 2026-10-27:

| Engine | Tasks done (passed) | Minutes per task, median | Generation, tok/s | First token per turn, median |
|---|---|---|---|---|
| Stock llama.cpp `d77256999` | 7 (6) | 3.8 | 5.4 | 10.2 s |
| Stock llama.cpp + MTP head | 9 (9) | 3.1 | 6.6 | 11.7 s |
| **Miner Llama v2.0** | **9 (7)** | **2.9** | **8.8** | **8.7 s** |
| Miner Llama v2.0 + MTP head | 9 (8) | 2.5 | 7.9 | 11.4 s |

Per task, v2.0 is 1.4× as fast as stock on the tasks both passed, and v2.0 with the MTP head ties v2.0 without it.

Details, the MTP head, memory and where the time goes: [the benchmark](../../docs/benchmarks/2026-10-27-ornith-122b.md).
Per-task and per-turn data, memory projections and GPU energy:
[docs/data/2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md).

Unsloth's [UD-Q2_K_XL](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF) (44.5 GB; 44,533,952,768 bytes, SHA-256
`93fb32cfb1c31b34df426b4a41d961bdaffcddbbd217d850262136cf3689ba66`, revision `745bd742473a`) runs too, with
`-ts 8,6,5,5,5,4,5,6,6`: in single requests it decodes ~10 % faster than the AD file and reads prompts slower; it has
not run the agent check yet.

**Not measured yet:** how far either 2–3-bit file is from the original weights, and how Ornith compares with Flint on
coding tasks beyond these twelve.

## What's next (v3)

1. Pascal kernels for the 2–3-bit expert formats: the routed experts take 43 % of a token and run at a sixth of the
   card's bandwidth (est. ~13 tok/s plain once they run like the rest).
2. Hybrid-model support for pipelined speculation and the card-built mask
   ([Q162](../../quants/experiments.md#registry)).
3. A Pascal-shaped Ornith file with smaller dense tensors, to free memory and bandwidth.
4. [Ornith-1.5-35B](../ornith-1.5-35b/README.md), which fits four or five cards with its full context.
