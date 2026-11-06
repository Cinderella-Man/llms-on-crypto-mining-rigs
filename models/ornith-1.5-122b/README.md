# Ornith-1.5-122B on Pascal mining rigs

[![Ornith-1.5-122B](https://img.shields.io/badge/Ornith--1.5--122B-v3%20release-2b6cb0)](#speed)

[Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B) is a fine-tune of Qwen3.5-122B-A10B aimed at
terminal coding agents: a 122 B-parameter mixture-of-experts model that uses about 10 B parameters per token. Like
Qwen3.6 it is a hybrid, with 36 linear-attention (Gated DeltaNet) layers and 12 full-attention layers. Its makers
release it under the MIT licence; this project is not affiliated with them.

**Status: released in Miner Llama v3, at 2–3 bits.** It is the project's ~120 B option and fits the rig's 54 GB of
GPU memory only in 2–3-bit files. On v3 a coding agent gets 16.3 tok/s out of it in my own file,
[Ornith-1.5-122B Pascal N2P GGUF](https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF) (`valid1`),
all twelve tasks passed, with the preset's keep-alive. Stock llama.cpp b12076 cannot load that file; on Unsloth's
UD-Q2_K_XL it gets 6.0 tok/s (11 of 12) against v3's 12.6 (12 of 12). Everything about the v3 release, with the
preset and one command: [release/ornith-v3](../../release/ornith-v3/README.md).

## Recommended setup

| | |
|---|---|
| Engine | [Miner Llama v3.0](https://github.com/Cinderella-Man/miner-llama/releases/tag/miner-llama-v3.0) (tag `miner-llama-v3.0`) with Ornith's preset, [release/ornith-v3/settings.env](../../release/ornith-v3/settings.env) |
| Model file | [Ornith-1.5-122B Pascal N2P GGUF](https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF) (48.2 GB), which only Miner Llama v3 loads: faster on the rig and closer to the makers' Q8_0 than UD-Q2_K_XL |
| Cards | all nine, split by bytes with `-ts 6,6,6,6,5,5,5,5,6` |
| Context | 49,152 tokens with `-ub 512` |
| Sampling | the makers' settings: temperature 0.6, top-p 0.95, top-k 20, min-p 0 |

```bash
MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/ornith-v3/settings.env && env $PRESET_ENV llama-p106-v3.0/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

**For other engines, or v2.0:** Unsloth's
[Ornith-1.5-122B-UD-Q2_K_XL](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF) (44.5 GB; 44,533,952,768 bytes,
SHA-256 `93fb32cfb1c31b34df426b4a41d961bdaffcddbbd217d850262136cf3689ba66`, repository revision `745bd742473a`;
[scripts/hf_get.sh](../../scripts/hf_get.sh) downloads it pinned and checked), split by bytes with
`-ts 8,6,5,5,5,4,5,6,6` (llama.cpp's default split overfills card 5):

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

### On Miner Llama v3 (2026-11-05 and 2026-11-06)

The [agent check](../../bench/agent_check/README.md), nine cards, `-c 49152`, one run per row
([2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md),
[2026-11-06_ornith_mtp_rig2](../../docs/data/2026-11-06_ornith_mtp_rig2/README.md)):

| File, engine | Tasks passed | Generation, tok/s | First token per turn, median |
|---|---|---|---|
| UD-Q2_K_XL, stock llama.cpp b12076 (`20a51f24a`), tuned (`-ub 256`) | 11 of 12 | 6.0 | 12.57 s |
| UD-Q2_K_XL, v3 (`1551f5fe1`), no switch | 12 of 12 | 12.6 | 7.90 s |
| Ornith-1.5-122B Pascal N2P GGUF, v3 (`1f7215948`), no switch | 12 of 12 | 14.9 | 9.41 s |
| **Ornith-1.5-122B Pascal N2P GGUF, v3 (`1f7215948`), the preset (+ the keep-alive)** | **12 of 12** | **16.3** | **8.39 s** |
| Ornith-1.5-122B Pascal N2P GGUF, v3 (`1551f5fe1`), the preset, a second rig | 12 of 12 | 16.42 | 9.34 s |

Ornith's own MTP layer as a draft is a loss on v3: 3.42 and 3.44 against the preset's 2.52 minutes a task at equal
work, since checking two tokens costs 1.76 times one.

### On Miner Llama v2.0 (2026-10-27)

The [agent check](../../bench/agent_check/README.md) (the pi coding agent fixing twelve small bugs, 35 minutes, nine
cards, `-c 49152`), 2026-10-27:

| File, engine | Tasks done (passed) | Minutes per task, median | Generation, tok/s | First token per turn, median |
|---|---|---|---|---|
| UD-Q2_K_XL, stock llama.cpp `d77256999` | 9 (8) | 3.5 | 6.2 | 9.8 s |
| **UD-Q2_K_XL, Miner Llama v2.0** | **12 (12)** | **2.1** | 9.7 | 7.6 s |
| UD-Q2_K_XL, Miner Llama v2.0 + v3's IQ2_S/IQ3_XXS kernels (then in development) | 12 (12) | 2.5 | **11.3** | **6.6 s** |
| AD-IQ3_XXS-IQ2_S, stock llama.cpp | 7 (6) | 3.8 | 5.4 | 10.2 s |
| AD-IQ3_XXS-IQ2_S, stock llama.cpp + MTP head | 9 (9) | 3.1 | 6.6 | 11.7 s |
| AD-IQ3_XXS-IQ2_S, Miner Llama v2.0 | 9 (7) | 2.9 | 8.8 | 8.7 s |
| AD-IQ3_XXS-IQ2_S, Miner Llama v2.0 + MTP head | 9 (8) | 2.5 | 7.9 | 11.4 s |

Per task, v2.0 is 1.67× as fast as stock with UD-Q2_K_XL on the eight tasks both passed (1.4× with the AD file), and
v2.0 with the MTP head ties v2.0 without it. v3's kernels for the 2–3-bit expert formats generate 1.17× faster than
v2.0 (88.6 against 103.4 ms per token); that run's answers came out longer, so its minutes per task do not show it yet.

Details, the MTP head, memory and where the time goes: [the benchmark](../../docs/benchmarks/2026-10-27-ornith-122b.md).
Per-task and per-turn data, memory projections and GPU energy:
[docs/data/2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md).

## Quality

Against the makers' Q8_0 (the BF16 model does not fit the quality machine), by KL divergence and how often the top
token matches ([docs/data/2026-10-27_ornith_kld](../../docs/data/2026-10-27_ornith_kld/README.md)):

| File | wikitext | Agent sessions | Exercism code |
|---|---|---|---|
| UD-Q2_K_XL | 0.184 / 83.5 % | 0.054 / 94.9 % | 0.071 / 96.3 % |
| AD-IQ3_XXS-IQ2_S | 0.233 / 81.4 % | 0.066 / 94.3 % | 0.081 / 96.2 % |
| **Ornith-1.5-122B Pascal N2P GGUF** ([its data](../../docs/data/2026-10-28_ornith_valid1/README.md)) | **0.1762 / 84.14 %** | **0.0518 / 95.04 %** | **0.0615 / 96.69 %** |

**Not measured yet:** how Ornith compares with Flint on coding tasks beyond these twelve.

## What v3 did, and what is next

1. **Done in v3:** Pascal kernels for the 2–3-bit expert formats. The routed experts took 43 % of a token at a sixth
   of the card's bandwidth. The kernels for IQ2_S and IQ3_XXS take the UD file's expert products from 31.6 to 17.3 ms
   of GPU time per token ([data](../../docs/data/2026-10-27_iq_kernels/README.md)), and my own N2P and N3P types with
   their kernels made Ornith-1.5-122B Pascal N2P GGUF, 16.3 tok/s with the preset.
2. **Done in v3:** a Pascal-shaped Ornith file, Ornith-1.5-122B Pascal N2P GGUF
   ([how it was made](../../docs/data/2026-10-28_ornith_valid1/README.md)).
3. **Tried in v3, a loss:** Ornith's own MTP layer as the draft
   ([data](../../docs/data/2026-11-06_ornith_mtp_rig2/README.md)); it needs a cheaper check of the draft.
4. Next: hybrid-model support for pipelined speculation and the card-built mask
   ([Q162](../../quants/experiments.md#registry)).
5. Next: [Ornith-1.5-35B](../ornith-1.5-35b/README.md), which fits four or five cards with its full context.
