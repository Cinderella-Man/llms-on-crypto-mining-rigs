# Qwen3.8-Flash-Next files: what each holds, read from the GGUF headers (2026-10-27)

The data behind sections 1 and 6 of [flash_next_variants.md](../../research/notes/flash_next_variants.md): for ten
published Flash-Next GGUF files and one MTP draft, how many bytes sit in the n-gram table, the routed experts and the
dense tensors, in which formats, and how many bytes one generated token reads. Only the files' headers were read; no
model was run and nothing was measured on the rig.

| File | What |
|---|---|
| `census/summary.tsv` | one row per file: experts, total, n-gram table, on the cards (total − n-gram table), routed experts, dense, read per token (GB) |
| `census/<file>.tensors.tsv` | per file: every tensor class (the tensor name with its layer number taken out) and format, with tensor count, shape and GB |
| `run/gguf_census.py` | reads a GGUF header (metadata, tensor names, shapes, types, offsets) and prints the report; standard library only |
| `run/census_all.sh` | runs it over one folder per model and writes the tables |

## Results

| File | Experts | Total | N-gram table | On the cards | Routed experts | Dense | Read per token |
|---|---|---|---|---|---|---|---|
| `flashnext_iq3s` | 512 | 83.61 | 28.80 | 54.81 | 50.29 | 4.18 | 5.16 |
| `flashnext_coder_iq1m` | 256 | 58.40 | 28.80 | 29.60 | 25.15 | 4.11 | 5.10 |
| `swift_flash_q2_0` | 512 | 66.54 | 28.80 | 37.74 | 33.97 | 3.49 | 4.16 |
| `swift_flash_iq2_xs` | 512 | 68.14 | 28.80 | 39.34 | 35.45 | 3.55 | 4.24 |
| `swift_flash_iq3_xxs` | 512 | 75.96 | 28.80 | 47.15 | 42.91 | 3.97 | 4.81 |
| `swift_flash_iq3_s` | 512 | 83.73 | 28.80 | 54.93 | 50.29 | 4.30 | 5.28 |
| `gyro_s` | 512 | 58.48 | 28.80 | 29.68 | 25.79 | 3.36 | 3.86 |
| `gyro_m` | 512 | 91.95 | 54.40 | 37.55 | 32.09 | 4.79 | 5.42 |
| `gyro_l` | 512 | 108.31 | 54.40 | 53.91 | 48.44 | 4.79 | 5.74 |
| `gyro_l_mtp` (MTP draft) | 512 | 2.69 | — | 2.69 | 1.42 | 0.57 | 0.60 |
| `takeonme_flash_q4k` | 512 | 136.90 | 54.40 | 82.50 | 77.02 | 4.81 | 6.31 |

GB are decimal. Definitions, as `gguf_census.py` applies them:
- **N-gram table:** `per_layer_token_embd.weight`, 160 × 320,001,536 in every file: IQ4_NL (28.80 GB) or Q8_0
  (54.40 GB).
- **Routed experts:** `blk.N.ffn_{gate,up,down}_exps`.
- **Dense:** everything else, except the token embeddings and any MTP tensors. It includes the shared experts, the
  router `ffn_gate_inp`, the hyper-connection tensors `hc_*` and the output layer.
- **Read per token:** dense + routed experts × used / count, with 10 experts used per token. The token embeddings and
  the n-gram table contribute a few rows per token and are left out.

What the tables show:
- **The dense tensors are 3.4–4.8 GB of every file's 3.9–6.3 GB read per token.** The routed experts are 0.5–1.0 GB
  of it.
- **In every GSQ-RCO file** (ISTA and Swift) the hyper-connection projections `hc_{attn,ffn}_{down,up}` are BF16:
  1.26 GB of the 1.35–1.53 GB in BF16. The router `ffn_gate_inp` is F32 (0.25 GB) in the Swift files, Gyro-M/L and
  TakeOnMe, and BF16 (0.13 GB) in ISTA's files and Gyro-S.
- **The Swift GSQ-RCO IQ3_S file uses the same expert formats as ISTA's IQ3_S, layer for layer:** gate/up in IQ2_S,
  IQ3_XXS, IQ3_S or IQ4_XS, down in IQ4_NL or Q2_0. The Swift Q2_0 file has every routed expert in Q2_0.
- **Gyro's expert formats print as type ids 144–149,** because they are not in llama.cpp. The agentionai fork names
  them TQ2_T (144), TQK6 (145), TQK7 (146) and TQP5–7 (147–149).

## How it was made

| Census file | Hugging Face repository and revision | Licence |
|---|---|---|
| `flashnext_iq3s` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) `ed59f92082b1e93c0e96d60a8b11aab089b52f09`, `IQ3_S/` (2 shards) | weights under the base model's licence |
| `flashnext_coder_iq1m` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) `bba88856f8c3c0c9e40e45f82d05e228b814ac43`, `IQ1_M/` (2 shards) | as above |
| `swift_flash_*` | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) `99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d` (2 shards each) | Swift Open License 1.0 + qwen-community-1.0 |
| `gyro_s`, `gyro_m`, `gyro_l`, `gyro_l_mtp` | [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) `273edc1d2ffae5241fe3cab69568d682dad4c718` | qwen-community-1.0 |
| `takeonme_flash_q4k` | [Pedro-TakeOnMe/TakeOnMe-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/Pedro-TakeOnMe/TakeOnMe-Qwen3.8-Flash-Next-GGUF) `ab02839fe46f7d67021a44fe22fc72d39d2f2410` (5 shards) | Swift Open License 1.0 + qwen-community-1.0 |

The run, on the 7-GPU quality machine, at low CPU priority (header reads only, seconds per file). Each argument is a
folder holding one model's shards, named as in the census:

```bash
GGUF_PY=<llama.cpp>/gguf-py OUT=<out dir> bash run/census_all.sh \
    $MODELS/flashnext_iq3s $MODELS/flashnext_coder_iq1m $MODELS/swift_flash_q2_0 $MODELS/swift_flash_iq2_xs \
    $MODELS/swift_flash_iq3_xxs $MODELS/swift_flash_iq3_s $MODELS/gyro_s $MODELS/gyro_m $MODELS/gyro_l \
    $MODELS/takeonme_flash_q4k
```

`GGUF_PY` points at a llama.cpp `gguf-py` folder; only its `gguf/constants.py` is loaded, for the format names.
The tensor sizes come from the offsets between tensors, so formats the local llama.cpp doesn't know still count.
The full reports also hold every file's metadata (architecture keys, licence fields, chat template); they are not
published here.
