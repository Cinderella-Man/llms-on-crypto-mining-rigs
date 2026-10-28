# Ornith-1.5-122B with N2P/N3P experts allocated by layer and a Q5_0 output layer (2026-10-28)

The data behind the registry row for this file in [quants/experiments.md](../../../quants/experiments.md). This file
combines two results from the night before:
- **Q167's valid1** ([allocation data](../2026-10-27_ornith_alloc/README.md)): Q164's file c with the routed experts'
  gate/up at N3P in layers 24–47 and their down at N2P in layers 0–31. Its KLD against the Q8_0 was better than
  Unsloth's UD-Q2_K_XL on all three texts.
- **A Q5_0 output layer**, which cost nothing measurable on UD-Q2_K_XL and AD-IQ3_XXS-IQ2_S
  ([Ornith KLD data](../2026-10-27_ornith_kld/README.md)). It saves 287 MB against file c's Q8_0.

This is the N2P/N3P candidate for the rig's agent check. The formats are Miner Llama's own
([pascal_lowbit_format.md](../../research/notes/pascal_lowbit_format.md)).

## Result

KLD against Ornith-1.5-122B Q8_0, as in the Ornith KLD data:
- wikitext-2, 80 × 512 tokens;
- held-out agent sessions, 12 × 8,192 tokens;
- the agent check's Exercism code, 80 × 512 tokens.

Each cell is mean KLD ± SE / same top token. File sizes are in GB of 10⁹ bytes; "on the cards" leaves out the token
embedding and the MTP block.

| File | File size | On the cards | wiki | agent | code |
|---|---|---|---|---|---|
| UD-Q2_K_XL (Unsloth) | 44.53 | — | 0.1843 ± 0.0026 / 83.55 % | 0.0541 ± 0.0011 / 94.93 % | 0.0713 ± 0.0028 / 96.32 % |
| AD-IQ3_XXS-IQ2_S (AtomicChat) | 47.64 | — | 0.2332 / 81.43 % | 0.0657 / 94.31 % | 0.0807 / 96.23 % |
| file c (Q164) | 47.58 | 45.81 | 0.2033 ± 0.0030 / 82.85 % | 0.0572 ± 0.0012 | 0.0721 ± 0.0028 |
| valid1 (Q167), Q8_0 output layer | 48.48 | 46.72 | 0.1747 ± 0.0028 / 84.28 % | 0.0513 ± 0.0010 / 95.15 % | 0.0613 ± 0.0026 / 96.68 % |
| **valid1, Q5_0 output layer (this file)** | **48.20** | **46.43** | **0.1762 ± 0.0028 / 84.14 %** | **0.0518 ± 0.0010 / 95.04 %** | **0.0615 ± 0.0026 / 96.69 %** |

Differences below are in standard errors of a difference, √(SE₁² + SE₂²). That is conservative, because both files
are scored on the same tokens.
- **Against UD-Q2_K_XL** this file is better on all three texts:
  - wikitext −0.0081 (2.1 standard errors);
  - agent sessions −0.0023 (1.5);
  - code −0.0098 (2.6);
  - same top token +0.6, +0.1 and +0.4 points.

  It is 3.66 GB larger.
- **Against valid1 with the Q8_0 output layer** it is level (+0.0015, +0.0005, +0.0002: 0.4, 0.4 and 0.1 standard
  errors) for 287 MB less. That matches what the Q5_0 head did on UD and AD.
- **Perplexity** is 5.5 % above the Q8_0's on wikitext, 0.7 % on agent sessions and 2.4 % on code (`kld.tsv`).
  UD-Q2_K_XL's falls slightly below the reference on all three, which is noise around a quantized reference; KL
  divergence is the measure here.
- **Speed is not measured here.** N2P and N3P are est. ~1.8× faster per expert weight than the IQ2_S/IQ3_XXS Pascal
  kernels on a P106; the rig measures them next, then this file in the agent check.

## The file

`Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`: 48,196,722,592 bytes, md5 `7985df46ec7bb3867e482eaf91aaef39`. It is not
published yet. Its `quantize.imatrix.file` key still holds the path on the machine that quantized files c and d;
rewrite that key before publishing the file.

| Tensors | Type | Count | GB |
|---|---|---|---|
| expert gate and up, layers 24–47 | N3P | 2 × 24 | 15.70 |
| expert gate and up, layers 0–23 and the MTP block | N2P | 2 × 25 | 13.53 |
| expert down, layers 0–31 | N2P | 32 | 8.66 |
| expert down, layers 32–47 and the MTP block | N3P | 17 | 5.56 |
| output layer | Q5_0 | 1 | 0.52 |
| token embedding, attention k and v | Q8_0 | 1 + 2 × 13 | 0.86 |
| every other quantized tensor (attention, Gated DeltaNet, shared experts) | Q5_0 | | 3.20 |
| routers, norms, convolution and SSM vectors | F32 | | 0.16 |

The full table, by role and type, is in `header-Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.txt`.

## Files

| File | What |
|---|---|
| `kld.tsv` | per text: perplexity of the file and of the reference, mean KLD ± SE, median, 99th percentile and maximum KLD, same top token ± SE, RMS of the probability change (the columns of the Ornith KLD data's `kld_122b.tsv`) |
| `header-Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.txt` | `scripts/gguf_summary.py` of the file: metadata, then tensor bytes per role and type, per block, the non-block tensors and the MTP block |
| `run/compose.sh` | the file from files c and d and the Q5_0 head, each tensor byte for byte (`scripts/compose_gguf.py`) |

## How it was made

```bash
# files c and d: docs/data/2026-10-27_lowbit/run/quantize.sh FILE=c, FILE=d; the head: docs/data/2026-10-27_ornith_kld/run/head_q5_0.py
QM=<ssh alias> FILE_C=<file c> FILE_D=<file d> HEAD=<output-q5_0.gguf> OUT=<folder>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  GGUFPY=<fork branch pascal-p106-v3-lowbit>/gguf-py bash run/compose.sh
# KLD with the Q8_0's existing bases in OUT, on two V100s (no -ts needed), with the low-bit branch's llama-perplexity:
BIN=<llama.cpp>/bin/llama-perplexity OUT=<folder with the bases> GPUS=3,4 WIKI=wiki.test.raw AGENT=agent-heldout.txt \
  CODE=code-exercism.txt bash ../2026-10-27_ornith_kld/run/kld_texts.sh Ornith-1.5-122B-Q8_0.gguf \
  Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf=<pascal-p106-v3-lowbit build>/bin/llama-perplexity
```

- **Composing:** CPU in Docker at low priority, 5 minutes. Rules apply in this order: gate/up of layers 24–47 and
  down of layers 0–31 from file d, `output.weight` from the Q5_0 head, all other 818 tensors from file c. These are
  alloc.py's switch regexes.
- **Scoring:** fork branch `pascal-p106-v3-lowbit` at `5fd8ffeec`, the build Q167 scored with, on two V100 32 GB. Flags:
  - `-c 512 -b 512 -ub 512 --chunks 80` for wikitext and code;
  - `-c 8192 -b 8192 -ub 512 --chunks 12` for the agent text;
  - on all three, `-ngl 99 -fa on --kl-divergence-base <Q8_0 base> --kl-divergence`.

  It took 19 minutes for the three texts, from a copy of the file on an NVMe disk.
- **Licences:** Ornith-1.5 is MIT; Unsloth's importance matrix (used for files c and d) is Apache-2.0; ik_llama.cpp's
  level tables in N2P/N3P are MIT; the scores and scripts are this project's.
