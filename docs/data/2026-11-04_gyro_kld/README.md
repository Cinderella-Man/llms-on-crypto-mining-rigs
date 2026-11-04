# agentionai's Gyro files on our KL-divergence texts (2026-11-04)

Registry row Q896 in [quants/experiments.md](../../../quants/experiments.md). agentionai's Gyro files store
Qwen3.8-Flash-Next's routed experts in a trellis-coded format of the QTIP family (TQ2_T, 2.125 bits per weight), which
stock llama.cpp cannot load. agentionai report KL divergence against a Q8_0 of the full model at `-c 2048` on their own
texts: Gyro-M (all 512 experts) 0.291 on held-out 2026 writing and code, Gyro-Coder (256 of 512 experts per layer) 0.579
there and 0.118 on C++. Here both files are scored on the three texts and the reference logits that every other
Flash-Next file in this repository was scored on
([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)), so they sit on the same scale.

## The files

| File | Repository, revision | Bytes | SHA-256 (Hugging Face LFS) |
|---|---|---:|---|
| `Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf` | [agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF) `b7d23f5036e9134b3b84e20e7040ba47bc5e0b4c` | 75,796,732,256 | `2ecaf3ba6ed7f142ce00bc4a3172052f01bcf4e10d3ff617d8ab4755f4f2d4c4` |
| `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` | [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) `05892758a67efff0895d23c6df10a6fdb6be8cf4` | 91,965,774,272 | `7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f` |

What their headers hold (`run/header.py`; GiB; "GPUs" is everything but the n-gram table and the token embeddings,
which llama.cpp keeps on the host):

| | Gyro-Coder | Gyro-M |
|---|---:|---:|
| `general.name` | Qwen3.8 Flash Next | Qwen3.8-Flash-Next Gyro-M |
| layers, routed experts per layer, experts per token | 48, **256**, 10 | 48, 512, 10 |
| routed experts (`ffn_{gate,up,down}_exps`, 144 tensors) | TQ2_T, 14.94 | TQ2_T, 29.88 |
| router (`ffn_gate_inp`) | F32, 2,560 × 256 | F32, 2,560 × 512 |
| every other matrix (attention, recurrent, shared expert, hyper-connections, n-gram keys, output) | Q8_0 | Q8_0 |
| indexer projections | BF16 | BF16 |
| n-gram table (`per_layer_token_embd`, 160 × 320,001,536) | Q8_0, 50.66 | Q8_0, 50.66 |
| token embeddings | Q8_0, 0.63 | Q8_0, 0.63 |
| on the GPUs | **19.29** | **34.35** |
| whole file | 70.58 | 85.64 |

The GPU figures on agentionai's cards (19.92 and 34.98 GiB) add the token embeddings. Beside our files: Swift-1.5's
GSQ-RCO files carry their dense matrices in K- and IQ-quants and an IQ4_NL n-gram table (28.8 GB); that table alone,
swapped into the Q8_0 reference, scores 0.027 / 0.035 / 0.007 on wiki / agent / code against a floor of
0.017 / 0.029 / 0.004 ([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md),
`q8_0-iq4nl-table` and `q8_0-ub256`). The Gyro files keep both at Q8_0, so only their routed experts are below 8 bits.

## Licence

Both repositories carry the Qwen Community License 1.0 (`LICENSE`, the same text in both). It grants the right to use,
copy, modify, publish and distribute the weights and derivative works, with the copyright notice kept. Its two
conditions concern commercial products above 100 million monthly users or US$20 million monthly revenue (show the model
name) and Model-as-a-Service or AI-work-assistant businesses (a separate licence from Qwen for commercial use). Running
the files to measure them and publishing the measurements is within the grant. This folder holds only scores and
header statistics; no weights are redistributed.

## The engine

[agentionai's llama.cpp fork](https://gitlab.com/agentionai-group/llama.cpp), `main` at
`9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f` (tag `main-b11377-9aa51ce4f`, 2026-10-12), built from source in Docker
(`run/build.sh`) for two backends: CUDA 12.9 for compute 8.6, the backend the reference logits were made with, and
Vulkan (Ubuntu 24.04's loader and glslc), the backend agentionai publish. Its `llama-perplexity` reads the reference
logits files of upstream's format unchanged (the reader is upstream's; only the batch API around it differs). In
KL-divergence mode it scores every chunk of the reference file and ignores `--chunks`. On a CPU without a GPU
(`run/cpucheck.sh`) both files load and score their first code chunks.

## The reference

The reference logits are those of Swift-1.5's Q8_0, a fine-tune of Qwen3.8-Flash-Next that thinks less. The Gyro
files are quantized from the original Qwen3.8-Flash-Next, so their KL divergence here holds the fine-tune's difference
as well as their quantization. ISTA's GSQ-RCO files are in the same position, and the reference chain measured how
large that difference is: ISTA's IQ3_S, made from the original model, and Swift-1.5's IQ3_S, made the same way from
the fine-tune, score nearly the same against these logits.

| Text | ISTA IQ3_S (original model) | Swift-1.5 IQ3_S | difference |
|---|---:|---:|---:|
| wiki | 0.161 | 0.153 | +0.008 |
| agent | 0.117 | 0.121 | −0.004 |
| code | 0.032 | 0.032 | 0.000 |

So the checkpoint adds about 0.01 or less at this level, and the Gyro files are read beside ISTA's IQ3_S and ISTA's
Coder (256 of 512 experts per layer, like Gyro-Coder), which share their checkpoint.

## How it ran

- **Texts and flags** of [2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md): wikitext-2 test
  (80 × 512 tokens), held-out agent sessions (12 × 8,192) and the agent check's Exercism code (80 × 512);
  `llama-perplexity` with `-ub 512 -fa on`, every layer on the GPUs; the reference logits carry their own tokens.
- **Two RTX 3090s**, layers split by llama.cpp's default. The reference ran on seven cards (3080s and 3090s) with
  its own split. A change of cards and split alone moves Flash-Next's scores by up to the floor that chain measured
  (the reference at `-ub 256` against itself: 0.017 / 0.029 / 0.004 on wiki / agent / code).
- **A control first** (`run/score.sh`): ISTA's Coder, which the reference chain scored with Miner Llama's own build,
  on this fork's CUDA build and these two cards. Each text had to land within the larger of that floor and two
  standard errors of the old score, or the run would stop before the Gyro files.
- The n-gram table is read at random, so each file is read from a copy on an NVMe drive or, when that drive is busy,
  in place with its table held in the page cache (see "How the run went").

## Results

KL divergence against Swift-1.5's Q8_0 (mean ± standard error) and the share of tokens whose top token matches it,
on the held-out agent sessions and the Exercism code. The rows from ISTA's Coder down are the reference chain's own scores of
the files this repository runs, on seven cards with Miner Llama's build
([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)). GiB on the GPUs is everything but the
n-gram table and the token embeddings (`run/header.py`).

| File | Experts per layer | GiB on the GPUs | agent KLD | agent same top | code KLD | code same top |
|---|---:|---:|---:|---:|---:|---:|
| **Gyro-M** (TQ2_T experts, the rest Q8_0) | 512 | 34.35 | **0.148 ± 0.003** | 92.5 % | **0.046 ± 0.002** | 97.5 % |
| **Gyro-Coder** (TQ2_T experts, the rest Q8_0) | 256 | 19.29 | **0.281 ± 0.005** | 90.1 % | **0.133 ± 0.005** | 95.6 % |
| control: ISTA GSQ-RCO Coder ("IQ1_M"), this fork and cards | 256 | 27.25 | 0.278 ± 0.005 | 90.8 % | 0.130 ± 0.005 | 95.6 % |
| ISTA GSQ-RCO Coder ("IQ1_M") | 256 | 27.25 | 0.274 ± 0.005 | 90.7 % | 0.131 ± 0.005 | 95.5 % |
| ISTA GSQ-RCO IQ3_S | 512 | 50.73 | 0.117 ± 0.002 | 93.4 % | 0.032 ± 0.001 | 98.0 % |
| Swift-1.5 GSQ-RCO IQ3_S | 512 | 50.85 | 0.121 ± 0.003 | 93.7 % | 0.032 ± 0.001 | 98.1 % |
| Swift-1.5 GSQ-RCO IQ3_XXS | 512 | 43.67 | 0.176 ± 0.004 | 92.3 % | 0.054 ± 0.002 | 97.5 % |
| Swift-1.5 GSQ-RCO Q2_0 | 512 | 34.89 | 0.285 ± 0.006 | 90.7 % | 0.112 ± 0.004 | 96.0 % |
| floor: the reference at `-ub 256` | 512 | — | 0.029 ± 0.001 | 97.1 % | 0.004 ± 0.000 | 99.2 % |

- **The control holds:** ISTA's Coder on this fork and these two cards lands 0.004 above its old agent score and 0.001
  below its old code score, inside the noise (the larger of the floor and two standard errors: 0.029 and 0.009). The
  fork's `llama-perplexity` and the change of cards and split leave the scale as it was.
- **Gyro-M at the size of Swift's Q2_0:** with the same GPU footprint (34.35 against 34.89 GiB), Gyro-M's KLD is
  about half of Q2_0's on agent text and 0.41 of it on code, and below Swift IQ3_XXS's on both, which needs 9.3 GiB
  more. The checkpoint difference counts against Gyro-M here (about 0.01 or less, from the ISTA-vs-Swift IQ3_S pair),
  so its quantization alone is at least this good. Part of the gap is the file's other tensors: Gyro-M keeps every
  non-expert matrix and the n-gram table in Q8_0, where the Swift files use K- and IQ-quants and an IQ4_NL table.
- **On the rig** Gyro-M would take the room Swift Q2_0 takes on nine 6 GB cards. No Pascal kernel for TQ2_T exists
  yet; registry row Q895 measures what one would cost per weight on a P106.
- **Gyro-Coder at 19.29 GiB** scores like ISTA's Coder, which keeps the same number of experts (256 of 512) in 27.25 GiB,
  and like Swift's Q2_0 on agent text at 55 % of its size; on code it is 0.021 behind Q2_0.
- **Not scored yet** (registry row Q909): the wiki text of both files and Gyro-Coder's code text on the Vulkan build;
  the run's card time ended. `run/score.sh` with `STEPS="coder:wiki m:wiki coder-vulkan:code"` scores them and skips
  what is done.

The scores as a table: `kld.tsv` (`table.py` of the reference chain over this run's logs).

## How the run went

- Every file ran from a copy on the NVMe drive at first. While other jobs wrote to that drive, its writes slowed to
  ~100 MB/s and its reads queued behind them: 80–90 s per agent chunk against ~15 s on a quiet drive. Gyro-M was then
  read in place from a hard disk instead. There its n-gram table, read at random, came in at ~110
  reads a second, one agent chunk in 565 s. `run/pin_range.py` holds the table's 54.4 GB in the page cache
  (`mlock`, filled by one sequential read in 338 s), after which the chunks took ~25 s.

## Files here

| File | What |
|---|---|
| `run/download.sh`, `run/files.list` | the two files at their pinned revisions, resumable, each checked against its SHA-256 |
| `run/build.sh` | the fork at the pinned commit, `BACKEND=cuda` or `vulkan`, in Docker |
| `run/header.py` | a GGUF's keys, tensor types and bytes, the n-gram table and the bytes on the GPUs (standard library; [scripts/gguf_reshard.py](../../../scripts/gguf_reshard.py)) |
| `run/cpucheck.sh` | the engine check without a GPU: one file's first code chunks on the CPU |
| `run/hot_copy.sh` | a file copied to a fast disk and checked against its SHA-256 |
| `run/kld.sh` | a file's KL divergence against the reference logits on the three texts with the fork's `llama-perplexity` |
| `run/score.sh` | the whole run: the copies, a wait for the GPUs, the control and its check, Gyro-Coder, Gyro-M, Gyro-Coder on Vulkan |
| `run/pin_range.py`, `run/pin_guard.sh` | a byte range of a file (the n-gram table) held in the page cache with `mlock`, and its release when free memory runs low or the run ends |
| `kld.tsv` | the scores: mean KLD and its error, the 99th and 99.9th percentile, same top token, RMS change of the top probability, perplexity of the file and of the reference |

## Commands

On the quality machine, with `W` a work folder, `M` the model folder, `HOT` a folder on a fast disk and `BASES` the
folder holding [2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)'s
`base.{wiki,agent,code}.kld`:

```bash
OUT=$M bash run/download.sh
WORK=$W BACKEND=cuda JOBS=8 DOCKER_ARGS="--cpu-shares=256" bash run/build.sh
WORK=$W BACKEND=vulkan JOBS=8 DOCKER_ARGS="--cpu-shares=256" bash run/build.sh
python3 -I run/header.py $M/Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf $M/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf
BIN=$W MODELS=$M BASES=$BASES OUT=$W/cpucheck bash run/cpucheck.sh
BIN=$W BASES=$BASES OUT=$W/kld HOT=$HOT GPUS=<two GPU ids> MOUNTS=<folders the files sit in> \
  CODER=$M/Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf GYRO_M=$M/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf \
  CONTROL_FILES="<ISTA Coder shard 1> <its sha256> <shard 2> <its sha256>" \
  CONTROL_REF="wiki:0.845520:0.009881 agent:0.273501:0.005248 code:0.130815:0.004663" \
  FLOOR="wiki:0.017012 agent:0.029142 code:0.003578" \
  STEPS="control:agent,code coder:agent,code m:agent,code" IN_PLACE=m WARM=1 bash run/score.sh
# Gyro-M read in place: its n-gram table held in RAM for at most 40 minutes, released early when the run ends
read start len <<< "$(python3 -I run/header.py --range per_layer_token_embd.weight $M/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf)"
timeout 2400 python3 -I run/pin_range.py $M/Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf $start $len 2400 &
OUT=$W/kld MIN_AVAIL_GB=15 bash run/pin_guard.sh &
python3 -I ../2026-11-03_flash_next_quality/run/table.py $W/kld gyro-m gyro-coder control-ista-coder > kld.tsv
```

In this run the pin started while Gyro-M's agent text was running, and that text was paused (`SIGSTOP`) for the 338 s
the pin took to read the table, so the hard disk read it in one pass.
