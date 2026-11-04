# file_choice: which GGUF file to run, from headers, KLD logs and kernel speeds

Puts quality, fit and an estimated decode speed side by side for several files of one model on a multi-card rig,
without loading any of them. It reads only GGUF headers (tensor names, types, shapes, offsets) and llama-perplexity's
KLD logs, and runs in under a few minutes of one CPU core. First use and result:
[docs/data/2026-11-04_file_choice](../../docs/data/2026-11-04_file_choice/README.md) (eight Qwen3.8-Flash-Next files
on nine P106-090 cards).

| File | What it does |
|---|---|
| `dump.py NAME SHARD...` | one file's tensor index as JSON: shape keys, shards, every tensor's name, type, shape, bytes and shard |
| `fit.py NAME=SHARD[,SHARD...] ...` | the best `-ts` (and `-ot` expert moves) per file over a grid of context, cache type, micro-batch and draft, through [scripts/split_plan.py](../../scripts/split_plan.py); per-card costs from llama.cpp's projections (Q213) |
| `model.py` | from the dumps, the fit grid and the KLD logs: what each file puts on the cards, the bytes a decoded token reads per role and type, the est. ms per token, the fit summary and the ranking |
| `efficiency.tsv` | the GB/s each weight type's kernel reaches on a P106, per role (routed gate/up, routed down, output, hyper-connections, any), "timed" and "expected", each row with its source |
| `run.sh` | all three steps: `FILES=<list> KLD=<folder> OUT=<folder> bash run.sh` |

The speed is a model: est. ms per token = Σ (MB a token reads of each type ÷ that type's GB/s) + the GPU time spent
outside weight products (`--nonweight`, 33.48 ms from the R13 profile of Flash-Next). For another card, model or
engine, replace `efficiency.tsv`'s rows and `--nonweight` with that setup's measurements; for another card count or
memory size, pass `fit.py --cards` and `--free`, and replace its `COSTS` table with that setup's per-layer projections
(`scripts/fit_probe.sh`, as in [2026-11-03_flash_next_fit](../../docs/data/2026-11-03_flash_next_fit/README.md)).

Standard-library Python 3; `dump.py` imports [scripts/gguf_reshard.py](../../scripts/gguf_reshard.py).
