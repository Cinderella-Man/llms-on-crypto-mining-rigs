# Q51: how many experts consecutive Flint tokens share (2026-10-23)

Q51 generated Flint's answers to 38 held-out prompts (10 chat, 10 code, 8 whole-file edits, 10 agent turns) and recorded
which 6 of the 256 experts every token routes to in each of the 47 MoE layers, to see whether consecutive tokens reuse
experts (they mostly don't: 35.6 distinct experts per layer for 8 tokens, 80 % of random routing). The result is in
[bench/expert_overlap](../../../bench/expert_overlap/README.md) and the registry row Q51
([quants/experiments.md](../../../quants/experiments.md)). The prompts and answers themselves are
[release/flint-v2/reproduce/data/q51](../../../release/flint-v2/reproduce/data/q51/README.md), the held-out test set every
drafter and guesser score in this repository is measured on; this folder holds the analysis behind the summary.

## Results

Distinct experts per layer for n consecutive tokens, and experts shared by adjacent tokens (`adj`, of 6):

| Scope | n = 2 | n = 4 | n = 8 | n = 16 | adj |
|---|---|---|---|---|---|
| chat | 10.96 | 19.84 | 35.16 | 59.51 | 1.04 |
| code | 11.01 | 20.03 | 35.40 | 59.51 | 0.99 |
| edits | 11.16 | 20.44 | 36.08 | 60.14 | 0.84 |
| agent | 11.02 | 19.96 | 35.29 | 59.56 | 0.98 |
| all generated tokens | 11.05 | 20.13 | 35.58 | 59.74 | 0.95 |
| prompt tokens | 11.13 | 20.24 | 35.56 | 59.12 | 0.87 |
| random routing | 11.86 | 23.17 | 44.24 | 80.84 | 0.14 |

- Router weights after renormalization, mean by rank: 0.305, 0.210, 0.156, 0.125, 0.107, 0.096; the lightest of the 6
  is below 0.05 for 6.4 % of tokens and below 0.10 for 52.9 %.
- Expert use per layer: the top 10 / 25 / 50 % of a layer's experts take 22.7 / 44.3 / 70.6 % of the routed reads;
  4.3 % of experts are never used.
- Generated tokens: chat 17,432, code 21,416, edits 23,747, agent 2,963; 8 of the 38 answers (2 code, 6 edits) stop at
  the 3,072-token limit (`generation_summary.txt`).

## Files

| File | What |
|---|---|
| `q51.json` | everything `analyze.py` computes: distinct experts per scope and window, per layer (`per_layer`), router weights by rank, usage concentration |
| `analysis.txt` | `analyze.py`'s printed summary (the table above) |
| `generation_summary.txt` | the sampling the server applied, and per prompt: category, prompt tokens, generated tokens, stop reason |
| `run/run_q51.sh` | build `llama-route-dump`, serve Flint Q8_0, generate the 38 answers, stop the server, route pass |
| `run/run_q51_dump.sh` | the route pass again with logits requested for every token (so the last layer is recorded too), then `analyze.py` |

## Made on 2026-10-23 with

[bench/expert_overlap](../../../bench/expert_overlap/README.md) (`prompts.py`, `generate.py`, `route_dump.cpp`,
`analyze.py`) and Unsloth's Flint Q8_0 in llama-server on two V100s, the sampling the GGUF carries (temperature 1.0,
top-k 20, top-p 0.95, min-p 0.05), up to 3,072 tokens per answer:

```bash
python -I generate.py http://127.0.0.1:18451 prompts.jsonl gen.jsonl seqs.txt 4 3072
ROUTE_IN=seqs.txt ROUTE_OUT=routes.bin llama-route-dump -m flint-Q8_0.gguf -ngl 99 -c 16384 -b 512 -ub 512 -fa on
python -I analyze.py routes.bin gen.jsonl q51.json > analysis.txt
```

`run/` holds the same steps as Docker jobs (`WORK`: the work folder with a llama.cpp tree and a copy of
bench/expert_overlap; `MODELS`: the folder holding `flint-Q8_0.gguf`). The routes file (`routes.bin`) is not included;
the commands above rebuild it from `seqs.txt`. The answers are Flint's outputs (Flint is Apache-2.0); the edit prompts
quote this repository's own scripts (MIT).
