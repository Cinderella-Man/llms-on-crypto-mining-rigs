# Q96: a continuation table behind the copy lookup, and Gumbel-coupled guesses (2026-10-24)

Q96 added guessers to `pipeline_sim.py`'s replay of Q51's 38 Flint transcripts: a table of next-token counts after 1- to
8-token contexts of Flint's other outputs (leave one transcript out), the request's own counts, and combinations behind
the copy lookup; then guesses coupled to the sampler's noise, using Flint Q8_0's top-20 log-probabilities along the
transcripts. A continuation table helps (copy lookup, else table and request counts: 1.44× on six cards and 1.54× on
nine, against 1.34× and 1.41×); coupling does not help n-gram guessers (39.7 → 38.6 % right) but lifts a model-like
drafter (90.0 → 98.7 % at σ 0). The write-up is [bench/ngram_replay](../../../bench/ngram_replay/README.md) ("A
continuation table behind the copy lookup (Q96)") and the registry row Q96
([quants/experiments.md](../../../quants/experiments.md)); this folder holds the full grids.

| File | What |
|---|---|
| `q96_table.md`, `.json` | the whole grid: every guesser of Q87 and Q96 on six cards and both nine-card splits at each depth, the cost-sensitivity rows (hop 0.1 / 0.6 / 1.0 ms, stages × 1.1), the oracle-p rows, then one token ahead per guesser (fresh and copied text, by class) with the table at a quarter, half and full size; speed-ups over plain decoding |
| `q96_coupled.md`, `.json` | the same pipelines with the guess as the sampled top guess (expected) against the Gumbel-coupled guess |
| `q96_coupling_study.md`, `.json` | one token ahead, expected share right per class: Flint's own top token, the table guesser top and coupled (counts to the power 1, 2, 4, 8), and a modelled drafter (Flint's log-probabilities with Gaussian noise σ 0–2 nats), top and coupled |
| `run/build_logit_dump.sh` | builds `llama-logit-dump` against a llama.cpp tree's CUDA build (CPU Docker) |
| `run/q96-logit-dump.job` | the GPU job: Flint Q8_0's top-20 log-probabilities along the 38 transcripts (`top20.bin`, 10.5 MB, not included) |

In the tables, `Cards` is the split of Flint's layers, `Depth` the number of tokens in flight, and the last column the
card time spent on guesses that were cancelled, per generated token.

## Made on 2026-10-24 with

[bench/ngram_replay/pipeline_sim.py](../../../bench/ngram_replay/pipeline_sim.py) (standard library only, CPU, 8
processes) and [logit_dump](../../../bench/ngram_replay/logit_dump/logit_dump.cpp) on
[Q51's transcripts](../../../release/flint-v2/reproduce/data/q51/README.md):

```bash
Q51=release/flint-v2/reproduce/data/q51
python -I bench/ngram_replay/pipeline_sim.py $Q51/gen.jsonl --jobs 8 --accuracy --out q96_table.json > q96_table.md
# Flint Q8_0's top-20 log-probabilities (llama.cpp d77256999, two RTX 3090s; see run/)
LOGIT_IN=$Q51/seqs.txt LOGIT_OUT=top20.bin LOGIT_K=20 llama-logit-dump -m flint-Q8_0.gguf -ngl 99 -c 6144 -b 512 -ub 512 -fa on
python -I bench/ngram_replay/pipeline_sim.py $Q51/gen.jsonl --jobs 8 --topk top20.bin --only 'Gumbel|expected' \
    --out q96_coupled.json > q96_coupled.md
python -I bench/ngram_replay/pipeline_sim.py $Q51/gen.jsonl --topk top20.bin --coupling-study \
    --out q96_coupling_study.json > q96_coupling_study.md
```

The pipeline model's stage times are the rig's measured per-card times (Q87); nothing here was timed on the GPU machine.
The tables hold counts and ratios only.
