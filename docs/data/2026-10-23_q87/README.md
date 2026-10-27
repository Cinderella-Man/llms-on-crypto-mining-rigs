# Q87: the first pipelined-speculation simulation, full sweep (2026-10-23)

Registry row Q87 ([quants/experiments.md](../../../quants/experiments.md)) started with a simulator: Flint's own
transcripts replayed through an event-by-event model of a layer split, where guessed tokens enter card 1 while later
cards finish earlier tokens, and a wrong guess cancels everything behind it. The method and selected rows are in
[bench/ngram_replay/README.md](../../../bench/ngram_replay/README.md) and
[pipelined_speculation.md](../../research/notes/pipelined_speculation.md). This folder holds the simulator's whole
first sweep.

| File | What |
|---|---|
| `pipeline.md` | the printed tables: the three card setups (stage time per card, plain tokens/s, the pipeline's ceiling), then one row per setup × guesser × depth with the speed-up over plain decoding for chat, code, edits, agent turns, all four, and file rewrites, the share of right and wrong guesses, and the card time spent on cancelled stages per token (101 configurations) |
| `pipeline.json` | the same per configuration plus tokens/s per workload and each setup's layers, stage times, hop and host gap |

Guessers: the copy lookup at minimum match lengths 2–12, a fixed 4-token key (the lookup of `ngram-simple` and
`ngram-map-k`), `ngram-cache` counts, the copy lookup backed by `ngram-cache`, and an oracle that guesses right with
probability p = 0.3–0.95, alone or behind the copy lookup; 1–8 guesses in flight (depth 5 for the full guesser sweep
on six cards, 8 on nine). For example, on six cards copy ≥ 4 backed by `ngram-cache` gives 1.34× overall at depth 5
(chat 1.11, code 1.26, edits 1.74, file rewrites 4.04), and an oracle at p = 0.7 behind the copy lookup 2.32×.

Made on 2026-10-23 on a PC's CPU (about 2 minutes) with
[bench/ngram_replay/pipeline_sim.py](../../../bench/ngram_replay/pipeline_sim.py) as committed in `36ce0f3`, on Q51's
38 held-out transcripts ([q51/gen.jsonl](../../../release/flint-v2/reproduce/data/q51/README.md)):

```bash
mkdir -p /tmp/q87
for f in pipeline_sim.py replay.py; do git show 36ce0f3:bench/ngram_replay/$f > /tmp/q87/$f; done
python3 -I /tmp/q87/pipeline_sim.py release/flint-v2/reproduce/data/q51/gen.jsonl --out pipeline.json > pipeline.md
```

Later versions of `pipeline_sim.py` add the continuation-table and Gumbel-coupled guessers (Q96) and measured card
setups, so their output has more rows. In `pipeline.md` the second table's header names the three configuration
columns (setup, guesser, depth); the simulator printed them under one heading.
