# Q64: offline replay of n-gram drafters on Flint's own transcripts (2026-10-23)

Q64 replayed Q51's 38 Flint transcripts (65,520 generated tokens at T 1.0) through every n-gram drafter of llama.cpp and
its settings with exact acceptance, first at the speculation note's model verify costs and then at the rig's measured
verify cost (Q19 stage 1). At the measured cost only 2-token drafts pay, about 1.03× overall; 28 % of Flint's output
sits in copies of 8 tokens or more. The write-up is [bench/ngram_replay](../../../bench/ngram_replay/README.md) ("What
Flint's output copies", "Speed-up at the measured verify cost") and the registry row Q64
([quants/experiments.md](../../../quants/experiments.md)); this folder holds the full printed reports.

| File | What |
|---|---|
| `report.md` | `replay.py`'s tables at the model verify-cost curves: copy structure per class and segment, copy runs, how often a 6-token match continues correctly, then per class (chat, code, edit, agent, all, file rewrites) and per segment every drafter and setting: steps with a draft, acceptance, tokens per step, draft widths and the speed-up under the optimistic, mid, Q51 and pessimistic curves |
| `report_measured.md` | the same with two more speed-up columns: `measured` (the rig's verify step times, the last slope above 8 tokens) and `measured+pp` (llama-bench's pp512 rate above 8 tokens) |

## Made on 2026-10-23 with

[bench/ngram_replay/replay.py](../../../bench/ngram_replay/replay.py) (standard library only, about a minute on one
core) on [Q51's gen.jsonl](../../../release/flint-v2/reproduce/data/q51/README.md):

```bash
python -I bench/ngram_replay/replay.py release/flint-v2/reproduce/data/q51/gen.jsonl --out results.json > report.md
# with the rig's verify step times (ms: server synth points on six cards, llama-bench pp5-pp8 as upper bounds)
python -I bench/ngram_replay/replay.py release/flint-v2/reproduce/data/q51/gen.jsonl \
    --vn 1=25.9,2.9=57.7,4.05=92.1 --vn-cap 5=163.6,6=170.0,7=207.6,8=212.1 \
    --out results_measured.json > report_measured.md
# the Python ports against the fork's C++ drafters: all 15 llama.cpp settings "same"
LLAMA_DIR=<llama.cpp fork checkout> bash bench/ngram_replay/fork_check.sh release/flint-v2/reproduce/data/q51/gen.jsonl
```

The replay is deterministic; the JSON outputs (with full verify-width histograms) are rebuilt by the commands above. The
reports hold counts and ratios only.
