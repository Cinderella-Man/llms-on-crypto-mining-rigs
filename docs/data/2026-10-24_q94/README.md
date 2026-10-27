# Q94: reflection words in Flint's reasoning, full counts

The JSON output of [reflections.py](../../../bench/reasoning_length/reflections.py) behind the tables of
[bench/reasoning_length/README.md](../../../bench/reasoning_length/README.md) (registry row Q94 in
[quants/experiments.md](../../../quants/experiments.md)): how often Flint's reasoning holds "Wait"-style reflection
words, per output kind and per BFCL category. Flint reflects seldom, 2.4 reflection words per 1,000 reasoning tokens in
Q51's transcripts ("wait" 18 and "hmm" 18 times in 43,373 tokens) and 3.0 per 1,000 words in 3,475 BFCL answers, so a
logit bias against them would cut little and Q94 was dropped.

| File | What |
|---|---|
| `q94.json` | Q51's 38 transcripts with token ids and Flint's vocabulary: per output kind (chat, code, edits, agent) and overall, reflection and contrast words per 1,000 reasoning tokens (all and sentence-initial), the share of reasoning in sentences and paragraphs a reflection word opens, counts per word, per sentence-initial word and per tenth of the reasoning |
| `q94_q51_text.json` | the same transcripts counted as text (`--text`), per 1,000 words: the bridge between the two ways of counting (1.62 tokens per word) |
| `q94_bfcl.json` | Q41's BFCL v4 single-turn answers as text, per BFCL category and overall, per 1,000 reasoning words |

Made on 2026-10-24 with `bench/reasoning_length/reflections.py` at commit `d45f643` (standard library only, seconds
on a CPU). Q51's transcripts are public in
[release/flint-v2/reproduce/data/q51](../../../release/flint-v2/reproduce/data/q51/README.md) (`gen.jsonl`, Flint
Q8_0 at its default sampling). The BFCL answers are attempt 1 of the first half of Q41's E1 for G3 and UD-Q4_K_XL
([2026-10-24_q41](../2026-10-24_q41/README.md); the raw outputs are not published). The pilot's multi-turn output
(`pilot/e2-ud-q4kxl.jsonl`) was also passed and contributed nothing: its records hold no reasoning block in `content`.

```bash
python -I reflections.py gen.jsonl flint-Q8_0.gguf --out q94.json
python -I reflections.py --text gen.jsonl --out q94_q51_text.json
python -I reflections.py --text raw/e1-g3-a1-p1of2.jsonl raw/e1-ud-a1-p1of2.jsonl pilot/e2-ud-q4kxl.jsonl \
  --out q94_bfcl.json
```

Counts only: no reasoning text is included. Q51's prompts and answers are described in their own folder; the BFCL
answers are Flint's outputs (Apache-2.0) on BFCL v4 tasks (Gorilla project, Apache-2.0).
