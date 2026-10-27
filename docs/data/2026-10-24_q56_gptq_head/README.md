# Q56 and Q91: the GPTQ output head and the screened head, offline (2026-10-23/24)

The full screen table and the scripts behind registry rows Q56 and Q91
([experiments.md](../../../quants/experiments.md)) and their write-up in
[quants/gptq_head](../../../quants/gptq_head/README.md), which also holds the tools and the
end-to-end `run.sh`. A GPTQ-rounded Q5_0 output head costs +0.002 KLD where round-to-nearest Q5_0 costs +0.023 (Q56),
and a GPTQ Q4_0 screen of 256 candidates holds the exact head's top 20 on all 40,527 held-out rows, so top-k 20
sampling draws exactly as from the full head (Q91). The public README shows six rows of the screen table;
`q91_screen.tsv` has all of them.

| File | What |
|---|---|
| `q91_screen.tsv` | per screen and number of candidates M: screen size (MB, share of the Q8_0 head's bytes), share of the head read per token, how often the screen's candidates hold the exact top 1, the exact top 20 and the top-p 0.95 nucleus, the exact probability of the sampled set, and the KLD of the screened logits for M = 256 and 1,024; a `#` line per screen gives the M each row needs for its full top 20. Screens: exact control, GPTQ Q5_0 and Q4_0, round-to-nearest Q4_0 and Q4_1, and H-weighted low-rank factors of rank 64–512 |
| `run/after_dump.sh` | after the head dump: dump check against F3's own logits, Hessians (wikitext, Flint transcripts, both), the seven candidate Q5_0 heads (imatrix, GPTQ with max or searched scales, column order, refit, per calibration set), offline check |
| `run/holdout_q51.sh` | the domain check: 8 of the 38 Q51 transcripts held out of every Hessian, heads rebuilt on the rest, scored on the held-out transcripts and wikitext rows |
| `run/q91_screen.sh` | the screen table above, on held-out rows only |

Made on 2026-10-23/24 with the tools in [quants/gptq_head](../../../quants/gptq_head/README.md) (`llama-head-dump`,
`gptq_head.py`, `screen_head.py`) on Q73's F3 (rescale off), Flint BF16 for the head weights, wikitext-2 train (chunks
0–499 for the Hessians, 2000–2039 held out) and the Q51 transcripts
([release/flint-v2/reproduce/data/q51](../../../release/flint-v2/reproduce/data/q51/README.md), `seqs.txt`):

```bash
# 1. head inputs of F3 (and the round-to-nearest head): quants/gptq_head/run.sh with
MODEL=flint-q73-F3.gguf BF16=flint-BF16.gguf TRAIN=wiki.train.raw OUT=$WORK SEQS=seqs.txt quants/gptq_head/run.sh
#    (file names used below: x-wiki-cal.f32, x-wiki-val.f32, x-q51.f32 for its x-cal, x-val, x-seqs)
# 2. the candidate heads, the held-out split, the Q4_0 screen and the screen table
export WORK BF16=flint-BF16.gguf F3=flint-q73-F3.gguf PYTHONPATH=<llama.cpp>/gguf-py
bash docs/data/2026-10-24_q56_gptq_head/run/after_dump.sh
bash docs/data/2026-10-24_q56_gptq_head/run/holdout_q51.sh
python quants/gptq_head/gptq_head.py quant $BF16 $WORK/head-gptq-search-mixa.q4_0 --mode gptq --hessian $WORK/H-mixa.npy \
  --scale search --type q4_0 > $WORK/quant-q4_0.log
bash docs/data/2026-10-24_q56_gptq_head/run/q91_screen.sh
# 3. full-model KLD of F3 with each head: gptq_head.py base (F3 without its head) once, then shard per head,
#    scored with docs/data/2026-10-23_q36_q45/run/kld_gpu.sh and ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1
```

Wikitext-2 is not included (llama.cpp's `scripts/get-wikitext-2.sh`). The transcripts are Flint's own outputs (Flint is
Apache-2.0); the table and scripts are this project's (MIT).
