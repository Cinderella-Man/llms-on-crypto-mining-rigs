#!/usr/bin/env bash
# Q91 (2026-10-24): screened output head, offline. Held-out rows only: wikitext-2 train chunks 2000-2039 (never in any
# H) and the 8 held-out Q51 transcripts (not in H-mixa). Screens calibrated with H-mixa (holdout_q51.sh); the Q4_0
# screen head-gptq-search-mixa.q4_0 is gptq_head.py quant with --type q4_0 (its log in quant-q4_0.log); output
# q91_screen.tsv. Env: WORK, BF16, F3 as for holdout_q51.sh; numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
set -euo pipefail
T=$(cd "$(dirname "$0")/../../../../quants/gptq_head" && pwd); cd "$WORK"
while [ ! -s head-gptq-search-mixa.q4_0 ] || ! grep -q "rows, q4_0" quant-q4_0.log; do sleep 10; done
python $T/screen_head.py $BF16 $F3 H-mixa.npy \
  x-wiki-val.f32 x-q51.f32@8,9,18,19,26,27,33,37 \
  --screen q5_0-gptq=head-gptq-search-mixa.q5_0 --screen q4_0-gptq=head-gptq-search-mixa.q4_0 \
  --ranks 64,128,256,512 --M 64,128,256,512,1024 --kld-M 256,1024 --out q91_screen.tsv
echo Q91_DONE
