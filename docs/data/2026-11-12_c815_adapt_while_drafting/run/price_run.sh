#!/bin/bash
# C815 (Q1424): LLAMA_SPEC_ADAPT_ONLINE=1 and =2 as built, priced on FN80's dumps (price_built.py), on a CPU at the
# lowest priority. WORK holds repo/ (this repository's bench/draft_count and docs/data) and dumps/work/dump-d6-*.jsonl,
# dump-d7-*.jsonl (FN80's LLAMA_SPEC_DUMP files, docs/data/2026-11-11_fn80_fn81_draft_yield); writes WORK/out/.
set -eu
cd "$WORK"
W=dumps/work
D=repo/docs/data
nice -n 19 python3 -I $D/2026-11-12_c815_adapt_while_drafting/run/price_built.py \
    $W/dump-d6-1.jsonl $W/dump-d6-2.jsonl $W/dump-d7-1.jsonl $W/dump-d7-2.jsonl \
    --n-max 6 --cols 8,10,12,14,16 --table v4est=$D/2026-11-12_fn82_adaptive_draft/v4_est.tsv --out out
