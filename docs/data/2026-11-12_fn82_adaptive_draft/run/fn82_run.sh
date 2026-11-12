#!/bin/bash
# FN82 (Q1402): LLAMA_SPEC_ADAPT's rule replayed on FN80's dumps, on a CPU at the lowest priority.
# WORK holds repo/ (this repository's bench/draft_count and docs/data), dumps/work/dump-d{3..7}-{1,2}.jsonl (FN80's
# LLAMA_SPEC_DUMP files) and receives tables/ (the est. v4 cost table), replay/ (the replay's tables) and main.md.
# Writes WORK/done or WORK/failed.
set -u
cd "$WORK" || exit 1
rm -f done failed
W=dumps/work
D=repo/docs/data
F=$D/2026-11-12_fn82_adaptive_draft/run
if nice -n 19 python3 -I repo/bench/draft_count/adapt_replay.py --selftest &&
   nice -n 19 python3 -I $F/v4_cost.py --r120b $D/2026-11-12_r120b_kern_speed --c805 $D/2026-11-12_c805_kcol_exact \
     --r41 $D/2026-11-10_release_profile --out tables &&
   nice -n 19 python3 -I repo/bench/draft_count/adapt_replay.py \
     $W/dump-d3-1.jsonl $W/dump-d3-2.jsonl $W/dump-d4-1.jsonl $W/dump-d4-2.jsonl $W/dump-d5-1.jsonl $W/dump-d5-2.jsonl \
     $W/dump-d6-1.jsonl $W/dump-d6-2.jsonl $W/dump-d7-1.jsonl $W/dump-d7-2.jsonl \
     --out replay --n-max 3,4,5,6,7 --fixed 2,3,4,5,6,7 --cols 8,10,12,14,16 --table v4est=tables/v4_est.tsv \
     --also 'adapt-n{N}-run-wdef-x1-online,pmin0.7-n{N}' \
     --check-cut $D/2026-11-11_fn80_fn81_draft_yield/tables/cut.tsv &&
   nice -n 19 python3 -I $F/main_table.py replay > main.md; then
  touch done
else
  touch failed
fi
