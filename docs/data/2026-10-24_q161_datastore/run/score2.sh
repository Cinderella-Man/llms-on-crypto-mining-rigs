#!/usr/bin/env bash
# Q161 (2026-10-25): strict v1 (data sets + stdlib/llama.cpp files), v2 (+ permissive site-packages) and v2 notool
# (without ToolACE/glaive/hermes) against the old store, Q51 held out, depth 3.
# WORK: work dir with src/ (a copy of bench/ngram_replay), libdatastore*.so (datastore.cpp built with -shared, see
# bench/ngram_replay/README.md) and the stores; Q51_GEN: release/flint-v2/reproduce/data/q51/gen.jsonl. CPU only.
: "${WORK:?}" "${Q51_GEN:?}"
cd "$WORK"
S="--store old=pstore/flint-answers-clean-13.8M.pstore --store strict-v1=pstore/flint-answers-strict-13.8M.pstore --store strict-v2=pstore/flint-answers-strict-v2-13.8M.pstore --store strict-v2-notool=pstore/flint-answers-strict-v2-notool-13.8M.pstore"
nice -n 19 python3 src/datastore_sim.py "$Q51_GEN" --lib ./libdatastore-v3.so $S \
    --policies "L >= 2, share >= 0.3" --kinds req2ds --accuracy --sim --depths 3 --jobs 8 --out strict/score2.json \
    > strict/score2.md 2> strict/score2.log
echo "exit $?" >> strict/score2.log
