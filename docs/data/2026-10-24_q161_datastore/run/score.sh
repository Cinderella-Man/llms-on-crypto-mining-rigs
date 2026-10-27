#!/usr/bin/env bash
# Q161 (2026-10-25): the strict store against the old one on Q51 held out: one-token accuracy and the nine-card
# pipeline sim at depth 3, the engine chain (copy >= 4, else request counts L >= 2, else store, else counts).
# WORK: work dir with src/ (a copy of bench/ngram_replay), libdatastore*.so (datastore.cpp built with -shared, see
# bench/ngram_replay/README.md) and the stores; Q51_GEN: release/flint-v2/reproduce/data/q51/gen.jsonl. CPU only.
: "${WORK:?}" "${Q51_GEN:?}"
cd "$WORK"
S="--store old=pstore/flint-answers-clean-13.8M.pstore --store strict=pstore/flint-answers-strict-13.8M.pstore"
nice -n 19 python3 src/datastore_sim.py "$Q51_GEN" --lib ./libdatastore-v3.so $S \
    --policies "L >= 2, share >= 0.3" --kinds req2ds --accuracy --sim --depths 3 --jobs 8 --out strict/score.json \
    > strict/score.md 2> strict/score.log
echo "exit $?" >> strict/score.log
