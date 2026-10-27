#!/usr/bin/env bash
# Q161 (2026-10-25): the checked core-b and release-b stores (both Q51 rules) against the old store, depth 3.
# WORK: work dir with src/ (a copy of bench/ngram_replay), libdatastore*.so (datastore.cpp built with -shared, see
# bench/ngram_replay/README.md) and the stores; Q51_GEN: release/flint-v2/reproduce/data/q51/gen.jsonl. CPU only.
: "${WORK:?}" "${Q51_GEN:?}"
cd "$WORK"
S="--store old=pstore/flint-answers-allsources-q51heldout-13.8M.pstore --store core-b=pstore/flint-answers-core-20261025b-13.8M.pstore --store release-b=pstore/flint-answers-release-20261025b-13.8M.pstore"
nice -n 19 python3 src/datastore_sim.py "$Q51_GEN" --lib ./libdatastore-v3.so $S \
    --policies "L >= 2, share >= 0.3" --kinds req2ds --accuracy --sim --depths 3 --jobs 8 --out strict/score5.json \
    > strict/score5.md 2> strict/score5.log
echo "exit $?" >> strict/score5.log
