#!/usr/bin/env bash
# Q161 (2026-10-25): the licence variants (store_variants.py) against the old store ("current") on Q51 held out: one-token
# accuracy and the nine-card pipeline sim at depth 3, the engine chain.
# WORK: work dir with src/ (a copy of bench/ngram_replay), libdatastore*.so (datastore.cpp built with -shared, see
# bench/ngram_replay/README.md) and the stores; Q51_GEN: release/flint-v2/reproduce/data/q51/gen.jsonl. CPU only.
: "${WORK:?}" "${Q51_GEN:?}"
cd "$WORK"
S="--store current=pstore/flint-answers-clean-13.8M.pstore --store noshare-same=pstore/flint-answers-noshare-same.pstore --store strict-same=pstore/flint-answers-strict-same.pstore --store noshare-all=pstore/flint-answers-noshare-all.pstore --store strict-all=pstore/flint-answers-strict-all.pstore"
nice -n 19 python3 src/datastore_sim.py "$Q51_GEN" --lib ./libdatastore-v3.so $S \
    --policies "L >= 2, share >= 0.3" --kinds req2ds --accuracy --sim --depths 3 --jobs 8 --out variants/score.json \
    > variants/score.md 2> variants/score.log
echo "exit $?" >> variants/score.log
