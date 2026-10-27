#!/usr/bin/env bash
# Q161 (2026-10-25): builds the core + OpenR1-Math (default) store with store_build.py (copied into $WORK/strict with
# its id list), then the Q51 sim next to core and strict v2 and the old store (-> strict/score4.{md,json}).
# WORK: work dir with src/ (a copy of bench/ngram_replay), libdatastore*.so (datastore.cpp built with -shared, see
# bench/ngram_replay/README.md) and the stores; Q51_GEN: release/flint-v2/reproduce/data/q51/gen.jsonl. CPU only.
: "${WORK:?}" "${Q51_GEN:?}"
cd "$WORK/strict"
nice -n 19 python3 -I store_build.py core-openr1-13.8M --ids strict-ids-notool.txt --allow core-openr1.allow > core-openr1-13.8M.out 2>&1
cd ..
nice -n 10 ./datastore build strict/core-openr1-13.8M.i32 99999999 pstore/flint-answers-core-openr1-13.8M.pstore > strict/core-openr1-13.8M.build.log 2>&1
md5sum pstore/flint-answers-core-openr1-13.8M.pstore >> strict/core-openr1-13.8M.out
S="--store old=pstore/flint-answers-clean-13.8M.pstore --store strict-v2=pstore/flint-answers-strict-v2-13.8M.pstore --store core=pstore/flint-answers-core-13.8M.pstore --store core-openr1=pstore/flint-answers-core-openr1-13.8M.pstore"
nice -n 19 python3 src/datastore_sim.py "$Q51_GEN" --lib ./libdatastore-v3.so $S \
    --policies "L >= 2, share >= 0.3" --kinds req2ds --accuracy --sim --depths 3 --jobs 8 --out strict/score4.json \
    > strict/score4.md 2> strict/score4.log
echo "exit $?" >> strict/score4.log
