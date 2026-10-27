#!/usr/bin/env bash
# Q161 (2026-10-24): nine-card pipeline simulation of the store chains (3/7/13.8 M-token stores, all-14M, noagent-13M;
# made with bench/ngram_replay/datastore_corpus.py), depths 1/2/3/4/8, 8 processes -> out/sim.{md,json}.
# WORK: work dir with src/ (a copy of bench/ngram_replay), libdatastore*.so (datastore.cpp built with -shared, see
# bench/ngram_replay/README.md) and the stores; Q51_GEN: release/flint-v2/reproduce/data/q51/gen.jsonl. CPU only.
: "${WORK:?}" "${Q51_GEN:?}"
cd "$WORK"
S="--store clean-3M=stores/clean-3M.ds --store clean-7M=stores/clean-7M.ds --store clean-14M=stores/clean-14M.ds --store all-14M=stores/all-14M.ds --store noagent-13M=stores/noagent-13M.ds"
nice -n 10 python3 src/datastore_sim.py "$Q51_GEN" --lib ./libdatastore.so $S \
  --policies "L >= 2, share >= 0.3" --kinds ds,req2ds --sim --depths 1,2,3,4,8 --jobs 8 --out out/sim.json > out/sim.md 2> out/sim.log
echo "rc=$?" >> out/sim.log
