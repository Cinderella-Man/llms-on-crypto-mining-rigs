#!/usr/bin/env bash
# Q550: expected gain of pipelined speculation for Qwen3.8-Flash-Next on nine P106 cards, from the event model of
# bench/ngram_replay/pipeline_sim.py replaying Q51's 38 Flint transcripts (chat, code, edits, agent; 65,558 generated
# tokens) with the guessers llama-server has (copy = copy >= 2; cont = copy >= 4, else the request's counts; staircase
# with --pipe-store = copy >= 4, else the request's counts of 2+ token contexts, else the store) and oracles.
# Stages: Flash-Next's GPU time per card from Q215's decode profile (v3, Swift Q2_0, -ts 6,5,5,6,5,5,6,5,6, ms per
# token: 15.57 14.77 13.96 18.09 14.27 14.98 16.23 11.99 16.09; card 8 holds the output layer, which the model adds as
# its 2.5 ms head), scaled to a plain token of 112, 140 and 185 ms (the 8.9, 7.1 and 5.4 tok/s Q215 measured), hops of
# 0.6 ms between cards and 3 ms of host time from a sample to the next launch. The transcripts are Flint's, so the
# guess rates are Flint's (same tasks, a different model and tokenizer): an estimate, not a measurement.
# Usage: SIM=<bench/ngram_replay> GEN=<q51 gen.jsonl> OUT=<dir> [JOBS=4] [HOP=0.6] [GAP=3] bash sim.sh
set -euo pipefail
: "${SIM:?set SIM}" "${GEN:?set GEN}" "${OUT:?set OUT}"
mkdir -p "$OUT"
L=1557,1477,1396,1809,1427,1498,1623,1199,1359
args=()
for T in 112 140 185; do args+=(--setup "fn-${T}ms=$T:$L"); done
nice -n 19 python3 -I "$SIM/pipeline_sim.py" "$GEN" "${args[@]}" --depths 1,2,3,4,5,6 --hop "${HOP:-0.6}" \
    --gap "${GAP:-3}" --jobs "${JOBS:-4}" \
    --only 'copy, match >= 2 \||request counts \(no table\)|share >= 0.3|oracle p (0.5|0.7|0.9|1.0) \| depth' \
    --out "$OUT/sim-fn.json" > "$OUT/sim-fn.txt" 2> "$OUT/sim-fn.err"
echo "exit $?"; tail -n 5 "$OUT/sim-fn.txt"
