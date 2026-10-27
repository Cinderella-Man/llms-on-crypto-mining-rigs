#!/usr/bin/env bash
# Offline estimate of the answer store (CPU only, nice 19), after build_stores.sh:
#   agentic  the 40 scored held-out agent sessions (RUN/seqs/test-rows.jsonl, 648 turns) with the store built without
#            the held-out modules' sessions (train), a threshold grid, and the pipeline model at the release split,
#            depths 3 and 4
#   sizes    the same sessions with prefixes of the train store's shuffled stream: 1, 2, 4 and 8 M tokens
#   q51      Q51's 38 transcripts with the full store and the train store (both leave out Q51's prompts)
# Env: BENCH, RUN, WORK as in build_stores.sh; Q51_GEN (Q51's generated answers, gen.jsonl); JOBS (default 8); STEPS.
set -euo pipefail
: "${BENCH:?}" "${RUN:?}" "${WORK:?}" "${Q51_GEN:?}"
O=$WORK/replay
TEST=$RUN/seqs/test-rows.jsonl
PY="nice -n 19 python3 -I $BENCH/ngram_replay/store_agentic.py"
JOBS=${JOBS:-8}
mkdir -p "$O"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$O/replay.log"; }
for m in 1 2 4 8; do   # prefixes of the shuffled stream: random subsets of the training turns
  [ -s "$WORK/train/agentic-${m}M.pstore" ] || nice -n 19 "$WORK/datastore" build "$WORK/train/corpus/clean.i32" \
    "${m}000000" "$WORK/train/agentic-${m}M.pstore" | tail -1 | tee -a "$O/replay.log"
done
sizes=(--store train-1M="$WORK/train/agentic-1M.pstore" --store train-2M="$WORK/train/agentic-2M.pstore"
       --store train-4M="$WORK/train/agentic-4M.pstore" --store train-8M="$WORK/train/agentic-8M.pstore")
for step in ${STEPS:-agentic sizes q51}; do
  case $step in
    agentic)
      log "agentic: held-out sessions, train store, threshold grid, pipeline at depths 3 and 4"
      $PY "$TEST" --lib "$WORK/libdatastore.so" --store train="$WORK/train/agentic.pstore" --accuracy \
        --grid 1,2,3,4,6:0,0.2,0.3,0.5,0.7 --sim --depths 3,4 --jobs "$JOBS" --out "$O/agentic.json" \
        > "$O/agentic.md" 2> "$O/agentic.err" ;;
    sizes)
      log "sizes: held-out sessions, train store prefixes of 1, 2, 4 and 8 M tokens"
      $PY "$TEST" --lib "$WORK/libdatastore.so" "${sizes[@]}" --accuracy --sim --depths 4 --jobs "$JOBS" \
        --out "$O/sizes.json" > "$O/sizes.md" 2> "$O/sizes.err" ;;
    q51)
      log "q51: Q51 transcripts, full and train stores"
      $PY "$Q51_GEN" --lib "$WORK/libdatastore.so" --store full="$WORK/full/agentic.pstore" \
        --store train="$WORK/train/agentic.pstore" --accuracy --grid 1,2,3:0,0.3,0.5 --sim --depths 3,4 --jobs "$JOBS" \
        --out "$O/q51.json" > "$O/q51.md" 2> "$O/q51.err" ;;
  esac
  log "$step done"
done
