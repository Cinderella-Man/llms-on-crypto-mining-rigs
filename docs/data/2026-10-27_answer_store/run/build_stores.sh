#!/usr/bin/env bash
# The answer store from agentic data run 1, built and gated as bench/agentic_data/run.sh's checks step does (CPU only,
# nice 19). Two stores:
#   full   every gated session (1,177): the store to ship
#   train  only the training sessions of the MTP head's split (RUN/ids/train.txt, 1,108): the held-out modules'
#          sessions are left out, for the offline estimate on RUN/seqs/test-rows.jsonl
# Env: BENCH (this repository's bench/), RUN (bench/staircase_drafter/agentic_head.sh's WORK: export/train/ with
# turns.jsonl, turn_prompts.jsonl, sessions_seq.jsonl and their gate records; ids/train.txt), Q51 (Q51's prompts.jsonl),
# WORK (output folder), STORES (default "full train").
set -euo pipefail
: "${BENCH:?}" "${RUN:?}" "${Q51:?}" "${WORK:?}"
T=$RUN/export/train
SRC=$BENCH/staircase_drafter/sources.json
PY="nice -n 19 python3 -I"
CS="$PY $BENCH/staircase_drafter/check_sources.py $SRC derived"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$WORK/build.log"; }
mkdir -p "$WORK"
nice -n 19 g++ -O2 -std=c++17 -DDS_MAIN -o "$WORK/datastore" "$BENCH/ngram_replay/datastore.cpp"
nice -n 19 g++ -O2 -std=c++17 -shared -fPIC -o "$WORK/libdatastore.so" "$BENCH/ngram_replay/datastore.cpp"

for name in ${STORES:-full train}; do
  C=$WORK/$name; mkdir -p "$C"
  parents=("$T/answers.gate.json" "$T/turn_prompts.gate.json"); extra=""
  if [ "$name" = train ]; then
    parents+=("$T/seq.gate.json"); extra=" --sessions $RUN/ids/train.txt $T/sessions_seq.jsonl"
  fi
  corpus_cmd="python3 -I datastore_corpus.py $C/corpus $T/turn_prompts.jsonl $Q51 $T/turns.jsonl$extra"
  log "$name: corpus ($corpus_cmd)"
  # shellcheck disable=SC2086
  $PY "$BENCH/ngram_replay/datastore_corpus.py" "$C/corpus" "$T/turn_prompts.jsonl" "$Q51" "$T/turns.jsonl" $extra \
    | tee -a "$WORK/build.log"
  $CS "$C"/corpus/{clean.i32,all.i32,noagent.i32,index.json} --kind other --parent "${parents[@]}" --test "$Q51" \
    --allow-dir "$C/corpus" --cmd "$corpus_cmd" --out "$C/corpus.gate.json" | tail -1 | tee -a "$WORK/build.log"
  log "$name: store"
  nice -n 19 "$WORK/datastore" build "$C/corpus/clean.i32" 99999999 "$C/agentic.pstore" | tail -3 | tee -a "$WORK/build.log"
  $CS "$C/agentic.pstore" --kind store --parent "$C/corpus.gate.json" --tool "$WORK/datastore" \
    --cmd "$WORK/datastore build $C/corpus/clean.i32 99999999 $C/agentic.pstore" --out "$C/store.gate.json" \
    | tail -1 | tee -a "$WORK/build.log"
  sha256sum "$C/agentic.pstore" | tee -a "$WORK/build.log"
done
# the whole provenance chain of the store to ship, and the held-out tests re-checked on its sessions
$PY "$BENCH/agentic_data/chain.py" "$WORK/full/store.gate.json" --out "$WORK/full/chain.json" | tee -a "$WORK/build.log"
if [ -n "${SESSIONS:-}" ]; then   # the runs' sessions/ folders (agent.py run), for heldout_check.py
  # shellcheck disable=SC2086
  $PY "$BENCH/agentic_data/heldout_check.py" $SESSIONS --agent-tasks "$BENCH/agent_check/tasks" --q51 "$Q51" \
    --only "$T/sessions_seq.jsonl" --out "$WORK/heldout_check.json" | tee -a "$WORK/build.log"
fi
log "done"
