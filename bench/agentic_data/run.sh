#!/usr/bin/env bash
# The agentic data pipeline end to end on a GPU machine with Docker, gated at every stage (README.md). The same script
# runs the pilot (the defaults) and the long run (more modules or code bases, more tasks per module, vLLM).
#
#   WORK=<folder> MODEL=<Flint GGUF> HF_DIR=<RecursiveSI/flint small files> Q51=<Q51 prompts.jsonl> \
#   LLAMA_SERVER=<llama-server binary> MOUNTS="-v a:a ..." bash bench/agentic_data/run.sh
#
# Environment:
#   WORK          output folder (scans/, ws/, texts.json, s1/, s2/, export/, run.log)
#   API           llama (default: llama-server in SERVER_IMAGE on GPUS) or vllm (a vLLM server already answering
#                 on PORT, e.g. bench/staircase_drafter/vllm/serve_vllm.sh; MODEL is then the NVFP4 GGUF, WRITER
#                 flint-nvfp4-vllm)
#   WRITER        the writer's id in sources.json (default flint-q8_0)
#   GPUS PORT NP  llama-server's cards (default 3,4), port (18640) and slots (16); CTX_SLOT tokens per slot (32768)
#   MODULES       the standard-library modules (workspace.py stdlib) for the stage-1 workspaces, or "all"
#   ATTEMPTS      stage-1 tasks per module (default 1); REPEAT sessions per task, other seeds (default 1)
#   TEXTS         the harness texts (default WORK/texts.json, written by the texts step; the long run uses the
#                 owner-reviewed bench/agentic_data/texts.json and leaves that step out)
#   CLAIMS        a folder shared with another run.sh on the same tasks (another writer): each task is claimed there
#   TOOLS         the tool set per session: agx (ours), pi (the pi agent's names and parameters) or mixed (default)
#   TEXTS_FROM    texts step only: an earlier texts.json whose texts are kept (only missing ones are written)
#   PER_MODULE IMPLEMENT  stage-2 repair and implementation tasks per module (3, 1)
#   CONC          sessions at a time (default NP); MAX_TURNS MAX_GEN MAX_CTX TIME_CAP (24, 20000, CTX_SLOT, 30 min)
#   STEPS         default "image workspaces server texts s1 s2 gate export checks stop"; any subset, in order
#   STOP_FILE     create it to start no new sessions (default WORK/STOP)
# Every container carries the label pascal-quants=1 and --cpu-shares 256; the script's own Python runs at nice 19.
# Quality and token counts only: times on a shared machine are not speed results.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${WORK:?}" "${HF_DIR:?}" "${Q51:?}"
API=${API:-llama}; WRITER=${WRITER:-flint-q8_0}
GPUS=${GPUS:-3,4}; PORT=${PORT:-18640}; NP=${NP:-16}; CTX_SLOT=${CTX_SLOT:-32768}
MODULES=${MODULES:-graphlib,getopt,reprlib,colorsys,fnmatch,shlex,fractions,netrc}
PER_MODULE=${PER_MODULE:-3}; IMPLEMENT=${IMPLEMENT:-1}
CONC=${CONC:-$NP}; MAX_TURNS=${MAX_TURNS:-24}; MAX_GEN=${MAX_GEN:-20000}; MAX_CTX=${MAX_CTX:-$CTX_SLOT}
ATTEMPTS=${ATTEMPTS:-1}; TEXTS=${TEXTS:-$WORK/texts.json}; CLAIMS=${CLAIMS:-}; TOOLS=${TOOLS:-mixed}
REPEAT=${REPEAT:-1}
TIME_CAP=${TIME_CAP:-30}
STEPS=${STEPS:-image workspaces server texts s1 s2 gate export checks stop}
STOP_FILE=${STOP_FILE:-$WORK/STOP}
SERVER_IMAGE=${SERVER_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PY_IMAGE=python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2
NAME=agx-srv-${GPUS//,/}
SRC=$HERE/../staircase_drafter/sources.json; CB=$HERE/codebases.json; AGT=$HERE/../agent_check/tasks
URL=http://127.0.0.1:$PORT
PY="nice -n 19 python3 -I"
mkdir -p "$WORK"
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$WORK/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
held=(--q51 "$Q51" --agent-tasks "$AGT")
common=(--server "$URL" --api "$API" --template "$HF_DIR/chat_template.jinja" --writer "$WRITER" --model-file "${MODEL:-}")

if has image; then
  log "sandbox image"
  docker build -q --label pascal-quants=1 -t agx-sandbox:py312 "$HERE/sandbox" >> "$WORK/run.log"
fi

if has workspaces; then
  log "licence scan of CPython's Lib and the stage-1 workspaces"
  mkdir -p "$WORK/scans"
  docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none -v "$HERE:$HERE:ro" -v "$WORK:$WORK" \
    "$PY_IMAGE" python3 -I "$HERE/license_scan.py" /usr/local/lib/python3.12 --license PSF \
    --holders "Python Software Foundation|PSF" --exclude "site-packages/*" "config-3.12*" "lib-dynload/*" \
    --out "$WORK/scans/cpython-3.12.15-lib" >> "$WORK/run.log"
  $PY "$HERE/workspace.py" stdlib --modules "$MODULES" --scan "$WORK/scans/cpython-3.12.15-lib.tsv" \
    --image "$PY_IMAGE" --out "$WORK/ws" >> "$WORK/run.log" 2>&1
  $PY "$HERE/gate.py" workspaces "$WORK"/ws/*.manifest.json --sources "$SRC" --codebases "$CB" \
    --scan "$WORK/scans/cpython-3.12.15-lib.tsv" "${held[@]}" --out "$WORK/ws.gate.json" | tee -a "$WORK/run.log"
fi

if has server && [ "$API" = llama ]; then
  : "${MODEL:?}" "${LLAMA_SERVER:?}"
  log "llama-server on GPUs $GPUS, $NP slots of $CTX_SLOT tokens"
  docker rm -f "$NAME" > /dev/null 2>&1 || true
  # shellcheck disable=SC2086
  docker run -d --name "$NAME" --label pascal-quants=1 --network host --gpus "\"device=$GPUS\"" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID ${MOUNTS:-} "$SERVER_IMAGE" "$LLAMA_SERVER" -m "$MODEL" -ngl 99 -fa on \
    -np "$NP" -c $((NP * CTX_SLOT)) -no-kvu -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 -bs --cache-ram 0 --jinja \
    --metrics --no-webui --host 127.0.0.1 --port "$PORT" > /dev/null
  until curl -sf "$URL/health" > /dev/null; do
    sleep 10
    [ "$(docker inspect -f '{{.State.Status}}' "$NAME" 2> /dev/null)" = running ] \
      || { log "server died while loading"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  done
  log "server ready"
fi

if has texts; then
  log "Flint writes the harness texts"
  $PY "$HERE/agent.py" texts "${common[@]}" --spec "$HERE/texts_spec.json" --out "$TEXTS" --conc "$CONC" \
    --heldout-q51 "$Q51" --heldout-tasks "$AGT" ${TEXTS_FROM:+--extend "$TEXTS_FROM"} | tee -a "$WORK/run.log"
fi

run_stage() {   # $1 stage folder, $2 tasks file, $3.. extra gate parents
  local d=$1 tasks=$2; shift 2
  $PY "$HERE/gate.py" tasks "$tasks" --parent "$WORK/ws.gate.json" "$@" --sources "$SRC" --codebases "$CB" \
    --texts "$TEXTS" "${held[@]}" --out "$d/tasks.gate.json" | tee -a "$WORK/run.log"
  $PY "$HERE/agent.py" run "${common[@]}" --tasks "$tasks" --texts "$TEXTS" --out "$d" \
    --work "$d/work" --conc "$CONC" --max-turns "$MAX_TURNS" --max-gen "$MAX_GEN" --max-ctx "$MAX_CTX" \
    --time-cap-min "$TIME_CAP" --run-tag "agx-$(basename "$d")" --stop-file "$STOP_FILE" \
    --tools "$TOOLS" --repeat "$REPEAT" ${CLAIMS:+--claims "$CLAIMS/$(basename "$d")"}
  tail -3 "$d/run.log" | tee -a "$WORK/run.log"
  $PY "$HERE/gate.py" sessions "$d/sessions" --parent "$d/tasks.gate.json" --sources "$SRC" --codebases "$CB" \
    --texts "$TEXTS" "${held[@]}" --out "$d/sessions.gate.json" | tee -a "$WORK/run.log" || true
}

if has s1; then
  log "stage 1: write-tests sessions"
  mkdir -p "$WORK/s1"
  $PY "$HERE/tasks.py" write-tests --workspaces "$WORK/ws" --texts "$TEXTS" --out "$WORK/s1/tasks.jsonl" \
    --attempts "$ATTEMPTS" \
    | tee -a "$WORK/run.log"
  run_stage "$WORK/s1" "$WORK/s1/tasks.jsonl"
fi

if has s2; then
  log "stage 2: repair and implementation tasks on the stage-1 tests"
  mkdir -p "$WORK/s2"
  $PY "$HERE/tasks.py" from-sessions --sessions "$WORK/s1/sessions" --workspaces "$WORK/ws" --texts "$TEXTS" \
    --out "$WORK/s2/tasks" --per-module "$PER_MODULE" --implement "$IMPLEMENT" | tee -a "$WORK/run.log"
  run_stage "$WORK/s2" "$WORK/s2/tasks/tasks.jsonl" "$WORK/s1/sessions.gate.json"
fi

if has gate; then
  log "the sessions gate over both stages"
  $PY "$HERE/gate.py" sessions "$WORK/s1/sessions" "$WORK/s2/sessions" --parent "$WORK/s1/tasks.gate.json" \
    "$WORK/s2/tasks.gate.json" --sources "$SRC" --codebases "$CB" --texts "$TEXTS" "${held[@]}" \
    --out "$WORK/sessions.gate.json" | tee -a "$WORK/run.log" || true
fi

if has export; then
  log "exports (store, dump, dataset), only sessions the gate passed"
  $PY "$HERE/export.py" "$WORK/sessions.gate.json" --codebases "$CB" --apache "$HF_DIR/LICENSE" \
    --out "$WORK/export" | tee -a "$WORK/run.log"
  $PY "$HERE/gate.py" answers "$WORK/export/train/turns.jsonl" "$WORK/export/train/sessions_seq.jsonl" \
    --parent "$WORK/sessions.gate.json" --sources "$SRC" --codebases "$CB" --out "$WORK/export/train/answers.gate.json" \
    | tee -a "$WORK/run.log"
  $PY "$HERE/gate.py" answers "$WORK/export/train/turn_prompts.jsonl" --parent "$WORK/sessions.gate.json" \
    --sources "$SRC" --codebases "$CB" --out "$WORK/export/train/turn_prompts.gate.json" | tee -a "$WORK/run.log"
  $PY "$HERE/gate.py" dataset "$WORK/export/hf" --parent "$WORK/sessions.gate.json" --sources "$SRC" --codebases "$CB" \
    "${held[@]}" --out "$WORK/export/hf.gate.json" | tee -a "$WORK/run.log"
fi

if has checks; then
  log "checks: the answer store, built and recorded through check_sources.py, and the dump's sequence file"
  mkdir -p "$WORK/checks"
  T=$WORK/export/train; C=$WORK/checks; CS="$PY $HERE/../staircase_drafter/check_sources.py $SRC derived"
  corpus_cmd="python3 -I datastore_corpus.py $C/corpus $T/turn_prompts.jsonl $Q51 $T/turns.jsonl"
  $PY "$HERE/../ngram_replay/datastore_corpus.py" "$C/corpus" "$T/turn_prompts.jsonl" "$Q51" "$T/turns.jsonl" \
    | tee -a "$WORK/run.log"
  $CS "$C"/corpus/{clean.i32,all.i32,noagent.i32,index.json} --kind other --parent "$T/answers.gate.json" \
    "$T/turn_prompts.gate.json" --test "$Q51" --allow-dir "$C/corpus" --cmd "$corpus_cmd" --out "$C/corpus.gate.json" \
    | tail -1 | tee -a "$WORK/run.log"
  nice -n 19 g++ -O2 -std=c++17 -DDS_MAIN -o "$C/datastore" "$HERE/../ngram_replay/datastore.cpp"
  nice -n 19 "$C/datastore" build "$C/corpus/clean.i32" 99999999 "$C/agentic.pstore" | tail -3 | tee -a "$WORK/run.log"
  $CS "$C/agentic.pstore" --kind store --parent "$C/corpus.gate.json" --tool "$C/datastore" \
    --cmd "$C/datastore build $C/corpus/clean.i32 99999999 $C/agentic.pstore" --out "$C/store.gate.json" \
    | tail -1 | tee -a "$WORK/run.log"
  docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none --entrypoint python3 \
    -v "$HERE/..:$HERE/..:ro" -v "$WORK:$WORK" "${FEATS_IMAGE:-vllm/vllm-openai@sha256:1440cb63fac5fa19040c295fcd5c6e6d551a506aa72a68fe3425cfd6ef8cf4b8}" \
    "$HERE/../staircase_drafter/feats.py" seqs "$WORK/checks/seqs.txt" "$WORK/checks/map.jsonl" \
    "$WORK/export/train/sessions_seq.jsonl" --max-tok 1000000 | tee -a "$WORK/run.log"
fi

if has stop; then
  docker rm -f "$NAME" > /dev/null 2>&1 || true
  docker ps -aq --filter "label=agx-run" | xargs -r docker rm -f > /dev/null 2>&1 || true
  log "server and sandboxes stopped"
fi
