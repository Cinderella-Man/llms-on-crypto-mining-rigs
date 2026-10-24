#!/usr/bin/env bash
# run_matrix.sh: the release benchmark matrix on the rig (bench/release_matrix/README.md). Stock llama.cpp, the v1
# engine and the v2 engine, each on every model file, on all nine cards: first at stock clocks (block "untuned"), then
# with a memory clock offset on every card (block "tuned"). The blocks never mix: every engine runs untuned before any
# offset is set, and the tuned block compares the engines under the same offsets.
#
# Per block, model and engine:
#   - llama-bench: a 512-token prompt, then 128 generated tokens in the same process (warm clocks), ROUNDS times;
#   - llama-server with the release command line (SERVER_ARGS; "<engine>+pipe" adds PIPE_ARGS): one greedy request,
#     whose hash in the tuned block must equal the untuned one (with stable memory the engine's output is
#     bit-identical); one warm-up request; then one chat request per prompt and seed with the makers' sampling.
# make_table.py turns $OUT/results.tsv into $OUT/tables.md, ready for the README.
#
# Run as root on the rig, from a copy of this folder, with nothing else on the cards:
#   BIN_v2=/opt/llama-builds/<v2 build>/bin ./run_matrix.sh                                        # untuned only
#   BIN_v2=... MEM_OFFSET=600 KIT=/tmp/oc/mem_offset-kit-20261024 ./run_matrix.sh                  # untuned, then tuned
#   BIN_v2=... MEM_OFFSET=600 KIT=... BLOCKS=tuned OUT=/tmp/release_matrix/<earlier TAG> ./run_matrix.sh   # tuned later
#
# Env [defaults]:
#   ENGINES [stock v1 v2]; BIN_<engine>: bin directories [stock: d77256999, v1: eeaa59660 under /opt/llama-builds; v2
#     must be given]; ENV_<engine>: extra environment [v2: LLAMA_NO_MOE_DOWN_RESCALE=tg; none for stock and v1]
#   SERVE [stock v1 v2 v2+pipe]: served configurations; PIPE_ARGS [--pipe-depth 3]
#   MODELS: LABEL=path pairs [UD-Q4_K_XL and K2h-r16 under /mnt/models]
#   GPUS [0,1,2,3,4,5,6,7,8], numbered in PCI bus order (CUDA_DEVICE_ORDER=PCI_BUS_ID), as NVML and nvidia-smi number them
#   ROUNDS [1], BENCH_ARGS [-ngl 99 -fa on -p 512 -n 128 -r 3]
#   SERVER_ARGS [the chat command line of the model pages, at the full 262,144-token context], N_GREEDY [128],
#     MAX_TOKENS [1000], SEEDS [1 2], PROMPTS [prompts/chat.txt prompts/code.txt], PORT [8089]
#   MEM_OFFSET (MT/s on every card of GPUS) or MEM_OFFSETS (one per card, comma-separated); KIT: the unpacked
#     memory-offset kit (bench/mem_offset), whose nvml_memoc.py sets the offsets; a card without its units file gets
#     the kit's `memoc_steps.sh units` first (~1 min)
#   BLOCKS [untuned, plus tuned when an offset is given], STRICT [1: a changed greedy hash ends the tuned block],
#   FORCE [0: refuse if a compute process is on the cards], TAG [date and time], OUT [/tmp/release_matrix/$TAG]
# Every offset goes back to 0 when the tuned block ends and whenever the script ends, fails or is interrupted.
# Time (est.): ~25 min per model and block with the defaults, so ~1 h 40 min for two models in both blocks.
# Exit status: 0 done, 1 a run failed, 2 usage or preflight, 3 the tuned block stopped on a changed output.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
R=/opt/llama-builds
ENGINES=${ENGINES:-stock v1 v2}
BIN_stock=${BIN_stock:-$R/llama.cpp-20261022-d77256999/bin}
BIN_v1=${BIN_v1:-$R/llama.cpp-20261023-eeaa59660-pascal-p106/bin}
BIN_v2=${BIN_v2:-}
ENV_v2=${ENV_v2-LLAMA_NO_MOE_DOWN_RESCALE=tg}
SERVE=${SERVE:-stock v1 v2 v2+pipe}
PIPE_ARGS=${PIPE_ARGS:---pipe-depth 3}
MODELS=${MODELS:-UD-Q4_K_XL=/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf K2h-r16=/mnt/models/flint/flint-q113-K2h-r16.gguf}
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8}
ROUNDS=${ROUNDS:-1}
BENCH_ARGS=${BENCH_ARGS:--ngl 99 -fa on -p 512 -n 128 -r 3}
SERVER_ARGS=${SERVER_ARGS:--ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 --no-webui}
N_GREEDY=${N_GREEDY:-128}
MAX_TOKENS=${MAX_TOKENS:-1000}
SEEDS=${SEEDS:-1 2}
PROMPTS=${PROMPTS:-$here/prompts/chat.txt $here/prompts/code.txt}
PORT=${PORT:-8089}
MEM_OFFSET=${MEM_OFFSET:-}
MEM_OFFSETS=${MEM_OFFSETS:-}
KIT=${KIT:-}
STRICT=${STRICT:-1}
FORCE=${FORCE:-0}
TAG=${TAG:-$(date +%Y%m%d-%H%M)}
OUT=${OUT:-/tmp/release_matrix/$TAG}
if [ -n "$MEM_OFFSET$MEM_OFFSETS" ]; then BLOCKS=${BLOCKS:-untuned tuned}; else BLOCKS=${BLOCKS:-untuned}; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT/raw"
RES=$OUT/results.tsv
META=$OUT/meta.tsv
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

log() { echo "$(date '+%F %T') $*" | tee -a "$OUT/matrix.log"; }
rec() { local IFS=$'\t'; echo "$*" >> "$RES"; }               # rec BLOCK MODEL CONFIG METRIC VALUE
meta() { printf '%s\t%s\n' "$1" "$2" >> "$META"; }
var() { local v="$1_$2"; echo "${!v:-}"; }                     # var BIN v2 -> $BIN_v2
on_cards() { local e=$1; shift; env CUDA_VISIBLE_DEVICES="$GPUS" $e "$@"; }

CHILD="" SRV="" WD="" SMI="" OFFSETS_SET=0 OFFSETS=()
killtree() { local p; for p in $(pgrep -P "$1"); do killtree "$p"; done; kill "$1" 2>/dev/null; }
reset_offsets() {
    local g
    for g in "${GPU_LIST[@]}"; do
        python3 "$KIT/nvml_memoc.py" reset "$g" >> "$OUT/offsets.log" 2>&1 \
            || log "WARNING: resetting gpu $g failed; run: python3 $KIT/nvml_memoc.py reset $g"
    done
    OFFSETS_SET=0
    log "memory offsets back to 0 on cards $GPUS"
}
cleanup() {
    local rc=$? p
    trap - EXIT INT TERM HUP
    for p in $CHILD $SRV $WD; do killtree "$p"; done
    [ -n "$SMI" ] && kill "$SMI" 2>/dev/null
    sleep 2  # let the CUDA processes exit before the offsets change under them
    [ "$OFFSETS_SET" = 1 ] && reset_offsets
    exit "$rc"
}
trap cleanup EXIT
trap 'log "interrupted"; exit 130' INT TERM HUP
# child CMD...: run CMD in the background and wait, so that a signal reaches the trap at once; the status is CMD's
child() { "$@" & CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=""; return "$rc"; }

preflight() {
    local e m bin bad=0
    command -v python3 > /dev/null && command -v curl > /dev/null || { echo "needs python3 and curl"; exit 2; }
    for e in $ENGINES; do
        bin=$(var BIN "$e")
        [ -n "$bin" ] || { echo "BIN_$e is not set"; bad=1; continue; }
        [ -x "$bin/llama-bench" ] && [ -x "$bin/llama-server" ] || { echo "no llama-bench or llama-server in $bin"; bad=1; }
    done
    for e in $SERVE; do case " $ENGINES " in *" ${e%+pipe} "*) ;; *) echo "SERVE has $e, whose engine is not in ENGINES"; bad=1;; esac; done
    for m in $MODELS; do [ -f "${m#*=}" ] || { echo "model file missing: ${m#*=}"; bad=1; }; done
    for e in $PROMPTS; do [ -f "$e" ] || { echo "prompt file missing: $e"; bad=1; }; done
    pgrep -x llama-server > /dev/null && { echo "a llama-server is running: stop it first"; bad=1; }
    if [ -n "$(nvidia-smi -i "$GPUS" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)" ] && [ "$FORCE" != 1 ]; then
        echo "compute processes on the cards (FORCE=1 to go on anyway):"; nvidia-smi -i "$GPUS" --query-compute-apps=pid,process_name --format=csv,noheader
        bad=1
    fi
    case " $BLOCKS " in *" tuned "*)
        [ "$(id -u)" = 0 ] || { echo "the tuned block needs root (NVML offsets)"; bad=1; }
        [ -n "$KIT" ] && [ -f "$KIT/nvml_memoc.py" ] && [ -f "$KIT/memoc_steps.sh" ] \
            || { echo "KIT must point to the unpacked memory-offset kit (nvml_memoc.py, memoc_steps.sh)"; bad=1; }
        if [ -n "$MEM_OFFSETS" ]; then IFS=',' read -r -a OFFSETS <<< "$MEM_OFFSETS"
        elif [ -n "$MEM_OFFSET" ]; then for e in "${GPU_LIST[@]}"; do OFFSETS+=("$MEM_OFFSET"); done
        else echo "the tuned block needs MEM_OFFSET or MEM_OFFSETS"; bad=1; fi
        [ "${#OFFSETS[@]}" = 0 ] || [ "${#OFFSETS[@]}" = "${#GPU_LIST[@]}" ] \
            || { echo "MEM_OFFSETS has ${#OFFSETS[@]} values for ${#GPU_LIST[@]} cards"; bad=1; }
        ;;
    esac
    [ "$bad" = 0 ] || exit 2
}

set_offsets() {
    local i g o
    for i in "${!GPU_LIST[@]}"; do
        g=${GPU_LIST[$i]} o=${OFFSETS[$i]}
        if [ ! -f "$KIT/memoc_units_gpu$g.json" ]; then
            log "gpu $g: no units file in the kit, measuring them (memoc_steps.sh units)"
            child env GPU="$g" OUT="$OUT/units" bash "$KIT/memoc_steps.sh" units >> "$OUT/offsets.log" 2>&1 \
                || { log "units on gpu $g failed, see $OUT/offsets.log"; return 1; }
        fi
        OFFSETS_SET=1
        python3 "$KIT/nvml_memoc.py" set "$g" "$o" >> "$OUT/offsets.log" 2>&1 \
            || { log "setting +$o MT/s on gpu $g was refused, see $OUT/offsets.log"; return 1; }
        log "gpu $g: memory +$o MT/s | $(python3 "$KIT/nvml_memoc.py" clocks "$g" 2>&1)"
    done
}

watchdog() {  # the 4 GB host: stop the server before the out-of-memory killer takes the machine
    while kill -0 "$1" 2> /dev/null; do
        local avail
        avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
        if [ "$avail" -lt "${MIN_AVAIL_MB:-400}" ]; then
            echo "watchdog: MemAvailable $avail MiB, stopping llama-server" >> "$OUT/matrix.log"; kill -9 "$1"; break
        fi
        sleep 1
    done
}

bench() {  # bench BLOCK LABEL MODEL ENGINE ROUND
    local blk=$1 lab=$2 m=$3 eng=$4 r=$5 f metric value
    f=$OUT/raw/$blk-$lab-$eng-bench$r
    log "$blk | $lab | $eng | llama-bench, round $r"
    if ! child on_cards "$(var ENV "$eng")" "$(var BIN "$eng")/llama-bench" -m "$m" $BENCH_ARGS -o jsonl > "$f.jsonl" 2> "$f.log"; then
        log "  llama-bench failed, see $f.log"; rec "$blk" "$lab" "$eng" error bench; return 1
    fi
    while IFS=$'\t' read -r metric value; do
        rec "$blk" "$lab" "$eng" "$metric" "$value"; log "  $metric $value tok/s"
    done < <(python3 "$here/make_table.py" bench "$f.jsonl")
}

serve() {  # serve BLOCK LABEL MODEL CONFIG; status 3 when the tuned output differs from the untuned one
    local blk=$1 lab=$2 m=$3 cfg=$4 eng=${4%+pipe} extra="" f sha ref med i
    [ "$cfg" != "$eng" ] && extra=$PIPE_ARGS
    f=$OUT/raw/$blk-$lab-$cfg
    log "$blk | $lab | $cfg | llama-server"
    # env execs the server, so $! is the server itself
    env CUDA_VISIBLE_DEVICES="$GPUS" $(var ENV "$eng") "$(var BIN "$eng")/llama-server" -m "$m" $SERVER_ARGS $extra \
        --host 127.0.0.1 --port "$PORT" > "$f-server.log" 2>&1 &
    SRV=$!
    watchdog "$SRV" & WD=$!
    for i in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
        kill -0 "$SRV" 2> /dev/null || break
        sleep 2
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
        log "  the server did not come up, see $f-server.log"; tail -5 "$f-server.log" | tee -a "$OUT/matrix.log"
        rec "$blk" "$lab" "$cfg" error server; stop_server; return 1
    fi
    if child python3 "$here/client.py" "$PORT" greedy "$N_GREEDY" ${PROMPTS%% *} > "$f-greedy.json"; then
        sha=$(python3 "$here/make_table.py" sha "$f-greedy.json")
        rec "$blk" "$lab" "$cfg" greedy_sha "$sha"
        ref=$(awk -F'\t' -v l="$lab" -v c="$cfg" '$1 == "untuned" && $2 == l && $3 == c && $4 == "greedy_sha" { s = $5 } END { print s }' "$RES")
        if [ "$blk" = tuned ]; then
            if [ -z "$ref" ]; then rec "$blk" "$lab" "$cfg" exact unknown; log "  greedy hash: no untuned reference"
            elif [ "$sha" = "$ref" ]; then rec "$blk" "$lab" "$cfg" exact same; log "  greedy output identical to untuned"
            else
                rec "$blk" "$lab" "$cfg" exact DIFFERENT; log "  greedy output DIFFERS from untuned: the offset is not stable"
                [ "$STRICT" = 1 ] && { stop_server; return 3; }
            fi
        fi
    else
        log "  the greedy request failed"; rec "$blk" "$lab" "$cfg" error greedy
    fi
    child python3 "$here/client.py" "$PORT" chat 64 99 ${PROMPTS%% *} > /dev/null  # warm-up
    if child python3 "$here/client.py" "$PORT" chat "$MAX_TOKENS" "$SEEDS" $PROMPTS > "$f-chat.jsonl"; then
        med=$(python3 "$here/make_table.py" served "$f-chat.jsonl")
        if [ -n "$med" ]; then
            IFS=$'\t' read -r m1 m2 m3 m4 <<< "$med"
            rec "$blk" "$lab" "$cfg" served "$m1"; rec "$blk" "$lab" "$cfg" served_range "$m2-$m3"; rec "$blk" "$lab" "$cfg" served_n "$m4"
            log "  served: median $m1 tok/s ($m2-$m3, $m4 requests)"
        fi
    else
        log "  a chat request failed, see $f-chat.jsonl"; rec "$blk" "$lab" "$cfg" error chat
    fi
    grep -E 'pipelined =' "$f-server.log" | tail -3 >> "$f-pipe.txt" 2>/dev/null
    stop_server
}
stop_server() {
    [ -n "$SRV" ] && { killtree "$SRV"; wait "$SRV" 2>/dev/null; }
    [ -n "$WD" ] && { kill "$WD" 2>/dev/null; wait "$WD" 2>/dev/null; }
    SRV="" WD=""
    sleep 5
}

block() {  # block untuned|tuned
    local blk=$1 xid0 t0 lm lab m e r order rc
    log "== block $blk"
    if [ "$blk" = tuned ]; then
        set_offsets || { reset_offsets; return 1; }
        meta tuned_offsets "$(IFS=,; echo "${OFFSETS[*]}")"
    fi
    xid0=$(dmesg 2>/dev/null | wc -l); t0=$(date +%s)
    nvidia-smi -i "$GPUS" --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
        --format=csv,noheader,nounits -lms 1000 >> "$OUT/smi-$blk.csv" 2>&1 &
    SMI=$!
    for lm in $MODELS; do
        lab=${lm%%=*} m=${lm#*=}
        order=$ENGINES
        for r in $(seq 1 "$ROUNDS"); do
            for e in $order; do bench "$blk" "$lab" "$m" "$e" "$r"; done
            order=$(echo $order | tr ' ' '\n' | tac | tr '\n' ' ')  # alternate the order between rounds
        done
        for e in $SERVE; do
            serve "$blk" "$lab" "$m" "$e"; rc=$?
            [ "$rc" = 3 ] && { kill "$SMI"; SMI=""; sleep 2; reset_offsets; return 3; }
        done
    done
    kill "$SMI" 2>/dev/null; SMI=""
    local xid
    xid=$({ dmesg 2>/dev/null | tail -n +"$((xid0 + 1))"; journalctl -k -q --since "@$t0" 2>/dev/null; } | grep -i 'NVRM: Xid' | sort -u | wc -l)
    meta "xid_$blk" "$xid"
    meta "mem_clock_max_$blk" "$(awk -F', ' '$5 + 0 > m { m = $5 + 0 } END { print m + 0 }' "$OUT/smi-$blk.csv")"
    log "block $blk done: $xid Xid lines; highest memory clock $(awk -F', ' '$5 + 0 > m { m = $5 + 0 } END { print m + 0 }' "$OUT/smi-$blk.csv") MHz"
    [ "$blk" = tuned ] && { sleep 2; reset_offsets; }
    return 0
}

preflight
log "run_matrix.sh: blocks [$BLOCKS], engines [$ENGINES], served [$SERVE], cards $GPUS, out $OUT"
meta cards "${#GPU_LIST[@]}"; meta n_greedy "$N_GREEDY"; meta max_tokens "$MAX_TOKENS"; meta seeds "$SEEDS"
meta prompts "$(for p in $PROMPTS; do basename "$p"; done | tr '\n' ' ')"; meta pipe_args "$PIPE_ARGS"
meta bench_args "$BENCH_ARGS"; meta server_args "$SERVER_ARGS"
for e in $ENGINES; do
    meta "build_$e" "$(basename "$(dirname "$(var BIN "$e")")")"
    meta "env_$e" "$(var ENV "$e")"
done
status=0
for b in $BLOCKS; do
    block "$b"; rc=$?
    [ "$rc" = 3 ] && { status=3; log "tuned block stopped: an output changed under the offset"; break; }
    [ "$rc" != 0 ] && status=1
done
grep -q $'\terror\t' "$RES" 2>/dev/null && [ "$status" = 0 ] && status=1
python3 "$here/make_table.py" tables "$OUT" > "$OUT/tables.md" && log "tables: $OUT/tables.md"
cat "$OUT/tables.md"
exit "$status"
