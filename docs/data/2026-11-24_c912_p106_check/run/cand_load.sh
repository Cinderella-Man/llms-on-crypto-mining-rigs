#!/usr/bin/env bash
# C912's P106 check (Q1649): ../../2026-11-17_v5_cand_p106_check/run/cand_load.sh with one change, the release lines
# taken from release/*-$REL (REL, default v5: the v5 presets) instead of release/*-v4, and a third greedy request
# ('long', PROMPTS) of about 400 prompt tokens, so each load also reads a prompt longer than one 64-token micro-batch
# (Flint's and Flash-Next's --ubatch-taper short=...:64 cut it into 64-token passes; Ornith reads it in one pass).
# Its own header follows.
# The P106 check of the v5 candidate (Q1525), parts (3) and (4) on one rig: do the release files give the same greedy
# replies, the same memory per card and the same launch groups per evaluation on a new build (NEW) as on a reference
# build (REF)? R153's ../../2026-11-13_c831_upstream/run/r153_load.sh with three changes:
#   - each file is loaded with its release line (release/*-$REL/settings.env; R153's run used v4.0's), every switch
#     the line does not set at its default; Flint Pascal Q5 GGUF on Flint's line without -ts (release/flint-v4/README.md);
#     no load without --spec-draft-n-ubatch (the v4 and v5 lines set 512);
#   - both builds' servers run with the fork's profiler in its light mode (PROFILE=1: GGML_CUDA_PROFILE_MODE=graph, one
#     event pair around each whole evaluation, CUDA graphs and fusions unchanged), so each load also writes every
#     evaluation's launch groups per card (part 4: a fusion that stops matching shows as more groups);
#   - the greedy step runs the builds in GREEDY_BUILDS (default "REF NEW"); with GREEDY_EXPECT ('<model>=<md5> ...',
#     a build's text recorded earlier with the same command) a model whose NEW text differs from it is run on REF too.
# Run from the repository root on a machine that only runs ssh. Per model, in this order:
#   load    one llama-server per build (NEW, then REF), untimed, under bench/agent_check/rig_server.sh (RAM watchdog,
#           monitors): memory per card (nvidia-smi) 20 s after it answers, the host's MemAvailable then, the load log;
#           then greedy requests to the server (/completion, temperature 0, 256 tokens, no prompt cache): a coding
#           request, a prose paragraph and the long review request (PROMPTS); the profile (prof.csv) comes back with
#           the logs
#   greedy  docs/data/2026-11-05_ladder5/run/rig_greedy.sh: llama-completion decodes the coding request greedily on
#           the model's no-draft split, -c 4096, every switch at its default, and compares the text byte for byte
# Env: RIG (ssh alias), NEW, REF (build folders under /opt/llama-builds), FN (Flash-Next's first shard: Swift Q2_0, its
#   n-gram table as shard 3), DRAFT (Flash-Next's MTP draft), FLINT_Q5, FLINT_Q8, ORNITH (the release files), RIGDIR (a
#   folder on the rig for logs, on a drive with room), OUT (a local folder), PORT (8091), MODELS_RUN ("flashnext flintq8
#   flintq5 ornith"), STEPS ("load greedy"), REL (v5: the presets' version), LABEL_NEW (cand), LABEL_REF (ref), PROFILE (1), GREEDY_BUILDS, GREEDY_EXPECT
set -uo pipefail
: "${RIG:?}" "${NEW:?}" "${REF:?}" "${FN:?}" "${DRAFT:?}" "${FLINT_Q5:?}" "${FLINT_Q8:?}" "${ORNITH:?}" "${RIGDIR:?}" "${OUT:?}"
PORT=${PORT:-8091}
MODELS_RUN=${MODELS_RUN:-flashnext flintq8 flintq5 ornith}
STEPS=${STEPS:-load greedy}
LABEL_NEW=${LABEL_NEW:-cand}; LABEL_REF=${LABEL_REF:-ref}
PROFILE=${PROFILE:-1}
GREEDY_BUILDS=${GREEDY_BUILDS:-$REF $NEW}
GREEDY_EXPECT=${GREEDY_EXPECT:-}
REL=${REL:-v5}
PROMPTS=${PROMPTS:-code prose long}
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "$(date +%H:%M:%S) $*"; }
mkdir -p "$OUT"

# the release lines
# shellcheck disable=SC1091
. release/flash-next-$REL/settings.env; FN_ENV=$PRESET_ENV; FN_ARGS="$PRESET_ARGS -md $DRAFT"
# shellcheck disable=SC1091
. release/flint-$REL/settings.env; FL_ENV=$PRESET_ENV; FL_ARGS=$PRESET_ARGS
# shellcheck disable=SC1091
. release/ornith-$REL/settings.env; OR_ENV=$PRESET_ENV; OR_ARGS=$PRESET_ARGS
# Flint Pascal Q5 GGUF: the same line with layers by count (release/flint-v4/README.md; the v5 line keeps the same -ts)
FL5_ARGS=${FL_ARGS/ -ts 6,6,5,5,6,5,5,6,6/}
[ "$FL5_ARGS" != "$FL_ARGS" ] || { say "Flint's line has no -ts 6,6,5,5,6,5,5,6,6 to drop"; exit 1; }

model() {  # <name>: sets M_FILE M_ENV M_ARGS M_SPLIT (the split: R153's, for the greedy step)
    case $1 in
        flashnext) M_FILE=$FN; M_ENV=$FN_ENV; M_ARGS=$FN_ARGS; M_SPLIT='-ngl 99 -fa on -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio' ;;
        flintq8) M_FILE=$FLINT_Q8; M_ENV=$FL_ENV; M_ARGS=$FL_ARGS; M_SPLIT='-ngl 99 -fa on -ts 6,6,5,5,6,5,5,6,6' ;;
        flintq5) M_FILE=$FLINT_Q5; M_ENV=$FL_ENV; M_ARGS=$FL5_ARGS; M_SPLIT='-ngl 99 -fa on' ;;
        ornith) M_FILE=$ORNITH; M_ENV=$OR_ENV; M_ARGS=$OR_ARGS; M_SPLIT='-ngl 99 -fa on -ts 6,6,6,6,5,5,5,5,6 -fit off' ;;
        *) say "unknown model $1"; return 1 ;;
    esac
}

P_CODE='Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest tests for it.'
P_PROSE='Explain in one paragraph how a hash map handles collisions, then give one example in Python.'
P_LONG='You are reviewing a small inventory service written in Python. It keeps a dictionary that maps each product code to a record with four fields: the product name, the number of units in stock, the unit price in cents, and the time of the last change. The service offers five operations. The first adds a product and fails if its code is already present. The second removes a product and fails if its code is missing. The third changes the number of units by a signed amount and fails if the result would drop below zero. The fourth changes the unit price and fails if the new price is not a positive whole number. The fifth returns every product whose stock is below a given threshold, sorted by code. Every operation writes one line to an audit log with the time, the operation, the product code and the outcome. Two background workers call these operations at the same time: one applies deliveries from a queue, the other applies sales from the shop. Last week the stock of one product went negative after a busy afternoon, and the audit log shows two sales of the last unit at almost the same moment, both reported as successful. A colleague suggests wrapping each operation in a single global lock; another suggests one lock per product code; a third suggests replacing the dictionary with a small SQLite database and letting it handle the transactions. Explain what most likely went wrong, compare the three suggestions for correctness, simplicity and throughput, pick one, and write the Python code for the stock change operation under your choice, with a short test that runs two threads selling the last unit and checks that exactly one of them succeeds.'

load() {  # <model> <label> <build folder> <server arguments>
    local arm=$1-$2 d=$RIGDIR/load-$1-$2 o=$OUT/$1/load-$2 up=0 p body env=$M_ENV
    mkdir -p "$o"
    [ "$PROFILE" = 1 ] && env="$env GGML_CUDA_PROFILE=$d/prof.csv GGML_CUDA_PROFILE_MODE=graph"
    say "load $arm: $3"
    ssh "$RIG" "LOGDIR=$d MIN_AVAIL_MB=60 ACTION=start ARM=$arm BIN=/opt/llama-builds/$3/bin/llama-server MODEL=$M_FILE \
        PORT=$PORT SERVER_ENV='${env# }' SERVER_ARGS='$4 --metrics --no-webui' bash -s" \
        < bench/agent_check/rig_server.sh > "$o/console.txt" 2>&1 && up=1
    if [ $up = 1 ]; then
        sleep 20
        ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits" > "$o/mem.csv"
        ssh "$RIG" "grep -E '^(MemTotal|MemAvailable|CmaFree|SwapTotal|SwapFree):' /proc/meminfo" > "$o/meminfo.txt"
        for p in $PROMPTS; do
            case $p in code) body=$P_CODE ;; prose) body=$P_PROSE ;; long) body=$P_LONG ;; *) continue ;; esac
            printf '{"prompt": "%s", "n_predict": 256, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": false}' "$body" |
                ssh "$RIG" "curl -s -m 900 -H 'Content-Type: application/json' -d @- http://127.0.0.1:$PORT/completion" \
                > "$o/reply-$p.json" 2> "$o/reply-$p.err"
            say "  $arm $p: $(wc -c < "$o/reply-$p.json") B of JSON"
        done
        ssh "$RIG" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits" > "$o/mem-after.csv"
    fi
    ssh "$RIG" "LOGDIR=$d ACTION=end PORT=$PORT bash -s" < bench/agent_check/rig_server.sh >> "$o/console.txt" 2>&1
    ssh "$RIG" "tar -C $d -cf - ." | tar -C "$o" -xf - && ssh "$RIG" "rm -rf $d"
    grep -aE 'buffer size|KV self size|compute buffer|graph|nodes|projected|failed|error|abort|assert|memory breakdown' \
        "$o/server-$arm.log" > "$o/buffers.txt"
    say "load $arm: $(grep -a 'up after\|exited\|timeout' "$o/console.txt" | head -n 1 | head -c 160)"
    [ $up = 1 ]
}

greedy() {  # <model> <builds> <out folder>
    RIG=$RIG MODEL=$M_FILE TAG=$1-code PROMPT=$P_CODE ARGS=$M_SPLIT DIR=$RIGDIR/greedy OUT=$3 \
        BUILDS="$2" bash docs/data/2026-11-05_ladder5/run/rig_greedy.sh > "$3.console" 2>&1
    say "$(tail -n 1 "$3.console")"
}

for m in $MODELS_RUN; do
    model "$m" || continue
    if has load; then
        load "$m" "$LABEL_NEW" "$NEW" "$M_ARGS"
        load "$m" "$LABEL_REF" "$REF" "$M_ARGS"
    fi
    if has greedy; then
        say "greedy $m: $GREEDY_BUILDS"
        greedy "$m" "$GREEDY_BUILDS" "$OUT/$m/greedy"
        exp=$(tr ' ' '\n' <<< "$GREEDY_EXPECT" | sed -n "s/^$m=//p")
        got=$(awk -F'\t' -v b="$NEW" '$1 == b {print $2}' "$OUT/$m/greedy/greedy-$m-code.tsv")
        if [ -n "$exp" ]; then
            say "greedy $m: NEW $got, expected $exp: $([ "$got" = "$exp" ] && echo SAME || echo DIFFERS)"
            if [ "$got" != "$exp" ] && [[ " $GREEDY_BUILDS " != *" $REF "* ]]; then
                say "greedy $m: REF too"
                greedy "$m" "$REF" "$OUT/$m/greedy-ref"
            fi
        fi
    fi
done
ssh "$RIG" "rmdir $RIGDIR/greedy 2> /dev/null; ps -eo comm | grep -c '^llama-' ; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '" \
    > "$OUT/clean.txt"
say "done: llama processes and card MiB after: $(tr '\n' ' ' < "$OUT/clean.txt")"
