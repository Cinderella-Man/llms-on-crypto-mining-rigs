#!/usr/bin/env bash
# T1 (Miner Llama v3's release tests): Flash-Next's release preset on the frozen release build -ladder5 1f7215948, one
# rig, S35's line (pmix-q2-fast, the k-column settings, the Q8_0 hop cast through the staged hop, the MTP draft with the
# head's shortlist; docs/data/2026-11-04_s35_stack) with C533's keep-alive. Run from the repository root on a machine
# that only runs ssh; it takes S35's machine file (docs/data/2026-11-03_flash_next_mtp/run/s12.env.example, filled in).
#   ident  greedy identity on the real file at every default: one llama-server per build (BUILD, then each of
#          IDENT_REF, e.g. -ladder4), no switch set, two prompts at temperature 0, 128 tokens each; the md5 of each reply
#          must agree between the builds (ident.tsv)
#   tk     llama-bench on S35's no-draft line with GGML_CUDA_KEEPALIVE=decode, -p 1,2,3,4,5 -n 128, five repetitions
#          after a warm-up, LOADS loads (default 1): plain decoding and T(k), against C533's rig run on -keepalive
#   prep   the agent check's bench files and sandbox image on the client
#   arms   the agent check, one arm a switch group (ARMS_T1, default "t1base t1skip t1neg"):
#            t1base  S35's line + GGML_CUDA_KEEPALIVE=decode (C533a's s35ka): the preset candidate
#            t1skip  t1base + the expert skip, LLAMA_MOE_TAIL=p=0.9,renorm=1,hist=0 (C120)
#            t1neg   t1base + every "Negligible gains" switch that applies to this line, together (C534's group): the
#                    expert skip, the draft length per step (LLAMA_SPEC_ADAPT=1 with C271's real-text T(k) for the
#                    -kshape kernels at these k-column settings, COST), one server thread (-t 1 -tb 1) and the
#                    keep-alive at each request (GGML_CUDA_KEEPALIVE=request instead of decode); the staged hop is on
#                    in every arm
#   report the agent check's report of the arms (REPORT_ARMS, default ARMS_T1), the client's folder collected to OUT
#   stock  stock llama.cpp (BUILD) on MODEL with the arms' server line and no switch, one arm per name in STOCK_ARMS
#          (stockq2: Swift's own Q2_0 file), STOCK_ARGS for another line (default S35's with the MTP draft)
#   swift  t1swift: t1base's line and switches exactly on MODEL (Swift's own Q2_0 file)
#   tune   stock's tuning pass: one llama-bench load of BUILD on MODEL, flash attention on, -ub 128/256/512, tg128,
#          pp4 and pp2048, three repetitions
# The steps run in the order above, whatever STEPS lists.
# Env: ENVFILE, MODEL (the first shard on the rig), BUILD (the release build's folder under /opt/llama-builds),
#      IDENT_REF (the reference builds' folders), OUT (a local folder), STEPS, LOADS, ARMS_T1, REPORT_ARMS, COST (the
#      cost table on the rig, for t1neg), STOCK_ARMS, STOCK_ARGS, IDENT_PORT (8191), and RIG_PORT RELAY_PORT CLIENT_PORT PROXY_PORT CLIENT_DIR
#      CLIENT_OUT RIG_LOGDIR BENCH as s35_arms.sh: the caller's values win over the machine file's.
set -uo pipefail
: "${ENVFILE:?}" "${MODEL:?}" "${BUILD:?}" "${OUT:?}"
M=$MODEL; T=$BUILD
keep=$(for v in RIG RIG_PORT RELAY_PORT CLIENT_PORT PROXY_PORT CLIENT_DIR CLIENT_OUT RIG_LOGDIR BENCH; do
           [ -n "${!v:-}" ] && printf '%s=%q ' "$v" "${!v}"; done)
set -a; . "$ENVFILE"; eval "$keep"; set +a
export MIN_AVAIL_MB=60 MODEL=$M REL_BIN=/opt/llama-builds/$T/bin/llama-server RIG_LOGDIR=${RIG_LOGDIR:-/tmp/t1-agent-logs}
STEPS=${STEPS:-ident tk prep arms report}
LOADS=${LOADS:-1}
ARMS_T1=${ARMS_T1:-t1base t1skip t1neg}
IDENT_PORT=${IDENT_PORT:-8191}
has() { [[ " $STEPS " == *" $1 "* ]]; }
KENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0"
HOP="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1"
SL="LLAMA_DRAFT_SHORTLIST=32768,8192,2"
SKIP="LLAMA_MOE_TAIL=p=0.9,renorm=1,hist=0"
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
LBOPT="-ngl 99 -fa 1 -b 2048 -ub 512 -ts 6/5/5/6/5/5/6/5/6 -lm dio"
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
mkdir -p "$OUT"
# <label> <build folder>: one llama-server on the rig at every default, two greedy prompts; prints "<label> <prompt>
# <md5> <tokens> <tok/s> <first 60 characters>" per prompt
ident_load() {
    ssh "$RIG" "bash -s" > "$OUT/ident-$1.out" 2>&1 <<EOF
mkdir -p /tmp/t1i && cd /tmp/t1i
/opt/llama-builds/$2/bin/llama-server -m $M -ngl 99 -fa on -c 8192 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio \
    -np 1 --no-webui --host 127.0.0.1 --port $IDENT_PORT > server-$1.log 2>&1 < /dev/null &
srv=\$!
for i in \$(seq 900); do curl -sf http://127.0.0.1:$IDENT_PORT/health > /dev/null && break; kill -0 \$srv 2> /dev/null || break; sleep 2; done
for p in code prose; do
    case \$p in
        code)  q='Write a Python function that checks whether a string is a palindrome, with a short docstring.' ;;
        prose) q='Explain in one paragraph how a hash map handles collisions.' ;;
    esac
    python3 -I -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": 128, "temperature": 0, "seed": 0, "cache_prompt": False}))' "\$q" > req.json
    curl -sf http://127.0.0.1:$IDENT_PORT/completion -H 'Content-Type: application/json' -d @req.json > rep-$1-\$p.json
    python3 -I -c 'import hashlib,json,sys; r=json.load(open(sys.argv[1])); c=r["content"]; t=r["timings"]
print(sys.argv[2], sys.argv[3], hashlib.md5(c.encode()).hexdigest()[:12], t["predicted_n"], round(t["predicted_per_second"], 2), repr(c[:60]), sep="\t")' rep-$1-\$p.json $1 \$p
done
kill -INT \$srv; sleep 8; pkill -9 -x llama-server
grep -cE 'out of memory|error' server-$1.log | sed 's/^/errors in the log: /'
EOF
    grep -P '^\S+\t(code|prose)\t' "$OUT/ident-$1.out" >> "$OUT/ident.tsv"
    ssh "$RIG" "cat /tmp/t1i/server-$1.log" > "$OUT/ident-server-$1.log"
}
# <name> <keep-alive setting or ""> <llama-bench options>: one llama-bench load on the rig under the RAM watchdog
bench_load() {
    ssh "$RIG" "mkdir -p /tmp/t1 && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
    ssh "$RIG" "LOG=/dev/shm/rw_$1.tsv PROC=llama-bench MIN_AVAIL_MB=60 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw.pid
        cd /tmp/t1 && env $KENV $HOP ${2:+GGML_CUDA_KEEPALIVE=$2} timeout 2400 /opt/llama-builds/$T/bin/llama-bench -m $M $LBOPT $3 \
            -o jsonl > $1.jsonl 2> $1.err; echo \"$1 rc=\$?\"
        kill \$(cat /dev/shm/rw.pid); mv /dev/shm/rw_$1.tsv /tmp/t1/; rm -f /dev/shm/rw.pid /dev/shm/ram_watchdog.sh"
    for f in "$1.jsonl" "$1.err" "rw_$1.tsv"; do ssh "$RIG" "cat /tmp/t1/$f" > "$OUT/$f"; done
}
if has ident; then
    : > "$OUT/ident.tsv"
    ident_load l5 "$T"
    for r in ${IDENT_REF:-}; do ident_load "ref-${r##*-}" "$r"; done
    cat "$OUT/ident.tsv"
    ssh "$RIG" "rm -rf /tmp/t1i"
fi
if has tk; then
    for i in $(seq "$LOADS"); do bench_load "tk-on-$i" decode "-p 1,2,3,4,5 -n 128 -r 5"; done
    ssh "$RIG" "rm -rf /tmp/t1"
fi
if has prep; then
    ARMS=t1base STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1
fi
run_arm() {  # <arm> <REL_ENV> <args>
    env "ARGS_$1=$3" "BUDGET_MIN_$1=40" REL_ENV="$2" ARMS=$1 STEPS=run bash $RC "$OUT/agent-$1" > "$OUT/agent-$1.console" 2>&1
    echo "arm $1: exit $?"
}
if has arms; then
    for a in $ARMS_T1; do
        case $a in
            t1base) run_arm t1base "$KENV $HOP $SL GGML_CUDA_KEEPALIVE=decode"       "$MT" ;;
            t1skip) run_arm t1skip "$KENV $HOP $SL GGML_CUDA_KEEPALIVE=decode $SKIP" "$MT" ;;
            t1neg)  run_arm t1neg  "$KENV $HOP $SL GGML_CUDA_KEEPALIVE=request $SKIP LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_COST=${COST:?COST=<the cost table on the rig>} LLAMA_SPEC_ADAPT_LOG=adapt-t1neg.jsonl" \
                                   "$MT -t 1 -tb 1" ;;
        esac
    done
fi
if has report; then
    ARMS="${REPORT_ARMS:-$ARMS_T1}" STEPS="report collect" bash $RC "$OUT/agent-report" > "$OUT/agent-report.console" 2>&1
fi
# stock llama.cpp (BUILD: upstream with no fork commit): the arms' server line with no switch; STOCK_ARMS names the
# arms (e.g. stockpmix on pmix-q2-fast, stockq2 on Swift's own Q2_0 file, MODEL), STOCK_ARGS the line (default S35's
# with the MTP draft)
if has stock; then
    for a in ${STOCK_ARMS:?}; do run_arm "$a" "" "${STOCK_ARGS:-$MT}"; done
fi
# t1swift: the release preset (t1base's line and switches exactly) on the file MODEL names, Swift's own public Q2_0
# file: the release's public-file number, paired with stockq2 on the same file
if has swift; then
    run_arm t1swift "$KENV $HOP $SL GGML_CUDA_KEEPALIVE=decode" "$MT"
fi
# tune: stock's line, one llama-bench load (BUILD, MODEL) with flash attention on: decode (tg128), a 4-token pass
# (pp4) and a 2,048-token prompt (pp2048) at -ub 128, 256 and 512, three repetitions, no switch (tune.jsonl)
if has tune; then
    KENV=""; HOP=""; LBOPT="-ngl 99 -fa 1 -b 2048 -ts 6/5/5/6/5/5/6/5/6 -lm dio"
    bench_load tune "" "-ub 128,256,512 -p 4,2048 -n 128 -r 3"
    ssh "$RIG" "rm -rf /tmp/t1"
fi
