#!/usr/bin/env bash
# Q860's speed on the rig: whole-model k-token passes (llama-bench -p 1,2,3,4,5 -n 0, five repetitions after a
# warm-up) of the -routeonce build in three arms, beside rig 1's k-curve (../../2026-11-04_kcurve_p106):
#   off  the switch unset (the -ladder3 base's T(k));
#   k1   LLAMA_MOE_ROUTE_ONCE=1: the masked routing with every token its own block's first, the same experts: the cost
#        of the switch's extra graph operations alone;
#   on   LLAMA_MOE_ROUTE_ONCE=8,max=8: a pass of up to 8 tokens is one block, so every layer but the last (cropped to
#        the one output llama-bench asks for) reads the first token's 10 routed experts instead of the k tokens' union.
# Each arm is verify_cost.sh's wall step (../../2026-11-03_verify_cost/run/) under its host-memory guard after a logged
# quiet check. First, with TEST (extra-<commit>/test-route-once of run/build_gpu.sh's rig step) and RANDOM_MODELS
# (run/cpu_checks.sh's random/ folder), the CUDA check on one P106: the CPU's checks, the exact arms within rounding.
# Run ON the rig, idle, detached:
#   ssh $RIG "BIN=<-routeonce build>/bin MODEL=<first shard> OUT=<dir> VC=<folder with verify_cost.sh, mem_guard.sh and
#     smoke_request.json> [TEST=... RANDOM_MODELS=...] setsid nohup bash rig_timing.sh > <dir>/rig_timing.out 2>&1 < /dev/null &"
# Env, optional: KS (1,2,3,4,5), REPS (5), ARMS ("off k1 on"), TS (verify_cost.sh's split). Ends with DONE in OUT.
# The table, on any machine with Python 3:
#   python3 -I ../../2026-11-04_kcurve_p106/run/tk.py off=$OUT/off/bench-wall.json k1=$OUT/k1/bench-wall.json \
#     on=$OUT/on/bench-wall.json > tk_route_once.tsv
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${OUT:?}" "${VC:?}"
KS=${KS:-1,2,3,4,5}
REPS=${REPS:-5}
ARMS=${ARMS:-off k1 on}
mkdir -p "$OUT"
rm -f "$OUT/DONE"

if [ -n "${TEST:-}" ] && [ -n "${RANDOM_MODELS:-}" ]; then
    for m in qwen4exp-moe qwen4exp-moe-q5k; do
        CUDA_VISIBLE_DEVICES=0 "$TEST" -m "$RANDOM_MODELS/$m.gguf" -d CUDA0 -n 16 --router-scale 30 \
            > "$OUT/cuda_unit-$m.log" 2>&1
        echo "cuda unit $m: exit $?; $(tail -n 1 "$OUT/cuda_unit-$m.log")"
    done
fi

for arm in $ARMS; do
    case $arm in
        off) ro="" ;;
        k1)  ro="1" ;;
        on)  ro="8,max=8" ;;
        *)   echo "unknown arm $arm"; continue ;;
    esac
    echo "== $arm $(date -Is): LLAMA_MOE_ROUTE_ONCE=${ro:-unset}"
    if [ -n "$ro" ]; then
        LLAMA_MOE_ROUTE_ONCE=$ro BIN=$BIN MODEL=$MODEL DRAFT=$MODEL OUT=$OUT/$arm STEPS=wall KS=$KS REPS=$REPS \
            bash "$VC/verify_cost.sh"
    else
        env -u LLAMA_MOE_ROUTE_ONCE BIN="$BIN" MODEL="$MODEL" DRAFT="$MODEL" OUT="$OUT/$arm" STEPS=wall KS="$KS" \
            REPS="$REPS" bash "$VC/verify_cost.sh"
    fi
done
echo "== end $(date -Is)"
touch "$OUT/DONE"
