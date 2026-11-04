#!/usr/bin/env bash
# Q895: agentionai's trellis kernels (TQ2_T) on one P106-090, from their llama.cpp fork built for compute 6.1 with the
# P106 perf cases added to test-backend-ops (p106perf.diff) and the trellis multi-row layout made available before
# Turing (p106rows.diff: GGML_CUDA_TQ_MMVQ_ROWS=1; unset, the fork's own choice). Parts, in order (PARTS, default
# "quiet check perf"):
#   quiet  the card empty, no llama or test process, MemAvailable; logged, and the run stops if any of them fails
#   check  test-backend-ops (test mode) on the card against the CPU: the TQ2_T MUL_MAT cases and the TQ2_T MUL_MAT_ID
#          cases of up to 8 tokens in the fork's own test list; then the one-token cases with GGML_CUDA_TQ_MMVQ_ROWS=1
#   perf   test-backend-ops perf: Flash-Next's routed experts, 10 of 128, at 1-8 tokens (gate/up 2,560 -> 640 with the
#          activations broadcast, down 640 -> 2,560), TQ2_T and Q2_0 in the same binary; the Q8_0 dense shapes Gyro
#          files keep at 1, 2, 4 and 8 tokens; the files' runtime Hadamard rotation of the experts' inputs
#          (MUL_MAT_HADAMARD, 128-wide blocks: 20 a token for gate/up, 50 for down, at 1, 2, 4 and 8 tokens); then
#          TQ2_T at one token in two more layouts: rows1
#          (GGML_CUDA_TQ_MMVQ_ROWS=1, one row per warp) and wide0 (GGML_CUDA_TQ_MMVQ_WIDE=0, the down projection's
#          16-row blocks off)
# The trellis encoder takes minutes per expert tensor on a 2-core host, so perf fills the weights from INIT_BLOCKS
# encoded blocks repeated over each tensor (the fork's GGML_TEST_QUANT_INIT_BLOCKS). Every TQ2_T byte string is a
# valid trellis path, so the decode does the same work on any data; check mode encodes every block.
#   (env: BIN OUT [CARD] [PARTS] [INIT_BLOCKS]) bash p106_tq.sh
# Output in OUT: run.log (states and verdicts), check_*.log, perf_*.txt (test-backend-ops console output: us per run),
# perf_*.log (its stderr).
set -euo pipefail
: "${BIN:?}" "${OUT:?}"
CARD=${CARD:-0}
PARTS=${PARTS:-"quiet check perf"}
INIT_BLOCKS=${INIT_BLOCKS:-4096}
TBO=$BIN/test-backend-ops
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD
log() { echo "[$(date -Is)] $*" | tee -a "$OUT/run.log"; }
state() {
    nvidia-smi -i "$CARD" --query-gpu=name,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate \
        --format=csv,noheader | tee -a "$OUT/run.log"
    grep -E '^(MemAvailable|SwapFree)' /proc/meminfo | tr -s ' ' | paste -sd ' ' | tee -a "$OUT/run.log"
}
has() { [[ " $PARTS " == *" $1 "* ]]; }

log "p106_tq.sh PARTS='$PARTS' CARD=$CARD INIT_BLOCKS=$INIT_BLOCKS BIN=$BIN"
md5sum "$TBO" | tee -a "$OUT/run.log"

if has quiet; then
    state
    used=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    procs=$(ps -eo comm | grep -cE '^(llama-|test-backend)' || true)
    avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    log "quiet: card $CARD ${used} MiB used, $procs llama/test processes, MemAvailable ${avail} MiB"
    if [ "$used" -gt 64 ] || [ "$procs" -gt 0 ] || [ "$avail" -lt 2048 ]; then
        log "NOT QUIET: stopping"
        exit 1
    fi
fi

if has check; then
    log "check: MUL_MAT"
    "$TBO" test -b CUDA0 -o MUL_MAT -p '^type_a=tq2_t,' > "$OUT/check_mul_mat.log" 2>&1 || true
    log "check MUL_MAT: $(grep -E 'tests passed' "$OUT/check_mul_mat.log" | tail -n 1)"
    log "check: MUL_MAT_ID up to 8 tokens"
    "$TBO" test -b CUDA0 -o MUL_MAT_ID -p '^type_a=tq2_t,.*,n=[1-8],' > "$OUT/check_mul_mat_id.log" 2>&1 || true
    log "check MUL_MAT_ID: $(grep -E 'tests passed' "$OUT/check_mul_mat_id.log" | tail -n 1)"
    log "check: one token, GGML_CUDA_TQ_MMVQ_ROWS=1"
    GGML_CUDA_TQ_MMVQ_ROWS=1 "$TBO" test -b CUDA0 -o MUL_MAT,MUL_MAT_ID -p '^type_a=tq2_t,.*,n=1,' \
        > "$OUT/check_rows1.log" 2>&1 || true
    log "check rows1: $(grep -E 'tests passed' "$OUT/check_rows1.log" | tail -n 1)"
    grep -E 'FAIL' "$OUT"/check_*.log | head -n 20 | tee -a "$OUT/run.log" || true
fi

perf() {  # name, op, params regex, extra env
    local name=$1 op=$2 re=$3
    shift 3
    log "perf $name"
    env GGML_TEST_QUANT_INIT_BLOCKS="$INIT_BLOCKS" "$@" "$TBO" perf -b CUDA0 -o "$op" -p "$re" \
        > "$OUT/perf_$name.txt" 2> "$OUT/perf_$name.log" || log "perf $name: exit $?"
    log "perf $name: $(grep -c "us/run" "$OUT/perf_$name.txt" || true) cases"
}

if has perf; then
    state
    perf experts MUL_MAT_ID '^type_a=(tq2_t|q2_0),type_b=f32,n_mats=128,n_used=10,'
    perf dense MUL_MAT '^type_a=q8_0,type_b=f32,(m=(6144|10240|12288),n=[1248],k=2560|m=2560,n=[1248],k=6144|m=10240,n=[1248],k=320|m=320,n=[1248],k=10240),'
    perf hadamard MUL_MAT_HADAMARD '^type_a=f32,type_b=f32,m=128,n=(20|40|80|160|50|100|200|400),k=128,'
    one='^type_a=tq2_t,type_b=f32,n_mats=128,n_used=10,b=[01],m=(640|2560),n=1,'
    perf rows1 MUL_MAT_ID "$one" GGML_CUDA_TQ_MMVQ_ROWS=1
    perf wide0 MUL_MAT_ID "$one" GGML_CUDA_TQ_MMVQ_WIDE=0
    state
fi
log "done"
touch "$OUT/done"
