#!/usr/bin/env bash
# run_block8.sh: the one-card microbenchmarks of the rig queue in one go (~45-60 min), on one P106-090 with nothing
# else on it. Run as root from the unpacked bundle (bench/rig_microbench/README.md); everything goes to $OUT.
#
#   1. Q101  the Pascal Q6_K kernel's gate in bench/mmvq_pascal: >= 130 GB/s on the dense K = 2048 shapes      ~2 min
#   2. Q101  the same products in the fork's q6k build (test-backend-ops perf, 1-4 tokens; llama.cpp's
#            kernels against the Pascal ones; Q6_K, Q5_K, Q8_0)                                                ~10-15 min
#   3. Q99   TLB probe (pointer chases over 2 MiB pages, alone and under load)                                 ~5-10 min
#   4. Q99   weights spread over 0.5-5 GB with fresh expert picks per call (MoE, attention, head)               ~8 min
#   5. Q103  per-instruction throughput (FFMA, dp4a, integer, shifts, conversions, shared memory, shuffles)    ~1 min
#   6. Q88   persistent MoE-layer kernel against the fork's 13 launches: check, then S4-L0 and G3 timings,
#            and one block per SM                                                                              ~6 min
# A failed step is logged and the next one runs; $OUT/summary.txt collects the deciding numbers.
#
# Env [defaults]: GPU [0], OUT [out/ next to this script], Q6K_BIN [the q6k build's bin directory on the rig],
#   STEPS [1 2 3 4 5 6], FORCE [0: refuse to start if a compute process is on the card]
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
GPU=${GPU:-0}
OUT=${OUT:-$here/out}
Q6K_BIN=${Q6K_BIN:-/opt/llama-builds/llama.cpp-20261024-03b92cf10-pascal-p106-v2-q6k/bin}
STEPS=${STEPS:-1 2 3 4 5 6}
FORCE=${FORCE:-0}
BIN=$here/bin
mkdir -p "$OUT"
LOG=$OUT/block8.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
on_gpu() { CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU "$@"; }

SMI=""
CHILD=""
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    [ -n "$CHILD" ] && pkill -P "$CHILD" 2>/dev/null; [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
    [ -n "$SMI" ] && kill "$SMI" 2>/dev/null
    exit "$rc"
}
trap cleanup EXIT
trap 'log "interrupted"; exit 130' INT TERM HUP

step() {  # step NAME CMD...: runs CMD with its output in $OUT/NAME.log, timed; never stops the block
    local name=$1 t0 rc
    shift
    t0=$(date +%s)
    log "== $name: $*"
    "$@" > "$OUT/$name.log" 2>&1 &
    CHILD=$!
    wait "$CHILD"; rc=$?
    CHILD=""
    log "   $name: exit $rc after $(( $(date +%s) - t0 )) s"
    return $rc
}

preflight() {
    local apps
    apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
    if [ -n "$apps" ] && [ "$FORCE" != 1 ]; then echo "compute processes on gpu $GPU (FORCE=1 to go on):"; echo "$apps"; exit 2; fi
    (cd "$here" && md5sum -c MD5SUMS) > "$OUT/md5.log" 2>&1 || { echo "MD5SUMS do not match: see $OUT/md5.log"; exit 2; }
    log "block 8 on gpu $GPU: steps $STEPS, OUT $OUT"
    nvidia-smi -i "$GPU" --query-gpu=index,name,pci.bus_id,driver_version,clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu \
        --format=csv | tee -a "$LOG"
    nvidia-smi -i "$GPU" --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active \
        --format=csv,noheader,nounits -lms 500 >> "$OUT/smi.csv" 2>&1 &
    SMI=$!
}

s1() { step q101_gate on_gpu "$BIN/mmvq_pascal" --types q6_K,q5_K,q8_0 --csv "$OUT/q101_gate.csv"; }
s2() {
    if [ ! -x "$Q6K_BIN/test-backend-ops" ]; then log "== q101_fork: $Q6K_BIN/test-backend-ops not found, skipped"; return 1; fi
    step q101_fork env CUDA_DEVICE_ORDER=PCI_BUS_ID BIN="$Q6K_BIN" GPU="$GPU" OUT="$OUT/q101_fork" TYPES='(q6_K|q5_K|q8_0)' RUNS="off on" \
        bash "$here/multitoken_perf.sh"
    python3 "$here/multitoken_perf_table.py" "$OUT/q101_fork" > "$OUT/q101_fork_tables.md" 2>&1
}
s3() { step q99_tlb on_gpu "$BIN/mmvq_pascal" --tlb-probe --csv "$OUT/q99_tlb.csv"; }
s4() {
    local gb
    for gb in 0.5 1 2 3 4 5; do
        step "q99_spread$gb" on_gpu "$BIN/mmvq_pascal" --types q5_0,q8_0,q4_1 --variants st_r1c2,st_r2c2,st_r2c4 \
            --shapes moe_gate_up,moe_down,attn_q,attn_output,output --spread-gb "$gb" --csv "$OUT/q99_spread$gb.csv"
    done
}
s5() { step q103_insn on_gpu "$BIN/insn_probe" --csv "$OUT/q103_insn.csv"; }
s6() {
    step q88_check on_gpu "$BIN/persistent_pascal" --check-only --sets 2 --trials 2
    step q88_s4l0 on_gpu "$BIN/persistent_pascal" --no-check --format s4l0 --csv "$OUT/q88_s4l0.csv"
    step q88_g3 on_gpu "$BIN/persistent_pascal" --no-check --format g3 --csv "$OUT/q88_g3.csv"
    step q88_grid5 on_gpu "$BIN/persistent_pascal" --no-check --no-trace --grid 5
}

summary() {
    {
        echo "block 8, gpu $GPU, $(date '+%F %T')"
        echo
        echo "Q101 gate (bench/mmvq_pascal; >= 130 GB/s on the dense K = 2048 shapes, Pascal kernel st_r1c1):"
        awk -F, '$9 == "q6_K" && $4 == 2048 && $6 == 0 && $10 == "st_r1c1" {
                     printf "  %-14s %7.1f GB/s  %s\n", $3, $15, $23; if ($15 < 130 || $23 != "PASS") bad = 1; n++ }
                 END { if (n) print "  gate: " (bad ? "NOT met" : "met"); else print "  no Q6_K rows (see q101_gate.log)" }' \
            "$OUT/q101_gate.csv" 2>/dev/null
        echo "  streaming read: $(awk -F, '$3 == "STREAM_READ_PLAIN" { print $15 " GB/s" }' "$OUT/q101_gate.csv" 2>/dev/null)"
        echo
        echo "Q101 fork (q6k build, us per call, one token; off = llama.cpp's kernels, on = Pascal):"
        grep -E '^\| (attn_q|attn_k/v|attn_output|shexp|experts)' "$OUT/q101_fork_tables.md" 2>/dev/null | awk -F'|' '$3 + 0 == 1' | head -24
        echo
        echo "Q99 spread: MoE and head calls, us per call by swept GB (csv per size in q99_spread*.csv):"
        for gb in 0.5 1 2 3 4 5; do
            awk -F, -v gb="$gb" '$3 ~ /^(moe_gate_up|moe_down|output)$/ && $12 != "" { printf "  %4s GB %-12s %-6s %-8s %8.2f us\n", gb, $3, $9, $10, $12 }' \
                "$OUT/q99_spread$gb.csv" 2>/dev/null
        done
        echo
        echo "Q99 TLB probe: $OUT/q99_tlb.csv"
        echo
        echo "Q103 instruction throughput:"
        sed -n '/^class/,/^$/p' "$OUT/q103_insn.log" 2>/dev/null | sed 's/^/  /'
        echo
        echo "Q88 persistent kernel:"
        grep -hiE 'speed-up|saving|per layer|PASS|FAIL' "$OUT"/q88_*.log 2>/dev/null | head -30 | sed 's/^/  /'
        echo
        echo "step times:"; grep -E '   [a-z0-9_.]+: exit' "$LOG" | sed 's/^/  /'
    } > "$OUT/summary.txt"
    cat "$OUT/summary.txt"
}

preflight
for s in $STEPS; do "s$s"; done
summary
log "block 8 done"
