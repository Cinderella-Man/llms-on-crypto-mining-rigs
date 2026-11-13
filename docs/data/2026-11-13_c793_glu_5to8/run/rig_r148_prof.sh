#!/usr/bin/env bash
# rig_r148_prof.sh: R148's perf step done again under the fork's profiler (GGML_CUDA_PROFILE). rig_r148.sh's perf step
# cannot time the launch: test-backend-ops perf repeats only the graph's last node, so a MUL_MAT_GLU case times one
# gate + up + SWIGLU (or one fused launch) and ~8,190 copies of the SWIGLU alone, and its us/run is the SWIGLU's in both
# arms. Here each case's perf loop runs under GGML_CUDA_PROFILE_MODE=group, which times every launch group of every
# evaluation of the captured CUDA graph, and each replayed evaluation (eval >= 3: eval 0 is the single-op warm-up, 1 the
# eager run, 2 the capture) gives the sum of its first K launch groups (default 10). Both arms then hold K groups, K
# profiler events and the same positions in the graph:
#   off (GLU_MT=1): gate, up, SWIGLU, K-3 SWIGLUs        on (GLU_MT=3): the fused launch, K-1 SWIGLUs
# so off - on = gate + up - fused - SWIGLU, and the time a served graph saves (gate + up + SWIGLU against the fused
# launch) is off - on + 2 x one SWIGLU's time without the profiler's events, which the sw step measures in
# GGML_CUDA_PROFILE_MODE=graph (an evaluation's GPU time over its launch groups; no event between the groups). The
# Q2_0/Q3_K pairs run the same fused launch in both arms: their off - on is the method's spread.
# Every arm has the preset's k-column switches and _DMT=1 _DMT_R=1 _Q5K_MT=1 _RO8=1, as rig_r148.sh's. Runs ON the rig,
# detached, the rig idle; ends with $OUT/DONE. The profile files go to TMP (default /dev/shm) and are deleted after each
# case.
# Steps (STEPS, default "sw prof"):
#   sw    graph mode, arm on, the pair q3_K+q3_K at each token count, REPS runs: $OUT/sw.tsv (n, rep, eval, eval_us,
#         eval_groups) of every replayed evaluation
#   prof  group mode, the arms off and on alternated over PASSES passes, each case (PAIRS x PERF_N) REPS times in a row:
#         $OUT/evals.tsv (arm, pass, pair, n, rep, eval, sum of the first K groups' us, groups 0-2's us, the mean us of
#         the groups from K on, the evaluation's groups, group 0's label) of every replayed evaluation
# Env: BIN (bin/ of the -c793 tarball or a later pascal-p106-v4 one), OUT, DEV (card, default 0;
#   CUDA_DEVICE_ORDER=PCI_BUS_ID), PASSES (2), REPS (3), K (10), PERF_N ("5 6 7 8"), PAIRS (gate+up, default the 14
#   pairs of Qwen3.8-Flash-Next's (Swift GSQ-RCO Q2_0) unfused shared-expert layers the launch takes), TMP
# Usage: BIN=... OUT=... setsid -f nohup bash rig_r148_prof.sh > console.txt 2>&1 < /dev/null
set -uo pipefail
: "${BIN:?}" "${OUT:?}"
STEPS=${STEPS:-sw prof}
DEV=${DEV:-0}
PASSES=${PASSES:-2}
REPS=${REPS:-3}
K=${K:-10}
PERF_N=${PERF_N:-5 6 7 8}
PAIRS=${PAIRS:-q2_0+q3_K q3_K+q3_K q3_K+q2_0 q2_0+q4_K q2_0+iq4_xs iq4_xs+q3_K q4_K+q4_K q4_K+q3_K q4_K+q2_0 q3_K+iq4_xs iq4_xs+q2_0 q5_K+q3_K q3_K+q5_K iq4_xs+q4_K}
TMP=${TMP:-/dev/shm}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
step() { [[ " $STEPS " == *" $1 "* ]]; }
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
P=GGML_CUDA_MMVQ_PASCAL
KP="${P}_KCOL_SMEM=32768 ${P}_KCOL_TABLE=0"
RO="${P}_DMT=1 ${P}_DMT_R=1 ${P}_Q5K_MT=1 ${P}_RO8=1"
M="${P}_GLU_MT"
arm_env() { case $1 in off) echo "$KP $RO $M=1" ;; on) echo "$KP $RO $M=3" ;; *) return 1 ;; esac; }
F="$TMP/r148_prof_$$.csv"
run_case() { # <arm> <pair> <n> <mode>: one test-backend-ops perf call for one case, its profile in $F
    local e; e=$(arm_env "$1") || return 1
    rm -f "$F"
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$DEV" GGML_TEST_RAW_QUANT=1 $e GGML_CUDA_PROFILE="$F" GGML_CUDA_PROFILE_MODE="$4" \
        "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT_GLU \
        -p "type_gate=${2%+*},type_up=${2#*+},m=640,n=$3,k=2560," > "$TMP/r148_prof_$$.out" 2>&1
    local rc=$?
    grep -q 'us/run' "$TMP/r148_prof_$$.out" || { log "no perf line: $1 $2 n=$3 $4 (exit $rc)"; return 1; }
}
quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-start.txt"

if step sw; then
    log "sw start"
    printf 'n\trep\teval\teval_us\teval_groups\n' > "$OUT/sw.tsv"
    for n in $PERF_N; do
        for rep in $(seq 1 "$REPS"); do
            run_case on q3_K+q3_K "$n" graph || continue
            # columns device,eval,host_ms,group,label,us,mode,graph,phase,how,groups,eval_groups,eval_us
            awk -F, -v n="$n" -v rep="$rep" 'NR > 1 && $10 == "replay" { printf "%s\t%s\t%s\t%s\t%s\n", n, rep, $2, $13, $12 }' \
                "$F" >> "$OUT/sw.tsv"
        done
    done
    log "sw end: $(($(wc -l < "$OUT/sw.tsv") - 1)) evaluations"
fi

if step prof; then
    printf 'arm\tpass\tpair\tn\trep\teval\tsum_k_us\tg0_us\tg1_us\tg2_us\trest_mean_us\tgroups\tg0_label\n' > "$OUT/evals.tsv"
    for pass in $(seq 1 "$PASSES"); do
        for arm in off on; do
            quiet
            log "prof $arm pass $pass start ($(arm_env "$arm"); K=$K, REPS=$REPS)"
            for pair in $PAIRS; do
                for n in $PERF_N; do
                    for rep in $(seq 1 "$REPS"); do
                        run_case "$arm" "$pair" "$n" group || continue
                        # columns device,eval,host_ms,group,label,us; evaluations from 3 on are the CUDA graph's replays
                        awk -F, -v K="$K" -v pre="$arm"$'\t'"$pass"$'\t'"$pair"$'\t'"$n"$'\t'"$rep" '
                            NR > 1 && $2 >= 3 {
                                e = $2; g = $4; us = $NF; ng[e]++
                                if (g < K) s[e] += us; else { r[e] += us; rc[e]++ }
                                if (g < 3) v[e, g] = us
                                if (g == 0) { l = $5; gsub(/[ \t]+/, "_", l); lab[e] = l }
                            }
                            END { for (e in ng) printf "%s\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.3f\t%d\t%s\n", pre, e, s[e], v[e, 0], v[e, 1], v[e, 2], (rc[e] ? r[e] / rc[e] : 0), ng[e], lab[e] }' \
                            "$F" | sort -t$'\t' -k6,6n >> "$OUT/evals.tsv"
                    done
                done
            done
            log "prof $arm pass $pass end: $(awk -F'\t' -v a="$arm" -v p="$pass" '$1 == a && $2 == p' "$OUT/evals.tsv" | wc -l) evaluations"
        done
    done
fi

rm -f "$F" "$TMP/r148_prof_$$.out"
quiet
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.gr,clocks.mem --format=csv,noheader > "$OUT/cards-end.txt"
touch "$OUT/DONE"
