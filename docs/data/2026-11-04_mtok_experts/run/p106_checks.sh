#!/usr/bin/env bash
# p106_checks.sh: the multi-token Q2_0 expert kernel (R7, Q591) on one P106-090, run ON the rig with the rig otherwise
# idle. Correctness first, then the timings; every step's console goes to OUT.
#   correct  test-backend-ops on the card against the rig's CPU: every Q2_0 MUL_MAT_ID, fused gate/up and routed +
#            shared case (1 to 8 tokens), at the default and at each GGML_CUDA_MMVQ_PASCAL_MTOK setting, and with the
#            kernel off (the earlier kernels) as the control. The 512-expert cases take ~0.5 GB of host memory.
#   bench    bench/mmvq_pascal --q2_0-mtok: pertok against mtok and its variants at k = 1 to 8 over the route
#            profiles (CSV and console)
#   perf     test-backend-ops perf of the routed products at 1 to 8 tokens (random and overlapping routing): the
#            kernel on, off (GGML_CUDA_MMVQ_PASCAL_MTOK=0: mul_mat_vec_q_pascal_mc at 2-4, llama.cpp's
#            mul_mat_vec_q_moe at 5-8), groups of one entry, and llama.cpp's kernels throughout (SKIP=q2_0)
# Environment: BIN (the build's bin/ with test-backend-ops), BENCH (the mmvq_pascal binary), OUT; optional DEV (card,
# default 0), STEPS ("correct bench perf"). Ends with "== end" and the file DONE in OUT.
#   ssh $RIG "BIN=<build>/bin BENCH=<dir>/mmvq_pascal OUT=<work dir> setsid nohup bash p106_checks.sh > <work dir>/p106.out 2>&1 < /dev/null &"
set -uo pipefail
: "${BIN:?}" "${BENCH:?}" "${OUT:?}"
DEV=${DEV:-0}
STEPS=${STEPS:-correct bench perf}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DEV
TBO=$BIN/test-backend-ops
OPS=MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED
PAT='type_a=q2_0|type=q2_0'

quiet() {
  echo "== quiet before $1 $(date -Is): load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); \
cards $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
}

for step in $STEPS; do
  quiet "$step"
  case $step in
    correct)
      # tag:environment
      for run in "default:" "cap1:GGML_CUDA_MMVQ_PASCAL_MTOK=1,1" "cap3:GGML_CUDA_MMVQ_PASCAL_MTOK=1,3" \
                 "rb1:GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,1" "rb8:GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,8" \
                 "gdir:GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,0,1" "off:GGML_CUDA_MMVQ_PASCAL_MTOK=0" ; do
        tag=${run%%:*}
        env ${run#*:} "$TBO" -b CUDA0 -o "$OPS" -p "$PAT" > "$OUT/tbo_$tag.log" 2>&1
        echo "correct $tag: $(grep -E 'tests passed' "$OUT/tbo_$tag.log" | tail -1) $(grep -c FAIL "$OUT/tbo_$tag.log") FAIL lines"
      done
      ;;
    bench)
      "$BENCH" --q2_0-mtok --csv "$OUT/q2_0_mtok.csv" > "$OUT/q2_0_mtok.txt" 2>&1
      echo "bench: rc $? $(grep -E '^correctness' "$OUT/q2_0_mtok.txt")"
      ;;
    perf)
      for run in "mtok:" "off:GGML_CUDA_MMVQ_PASCAL_MTOK=0" "cap1:GGML_CUDA_MMVQ_PASCAL_MTOK=1,1" \
                 "llama:GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0" ; do
        tag=${run%%:*}
        env ${run#*:} "$TBO" perf -b CUDA0 -o MUL_MAT_ID -p type_a=q2_0 > "$OUT/perf_$tag.txt" 2>&1
        echo "perf $tag: $(grep -c 'us/run' "$OUT/perf_$tag.txt") cases"
      done
      ;;
  esac
done
echo "== end $(date -Is)"
touch "$OUT/DONE"
