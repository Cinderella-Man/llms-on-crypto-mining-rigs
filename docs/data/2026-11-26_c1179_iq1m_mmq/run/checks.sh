#!/usr/bin/env bash
# C1179's checks on one CUDA GPU: test-backend-ops MUL_MAT, MUL_MAT_ID and MUL_MAT_VEC_FUSION against the CPU, and
# which kernels one IQ1_M expert product launches.
#
#   BIN_A=<build A>/bin BIN_B=<build B>/bin OUT=<log folder> bash checks.sh
#
# Build A: build.sh with ARCHS="61-virtual;70" (on a V100 its sm_70 machine code runs).
# Build B: build.sh with ARCHS="61-virtual" FORCE_MMQ=1: the P106's compute-6.1 PTX, which the driver JIT-compiles
#   because the build has no other code for the card, with MMQ at every product size as a P106 picks it. Before Turing,
#   MMQ's integer path is DP4A on both cards. (CUDA_FORCE_PTX_JIT=1 is not needed for that, and it also forces
#   cuBLAS's own kernels through the JIT: on a V100 with driver 580, cuBLAS then fails to start.)
# GGML_CUDA_MMQ_IQ1_M=2 puts IQ1_M on MMQ on any NVIDIA GPU; 1 is compute 6.1 only, so on a V100 it leaves it off.
# Set CUDA_CACHE_PATH and CUDA_CACHE_MAXSIZE to keep the driver's JIT cache next to the work, and TMPDIR for nsys.
# STEPS (default all): a_all b_all a_on b_on trace (a_trace and b_trace: one build's); a_rest after a stopped a_all. NSYS: the nsys binary (default nsys).
set -u
: "${BIN_A:?}" "${BIN_B:?}" "${OUT:?}"
mkdir -p "$OUT"
OPS=MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION
IQ1M='type_a=iq1_m|type=iq1_m'
STEPS=${STEPS:-a_all b_all a_on b_on trace}
NSYS=${NSYS:-nsys}

# name, then the command (environment assignments first); one summary line per run: exit code, time, the pass count
run() {
    local name=$1; shift
    local t0=$SECONDS
    env "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    local passed fails
    passed=$(grep -E '^ *[0-9]+/[0-9]+ tests passed' "$OUT/$name.log" | tail -1 | sed 's/^ *//')
    fails=$(grep -c 'FAIL' "$OUT/$name.log")
    printf '%s\texit %d\t%d s\t%s\tFAIL lines %d\n' "$name" "$rc" $((SECONDS - t0)) "$passed" "$fails" | tee -a "$OUT/summary.tsv"
}

# name, build's bin folder, then environment assignments: the kernels and CUDA calls of one IQ1_M expert product
# (Qwen3.8-Flash-Next's routed gate/up shape, 512 tokens), by name and count only
trace() {
    local name=$1 bin=$2; shift 2
    local case='type_a=iq1_m,type_b=f32,n_mats=32,n_used=10,b=0,m=640,n=512,k=2560,'
    rm -f "$OUT/$name.nsys-rep" "$OUT/$name.sqlite"
    env "$@" "$NSYS" profile -t cuda -o "$OUT/$name" "$bin/test-backend-ops" -b CUDA0 -o MUL_MAT_ID -p "$case" \
        > "$OUT/$name.log" 2>&1
    echo "exit $?" >> "$OUT/$name.log"
    "$NSYS" stats -q -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o "$OUT/$name" "$OUT/$name.nsys-rep" >> "$OUT/$name.log" 2>&1
    rm -f "$OUT/$name.nsys-rep" "$OUT/$name.sqlite"
}

for step in $STEPS; do
    case $step in
        # the switch off: build A every type's cases of the three ops; build B every type's MUL_MAT cases and IQ1_M's
        # MUL_MAT_ID and MUL_MAT_VEC_FUSION cases (the CPU reference, built without AVX as for the rig, takes over an
        # hour for every type's MUL_MAT_ID cases)
        a_all) run a_all -u GGML_CUDA_MMQ_IQ1_M "$BIN_A/test-backend-ops" -b CUDA0 -o "$OPS" ;;
        b_all)
            run b_all_mm -u GGML_CUDA_MMQ_IQ1_M "$BIN_B/test-backend-ops" -b CUDA0 -o MUL_MAT
            run b_all_iq1m -u GGML_CUDA_MMQ_IQ1_M "$BIN_B/test-backend-ops" -b CUDA0 \
                -o MUL_MAT_ID,MUL_MAT_VEC_FUSION -p "$IQ1M"
            ;;
        # after a stopped a_all: the MUL_MAT cases it did not reach (every type, by their full names), and IQ1_M's
        # MUL_MAT_ID and MUL_MAT_VEC_FUSION cases, the switch off
        a_rest)
            "$BIN_A/test-backend-ops" support -b CUDA0 -o MUL_MAT 2> /dev/null \
                | sed -n 's/^  \(MUL_MAT(.*)\): .*/\1/p' | sort -u > "$OUT/a_mm_cases.txt"
            sed -n 's/^  \(MUL_MAT(.*)\): .*/\1/p' "$OUT/a_all.log" | sort -u > "$OUT/a_all_mm_done.txt"
            comm -23 "$OUT/a_mm_cases.txt" "$OUT/a_all_mm_done.txt" > "$OUT/a_mm_rest.txt"
            run a_mm_rest -u GGML_CUDA_MMQ_IQ1_M "$BIN_A/test-backend-ops" -b CUDA0 -o "$(paste -sd, "$OUT/a_mm_rest.txt")"
            run a_off -u GGML_CUDA_MMQ_IQ1_M "$BIN_A/test-backend-ops" -b CUDA0 -o MUL_MAT_ID,MUL_MAT_VEC_FUSION -p "$IQ1M"
            ;;
        # IQ1_M, the switch on
        a_on)  run a_on GGML_CUDA_MMQ_IQ1_M=2 "$BIN_A/test-backend-ops" -b CUDA0 -o "$OPS" -p "$IQ1M" ;;
        b_on)  run b_on GGML_CUDA_MMQ_IQ1_M=2 "$BIN_B/test-backend-ops" -b CUDA0 -o "$OPS" -p "$IQ1M" ;;
        trace|a_trace|b_trace)
            if [ "$step" != b_trace ]; then
                trace a_trace_off "$BIN_A" -u GGML_CUDA_MMQ_IQ1_M
                trace a_trace_1   "$BIN_A" GGML_CUDA_MMQ_IQ1_M=1
                trace a_trace_on  "$BIN_A" GGML_CUDA_MMQ_IQ1_M=2
            fi
            if [ "$step" != a_trace ]; then
                trace b_trace_off "$BIN_B" -u GGML_CUDA_MMQ_IQ1_M
                trace b_trace_on  "$BIN_B" GGML_CUDA_MMQ_IQ1_M=2
            fi
            ;;
        *) echo "unknown step $step" >&2; exit 2 ;;
    esac
done
