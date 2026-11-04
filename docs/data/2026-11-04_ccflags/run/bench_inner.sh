#!/usr/bin/env bash
# Inside the CUDA container (build_bench_arms.sh starts it): builds the jobs of /o/jobs.txt, $JOBS at a time.
set -uo pipefail
one() { # <arm> <binary>:<source> <flags>
    local arm=$1 bin=${2%%:*} src=${2##*:} flags=$3
    mkdir -p "/o/$arm"
    # shellcheck disable=SC2086
    nice -n 19 nvcc -O3 -std=c++17 -use_fast_math $flags -gencode arch=compute_61,code=sm_61 -Wno-deprecated-gpu-targets \
        -cudart static -Xcompiler "-O2 -march=x86-64 -mno-avx -mno-avx2 -mno-fma -mno-f16c" -Xptxas -v \
        -o "/o/$arm/$bin" "/b/src/$src.cu" 2> "/o/$arm/ptxas-$bin.log" \
        || { echo "FAILED $arm $bin"; tail -n 5 "/o/$arm/ptxas-$bin.log"; return 1; }
    cuobjdump -res-usage "/o/$arm/$bin" > "/o/$arm/res-usage-$bin.txt"
    echo "built $arm $bin"
}
export -f one
nvcc --version | tail -n 2 > /o/BUILDINFO.txt
xargs -P "${JOBS:-3}" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r a t f <<< "$1"; one "$a" "$t" "$f"' _ {} < /o/jobs.txt
