#!/usr/bin/env bash
# Register census of the Pascal decode attention kernel (fattn-pascal.cu) for compute 6.1: compiles that one file with
# the exact flags of the rig build (scripts/build_llamacpp_docker.sh's cmake configuration, CUDA 12.9 in Docker) plus
# -Xptxas -v, and prints per kernel registers, stack, spill stores and loads, local memory and shared memory
# (ptxas) and cuobjdump's resource usage of the object. CPU only, no GPU.
# WORK: a folder holding one llama.cpp source tree per arm, $WORK/<arm>/src (e.g. `git archive` of a fork commit);
# ARMS: the arm names (default "before after"); FILES: the CUDA files to census, relative to ggml/src/ggml-cuda
# (default fattn-pascal.cu). Output: $WORK/census/<arm>.ptxas.txt, <arm>.res.txt, and census.tsv over the arms.
# Usage: WORK=<dir> [ARMS="before after"] [FILES=fattn-pascal.cu] [DOCKER_ARGS=...] census.sh
set -euo pipefail
WORK=${WORK:?WORK=<folder with <arm>/src trees>}
ARMS=${ARMS:-before after}
FILES=${FILES:-fattn-pascal.cu}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$WORK/census"

for arm in $ARMS; do
    [ -d "$WORK/$arm/src/ggml" ] || { echo "no source tree $WORK/$arm/src" >&2; exit 1; }
    docker run --rm ${DOCKER_ARGS:-} -e ARM="$arm" -e FILES="$FILES" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        -v "$WORK:/work" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq cmake build-essential python3 > /dev/null
        cd /work/$ARM/src
        rm -rf /tmp/b
        # the rig build'"'"'s configuration (scripts/build_llamacpp_docker.sh), with the compile commands exported
        cmake -S . -B /tmp/b -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
            -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
            -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
            -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF \
            -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > /dev/null
        out=/work/census/$ARM
        : > $out.ptxas.txt; : > $out.res.txt
        for f in $FILES; do
            cmd=$(python3 -c "
import json, sys
for e in json.load(open(\"/tmp/b/compile_commands.json\")):
    if e[\"file\"].endswith(\"/ggml-cuda/\" + sys.argv[1]):
        print(e[\"command\"]); break
" "$f")
            [ -n "$cmd" ] || { echo "no compile command for $f" >&2; exit 1; }
            o=/tmp/b/census-${f%.cu}.o
            cmd=$(echo "$cmd" | sed -E "s# -o [^ ]+# -o $o#")
            echo "== $f" >> $out.ptxas.txt
            (cd /tmp/b/ggml/src/ggml-cuda 2>/dev/null || cd /tmp/b; eval "$cmd -Xptxas -v") 2>&1 | grep -E "ptxas|Compiling|Function properties|spill|Used" >> $out.ptxas.txt
            echo "== $f" >> $out.res.txt
            cuobjdump -res-usage $o >> $out.res.txt
            cuobjdump -sass $o > /work/census/$ARM.${f%.cu}.sass
        done
        chown -R "$HOST_UID:$HOST_GID" /work/census
    '
done

# one row per arm and kernel (sm_61): registers, stack, spill stores/loads, shared memory, from ptxas
{
    printf "arm\tkernel\tregisters\tstack_bytes\tspill_stores\tspill_loads\tshared_bytes\n"
    for arm in $ARMS; do
        awk -v arm="$arm" '
            /Compiling entry function/ { match($0, /'"'"'[^'"'"']+'"'"'/); k = substr($0, RSTART+1, RLENGTH-2); next }
            /bytes stack frame/ && k != "" { split($0, w, /[ ,]+/); for (i in w) { if (w[i] == "bytes" && w[i+1] == "stack") st = w[i-1]; if (w[i] == "spill" && w[i+1] == "stores") ss = w[i-2]; if (w[i] == "spill" && w[i+1] == "loads") sl = w[i-2] } next }
            /Used [0-9]+ registers/ && k != "" { match($0, /Used [0-9]+ registers/); r = substr($0, RSTART+5, RLENGTH-15); sm = 0; if (match($0, /[0-9]+ bytes smem/)) sm = substr($0, RSTART, RLENGTH-11); printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", arm, k, r, st, ss, sl, sm; k = "" }
        ' "$WORK/census/$arm.ptxas.txt"
    done
} > "$WORK/census/census.tsv"
cat "$WORK/census/census.tsv"

# instructions per kernel in the sm_61 machine code (static counts): local stores and loads (STL, LDL; a stack frame
# without spills is an array kept in local memory), the FP32 arithmetic, shuffles, selects and branches
{
    printf "arm\tkernel\tinstructions\tSTL\tLDL\tFFMA\tFMUL\tFADD\tSHFL\tSEL\tBRA\tLDG\tLDS\tSTS\n"
    for arm in $ARMS; do
        for s in "$WORK/census/$arm".*.sass; do
            awk -v arm="$arm" '
                function flush() { if (k != "") printf "%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", arm, k, n,
                    c["STL"], c["LDL"], c["FFMA"], c["FMUL"], c["FADD"], c["SHFL"], c["SEL"], c["BRA"], c["LDG"], c["LDS"], c["STS"] }
                /Function : / { flush(); k = $3; n = 0; split("", c); next }
                k != "" && $1 ~ /^\/\*[0-9a-f]+\*\/$/ { op = ($2 ~ /^@/) ? $3 : $2; sub(/\..*/, "", op); sub(/;$/, "", op); c[op]++; n++ }
                END { flush() }
            ' "$s"
        done
    done
} > "$WORK/census/sass_counts.tsv"
cut -f1-6 "$WORK/census/sass_counts.tsv"
