#!/usr/bin/env bash
# Q671: checks the fork branch's GGML_CUDA_PASCAL_FLAGS option (ggml/src/ggml-cuda/CMakeLists.txt): configures the rig
# build (scripts/build_llamacpp_docker.sh's cmake settings) with two options for the Pascal kernel files, lists which
# CUDA files' compile commands carry them (exactly the Pascal kernel files must), compiles one of those files with them,
# and configures again without the option to show that no command carries it. CPU only, no GPU.
# Usage: SRC=<llama.cpp tree of the branch> [LOCK=<file>] [DOCKER_ARGS=...] cmake_hook_check.sh
set -uo pipefail
SRC=$(realpath "${SRC:?SRC=<llama.cpp tree>}")
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
locked docker run --rm $D -v "$SRC:/work/src:ro" "$IMAGE" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq cmake build-essential python3 > /dev/null
    conf() { # <build dir> [extra cmake args]
        local b=$1; shift
        cmake -S /work/src -B "$b" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
            -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets \
            -DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF \
            -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF \
            -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "$@" > "$b.log" 2>&1
        grep -E "Extra nvcc options|CMake Error" "$b.log" || true
    }
    list() { # <build dir> <marker>
        python3 - "$1/compile_commands.json" "$2" <<PY
import json, os, sys
hit, miss = [], 0
for e in json.load(open(sys.argv[1])):
    if "/ggml-cuda/" in e["file"] and e["file"].endswith(".cu"):
        if sys.argv[2] in e["command"]:
            hit.append(os.path.basename(e["file"]))
        else:
            miss += 1
print(len(hit), "CUDA files carry the options:", " ".join(sorted(hit)))
print(miss, "other CUDA files do not")
PY
    }
    conf /tmp/b1 "-DGGML_CUDA_PASCAL_FLAGS=--ptxas-options=-dlcm=ca;-DMMVQP_IQ_MIN_BLOCKS=2"
    list /tmp/b1 "ptxas-options=-dlcm=ca"
    cmd=$(python3 -c "
import json
for e in json.load(open(\"/tmp/b1/compile_commands.json\")):
    if e[\"file\"].endswith(\"/mmvq-pascal-mtok.cu\"):
        print(e[\"command\"]); break")
    echo "--- mmvq-pascal-mtok.cu: ${cmd##*-Wno-deprecated-gpu-targets}" | cut -c1-300
    (cd /tmp/b1/ggml/src/ggml-cuda && eval "$cmd" 2>&1 | tail -n 3; echo "compile exit ${PIPESTATUS[0]}"
     cuobjdump -res-usage CMakeFiles/ggml-cuda.dir/mmvq-pascal-mtok.cu.o | grep -o "REG:[0-9]*" | sort -t: -k2 -n | uniq -c | tail -n 3)
    conf /tmp/b2
    list /tmp/b2 "ptxas-options=-dlcm=ca"
'
