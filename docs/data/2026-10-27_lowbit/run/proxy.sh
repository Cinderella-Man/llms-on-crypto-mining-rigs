#!/usr/bin/env bash
# Q164's CPU quality proxy as it ran. Everything executes on the quality machine (SSH alias QM) in Docker at
# --cpu-shares 256 with no GPU; the machine that runs this script only copies files over ssh.
#   lib      CPU-only libggml-base.so from the llama.cpp tree SRC (llama.cpp's reference quantizers for lowbit_proxy.py;
#            from fork branch pascal-p106-v3-lowbit it also has N2P and N3P)
#   proxy    scripts/lowbit_proxy.py over the sampled experts, output in WORK/RUN/proxy.tsv; ARGS adds options
#            (--layers, --formats); the non-uniform tables are fitted once into WORK/RUN/proxy.tables.json, so copy
#            the first run's tables.json there to reuse them
#   summary  scripts/lowbit_summary.py over WORK/RUN/proxy.tsv; extra arguments go to it
# Env: QM, WORK (work dir on QM), SRC (llama.cpp tree on QM), MODEL (Ornith-1.5-122B BF16 safetensors dir on QM),
#      IMATRIX (Unsloth's imatrix_unsloth.gguf_file on QM), RUN (run name), ARGS, REPO (this repository, local).
set -euo pipefail
QM=${QM:?QM=<ssh alias of the quality machine>}
WORK=${WORK:?WORK=<work dir on QM>}
RUN=${RUN:-run2}
REPO=${REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)}
IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04
DATA_MOUNTS="-v $(dirname "${MODEL:-/x}"):$(dirname "${MODEL:-/x}"):ro -v $(dirname "${IMATRIX:-/x}"):$(dirname "${IMATRIX:-/x}"):ro"

case "${1:-}" in
lib)
    SRC=${SRC:?SRC=<llama.cpp tree on QM>}
    ssh "$QM" "mkdir -p $WORK && docker run --rm --label pascal-quants=1 --cpu-shares 256 -v $SRC:/src:ro -v $WORK:/out $IMG \
        bash -euo pipefail -c 'export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq cmake > /dev/null
        cmake -S /src -B /out/build-ggml -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
            -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF > /out/cmake.log 2>&1
        cmake --build /out/build-ggml -j 4 --target ggml-base > /out/build.log 2>&1'"
    ;;
proxy)
    MODEL=${MODEL:?MODEL=<BF16 safetensors dir on QM>}
    IMATRIX=${IMATRIX:?IMATRIX=<imatrix.gguf on QM>}
    ssh "$QM" "cat > $WORK/lowbit_proxy.py" < "$REPO/scripts/lowbit_proxy.py"
    ssh "$QM" "mkdir -p $WORK/$RUN && docker run --rm --label pascal-quants=1 --cpu-shares 256 -v $WORK:/w $DATA_MOUNTS $IMG bash -c '
        export DEBIAN_FRONTEND=noninteractive; apt-get update -qq >/dev/null && apt-get install -y -qq python3-numpy >/dev/null 2>&1
        OPENBLAS_NUM_THREADS=1 python3 -I /w/lowbit_proxy.py --model $MODEL --imatrix $IMATRIX --lib /w/build-ggml/bin/libggml-base.so \
            --out /w/$RUN/proxy.tsv --jobs 8 ${ARGS:-} > /w/$RUN/proxy.log 2>&1'"
    ;;
summary)
    shift
    ssh "$QM" "cat > $WORK/lowbit_summary.py" < "$REPO/scripts/lowbit_summary.py"
    ssh "$QM" "docker run --rm --label pascal-quants=1 --cpu-shares 256 -v $WORK:/w python:3.12-slim \
        python3 -I /w/lowbit_summary.py /w/$RUN/proxy.tsv $*"
    ;;
*)
    echo "usage: $0 lib | proxy | summary [lowbit_summary.py options]" >&2
    exit 1
    ;;
esac
