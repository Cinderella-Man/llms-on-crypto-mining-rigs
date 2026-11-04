#!/usr/bin/env bash
# The Pascal trellis proxy (Q929) as it ran. Everything executes on a separate CPU machine (SSH alias QM) in Docker at
# --cpu-shares 256 with no GPU, 8 worker processes; the machine that runs this script only copies files over ssh.
#   image    the run image: Ubuntu 24.04 with gcc and numpy (python3-numpy)
#   lib      libptq.so from run/ptq.c (the Viterbi encoder), into WORK
#   run      run/ptq_proxy.py over the sampled experts, detached; WORK/RUN/proxy.tsv, proxy.log, a done marker
#            proxy.done; ARGS adds options (--codes, --refs, --layers, ...)
#   summary  run/summary.py over WORK/RUN/proxy.tsv (with run/decode_cost.py's estimates); extra arguments go to it
# Env: QM, WORK (work dir on QM), RUN (run name), MODEL (Swift-1.5 BF16 safetensors dir), IMATRIX (FN4's
#      imatrix-swift15-q8_0-agent-train.gguf), LIB (CPU-only libggml-base.so of llama.cpp), LIB_N (the same of fork
#      branch pascal-p106-v3-lowbit, with N2P/N3P), GYRO_SRC (agentionai's llama.cpp tree at 9aa51ce4f), and the files
#      whose own bytes are read back: Q2_0 (Swift-1.5 GSQ-RCO Q2_0 shard 1), GYRO_M (Gyro-M), IQ3S_ISTA and IQ3S_SWIFT
#      (the GSQ-RCO IQ3_S files of the original model and of Swift-1.5, shard 1); MOUNTS: every folder those live in.
set -euo pipefail
QM=${QM:?QM=<ssh destination of the CPU machine, options allowed>}
WORK=${WORK:?WORK=<work dir on QM>}
RUN=${RUN:-main}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
IMG=pq-ptq:24.04
VOLS="-v $WORK:/w"
for m in ${MOUNTS:-}; do VOLS="$VOLS -v $m:$m:ro"; done

case "${1:-}" in
image)
    ssh $QM "docker image inspect $IMG > /dev/null 2>&1 || printf 'FROM ubuntu:24.04\nRUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libc6-dev python3-numpy > /dev/null && rm -rf /var/lib/apt/lists/*\nLABEL pascal-quants=1\n' | docker build -q -t $IMG -"
    ;;
lib)
    ssh $QM "mkdir -p $WORK/run" && ssh $QM "cat > $WORK/run/ptq.c" < "$HERE/ptq.c"
    ssh $QM "docker run --rm --label pascal-quants=1 --cpu-shares 256 $VOLS $IMG gcc -O3 -march=x86-64-v3 -fPIC -shared -o /w/libptq.so /w/run/ptq.c -lm"
    ;;
run)
    for f in ptq_proxy.py summary.py decode_cost.py; do ssh $QM "cat > $WORK/run/$f" < "$HERE/$f"; done
    ssh $QM "cat > $WORK/run/lowbit_proxy.py" < "$REPO/scripts/lowbit_proxy.py"
    ssh $QM "mkdir -p $WORK/$RUN && rm -f $WORK/$RUN/proxy.done && cd $WORK/$RUN && docker run -d --rm --name ptq-$RUN \
        --label pascal-quants=1 --cpu-shares 256 --cpus 8 $VOLS $IMG bash -c 'OPENBLAS_NUM_THREADS=1 python3 -I /w/run/ptq_proxy.py \
        --model $MODEL --imatrix $IMATRIX --lowbit /w/run/lowbit_proxy.py --ptq /w/libptq.so --lib $LIB --lib-n ${LIB_N:-} \
        --gyro-src $GYRO_SRC --q2-0 ${Q2_0:-} --gyro-m ${GYRO_M:-} --iq3s-ista ${IQ3S_ISTA:-} --iq3s-swift ${IQ3S_SWIFT:-} \
        --out /w/$RUN/proxy.tsv --jobs 8 ${ARGS:-} > /w/$RUN/proxy.log 2>&1; echo \$? > /w/$RUN/proxy.done' > /dev/null"
    ;;
summary)
    shift
    for f in summary.py decode_cost.py; do ssh $QM "cat > $WORK/run/$f" < "$HERE/$f"; done
    ssh $QM "docker run --rm --label pascal-quants=1 --cpu-shares 256 $VOLS $IMG python3 -I /w/run/summary.py /w/$RUN/proxy.tsv $*"
    ;;
*)
    echo "usage: $0 image | lib | run | summary [summary.py options]" >&2
    exit 1
    ;;
esac
