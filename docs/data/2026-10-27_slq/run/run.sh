#!/usr/bin/env bash
# The tallies behind docs/research/notes/slq.md, as they ran. Everything executes on the quality machine (SSH alias QM)
# in Docker at --cpu-shares 256 with no GPU; the machine that runs this script only copies files over ssh.
#   gguf    fetch the three precision maps of kaitchup/Qwen3.8-27B-SLQ-GGUF at revision GGUF_REV (public, Apache-2.0)
#           into WORK and tally them with slq_bytes.py svg
#   vllm    tally the Humming-packed checkpoints kaitchup/Qwen3.8-27B-SLQ-EAR98, -EAR98.5 and -EAR99 with
#           slq_bytes.py safetensors (headers and config.json only; no weights are read)
#   fetch   copy WORK/out/*.tsv to OUT (default: this script's data folder)
# Env: QM, WORK (work dir on QM), CKPTS ("EAR98=<dir> EAR98.5=<dir> EAR99=<dir>", checkpoint dirs on QM), OUT.
set -euo pipefail
QM=${QM:?QM=<ssh alias of the quality machine>}
WORK=${WORK:?WORK=<work dir on QM>}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$(dirname "$HERE")}
GGUF_REV=${GGUF_REV:-8528ce42479c946842d7b5868dfb592fb7d4a52b}
IMG=python:3.12-slim

ssh "$QM" "mkdir -p $WORK/out $WORK/svg && cat > $WORK/slq_bytes.py" < "$HERE/slq_bytes.py"
case "${1:-}" in
gguf)
    ssh "$QM" "cd $WORK/svg && for e in ear98 ear98.5 ear99; do
        curl -sfL --max-time 120 -o precision-\$e.svg \
            https://huggingface.co/kaitchup/Qwen3.8-27B-SLQ-GGUF/resolve/$GGUF_REV/assets/precision-\$e.svg; done
        docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none -v $WORK:/w $IMG python3 -I /w/slq_bytes.py \
            svg /w/out EAR98=/w/svg/precision-ear98.svg EAR98.5=/w/svg/precision-ear98.5.svg EAR99=/w/svg/precision-ear99.svg"
    ;;
vllm)
    CKPTS=${CKPTS:?CKPTS=\"EAR98=<dir> EAR98.5=<dir> EAR99=<dir>\"}
    MOUNTS="" ARGS=""
    for c in $CKPTS; do MOUNTS="$MOUNTS -v ${c#*=}:${c#*=}:ro"; ARGS="$ARGS $c"; done
    ssh "$QM" "docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none -v $WORK:/w $MOUNTS $IMG \
        python3 -I /w/slq_bytes.py safetensors /w/out $ARGS"
    ;;
fetch)
    for f in gguf_tensors gguf_formats gguf_summary vllm_classes vllm_modules vllm_summary; do
        ssh "$QM" "cat $WORK/out/$f.tsv" > "$OUT/$f.tsv"
    done
    ;;
*)
    echo "usage: $0 gguf | vllm | fetch" >&2
    exit 1
    ;;
esac
