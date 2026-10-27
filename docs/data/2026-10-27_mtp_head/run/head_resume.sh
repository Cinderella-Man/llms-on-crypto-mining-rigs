#!/usr/bin/env bash
# Flint's MTP heads, the overnight round after the 240-minute heads (resumed 15:37 and stopped at 16:35 on 2026-10-27,
# when the owner gave the machine to another job). HEAD_ENV: an env file like head.env.example.
#   bash head_resume.sh start [TOTAL_MINUTES]
#       snapshot the first round (heads/<name>-240min, gguf/flint-mtp-<name>-240min.gguf, scores-240min), then
#       train.py --resume with a total budget of TOTAL_MINUTES (default 1000: ~3.6 epochs of 14.3 M targets at ~870
#       targets/s per head; the 240 minutes already spent count), same GPUs and options as the first round (cmd.txt)
#   bash head_resume.sh finish
#       after both have stopped: gate each head.pt again, then agentic_head.sh STEPS="score gguf" (scores on Q51 and
#       the held-out sessions, GGUF files)
# The time-based cosine schedule restarts from the resumed budget's position (a warm restart: the learning rate rises
# from 10 % back to ~87 % of its peak, then decays again).
set -euo pipefail
. "${HEAD_ENV:?}"
HERE=$BENCH/staircase_drafter
TORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
mounts=(-v "$BENCH:$BENCH:ro" -v "$WORK:$WORK")
for d in $FEATS_DIRS; do mounts+=(-v "$d:$d"); done
for r in $RUNS; do mounts+=(-v "$r:$r:ro"); done
IFS='|' read -ra hs <<< "$HEADS"
case "${1:-}" in
start)
    TOTAL=${2:-1000}
    [ -d "$WORK/scores" ] && [ ! -d "$WORK/scores-240min" ] && cp -a "$WORK/scores" "$WORK/scores-240min"
    for h in "${hs[@]}"; do
        name=${h%%:*}; rest=${h#*:}; gpu=${rest%%:*}
        H=$WORK/heads/$name
        [ -f "$H/ckpt.pt" ] || { echo "no checkpoint in $H"; exit 1; }
        if docker ps --format '{{.Names}}' | grep -qx "pq-head-$name"; then echo "pq-head-$name is running"; exit 1; fi
        [ -d "$H-240min" ] || cp -a "$H" "$H-240min"
        g=$WORK/gguf/flint-mtp-$name.gguf
        [ -f "$g" ] && [ ! -f "${g%.gguf}-240min.gguf" ] && cp -a "$g" "${g%.gguf}-240min.gguf"
        cmd="$(sed -E "s/--minutes [0-9.]+/--minutes $TOTAL/" "$H/cmd.txt") --resume"
        echo "$cmd" > "$H/cmd-resume.txt"
        docker rm -f "pq-head-$name" > /dev/null 2>&1 || true
        docker run -d --name "pq-head-$name" --label pascal-quants=1 --cpu-shares 256 --gpus "\"device=$gpu\"" \
            -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --shm-size 8g \
            --network none "${mounts[@]}" -w "$HERE" "$TORCH_IMAGE" bash -c "$cmd >> $H/train.log 2>&1" > /dev/null
        echo "$(date +%T) resumed head $name on GPU $gpu to $TOTAL minutes in all" | tee -a "$WORK/head.out"
    done
    ;;
finish)
    for h in "${hs[@]}"; do
        name=${h%%:*}; H=$WORK/heads/$name
        if docker ps --format '{{.Names}}' | grep -qx "pq-head-$name"; then echo "pq-head-$name still runs"; exit 1; fi
        docker rm "pq-head-$name" > /dev/null 2>&1 || true
        docker run --rm --label pascal-quants=1 --cpu-shares 256 --network none "${mounts[@]}" -w "$HERE" "$TORCH_IMAGE" \
            python -I check_sources.py "$HERE/sources.json" derived "$H/head.pt" --kind head \
            --parent "$WORK/feats.gate.json" "$WORK/vocab.gate.json" "$WORK/export/train/seq.gate.json" \
            --cmd "$(cat "$H/cmd-resume.txt")" --allow-dir "$H" "$WORK/flint" --out "$H/head.gate.json" \
            | tail -1 | tee -a "$WORK/head.out"
    done
    export STEPS="score gguf"
    bash "$HERE/agentic_head.sh" >> "$WORK/head.out" 2>&1
    ;;
*)
    echo "usage: head_resume.sh start [TOTAL_MINUTES] | finish"; exit 1 ;;
esac
