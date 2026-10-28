#!/usr/bin/env bash
# The fix branch's server checks (pascal-p106-v3-promptpath): three llama-servers on a tiny random Flint, one slot, the
# rig's -b 512 -ub 128 and --ctx-checkpoints 4 --cache-ram 0: "on" with LLAMA_SERVER_TEXT_REUSE=1 and ON_EXTRA, "off"
# and "cold" without; then textreuse_check.py against them.
#   CPU (Q260): BIN = build_cpu.sh's SRC=$WORK/src-pp OUT=$WORK/out-cpu-pp build, -fa off (batch-invariant on the CPU,
#               so the warm answer must equal the cold server's);
#   GPU (Q261): GPUS=3,4 and BIN = the rig tarball's bin/ (compute 6.1 PTX, compiled for the card at load), the model
#               layer-split over both cards, -fa on as on the rig; the cold comparison is reported, not required (other
#               micro-batch shapes round differently on a GPU, and a random model's flat logits flip greedy picks).
# ON_EXTRA adds variables to the "on" server: LLAMA_SERVER_TAIL_CKPT_MIN=4096 TAG=_tail (Q263);
# "LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_MAX=16 LLAMA_GPU_KQ_MASK_CHECK=1" N_PREDICT=1100 TAG=_mask (Q261: the card-built
# mask in decode steps only, past the 768-cell window ring, every card mask compared with the host mask; on and off must
# then sample the same first reply from the same seed).
#   WORK=... MODEL=/path/tiny-flint.gguf [GPUS=3,4 BIN=...] bash check_server.sh
# Output: $WORK/out/textreuse_check$TAG.tsv and the servers' logs.
set -euo pipefail
: "${WORK:?set WORK}" "${MODEL:?set MODEL}"
BIN=${BIN:-$WORK/out-cpu-pp/build/bin}
mkdir -p "$WORK/out" "$WORK/cuda-cache"
if [ -n "${GPUS:-}" ]; then
    DEV=(--gpus "\"device=$GPUS\"" -e CUDA_CACHE_PATH=/cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 -v "$WORK/cuda-cache:/cuda-cache")
    FA="-fa on -ngl 99 -ts 1,1"; COLD=--cold-informational
else
    DEV=(); FA="-fa off -ngl 99"; COLD=
fi
start() {  # name port env...
    local name=$1 port=$2 envs=() e; shift 2
    for e in "$@"; do [ -n "$e" ] && envs+=(-e "$e"); done
    docker rm -f "pq-rereads-$name" >/dev/null 2>&1 || true
    docker run -d --rm --cpu-shares=256 --label pascal-quants=1 --name "pq-rereads-$name" --network host "${DEV[@]}" \
        "${envs[@]}" -v "$BIN:/bin-llama:ro" -v "$MODEL:/model.gguf:ro" nvidia/cuda:12.9.2-devel-ubuntu24.04 \
        /bin-llama/llama-server -m /model.gguf $FA -c 8192 -np 1 -t 2 -b 512 -ub 128 --ctx-checkpoints 4 \
        --cache-ram 0 --jinja --reasoning-format deepseek --host 127.0.0.1 --port "$port" --no-webui >/dev/null
}
start on 18441 LLAMA_SERVER_TEXT_REUSE=1 ${ON_EXTRA:-}
start off 18442 LLAMA_SERVER_TEXT_REUSE=0
start cold 18443 LLAMA_SERVER_TEXT_REUSE=0
for p in 18441 18442 18443; do
    for i in $(seq 600); do curl -sf "http://127.0.0.1:$p/health" >/dev/null && break; sleep 1; done
done
rc=0
docker run --rm --cpu-shares=256 --label pascal-quants=1 --network host -v "$WORK:/w" -w /w python:3.12-slim \
    python3 -I run/textreuse_check.py --on http://127.0.0.1:18441 --off http://127.0.0.1:18442 \
    --cold http://127.0.0.1:18443 --trials "${TRIALS:-6}" --n-predict "${N_PREDICT:-700}" $COLD \
    > "$WORK/out/textreuse_check${TAG:-}.tsv" || rc=$?
for s in on off cold; do docker logs "pq-rereads-$s" > "$WORK/out/textreuse${TAG:-}_$s.log" 2>&1; docker rm -f "pq-rereads-$s" >/dev/null; done
L=$WORK/out/textreuse${TAG:-}_on.log
echo "card masks identical to the host mask: $(grep -a -c 'identical to the host mask' "$L"), by micro-batch size:" \
    "$(grep -a -o 'copy [0-9]*: [0-9]* tokens' "$L" | awk '{print $3}' | sort -n | uniq -c | tr -s ' \n' ' ')" \
    >> "$WORK/out/textreuse_check${TAG:-}.tsv"
tail -2 "$WORK/out/textreuse_check${TAG:-}.tsv"
exit $rc
