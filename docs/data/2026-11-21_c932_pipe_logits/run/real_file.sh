#!/usr/bin/env bash
# C932 (2026-11-21): bench/pipespec/check_identity.sh on a real Flash-Next file on a build machine's CPU, plain against
# pipelined, once per build of BUILDS: greedy, the model's chat template, 64 tokens, depth 1, 0 and 50 % wrong guesses
# (plain, pipe-d0, two oracles and four guessers). Correctness only: nothing is timed.
# Usage: W=<work dir with bin/<commit>/ from check_at.sh> MODEL=<shard 1 of the gguf> REPO=<folder with pipespec/>
#   [BUILDS='base:3dfad876f head:80d83fa1a'] [THREADS=8] [MEM=56g] bash real_file.sh
# Each build in a container (labels pascal-quants=1 and c932=1, --cpu-shares=256, --memory=$MEM: the file's pages count
# toward it, no network), its output in $W/real/<tag>/ and $W/real/<tag>.log. Done when $W/real/DONE exists.
set -uo pipefail
W=${W:?W=<work dir>}
MODEL=${MODEL:?MODEL=<gguf shard 1>}
REPO=${REPO:?REPO=<scripts folder>}
BUILDS=${BUILDS:-base:3dfad876f head:80d83fa1a}
THREADS=${THREADS:-8}
MEM=${MEM:-56g}
MD=$(dirname "$MODEL")
mkdir -p "$W/real"
rm -f "$W/real/DONE"
for t in $BUILDS; do
    tag=${t%%:*} BB=$W/bin/${t#*:}
    echo "=== $tag $(date -Is)"
    docker run --rm --label pascal-quants=1 --label c932=1 --cpu-shares=256 --memory="$MEM" --memory-swap="$MEM" --network none \
        -v "$W:$W" -v "$REPO:$REPO:ro" -v "$MD:$MD:ro" -e LD_LIBRARY_PATH="$BB" pq-c565-cpu:24.04 \
        env BIN="$BB" MODEL="$MODEL" CHAT=1 SPEED=0 TIMEOUT=5400 PROMPT="$REPO/pipespec/prompts/code.txt" OUT="$W/real/$tag" \
        ARGS="-ngl 0 -t $THREADS -c 4096 -fa on" N=64 DEPTHS=1 MISSES="0 0.5" bash "$REPO/pipespec/check_identity.sh" \
        > "$W/real/$tag.log" 2>&1
    echo "$tag rc=$?"
    cat "$W/real/$tag.log"
done
date -Is > "$W/real/DONE"
