#!/usr/bin/env bash
# The agent cache (Q111) on one GPU with the real Flint: correctness against a cold slot, then a short replay of real
# sessions in two configurations, for the token counts and the replies (not for speed).
#
#   WORK=<dir> GPU=<index> MODEL=<flint.gguf> [MOUNTS="-v <dir>:<dir>:ro ..."] [LOCK=<lock file>] [SESSIONS=6] \
#       bench/agent_cache/gpu_check.sh [check|replay|all]
# (MOUNTS: the folders the container needs to see MODEL, at the same paths)
#
# WORK holds out-cuda/build/bin/llama-server (+ out-cuda/cuda12-libs), sessions-e2-ud.jsonl and bench/agent_cache/;
# results go to WORK/gpu-check/. Everything runs in one container, under `flock LOCK` when LOCK is set (a shared card:
# one acquisition, ~15-20 min for all).
# - check: one server with two slots, slot 0 under test (prefix cache, ring checkpoints, generation checkpoints) and
#   slot 1 cold (cache_prompt off); check_exact.py's prefix, turns and trim scenarios, then the prefix again after a
#   restart (from its file). GPU kernels sum in other orders for other batch sizes, so only paths whose reused cells
#   come from the same prompt evaluations as a cold slot's are expected to match bit for bit (the prefix restore); the
#   others report the first token's largest log-probability difference.
# - replay: replay_sessions.py on SESSIONS sessions (two groups that share a tool list), first as stock llama.cpp serves
#   them (a reserve per request, two tail checkpoints, no prefix files or generation checkpoints), then with the agent
#   cache; the summary compares prompt tokens read per generated token, reuse classes and replies.
# The Pascal kernels are forced on (they are written for compute 6.1), as in the rig builds' own checks.
set -uo pipefail
WORK=${WORK:?}; GPU=${GPU:?}; MODEL=${MODEL:?}
MODE=${1:-all}
OUT=$WORK/gpu-check
mkdir -p "$OUT" "$WORK/cuda-cache"
docker run --rm --label pascal-quants=1 -v "$WORK:/ac" python:3.12-slim python3 /ac/bench/agent_cache/replay_sessions.py \
    pick /ac/sessions-e2-ud.jsonl --groups 2 --per-group $((${SESSIONS:-6} / 2)) --write /ac/gpu-check/sessions.jsonl \
    > "$OUT/pick.txt" 2>&1
locked() { if [ -n "${LOCK:-}" ]; then flock "$LOCK" "$@"; else "$@"; fi; }
locked docker run --rm --name pq-q111-gpu-$MODE --label pascal-quants=1 --gpus "\"device=$GPU\"" --network host \
    -e LD_LIBRARY_PATH=/ac/out-cuda/cuda12-libs -e CUDA_CACHE_PATH=/ac/cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 \
    -e GGML_CUDA_MMVQ_PASCAL=2 -e GGML_CUDA_PERSISTENT_MOE=2 -e LLAMA_NO_MOE_DOWN_RESCALE=tg -e LLAMA_INPUT_ON_GPU=1 \
    -e LLAMA_INPUT_UPLOAD=2 -e MODEL="$MODEL" -e MODE="$MODE" \
    -v "$WORK:/ac" ${MOUNTS:-} \
    nvidia/cuda:12.9.2-devel-ubuntu24.04 bash /ac/bench/agent_cache/gpu_inner.sh 2>&1 | tee "$OUT/gpu-$MODE.txt"
exit "${PIPESTATUS[0]}"
