# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: ran (queued as 23-q43_plain).
. /tmp/rq/jobs/common.inc
M=/mnt/models2/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF
 BIN=$FORK GPUS=0,1,2,3,4 MODEL=$M/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf TAG=q43_plain MAX_TOKENS=512 \
    SERVER_ARGS="-c 32768 -b 512 -ub 128 -bs  " bash $SC/server_bench.sh
