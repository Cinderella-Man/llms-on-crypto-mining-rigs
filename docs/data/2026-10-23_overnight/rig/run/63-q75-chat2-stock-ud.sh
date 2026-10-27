# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: never ran (the queue stopped first).
. /tmp/rq/jobs/common.inc
UD=/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf
Q8=/mnt/models2/unsloth/flint-GGUF/flint-Q8_0.gguf
 BIN=$STOCK MODEL=$UD TAG=chat2-stock-ud PROMPTS=chat MAX_TOKENS=65536 GPUS=0,1,2,3,4,5,6,7,8 \
    SERVER_ARGS="-c 262144 -b 512 -ub 128 -bs --ctx-checkpoints 4 " bash $SC/server_bench.sh
