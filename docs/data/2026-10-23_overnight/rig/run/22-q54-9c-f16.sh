# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: ran (queued as 265-q54-9c-f16).
. /tmp/rq/jobs/common.inc
mkdir -p $OUTD/q54
LLAMA_NO_MOE_DOWN_RESCALE=1 $FORK/llama-bench -m $S4L0 -ngl 99 -fa on -ub 128 \
    -p 0 -n 64 -r 1 -d 32768,65536,131072 -o jsonl > $OUTD/q54/9c-f16.jsonl
