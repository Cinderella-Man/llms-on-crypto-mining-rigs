# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: never ran (the queue stopped first).
. /tmp/rq/jobs/common.inc
mkdir -p $OUTD/q54
LLAMA_NO_MOE_DOWN_RESCALE=1 $FORK/llama-bench -m $S4L0 -ngl 99 -fa on -ub 128 -ctk q8_0 -ctv q8_0 \
    -p 0 -n 64 -r 1 -d 131072 -o jsonl > $OUTD/q54/9c-q8.jsonl
