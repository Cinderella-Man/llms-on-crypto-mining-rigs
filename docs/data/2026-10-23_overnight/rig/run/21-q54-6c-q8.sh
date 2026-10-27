# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: ran (queued as 22-q54-6c-q8).
. /tmp/rq/jobs/common.inc
mkdir -p $OUTD/q54
LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $FORK/llama-bench -m $S4L0 -ngl 99 -fa on -ub 128 \
    -ctk q8_0 -ctv q8_0 -p 0 -n 64 -r 1 -d 0,8192,32768 -o jsonl > $OUTD/q54/6c-q8.jsonl
