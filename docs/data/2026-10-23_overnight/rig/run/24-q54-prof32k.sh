# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: never ran (the queue stopped first).
. /tmp/rq/jobs/common.inc
mkdir -p $OUTD/q54
LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 GGML_CUDA_PROFILE=$OUTD/q54/prof-d32k.csv \
    $FORK/llama-bench -m $S4L0 -ngl 99 -fa on -ub 128 -p 0 -n 32 -r 1 -d 32768
LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 GGML_CUDA_PROFILE_HOST=$OUTD/q54/phost-d32k.csv \
    $FORK/llama-bench -m $S4L0 -ngl 99 -fa on -ub 128 -p 0 -n 32 -r 1 -d 32768
