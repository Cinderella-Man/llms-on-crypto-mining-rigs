# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: ran.
. /tmp/rq/jobs/common.inc
mkdir -p $OUTD/q19
LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 GGML_CUDA_PROFILE=$OUTD/q19/prof-pp4.csv \
    $FORK/llama-bench -m $S4L0 -ngl 99 -fa on -ub 128 -p 4 -n 0 -r 24
