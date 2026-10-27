# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: never ran (the queue stopped first).
. /tmp/rq/jobs/common.inc
mkdir -p $OUTD/q65
GGML_CUDA_PROFILE_HOST=$OUTD/q65/phost-synth-w4.csv LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$FORK MODEL=$S4L0 GPUS=0,1,2,3,4,5 \
    PARTS=synth WIDTHS="4" CTX=8192 TAG=q65_host_w4 bash $SC/verify_cost.sh
