# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: never ran (the queue stopped first).
. /tmp/rq/jobs/common.inc
LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$FORK MODEL=$S4L0 GPUS=0,1,2,3,4,5 TAG=q55_s4l0_6c bash $SC/tool_request_ab.sh
