# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: ran.
. /tmp/rq/jobs/common.inc
LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$FORK MODEL=$S4L0 GPUS=0,1,2,3,4,5 PARTS=synth WIDTHS="1 3 5" CTX=8192 TAG=q19_s4l0_synth bash $SC/verify_cost.sh
