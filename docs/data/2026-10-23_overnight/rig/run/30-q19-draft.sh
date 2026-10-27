# Overnight rig queue, 2026-10-23: job for scripts/rig_queue.sh (copy into /tmp/rq/jobs/ on the rig, with common.inc).
# Status: never ran (Q19 was decided without it).
# HEADS is a random-weight EAGLE-3 head with Flint's shapes (registry row Q19, stage 1c); VERIFY_TSV is job 13's output.
. /tmp/rq/jobs/common.inc
LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$FORK MODEL=$S4L0 GPUS=0,1,2,3,4,5,6 DRAFT_DEV=CUDA6 SERVER_ARGS="-ts 1,1,1,1,1,1,0" \
    HEADS=/mnt/models/eagle3/flint-eagle3-rand-v32k-Q8_0.gguf DRAFT_N="1 2 3" CTX=8192 \
    VERIFY_TSV=/tmp/eagle3_flint/q19_s4l0_synth/results.tsv TAG=q19_draft_s4l0 bash $SC/draft_cost.sh
