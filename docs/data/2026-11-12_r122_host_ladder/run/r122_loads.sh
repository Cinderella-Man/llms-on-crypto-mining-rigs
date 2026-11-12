#!/usr/bin/env bash
# r122_loads.sh: R122 (Q1400), the decode step's eight host-side switches of pascal-p106-v4 as one ladder on a P106 rig,
# untimed by anything else. Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# It is R108's driver (docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh: every load the preset exactly through
# bench/agent_check/rig_server.sh with the host profile, the n-gram table's file dropped from the page cache before the
# load, C761's vm_sampler.sh every 0.5 s, S37's greedy request then the first-turn request) with the ladder's loads:
#   base  the preset (run/preset.env)
#   a     + LLAMA_DRAFT_PREFETCH=1        C615: each drafted token's n-gram rows issued as the drafter has them
#   b     + LLAMA_VERIFY_PREPARE=2        C738: the verify pass prepared during the last drafting pass, inputs uploaded
#   c     + LLAMA_MTP_GRAPH_KEEP=1        C619: the draft's graph kept between its passes
#   d     + LLAMA_KV_SEQ_RM_BOUND=1       C740: seq_rm reads one sequence's cells from its index
#   e     + LLAMA_SERVER_YIELD=2          C741: one yield to the task queue a step, the worker not woken
#   f     + LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1
#                                         C771, C744, C742: the draft's gather first, the slot's cap, no last hidden row
# then the seven again (base-2 … f-2), so that every rung has two loads and a spread.
# Usage: BIN=<bin/ of the build> MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> TABLE=<the shard with the n-gram
#   table> PRESET=<run/preset.env> RIG_SERVER=<bench/agent_check/rig_server.sh> VM_SAMPLER=<vm_sampler.sh>
#   RIG_LOADS=<R108's rig_loads.sh> REQ_GREEDY=<json> REQ_LONG=<json> OUT=<dir> bash r122_loads.sh
set -uo pipefail
: "${RIG_LOADS:?RIG_LOADS=<docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh>}"
A=LLAMA_DRAFT_PREFETCH=1
B=$A,LLAMA_VERIFY_PREPARE=2
C=$B,LLAMA_MTP_GRAPH_KEEP=1
D=$C,LLAMA_KV_SEQ_RM_BOUND=1
E=$D,LLAMA_SERVER_YIELD=2
F=$E,LLAMA_MTP_EMBD_FIRST=1,LLAMA_MTP_DRAFT_CAP=1,LLAMA_MTP_SKIP_LAST_ROW=1
export LOADS=${LOADS:-base a b c d e f base-2 a-2 b-2 c-2 d-2 e-2 f-2}
export LOAD_ENV="a:$A b:$B c:$C d:$D e:$E f:$F a-2:$A b-2:$B c-2:$C d-2:$D e-2:$E f-2:$F"
exec bash "$RIG_LOADS"
