#!/usr/bin/env bash
# r137_loads.sh: R137 (Q1429), GGML_CUDA_KEEPALIVE_MS=250 as a plain setting, alone and with C811's switch, on a P106
# rig, untimed by anything else. Runs ON the rig, detached, the rig idle; ends with $OUT/DONE.
# It is R128's driver (docs/data/2026-11-12_r128_keepalive_settings/run/r128_loads.sh, which runs R108's
# docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh unchanged: every load the preset exactly through
# bench/agent_check/rig_server.sh with the host profile, the n-gram table's file dropped from the page cache before the
# load, C761's vm_sampler.sh every 0.5 s, S37's greedy request then the first-turn request) with these loads at R122's
# rung f (docs/data/2026-11-12_r122_host_ladder/run/preset.env plus rung f's eight switches), on a build with C811 and
# C817 (GGML_CUDA_KEEPALIVE_EAGER: the keep-alive block of a card starts no sooner than the card's first graph and stops
# at a held start without a wait; docs/data/2026-11-12_c811_keepalive_wait/, docs/data/2026-11-12_c817_keepalive_held/):
#   f1         rung f + GGML_CUDA_KEEPALIVE_EAGER=1 + GGML_CUDA_KEEPALIVE_MS=1000   the reference (1,000 is the default)
#   f1ka250    rung f + GGML_CUDA_KEEPALIVE_EAGER=1 + GGML_CUDA_KEEPALIVE_MS=250    the switch with the shorter timeout
#   f0ka250    rung f + GGML_CUDA_KEEPALIVE_EAGER=0 + GGML_CUDA_KEEPALIVE_MS=250    the setting alone (R128's fka250)
#   <load>-2   the same again; <load>np the same without the host profile (GGML_CUDA_PROFILE_HOST set empty, which the
#              engine reads as off; FN87: the profile lengthens the long prompt)
# Order: the three profiled, the three unprofiled, the three profiled again, so that a drift over the run falls on
# every setting alike.
# Usage: BIN=<bin/ of the build> MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> TABLE=<the shard with the n-gram
#   table> PRESET=<R122's run/preset.env> RIG_SERVER=<bench/agent_check/rig_server.sh> VM_SAMPLER=<vm_sampler.sh>
#   RIG_LOADS=<R108's rig_loads.sh> REQ_GREEDY=<json> REQ_LONG=<json> OUT=<dir> [LOADS=...] bash r137_loads.sh
set -uo pipefail
: "${RIG_LOADS:?RIG_LOADS=<docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh>}"
E=LLAMA_DRAFT_PREFETCH=1,LLAMA_VERIFY_PREPARE=2,LLAMA_MTP_GRAPH_KEEP=1,LLAMA_KV_SEQ_RM_BOUND=1,LLAMA_SERVER_YIELD=2
F=$E,LLAMA_MTP_EMBD_FIRST=1,LLAMA_MTP_DRAFT_CAP=1,LLAMA_MTP_SKIP_LAST_ROW=1
A=$F,GGML_CUDA_KEEPALIVE_EAGER=1,GGML_CUDA_KEEPALIVE_MS=1000
B=$F,GGML_CUDA_KEEPALIVE_EAGER=1,GGML_CUDA_KEEPALIVE_MS=250
C=$F,GGML_CUDA_KEEPALIVE_EAGER=0,GGML_CUDA_KEEPALIVE_MS=250
NP=GGML_CUDA_PROFILE_HOST=
export LOADS=${LOADS:-f1 f1ka250 f0ka250 f1np f1ka250np f0ka250np f1-2 f1ka250-2 f0ka250-2}
export LOAD_ENV="f1:$A f1-2:$A f1np:$A,$NP f1ka250:$B f1ka250-2:$B f1ka250np:$B,$NP f0ka250:$C f0ka250-2:$C f0ka250np:$C,$NP"
exec bash "$RIG_LOADS"
