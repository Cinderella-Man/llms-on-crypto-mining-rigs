#!/usr/bin/env bash
# r128_loads.sh: R128 (Q1419), the keep-alive block's first-call waits (FN85, docs/data/2026-11-12_fn85_server_gap/)
# without code, and rung f's three switches one at a time, on a P106 rig, untimed by anything else. Runs ON the rig,
# detached, the rig idle; ends with $OUT/DONE.
# It is R122's driver (docs/data/2026-11-12_r122_host_ladder/run/r122_loads.sh, which runs R108's
# docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh unchanged: every load the preset exactly through
# bench/agent_check/rig_server.sh with the host profile, the n-gram table's file dropped from the page cache before the
# load, C761's vm_sampler.sh every 0.5 s, S37's greedy request then the first-turn request) with these loads, on R122's
# preset (run/preset.env there):
#   fka250   rung f + GGML_CUDA_KEEPALIVE_MS=250   the block ends 250 ms after its card's last graph, not 1,000
#   feager   rung f + CUDA_MODULE_LOADING=EAGER     every kernel loaded on every card when the context is made
#   e771     rung e + LLAMA_MTP_EMBD_FIRST=1        C771 alone
#   e744     rung e + LLAMA_MTP_DRAFT_CAP=1         C744 alone
#   e742     rung e + LLAMA_MTP_SKIP_LAST_ROW=1     C742 alone
#   fnoprof  rung f without the host profile        GGML_CUDA_PROFILE_HOST set empty (the engine reads empty as off)
# Rung e is the preset + C615 C738 C619 C740 C741, rung f e + C771 C744 C742 (R122's ladder). The reference loads are
# R122's own e, e-2, f and f-2 (same rig, build and preset).
# Usage: BIN=<bin/ of the build> MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> TABLE=<the shard with the n-gram
#   table> PRESET=<R122's run/preset.env> RIG_SERVER=<bench/agent_check/rig_server.sh> VM_SAMPLER=<vm_sampler.sh>
#   RIG_LOADS=<R108's rig_loads.sh> REQ_GREEDY=<json> REQ_LONG=<json> OUT=<dir> bash r128_loads.sh
set -uo pipefail
: "${RIG_LOADS:?RIG_LOADS=<docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh>}"
E=LLAMA_DRAFT_PREFETCH=1,LLAMA_VERIFY_PREPARE=2,LLAMA_MTP_GRAPH_KEEP=1,LLAMA_KV_SEQ_RM_BOUND=1,LLAMA_SERVER_YIELD=2
F=$E,LLAMA_MTP_EMBD_FIRST=1,LLAMA_MTP_DRAFT_CAP=1,LLAMA_MTP_SKIP_LAST_ROW=1
export LOADS=${LOADS:-fka250 feager e771 e744 e742 fnoprof}
export LOAD_ENV="fka250:$F,GGML_CUDA_KEEPALIVE_MS=250 feager:$F,CUDA_MODULE_LOADING=EAGER e771:$E,LLAMA_MTP_EMBD_FIRST=1 e744:$E,LLAMA_MTP_DRAFT_CAP=1 e742:$E,LLAMA_MTP_SKIP_LAST_ROW=1 fnoprof:$F,GGML_CUDA_PROFILE_HOST="
exec bash "$RIG_LOADS"
