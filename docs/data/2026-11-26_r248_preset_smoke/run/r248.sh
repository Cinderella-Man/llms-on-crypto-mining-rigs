#!/usr/bin/env bash
# R248 (Q1938): one coding-agent run of Flash-Next's v5 preset as it stands (release/flash-next-v5/settings.env at public
# v5 fa31e1b8: -c 131072 -ub 256, GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.45) through bench/agent_check/preset_agent.sh,
# arm pre, the 12 bugfix tasks; the machine that runs this is also the agent's client (CLIENT=local, Docker with the
# agent-check:0.87.1 image). Run from the repository root:
#   RIG=<ssh alias> MODELS=<the rig's Flash-Next folder> NEW=<build folder under /opt/llama-builds> WORK=<local folder> \
#     bash docs/data/2026-11-26_r248_preset_smoke/run/r248.sh            (DRY=1 first: the line only)
# MODELS holds swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf and
# mtp-Qwen3.8-Flash-Next-draft.gguf. The rig's logs go to /var/tmp/r248-pre on the rig and are removed after the run;
# the client's results to WORK/client-pre (its bench files to WORK/client-bench). Ports: the rig's 8282, here 8308 and
# 18808 (they must be free). AGENT_BY=21:15: no arm starts later (an arm of at most 52 minutes ends by the rig's 22:00
# stop).
set -u
: "${RIG:?}" "${MODELS:?}" "${NEW:?}" "${WORK:?}"
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
export ARMS=pre AGENT_BY=${AGENT_BY:-21:15} PREFIX=acr-r248-pre RIG CLIENT=local \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf NEW RIGDIR=/var/tmp/r248-pre OUT=$WORK/out \
  RIG_PORT=8282 RELAY_PORT=8308 PROXY_PORT=18808 CLIENT_DIR=$WORK/client-bench CLIENT_OUT=$WORK/client-pre \
  STEPS="files agent report"
unset BIN PRESET ENV_pre SET_pre REPORT_ARMS
timeout 3600 bash bench/agent_check/preset_agent.sh
