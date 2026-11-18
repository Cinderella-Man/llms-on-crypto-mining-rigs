#!/usr/bin/env bash
# C854 step 4 (Q1525), Flash-Next: R156's driver (docs/data/2026-11-13_r156_agent_v4r/run/r156.sh), arm v4r, which is
# the released v4.0 preset (release/flash-next-v4/settings.env), with the candidate's llama-server; then its report.
# Run from the repository root on the machine that drives the rig. Machine-specific values come from the environment:
#   RIG          ssh alias of the rig
#   CLIENT       local, or the ssh alias of the machine that runs the agent's sandboxes
#   MODEL, DRAFT Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf and mtp-Qwen3.8-Flash-Next-draft.gguf, paths
#                on the rig
#   NEW          the candidate's build folder under /opt/llama-builds on the rig
#   REF          v4.0's build folder there (r156.sh requires it; the v4r arm does not run it)
#   RIGDIR       a folder on the rig for the server's and the host's logs (moved to the client at the end, then removed)
#   CLIENT_DIR, CLIENT_OUT   the bench files and the results folder on the client
#   OUT          a local folder
export RIG=${RIG:?} CLIENT=${CLIENT:?} MODEL=${MODEL:?} DRAFT=${DRAFT:?} NEW=${NEW:?} REF=${REF:?} RIGDIR=${RIGDIR:?} \
  OUT=${OUT:?} STEPS="agent report" AGENT_ORDER=v4r V4R_DRAFT_UB=512 REPORT_ARMS=v4r \
  RIG_PORT=${RIG_PORT:-8282} RELAY_PORT=${RELAY_PORT:-8286} PROXY_PORT=${PROXY_PORT:-18783} \
  CLIENT_DIR=${CLIENT_DIR:?} CLIENT_OUT=${CLIENT_OUT:?} CPU_SHARES=256
exec timeout 5400 bash docs/data/2026-11-13_r156_agent_v4r/run/r156.sh
