#!/usr/bin/env bash
# agent_arms.sh OUT: Q230 on the agent check. Two arms with the same build, file and flags: sparse0 (the sparse decode
# attention off: the dense kernels read every cell of the 12 QSA layers, as -ladder does) and sparse1 (on), one arm at a
# time through ../../2026-11-03_flash_next_pipelined_rig/run/agent_check.sh (the rig's quiet check, mem_guard.sh,
# release_check_s12.sh), then one report over both. Both arms log the QSA k-pool inputs' host time and bytes
# (LLAMA_QSA_INPUT_STATS, Q441's measurement) into their server logs; qsa_inputs-<arm>.txt collects those lines.
#
# Runs on the machine that drives the check (it only runs ssh), with agent_check.sh's environment (RIG, CLIENT,
# CLIENT_OUT, the ports, CLIENT_DIR, REL_BIN = this build's llama-server, MODEL, ...; see that script and
# release_check_s12.sh) plus ARGS (the server flags of both arms), ARMS (default "sparse0 sparse1"), STATS (default 64).
set -uo pipefail
OUT=$(realpath -m "${1:?usage: agent_arms.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
AC=${AC:-$HERE/../../2026-11-03_flash_next_pipelined_rig/run/agent_check.sh}
ARMS=${ARMS:-sparse0 sparse1}
: "${ARGS:?ARGS=<server flags>}" "${CLIENT:?}" "${CLIENT_OUT:?}"
for arm in $ARMS; do
    export "ARGS_$arm=$ARGS"
done
mkdir -p "$OUT"

STEPS=prep ARMS="$ARMS" bash "$AC" "$OUT" || exit 1
for arm in $ARMS; do
    v=${arm#sparse}
    REL_ENV="GGML_CUDA_FATTN_SPARSE_PASCAL=$v LLAMA_QSA_INPUT_STATS=${STATS:-64}" STEPS=arms ARMS=$arm bash "$AC" "$OUT" \
        || exit 1
done
STEPS=report ARMS="$ARMS" bash "$AC" "$OUT"
for arm in $ARMS; do
    grep -h "QSA k-pool inputs" "$OUT/rig/server-$arm.log" 2> /dev/null | sed 's/^.*QSA k-pool inputs/QSA k-pool inputs/' \
        > "$OUT/qsa_inputs-$arm.txt"
done
