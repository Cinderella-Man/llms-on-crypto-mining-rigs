#!/usr/bin/env bash
# Q213's grid of llama.cpp memory projections for one Flash-Next file on nine cards: every combination of context,
# KV cache type, micro-batch and MTP draft (none, or a separate draft GGUF), through scripts/fit_probe.sh, which stops
# the server before the weights load. The files on the rig can be header-only sparse copies (README: "How it was made").
#   RIG=rig BUILDS=<rig builds folder> BUILD=<build folder> MODEL=<rig path of the first shard> TAG=<label prefix> \
#   OUT=<log> [TS=<-ts list> | PLANS=<plan_splits.sh output>] [DRAFT=<rig path of the MTP draft>] [DEVD=<draft's
#   device, e.g. CUDA8>] [CTXS="65536 131072 262144"] [UBS="128 256 512"] [KVS="f16 q8_0"] [MTP="0 1"] [NMAX=3] \
#   bash fit_grid.sh
# PLANS takes each configuration's -ts (and -ot, if the plan moves expert tensors) from plan_splits.sh's best plan for
# it, and skips configurations it has no plan for. Without TS or PLANS llama.cpp splits the layers by free memory
# (equal shares here). With MTP=1 the draft gets the same KV cache type as the main model and, without DEVD, the main
# model's -ts (its one layer and output then land on the last card).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${MODEL:?}" "${BUILDS:?}" "${BUILD:?}" "${TAG:?}" "${OUT:?}"
cfgs=()
for mtp in ${MTP:-0}; do
  for kv in ${KVS:-f16 q8_0}; do
    for ub in ${UBS:-128 256 512}; do
      for ctx in ${CTXS:-65536 131072 262144}; do
        a="-b 2048 -ub $ub"
        ts=${TS:-} ot=
        if [ -n "${PLANS:-}" ]; then
          plan=$(awk -v key="-$kv-ub$ub-$ctx$([ "$mtp" = 1 ] && echo -mtp) " \
            '/^== /{hit = index($0, key) > 0} hit && /^#1: -ts /{print $3, ($4 == "-ot" ? $5 : ""); exit}' "$PLANS")
          [ -n "$plan" ] || { echo "no plan for $kv-ub$ub-$ctx in $PLANS" >&2; continue; }
          read -r ts ot <<< "$plan"
        fi
        [ -n "$ts" ] && a="$a -ts $ts"
        [ -n "$ot" ] && a="$a -ot '$ot'"
        [ "$kv" = f16 ] || a="$a -ctk $kv -ctv $kv"
        case $ctx in 262144) k=262k ;; *) k=$((ctx / 1024))k ;; esac
        label="$TAG-$kv-ub$ub-$k"
        if [ "$mtp" = 1 ]; then
          a="$a -md ${DRAFT:?} --spec-type draft-mtp --spec-draft-n-max ${NMAX:-3}"
          [ "$kv" = f16 ] || a="$a -ctkd $kv -ctvd $kv"
          [ -n "${DEVD:-}" ] && a="$a -devd $DEVD"
          label="$label-mtp"
        fi
        cfgs+=("$label|$BUILD|$ctx|$a")
      done
    done
  done
done
CONFIGS=$(IFS=';'; echo "${cfgs[*]}") KEEP=${KEEP:-400} bash "$here/../../../../scripts/fit_probe.sh"
