#!/usr/bin/env bash
# The P106 check of the v5 candidate (Q1525), part (2): the kernel suite on one P106-090 (card 0, CUDA_DEVICE_ORDER=
# PCI_BUS_ID), natively, every switch a run does not set at its default. R153's rig driver
# (../../2026-11-13_c831_upstream/README.md, "On a P106") with its steps v4kern, ro8kt and archs on the candidate, plus
# the -moeskip check's archs-off run against the reference build. Started detached on the rig:
#   R=<work folder> REPO=<copy of this repository's scripts> SETS=<folder with v4kern/ and ro8kt/> NEW=<build folder> \
#     REF=<build folder> STEPS='v4kern ro8kt archs off' setsid -f nohup bash rig_suite.sh > console.txt 2>&1 < /dev/null
# Steps, one after another under a lock on card 0:
#   v4kern  scripts/replay_checks.sh, the set v4kern (C796's three kernel suites and the three switches together,
#           recorded with v4-rc1's test-backend-ops for R153; MUL_MAT cases only), NEW's build
#   ro8kt   the same, the set ro8kt (C794's K tiles at 5-8 tokens)
#   archs   check.sh NATIVE=1 RUNS='archs archs-on archs-mask', test-llama-archs -b CUDA0, NEW's build
#   off     the -moeskip gpu_checks.sh run archs-off: REF's test-llama-archs writes every random model's logits on the
#           card, NEW's compares with them (nmse per model; 0 of N values differing means the same bits)
# Env: R, REPO, SETS, NEW, REF (folders holding bin/), STEPS. Output: R/out/ (driver.log, one folder or console per step)
set -uo pipefail
R=${R:?R=<work folder>}
REPO=${REPO:?REPO=<copy of the repository scripts>}
SETS=${SETS:?SETS=<folder with the record sets>}
NEW=${NEW:?NEW=<build folder>}
REF=${REF:?REF=<build folder>}
STEPS=${STEPS:-v4kern ro8kt archs off}
O=$R/out
mkdir -p "$O"
L=$O/driver.log
log() { echo "$(date -Is) $*" | tee -a "$L"; }
want() { [[ " $STEPS " == *" $1 "* ]]; }
gpu() { nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader -i 0 | tr '\n' ' '; }
exec 9> "$R/card0.lock"
flock 9
busy=$(ps -eo comm | grep -x -E 'llama-server|llama-cli|llama-completion|llama-bench|test-backend-ops|test-llama-archs' | sort | uniq -c | tr '\n' ' ')
log "start STEPS='$STEPS' NEW=$NEW REF=$REF card 0: $(gpu); other processes: ${busy:-none}"
rep() { # <out name> <set> <bin>
    local out=$O/$1
    rm -rf "$out"
    env SET="$2" BIN="$3" OUT="$out" DEV=0 RUNS_FILE="$2/runs.tsv" bash "$REPO/scripts/replay_checks.sh" > "$out.console" 2>&1
    log "replay $1 exit $?: $(awk -F'\t' 'NR>1{print $8}' "$out/results.tsv" 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
}
want v4kern && rep v4kern "$SETS/v4kern" "$NEW/bin"
want ro8kt && rep ro8kt "$SETS/ro8kt" "$NEW/bin"
if want archs; then
    NATIVE=1 BIN=$NEW ARCHS=$NEW/bin/test-llama-archs WORK=$O/archs GPU=0 ARCHS_ARGS='-b CUDA0' \
        RUNS='archs archs-on archs-mask' bash "$REPO/docs/data/2026-11-03_upstream_rebase/run/check.sh" > "$O/archs.console" 2>&1
    log "archs exit $?:"; sed 's/^/    /' "$O/archs.console" | tee -a "$L"
fi
if want off; then
    rm -rf "$O/off"
    MODE=native DEV=0 BIN="$NEW/bin" BASE_BIN="$REF/bin" OUT="$O/off" RUNS=archs-off \
        bash "$REPO/docs/data/2026-11-04_moe_skip/run/gpu_checks.sh" > "$O/off.console" 2>&1
    log "off exit $?:"; sed -n '3,$p' "$O/off/console.txt" | sed 's/^/    /' | tee -a "$L"
fi
log "end STEPS='$STEPS' card 0: $(gpu)"
echo "$(date -Is) $STEPS" >> "$O/done"
