#!/usr/bin/env bash
# C28 (Q548, Q448) and C22 (Q440) on the rig, single requests: Q215's server line (fn_line.sh with MODE=speed: prompts of
# ~1k, ~8k and ~32k tokens, then restore32k, a follow-up that changes the 32k prompt's last ~6,000 characters, ~1,500
# tokens) once per checkpoint setting, one after the other, run ON the rig and detached. The release line re-reads
# restore32k from its 42-token prefix checkpoint (Q215: 209 s to the first token); with checkpoints inside the prompt it
# resumes from the last one before the change. Every line logs its checkpoint saves and restores with their times
# (-lv 4), and with --ctx-checkpoints-vram the stores it allocates on each card.
# Lines (LINES: "label|extra server arguments", separated by ";"), by default:
#   ck0   this build with no switch (the release line's checkpoints, on the host)
#   cks8  --ctx-checkpoints-spread 8192 --ctx-checkpoints-vram 4
#   cks4  --ctx-checkpoints-spread 4096 --ctx-checkpoints-vram 4
# Usage, on the rig: FN=<folder with fn_line.sh and fn_requests.py> BUILD=<build folder> BUILDS=<builds folder> \
#   MODEL=<first shard> OUT=<work dir> [ARGS='<the server line>'] [LINES=...] [CTX=65536] \
#   setsid -f nohup bash history_edit.sh > <work dir>/history_edit.out 2>&1 < /dev/null
# Done when history_edit.out ends with "== history_edit done"; per line OUT/line-<label>.out (fn_line.sh's report and
# fn_requests.py's JSON summary) and OUT/server-<label>.log.
set -uo pipefail
: "${FN:?}" "${BUILD:?}" "${BUILDS:?}" "${MODEL:?}" "${OUT:?}"
ARGS=${ARGS:--b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio}
LINES=${LINES:-ck0|;cks8|--ctx-checkpoints-spread 8192 --ctx-checkpoints-vram 4;cks4|--ctx-checkpoints-spread 4096 --ctx-checkpoints-vram 4}
IFS=';' read -ra L <<< "$LINES"
for l in "${L[@]}"; do
    label=${l%%|*}
    extra=${l#*|}
    LABEL=$label BUILD=$BUILD BUILDS=$BUILDS MODEL=$MODEL CTX=${CTX:-65536} ARGS="$ARGS -lv 4 $extra" MODE=speed OUT=$OUT \
        bash "$FN/fn_line.sh" > "$OUT/line-$label.out" 2>&1 < /dev/null
    echo "== $label $(date -Is): $(grep -c 'created context checkpoint' "$OUT/server-$label.log") saves," \
        "$(grep -c 'restored context checkpoint' "$OUT/server-$label.log") restores," \
        "$(grep -c 'allocated a state store' "$OUT/server-$label.log") stores allocated"
    grep -E 'restored context checkpoint|allocated a state store' "$OUT/server-$label.log" | cut -c1-220
done
pkill -x llama-server
echo "== history_edit done $(date -Is)"
