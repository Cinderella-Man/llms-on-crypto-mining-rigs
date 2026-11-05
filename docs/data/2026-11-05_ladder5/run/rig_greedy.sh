#!/usr/bin/env bash
# T0c on one rig: does a build change a model's greedy tokens? llama-completion decodes N tokens greedily (one token a
# pass, so every expert product takes the one-token path the T0c fix touches) from one prompt with every build of BUILDS,
# at every switch's default and the model's own split; the generated text must be the same bytes for every build. Run
# from the repository root on a machine that only runs ssh; every rig command runs under the RAM watchdog
# (scripts/ram_watchdog.sh), as rig_ladder5.sh.
# Env: RIG (ssh alias), MODEL (the first shard on the rig), BUILDS ('<folder under /opt/llama-builds> ...': the first is
#   the base), ARGS (the model's split, e.g. '-ngl 99 -fa on -ts 6,6,5,5,6,5,5,6,6'), DIR (a folder on the rig),
#   OUT (a local folder), TAG (a name for the model), N (256), CTX (4096), PROMPT (a coding request)
# Output: OUT/greedy-<TAG>.tsv (build, md5 of the generated text, its bytes, the eval line), OUT/<TAG>-<build>.{out,err},
#   OUT/rw_*.tsv (the watchdog's host-memory log); the last line says SAME or DIFFERS
set -uo pipefail
: "${RIG:?}" "${MODEL:?}" "${BUILDS:?}" "${ARGS:?}" "${DIR:?}" "${OUT:?}" "${TAG:?}"
N=${N:-256}
CTX=${CTX:-4096}
PROMPT=${PROMPT:-Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest tests for it.}
mkdir -p "$OUT"
T=$OUT/greedy-$TAG.tsv
printf 'build\tmd5\tbytes\teval\n' > "$T"
for b in $BUILDS; do
    n=$TAG-$(echo "$b" | cut -d- -f3)
    ssh "$RIG" "mkdir -p $DIR && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
    ssh "$RIG" "LOG=/dev/shm/rw_$n.tsv PROC=llama-completion MIN_AVAIL_MB=60 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw.pid
        cd $DIR && timeout 3600 /opt/llama-builds/$b/bin/llama-completion -m $MODEL $ARGS -c $CTX -n $N --temp 0 --seed 1 \
            -no-cnv --no-display-prompt -p '$PROMPT' > $n.out 2> $n.err < /dev/null; echo \"$n rc=\$?\"
        kill \$(cat /dev/shm/rw.pid); mv /dev/shm/rw_$n.tsv $DIR/; rm -f /dev/shm/rw.pid /dev/shm/ram_watchdog.sh"
    for f in "$n.out" "$n.err" "rw_$n.tsv"; do ssh "$RIG" "cat $DIR/$f" > "$OUT/$f"; done
    printf '%s\t%s\t%s\t%s\n' "$b" "$(md5sum < "$OUT/$n.out" | cut -d' ' -f1)" "$(wc -c < "$OUT/$n.out")" \
        "$(grep -a ' eval time' "$OUT/$n.err" | grep -av prompt | tail -n 1 | sed 's/.*eval time *= *//')" >> "$T"
done
ssh "$RIG" "rm -f $DIR/$TAG-*.out $DIR/$TAG-*.err $DIR/rw_$TAG-*.tsv"
nm=$(tail -n +2 "$T" | cut -f2 | sort -u | wc -l)
echo "greedy $TAG: $([ "$nm" = 1 ] && [ "$(tail -n +2 "$T" | awk -F'\t' '$3 > 0' | wc -l)" -gt 0 ] && echo SAME || echo DIFFERS) ($nm distinct md5 over $(echo $BUILDS | wc -w) builds)"
