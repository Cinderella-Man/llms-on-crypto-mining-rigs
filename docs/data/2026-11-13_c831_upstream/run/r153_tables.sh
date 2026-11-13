#!/usr/bin/env bash
# R153 (Q1463): the rig tables of this folder from the rig's output folder ($O: rig_r153's out/), the record logs
# ($LOGS: record_checks.sh's <suite>.log and the drivers' rec-<suite>.out), the comparison tables ($CMP: compare_builds.sh's
# cmp-<set>.tsv) and C831's CPU table (logits-by-model.tsv). Writes into DEST:
#   rig-sets.tsv     per recorded set: cases, bytes, cases linked from another set, record start and end
#   rig-replays.tsv  every replayed run: set, build, run, switches, expect, cases passed, FAIL lines, exit, verdict,
#                    unrecorded cases (the oldset rows: the sets of 2026-11-11 on the tip, before they were recorded again)
#   rig-compare.tsv  part (1): per set and run, the two builds' cases, those in both, the same verdict and errors
#   rig-archs-off.tsv  per random model: the tip's logits against 80c94fe0d's on the card and on the CPU (C831)
#   rig-part2.txt    the consoles of check.sh (archs, moe) and gpu_checks.sh (gpu-a, gpu-b with moeskip's)
#   rig-repeat.tsv   the case held twice, five replays a build (case_repeat.sh)
# Usage: O=<out> LOGS=<dir> CMP=<dir> DEST=<this folder> bash r153_tables.sh
set -uo pipefail
O=${O:?}; LOGS=${LOGS:?}; CMP=${CMP:?}; DEST=${DEST:?}
{
    printf 'set\tcases\tbytes\tlinked\trecorded_from\trecorded_to\n'
    for s in kshape flashnext-moe n2p onecard v4kern ro8kt; do
        f=$LOGS/rec-$s.out
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$s" \
            "$(grep -o 'manifest: [0-9]* cases' "$f" | grep -o '[0-9]*')" \
            "$(grep -o 'cases, [0-9]* bytes' "$f" | grep -o '[0-9]*')" \
            "$(grep -o 'linked [0-9]* of [0-9]* selected cases in all' "$f" | cut -d' ' -f2)" \
            "$(head -n 1 "$f" | cut -d' ' -f1)" "$(grep 'manifest:' "$f" | cut -d' ' -f1)"
    done
} > "$DEST/rig-sets.tsv"
{
    printf 'set\tbuild\trun\tswitches\texpect\tcases\tfail_lines\texit\tverdict\tunrecorded\n'
    for d in "$O"/p1-*-tip "$O"/p1-*-base "$O"/v4kern-tip "$O"/ro8kt-tip "$O"/v4kern-oldset-tip "$O"/ro8kt-oldset-tip; do
        n=$(basename "$d"); b=${n##*-}; s=${n%-*}; s=${s#p1-}
        awk -F'\t' -v s="$s" -v b="$b" 'NR > 1 { print s "\t" b "\t" $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $8 "\t" $10 }' "$d/results.tsv"
    done
} > "$DEST/rig-replays.tsv"
{
    printf 'set\t'; head -n 1 "$CMP/cmp-kshape.tsv"
    for s in kshape flashnext-moe n2p; do awk -v s="$s" 'NR > 1 { print s "\t" $0 }' "$CMP/cmp-$s.tsv"; done
} > "$DEST/rig-compare.tsv"
{
    printf 'arch\tcard_nmse\tcard_max_abs\tcard_differ\tcpu_nmse\tcpu_max_abs\tcpu_differ\n'
    grep -a 'logits against' "$O/gpu-a/moeskip/archs-off.log" \
        | sed -E 's/^([^:]+): logits against [^:]+: nmse ([^,]+), max abs diff ([^,]+), ([0-9]+) of [0-9]+ differ.*/\1\t\2\t\3\t\4/' \
        | awk -F'\t' 'NR == FNR { if (FNR > 1) c[$1] = $2 "\t" $3 "\t" $4; next } { print $0 "\t" c[$1] }' \
            "$DEST/logits-by-model.tsv" -
} > "$DEST/rig-archs-off.tsv"
{
    for f in archs.console moe.console gpu-a/console.txt gpu-a/moeskip/console.txt gpu-b/console.txt gpu-b/moeskip/console.txt; do
        echo "== $f"; sed -E "s#/[^ ]*/opt/llama-builds/#<builds>/#g; s#/mnt/[^ ]*/r153/#<work>/#g" "$O/$f"
    done
} > "$DEST/rig-part2.txt"
{ printf 'build\trun\tcase\tverdict\terrors\n'; cat "$O/repeat-q2_0-n32/repeat.tsv"; } > "$DEST/rig-repeat.tsv"
wc -l "$DEST"/rig-*
