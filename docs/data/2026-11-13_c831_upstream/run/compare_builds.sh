#!/usr/bin/env bash
# R153 (Q1463): two builds' replays of one record compared case by case. TIP and BASE are replay_checks.sh output
# folders of the v4-rc1 build (on a set) and of 80c94fe0d (on the same set under its own names, alias_base.py); each
# run's <run>.tsv lists per case its id, its name, its verdict and every node's error as printed. Cases are joined by
# id through ALIAS (alias_base.py's alias.tsv: the v4-rc1 build's id, the base's id), which also keeps apart the cases
# a test list holds twice under one name (their keys end in #0 and #1, and their stored inputs differ); NOREC lines are
# left out. Per run: the cases each build replayed, the cases in both, those with the same verdict, those with the same
# errors (every node's error the same text, so the same value as printed), and each build's failures.
# Usage: TIP=<dir> BASE=<dir> ALIAS=<alias.tsv> [DIFF=<file for the cases whose errors differ>] bash compare_builds.sh
set -uo pipefail
TIP=${TIP:?TIP=<replay output of the v4-rc1 build>}
BASE=${BASE:?BASE=<replay output of 80c94fe0d>}
ALIAS=${ALIAS:?ALIAS=<alias.tsv of the base set>}
DIFF=${DIFF:-/dev/null}
[ "$DIFF" = /dev/null ] || : > "$DIFF"
printf 'run\tcases_tip\tcases_base\tin_both\tsame_verdict\tsame_errors\tfailed_tip\tfailed_base\n'
for t in "$TIP"/*.tsv; do
    r=$(basename "$t" .tsv)
    case $r in results|unrecorded) continue ;; esac
    b=$BASE/$r.tsv
    if [ ! -f "$b" ]; then printf '%s\tno run of the base\n' "$r"; continue; fi
    awk -F'\t' -v run="$r" -v diff="$DIFF" '
        FILENAME == ARGV[1] { base_of[$1] = $2; next }
        FILENAME == ARGV[2] { if ($3 == "NOREC") next; nb++; if ($3 != "OK") fb++; bv[$1] = $3; be[$1] = $4; next }
        {
            if ($3 == "NOREC") next
            nt++; if ($3 != "OK") ft++
            if (!($1 in base_of) || !(base_of[$1] in bv)) next
            k = base_of[$1]; j++; if (bv[k] == $3) sv++
            if (be[k] == $4) se++; else print run "\t" $2 "\t" $3 "\t" $4 "\t" bv[k] "\t" be[k] >> diff
        }
        END { printf "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", run, nt, nb, j, sv, se, ft, fb }' "$ALIAS" "$b" "$t"
done
