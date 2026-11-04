#!/usr/bin/env bash
# C372: sets.tsv from replay_checks.sh's console lines (one line per run), for the runs made before replay_checks.sh
# wrote its own results.tsv. Each argument is <build tag>:<suite folder with runs.tsv>:<OUT of replay_checks.sh>.
# Output (stdout): build, set, run, switches, expect, cases passed, FAIL lines, exit status, memcheck errors, verdict
set -uo pipefail
printf 'build\tset\trun\tswitches\texpect\tcases\tfail_lines\texit\tmemcheck_errors\tverdict\n'
for a in "$@"; do
    IFS=: read -r tag set out <<< "$a"
    awk -F'\t' -v tag="$tag" -v set="$(basename "$set")" '
        FNR == NR { if (!($1 in sw)) sw[$1] = ($2 == "-" ? "defaults" : $2); next }
        /^(==|   )/ { next }
        {
            n = split($0, f, " +")
            run = f[1]; v = f[n]; e = f[n - 1]; x = f[n - 3]; fl = f[n - 6]
            c = "none"
            if (match($0, /[0-9]+\/[0-9]+ tests passed/)) c = substr($0, RSTART, RLENGTH - 13)
            else if ($0 ~ /NO CASES RAN/) c = "no cases ran"
            m = "-"
            if (match($0, /ERROR SUMMARY: [0-9]+/)) m = substr($0, RSTART + 15, RLENGTH - 15)
            s = (run in sw) ? sw[run] : "defaults"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", tag, set, run, s, e, c, fl, x, m, v
        }' "$set/runs.tsv" FS=' ' "$out/console.txt"
done
