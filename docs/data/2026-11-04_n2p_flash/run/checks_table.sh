#!/usr/bin/env bash
# gpu_checks.sh's output folders to two tables: per check and run the cases passed, run and not supported with the
# run's exit status, and every failing case by name with its error and the limit it missed.
#   bash checks_table.sh <folder for the tables> <check>=<OUT of gpu_checks.sh> ...
# Output: <folder>/gpu_checks.tsv (check, run, switch, passed, cases, not_supported, failed, exit) and
#   <folder>/gpu_check_failures.tsv (check, run, case, err, limit)
set -euo pipefail
DIR=${1:?folder for the tables}
shift
mkdir -p "$DIR"
plain() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
T=$(printf '\t')
printf 'check\trun\tswitch\tpassed\tcases\tnot_supported\tfailed\texit\n' > "$DIR/gpu_checks.tsv"
printf 'check\trun\tcase\terr\tlimit\n' > "$DIR/gpu_check_failures.tsv"
for arg in "$@"; do
    check=${arg%%=*} out=${arg#*=}
    # console.txt: one line per run, the run in columns 1-12 and its switch in columns 14-57
    grep -a -E '^[a-z0-9-]+ +.* exit [0-9]+' "$out/console.txt" | while IFS= read -r line; do
        run=${line%% *}
        what=$(printf '%s' "${line:13:44}" | sed -E 's/ +$//')
        log=$out/$run.log
        read -r passed cases < <(plain "$log" | grep -a -o -E '[0-9]+/[0-9]+ tests passed' | head -n 1 | tr '/' ' ' | cut -d' ' -f1,2)
        ns=$(plain "$log" | grep -a -c 'not supported' || true)
        rc=$(tail -n 1 "$log" | sed -E 's/^exit //')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$check" "$run" "$what" "$passed" "$cases" "$ns" "$((cases - passed))" "$rc" >> "$DIR/gpu_checks.tsv"
        plain "$log" | grep -a -E '^\[[A-Z_]+\] (ERR|NMSE) = ' |
            sed -E "s/^\[[A-Z_]+\] [A-Z]+ = ([0-9.e+-]+) > ([0-9.e+-]+) +(.*): FAIL.*/$check$T$run$T\3$T\1$T\2/" >> "$DIR/gpu_check_failures.tsv" || true
    done
done
