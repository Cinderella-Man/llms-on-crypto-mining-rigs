#!/usr/bin/env bash
# R153 (Q1463): the run list of a set recorded with EXTRA_P (scripts/record_checks.sh), cut to the cases the record
# holds. Each run's -p filter must also match EXTRA_P, exactly as the recorder's own filters did, and a run that
# selects none of the set's cases is left out (replay_checks.sh would class it WRONG: no case replayed). The count per
# run comes from the manifest: the case strings whose op is one of the run's -o ops and whose parameters match its
# filter (grep -P; test-backend-ops matches -p against the parameters inside the case string's parentheses).
# Output: SET/NAME (the run list for replay_checks.sh's RUNS_FILE), SET/NAME.counts (run, cases it selects).
# Usage: SET=<a recorded set> EXTRA_P='n=(9|[1-9][0-9]+),' [IN=runs.tsv] [NAME=runs-past8.tsv] bash runs_past8.sh
set -uo pipefail
SET=${SET:?SET=<a recorded set>}
EXTRA_P=${EXTRA_P:?EXTRA_P=<the regex the set was recorded with>}
IN=${IN:-runs.tsv}
NAME=${NAME:-runs-past8.tsv}
[ -f "$SET/manifest.tsv" ] && [ -f "$SET/$IN" ] || { echo "no manifest.tsv or $IN in $SET" >&2; exit 1; }
: > "$SET/$NAME"
: > "$SET/$NAME.counts"
while IFS=$'\t' read -r run envs ops params expect wrap known; do
    [ "$params" = - ] && params=''
    if [ -n "$params" ]; then p="(?=.*($EXTRA_P))(?:.*)($params)"; else p=$EXTRA_P; fi
    ore="^(${ops//,/|})\\("
    n=$(cut -f2 "$SET/manifest.tsv" | grep -P -- "$ore" | sed -E 's/^[A-Z_0-9]+\((.*)\)$/\1/' | grep -c -P -- "$p")
    printf '%s\t%s\n' "$run" "$n" >> "$SET/$NAME.counts"
    [ "$n" -gt 0 ] || continue
    if [ -n "${known:-}" ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$envs" "$ops" "$p" "$expect" "$wrap" "$known"
    else
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$envs" "$ops" "$p" "$expect" "$wrap"
    fi >> "$SET/$NAME"
done < "$SET/$IN"
echo "$SET/$NAME: $(wc -l < "$SET/$NAME") of $(wc -l < "$SET/$IN") runs select recorded cases"
