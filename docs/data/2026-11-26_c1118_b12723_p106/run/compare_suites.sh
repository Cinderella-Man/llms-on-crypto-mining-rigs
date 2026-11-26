#!/usr/bin/env bash
# C1118's P106 check (Q1851): part (2)'s per-case results against an earlier check's, file by file (standard tools only).
# replay_checks.sh writes one table per run (case id, case, verdict, error values); two checks that replay the same
# record sets on the same kind of card give byte-identical tables when the card computes every case the same.
# Usage: compare_suites.sh LAST THIS   (each a rig_suite.sh out/ folder with v4kern/ and ro8kt/)
# Prints, per set, the run tables compared and those that differ, and whether results.tsv (run, expect, cases, FAIL
# lines, exit, verdict, known fail, unrecorded) and the unrecorded cases are the same.
set -uo pipefail
LAST=${1:?LAST out/ folder}; THIS=${2:?THIS out/ folder}
for s in v4kern ro8kt; do
    n=0; d=0
    for f in "$THIS/$s"/*.tsv; do
        b=$(basename "$f")
        case $b in results.tsv|unrecorded.tsv) continue ;; esac
        n=$((n + 1))
        cmp -s "$f" "$LAST/$s/$b" || { d=$((d + 1)); echo "$s/$b differs"; }
    done
    r=same; diff -q <(cut -f1,3-6,8- "$LAST/$s/results.tsv") <(cut -f1,3-6,8- "$THIS/$s/results.tsv") > /dev/null || r=DIFFERS
    u=same; diff -q <(cut -f2- "$LAST/$s/unrecorded.tsv" | sort -u) <(cut -f2- "$THIS/$s/unrecorded.tsv" | sort -u) > /dev/null || u=DIFFERS
    echo "$s: $n run tables, $d differ; results.tsv $r; unrecorded cases $u"
done
