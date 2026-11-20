#!/usr/bin/env bash
# C881 (Q1590): test-llama-archs --kv-seq-rm (C740's check) over the arms of one build, each run writing its trace
# (LLAMA_ARCHS_SEQ_RM_TRACE) so that two builds can be compared line by line. After C740's run/bound_loop.sh: a test
# that aborts the process is run again for its architecture alone with the switch off in both contexts
# (LLAMA_ARCHS_SEQ_RM_BOUND_ON=0); when it aborts at the same test there too, the abort is the architecture's, the test
# is left out (LLAMA_ARCHS_SKIP) and the arm runs again from the start. Each abort gets its own verdict line.
# Usage: seqrm_loop.sh <bin dir> <out dir> <tree> '<arms>'
#   arms  '<name>@<architecture regex, empty: all>@<seed>@<VAR=value>[,<VAR=value>...]@<skip regex, empty: none> ...'
#         an arm whose name starts with "control" must fail (the switch's test hook =2)
# Output: <out dir>/<tree>-<arm>.log, .trace, .skip (the final LLAMA_ARCHS_SKIP), .abortN and .baseN of each abort;
#   verdict lines (check, verdict, detail) in <out dir>/verdicts-<tree>.tsv
set -uo pipefail
bin=$1 out=$2 tree=$3 arms=$4
V=$out/verdicts-$tree.tsv
mkdir -p "$out"
: > "$V"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$V"; }

# one <log> <filter> <seed> <skip regex> <trace file or empty> <VAR=value ...>: rc
one() {
    local a=()
    [ -n "$2" ] && a=(-a "$2")
    # shellcheck disable=SC2086
    env $6 LLAMA_ARCHS_SKIP="$4" ${5:+LLAMA_ARCHS_SEQ_RM_TRACE=$5} nice -n 19 timeout 5400 \
        "$bin/test-llama-archs" --kv-seq-rm "${a[@]}" -s "$3" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$1"
    return "${PIPESTATUS[0]}"
}

last_test() { grep -a '^kv-seq-rm start: ' "$1" | tail -n 1 | sed 's/^kv-seq-rm start: //'; }

set -f  # the arms hold regexes
for a in $arms; do
    IFS=@ read -r name filter seed envs skip <<< "$a"
    envs=${envs//,/ }
    skip=${skip:-^\$}
    log=$out/$tree-$name.log
    n_abort=0
    s0=$(date +%s)
    while true; do
        one "$log" "$filter" "$seed" "$skip" "$log.trace" "$envs"
        rc=$?
        t=$(last_test "$log")
        if [ "$rc" -lt 128 ] || [ -z "$t" ] || [ "$n_abort" -ge 20 ]; then
            break
        fi
        n_abort=$((n_abort + 1))
        msg=$(grep -a -m 1 -E 'GGML_ASSERT|GGML_ABORT|: [a-z].*(failed|cannot|error)' "$log" | sed 's/^.*src\/src\///; s/^.*src\/ggml\///')
        mv "$log" "$log.abort$n_abort"
        arch=${t%% *}
        one "$log.base$n_abort" "^$arch\$" "$seed" "$skip" "" "$envs LLAMA_ARCHS_SEQ_RM_BOUND_ON=0"
        rc0=$?
        t0=$(last_test "$log.base$n_abort")
        verdict "seqrm-$tree-$name-abort$n_abort" "$([ "$rc0" -ge 128 ] && [ "$t0" = "$t" ] && echo BASE-ABORT || echo ABORT)" \
            "$t: rc=$rc ($msg); the switch off in both contexts: rc=$rc0, last test: $t0"
        skip="$skip|^$t\$"
    done
    echo "$skip" > "$log.skip"
    sum=$(grep -a -m 1 '^Summary' "$log")
    st=$(grep -a -m 1 '^seq_rm with the switch' "$log")
    if [[ $name == control* ]]; then
        v=$([ "$rc" != 0 ] && grep -aq 'FAIL edit' "$log" && echo AS-EXPECTED || echo UNEXPECTED)
    else
        v=$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)
    fi
    verdict "seqrm-$tree-$name" "$v" "rc=$rc; ${filter:-every architecture}; seed $seed; ${envs:-no switch}; $sum; ${st#seq_rm with the switch, all tests: }; skip $skip; $(( $(date +%s) - s0 )) s"
done
