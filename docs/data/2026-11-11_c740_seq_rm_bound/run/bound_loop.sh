#!/usr/bin/env bash
# C740 (Q1309): test-llama-archs --kv-seq-rm over every arm and seed, in one go (checks.sh runs it in one container).
# A test that aborts the process is run again for its architecture alone with the switch off in both contexts
# (LLAMA_ARCHS_SEQ_RM_BOUND_ON=0): when it aborts at the same test there too, the abort is the architecture's (a limit
# of its memory or graph that the random edits reach), not the switch's; the test is then left out (LLAMA_ARCHS_SKIP)
# and the arm runs again. Each abort gets its own verdict line.
# Usage: bound_loop.sh <bin dir> <out dir> <tag> '<arms>' [VAR=value]
#   arms   '<name>@<architecture regex>@<seeds, comma-separated>@<VAR=value>[,<VAR=value>...] ...'
#   VAR=value  added to every run of the arms (the control: LLAMA_ARCHS_SEQ_RM_BOUND_ON=2, which must be caught)
# Output: <out dir>/<tag>-<arm>-s<seed>.log (and .abortN, .baseN of each abort), verdict lines (check, verdict, detail)
#   in <out dir>/verdicts-<tag>.tsv
set -uo pipefail
bin=$1 out=$2 b=$3 arms=$4 extra=${5:-}
V=$out/verdicts-$b.tsv
mkdir -p "$out"
: > "$V"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$V"; }

# one <log> <filter> <seed> <skip regex> <VAR=value ...>: rc
one() {
    # shellcheck disable=SC2086
    env $5 LLAMA_ARCHS_SKIP="$4" nice -n 19 timeout 5400 "$bin/test-llama-archs" --kv-seq-rm -a "$2" -s "$3" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' > "$1"
    return "${PIPESTATUS[0]}"
}

last_test() { grep -a '^kv-seq-rm start: ' "$1" | tail -n 1 | sed 's/^kv-seq-rm start: //'; }

for a in $arms; do
    IFS=@ read -r name filter seeds envs <<< "$a"
    envs=${envs//,/ }
    for s in ${seeds//,/ }; do
        log=$out/$b-$name-s$s.log
        skip='^$'
        n_abort=0
        while true; do
            one "$log" "$filter" "$s" "$skip" "$envs $extra"
            rc=$?
            t=$(last_test "$log")
            if [ "$rc" -lt 128 ] || [ -z "$t" ] || [ "$n_abort" -ge 20 ]; then
                break
            fi
            n_abort=$((n_abort + 1))
            msg=$(grep -a -m 1 -E 'GGML_ASSERT|GGML_ABORT|: [a-z].*(failed|cannot|error)' "$log" | sed 's/^.*src\/src\///; s/^.*src\/ggml\///')
            mv "$log" "$log.abort$n_abort"
            arch=${t%% *}
            one "$log.base$n_abort" "^$arch\$" "$s" "$skip" "$envs LLAMA_ARCHS_SEQ_RM_BOUND_ON=0"
            rc0=$?
            t0=$(last_test "$log.base$n_abort")
            verdict "bound-$b-$name-s$s-abort$n_abort" "$([ "$rc0" -ge 128 ] && [ "$t0" = "$t" ] && echo BASE-ABORT || echo ABORT)" \
                "$t: rc=$rc ($msg); the switch off in both contexts: rc=$rc0, last test: $t0"
            skip="$skip|^$t\$"
        done
        sum=$(grep -a -m 1 '^Summary' "$log")
        st=$(grep -a -m 1 '^seq_rm with the switch' "$log")
        if [ -n "$extra" ]; then
            # the control: the hook must make some test fail on a difference
            v=$([ "$rc" != 0 ] && grep -aq 'FAIL edit' "$log" && echo AS-EXPECTED || echo UNEXPECTED)
        else
            v=$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)
        fi
        verdict "bound-$b-$name-s$s" "$v" "rc=$rc; -a $filter; ${envs:-no switch} $extra; $sum; ${st#seq_rm with the switch, all tests: }"
    done
done
