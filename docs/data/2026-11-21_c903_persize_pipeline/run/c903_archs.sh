#!/usr/bin/env bash
# C903 (Q1632): every architecture's random model on one CPU, decoded in steps of 1 to 8 tokens (test-llama-archs with
# LLAMA_ARCHS_LOGITS, which also decodes the same tokens as one batch), with the per-size schedulers off, on as before
# C903, on with their pipeline-parallel input copies (LLAMA_SCHED_PER_SIZE_PP=2, the test mode, since one CPU has no
# pipeline parallelism), and the same with Flint's preset switch LLAMA_SCHED_PER_SIZE_RESERVE=2. Each arm's logits are
# compared value for value with the first arm's (the file test-llama-archs writes on its first run).
#
# Then one architecture (VARCH, default Flint's) per arm with LLAMA_UBATCH_TRACE=1 and -v 3 (test-llama-archs prints
# warnings from verbosity 2 on, never libllama's info lines): the change's build logs each per-size scheduler as it is
# made, with its input copies, and the scheduler logs each pass's input copy (sched-trace: copy = N). Here each
# per-size scheduler builds its graph once and reuses it, so only its first copy is used; c903_server.sh's long reply
# crosses a padded length of the cache, where the graphs are built again and take their next copies.
# Env: NEW and BASE, the test-llama-archs binaries of the change and of its base; OUT, the output folder; SEED (1).
# Output: OUT/<arm>.log and OUT/<arm>.RC (rc and seconds), OUT/summary.txt; OUT/v_<arm>.log and OUT/verbose.txt.
set -uo pipefail
NEW=${NEW:?NEW=<test-llama-archs of the change>}
BASE=${BASE:?BASE=<test-llama-archs of the base>}
OUT=${OUT:?OUT=<output folder>}
SEED=${SEED:-1}
VARCH=${VARCH:-flint}
mkdir -p "$OUT/logits"
rm -f "$OUT"/logits/*.f32 "$OUT"/*.log "$OUT"/*.RC "$OUT/summary.txt" "$OUT/verbose.txt"

ARMS="off base_on new_on new_pp0 new_pp2 new_pp2_r"
arm_spec() { # <arm>: the binary and the switches
    case $1 in
        off)       echo "$NEW" ;;
        base_on)   echo "$BASE LLAMA_SCHED_PER_SIZE=1" ;;
        new_on)    echo "$NEW LLAMA_SCHED_PER_SIZE=1" ;;
        new_pp0)   echo "$NEW LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_PP=0" ;;
        new_pp2)   echo "$NEW LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_PP=2" ;;
        new_pp2_r) echo "$NEW LLAMA_SCHED_PER_SIZE=1 LLAMA_SCHED_PER_SIZE_PP=2 LLAMA_SCHED_PER_SIZE_RESERVE=2" ;;
    esac
}

run() { # <log name> <logits prefix> <extra env, or -> <extra args, or -> <arm>
    local name=$1 prefix=$2 xenv=$3 xargs=$4 a=$5
    [ "$xenv" = - ] && xenv=
    [ "$xargs" = - ] && xargs=
    set -- $(arm_spec "$a")
    local bin=$1
    shift
    local s
    s=$(date +%s)
    env LLAMA_ARCHS_LOGITS="$prefix" $xenv "$@" nice -n 19 "$bin" -s "$SEED" $xargs > "$OUT/$name.log" 2>&1
    echo "rc=$? s=$(( $(date +%s) - s ))" > "$OUT/$name.RC"
}

# the first arm writes the reference logits; every later arm compares with them
for a in $ARMS; do run "$a" "$OUT/logits/" - - "$a"; done

{
    echo "arm rc seconds wrote compared differing_logits_files OK_rows FAIL_rows"
    for a in $ARMS; do for n in "$a"; do
        rc=$(sed -n 's/^rc=\([0-9]*\).*/\1/p' "$OUT/$n.RC")
        sec=$(sed -n 's/.* s=\([0-9]*\).*/\1/p' "$OUT/$n.RC")
        wrote=$(grep -ac 'wrote .* logits to' "$OUT/$n.log")
        compared=$(grep -ac 'logits against' "$OUT/$n.log")
        differ=$(grep -a 'logits against' "$OUT/$n.log" | grep -avc ' 0 of [0-9]* differ')
        ok=$(grep -ac 'OK' "$OUT/$n.log")
        fail=$(grep -ac 'FAIL' "$OUT/$n.log")
        echo "$n $rc $sec $wrote $compared $differ $ok $fail"
    done; done
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"

for a in $ARMS; do
    run "v_$a" "$OUT/logits/" LLAMA_UBATCH_TRACE=1 "-a ^$VARCH\$ -v 3" "$a"
done
{
    echo "arm (-a ^$VARCH\$, LLAMA_UBATCH_TRACE=1, -v 3) rc per_size_schedulers_logged with_1_copy with_4_copies passes_traced passes_on_copies_1_to_3 rebuilt_passes"
    for a in $ARMS; do for n in "v_$a"; do
        f=$OUT/$n.log
        echo "$n $(sed -n 's/^rc=\([0-9]*\).*/\1/p' "$OUT/$n.RC") $(grep -ac 'scheduler for [0-9]*-token batches' "$f") $(grep -ac ' 1 input copies' "$f") $(grep -ac ' 4 input copies' "$f") $(grep -ac 'sched-trace: copy = ' "$f") $(grep -ac 'sched-trace: copy = [1-3]' "$f") $(grep -ac 'ubatch-trace: .* graph = rebuilt' "$f")"
    done; done
    echo "logits lines of the traced runs (each against the reference):"
    grep -aH 'logits against' "$OUT"/v_*.log | sed "s#$OUT/##; s#against [^:]*: #: #"
} > "$OUT/verbose.txt"
cat "$OUT/verbose.txt"
