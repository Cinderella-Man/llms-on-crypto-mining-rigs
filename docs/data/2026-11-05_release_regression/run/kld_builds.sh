#!/usr/bin/env bash
# Release tests T2 and T3 (2026-11-05), the size of a greedy difference between engine builds: llama-perplexity on
# TEXT (CHUNKS chunks of CTX tokens) with the reference build writing its next-token distributions
# (--kl-divergence-base), then every other build scored against them (--kl-divergence: mean KLD, same top token,
# perplexity). Each run on the rig under scripts/ram_watchdog.sh. Run from the repository root on a machine that only
# runs ssh, with nothing else on the rig.
# Env: RIG, MODEL (on the rig), BUILDS ('<folder under /opt/llama-builds> ...': the first is the reference), ARGS (the
#   layer split and batch flags, e.g. '-b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6'), TEXT (on the rig), DIR (a rig folder
#   with room for the distributions, ~n_vocab x 2 bytes a scored token), OUT (local); optional CTX (512), CHUNKS (16),
#   BASE (a base file already on the rig, written with the same CTX on another machine, e.g. a Q8_0 reference: every
#   build is then scored against it and none writes one; a BASE inside DIR is removed with DIR at the end)
# Output: OUT/kld-<build tag>.out/.err (the tool's output), OUT/kld.tsv (build, mean KLD, same top token, PPL lines)
set -uo pipefail
: "${RIG:?}" "${MODEL:?}" "${BUILDS:?}" "${ARGS:?}" "${TEXT:?}" "${DIR:?}" "${OUT:?}"
CTX=${CTX:-512}; CHUNKS=${CHUNKS:-16}
mkdir -p "$OUT"
ref=""
for b in $BUILDS; do
    t=${b##*-pascal-p106-}
    if [ -n "${BASE:-}" ]; then [ -n "$ref" ] || ref=$(basename "$BASE"); k="--kl-divergence-base $BASE --kl-divergence"
    elif [ -z "$ref" ]; then ref=$t; k="--kl-divergence-base $DIR/base.kld"; else k="--kl-divergence-base $DIR/base.kld --kl-divergence"; fi
    echo "=== kld $t $(date -Is)"
    ssh "$RIG" "mkdir -p $DIR && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
    ssh "$RIG" "LOG=/dev/shm/rw_kld.tsv PROC=llama-perplexity MIN_AVAIL_MB=200 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw.pid
        cd $DIR && timeout 3600 /opt/llama-builds/$b/bin/llama-perplexity -m $MODEL -ngl 99 -fa on $ARGS -f $TEXT -c $CTX \
            --chunks $CHUNKS $k > kld-$t.out 2> kld-$t.err; echo \"kld $t rc=\$?\"
        kill \$(cat /dev/shm/rw.pid); mv /dev/shm/rw_kld.tsv $DIR/rw_kld-$t.tsv; rm -f /dev/shm/rw.pid /dev/shm/ram_watchdog.sh"
    for f in kld-$t.out kld-$t.err rw_kld-$t.tsv; do ssh "$RIG" "cat $DIR/$f" > "$OUT/$f"; done
done
{
    printf 'build\tagainst\tmean KLD\tsame top p\tPPL\n'
    for b in $BUILDS; do
        t=${b##*-pascal-p106-}; f="$OUT/kld-$t.err"
        printf '%s\t%s\t%s\t%s\t%s\n' "$t" "$ref" "$(grep -a -m1 '^Mean    KLD:' "$f" "$OUT/kld-$t.out" | cut -d: -f3- | xargs)" \
            "$(grep -a -m1 '^Same top p:' "$f" "$OUT/kld-$t.out" | cut -d: -f3- | xargs)" \
            "$( { grep -a -h -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$f" "$OUT/kld-$t.out" | sed 's/.*PPL = //'
                  grep -a -h -E '^Mean PPL\(Q\) +:' "$f" "$OUT/kld-$t.out" | cut -d: -f2- | xargs; } | tail -n 1)"
    done
} > "$OUT/kld.tsv"
ssh "$RIG" "rm -rf $DIR"
cat "$OUT/kld.tsv"
