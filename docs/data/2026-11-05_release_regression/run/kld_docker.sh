#!/usr/bin/env bash
# The method of kld_builds.sh on a build machine's GPUs instead of the rig: llama-perplexity on TEXT (CHUNKS chunks of
# CTX tokens), the reference build writing its next-token distributions (--kl-divergence-base), every other build
# scored against them (--kl-divergence). The rig builds carry compute 6.1 PTX, which the driver compiles for the card
# at the first launch (the compiled kernels kept in WORK/cache). One container per build, labelled, removed at exit.
# Runs on the build machine itself.
# Env: MODEL, BUILDS ('<build folder with bin/> ...': the first is the reference), ARGS (e.g. '-ngl 44 -ts 1,1 -b 2048
#   -ub 512'), TEXT, WORK (a scratch folder with room for the distributions, ~n_vocab x 2 bytes a scored token), GPUS
#   (Docker's device list: UUIDs, comma-separated); optional CTX (512), CHUNKS (16), IMAGE
#   (nvidia/cuda:12.9.2-devel-ubuntu24.04), MOUNTS (more "-v a:a" for the model's and text's drives), LABEL
#   (pascal-quants=1), CPU_SHARES (256), REUSE_BASE (1: keep the reference's finished run)
# Output in WORK: kld-<build tag>.out/.err, kld.tsv (build, mean KLD, same top token, PPL), KLD_DONE
set -uo pipefail
: "${MODEL:?}" "${BUILDS:?}" "${ARGS:?}" "${TEXT:?}" "${WORK:?}" "${GPUS:?}"
CTX=${CTX:-512}; CHUNKS=${CHUNKS:-16}; IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$WORK/cache"; rm -f "$WORK/KLD_DONE"
ref=""
for b in $BUILDS; do
    t=$(basename "$b"); t=${t##*-pascal-p106-}
    if [ -z "$ref" ]; then
        ref=$t; k="--kl-divergence-base $WORK/base.kld"
        # REUSE_BASE=1: the reference's finished run (its base.kld and a final estimate in its log) is kept
        if [ -n "${REUSE_BASE:-}" ] && [ -s "$WORK/base.kld" ] && grep -a -q 'Final estimate' "$WORK/kld-$t.err" "$WORK/kld-$t.out"; then
            echo "=== kld $t: reusing its base.kld"; continue
        fi
    else k="--kl-divergence-base $WORK/base.kld --kl-divergence"; fi
    echo "=== kld $t $(date -Is)"
    # shellcheck disable=SC2086
    docker run --rm --name "t23kld-$t" --label "${LABEL:-pascal-quants=1}" --cpu-shares "${CPU_SHARES:-256}" \
        --gpus "\"device=$GPUS\"" -e CUDA_CACHE_PATH="$WORK/cache" -e CUDA_CACHE_MAXSIZE=4294967296 \
        -v "$WORK:$WORK" -v "$b:$b:ro" ${MOUNTS:-} "$IMAGE" \
        "$b/bin/llama-perplexity" -m "$MODEL" -fa on $ARGS -f "$TEXT" -c "$CTX" --chunks "$CHUNKS" $k \
        > "$WORK/kld-$t.out" 2> "$WORK/kld-$t.err"
    echo "kld $t rc=$?"
done
{
    printf 'build\tagainst\tmean KLD\tsame top p\tPPL\n'
    for b in $BUILDS; do
        t=$(basename "$b"); t=${t##*-pascal-p106-}; f="$WORK/kld-$t.err"
        printf '%s\t%s\t%s\t%s\t%s\n' "$t" "$ref" "$(grep -a -h -m1 '^Mean    KLD:' "$f" "$WORK/kld-$t.out" | cut -d: -f2- | xargs)" \
            "$(grep -a -h -m1 '^Same top p:' "$f" "$WORK/kld-$t.out" | cut -d: -f2- | xargs)" \
            "$(grep -a -h 'Final estimate' "$f" "$WORK/kld-$t.out" | tail -n 1 | cut -d: -f2- | xargs)"
    done
} > "$WORK/kld.tsv"
rm -f "$WORK/base.kld"
date -Is > "$WORK/KLD_DONE"
cat "$WORK/kld.tsv"
