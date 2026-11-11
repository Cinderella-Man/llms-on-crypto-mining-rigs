#!/usr/bin/env bash
# kld_depth.sh: Q1248's check. KL divergence and same top token of one engine line against another at several context
# depths, on a machine whose cards are newer than Pascal: llama-perplexity runs in a container through the build's
# compute 6.1 PTX (the build holds only sm_61 code and compute 6.1 PTX, so a newer card's driver compiles the PTX; OUT/jit
# keeps the result between loads). One container and one model load per arm and depth.
# A depth D is one chunk of the text's first D tokens (-c D --chunks 1), scored on its second half (positions D/2 to
# D-1), as llama-perplexity scores every chunk. The first arm of ARMS writes the reference log-probabilities
# (OUT/base.<D>.kld); every later arm is scored against them on the same tokens.
# An arm is a name with two variables: ARM_UB_<name> (its micro-batch, -ub; default 512) and ARM_ENV_<name> (its
# variables, space-separated VAR=value). Every arm gets PRESET_ENV, GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=$NSM (the sparse
# kernel cuts its lists as on a 5-SM P106-090) and the JIT cache variables. Correctness only: no time is read.
#   BIN=<build>/bin MODEL=<first shard> TEXT=<text file> OUT=<work dir> GPUS=<cards, comma-separated> \
#     MOUNTS="<folders the container reads>" ARMS="a b c" ARM_UB_a=128 ARM_ENV_a="VAR=1" ... \
#     setsid -f nohup bash kld_depth.sh > <log> 2>&1 < /dev/null
# Env: BIN, MODEL, TEXT, OUT, GPUS (docker's device list: indices or UUIDs), ARMS; MOUNTS (mounted read-only at their
#      own paths; OUT is mounted writable); DEPTHS (default "8192 16384 32768 49152"); BATCH (-b, 2048: the Flash-Next
#      preset's); ARGS (-ngl 99 -fa on -fit off --no-host); PRESET_ENV; NSM (5); IMAGE
#      (nvidia/cuda:12.9.2-devel-ubuntu24.04); DOCKER_ARGS (labels, CPU share); CNAME (kld-depth)
# Output: OUT/base.<D>.kld, OUT/<arm>.<D>.log (llama-perplexity's log), OUT/runs.tsv (arm, depth, -ub, exit status,
#         start and end), OUT/done, and OUT/failed if any run failed. Finished runs are skipped when run again.
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${OUT:?}" "${GPUS:?}" "${ARMS:?}"
DEPTHS=${DEPTHS:-8192 16384 32768 49152}
BATCH=${BATCH:-2048}
ARGS=${ARGS:--ngl 99 -fa on -fit off --no-host}
NSM=${NSM:-5}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CNAME=${CNAME:-kld-depth}
mkdir -p "$OUT/jit"
rm -f "$OUT/done" "$OUT/failed"
vols=(-v "$OUT:$OUT")
for m in ${MOUNTS:-}; do vols+=(-v "$m:$m:ro"); done
[ -s "$OUT/runs.tsv" ] || printf 'arm\tdepth\tub\texit\tstart\tend\n' > "$OUT/runs.tsv"
base_arm=${ARMS%% *}
failed=0
for arm in $ARMS; do
    ubv=ARM_UB_$arm envv=ARM_ENV_$arm
    ub=${!ubv:-512} aenv=${!envv:-}
    envs="CUDA_CACHE_PATH=$OUT/jit CUDA_CACHE_MAXSIZE=4294967296 ${PRESET_ENV:-} GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=$NSM $aenv"
    eargs=(); for e in $envs; do eargs+=(-e "$e"); done
    for d in $DEPTHS; do
        base=$OUT/base.$d.kld log=$OUT/$arm.$d.log
        if [ "$arm" = "$base_arm" ]; then
            [ -f "$base" ] && continue
            kargs="--kl-divergence-base $base.partial"
        else
            [ -f "$base" ] || { echo "no base for depth $d: $arm skipped"; failed=1; continue; }
            grep -q "^Mean    KLD" "$log" 2> /dev/null && continue
            kargs="--kl-divergence-base $base --kl-divergence"
        fi
        start=$(date -Is)
        echo "== $arm $d: -ub $ub $aenv $start"
        docker rm -f "$CNAME" > /dev/null 2>&1
        while docker container inspect "$CNAME" > /dev/null 2>&1; do sleep 1; done
        # shellcheck disable=SC2086
        docker run --rm --name "$CNAME" ${DOCKER_ARGS:-} --gpus "\"device=$GPUS\"" "${vols[@]}" "${eargs[@]}" "$IMAGE" \
            "$BIN/llama-perplexity" -m "$MODEL" -f "$TEXT" -c "$d" --chunks 1 -b "$BATCH" -ub "$ub" $ARGS $kargs \
            > "$log" 2>&1 < /dev/null
        rc=$?
        if [ "$arm" = "$base_arm" ]; then
            if [ $rc -eq 0 ] && grep -q "Final estimate" "$log"; then
                mv "$base.partial" "$base"
                echo "   base $d: $(grep "Final estimate" "$log")"
            else
                rm -f "$base.partial"; echo "   FAILED base $d (exit $rc)"; tail -n 5 "$log"; failed=1
            fi
        elif [ $rc -eq 0 ] && grep -q "^Mean    KLD" "$log"; then
            echo "   $(grep -E "^(Mean    KLD|Same top p)" "$log" | tr -s ' ' | tr '\n' ' ')"
        else
            echo "   FAILED $arm $d (exit $rc)"; tail -n 5 "$log"; failed=1
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$d" "$ub" "$rc" "$start" "$(date -Is)" >> "$OUT/runs.tsv"
    done
done
echo "== end $(date -Is)"
[ $failed -eq 0 ] || touch "$OUT/failed"
touch "$OUT/done"
