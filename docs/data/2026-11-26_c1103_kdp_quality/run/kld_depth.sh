#!/usr/bin/env bash
# kld_depth.sh: C976's driver (Q1707), FN64's (docs/data/2026-11-11_fn64_sparse_prompt_ptx/run/kld_depth.sh) with four
# additions: per-arm arguments (ARM_ARGS_<name>, e.g. the cache types) and binary folder (ARM_BIN_<name>), a chunk count
# (CHUNKS) and an outside reference (BASE_FILE). Everything else is FN64's: KL divergence and same top token of one
# engine line against another at several context depths, on cards newer than Pascal, with llama-perplexity in a
# container running the build's compute 6.1 PTX (the build holds only sm_61 code and compute 6.1 PTX; JIT keeps the
# compiled result between loads). One container and one model load per arm and depth. Correctness only: no time is read.
# A depth D is CHUNKS chunks of D tokens of the text (-c D --chunks CHUNKS), each scored on its second half (positions
# D/2 to D-1), as llama-perplexity scores every chunk. Without BASE_FILE the first arm of ARMS writes the reference
# log-probabilities (OUT/base.<D>.kld) and every later arm is scored against them on the same tokens; with BASE_FILE
# every arm is scored against that file (one depth: the file's own -c and chunks).
# An arm is a name with up to four variables: ARM_UB_<name> (its micro-batch, -ub; default 512), ARM_ENV_<name> (its
# variables, space-separated VAR=value), ARM_ARGS_<name> (extra llama-perplexity arguments) and ARM_BIN_<name> (its
# build's bin folder; default BIN). Every arm gets PRESET_ENV and the JIT cache variables.
#   BIN=<build>/bin MODEL=<gguf> TEXT=<text file> OUT=<work dir> GPUS=<cards, comma-separated> \
#     MOUNTS="<folders the container reads>" ARMS="a b c" ARM_UB_a=512 ARM_ARGS_a="-ctk f16 -ctv f16" ... \
#     setsid -f nohup bash kld_depth.sh > <log> 2>&1 < /dev/null
# Env: BIN, MODEL, TEXT, OUT, GPUS (docker's device list: indices or UUIDs), ARMS; MOUNTS (mounted read-only at their
#      own paths; OUT and JIT are mounted writable); DEPTHS (default "8192 16384 32768 49152"); CHUNKS (1); BASE_FILE;
#      BATCH (-b, 4096: Ornith's v5 preset's); ARGS (-ngl 99 -fa on -fit off --no-host); PRESET_ENV; JIT (OUT/jit);
#      IMAGE (nvidia/cuda:12.9.2-devel-ubuntu24.04); DOCKER_ARGS (labels, CPU share); CNAME (kld-depth)
# Output: OUT/base.<D>.kld, OUT/<arm>.<D>.log (llama-perplexity's log), OUT/runs.tsv (arm, depth, -ub, exit status,
#         start and end), OUT/done, and OUT/failed if any run failed. Finished runs are skipped when run again.
# Every run has GGML_SCHED_DEBUG_REALLOC=$SCHED_DEBUG_REALLOC (default 1; 0 leaves it out), as FN64's since C804.
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${TEXT:?}" "${OUT:?}" "${GPUS:?}" "${ARMS:?}"
DEPTHS=${DEPTHS:-8192 16384 32768 49152}
CHUNKS=${CHUNKS:-1}
BATCH=${BATCH:-4096}
ARGS=${ARGS:--ngl 99 -fa on -fit off --no-host}
JIT=${JIT:-$OUT/jit}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CNAME=${CNAME:-kld-depth}
mkdir -p "$OUT" "$JIT"
rm -f "$OUT/done" "$OUT/failed"
vols=(-v "$OUT:$OUT" -v "$JIT:$JIT")
for m in ${MOUNTS:-}; do vols+=(-v "$m:$m:ro"); done
[ -s "$OUT/runs.tsv" ] || printf 'arm\tdepth\tub\texit\tstart\tend\n' > "$OUT/runs.tsv"
base_arm=${ARMS%% *}
[ -n "${BASE_FILE:-}" ] && base_arm=
failed=0
for arm in $ARMS; do
    ubv=ARM_UB_$arm envv=ARM_ENV_$arm argv=ARM_ARGS_$arm binv=ARM_BIN_$arm
    ub=${!ubv:-512} aenv=${!envv:-} aargs=${!argv:-} abin=${!binv:-$BIN}
    envs="CUDA_CACHE_PATH=$JIT CUDA_CACHE_MAXSIZE=4294967296 ${PRESET_ENV:-} $aenv"
    [ "${SCHED_DEBUG_REALLOC:-1}" = 0 ] || envs="$envs GGML_SCHED_DEBUG_REALLOC=${SCHED_DEBUG_REALLOC:-1}"
    eargs=(); for e in $envs; do eargs+=(-e "$e"); done
    for d in $DEPTHS; do
        base=${BASE_FILE:-$OUT/base.$d.kld} log=$OUT/$arm.$d.log
        if [ "$arm" = "$base_arm" ]; then
            [ -f "$base" ] && continue
            kargs="--kl-divergence-base $base.partial"
        else
            [ -f "$base" ] || { echo "no base for depth $d: $arm skipped"; failed=1; continue; }
            grep -q "^Mean    KLD" "$log" 2> /dev/null && continue
            kargs="--kl-divergence-base $base --kl-divergence"
        fi
        start=$(date -Is)
        echo "== $arm $d: -ub $ub $aargs $aenv $abin $start"
        docker rm -f "$CNAME" > /dev/null 2>&1
        while docker container inspect "$CNAME" > /dev/null 2>&1; do sleep 1; done
        # shellcheck disable=SC2086
        docker run --rm --name "$CNAME" ${DOCKER_ARGS:-} --gpus "\"device=$GPUS\"" "${vols[@]}" "${eargs[@]}" "$IMAGE" \
            "$abin/llama-perplexity" -m "$MODEL" -f "$TEXT" -c "$d" --chunks "$CHUNKS" -b "$BATCH" -ub "$ub" $ARGS \
            $aargs $kargs > "$log" 2>&1 < /dev/null
        rc=$?
        grep -a -q 'unexpected graph reallocation' "$log" \
            && echo "   graph reallocation at an unchanged size in $arm $d: $(grep -a -m1 -o 'unexpected graph reallocation.*' "$log")"
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
