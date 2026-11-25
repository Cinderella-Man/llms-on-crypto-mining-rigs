#!/usr/bin/env bash
# land-v5-batch3: one test-backend-ops case run again and again on each build, to tell an intermittent failure from a
# build's own. The case failed once (ERR 0.4995) in the head's MUL_MAT replay with --cpu-check 25 and passed in every
# other replay of both builds. Each repeat is its own test-backend-ops process in its own container, with the fork's
# matrix-vector kernels forced at a P106's shapes (ON, as run.sh's P106_ENV), N runs per build and mode (MODES):
#   replay    the kept record's inputs and references (--replay REC)
#   live      inputs drawn by the test, the reference on this host's processor
#   cpucheck  --replay REC --cpu-check CPU_CHECK: the cases the check picks (about one in CPU_CHECK) run live first, as
#             in the run where the case failed; with CASE matching the failing case's shape in every type, its F16
#             sibling is picked and run live just before it, as there
# Env: W (outputs), BUILDS ("tag=bin-folder ..."), REC (the record's mul_mat folder), GPU (one card), N (40),
#      MODES ("replay live"), CPU_CHECK (25), TAG (a prefix for the log names),
#      CASE (a -p regular expression on the case's parameters), MOUNTS (folders mounted read-only at the same paths), DOCKER_ARGS, JIT (a PTX JIT
#      cache folder), IMG.
set -uo pipefail
: "${W:?}" "${BUILDS:?}" "${REC:?}"
GPU=${GPU:-0}
N=${N:-40}
MODES=${MODES:-replay live}
CPU_CHECK=${CPU_CHECK:-25}
TAG=${TAG:-}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CASE=${CASE:-'^type_a=bf16,type_b=f32,m=1057,n=1,k=128,bs=\[8,1\],nr=\[4,1\],per=\[0,2,1,3\],'} # -p matches the parameters
ON=${ON:-"GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 \
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"}
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$W:$W")
mkdir -p "$W/logs"
[ -n "${JIT:-}" ] && mkdir -p "$JIT"
[ -f "$W/repeats.tsv" ] || printf 'build\tmode\trun\trc\tverdict\terror\tpassed\n' > "$W/repeats.tsv"
for b in $BUILDS; do
    tag=${b%%=*} bin=${b#*=}
    for mode in $MODES; do
        case $mode in
            replay) extra="--replay $REC" ;;
            cpucheck) extra="--replay $REC --cpu-check $CPU_CHECK" ;;
            *) extra="" ;;
        esac
        for i in $(seq 1 "$N"); do
            f=$W/logs/$TAG$tag-$mode-$i.log
            # shellcheck disable=SC2086
            docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" ${JIT:+-e CUDA_CACHE_PATH=$JIT -v $JIT:$JIT} --gpus "\"device=$GPU\"" "$IMG" \
                bash -c "env $ON timeout 300 $bin/test-backend-ops test -b CUDA0 -o MUL_MAT -p '$CASE' $extra > $f 2>&1; echo rc=\$? >> $f" \
                > /dev/null 2>&1
            line=$(sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -a -E 'MUL_MAT\(type_a=bf16.*: (OK|FAIL)' | tail -n 1)
            v=$(grep -oE '(OK|FAIL)$' <<< "$line")
            e=$(grep -oE 'ERR = [0-9.e+-]+' <<< "$line" | cut -d' ' -f3)
            pass=$(grep -a -oE '[0-9]+/[0-9]+ tests passed' "$f" | tail -n 1 | cut -d' ' -f1)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$TAG$tag" "$mode" "$i" "$(grep -a -o 'rc=[0-9]*' "$f" | tail -n 1 | cut -d= -f2)" "${v:--}" "${e:--}" "${pass:--}" >> "$W/repeats.tsv"
        done
        echo "$(date +%H:%M:%S) $TAG$tag $mode: $(awk -F'\t' -v t="$TAG$tag" -v m="$mode" '$1==t && $2==m {c[$5]++} END {for (k in c) printf "%s %d ", k, c[k]}' "$W/repeats.tsv")"
    done
done
date -Is > "$W/DONE"
