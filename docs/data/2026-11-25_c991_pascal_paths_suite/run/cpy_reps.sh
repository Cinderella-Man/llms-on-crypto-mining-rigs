#!/usr/bin/env bash
# C991 (Q1724): the quantizing copies of the hop cast's shape (CPY from F32 at [2560,4,512,1]) repeated, with every
# compute-6.1-only path forced (scripts/pascal_forced.sh) and with nothing set, to show which of their cases fail
# changes from run to run (a float landing on a rounding boundary, rounded apart by the card and the processor) whatever
# the switches. None of the forced switches reaches the copy kernels.
# Usage (on the build machine): WORK=<dir> NEW=<bin/ of the rig build> REPO=<this repository> CARD=<card> \
#   [N=5] [MOUNTS=...] [DOCKER_ARGS=...] [IMG=nvidia/cuda:12.9.2-devel-ubuntu24.04] bash cpy_reps.sh
# Output: WORK/cpy/<setting>-<n>.log and WORK/cpy/summary.tsv (setting, repeat, cases, failed, failing cases)
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REPO:?}" "${CARD:?}"
N=${N:-5}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
. "$REPO/scripts/pascal_forced.sh"
o=$WORK/cpy
mkdir -p "$o" "$WORK/jitcache"
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
P='^type_src=f32,.*ne_src=\[2560,4,512,1\]'
printf 'setting\trepeat\tcases\tfailed\tfailing_cases\n' > "$o/summary.tsv"
for i in $(seq 1 "$N"); do
    for s in forced unset; do
        env=""; [ $s = forced ] && env=$PASCAL_FORCED
        # shellcheck disable=SC2086
        docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" -e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296 \
            --gpus "\"device=$CARD\"" "$IMG" bash -c \
            "env $env X=1 nice -n 19 timeout 1800 $NEW/test-backend-ops test -b CUDA0 -o CPY -p '$P' > $o/$s-$i.log 2>&1" > /dev/null 2>&1
        l=$(sed 's/\x1b\[[0-9;]*m//g' "$o/$s-$i.log" | grep -av 'Backend ')
        printf '%s\t%s\t%s\t%s\t%s\n' "$s" "$i" "$(grep -a -c -E ': (OK|FAIL)$' <<< "$l")" "$(grep -a -c ': FAIL$' <<< "$l")" \
            "$(grep -a ': FAIL$' <<< "$l" | sed -E 's/.*type_dst=([a-z0-9_]+).*/\1/' | paste -sd' ')" >> "$o/summary.tsv"
    done
done
cat "$o/summary.tsv"
