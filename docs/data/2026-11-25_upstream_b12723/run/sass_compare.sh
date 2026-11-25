#!/usr/bin/env bash
# upstream-b12723 (from land-v5-batch3): the compute 6.1 machine code (SASS) of one program in two builds, kernel by kernel. cuobjdump -sass of
# each build's binary in a CUDA container without a card, then sass_compare.py on the host: each kernel's listing (from
# its "Function :" line to the next) hashed, the two builds' kernels matched by name and copy number. Writes
# $W/sass_compare.tsv (one row per kernel family and verdict) and prints the counts.
# Env: W (outputs), BASE and HEAD (bin folders), PROG (test-backend-ops), MOUNTS, DOCKER_ARGS, IMG.
set -uo pipefail
: "${W:?}" "${BASE:?}" "${HEAD:?}"
PROG=${PROG:-test-backend-ops}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
HERE=$(cd "$(dirname "$0")" && pwd)
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
mkdir -p "$W"
VOL+=(-v "$W:$W")
for t in base head; do
    b=$BASE; [ $t = head ] && b=$HEAD
    md5sum "$b/$PROG" > "$W/$t.md5"
    # shellcheck disable=SC2086
    docker run --rm ${DOCKER_ARGS:-} --network none -e NVIDIA_VISIBLE_DEVICES=void "${VOL[@]}" "$IMG" \
        bash -c "nice -n 19 cuobjdump -sass $b/$PROG > $W/$t.sass 2> $W/$t.err; echo rc=\$? > $W/$t.rc" &
done
wait
cat "$W/base.rc" "$W/head.rc"
python3 -I "$HERE/sass_compare.py" "$W/base.sass" "$W/head.sass" "$W/sass_compare.tsv"
rm -f "$W/base.sass" "$W/head.sass"
