#!/usr/bin/env bash
# Every kernel's sm_61 SASS in two builds' test-backend-ops, kernel by kernel (C1138): the base build and the branch's.
# Prints how many kernels are the same bytes (instruction offsets stripped), which differ, which are only in one build;
# the branch's kernels of a NEW pattern (default: the fast kernels) are expected to be new. The hand-written PTX kernels
# are not in test-backend-ops' own SASS (they are a separate fatbin in the binary's data): EXTRA_FATBIN, a fatbin file,
# adds its kernels to the branch's list (dumped with cuobjdump -sass). CPU only, in the CUDA container.
#
# Env: WORK (output folder), BASE and HEAD (the two builds' bin/), NEW (an awk regex of the new kernels' names, default
# iq2_xxs_pascal_mtok_fast|mtok_iq2xxs_fast_ptx), EXTRA_FATBIN (optional), DOCKER_ARGS, IMAGE
set -euo pipefail
: "${WORK:?}" "${BASE:?}" "${HEAD:?}"
NEW=${NEW:-iq2_xxs_pascal_mtok_fast|mtok_iq2xxs_fast_ptx}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$WORK"
docker run --rm ${DOCKER_ARGS:-} --network none -v "$WORK:$WORK" -v "$BASE:/base:ro" -v "$HEAD:/head:ro" \
    ${EXTRA_FATBIN:+-v "$EXTRA_FATBIN:/extra.fatbin:ro"} -e W="$WORK" -e NEW="$NEW" "$IMAGE" bash -euo pipefail -c '
split() {  # <sass file> <out dir>: one file per kernel, offsets stripped
    rm -rf "$2"; mkdir -p "$2"
    awk -v out="$2" "/Function : /{ f = \$3; file = out \"/\" f \".sass\" } f { sub(/\\/\\*[0-9a-f]+\\*\\//, \"\"); print > file }" "$1"
}
cuobjdump -sass -arch sm_61 /base/test-backend-ops > $W/base.sass
cuobjdump -sass -arch sm_61 /head/test-backend-ops > $W/head.sass
[ -f /extra.fatbin ] && cuobjdump -sass /extra.fatbin >> $W/head.sass
split $W/base.sass $W/base; split $W/head.sass $W/head
cd $W
same=0; diff=0; newk=0; other_new=0
: > differ.txt; : > new.txt
for f in head/*.sass; do
    n=${f#head/}
    if [ ! -f base/$n ]; then newk=$((newk+1)); echo "$n" >> new.txt; echo "$n" | grep -Eq "$NEW" || other_new=$((other_new+1))
    elif cmp -s $f base/$n; then same=$((same+1)); else diff=$((diff+1)); echo "$n" >> differ.txt; fi
done
gone=$(for f in base/*.sass; do [ -f head/${f#base/} ] || echo x; done | wc -l)
echo "kernels in base $(ls base | wc -l), in head $(ls head | wc -l): the same bytes $same, differing $diff, only in head $newk ($other_new not matching NEW), only in base $gone"
head -n 20 differ.txt
'
