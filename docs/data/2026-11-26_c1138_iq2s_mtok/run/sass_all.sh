#!/usr/bin/env bash
# Every kernel's sm_61 SASS in two builds' test-backend-ops, kernel by kernel (C1138): the base build and the branch's.
# Prints how many kernels are the same bytes (instruction offsets stripped), which differ, which are only in one build;
# the branch's kernels of a NEW pattern (default: IQ2_S's fast kernels) are expected to be new. The hand-written PTX
# kernels are not in test-backend-ops' own SASS (they are separate fatbins in the binary's data): BASE_EXTRA and
# HEAD_EXTRA, space-separated fatbin files (the build tree's ggml/src/ggml-cuda/ptx/*.fatbin), add their kernels to the
# base's and the branch's lists (dumped with cuobjdump -sass). CPU only, in the CUDA container.
#
# Env: WORK (output folder), BASE and HEAD (the two builds' bin/), NEW (a grep -E regex of the new kernels' names,
# default iq2_s_pascal_mtok_fast|mtok_iq2s_fast_ptx), BASE_EXTRA, HEAD_EXTRA (optional), REUSE=1 (keep WORK's base.sass and
# head.sass from an earlier run instead of dumping again), DOCKER_ARGS, IMAGE
# The hashes leave out the fatbin section headers and blank lines between cubins: the kernel that ends a cubin carries
# the next cubin's header in cuobjdump's text, and which kernel that is changes when a build adds or renames sources.
set -euo pipefail
: "${WORK:?}" "${BASE:?}" "${HEAD:?}"
NEW=${NEW:-iq2_s_pascal_mtok_fast|mtok_iq2s_fast_ptx}
XV=()
i=0
for f in ${BASE_EXTRA:-}; do XV+=(-v "$f:/extra/base$i.fatbin:ro"); i=$((i + 1)); done
i=0
for f in ${HEAD_EXTRA:-}; do XV+=(-v "$f:/extra/head$i.fatbin:ro"); i=$((i + 1)); done
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$WORK"
docker run --rm ${DOCKER_ARGS:-} --network none -v "$WORK:$WORK" -v "$BASE:/base:ro" -v "$HEAD:/head:ro" \
    ${XV[@]+"${XV[@]}"} -e W="$WORK" -e NEW="$NEW" -e REUSE="${REUSE:-0}" "$IMAGE" bash -euo pipefail -c '
# <sass file> <out.tsv>: one line per kernel, its name and the md5 of its instructions (offsets and the fatbin section
# headers between cubins stripped), sorted by name
kernels() {
    awk "/Function : /{ if (f != \"\") print f \"\t\" b; f = \$3; b = \"\"; next } /^(Fatbin|=+\$|arch =|code version|host =|compile_size|compressed|ptxasOptions|identifier|[ \t]*code for sm_|[ \t]*\$)/ { next } f != \"\" { sub(/\\/\\*[0-9a-f]+\\*\\//, \"\"); gsub(/[ \t]+/, \" \"); b = b \"|\" \$0 } END { if (f != \"\") print f \"\t\" b }" "$1" |
        while IFS="$(printf "\t")" read -r name text; do printf "%s\t%s\n" "$name" "$(printf "%s" "$text" | md5sum | cut -c1-32)"; done |
        LC_ALL=C sort -t "$(printf "\t")" -k1,1 > "$2"
}
for t in base head; do
    [ "$REUSE" = 1 ] && [ -s $W/$t.sass ] || cuobjdump -sass -arch sm_61 /$t/test-backend-ops > $W/$t.sass
done
kernels $W/base.sass $W/base.tsv; kernels $W/head.sass $W/head.tsv
for t in base head; do
    for f in /extra/$t*.fatbin; do
        [ -f "$f" ] || continue
        x=$(basename "$f" .fatbin)
        cuobjdump -sass "$f" > $W/$x.sass
        kernels $W/$x.sass $W/$x.tsv
        LC_ALL=C sort -t "$(printf "\t")" -k1,1 $W/$t.tsv $W/$x.tsv -o $W/$t.tsv
    done
done
cd $W
# a kernel name can have several copies (one per source file that instantiates it): compare the copies of each name as a
# set of instruction md5s
for t in base head; do
    LC_ALL=C sort -t "$(printf "\t")" -k1,1 -k2,2 $t.tsv | awk -F "\t" "{ if (\$1 != n) { if (n != \"\") print n \"\t\" c \"\t\" m; n = \$1; m = \$2; c = 1 } else { m = m \",\" \$2; c++ } } END { if (n != \"\") print n \"\t\" c \"\t\" m }" > $t.names.tsv
done
LC_ALL=C join -t "$(printf "\t")" base.names.tsv head.names.tsv > both.names.tsv
same=$(awk -F "\t" "\$2 == \$4 && \$3 == \$5" both.names.tsv | wc -l)
awk -F "\t" "\$2 != \$4 || \$3 != \$5 { print \$1 }" both.names.tsv > differ.txt
LC_ALL=C join -t "$(printf "\t")" -v 2 base.names.tsv head.names.tsv | cut -f1 > new.txt
LC_ALL=C join -t "$(printf "\t")" -v 1 base.names.tsv head.names.tsv | cut -f1 > gone.txt
other_new=$(grep -Evc "$NEW" new.txt || true)
echo "kernel names in base $(wc -l < base.names.tsv) ($(wc -l < base.tsv) copies), in head $(wc -l < head.names.tsv) ($(wc -l < head.tsv) copies): every copy the same bytes $same, differing $(wc -l < differ.txt), only in head $(wc -l < new.txt) ($other_new not matching NEW), only in base $(wc -l < gone.txt)"
head -n 20 differ.txt
'
