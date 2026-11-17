#!/usr/bin/env bash
# The v5 candidate on upstream b12483 (2026-11-17): every GPU kernel of pascal-p106-v5's llama-server (1f1110c67,
# upstream b12410) against the candidate's (pascal-p106-v5-cand-b12483: the same plus upstream b12483 and #32434),
# compared one kernel at a time on a CPU (cuobjdump in the CUDA image, no GPU).
#   OUT=<folder> MNT=<folder holding the binaries> BASE=<1f1110c67's llama-server> NEW=<candidate's llama-server> \
#       bash sass_check.sh
# sass_compare.py is C861's (docs/data/2026-11-17_c861a_fp32_scale/run/), sass_classify.py this folder's; copy both
# into OUT first.
# Out in OUT: md5.txt, h-B.tsv and h-N.tsv (one row per kernel: name, SHA-1 of its SASS with whitespace normalized,
# instruction counts), summary.txt, cmp.identity.tsv, differ.list (base kernels whose SASS differs), missing.list
# (base kernels the new build lacks), only_new.list (kernels only the new build has), ext-B.txt and ext-N.txt (the
# differing kernels' SASS without addresses and encodings), linediff.txt (each differing kernel classified: only
# MOV32I constants differ, i.e. a printed source line or a string address, or other lines differ), kq.txt (C861's
# kq_fp32=true kernels, base against new; the kernel-by-kernel check of these is in the README), h-B-off.tsv (the
# base without its kq_fp32=true kernels), classes.tsv and classes_summary.txt (each differing kernel's family, sizes
# and kind of difference), DONE.
set -u
S=${OUT:?out folder}
B=${BASE:?1f1110c67 llama-server}
N=${NEW:?candidate llama-server}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
MNT=${MNT:?a folder that holds the binaries, mounted read-only into the container}
dump() { docker run --rm --label pascal-quants=1 --network none ${DOCKER_ARGS:---cpu-shares=256} \
    -v "$MNT:$MNT:ro" "$IMG" cuobjdump -sass "$1" | nice -n 19 c++filt; }
rm -f "$S/DONE"
md5sum "$B" "$N" > "$S/md5.txt"
dump "$B" > "$S/dump-B.txt"
dump "$N" > "$S/dump-N.txt"
nice -n 19 python3 "$S/sass_compare.py" hash < "$S/dump-B.txt" > "$S/h-B.tsv"
nice -n 19 python3 "$S/sass_compare.py" hash < "$S/dump-N.txt" > "$S/h-N.tsv"
# the base build carries C861's kq_fp32=true kernels, which compare would count as extra copies of their
# switched-off twins: compare the switched-off kernels only, and the kq_fp32=true ones apart (kq.txt)
awk -F'\t' 'NR == 1 || $2 != "true"' "$S/h-B.tsv" > "$S/h-B-off.tsv"
nice -n 19 python3 "$S/sass_compare.py" compare "$S/h-B-off.tsv" "$S/h-N.tsv" "$S/cmp" > "$S/summary.txt" 2>&1
awk -F'\t' '$4 == "NO" {print $1}' "$S/cmp.identity.tsv" > "$S/differ.list"
awk -F'\t' '$4 == "missing in new" {print $1}' "$S/cmp.identity.tsv" > "$S/missing.list"
awk -F'\t' 'FNR == NR {if (FNR > 1 && $2 != "true") b[$1] = 1; next} FNR > 1 && $2 != "true" && !($1 in b) {print $1}' \
    "$S/h-B.tsv" "$S/h-N.tsv" | sort -u > "$S/only_new.list"
nice -n 19 python3 "$S/sass_compare.py" extract "$S/differ.list" < "$S/dump-B.txt" > "$S/ext-B.txt"
nice -n 19 python3 "$S/sass_compare.py" extract "$S/differ.list" < "$S/dump-N.txt" > "$S/ext-N.txt"
nice -n 19 python3 "$S/sass_compare.py" linediff "$S/ext-B.txt" "$S/ext-N.txt" > "$S/linediff.txt" 2>&1
echo "kq_fp32=true kernels' SHA-1, base against new: $(diff <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-B.tsv" | sort) \
    <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-N.tsv" | sort) > /dev/null && echo identical || echo DIFFERENT)" > "$S/kq.txt"
nice -n 19 python3 "$S/sass_classify.py" "$S/h-B-off.tsv" "$S/h-N.tsv" "$S/ext-B.txt" "$S/ext-N.txt" \
    > "$S/classes.tsv" 2> "$S/classes_summary.txt"
rm -f "$S/dump-B.txt" "$S/dump-N.txt"
echo done > "$S/DONE"
