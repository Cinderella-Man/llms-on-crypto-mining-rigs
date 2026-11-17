#!/usr/bin/env bash
# The v5 switch merge (2026-11-17): every GPU kernel of v4.0's llama-server against the merged pascal-p106-v5's,
# compared one kernel at a time on a CPU (cuobjdump in the CUDA image, no GPU).
#   OUT=<folder> MNT=<folder holding both> BASE=<v4.0 llama-server> NEW=<merged llama-server> \
#       [C861=<C861's sass folder>] bash sass_check.sh
# sass_compare.py is C861's (docs/data/2026-11-17_c861a_fp32_scale/run/); copy it into OUT first.
# Out in OUT: md5.txt, h-B.tsv and h-N.tsv (one row per kernel: name, SHA-1 of its SASS with whitespace normalized,
# instruction counts), summary.txt, cmp.identity.tsv, differ.list (base kernels whose SASS differs), size.txt
# (instruction counts of those), kq_on.list (kernels only the new build has: kq_fp32=true), ext-B.txt and ext-N.txt
# (the differing kernels' SASS without addresses and encodings), linediff.txt (each differing line classified),
# c861.txt (the differing kernels against C861's own build, when C861 is given), DONE.
set -u
S=${OUT:?out folder}
B=${BASE:?v4.0 llama-server}
N=${NEW:?merged llama-server}
C861=${C861:-}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
MNT=${MNT:?a folder that holds both binaries, mounted read-only into the container}
dump() { docker run --rm --label pascal-quants=1 ${DOCKER_ARGS:---cpu-shares=256} \
    -v "$MNT:$MNT:ro" "$IMG" cuobjdump -sass "$1" | nice -n 19 c++filt; }
rm -f "$S/DONE"
md5sum "$B" "$N" > "$S/md5.txt"
dump "$B" | nice -n 19 python3 "$S/sass_compare.py" hash > "$S/h-B.tsv"
dump "$N" | nice -n 19 python3 "$S/sass_compare.py" hash > "$S/h-N.tsv"
nice -n 19 python3 "$S/sass_compare.py" compare "$S/h-B.tsv" "$S/h-N.tsv" "$S/cmp" > "$S/summary.txt" 2>&1
awk -F'\t' '$4 != "yes" && NR > 1 {print $1}' "$S/cmp.identity.tsv" > "$S/differ.list"
awk -F'\t' '$2 == "true" {print $1}' "$S/h-N.tsv" | sort > "$S/kq_on.list"
# instruction counts (base build) of the differing kernels, and how many long kernels differ
awk -F'\t' 'FNR == NR {d[$1] = 1; next} FNR > 1 && $2 != "true" {
        n = $4 + 0; if ($1 in d) {k++; if (n > mx) mx = n} if (n > 200) {L++; if ($1 in d) Ld++} }
    END {printf "differing kernels: %d, longest %d instructions; kernels over 200 instructions: %d, differing: %d\n", k, mx, L, Ld}' \
    "$S/differ.list" "$S/h-B.tsv" > "$S/size.txt"
dump "$B" | nice -n 19 python3 "$S/sass_compare.py" extract "$S/differ.list" > "$S/ext-B.txt"
dump "$N" | nice -n 19 python3 "$S/sass_compare.py" extract "$S/differ.list" > "$S/ext-N.txt"
nice -n 19 python3 "$S/sass_compare.py" linediff "$S/ext-B.txt" "$S/ext-N.txt" > "$S/linediff.txt" 2>&1
if [ -n "$C861" ]; then
    {
        echo "differ.list vs C861's differ2.list (sorted): $(diff <(sort "$C861/differ2.list") <(sort "$S/differ.list") > /dev/null && echo same || echo DIFFERENT)"
        echo "ext-N.txt vs C861's ext2-N.txt: $(cmp -s "$C861/ext2-N.txt" "$S/ext-N.txt" && echo identical || echo DIFFERENT)"
        echo "ext-B.txt vs C861's ext2-B.txt: $(cmp -s "$C861/ext2-B.txt" "$S/ext-B.txt" && echo identical || echo DIFFERENT)"
        echo "kq_on.list: $(wc -l < "$S/kq_on.list") kernels; C861's build: $(awk -F'\t' '$2 == "true"' "$C861/h2-N.tsv" | wc -l)"
    } > "$S/c861.txt"
fi
echo done > "$S/DONE"
