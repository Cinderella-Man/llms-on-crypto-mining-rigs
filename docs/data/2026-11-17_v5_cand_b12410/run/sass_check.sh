#!/usr/bin/env bash
# The v5 candidate on upstream b12410 (2026-11-17): every GPU kernel of C854's llama-server (upstream b12410 on v4.0,
# pascal-p106-v5-b12410-up) against the candidate's (pascal-p106-v5-cand-b12410: the same plus the v5 switches),
# compared one kernel at a time on a CPU (cuobjdump in the CUDA image, no GPU).
#   OUT=<folder> MNT=<folder holding the binaries> BASE=<C854's llama-server> NEW=<candidate's llama-server> \
#       [MERGE=<the v5 switch merge's sass folder>] bash sass_check.sh
# sass_compare.py is C861's (docs/data/2026-11-17_c861a_fp32_scale/run/); copy it into OUT first. MERGE is the OUT
# folder of docs/data/2026-11-17_v5_merge_switches/run/sass_check.sh (v4.0 against the switches on v4.0).
# Out in OUT: md5.txt, h-B.tsv and h-N.tsv (one row per kernel: name, SHA-1 of its SASS with whitespace normalized,
# instruction counts), summary.txt, cmp.identity.tsv, cmp.kq.tsv, differ.list (base kernels whose SASS differs),
# size.txt, kq_on.list (kernels only the new build has: kq_fp32=true), ext-B.txt and ext-N.txt (the differing
# kernels' SASS without addresses and encodings), linediff.txt (each differing line classified), and with MERGE:
# merge.txt and linediff-merge.txt (the differing kernels and the kq_fp32 kernels against the switch merge's build),
# DONE.
set -u
S=${OUT:?out folder}
B=${BASE:?C854 llama-server}
N=${NEW:?candidate llama-server}
MERGE=${MERGE:-}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
MNT=${MNT:?a folder that holds the binaries, mounted read-only into the container}
dump() { docker run --rm --label pascal-quants=1 --network none ${DOCKER_ARGS:---cpu-shares=256} \
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
if [ -n "$MERGE" ]; then
    nice -n 19 python3 "$S/sass_compare.py" linediff "$MERGE/ext-N.txt" "$S/ext-N.txt" > "$S/linediff-merge.txt" 2>&1
    {
        echo "differ.list vs the merge's differ.list (sorted): $(diff <(sort "$MERGE/differ.list") <(sort "$S/differ.list") > /dev/null && echo same || echo DIFFERENT)"
        echo "ext-N.txt vs the merge's ext-N.txt: $(cmp -s "$MERGE/ext-N.txt" "$S/ext-N.txt" && echo identical || echo DIFFERENT)"
        echo "ext-B.txt (C854) vs the merge's ext-B.txt (v4.0): $(cmp -s "$MERGE/ext-B.txt" "$S/ext-B.txt" && echo identical || echo DIFFERENT)"
        echo "kq_on.list vs the merge's kq_on.list: $(cmp -s "$MERGE/kq_on.list" "$S/kq_on.list" && echo same || echo DIFFERENT) ($(wc -l < "$S/kq_on.list") kernels)"
        echo "cmp.kq.tsv vs the merge's cmp.kq.tsv: $(cmp -s "$MERGE/cmp.kq.tsv" "$S/cmp.kq.tsv" && echo identical || echo DIFFERENT)"
        echo "kq_fp32=true kernels' SHA-1 vs the merge's: $(diff <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$MERGE/h-N.tsv" | sort) \
            <(awk -F'\t' '$2 == "true" {print $1 "\t" $3}' "$S/h-N.tsv" | sort) > /dev/null && echo identical || echo DIFFERENT)"
    } > "$S/merge.txt"
fi
echo done > "$S/DONE"
