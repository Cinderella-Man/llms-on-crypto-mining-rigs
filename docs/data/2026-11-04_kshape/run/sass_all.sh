#!/usr/bin/env bash
# Per-kernel SASS identity of two builds' test-backend-ops: every kernel's sm_61 SASS, with the instruction offsets,
# the encodings and the module constant bank's addresses (c[0x3], which move with the module's globals) stripped,
# hashed and compared by kernel name. Kernels whose SASS is identical compute the same results; for Q916 the only
# differing kernels must be llama.cpp's mul_mat_vec_q instances of the five types whose dot products it changed. CPU only
# (cuobjdump of the CUDA container).
# Usage: OUT=<dir> sass_all.sh <old bin/test-backend-ops> <new bin/test-backend-ops>   (run where cuobjdump is)
# Output: OUT/{old,new}.tsv (kernel, md5 of its SASS; a name compiled in several translation units appears once per
#   copy), OUT/diff.tsv (kernel name, identical|differs|only-old|only-new: the sets of its copies' hashes
#   compared), and a tally on stdout. COMPARE=1 compares existing OUT/{old,new}.tsv only.
set -euo pipefail
OUT=${OUT:?OUT=<dir>}
mkdir -p "$OUT"
[ "${COMPARE:-0}" = 1 ] || for x in old:$1 new:$2; do
    tag=${x%%:*}
    rm -rf "$OUT/$tag.k" && mkdir -p "$OUT/$tag.k"
    # one file per kernel (numbered; the name list beside them)
    cuobjdump -sass -arch sm_61 "${x#*:}" | awk -v dir="$OUT/$tag.k" '
        /^[ \t]*Function : / { if (f != "") close(f); n++; f = dir "/" n; print $3 > (dir "/names"); printf "" > f; next }
        /^[ \t]*\/\*[0-9a-f]+\*\// && f != "" { line = $0; sub(/^[ \t]*\/\*[0-9a-f]+\*\/[ \t]*/, "", line);
            sub(/[ \t]*\/\*[^*]*\*\/[ \t]*$/, "", line); gsub(/c\[0x3\]\[0x[0-9a-f]+\]/, "c[0x3][]", line); print line > f }'
    n=$(wc -l < "$OUT/$tag.k/names")
    paste "$OUT/$tag.k/names" <(for i in $(seq 1 "$n"); do md5sum < "$OUT/$tag.k/$i" | cut -c1-32; done) | sort > "$OUT/$tag.tsv"
    rm -rf "$OUT/$tag.k"
done
ms() { sort -u "$1" | awk -F'\t' '{ a[$1] = a[$1] $2 "," } END { for (k in a) print k "\t" a[k] }' | sort; }
join -t $'\t' -a 1 -a 2 -e - -o 0,1.2,2.2 <(ms "$OUT/old.tsv") <(ms "$OUT/new.tsv") | awk -F'\t' '{
    s = ($2 == "-") ? "only-new" : ($3 == "-") ? "only-old" : ($2 == $3) ? "identical" : "differs"; print $1 "\t" s }' > "$OUT/diff.tsv"
cut -f2 "$OUT/diff.tsv" | sort | uniq -c
grep -v $'\tidentical$' "$OUT/diff.tsv" | cut -c1-200 || true
