#!/usr/bin/env bash
# ptx_names.sh: C779's PTX against the base's with the numbers in internal-linkage names masked. checks.sh's sass step
# masks the hash nvcc puts in an internal name (_cu_<8 hex>); it keeps the number after it, which nvcc counts over the
# entities a translation unit declares before the name. C779 declares more in common.cuh (graph-patch.h), so every
# function-local __shared__ array of a kernel compiled after it gets another number: the PTX of ten flash_attn_tile
# instances differs only in that name (the SASS is the same file). This dumps both binaries' PTX whole, masks
# _cu_<8 hex>_<number> as well, and compares the two files.
# Env: NEW and BASE (the llama-server binaries), OUT (a folder for the dumps), LOCK (flock file, optional), IMAGE
# Output: OUT/{new,base}.ptx, OUT/ptx_names.txt (the differing lines' names and counts, and the masked files' md5s)
set -uo pipefail
: "${NEW:?}" "${BASE:?}" "${OUT:?}"
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
mkdir -p "$OUT"
run() { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
for p in "NEW:$NEW:new" "BASE:$BASE:base"; do
    f=${p#*:}; f=${f%:*}; w=${p##*:}
    d=$(dirname "$f")
    run docker run --rm --label pascal-quants=1 --cpu-shares=256 --network none -v "$d:$d:ro" -v "$OUT:$OUT" "$IMAGE" \
        bash -c "cuobjdump -ptx $f > $OUT/$w.ptx" > /dev/null 2>&1
done
{
    echo "lines: new $(wc -l < "$OUT/new.ptx"), base $(wc -l < "$OUT/base.ptx")"
    echo "differing lines: $(diff "$OUT/base.ptx" "$OUT/new.ptx" | grep -c '^[<>]')"
    echo "their internal names (with the number after the hash):"
    diff "$OUT/base.ptx" "$OUT/new.ptx" | grep '^[<>]' | grep -oE '_cu_[0-9a-f]{8}_[0-9]+' | sort | uniq -c
    nm=$(sed -E 's/_cu_[0-9a-f]{8}_[0-9]+/_cu_X_N/g' "$OUT/new.ptx" | md5sum | cut -d' ' -f1)
    bm=$(sed -E 's/_cu_[0-9a-f]{8}_[0-9]+/_cu_X_N/g' "$OUT/base.ptx" | md5sum | cut -d' ' -f1)
    echo "masked md5: new $nm, base $bm"
    echo "verdict: $([ "$nm" = "$bm" ] && echo SAME || echo DIFFERS)"
} | tee "$OUT/ptx_names.txt"
