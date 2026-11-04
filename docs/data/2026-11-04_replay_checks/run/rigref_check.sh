#!/usr/bin/env bash
# C372 (Q931): the references of a recorded set against the rig tarball's own test-backend-ops. A sample of each set's
# cases (--sample N: about one in N) is replayed on the CPU with the rig binary and re-recorded (--replay with --record
# and -b CPU: the recorded inputs, the reference computed again); each re-recorded .ref blob must equal the set's, bit
# for bit (md5). Runs in a container without GPUs, CPU only.
# Env: BIN (bin/ of the rig tarball), SETS (folder with the recorded suites), W (scratch), SAMPLE (default 20),
#   SUITES (default: every folder in SETS with a record.txt), JOBS (default 4)
# Output: W/rigref.txt (one line per set: cases re-recorded, identical .ref blobs, PASS or FAIL)
set -uo pipefail
BIN=${BIN:?}; SETS=${SETS:?}; W=${W:?}
SAMPLE=${SAMPLE:-20}
JOBS=${JOBS:-4}
mkdir -p "$W"
R=$W/rigref.txt
# the rig build links the CUDA driver library, which a container without GPUs lacks: the toolkit's stub stands in
# (CUDA then finds no device; only the CPU backend is present)
if [ -e /usr/local/cuda/lib64/stubs/libcuda.so ] && ! ldconfig -p | grep -q "libcuda.so.1 "; then
    mkdir -p "$W/stub" && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "$W/stub/libcuda.so.1"
    export LD_LIBRARY_PATH="$W/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
for s in ${SUITES:-$(cd "$SETS" && ls -d */ | tr -d /)}; do
    [ -f "$SETS/$s/record.txt" ] || continue
    d=$W/$s
    rm -rf "$d"
    n=0
    # the set's filters: the record.txt lines after "filters", "  <-o><TAB><-p>"
    while IFS=$'\t' read -r ops p; do
        n=$((n + 1))
        "$BIN/test-backend-ops" -b CPU -o "${ops# }" -p "$p" --replay "$SETS/$s" --record "$d" --sample "$SAMPLE" -j "$JOBS" \
            > "$W/$s-$n.log" 2>&1
    done < <(sed -n '/^filters/,$p' "$SETS/$s/record.txt" | tail -n +2 | sed 's/^  //')
    c=0; same=0
    for f in "$d"/cases/*.ref; do
        [ -e "$f" ] || continue
        c=$((c + 1))
        cmp -s "$f" "$SETS/$s/cases/$(basename "$f")" && same=$((same + 1))
    done
    [ "$c" -gt 0 ] && [ "$same" = "$c" ] && v=PASS || v=FAIL
    echo "$s $v: $c cases re-recorded with $(basename "$(dirname "$BIN")"), $same .ref blobs identical to the set's" | tee -a "$R"
done
