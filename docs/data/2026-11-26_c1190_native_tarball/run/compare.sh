#!/usr/bin/env bash
# C1190: a rig tarball built natively (NATIVE=1) against a Docker-built one of the same commit and settings, both
# unpacked. Runs where the CUDA 12.9 toolkit is (cuobjdump) at the lowest CPU priority; needs no card.
#
#   A=<Docker build folder> B=<native build folder> OUT=<folder> [CUDA=/usr/local/cuda] \
#       [PROGS="llama-server test-backend-ops"] bash compare.sh
#
# Writes to OUT:
#   buildinfo.diff     BUILDINFO.txt of A against B
#   programs.tsv       every program of either build: size and md5 in each
#   elf.tsv            per program and build: AVX instructions (ymm registers, as BUILDINFO.txt counts them, and every
#                      VEX-coded "v..." instruction), the highest GLIBC_ symbol version, the shared libraries it needs,
#                      its RUNPATH and the compiler named in its .comment section
#   host_<prog>.tsv    PROGS' defined symbols (nm -S, demangled) whose size differs, and host_totals.tsv the counts
#   card_<prog>*.tsv   PROGS' card code, function by function: cuobjdump -sass and -ptx of each build compared by
#                      card_compare.py (identical / same / const / code, after C1179's code_compare.py)
set -euo pipefail
: "${A:?}" "${B:?}" "${OUT:?}"
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
CUDA=${CUDA:-/usr/local/cuda}
PROGS=${PROGS:-llama-server test-backend-ops}
mkdir -p "$OUT"
printf 'program\tbuild\tlisting\tbytes\tcuobjdump error lines\n' > "$OUT/listings.tsv"
# the listings run to gigabytes: scratch next to the results, not in /tmp
T=$(mktemp -d -p "$OUT")
trap 'rm -rf "$T"' EXIT
n() { nice -n 19 "$@"; }

diff "$A/BUILDINFO.txt" "$B/BUILDINFO.txt" > "$OUT/buildinfo.diff" || true

printf 'program\tsize A\tsize B\tmd5 A\tmd5 B\n' > "$OUT/programs.tsv"
for p in $( (ls "$A/bin"; ls "$B/bin") | sort -u); do
    sa=- sb=- ma=- mb=-
    [ -f "$A/bin/$p" ] && sa=$(stat -c %s "$A/bin/$p") && ma=$(md5sum < "$A/bin/$p" | cut -c1-32)
    [ -f "$B/bin/$p" ] && sb=$(stat -c %s "$B/bin/$p") && mb=$(md5sum < "$B/bin/$p" | cut -c1-32)
    printf '%s\t%s\t%s\t%s\t%s\n' "$p" "$sa" "$sb" "$ma" "$mb" >> "$OUT/programs.tsv"
done

# one program of one build: its elf.tsv row
elf_row() {
    local p=$1 side=$2 dir=$A
    [ "$side" = B ] && dir=$B
    local f=$dir/bin/$p dis=$T/dis.$p.$side
    n objdump -d --no-show-raw-insn "$f" > "$dis"
    local ymm vex glibc needed runpath comp
    ymm=$(grep -c ymm "$dis" || true)
    vex=$(grep -cE '^ +[0-9a-f]+:'$'\t''v[a-z0-9]+ ' "$dis" || true)
    rm -f "$dis"
    glibc=$(objdump -T "$f" | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -n 1)
    needed=$(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | paste -sd' ')
    runpath=$(readelf -d "$f" | sed -n 's/.*(\(RUNPATH\|RPATH\)).*\[\(.*\)\]/\2/p')
    comp=$(readelf -p .comment "$f" | sed -n 's/^ *\[ *[0-9a-f]*\] *//p' | sort -u | paste -sd'|')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$p" "$side" "$ymm" "$vex" "$glibc" "$needed" "${runpath:--}" "$comp"
}
JOBS=${JOBS:-6}
for p in $(ls "$B/bin"); do
    for side in A B; do
        [ "$side" = A ] && [ ! -f "$A/bin/$p" ] && continue
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done
        elf_row "$p" "$side" > "$T/elf.$p.$side" &
    done
done
wait
printf 'program\tbuild\tymm instructions\tVEX-coded instructions\thighest GLIBC_\tneeded\trunpath\tcompiler\n' > "$OUT/elf.tsv"
for p in $(ls "$B/bin"); do cat "$T/elf.$p.A" "$T/elf.$p.B" 2> /dev/null; done >> "$OUT/elf.tsv"
rm -f "$T"/elf.*

syms() { n nm -S --defined-only -C "$1" 2> /dev/null \
    | awk 'NF >= 4 { s = $2; t = $3; $1 = ""; $2 = ""; $3 = ""; sub(/^ +/, ""); print $0 "\t" t "\t" s }' | sort; }
printf 'program\tsymbols A\tsymbols B\tin both\tsame size\tsize differs\tonly in A\tonly in B\n' > "$OUT/host_totals.tsv"
for p in $PROGS; do
    syms "$A/bin/$p" > "$T/a"
    syms "$B/bin/$p" > "$T/b"
    python3 -I - "$T/a" "$T/b" "$OUT/host_$p.tsv" "$p" >> "$OUT/host_totals.tsv" << 'PY'
import collections, sys
def load(path):
    d = collections.defaultdict(list)
    for line in open(path, errors='replace'):
        name, kind, size = line.rstrip('\n').rsplit('\t', 2)
        d[(name, kind)].append(int(size, 16))
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
both = set(a) & set(b)
same = [k for k in both if sorted(a[k]) == sorted(b[k])]
diff = sorted(k for k in both if sorted(a[k]) != sorted(b[k]))
with open(sys.argv[3], 'w') as f:
    f.write('symbol\tkind\tsizes A\tsizes B\n')
    for k in diff:
        f.write(f'{k[0]}\t{k[1]}\t{",".join(map(str, sorted(a[k])))}\t{",".join(map(str, sorted(b[k])))}\n')
    for k in sorted(set(a) - set(b)):
        f.write(f'{k[0]}\t{k[1]}\t{",".join(map(str, sorted(a[k])))}\t-\n')
    for k in sorted(set(b) - set(a)):
        f.write(f'{k[0]}\t{k[1]}\t-\t{",".join(map(str, sorted(b[k])))}\n')
print('\t'.join(map(str, [sys.argv[4], len(a), len(b), len(both), len(same), len(diff), len(set(a) - set(b)), len(set(b) - set(a))])))
PY
done

for p in $PROGS; do
    for side in A B; do
        dir=$A
        [ "$side" = B ] && dir=$B
        for kind in sass ptx; do
            { n "$CUDA/bin/cuobjdump" "-$kind" "$dir/bin/$p" > "$T/$side.$kind" 2> "$T/err.$side.$kind" || true; } &
        done
    done
    wait
    for side in A B; do
        for kind in sass ptx; do
            printf '%s\t%s\t%s\t%s\t%s\n' "$p" "$side" "$kind" "$(stat -c %s "$T/$side.$kind")" \
                "$(wc -l < "$T/err.$side.$kind")" >> "$OUT/listings.tsv"
        done
    done
    n python3 -I "$here/card_compare.py" "$T/A.sass" "$T/B.sass" "$T/A.ptx" "$T/B.ptx" "$OUT/card_$p" \
        > "$OUT/card_${p}_totals.txt"
    rm -f "$T"/A.* "$T"/B.* "$T"/err.*
done
for f in "$OUT"/card_*_totals.txt; do echo "== $f"; cat "$f"; done
cat "$OUT/host_totals.tsv"
