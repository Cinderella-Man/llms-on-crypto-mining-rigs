#!/usr/bin/env bash
# C937 (Q1677, 2026-11-24): where two builds of the same commit differ, binary by binary. For each file in both bin/
# folders: the md5; every ELF section's size and md5 (objcopy -O binary --only-section); the embedded device code
# (cuobjdump -xelf all / -xptx all: each cubin and PTX file's md5); and for a PTX file that differs, the number of
# changed lines and how many of them differ only in nvcc's per-compile names (_INTERNAL_<hash>_..._cu_<hash>_<n>...).
# Needs binutils and CUDA's cuobjdump. Usage: A=<bin folder> B=<bin folder> OUT=<folder> [TMP=<scratch>]
#   [CUOBJDUMP=cuobjdump] bash compare_build.sh
# Writes $OUT/sections.tsv, $OUT/device_code.tsv and $OUT/ptx_diff.tsv; prints a summary per binary.
set -uo pipefail
A=${A:?A=<bin folder>}
B=${B:?B=<bin folder>}
OUT=${OUT:?OUT=<folder>}
TMP=${TMP:-$OUT/tmp}
CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
mkdir -p "$OUT"
printf 'file\tsection\tsize A\tsize B\tsame\n' > "$OUT/sections.tsv"
printf 'file\tcubins\tcubins same\tptx\tptx same\n' > "$OUT/device_code.tsv"
printf 'file\tptx\tchanged lines\tonly nvcc names\n' > "$OUT/ptx_diff.tsv"
sections() { readelf -SW "$1" | sed -n 's/^ *\[ *[0-9]*\] *\([^ ]*\) *[A-Z_0-9]* *[0-9a-f]* *[0-9a-f]* *\([0-9a-f]*\).*/\1 \2/p' | grep -v '^NULL '; }
secmd5() { objcopy -O binary --only-section="$2" "$1" "$TMP/sec" 2> /dev/null && md5sum < "$TMP/sec" | cut -d' ' -f1; rm -f "$TMP/sec"; }
for fa in "$A"/*; do
    f=$(basename "$fa"); fb=$B/$f
    [ -f "$fb" ] || { echo "$f: only in A"; continue; }
    rm -rf "$TMP" && mkdir -p "$TMP/a" "$TMP/b"
    same=$([ "$(md5sum < "$fa")" = "$(md5sum < "$fb")" ] && echo yes || echo no)
    # sections: name and size from readelf, content md5 from objcopy
    nsec=0 nsame=0 diffs=""
    while read -r name size_a; do
        size_b=$(sections "$fb" | awk -v n="$name" '$1 == n {print $2}' | head -n 1)
        s=no
        [ "$size_a" = "$size_b" ] && [ "$(secmd5 "$fa" "$name")" = "$(secmd5 "$fb" "$name")" ] && s=yes
        nsec=$((nsec + 1)); [ $s = yes ] && nsame=$((nsame + 1)) || diffs="$diffs $name"
        printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$name" "$((16#$size_a))" "$([ -n "$size_b" ] && echo $((16#$size_b)) || echo -)" "$s" >> "$OUT/sections.tsv"
    done < <(sections "$fa")
    line="$f: md5 $same, sections the same $nsame of $nsec${diffs:+ (differ:$diffs)}"
    # device code
    if readelf -SW "$fa" | grep -q ' \.nv_fatbin '; then
        (cd "$TMP/a" && "$CUOBJDUMP" -xelf all "$fa" > /dev/null 2>&1; "$CUOBJDUMP" -xptx all "$fa" > /dev/null 2>&1)
        (cd "$TMP/b" && "$CUOBJDUMP" -xelf all "$fb" > /dev/null 2>&1; "$CUOBJDUMP" -xptx all "$fb" > /dev/null 2>&1)
        cub=0 cubs=0 ptx=0 ptxs=0
        for x in "$TMP"/a/*; do
            n=$(basename "$x"); y=$TMP/b/$n
            s=$([ -f "$y" ] && cmp -s "$x" "$y" && echo 1 || echo 0)
            case "$n" in
                *.cubin) cub=$((cub + 1)); cubs=$((cubs + s)) ;;
                *.ptx) ptx=$((ptx + 1)); ptxs=$((ptxs + s))
                    if [ "$s" = 0 ] && [ -f "$y" ]; then
                        ch=$(diff "$x" "$y" | grep -c '^<')
                        norm() { sed -E 's/_cu_[0-9a-f]{8}_[0-9]+/_cu_X_N/g' "$1"; }
                        left=$(diff <(norm "$x") <(norm "$y") | grep -c '^<')
                        printf '%s\t%s\t%s\t%s\n' "$f" "${n#"$f".}" "$ch" "$([ "$left" = 0 ] && echo yes || echo "no ($left lines differ otherwise)")" >> "$OUT/ptx_diff.tsv"
                    fi ;;
            esac
        done
        printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$cub" "$cubs" "$ptx" "$ptxs" >> "$OUT/device_code.tsv"
        line="$line; cubins the same $cubs of $cub, PTX $ptxs of $ptx"
    fi
    echo "$line"
done
rm -rf "$TMP"
