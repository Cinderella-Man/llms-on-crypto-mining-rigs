#!/usr/bin/env bash
# b12723-finish (2026-11-25, C1118): two rig builds' programs compared byte by byte and function by function, for a
# commit that changes only host code (here caf27fdae against 0a6a04e22). Every program carries its commit id and an ELF
# build id, so two builds' md5s always differ; this counts the bytes that differ and lists the defined symbols whose
# size differs (nm -S, demangled). The programs link libllama statically, so a function that grows moves the code after
# it and changes many bytes at the same file size; the symbol list shows which functions changed.
# Env: A and B (the two builds' bin folders), OUT (the TSV to write). Runs where the binaries are: cmp, nm, grep.
set -uo pipefail
: "${A:?A=<first bin folder>}" "${B:?B=<second bin folder>}" "${OUT:?OUT=<tsv>}"
export LC_ALL=C # sort and comm in one collation
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
syms() {  # <program>: "name<TAB>size" of every defined symbol with a size, sorted
    nice -n 19 nm -S --defined-only -C "$1" 2>/dev/null \
        | awk 'NF >= 4 { s = $2; $1 = ""; $2 = ""; $3 = ""; sub(/^ +/, ""); print $0 "\t" s }' | sort -u
}
printf 'program\tsize first\tsize second\tbytes differ\tsymbols first\tsymbols second\tsymbols whose size differs\twhich\n' > "$OUT"
for p in $(ls "$A"); do
    if [ ! -f "$B/$p" ]; then
        printf '%s\t%s\t-\t-\t-\t-\t-\tonly in the first\n' "$p" "$(stat -c %s "$A/$p")" >> "$OUT"
        continue
    fi
    nd=$(nice -n 19 cmp -l "$A/$p" "$B/$p" | wc -l)
    syms "$A/$p" > "$T/a"
    syms "$B/$p" > "$T/b"
    comm -3 "$T/a" "$T/b" | sed 's/^\t//' | cut -f1 | sort -u > "$T/d"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$p" "$(stat -c %s "$A/$p")" "$(stat -c %s "$B/$p")" "$nd" \
        "$(wc -l < "$T/a")" "$(wc -l < "$T/b")" "$(wc -l < "$T/d")" "$(paste -sd'|' "$T/d")" >> "$OUT"
done
for p in $(ls "$B"); do
    [ -f "$A/$p" ] || printf '%s\t-\t%s\t-\t-\t-\t-\tonly in the second\n' "$p" "$(stat -c %s "$B/$p")" >> "$OUT"
done
cut -f1,4,7,8 "$OUT"
