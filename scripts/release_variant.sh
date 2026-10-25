#!/usr/bin/env bash
# release_variant.sh: picks one engine variant in the release drafts (README.md, models/flint/README.md,
# release/notes-<version>.md). A line that ends in "<!-- only:TAG -->", and a block between the lines
# "<!-- begin:TAG -->" and "<!-- end:TAG -->", belong to TAG. The tags given stay, without their markers; every other
# tagged line and block is removed. Tags in use: the engine commits rc1, rc3 and rc4; the features q163-q111 (prompt
# attention and the agent prompt cache, in rc3 and rc4), drafter (a release with the drafter and store files, v2.1)
# and nodrafter (the lines that say they come later).
#
#   scripts/release_variant.sh rc4 README.md models/flint/README.md          edit the files in place
#   scripts/release_variant.sh "rc3 q163-q111 nodrafter" - < in.md > out.md  filter standard input
# Presets (the name alone): rc4 = "rc4 q163-q111 nodrafter" (v2.0), rc4-drafter = "rc4 q163-q111 drafter" (v2.1),
# rc3 = "rc3 q163-q111 nodrafter", rc1 = "rc1 nodrafter".
# Prints, per file, the tags kept and removed. Exit status: 0 done, 1 a marker survived, 2 usage.
set -euo pipefail
[ $# -ge 2 ] || { sed -n '2,/^# Prints/p' "$0"; exit 2; }
keep=$1; shift
case $keep in
    rc4) keep="rc4 q163-q111 nodrafter" ;;
    rc4-drafter) keep="rc4 q163-q111 drafter" ;;
    rc3) keep="rc3 q163-q111 nodrafter" ;;
    rc1) keep="rc1 nodrafter" ;;
esac

filter() {  # filter FILE: the variant of FILE on standard output; tags kept and removed on standard error
    local f=$1 t kept="" gone="" prog=(-e '')
    for t in $(grep -oE '<!-- (only|begin|end):[A-Za-z0-9_-]+ -->' "$f" | sed -E 's/<!-- [a-z]+:([^ ]+) -->/\1/' | sort -u); do
        if [[ " $keep " == *" $t "* ]]; then
            prog+=(-e "/^<!-- begin:$t -->\$/d" -e "/^<!-- end:$t -->\$/d" -e "s/ <!-- only:$t -->\$//")
            kept="$kept $t"
        else
            prog+=(-e "/^<!-- begin:$t -->\$/,/^<!-- end:$t -->\$/d" -e "/<!-- only:$t -->\$/d")
            gone="$gone $t"
        fi
    done
    sed "${prog[@]}" "$f"
    echo "$2: kept [${kept# }], removed [${gone# }]" >&2
}

status=0
for f in "$@"; do
    if [ "$f" = - ]; then
        tmp=$(mktemp); cat > "$tmp"
        filter "$tmp" "standard input" | tee "$tmp.out"
        grep -qE '<!-- (only|begin|end):' "$tmp.out" && { echo "markers left in the output" >&2; status=1; }
        rm -f "$tmp" "$tmp.out"
    else
        filter "$f" "$f" > "$f.variant"
        mv "$f.variant" "$f"
        grep -qE '<!-- (only|begin|end):' "$f" && { echo "markers left in $f" >&2; status=1; }
    fi
done
exit "$status"
