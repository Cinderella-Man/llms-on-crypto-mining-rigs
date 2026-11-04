#!/usr/bin/env bash
# reshard_iq3_s.sh: Swift IQ3_S (two shards as published, the n-gram table in shard 1 with 372 other tensors) into three
# shards with the table alone in the third, so the rig reads the table from its SSD (scripts/gguf_reshard.py, C160's
# method). Swift IQ3_S's table is byte for byte the table of the rig's Q2_0 and IQ3_XXS sets (table_md5.tsv), and its
# tensor count is theirs (1,224), so the new third shard should be the very file the rig already keeps on its SSD:
# the script checks that against TABLE_MD5 and then deletes its copy, leaving shards 1 and 2 to copy to the rig.
#   SRC="S-00001-of-00002.gguf S-00002-of-00002.gguf" OUT=dir TABLE_MD5=<md5 of the rig's table shard> bash reshard_iq3_s.sh
# Then `gguf_reshard.py verify` rebuilds both old shards from the new ones and compares SHA-256 with the originals.
# Output: OUT/swift_iq3_s_3shard/ (shards 1 and 2), OUT/reshard.tsv (file, bytes, md5), OUT/verify.txt, OUT/DONE.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
: "${SRC:?}" "${OUT:?}" "${TABLE_MD5:?}"
D=$OUT/swift_iq3_s_3shard
T=$OUT/table_check
mkdir -p "$D" "$T"
rm -f "$OUT/DONE"
echo "=== reshard $(date -Is)"
# shellcheck disable=SC2086
ionice -c3 nice -n 19 python3 -I "$REPO/scripts/gguf_reshard.py" split --isolate '^per_layer_token_embd\.' \
    --out-dir "$D" --isolated-dir "$T" $SRC || { echo "reshard FAILED"; echo FAIL > "$OUT/DONE"; exit 1; }
printf 'file\tbytes\tmd5\n' > "$OUT/reshard.tsv"
for f in "$D"/*.gguf; do
    r=$(readlink -f "$f")
    printf '%s\t%s\t%s\n' "$(basename "$f")" "$(stat -c %s "$r")" "$(ionice -c3 nice -n 19 md5sum < "$r" | cut -d' ' -f1)" \
        >> "$OUT/reshard.tsv"
done
cat "$OUT/reshard.tsv"
echo "=== verify $(date -Is)"
# shellcheck disable=SC2086
ionice -c3 nice -n 19 python3 -I "$REPO/scripts/gguf_reshard.py" verify $SRC --new "$D"/*.gguf > "$OUT/verify.txt" 2>&1
echo "verify rc $?"
for f in $SRC; do
    echo "$(ionice -c3 nice -n 19 sha256sum < "$f" | cut -d' ' -f1)  $(basename "$f") (original)"
done >> "$OUT/verify.txt"
cat "$OUT/verify.txt"
t=$(awk -F '\t' '$1 ~ /00003-of-00003/ {print $3}' "$OUT/reshard.tsv")
if [ "$t" = "$TABLE_MD5" ]; then
    echo "table shard = the rig's table file ($t): deleting this copy; on the rig, shard 3 is a symlink to the SSD table"
    rm -f "$T"/*.gguf "$D"/*-00003-of-00003.gguf
    echo OK > "$OUT/DONE"
else
    echo "table shard md5 $t differs from $TABLE_MD5: keeping it ($T)"
    echo DIFFERENT > "$OUT/DONE"
fi
