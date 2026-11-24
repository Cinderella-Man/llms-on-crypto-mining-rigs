#!/usr/bin/env bash
# C937 (Q1677, 2026-11-24): which upstream llama.cpp releases have a prebuilt web page archive (llama-server's chat UI)
# in the Hugging Face bucket ggml-org/llama-ui. Needs curl, python3 and git; no token.
#   1. every top-level folder of the bucket (one per release, b<N>, plus latest), from the bucket's listing API, which
#      pages 1000 at a time;
#   2. for the folders FROM..TO (by number), whether dist.tar.gz.sha256 is there (what a llama.cpp tree downloads
#      first), one request each;
#   3. which numbers FROM..TO are upstream release tags (git ls-remote of upstream), joined with step 2.
# Usage: OUT=<folder> [FROM=11700] [TO=12500] [STEPS="folders probe tags"] bash list_archives.sh
# Writes $OUT/folders.tsv (folder, uploadedAt), $OUT/archives_range.tsv (release, folder, sha256 http code) and
# $OUT/releases.tsv (release, upstream tag, folder, sha256 http code). Each step reads the files of the steps before.
set -euo pipefail
STEPS=${STEPS:-folders probe tags}
OUT=${OUT:?OUT=<folder>}
FROM=${FROM:-11700}
TO=${TO:-12500}
BUCKET=${BUCKET:-ggml-org/llama-ui}
mkdir -p "$OUT"
if [[ " $STEPS " == *" folders "* ]]; then
api="https://huggingface.co/api/buckets/$BUCKET/tree?recursive=false"
: > "$OUT/pages.jsonl"
n=0
while [ -n "$api" ]; do
    n=$((n + 1))
    hdr=$(mktemp)
    curl -sf -D "$hdr" "$api" >> "$OUT/pages.jsonl"
    echo >> "$OUT/pages.jsonl"
    api=$(sed -n 's/^[Ll]ink: *<\([^>]*\)>; *rel="next".*/\1/p' "$hdr" | tr -d '\r' | head -n 1)
    rm -f "$hdr"
    [ "$n" -lt 50 ] || { echo "more than 50 pages: stop" >&2; exit 1; }
done
python3 -I -c '
import json, sys
rows = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        for e in json.loads(line):
            if e.get("type") == "directory":
                rows.append((e["path"], e.get("uploadedAt", "")))
def key(r):
    p = r[0]
    return (0, int(p[1:])) if p[:1] == "b" and p[1:].isdigit() else (1, 0)
rows.sort(key=key)
with open(sys.argv[2], "w") as f:
    f.write("folder\tuploadedAt\n")
    for p, u in rows:
        f.write(f"{p}\t{u}\n")
' "$OUT/pages.jsonl" "$OUT/folders.tsv"
rm -f "$OUT/pages.jsonl"
echo "pages: $n, folders: $(($(wc -l < "$OUT/folders.tsv") - 1))"
fi

if [[ " $STEPS " == *" probe "* ]]; then

printf 'release\tfolder\tdist.tar.gz.sha256\n' > "$OUT/archives_range.tsv"
for ((v = FROM; v <= TO; v++)); do
    if grep -q "^b$v	" "$OUT/folders.tsv"; then
        folder=yes
        code=$(curl -s -o /dev/null -w '%{http_code}' -L \
            "https://huggingface.co/buckets/$BUCKET/resolve/b$v/dist.tar.gz.sha256?download=true")
    else
        folder=no
        code=-
    fi
    printf 'b%s\t%s\t%s\n' "$v" "$folder" "$code" >> "$OUT/archives_range.tsv"
done
echo "releases b$FROM..b$TO: folder $(awk -F'\t' '$2=="yes"' "$OUT/archives_range.tsv" | wc -l)," \
    "archive $(awk -F'\t' '$3=="200"' "$OUT/archives_range.tsv" | wc -l)," \
    "folder without archive $(awk -F'\t' '$2=="yes" && $3!="200"' "$OUT/archives_range.tsv" | wc -l)"
fi

if [[ " $STEPS " == *" tags "* ]]; then
git ls-remote --tags https://github.com/ggml-org/llama.cpp 'refs/tags/b*' | sed -n 's|.*refs/tags/\(b[0-9]*\)$|\1|p' \
    | sort -u > "$OUT/upstream_tags.txt"
printf 'release\tupstream tag\tfolder\tdist.tar.gz.sha256\n' > "$OUT/releases.tsv"
tail -n +2 "$OUT/archives_range.tsv" | while IFS=$'\t' read -r v folder code; do
    printf '%s\t%s\t%s\t%s\n' "$v" "$(grep -qx "$v" "$OUT/upstream_tags.txt" && echo yes || echo no)" "$folder" "$code"
done >> "$OUT/releases.tsv"
rm -f "$OUT/upstream_tags.txt"
echo "upstream tags b$FROM..b$TO: $(awk -F'\t' '$2=="yes"' "$OUT/releases.tsv" | wc -l)," \
    "with an archive $(awk -F'\t' '$2=="yes" && $4=="200"' "$OUT/releases.tsv" | wc -l)," \
    "without $(awk -F'\t' '$2=="yes" && $4!="200"' "$OUT/releases.tsv" | wc -l);" \
    "archives that are no tag $(awk -F'\t' '$2=="no" && $4=="200"' "$OUT/releases.tsv" | wc -l)"
fi
