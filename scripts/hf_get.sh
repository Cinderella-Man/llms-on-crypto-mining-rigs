#!/usr/bin/env bash
# Downloads the files of a manifest written by scripts/hf_manifest.py (repo, pinned revision, path, size, LFS SHA-256)
# from Hugging Face. Each file goes to $DEST/<org>__<repo>/<path>, resumes after an interruption (curl -C -), and must
# match its size and SHA-256; a file already complete (a .ok file beside it) is skipped. Exit status 1 if any file
# failed. Used on 2026-10-27 for the Ornith-1.5 files (106 files, 545 GB).
#   DEST=<folder> [MANIFEST=$DEST/manifest.tsv] [ORDER='<regex> ...'] bash scripts/hf_get.sh
# ORDER: extended regexes over "<repo>/<path>", the files matching the first one download first, then the second's, and
# so on; the rest keep the manifest's order at the end. The Ornith download ran in the order of
#   ORDER='^AtomicChat/ ^unsloth/.*UD-Q2_K_XL 35B-A3B-GGUF ^unsloth/.*imatrix 122B-GGUF'
set -uo pipefail
shopt -s lastpipe   # the last loop runs in this shell, so its failures set the exit status
DEST=${DEST:?download folder}
M=${MANIFEST:-$DEST/manifest.tsv}
read -ra ORD <<< "${ORDER:-}"   # read, not word splitting: no glob expansion of the regexes
order() {  # priority of a manifest row: the index of the first ORDER regex that matches repo/path, lower first
    local i=1 re
    for re in "${ORD[@]}"; do
        if [[ "$1/$2" =~ $re ]]; then echo $i; return; fi
        i=$((i + 1))
    done
    echo $i
}
bad=0
while IFS=$'\t' read -r repo rev path size sha; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(order "$repo" "$path")" "$repo" "$rev" "$path" "$size" "$sha"
done < "$M" | sort -s -k1,1n | while IFS=$'\t' read -r _ repo rev path size sha; do
    out="$DEST/${repo//\//__}/$path"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out.ok" ]; then continue; fi
    echo "== $(date +%T) $repo $path ($((size / 1000000)) MB)"
    # --retry-all-errors: a dropped HTTP/2 stream (curl 92, at 96.7 of 132.6 GB) resumes too
    curl -fsSL --retry 20 --retry-all-errors --retry-delay 10 -C - -o "$out" \
        "https://huggingface.co/$repo/resolve/$rev/$path" \
        || { echo "FAILED download $repo $path"; bad=1; continue; }
    got=$(stat -c %s "$out")
    if [ "$got" != "$size" ]; then echo "FAILED size $repo $path: $got != $size"; bad=1; continue; fi
    if [ -n "$sha" ]; then
        h=$(nice -n 19 sha256sum "$out" | cut -d' ' -f1)
        [ "$h" = "$sha" ] || { echo "FAILED sha256 $repo $path"; bad=1; continue; }
    fi
    echo "$rev $sha" > "$out.ok"
    echo "ok $(date +%T) $path"
done
echo "== done $(date +%T)"
exit $bad
