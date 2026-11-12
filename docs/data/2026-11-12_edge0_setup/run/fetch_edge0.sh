#!/usr/bin/env bash
# Fetch one Edge0 model folder from Hugging Face with curl (resume-safe), then check every file's size and every
# large file's SHA-256 against the repository's own listing. Nothing from the folder is executed.
#
#   WORK=/path/to/edge0 REPO=Edge0/Edge0-145B-q3k bash fetch_edge0.sh
#
# Optional: REV (default main), JOBS (parallel downloads, default 2), MIN_FREE_GB (default 160),
#           RATE (curl's --limit-rate per download, e.g. 35M: keep the sum below what the target disks write
#           steadily; a QLC system disk stalled the whole machine under an unthrottled 26 GB),
#           REUSE_DIRS (space-separated folders searched for an identical ple-fp8.gguf before downloading it),
#           LINK_FROM (a finished sibling model folder: files with the same SHA-256 are hard-linked, not fetched).
# Output:   $WORK/models/<name>/        the model folder (data only)
#           $WORK/dl/<name>.manifest.tsv  path, bytes, sha256 (or -) from the Hugging Face API
#           $WORK/dl/<name>.log           progress
#           $WORK/dl/<name>.verify        one line per file: OK or FAIL
#           $WORK/dl/<name>.done          written last: "OK <date>" or "FAIL <date>"
set -u
WORK=${WORK:?set WORK}; REPO=${REPO:?set REPO}; REV=${REV:-main}; JOBS=${JOBS:-2}; MIN_FREE_GB=${MIN_FREE_GB:-160}
REUSE_DIRS=${REUSE_DIRS:-}; LINK_FROM=${LINK_FROM:-}
NAME=${REPO##*/}
DEST=$WORK/models/$NAME
DL=$WORK/dl
MAN=$DL/$NAME.manifest.tsv
LOG=$DL/$NAME.log
mkdir -p "$DEST" "$DL"
rm -f "$DL/$NAME.done" "$DL/$NAME.verify"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "start $REPO@$REV -> $DEST"
free_gb=$(df -Pk "$WORK" | awk 'NR==2{print int($4/1048576)}')
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then say "only $free_gb GB free, need $MIN_FREE_GB"; echo "FAIL $(date '+%F %T') space" > "$DL/$NAME.done"; exit 1; fi

# 1. the file list, sizes and LFS hashes from the API (assets/ holds only the card's pictures)
curl -fsSL --retry 5 "https://huggingface.co/api/models/$REPO/tree/$REV?recursive=1" -o "$DL/$NAME.tree.json" || { say "tree API failed"; echo "FAIL $(date '+%F %T') api" > "$DL/$NAME.done"; exit 1; }
python3 -I - "$DL/$NAME.tree.json" > "$MAN" <<'PY'
import json, sys
for e in sorted(json.load(open(sys.argv[1])), key=lambda e: e.get("size", 0)):
    if e.get("type") != "file" or e["path"].startswith("assets/"):
        continue
    print(e["path"], e["size"], (e.get("lfs") or {}).get("oid", "-"), sep="\t")
PY
say "manifest: $(wc -l < "$MAN") files, $(awk -F'\t' '{s+=$2} END{printf "%.2f GB", s/1e9}' "$MAN")"

sha_of() { nice -n 19 ionice -c3 sha256sum "$1" | cut -d' ' -f1; }

# 2. reuse before fetching: identical files from a sibling folder (hard links) or another local copy of the table
while IFS=$'\t' read -r path size oid; do
  [ "$oid" = "-" ] && continue
  [ -e "$DEST/$path" ] && continue
  mkdir -p "$(dirname "$DEST/$path")"
  if [ -n "$LINK_FROM" ] && [ -f "$LINK_FROM/$path" ] && [ "$(stat -L -c %s "$LINK_FROM/$path")" = "$size" ]; then
    if [ "$(sha_of "$LINK_FROM/$path")" = "$oid" ]; then ln "$LINK_FROM/$path" "$DEST/$path" && say "linked $path from $LINK_FROM"; fi
  fi
done < "$MAN"
if [ -n "$REUSE_DIRS" ] && [ ! -e "$DEST/ple-fp8.gguf" ]; then
  want=$(awk -F'\t' '$1=="ple-fp8.gguf"{print $2}' "$MAN"); oid=$(awk -F'\t' '$1=="ple-fp8.gguf"{print $3}' "$MAN")
  # shellcheck disable=SC2086
  for cand in $(find $REUSE_DIRS -maxdepth 6 -type f -name 'ple*.gguf' -size "${want}c" 2>/dev/null); do
    say "candidate table $cand, hashing"
    if [ "$(sha_of "$cand")" = "$oid" ]; then cp --reflink=auto "$cand" "$DEST/ple-fp8.gguf" && say "copied ple-fp8.gguf from $cand"; break; fi
  done
fi

# 3. download what is missing or short, smallest first, JOBS at a time
fetch_one() {
  path=$1; size=$2
  out="$DEST/$path"; mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -L -c %s "$out")" = "$size" ]; then return 0; fi
  for try in 1 2 3 4 5 6; do
    curl -fsSL --retry 5 --retry-delay 10 ${RATE:+--limit-rate "$RATE"} -C - -o "$out" "https://huggingface.co/$REPO/resolve/$REV/$path" && break
    echo "$(date '+%F %T') retry $try $path" >> "$LOG"; sleep 15
  done
  echo "$(date '+%F %T') got $path $(stat -L -c %s "$out" 2>/dev/null)" >> "$LOG"
}
export -f fetch_one; export DEST REPO REV LOG; export RATE=${RATE:-}
awk -F'\t' '{print $1"\t"$2}' "$MAN" | nice -n 19 ionice -c3 xargs -P "$JOBS" -d '\n' -n 1 bash -c 'IFS=$'"'"'\t'"'"' read -r p s <<< "$1"; fetch_one "$p" "$s"' _

# 4. verify: every size, and the SHA-256 of every file the repository stores by hash
bad=0
while IFS=$'\t' read -r path size oid; do
  f="$DEST/$path"
  if [ ! -f "$f" ] || [ "$(stat -L -c %s "$f")" != "$size" ]; then echo "FAIL size $path" >> "$DL/$NAME.verify"; bad=1; continue; fi
  if [ "$oid" != "-" ]; then
    got=$(sha_of "$f")
    if [ "$got" != "$oid" ]; then echo "FAIL sha256 $path $got" >> "$DL/$NAME.verify"; bad=1; continue; fi
    echo "OK sha256 $path $got" >> "$DL/$NAME.verify"
  else
    echo "OK size $path $(sha_of "$f")" >> "$DL/$NAME.verify"
  fi
done < "$MAN"
chmod -R a-x,a+X "$DEST"
if [ "$bad" = 0 ]; then say "verified"; echo "OK $(date '+%F %T')" > "$DL/$NAME.done"; else say "verification failed"; echo "FAIL $(date '+%F %T') verify" > "$DL/$NAME.done"; fi
