#!/usr/bin/env bash
# The dry run of lightprof_rig.sh on any Linux machine without a GPU: the rig script runs unchanged against
# fake_llama_server.py (and stand-ins for curl and nvidia-smi), every arm, then its tables. Run from the repository's
# root, e.g. in a container: docker run --rm -v "$PWD":/r:ro python:3.12-slim-bookworm bash /r/docs/data/2026-11-04_light_profiler/run/dryrun.sh
# Prints the rig script's output, its check table, and "dry run: pass" or "dry run: FAIL <reasons>".
set -uo pipefail
R=$(cd "$(dirname "$0")/../../../.." && pwd)
RUN=$R/docs/data/2026-11-04_light_profiler/run
T=$(mktemp -d)
mkdir -p "$T/bundle" "$T/bin"
cp "$RUN/lightprof_rig.sh" "$RUN/lightprof_requests.py" "$R/scripts/prof_summary.py" "$R/scripts/ram_watchdog.sh" "$T/bundle/"
cp "$RUN/fake_llama_server.py" "$T/bin/llama-server"
chmod +x "$T/bin/llama-server"
cat > "$T/bin/curl" << 'EOF'
#!/usr/bin/env bash
python3 -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' "${@: -1}" > /dev/null 2>&1
EOF
printf '#!/bin/sh\necho "0, 5 MiB"\n' > "$T/bin/nvidia-smi"
chmod +x "$T/bin/curl" "$T/bin/nvidia-smi"
PATH=$T/bin:$PATH PROF_SUMMARY_DIR=$T/bundle BIN=$T/bin MODEL=/dev/null OUT=$T/out ARMS="off graph layer sample=8 group off" \
  N_WARM=4 N_TOK=8 REPS=2 EVENT_US=8 bash "$T/bundle/lightprof_rig.sh" 2>&1 | grep -v -e '^chrt' -e 'pkill: command not found' -e 'free: command not found'
echo "----- check.tsv"
column -t -s $'\t' "$T/out/check.tsv" 2> /dev/null || cat "$T/out/check.tsv"
fail=""
[ -f "$T/out/DONE" ] || fail="$fail no-DONE"
[ "$(awk 'END { print NR }' "$T/out/check.tsv")" = 7 ] || fail="$fail rows"
for a in graph layer sample8 group; do
  [ -s "$T/out/summary-$a.txt" ] || fail="$fail no-summary-$a"
  awk -F'\t' -v a="$a" '$1 == a && $13 == "yes" { ok = 1 } END { exit !ok }' "$T/out/check.tsv" || fail="$fail output-$a"
done
for a in graph layer; do
  awk -F'\t' -v a="$a" '$1 == a && $14 == "pass" { ok = 1 } END { exit !ok }' "$T/out/check.tsv" || fail="$fail check-$a"
done
awk -F'\t' '$1 == "sample8" && $9 + 0 == 1 { ok = 1 } END { exit !ok }' "$T/out/check.tsv" || fail="$fail coverage"
awk -F'\t' '$1 == "group" && $3 + 0 > 25 { ok = 1 } END { exit !ok }' "$T/out/check.tsv" || fail="$fail group-cost"
awk -F'\t' '$1 == "off2" && $3 + 0 == 0 { ok = 1 } END { exit !ok }' "$T/out/check.tsv" || fail="$fail off2"
grep -q "C142's check" "$T/out/check.txt" || fail="$fail check-txt"
rm -rf "$T"
[ -z "$fail" ] && echo "dry run: pass" || { echo "dry run: FAIL$fail"; exit 1; }
