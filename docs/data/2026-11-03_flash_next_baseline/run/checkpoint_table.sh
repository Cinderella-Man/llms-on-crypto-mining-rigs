#!/usr/bin/env bash
# checkpoint_table.sh V3_LOG V3DIAG_LOG(.gz) > checkpoints.csv: context-checkpoint saves and restores from llama-server
# logs. Restores log their own duration ("restored context checkpoint (… size = S MiB, T ms)"); saves log only at
# trace level (-lv 5) and without a duration, so a save's time is the gap between its line's timestamp
# (minutes.seconds.ms.us since the server started) and the line before it, which the server prints just before it
# copies the state.
set -euo pipefail
echo "log,event,n_tokens,size_mib,ms"
for f in "$@"; do
  cat_=cat; [[ $f == *.gz ]] && cat_="gunzip -c"
  $cat_ "$f" | awk -v src="$(basename "$f" .gz)" '
    function us(ts,  a) { split(ts, a, "."); return ((a[1] * 60 + a[2]) * 1000 + a[3]) * 1000 + a[4] }
    /restored context checkpoint/ {
      match($0, /n_tokens = [0-9]+/); n = substr($0, RSTART + 11, RLENGTH - 11)
      match($0, /size = [0-9.]+ MiB, [0-9.]+ ms/); split(substr($0, RSTART, RLENGTH), p, " ")
      print src ",restore," n "," p[3] "," p[5] }
    /created context checkpoint/ {
      match($0, /n_tokens = [0-9]+/); n = substr($0, RSTART + 11, RLENGTH - 11)
      match($0, /size = [0-9.]+ MiB/); s = substr($0, RSTART + 7, RLENGTH - 11)
      printf "%s,save,%s,%s,%.1f\n", src, n, s, (us($1) - us(prev)) / 1000 }
    { prev = $1 }'
done
