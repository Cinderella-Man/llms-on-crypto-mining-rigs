#!/usr/bin/env bash
# rig_loads.sh: C615 (Q1100) on a P106 rig, untimed: LLAMA_DRAFT_PREFETCH off and on in R86's loads
# (docs/data/2026-11-11_c405_server_gap/run/rig_trace.sh, from which this is made), with the n-gram table's pages
# dropped from the page cache before every load, so that each load reads its rows from the disk as a fresh session's
# new n-grams are (the same requests in every load would otherwise find the rows the previous load read). Runs ON the
# rig, detached, the rig idle; ends with $OUT/DONE.
# Every load is the preset exactly (settings.env: PRESET_ENV, PRESET_ARGS, the MTP draft), started through
# bench/agent_check/rig_server.sh (its monitors, meminfo.log with MemAvailable and swap every 2 s, and the RAM
# watchdog), plus the host profile (GGML_CUDA_PROFILE_HOST). A load whose name starts with "trace" also has the step
# trace (LLAMA_STEP_TRACE); LOAD_ENV adds switches to named loads ('<load>:VAR=value[,VAR=value] ...'; a comma separates
# switches, so values hold none). Each load answers REQ_GREEDY then REQ_LONG, PAUSE seconds apart.
# Before each load: the table file's resident pages (fincore), its cache dropped (dd iflag=nocache count=0: the kernel's
# POSIX_FADV_DONTNEED for the whole file, clean pages only; nothing else is dropped), the resident pages again. Through
# each load C761's vm_sampler.sh (VM_SAMPLER) logs MemAvailable, Cached, swap and paging (pswpin, pswpout, pgmajfault)
# every 0.5 s.
# Usage: BIN=<bin/ of the build> MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> TABLE=<the shard with the n-gram
#   table> PRESET=<settings.env> RIG_SERVER=<bench/agent_check/rig_server.sh> VM_SAMPLER=<vm_sampler.sh>
#   REQ_GREEDY=<json> REQ_LONG=<json> OUT=<dir> [LOADS='trace tracedp prof dp prof-2 dp-2'] [LOAD_ENV=...]
#   [PORT=8193] [PAUSE=4] [MIN_AVAIL_MB=60] bash rig_loads.sh
# Output: per load OUT/<load>/ with host_profile.csv, step_trace.jsonl (trace loads), reply_{greedy,long}.json,
#   server-<load>.log, vm.tsv and rig_server.sh's monitor files; OUT/status.txt (the epoch time around every request,
#   the table's resident pages around each drop and after each request)
# Since C804 (Q1408, 2026-11-12) every load also counts the target graph's reallocations per request from its server
# log's print_timing lines (`graph reallocs = S / D`, as scripts/realloc_counts.py counts them): OUT/<load>/reallocs.tsv
# and a line in status.txt. At the end each load is marked against its line without its switch, the loads whose
# switches (LOAD_ENV's, and the step trace of a trace load) are the largest proper subset of its own (with R122's ladder
# the rung before, with the default loads dp against prof and prof-2): OUT/realloc_loads.tsv, OUT/realloc_marks.tsv and
# a line in status.txt per load that reallocates where its line does not. The loads and requests are the same as before.
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${DRAFT:?}" "${TABLE:?}" "${PRESET:?}" "${RIG_SERVER:?}" "${VM_SAMPLER:?}" "${REQ_GREEDY:?}" "${REQ_LONG:?}" "${OUT:?}"
LOADS=${LOADS:-trace tracedp prof dp prof-2 dp-2}
PORT=${PORT:-8193}; PAUSE=${PAUSE:-4}
# shellcheck disable=SC1090
. "$PRESET"
: "${PRESET_ENV:?}" "${PRESET_ARGS:?}"
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
log() { echo "$(date -Is) $*" | tee -a "$OUT/status.txt"; }
quiet() { log "quiet: load $(cut -d' ' -f1-3 /proc/loadavg); llama processes $(pgrep -c '^llama-'); $(grep -E '^MemAvailable:' /proc/meminfo | tr -s ' ')"; }
resident() { fincore --bytes --noheadings --output RES "$TABLE" 2> /dev/null | tr -d ' ' || echo '?'; }
# realloc_count SERVER_LOG TSV (C804): the target graph's reallocations per request from print_timing's running count
# (`graph reallocs = S / D` since the load, after the request's `prompt eval time = X ms / N tokens`); writes one row
# per request to TSV and prints the summary
realloc_count() {
    sed 's/\x1b\[[0-9;]*m//g' "$1" 2> /dev/null | awk -v out="$2" '
        BEGIN { print "req\tprompt_tokens\tprompt_ms\treallocs_same\treallocs_changed" > out }
        /prompt eval time =/ { t = $0; sub(/.*prompt eval time = */, "", t); split(t, p, " "); ms = p[1]; n = p[4]; cur = 1; next }
        cur && /graph reallocs =/ { t = $0; sub(/.*graph reallocs = */, "", t); split(t, q, " ")
            s = q[1] - ps; d = q[3] - pd; ps = q[1]; pd = q[3]; cur = 0
            printf "%d\t%s\t%s\t%d\t%d\n", k, n, ms, s, d > out
            k++; ts += s; td += d; if (s + d > 0) w++ }
        END { printf "graph reallocations: %d of %d requests (%d at the same size, %d changed)\n", w, k, ts, td }'
}
# realloc_marks (C804): each load of realloc_loads.tsv against the loads whose switches are the largest proper subset
# of its own; marks as scripts/realloc_counts.py --marks gives them
realloc_marks() {
    awk -F'\t' '
        function subset(a, b,   x, y, i, j, nx, ny, f) {
            if (a == "-") return 1
            nx = split(a, x, ","); ny = split(b, y, ",")
            for (i = 1; i <= nx; i++) { f = 0; for (j = 1; j <= ny; j++) if (x[i] == y[j]) f = 1; if (!f) return 0 }
            return 1 }
        NR > 1 { m++; nm[m] = $1; sw[m] = $2; n[m] = $3; w[m] = $4; s[m] = $5; d[m] = $6; k[m] = ($2 == "-") ? 0 : split($2, t, ",") }
        END {
            print "load\tswitches\tagainst\trequests\trequests_with_realloc\treallocs_same\treallocs_changed\tagainst_requests_with_realloc\tmark"
            for (i = 1; i <= m; i++) {
                best = -1
                for (j = 1; j <= m; j++) if (k[j] < k[i] && k[j] > best && subset(sw[j], sw[i])) best = k[j]
                refs = ""; rw = ""; zero = 1; more = 1
                for (j = 1; j <= m; j++) if (k[j] < k[i] && k[j] == best && subset(sw[j], sw[i])) {
                    refs = refs (refs == "" ? "" : ",") nm[j]; rw = rw (rw == "" ? "" : ",") w[j] " of " n[j]
                    if (w[j] > 0) zero = 0
                    if (w[i] / (n[i] > 0 ? n[i] : 1) <= w[j] / (n[j] > 0 ? n[j] : 1)) more = 0 }
                mk = (refs == "") ? "-" : (w[i] == 0 ? "none" : (zero ? "reallocates" : (more ? "reallocates more" : "as its line")))
                printf "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\t%s\n", nm[i], sw[i], (refs == "" ? "-" : refs), n[i], w[i], s[i], d[i], (rw == "" ? "-" : rw), mk
            } }' "$OUT/realloc_loads.tsv"
}
rs() { # <load> VAR=value ...: rig_server.sh in the load's folder
    local d=$OUT/$1
    shift
    env LOGDIR="$d" PORT="$PORT" MIN_AVAIL_MB="${MIN_AVAIL_MB:-60}" "$@" bash "$RIG_SERVER" >> "$OUT/status.txt" 2>&1
}

printf 'load\tswitches\trequests\trequests_with_realloc\treallocs_same\treallocs_changed\n' > "$OUT/realloc_loads.tsv"
for load in $LOADS; do
    d=$OUT/$load
    mkdir -p "$d"
    sw="GGML_CUDA_PROFILE_HOST=$d/host_profile.csv"
    case $load in
        trace*) sw="$sw LLAMA_STEP_TRACE=$d/step_trace.jsonl" ;;
    esac
    for e in ${LOAD_ENV:-}; do
        [ "${e%%:*}" = "$load" ] && sw="$sw ${e#*:}" && sw=${sw//,/ }
    done
    # the load's own switches for C804's marks: LOAD_ENV's, and the step trace (by name) of a trace load
    sws=$( { case $load in trace*) echo LLAMA_STEP_TRACE ;; esac
             for e in ${LOAD_ENV:-}; do [ "${e%%:*}" = "$load" ] && echo "${e#*:}" | tr ',' '\n'; done; } | sort -u | paste -sd, -)
    pkill -x llama-server; sleep 2
    before=$(resident)
    dd if="$TABLE" iflag=nocache count=0 status=none
    log "$load: n-gram table cache dropped, resident bytes $before -> $(resident)"
    quiet
    rm -f "$d/vm.stop"
    OUT="$d/vm.tsv" STOP="$d/vm.stop" bash "$VM_SAMPLER" &
    vm=$!
    log "$load: server start $(date +%s.%3N)"
    if ! rs "$load" ACTION=start ARM="$load" BIN="$BIN/llama-server" MODEL="$MODEL" \
            "SERVER_ENV=$PRESET_ENV $sw" "SERVER_ARGS=$PRESET_ARGS -md $DRAFT"; then
        log "$load: server did not start"
        rs "$load" ACTION=end
        pkill -9 -x llama-server
        touch "$d/vm.stop"; wait "$vm"
        continue
    fi
    log "$load: server up ($PRESET_ENV $sw; $PRESET_ARGS -md $DRAFT); table resident bytes $(resident)"
    for r in greedy long; do
        case $r in greedy) f=$REQ_GREEDY ;; long) f=$REQ_LONG ;; esac
        sleep "$PAUSE"
        log "$load $r: request start $(date +%s.%3N)"
        curl -s -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$f" \
            > "$d/reply_$r.json"
        log "$load $r: request exit $? end $(date +%s.%3N) ($(wc -c < "$d/reply_$r.json") bytes); table resident bytes $(resident)"
    done
    # rig_server.sh ends the server with SIGTERM and waits for it, so that both files are whole (and the C615 line)
    rs "$load" ACTION=end
    if pgrep -x llama-server > /dev/null; then log "$load: server still up, killing it"; pkill -9 -x llama-server; fi
    touch "$d/vm.stop"; wait "$vm"
    log "$load: stopped"
    log "$load: $(realloc_count "$d/server-$load.log" "$d/reallocs.tsv")"
    awk -F'\t' -v l="$load" -v sw="${sws:--}" 'NR > 1 { n++; if ($4 + $5 > 0) w++; s += $4; c += $5 }
        END { printf "%s\t%s\t%d\t%d\t%d\t%d\n", l, sw, n, w, s, c }' "$d/reallocs.tsv" >> "$OUT/realloc_loads.tsv"
done
realloc_marks > "$OUT/realloc_marks.tsv"
awk -F'\t' 'NR > 1 && $9 ~ /^reallocates/ { r++; printf "graph reallocations: %s %s (%d of %d requests; its line without the switch, %s: %s)\n", $1, $9, $5, $4, $3, $8 }
    END { if (!r) print "graph reallocations: no load reallocates where its line without its switch does not" }' \
    "$OUT/realloc_marks.tsv" | while IFS= read -r line; do log "$line"; done
sleep 2; quiet
touch "$OUT/DONE"
