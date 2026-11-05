#!/usr/bin/env bash
# tk_tables.sh: rig_keepalive.sh's T(k) loads as tables. Run it where the files are (it needs bash, awk and date).
#   tk_tables.sh speed  <OUT>   one row a load and test: T(k) at k = 1-5 (ms a pass) and tg128 (ms a token)
#   tk_tables.sh clocks <OUT>   one row a load, phase and card, from smi_<load>.csv (rig_keepalive.sh's SMI_MS):
#                                the phases are "load" (before the first test), "tk" (the -p 1-5 tests) and "tg128"
#                                (from its first warm-up to the log's end); samples in P8, at base (P0 below 1,400 MHz),
#                                between and at boost (1,800 MHz or more), the median SM clock and the mean power
#   tk_tables.sh blocks <OUT>   the keep-alive's own per-card lines from the "on" loads' -v logs (ggml_cuda_keepalive)
set -euo pipefail
mode=$1
out=$2
cd "$out"
# the value of a key in one llama-bench jsonl line
jget='function jget(line, key,    s) {
    if (!match(line, "\"" key "\": *[^,}]*")) return ""
    s = substr(line, RSTART, RLENGTH); sub(/^[^:]*: */, "", s); gsub(/"/, "", s); return s }'
case $mode in
speed)
    printf 'load\tkeepalive\ttest\tms_avg\tms_sd\treps\n'
    for f in tk-*.jsonl; do
        l=${f%.jsonl}
        set=off; [[ $l == tk-on-* ]] && set=decode
        awk -v l="$l" -v set="$set" "$jget"'
            { np = jget($0, "n_prompt") + 0; ng = jget($0, "n_gen") + 0; ns = jget($0, "avg_ns") + 0; sd = jget($0, "stddev_ns") + 0
              reps = 0
              if (match($0, /"samples_ns": *\[[^]]*\]/)) { s = substr($0, RSTART, RLENGTH); reps = gsub(/,/, ",", s) + 1 }
              if (np > 0) printf "%s\t%s\tT(%d)\t%.2f\t%.2f\t%d\n", l, set, np, ns / 1e6, sd / 1e6, reps
              else printf "%s\t%s\ttg%d\t%.2f\t%.2f\t%d\n", l, set, ng, ns / 1e6 / ng, sd / 1e6 / ng, reps }' "$f"
    done
    ;;
clocks)
    printf 'load\tkeepalive\tphase\tcard\tsamples\tP8\tbase\tbetween\tboost\tmedian_MHz\tW_mean\n'
    for f in smi_tk-*.csv; do
        l=${f#smi_}; l=${l%.csv}
        set=off; [[ $l == tk-on-* ]] && set=decode
        # the tests' start times (UTC in the jsonl) as the rig's local time of day
        starts=$(awk "$jget"'{ print jget($0, "test_time") "|" jget($0, "n_prompt") "|" jget($0, "n_gen") }' "$l.jsonl" |
            while IFS='|' read -r t np ng; do echo "$(date -d "$t" +%H:%M:%S) $np $ng"; done)
        STARTS=$starts awk -F', ' -v l="$l" -v set="$set" '
            function sec(t) { split(t, h, ":"); return h[1] * 3600 + h[2] * 60 + h[3] }
            BEGIN {
                n = split(ENVIRON["STARTS"], s, "\n")
                tk0 = -1; tg0 = -1
                for (i = 1; i <= n; i++) {
                    split(s[i], f, " ")
                    if (f[2] + 0 > 0 && tk0 < 0) tk0 = sec(f[1])
                    if (f[2] + 0 == 0 && f[3] + 0 > 0 && tg0 < 0) tg0 = sec(f[1])
                }
            }
            $1 ~ /^[0-9]/ {
                t = sec(substr($1, 12, 8))
                ph = "load"
                if (tk0 >= 0 && t >= tk0) ph = "tk"
                if (tg0 >= 0 && t >= tg0) ph = "tg128"
                k = ph SUBSEP ($2 + 0)
                if (!(k in cnt)) { order[++m] = k }
                cnt[k]++
                c = $4 + 0
                if ($3 == "P8") p8[k]++
                else if (c < 1400) lo[k]++
                else if (c >= 1800) hi[k]++
                else mid[k]++
                w[k] += $5   # rig_keepalive.sh logs timestamp, index, pstate, clocks.sm, power.draw, reasons
                hist[k, c]++
                if (!((k, c) in seen)) { seen[k, c] = 1; nv[k]++; v[k, nv[k]] = c }
            }
            END {
                for (i = 1; i <= m; i++) {
                    k = order[i]
                    # the median: the distinct clocks in order, counted up to half the samples
                    for (a = 1; a <= nv[k]; a++) for (b = a + 1; b <= nv[k]; b++)
                        if (v[k, b] < v[k, a]) { x = v[k, a]; v[k, a] = v[k, b]; v[k, b] = x }
                    acc = 0; med = 0
                    for (a = 1; a <= nv[k]; a++) { acc += hist[k, v[k, a]]; if (acc * 2 >= cnt[k]) { med = v[k, a]; break } }
                    split(k, kk, SUBSEP)
                    printf "%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.1f\n", l, set, kk[1], kk[2], cnt[k], p8[k], lo[k], mid[k], hi[k], med, w[k] / cnt[k]
                }
            }' "$f"
    done
    ;;
blocks)
    for f in smoke.err tk-on-*.err; do
        [ -e "$f" ] || continue
        grep -a 'ggml_cuda_keepalive' "$f" | sed "s/^/${f%.err}\t/"
    done
    ;;
esac
