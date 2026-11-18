#!/usr/bin/env bash
# R207b (Q1599), the second run of pascal-p106-v5's 1f1110c67 and the verdict's tables (part (9)), standard library only
# (python3 -I), from the repository root:
#   O4V5B_FROM="<first request>" O4V5B_TO="<last request>" bash docs/data/2026-11-18_c854b_p106_check/run/r207b_tables.sh TABLES
# TABLES is a folder holding
#   res/turns.csv, res/attempts.csv    r207_tables.sh's rows plus o4v5b's: its turns.csv rows, and only the attempts.csv
#                                      rows whose tag starts with o4v5b/
#   rig/server-{o4c2,o4v5a,o4c3,o4v5b}.log   the four rig-1 runs' server logs
#   rig_o4v5b/{meminfo.log,cards.csv}  o4v5b's lines of the rig's meminfo.log (from the server's start) and its card log
#   rig_{o4v5a,o4c2,o4c3}/cards.csv    the other three runs' card logs (r207_tables.sh's)
# and writes its tables there. The per-request tables come from the published agent/<arm>/turns.csv.
set -u
T=${1:?}; D=docs/data/2026-11-18_c854b_p106_check; R166=docs/data/2026-11-16_r166_ornith_stack_no_taper
py() { nice -n 19 python3 -I "$@"; }
O4V5A=$D/agent/o4v5a/turns.csv O4V5B=$D/agent/o4v5b/turns.csv O4C2=$D/agent/o4c2/turns.csv O4C3=$D/agent/o4c3/turns.csv
O4SNA=$R166/agent/o4sna/turns.csv O4SNB=$R166/agent/o4snb/turns.csv
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb o4c2 o4v5a o4c3 o4v5b \
  > "$T/agent_tables.out" 2>&1                                                       # agent.tsv
py scripts/realloc_counts.py o4v5b="$T/rig/server-o4v5b.log" o4c3="$T/rig/server-o4c3.log" o4v5a="$T/rig/server-o4v5a.log" \
  o4c2="$T/rig/server-o4c2.log" > "$T/r207b_reallocs.tsv"
py $D/run/host_cards.py "$T/rig_o4v5b" R207b o4v5b > "$T/r207b_host.tsv"
py $D/run/busy_clocks.py o4v5b="$T/rig_o4v5b/cards.csv" "${O4V5B_FROM:?}" "${O4V5B_TO:?}" \
  o4c3="$T/rig_o4c3/cards.csv" "2026-11-18 16:57:49" "2026-11-18 17:27:33" \
  o4v5a="$T/rig_o4v5a/cards.csv" "2026-11-18 16:06:27" "2026-11-18 16:31:32" \
  o4c2="$T/rig_o4c2/cards.csv" "2026-11-18 14:11:18" "2026-11-18 14:39:27" > "$T/r207b_clocks.tsv"
py bench/agent_check/report.py work o4v5b=$O4V5B o4v5a=$O4V5A o4c2=$O4C2 o4c3=$O4C3 o4sna=$O4SNA o4snb=$O4SNB \
  --tsv "$T/r207b_work.tsv" > "$T/r207b_work.md"
# the requests read on the same prompt in the four rig-1 runs, and in those four and v4.0's two
py $D/run/same_prompts.py o4v5a=$O4V5A o4v5b=$O4V5B o4c2=$O4C2 o4c3=$O4C3 > "$T/r207b_same_prompts4.md"
py $D/run/same_prompts.py o4v5a=$O4V5A o4v5b=$O4V5B o4c2=$O4C2 o4c3=$O4C3 o4sna=$O4SNA o4snb=$O4SNB > "$T/r207b_same_prompts6.md"
# request by request on the requests matched in all four rig-1 runs: each candidate run against each 1f1110c67 run, and
# each build's two runs against each other (its own spread on this rig); exactly, and with the cached tokens within one
pair() {  # pair FILE REF RUN: REF's and RUN's turns.csv as name=path, the other rig-1 runs as keys
  local f=$1 a=$2 b=$3 k=() x
  for x in o4v5a=$O4V5A o4v5b=$O4V5B o4c2=$O4C2 o4c3=$O4C3; do
    [ "$x" = "$a" ] || [ "$x" = "$b" ] || k+=("$x")
  done
  py $D/run/paired_prompts.py "$a" "$b" "${k[@]}" > "$T/$f.md"
  CACHE_TOL=1 py $D/run/paired_prompts.py "$a" "$b" "${k[@]}" > "$T/${f}_tol1.md"
}
pair r207b_paired_v5a_c2 o4v5a=$O4V5A o4c2=$O4C2
pair r207b_paired_v5a_c3 o4v5a=$O4V5A o4c3=$O4C3
pair r207b_paired_v5b_c2 o4v5b=$O4V5B o4c2=$O4C2
pair r207b_paired_v5b_c3 o4v5b=$O4V5B o4c3=$O4C3
pair r207b_paired_v5 o4v5a=$O4V5A o4v5b=$O4V5B
pair r207b_paired_cand o4c2=$O4C2 o4c3=$O4C3
# the six pairs' totals in one table
for s in "" _tol1; do
  for f in v5a_c2 v5a_c3 v5b_c2 v5b_c3 v5 cand; do
    awk -v f="$f" '/^[|] requests/ { t = 1; next }
      t && /^[|] (all matched|first turns|with a cached prefix)/ {
        n = split($0, c, / [|] /); sub(/^[|] /, "", c[1]); sub(/ [|]$/, "", c[n]); m = split(c[1], l, ", ")
        lab = l[1] ~ /^all/ ? "all matched" : l[1] ~ /^first/ ? "first turns" : "with a cached prefix"
        printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", f, lab, l[m], c[3], c[4], c[5], c[6], c[7] }' \
      "$T/r207b_paired_$f$s.md"
  done > "$T/r207b_pairs$s.rows"
  { echo "| pair (RUN against REF) | requests | n | REF s | RUN s | RUN / REF | RUN slower / same / faster | median of the ratios |"
    echo "|---|---|---:|---:|---:|---:|---|---:|"
    sed -e 's/^| v5a_c2 |/| o4c2 against o4v5a |/' -e 's/^| v5a_c3 |/| o4c3 against o4v5a |/' \
        -e 's/^| v5b_c2 |/| o4c2 against o4v5b |/' -e 's/^| v5b_c3 |/| o4c3 against o4v5b |/' \
        -e 's/^| v5 |/| o4v5b against o4v5a |/' -e 's/^| cand |/| o4c3 against o4c2 |/' "$T/r207b_pairs$s.rows"
  } > "$T/r207b_pairs$s.md"
  rm "$T/r207b_pairs$s.rows"
done
# part (7)'s 21 requests (r206_paired4.md's rows): each of the four rig-1 runs' seconds, o4c3's and o4v5b's from the
# tables that match the cached tokens within one, then each build's two-run mean and spread
CACHE_TOL=1 py $D/run/paired_prompts.py o4v5a=$O4V5A o4v5b=$O4V5B > "$T/r207b_v5_tol1_all.md"
awk -F' [|] ' 'FNR == 1 { f++ }
  /^[|] bugfix/ { k = $1 SUBSEP $2 SUBSEP $3 }
  f == 1 && /^[|] bugfix/ { v5[k] = $5; c2[k] = $6; cold[k] = ($4 == 0); first[k] = ($2 == 1) }
  f == 2 && /^[|] bugfix/ && (k in v5) { c3[k] = $6; if ($5 != v5[k]) bad++ }
  f == 3 && /^[|] bugfix/ && (k in v5) { v5b[k] = $6; if ($5 != v5[k]) bad++ }
  END {
    print "| requests | o4v5a s | o4v5b s | o4c2 s | o4c3 s | 1f1110c67, mean of two | the candidate, mean of two | candidate / 1f1110c67 | 1f1110c67: o4v5b / o4v5a | the candidate: o4c3 / o4c2 |"
    print "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for (s = 0; s < 3; s++) {
      n = a = b = c = d = 0
      for (k in v5) {
        if ((s == 1 && !first[k]) || (s == 2 && cold[k])) continue
        if (!(k in c3) || !(k in v5b)) { miss++; continue }
        n++; a += v5[k]; d += v5b[k]; b += c2[k]; c += c3[k]
      }
      lab = s == 0 ? "part (7)'"'"'s 21" : s == 1 ? "of them, the first turns" : "of them, with a cached prefix"
      printf "| %s, %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.4f | %.4f | %.4f |\n", lab, n, a / 1000, d / 1000,
        b / 1000, c / 1000, (a + d) / 2000, (b + c) / 2000, (b + c) / (a + d), d / a, c / b
    }
    if (bad + miss) printf "\n%d requests with a different o4v5a time, %d missing in o4c3 or o4v5b\n", bad, miss
  }' $D/r206_paired4.md $D/r207_paired5_tol1.md "$T/r207b_v5_tol1_all.md" > "$T/r207b_r206set.md"
rm "$T/r207b_v5_tol1_all.md"
# generation by context: the four rig-1 runs' decode speed per middle-context bucket, and at one shared mix of contexts
py $D/run/decode_by_context.py o4v5a=$O4V5A o4v5b=$O4V5B o4c2=$O4C2 o4c3=$O4C3 > "$T/r207b_decode.md"
