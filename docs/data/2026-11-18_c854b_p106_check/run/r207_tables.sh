#!/usr/bin/env bash
# R207a (Q1599), the second candidate run's tables (part (8)), standard library only (python3 -I), from the repository root:
#   bash docs/data/2026-11-18_c854b_p106_check/run/r207_tables.sh TABLES
# TABLES is a folder holding
#   res/turns.csv, res/attempts.csv    r206_tables.sh's rows plus o4c3's: its turns.csv rows, and only the attempts.csv
#                                      rows whose tag starts with o4c3/
#   rig/server-{o4c2,o4v5a,o4c3}.log   the three rig-1 runs' server logs
#   rig_o4c3/{meminfo.log,cards.csv}   o4c3's lines of the rig's meminfo.log (from 16:46:27) and its card log
#   rig_o4v5a/cards.csv, rig_o4c2/cards.csv   the other two runs' card logs (r206_tables.sh's)
# and writes its tables there. The per-request tables come from the published agent/<arm>/turns.csv.
set -u
T=${1:?}; D=docs/data/2026-11-18_c854b_p106_check; R166=docs/data/2026-11-16_r166_ornith_stack_no_taper
py() { nice -n 19 python3 -I "$@"; }
O4V5A=$D/agent/o4v5a/turns.csv O4C2=$D/agent/o4c2/turns.csv O4C3=$D/agent/o4c3/turns.csv
O4SNA=$R166/agent/o4sna/turns.csv O4SNB=$R166/agent/o4snb/turns.csv
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb o4c2 o4v5a o4c3 \
  > "$T/agent_tables.out" 2>&1                                                       # agent.tsv
py scripts/realloc_counts.py o4c3="$T/rig/server-o4c3.log" o4v5a="$T/rig/server-o4v5a.log" o4c2="$T/rig/server-o4c2.log" \
  > "$T/r207_reallocs.tsv"
py $D/run/host_cards.py "$T/rig_o4c3" R207a o4c3 > "$T/r207_host.tsv"
py $D/run/busy_clocks.py o4c3="$T/rig_o4c3/cards.csv" "${O4C3_FROM:?}" "${O4C3_TO:?}" \
  o4v5a="$T/rig_o4v5a/cards.csv" "2026-11-18 16:06:27" "2026-11-18 16:31:32" \
  o4c2="$T/rig_o4c2/cards.csv" "2026-11-18 14:11:18" "2026-11-18 14:39:27" > "$T/r207_clocks.tsv"
py bench/agent_check/report.py work o4c3=$O4C3 o4v5a=$O4V5A o4c2=$O4C2 o4sna=$O4SNA o4snb=$O4SNB \
  --tsv "$T/r207_work.tsv" > "$T/r207_work.md"
# the requests read on the same prompt: in the three rig-1 runs, and in those three and v4.0's two
py $D/run/same_prompts.py o4v5a=$O4V5A o4c2=$O4C2 o4c3=$O4C3 > "$T/r207_same_prompts.md"
py $D/run/same_prompts.py o4v5a=$O4V5A o4c2=$O4C2 o4c3=$O4C3 o4sna=$O4SNA o4snb=$O4SNB > "$T/r207_same_prompts5.md"
# request by request against 1f1110c67's o4v5a: this run, and part (7)'s o4c2 on the same set
py $D/run/paired_prompts.py o4v5a=$O4V5A o4c3=$O4C3 o4c2=$O4C2 o4sna=$O4SNA o4snb=$O4SNB > "$T/r207_paired5.md"
py $D/run/paired_prompts.py o4v5a=$O4V5A o4c2=$O4C2 o4c3=$O4C3 o4sna=$O4SNA o4snb=$O4SNB > "$T/r207_paired5_o4c2.md"
py $D/run/paired_prompts.py o4v5a=$O4V5A o4c3=$O4C3 o4c2=$O4C2 > "$T/r207_paired.md"
py $D/run/paired_prompts.py o4v5a=$O4V5A o4c2=$O4C2 o4c3=$O4C3 > "$T/r207_paired_o4c2.md"
# the candidate's two runs against each other (one build's spread on this rig), and v4.0's on the same set
py $D/run/paired_prompts.py o4c2=$O4C2 o4c3=$O4C3 o4v5a=$O4V5A o4sna=$O4SNA o4snb=$O4SNB > "$T/r207_paired5_cand.md"
py $D/run/paired_prompts.py o4sna=$O4SNA o4snb=$O4SNB o4v5a=$O4V5A o4c2=$O4C2 o4c3=$O4C3 > "$T/r207_paired5_r166.md"
py $D/run/paired_prompts.py o4c2=$O4C2 o4c3=$O4C3 o4v5a=$O4V5A > "$T/r207_paired_cand.md"
# the cached tokens within one (o4c3 read Python affine-cipher's turn 2 with one token more from the cache, 2,811 against 2,810)
CACHE_TOL=1 py $D/run/paired_prompts.py o4v5a=$O4V5A o4c3=$O4C3 o4c2=$O4C2 o4sna=$O4SNA o4snb=$O4SNB > "$T/r207_paired5_tol1.md"
CACHE_TOL=1 py $D/run/paired_prompts.py o4v5a=$O4V5A o4c2=$O4C2 o4c3=$O4C3 o4sna=$O4SNA o4snb=$O4SNB > "$T/r207_paired5_tol1_o4c2.md"
CACHE_TOL=1 py $D/run/paired_prompts.py o4c2=$O4C2 o4c3=$O4C3 o4v5a=$O4V5A o4sna=$O4SNA o4snb=$O4SNB > "$T/r207_paired5_tol1_cand.md"
CACHE_TOL=1 py $D/run/paired_prompts.py o4sna=$O4SNA o4snb=$O4SNB o4v5a=$O4V5A o4c2=$O4C2 o4c3=$O4C3 > "$T/r207_paired5_tol1_r166.md"
# part (7)'s 21 requests (r206_paired4.md's rows), with o4c3's prompt ms from the one-token table: each run's seconds
awk -F' [|] ' 'FNR == 1 { f++ }
  /^[|] bugfix/ { k = $1 SUBSEP $2 SUBSEP $3 }
  f == 1 && /^[|] bugfix/ { v5[k] = $5; c2[k] = $6; cold[k] = ($4 == 0); first[k] = ($2 == 1) }
  f == 2 && /^[|] bugfix/ && (k in v5) { c3[k] = $6; if ($5 != v5[k]) bad++ }
  END {
    print "| requests | o4v5a s | o4c2 s | o4c3 s | o4c2 / o4v5a | o4c3 / o4v5a | o4c3 / o4c2 |"
    print "|---|---:|---:|---:|---:|---:|---:|"
    for (s = 0; s < 3; s++) {
      n = a = b = c = 0
      for (k in v5) {
        if ((s == 1 && !first[k]) || (s == 2 && cold[k])) continue
        if (!(k in c3)) { miss++; continue }
        n++; a += v5[k]; b += c2[k]; c += c3[k]
      }
      lab = s == 0 ? "part (7)'"'"'s 21" : s == 1 ? "of them, the first turns" : "of them, with a cached prefix"
      printf "| %s, %d | %.2f | %.2f | %.2f | %.4f | %.4f | %.4f |\n", lab, n, a / 1000, b / 1000, c / 1000, b / a, c / a, c / b
    }
    if (bad + miss) printf "\n%d requests with a different o4v5a time, %d missing in o4c3\n", bad, miss
  }' $D/r206_paired4.md "$T/r207_paired5_tol1.md" > "$T/r207_r206set.md"
