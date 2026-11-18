#!/usr/bin/env bash
# R204b (Q1593), run 2 of 2 with GGML_CUDA_MMID_GLU_UNFUSE=2 (arm o4g2b) and the verdict's tables: the switch's two runs
# (o4g2a, o4g2b) against the same tree's two runs with the switch unset (o4c2, o4c3,
# docs/data/2026-11-18_c854b_p106_check parts (5) and (8)); standard library only (python3 -I), from the repository root:
#   O4G2B_FROM="<first request>" O4G2B_TO="<last request>" bash docs/data/2026-11-18_r204_ornith_glu_agent/run/r204b_tables.sh TABLES
# TABLES is a folder holding
#   res/turns.csv, res/attempts.csv   r204a_tables.sh's rows plus o4g2b's: its turns.csv rows, and only the attempts.csv
#                                     rows whose tag starts with o4g2b/
#   rig/server-{o4c2,o4c3,o4g2a,o4g2b}.log  the four runs' server logs (and r204a_tables.sh's others)
#   rig_o4g2b/{meminfo.log,cards.csv} o4g2b's lines of the rig's meminfo.log (from the server's start) and its card log
#   rig_{o4c2,o4c3,o4g2a}/cards.csv   the other three runs' card logs (r204a_tables.sh's)
# and writes its tables there. The per-request tables come from the published agent/<arm>/turns.csv.
set -u
T=${1:?}; D=docs/data/2026-11-18_r204_ornith_glu_agent; C=docs/data/2026-11-18_c854b_p106_check
py() { nice -n 19 python3 -I "$@"; }
O4G2A=$D/agent/o4g2a/turns.csv O4G2B=$D/agent/o4g2b/turns.csv O4C2=$C/agent/o4c2/turns.csv O4C3=$C/agent/o4c3/turns.csv
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb o4c2 o4v5a o4c3 o4v5b \
  o4g2a o4g2b > "$T/agent_tables.out" 2>&1                                           # agent.tsv
py scripts/realloc_counts.py o4g2b="$T/rig/server-o4g2b.log" o4g2a="$T/rig/server-o4g2a.log" o4c3="$T/rig/server-o4c3.log" \
  o4c2="$T/rig/server-o4c2.log" > "$T/r204b_reallocs.tsv"
py $C/run/host_cards.py "$T/rig_o4g2b" R204b o4g2b > "$T/r204b_host.tsv"
py $C/run/busy_clocks.py o4g2b="$T/rig_o4g2b/cards.csv" "${O4G2B_FROM:?}" "${O4G2B_TO:?}" \
  o4g2a="$T/rig_o4g2a/cards.csv" "2026-11-18 18:51:07" "2026-11-18 19:13:52" \
  o4c3="$T/rig_o4c3/cards.csv" "2026-11-18 16:57:49" "2026-11-18 17:27:33" \
  o4c2="$T/rig_o4c2/cards.csv" "2026-11-18 14:11:18" "2026-11-18 14:39:27" > "$T/r204b_clocks.tsv"
py bench/agent_check/report.py work o4g2b=$O4G2B o4g2a=$O4G2A o4c2=$O4C2 o4c3=$O4C3 \
  --tsv "$T/r204b_work.tsv" > "$T/r204b_work.md"
# the requests read on the same prompt in the four runs on this tree, exactly and with the cached tokens within one
py $C/run/same_prompts.py o4c2=$O4C2 o4c3=$O4C3 o4g2a=$O4G2A o4g2b=$O4G2B > "$T/r204b_same_prompts.md"
CACHE_TOL=1 py $C/run/same_prompts.py o4c2=$O4C2 o4c3=$O4C3 o4g2a=$O4G2A o4g2b=$O4G2B > "$T/r204b_same_prompts_tol1.md"
# request by request on the requests matched in all four: each switch run against each control run, and each pair's
# own two runs against each other (its spread on this rig); exactly, and with the cached tokens within one
pair() {  # pair FILE REF RUN: REF's and RUN's turns.csv as name=path, the other two runs as keys
  local f=$1 a=$2 b=$3 k=() x
  for x in o4c2=$O4C2 o4c3=$O4C3 o4g2a=$O4G2A o4g2b=$O4G2B; do
    [ "$x" = "$a" ] || [ "$x" = "$b" ] || k+=("$x")
  done
  py $C/run/paired_prompts.py "$a" "$b" "${k[@]}" > "$T/$f.md"
  CACHE_TOL=1 py $C/run/paired_prompts.py "$a" "$b" "${k[@]}" > "$T/${f}_tol1.md"
}
pair r204b_paired_c2_g2a o4c2=$O4C2 o4g2a=$O4G2A
pair r204b_paired_c3_g2a o4c3=$O4C3 o4g2a=$O4G2A
pair r204b_paired_c2_g2b o4c2=$O4C2 o4g2b=$O4G2B
pair r204b_paired_c3_g2b o4c3=$O4C3 o4g2b=$O4G2B
pair r204b_paired_g2 o4g2a=$O4G2A o4g2b=$O4G2B
pair r204b_paired_c o4c2=$O4C2 o4c3=$O4C3
# the six pairs' totals in one table, then the two-run means: the switch's mean against the controls' mean, with each
# pair's own spread
for s in "" _tol1; do
  for f in c2_g2a c3_g2a c2_g2b c3_g2b g2 c; do
    awk -v f="$f" '/^[|] requests/ { t = 1; next }
      t && /^[|] (all matched|first turns|with a cached prefix)/ {
        n = split($0, c, / [|] /); sub(/^[|] /, "", c[1]); sub(/ [|]$/, "", c[n]); m = split(c[1], l, ", ")
        lab = l[1] ~ /^all/ ? "all matched" : l[1] ~ /^first/ ? "first turns" : "with a cached prefix"
        printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", f, lab, l[m], c[3], c[4], c[5], c[6], c[7] }' \
      "$T/r204b_paired_$f$s.md"
  done > "$T/r204b_pairs$s.rows"
  { echo "| pair (RUN against REF) | requests | n | REF s | RUN s | RUN / REF | RUN slower / same / faster | median of the ratios |"
    echo "|---|---|---:|---:|---:|---:|---|---:|"
    sed -e 's/^| c2_g2a |/| o4g2a against o4c2 |/' -e 's/^| c3_g2a |/| o4g2a against o4c3 |/' \
        -e 's/^| c2_g2b |/| o4g2b against o4c2 |/' -e 's/^| c3_g2b |/| o4g2b against o4c3 |/' \
        -e 's/^| g2 |/| o4g2b against o4g2a |/' -e 's/^| c |/| o4c3 against o4c2 |/' "$T/r204b_pairs$s.rows"
  } > "$T/r204b_pairs$s.md"
  awk -F' [|] ' '$1 == "| g2" { g2a[$2] = $4; g2b[$2] = $5; n[$2] = $3 }
    $1 == "| c" { c2[$2] = $4; c3[$2] = $5; m[$2] = $3 }
    END {
      print "| requests | n | o4c2 s | o4c3 s | o4g2a s | o4g2b s | unset, mean of two | the switch, mean of two | switch / unset | the switch: o4g2b / o4g2a | unset: o4c3 / o4c2 |"
      print "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
      split("all matched|first turns|with a cached prefix", L, "|")
      for (i = 1; i <= 3; i++) {
        k = L[i]; if (!(k in g2a) || !(k in c2) || n[k] != m[k]) { printf "| %s | missing or different counts |\n", k; continue }
        printf "| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.4f | %.4f | %.4f |\n", k, n[k], c2[k], c3[k], g2a[k],
          g2b[k], (c2[k] + c3[k]) / 2, (g2a[k] + g2b[k]) / 2, (g2a[k] + g2b[k]) / (c2[k] + c3[k]), g2b[k] / g2a[k], c3[k] / c2[k]
      } }' "$T/r204b_pairs$s.rows" > "$T/r204b_means$s.md"
  rm "$T/r204b_pairs$s.rows"
done
# generation by context: the four runs' decode speed per middle-context bucket and at one shared mix of contexts, then
# the two-run means at that mix
py $C/run/decode_by_context.py o4c2=$O4C2 o4c3=$O4C3 o4g2a=$O4G2A o4g2b=$O4G2B > "$T/r204b_decode.md"
awk -F' [|] ' '/ms a token at the same mix/ { t = 1; next }
  t && /^[|] o4/ { r = $1; sub(/^[|] /, "", r); sub(/ [|]$/, "", $3); ms[r] = $2; ts[r] = $3 }
  END {
    u = (ms["o4c2"] + ms["o4c3"]) / 2; s = (ms["o4g2a"] + ms["o4g2b"]) / 2
    print "| at the same mix of contexts | o4c2 | o4c3 | o4g2a | o4g2b | unset, mean of two | the switch, mean of two | switch / unset | the switch: o4g2b / o4g2a | unset: o4c3 / o4c2 |"
    print "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    printf "| ms a decode step | %s | %s | %s | %s | %.2f | %.2f | %.4f | %.4f | %.4f |\n", ms["o4c2"], ms["o4c3"], ms["o4g2a"],
      ms["o4g2b"], u, s, s / u, ms["o4g2b"] / ms["o4g2a"], ms["o4c3"] / ms["o4c2"]
    u = (ts["o4c2"] + ts["o4c3"]) / 2; s = (ts["o4g2a"] + ts["o4g2b"]) / 2
    printf "| generation tok/s | %s | %s | %s | %s | %.2f | %.2f | %.4f | %.4f | %.4f |\n", ts["o4c2"], ts["o4c3"], ts["o4g2a"],
      ts["o4g2b"], u, s, s / u, ts["o4g2b"] / ts["o4g2a"], ts["o4c3"] / ts["o4c2"]
  }' "$T/r204b_decode.md" > "$T/r204b_decode_means.md"
