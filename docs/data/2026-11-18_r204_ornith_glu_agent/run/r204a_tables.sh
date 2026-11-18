#!/usr/bin/env bash
# R204a (Q1593), run 1 of 2 with GGML_CUDA_MMID_GLU_UNFUSE=2 (arm o4g2a), against the same tree's two runs with the
# switch unset (o4c2, o4c3, docs/data/2026-11-18_c854b_p106_check parts (5) and (8)); standard library only
# (python3 -I), from the repository root:
#   O4G2A_FROM="<first request>" O4G2A_TO="<last request>" bash docs/data/2026-11-18_r204_ornith_glu_agent/run/r204a_tables.sh TABLES
# TABLES is a folder holding
#   res/turns.csv, res/attempts.csv   r207b_tables.sh's rows plus o4g2a's: its turns.csv rows, and only the attempts.csv
#                                     rows whose tag starts with o4g2a/
#   rig/server-{o4c2,o4c3,o4g2a}.log  the three runs' server logs (and r207b_tables.sh's others)
#   rig_o4g2a/{meminfo.log,cards.csv} o4g2a's lines of the rig's meminfo.log (from the server's start) and its card log
#   rig_{o4c2,o4c3}/cards.csv         the two control runs' card logs
# and writes its tables there. The per-request tables come from the published agent/<arm>/turns.csv.
set -u
T=${1:?}; D=docs/data/2026-11-18_r204_ornith_glu_agent; C=docs/data/2026-11-18_c854b_p106_check
py() { nice -n 19 python3 -I "$@"; }
O4G2A=$D/agent/o4g2a/turns.csv O4C2=$C/agent/o4c2/turns.csv O4C3=$C/agent/o4c3/turns.csv
O4V5A=$C/agent/o4v5a/turns.csv O4V5B=$C/agent/o4v5b/turns.csv
py docs/data/2026-11-05_keepalive/run/agent_tables.py "$T" v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb o4c2 o4v5a o4c3 o4v5b \
  o4g2a > "$T/agent_tables.out" 2>&1                                                 # agent.tsv
py scripts/realloc_counts.py o4g2a="$T/rig/server-o4g2a.log" o4c3="$T/rig/server-o4c3.log" o4c2="$T/rig/server-o4c2.log" \
  > "$T/r204a_reallocs.tsv"
py $C/run/host_cards.py "$T/rig_o4g2a" R204a o4g2a > "$T/r204a_host.tsv"
py $C/run/busy_clocks.py o4g2a="$T/rig_o4g2a/cards.csv" "${O4G2A_FROM:?}" "${O4G2A_TO:?}" \
  o4c3="$T/rig_o4c3/cards.csv" "2026-11-18 16:57:49" "2026-11-18 17:27:33" \
  o4c2="$T/rig_o4c2/cards.csv" "2026-11-18 14:11:18" "2026-11-18 14:39:27" > "$T/r204a_clocks.tsv"
py bench/agent_check/report.py work o4g2a=$O4G2A o4c2=$O4C2 o4c3=$O4C3 o4v5a=$O4V5A o4v5b=$O4V5B \
  --tsv "$T/r204a_work.tsv" > "$T/r204a_work.md"
# the requests read on the same prompt in the three runs on this tree, exactly and with the cached tokens within one
py $C/run/same_prompts.py o4c2=$O4C2 o4c3=$O4C3 o4g2a=$O4G2A > "$T/r204a_same_prompts.md"
CACHE_TOL=1 py $C/run/same_prompts.py o4c2=$O4C2 o4c3=$O4C3 o4g2a=$O4G2A > "$T/r204a_same_prompts_tol1.md"
# request by request on the requests matched in all three: the switch's run against each control run, and the two
# control runs against each other (the tree's own spread on this rig); exactly, and with the cached tokens within one
pair() {  # pair FILE REF RUN: REF's and RUN's turns.csv as name=path, the third run as the key
  local f=$1 a=$2 b=$3 k=() x
  for x in o4c2=$O4C2 o4c3=$O4C3 o4g2a=$O4G2A; do
    [ "$x" = "$a" ] || [ "$x" = "$b" ] || k+=("$x")
  done
  py $C/run/paired_prompts.py "$a" "$b" "${k[@]}" > "$T/$f.md"
  CACHE_TOL=1 py $C/run/paired_prompts.py "$a" "$b" "${k[@]}" > "$T/${f}_tol1.md"
}
pair r204a_paired_c2_g2a o4c2=$O4C2 o4g2a=$O4G2A
pair r204a_paired_c3_g2a o4c3=$O4C3 o4g2a=$O4G2A
pair r204a_paired_c2_c3 o4c2=$O4C2 o4c3=$O4C3
# the three pairs' totals in one table
for s in "" _tol1; do
  for f in c2_g2a c3_g2a c2_c3; do
    awk -v f="$f" '/^[|] requests/ { t = 1; next }
      t && /^[|] (all matched|first turns|with a cached prefix)/ {
        n = split($0, c, / [|] /); sub(/^[|] /, "", c[1]); sub(/ [|]$/, "", c[n]); m = split(c[1], l, ", ")
        lab = l[1] ~ /^all/ ? "all matched" : l[1] ~ /^first/ ? "first turns" : "with a cached prefix"
        printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", f, lab, l[m], c[3], c[4], c[5], c[6], c[7] }' \
      "$T/r204a_paired_$f$s.md"
  done > "$T/r204a_pairs$s.rows"
  { echo "| pair (RUN against REF) | requests | n | REF s | RUN s | RUN / REF | RUN slower / same / faster | median of the ratios |"
    echo "|---|---|---:|---:|---:|---:|---|---:|"
    sed -e 's/^| c2_g2a |/| o4g2a against o4c2 |/' -e 's/^| c3_g2a |/| o4g2a against o4c3 |/' \
        -e 's/^| c2_c3 |/| o4c3 against o4c2 |/' "$T/r204a_pairs$s.rows"
  } > "$T/r204a_pairs$s.md"
  rm "$T/r204a_pairs$s.rows"
done
# generation by context: decode speed per middle-context bucket and at one shared mix of contexts, the three runs on
# this tree, and the five rig-1 runs of the day (with pascal-p106-v5's 1f1110c67, parts (7) and (9))
py $C/run/decode_by_context.py o4c2=$O4C2 o4c3=$O4C3 o4g2a=$O4G2A > "$T/r204a_decode.md"
py $C/run/decode_by_context.py o4c2=$O4C2 o4c3=$O4C3 o4v5a=$O4V5A o4v5b=$O4V5B o4g2a=$O4G2A > "$T/r204a_decode5.md"
