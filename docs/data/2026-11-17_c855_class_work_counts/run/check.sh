#!/usr/bin/env bash
# check.sh: C855's check (Q1526). Every published agent-check table whose rows are all published is rebuilt from those
# rows twice: with the agent_tables.py that made it (docs/data/2026-11-05_keepalive/run) and with
# bench/agent_check/agent_tables.py. Per table it prints whether the new table's first 14 columns are the old script's
# table byte for byte, and whether the old script's table is the published one (a table whose acceptance came from
# server logs that are not published differs there only, and says so). Then the class: the class_arms.py that made
# R159's class.tsv and bench/agent_check/class_arms.py on R159's published agent.tsv, each against its class.tsv, and
# the new class_arms.py on the new table, whose first lines must be that class.tsv. Last, the classes on the work
# measures that the result page quotes. The new tables and classes go to OUT/tables (default: this folder's tables/).
# WORK is a scratch folder (default: a new temporary one). Exits non-zero if an old column or class line differs.
# Runs with Python 3 (no packages), from anywhere.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
OUT=${OUT:-$(cd "$(dirname "$0")/.." && pwd)}
WORK=${WORK:-$(mktemp -d)}
T=$OUT/tables
mkdir -p "$T" "$WORK"
PY=(python3 -I)
D=$ROOT/docs/data
OLD_T=$D/2026-11-05_keepalive/run/agent_tables.py
OLD_C=$D/2026-11-14_r159_flint_release_agent/run/class_arms.py
NEW_T=$ROOT/bench/agent_check/agent_tables.py
NEW_C=$ROOT/bench/agent_check/class_arms.py
MERGE=$D/2026-11-17_r17_flint_textreuse/run/merge_rows.py
RR=$D/2026-11-05_release_regression/agent
R159=$D/2026-11-14_r159_flint_release_agent
R160=$D/2026-11-14_r160_ornith_release_agent
R166=$D/2026-11-16_r166_ornith_stack_no_taper
R169=$D/2026-11-16_r169_ornith_no_eager
R170=$D/2026-11-16_r170_flint_q5
R17=$D/2026-11-17_r17_flint_textreuse
SIDE=$D/2026-11-06_flint_side_rig2
MTP=$D/2026-11-06_ornith_mtp_rig2
fail=0

same() {  # LABEL A B: A and B byte for byte
  if cmp -s "$2" "$3"; then echo "same: $1"; else echo "DIFFERS: $1"; fail=1; fi
}

cols_differ() {  # A B: the header names of the columns in which two TSVs of the same shape differ
  awk -F'\t' 'NR == FNR { a[FNR] = $0; next }
    FNR == 1 { for (i = 1; i <= NF; i++) h[i] = $i }
    { n = split(a[FNR], x, "\t"); for (i = 1; i <= NF; i++) if (x[i] != $i) d[h[i]] = 1 }
    END { s = ""; for (k in d) s = s (s ? ", " : "") k; print s }' "$1" "$2"
}

merge() {  # FOLD NAME=DIR[:ARM]...: a table folder from published rows (R17's merge_rows.py)
  local fold=$1; shift
  rm -rf "${WORK:?}/$fold"
  "${PY[@]}" "$MERGE" "$WORK/$fold" "$@" > /dev/null
}

table() {  # FOLD PUBLISHED NAME REF ARM...: both scripts over the folder (OUT_TAG, SKIP_TASKS, NMAX from the call)
  local fold=$1 pub=$2 name=$3 tag=${OUT_TAG:-}
  shift 3
  "${PY[@]}" "$OLD_T" "$WORK/$fold" "$@" > /dev/null
  mv "$WORK/$fold/agent$tag.tsv" "$WORK/$name.old.tsv"
  "${PY[@]}" "$NEW_T" "$WORK/$fold" "$@" > /dev/null
  mv "$WORK/$fold/agent$tag.tsv" "$T/$name.tsv"
  cut -f1-14 "$T/$name.tsv" > "$WORK/$name.new14.tsv"
  same "$name: the new table's first 14 columns are the old script's table" "$WORK/$name.new14.tsv" "$WORK/$name.old.tsv"
  if cmp -s "$WORK/$name.old.tsv" "$pub"; then
    echo "  and the old script's table is the published ${pub#"$D/"}"
  else
    echo "  the old script's table differs from the published ${pub#"$D/"} in: $(cols_differ "$WORK/$name.old.tsv" "$pub")"
  fi
}

# Flint Pascal Q8 GGUF: R159's seven runs (v3.0's reference run, v3.0, v4-rc2, the tuned v4 preset)
merge flint v3p065="$RR/flint-p065-v3base:rel" f4a="$R159/agent/f4a" f3a="$R159/agent/f3a" f4b="$R159/agent/f4b" \
  f3b="$R159/agent/f3b" f4ta="$R159/agent/f4ta" f4tb="$R159/agent/f4tb" f4tr="$R17/agent/f4tr"
table flint "$R159/agent.tsv" r159_agent v3p065 f4a f3a f4b f3b f4ta f4tb
# R17a: text reuse, beside the tuned preset's two runs
table flint "$R17/agent.tsv" r17_agent v3p065 f4ta f4tb f4tr
# Ornith: v3.0's reference run (arm ka of the MTP runs on rig 2, renamed v3ka), R160's six runs, R166's two, R169's two
merge ornith v3ka="$MTP/agent/ornith-ka:rel" o4a="$R160/agent/o4a" o4sa="$R160/agent/o4sa" o3a="$R160/agent/o3a" \
  o4b="$R160/agent/o4b" o4sb="$R160/agent/o4sb" o3b="$R160/agent/o3b" o4sna="$R166/agent/o4sna" \
  o4snb="$R166/agent/o4snb" o4nea="$R169/agent/o4nea" o4neb="$R169/agent/o4neb"
table ornith "$R160/agent.tsv" r160_agent v3ka o4a o4sa o3a o4b o4sb o3b
OUT_TAG=_o3aref table ornith "$R160/agent_o3aref.tsv" r160_agent_o3aref o3a o3b o4a o4b o4sa o4sb
table ornith "$R166/agent.tsv" r166_agent v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb
table ornith "$R169/agent.tsv" r169_agent v3ka o4a o4sa o3a o4b o4sb o3b o4sna o4snb o4nea o4neb
# Flint Pascal Q5 GGUF: R170
merge q5 v3q5="$RR/flint-pq5-v3base:rel" q5f4t="$R170/agent/q5f4t" q5f3="$R170/agent/q5f3"
table q5 "$R170/agent.tsv" r170_agent v3q5 q5f4t q5f3
OUT_TAG=_ref_q5f3 table q5 "$R170/agent_ref_q5f3.tsv" r170_agent_ref_q5f3 q5f3 q5f4t
# v3.0's text-reuse run on rig 2 (Unsloth Q8_0)
merge side v3base="$SIDE/agent/flint-v3base:rel" reread="$SIDE/agent/flint-reread:rel"
table side "$SIDE/agent_t2d.tsv" side_agent_t2d v3base v3base reread
# Ornith's MTP draft on rig 2 (NMAX=2; their acceptance came from server logs that are not published)
merge mtp ka="$MTP/agent/ornith-ka:rel" mtpka="$MTP/agent/ornith-mtpka:rel" mtpadapt="$MTP/agent/ornith-mtpadapt:rel"
NMAX=2 table mtp "$MTP/agent_part1.tsv" mtp_agent_part1 ka ka mtpka
NMAX=2 OUT_TAG=_9tasks SKIP_TASKS=bugfix-javascript-complex-numbers,bugfix-python-pig-latin,bugfix-javascript-variable-length-quantity \
  table mtp "$MTP/agent_part1_9tasks.tsv" mtp_agent_part1_9tasks ka ka mtpka
NMAX=2 table mtp "$MTP/agent_mtpadapt.tsv" mtp_agent_mtpadapt ka ka mtpka mtpadapt
NMAX=2 OUT_TAG=_11tasks SKIP_TASKS=bugfix-javascript-variable-length-quantity \
  table mtp "$MTP/agent_mtpadapt_11tasks.tsv" mtp_agent_mtpadapt_11tasks ka ka mtpadapt

# R159's class: the old and the new class_arms.py on the published table, then the new one on the new table
CLS=(f4t=f4ta,f4tb f4=f4a,f4b v3.0=f3a,f3b)
"${PY[@]}" "$OLD_C" "$R159/agent.tsv" "${CLS[@]}" > "$WORK/class_old.tsv"
same "R159's class.tsv from the published agent.tsv, old class_arms.py" "$WORK/class_old.tsv" "$R159/class.tsv"
"${PY[@]}" "$NEW_C" "$R159/agent.tsv" "${CLS[@]}" > "$WORK/class_new_pub.tsv"
same "R159's class.tsv from the published agent.tsv, new class_arms.py" "$WORK/class_new_pub.tsv" "$R159/class.tsv"
"${PY[@]}" "$NEW_C" "$T/r159_agent.tsv" "${CLS[@]}" > "$T/r159_class.tsv"
head -n "$(wc -l < "$R159/class.tsv")" "$T/r159_class.tsv" > "$WORK/class_new_head.tsv"
same "R159's class.tsv: the new class's first $(wc -l < "$R159/class.tsv") lines, from the new table" \
  "$WORK/class_new_head.tsv" "$R159/class.tsv"

# the classes the result page quotes: R17a's one run against the tuned preset's two, v4-rc2's preset against v3.0
# (Q1488: they sample the same tokens), and Ornith's pairs
"${PY[@]}" "$NEW_C" "$T/r17_agent.tsv" f4tr=f4tr f4t=f4ta,f4tb > "$T/r17_class.tsv"
"${PY[@]}" "$NEW_C" "$T/r159_agent.tsv" f4=f4a,f4b v3.0=f3a,f3b > "$T/r159_class_f4.tsv"
"${PY[@]}" "$NEW_C" "$T/r169_agent.tsv" o4s=o4sa,o4sb o4=o4a,o4b v3.0=o3a,o3b > "$T/r160_class.tsv"
"${PY[@]}" "$NEW_C" "$T/r169_agent.tsv" o4sn=o4sna,o4snb o4s=o4sa,o4sb > "$T/r166_class.tsv"
"${PY[@]}" "$NEW_C" "$T/r169_agent.tsv" o4ne=o4nea,o4neb o4sn=o4sna,o4snb > "$T/r169_class.tsv"
echo "tables and classes in ${T#"$ROOT/"}"
[ "$fail" = 0 ] && echo "every old column and class line the same" || echo "SOME OLD COLUMNS OR CLASS LINES DIFFER"
exit "$fail"
