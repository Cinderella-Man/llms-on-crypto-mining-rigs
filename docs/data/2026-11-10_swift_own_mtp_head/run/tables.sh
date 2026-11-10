#!/usr/bin/env bash
# tables.sh: the tables of the own-head run (registry Q1168), made with pv_table.py of
# docs/data/2026-11-10_prob_verify_acceptance (Q1173), whose four loads of the draft in use are the other side of
# each pair.
#   WORK=<run_own_head.sh's work folder> PRIOR=<a folder with the earlier run's replay-pv0-{1,2}.jsonl,
#   replay-pv1-{1,2}.jsonl and server-*.log of the same loads> REPO=<checkout of this repository> bash tables.sh
# pv_table.py compares every arm of a folder with one control, so the loads are linked into one folder per
# comparison (WORK/cmp/<name>), each with its tables in WORK/tables/<name>:
#   all    the nine loads, control pv0: every load's counters; own-pv0 and gyro-pv0 against the earlier pv0 loads
#   pre3   own-pv0 against the preset's three loads: the earlier two and this run's control (linked as pv0-3)
#   night  own-pv0 against this run's control alone
#   pv1    own-pv1 against the earlier pv1 loads
#   own    own-pv1 against own-pv0: probabilistic verification on the new head
# WORK/tables/compare.tsv holds the rows this experiment asks for under one header, the comparison's name first.
set -euo pipefail
: "${WORK:?}" "${PRIOR:?}" "${REPO:?}"
PV=$REPO/docs/data/2026-11-10_prob_verify_acceptance/run/pv_table.py
link() {  # link NAME FILE...: the loads of one comparison
  local name=$1 f; shift
  rm -rf "$WORK/cmp/$name"; mkdir -p "$WORK/cmp/$name"
  for f in "$@"; do ln -s "$f" "$WORK/cmp/$name/"; done
}
link all "$PRIOR"/replay-pv[01]-[12].jsonl "$PRIOR"/server-pv[01]-[12].log "$WORK"/replay-*.jsonl "$WORK"/server-*.log
link pre3 "$PRIOR"/replay-pv0-[12].jsonl "$PRIOR"/server-pv0-[12].log "$WORK"/replay-own-pv0-*.jsonl "$WORK"/server-own-pv0-*.log
ln -s "$WORK/replay-gyro-pv0-1.jsonl" "$WORK/cmp/pre3/replay-pv0-3.jsonl"
ln -s "$WORK/server-gyro-pv0-1.log" "$WORK/cmp/pre3/server-pv0-3.log"
link night "$WORK"/replay-gyro-pv0-1.jsonl "$WORK"/server-gyro-pv0-1.log "$WORK"/replay-own-pv0-*.jsonl "$WORK"/server-own-pv0-*.log
link pv1 "$PRIOR"/replay-pv1-[12].jsonl "$PRIOR"/server-pv1-[12].log "$WORK"/replay-own-pv1-*.jsonl "$WORK"/server-own-pv1-*.log
link own "$WORK"/replay-own-*.jsonl "$WORK"/server-own-*.log
for c in all:pv0 pre3:pv0 night:gyro-pv0 pv1:pv1 own:own-pv0; do
  nice -n 19 python3 -I "$PV" "$WORK/cmp/${c%%:*}" --control "${c#*:}" --out "$WORK/tables/${c%%:*}" \
    > "$WORK/tables.${c%%:*}.console.txt"
done
rows() {  # rows NAME AWK-CONDITION: the comparison's rows, its name in front
  awk -F'\t' -v OFS='\t' -v c="$1" "$2 { print c, \$0 }" "$WORK/tables/$1/compare.tsv"
}
{
  printf 'comparison\t'; head -n 1 "$WORK/tables/all/compare.tsv"
  rows all '$1 == "arm against the control" && ($2 == "gyro-pv0" || $2 == "own-pv0")'
  rows pre3 '$1 == "arm against the control"'
  rows night '$1 == "arm against the control"'
  rows pv1 '$1 == "arm against the control" && $2 == "own-pv1"'
  rows own '$1 == "arm against the control"'
  rows all '$1 == "second load against the first" && $2 ~ /^own/'
  rows pre3 '$1 == "second load against the first" && $2 == "pv0-2"'
} > "$WORK/tables/compare.tsv"
# by_position.tsv: the verify steps of a three-token draft by how many tokens were kept, and the share of the drafts
# whose first, second and third token was accepted (a token counts only when every token before it was accepted)
python3 -I - "$WORK/tables/all/steps.tsv" > "$WORK/tables/by_position.tsv" << 'EOF'
import collections, sys
steps = collections.defaultdict(lambda: [0, 0, 0, 0])
order = []
for line in list(open(sys.argv[1]))[1:]:
    load, drafted, accepted, _restore, n = line.split("\t")
    if load not in order:
        order.append(load)
    if int(drafted) == 3:
        steps[load][int(accepted)] += int(n)
print("load\tsteps_of_3\tkept_0\tkept_1\tkept_2\tkept_3\tfirst_accepted\tsecond_accepted\tthird_accepted\t"
      "second_given_first\tthird_given_second")
for load in order:
    k = steps[load]
    n = sum(k)
    p1, p2, p3 = (k[1] + k[2] + k[3]) / n, (k[2] + k[3]) / n, k[3] / n
    print(f"{load}\t{n}\t{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{p1:.4f}\t{p2:.4f}\t{p3:.4f}\t{p2 / p1:.4f}\t{p3 / p2:.4f}")
EOF
cat "$WORK/tables/all/loads.tsv"
cut -f 1,3-10,14-19,21-26 "$WORK/tables/compare.tsv"
cat "$WORK/tables/by_position.tsv"
