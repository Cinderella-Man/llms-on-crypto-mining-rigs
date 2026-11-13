#!/usr/bin/env bash
# R163 (Q1483): tokens a step and ms a step from the server logs, counted per reply. The agent-check tables give
# tokens a step as 1 + 3 x acceptance (a fixed draft of 3 tokens), which is wrong for a draft of 4. Here each reply's
# steps are its generated tokens over its "mean len" (the server's tokens a step for that reply), so a draft of any
# length counts as it ran. ms a step is the replies' eval time over those steps (est.: the eval time holds the
# drafting passes, the verify passes and the sampling); drafted per step is the drafted tokens over the steps.
# Usage: bash steps.sh <folder with server-<arm>.log> <arm>... > steps.tsv
set -euo pipefail
dir=$1; shift
printf 'arm\treplies\tgenerated_tokens\teval_ms\tsteps\ttokens_per_step\tms_per_step\tgen_tok_s\taccepted\tdrafted\tacceptance\tdrafted_per_step\n'
for a in "$@"; do
  nice -n 19 awk -v arm="$a" '
    /print_timing/ && / eval time =/ && !/prompt eval/ {
      match($0, /task +[0-9]+/); t = substr($0, RSTART, RLENGTH)
      split($0, x, "eval time ="); split(x[2], y, " "); ms[t] = y[1]; n[t] = y[4]; order[++k] = t }
    /draft acceptance =/ {
      match($0, /task +[0-9]+/); t = substr($0, RSTART, RLENGTH)
      match($0, /mean len = +[0-9.]+/); m = substr($0, RSTART, RLENGTH); sub(/mean len = +/, "", m); ml[t] = m
      match($0, /\( *[0-9]+ accepted/); s = substr($0, RSTART + 1); split(s, z, " "); acc += z[1]
      match($0, /\/ *[0-9]+ generated/); s = substr($0, RSTART + 1); split(s, w, " "); gen += w[1] }
    END {
      for (i = 1; i <= k; i++) { t = order[i]; T += n[t]; M += ms[t]; S += (ml[t] > 0) ? n[t] / ml[t] : n[t] }
      printf "%s\t%d\t%d\t%.1f\t%.1f\t%.4f\t%.2f\t%.2f\t%d\t%d\t%.4f\t%.3f\n", arm, k, T, M, S, T / S, M / S, 1000 * T / M,
        acc, gen, acc / gen, gen / S }' "$dir/server-$a.log"
done
