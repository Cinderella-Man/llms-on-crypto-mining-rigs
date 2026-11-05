#!/usr/bin/env bash
# acceptance.sh <server-ARM.log> ...: the draft's acceptance and tokens per step over a whole agent-check session, from
# llama-server's per-request lines ("draft acceptance = a (A accepted / G generated), mean len = L"). Steps per request
# are A / (L - 1). The server counts only the steps that drafted: an adaptive arm's width-1 steps (no draft) are in
# its step log (adapt_log.py), not here. Prints per log: requests, accepted, drafted, acceptance, steps, tokens per step.
for f in "$@"; do
  grep 'draft acceptance' "$f" \
    | sed -E 's/.*\( *([0-9]+) accepted \/ *([0-9]+) generated\), mean len = *([0-9.]+).*/\1 \2 \3/' \
    | awk -v f="$(basename "$f")" '{ a += $1; g += $2; if ($3 > 1) n += $1 / ($3 - 1) }
        END { printf "%s\t%d\t%d\t%d\t%.3f\t%.0f\t%.3f\n", f, NR, a, g, a / g, n, (a + n) / n }'
done
