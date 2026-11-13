#!/usr/bin/env bash
# draft_widths.sh: R154's draft table (Q1465) from one llama-server log of an arm run with LLAMA_SPEC_ADAPT=1. Each
# request's print_timing lines give its draft acceptance (accepted / drafted tokens, the drafted ones being those the
# verify pass checked, after the rule's cut), its graph reallocations, and the adaptive rule's steps by verify width
# ("adaptive draft (on) = N drafted steps, ..., steps by verify width (1: a, 3: b, ...)"). It prints one row per request
# and a total row: steps, tokens a step ((accepted + steps) / steps: the accepted draft tokens plus the verify pass's own
# token), drafted tokens a step, and the share of steps at each verify width 1 to 8. A log without the adaptive lines
# (a fixed draft length) gives the acceptance and reallocations only. Run on any CPU at nice 19:
#   nice -n 19 bash draft_widths.sh <server log> > draft_widths.tsv
set -euo pipefail
LOG=${1:?server log}
awk '
function num(re,   s) { if (!match($0, re)) return 0; s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); return s + 0 }
function task(   s) { match($0, /task [0-9]+ \|/); s = substr($0, RSTART + 5, RLENGTH - 7); return s }
BEGIN { OFS = "\t"; print "request", "steps", "accepted", "drafted", "acceptance", "tokens_a_step", "drafted_a_step",
        "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "graph_reallocs" }
/graph reallocs =/ { t = task(); r = $0; sub(/.*graph reallocs = */, "", r); split(r, rr, "/"); re[t] = (rr[1] + 0) + (rr[2] + 0) }
/draft acceptance = / { t = task(); acc[t] = num("\\( *[0-9]+ accepted"); dr[t] = num("/ *[0-9]+ generated");
                        if (!(t in seen)) { seen[t] = 1; order[++n] = t } }
/adaptive draft \(on\)/ { t = task(); st[t] = num("= [0-9]+ drafted steps"); w = $0; sub(/.*steps by verify width \(/, "", w);
                          sub(/\).*/, "", w); k = split(w, p, ", "); for (i = 1; i <= k; i++) { split(p[i], q, ": "); mix[t, q[1] + 0] = q[2] + 0 } }
function row(name, s, a, d, r, m,   i, line) {
  line = name OFS s OFS a OFS d OFS (d ? sprintf("%.4f", a / d) : "") OFS (s ? sprintf("%.3f", (a + s) / s) : "") OFS (s ? sprintf("%.3f", d / s) : "")
  for (i = 1; i <= 8; i++) line = line OFS m[i]
  print line, r
}
END {
  for (j = 1; j <= n; j++) { t = order[j]; delete m; for (i = 1; i <= 8; i++) { m[i] = mix[t, i] + 0; tm[i] += m[i] }
    row(t, st[t] + 0, acc[t], dr[t], re[t] + 0, m); S += st[t]; A += acc[t]; D += dr[t]; R += re[t] }
  row("all", S, A, D, R, tm)
  delete m; for (i = 1; i <= 8; i++) m[i] = S ? sprintf("%.1f %%", 100 * tm[i] / S) : ""
  row("share", "", "", "", "", m)
}' "$LOG"
