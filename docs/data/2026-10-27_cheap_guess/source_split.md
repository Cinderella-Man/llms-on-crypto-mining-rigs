# The head's acceptance by guess source (held-out sessions)

`source_split.py` on the 40 held-out sessions of agentic run 1 (680,600 scored positions), the nine-card head after
240 minutes, inverse-CDF coupling (the served sampler's, metric `uniform`). Source: what llama-server's lookups would
guess at the position, in their order (the copy lookup, a continuation of a 2-token or longer context, of a 1-token
context, none), computed on the true prefix; "lookup right": how often that lookup's guess is the token.

Window g9 (the head right after card 1's stage on nine cards):

| source | positions | share | head right | lookup right |
|---|---|---|---|---|
| copy | 354878 | 0.521 | 0.310 | 0.815 |
| cont2 | 167738 | 0.246 | 0.266 | 0.434 |
| cont1 | 128194 | 0.188 | 0.244 | 0.191 |
| none | 29790 | 0.044 | 0.273 | 0.000 |
| not copy (the old order's head positions) | 325722 | 0.479 | 0.258 | |

Window g9miss (the first guess after a miss: one token in flight):

| source | positions | share | head right | lookup right |
|---|---|---|---|---|
| copy | 354878 | 0.521 | 0.283 | 0.815 |
| cont2 | 167738 | 0.246 | 0.243 | 0.434 |
| cont1 | 128194 | 0.188 | 0.220 | 0.191 |
| none | 29790 | 0.044 | 0.248 | 0.000 |
| not copy (the old order's head positions) | 325722 | 0.479 | 0.235 | |

The head in place of the lookup wherever its top probability (at the scored alpha) is at least t, else the lookup,
window g9: right guesses per position (the share of positions the head takes):

| source | t 0 | t 0.3 | t 0.5 | t 0.6 | t 0.7 | t 0.8 | t 0.9 | t 0.95 | t 1.01 |
|---|---|---|---|---|---|---|---|---|---|
| copy | 0.310 (1.00) | 0.503 (0.70) | 0.629 (0.50) | 0.681 (0.41) | 0.722 (0.34) | 0.765 (0.26) | 0.793 (0.19) | 0.805 (0.14) | 0.815 (0.00) |
| cont2 | 0.266 (1.00) | 0.387 (0.52) | 0.423 (0.31) | 0.432 (0.24) | 0.436 (0.19) | 0.439 (0.14) | 0.440 (0.10) | 0.439 (0.07) | 0.434 (0.00) |
| cont1 | 0.244 (1.00) | 0.265 (0.46) | 0.253 (0.25) | 0.245 (0.19) | 0.236 (0.14) | 0.227 (0.10) | 0.216 (0.07) | 0.208 (0.05) | 0.191 (0.00) |
