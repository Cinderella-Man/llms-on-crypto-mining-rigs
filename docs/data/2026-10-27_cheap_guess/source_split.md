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
