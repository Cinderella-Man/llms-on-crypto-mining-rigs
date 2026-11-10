# The line with the draft (draft) against the line without it (nodraft): tables

The line for rounding: 0.05 in log-probability.

## Verdict per depth

- depth 0: a lead beyond rounding (0.6213 between the two candidates); teacher-forced, the largest lead a token of the draft line's reply loses is 0.3781
- depth 8,192: a lead beyond rounding (1.7798 between the two candidates); teacher-forced, the largest lead a token of the draft line's reply loses is 0.3936
- depth 16,384: a lead beyond rounding (0.4698 between the two candidates); teacher-forced, the largest lead a token of the draft line's reply loses is 0.0825
- depth 32,768: a lead beyond rounding (0.1086 between the two candidates); teacher-forced, the largest lead a token of the draft line's reply loses is 0.3640
- depth 49,152: a lead beyond rounding (0.2151 between the two candidates); teacher-forced, the largest lead a token of the draft line's reply loses is 0.9267

## Where the two replies part

| depth | tokens | first differing token | plain token id | plain piece | draft token id | draft piece | plain generating lp plain | plain generating lp draft | plain generating lead of plain | draft generating lp draft | draft generating lp plain | draft generating lead of draft | read back tokens | read back first choice | read back lp plain | read back lp draft | read back lead of plain | plain reply same as earlier run | draft reply same as earlier run | plain second request parts at | plain second request lead | draft second request parts at | draft second request lead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 256 | 7 | 9845 | " verify" | 1542 | " run" | -1.0734 | -1.2834 | 0.2100 | -1.1357 | -1.3389 | 0.2032 | 11 | draft | -1.6415 | -1.0201 | -0.6213 | yes | yes | nowhere |  | nowhere |  |
| 8192 | 256 | 2 | 1965 | " response" | 14408 | " listing" | -1.0346 | -2.8144 | 1.7798 | -1.3152 | -2.0688 | 0.7537 | 6 | plain | -0.9327 | -2.6012 | 1.6685 | no | no | nowhere |  | nowhere |  |
| 16384 | 256 | 17 | 1228 | " test" | 4220 | " whole" | -1.1623 | -1.3140 | 0.1516 | -1.1604 | -1.4569 | 0.2965 | 21 | plain | -1.0098 | -1.4796 | 0.4698 | no | no | nowhere |  | nowhere |  |
| 32768 | 256 | 26 | 353 | " I" | 561 | " The" | -1.0135 | -1.0828 | 0.0692 | -1.0428 | -1.1514 | 0.1086 | 30 | plain | -1.0277 | -1.1321 | 0.1044 | no | no | nowhere |  | nowhere |  |
| 49152 | 256 | 43 | 682 | " has" | 5533 | " uses" | -0.6262 | -0.7704 | 0.1443 | -0.6197 | -0.7778 | 0.1581 | 47 | plain | -0.5942 | -0.8093 | 0.2151 | no | no | nowhere |  | nowhere |  |

## Both replies teacher-forced on the plain line

| depth | reply | first token scored | tokens scored | end of text most probable | no probabilities | not first choice | above line | largest lead lost | at token | lead unknown | lead lost at first differing token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | nodraft | 0 | 256 | 0 | 0 | 0 | 0 |  |  | 0 | 0.0000 |
| 0 | draft | 7 | 249 | 0 | 0 | 5 | 3 | 0.3781 | 37 | 0 | 0.0000 |
| 8192 | nodraft | 0 | 256 | 2 | 0 | 4 | 3 | 0.5409 | 206 | 0 | 0.0000 |
| 8192 | draft | 2 | 254 | 0 | 1 | 5 | 5 | 0.3936 | 3 | 0 | 0.2052 |
| 16384 | nodraft | 0 | 256 | 5 | 0 | 3 | 3 | 0.5914 | 17 | 0 | 0.5914 |
| 16384 | draft | 17 | 239 | 4 | 0 | 3 | 2 | 0.0825 | 24 | 0 | 0.0208 |
| 32768 | nodraft | 0 | 256 | 4 | 0 | 5 | 5 | 0.4607 | 68 | 0 | 0.2733 |
| 32768 | draft | 26 | 230 | 1 | 0 | 10 | 10 | 0.3640 | 44 | 0 | 0.0000 |
| 49152 | nodraft | 0 | 256 | 0 | 1 | 5 | 4 | 0.5577 | 141 | 0 | 0.2750 |
| 49152 | draft | 43 | 213 | 0 | 1 | 3 | 3 | 0.9267 | 118 | 0 | 0.0000 |
| all | nodraft |  | 1280 | 11 | 1 | 17 | 15 | 0.5914 | 17 at depth 16384 | 0 |  |
| all | draft |  | 1185 | 5 | 2 | 26 | 23 | 0.9267 | 118 at depth 49152 | 0 |  |

## The same token after the same tokens, two paths

| depth | pair | tokens | exactly equal | median abs shift | p90 abs shift | max abs shift | at token | mean kld over listed |
|---|---|---|---|---|---|---|---|---|
| 0 | draft generating against nodraft generating, the common tokens | 7 | 1 | 0.0073 | 0.0138 | 0.3214 | 6 | 0.0038 |
| 0 | nodraft generating against nodraft teacher-forced, its own reply | 256 | 256 | 0.0000 | 0.0000 | 0.0000 | 255 | 0.0000 |
| 0 | draft generating against nodraft teacher-forced, the reply of draft | 249 | 6 | 0.0005 | 0.0925 | 0.5714 | 251 | 0.0055 |
| 8192 | draft generating against nodraft generating, the common tokens | 2 | 0 | 0.2773 | 0.1750 | 0.3797 | 1 | 0.1559 |
| 8192 | nodraft generating against nodraft teacher-forced, its own reply | 256 | 22 | 0.0000 | 0.0431 | 0.3806 | 206 | 0.0049 |
| 8192 | draft generating against nodraft teacher-forced, the reply of draft | 253 | 18 | 0.0000 | 0.0613 | 0.3788 | 92 | 0.0044 |
| 16384 | draft generating against nodraft generating, the common tokens | 17 | 0 | 0.0781 | 0.1738 | 0.3079 | 9 | 0.0263 |
| 16384 | nodraft generating against nodraft teacher-forced, its own reply | 256 | 44 | 0.0000 | 0.0641 | 1.5620 | 245 | 0.0057 |
| 16384 | draft generating against nodraft teacher-forced, the reply of draft | 239 | 46 | 0.0000 | 0.0154 | 1.2704 | 223 | 0.0011 |
| 32768 | draft generating against nodraft generating, the common tokens | 26 | 0 | 0.0493 | 0.1232 | 0.1796 | 15 | 0.0156 |
| 32768 | nodraft generating against nodraft teacher-forced, its own reply | 256 | 23 | 0.0007 | 0.1225 | 0.5297 | 128 | 0.0057 |
| 32768 | draft generating against nodraft teacher-forced, the reply of draft | 230 | 2 | 0.0108 | 0.1271 | 0.6286 | 217 | 0.0101 |
| 49152 | draft generating against nodraft generating, the common tokens | 43 | 1 | 0.0010 | 0.0323 | 0.6362 | 5 | 0.0108 |
| 49152 | nodraft generating against nodraft teacher-forced, its own reply | 255 | 19 | 0.0000 | 0.0630 | 0.5054 | 141 | 0.0049 |
| 49152 | draft generating against nodraft teacher-forced, the reply of draft | 212 | 25 | 0.0000 | 0.0402 | 0.6422 | 118 | 0.0056 |

## Every token that is not the plain line's first choice

| depth | reply | token | lead lost | note |
|---|---|---|---|---|
| 0 | draft | 27 | 0.1166 |  |
| 0 | draft | 37 | 0.3781 |  |
| 0 | draft | 39 | 0.2124 |  |
| 0 | draft | 64 | 0.0499 |  |
| 0 | draft | 125 | 0.0382 |  |
| 8192 | nodraft | 0 | 0.4449 |  |
| 8192 | nodraft | 27 | 0.0094 |  |
| 8192 | nodraft | 206 | 0.5409 |  |
| 8192 | nodraft | 208 | 0.1406 |  |
| 8192 | draft | 2 | 0.2052 |  |
| 8192 | draft | 3 | 0.3936 |  |
| 8192 | draft | 57 | 0.3005 |  |
| 8192 | draft | 92 | 0.1128 |  |
| 8192 | draft | 247 | 0.0876 |  |
| 16384 | nodraft | 9 | 0.5642 |  |
| 16384 | nodraft | 17 | 0.5914 |  |
| 16384 | nodraft | 23 | 0.1848 |  |
| 16384 | draft | 17 | 0.0208 |  |
| 16384 | draft | 24 | 0.0825 |  |
| 16384 | draft | 201 | 0.0620 |  |
| 32768 | nodraft | 26 | 0.2733 |  |
| 32768 | nodraft | 56 | 0.2014 |  |
| 32768 | nodraft | 68 | 0.4607 |  |
| 32768 | nodraft | 147 | 0.2372 |  |
| 32768 | nodraft | 198 | 0.1745 |  |
| 32768 | draft | 33 | 0.1338 |  |
| 32768 | draft | 44 | 0.3640 |  |
| 32768 | draft | 47 | 0.0790 |  |
| 32768 | draft | 49 | 0.1454 |  |
| 32768 | draft | 98 | 0.0638 |  |
| 32768 | draft | 115 | 0.1970 |  |
| 32768 | draft | 147 | 0.3398 |  |
| 32768 | draft | 196 | 0.0614 |  |
| 32768 | draft | 209 | 0.1118 |  |
| 32768 | draft | 217 | 0.0779 |  |
| 49152 | nodraft | 43 | 0.2750 |  |
| 49152 | nodraft | 73 | 0.2206 |  |
| 49152 | nodraft | 141 | 0.5577 |  |
| 49152 | nodraft | 152 | 0.0062 |  |
| 49152 | nodraft | 188 | 0.0808 |  |
| 49152 | draft | 55 | 0.5913 |  |
| 49152 | draft | 118 | 0.9267 |  |
| 49152 | draft | 213 | 0.5351 |  |
