# The price of a first-finish race and of a length cap on these replies

Tokens of whole replies (reasoning and answer). A prompt's samples are its replies in every load of the
arm. "Share" is against one sample (N = 1).

## First finisher of N samples of one request

| Arm | Prompts | Samples a prompt | N = 1 | N = 2 | N = 3 | N = 4 | N = 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pv0` | 12 | 10 | 15796 (1.000) | 9814 (0.621) | 7860 (0.498) | 6891 (0.436) | 6296 (0.399) |
| `pv1` | 12 | 10 | 16823 (1.000) | 9979 (0.593) | 7944 (0.472) | 7023 (0.417) | 6454 (0.384) |

Expected tokens summed over the prompts, with the share in brackets.

## Second finisher of N (a vote of two needs it)

| Arm | N = 2 | N = 3 | N = 4 | N = 5 |
|---|---:|---:|---:|---:|
| `pv0` | 21779 (1.379) | 13720 (0.869) | 10769 (0.682) | 9271 (0.587) |
| `pv1` | 23668 (1.407) | 14050 (0.835) | 10705 (0.636) | 9300 (0.553) |

## Per prompt, every arm's samples together

| Prompt | Samples | Shortest | Median | Mean | Longest |
|---|---:|---:|---:|---:|---:|
| bugfix-javascript-variable-length-quantity | 20 | 1073 | 4044.5 | 5112.3 | 13639 |
| bugfix-python-scale-generator | 20 | 1244 | 2421.0 | 4503.4 | 19994 |
| bugfix-python-variable-length-quantity | 20 | 770 | 1792.5 | 2603.2 | 7452 |
| bugfix-python-pig-latin | 20 | 367 | 1055.5 | 1250.6 | 3601 |
| bugfix-python-list-ops | 20 | 445 | 887.0 | 908.8 | 1681 |
| bugfix-python-affine-cipher | 20 | 205 | 629.5 | 896.0 | 3504 |
| bugfix-javascript-phone-number | 20 | 119 | 248.5 | 249.7 | 443 |
| bugfix-python-phone-number | 20 | 96 | 189.0 | 231.7 | 466 |
| bugfix-javascript-triangle | 20 | 134 | 193.0 | 206.8 | 344 |
| bugfix-javascript-space-age | 20 | 124 | 164.0 | 192.9 | 398 |
| bugfix-javascript-grade-school | 20 | 87 | 114.5 | 112.8 | 161 |
| bugfix-javascript-complex-numbers | 20 | 38 | 38.0 | 41.9 | 101 |

## A cap on a reply's length

| Cap, tokens | Tokens kept, share of all | Replies over the cap |
|---:|---:|---:|
| 1000 | 0.390 | 79 of 240 |
| 2000 | 0.563 | 40 of 240 |
| 4000 | 0.744 | 19 of 240 |
| 8000 | 0.897 | 7 of 240 |

A capped reply is counted at the cap: what the model writes after a forced end of its reasoning is not
in these numbers.
