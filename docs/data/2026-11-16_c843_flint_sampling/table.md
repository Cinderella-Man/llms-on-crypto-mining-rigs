# C843: the arms against `v3`, request by request

Each row is one request (a recorded agent request, label, at a fixed seed; `rep` 1 a repeat sent again at the end of the load). For each arm: the reply's length in generated tokens and how it ended; "same" when its token ids are those of `v3`'s reply to the same request, else the first token (0-based) where they differ.

| request | seed | rep | prompt tokens | `v3` | `v4` | `v4n` |
|---|---|---|---|---|---|---|
| A1 | 11 | 0 | 1773 | 36, tool_calls (1) | 36, tool_calls (1), same | 36, tool_calls (1), same |
| A1 | 12 | 0 | 1773 | 28, tool_calls (2) | 28, tool_calls (2), same | 28, tool_calls (2), same |
| A1 | 13 | 0 | 1773 | 28, tool_calls (2) | 28, tool_calls (2), same | 28, tool_calls (2), same |
| A2 | 11 | 0 | 2005 | 27, tool_calls (1) | 27, tool_calls (1), same | 27, tool_calls (1), same |
| A2 | 12 | 0 | 2005 | 37, tool_calls (2) | 37, tool_calls (2), same | 37, tool_calls (2), same |
| A2 | 13 | 0 | 2005 | 33, tool_calls (2) | 33, tool_calls (2), same | 33, tool_calls (2), same |
| A3 | 11 | 0 | 1709 | 13, tool_calls (1) | 13, tool_calls (1), same | 13, tool_calls (1), same |
| A3 | 12 | 0 | 1709 | 26, tool_calls (2) | 26, tool_calls (2), same | 26, tool_calls (2), same |
| A3 | 13 | 0 | 1709 | 21, tool_calls (2) | 21, tool_calls (2), same | 21, tool_calls (2), same |
| A4 | 11 | 0 | 2313 | 34, tool_calls (2) | 34, tool_calls (2), same | 34, tool_calls (2), same |
| A4 | 12 | 0 | 2313 | 14, tool_calls (1) | 14, tool_calls (1), same | 14, tool_calls (1), same |
| A4 | 13 | 0 | 2313 | 45, tool_calls (2) | 45, tool_calls (2), same | 45, tool_calls (2), same |
| B1 | 11 | 0 | 3567 | 2048, length | 2048, length, same | 2048, length, same |
| B1 | 12 | 0 | 3567 | 1461, tool_calls (1) | 1461, tool_calls (1), same | 1461, tool_calls (1), same |
| B1 | 13 | 0 | 3567 | 2048, length | 2048, length, same | 2048, length, same |
| B2 | 11 | 0 | 2878 | 24, tool_calls (1) | 24, tool_calls (1), same | 24, tool_calls (1), same |
| B2 | 12 | 0 | 2878 | 15, tool_calls (1) | 15, tool_calls (1), same | 15, tool_calls (1), same |
| B2 | 13 | 0 | 2878 | 28, tool_calls (1) | 28, tool_calls (1), same | 28, tool_calls (1), same |
| B3 | 11 | 0 | 3067 | 24, tool_calls (1) | 24, tool_calls (1), same | 24, tool_calls (1), same |
| B3 | 12 | 0 | 3067 | 24, tool_calls (1) | 24, tool_calls (1), same | 24, tool_calls (1), same |
| B3 | 13 | 0 | 3067 | 2048, length | 2048, length, same | 2048, length, same |
| B4 | 11 | 0 | 3684 | 2048, length | 2048, length, same | 2048, length, same |
| B4 | 12 | 0 | 3684 | 2048, length | 2048, length, same | 2048, length, same |
| B4 | 13 | 0 | 3684 | 2048, length | 2048, length, same | 2048, length, same |
| A1 | 11 | 1 | 1773 | 36, tool_calls (1) | 36, tool_calls (1), same | 36, tool_calls (1), same |
| B1 | 11 | 1 | 3567 | 2048, length | 2048, length, same | 2048, length, same |

## Totals against the control

| arm | requests | the same token ids | tokens compared | tokens the same | same length | same end | same prompt |
|---|---|---|---|---|---|---|---|
| `v4` | 26 | 26 | 16290 | 16290 | 26 | 26 | 26 |
| `v4n` | 26 | 26 | 16290 | 16290 | 26 | 26 | 26 |

## Repeats within a load

| arm | request | seed | the same token ids as the load's first reply |
|---|---|---|---|
| `v3` | A1 | 11 | yes |
| `v3` | B1 | 11 | yes |
| `v4` | A1 | 11 | yes |
| `v4` | B1 | 11 | yes |
| `v4n` | A1 | 11 | yes |
| `v4n` | B1 | 11 | yes |

## The sampler path, from the server log

| arm | requests | tokens | drawn on the device | while the grammar applied | rejected (after a device draw) | forced | pipelined guesses, right, wrong |
|---|---|---|---|---|---|---|---|
| `v3` | 26 | 487 | 487 | 461 | 0 (0) | 0 | 41318, 5285, 9011 |
| `v4` | 26 | 487 | 487 | 461 | 0 (0) | 0 | 41318, 5285, 9011 |
| `v4n` | 26 | 487 | 487 | 461 | 0 (0) | 0 | 41318, 5285, 9011 |

## Settings each reply reports

| arm | seeds echoed | top-k, top-p, min-p, temperature | samplers | backend sampling | chat format | lazy grammar | grammars |
|---|---|---|---|---|---|---|---|
| `v3` | all | [(20, 0.949999988079071, 0.0, 1.0)] | ['penalties dry top_n_sigma top_k typ_p top_p min_p xtc temperature'] | ['True'] | ['peg-native'] | ['True'] | 1 distinct |
| `v4` | all | [(20, 0.949999988079071, 0.0, 1.0)] | ['penalties dry top_n_sigma top_k typ_p top_p min_p xtc temperature'] | ['True'] | ['peg-native'] | ['True'] | 1 distinct |
| `v4n` | all | [(20, 0.949999988079071, 0.0, 1.0)] | ['penalties dry top_n_sigma top_k typ_p top_p min_p xtc temperature'] | ['True'] | ['peg-native'] | ['True'] | 1 distinct |
