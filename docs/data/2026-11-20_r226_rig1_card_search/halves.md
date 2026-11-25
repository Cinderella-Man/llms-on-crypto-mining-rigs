| half | load, s | rounds | the stock answer | first differing round | minutes after the load | the rounds' time, s | generation tok/s, median of rounds 2 on (prompts 1, 2) | against stock's round 2 |
|---|---:|---:|---:|---|---:|---:|---|---|
| a | 673 | 56 | 56 of 56 | none |  | 1831 | 17.34, 17.34 (55 rounds) | +4.21 %, +4.21 % |
| b | 672 | 55 | 55 of 55 | none |  | 1832 | 17.04, 17.04 (54 rounds) | +2.40 %, +2.40 % |
| cores120 | 675 | 5 | 4 of 5 | round 5 (`6f45f3ece7f7d06a`) | 2.8 | 174 | 17.23, 17.23 (4 rounds) | +3.55 %, +3.55 % |
| full90 | 671 | 137 | 136 of 137 | round 137 (`0e9263fcf6d12ba1`) | 72.5 | 4358 | 17.79, 17.79 (136 rounds) | +6.91 %, +6.91 % |
| mem120 | 675 | 22 | 21 of 22 | round 22 (`86422e3068f83d65`) | 11.8 | 712 | 17.70, 17.69 (21 rounds) | +6.37 %, +6.31 % |
| memalone120 | 675 | 217 | 217 of 217 | none |  | 7226 | 17.01, 17.01 (216 rounds) | +2.22 %, +2.22 % |

The stock reference is `adf47784f9b02d4b`, stock's round 1 in 2026-11-20_r224_apply_rig1/model_rounds.tsv; stock's round 2 generated 16.64 and 16.64 tok/s.
