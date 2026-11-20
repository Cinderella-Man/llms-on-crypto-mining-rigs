| half | load, s | rounds | the stock answer | first differing round | minutes after the load | the rounds' time, s | generation tok/s, median of rounds 2 on (prompts 1, 2) | against stock's round 2 |
|---|---:|---:|---:|---|---:|---:|---|---|
| a | 673 | 56 | 56 of 56 | none |  | 1831 | 17.34, 17.34 (55 rounds) | +4.21 %, +4.21 % |
| b | 672 | 55 | 55 of 55 | none |  | 1832 | 17.04, 17.04 (54 rounds) | +2.40 %, +2.40 % |
| full90 | 671 | 137 | 136 of 137 | round 137 (`0e9263fcf6d12ba1`) | 72.5 | 4358 | 17.79, 17.79 (136 rounds) | +6.91 %, +6.91 % |

The stock reference is `adf47784f9b02d4b`, stock's round 1 in 2026-11-20_r224_apply_rig1/model_rounds.tsv; stock's round 2 generated 16.64 and 16.64 tok/s.
