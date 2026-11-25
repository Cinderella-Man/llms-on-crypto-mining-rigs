| Request | Depth | New tokens per load | `c1` loads: accepted/drafted, steps, ms a step | `s0` loads | Same reply in all loads |
|---|---:|---|---|---|---|
| greedy, pass 1 | 0 | 7, 7, 7, 7 | c1a 148/318, 108, 100.82; c1b 148/318, 108, 101.85 | s0a 148/318, 108, 101.38; s0b 148/318, 108, 101.43 | yes |
| greedy, pass 2 | 0 | 0, 0, 0, 0 | c1a 149/315, 107, 97.88; c1b 149/315, 107, 98.04 | s0a 149/315, 107, 98.59; s0b 149/315, 107, 98.12 | yes |
| sampled | 0 | 9, 9, 9, 9 | c1a 145/329, 111, 101.83; c1b 145/329, 111, 102.42 | s0a 145/329, 111, 101.76; s0b 145/329, 111, 101.73 | yes |
| greedy, pass 1 | 4,096 | 1, 1, 1, 1 | c1a 177/231, 79, 113.16; c1b 177/231, 79, 112.89 | s0a 177/231, 79, 116.71; s0b 177/231, 79, 123.70 | yes |
| greedy, pass 2 | 4,096 | 0, 0, 0, 0 | c1a 177/231, 79, 108.78; c1b 177/231, 79, 108.82 | s0a 177/231, 79, 108.70; s0b 177/231, 79, 109.01 | yes |
| sampled | 4,096 | 3, 3, 3, 3 | c1a 179/228, 77, 110.75; c1b 179/228, 77, 110.28 | s0a 178/230, 78, 111.50; s0b 178/230, 78, 111.80 | no |
| greedy, pass 1 | 8,192 | 14, 14, 14, 14 | c1a 173/245, 83, 119.42; c1b 173/245, 83, 119.52 | s0a 170/254, 86, 117.74; s0b 170/254, 86, 117.87 | yes |
| greedy, pass 2 | 8,192 | 0, 0, 0, 0 | c1a 178/230, 78, 113.34; c1b 178/230, 78, 113.24 | s0a 178/230, 78, 113.22; s0b 178/230, 78, 113.46 | yes |
| sampled | 8,192 | 1, 1, 1, 1 | c1a 181/220, 75, 120.07; c1b 181/220, 75, 115.32 | s0a 181/220, 75, 115.24; s0b 181/220, 75, 116.13 | yes |

| Load | Arm | New tokens in all replies (warm-up and primes too) | Of them `<…>` tokens | Server: target tokens outside the list | Server: drafts with the full head |
|---|---|---:|---:|---:|---:|
| c1a | c1 | 37 | 2 | 31 | 31 |
| c1b | c1 | 37 | 2 | 31 | 31 |
| s0a | s0 | 37 | 2 | 35 | 0 |
| s0b | s0 | 37 | 2 | 35 | 0 |

| Request | Depth | Load | New tokens: place in the reply and text (the first eight) |
|---|---:|---|---|
| greedy, pass 1 | 0 | c1a | 46: `\u27e9`, 50: ` \u2713`, 163: ` appended`, 171: `_chat`, 211: `</think>`, 213: `<tool_call>`, 217: `=B` |
| greedy, pass 1 | 0 | c1b | 46: `\u27e9`, 50: ` \u2713`, 163: ` appended`, 171: `_chat`, 211: `</think>`, 213: `<tool_call>`, 217: `=B` |
| greedy, pass 1 | 0 | s0a | 46: `\u27e9`, 50: ` \u2713`, 163: ` appended`, 171: `_chat`, 211: `</think>`, 213: `<tool_call>`, 217: `=B` |
| greedy, pass 1 | 0 | s0b | 46: `\u27e9`, 50: ` \u2713`, 163: ` appended`, 171: `_chat`, 211: `</think>`, 213: `<tool_call>`, 217: `=B` |
| sampled | 0 | c1a | 22: ` `_`, 25: ``:`, 71: `)``, 101: `(seq`, 102: `_lens`, 162: ` \u2264`, 167: ` enforced`, 217: `_cos` |
| sampled | 0 | c1b | 22: ` `_`, 25: ``:`, 71: `)``, 101: `(seq`, 102: `_lens`, 162: ` \u2264`, 167: ` enforced`, 217: `_cos` |
| sampled | 0 | s0a | 22: ` `_`, 25: ``:`, 71: `)``, 101: `(seq`, 102: `_lens`, 162: ` \u2264`, 167: ` enforced`, 217: `_cos` |
| sampled | 0 | s0b | 22: ` `_`, 25: ``:`, 71: `)``, 101: `(seq`, 102: `_lens`, 162: ` \u2264`, 167: ` enforced`, 217: `_cos` |
| greedy, pass 1 | 4,096 | c1a | 22: ` oddly` |
| greedy, pass 1 | 4,096 | c1b | 22: ` oddly` |
| greedy, pass 1 | 4,096 | s0a | 22: ` oddly` |
| greedy, pass 1 | 4,096 | s0b | 22: ` oddly` |
| sampled | 4,096 | c1a | 63: `(py`, 206: `/site`, 207: `-packages` |
| sampled | 4,096 | c1b | 63: `(py`, 206: `/site`, 207: `-packages` |
| sampled | 4,096 | s0a | 63: `(py`, 206: `/site`, 207: `-packages` |
| sampled | 4,096 | s0b | 63: `(py`, 206: `/site`, 207: `-packages` |
| greedy, pass 1 | 8,192 | c1a | 5: `bled`, 7: `uncated`, 35: ` summarize`, 52: `Bug`, 53: `:**`, 104: `\ufffd`, 106: `\u00b9`, 131: ` multiplying` |
| greedy, pass 1 | 8,192 | c1b | 5: `bled`, 7: `uncated`, 35: ` summarize`, 52: `Bug`, 53: `:**`, 104: `\ufffd`, 106: `\u00b9`, 131: ` multiplying` |
| greedy, pass 1 | 8,192 | s0a | 5: `bled`, 7: `uncated`, 35: ` summarize`, 52: `Bug`, 53: `:**`, 104: `\ufffd`, 106: `\u00b9`, 131: ` multiplying` |
| greedy, pass 1 | 8,192 | s0b | 5: `bled`, 7: `uncated`, 35: ` summarize`, 52: `Bug`, 53: `:**`, 104: `\ufffd`, 106: `\u00b9`, 131: ` multiplying` |
| sampled | 8,192 | c1a | 9: `rupted` |
| sampled | 8,192 | c1b | 9: `rupted` |
| sampled | 8,192 | s0a | 9: `rupted` |
| sampled | 8,192 | s0b | 9: `rupted` |
