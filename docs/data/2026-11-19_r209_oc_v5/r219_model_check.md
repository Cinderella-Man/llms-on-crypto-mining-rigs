| | stock, earlier boot | with the offsets, earlier boot | with the offsets, previous later boot | **with the offsets, this boot** | this boot against the earlier boot's offsets | this boot against the previous later boot | this boot against stock |
|---|---|---|---|---|---|---|---|
| load, s | 658 | 663 | 543 | **541** | |  | |
| host memory available when loaded, MiB | 1,496 | 1,492 | 1,502 | **1,492** | |  | |
| MiB used, cards 0 to 8 | 5603 5635 5549 5635 5243 5351 5513 5601 5009 | 5603 5635 5549 5635 5243 5351 5513 5601 5009 | 5603 5635 5549 5635 5243 5351 5513 5601 5009 | **5603 5635 5549 5635 5243 5351 5513 5601 5009** | |  | |
| rounds; the stock hash | 2; `adf47784f9b02d4b` | 58; 58 of 58 | 2; 2 of 2 | **2; 2 of 2** | |  | |
| hash in each round | `adf47784f9b02d4b`, `adf47784f9b02d4b` | `adf47784f9b02d4b` in every round | `adf47784f9b02d4b`, `adf47784f9b02d4b` | **`adf47784f9b02d4b`, `adf47784f9b02d4b`** | |  | |
| generation tok/s in round 1 (prompts 1, 2) | 15.27, 16.15 | 16.42, 17.44 | 16.54, 17.47 | **16.61, 17.45** | +1.16 %, +0.06 % | +0.42 %, -0.11 % | +8.78 %, +8.05 % |
| generation tok/s in round 2 (prompts 1, 2) | 16.66, 16.66 | 18.02, 18.01 | 18.01, 18.01 | **18.00, 17.99** | -0.11 %, -0.11 % | -0.06 %, -0.11 % | +8.04 %, +7.98 % |
| prompt processing tok/s in round 1 (prompts 1, 2) | 13.05, 27.45 | 12.05, 29.40 | 13.31, 29.58 | **14.80, 29.60** | +22.82 %, +0.68 % | +11.19 %, +0.07 % | +13.41 %, +7.83 % |
| prompt processing tok/s in round 2 (prompts 1, 2) | 26.95, 29.43 | 29.09, 31.84 | 29.10, 32.01 | **29.01, 31.66** | -0.28 %, -0.57 % | -0.31 %, -1.09 % | +7.64 %, +7.58 % |

This boot's check: 2 rounds in 71 s, exit 0.
