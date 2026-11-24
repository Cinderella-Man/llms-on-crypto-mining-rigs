| task | turn | prompt tokens read | from the cache | t1400a prompt ms | t1400b prompt ms | f5cand prompt ms | f5ref prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 5998.3 | 5023.4 | 6336.8 | 6381.1 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 2723.6 | 2786.8 | 2945.9 | 2993.9 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 3837.7 | 3808.7 | 4091.4 | 4023.1 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 4794.9 | 5238.7 | 6215.9 | 5207.3 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 2863.2 | 2807.9 | 3074.0 | 3122.2 |
| bugfix-javascript-triangle | 2 | 1705 | 2031 | 7778.8 | 7815.4 | 8321.7 | 8778.0 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3385.1 | 3370.2 | 3632.9 | 3639.3 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 27334.5 | 27406.9 | 28535.5 | 28481.6 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 2905.5 | 2892.7 | 3061.9 | 3056.1 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 6274.1 | 6192.6 | 7303.4 | 6808.6 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 6637.3 | 5965.8 | 7256.8 | 7259.9 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 4536.8 | 4516.1 | 4988.1 | 4992.9 |
| **all 12 matched requests** | | 11676 | | **79069.7** (147.7 tok/s) | **77825.2** (150.0 tok/s) | **85764.3** (136.1 tok/s) | **84744.2** (137.8 tok/s) |
| **of them, the 11 first turns** | | 9971 | | **71290.9** (139.9 tok/s) | **70009.8** (142.4 tok/s) | **77442.6** (128.8 tok/s) | **75966.1** (131.3 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| t1400a | 66 | 31360 | 1021.1 | 30.71 |
| t1400b | 67 | 18328 | 593.8 | 30.87 |
| f5cand | 62 | 32077 | 1121.9 | 28.59 |
| f5ref | 95 | 26616 | 993.8 | 26.78 |
