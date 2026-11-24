| task | turn | prompt tokens read | from the cache | f5cand prompt ms | f5ref prompt ms |
|---|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6336.8 | 6381.1 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 2945.9 | 2993.9 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 4091.4 | 4023.1 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6215.9 | 5207.3 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 3074.0 | 3122.2 |
| bugfix-javascript-triangle | 2 | 1705 | 2031 | 8321.7 | 8778.0 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3632.9 | 3639.3 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 28535.5 | 28481.6 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 3061.9 | 3056.1 |
| bugfix-python-list-ops | 2 | 1557 | 2064 | 7944.7 | 8009.7 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 7303.4 | 6808.6 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 7256.8 | 7259.9 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 4988.1 | 4992.9 |
| **all 13 matched requests** | | 13233 | | **93708.9** (141.2 tok/s) | **92753.8** (142.7 tok/s) |
| **of them, the 11 first turns** | | 9971 | | **77442.6** (128.8 tok/s) | **75966.1** (131.3 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| f5cand | 62 | 32077 | 1121.9 | 28.59 |
| f5ref | 95 | 26616 | 993.8 | 26.78 |
