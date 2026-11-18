| task | turn | prompt tokens read | from the cache | fnoc0 prompt ms | fnoc1 prompt ms | r156a-v4r prompt ms |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6786.0 | 6705.5 | 6670.3 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 3065.2 | 2923.1 | 2988.0 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 6272.6 | 6129.0 | 6437.4 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6739.5 | 6725.8 | 6651.6 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 3129.7 | 3017.2 | 3078.3 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 4861.5 | 4625.2 | 3689.6 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 28824.8 | 28591.6 | 28089.2 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 3132.3 | 3057.8 | 3115.9 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 7542.8 | 7450.2 | 7371.0 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 7263.3 | 7809.2 | 7193.1 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 6653.1 | 6404.0 | 6554.9 |
| **all 11 matched requests** | | 9971 | | **84271.0** (118.3 tok/s) | **83438.6** (119.5 tok/s) | **81839.3** (121.8 tok/s) |
| **of them, the 11 first turns** | | 9971 | | **84271.0** (118.3 tok/s) | **83438.6** (119.5 tok/s) | **81839.3** (121.8 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| fnoc0 | 60 | 22860 | 814.1 | 28.08 |
| fnoc1 | 62 | 39570 | 1458.7 | 27.13 |
| r156a-v4r | 95 | 41836 | 1568.7 | 26.67 |
