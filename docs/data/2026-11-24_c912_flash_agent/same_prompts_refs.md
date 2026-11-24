| task | turn | prompt tokens read | from the cache | t1400a prompt ms | t1400b prompt ms |
|---|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 5998.3 | 5023.4 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 2723.6 | 2786.8 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 3837.7 | 3808.7 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 4794.9 | 5238.7 |
| bugfix-javascript-space-age | 2 | 931 | 2409 | 5050.9 | 5024.0 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 2863.2 | 2807.9 |
| bugfix-javascript-triangle | 2 | 1705 | 2031 | 7778.8 | 7815.4 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3385.1 | 3370.2 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 27334.5 | 27406.9 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 2905.5 | 2892.7 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 6274.1 | 6192.6 |
| bugfix-python-phone-number | 2 | 622 | 2728 | 3897.8 | 3825.0 |
| bugfix-python-pig-latin | 1 | 775 | 1549 | 4722.1 | 4648.8 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 6637.3 | 5965.8 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 4536.8 | 4516.1 |
| **all 15 matched requests** | | 14004 | | **92740.5** (151.0 tok/s) | **91323.1** (153.3 tok/s) |
| **of them, the 12 first turns** | | 10746 | | **76013.0** (141.4 tok/s) | **74658.7** (143.9 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| t1400a | 66 | 31360 | 1021.1 | 30.71 |
| t1400b | 67 | 18328 | 593.8 | 30.87 |
