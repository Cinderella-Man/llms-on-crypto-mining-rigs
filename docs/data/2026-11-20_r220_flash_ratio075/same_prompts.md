| task | turn | prompt tokens read | from the cache | f5oc1 prompt ms | c10a prompt ms |
|---|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6363.0 | 6340.3 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 2716.7 | 2711.6 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 5850.1 | 5967.0 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6357.0 | 6334.9 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 2809.0 | 2999.9 |
| bugfix-javascript-triangle | 2 | 1705 | 2031 | 7841.3 | 7837.6 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 4359.6 | 4181.4 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 27494.3 | 27140.2 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 2840.7 | 2852.1 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 6985.6 | 6957.1 |
| bugfix-python-pig-latin | 1 | 775 | 1549 | 6136.7 | 6278.6 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 6653.5 | 6862.9 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 6263.5 | 6130.4 |
| **all 13 matched requests** | | 12451 | | **92670.9** (134.4 tok/s) | **92594.0** (134.5 tok/s) |
| **of them, the 12 first turns** | | 10746 | | **84829.6** (126.7 tok/s) | **84756.5** (126.8 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| f5oc1 | 61 | 22262 | 725.5 | 30.69 |
| c10a | 64 | 16406 | 541.9 | 30.28 |
