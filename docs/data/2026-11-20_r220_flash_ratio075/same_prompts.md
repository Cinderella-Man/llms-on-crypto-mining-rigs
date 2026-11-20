| task | turn | prompt tokens read | from the cache | f5oc1 prompt ms | c10a prompt ms | r075a prompt ms | c10b prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6363.0 | 6340.3 | 6360.9 | 8672.7 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 2716.7 | 2711.6 | 2765.4 | 2714.7 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 5850.1 | 5967.0 | 5696.2 | 5831.0 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6357.0 | 6334.9 | 6432.6 | 6561.5 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 2809.0 | 2999.9 | 2937.3 | 2815.3 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 4359.6 | 4181.4 | 4306.4 | 4199.5 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 27494.3 | 27140.2 | 27417.3 | 27382.1 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 2840.7 | 2852.1 | 2858.6 | 2843.3 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 6985.6 | 6957.1 | 7121.8 | 7216.0 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 6653.5 | 6862.9 | 6818.3 | 6995.7 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 6263.5 | 6130.4 | 6123.1 | 6311.6 |
| **all 11 matched requests** | | 9971 | | **78692.9** (126.7 tok/s) | **78477.9** (127.1 tok/s) | **78837.8** (126.5 tok/s) | **81543.4** (122.3 tok/s) |
| **of them, the 11 first turns** | | 9971 | | **78692.9** (126.7 tok/s) | **78477.9** (127.1 tok/s) | **78837.8** (126.5 tok/s) | **81543.4** (122.3 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| f5oc1 | 61 | 22262 | 725.5 | 30.69 |
| c10a | 64 | 16406 | 541.9 | 30.28 |
| r075a | 63 | 20391 | 666.4 | 30.60 |
| c10b | 98 | 37544 | 1274.4 | 29.46 |
