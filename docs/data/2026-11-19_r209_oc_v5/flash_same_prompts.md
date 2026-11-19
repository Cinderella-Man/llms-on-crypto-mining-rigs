| task | turn | prompt tokens read | from the cache | fnoc0 prompt ms | v4rc2 prompt ms | f5oc0 prompt ms | f5oc1 prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6786.0 | 6736.7 | 6773.6 | 6363.0 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 3065.2 | 3006.7 | 3021.0 | 2716.7 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 6272.6 | 6203.7 | 6225.1 | 5850.1 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6739.5 | 6650.7 | 6753.3 | 6357.0 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 3129.7 | 3088.7 | 3203.8 | 2809.0 |
| bugfix-javascript-triangle | 2 | 1705 | 2031 | 8241.2 | 8447.2 | 8248.7 | 7841.3 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 4861.5 | 3662.0 | 4670.1 | 4359.6 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 28824.8 | 28567.5 | 28661.0 | 27494.3 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 3132.3 | 3163.1 | 3147.8 | 2840.7 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 7542.8 | 7386.1 | 7442.1 | 6985.6 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 7263.3 | 7198.7 | 7203.4 | 6653.5 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 6653.1 | 6616.2 | 6494.1 | 6263.5 |
| **all 12 matched requests** | | 11676 | | **92512.2** (126.2 tok/s) | **90727.2** (128.7 tok/s) | **91844.1** (127.1 tok/s) | **86534.2** (134.9 tok/s) |
| **of them, the 11 first turns** | | 9971 | | **84271.0** (118.3 tok/s) | **82280.0** (121.2 tok/s) | **83595.4** (119.3 tok/s) | **78692.9** (126.7 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| fnoc0 | 60 | 22860 | 814.1 | 28.08 |
| v4rc2 | 91 | 39678 | 1469.6 | 27.00 |
| f5oc0 | 66 | 22146 | 796.1 | 27.82 |
| f5oc1 | 61 | 22262 | 725.5 | 30.69 |
