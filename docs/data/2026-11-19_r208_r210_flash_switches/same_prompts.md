| task | turn | prompt tokens read | from the cache | v4rc2 prompt ms | f5c1 prompt ms | f5s1 prompt ms |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6736.7 | 6797.1 | 6665.9 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 3006.7 | 3000.2 | 3039.1 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 6203.7 | 6308.7 | 6550.7 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6650.7 | 6615.4 | 6611.6 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 3088.7 | 3068.9 | 3151.5 |
| bugfix-javascript-triangle | 2 | 1705 | 2031 | 8447.2 | 8252.9 | 8225.1 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3662.0 | 4817.4 | 4806.2 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 28567.5 | 27933.5 | 27954.2 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 3163.1 | 3074.7 | 3116.7 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 7386.1 | 7326.6 | 7245.0 |
| bugfix-python-pig-latin | 2 | 1051 | 2406 | 6942.3 | 6929.4 | 6892.9 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 7198.7 | 7924.0 | 7448.8 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 6616.2 | 6437.8 | 6453.8 |
| **all 13 matched requests** | | 12727 | | **97669.5** (130.3 tok/s) | **98486.5** (129.2 tok/s) | **98161.6** (129.7 tok/s) |
| **of them, the 11 first turns** | | 9971 | | **82280.0** (121.2 tok/s) | **83304.2** (119.7 tok/s) | **83043.6** (120.1 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| v4rc2 | 91 | 39678 | 1469.6 | 27.00 |
| f5c1 | 103 | 40857 | 1514.5 | 26.98 |
| f5s1 | 82 | 25651 | 921.0 | 27.85 |
