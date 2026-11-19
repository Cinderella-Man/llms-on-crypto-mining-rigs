| task | turn | prompt tokens read | from the cache | v4rc2 prompt ms | f5c1 prompt ms |
|---|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6736.7 | 6797.1 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 3006.7 | 3000.2 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 6203.7 | 6308.7 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6650.7 | 6615.4 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 3088.7 | 3068.9 |
| bugfix-javascript-triangle | 2 | 1705 | 2031 | 8447.2 | 8252.9 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3662.0 | 4817.4 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 28567.5 | 27933.5 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 3163.1 | 3074.7 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 7386.1 | 7326.6 |
| bugfix-python-phone-number | 2 | 2048 | 2765 | 9045.1 | 8993.6 |
| bugfix-python-pig-latin | 1 | 2324 | 0 | 12336.3 | 12662.1 |
| bugfix-python-pig-latin | 2 | 1051 | 2406 | 6942.3 | 6929.4 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 7198.7 | 7924.0 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 6616.2 | 6437.8 |
| bugfix-python-variable-length-quantity | 2 | 2030 | 2446 | 8796.6 | 8771.6 |
| **all 16 matched requests** | | 19129 | | **127847.5** (149.6 tok/s) | **128913.9** (148.4 tok/s) |
| **of them, the 12 first turns** | | 12295 | | **94616.3** (129.9 tok/s) | **95966.4** (128.1 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| v4rc2 | 91 | 39678 | 1469.6 | 27.00 |
| f5c1 | 103 | 40857 | 1514.5 | 26.98 |
