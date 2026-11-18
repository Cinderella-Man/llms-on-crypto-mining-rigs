| task | turn | prompt tokens read | from the cache | v4rc2 prompt ms | r156a-v4r prompt ms | r156b-v4r2 prompt ms |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6736.7 | 6670.3 | 6857.7 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 3006.7 | 2988.0 | 3072.0 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 6203.7 | 6437.4 | 6246.4 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6650.7 | 6651.6 | 6606.9 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 3088.7 | 3078.3 | 3077.3 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3662.0 | 3689.6 | 5385.3 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 28567.5 | 28089.2 | 29339.5 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 3163.1 | 3115.9 | 3058.4 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 7386.1 | 7371.0 | 7480.5 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 7198.7 | 7193.1 | 7250.5 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 6616.2 | 6554.9 | 6737.8 |
| **all 11 matched requests** | | 9971 | | **82280.0** (121.2 tok/s) | **81839.3** (121.8 tok/s) | **85112.3** (117.2 tok/s) |
| **of them, the 11 first turns** | | 9971 | | **82280.0** (121.2 tok/s) | **81839.3** (121.8 tok/s) | **85112.3** (117.2 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| v4rc2 | 91 | 39678 | 1469.6 | 27.00 |
| r156a-v4r | 95 | 41836 | 1568.7 | 26.67 |
| r156b-v4r2 | 68 | 25747 | 932.5 | 27.61 |
