| task | turn | prompt tokens read | from the cache | o4v5a prompt ms | o4c2 prompt ms | o4sna prompt ms | o4snb prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8597.3 | 8615.8 | 8537.5 | 8570.8 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16445.1 | 16483.1 | 16318.3 | 16403.0 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6510.1 | 6815.3 | 6498.8 | 6474.8 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 11027.7 | 11054.9 | 10866.7 | 10872.6 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8282.2 | 8281.7 | 8233.6 | 8343.7 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10203.8 | 10291.1 | 10088.3 | 10066.4 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8630.0 | 8672.6 | 8436.5 | 8453.2 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8484.7 | 8379.0 | 8318.4 | 8311.6 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7243.7 | 7945.9 | 7105.1 | 6754.4 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9952.5 | 9956.1 | 9824.7 | 9836.2 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8192.7 | 8222.8 | 8106.5 | 8141.8 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11785.6 | 11872.5 | 11630.0 | 11601.2 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 29185.8 | 27818.2 | 26726.8 | 26506.6 |
| bugfix-python-affine-cipher | 2 | 1185 | 2810 | 9954.0 | 10097.6 | 9713.0 | 9722.7 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7105.5 | 7032.7 | 7042.9 | 6849.2 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10076.5 | 10088.2 | 9960.7 | 9945.2 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9539.2 | 9550.5 | 9281.8 | 9276.3 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8407.8 | 8419.3 | 8335.5 | 8360.2 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8862.6 | 8947.3 | 8796.9 | 8809.6 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9356.7 | 9301.9 | 9143.7 | 9164.1 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8540.9 | 8525.2 | 8385.6 | 8393.8 |
| **all 21 matched requests** | | 24553 | | **216384.3** (113.5 tok/s) | **216371.3** (113.5 tok/s) | **211351.3** (116.2 tok/s) | **210857.2** (116.4 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **119591.9** (89.5 tok/s) | **119201.7** (89.8 tok/s) | **115834.2** (92.4 tok/s) | **115288.9** (92.9 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o4v5a | 61 | 18007 | 1090.2 | 16.52 |
| o4c2 | 63 | 20266 | 1243.6 | 16.30 |
| o4sna | 63 | 21810 | 1320.3 | 16.52 |
| o4snb | 73 | 27605 | 1711.9 | 16.13 |
