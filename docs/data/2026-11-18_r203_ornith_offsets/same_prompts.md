| task | turn | prompt tokens read | from the cache | o4sna prompt ms | o4snb prompt ms | o4oc0 prompt ms |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8537.5 | 8570.8 | 8584.7 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16318.3 | 16403.0 | 16759.8 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6498.8 | 6474.8 | 6541.6 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8233.6 | 8343.7 | 8313.1 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10088.3 | 10066.4 | 10102.4 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8436.5 | 8453.2 | 8483.7 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8318.4 | 8311.6 | 8351.6 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7105.1 | 6754.4 | 6979.0 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9824.7 | 9836.2 | 9861.1 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8106.5 | 8141.8 | 8200.5 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11630.0 | 11601.2 | 11695.5 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 26726.8 | 26506.6 | 27184.6 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7042.9 | 6849.2 | 7273.6 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 9960.7 | 9945.2 | 9981.4 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9281.8 | 9276.3 | 9315.9 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8335.5 | 8360.2 | 8378.1 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8796.9 | 8809.6 | 8927.8 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9143.7 | 9164.1 | 9195.3 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8385.6 | 8393.8 | 8419.4 |
| **all 19 matched requests** | | 21618 | | **190771.6** (113.3 tok/s) | **190262.0** (113.6 tok/s) | **192549.1** (112.3 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **115834.2** (92.4 tok/s) | **115288.9** (92.9 tok/s) | **116869.5** (91.6 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o4sna | 63 | 21810 | 1320.3 | 16.52 |
| o4snb | 73 | 27605 | 1711.9 | 16.13 |
| o4oc0 | 67 | 23764 | 1446.9 | 16.42 |
