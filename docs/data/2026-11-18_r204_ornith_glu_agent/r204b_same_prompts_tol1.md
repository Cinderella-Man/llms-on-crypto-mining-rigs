| task | turn | prompt tokens read | from the cache | o4c2 prompt ms | o4c3 prompt ms | o4g2a prompt ms | o4g2b prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8615.8 | 8602.4 | 8586.7 | 8586.3 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16483.1 | 16450.8 | 16550.7 | 16441.2 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6815.3 | 6881.2 | 6553.0 | 6922.2 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8281.7 | 8010.1 | 8360.3 | 8322.7 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10291.1 | 10276.3 | 10201.6 | 10320.5 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8672.6 | 8549.5 | 8603.4 | 8607.8 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8379.0 | 8409.1 | 8564.1 | 8455.0 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7945.9 | 7022.7 | 7201.7 | 7351.2 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9956.1 | 9949.7 | 9927.0 | 9929.8 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8222.8 | 7855.6 | 7858.7 | 8097.2 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11872.5 | 11898.5 | 11880.6 | 11847.3 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27818.2 | 27867.1 | 28723.2 | 27831.8 |
| bugfix-python-affine-cipher | 2 | 1185 | 2810 | 10097.6 | 9937.8 | 10027.1 | 9998.7 |
| bugfix-python-affine-cipher | 3 | 448 | 4052 | 7675.8 | 7831.3 | 7685.6 | 7798.5 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7032.7 | 7402.7 | 7056.7 | 7063.7 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10088.2 | 10078.1 | 10064.2 | 10058.3 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9550.5 | 9517.7 | 9513.3 | 9509.8 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8419.3 | 8398.3 | 8397.2 | 8384.6 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8947.3 | 8964.1 | 8960.8 | 8871.2 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9301.9 | 9354.0 | 9330.0 | 9481.3 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8525.2 | 8562.2 | 8529.2 | 8519.7 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2394 | 11506.5 | 11504.8 | 11458.9 | 11482.0 |
| **all 22 matched requests** | | 25137 | | **224498.7** (112.0 tok/s) | **223324.3** (112.6 tok/s) | **224034.0** (112.2 tok/s) | **223880.6** (112.3 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **119201.7** (89.8 tok/s) | **118023.6** (90.7 tok/s) | **118713.6** (90.2 tok/s) | **118678.2** (90.2 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o4c2 | 63 | 20266 | 1243.6 | 16.30 |
| o4c3 | 72 | 21608 | 1334.4 | 16.19 |
| o4g2a | 62 | 16516 | 935.4 | 17.66 |
| o4g2b | 59 | 20411 | 1181.1 | 17.28 |
