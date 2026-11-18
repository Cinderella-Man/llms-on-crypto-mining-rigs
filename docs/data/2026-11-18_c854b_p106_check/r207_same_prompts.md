| task | turn | prompt tokens read | from the cache | o4v5a prompt ms | o4c2 prompt ms | o4c3 prompt ms |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8597.3 | 8615.8 | 8602.4 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16445.1 | 16483.1 | 16450.8 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6510.1 | 6815.3 | 6881.2 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 11027.7 | 11054.9 | 11030.9 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8282.2 | 8281.7 | 8010.1 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10203.8 | 10291.1 | 10276.3 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8630.0 | 8672.6 | 8549.5 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8484.7 | 8379.0 | 8409.1 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7243.7 | 7945.9 | 7022.7 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9952.5 | 9956.1 | 9949.7 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8192.7 | 8222.8 | 7855.6 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11785.6 | 11872.5 | 11898.5 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 29185.8 | 27818.2 | 27867.1 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7105.5 | 7032.7 | 7402.7 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10076.5 | 10088.2 | 10078.1 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9539.2 | 9550.5 | 9517.7 |
| bugfix-python-phone-number | 2 | 1906 | 2703 | 11655.1 | 11634.2 | 11596.6 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8407.8 | 8419.3 | 8398.3 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8862.6 | 8947.3 | 8964.1 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9356.7 | 9301.9 | 9354.0 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8540.9 | 8525.2 | 8562.2 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2394 | 11439.1 | 11506.5 | 11504.8 |
| **all 22 matched requests** | | 27160 | | **229524.5** (118.3 tok/s) | **229414.4** (118.4 tok/s) | **228182.7** (119.0 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **119591.9** (89.5 tok/s) | **119201.7** (89.8 tok/s) | **118023.6** (90.7 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o4v5a | 61 | 18007 | 1090.2 | 16.52 |
| o4c2 | 63 | 20266 | 1243.6 | 16.30 |
| o4c3 | 72 | 21608 | 1334.4 | 16.19 |
