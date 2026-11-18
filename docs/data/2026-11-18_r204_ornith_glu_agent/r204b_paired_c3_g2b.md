| task | turn | prompt tokens read | from the cache | o4c3 prompt ms | o4g2b prompt ms | o4g2b / o4c3 |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8602.4 | 8586.3 | 0.9981 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16450.8 | 16441.2 | 0.9994 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6881.2 | 6922.2 | 1.0060 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8010.1 | 8322.7 | 1.0390 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10276.3 | 10320.5 | 1.0043 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8549.5 | 8607.8 | 1.0068 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8409.1 | 8455.0 | 1.0055 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7022.7 | 7351.2 | 1.0468 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9949.7 | 9929.8 | 0.9980 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 7855.6 | 8097.2 | 1.0308 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11898.5 | 11847.3 | 0.9957 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27867.1 | 27831.8 | 0.9987 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7402.7 | 7063.7 | 0.9542 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10078.1 | 10058.3 | 0.9980 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9517.7 | 9509.8 | 0.9992 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8398.3 | 8384.6 | 0.9984 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8964.1 | 8871.2 | 0.9896 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9354.0 | 9481.3 | 1.0136 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8562.2 | 8519.7 | 0.9950 |

| requests | prompt tokens read | o4c3 s | o4g2b s | o4g2b / o4c3 | o4g2b slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 19 | 21618 | 194.05 | 194.60 | 1.0028 | 8 / 0 / 11 | 0.9992 |
| first turns, 12 | 10708 | 118.02 | 118.68 | 1.0055 | 6 / 0 / 6 | 1.0026 |
| with a cached prefix (all but a session's cold first request), 18 | 18876 | 166.18 | 166.77 | 1.0035 | 8 / 0 / 10 | 0.9993 |
