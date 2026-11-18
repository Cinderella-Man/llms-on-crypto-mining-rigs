| task | turn | prompt tokens read | from the cache | o4c2 prompt ms | o4g2b prompt ms | o4g2b / o4c2 |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8615.8 | 8586.3 | 0.9966 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16483.1 | 16441.2 | 0.9975 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6815.3 | 6922.2 | 1.0157 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8281.7 | 8322.7 | 1.0049 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10291.1 | 10320.5 | 1.0029 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8672.6 | 8607.8 | 0.9925 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8379.0 | 8455.0 | 1.0091 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7945.9 | 7351.2 | 0.9252 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9956.1 | 9929.8 | 0.9974 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8222.8 | 8097.2 | 0.9847 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11872.5 | 11847.3 | 0.9979 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27818.2 | 27831.8 | 1.0005 |
| bugfix-python-affine-cipher | 2 | 1185 | 2810 | 10097.6 | 9998.7 | 0.9902 |
| bugfix-python-affine-cipher | 3 | 448 | 4052 | 7675.8 | 7798.5 | 1.0160 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7032.7 | 7063.7 | 1.0044 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10088.2 | 10058.3 | 0.9970 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9550.5 | 9509.8 | 0.9957 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8419.3 | 8384.6 | 0.9959 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8947.3 | 8871.2 | 0.9915 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9301.9 | 9481.3 | 1.0193 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8525.2 | 8519.7 | 0.9994 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2394 | 11506.5 | 11482.0 | 0.9979 |

| requests | prompt tokens read | o4c2 s | o4g2b s | o4g2b / o4c2 | o4g2b slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 22 | 25137 | 224.50 | 223.88 | 0.9972 | 8 / 0 / 14 | 0.9977 |
| first turns, 12 | 10708 | 119.20 | 118.68 | 0.9956 | 5 / 0 / 7 | 0.9980 |
| with a cached prefix (all but a session's cold first request), 21 | 22395 | 196.68 | 196.05 | 0.9968 | 7 / 0 / 14 | 0.9975 |
