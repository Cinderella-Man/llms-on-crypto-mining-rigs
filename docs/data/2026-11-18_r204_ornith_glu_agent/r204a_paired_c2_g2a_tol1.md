| task | turn | prompt tokens read | from the cache | o4c2 prompt ms | o4g2a prompt ms | o4g2a / o4c2 |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8615.8 | 8586.7 | 0.9966 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16483.1 | 16550.7 | 1.0041 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6815.3 | 6553.0 | 0.9615 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8281.7 | 8360.3 | 1.0095 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10291.1 | 10201.6 | 0.9913 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8672.6 | 8603.4 | 0.9920 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8379.0 | 8564.1 | 1.0221 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7945.9 | 7201.7 | 0.9063 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9956.1 | 9927.0 | 0.9971 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8222.8 | 7858.7 | 0.9557 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11872.5 | 11880.6 | 1.0007 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27818.2 | 28723.2 | 1.0325 |
| bugfix-python-affine-cipher | 2 | 1185 | 2810 | 10097.6 | 10027.1 | 0.9930 |
| bugfix-python-affine-cipher | 3 | 448 | 4052 | 7675.8 | 7685.6 | 1.0013 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7032.7 | 7056.7 | 1.0034 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10088.2 | 10064.2 | 0.9976 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9550.5 | 9513.3 | 0.9961 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8419.3 | 8397.2 | 0.9974 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8947.3 | 8960.8 | 1.0015 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9301.9 | 9330.0 | 1.0030 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8525.2 | 8529.2 | 1.0005 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2394 | 11506.5 | 11458.9 | 0.9959 |

| requests | prompt tokens read | o4c2 s | o4g2a s | o4g2a / o4c2 | o4g2a slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 22 | 25137 | 224.50 | 224.03 | 0.9979 | 10 / 0 / 12 | 0.9975 |
| first turns, 12 | 10708 | 119.20 | 118.71 | 0.9959 | 5 / 0 / 7 | 0.9970 |
| with a cached prefix (all but a session's cold first request), 21 | 22395 | 196.68 | 195.31 | 0.9930 | 9 / 0 / 12 | 0.9974 |
