| task | turn | prompt tokens read | from the cache | o4c3 prompt ms | o4g2a prompt ms | o4g2a / o4c3 |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8602.4 | 8586.7 | 0.9982 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16450.8 | 16550.7 | 1.0061 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6881.2 | 6553.0 | 0.9523 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8010.1 | 8360.3 | 1.0437 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10276.3 | 10201.6 | 0.9927 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8549.5 | 8603.4 | 1.0063 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8409.1 | 8564.1 | 1.0184 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7022.7 | 7201.7 | 1.0255 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9949.7 | 9927.0 | 0.9977 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 7855.6 | 7858.7 | 1.0004 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11898.5 | 11880.6 | 0.9985 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27867.1 | 28723.2 | 1.0307 |
| bugfix-python-affine-cipher | 2 | 1185 | 2811 | 9937.8 | 10027.1 | 1.0090 |
| bugfix-python-affine-cipher | 3 | 448 | 4053 | 7831.3 | 7685.6 | 0.9814 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7402.7 | 7056.7 | 0.9533 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10078.1 | 10064.2 | 0.9986 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9517.7 | 9513.3 | 0.9995 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8398.3 | 8397.2 | 0.9999 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8964.1 | 8960.8 | 0.9996 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9354.0 | 9330.0 | 0.9974 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8562.2 | 8529.2 | 0.9961 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2394 | 11504.8 | 11458.9 | 0.9960 |

| requests | prompt tokens read | o4c3 s | o4g2a s | o4g2a / o4c3 | o4g2a slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 22 | 25137 | 223.32 | 224.03 | 1.0032 | 8 / 0 / 14 | 0.9991 |
| first turns, 12 | 10708 | 118.02 | 118.71 | 1.0058 | 5 / 0 / 7 | 0.9997 |
| with a cached prefix (all but a session's cold first request), 21 | 22395 | 195.46 | 195.31 | 0.9993 | 7 / 0 / 14 | 0.9986 |
