| task | turn | prompt tokens read | from the cache | o4g2a prompt ms | o4g2b prompt ms | o4g2b / o4g2a |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8586.7 | 8586.3 | 1.0000 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16550.7 | 16441.2 | 0.9934 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6553.0 | 6922.2 | 1.0563 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8360.3 | 8322.7 | 0.9955 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10201.6 | 10320.5 | 1.0117 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8603.4 | 8607.8 | 1.0005 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8564.1 | 8455.0 | 0.9873 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 7201.7 | 7351.2 | 1.0208 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9927.0 | 9929.8 | 1.0003 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 7858.7 | 8097.2 | 1.0303 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11880.6 | 11847.3 | 0.9972 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 28723.2 | 27831.8 | 0.9690 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7056.7 | 7063.7 | 1.0010 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 10064.2 | 10058.3 | 0.9994 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9513.3 | 9509.8 | 0.9996 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8397.2 | 8384.6 | 0.9985 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8960.8 | 8871.2 | 0.9900 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9330.0 | 9481.3 | 1.0162 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8529.2 | 8519.7 | 0.9989 |

| requests | prompt tokens read | o4g2a s | o4g2b s | o4g2b / o4g2a | o4g2b slower / same / faster | median of the ratios |
|---|---:|---:|---:|---:|---|---:|
| all matched, 19 | 21618 | 194.86 | 194.60 | 0.9987 | 8 / 0 / 11 | 0.9996 |
| first turns, 12 | 10708 | 118.71 | 118.68 | 0.9997 | 6 / 0 / 6 | 1.0002 |
| with a cached prefix (all but a session's cold first request), 18 | 18876 | 166.14 | 166.77 | 1.0038 | 8 / 0 / 10 | 0.9998 |
