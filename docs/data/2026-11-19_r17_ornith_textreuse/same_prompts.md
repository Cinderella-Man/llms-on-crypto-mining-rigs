| task | turn | prompt tokens read | from the cache | o5oc1 prompt ms | o5t0a prompt ms | o5t1a prompt ms |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8034.4 | 8059.0 | 8049.5 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 15391.0 | 15404.4 | 15411.2 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6691.9 | 6625.1 | 6643.7 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 7884.8 | 7505.9 | 7976.3 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 9449.0 | 9491.2 | 9529.5 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 7936.2 | 7976.1 | 7970.8 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 7812.1 | 7822.1 | 7835.7 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 6911.6 | 6350.4 | 6383.4 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9217.6 | 9225.2 | 9233.5 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 7461.0 | 7346.3 | 7836.7 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 10972.7 | 10937.7 | 10960.9 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 25944.4 | 25521.3 | 25814.0 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 6675.1 | 6681.1 | 6567.5 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 9362.9 | 9358.5 | 9361.3 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 8755.6 | 8758.2 | 8784.7 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 7848.8 | 7858.7 | 7868.3 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8276.7 | 8297.4 | 8292.9 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 8624.8 | 8616.8 | 8616.0 |
| bugfix-python-scale-generator | 2 | 1448 | 2659 | 9509.3 | 9507.9 | 9504.7 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 7881.8 | 7891.1 | 7879.1 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2395 | 10503.3 | 10536.5 | 10529.1 |
| **all 21 matched requests** | | 24952 | | **201144.9** (124.0 tok/s) | **199771.0** (124.9 tok/s) | **201048.7** (124.1 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **110650.4** (96.8 tok/s) | **109190.1** (98.1 tok/s) | **110390.0** (97.0 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o5oc1 | 67 | 20222 | 1050.0 | 19.26 |
| o5t0a | 58 | 20554 | 1074.9 | 19.12 |
| o5t1a | 64 | 17458 | 899.4 | 19.41 |
