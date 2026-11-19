| task | turn | prompt tokens read | from the cache | o5oc1 prompt ms | o5t0a prompt ms | o5t1a prompt ms | o5t0b prompt ms | o5t1b prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8034.4 | 8059.0 | 8049.5 | 8028.5 | 8047.2 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 15391.0 | 15404.4 | 15411.2 | 15452.4 | 15594.5 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6691.9 | 6625.1 | 6643.7 | 6239.1 | 6745.4 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 7884.8 | 7505.9 | 7976.3 | 7923.8 | 7519.4 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 9449.0 | 9491.2 | 9529.5 | 9486.2 | 9468.0 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 7936.2 | 7976.1 | 7970.8 | 7991.8 | 7946.1 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 7812.1 | 7822.1 | 7835.7 | 7837.4 | 7831.2 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 6911.6 | 6350.4 | 6383.4 | 6946.5 | 6903.0 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9217.6 | 9225.2 | 9233.5 | 9239.5 | 9227.2 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 7461.0 | 7346.3 | 7836.7 | 7859.9 | 7838.8 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 10972.7 | 10937.7 | 10960.9 | 10994.0 | 10968.5 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 25944.4 | 25521.3 | 25814.0 | 25917.3 | 25828.8 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 6675.1 | 6681.1 | 6567.5 | 6682.0 | 6778.1 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 9362.9 | 9358.5 | 9361.3 | 9374.2 | 9359.5 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 8755.6 | 8758.2 | 8784.7 | 8788.6 | 8765.4 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 7848.8 | 7858.7 | 7868.3 | 7853.3 | 7882.0 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8276.7 | 8297.4 | 8292.9 | 8297.8 | 8302.9 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 8624.8 | 8616.8 | 8616.0 | 8614.5 | 8622.6 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 7881.8 | 7891.1 | 7879.1 | 7893.8 | 7888.8 |
| **all 19 matched requests** | | 21618 | | **181132.3** (119.3 tok/s) | **179726.6** (120.3 tok/s) | **181014.9** (119.4 tok/s) | **181420.8** (119.2 tok/s) | **181517.4** (119.1 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **110650.4** (96.8 tok/s) | **109190.1** (98.1 tok/s) | **110390.0** (97.0 tok/s) | **110739.3** (96.7 tok/s) | **110765.5** (96.7 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o5oc1 | 67 | 20222 | 1050.0 | 19.26 |
| o5t0a | 58 | 20554 | 1074.9 | 19.12 |
| o5t1a | 64 | 17458 | 899.4 | 19.41 |
| o5t0b | 66 | 16209 | 831.1 | 19.50 |
| o5t1b | 54 | 21478 | 1129.6 | 19.01 |
