| task | turn | prompt tokens read | from the cache | o5t1a prompt ms | o5t1b prompt ms | o5f16a prompt ms | o5q8a prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8049.5 | 8047.2 | 8038.4 | 8198.8 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 15411.2 | 15594.5 | 15509.2 | 16455.3 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6643.7 | 6745.4 | 6140.2 | 6252.5 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 10231.5 | 10250.6 | 10208.8 | 10558.4 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 7976.3 | 7519.4 | 8016.9 | 8139.4 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 9529.5 | 9468.0 | 9464.5 | 9729.2 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 7970.8 | 7946.1 | 7966.8 | 8122.7 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 7835.7 | 7831.2 | 7811.8 | 7977.2 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 6383.4 | 6903.0 | 6976.5 | 6973.0 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9233.5 | 9227.2 | 9226.4 | 9480.7 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 7836.7 | 7838.8 | 7846.9 | 7944.2 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 25814.0 | 25828.8 | 25218.3 | 25619.2 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 6567.5 | 6778.1 | 6686.2 | 6878.6 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 9361.3 | 9359.5 | 9346.9 | 9606.8 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 8784.7 | 8765.4 | 8774.3 | 8941.6 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 7868.3 | 7882.0 | 7843.9 | 7983.6 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8292.9 | 8302.9 | 8277.6 | 8549.6 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 8616.0 | 8622.6 | 8632.4 | 8774.4 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 7879.1 | 7888.8 | 7872.5 | 8028.8 |
| **all 19 matched requests** | | 21303 | | **180285.5** (118.2 tok/s) | **180799.6** (117.8 tok/s) | **179858.4** (118.4 tok/s) | **184214.0** (115.6 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **110390.0** (97.0 tok/s) | **110765.5** (96.7 tok/s) | **110013.2** (97.3 tok/s) | **111856.8** (95.7 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o5t1a | 64 | 17458 | 899.4 | 19.41 |
| o5t1b | 54 | 21478 | 1129.6 | 19.01 |
| o5f16a | 59 | 18433 | 949.1 | 19.42 |
| o5q8a | 65 | 22559 | 1106.1 | 20.39 |
