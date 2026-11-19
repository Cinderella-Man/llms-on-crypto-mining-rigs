| task | turn | prompt tokens read | from the cache | o4oc0 prompt ms | o4g2a prompt ms | o4g2b prompt ms | o5oc0 prompt ms | o5oc1 prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8584.7 | 8586.7 | 8586.3 | 8554.4 | 8034.4 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 16759.8 | 16550.7 | 16441.2 | 16368.1 | 15391.0 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6541.6 | 6553.0 | 6922.2 | 6461.2 | 6691.9 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8313.1 | 8360.3 | 8322.7 | 7974.4 | 7884.8 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10102.4 | 10201.6 | 10320.5 | 10101.2 | 9449.0 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8483.7 | 8603.4 | 8607.8 | 8439.9 | 7936.2 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8351.6 | 8564.1 | 8455.0 | 8310.6 | 7812.1 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 6979.0 | 7201.7 | 7351.2 | 6908.9 | 6911.6 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9861.1 | 9927.0 | 9929.8 | 9851.9 | 9217.6 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8200.5 | 7858.7 | 8097.2 | 7825.1 | 7461.0 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11695.5 | 11880.6 | 11847.3 | 11627.1 | 10972.7 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27184.6 | 28723.2 | 27831.8 | 27147.0 | 25944.4 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7273.6 | 7056.7 | 7063.7 | 7037.0 | 6675.1 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 9981.4 | 10064.2 | 10058.3 | 9973.3 | 9362.9 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9315.9 | 9513.3 | 9509.8 | 9273.4 | 8755.6 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8378.1 | 8397.2 | 8384.6 | 8356.9 | 7848.8 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8927.8 | 8960.8 | 8871.2 | 8810.5 | 8276.7 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9195.3 | 9330.0 | 9481.3 | 9133.4 | 8624.8 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8419.4 | 8529.2 | 8519.7 | 8381.1 | 7881.8 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2395 | 11265.5 | 11458.9 | 11482.0 | 11231.7 | 10503.3 |
| **all 20 matched requests** | | 23504 | | **203814.6** (115.3 tok/s) | **206321.3** (113.9 tok/s) | **206083.5** (114.1 tok/s) | **201767.1** (116.5 tok/s) | **191635.7** (122.6 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **116869.5** (91.6 tok/s) | **118713.6** (90.2 tok/s) | **118678.2** (90.2 tok/s) | **115492.5** (92.7 tok/s) | **110650.4** (96.8 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o4oc0 | 67 | 23764 | 1446.9 | 16.42 |
| o4g2a | 62 | 16516 | 935.4 | 17.66 |
| o4g2b | 59 | 20411 | 1181.1 | 17.28 |
| o5oc0 | 60 | 14253 | 797.6 | 17.87 |
| o5oc1 | 67 | 20222 | 1050.0 | 19.26 |
