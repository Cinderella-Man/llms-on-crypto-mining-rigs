| task | turn | prompt tokens read | from the cache | o4oc0 prompt ms | o4oc1 prompt ms | o4oc2 prompt ms |
|---|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 8584.7 | 8390.5 | 8029.0 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6541.6 | 6808.1 | 6546.0 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 8313.1 | 8240.0 | 7824.5 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 10102.4 | 9970.8 | 9462.5 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 8483.7 | 8310.6 | 7966.3 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 8351.6 | 8205.9 | 7807.8 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 6979.0 | 7118.1 | 6362.5 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9861.1 | 9727.3 | 9241.4 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 8200.5 | 8098.8 | 7345.4 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 11695.5 | 11495.7 | 10949.0 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 27184.6 | 26746.8 | 25691.4 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 7273.6 | 6976.9 | 6528.2 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 9981.4 | 9859.1 | 9351.4 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 9315.9 | 9160.4 | 8728.5 |
| bugfix-python-phone-number | 2 | 1906 | 2703 | 11423.2 | 11272.3 | 10661.3 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 8378.1 | 8210.9 | 7859.6 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8927.8 | 8630.1 | 8265.0 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 9195.3 | 9053.6 | 8614.0 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 8419.4 | 8240.5 | 7903.4 |
| **all 19 matched requests** | | 20456 | | **187212.6** (109.3 tok/s) | **184516.4** (110.9 tok/s) | **175137.1** (116.8 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **116869.5** (91.6 tok/s) | **115355.2** (92.8 tok/s) | **109398.9** (97.9 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o4oc0 | 67 | 23764 | 1446.9 | 16.42 |
| o4oc1 | 62 | 17016 | 982.8 | 17.31 |
| o4oc2 | 63 | 21519 | 1202.6 | 17.89 |
