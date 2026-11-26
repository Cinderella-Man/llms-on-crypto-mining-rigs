| task | turn | prompt tokens read | from the cache | gmixa prompt ms | pmixa prompt ms | gmixb prompt ms | pmixb prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 8669.5 | 6318.6 | 8690.6 | 5383.3 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 4244.3 | 2926.6 | 4334.7 | 3008.1 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 6383.3 | 4164.3 | 6374.8 | 4142.6 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 8127.7 | 5704.7 | 8219.9 | 5148.1 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 4610.0 | 3052.3 | 4641.5 | 3106.0 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 5810.7 | 3656.2 | 5816.6 | 3651.2 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 31070.9 | 27092.0 | 31344.0 | 27171.9 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 4604.3 | 3119.1 | 4589.1 | 3106.5 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 10785.1 | 6808.1 | 10991.7 | 6753.2 |
| bugfix-python-pig-latin | 1 | 775 | 1549 | 7873.0 | 5058.1 | 7857.1 | 5042.6 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 10644.4 | 7186.6 | 10647.7 | 6992.9 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 8055.2 | 4937.3 | 7982.7 | 4993.4 |
| **all 12 matched requests** | | 10746 | | **110878.5** (96.9 tok/s) | **80024.0** (134.3 tok/s) | **111490.3** (96.4 tok/s) | **78499.8** (136.9 tok/s) |
| **of them, the 12 first turns** | | 10746 | | **110878.5** (96.9 tok/s) | **80024.0** (134.3 tok/s) | **111490.3** (96.4 tok/s) | **78499.8** (136.9 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| gmixa | 67 | 19217 | 687.6 | 27.95 |
| pmixa | 65 | 21476 | 723.8 | 29.67 |
| gmixb | 73 | 24765 | 892.8 | 27.74 |
| pmixb | 59 | 18459 | 609.4 | 30.29 |
