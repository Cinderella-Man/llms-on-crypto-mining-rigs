| task | turn | prompt tokens read | from the cache | rig1-valid1 prompt ms | rig2-ka prompt ms | rig2-o3a prompt ms | rig2-o3b prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 17557.1 | 17571.8 | 17663.6 | 17493.7 |
| bugfix-javascript-complex-numbers | 2 | 3068 | 2471 | 22941.9 | 22967.7 | 22971.1 | 22989.6 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 9165.1 | 9287.0 | 9337.0 | 9300.5 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 13637.4 | 13857.8 | 14029.1 | 14049.6 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 18621.6 | 18136.6 | 18121.2 | 18145.2 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 16903.2 | 16563.8 | 16596.8 | 16573.7 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 13222.5 | 12758.4 | 12702.8 | 12700.8 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 9815.4 | 9609.4 | 9607.8 | 9665.5 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 17982.6 | 17796.8 | 17793.3 | 17824.4 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 12898.1 | 13242.6 | 13248.2 | 13284.8 |
| bugfix-javascript-variable-length-quantity | 2 | 2065 | 2158 | 20968.5 | 20714.2 | 20137.6 | 20149.3 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 38292.3 | 41523.3 | 36564.0 | 37477.5 |
| bugfix-python-affine-cipher | 2 | 1185 | 2810 | 18668.8 | 18933.1 | 18189.2 | 18236.6 |
| bugfix-python-affine-cipher | 3 | 448 | 4052 | 9523.0 | 9237.9 | 9144.2 | 9150.2 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 9951.0 | 9834.8 | 9514.5 | 9541.1 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 19080.9 | 18022.8 | 17999.8 | 18238.3 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 19644.3 | 19193.7 | 19086.8 | 19133.0 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 16628.3 | 16160.7 | 16038.7 | 16410.0 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 16853.3 | 16467.1 | 16585.3 | 16300.8 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 19152.8 | 19037.8 | 19074.2 | 18931.5 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 16802.7 | 16474.5 | 16505.1 | 16496.9 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2395 | 19791.9 | 19514.0 | 19495.7 | 19553.7 |
| **all 22 matched requests** | | 25137 | | **378102.5** (66.5 tok/s) | **376905.9** (66.7 tok/s) | **370406.1** (67.9 tok/s) | **371646.7** (67.6 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **200447.7** (53.4 tok/s) | **202357.3** (52.9 tok/s) | **197265.8** (54.3 tok/s) | **198357.8** (54.0 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| rig1-valid1 | 62 | 18276 | 1230.1 | 14.86 |
| rig2-ka | 63 | 18032 | 1097.9 | 16.42 |
| rig2-o3a | 71 | 22823 | 1407.2 | 16.22 |
| rig2-o3b | 60 | 15969 | 968.6 | 16.49 |
