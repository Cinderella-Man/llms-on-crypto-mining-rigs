| task | turn | prompt tokens read | from the cache | r075a prompt ms | r075b prompt ms | c600a prompt ms | t1400a prompt ms | c600b prompt ms | t1400b prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6360.9 | 6378.5 | 6509.9 | 5998.3 | 6356.6 | 5023.4 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 2765.4 | 2783.7 | 2780.0 | 2723.6 | 2769.9 | 2786.8 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 5696.2 | 6314.2 | 6328.3 | 3837.7 | 6436.9 | 3808.7 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6432.6 | 6303.2 | 6435.7 | 4794.9 | 6438.8 | 5238.7 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 2937.3 | 2879.3 | 2844.4 | 2863.2 | 2827.4 | 2807.9 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 4306.4 | 4418.2 | 4365.3 | 3385.1 | 4407.1 | 3370.2 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 27417.3 | 27446.0 | 27406.7 | 27334.5 | 27224.0 | 27406.9 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 2858.6 | 2849.9 | 2892.1 | 2905.5 | 2869.6 | 2892.7 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 7121.8 | 7747.2 | 7254.9 | 6274.1 | 7206.8 | 6192.6 |
| bugfix-python-pig-latin | 1 | 775 | 1549 | 6133.0 | 6170.4 | 6171.0 | 4722.1 | 6148.1 | 4648.8 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 6818.3 | 6851.4 | 6861.0 | 6637.3 | 6865.3 | 5965.8 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 6123.1 | 6314.4 | 6216.9 | 4536.8 | 6159.3 | 4516.1 |
| **all 12 matched requests** | | 10746 | | **84970.8** (126.5 tok/s) | **86456.5** (124.3 tok/s) | **86066.3** (124.9 tok/s) | **76013.0** (141.4 tok/s) | **85709.8** (125.4 tok/s) | **74658.7** (143.9 tok/s) |
| **of them, the 12 first turns** | | 10746 | | **84970.8** (126.5 tok/s) | **86456.5** (124.3 tok/s) | **86066.3** (124.9 tok/s) | **76013.0** (141.4 tok/s) | **85709.8** (125.4 tok/s) | **74658.7** (143.9 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| r075a | 63 | 20391 | 666.4 | 30.60 |
| r075b | 64 | 27522 | 905.7 | 30.39 |
| c600a | 60 | 22271 | 717.2 | 31.05 |
| t1400a | 66 | 31360 | 1021.1 | 30.71 |
| c600b | 68 | 29745 | 945.6 | 31.46 |
| t1400b | 67 | 18328 | 593.8 | 30.87 |
