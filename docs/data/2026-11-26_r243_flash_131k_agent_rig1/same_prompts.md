| task | turn | prompt tokens read | from the cache | ca prompt ms | l131a prompt ms | cb prompt ms | l131b prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6390.5 | 5860.5 | 5497.4 | 5381.9 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 3044.0 | 3292.6 | 2990.9 | 3306.3 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 4209.2 | 4138.7 | 4212.5 | 4143.8 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 5860.5 | 5705.5 | 5279.2 | 5277.0 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 3125.8 | 3539.1 | 3131.5 | 3431.2 |
| bugfix-javascript-triangle | 2 | 1705 | 2031 | 8324.1 | 8255.0 | 8262.7 | 8296.2 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3721.5 | 3818.4 | 3699.8 | 3845.4 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 27406.9 | 19480.4 | 28128.4 | 19460.9 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 3141.6 | 3352.4 | 3117.0 | 3436.4 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 6866.2 | 6829.0 | 6906.0 | 6750.6 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 7265.7 | 7138.1 | 7087.8 | 6959.5 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 4949.6 | 5142.2 | 4952.9 | 5158.1 |
| **all 12 matched requests** | | 11676 | | **84305.7** (138.5 tok/s) | **76551.8** (152.5 tok/s) | **83266.1** (140.2 tok/s) | **75447.3** (154.8 tok/s) |
| **of them, the 11 first turns** | | 9971 | | **75981.5** (131.2 tok/s) | **68296.8** (146.0 tok/s) | **75003.4** (132.9 tok/s) | **67151.1** (148.5 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| ca | 73 | 24497 | 842.8 | 29.07 |
| l131a | 64 | 24305 | 836.6 | 29.05 |
| cb | 64 | 25305 | 878.4 | 28.81 |
| l131b | 85 | 34440 | 1274.7 | 27.02 |
