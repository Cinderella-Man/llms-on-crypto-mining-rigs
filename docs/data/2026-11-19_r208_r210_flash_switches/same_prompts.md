| task | turn | prompt tokens read | from the cache | v4rc2 prompt ms | f5c1 prompt ms | f5s1 prompt ms | f5t1 prompt ms | f5c2 prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 6736.7 | 6797.1 | 6665.9 | 6688.6 | 6734.3 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 3006.7 | 3000.2 | 3039.1 | 3011.2 | 3043.5 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 6203.7 | 6308.7 | 6550.7 | 6497.5 | 6784.2 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 6650.7 | 6615.4 | 6611.6 | 6614.6 | 7286.2 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 3088.7 | 3068.9 | 3151.5 | 3098.3 | 3104.9 |
| bugfix-javascript-triangle | 2 | 1705 | 2031 | 8447.2 | 8252.9 | 8225.1 | 8248.2 | 8286.2 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3662.0 | 4817.4 | 4806.2 | 4775.9 | 4934.5 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 28567.5 | 27933.5 | 27954.2 | 28132.4 | 28128.3 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 3163.1 | 3074.7 | 3116.7 | 3072.4 | 3132.1 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 7386.1 | 7326.6 | 7245.0 | 7377.5 | 7502.9 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 7198.7 | 7924.0 | 7448.8 | 7972.7 | 7018.9 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 6616.2 | 6437.8 | 6453.8 | 6740.5 | 6470.6 |
| **all 12 matched requests** | | 11676 | | **90727.2** (128.7 tok/s) | **91557.1** (127.5 tok/s) | **91268.7** (127.9 tok/s) | **92229.9** (126.6 tok/s) | **92426.7** (126.3 tok/s) |
| **of them, the 11 first turns** | | 9971 | | **82280.0** (121.2 tok/s) | **83304.2** (119.7 tok/s) | **83043.6** (120.1 tok/s) | **83981.7** (118.7 tok/s) | **84140.4** (118.5 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| v4rc2 | 91 | 39678 | 1469.6 | 27.00 |
| f5c1 | 103 | 40857 | 1514.5 | 26.98 |
| f5s1 | 82 | 25651 | 921.0 | 27.85 |
| f5t1 | 102 | 32381 | 1196.4 | 27.07 |
| f5c2 | 69 | 37889 | 1382.1 | 27.41 |
