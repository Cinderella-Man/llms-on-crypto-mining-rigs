| task | turn | prompt tokens read | from the cache | o5q8a prompt ms | o5kdpa prompt ms | o5q8b prompt ms | o5kdpb prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1511 | 7911.1 | 7673.3 | 7945.6 | 7718.5 |
| bugfix-javascript-grade-school | 1 | 403 | 1511 | 6537.1 | 6339.1 | 6449.0 | 6376.9 |
| bugfix-javascript-grade-school | 2 | 1750 | 1980 | 10194.9 | 9693.5 | 10232.6 | 9750.2 |
| bugfix-javascript-phone-number | 1 | 610 | 1511 | 7773.1 | 7533.5 | 7735.5 | 7203.6 |
| bugfix-javascript-phone-number | 2 | 1455 | 2183 | 9430.8 | 8991.5 | 9473.8 | 9025.4 |
| bugfix-javascript-space-age | 1 | 803 | 1511 | 7852.8 | 7603.2 | 7843.0 | 7641.7 |
| bugfix-javascript-space-age | 2 | 642 | 2376 | 7714.7 | 7480.1 | 7762.2 | 7472.8 |
| bugfix-javascript-triangle | 1 | 430 | 1511 | 6712.7 | 6702.3 | 6804.6 | 6689.6 |
| bugfix-javascript-triangle | 2 | 1358 | 2004 | 9198.0 | 8817.4 | 9188.4 | 8871.5 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1511 | 7577.4 | 7496.6 | 7585.4 | 7533.8 |
| bugfix-python-affine-cipher | 1 | 2742 | 0 | 24786.9 | 24701.5 | 25341.0 | 24809.3 |
| bugfix-python-affine-cipher | 2 | 1185 | 2811 | 9172.5 | 8681.3 | 9212.0 | 8710.7 |
| bugfix-python-list-ops | 1 | 441 | 1511 | 6795.8 | 6300.3 | 6743.3 | 6548.5 |
| bugfix-python-list-ops | 2 | 1415 | 2016 | 9297.9 | 8883.3 | 9325.0 | 8935.7 |
| bugfix-python-phone-number | 1 | 1130 | 1511 | 8615.0 | 8295.5 | 8612.2 | 8338.4 |
| bugfix-python-pig-latin | 1 | 775 | 1511 | 7703.9 | 7490.9 | 7766.9 | 7555.2 |
| bugfix-python-pig-latin | 2 | 907 | 2353 | 8230.1 | 7895.5 | 8294.3 | 7942.1 |
| bugfix-python-scale-generator | 1 | 1086 | 1511 | 8519.9 | 8206.6 | 8523.1 | 8261.4 |
| bugfix-python-scale-generator | 2 | 1448 | 2659 | 9559.5 | 9022.0 | 9595.8 | 9072.4 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1511 | 7756.3 | 7535.8 | 7792.9 | 7574.2 |
| bugfix-python-variable-length-quantity | 2 | 1886 | 2395 | 10603.5 | 9952.8 | 10616.2 | 10033.4 |
| **all 21 matched requests** | | 22754 | | **191943.9** (118.5 tok/s) | **185295.9** (122.8 tok/s) | **192842.7** (118.0 tok/s) | **186065.4** (122.3 tok/s) |
| **of them, the 12 first turns** | | 10708 | | **108542.0** (98.7 tok/s) | **105878.6** (101.1 tok/s) | **109142.5** (98.1 tok/s) | **106251.2** (100.8 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| o5q8a | 60 | 25385 | 1244.2 | 20.40 |
| o5kdpa | 68 | 20995 | 1024.1 | 20.50 |
| o5q8b | 64 | 23411 | 1162.0 | 20.15 |
| o5kdpb | 66 | 20467 | 1004.6 | 20.37 |
