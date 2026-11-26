| task | turn | prompt tokens read | from the cache | ca prompt ms | r05a prompt ms | cb prompt ms | r05b prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 5805.7 | 5771.6 | 5833.3 | 4997.3 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 2705.7 | 2733.3 | 2779.6 | 2792.1 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 5329.3 | 5428.2 | 5289.0 | 4795.4 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 2803.0 | 2826.4 | 2870.0 | 2875.9 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3357.4 | 3369.7 | 3372.9 | 3410.4 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 26709.0 | 27005.7 | 27192.2 | 27267.7 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 2851.7 | 2848.5 | 2836.7 | 2896.2 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 6214.6 | 6150.0 | 6280.4 | 6308.2 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 6691.4 | 6566.8 | 6626.1 | 6510.8 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 4438.1 | 4437.4 | 4535.8 | 4606.7 |
| **all 10 matched requests** | | 9361 | | **66906.0** (139.9 tok/s) | **67137.5** (139.4 tok/s) | **67616.0** (138.4 tok/s) | **66460.8** (140.8 tok/s) |
| **of them, the 10 first turns** | | 9361 | | **66906.0** (139.9 tok/s) | **67137.5** (139.4 tok/s) | **67616.0** (138.4 tok/s) | **66460.8** (140.8 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| ca | 84 | 50096 | 1724.3 | 29.05 |
| r05a | 63 | 19756 | 626.5 | 31.54 |
| cb | 71 | 33242 | 1053.3 | 31.56 |
| r05b | 73 | 40493 | 1264.0 | 32.04 |
