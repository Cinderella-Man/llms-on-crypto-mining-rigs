| task | turn | prompt tokens read | from the cache | s2a prompt ms | s1a prompt ms | s2b prompt ms | s1b prompt ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bugfix-javascript-complex-numbers | 1 | 896 | 1549 | 4941.0 | 5138.9 | 5001.5 | 5859.9 |
| bugfix-javascript-grade-school | 1 | 403 | 1549 | 2726.1 | 2741.8 | 2750.9 | 2746.8 |
| bugfix-javascript-phone-number | 1 | 610 | 1549 | 3713.6 | 3703.6 | 3738.8 | 3853.8 |
| bugfix-javascript-space-age | 1 | 803 | 1549 | 5275.5 | 4781.7 | 4899.9 | 4779.5 |
| bugfix-javascript-triangle | 1 | 430 | 1549 | 2835.8 | 2855.6 | 2834.2 | 2890.0 |
| bugfix-javascript-variable-length-quantity | 1 | 577 | 1549 | 3308.7 | 3300.6 | 3304.1 | 3416.9 |
| bugfix-python-affine-cipher | 1 | 2780 | 0 | 27655.0 | 27303.8 | 27521.0 | 27290.1 |
| bugfix-python-list-ops | 1 | 441 | 1549 | 2868.8 | 2864.3 | 2851.5 | 2904.4 |
| bugfix-python-phone-number | 1 | 1130 | 1549 | 6896.5 | 6210.9 | 6403.6 | 6338.9 |
| bugfix-python-pig-latin | 1 | 775 | 1549 | 4632.3 | 4691.3 | 4654.9 | 4763.1 |
| bugfix-python-scale-generator | 1 | 1086 | 1549 | 6452.7 | 6655.0 | 6494.2 | 6666.2 |
| bugfix-python-variable-length-quantity | 1 | 815 | 1549 | 4450.3 | 4672.9 | 4862.1 | 4638.1 |
| **all 12 matched requests** | | 10746 | | **75756.2** (141.8 tok/s) | **74920.4** (143.4 tok/s) | **75316.5** (142.7 tok/s) | **76147.4** (141.1 tok/s) |
| **of them, the 12 first turns** | | 10746 | | **75756.2** (141.8 tok/s) | **74920.4** (143.4 tok/s) | **75316.5** (142.7 tok/s) | **76147.4** (141.1 tok/s) |

| run | requests | tokens generated | decode s | generation tok/s |
|---|---:|---:|---:|---:|
| s2a | 62 | 25570 | 823.0 | 31.07 |
| s1a | 66 | 27587 | 917.9 | 30.05 |
| s2b | 68 | 22062 | 724.9 | 30.43 |
| s1b | 67 | 38682 | 1255.1 | 30.82 |
