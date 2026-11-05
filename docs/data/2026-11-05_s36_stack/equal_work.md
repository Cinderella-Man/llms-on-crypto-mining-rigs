
Tasks every arm scored: 12 tasks (javascript-complex-numbers, javascript-grade-school, javascript-phone-number, javascript-space-age, javascript-triangle, javascript-variable-length-quantity, python-affine-cipher, python-list-ops, python-phone-number, python-pig-latin, python-scale-generator, python-variable-length-quantity)

| arm | passed | minutes | per task vs s35stack (geomean) | generated tokens | decode s | gen tok/s | new prompt tokens | prefill s | prompt tok/s | rest s | s35stack's work at this arm's rates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s35stack | 12 / 12 | 34.8 | 1.00× | 22654 | 1208 | 18.76 | 37849 | 558 | 67.8 | 322 | 34.8 min, 1.00× |
| s36ref | 12 / 12 | 29.5 | 1.15× | 21855 | 1152 | 18.97 | 38829 | 563 | 69.0 | 56 | 34.4 min, 1.01× |
| s36adapt | 12 / 12 | 35.1 | 1.12× | 25957 | 1437 | 18.07 | 40719 | 582 | 69.9 | 86 | 35.3 min, 0.99× |

The same without the tasks any arm ran away on: 11 tasks (javascript-complex-numbers, javascript-grade-school, javascript-phone-number, javascript-space-age, javascript-triangle, javascript-variable-length-quantity, python-affine-cipher, python-list-ops, python-phone-number, python-scale-generator, python-variable-length-quantity)

| arm | passed | minutes | per task vs s35stack (geomean) | generated tokens | decode s | gen tok/s | new prompt tokens | prefill s | prompt tok/s | rest s | s35stack's work at this arm's rates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s35stack | 11 / 11 | 30.1 | 1.00× | 19016 | 998 | 19.05 | 33531 | 493 | 68.1 | 315 | 30.1 min, 1.00× |
| s36ref | 11 / 11 | 26.2 | 1.13× | 19333 | 1013 | 19.08 | 35012 | 504 | 69.5 | 53 | 29.9 min, 1.01× |
| s36adapt | 11 / 11 | 23.1 | 1.23× | 16444 | 832 | 19.76 | 32812 | 480 | 68.3 | 72 | 29.5 min, 1.02× |
