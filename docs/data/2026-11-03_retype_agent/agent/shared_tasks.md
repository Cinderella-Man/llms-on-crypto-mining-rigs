
Tasks every arm scored: 8 tasks (javascript-grade-school, javascript-phone-number, javascript-space-age, javascript-triangle, python-affine-cipher, python-list-ops, python-phone-number, python-variable-length-quantity)

| arm | passed | minutes | per task vs rel (geomean) | generated tokens | decode s | gen tok/s | new prompt tokens | prefill s | prompt tok/s | rest s | rel's work at this arm's rates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rel | 8 / 8 | 38.6 | 1.00× | 12153 | 1549 | 7.85 | 25135 | 422 | 59.6 | 344 | 38.6 min, 1.00× |
| pipe0 | 7 / 8 | 42.3 | 0.89× | 10142 | 1287 | 7.88 | 23543 | 387 | 60.8 | 863 | 38.3 min, 1.01× |
| hcq8 | 8 / 8 | 25.6 | 1.35× | 9652 | 1070 | 9.02 | 22682 | 374 | 60.7 | 91 | 35.1 min, 1.10× |

The same without the tasks any arm ran away on: 6 tasks (javascript-grade-school, javascript-phone-number, javascript-space-age, javascript-triangle, python-affine-cipher, python-phone-number)

| arm | passed | minutes | per task vs rel (geomean) | generated tokens | decode s | gen tok/s | new prompt tokens | prefill s | prompt tok/s | rest s | rel's work at this arm's rates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rel | 6 / 6 | 18.4 | 1.00× | 5794 | 698 | 8.31 | 19427 | 330 | 58.8 | 77 | 18.4 min, 1.00× |
| pipe0 | 6 / 6 | 23.3 | 0.89× | 7107 | 886 | 8.02 | 19695 | 325 | 60.6 | 187 | 18.7 min, 0.99× |
| hcq8 | 6 / 6 | 14.0 | 1.32× | 4998 | 533 | 9.37 | 17237 | 285 | 60.5 | 24 | 16.9 min, 1.09× |
