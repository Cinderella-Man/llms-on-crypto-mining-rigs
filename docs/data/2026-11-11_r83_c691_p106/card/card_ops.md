## Card 8: the selected launch groups over each whole load (PAD, TOP_K top_k, GET_ROWS top_k_rows; 7.6 µs a group subtracted)

| Load | Label | Groups | Sum, µs | Net, ms |
|---|---|---:|---:|---:|
| 1-off | GET_ROWS top_k_rows | 1178 | 12391.3 | 3.439 |
| 1-off | PAD node_166 | 43 | 3779.5 | 3.453 |
| 1-off | PAD node_176 | 462 | 41204.5 | 37.693 |
| 1-off | PAD node_6219 | 170 | 38135.8 | 36.844 |
| 1-off | PAD node_6221 | 1 | 381.9 | 0.374 |
| 1-off | TOP_K top_k | 1178 | 105506.9 | 96.554 |
| 2-on | GET_ROWS top_k_rows | 676 | 7496.2 | 2.359 |
| 2-on | TOP_K top_k | 676 | 87611.5 | 82.474 |
| 3-on | GET_ROWS top_k_rows | 676 | 7510.5 | 2.373 |
| 3-on | TOP_K top_k | 676 | 87737.3 | 82.600 |
| 4-off | GET_ROWS top_k_rows | 1178 | 12238.7 | 3.286 |
| 4-off | PAD node_166 | 43 | 3774.4 | 3.448 |
| 4-off | PAD node_176 | 462 | 41168.8 | 37.658 |
| 4-off | PAD node_6219 | 170 | 38047.8 | 36.756 |
| 4-off | PAD node_6221 | 1 | 384.0 | 0.376 |
| 4-off | TOP_K top_k | 1178 | 105638.9 | 96.686 |

| Load | Groups | Net, ms | Generation steps | Net ms a generation step |
|---|---:|---:|---:|---:|
| 1-off | 3032 | 178.357 | 167 | 1.068 |
| 2-on | 1352 | 84.833 | 167 | 0.508 |
| 3-on | 1352 | 84.973 | 167 | 0.509 |
| 4-off | 3032 | 178.209 | 167 | 1.067 |
