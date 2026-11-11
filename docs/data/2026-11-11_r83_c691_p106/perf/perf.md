## The operator times, card 0, µs a launch

| Op | Rows | Pass 1 | Pass 2 |
|---|---:|---:|---:|
| TOP_K | 1 | 109.15 | 105.55 |
| TOP_K | 2 | 142.41 | 145.14 |
| TOP_K | 4 | 227.05 | 227.38 |
| PAD | 4 | 213.59 | 214.96 |

## A verify pass's four chains: the launches the switch changes, µs

| Pass | One-row TOP_K | Four of them | PAD (4 rows) | Off: four TOP_K + PAD | On: one four-row TOP_K | Saving | Four rows / one row |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 109.15 | 436.60 | 213.59 | 650.19 | 227.05 | 423.14 | 2.08 |
| 2 | 105.55 | 422.20 | 214.96 | 637.16 | 227.38 | 409.78 | 2.15 |
