### The first request (79 prompt tokens, 256 generated, temperature 0)

| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting pass | idle before the first verify pass | other, head | wait, card 0, last verify pass | idle after it | wait, card 8, last catch-up | other, tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fka250 | 10094.1 | 9140.1 | **954.0** | — | 287.0 | 129.5 | 283.8 | 248.9 | — | 4.7 |

### The second request (2,760 prompt tokens, 256 generated, temperature 1.0)

| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting passes | other, head | wait, card 4, last step | other, tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| fka250 | 9535.1 | 9025.9 | **509.2** | 265.7 | 21.8 | — | 221.6 |
