### The first request (79 prompt tokens, 256 generated, temperature 0)

| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting pass | idle before the first verify pass | other, head | wait, card 0, last verify pass | idle after it | wait, card 8, last catch-up | other, tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f1 | 10044.5 | 9173.6 | **870.9** | — | 375.5 | 203.2 | — | 228.4 | — | 63.6 |
| f1-2 | 10095.4 | 9145.6 | **949.8** | — | 460.1 | 197.5 | — | 231.4 | — | 60.7 |

### The second request (2,760 prompt tokens, 256 generated, temperature 1.0)

| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting passes | other, head | wait, card 4, last step | other, tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| f1 | 9859.0 | 9314.0 | **544.9** | — | 303.0 | — | 241.9 |
| f1-2 | 10650.9 | 10121.5 | **529.3** | — | 170.2 | — | 359.1 |
