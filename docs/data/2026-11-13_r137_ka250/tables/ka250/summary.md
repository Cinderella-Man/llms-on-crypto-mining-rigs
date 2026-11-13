### The first request (79 prompt tokens, 256 generated, temperature 0)

| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting pass | idle before the first verify pass | other, head | wait, card 0, last verify pass | idle after it | wait, card 8, last catch-up | other, tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| f1ka250 | 10111.1 | 9158.3 | **952.8** | — | 454.0 | 205.6 | — | 0.0 | — | 64.1 |
| f1ka250-2 | 10104.1 | 9147.2 | **956.9** | — | 467.4 | 199.5 | — | 0.0 | — | 63.1 |
| f0ka250 | 10064.2 | 9110.5 | **953.7** | — | 285.2 | 129.8 | 283.6 | 250.5 | — | 4.7 |
| f0ka250-2 | 10064.6 | 9112.7 | **951.9** | — | 285.0 | 130.0 | 283.3 | 249.0 | — | 4.7 |

### The second request (2,760 prompt tokens, 256 generated, temperature 1.0)

| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting passes | other, head | wait, card 4, last step | other, tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| f1ka250 | 9925.8 | 9398.9 | **527.0** | — | 285.9 | — | 241.1 |
| f1ka250-2 | 10654.1 | 10127.1 | **527.0** | — | 169.4 | — | 248.7 |
| f0ka250 | 2407.5 | 1890.4 | **517.1** | 253.8 | 94.0 | — | 169.3 |
| f0ka250-2 | 10715.1 | 9972.6 | **742.6** | 253.9 | 21.5 | 264.4 | 202.8 |
