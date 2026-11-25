## The greedy decode step (step.sh)

Per load: ms a token on pass 1, pass 2 → both passes (their time over their tokens). Class on both passes.

| Depth (tokens) | Arm | Per load (ms a token) | Spread | Against unset | Class |
|---:|---|---|---:|---:|---|
| 0 | unset | unset1: 21.011, 18.700 → **19.856**; unset2: 20.993, 18.144 → **19.568** | 0.287 |  |  |
|  | sort | sort1: 20.985, 18.206 → **19.595**; sort2: 21.062, 18.034 → **19.548** | 0.047 | -0.140 (-0.71 %) | negligible |
| 8,192 | unset | unset1: 20.969, 19.166 → **20.068**; unset2: 22.663, 19.253 → **20.958** | 0.890 |  |  |
|  | sort | sort1: 21.286, 19.180 → **20.233**; sort2: 21.995, 19.230 → **20.612** | 0.379 | -0.090 (-0.44 %) | negligible |

## The sampled pass (step.sh, SAMPLER_PASS=1)

One request a depth and load with the server line's own sampler and a fixed seed a depth; ms a token (generation time over tokens).

| Depth (tokens) | Arm | Per load (ms a token) | Spread | Against unset | Class |
|---:|---|---|---:|---:|---|
| 0 | unset | unset1: 15.930; unset2: 16.491 | 0.561 |  |  |
|  | sort | sort1: 17.501; sort2: 16.427 | 1.075 | +0.753 (+4.65 %) | negligible |
| 8,192 | unset | unset1: 22.853; unset2: 22.700 | 0.153 |  |  |
|  | sort | sort1: 22.685; sort2: 22.632 | 0.052 | -0.118 (-0.52 %) | negligible |

### The sampled replies

| Depth (tokens) | Arm | Per load (md5 of the token ids) | One reply |
|---:|---|---|---|
| 0 | unset | unset1: `eb1751516ccb`; unset2: `eb1751516ccb` | yes |
|  | sort | sort1: `eb1751516ccb`; sort2: `eb1751516ccb` | yes |
| 8,192 | unset | unset1: `a3384f1e668d`; unset2: `a3384f1e668d` | yes |
|  | sort | sort1: `a3384f1e668d`; sort2: `a3384f1e668d` | yes |

- Tokens out: 256 on every sampled request.
