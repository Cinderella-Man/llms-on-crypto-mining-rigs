| run | middle context, tokens | requests | tokens generated | decode s | tok/s | ms a token |
|---|---|---:|---:|---:|---:|---:|
| o5oc1 | under 3,000 | 12 | 791 | 38.3 | 20.65 | 48.43 |
| o5oc1 | 3,000 to 4,500 | 18 | 3508 | 174.4 | 20.12 | 49.71 |
| o5oc1 | 4,500 to 6,000 | 22 | 7309 | 375.7 | 19.45 | 51.41 |
| o5oc1 | 6,000 to 8,000 | 12 | 7910 | 421.0 | 18.79 | 53.23 |
| o5oc1 | 8,000 and over | 3 | 704 | 40.6 | 17.35 | 57.64 |
| o5oc1 | **all** | 67 | 20222 | 1050.0 | **19.26** | 51.93 |
| o5t0a | under 3,000 | 12 | 788 | 37.9 | 20.77 | 48.16 |
| o5t0a | 3,000 to 4,500 | 17 | 4543 | 226.8 | 20.03 | 49.93 |
| o5t0a | 4,500 to 6,000 | 18 | 5508 | 282.4 | 19.50 | 51.28 |
| o5t0a | 6,000 to 8,000 | 6 | 4945 | 261.3 | 18.92 | 52.84 |
| o5t0a | 8,000 and over | 5 | 4770 | 266.4 | 17.91 | 55.84 |
| o5t0a | **all** | 58 | 20554 | 1074.9 | **19.12** | 52.30 |

At the same mix of contexts (the 5 buckets every run decoded in, weighted by the 40,776 tokens all runs generated there: under 3,000 3.9 %, 3,000 to 4,500 19.7 %, 4,500 to 6,000 31.4 %, 6,000 to 8,000 31.5 %, 8,000 and over 13.4 %):

| run | ms a token at the same mix | tok/s at the same mix |
|---|---:|---:|
| o5oc1 | 52.37 | 19.10 |
| o5t0a | 52.00 | 19.23 |
