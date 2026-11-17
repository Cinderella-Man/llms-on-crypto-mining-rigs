| Prompt | Try | Tokens (server) | Fact depths, tokens from the end | pumpernickel (48,000) | dundee (32,000) | volvo (16,000) | lighthouse (4,000) | metropolis (1,000) | hibiscus (300) | Found | Finish | Reply tokens | Prompt tok/s | Reply's first line |
|---|---:|---:|---|---|---|---|---|---|---|---:|---|---:|---:|---|
| recall-8000 | 1 | 8,074 | 8,029, 7,992, 7,950, 4,009, 993, 286 | yes | yes | yes | yes | yes | yes | 6/6 | stop | 1621 | 329.71 | - Has a cat named Pumpernickel |
| recall-56460 | 1 | 56,437 | 47,998, 32,014, 15,987, 4,009, 993, 286 | **no** | **no** | **no** | **no** | **no** | **no** | 0/6 | length | 8192 | 204.81 | (no answer: the reply has reasoning only) |
| recall-56460 | 2 | 56,437 | 47,998, 32,014, 15,987, 4,009, 993, 286 | **no** | **no** | **no** | **no** | **no** | **no** | 0/6 | length | 8192 | 205.59 | (no answer: the reply has reasoning only) |
