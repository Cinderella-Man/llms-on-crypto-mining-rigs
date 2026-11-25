| phase | rep | arm | rc | wall s | peak RSS KiB | probe | open+init ms | close ms | RSS after init MiB | after close MiB | found s | device 0 s | /health s | VmHWM at /health KiB | MemAvailable before / lowest KiB | stopped | reply tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| list | 1 | unset | 0 | 2.9965 | 1293372 | skipped (no device has fast BF16; GGML_CUDA_NVML_PROBE=1 runs it) |  |  |  |  | 0.207054 | 0.207495 |  |  |  /  |  |  |
| list | 1 | probe | 0 | 2.5959 | 1293384 | ran (GGML_CUDA_NVML_PROBE=1; library loaded, init ok) | 19.1 | 62.8 | +14.2 | +0.1 | 0.175153 | 0.194717 |  |  |  /  |  |  |
| list | 2 | probe | 0 | 2.4755 | 1293444 | ran (GGML_CUDA_NVML_PROBE=1; library loaded, init ok) | 19.4 | 63.4 | +14.2 | +0.1 | 0.171651 | 0.191493 |  |  |  /  |  |  |
| list | 2 | unset | 0 | 2.4383 | 1293596 | skipped (no device has fast BF16; GGML_CUDA_NVML_PROBE=1 runs it) |  |  |  |  | 0.172023 | 0.172440 |  |  |  /  |  |  |
| list | 3 | unset | 0 | 2.4138 | 1293736 | skipped (no device has fast BF16; GGML_CUDA_NVML_PROBE=1 runs it) |  |  |  |  | 0.172614 | 0.173039 |  |  |  /  |  |  |
| list | 3 | probe | 0 | 2.4850 | 1293400 | ran (GGML_CUDA_NVML_PROBE=1; library loaded, init ok) | 19.0 | 69.2 | +14.2 | +0.1 | 0.186432 | 0.205824 |  |  |  /  |  |  |
| list | 4 | probe | 0 | 2.4843 | 1293952 | ran (GGML_CUDA_NVML_PROBE=1; library loaded, init ok) | 19.1 | 63.7 | +14.2 | +0.1 | 0.197959 | 0.217511 |  |  |  /  |  |  |
| list | 4 | unset | 0 | 2.4647 | 1293732 | skipped (no device has fast BF16; GGML_CUDA_NVML_PROBE=1 runs it) |  |  |  |  | 0.204706 | 0.205104 |  |  |  /  |  |  |
| list | 5 | unset | 0 | 2.4324 | 1293624 | skipped (no device has fast BF16; GGML_CUDA_NVML_PROBE=1 runs it) |  |  |  |  | 0.199252 | 0.199664 |  |  |  /  |  |  |
| list | 5 | probe | 0 | 2.5596 | 1293700 | ran (GGML_CUDA_NVML_PROBE=1; library loaded, init ok) | 19.1 | 63.6 | +14.2 | +0.1 | 0.209035 | 0.228506 |  |  |  /  |  |  |
| load | 1 | unset | 0 | 547.655 | 2705340 | skipped (no device has fast BF16; GGML_CUDA_NVML_PROBE=1 runs it) |  |  |  |  | 0.200983 | 0.201417 | 541.148 | 2705340 | 3022548 / 1407412 |  | 11751 13 1061 369 264 1575 61616 2029 13 271 6820 11 1042 579 11290 279 |
| load | 1 | probe | 0 | 553.363 | 2996188 | ran (GGML_CUDA_NVML_PROBE=1; library loaded, init ok) | 28.0 | 90.3 | +14.0 | +0.1 | 0.217226 | 0.245633 | 546.849 | 2996188 | 3273128 / 1415944 |  | 11751 13 1061 369 264 1575 61616 2029 13 271 6820 11 1042 579 11290 279 |

The two loads' greedy replies: the same token ids.
