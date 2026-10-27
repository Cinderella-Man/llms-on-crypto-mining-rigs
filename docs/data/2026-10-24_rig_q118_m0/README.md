# Q118 M0: the host's share of a served token (2026-10-24)

How much of a decoded token llama-server spends on the rig's two-core host: llama-bench tg128 at one and two threads,
then llama-server with backend sampling (`-bs`) at `-t 2` and `-t 1`, one 1,000-token chat request without and with
streaming each, while the server's threads (`top -H`), the cards' utilization (nvidia-smi every 200 ms), vmstat and the
process's page-fault counters are logged. Flint UD-Q4_K_XL on nine cards, the v2 base build `032b130f4`. Result:
registry row Q118 in [quants/experiments.md](../../../quants/experiments.md) and
[host_decode_loop.md](../../research/notes/host_decode_loop.md). A diagnostic, not an agentic number.

| File | What |
|---|---|
| `server_host_share.csv` | per server thread count (`-t`) and mode (no streaming, streaming): generated tokens, tok/s and ms per token from the server's timings, median and 95th-percentile gap between streamed chunks, wall time; minor and major page faults during the request (and minor faults per prompt-plus-generated token); from `top -H` over the busy seconds: the process's CPU (% of one core), its main thread, the CUDA threads and the rest, thread count, and the system's user / system / I/O-wait shares; the cards' mean utilization while busy; swap-in and swap-out (KiB) during the request |
| `run/m0.sh` | the driver; the client is [bench/server_mmap/client.py](../../../bench/server_mmap/client.py) |

## What it shows

| Question | Result |
|---|---|
| llama-bench tg128 | 30.95 ± 0.22 tok/s at `-t 2`, 28.85 ± 1.52 at `-t 1` |
| llama-server, 1,000 tokens, `-t 2` / `-t 1` | not streamed 31.42 / 32.57 tok/s, streamed 33.31 / 33.59; chunk gaps median 30.3 / 30.0 ms, 95th percentile 30.8 / 30.5 ms |
| Host CPU per token | the process uses 95–100 % of one core, almost all of it the main thread (94.7–98.6 %); the CUDA threads 0.2–0.3 %; system-wide 28–30 % user and 21–22 % system time over both cores, about one core in all |
| Page faults and swap | 28,000–33,000 minor faults per request (27–32 per token); without streaming 226–236 major faults and 4.4–4.6 MB swapped in, with streaming 15–24 and 0.5–0.6 MB: the host has 4 GB of RAM and runs with swap in use |
| The cards | 10.4–10.8 % mean utilization each, the layer split's one card busy at a time |

The server decodes faster than llama-bench (31–34 against 29–31 tok/s), one thread is enough, and streaming costs
nothing; the per-token host work runs on one core and is the main thread's.

## How it was made

On 2026-10-24 10:01–10:08, nothing else on the rig, the build unpacked in `/opt/llama-builds`
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh)) and the model at
`/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf`:

```bash
RIG=rig OUT=. docs/data/2026-10-24_rig_q118_m0/run/m0.sh
```

The server runs with `-c 16384 -np 1 -bs --cache-ram 0 --ctx-checkpoints 0 -t <1|2>` and
`LLAMA_NO_MOE_DOWN_RESCALE=tg`; the client sends one chat request of 1,000 tokens per mode after a 128-token warm-up.
`server_host_share.csv` was computed from the request summaries and the logs in `raw/` (top, nvidia-smi, vmstat and
fault counters, not included). Flint UD-Q4_K_XL is Unsloth's quant of Flint (Apache-2.0).
