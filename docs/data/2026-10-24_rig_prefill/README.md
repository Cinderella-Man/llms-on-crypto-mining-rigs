# Q109/Q89: the prompt micro-batch size and the scheduler's copies on the combined build (2026-10-24)

Which `-ub` and how many scheduler copies read prompts fastest on the combined v2 build `b3052a4f8`: llama-bench pp2048
and pp8192 at `-b 8192` and `-ub` 128, 256, 512 and 1024 on nine and six cards, then the prefill build with four
scheduler copies (`82bbb3aa4`, llama.cpp's default) against the same build compiled with one copy, nine cards. Flint
K2h-r16. Results: registry rows Q109 and Q89 in [quants/experiments.md](../../../quants/experiments.md) and
[prefill_pascal.md](../../research/notes/prefill_pascal.md). Diagnostics, not agentic numbers.

| File | What |
|---|---|
| `llama_bench.csv` | 20 rows: build (`combined`, `prefill-4copies`, `prefill-1copy`), cards, start time, `n_batch`, `n_ubatch`, test (pp2048, pp8192), tok/s, standard deviation (two repetitions) |
| `run/prefill_grid.sh` | the driver |

## What it shows

| Prompt tokens per second | `-ub` 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| nine cards, pp2048 | 566.5 | 598.2 | **611.9** | 415.0 |
| nine cards, pp8192 | 564.5 | 635.7 | **714.5** | 691.9 |
| six cards, pp2048 | 484.2 | **513.3** | 504.2 | 414.3 |
| six cards, pp8192 | 486.8 | 546.0 | **608.2** | 601.9 |

`-ub 512` is fastest for long prompts on both card counts (pp2048 on six cards slightly prefers 256); `-ub 1024` loses,
most at 2,048 tokens, where only two micro-batches fill nine cards. One scheduler copy instead of four reads an
8,192-token prompt at 131.7 (`-ub 128`) and 178.2 (`-ub 512`) tok/s against 555.7 and 694.6: the four copies let the
cards work on different micro-batches at once (pipeline parallelism), worth 3.9–4.2×.

## How it was made

On 2026-10-24 17:29–18:04, nothing else on the rig, the three builds unpacked in `/opt/llama-builds`
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh); the one-copy build with
`SCHED_COPIES=1`, which sets `GGML_SCHED_MAX_COPIES=1`) and K2h-r16 at `/mnt/models/flint/flint-q113-K2h-r16.gguf`:

```bash
RIG=rig OUT=. docs/data/2026-10-24_rig_prefill/run/prefill_grid.sh    # prefill_grid.md
```

llama-bench `-ngl 99 -fa on -p 2048,8192 -n 0 -b 8192 -ub 128,256,512,1024 -r 2` (the first prompt run warms the
cards), `LLAMA_NO_MOE_DOWN_RESCALE=tg`; its rows were turned into `llama_bench.csv`. Flint K2h-r16: recipe in
[quants/experiments.md](../../../quants/experiments.md) (Q113), Flint is Apache-2.0.
