# Q116/Q79: long context on nine cards, host mask against the card-built mask, with host memory and swap (2026-10-24)

llama-batched-bench with Flint K2h-r16 on nine cards at 32k and 64k prompt tokens (a 131k run was stopped after 36 s:
it would take hours), the attention mask built on the host (`LLAMA_GPU_KQ_MASK=0`) against built on each card (`1`),
on the gpumask2 build `6f943f792` (fork branch `pascal-p106-v2-gpumask2`, Q110's prompt path and Q116's decode path).
Host memory and swap were logged every second to answer Q79: is the slower 64k prompt host swapping? Results: registry
rows Q116 and Q79 in [quants/experiments.md](../../../quants/experiments.md),
[prefill_pascal.md](../../research/notes/prefill_pascal.md) and, for the prompt attention that followed (Q163),
[pascal_attention_long_context.md](../../research/notes/pascal_attention_long_context.md).

| File | What |
|---|---|
| `batched_bench.csv` | llama-batched-bench's rows: mask (host, cards), `-ub`, start time, prompt tokens, generated tokens, batch, KV cells, prompt time and tok/s, decode time and tok/s, total. The 32k card-mask row is missing: the log kept only the last four lines of that run, which held warnings |
| `memory_swap.csv` | per run, from `/proc/meminfo` and vmstat every second: seconds logged, lowest and median MemAvailable, swap free at the start and its lowest value, swap in use (vmstat `swpd`) lowest and highest, MiB swapped in and out, seconds with swap traffic, mean I/O wait, MiB read from block devices (the model load included) |
| `run/longctx.sh` | the driver: `test-backend-ops -o KQ_MASK` on card 0, then the runs |

## What it shows

| Run | Prompt tok/s | Decode tok/s after the prompt | MemAvailable, lowest | Swap in / out during the run |
|---|---|---|---|---|
| 32k, host mask, `-ub 512` | 106.98 | 23.10 | 1,365 MiB | 178 / 265 MiB |
| 32k, card mask, `-ub 512` | (not kept) | (not kept) | 1,489 MiB | 176 / 257 MiB |
| 64k, host mask, `-ub 256` | 67.79 | 15.96 | 1,438 MiB | 126 / 261 MiB |
| 64k, card mask, `-ub 256` | 68.91 | **17.90** (+12 %) | 1,573 MiB | 139 / 264 MiB |

The host never ran short of memory (at least 1,365 MiB available, swap in use 355–601 MiB of 3,840 MiB) and the
swap traffic is the same at 32k and 64k and for both masks, so swapping does not explain why a 64k prompt reads at
68 tok/s against 107 at 32k: long prompts are attention-bound (Q163). The card-built mask leaves prompts unchanged and
speeds up decoding at 64k depth by 12 %. `KQ_MASK` passed its operator test on a P106 (2 of 2 backends).

## How it was made

On 2026-10-24 18:04–19:03, nothing else on the rig, the build unpacked in `/opt/llama-builds`
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) with `TESTS=1`) and K2h-r16 at
`/mnt/models/flint/flint-q113-K2h-r16.gguf`:

```bash
RIG=rig OUT=. docs/data/2026-10-24_rig_longctx/run/longctx.sh
```

Each run: `llama-batched-bench -ngl 99 -fa on -c <prompt + 512> -b 2048 -ub <512 at 32k, 256 above> -npp <prompt>
-ntg 64 -npl 1` with `LLAMA_NO_MOE_DOWN_RESCALE=tg`, beside `vmstat 1` and a one-second loop over MemAvailable and
SwapFree. `memory_swap.csv` was computed from those logs (not included). Flint K2h-r16: recipe in
[quants/experiments.md](../../../quants/experiments.md) (Q113), Flint is Apache-2.0.
