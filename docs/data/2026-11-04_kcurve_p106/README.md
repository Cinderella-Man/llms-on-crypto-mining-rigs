# Q591: the k-curve on the P106s, T(k) of a whole-model k-token pass (2026-11-04)

What a verify pass of k tokens costs on nine P106-090s, the number that decides how long a draft pays
([adaptive draft length](../2026-11-04_adaptive_draft/README.md) reads it as `LLAMA_SPEC_ADAPT_COST`). Fork branch
`pascal-p106-v3-b12076-ladder3` `7bc1552f4` (every switch at its default unless an arm says otherwise), Swift
Qwen3.8-Flash-Next GSQ-RCO Q2_0 as published (BF16 hyper-connections) and its hcq8 retype (hyper-connections in Q8_0,
router in F16; [retype](../2026-11-03_retype_agent/README.md)). The method is S21's
([verify cost](../2026-11-03_verify_cost/README.md)): `llama-bench -p 1,2,...,8 -n 0`, five repetitions after a warm-up.

## T(k), ms per pass (median of 5; standard deviation 2–4 ms)

| k | Swift Q2_0 | hcq8 retype | hcq8, k-column and multi-token kernels off | S21, the old base (Swift Q2_0) |
|---:|---:|---:|---:|---:|
| 1 | **82.4** | 98.8 | 92.1 | 136.0 |
| 2 | 122.1 | 149.8 | 130.9 | 185.8 |
| 3 | 149.4 | 177.5 | 170.6 | 244.6 |
| 4 | **173.0** | 204.8 | 216.4 | 305.4 |
| 5 | 217.1 | 243.8 | 249.6 | 359.9 |
| 6 | 249.1 | 277.6 | 301.7 | |
| 7 | 280.1 | 313.1 | 343.8 | |
| 8 | 319.1 | 353.5 | 387.9 | |
| T(5) / T(1) | 2.63 | 2.47 | 2.71 | 2.65 |

"Off" is `GGML_CUDA_MMVQ_PASCAL_KCOL=0 GGML_CUDA_MMVQ_PASCAL_MTOK=0`: the dense products and the routed experts of 2–8
tokens on the kernels before Q591 (and the one-column Q3_K and short-row Q8_0 products on the one-token kernels).

- **The published Q2_0 is faster than the retype at every k** (−16 % at one token): with the Pascal BF16 kernel (C54)
  the exact hyper-connections cost less than the retype's 320-wide Q8_0 rows. One token in 82.4 ms is ~12 tok/s before
  any draft.
- **The curve is far from flat:** each extra token adds 24–44 ms (T(8) / T(1) = 3.9). The cheapest step is k = 3 → 4
  (+23.6 ms); k = 4 → 5 costs +44. A verify of width 4 (a draft of 3) is where a draft pays most: with a drafter step of
  20 ms, est. ~15 tok/s at a per-position acceptance of 0.8 and ~18 at 0.9.
- **On the retype the new kernels lose at 1–3 tokens and win from 4** (+6.7, +18.9, +6.9 ms at k = 1–3; −11.6 to −34.4
  at k = 4–8).

## Files

| File | What it is |
|---|---|
| [tk_wall.tsv](tk_wall.tsv) | per arm and k: mean and median ms, standard deviation, ms per token, T(k) / T(1), the step from k − 1 (`hcq8_prof`: the same passes under `GGML_CUDA_PROFILE`, whose events add ~20 ms at one token) |
| [cost/t_k_q2_0.txt](cost/t_k_q2_0.txt), [cost/t_k_hcq8.txt](cost/t_k_hcq8.txt), [cost/t_k_hcq8_off.txt](cost/t_k_hcq8_off.txt) | `LLAMA_SPEC_ADAPT_COST` files: one T(k) in ms per line, k = 1–8 (past the end the server takes the last slope) |
| [run/tk.py](run/tk.py) | the table and the cost files from llama-bench's JSON |

The cost files hold llama-bench's passes, which ask for the last token's logits only; a verify needs all k rows of the
output layer, which the k-column Q5_K kernel adds at a small cost (`--verify-extra` in `tk.py` adds it per k).

## Run

On the rig, idle, from a folder holding S21's `verify_cost.sh`, `mem_guard.sh` and `smoke_request.json`
([verify cost run/](../2026-11-03_verify_cost/run/), [MTP run/](../2026-11-03_flash_next_mtp/run/)):

```bash
BIN=<-ladder3 build>/bin MODEL=<first shard> DRAFT=<any gguf; unused by these steps> OUT=<dir> STEPS='wall prof' KS=1,2,3,4,5,6,7,8 \
  setsid nohup bash run/verify_cost.sh > verify_cost.out 2>&1 < /dev/null &
# the off arm: the same with GGML_CUDA_MMVQ_PASCAL_KCOL=0 GGML_CUDA_MMVQ_PASCAL_MTOK=0 in the environment and STEPS=wall
python3 -I run/tk.py --cost cost q2_0=<OUT>/bench-wall.json ...
```

`verify_cost.sh` runs `llama-bench -m MODEL -ngl 99 -fa on -b 2048 -ub 512 -ts 6/6/6/6/5/6/5/6/3 -lm dio --no-host 1 -p KS -n 0
-r 5 -o json` under the host-memory guard after a logged quiet check (no llama process, the cards empty). Each arm took
~5 minutes including the load.
