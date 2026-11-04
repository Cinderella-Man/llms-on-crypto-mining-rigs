# Q591: the k-curve on the P106s, T(k) of a whole-model k-token pass (2026-11-04)

What a verify pass of k tokens costs on nine P106-090s, the number that decides how long a draft pays
([adaptive draft length](../2026-11-04_adaptive_draft/README.md) reads it as `LLAMA_SPEC_ADAPT_COST`), and where the
Pascal k-column and multi-token kernels of Q591 win and lose. Fork branch `pascal-p106-v3-b12076-ladder3` `7bc1552f4`,
Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0 as published (BF16 hyper-connections) and its hcq8 retype (hyper-connections in
Q8_0, router in F16; [retype](../2026-11-03_retype_agent/README.md)). The whole-model method is S21's
([verify cost](../2026-11-03_verify_cost/README.md)): `llama-bench -p 1,2,...,8 -n 0`, five repetitions after a warm-up.
Why the curve is this steep, part by part: [k-pass profile](../2026-11-04_kpass_profile/README.md).

## T(k), ms per pass (median of 5; standard deviation 2–4 ms)

| k | Swift Q2_0, defaults | **Swift Q2_0, `KCOL_SMEM=32768`** | Q2_0, `KCOL_R=1` | Q2_0, `KCOL=0` | Q2_0, `MTOK=0` | Q2_0, both off | hcq8, defaults | hcq8, `R=1` + `SMEM=32768` | hcq8, both off | S21, old base |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82.4 | **74.9** | 74.8 | 78.1 | 83.0 | 82.5 | 98.8 | 78.7 | 92.1 | 136.0 |
| 2 | 122.1 | **107.7** | 111.3 | 113.1 | 129.3 | 125.6 | 149.8 | 117.7 | 130.9 | 185.8 |
| 3 | 149.4 | **138.5** | 141.6 | 148.0 | 160.9 | 167.1 | 177.5 | 148.6 | 170.6 | 244.6 |
| 4 | 173.0 | **162.8** | 166.5 | 186.2 | 190.1 | 210.8 | 204.8 | 177.5 | 216.4 | 305.4 |
| 5 | 217.1 | **205.3** | 216.7 | 225.6 | 233.3 | 250.7 | 243.8 | 214.6 | 249.6 | 359.9 |
| 6 | 249.1 | **234.7** | 246.6 | 269.3 | 271.9 | 303.4 | 277.6 | 244.8 | 301.7 | |
| 7 | 280.1 | **261.7** | 270.8 | 301.5 | 309.4 | 341.4 | 313.1 | 279.0 | 343.8 | |
| 8 | 319.1 | **297.3** | 304.9 | 347.6 | 344.5 | 383.3 | 353.5 | 316.4 | 387.9 | |
| T(5) / T(1) | 2.63 | 2.74 | 2.90 | 2.89 | 2.81 | 3.04 | 2.47 | 2.73 | 2.71 | 2.65 |

The switches (all `GGML_CUDA_MMVQ_PASCAL_…`): `KCOL=0` puts the dense products of 2–8 tokens back on the kernels before
Q591 (and the one-column Q3_K, IQ4_NL and short-row Q8_0 products on the one-token kernels); `MTOK=0` does the same for
the routed experts; `KCOL_R=1` forces one row per lane group in the k-column kernels; `KCOL_SMEM=32768` gives their
launch shape a 32 KB shared-memory budget instead of 48 KB (3 blocks per SM where registers allow). `R=1` together with
`SMEM=32768` measures the same as `SMEM=32768` alone (205.2 at k = 5).

- **The published Q2_0 with `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768` is the fastest at every k**: 74.9 ms at one token
  (13.4 tok/s before any draft), 7.5–21.8 ms below the defaults at k = 1–8, with no rebuild. The retype is slower at
  every k in every setting (78.7 at best): with the Pascal BF16 kernel (C54) the exact hyper-connections cost less than
  the retype's 320-wide Q8_0 rows.
- **The curve is far from flat**: each extra token adds 24–44 ms; T(5) / T(1) is 2.6–2.9 in every setting, against the
  ≤ 1.3 the kernels aimed at. The cheapest steps are 3 → 4 (+24 ms); 4 → 5 costs +42. A verify of width 4 (a draft of 3)
  is where a draft pays most: with a drafter step of 20 ms, est. ~16 tok/s at a per-position acceptance of 0.8 and ~19
  at 0.9.
- **The multi-token expert kernel pays at every k ≥ 2** (`MTOK=0` is 7–25 ms slower); **the dense k-column kernels
  pay from k = 4** at their default shape and from k = 1 with the 32 KB budget.

## The kernels, one P106 (card 0)

**Bench** (`bench/mmvq_pascal`, each call the quantizer and the product, weights cycling beyond the L2, every row
checked against a double-precision reference; every variant passed). The dense k-column kernels at Flash-Next's shapes
([kcol_dense.csv](kcol_dense.csv), C93's build with Q4_0/Q5_0/IQ4_NL; [kcol_iq.csv](kcol_iq.csv), the six IQ types):
[kcol_best.tsv](kcol_best.tsv) lists every case where another launch shape beats the default by ≥ 5 % and what it is
worth per token. The default loses to one row per lane group or the 32 KB budget at almost every shape: the Q8_0 rows
320 → 10,240 by 2.2× (162 → 73 µs, 97 calls a token), Q3_K and IQ4_XS attention by 1.3–1.9×, the Q5_K output layer by
1.4×; 19–24 ms a token over the bench's shapes (Q823). The BF16 kernel at 1–8 tokens and every launch shape:
[bf16.csv](bf16.csv) (8.7 ms a token of hyper-connection products at one token, 10.7 at four, 14.2 at five; the
default within 4–32 % of the best shape). The routed experts ([q2_0_mtok.csv](q2_0_mtok.csv), R7's kernel): with
llama-bench-like random routing a 5-token pass reads 48.3 distinct experts per layer of its 50 slots, so the experts'
cost grows ~4.4× from one token to five (6.3 → 27.8 ms over 48 layers) however they are read; the multi-token kernel is
bit-identical to the one-token kernel and 0–9 % below it. The one-token Q2_0 shapes ([q2_0_onetoken.csv](q2_0_onetoken.csv)):
the shared expert's fused gate/up takes one row (7.78 µs against 11.33 at two).

**Inside ggml** (test-backend-ops perf, the quantizer and the product as the model calls them, the new kernels off and
on on one binary):

| File | What | The k-column / multi-token kernels win | They lose |
|---|---|---|---|
| [perf_dense.tsv](perf_dense.tsv) | Flash-Next's dense products, 1–8 tokens; BF16 | Q3_K at 3–8 tokens (1.1–1.6×; 3.4× at 2 tokens for 12,288 rows), Q2_0 at 5–8 (2.8–4.1×), IQ4_XS at 6–8 (1.1–1.3×), the Q5_K output layer at 5–8 (to 1.7×); the BF16 kernel at every shape and token count (1.4–6.3×) | Q3_K at one token (0.75–0.78×), IQ4_XS at 2–5 (0.72–0.93), Q4_K at 2–6 (0.68–0.93), the Q5_K and Q6_K output layers at 2–4 (0.55–0.74), the 320-row Q8_0 at 2–8 (0.48–0.80) |
| [perf_cov.tsv](perf_cov.tsv) | the shared expert's down types Q4_0, Q5_0, IQ4_NL (2,560 × 640) | at 5–8 tokens (1.1–1.5×), IQ4_NL from 3 (1.2–1.8×) | Q4_0 and Q5_0 at 2–4 (0.54–0.84), IQ4_NL at one token (0.79) |
| [perf_experts.tsv](perf_experts.tsv) | the routed Q2_0 experts (`MUL_MAT_ID`, 10 of 128, random routing and two overlap profiles) | gate/up at 2–8 tokens (1.4–2.4×) | the down projection is identical off and on (0.99–1.02 at every k): the multi-token kernel never takes it (Q825) |
| [perf_iq.md](perf_iq.md), [perf_iq.csv](perf_iq.csv) | the six IQ types' routed and dense products, and per Swift IQ file the timed products per k-token pass | mostly from 4–6 tokens on (to 2.5× for IQ1_M) | at 2–3 tokens for several types; the fused routed gate/up of a 5-token pass costs 6.8× its one-token value in the IQ3_S file's mix |

Correctness on the card in the same runs: C93's Q2_0 fusion cases 157/157 and 157/157 (`Q2_0=2`), every bench variant
of all four benches against its CPU reference, the multi-token expert kernel bit for bit against the one-token kernel.

## Files

| File | What it is |
|---|---|
| [tk_wall.tsv](tk_wall.tsv) | per arm and k: mean and median ms, standard deviation, ms per token, T(k) / T(1), the step from k − 1 (`hcq8_prof`: the same passes under `GGML_CUDA_PROFILE`) |
| [cost/](cost/) | `LLAMA_SPEC_ADAPT_COST` files, one per arm: one T(k) in ms per line, k = 1–8 (past the end the server takes the last slope); [t_k_q2_0_s32k.txt](cost/t_k_q2_0_s32k.txt) for the fastest line |
| [kcol_best.tsv](kcol_best.tsv), [kcol_dense.csv](kcol_dense.csv), [kcol_iq.csv](kcol_iq.csv), [bf16.csv](bf16.csv), [q2_0_mtok.csv](q2_0_mtok.csv), [q2_0_onetoken.csv](q2_0_onetoken.csv) | the benches (above) |
| [perf_dense.tsv](perf_dense.tsv), [perf_cov.tsv](perf_cov.tsv), [perf_experts.tsv](perf_experts.tsv), [perf_iq.md](perf_iq.md), [perf_iq.csv](perf_iq.csv) | inside ggml, off against on (columns: µs off, µs on, speed-up, each arm against its own one token) |
| [run/tk.py](run/tk.py) | the T(k) table and the cost files from llama-bench's JSON |
| [run/kcol_best.py](run/kcol_best.py) | the per-shape launch-shape comparison from `mmvq_kcol`'s CSV |

The cost files hold llama-bench's passes, which ask for the last token's logits only; a verify needs all k rows of the
output layer (`--verify-extra` in `tk.py` adds it per k; the Q5_K output layer at k rows costs ~3.1 ms at one, 10.6 at
two and 12.0 at five inside ggml, so with the defaults a verify of k ≥ 2 is ~7–9 ms above the pass).

## Run

On the rig, idle; the benches and scripts as their own folders give them
([kcol dense](../2026-11-04_kcol_dense/README.md) `run/rig_kcol.sh` with `PARTS="bench bf16 perf"`,
[kcol coverage](../2026-11-04_kcol_coverage/README.md) `run/rig_kcol_cov.sh` with `PARTS="q2_0 perf"`,
[multi-token experts](../2026-11-04_mtok_experts/README.md) `run/p106_checks.sh` with `STEPS="bench perf"` and its
perf arms `mtok` and `off`, [IQ k-column](../2026-11-04_iq_kcol/README.md) `bench/mmvq_pascal/iq_kcol_perf.sh` and
`mmvq_kcol --types iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq1_m --tokens 1,2,3,4,5,6,7,8`), with `BIN` the `-ladder3`
build's `bin/`. The whole-model arms, from a folder holding S21's `verify_cost.sh`, `mem_guard.sh` and
`smoke_request.json` ([verify cost run/](../2026-11-03_verify_cost/run/), [MTP run/](../2026-11-03_flash_next_mtp/run/)):

```bash
BIN=<-ladder3 build>/bin MODEL=<first shard> DRAFT=<any gguf; unused by these steps> OUT=<dir> STEPS=wall KS=1,2,3,4,5,6,7,8 \
  setsid nohup bash run/verify_cost.sh > verify_cost.out 2>&1 < /dev/null &
# an arm: the same with its switches in the environment, e.g. GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768
python3 -I run/tk.py --cost cost q2_0=<OUT>/bench-wall.json ...
python3 -I run/kcol_best.py kcol.csv 5 > kcol_best.tsv
python3 -I ../2026-11-04_kcol_dense/run/perf_table.py <rig_kcol.sh's OUT> > perf_dense.tsv
```

`verify_cost.sh` runs `llama-bench -m MODEL -ngl 99 -fa on -b 2048 -ub 512 -ts 6/6/6/6/5/6/5/6/3 -lm dio --no-host 1 -p KS -n 0
-r 5 -o json` under the host-memory guard after a logged quiet check (no llama process, the cards empty). Each arm took
~5 minutes including the load; the benches and perf arms ~3 hours in all, most of it the CPU's preparation of
test-backend-ops' weights.
