# 2026-11-03 kernel timing: the Pascal Q2_0 and BF16 kernels on one P106 (Q223, Q546)

The two Pascal kernels written for Qwen3.8-Flash-Next on 2026-11-03,
[the Q2_0 expert kernel](../2026-11-03_q2_0_kernel/README.md) (Q223, branch `pascal-p106-v3-q2_0mv` `7a9fb7cff`) and
[the BF16 short-row kernel](../2026-11-03_bf16_kernel/README.md) (Q546, branch `pascal-p106-v3-bf16mv` `a4acc80a3`),
timed against llama.cpp's own kernels at Flash-Next's shapes on one P106-090 (card 0 of the rig, every other card idle),
then checked and timed inside ggml with each branch's `test-backend-ops`. Result page: the registry rows Q223 and Q546
in [quants/experiments.md](../../../quants/experiments.md).

## Result

| | llama.cpp's kernel | Pascal kernel | Speed-up | Gate (handoff) |
|---|---:|---:|---:|---|
| Q2_0 experts, ms per token (4 shapes × 48 layers, one token) | 25.23 | **6.96** (best variant 6.66) | 3.62× | ≥ ~50 GB/s on the routed shapes: 132 and 82 GB/s |
| Q2_0 routed gate/up, ps per weight | 7.12 | **2.13** | 3.34× | estimate 3–6 ps |
| Q2_0 routed down, ps per weight | 14.80 | **3.45** | 4.29× | estimate 3–6 ps |
| BF16 tensors, ms per decoding step, 1 token | 25.46 | **8.93** (best 8.64) | 2.85× | ≥ ~100 GB/s on `hc_up`, `hc_down`: 159 and 157 GB/s |
| BF16 tensors, 2 / 3 / 4 tokens | 35.05 / 54.85 / 71.07 | **8.89 / 12.00 / 14.17** | 3.9× / 4.6× / 5.0× | |
| BF16 `hc_up` (320 → 10,240), ps per weight | 60.17 | **12.59** | 4.78× | |
| Correctness on the P106 (test-backend-ops) | | Q2_0 339/339, BF16 265/265 | | must pass all |

- **Both kernels pass their gates by a wide margin.** The Q2_0 kernel reads the routed experts at 2.1–3.5 ps per weight,
  better than the 3–6 ps estimated; the BF16 kernel runs the hyper-connection projections at 157–164 GB/s, 89–92 % of the
  card's 179 GB/s streaming read, at one and two tokens.
- **Every variant of both benches matches the double-precision reference on the P106**, and the branches' own
  test-backend-ops cases pass on the card's native compute 6.1 code (they were checked only through the PTX JIT on an
  RTX 3090 before).
- **At 2–4 tokens the Pascal Q2_0 gate/up product is slower than llama.cpp's inside ggml:** the unfused
  `MUL_MAT_ID` case (10 of 128 experts, 640 × 2,560) takes 175 / 293 / 428 µs against 141 / 209 / 280 µs (0.81× /
  0.71× / 0.65×), while the down product stays faster (1.41× / 1.31× / 1.21×) and so do the dense products. One token
  is 1.35× faster. A verify step of a speculative decoder runs 2–4 tokens, so this decides how the speed-ladder build
  dispatches gate/up at those batch sizes (Q610). The bench times the fused gate/up only at one token.
- **The injects (10,240 × 4) are the one BF16 shape the kernel loses at 1–2 tokens** (4.05 against 3.68 µs, 0.04 ms
  per token); fusing them into the down projection (Q557) is worth at most the 0.35–0.39 ms per token they take.
- **Launch shapes:** the fork's defaults are within 5 % of the best variant on every large shape at one token. The
  measurable misses: the shared expert's gate/up (R = 1 takes 7.78 against 10.98 µs, −0.15 ms per token; Q612) and, at
  3–4 tokens, the BF16 down projection and recurrent alpha/beta (G = 32, R = 2 and G = 128, R = 2: −0.96 ms per 3-token
  step and −1.31 ms per 4-token step; Q611).

## In the per-token budget

GPU time per decoded token, summed over the nine cards, from the
[R13 profile](../2026-11-03_flash_next_baseline/README.md) of the unmodified Swift Q2_0 file (136 ms) and the
[R28 profile](../2026-11-03_retype_agent/README.md) of Q218's retyped file (120.7 ms, hyper-connections in Q8_0):

| Tensors | In the model now (profile) | llama.cpp's kernel in the bench | Pascal kernel in the bench | Change in the bench |
|---|---:|---:|---:|---:|
| Q2_0 experts (routed gate/up and down, shared gate/up and down) | ~28 (R13; gate/up 12.86 + down 10.85 + the shared expert); the same in R28's file | 25.23 | 6.96 | **−18.3** |
| BF16, unmodified file: `hc_*_up`, `hc_*_down`, injects, indexer, alpha/beta | 43 (R13) | 25.46 | 8.93 | **−16.5** |
| The same projections in R28's retyped file (`hc_*` in Q8_0, llama.cpp's Q8_0 kernel) | 24.5 (18.61 + 4.00 + 1.90) | | | |

- The bench's llama.cpp times sit close to the profile for the Q2_0 experts (25.2 against ~26–28 ms) but well below
  it for the BF16 tensors (25.5 against 43 ms: `hc_*_up` takes ~330 µs a call in the model and 197 µs in the bench,
  the in-model excess of Q99). How much of that excess the Pascal kernel keeps shows only in the model's profile.
- **Estimate for the speed ladder:** with both kernels the unmodified, exact file goes from 136 to roughly 84–90 ms
  of GPU time per token (−18 for the experts; 43 → ~9–15 ms for the BF16 tensors, the bench's 8.9 ms plus up to ~6 ms
  of in-model excess). R28's retyped file with the Q2_0 kernel would be ~102 ms, since its Q8_0 hyper-connections stay
  on llama.cpp's kernel at 24.5 ms. So the BF16 kernel on the exact file should beat the retype by ~10–15 ms per token
  without its KLD cost; a short-row Q8_0 kernel (Q604) is what would let the retyped file catch up. These are estimates
  from one card's bench; the agent check of the ladder build (R29) measures the real line.

## Files

| File | What |
|---|---|
| [tables.md](tables.md) | every table: per shape and variant, µs per call, ps per weight, GB/s, ms per token; both perf comparisons |
| [q2_0.csv](q2_0.csv), [q2_0_rig.txt](q2_0_rig.txt) | `bench/mmvq_pascal --q2_0` on the P106: CSV and console (24 variant × shape cells, all PASS) |
| [bf16.csv](bf16.csv), [bf16_rig.txt](bf16_rig.txt) | `bench/mmvq_pascal --bf16 --tokens 1,2,3,4`: CSV and console (190 cells, all PASS) |
| [perf_q2_0_stock.txt](perf_q2_0_stock.txt), [perf_q2_0_pascal.txt](perf_q2_0_pascal.txt) | `test-backend-ops perf -o MUL_MAT,MUL_MAT_ID -p type_a=q2_0`, llama.cpp's kernel (`GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0`) and the Pascal one (default), same binary |
| [perf_bf16_mmvf0.txt](perf_bf16_mmvf0.txt), [perf_bf16_mmvf1.txt](perf_bf16_mmvf1.txt) | `test-backend-ops perf -o MUL_MAT -p type_a=bf16`, `GGML_CUDA_MMVF_PASCAL=0` and `1`, same binary |
| [run/summarize.py](run/summarize.py) | builds `tables.md` from the CSVs and perf logs: `python3 run/summarize.py <folder>` |

The correctness logs (one line per case, all OK) are summarized above and not published.

## Commands

Binaries: the bench from this repository's `bench/mmvq_pascal` (`bench/mmvq_pascal/build.sh`, sm_61) at `v3`
`02e8d94` for `--q2_0` (md5 `8d4dc0193c15c4a94fbf1155e8665375`) and at `e077d34` for `--bf16` (md5
`8b04266bd697dd2936f8d9f1c79426c0`); the engines are the rig builds of the two fork branches
(`scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-q2_0mv`, `fork/pascal-p106-v3-bf16mv`, `TESTS=1`).
Everything ran from the rig's `/tmp`, card 0, nothing else on the rig, and was removed afterwards.

```bash
RIG=rig; G='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0'
Q=llama.cpp-20261103-7a9fb7cff-pascal-p106-v3-q2_0mv; F=llama.cpp-20261103-a4acc80a3-pascal-p106-v3-bf16mv
# copied to the rig: /tmp/mmvq_pascal_q2, /tmp/mmvq_pascal_bf16, and both tarballs unpacked under /tmp/r16
ssh $RIG "$G /tmp/mmvq_pascal_q2 --q2_0 --csv /tmp/q2_0.csv" > q2_0_rig.txt
ssh $RIG "$G /tmp/mmvq_pascal_bf16 --bf16 --tokens 1,2,3,4 --csv /tmp/bf16.csv" > bf16_rig.txt
ssh $RIG "cd /tmp/r16 && $G ./$Q/bin/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED -p 'type_a=q2_0|type=q2_0'"  # 339/339
ssh $RIG "cd /tmp/r16 && $G ./$F/bin/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_VEC_FUSION -p 'type_a=bf16|type=bf16'"  # 265/265
for t in stock:GGML_CUDA_MMVQ_PASCAL_SKIP=q2_0 pascal:GGML_CUDA_MMVQ_PASCAL=1; do
  ssh $RIG "cd /tmp/r16 && $G ${t#*:} ./$Q/bin/test-backend-ops perf -b CUDA0 -o MUL_MAT,MUL_MAT_ID -p type_a=q2_0" > perf_q2_0_${t%%:*}.txt
done
for m in 0 1; do
  ssh $RIG "cd /tmp/r16 && $G GGML_CUDA_MMVF_PASCAL=$m ./$F/bin/test-backend-ops perf -b CUDA0 -o MUL_MAT -p type_a=bf16" > perf_bf16_mmvf$m.txt
done
ssh $RIG 'rm -rf /tmp/r16 /tmp/mmvq_pascal_q2 /tmp/mmvq_pascal_bf16 /tmp/q2_0.csv /tmp/bf16.csv'
python3 run/summarize.py . > tables.md   # on any machine but the rig; ANSI colours stripped from the perf logs first
```

Ps per weight is µs per call × 10⁶ over the weights a call reads (Q2_0: 18 bytes per 64 weights; BF16: 2 bytes);
ms per token is calls per token × µs per call, with the bench's calls per token (48 per Q2_0 shape; 97, 97, 96, 12 and
72 for the BF16 shapes). The bench's shapes and method are in
[bench/mmvq_pascal/README.md](../../../bench/mmvq_pascal/README.md) (Q2_0: 512 experts, 64 random sets of 10 replayed,
no call in L2; BF16: copies cycled past L2). test-backend-ops' `MUL_MAT_ID` perf cases use 128 experts and the unfused
product, so their absolute times differ from the bench's.

Provenance and licence: measurements and scripts of this repository (its licence); no third-party data.
