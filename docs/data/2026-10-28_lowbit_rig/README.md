# N2P/N3P on the rig: kernels on a P106 (Q188) and the agent check (Q189), 2026-10-28

The rig tests of Miner Llama's 2- and 3-bit formats ([pascal_lowbit_format.md](../../research/notes/pascal_lowbit_format.md)),
registry rows Q188 and Q189 in [quants/experiments.md](../../../quants/experiments.md):
- **Q188:** correctness and kernel time of fork branch `pascal-p106-v3-lowbit` (build `d7a75deb3`) on one P106-090,
  at Ornith-1.5-122B's shapes, against the formats it competes with.
- **Q189:** the agent check of Q212's file ([its data](../2026-10-28_ornith_valid1/README.md)) on nine cards, against
  UD-Q2_K_XL on the tuned IQ2_S/IQ3_XXS kernels ([ud-v3](../2026-10-27_ornith/agent/ud-v3/compare.md)).

## Q188: kernels on one P106

**Correctness** (`test-backend-ops` against the rig's CPU, card 0): MUL_MAT 30/30, MUL_MAT_ID 18/18 and GET_ROWS 16/16
for N2P and N3P. These are the cases small enough for the rig's 2-core CPU to quantize, with MMQ from 9 tokens up. The
fused gate/up has no case that small; it passed on a V100, natively and as compute 6.1 code
([correctness-v100.md](../2026-10-27_lowbit/correctness-v100.md)).

**Time per weight at one token**, in ps, Ornith's shapes. Expert products use 8 of 64 experts, ≥ 25 M weights per call
(beyond the card's 1.5 MB L2); weights are random bytes. The fused gate/up was timed under GGML_CUDA_PROFILE, as in the
IQ kernels' round 2.

| Type, kernel | fused gate/up 3072 → 1024 | expert gate alone | expert down 1024 → 3072 | dense 3072 → 8192 |
|---|---|---|---|---|
| N2P, Pascal | 5.53 | 4.18 | 4.66 | 4.16 |
| N3P, Pascal | 5.09 | 3.80 | 4.27 | 3.81 |
| Q4_0, Pascal | 3.97 | 3.85 | 5.07 | 3.87 |
| Q8_0, Pascal | 6.67 | 6.92 | 7.46 | 6.96 |
| N2P, llama.cpp's kernel | 10.80 | 5.47 | 10.49 | 5.47 |
| N3P, llama.cpp's kernel | 10.70 | 5.21 | 10.38 | 5.17 |
| IQ2_S, llama.cpp's kernel | 9.97 | 11.12 | 17.41 | 11.03 |
| IQ3_XXS, llama.cpp's kernel | 11.07 | 13.11 | 18.67 | 12.99 |
| IQ2_S / IQ3_XXS, the tuned Pascal kernels ([Q166](../2026-10-27_iq_kernels/README.md)) | 6.67 / 7.28 | — | 9.19 / 7.07 | 7.16 / 6.38 |

- **N2P and N3P run at Q4_0's speed per weight when unfused.** The estimates were 3.7 and 4.0 ps: below ~3.5 bits the
  P106 is issue-bound, so fewer bits do not mean less time. The measured 3.8–4.7 ps confirm it.
- **Against the tuned IQ kernels:**
  - the down product is 1.5–1.7× faster (N2P 4.66 against IQ3_XXS's 7.07, N3P 4.27);
  - the fused gate/up only 1.2–1.4× (N2P 5.53 against IQ2_S's 6.67, N3P 5.09 against IQ3_XXS's 7.28).
- **The fused gate/up is the gap.** It takes 278.5 µs for N2P against 2 × 105.2 µs for the two products alone (N3P
  256.0 against 2 × 95.5), so the fusion costs ~1.3× instead of saving. The IQ kernels had the same problem until both
  matrices were staged one after the other (fork commit `baf2e8ded`). At the unfused speed, gate/up would be 1.6–1.9×
  faster than the tuned IQ kernels.
- **At 2 and 4 tokens llama.cpp's kernel is faster than the Pascal kernels on N2P/N3P.** For example, N2P's down at 2
  tokens takes 660 µs on the Pascal kernel against 291 µs on llama.cpp's. Every call reads up to 8 experts per token,
  so 2 and 4 tokens read more weights and have no per-weight column. One-token decode is unaffected; MTP verification
  needs the n ≥ 2 cases handed to llama.cpp's kernel, or a Pascal kernel for them.
- The cross-check holds: llama.cpp's kernel times IQ2_S and IQ3_XXS exactly as the IQ kernels' rig runs did.

All 174 rows (every type, product, token count and kernel) are in `kernels.tsv`. The dense 3072 → 1024 and 1024 → 3072
cases read only 3 M weights per call, so launch time dominates them; they are kept for completeness.

## Q189: the agent check

Q212's file (Ornith-1.5-122B, N2P/N3P experts allocated by layer, Q5_0 output layer; KLD better than UD-Q2_K_XL's
on all three texts) on nine cards. The server is the low-bit build `d7a75deb3`, against ud-v3: UD-Q2_K_XL on v2.0 with
the tuned IQ2_S/IQ3_XXS kernels (`ea55c7865`). Both are one run each of the agent check
([bench/agent_check](../../../bench/agent_check/README.md): the pi agent on 12 Exercism bug fixes, one attempt each,
12-minute cap, 35-minute budget), with the same settings:
- the makers' sampling (T 0.6, top-p 0.95, top-k 20);
- a 49,152-token context;
- `-b 2048 -ub 512 -fit off`.

The card split differs. UD's split, 8,6,5,5,5,4,5,6,6, would put 7,323 MiB on card 0 of a 6,068 MiB card, because
valid1's layers 24–47 are larger (1.00–1.05 GB against 0.88). The fit projection gave 6,6,6,6,5,5,5,5,6 instead: 4,972–5,602 MiB
per card, at least 397 MiB free on each.

| | ud-v3 (UD-Q2_K_XL, IQ kernels) | valid1 (Q212, N2P/N3P) |
|---|---|---|
| tasks passed | 12 / 12 | 12 / 12 |
| minutes, all tasks | 36.1 | **32.5** |
| minutes per task (median, p90) | 2.5, 4.9 | 2.4, 4.7 |
| generation tok/s | 11.3 | **14.5** |
| ms per generated token | 88.6 | **69.0** |
| prefill / decode s per task (mean) | 44 / 135 | 56 / 104 |
| first token per turn s (median, p90) | 6.60, 17.14 | 9.57, 20.29 |
| first token, turns 1 and 2 s (789 and 1,452 new prompt tokens in both) | 14.26, 16.27 | 16.49, 18.81 |
| new prompt tokens, turns 3 and 4 (median) | 32, 34 | 266, 216 |
| prompt tokens processed per task | 2,818 | 3,408 |
| cache hit | 86.4 % | 84.2 % |
| tool calls / failed / malformed | 59 / 1 / 0 | 58 / 0 / 0 |
| lowest host MemAvailable, swap used | 490 MB, 345 → 649 MB | 563 MB, 350 → 651 MB |

- **Decode is 1.28× faster per generated token** (69.0 against 88.6 ms), more than the kernel times above suggested
  (expert products about level with the tuned IQ kernels' 17.3 ms per token, as fused). My reading: UD's mix also has
  IQ1_S, IQ2_XXS, Q2_K, Q3_K and IQ3_S experts on llama.cpp's slower kernels, and K-quant dense tensors. Valid1 has
  two expert formats and Q5_0 dense tensors, all on Pascal kernels. A profile of both would show it.
- **The whole run is 10 % shorter** (32.5 against 36.1 minutes). Minutes per task are level, because the answers
  differ in length in a single run at T 0.6; a paired replay of the same traces would separate per-token speed from
  answer length.
- **The first token is slower:**
  - On the turns with the same new prompt tokens in both runs (turns 1 and 2), prompt processing takes ~16 % longer.
    That is the MMQ path, whose N2P/N3P tile configurations are copied from IQ4_XS and untuned.
  - In turns 3 and 4 the server re-read 216–266 new prompt tokens where ud-v3's re-read ~33, so the median first token
    rose more (9.57 against 6.60 s).
- **Memory** is like every Ornith line so far: below the harness's 600 MB gate with ~300 MB of swap growth, no
  watchdog kill (`gonogo.md`).

The tables are in [agent/valid1](../2026-10-27_ornith/agent/valid1/compare.md) next to ud-v3's: `compare.md`,
`table.md`, `gonogo.md`, `attempts.csv`, `turns.csv` and `validation.txt`.

## Files

| File | What |
|---|---|
| `kernels.tsv` | Q188: per type, product, token count and kernel, microseconds per call and, at one token, picoseconds per weight |
| `run/table.sh` | `kernels.tsv` from the output of [rig_kernels.sh](../2026-10-27_lowbit/run/rig_kernels.sh) |
| [../2026-10-27_ornith/agent/valid1/](../2026-10-27_ornith/agent/valid1/compare.md) | Q189's agent-check tables, in the layout of the other Ornith lines |

## How it was made

```bash
# Q188: install, correctness, kernel timing on card 0 (nothing else on the rig)
RIG=rig QM=<build machine> WORK=<its work folder> NAME=llama.cpp-20261027-d7a75deb3-pascal-p106-v3-lowbit \
  MD5=b87fddc30c0da6613c9c42ff0b1b9b48 BUILDS=/opt/llama-builds bash ../2026-10-27_lowbit/run/rig_kernels.sh > rig_kernels.txt
bash run/table.sh < rig_kernels.txt > kernels.tsv
# Q189: the split, then the agent check (the client machine runs the agent; see agent_ornith.sh for its variables)
RIG=rig MODEL=<rig path of the file> BUILDS=/opt/llama-builds OUT=fit_probe.md \
  CONFIGS='ts|llama.cpp-20261027-d7a75deb3-pascal-p106-v3-lowbit|49152|-b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6' bash scripts/fit_probe.sh
LOWBIT=/opt/llama-builds/llama.cpp-20261027-d7a75deb3-pascal-p106-v3-lowbit/bin/llama-server VALID1=<rig path of the file> \
  VALID1_LINE="-b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off" CANDS=valid1 <the harness's variables> \
  bash ../2026-10-27_ornith/run/agent_ornith.sh
```

- **Build:** fork branch `pascal-p106-v3-lowbit` at `d7a75deb3` ([run/build.sh](../2026-10-27_lowbit/run/build.sh)
  `KIND=cuda`): static, compute 6.1 and sm_70, no AVX. Same kernels as `5fd8ffeec`, plus `GGML_TEST_RAW_QUANT`.
- **Card:** P106-090 card 0 of the rig, with no other process on the rig.
