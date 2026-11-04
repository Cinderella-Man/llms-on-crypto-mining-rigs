# A trellis code for Pascal: the CPU proxy (2026-11-04)

Registry row Q929 in [quants/experiments.md](../../../quants/experiments.md), step 1. agentionai's Gyro-M stores
Qwen3.8-Flash-Next's routed experts in TQ2_T, a trellis code of the QTIP family at 2.125 bits per weight, and holds
about half the KL divergence of Swift-1.5's Q2_0 at the same size (agent text 0.148 against 0.285,
[gyro KLD](../2026-11-04_gyro_kld/README.md)). On a P106 its decode costs 10–38 ps per weight against Miner Llama's
Q2_0 kernel's 3.3 ([gyro on a P106](../2026-11-04_gyro_p106/README.md)): a 32-bit multiply per 4 weights, which Pascal
splits into three 16-bit multiply-adds, and two random reads from a codebook. This folder asks, before anyone writes an
encoder or a kernel, which trellis code built from the operations a Pascal GPU does cheaply keeps the most of TQ2_T's
quality per bit, measured on Swift-1.5's own expert weights, and what each one would cost to decode on a P106.

## Result

**Yes: HP4S, a trellis code decoded entirely in registers, holds TQ2_T's error within 4 % at the same 2.125 bits,
with an estimated decode of 4.6 ps per weight on a P106** (TQ2_T: 10.0 ps measured, 11.2 estimated; Miner Llama's
Q2_0: 3.3 measured). HP4S hashes the 16-bit trellis state with one 16 × 16-bit multiply and uses four 3-bit fields of
the hash to pick each of 4 weights from one of two 8-level int8 tables held in registers, with one byte permute. It needs
no shared memory and no 32-bit multiply. Its error is 24 % below Gyro-M's own TQ2_T bytes and 54 % below Swift-1.5's
own Q2_0, at fewer bits than Q2_0 (2.125 against 2.25). Step 2 times it on a P106 (below).

Importance-weighted error relative to Swift-1.5's own Q2_0 on the same 84 matrices (Q2_0 = 1.0; geometric mean of the
paired ratios), and the estimated one-token decode cost (`run/decode_cost.py`; "measured" where a kernel exists).
Every format is in `summary.md`.

| Code (2.125–2.25 bits) | bpw | gate/up | down | all | vs TQ2_T | ps a weight, P106 |
|---|---:|---:|---:|---:|---:|---:|
| Swift-1.5's own Q2_0 (the reference) | 2.25 | 1.000 | 1.000 | 1.000 | 2.26 | 3.3 measured |
| TQ2_T, this folder's Viterbi encoder | 2.125 | 0.472 | 0.387 | **0.442** | 1.000 | 10.0 measured |
| TQ2_T, the same with the 128-wide rotation (the Gyro files' way) | 2.125 | 0.532 | 0.504 | 0.523 | 1.18 | 10.0 + the rotation |
| TQ2_T, Gyro-M's own bytes (the original checkpoint) | 2.125 | 0.624 | 0.557 | 0.601 | 1.36 | 10.0 measured |
| **HP4S** `hp4s.n8.i8`: two 8-level tables | 2.125 | 0.490 | 0.404 | **0.459** | **1.039** | **4.6** |
| `hp4.n16.i8`: 16 levels, 4-bit fields | 2.125 | 0.490 | 0.401 | 0.458 | 1.036 | 4.8 |
| `hp4.n8.i8`: 8 levels | 2.125 | 0.551 | 0.459 | 0.519 | 1.173 | 4.0 |
| `hp4m.n8.i8`: 8 levels, Gyro's 32-bit multiply | 2.125 | 0.549 | 0.457 | 0.517 | 1.168 | 4.4 |
| `lut2.L12.i8`: 4,096 2-D entries in shared memory | 2.125 | 0.473 | 0.387 | 0.442 | 1.000 | 10.0 |
| `hash4.n2048.i8`: 2,048 4-D entries in shared memory | 2.125 | 0.474 | 0.390 | 0.444 | 1.004 | 6.8 |
| `lut4.L11.i8`: 2,048 4-D entries, indexed by the state | 2.125 | 0.529 | 0.435 | 0.495 | 1.120 | 6.4 |
| `lut4.L10.i8`: 1,024 4-D entries | 2.125 | 0.561 | 0.462 | 0.526 | 1.189 | 6.4 |
| `tcq16.L12.i8`: TCQ, 16 levels | 2.125 | 0.590 | 0.486 | 0.553 | 1.252 | 5.7 |
| `tcq8.L12.i8`: TCQ, 8 levels | 2.125 | 0.614 | 0.514 | 0.579 | 1.309 | 4.8 |
| `tcq8u.L12`: TCQ, 8 uniform levels | 2.125 | 0.715 | 0.592 | 0.671 | 1.518 | 4.2 |
| `hash4.n128.i8`: 128 entries, a copy per two lanes | 2.125 | 0.928 | 0.769 | 0.872 | 1.972 | 5.3 |
| `hash4.n64.i8`: 64 entries, a copy per lane | 2.125 | 1.303 | 1.082 | 1.225 | 2.771 | 4.8 |
| llama.cpp's own Q2_0 quantizer, FN4's importance matrix | 2.25 | 3.697 | 3.455 | 3.615 | 8.18 | 3.3 measured |

| Code (2.5–3.3 bits) | bpw | gate/up | down | all | ps a weight, P106 |
|---|---:|---:|---:|---:|---:|
| IQ2_S, llama.cpp's quantizer | 2.5625 | 0.421 | — | — | 6.67 measured (Q164's shared-memory kernel) |
| `hp4.n8.k25.i8` | 2.625 | 0.333 | 0.284 | 0.316 | 4.2 |
| `hp4.n16.k25.i8` | 2.625 | 0.265 | 0.218 | **0.249** | 4.8 |
| `hash4.n2048.k25.i8` | 2.625 | 0.279 | 0.231 | 0.262 | 7.0 |
| `lut2.L12.k25.i8` | 2.625 | 0.245 | 0.201 | 0.229 | 10.0 |
| N2P, Miner Llama's quantizer | 2.6875 | 0.351 | — | — | 4.18 measured (Ornith's shapes) |
| N2P-128 | 2.75 | 0.345 | 0.288 | 0.324 | — |
| IQ3_XXS, llama.cpp's quantizer | 3.0625 | 0.214 | — | — | 7.07 measured |
| N3P, Miner Llama's quantizer | 3.25 | 0.151 | — | — | 3.80 measured (Ornith's shapes) |
| N3P-128 | 3.3125 | 0.150 | 0.128 | 0.142 | — |
| Swift-1.5's own IQ3_S mix (IQ2_S to IQ3_S by layer) | mix | 0.261 | 0.093 | 0.185 | — |
| ISTA's IQ3_S mix of the original model, made the same way | mix | 0.292 | 0.089 | 0.196 | — |

The 256-weight types (IQ2_S, IQ3_XXS, N2P, N3P) do not fit the 640-wide down rows; their 128-weight variants do.

What the table says:
- **The shared-memory gather, not the multiply, is what makes a trellis code slow on Pascal.** Codebooks in shared
  memory reach TQ2_T's error (`lut2` exactly, `hash4` +0.4 %), but each random read per 4 weights costs ~3 ps a weight
  by itself, so they land at an est. 6.4–10 ps. The hash does not matter: one 16-bit XMAD scores like Gyro's 32-bit
  multiply (0.519 against 0.517).
- **Quality comes from the trellis memory and the levels each weight can take.** The codes that keep 8 bits of the
  previous steps in the state (TQ2_T, `lut2`, `hash4`, HP4) are within 4 % of each other once a weight has 16 levels to
  take; with 8 levels a weight HP4 loses 17 %. `lut4` keeps 3 or 2 bits and loses 12–19 %; TCQ's scalar levels lose
  25–52 % with 10 bits of memory. Small codebooks copied per lane, which read without conflicts, have too few codewords
  (2.0–2.8× TQ2_T's error).
- **The rotation hurts every trellis code here** (+10 to +18 %: TQ2_T 0.442 → 0.523, `hash4` 0.444 → 0.518, `hp4.n16`
  0.458 → 0.526, `hp4.n8` 0.519 → 0.576, `lut4.L11` 0.495 → 0.576, `tcq8` 0.579 → 0.636). With FN4's importance matrix
  the weights' importance is far from uniform across columns, and a rotation spreads it; Q304 found the same for scalar
  codes. A Pascal trellis code needs no run-time rotation, which also drops the Gyro files' four FWHT launches a layer.
- **int8 codebook entries cost nothing** (≤ 0.1 % against FP32 entries, TQ2_T's int8 copy included), so the `dp4a`
  path loses no quality.
- **The encoder matters as much as the format:** TQ2_T from this folder's imatrix-weighted Viterbi, without the
  rotation, scores 0.442 against Gyro-M's own bytes' 0.601. The checkpoint difference adds only ~2 % to Gyro's number:
  ISTA's and Swift's IQ3_S, made the same way from the two checkpoints, differ by 0.0016 in E, against Gyro-M's 0.086.
- At 2.625 bits HP4 with 16 levels (0.249) is below N2P-128 at 2.75 bits (0.324) and near IQ3_XXS at 3.06 on gate and
  up (0.265 against 0.214).
- The proxy ranks, it does not predict KL divergence: Gyro-M's bytes at 0.60 of Q2_0's error go with agent KL divergence
  0.148 against Q2_0's 0.285 (0.52).

## The metric and the weights

The metric is Q164's ([2026-10-27_lowbit](../2026-10-27_lowbit/README.md), `scripts/lowbit_proxy.py`, imported here):
the importance-weighted relative reconstruction error

    E = sum over rows and columns of imp_j * (w_ij - q_ij)^2  /  sum of imp_j * w_ij^2,

with imp_j the importance matrix's mean squared activation of input column j for that expert. Every quantizer weighs
column j of a row by imp_j * sqrt(sigma2 + w_ij^2) (sigma2 the row's mean square), as llama.cpp's do. E ranks formats
on the same weights; it is not a KL divergence. Tables give each format's E relative to Swift-1.5's own Q2_0 bytes on the
same matrices (geometric mean of the paired ratios), so **Q2_0 = 1.0**.

- **Weights:** `ukisai/Swift-Qwen3.8-Flash-Next` (Swift-1.5) BF16 safetensors, routed experts of layers 2, 10, 18, 26,
  34, 42 and 47 (Q164's depths of 48 layers), four experts each (the two the importance matrix routed to most and two
  random ones, seed 7), gate, up and down: 84 matrices. From each, 64 random rows of gate and up (2,560 wide) and 256 of
  down (640 wide): 163,840 weights, 1,280 blocks of 128, per matrix.
- **Importance matrix:** FN4's, `llama-imatrix` on Swift-1.5's Q8_0 over 200 chunks of 512 tokens of held-out agent
  sessions ([2026-11-04_expert_usage](../2026-11-04_expert_usage/README.md)).
- **Codebooks** are trained on two other layers (6 and 30, their two most-routed experts, 48 rows of gate and up and
  192 of down: 11,520 blocks), never on the matrices they are scored on.

## The codes

Every trellis code here stores 128-weight blocks with one FP16 scale (0.125 bits per weight) and a tail-biting
bitshift trellis: step t's state is the last L bits of the block's bit stream, which advances by kV bits per step of V
weights, and the state alone gives the step's V values through the code's codebook. Any bit string is a valid block,
so a decoder needs no search; the encoder is a Viterbi search (`run/ptq.c`, C, CPU) that minimizes the weighted error
above, with QTIP's two-pass tail-biting and a least-squares fit of the scale.

| Code | State, bits a step, weights a step | Codebook | Decode per 4 weights (one token) |
|---|---|---|---|
| `tq2t` | 16, 8, 4 | Gyro's TQ2_T itself: 32-bit multiply of the state, two 11-bit indices into agentionai's 2,048-point 2-D FP16 codebook; `tq2t.i8` their int8 copy (the CUDA kernels' table) | 32-bit multiply (3 XMAD), two random 2-byte shared-memory reads, dp4a |
| `hp4.n8`, `hp4.n16` | 16, 8, 4 | **computed in registers**: one 16 × 16-bit multiply (XMAD) of the state, folded (`p ^ p >> 16`), gives four 3-bit (4-bit) fields that pick each weight's int8 level from a trained table of 8 (16) levels held in two (four) registers with `PRMT` | ~6 (~10) ALU instructions, no shared memory |
| `hp4m.n8` | 16, 8, 4 | as `hp4.n8` with Gyro's 32-bit multiply | ~8 ALU |
| `hp4s.n8` | 16, 8, 4 | as `hp4.n8`, a hash bit switches between two trained 8-level tables | ~9 ALU |
| `hash4.n<N>` | 16, 8, 4 | the XMAD-hashed state indexes a trained N-entry 4-D int8 codebook in shared memory (N = 2,048: 8 KB; N = 128 and 64: small enough to copy per lane group, so the reads do not conflict) | ~5 ALU + one random 4-byte read |
| `lut4.L11`, `lut4.L10` | 11 or 10, 8, 4 | the state indexes a trained 2^L-entry 4-D int8 codebook directly (QTIP's lookup form; 8 or 4 KB) | ~3 ALU + one random 4-byte read |
| `lut2.L12` | 12, 4, 2 | the state indexes a trained 4,096-entry 2-D codebook (8 KB) | ~6 ALU + two random 2-byte reads |
| `tcq8.L12`, `tcq16.L12` | 12, 2, 1 | trellis-coded quantization: the 2 new bits pick one of 4 levels, a parity of older state bits picks which 4 of 8 trained levels (two parities: of 16); 4 interleaved chains of 32 weights a block; `tcq8u` uniform levels | ~7–14 ALU, no shared memory |
| `.k25` | as above, 10 bits a step for 4 weights, 5 for 2 | — | 2.625 bits per weight |

Suffixes: `.i8` int8 codebook entries with one global scale (what a dp4a kernel uses; without it, FP32 entries);
`+r128` a random-sign 128-wide Hadamard rotation of each row before the code, undone before the error is measured (the
Gyro files rotate every expert input at run time; Q304 found it hurt Q164's scalar codes).

Baselines: Swift-1.5's own Q2_0 bytes (the reference), llama.cpp's own Q2_0, IQ2_S and IQ3_XXS quantizers with FN4's
importance matrix, N2P and N3P (fork branch `pascal-p106-v3-lowbit`'s quantizer; gate and up only, since 640-wide down
rows do not fit their 256-weight super-blocks), N2P-128 and N3P-128 (Q164's numpy versions with 128-weight
super-blocks, Q220's variants for the down rows), and three files' own bytes read back: Gyro-M's TQ2_T (made from the
original Qwen3.8-Flash-Next, so its error includes the checkpoint difference), and the IQ3_S files of ISTA (original
model) and Swift-1.5, made the same way, which price that difference.

## The decode estimate

`run/decode_cost.py` counts, per 4 weights and one activation column, the issue slots (shifts, logic, byte permutes,
16-bit multiply-adds, dp4a, the per-block scale work) and the random shared-memory reads (32 lanes at random addresses
load the busiest of the 32 banks with ~3.5 lanes, so one warp read takes ~3.5 cycles of the SM's single port), and
estimates

    ps per weight = F + 0.217 * ALU slots + 3.04 * random reads (+ 0.87 per conflict-free read),

the two factors from the P106's 5 SMs × 128 lanes at ~1.8 GHz, F (activations, weight loads, the reduction and the
launch of a one-token product on Flash-Next's expert shapes) fitted to Miner Llama's measured Q2_0 kernel. The other
kernels measured on a P106 at one token check it:

| Kernel on a P106, one token | ALU slots per 4 weights | random reads per 4 weights | measured ps | model ps |
|---|---:|---:|---:|---:|
| Q2_0, Miner Llama's Pascal kernel (the fit) | 3.0 | 0 | 3.30 | 3.30 |
| N2P, Miner Llama's Pascal kernel | 4.75 | 0 | 4.18 | 3.68 (−12 %) |
| IQ2_S, Pascal kernel with its grid in shared memory | 7.0 | 1 | 6.67 | 7.21 (+8 %) |
| IQ3_XXS, Pascal kernel with its grid in shared memory | 6.25 | 1 | 7.07 | 7.04 (0 %) |
| TQ2_T, agentionai's kernel (one row per warp) | 11.5 | 2 | 10.0 | 11.22 (+12 %) |

The counts and every code's estimate are in `decode_cost.txt`.

So the model holds the measured kernels to within about ±15 %, and it says one thing plainly: a random shared-memory
read per 4 weights costs ~3 ps per weight by itself on this card, so every code that reads its codebook from shared
memory at random addresses lands at 6 ps or more, and only codes decoded in registers (or from conflict-free copies too
small to hold a useful codebook) come in under ~5 ps.

## Step 2: HP4S's decode on a P106

The one candidate is `hp4s.n8.i8` (HP4S below). Step 2 measures its decode on one P106 before anyone writes an encoder
for files or a llama.cpp type.

**The format.** A block is 128 weights in 34 bytes, TQ2_T's layout: an FP16 scale d and 32 bytes qs, one per step of 4
weights. Step t's 16-bit state is `s = qs[(t + 31) % 32] << 8 | qs[t]` (tail-biting; any byte string is valid). Then

    p = s * 0x9E37                      (16 x 16 -> 32 bits: one XMAD)
    h = p ^ (p >> 16)
    T = (h >> 15) & 1 ? T1 : T0         (two tables of 8 int8 levels, each in two registers)
    weight 4t + j = d * scale * T[(h >> 4j) & 7],  j = 0..3

with T0, T1 and the global `scale` from `codebooks.npz` (`hp4s.n8.i8`: 16 levels, the first 8 are T0). The decode of
step t against one activation column is one `PRMT` (the state), one `XMAD`, a shift and a `LOP3` (fold and mask the four
3-bit selectors into `PRMT`'s nibbles), a `BFE.S32` of bit 15 and two `LOP3` blends (the table pair), one `PRMT` lookup and
one `dp4a`; per block one FMA with d × the activations' scale. No shared memory, no 32-bit multiply.

**The bench** (`bench/mmvq_pascal`, a new type `hp4s` beside `q2_0`):
- the kernel on the frame of Miner Llama's Q2_0 Pascal kernel (`mmvq-pascal-impl.cuh`: activations held in registers,
  rows per lane group, 128-bit weight loads), with the lookup a template switch so that the same build also times
  `hp4.n8` (one 8-level table, est. 4.0 ps) and `hp4.n16` (4-bit fields, two lookups and a byte select, est. 4.8);
- Flash-Next's routed-expert shapes at 1 token (gate and up fused, 2,560 → 640, 10 of 512 experts; down 640 → 2,560),
  then 2, 4 and 8 tokens; Q2_0 in the same binary as the control;
- correctness: random bytes as blocks against a CPU decode of the same bytes (integer sums exact); the SASS checked for
  one `XMAD` per step, no `LDS` or local memory in the inner loop;
- built on the quant machine (`bench/mmvq_pascal/build.sh`, compute 6.1), checked through the 6.1 PTX on a newer card
  or by the CPU decode, timed on one P106 of an idle rig (~20 minutes).

**Pass line:** ≤ 5 ps per weight at one token on both shapes (est. 4.6; Q2_0 measures 3.3 on the same card). If it
passes, step 3 is the encoder (`run/ptq.c`'s Viterbi with FN4's importance matrix and the trained tables, as a
llama-quantize type in Miner Llama), the CUDA paths (one token, the 2–8-token MoE kernel, MMQ), a Swift-1.5 file and
its KL divergence on our three texts, then T(k) and the agent check on a rig.

## Files

| File | What |
|---|---|
| `proxy.tsv` | every format × 84 matrices: `layer`, `expert`, `kind`, `count` (FN4 routings of the expert), `format`, `bpw`, `err_imp` (E, the result), `err_plain` (unweighted), `seconds` |
| `summary.md` | `run/summary.py --md` over `proxy.tsv`: the table above with every format |
| `decode_cost.txt` | `run/decode_cost.py`: the anchors, the fit and every code's count and estimate |
| `codebooks.npz`, `codes.json` | the trained codebooks (a weight is d × entry; for `.i8` codes the int8 value is round(entry × 127 / max\|entry\|); `<code>.g0` the scale multiplier the search starts from) and the TCQ parity masks; Gyro's own TQ2_T codebook is not repeated here (agentionai's llama.cpp fork, `ggml/src/ggml-common.h`, Apache 2.0) |
| `run/ptq.c` | the Viterbi encoder (tail-biting, weighted, scale search), built as `libptq.so` |
| `run/ptq_proxy.py` | the proxy: codes, training, baselines, the files' own bytes, the evaluation |
| `run/summary.py`, `run/decode_cost.py` | the table and the decode estimates |
| `run/proxy.sh` | the run as it ran: the image, the library, the detached run, the summary |

## How it ran

On a separate CPU machine in Docker (`--cpu-shares 256 --cpus 8`, eight worker processes, Ubuntu 24.04's gcc 13 and
numpy 1.26), no GPU. `run/proxy.sh` takes the machine and paths from the environment (`QM`, `WORK`, `MODEL`, `IMATRIX`,
`LIB`, `LIB_N`, `GYRO_SRC`, `Q2_0`, `GYRO_M`, `IQ3S_ISTA`, `IQ3S_SWIFT`, `MOUNTS`):

```bash
run/proxy.sh image && run/proxy.sh lib
RUN=main ARGS="--codes $CODES" run/proxy.sh run     # CODES: the list below, one argument
RUN=main run/proxy.sh summary --md --ref2 tq2t --three-term tcq8.L12 > summary.md
run/decode_cost.py > decode_cost.txt
```

`CODES` = `tq2t,tq2t.i8,tq2t+r128,tq2t.i8+r128,hp4.n8,hp4.n8.i8,hp4.n8.i8+r128,hp4.n16,hp4.n16.i8,hp4.n16.i8+r128,
hp4m.n8.i8,hp4s.n8.i8,hash4.n2048,hash4.n2048.i8,hash4.n2048.i8+r128,hash4.n128.i8,hash4.n64.i8,lut4.L11,lut4.L11.i8,
lut4.L11.i8+r128,lut4.L10.i8,lut2.L12,lut2.L12.i8,lut2.L12.i8+r128,tcq8.L12,tcq8.L12.i8,tcq8.L12.i8+r128,tcq8u.L12,
tcq16.L12.i8,hp4.n8.k25.i8,hp4.n16.k25.i8,hash4.n2048.k25.i8,lut2.L12.k25.i8` (without the line breaks); every other
option is `ptq_proxy.py`'s default (the layers, experts and rows above, 6 training iterations, scale grid 0.9 and 1.1
around the trained start, one refit).

`LIB` is a CPU-only `libggml-base.so` of llama.cpp `d77256999` and `LIB_N` the same of fork branch
`pascal-p106-v3-lowbit` at `67181fcd9` (both built by [2026-10-27_lowbit](../2026-10-27_lowbit/README.md)'s
`run/proxy.sh lib`); `GYRO_SRC` is agentionai's llama.cpp at `9aa51ce4f`, read for TQ2_T's two codebook tables.
The run took 2 hours: the TCQ parity search on Gaussian data, then the codebooks (31 minutes; ~80–150 s for each 16-bit
state code from scratch), then the 84 matrices (87 minutes, on a CPU shared with other jobs). A smoke run on a few rows
first checked the readers: Swift's own Q2_0 against the BF16's gate rows (the fused `gate_up_proj` holds gate first)
and Gyro-M's rotation, undone as `(W' H) * signs`; the other order scores 13.5× Q2_0's error (`file:gyro_m.alt`).

The TCQ codes' parity masks (`codes.json`) are the best of every 2- and 3-term XOR of state bits on Gaussian data:
`tcq8.L12` takes bits 6, 8 and 11 (three terms, est. 4.8 ps), `tcq16.L12` bits 2 and 11, and 6 and 11.

## Provenance and licence

- Swift-1.5 BF16 weights: `ukisai/Swift-Qwen3.8-Flash-Next` (Swift Open License 1.0); Swift-1.5's GSQ-RCO Q2_0 and
  IQ3_S files: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF`; ISTA's GSQ-RCO IQ3_S of the original model; Gyro-M:
  `agentionai/Qwen3.8-Flash-Next-Gyro-GGUF` `05892758a67efff0895d23c6df10a6fdb6be8cf4` (Qwen Community License 1.0).
  Only error statistics of their weights are published here; no weights are redistributed.
- FN4's importance matrix is this repository's own.
- TQ2_T's format and codebooks: agentionai's llama.cpp fork (Apache 2.0), read at run time.
