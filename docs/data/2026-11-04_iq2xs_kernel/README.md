# Pascal kernels for IQ2_XXS, IQ2_XS and IQ3_S, and what Swift's IQ2_XS file puts on nine cards (2026-11-04)

Registry row Q222. The Flash-Next quality chain (FN2,
[2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)) scored Swift-1.5's GSQ-RCO "IQ2_XS" file
closer to the Q8_0 reference on held-out agent text than the Q2_0 file the rig runs (KL divergence 0.225 against
0.285, the same top token 91.5 % against 90.7 %; IQ3_XXS 0.176, 92.3 %). This folder answers what that file needs on
nine 6 GB P106-090 cards, from its GGUF headers only, and adds Pascal matrix-vector kernels for the grid types it and
its siblings decode through llama.cpp's stock kernels: IQ2_XXS, IQ2_XS and IQ3_S, beside the IQ2_S and IQ3_XXS kernels
of [2026-10-27_iq_kernels](../2026-10-27_iq_kernels/README.md). Fork branch `pascal-p106-v3-b12076-iq2xs` of
[Miner Llama](https://github.com/Cinderella-Man/llama.cpp), cut from `pascal-p106-v3-b12076-ladder2` (`a330672c1`,
[2026-11-04_ladder2_merge](../2026-11-04_ladder2_merge/README.md)).

## What the files hold

Read from the headers with [run/iq_layout.py](run/iq_layout.py): every tensor kind's types and shapes, and per weight
type the bytes on the cards and the bytes one decoded token reads there (dense tensors whole, routed experts 10 of 512).
Full listings: [layout_swift-iq2_xs.txt](layout_swift-iq2_xs.txt), [layout_swift-iq3_xxs.txt](layout_swift-iq3_xxs.txt),
[layout_swift-iq3_s.txt](layout_swift-iq3_s.txt), [layout_swift-q2_0.txt](layout_swift-q2_0.txt).

- **The "IQ2_XS" file holds no IQ2_XS tensor.** Its name is the mix's (`general.file_type` 20). The routed gate and up
  projections are IQ2_S in 34 layers, IQ2_XXS in 11 and IQ1_M in 3; every routed down projection is Q2_0 (640-wide
  rows, which no 256-weight type fits); attention, the recurrent layers' output and the shared expert are IQ4_XS, IQ3_S
  and Q6_K, the shared expert's down projection IQ4_NL, Q8_0 or Q2_0.
- **IQ2_XS is in the IQ3_XXS file:** routed gate/up in IQ2_XS (10 layers), IQ2_XXS (9), IQ3_S (13), IQ2_S and IQ3_XXS;
  IQ3_S also in its attention. The IQ3_S file has IQ3_S gate/up in 10 layers.

| Swift-1.5 GSQ-RCO file | On the cards | A token reads | Of it in IQ2_XXS, IQ2_XS, IQ3_S | Bytes on Pascal kernels: before, with these |
|---|---|---|---|---|
| Q2_0 (the rig's file) | 37.47 GB | 4.16 GB | — | 93.5 %, 93.5 % |
| IQ2_XS | 39.00 GB | 4.24 GB | 256 MB (6.0 %) | 86.5 %, **92.5 %** |
| IQ3_XXS | 46.88 GB | 4.81 GB | 484 MB (10.1 %) | 80.3 %, **90.3 %** |
| IQ3_S | 54.59 GB | 5.29 GB | 141 MB (2.7 %) | 84.8 %, 87.4 % |

What stays on llama.cpp's kernels: the routers (F32 `ffn_gate_inp`, 2,560 × 512 per layer, 252 MB a token in every
file), IQ4_NL (the IQ3_S file's routed down projections, 403 MB a token; the shared expert's down elsewhere) and IQ1_M
(22 MB in the IQ2_XS file).

## What must sit on nine 6 GB cards

The IQ2_XS file is two shards, 39.79 GB and 28.36 GB. **The n-gram table (`per_layer_token_embd`, IQ4_NL,
160 × 320,001,536, 28.80 GB) is in the first shard, with 341 other tensors**; the second holds the rest of the
weights. The table and the token embeddings (IQ4_XS, 0.34 GB) stay in host memory, so to read the table from the SSD as
the rig does for the Q2_0 file it needs its own shard first (`scripts/gguf_reshard.py split --isolate
'^per_layer_token_embd\.'`, as in [2026-11-03_flash_next_fit](../2026-11-03_flash_next_fit/README.md)).

Everything else goes to the cards: **37,197 MiB** (36,862 MiB in the 48 blocks, 335 MiB of output layers), 1,467 MiB
more than the Q2_0 file's 35,730. [scripts/split_plan.py](../../../scripts/split_plan.py) with the costs of Q213's fit
probes (64k context, `-ub 512`, F16 cache: 161 MiB of KV per attention layer, 3 MiB of state per recurrent layer,
compute buffers of 600 MiB and 674 MiB on the last card, 6,001 MiB free per card)
([fit_nine_cards.txt](fit_nine_cards.txt)):

| File | Whole layers: best `-ts` | Smallest free | With routed-expert tensors moved by `-ot` | Smallest free |
|---|---|---|---|---|
| Q2_0 | 6,5,5,6,5,5,6,5,6 | 737 MiB | 10 tensors | 1,032 MiB |
| **IQ2_XS** | **6,6,6,5,5,5,5,5,6** | **569 MiB** | 11 tensors | **877 MiB** |
| IQ3_XXS | — (−409 MiB) | | 8 tensors | 51 MiB |
| IQ3_S | — (−1,253 MiB) | | — (−860 MiB) | |

So the IQ2_XS file fits the rig at 64k context on whole layers with ~570 MiB to spare on the fullest card (cards 0, 1,
2 and 8 hold 5,342–5,432 MiB), or ~880 MiB with 11 expert tensors moved; the IQ3_XXS file only with moved experts and ~50 MiB
to spare; the IQ3_S file not at all.

## The kernels

`mmvq-pascal.cu` of the fork runs IQ2_XXS, IQ2_XS and IQ3_S as it runs IQ2_S and IQ3_XXS: one to four tokens, plain
products, expert products (`MUL_MAT_ID`), the fused expert gate/up with its GLU and the routed + shared expert launch;
one 256-weight super-block per lane, blocks staged through shared memory with the widest loads the rows allow (66-,
74- and 110-byte blocks start at 2-byte boundaries, like Q6_K's), registers for three blocks per SM (round 2's
default). Each block copies its type's grid into shared memory (IQ2_XXS 256 entries of 8 bytes, IQ2_XS 512 of 8,
IQ3_S 512 of 4). A weight's sign is applied as `sum(g * u) - 2 * sum((g & m) * u)`, two dp4a and one AND per 4 weights,
with the byte masks m computed from the sign bits in registers (IQ2_XXS and IQ2_XS: 7-bit patterns whose eighth sign
is their parity; IQ3_S: a sign byte per 8 weights; computing them won round 2 for the older types). Each 16- or
32-weight scale multiplies its integer sum in FP32.

The decoding is in a new header, `ggml/src/ggml-cuda/mmvq-pascal-iq.cuh`, free of ggml's headers, so that
[bench/mmvq_pascal/verify/iq_unit_check.cu](../../../bench/mmvq_pascal/verify/iq_unit_check.cu) builds it for the host
and checks it against ggml's own dequantization on a CPU. The other types' code is unchanged.

At Flash-Next's widths the rows of 10 super-blocks (K = 2,560) have 2 lanes per row and stage with 4-byte loads; K =
6,144 stages with 16-byte loads.

## Checks without a GPU

All on a build machine's CPU in the CUDA 12.9 container ([run/cpu_checks.sh](run/cpu_checks.sh)):

| Check | Result |
|---|---|
| Decoding against ggml's dequantization ([decoding_check.md](decoding_check.md)) | IQ2_XXS, IQ2_XS, IQ3_S: 400 cases each, worst 0.0016 of the tolerance; the mutation run fails exactly IQ2_XXS and IQ3_S (worst ~9,000 and ~6,900 times the tolerance) |
| Rig tarball (`scripts/build_llamacpp_docker.sh`, `TESTS=1`) | `llama.cpp-20261103-6c88b6991-pascal-p106-v3-b12076-iq2xs.tar.gz`, 360,853,175 bytes, md5 `fbc474e806ff4d47d9cb486a2f1bac2e`; compute 6.1, no AVX, glibc up to 2.38 |
| Registers ([registers.tsv](registers.tsv), `cuobjdump -res-usage` of test-backend-ops) | 99 grid-type kernels, none with local memory; the one-token kernels 90–168 registers and no stack; every 2–4-token kernel (`_mc`, all five types, the old ones too) at the 168-register cap with an 80–224-byte stack frame (Q714) |
| SASS against the base build `-ladder2` ([sass_summary.txt](sass_summary.txt)) | 7,326 kernels: all 7,263 of the base's identical, 63 new, every one IQ2_XXS, IQ2_XS or IQ3_S |

So the decoding is right, and nothing the rig ran before changed: every earlier kernel compiles to the same machine
code.

## Not yet run: on a GPU

Nothing ran on a GPU: the quant machine's cards held the quality chain and the rigs were off. Next, in order:

1. **Correctness through the compute 6.1 PTX on any free quant-machine GPU** ([run/gpu_checks.sh](run/gpu_checks.sh),
   `MODE=ptx`, ~30–60 min, mostly the CPU's IQ quantization of the reference weights): test-backend-ops of the five
   grid types at 1–4 tokens (Flash-Next's shapes and the earlier Ornith ones), the kernels on, then the new types on
   llama.cpp's kernels (`skip`), then compute-sanitizer on 15 cases. Every run must pass every case.
2. **One P106** (`MODE=native`, the same cases without the dense shapes of 6,144 rows or more, which a 2-core host
   quantizes for minutes; est. ~1 h), then the timing:
   [bench/mmvq_pascal/iq_perf.sh](../../../bench/mmvq_pascal/iq_perf.sh) (~10 min), tables by
   [iq_perf_table.py](../../../bench/mmvq_pascal/iq_perf_table.py) with the layouts above for the per-token sums.
   Expected from round 2 (IQ2_S and IQ3_XXS at Ornith's shapes, one token: 1.2–1.8× llama.cpp's kernels, 28–53 GB/s):
   est. 2–5 ms less per token for the IQ2_XS file (256 MB of the three types a token) and 3–9 ms for IQ3_XXS (484 MB).
3. **In the model:** the IQ2_XS file resharded with its table alone (Q710), then the agent check against the Q2_0 line.

## Files

| File | What |
|---|---|
| `layout_swift-*.txt` | [run/iq_layout.py](run/iq_layout.py) on four Swift-1.5 GSQ-RCO files: tensor kinds, types, shapes, bytes; per type the bytes a token reads and whether a Pascal kernel takes it; the share covered before Q222 |
| `fit_nine_cards.txt` | [scripts/split_plan.py](../../../scripts/split_plan.py) on the same four files: the best whole-layer splits and the best with expert tensors moved, for nine 6 GB cards at 64k context |
| `decoding_check.md` | the host decode check and its mutation run |
| `registers.tsv` | registers, stack, shared and local memory of every grid-type kernel (sm_61) |
| `sass_summary.txt` | the per-kernel SASS identity of the new build against the base's |
| `run/iq_layout.py` | per-type bytes a token reads, from GGUF headers (standard library; imports `scripts/gguf_reshard.py`) |
| `run/cpu_checks.sh` | the checks without a GPU: decode check, rig tarball, registers, SASS identity |
| `run/gpu_checks.sh` | tomorrow's correctness on one GPU (`MODE=ptx` or `native`) |

## Commands

```bash
# headers only (any machine with the files; nothing is read past the tensor index)
python3 -I run/iq_layout.py <shard 1> <shard 2> > layout.txt
python3 -I ../../../scripts/split_plan.py --cards 9 --free 6001 --kv-mib 161 --state-mib 3 --compute 600,674 --top 2 <shards>
python3 -I ../../../scripts/split_plan.py --cards 9 --free 6001 --kv-mib 161 --state-mib 3 --compute 600,674 \
    --split-experts --top 1 <shards>

# the checks without a GPU, on a machine with Docker (WORK a folder there; REPO a checkout of this repository;
# TREE the branch's source, e.g. `git archive pascal-p106-v3-b12076-iq2xs | tar -x -C $TREE`; BASE_BUILD the unpacked
# tarball of the base, pascal-p106-v3-b12076-ladder2)
WORK=$WORK REPO=$REPO TREE=$TREE REF=fork/pascal-p106-v3-b12076-iq2xs BASE_BUILD=$BASE_BUILD JOBS=6 \
    bash run/cpu_checks.sh

# GPU correctness: a newer GPU through the compute 6.1 PTX, then a P106
docker run --rm --gpus device=<n> -v $WORK:$WORK nvidia/cuda:12.9.2-devel-ubuntu24.04 \
    env MODE=ptx BIN=$WORK/<tarball dir>/bin OUT=$WORK/gpu-ptx bash $REPO/docs/data/2026-11-04_iq2xs_kernel/run/gpu_checks.sh
ssh $RIG "MODE=native BIN=<builds>/<tarball dir>/bin OUT=/tmp/iq_gpu bash -s" < run/gpu_checks.sh
# timing on the P106
ssh $RIG "BIN=<builds>/<tarball dir>/bin OUT=/tmp/iq_perf bash -s" < ../../../bench/mmvq_pascal/iq_perf.sh
python3 -I ../../../bench/mmvq_pascal/iq_perf_table.py iq_perf --layout layout_swift-iq2_xs.txt \
    --layout layout_swift-iq3_xxs.txt --layout layout_swift-iq3_s.txt > iq_perf.csv
```

## Provenance

The model files: [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(IQ2_XS, IQ3_S, IQ3_XXS, Q2_0; the IQ3_XXS and Q2_0 files as resharded for the rig with their tables alone); only
their headers were read. Kernel code: Miner Llama (MIT, on llama.cpp by the ggml authors); its grids and block layouts
are ggml's. Tables and scripts of this repository (MIT).
