# C408 / Q963: the card-hop cast below 8 bits (2026-11-04)

On the nine-P106 rig every Flash-Next token crosses eight card boundaries in a layer split, and each crossing carries
the residual: four hyper-connection streams of 2,560 values. C200's `LLAMA_HOP_CAST=q8_0`
([2026-11-04_card_hops](../2026-11-04_card_hops/README.md)) sends them in Q8_0 instead of F32, a quarter of the bytes,
at no measurable KL divergence (C200k), for +10–15 % decode and +22–31 % prompt speed on a rig (C200r). This folder
takes the same switch below 8 bits: the residual crosses in Q6_K, Q5_1, Q5_0, Q4_1, Q4_0 or IQ4_NL. Each type is lossy,
so each is gated by C200k's paired KL divergence before it goes near a rig.

The registry rows are Q963 (this work) and Q750 / Q231 (the Q8_0 and F16 casts) in
[quants/experiments.md](../../../quants/experiments.md); the queue row is C408.

**Status:** built for the rig and for RTX 30 cards, every check on the CPU passes (below). The KL-divergence job
([run/kld_lowbit.sh](run/kld_lowbit.sh)) and the rig lines ([run/rig_lowbit.sh](run/rig_lowbit.sh)) come next.

## The bytes

A token's residual is 10,240 values; a 4-token verify pass carries four times that, a 512-token prompt micro-batch 512
times. Q6_K packs 256 values with 16 int8 sub-scales and one F16 scale; the others pack 32 values with an F16 scale
(and an F16 minimum for Q5_1 and Q4_1).

| Type | Bits a value | Bytes a token-hop | Against Q8_0 | A 4-token pass | A 512-token micro-batch |
|---|---:|---:|---:|---:|---:|
| F32 (no cast) | 32 | 40,960 | +276 % | 163,840 | 20,971,520 |
| F16, BF16 | 16 | 20,480 | +88 % | 81,920 | 10,485,760 |
| Q8_0 (the release line's cast) | 8.5 | 10,880 | — | 43,520 | 5,570,560 |
| Q6_K | 6.5625 | 8,400 | −22.8 % | 33,600 | 4,300,800 |
| Q5_1 | 6 | 7,680 | −29.4 % | 30,720 | 3,932,160 |
| Q5_0 | 5.5 | 7,040 | −35.3 % | 28,160 | 3,604,480 |
| Q4_1 | 5 | 6,400 | −41.2 % | 25,600 | 3,276,800 |
| Q4_0, IQ4_NL | 4.5 | 5,760 | −47.1 % | 23,040 | 2,949,120 |

F32 → Q8_0 took 30,080 bytes off each token-hop; Q8_0 → Q4_0 takes another 5,120, a sixth of that step. So if a type
passes, expect a fraction of C200r's gain: most where a pass carries many tokens (prompts, verify batches), little in
plain decode, minus what the cast kernels cost.

## The build

Fork branch `pascal-p106-v3-b12076-hopcast4`, one commit on `pascal-p106-v3-b12076-kshape` (`d27845b67`):

| Commit | What it does |
|---|---|
| `3dbf1105a` | `LLAMA_HOP_CAST` takes nine types; the CUDA copy kernels F32 → Q6_K, Q6_K → F32 and IQ4_NL → F32; `supports_op` lists them; `test-backend-ops` copy cases at the hop shapes for every type; `test-hop-cast` round trips for every type |

## The switch

| Switch | Default | Values |
|---|---|---|
| `LLAMA_HOP_CAST` | off | `f16`, `bf16`, `q8_0` (C200), and `q6_k`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl`, in any case (`q6_K` too). Unset, empty, `0`, `off` or `f32`: off. Any other value logs a warning and leaves the hops in F32 |
| `LLAMA_HOP_CAST_AT` | the layer split's boundaries | as in C200: the layers to cast before, for a rig's boundaries in a run with another split |

A type's block must divide the residual's rows (Flash-Next's 2,560 holds 10 Q6_K super-blocks of 256 and 80 blocks of
32), or that hop stays F32. At context creation the switch logs `card hops in <type> before layers <list>`.

## The kernels

The graph is C200's: two `CPY` nodes per boundary, the narrow one pinned to the card before it and the wide one to the
card after it, so the scheduler copies the narrow tensor. What the new types needed was copy kernels on the CUDA side;
the CPU has them all (`from_float` / `to_float` of each type).

- **Q5_0, Q5_1, Q4_0, Q4_1, F32 → IQ4_NL:** the CUDA copy kernels were already there (one thread per 32-value block,
  the same rounding as the CPU's reference quantizers).
- **IQ4_NL → F32:** a pair dequantizer in the form of the other types' (`dequantize_iq4_nl_pair`), through the existing
  `cpy_q_f32`. Bit for bit the CPU's `d * kvalues_iq4nl[q]`.
- **F32 → Q6_K:** the CPU's `quantize_row_q6_K_ref` on 16 threads per 256-value super-block, one 16-value sub-block
  each: `make_qx_quants(16, 32, x, L, 1, NULL)` (weights x², the scale searched over 32 + 0.1 i for i = −9..9) written
  with IEEE division and without FMA contraction (`__fdiv_rn`, `__fmul_rn`, `__fadd_rn`), so that under the build's
  `-use_fast_math` it does the reference's arithmetic; the super-block's scale (the sub-block scale of largest
  magnitude, the first on ties, as the CPU's loop) by shuffles within the half-warp; the int8 sub-scales and the 6-bit
  codes as the reference; the interleaved low and high bits packed from the codes in shared memory, 8 `ql` and 4 `qh`
  bytes a thread. Eight super-blocks a thread block of 128.
- **Q6_K → F32:** `dequantize_q6_K`'s 64 threads per super-block, eight super-blocks a thread block. Bit for bit the
  CPU's `d * sc * q`.

## The checks without a GPU

The build machine ran [run/cpu_checks.sh](run/cpu_checks.sh) in containers without a GPU: the rig tarball, the RTX 30
build, CPU-only builds of the branch and of its base `d27845b67`, and these checks ([checks.tsv](checks.tsv)). Every
check passes:

| Check | Result |
|---|---|
| Rig build | `llama.cpp-20261104-3dbf1105a-pascal-p106-v3-b12076-hopcast4.tar.gz`: 515,662,898 bytes, md5 `1e29d4fb5db386b429f4f865daeb3d02`, sm_61 code and compute_61 PTX, no AVX, GLIBC_2.38; the test binaries from its tree in `extra-3dbf1105a.tar.gz` (md5 `9faa658cc351f86ecb02b2fb7f50480a`). On sm_61 the F32 → Q6_K kernel takes 88 registers, 2,048 bytes of shared memory and no local memory; Q6_K → F32 takes 32 registers |
| RTX 30 build (sm_86) | `llama-perplexity` (md5 `8bf5a31359da76396bdae5646f64fd08`), `test-backend-ops`, `test-hop-cast`, `test-llama-archs` of the same commit, for the KLD job |
| `test-backend-ops -b CPU -o CPY` | 591 / 591, the 54 cases at the hop shapes among them (nine types, to and from, T = 1, 4, 512), and Q6_K and IQ4_NL now in the all-types copy cases ([ops_cpu.txt](ops_cpu.txt)) |
| Round trips (`test-hop-cast --unit-only`) | all nine types at T = 1, 4 and 512: data the type represents comes back bit for bit; residual-like data (normal, with channels at 8× and 60×) stays within each type's bound, at worst 0.69 of it for Q6_K and 0.52 for IQ4_NL, 0.97–1.00 for the others; 54 of 54 checks ([unit.txt](unit.txt)) |
| The model (`test-hop-cast`, random qwen4exp) | the reference twice, the staged hop, and Q6_K and Q4_0 without a boundary: the same logits bit for bit. The round trip before layer 1 in every type: every one of the 10 hops per arm within the type's bound, layer 0 unchanged, 121 of 121 checks ([model.txt](model.txt), [model.tsv](model.tsv)); the error and the logits' change below |
| Identity (`test-llama-archs -b CPU`, every architecture) | with the switch off all 136 architectures' logits equal `-kshape`'s bit for bit; with `GGML_CUDA_HOP_STAGED=1 LLAMA_HOP_CAST=q4_0` (no boundary on the CPU) the same; with `LLAMA_HOP_CAST=q6_k` or `q4_0` and `LLAMA_HOP_CAST_AT=1` only qwen4exp changes. The 7 failures in each run's summary are the test's own (C200's note: a second model of 7 families compared against the first one's logits file; each run repeats the base's numbers for them), the 8th with a cast is qwen4exp's change ([identity.tsv](identity.tsv), [identity_summary.txt](identity_summary.txt)) |
| `test-llama-context` | the random qwen4exp: 19 passed, 1 skipped, 0 failed at the default and with the Q6_K and Q4_0 round trips before layer 1 ([ctx.txt](ctx.txt)) |
| llama-server | speculative decoding with verify passes of 1–4 tokens (392 tokens drafted) and prompts in 32-token micro-batches: `off`, `staged`, `q8auto` and `q4auto` give `-kshape`'s tokens and top-3 probabilities in all four requests; the round trip before layer 1 in F16, Q8_0, Q6_K, Q5_0, Q4_0 and IQ4_NL logs the hop in the target's and the draft's contexts and completes every request, with tokens that differ from the base's (a random model's near-tied logits) ([server.tsv](server.tsv), [server_verdict.txt](server_verdict.txt)) |

**What each type costs on the random model** (the round trip before layer 1; the KLD from the switch-off arm over 58
positions, and the hop's error as a share of the residual's rms):

| Type | Mean KLD | Top-1 agreement | Error / rms of the residual |
|---|---:|---:|---:|
| F16 | 3.2e-11 | 100 % | 0.021 % |
| BF16 | 2.1e-09 | 100 % | 0.17 % |
| Q8_0 | 2.2e-08 | 100 % | 0.54 % |
| Q6_K | 2.3e-07 | 100 % | 1.8 % |
| Q5_1 | 1.1e-06 | 96.6 % | 3.8 % |
| Q5_0 | 1.4e-06 | 96.6 % | 4.3 % |
| Q4_1 | 4.8e-06 | 93.1 % | 7.9 % |
| IQ4_NL | 4.8e-06 | 91.4 % | 7.9 % |
| Q4_0 | 5.8e-06 | 91.4 % | 8.7 % |

On the random-weight model one hop of Q6_K costs ten times Q8_0's KLD, the 5-bit types fifty times and the 4-bit ones
over two hundred times; only the real model at the rig's eight boundaries decides (the KLD job below). Residual-like data with 60× outlier channels
widens the gaps ([unit.txt](unit.txt): Q8_0 1.2 %, Q6_K 3.2 %, Q5_1 5.5 %, Q5_0 7 %, Q4_0 9 %, IQ4_NL 9 %, Q4_1 11 % of
the rms), because a 32-value block's step follows its largest value; Q6_K's 16-value sub-scales follow it less.

## Next: the KL divergence, then the rig

1. **The KL divergence** ([run/kld_lowbit.sh](run/kld_lowbit.sh), two RTX 3090s, ~1 h 45 min; or two pairs, ~1 h):
   C200k exactly, generalised to any type: Swift-1.5 Q2_0 of Flash-Next with the round trip at the rig's eight
   boundaries (`LLAMA_HOP_CAST_AT=6,11,16,22,27,32,38,43`) in each type, against the quality chain's Q8_0 bases
   ([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)) with its `kld.sh` unchanged, each arm
   paired by chunk against `off` on the same cards ([table.sh](../2026-11-04_moe_tail_kld/run/table.sh)). Gate, per
   arm and text: the paired change within noise (d − 2 se ≤ 0), as Q8_0's was. Before the arms, on the same cards:
   `test-backend-ops -b CUDA0 -o CPY` (the new kernels against the CPU) and `test-hop-cast --unit-only` (each type's
   round trip on the card against the CPU's), then `test-hop-cast` on the model itself with the round trip at the eight
   boundaries in every type: per boundary the residual's largest value and rms and the round trip's error (Q757).
2. **The rig** ([run/rig_lowbit.sh](run/rig_lowbit.sh), ~15 min an arm, nothing else running): C200r's server line
   (pmix-q2-fast, S12's no-draft split, the release line's kernel settings and staged hop) once per type that passed,
   with Q8_0 as the control: decode, ~1k / 8k / 32k prompts and one from a checkpoint. Then, for the best type, the
   agent check against S35's `s35stack`.

## Files

| File | What it is |
|---|---|
| [checks.tsv](checks.tsv) | one line per CPU check: verdict and detail |
| [ops_cpu.txt](ops_cpu.txt) | the hop-shape `CPY` cases of `test-backend-ops -b CPU` and its totals |
| [unit.txt](unit.txt) | `test-hop-cast --unit-only` on the CPU: the round trips of the nine types at [2560, 4, T] |
| [model.tsv](model.tsv), [model.txt](model.txt) | `test-hop-cast` on the random qwen4exp, per arm: identity with the reference, logit differences, KLD, top-1 agreement, hops and their bound; the boundary's range and error |
| [identity.tsv](identity.tsv), [identity_summary.txt](identity_summary.txt) | `test-llama-archs` logits of every architecture, the base's own lines and the branch's four settings against them; each run's summary line |
| [ctx.txt](ctx.txt) | `test-llama-context` totals at the three settings |
| [server.tsv](server.tsv), [server_verdict.txt](server_verdict.txt) | llama-server's speculative decoding, the base build against the branch's arms |
| [run/cpu_checks.sh](run/cpu_checks.sh) | the build-and-check chain (steps 1–11) |
| [run/server_check.sh](run/server_check.sh) | the llama-server check (C200's, with the new types' arms) |
| [run/kld_lowbit.sh](run/kld_lowbit.sh) | the GPU check, the boundaries on the real model and the KLD arms |
| [run/rig_lowbit.sh](run/rig_lowbit.sh) | the rig lines, one per type |

## Commands

```bash
# build and CPU checks (a machine with Docker; WORK holds a llama.cpp clone with the fork as remote "fork")
WORK=<work dir> REPO=<this repository> JOBS=8 bash docs/data/2026-11-04_hopcast_lowbit/run/cpu_checks.sh
# the KLD job (the quality chain's environment: MOUNTS, WIKI, AGENT, CODE; BASES = its kld folder; two free cards)
BIN=<work dir>/sm86/bin BASES=<bases> MODEL=<Swift Q2_0 first shard> OUT=<dir> GPUS=<a>,<b> TAG=lowbit \
  MOUNTS=... WIKI=... AGENT=... CODE=... bash docs/data/2026-11-04_hopcast_lowbit/run/kld_lowbit.sh
# the rig (a folder with rig_lowbit.sh, fn_line.sh and fn_requests.py), detached on the rig
BUILDS=<folder of builds> BUILD=<this build> MODEL=<Flash-Next first shard> ARMS="q8_0 q6_K ..." \
  setsid nohup bash rig_lowbit.sh > console.log 2>&1 < /dev/null &
```

Provenance: the fork's code and this folder's scripts are this project's; the random-weight models are generated by
`test-llama-archs`; no model was downloaded for this work.
