# Skipping the routed experts that `LLAMA_MOE_TAIL` drops: the kernels (Q660)

Registry rows: [Q660](../../../quants/experiments.md) (the kernels skip the dropped experts), Q661 (dropped slots read
their token's last kept expert instead), [Q618](../../../quants/experiments.md) (the cut itself). The cut's quality:
[2026-11-04_moe_tail_kld](../2026-11-04_moe_tail_kld/README.md) (p = 0.9 renormalized keeps 8.77 of Flash-Next's 10
routed experts per token at no measurable KL divergence, p = 0.8 keeps 7.27 at a small cost). The graph-level switch
it replaces: [2026-11-04_bytes_switches](../2026-11-04_bytes_switches/README.md).

Qwen3.8-Flash-Next routes each token to 10 of 512 experts per layer. `LLAMA_MOE_TAIL` drops a token's low-weight tail
of those 10 (a slot goes once the slots ranked above it hold a share p of the token's routed weight, or when its own
share is below w). Until now it worked at graph level: about nine small ops per MoE layer set the dropped slots'
weights to 0, and the expert kernels still read their rows, so the cut cost launches and saved no bytes. This branch
makes the cut one op that the router kernel computes in registers, and makes the expert kernels skip the dropped
experts' rows.

## How it works (fork branch `pascal-p106-v3-b12076-moeskip`)

- **`GGML_OP_MOE_TAIL`** (`ggml_moe_tail`, `ggml_moe_tail_ids`): two nodes right after the router's top-k chain, the
  cut weights (renormalized to the token's sum or not) and the cut ids. A dropped slot's id is **-1** in micro-batches of
  up to `n_skip` tokens (8: generation, verification and the short micro-batches of a prompt); larger ones keep every
  slot's own expert at weight 0 (as the graph-level ops compute it), so that the prompt path's kernels (MMQ) see the
  router's ids unchanged.
- **The arithmetic** is that of the graph-level ops it replaces (a double sum of the weights, the shares, a float
  running sum minus the share, `step(p - before)`, `step(share - w)`, the renormalization), so on the CPU backend the
  op's logits are the graph's bit for bit. CUDA computes the same with IEEE intrinsics (`__fdiv_rn`, `__dadd_rn`), in
  its own kernel and in the fused router kernel (`topk-moe.cu`): when the two nodes follow the chain, the same launch
  that picks the top-k writes the cut weights and ids from registers (up to 32 experts used).
- **Skipping -1 on CUDA** (llama.cpp's CPU backend already wrote zeros for -1): the one-token and multi-token expert
  kernels a P106 runs (`mul_mat_vec_q` and `mul_mat_vec_q_moe`, the fork's Pascal kernels in `mmvq-pascal.cu` and the
  multi-token `-mtok` kernels for Q2_0 and the IQ types) read a slot's id first and, for -1, write zeros to its output
  rows and return before touching the weights. Zeros, not nothing: the SwiGLU and the down projection read every
  slot. Kernels that read every slot's expert (MMQ, MMF, MMVF, the host fallback) run on a copy in which the slot names
  an expert its token does not (llama.cpp's MMQ reads past its buffers when a token names an expert twice, Q875), and
  then zero the slot's rows; ids that come from the router itself never take that detour. `ADD_ID` adds nothing for -1,
  and `GET_ROWS` reads row -1 as zeros (an expert's scale, looked up by the cut ids) and as -1 from I32 rows (the MoE
  cache's slot lookup, whose upload plan skips -1).
- **Switches:** `LLAMA_MOE_TAIL=p=<p>,w=<w>,renorm=<0|1>,hist=<0|1>` as before, plus `graph=1` (the graph-level ops,
  for comparison), `skip=<n>` (`n_skip`; `skip=0`: no -1, every slot keeps its expert at weight 0) and `repeat=1`
  (Q661's variant: up to `n_skip` tokens a dropped slot reads its token's last kept expert again instead of being
  skipped, the kernels unchanged; for the fork's kernels, which take a repeated expert). `hist=1` reads the experts
  kept back after every micro-batch and is a diagnostic: time with `hist=0`.

## Results

**The op gives the graph-level cut's logits bit for bit, on the CPU backend and on CUDA**, in micro-batches of 64, 4 and
1 tokens, with -1 ids, with `repeat` and with `skip=0`, across every architecture of `test-llama-archs`; **every expert
kernel takes -1**, and with the switch off the logits are `-kshape`'s bit for bit. Fork commit `f3072cbe8` (one commit
on `-kshape` `d27845b67`). Every check: [checks.tsv](checks.tsv). Timed on nine P106s below
([On the rig](#on-the-rig-nine-p106s-c120r-2026-11-05)).

| Check | Where | Result |
|---|---|---|
| `test-bytes-switches`, random qwen4exp (64 experts, 10 used; F32 and Q5_K heads; routers × 30) | CPU | 2 of 2 models pass, 0 failed checks: every op arm (7 cuts × 3 micro-batch sizes × 3 modes per model) gives the graph-level ops' logits and experts-kept histogram bit for bit; at p = 1 the switch on equals the switch off |
| `test-llama-archs -b CPU`, p = 0.8 renormalized and a 5 % floor; the op, `skip=0`, `repeat=1` | CPU | every architecture's logits equal the graph-level cut's (136 comparisons per run equal; the other 7 compare a dense model with a MoE model of the same architecture name, identically in the graph run itself) |
| test-backend-ops, skipped slots past 8 tokens (MMQ, MMF, the host fallback) | RTX 3080, PTX | 100 of 100 |
| the same, every skipped-slot case: defaults (the Pascal kernels, `-mtok`); llama.cpp's kernels (`MMVQ_PASCAL=0`); Q2_0 on the earlier multi-token kernel (`MTOK=0`) | RTX 3080, PTX | 568 of 568; 568 of 568; 41 of 41 |
| `MOE_TAIL`, `TOPK_MOE_TAIL` (the fused router chain), `GET_ROWS` (row -1), `ADD_ID` | RTX 3080, PTX | 493 of 493 |
| every expert case with nothing skipped (`MUL_MAT_ID`, `MUL_MAT_VEC_FUSION`, `MUL_MAT_ID_SHARED`, `TOPK_MOE`) | RTX 3080, PTX | 4,237 of 4,237 |
| `test-llama-archs -b CUDA0`, the switch off, against `-kshape`'s build | RTX 3080, PTX | 136 of 136 bit for bit (the 7 name collisions as in the base run) |
| `test-llama-archs -b CUDA0`, p = 0.8 renormalized: the op (the fused router kernel, -1 ids), `skip=0`, `repeat=1` against the graph-level cut | RTX 3080, PTX | each 136 of 136 bit for bit |

The PTX runs used the rig build's compute 6.1 PTX on the newer card with the fork's Pascal kernels forced on and a
P106's launch shapes (5 SMs, 96 KB of shared memory per SM), as for `-kshape`.

What the first runs found, and changed:

- **An expert's scale lookup takes the cut ids.** qwen4exp multiplies each routed expert's product by its scale
  (`ggml_get_rows(scales, ids)` in `build_lora_mm_id`), so `GET_ROWS` must take -1 too: it reads zeros (and -1 from
  I32 rows).
- **llama.cpp's MMQ cannot take a repeated expert** ([Q875](../2026-11-04_ladder3_merge/gpu_ptx.md): it reads past its
  buffers): the first design gave a dropped slot its token's last kept expert in micro-batches over 8 tokens, and the
  ids copy for MMQ put expert 0 in place of -1, both of which repeat an expert within a token; through the PTX the
  skipped-slot MMQ cases faulted (Xid 31, an illegal address). Now micro-batches over `n_skip` tokens keep every slot's
  own expert (at weight 0, as the graph-level ops compute it), the ids copy names experts the token does not, and the
  last kept expert is the opt-in `repeat=1` (Q661) for 1 to 8 tokens, where the fork's kernels take repeats.

## On the rig: nine P106s (C120r, 2026-11-05)

T(k) on real agent text, [C271's method](../2026-11-04_kreal_p106/README.md): Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0 on
nine P106-090s, the tarball of `f3072cbe8`, one llama-server load per arm, per k one request whose 1,680 prompt tokens
run in micro-batches of exactly k (the pipelining off, as a verify pass runs); T(k) = the request's prompt time /
(1,680 / k). Every other switch at its default (the release line's `KCOL_SMEM=32768 KCOL_TABLE=0` is within ~3 ms of
the defaults on real text, [C350](../2026-11-04_kreal_p106/README.md)). The arms ran in this order on the idle rig:
`off`, `p090r`, `p080r`, `p090rr`, `off2`; `off` and `off2` are the same setting loaded at the start and at the end, and
their difference is the spread between two loads. The changes are against the mean of the two. The `check` step
(test-backend-ops on one card) was left out: it computes its references on the host's two-core Celeron, and the PTX runs
above cover the same cases.

ms per pass of k real tokens ([rig/tk.tsv](rig/tk.tsv), [rig/requests.tsv](rig/requests.tsv)):

| k | `off` | `off2` | spread: `off2` − `off` | `p090r` | `p080r` | `p090rr` | `p090r` − off | `p080r` − off | `p090rr` − off | `p090rr` − `p090r` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67.5 | 65.3 | −2.2 | 64.5 | 64.3 | 65.3 | −1.8 (−2.8 %) | −2.1 (−3.1 %) | −1.1 (−1.6 %) | +0.8 |
| 2 | 92.7 | 92.7 | −0.0 | 92.3 | 89.7 | 92.4 | −0.4 (−0.4 %) | −3.0 (−3.2 %) | −0.2 (−0.2 %) | +0.1 |
| 3 | 120.3 | 119.6 | −0.7 | 117.9 | 115.2 | 117.9 | −2.1 (−1.7 %) | −4.7 (−4.0 %) | −2.1 (−1.7 %) | +0.0 |
| 4 | 139.6 | 139.9 | +0.3 | 137.3 | 133.1 | 139.2 | −2.5 (−1.8 %) | −6.6 (−4.7 %) | −0.6 (−0.4 %) | +1.9 |
| 5 | 175.2 | 171.3 | −3.9 | 170.5 | 162.8 | 170.3 | −2.8 (−1.6 %) | **−10.5 (−6.0 %)** | −3.0 (−1.7 %) | −0.2 |
| 6 | 197.4 | 193.8 | −3.7 | 194.2 | 185.1 | 192.1 | −1.4 (−0.7 %) | **−10.4 (−5.3 %)** | −3.5 (−1.8 %) | −2.1 |
| 7 | 214.5 | 215.7 | +1.2 | 211.8 | 208.3 | 211.5 | −3.3 (−1.5 %) | −6.8 (−3.2 %) | −3.6 (−1.7 %) | −0.3 |
| 8 | 241.6 | 245.2 | +3.6 | 244.7 | 236.3 | 244.2 | +1.3 (+0.5 %) | −7.1 (−2.9 %) | +0.8 (+0.3 %) | −0.5 |

Where the time goes, the first 340 tokens at k = 1 and 5 under `GGML_CUDA_PROFILE_MODE=group` (one load per arm; the
cards' GPU time per pass, 7.6 µs per span subtracted; [C266](../2026-11-04_kpass_profile/README.md)'s parts;
[rig/parts.tsv](rig/parts.tsv), [rig/summaries/](rig/summaries/)):

| Part | `off`, k = 1 | `p090r`, k = 1 | change | `off`, k = 5 | `p090r`, k = 5 | change |
|---|---:|---:|---:|---:|---:|---:|
| routed experts | 8.94 | 7.95 | **−0.99 (−11 %)** | 30.55 | 27.11 | **−3.44 (−11 %)** |
| attention and recurrent projections | 20.08 | 20.13 | +0.05 | 49.85 | 49.89 | +0.04 |
| hyper-connections (BF16) | 10.65 | 10.66 | +0.01 | 22.82 | 22.81 | −0.01 |
| shared expert | 3.33 | 3.33 | 0 | 6.41 | 6.41 | 0 |
| router | 2.88 | 2.87 | −0.01 | 6.06 | 6.06 | 0 |
| Gated DeltaNet state and conv | 3.52 | 3.53 | +0.01 | 5.45 | 5.45 | 0 |
| attention (FLASH_ATTN_EXT) | 0.83 | 0.83 | 0 | 1.97 | 1.97 | 0 |
| other operations without weights | 6.75 | 6.84 | +0.09 | 8.89 | 9.06 | +0.17 |
| other weights, output layer | 0.78 | 0.78 | 0 | 1.40 | 1.40 | 0 |
| the cards' GPU time | 57.75 | 56.91 | −0.84 | 133.41 | 130.17 | −3.24 |

Each setting in one of three classes (gain: faster beyond the spread of the two `off` loads; negligible: within it and
not slower; loss: measurably slower):

- **`p090r`, the skip at p = 0.9 renormalized: negligible, the sign right.** −1.8 ms a pass at one token and −2.8 ms at
  five, against `off` loads 2.2 and 3.9 ms apart; below both `off` loads at k = 1–5 and 7. The profile shows where:
  the routed experts take 11 % less GPU time at one token and at five (p = 0.9 keeps 8.77 of 10 experts on this text,
  12 % fewer reads), every other part within 0.2 ms. No quality cost ([KLD](../2026-11-04_moe_tail_kld/README.md)).
  With one pair of loads as the spread (about 2.8 ms of noise a load at k = 5), about seven alternated loads per arm
  would settle a 3 ms difference at k = 5 on its own; it is tested with the other negligible changes as one group
  instead (Q1023).
- **`p090rr`, the same cut with `repeat=1` (Q661): negligible, equal to the skip.** It differs from `p090r` by −2.1 to
  +1.9 ms at every k, −0.04 ms on average: the dropped slots' second reads of the last kept expert cost nothing
  measurable, so the P106's L2 serves them. `repeat=1` adds nothing over the skip and stays the fallback for expert
  kernels that cannot take −1.
- **`p080r`, p = 0.8 renormalized: a speed gain.** −3.0 to −10.5 ms a pass at k = 2–8, beyond the largest spread
  (3.9 ms) at k = 3–8; −6.0 % at five tokens; within the spread at one token (−2.1 ms). It keeps 7.27 of 10 experts and
  costs next-token KLD (wiki +0.025 ± 0.003, code +0.011 ± 0.002, the agent text within its noise), so whether it
  goes into the line is a quality question, which the agent check on real tasks answers (C371).
- **The switch off against `-kshape`:** the mean of `off` and `off2` is within 1.7 ms of `-kshape`'s T(k) on the same
  text ([C350](../2026-11-04_kreal_p106/README.md), one load on 2026-11-04) at k = 1–4, 7 and 8, and 4–5 ms above it at
  k = 5 and 6 (173.3 against 168.0, 195.6 against 191.4). Alternated loads of the two builds on one day
  ([below](#the-switch-off-against--kshape-c556-2026-11-05)) put that gap down to `off`'s cold first load and the
  spread between days, and find the real cost of the switch-off path: 0.2 ms a pass.

Cost files for the adaptive draft length (`LLAMA_SPEC_ADAPT_COST`), from these passes:
[t_k_moeskip_p090r_real.txt](../2026-11-04_kcurve_p106/cost/t_k_moeskip_p090r_real.txt),
[t_k_moeskip_p080r_real.txt](../2026-11-04_kcurve_p106/cost/t_k_moeskip_p080r_real.txt).

## The switch off against `-kshape` (C556, 2026-11-05)

With `LLAMA_MOE_TAIL` unset this branch builds `-kshape`'s graphs, but its expert kernels still read each slot's id and
test it for −1, and the fused router kernel carries the cut's code. Every cumulative build that holds the skip pays for
that path whenever the switch is off. Here the two tarballs (`-kshape` `39c3ab5f1`, this branch `f3072cbe8`) ran on the
idle rig on one day, the switch off in both: eight llama-server loads in the order `-kshape`, `-moeskip`, `-moeskip`,
`-kshape`, `-moeskip`, `-kshape`, `-kshape`, `-moeskip`. Each pair of consecutive loads holds one of each, and a steady
drift over the run cancels in the mean of the pairs. Each load ran T(k) at k = 4, 5 and 6 on the same agent text, with
the same method as above ([run/c556_alt.sh](run/c556_alt.sh), which runs `run/rig_tk.sh` `STEPS=tk ARMS=off` once per
load). Then one load of each build ran under the profiler at k = 4 and 5 (the first 340 tokens, group mode).

ms per pass of k real tokens, per load ([rig/c556_loads.tsv](rig/c556_loads.tsv),
[rig/c556_requests.tsv](rig/c556_requests.tsv)):

| k | 1: `-kshape` | 2: `-moeskip` | 3: `-moeskip` | 4: `-kshape` | 5: `-moeskip` | 6: `-kshape` | 7: `-kshape` | 8: `-moeskip` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 145.4 | 141.4 | 141.5 | 141.3 | 141.4 | 141.2 | 141.2 | 141.4 |
| 5 | 172.0 | 169.1 | 171.0 | 169.7 | 167.2 | 170.7 | 168.2 | 169.1 |
| 6 | 194.6 | 193.7 | 190.9 | 191.8 | 188.9 | 190.7 | 190.8 | 192.3 |

Per build, all eight loads ([rig/c556_tk.tsv](rig/c556_tk.tsv); the 95 % interval is Student's t over the four pairs),
and without load 1:

| k | `-kshape` mean ± sd (range), 4 loads | `-moeskip` mean ± sd (range), 4 loads | `-moeskip` − `-kshape` per pair | mean of the pairs (95 % interval) | without load 1: `-kshape` (3 loads) | `-moeskip` − `-kshape` |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 142.2 ± 2.1 (141.2–145.4) | 141.4 ± 0.06 (141.4–141.5) | −4.0, +0.2, +0.2, +0.2 | −0.8 (−4.2 to +2.5) | 141.21 ± 0.05 | **+0.21 (+0.15 %)** |
| 5 | 170.2 ± 1.6 (168.2–172.0) | 169.1 ± 1.5 (167.2–171.0) | −2.9, +1.3, −3.4, +0.9 | −1.0 (−5.0 to +2.9) | 169.5 ± 1.2 | −0.4 |
| 6 | 192.0 ± 1.8 (190.7–194.6) | 191.4 ± 2.0 (188.9–193.7) | −0.9, −0.9, −1.8, +1.5 | −0.5 (−2.8 to +1.8) | 191.1 ± 0.6 | +0.4 |

Where the time goes, one profiled load of each (the cards' GPU time per pass, 7.6 µs per span subtracted, C266's parts
and the expert launch groups; [rig/c556_parts.tsv](rig/c556_parts.tsv), [rig/c556_summaries/](rig/c556_summaries/),
[rig/c556_prof_requests.tsv](rig/c556_prof_requests.tsv)):

| Part or launch group | `-kshape`, k = 4 | `-moeskip`, k = 4 | change | `-kshape`, k = 5 | `-moeskip`, k = 5 | change |
|---|---:|---:|---:|---:|---:|---:|
| **routed experts** | 24.30 | 24.42 | **+0.12 (+0.5 %)** | 32.79 | 32.95 | **+0.16 (+0.5 %)** |
| of which gate and up (+ SwiGLU) | 13.48 | 13.54 | +0.06 | 18.66 | 18.76 | +0.10 |
| of which down | 8.47 | 8.53 | +0.06 | 11.52 | 11.57 | +0.05 |
| of which gate and up fused with the shared expert's | 2.34 | 2.35 | +0.01 | 2.61 | 2.62 | +0.01 |
| **the router's fused top-k chain** (`ffn_moe_probs`, in "other operations") | 0.88 | 0.92 | **+0.04 (+4 %)** | 0.81 | 0.84 | **+0.03 (+4 %)** |
| attention and recurrent projections | 43.44 | 43.25 | −0.19 | 53.56 | 53.52 | −0.04 |
| hyper-connections (BF16) | 16.20 | 16.16 | −0.04 | 24.27 | 24.28 | +0.01 |
| shared expert | 6.08 | 6.07 | −0.01 | 7.30 | 7.31 | +0.01 |
| router | 5.91 | 5.93 | +0.02 | 6.06 | 6.05 | −0.01 |
| Gated DeltaNet state and conv | 4.66 | 4.66 | 0 | 6.00 | 6.00 | 0 |
| attention (FLASH_ATTN_EXT) | 1.72 | 1.72 | 0 | 1.91 | 1.90 | −0.01 |
| other operations without weights | 11.80 | 11.92 | +0.12 | 10.02 | 10.09 | +0.07 |
| other weights, output layer | 2.05 | 2.07 | +0.02 | 1.37 | 1.36 | −0.01 |
| the cards' GPU time | 116.15 | 116.20 | +0.05 | 143.28 | 143.47 | +0.19 |

`kpass_parts.py` cuts each card's evaluations into the k = 4 and k = 5 requests where their GPU times separate best. On
the last two cards (9 of the 48 layers) the cut fell elsewhere, card 7's time counting mostly under k = 5 and card 8's
under k = 4, at the same place in both traces, so the comparison between the builds holds and the per-k sums are
approximate. The profiled requests ran in the same time on both builds (174.9 and 214.4 ms a pass under the profiler).

- **The first load ran slow at every k**, both builds alike: +4.0 ms (+2.9 %) at k = 4, +2.7 at 5 and +3.3 at 6 against
  the other seven loads' mean. It came after the rig had idled for an hour (the cards at 22–29 °C and the idle clock
  before it, 27–39 °C before the later loads); C120r's first load (`off`) ran 2.2–3.9 ms above its last (`off2`) at
  k = 1, 5 and 6 the same way. Each request at k = 4 here was the first of its load. On the earlier days, when k = 4
  ran as the fourth request of a load, it ran 0.8–1.8 ms faster (C350's 140.4 for `-kshape`, C120r's 139.6 and 139.9
  for this branch).
- **At k = 4 the switch-off path is 0.21 ms slower (+0.15 %).** Without the cold first load, the four `-moeskip` loads
  (141.38–141.51) all lie above the three `-kshape` loads (141.17–141.26), and the three later pairs give +0.20, +0.23
  and +0.25 ms. At this k the loads vary by only 0.05 ms (sd).
- **At k = 5 and 6 no cost is seen** (−1.0 and −0.5 ms over all eight loads, −0.4 and +0.4 without load 1): the loads
  vary by 1.3 and 1.5 ms (sd over the seven warm loads), far more than 0.2 ms. The 4–5 ms that C120r saw against the
  day before was its cold first load (`off`: 175.2 and 197.4 against `off2`'s 171.3 and 193.8) and the spread between
  days: today `-kshape` ran 169.5 and 191.1 ms over the three warm loads (168.0 and 191.4 on 2026-11-04) and `-moeskip`
  169.1 and 191.4.
- **The profile shows where the cost is:** every routed-expert launch group takes 0.5 % more GPU time at both k
  (+0.12 and +0.16 ms in all), and the router's fused top-k chain takes 4 % more (+0.03 to +0.04 ms). The other parts
  move both ways, by up to 0.19 ms.

**The class: a loss, small.** The switch-off path costs about 0.2 ms a pass (+0.15 % at k = 4), in the expert kernels'
−1 check and the fused router kernel. That is 20 times smaller than the gap that raised the question. The skip's
numbers above are measured against this build's own switch-off path, so against `-kshape` they shrink by the same
0.2 ms. The fix (Q1057) is to run the expert kernels and the fused router kernel in their `-kshape` form whenever the
graph holds no tail op (a template flag the host sets from where the ids come from), so that the switch-off path runs
`-kshape`'s machine code, checked by comparing the SASS.

## Files

| File | What it holds |
|---|---|
| [checks.tsv](checks.tsv) | every check: commit, where, run, what, result (the two first runs' failures included) |
| [run/cpu_checks.sh](run/cpu_checks.sh) | the CPU build and checks: `test-bytes-switches` (the op against the graph-level ops in micro-batches of 64, 4 and 1 tokens) and `test-llama-archs` over every architecture (graph-level logits written, the op's compared) |
| [run/build.sh](run/build.sh) | the rig tarball and the test extras |
| [run/gpu_checks.sh](run/gpu_checks.sh) | the GPU checks: test-backend-ops (skipped slots through every expert kernel, MMQ's first, the tail op, the fused router chain) and `test-llama-archs` on the card (the switch off against `-kshape`, the op against the graph-level ops), through the compute 6.1 PTX or natively on a P106 |
| [run/rig_tk.sh](run/rig_tk.sh) | the rig run: a native check on one P106, T(k) on real agent text for the switch off, p = 0.9 and p = 0.8 renormalized and p = 0.9 with `repeat=1` (Q661), the switch off loaded again (`off2`), and the profile of the routed experts at k = 1 and 5 |
| [run/rig_tables.py](run/rig_tables.py) | the rig tables: T(k) per arm and each arm's change against the mean of `off` and `off2`, the profile's parts |
| [rig/requests.tsv](rig/requests.tsv) | the rig run's requests: arm, k, prompt tokens, prompt time, micro-batches, ms per micro-batch (T(k)); `prof-*` rows are the profiled requests |
| [rig/tk.tsv](rig/tk.tsv) | T(k) per arm, the spread between the two `off` loads, each arm's change against their mean |
| [rig/parts.tsv](rig/parts.tsv), [rig/parts_k15.tsv](rig/parts_k15.tsv) | the profile's parts, ms per pass, `off` and `p090r` at k = 1 and 5 (long, and one column per arm and k) |
| [rig/summaries/](rig/summaries/) | `prof_summary.py`'s report per arm and k (per card, the first ops, the weight types, the top launch groups) |
| [../2026-11-04_kcurve_p106/cost/t_k_moeskip_p090r_real.txt](../2026-11-04_kcurve_p106/cost/t_k_moeskip_p090r_real.txt), [t_k_moeskip_p080r_real.txt](../2026-11-04_kcurve_p106/cost/t_k_moeskip_p080r_real.txt) | `LLAMA_SPEC_ADAPT_COST` files from the `p090r` and `p080r` passes |
| [run/c556_alt.sh](run/c556_alt.sh) | C556: eight loads alternating between two builds (`BIN_A`, `BIN_B`, `ORDER`), each `rig_tk.sh` `STEPS=tk ARMS=off` at k = 4–6; one table of every request, the cards' temperature and clock before each load |
| [run/c556_tables.py](run/c556_tables.py) | C556's tables: T(k) per load, per build mean, sd and range, the difference per pair of consecutive loads and its mean with a 95 % interval |
| [rig/c556_requests.tsv](rig/c556_requests.tsv) | C556's requests: load, build, then `rig_tk.sh`'s columns |
| [rig/c556_loads.tsv](rig/c556_loads.tsv), [rig/c556_tk.tsv](rig/c556_tk.tsv) | T(k) per load; per build and k the loads' n, mean, sd, min, max, the difference of the means, the pairs, their mean, sd and 95 % interval |
| [rig/c556_prof_requests.tsv](rig/c556_prof_requests.tsv), [rig/c556_parts.tsv](rig/c556_parts.tsv), [rig/c556_summaries/](rig/c556_summaries/) | C556's profiled requests (one load per build, k = 4 and 5, 340 tokens), their parts per pass, and `prof_summary.py`'s report per build and k |

## Commands

```bash
# CPU checks (a build machine; Docker)
WORK=<folder with src/ = the branch> JOBS=8 THREADS=16 [LOCK=<flock file>] [DOCKER_ARGS=...] bash run/cpu_checks.sh
# the rig tarball and the extras
WORK=<build folder> REPO=<this repository> [REMOTE=fork] JOBS=8 bash run/build.sh
# GPU checks: through the PTX on a newer GPU, or natively on one P106
BIN=<bin/ of the tarball> BASE_BIN=<bin/ of the -kshape tarball> OUT=<dir> MODE=ptx DEV=<card> bash run/gpu_checks.sh
# the rig (idle), detached; the agent text of 2026-11-04_kreal_p106 and its run/kreal_req.py beside this folder
BIN=<bin/> MODEL=<Swift Q2_0 first shard> TEXT=<agent text> OUT=<dir> setsid nohup bash run/rig_tk.sh > rig_tk.out 2>&1 < /dev/null &
# as run on 2026-11-05 (about 80 minutes): the steps tk and prof, the switch off loaded again at the end
BIN=<bin/> MODEL=<Swift Q2_0 first shard> TEXT=<agent text> OUT=out STEPS='tk prof' ARMS='off p090r p080r p090rr off2' \
  setsid nohup bash run/rig_tk.sh > rig_tk.out 2>&1 < /dev/null &
# then the tables (2026-11-04_kpass_profile's run/kpass_parts.py, the repository's scripts/prof_summary.py; --wall the
# unprofiled T(1) and T(5): for off the mean of off and off2)
python3 -I ../2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary <scripts/prof_summary.py> --event-us 7.6 --ks 1,5 \
  --wall 1=66.39,5=173.28 --out res off=out/prof-off.csv
python3 -I ../2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary <scripts/prof_summary.py> --event-us 7.6 --ks 1,5 \
  --wall 1=64.55,5=170.49 --out res-p090r p090r=out/prof-p090r.csv
(cat res/parts.tsv; tail -n +2 res-p090r/parts.tsv) > rig/parts.tsv
python3 -I run/rig_tables.py --requests out/requests.tsv --parts rig/parts.tsv --out rig
# C556 (about 65 minutes): eight loads, -kshape and this branch alternated, the switch off, k = 4-6
BIN_A=<bin/ of the -kshape tarball> BIN_B=<bin/ of this tarball> MODEL=<Swift Q2_0 first shard> TEXT=<agent text> \
  OUT=c556 setsid nohup bash run/c556_alt.sh > c556.out 2>&1 < /dev/null &
python3 -I run/c556_tables.py --requests c556/requests.tsv --out rig
# then one profiled load of each at k = 4 and 5 (about 12 minutes), and its parts (--wall: the unprofiled T(4) and
# T(5), the mean of each build's loads without load 1)
for b in kshape moeskip; do BIN=<bin/ of $b> MODEL=<Swift Q2_0 first shard> TEXT=<agent text> OUT=c556/prof-$b \
  STEPS=prof PROF_ARMS=off PROF_KS='4 5' bash run/rig_tk.sh; done
python3 -I ../2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary <scripts/prof_summary.py> --event-us 7.6 --ks 4,5 \
  --wall 4=141.21,5=169.53 --out c556/res-kshape kshape=c556/prof-kshape/prof-off.csv
python3 -I ../2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary <scripts/prof_summary.py> --event-us 7.6 --ks 4,5 \
  --wall 4=141.43,5=169.12 --out c556/res-moeskip moeskip=c556/prof-moeskip/prof-off.csv
(cat c556/res-kshape/parts.tsv; tail -n +2 c556/res-moeskip/parts.tsv) > rig/c556_parts.tsv
```

## Provenance and licence

Fork branch `pascal-p106-v3-b12076-moeskip` of llama.cpp (MIT). The random models are test-llama-archs' own. Swift-1.5
GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the agent text as in [2026-11-04_kreal_p106](../2026-11-04_kreal_p106/README.md).
