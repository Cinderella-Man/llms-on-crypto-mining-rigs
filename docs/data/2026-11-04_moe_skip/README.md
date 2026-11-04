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
on `-kshape` `d27845b67`). Not yet timed on a P106 (`run/rig_tk.sh`). Every check: [checks.tsv](checks.tsv).

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

## Files

| File | What it holds |
|---|---|
| [checks.tsv](checks.tsv) | every check: commit, where, run, what, result (the two first runs' failures included) |
| [run/cpu_checks.sh](run/cpu_checks.sh) | the CPU build and checks: `test-bytes-switches` (the op against the graph-level ops in micro-batches of 64, 4 and 1 tokens) and `test-llama-archs` over every architecture (graph-level logits written, the op's compared) |
| [run/build.sh](run/build.sh) | the rig tarball and the test extras |
| [run/gpu_checks.sh](run/gpu_checks.sh) | the GPU checks: test-backend-ops (skipped slots through every expert kernel, MMQ's first, the tail op, the fused router chain) and `test-llama-archs` on the card (the switch off against `-kshape`, the op against the graph-level ops), through the compute 6.1 PTX or natively on a P106 |
| [run/rig_tk.sh](run/rig_tk.sh) | the rig run: a native check on one P106, T(k) on real agent text for the switch off, p = 0.9 and p = 0.8 renormalized and p = 0.9 with `repeat=1` (Q661), and the profile of the routed experts at k = 1 and 5 |

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
```

## Provenance and licence

Fork branch `pascal-p106-v3-b12076-moeskip` of llama.cpp (MIT). The random models are test-llama-archs' own. Swift-1.5
GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the agent text as in [2026-11-04_kreal_p106](../2026-11-04_kreal_p106/README.md).
