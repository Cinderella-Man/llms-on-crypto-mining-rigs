# Read-once kernels for `gmix-q2`'s IQ products: none missing (C827, Q1451, 2026-11-13)

What this is: a check of which tensors of Swift Qwen3.8-Flash-Next's `gmix-q2` file (the v3.1 quality pick,
[2026-11-04_fmt_mixes](../2026-11-04_fmt_mixes/README.md)) are IQ2_S and IQ3_XXS, and which kernel of the Miner Llama
line `pascal-p106-v4` serves each of its products at 1–8 tokens. It was meant to give those products a read-once path
at 2–8 tokens (each weight read and decoded once for all tokens, each token's sums the one-token kernel's), as C722 and
C729 did for Q2_0, Q3_K, Q4_K, Q5_K, Q6_K and IQ4_XS. The reason: on a P106 `gmix-q2` takes 18.06 ms a step more than
`pmix-q2-fast` (+16.75 %) and reads an 8,128-token prompt 17.43 % slower
([2026-11-12_r115_gmix_step](../2026-11-12_r115_gmix_step/README.md)). Registry row [Q1451](../../../quants/experiments.md).

## Result

**There is nothing to add. `gmix-q2` holds no IQ3_XXS tensor and no IQ2_S product outside the routed experts. Its
routed IQ products already have a read-once kernel at 2–8 tokens in `pascal-p106-v4`, on by default. No engine code
was written, and no switch or rig row was added.**

- **The types** ([type_counts.tsv](type_counts.tsv), [changed.tsv](changed.tsv)): 121 of 1,223 tensors differ from
  `pmix-q2-fast`.
  - **48 are routed gate/up projections** (`ffn_gate_exps`, `ffn_up_exps`, 2,560 → 640, 512 experts), all Q2_0 in
    `pmix-q2-fast`. In `gmix-q2` they are:
    - IQ2_XXS in 11 layers (1, 3, 6, 7, 9, 10, 11, 14, 27, 29, 30);
    - IQ2_S in 7 layers (12, 15, 24, 25, 26, 28, 31);
    - IQ2_XS in 4 layers (0, 2, 4, 5);
    - IQ1_M in 2 layers (8, 13).

    The donors' "IQ3_XXS" and "IQ2_XS" builds give each layer's experts their own type, so the picked groups bring
    these four types and no IQ3_XXS.
  - **73 are dense:** mostly IQ4_XS, Q4_K, Q5_K and Q6_K traded among each other. The largest moves are 23 IQ4_XS →
    Q4_K, 8 IQ4_XS → Q6_K, 8 Q4_K → IQ4_XS, 6 Q4_K → Q6_K and 5 Q2_0 → IQ4_XS. The only dense IQ grid tensors are the
    shared expert's gate and up in two layers, in IQ3_S (4 tensors, 2,560 → 640).
- **The kernels on `pascal-p106-v4`** (fork tip `2144cf760`, with `v4c2`'s switches: `GGML_CUDA_MMVQ_PASCAL_DMT`,
  `_Q5K_MT`, `_Q6K_MT`, `_Q3K_MT`, `_GLU_MT` on, `_RO8` and `_MT` off):

  | Products | 1 token | 2–4 tokens | 5–8 tokens |
  |---|---|---|---|
  | routed gate/up in IQ2_XXS, IQ2_XS, IQ2_S (44 tensors) | the one-token Pascal kernel | `mmvq-pascal-mtok-iq` (Q713): each distinct expert read and decoded once for all its tokens, bit-identical per token | the same |
  | routed gate/up in IQ1_M (4) | llama.cpp's `mul_mat_vec_q` (no one-token Pascal kernel) | `mmvq-pascal-mtok-iq` | the same |
  | `pmix-q2-fast`'s routed gate/up in Q2_0 (48) | the one-token Pascal kernel | `mmvq-pascal-mtok` (R7): the same scheme | the same |
  | dense IQ4_XS, Q4_K, Q5_K, Q6_K | the one-token kernels | read once (`_DMT`, `_Q5K_MT`, `_Q6K_MT`) | the k-column kernel (`_RO8` is off) |
  | the shared expert's fused gate/up in IQ3_S (2 layers) and Q6_K (1 layer) | the one-token kernels | per-token kernels (`_GLU_MT` takes Q2_0, Q3_K, Q4_K, Q5_K and IQ4_XS) | per-token kernels |

  `ggml_cuda_mul_mat_vec_q` (`mmvq.cu`) asks `ggml_cuda_mmvq_pascal_mtok` before `ggml_cuda_mmvq_pascal`. So the
  2–4-token `mul_mat_vec_q_pascal_mt` kernels behind `GGML_CUDA_MMVQ_PASCAL_MT=1` never receive a routed IQ product.
  Their types (IQ2_S, IQ3_XXS, N2P, N3P, Q8_0, Q5_0) are those of Ornith-1.5's dense tensors. In these two files only the
  shared expert's Q5_0 and Q8_0 down projections are of those types: 21 and 1 in `gmix-q2`, 14 and 2 in `pmix-q2-fast`.
  C755 is the row for those downs.
- **Why read-once cannot pay the 18 ms back:**
  - In a four-token verify pass each token picks 10 of 512 experts. If the picks were independent, the 40
    (token, expert) entries would hit est. 38.84 distinct experts, so reading each expert once saves est. 2.9 % of
    the routed reads. Real picks overlap more, but the kernel already does this.
  - The 24 changed layers' gate/up hold 2.224 bits a weight on average, against Q2_0's 2.25 (IQ2_XXS 2.0625, IQ2_XS
    2.3125, IQ2_S 2.5625, IQ1_M 1.75). So `gmix-q2` reads no more bytes there.
  - The extra time is spent decoding: grid lookups, signs and scales per 8 weights, against Q2_0's shifts and masks.
- **Est. share of the 18.06 ms:** FN5 priced `gmix-q2` at est. +5.91 ms a token on v3.0's one-token kernels, 4.68 of
  them (79 %) from the routed gate/up switches ([fmt_mixes](../2026-11-04_fmt_mixes/README.md)). If the measured step
  splits the same way, that is est. 14.3 ms for the routed gate/up decode and est. 3.8 ms for the dense type changes.
  R115's line also had `_Q6K_MT` off, and `gmix-q2` has 18 more Q6_K dense tensors, which then ran on the k-column
  kernel at four tokens. R125's per-part profile of the verify pass (Q1409) measures the split. Its answer names the
  kernel work that would pay the quality back: the grid types' decode in `mmvq-pascal-mtok-iq`, and C2's and C161's
  kernels.
- **For R125:** its second arm (the step pair with `GGML_CUDA_MMVQ_PASCAL_MT=1`) can only move the shared expert's
  Q5_0 and Q8_0 downs at 2–4 tokens in both files. It does not test the IQ products.

## Files

| File | What it holds |
|---|---|
| [type_counts.tsv](type_counts.tsv) | tensors of each ggml type in `pmix-q2-fast` and `gmix-q2` (shards 1 and 2; shard 3 is the shared n-gram table) |
| [changed.tsv](changed.tsv) | the 121 tensors whose type differs: name, `pmix-q2-fast`'s type, shape (ne0 × ne1 × ne2), `gmix-q2`'s type |
| [run/gguf_types.py](run/gguf_types.py) | lists every tensor of GGUF files (name, type, shape) from the header alone, with Python's standard library |
| [run/types.sh](run/types.sh) | the commands that made the two tables |

## Commands

On any CPU host that holds both files, or the header copies of their shards:

```bash
GMIX=<dir with gmix-q2's shards> PMIX=<dir with pmix-q2-fast's shards> OUT=<out dir> bash run/types.sh
```

The dispatch was read in the fork's `ggml/src/ggml-cuda/`, at `pascal-p106-v4` `2144cf760`:
- `mmvq.cu`, `ggml_cuda_mul_mat_vec_q`: the order k-column, `_Q5K_MT`, `_DMT`/`_Q6K_MT`, `_Q3K_MT`, `-mtok`, then the
  Pascal kernels;
- `mmvq-pascal-mtok.cu`, `mtok_iq_type` and `mtok_iq_run`;
- `mmvq-pascal.cu`, the `mmvqp_mt_type` comment and `mmvqp_mt_enabled`;
- `mmvq-pascal-glu.cu`: the GLU kernel's type list.

## Provenance and licence

The model files are this project's mixes of Swift's GSQ-RCO quants of Qwen3.8-Flash-Next
([2026-11-04_fmt_mixes](../2026-11-04_fmt_mixes/README.md)). The tables list only tensor names, types and shapes.
Scripts: this repository's licence.
