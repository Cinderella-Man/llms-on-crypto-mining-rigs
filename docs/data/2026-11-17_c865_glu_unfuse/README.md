# C865: Ornith's routed gate, up and GLU run unfused on Pascal (`GGML_CUDA_MMID_GLU_UNFUSE`)

Registry rows [Q1559](../../../quants/experiments.md) (this switch) and [Q350](../../../quants/experiments.md) (whose
first arm it makes). The switch is described in [docs/switches.md](../../switches.md), "Fusions and numerics".

**What it changes.** At one token, an MoE layer's routed gate product, up product (both `MUL_MAT_ID`) and their GLU
run as one launch of the matrix-vector kernel (`mul_mat_vec_q` with a fused gate; for N2P and N3P the fork's Pascal
kernel). Upstream's guard against this fusion on Pascal, in `ggml_cuda_should_fuse_mul_mat_vec_q`, refuses only
compute capability 6.0 and below, so a P106 (6.1) fuses. Measured alone, the fused N2P product costs 5.53 ps a weight
against 4.18 for the two products apart (Q188). No earlier switch turned off this fusion alone:
`GGML_CUDA_GLU_ALIAS_FUSE=0` refuses only a fusion whose GLU output reuses the input's memory, and
`GGML_CUDA_DISABLE_FUSION=1` turns off every CUDA fusion.

`GGML_CUDA_MMID_GLU_UNFUSE`, read once at the first fusion decision, acts on Pascal cards above compute 6.0:

| Value | What runs |
|---|---|
| unset or `0` | v4.0's fusions |
| `1` | A routed gate/up/GLU (`MUL_MAT_ID`) is not fused: two `mul_mat_vec_q` launches, then the GLU. A shared expert and a dense FFN keep their fused launch. The one launch for a routed and a shared expert together (the shared-expert match) fuses the routed triple, so under `1` that match is not made at all: `graph_optimize` does not move the shared expert's three nodes next to the routed GLU, and the shared expert is fused on its own where the model builds it |
| `2` | Upstream's open [#32239](https://github.com/ggml-org/llama.cpp/pull/32239): the guard refuses every Pascal card (`cc < GGML_CUDA_CC_VOLTA`), so no fused `mul_mat_vec_q` runs: routed and shared gate/up/GLU, the shared-expert launch, a product with its bias or scale. There a Tesla P40 went from 30.18 to 33.75 tok/s (tg128, Mixtral 8x7B Q3_K_M) and from 50.74 to 50.08 (Llama 3.1 8B Q4_K_M); an MX230 (2 SMs) from 47.7 to 67.8 |

Fork branch `pascal-p106-v5-mmid-glu-unfuse`, commit `9550c62d6`, on `pascal-p106-v5` (`eb82916d3`, the v4.0 code); one
file, `ggml/src/ggml-cuda/ggml-cuda.cu`, host code only:

| Line (at `9550c62d6`) | Change |
|---|---|
| 212 | `ggml_cuda_mmid_glu_unfuse_mode()` reads the switch once, prints one INFO line when it is `1` or `2` and a warning for any other value but `0` |
| 230 | `ggml_cuda_mmid_glu_unfused_here()`: `1` and the current card below compute 7.0 |
| 2608 | `2`: upstream #32239's guard in `ggml_cuda_should_fuse_mul_mat_vec_q` |
| 2624 | `ggml_cuda_should_fuse_glu_mul_mat_vec_q()`: refuses a `MUL_MAT_ID` triple under `1`, otherwise `ggml_cuda_should_fuse_mul_mat_vec_q` as before |
| 2637 | `ggml_cuda_match_shared_expert` refuses under `1`; both its callers (`ggml_cuda_try_fuse` and `graph_optimize`) go through it |
| 6215, 6286, 6327 | the three gate/up/GLU sites that can see a `MUL_MAT_ID` (with weight scales, with biases, plain) call the new check |

With the switch unset, the mode is read first and is `0`, so every check returns what it returned before and the same
launches run. The routed products apart are the kernels llama.cpp already runs for an unfused `MUL_MAT_ID`; both read
the norm's Q8_1 row (`GGML_CUDA_NORM_Q8`) by lookup, so the second product finds it as the first did.

## Results (CPU only: the file's tensors and the build, no card)

1. **What it reaches on Ornith-1.5-122B Pascal N2P GGUF (`valid1`)** (`ornith_ffn_types.tsv`, from the GGUF header by
   `run/ffn_types.py`):

   | Weights | Type | Layers |
   |---|---|---|
   | routed gate and up (`ffn_gate_exps`, `ffn_up_exps`, 3072 × 1024 × 256) | N2P / N3P | 25 / 24; gate and up of one type in every layer |
   | shared gate and up (`ffn_gate_shexp`, `ffn_up_shexp`, 3072 × 1024) | Q5_0 | 49 |

   49 layers: the 48 of the model and the MTP layer, which does not run in the preset (no draft). Gate and up share a
   type in every layer, so v4.0 fuses the routed triple in all 48 decode layers. The shared-expert match needs the
   routed and the shared weights of one type, so on this file it never applies: v4.0 runs the routed triple and the
   shared triple as two fused launches a layer. `1` makes the routed one three launches and leaves the shared one
   fused; `2` makes both three launches.

2. **The build** (`TESTS=1`, `scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-mmid-glu-unfuse`, image
   `nvidia/cuda:12.9.2-devel-ubuntu24.04`, sm_61): rig tarball `llama.cpp-20261117-9550c62d6-pascal-p106-v5-mmid-glu-unfuse.tar.gz`,
   834,161,259 B, md5 `6576e7056bc4704c72960222a06785a1`; 0 AVX instructions in `llama-bench`, highest glibc symbol
   GLIBC_2.38, 34 warning lines (the v5 builds before it: 34). md5s: `llama-server` `dc9dcfa77d1c6dddab3af280d5dd4651`,
   `test-backend-ops` `681a98c5df63314bdffb5ec47deeca07`, `test-llama-archs` `a5524de7b0df35147d5846640f0f5ac0`,
   `test-cuda-keepalive` `090a5adc779906e101e646f444e6036c`, `llama-bench` `e78e71e97fee7584a02690c0554c0cd1`,
   `llama-completion` `64a5ac24100252f95c77a4616428f6cf`.

3. **The device code is v4.0's, byte for byte** (`device_code_md5.tsv`): the `.nv_fatbin` section (every GPU kernel
   of the binary) of `llama-server`, `test-backend-ops`, `llama-bench` and `llama-completion` has md5
   `a5a3a5cea79a07c6adc1f89b7cee7735` (61,270,464 B) in this build and in the v4.0 release build. Every arm, the switch
   unset, `1` or `2`, runs v4.0's kernels; only the choice of launches differs.

## On a P106 (the next row, not run here)

One rig with the Ornith file, its v4 preset (`release/ornith-v4/settings.env`), est. 1 h 50 min: about 10 min of
kernel cases, then seven loads of about 13 min each (a load of this file took 650 s in R162).
`run/c865.sh` drives it from a machine that runs only ssh and tar:

```bash
export RIG=<rig alias> WORK=<run folder on the rig> \
  BIN=<this build's tree on the rig>/bin/llama-server \
  MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz>
STEP=tbo bash docs/data/2026-11-17_c865_glu_unfuse/run/c865.sh     # every case passes in every arm
STEP=dry bash docs/data/2026-11-17_c865_glu_unfuse/run/c865.sh     # the three server lines
STEP=start bash docs/data/2026-11-17_c865_glu_unfuse/run/c865.sh
STEP=wait bash docs/data/2026-11-17_c865_glu_unfuse/run/c865.sh
STEP=tables bash docs/data/2026-11-17_c865_glu_unfuse/run/c865.sh
OUT=<folder> STEP=collect bash docs/data/2026-11-17_c865_glu_unfuse/run/c865.sh
```

- **Kernel cases** (`STEP=tbo`, card 0): `test-backend-ops -o MUL_MAT_VEC_FUSION,MUL_MAT_ID_SHARED,MUL_MAT_ID,ADD_RMS_NORM_MUL_Q8
  -p 'type(_a)?=(n2p|n3p|q5_0),'` (among them Ornith's expert shapes: rows of 3072, experts of width 1024, 8 experts
  a token, 1 to 4 tokens), then `-o GLU`, for the switch unset, `1` and `2`. Every case passes in every arm.
- **Decode at depth** (`STEP=start`, [bench/deep_lane](../../../bench/deep_lane/README.md)): arms `off` (unset), `u1`,
  `u2`; loads `warm:off`, then `off`, `u1`, `u2` twice, alternated; depths 0 and 8,192 tokens of recorded agent text;
  256 tokens at temperature 0 twice per depth. Each arm is classed against `off` at each depth: a gain when faster by
  more than the larger of the two arms' load-to-load spreads, a loss when slower by more, else negligible.
- **Replies**: the lane checks that every arm writes the same greedy reply at every depth. The separate GLU computes
  `silu(gate) * up` in its own kernel instead of the fused epilogue, so the replies may differ in the last bits; if they
  differ, the next step is the KL divergence of `u1` and `u2` against `off` (`llama-perplexity --kl-divergence`, as
  [FN93c](../2026-11-16_fn93c_ornith_taper_kld/README.md) ran it on this file), and the verdict waits for it.

## Files

| File | What |
|---|---|
| `ornith_ffn_types.tsv` | the FFN gate and up weights of the Ornith file by role: type, shape, count; and the layers whose gate and up share a type |
| `device_code_md5.tsv` | md5 of the `.nv_fatbin` section of four binaries, the v4.0 release build against this one |
| `run/ffn_types.py` | reads a GGUF header with the standard library only and prints `ornith_ffn_types.tsv` (`python3 -I ffn_types.py <ggml.h> <file.gguf>`; `ggml.h` names the type ids) |
| `run/c865.sh` | the P106 row: kernel cases, the deep lane, its tables and the collect |

Sources: the fork's code (MIT, as llama.cpp); upstream's #32239 read on GitHub (its diff is one line; the P40 numbers
are from its discussion). The Ornith file is Ornith-1.5-122B Pascal N2P GGUF (`valid1`); only its header was read.
