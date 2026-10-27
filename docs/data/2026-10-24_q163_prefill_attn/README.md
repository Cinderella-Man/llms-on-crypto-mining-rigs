# Q163: the compute-6.1 prompt attention kernel, its checks and its rig kit (2026-10-24/25)

The scripts behind Q163, the register-tiled FP32 attention kernel for prompt-size batches on compute 6.1
(`fattn-tile-pascal.cu`, fork branch `pascal-p106-v2-prefill-attn` `79613a54e`, in v2.0 and on by default;
`GGML_CUDA_FATTN_TILE_PASCAL=0` turns it off): the operator checks and model checks on a GPU running the compute_61
PTX, the SASS comparison behind the shared-memory-bytes-per-FMA figures, and the rig script. Design and results:
[pascal_attention_long_context.md](../../research/notes/pascal_attention_long_context.md) ("Prompt kernel (Q163)"),
registry row Q163 in [quants/experiments.md](../../../quants/experiments.md) and
[docs/engine_features.md](../../engine_features.md).

| File | What |
|---|---|
| `run/t1.sh` | operator checks: every `FLASH_ATTN_EXT` case with the kernel forced; the kernel's own 182 cases with it off and with the plain score layout at head 128 (`GGML_CUDA_FATTN_TILE_PASCAL_QK=1`); faults 1–3 (output × 1.05, sinks dropped, last key tile dropped) on those cases, which must fail |
| `run/t2.sh` | model checks over one 16,384-token chunk at `-ub 512`: Flint (head 128, sinks, 513-token window) with an F16 cache, kernel against stock, and with a Q8_0 cache, kernel off and on, against the F16 stock logits; Qwen3.6-35B-A3B (head 256) with an F16 cache |
| `run/sass61.sh` | the kernel compiled alone for sm_61: ptxas registers, spills and shared memory per instance, and its SASS |
| `run/sass_stock61.sh` | the stock TILE kernels compute 6.1 runs for prompts (head 128 and 256), compiled alone for sm_61: ptxas resources and the instruction mix of each loop body (FFMA, LDS, STS, BAR, F2F), next to the new kernel's |
| `run/rig_q163.sh` | the rig kit, kernel on against off in one binary: kernel speed on one card for both score layouts (step 0), K2h-r16 KLD over one 8k chunk (1), prompt speed at 8k/32k/64k for K2h-r16 (2) and Qwen3.6 Q4_0-noK (3), a Q8_0 cache's compute buffers at 64k (4) |

## What it shows

| Check | Result |
|---|---|
| Operator tests, V100 through the compute_61 PTX (`t1.sh`) | `FLASH_ATTN_EXT` 4172 of 4172 with the kernel forced, among them 168 new prompt cases (head 128/256, F16/Q8_0, causal and 513-key window masks); the 182 cases at these shapes pass with the kernel off and with the plain layout |
| Faults on those 182 cases | output × 1.05 fails 145, every case the kernel takes (the 37 passing ones are decode, ALiBi, softcap, BF16, unpadded-KV and 3-token F16 cases the stock kernels keep); dropped sinks fail 66, the kernel's sink cases except the 12 large-logit ones where a sink weighs nothing; a dropped last key tile fails 145 |
| Flint, 16k chunk, F16 cache, kernel against stock (`t2.sh`) | mean KLD 0.0522 ± 0.0044, same top token 93.9 % |
| Flint, Q8_0 cache against the F16 stock logits | stock kernels 0.0518 ± 0.0035 (93.4 %), the kernel reading Q8_0 directly 0.0525 ± 0.0040 (93.9 %): the kernel moves Flint's output as much as Flint's own rounding floor does |
| Qwen3.6-35B-A3B, 16k chunk, F16 cache | mean KLD 0.0021 ± 0.0001, same top token 98.3 % |
| sm_61 resources (`sass61.sh`) | 155–168 registers, 30,720 bytes of shared memory per block (3 blocks per SM), spill-free except 40 bytes in the split Q8_0 instance |
| Inner loops (`sass_stock61.sh`) | the stock TILE loop moves 2.5 bytes from shared memory per FMA, the new loops 1.5 |
| On the P106 (release candidates, [2026-10-24_q88_persistent](../2026-10-24_q88_persistent/README.md)) | the kernel's first-use line at the defaults, and logits that differ with it turned off |
| Prompt speed on the rig, K2h-r16, nine cards, llama-batched-bench `-b 2048 -ub 512` | 8,192 tokens 178.5 → 204.2 tok/s (+14 %), 32,768 tokens 120.6 → 138.7 (+15 %) on rc4 `bff28aba6`; whole-prompt batches on v2.0 (`-b` = prompt length) 699 → 766 at 8k, 447 → 461 at 32k ([2026-10-25_release_rig](../2026-10-25_release_rig/README.md)) |

`rig_q163.sh` as a whole did not run: its step 2 ran inside the release candidates' rig script (`rig_rc4.sh` step 4)
and in the release's prompt runs; the kernel-speed, Qwen3.6 and 64k steps have no rig result.

## How it was made

Checks on a CUDA host with Docker, the fork's test build from
[scripts/build_llamacpp_tests_docker.sh](../../../scripts/build_llamacpp_tests_docker.sh) (the host's GPU plus compute_61
PTX; on a non-Pascal GPU the kernel needs `GGML_CUDA_FATTN_TILE_PASCAL=2`), the night of 2026-10-24/25:

```bash
BUILD=<test build out dir> WORK=<dir> GPU=<n> docs/data/2026-10-24_q163_prefill_attn/run/t1.sh
BUILD=<test build out dir> WORK=<dir> GPU=<n> FLINT=<flint-imatU-Q4_0.gguf> QWEN=<Qwen3.6-35B-A3B-Q4_0-noK.gguf> \
    TEXT=<wikitext-2 wiki.test.raw> docs/data/2026-10-24_q163_prefill_attn/run/t2.sh
SRC=<fork source tree> WORK=<dir> docs/data/2026-10-24_q163_prefill_attn/run/sass61.sh
SRC=<fork source tree> WORK=<dir> docs/data/2026-10-24_q163_prefill_attn/run/sass_stock61.sh
```

The rig kit:

```bash
T=<tarball name> [STEPS='0 1 2 3 4'] [DEPTHS='8192 32768'] [QCARDS=0,1,2,3,4] docs/data/2026-10-24_q163_prefill_attn/run/rig_q163.sh
```

with the tarball of [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) unpacked in
`/opt/llama-builds` on the rig. Models: a Flint Q4_0 mix with an importance matrix (`imatU-Q4_0`) and K2h-r16 (Flint is
Apache-2.0), and a Qwen3.6-35B-A3B Q4_0 file (`Q4_0-noK`, Apache-2.0); the text is wikitext-2's test split
(CC BY-SA). The logs stay out; the numbers above are their summary lines.
