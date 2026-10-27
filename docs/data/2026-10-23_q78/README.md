# Q78: S4-L0's and UD-Q4_K_XL's matrix products at 2–4 tokens per step (2026-10-23)

Registry row Q78 ([quants/experiments.md](../../../quants/experiments.md)) added 2–4-token Pascal kernels for
`MUL_MAT` and `MUL_MAT_ID` to the fork (branch `pascal-p106-v2-multicol`). Their correctness checks are in
[pascal_mmvq_integration.md](../../research/notes/pascal_mmvq_integration.md), section 14; one of them replays the two
models' own matrix products at 2, 3 and 4 tokens per step with `test-backend-ops`. This folder holds those replay
files. The register and SASS helpers used for the same work are in
[bench/mmvq_pascal/tools/](../../../bench/mmvq_pascal/tools/) (`regs.py`, `loops.py`, `loops2.py`, `classify.py`).

| File | What |
|---|---|
| `replay/s4l0-ub{2,3,4}-matmul.txt` | Miner S4-L0's 13 distinct matrix products at 2, 3 and 4 tokens per step, `test-export-graph-ops` format |
| `replay/ud-ub{2,3,4}-matmul.txt` | Unsloth UD-Q4_K_XL's 24 |
| `replay/*-matmul-quant.txt` | the same without the F16/BF16 router product, whose batched kernel has no compute_61 code (for runs through the compute_61 PTX) |

Made on 2026-10-23 with the fork's `test-export-graph-ops` (it reads only the GGUF metadata, so a sparse copy holding a
file's first 64 MB at its full length is enough). For each N in 2, 3, 4 the export
`test-export-graph-ops -m flint-S4L0.gguf -ub N -b N -c 4096 -o s4l0-ubN.txt` was reduced to its matrix products
(lines starting with operation 29, `GGML_OP_MUL_MAT`, or 30, `GGML_OP_MUL_MAT_ID`) minus the lines that the one-token
export also has; the same for UD-Q4_K_XL. The `-quant` files drop the lines whose
weight (column 25, src0 type) is F16 (1) or BF16 (30):

```bash
grep -v -E '^([^ ]+ ){24}(1|30) ' s4l0-ub2-matmul.txt > s4l0-ub2-matmul-quant.txt
GGML_CUDA_MMVQ_PASCAL=2 test-backend-ops -b CUDA0 --test-file s4l0-ub2-matmul.txt
``` `GGML_CUDA_MMVQ_PASCAL=2` forces the Pascal kernels on any GPU; with
`CUDA_FORCE_PTX_JIT=1` a newer GPU runs the compute_61 code path. The test logs are not published; the results are in
section 14 of the note.
