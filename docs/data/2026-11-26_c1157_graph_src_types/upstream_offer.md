# CUDA: compare each source's type in the CUDA graph update check

The text of a pull request to [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), ready to post with
[upstream.diff](upstream.diff) (written against `master` at `a8ad690ca`, 2026-11-25; the newest tag then was b12756).
It is not posted: the owner decides.

---

**Title:** CUDA: compare each source's type in the CUDA graph update check

`ggml_cuda_graph_update_required` decides whether a captured CUDA graph can be replayed. For every node it keeps the
whole `ggml_tensor` and, for each source, the data pointer, `ne` and `nb`, and it replays the graph while they all
match the last evaluation. The sources' types are not kept.

Two types with the same element or block size give a tensor of the same shape the same strides: F16 and BF16 (2
bytes an element), Q4_0 and IQ4_NL (18 bytes a block of 32). So when a graph's source is replaced by a tensor of the
other type at the same address, with the same shape, and the node tensors sit at the same host addresses (the graph
key is the first node's address), every recorded property matches, and the graph captured for the first type runs
its kernels on the second type's bytes.

**Where it shows.** `test-backend-ops` builds each case in a fresh context, and consecutive cases often get the same
host and device addresses. A BF16 `MUL_MAT` case right after its F16 sibling (the same shapes) can then replay the
F16 case's graph, and fails with an NMSE near 0.5 (the card's F16 kernels read the BF16 weights). It is
intermittent, it depends on the heap layout, and it never happens with `GGML_CUDA_DISABLE_GRAPHS=1`. I first saw it
as `MUL_MAT(type_a=bf16,type_b=f32,m=1057,n=1,k=128,bs=[8,1],nr=[4,1],per=[0,2,1,3])` with ERR 0.4995 (limit 5e-4) on
an RTX 3090, and then reproduced it on demand: a set of 12 F16/BF16 `MUL_MAT` sibling cases, each checked
against the CPU, failed in 17 of 20 runs (always the same BF16 case right after its F16 sibling, always ERR
0.505182503) and in 0 of 20 with graphs off. A model's graphs keep their tensors' types from one
evaluation to the next, so inference does not hit it.

**The fix:** keep each source's type in `node_properties`, so such an evaluation counts as a change (the warm-up
starts again and the graph is captured anew). The array of ten 4-byte types follows the `size_t` arrays, so the
struct, compared with `memcmp`, gets no padding.

**Testing** (RTX 3090, CUDA 12.9, on my fork [Miner Llama](https://github.com/Cinderella-Man/miner-llama), whose graph
check is upstream's line for line; the diff applies to `master` as it is):
- A deterministic reproduction: evaluate `mul_mat` (a 512 × 128 matrix times a vector) until its graph is captured
  and replayed, then change only the matrix's type (`a->type`, the same buffer and strides, the data written again in
  the new type) and evaluate again. Without the fix every evaluation after the change is wrong in all four orders
  (F16→BF16 NMSE 4.29, BF16→F16 0.995, Q4_0→IQ4_NL 0.869, IQ4_NL→Q4_0 191; 20 of 32 evaluations right, three runs out
  of three); with the fix 32 of 32, three runs out of three; with `GGML_CUDA_DISABLE_GRAPHS=1` 32 of 32 either way.
- The 12-case sibling set above with the fix: 20 of 20 runs pass. The graph debug log shows the BF16 case still
  landing on its F16 sibling's addresses in 18 of them, now with one more warm-up reset instead of a replay.
- `test-backend-ops` `MUL_MAT`, `MUL_MAT_ID`, `FLASH_ATTN_EXT` and the fused-op cases: every case passes, as
  before. Every kernel compiles to the same machine code (host code only). Model inference gives the same tokens as
  before (a 256-token greedy reply, token for token, with CUDA graphs on).

The SYCL backend's `ggml_sycl_graph_update_required` keeps the same properties and probably has the same gap; this PR
does not touch it (I have no SYCL device to test on).

The standalone test I used (`tests/test-cuda-graph-src-types.cpp`, about 130 lines: evaluate a matrix-vector
product until its graph is captured, change only the matrix's type, write the data again in the new type, check
every evaluation against the host) can be added to this PR if wanted.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
