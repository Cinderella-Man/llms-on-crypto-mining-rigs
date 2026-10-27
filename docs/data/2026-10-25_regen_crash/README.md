# The pipelined-decoding crash of the release candidates: cause, fix and reproduction (2026-10-25)

The release blocker of 2026-10-25: on the rig, release candidates rc3 and rc4 crashed in pipelined decoding.
llama-server with `--pipe-depth` died on the third request with the same prompt (request 6 of
[bench/pipespec/check_server.sh](../../../bench/pipespec/check_server.sh), after "superseding context checkpoint"), and
llama-pipespec after 3 of the 34 jobs of rig_run.sh, both without a log line: a segfault at address 0x59 in
`ggml_gallocr_alloc_graph`. The other GPUs never showed it. This folder holds the fix and the two scripts that pinned it
down off the rig. Summary: [docs/engine_features.md](../../engine_features.md#the-pipelined-decoding-crash); the rig's
runs before and after the fix are in [2026-10-24_q88_persistent](../2026-10-24_q88_persistent/README.md) and
[2026-10-25_release_rig](../2026-10-25_release_rig/README.md).

| File | What |
|---|---|
| `run/leafcheck.diff` | the fix in `ggml/src/ggml-alloc.c`: `ggml_gallocr_needs_realloc` also checks every graph leaf against its plan entry (fork commit `2d53179aa`, in v2.0, which also adds `test_unread_leaf_needs_memory` to `tests/test-alloc.cpp`) |
| `run/alloc_test.sh` | `tests/test-alloc` on the CPU for the fixed tree and for its parent with only the new test case |
| `run/regen_gpu4.sh` | [bench/pipespec/rig_regenerate.sh](../../../bench/pipespec/rig_regenerate.sh) (the server under gdb, the same prompts three times each: T 0, then T 1 with two seeds) on one CUDA GPU with `--device CUDA0,CUDA0` (two llama devices on one card, so the scheduler runs with pipeline parallelism) and the packed uploads of compute 6.1 set by hand |

## The cause

1. With two or more devices the scheduler runs with pipeline parallelism and keeps four copies of every split input.
   `split_graph` adds the copies of all four as graph leaves; only the current copy is read by a node, the other three
   are leaves no node reads.
2. Packed input uploads (`LLAMA_INPUT_UPLOAD=2`) had become the default on compute 6.1 (Q150). An evaluation with nothing
   in flight packs the host inputs before allocation, which places the current copy of each input in the pack buffer:
   that copy already has memory.
3. When the allocator plans inside such an allocation, the current copy's leaf entry records "has memory of its own"
   (buffer −1). An explicit reserve never packs, so only re-plans inside an allocation do this; they happen when the
   graph changes shape (a backend sampler attached without a reserve, the device mask switching between a stale and an
   incremental update) or outgrows the previous plan.
4. The next graph that fits the plan, under another current copy, has a fresh copy tensor without memory at that leaf
   index. `ggml_gallocr_needs_realloc` checked nodes and node sources only, and no node reads that leaf, so the stale
   entry was used: `buffers[-1]`. The fault address 0x59 is 0x61 − 8, the malloc header in front of the rig's 10-pointer
   `buffers` array (nine cards and the CPU) minus one pointer.

Confirmed on the rig with rc4 and K2h-r16 on nine cards: request 6 (rewrite, T 1, second seed) segfaulted at leaf
`CUDA0#inp_tokens#0` (leaf 61, the scheduler's copy 0 of the token ids on card 1, read by no node), called from
`llama_context::decode`; with `LLAMA_INPUT_UPLOAD=0` all 6 requests were served. It appeared only on the P106 rig because
packed uploads were the default only on compute 6.1 and need two or more devices; the pipelined executor's own slots
run with one copy and cannot hit it.

**The fix** checks every leaf against its plan entry with the same test the nodes get. A stale entry forces a re-plan
(single buffer) or makes `alloc_graph` return false, and the scheduler reserves again, as for any other change. The cost
is one loop over the leaves per non-reused graph allocation; reused graphs (every decode token of the pipe slots, plain
decoding between KV steps) skip it, and the drafter's graphs are allocated once and not touched. v2.0 also ships packed
uploads off by default in plain decoding ([docs/engine_features.md](../../engine_features.md)).

## What the checks show

| Check | Result |
|---|---|
| `alloc_test.sh`, CPU | the fixed tree: 34 of 34 test-alloc cases pass; the parent with only the new case: segfault (exit 139) in `test_unread_leaf_needs_memory` |
| `regen_gpu4.sh`, one V100, granite-3.1-3b-a800m-instruct, `--device CUDA0,CUDA0`, `LLAMA_INPUT_UPLOAD=2` | the parent (`base`, `plain` runs) and uploads 0: 6 of 6 requests each, no crash (this combination does not reach the re-plan inside a packed allocation that the rig's request 5 reaches after its explicit reserve); the fixed build: 6 of 6 |
| check_server.sh's regenerate requests, CPU, granite | reference reproducible; depth 3 `cont` 14 of 14 identical; no `--pipe-depth` at T 0 6 of 6, the regenerate requests included |
| The fixed build on the rig (rc4 `2d53179aa`, nine cards) | llama-pipespec 33 of 33 jobs identical to plain decoding, K2h-r16 and UD-Q4_K_XL; llama-server 13 of 13 requests identical with `--pipe-depth 3`, regenerate requests included |

Earlier attempts to reproduce on the CPU, one V100 and two RTX 3090s with Flint UD-Q4_K_XL and granite all passed:
packed uploads were off there by default.

## How it was made

On 2026-10-25, on a CUDA host with Docker, with CPU and CUDA builds of the fork's rc4 parent `460160e93` and of
`2d53179aa`:

```bash
WORK=<dir> docs/data/2026-10-25_regen_crash/run/alloc_test.sh <tree with the fix> <parent tree with only the new test>
MODEL=<granite-3.1-3b-a800m-instruct-Q4_K_M.gguf> WORK=<dir> GPU=<n> \
    docs/data/2026-10-25_regen_crash/run/regen_gpu4.sh <build>/bin - base 2 'base plain'
```

The fifth argument is rig_regenerate.sh's `RUNS`; the fourth sets `LLAMA_INPUT_UPLOAD` (0 or 2). Run with a pre-fix
build to see that one GPU does not reproduce the crash, and with `2d53179aa` for the fixed path. The rig run used
`bench/pipespec/rig_regenerate.sh` with `RUNS='base upload0'` on rc4. granite-3.1-3b-a800m-instruct is IBM's
(Apache-2.0); the prompts are bench/pipespec's.
