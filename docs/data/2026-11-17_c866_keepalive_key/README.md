# C866 (Q1562): the keep-alive's new-shape key without the cache's length (2026-11-17)

**Built, not yet run on a rig.** On v4.0 every agent turn's first passes at a new depth stop the cards' clock-holding
blocks, and those passes then take about twice as long. The new value `GGML_CUDA_KEEPALIVE_EAGER=3` stops a block only
when a pass could launch a kernel the card has not run, not merely because the cache grew. With the switch unset or at
`0`, `1` or `2`, the build behaves exactly as v4.0: every GPU kernel in it is v4.0's machine code.

- Fork branch [`pascal-p106-v5-keepalive-key`](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v5-keepalive-key),
  commit `087eb3b67`, one commit on `pascal-p106-v5` (`eb82916d3`, v4.0's code).
- Rig build `llama.cpp-20261117-087eb3b67-pascal-p106-v5-keepalive-key.tar.gz`, md5 `3d731bf8e31976a606af246889a6fda3`;
  `bin/llama-server` md5 `3defbfcff219a50dd48f79c7bb19f19a`. CUDA 12.9.2 image, `sm_61`, 0 AVX instructions in
  `llama-bench`, highest GLIBC symbol `GLIBC_2.38`.
- The rig check is the next row (rig 2, below).

## Why (R192, R191)

- `GGML_CUDA_KEEPALIVE_EAGER=1` (C811) ends a card's keep-alive block before an evaluation that runs kernel by kernel at
  shapes the card has not run. Reason: under CUDA's lazy loading, a kernel's first launch waits for every running
  kernel, and the block runs until its 1 s timeout (FN85).
- Its shape key holds every size, the cache's length too. So every new depth is new on every card.
  [R192](../2026-11-17_r192_first_request_graphs/README.md) counted 30 stops in a first request at 4,096 tokens, none
  in its repeat. Those three passes cost 235.65, 219.92 and 188.19 ms of GPU time on the nine cards.
- [R191](../2026-11-17_r191_keepalive_stop/README.md) ran the same pair with no stops at all. The same passes cost
  102.29, 100.67 and 70.26 ms. But the MTP draft's first evaluation after the prompt waited 1,003.87 ms on card 8: a
  first launch with the block running.

## The change

`GGML_CUDA_KEEPALIVE_EAGER=3`: as `1`, with a different key for the evaluation (`ggml_cuda_keepalive_shapes`,
`ggml/src/ggml-cuda/ggml-cuda.cu` lines 7688 to 7800 at `087eb3b67`).

- **Left out:** every size along the cache's length, by position, in the ops that carry it:
  - the attention: K's and V's length, the mask's width, `n_kv_max` (only whether it is set is kept);
  - the QSA indexer: its top-k, top-k mask, scores and range mask (the pools, which follow the cache);
  - the on-device KQ mask, and fills.
  Positions, not values: at the warm-up's 256 cells the cache's length equals a 256-wide head, and the indexer's 64 pools
  equal its 64 heads.
- **Kept:** every other op's sizes, and every token count, exactly as at `1`.
- **Mixed in instead of the length: what the length picks** on the card:
  - the attention (`ggml_cuda_flash_attn_ext_length_choice`, `fattn.cu` line 873): whether the compute-6.1 sparse
    kernels take the call (`ggml_cuda_flash_attn_ext_sparse_pascal_takes`, the same checks the kernel's own dispatch
    makes, moved into `fasp_check`, `fattn-sparse-pascal.cu` line 71), which dense kernel runs, whether the compute-6.1
    tile kernel takes a tile call, and the length's alignment to 256 (which the vector and one-token Pascal kernels need);
  - the indexer's top-k (`ggml_cuda_top_k_path_of`, `top-k.cu` line 364): bitonic, radix or CUB argsort, through
    `top_k_pick`, the one function the op itself now dispatches with.
- **How the key tells the draft's case apart.** The draft's one-token passes in the warm-up run at 256 cells, below the
  sparse threshold, so the dense kernels take them. Its first one-token pass at 4,096 cells is past the threshold
  (2,051 cells for one token), so the sparse kernels take it. That pick is in the key, so the key is new, and the block
  is stopped before it, as at `1`.
- **The policy:** `keepalive-policy.h` reads `cfg.eager != 2` where it read `cfg.eager < 2` (line 343). That is the
  same test for every earlier value, and at `3` it acts on new keys only. Work outside a graph (copies, sets) is
  handled as at `1`.
- **The value:** `2` already means "before every evaluation that runs kernel by kernel" (C811), so the new value is
  `3`.
- **Not in the key:** whether the dense attention splits the cache over several blocks, which decides whether its
  combine kernel runs. That needs the kernel's occupancy. Est. it changes only while the cache is a few tiles long,
  and R191 saw no main-model pass wait at 4,096 cells with no stop at all.

### Where the picks change on Flash-Next's file (P106-090, the v4 preset)

The QSA layers keep 4 x min(pools, 512) + 3 cells (2,051 from 512 pools). Pools are 4 cells each, padded up to a
multiple of 64 (`kpool_pad`). The build has no CUB DeviceTopK (CUDA 12.9's CCCL is older than 3.4.3).

| Pass | Pick | Changes at |
|---|---|---|
| one token (the reply's last pass, the draft's passes) | the sparse attention takes it | 2,051 cells (`_MIN_RATIO` 1.0 x 2,051 x 1) |
| one token | the indexer's top-k: bitonic, then CUB argsort, then radix | 512 pools (about 2,044 cells), 4,096 pools (about 16,380) |
| 4 and 5 tokens (verify, the prompt's last 4) | the indexer's top-k: bitonic, then radix | 1,024 pools (about 4,092 cells) |
| 4 tokens | the sparse attention takes it | 8,204 cells (4 x 2,051) |
| 5 tokens | the sparse attention takes it | 10,255 cells |
| 9 to 128 tokens (prompt micro-batches) | the sparse attention takes it | 2,051 cells |

At each of these, a card's first such pass is new and stops its block, once a server. Est. an agent session from 4,096
to 32,768 tokens crosses four of them. At `1`, every turn stops every card.

Two of these sit right at R192's depths:

- **At 4,096,** R192's prime ends at 4,032 tokens (1,024 padded pools, bitonic) and its first request reaches 4,096
  (1,088 pools, radix). So the first request's 4-token passes, its last pass and the draft's first pass all have new
  picks there.
- **At 8,192,** the 4-token passes cross the sparse threshold.

So the rig check adds a depth with no change in between: 6,144, in the same load.

## Machine code: off is v4.0

`cuobjdump -sass` of v4.0's `llama-server` (the release build, md5 `e7afd26d…`) and of this build's, every kernel
hashed with whitespace normalized: **8,797 kernels in each, all 8,797 identical; the two hash tables are the same file**
(md5 `27cde4d20bf08456e7ded9266d09230a`). The change is host code only ([sass_identity.txt](sass_identity.txt)).
C861 (a)'s `sass_compare.py` ran on the quant machine's CPU in a CUDA 12.9.2 container (`--cpu-shares=256`).

## The build (quant machine, CPU only)

`TESTS=1 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs" scripts/build_llamacpp_docker.sh
fork/pascal-p106-v5-keepalive-key` with `JOBS=8` and `--cpu-shares=256`, under the build lock, seeded from C861 (a)'s
tree and ccache. Two builds: a trial of the same commit pushed straight to the build tree (rc 0, the same
`llama-server` md5), then the rig build from the fork (rc 0). Both had 34 warning lines, as C861 (a)'s builds did;
none is in a changed line.

| File in the tarball | md5 |
|---|---|
| `bin/llama-server` | `3defbfcff219a50dd48f79c7bb19f19a` |
| `bin/test-backend-ops` | `62c877eac6d9eb6279feda6a28f45e9c` |
| `bin/test-cuda-keepalive` | `136873de01845b9d965171abfd559016` |
| `bin/test-llama-archs` | `b107e540015b37604056894279e28765` |
| `bin/llama-bench` | `c0311a4257a834a5f6c79b82854484f0` |
| `bin/llama-completion` | `f325bdd69717ad511998044add119173` |

`test-cuda-keepalive` (CPU, a thread per card) gained a parse check for `3` and `test_eager_no_depth`: at `3` the
controller stops a block before a new key only. It was built, not run: the quant machine runs no tests for now, and
the laptop runs none. It runs first in the rig check (seconds, CPU).

## The rig check (next row, R194; not run here)

On rig 2, with Flash-Next's v4 preset and `GGML_CUDA_KEEPALIVE_EAGER=3`:

- R191's deep-lane load ([run/c866.sh](run/c866.sh)), at 4,096, 6,144 and 8,192 tokens in one load: prime, first
  request, repeat at each depth, 256 tokens at temperature 0;
- R192's logging (graph debug, the graph-mode profile, the host profile, the step trace);
- tables: [run/tables.sh](run/tables.sh), R192's and R121's scripts unchanged.

**Pass:**

- at 6,144: 0 block stops in the first request, and its kernel-by-kernel passes within 200 ms of the repeat's in total;
- at 4,096 and 8,192: stops only before the passes whose picks changed (the table above);
- nowhere an evaluation about 1 s slower than the same pass in the repeat;
- the same reply in each pair.

**Est.:** at 6,144 the first request about 52 ms slower than its repeat in generation, the prompt's tail about 2 ms
(R191's numbers without the draft's wait). Against them, at 4,096 on v4.0 (R192): 291 ms and 117 ms.

## Files

- [run/c866.sh](run/c866.sh): the load (R191's `r191.sh` with `=3` and three depths).
- [run/tables.sh](run/tables.sh): the tables.
- [sass_identity.txt](sass_identity.txt): the machine-code comparison.
