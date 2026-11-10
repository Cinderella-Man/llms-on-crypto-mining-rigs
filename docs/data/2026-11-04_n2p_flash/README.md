# Q928 / Q220: N2P and N3P for Qwen3.8-Flash-Next, with 128-weight variants for its 640-wide down projections (2026-11-04)

N2P and N3P are this project's own 2- and 3-bit formats for Pascal ([Q164](../../research/notes/pascal_lowbit_format.md),
[Q185–Q189](../2026-10-27_lowbit/README.md)): a handful of non-uniform int8 levels and a shifted copy held in registers,
decoded with one byte permute per 4 weights, so a P106 reads a weight in 3.8–4.7 ps at one token, Q4_0's speed. On
Ornith-1.5-122B a file with N2P/N3P experts was better than Unsloth's UD-Q2_K_XL on quality and 1.28× faster on the agent
check ([Q189](../2026-10-28_lowbit_rig/README.md)). This folder brings them to Swift-1.5 Qwen3.8-Flash-Next (hidden 2560,
512 routed experts of width 640, 10 used per token):

- **the types on the current engine:** the fork branch `pascal-p106-v3-lowbit` (ids 500, 501) ported onto the newest
  kernels, `pascal-p106-v3-b12076-kshape`, as `pascal-p106-v3-b12076-n2p`;
- **N2P-128 and N3P-128 (Q220):** the same formats in 128-weight blocks. Flash-Next's routed down projections have
  640-wide rows, which hold no whole 256-weight block, so before this only Q2_0 (2.25 bits) and IQ4_NL (4.5) fit them;
- **the multi-token expert kernel** for all four types: 2–8 tokens (a speculative verify) read each distinct expert once
  for every token that picked it;
- **two Flash-Next files:** the routed experts requantized from the Swift-1.5 Q8_0 with FN4's importance matrix (gate and
  up N2P; down N2P-128 in one file, N3P-128 in the other), every other tensor pmix-q2-fast's byte for byte.

## The formats

| Type | Id | Weights a block | Bytes | Bits a weight | Levels | Scales | Block alignment |
|---|---:|---:|---:|---:|---|---|---|
| N2P | 500 | 256 | 86 | 2.6875 | IQ2_K's 4 and a +5 copy | 4-bit signed + shift bit per 8 weights, FP16 per block | 2 B |
| N3P | 501 | 256 | 104 | 3.25 | IQ3_K's 8 and a +4 copy | 5-bit signed + shift bit per 32, FP16 per block | 8 B |
| **N2P-128** | 502 | 128 | 44 | 2.75 | as N2P | as N2P | 4 B |
| **N3P-128** | 503 | 128 | 54 | 3.375 | as N3P | as N3P | 2 B |

The 128-weight types are N2P and N3P with half the codes, scales and shift bits and their own FP16 scale; the quantizer
is the same code with the block size as a parameter (N2P and N3P come out byte for byte as before). Their 32-bit
fields come first, so every field is word-aligned relative to its block: N2P-128 is codes (8 words), scales (2 words),
then the shift bits and d sharing the last word; N3P-128 is the 2-bit plane (8 words), the 1-bit plane (4 words), then
N3P's first word (d, shift bits, bit 4 of the scales) and its 4-bit scale half-word. A 640-wide row is five blocks: 220
bytes in N2P-128, 270 in N3P-128 (Q2_0: 180). `llama-quantize` now falls back from N2P and N3P to their 128-weight types
where a row is a multiple of 128 wide (else to IQ4_NL as before), so `--tensor-type ffn_down_exps=n2p` also gives N2P-128.

## The code (fork branch `pascal-p106-v3-b12076-n2p`)

| Commit | What |
|---|---|
| `c1266a056` | the types (ids 500, 501), CPU reference, importance-matrix quantizer, CPU dot products (from `-lowbit` `d2c86e307`; Q4_1_64, retired on this base, dropped; upstream's `best_index_int8` is gone, so the quantizer has its own) |
| `4127220b1` | CUDA dequantize, get_rows, llama.cpp's matrix-vector kernel, the Pascal one-token and 2–4-token kernels (from `f40132341`, merged beside the grid types' and Q2_0's tile staging of this base) |
| `e51b90f02` | MMQ tile loaders (from `67181fcd9`; upstream's loaders take a gate argument now) |
| `db9b059ed` | test-backend-ops perf cases at Ornith's shapes (from `2942ef390`, `37aae03b2`, `5fd8ffeec`) |
| `0b8427657` | N2P-128 and N3P-128: CPU reference and quantizer, CPU dot products against Q8_0, gguf-py ids, the quantize fallback |
| `c722aa389` | N2P-128 and N3P-128 on CUDA: dequantize, get_rows, llama.cpp's kernel, the Pascal kernels (warp-tile staging of the 640-wide rows, as Q2_0's), MMQ |
| `efce6a24e` | the multi-token expert kernel for N2P, N3P, N2P-128 and N3P-128 (`mmvq-pascal-mtok-np.cuh`) |
| `a85c08643` | test-backend-ops cases at Flash-Next's shapes, 1–8 tokens and prompt batches, eval and perf |
| `1a0832375` | the CPU dequantizers read through const views (no warning; output unchanged) |

**One token** (`mmvq-pascal.cu`): N2P and N3P as before (N2P staged as a half-word type, N3P like Q4_K). N2P-128 loads
44-byte blocks as words; N3P-128 is a half-word type (54-byte blocks at 2-byte boundaries). At K = 640 a row is five
blocks, one lane per row, so the row segments are never whole 16-byte (N2P-128) or 4-byte (N3P-128) units: like Q2_0's
down projection (Q825), each warp stages its 32 rows at once as a tile with coalesced 128-bit loads and every lane reads
its block from shared memory. With a fused gate N2P-128 loads directly.

**2–8 tokens of experts** (`mmvq-pascal-mtok-np.cuh`): the grid types' multi-token kernel (Q713) generalized to the N
types and to 128-weight blocks: each distinct expert of the step is read once, each 32 weights decoded once into int8
words and multiplied by every token that picked it (8 dp4a a token), with the same integer sums, the same FP32
operations in the same order and the same lane reduction as the one-token kernel, so every token's result is
bit-identical to it. Row segments are staged with the widest loads the rows allow; N3P-128's 270-byte rows (no gate)
as warp tiles. `GGML_CUDA_MMVQ_PASCAL_MTOK=0` runs the earlier kernels.
Why: on Ornith's shapes Q188 timed the earlier 2- and 4-token Pascal kernel, which reads every (token, slot) entry's
expert on its own, at 3.5× and 8.1× the one-token time for N2P's expert gate/up (367 and 853 µs against 105) and 5.6×
and 14.9× for the down projection (660 and 1,744 against 117), behind llama.cpp's kernel ([Q302](../../../quants/experiments.md)).
The multi-token kernel's cost follows the distinct experts a step touches.

**Prompt batches** (MMQ): the N2P/N3P tile loaders take the block type; a 256-weight MMQ iteration spans two 128-weight
blocks. Both load into the Q8_0 tile layout as N2P and N3P do.

## Status

**Both files score below Swift IQ2_XS on all three texts, and the build passes its GPU checks at every shape the files
hold; one test case at another row count fails on the `-n2p` build and passes on the release build, whose last commit
fixes its cause (Q1181).** On held-out agent text the KL divergence against the Q8_0
reference is 0.1931 ± 0.0044 for file 1 (down N2P-128) and 0.1727 ± 0.0040 for file 2 (down N3P-128), against
0.2792 ± 0.0054 for pmix-q2-fast, 0.2851 ± 0.0056 for Swift Q2_0 and 0.2248 ± 0.0046 for Swift IQ2_XS
([KL divergence](#kl-divergence), [GPU checks](#gpu-checks)). Both files fit nine 6 GB cards at 64k on whole layers
(file 1 to `-ub 256`, file 2 to `-ub 128`). Still to come: the native check and the timing on a P106 (C394).

- **File 1** (2026-11-05): `gguf_compose verify` passes (1,224 tensors in 3 shards, the 144 routed-expert tensors
  identical to llama-quantize's, the other 1,080 to pmix-q2-fast's). The routed experts take 40,894,464,000 bytes against
  pmix-q2-fast's 33,973,862,400 (+6,920,601,600, +6.45 GiB). Shard 1 is 12,926,129,760 bytes (md5
  `b2ddb62733c82a30d211d65d9ab663fe`), shard 2 31,749,037,696 bytes (md5 `c8712cd348e6d08a5ed336e945d853f8`), and shard
  3 is the unchanged n-gram table (Swift Q2_0's, 28,800,138,432 bytes, md5 `98ce140e…`). In the headers, every layer's
  `ffn_gate_exps` and `ffn_up_exps` are N2P (type 500) and `ffn_down_exps` is N2P-128 (502).
- **Fit of file 1** ([fit/fit-n2p-dn2p128.tsv](fit/fit-n2p-dn2p128.tsv), beside [pmix-q2-fast's](fit/fit-pmix-q2-fast.tsv)).
  At 64k with an f16 cache on whole layers, `-ts 5,6,5,6,5,5,6,5,6` (pmix-q2-fast's split) fits up to `-ub 256`, with
  242 MiB free on the fullest card (474 at `-ub 64`; pmix-q2-fast: 1,067 and 1,299). With the DFlash2 drafter it fits up
  to `-ub 128` (238 MiB), with the MTP draft also up to `-ub 128` (217 MiB). `-ub 512` misses on every line.
- **File 2** (2026-11-05 16:32): composed onto file 1, so `verify` counts the 48 down projections identical to
  llama-quantize's N3P-128 and the other 1,176 tensors to file 1's. Shard 1 is 13,843,633,760 bytes (md5
  `5fded2e6f456f2aa4bd2ddce9eb943f2`), shard 2 33,977,261,696 bytes (md5 `df4809b9fe535bf7858dbf96aa116cbc`), shard 3
  the same n-gram table: 3,145,728,000 bytes (2.93 GiB) more than file 1.
- **Fit of file 2** ([fit/fit-n2p-dn3p128.tsv](fit/fit-n2p-dn3p128.tsv)). At 64k with an f16 cache on whole layers, the
  same `-ts 5,6,5,6,5,5,6,5,6` fits up to `-ub 128`, with 31 MiB free on the fullest card (99 at `-ub 64`). With the
  DFlash2 drafter or the MTP draft no whole-layer split fits at 64k (−69 and −90 MiB at `-ub 64`); with expert tensors
  placed one by one (the `ot` lines) the drafter fits, 395 MiB free at `-ub 64`.

- **Build:** the rig tarball (sm_61 code and compute_61 PTX, `TESTS=1`, no AVX, GLIBC 2.38) of `1a0832375` compiles
  cleanly. The files were made with the CPU tools of `0b8427657` (`llama-quantize`), whose quantizer is the final
  branch's.
- **CPU check** (`test-quantize-fns`, every type, a build without AVX like the rig's): all four pass
  ([cpu_check.tsv](cpu_check.tsv)). N2P-128 and N3P-128 quantize random data a little more closely than N2P and N3P
  (0.004376 against 0.004383, 0.002464 against 0.002475: one FP16 scale per 128 weights instead of 256), and their
  reference and fast quantizers agree exactly. test-backend-ops compares a GPU with the CPU backend, so its runs are
  the GPU checks below.

## KL divergence

Against the Swift-1.5 Q8_0 reference's logits on the three texts of
[2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md): 12 chunks of 8,192 tokens of held-out
agent sessions, 80 chunks of 512 tokens of wikitext-2 and 80 of the agent check's Exercism code. Scored with that
folder's `kld.sh` (`-ub 512`, flash attention, the layers placed by `-fit on`) and this branch's compute 8.6
llama-perplexity on two RTX 3080s and two RTX 3090s, each file read in place ([run/kld.sh](run/kld.sh)). The table
gives the mean ± its error and the share of tokens whose top choice is the reference's; every column is in
[kld.tsv](kld.tsv). The other files' rows are from [flash_next_quality](../2026-11-03_flash_next_quality/kld.tsv) and
[pascal_mix](../2026-11-04_pascal_mix/kld.tsv), scored the same way with the v3 base's llama-perplexity on seven and on
five cards.

| File | Agent | Wiki | Code |
|---|---|---|---|
| Swift Q2_0 (the maker's) | 0.2851 ± 0.0056, 90.71 % | 0.4283 ± 0.0059, 77.94 % | 0.1124 ± 0.0040, 95.98 % |
| pmix-q2-fast | 0.2792 ± 0.0054, 90.70 % | 0.4264 ± 0.0059, 77.71 % | 0.1080 ± 0.0038, 96.12 % |
| pmix-q2-fast, this run's build and cards (control) | 0.2773 ± 0.0054, 90.69 % | 0.4264 ± 0.0059, 77.66 % | 0.1075 ± 0.0038, 96.08 % |
| Swift IQ2_XS (the maker's) | 0.2248 ± 0.0046, 91.53 % | 0.3512 ± 0.0050, 79.57 % | 0.0750 ± 0.0028, 96.83 % |
| **N2P-N2P128** (file 1) | **0.1931 ± 0.0044**, 92.40 % | **0.2555 ± 0.0040**, 83.00 % | **0.0521 ± 0.0021**, 97.45 % |
| Swift IQ3_XXS (the maker's) | 0.1755 ± 0.0038, 92.32 % | 0.2477 ± 0.0039, 83.33 % | 0.0544 ± 0.0023, 97.48 % |
| **N2P-N3P128** (file 2) | **0.1727 ± 0.0040**, 92.79 % | **0.2252 ± 0.0037**, 84.16 % | **0.0460 ± 0.0019**, 97.53 % |

- **Both files are a gain against pmix-q2-fast on every text** (below it by more than its error, the classes of
  [2026-11-04_fmt_mixes](../2026-11-04_fmt_mixes/README.md)). File 1 is 0.0862 lower on agent text, 0.1708 on wiki and
  0.0559 on code; file 2 is 0.1066, 0.2012 and 0.0620 lower. Each file differs from pmix-q2-fast only in its routed
  experts: N2P from the Q8_0 with FN4's importance matrix in place of the makers' Q2_0.
- **Both are below Swift IQ2_XS.** File 1 by 0.0317 on agent text (0.1931 against 0.2248), 0.0957 on wiki and 0.0230 on
  code; file 2 by 0.0521, 0.1260 and 0.0290.
- **File 2 is level with Swift IQ3_XXS on agent text** (0.1727 against 0.1755, a difference of 0.0029, inside either
  error) and below it on wiki (0.2252 against 0.2477) and on code (0.0460 against 0.0544). File 1 is above IQ3_XXS by
  0.0176 on agent text and 0.0079 on wiki, and level with it on code (0.0521 against 0.0544).
- **The third bit in the down projections is worth 0.0204 on agent text** (file 2's 0.1727 against file 1's 0.1931,
  10.6 % lower), 0.0304 on wiki and 0.0061 on code, for 2.93 GiB.
- **The control holds, so the rows compare.** pmix-q2-fast scored with this run's build and cards gives 0.426375 ±
  0.005853 on wiki (0.426376 ± 0.005852 in its own run), 0.277337 ± 0.005369 on agent text (0.279228 ± 0.005378) and
  0.107518 ± 0.003823 on code (0.107982 ± 0.003843): the build and the card set move a score by 0.0019 at most, a
  third of its error.
- **Each file ran whole on the cards** ([mem_peak.tsv](mem_peak.tsv)). llama-perplexity's log does not print where
  `-fit on` put the layers, so the cards' memory was sampled every 5 s. The peaks add up to 45,692 MiB for file 1
  (its two weight shards: 42,606 MiB), 48,690 MiB for file 2 (45,605 MiB) and 39,088 MiB for the control (36,005 MiB):
  the same 3.08 GiB of context and work buffers above the weights in all three.
- These scores say nothing about speed: the timing on a P106 is C394.

## GPU checks

test-backend-ops runs each case on the card and on the CPU backend and compares the two results (the limit is a
normalized squared error of 0.0005). Two checks ran, each on one RTX 3080 ([run/gpu_checks.sh](run/gpu_checks.sh)):
the rig build through its compute 6.1 PTX, with the fork's Pascal kernels forced on at a P106's launch shapes (1–8
tokens, one run per switch setting), and a compute 8.6 build natively (llama.cpp's own kernels, and MMQ for the prompt
batches of 9, 33 and 128 tokens). Every run is in [gpu_checks.tsv](gpu_checks.tsv):

| Check | Run | Passed | Cases | Failing |
|---|---|---:|---:|---|
| PTX | `np` (the four types, defaults) | 717 | 718 | N3P-128 at 2,561 rows, one token |
| PTX | `np-mtok-off` (the earlier multi-token kernels) | 717 | 718 | the same case |
| PTX | `np-cap1` (groups of one) | 717 | 718 | the same case |
| PTX | `np-rb1` (one row block) | 717 | 718 | the same case |
| PTX | `np-notile` (no multi-token tiles) | 717 | 718 | the same case |
| PTX | `np-skip` (llama.cpp's matrix-vector kernel) | 716 | 718 | the same case, N2P-128 and N3P-128 |
| PTX | `reg-half` (Q2_0, Q6_K, the grid types) | 2,029 | 2,029 | |
| PTX | `reg-rows` (GET_ROWS of every type) | 248 | 248 | |
| native | `np` (prompt batches included) | 704 | 706 | the same case, N2P-128 and N3P-128 |
| native | `reg-rows` | 248 | 248 | |

- **Every failure is one test case, in its two 128-weight types:**
  `MUL_MAT_ID(type_a=n2p_128|n3p_128,type_b=f32,n_mats=32,n_used=10,b=0,m=2561,n=1,k=640)`, one token's product with 10
  of 32 experts whose matrices have 2,561 rows of 640 weights. [gpu_check_failures.tsv](gpu_check_failures.tsv) lists
  it per run with its error.
- **It fails with the fork's kernels and with llama.cpp's, through the PTX and natively, and its error moves from run
  to run.** With the fork's kernels on, N3P-128's error is 0.000762 (`np`), 0.000563 (`np-mtok-off`), 0.000805
  (`np-cap1`), 0.000697 (`np-rb1`) and 1.000003 (`np-notile`); those switches set the multi-token kernel, and the case
  has one token. N2P-128 passes it in those five runs. On llama.cpp's kernel the error is 0.000518 for N2P-128 and
  1.000000 for N3P-128 through the PTX (`np-skip`), and 0.989390 and 0.001466 natively. The cause is a fault in
  llama.cpp's `mul_mat_vec_q` that the release build's last commit fixes: on that build the case passes
  ([below](#the-check-set-for-a-p106); Q1181 in the [registry](../../../quants/experiments.md)).
- **Everything around it passes:** the same matrices without expert ids (`MUL_MAT` at 2,561 rows), with expert ids at
  2 to 8 tokens, and every case at 2,560 rows at every token count, in every run.
- **No tensor in either file has 2,561 rows.** The routed down projections have 2,560 rows of 640 weights and the gate
  and up projections 640 rows of 2,560. In the types the files use (N2P for gate and up, N2P-128 and N3P-128 for
  down), the cases at those shapes pass at 1 to 8 tokens through the PTX and at 1 to 8, 9, 33 and 128 tokens natively.
- **The KL divergence above does not run the failing case's path.** llama-perplexity scores in micro-batches of 512
  tokens, and on these cards a routed-expert product of more than 8 tokens goes to MMQ, not to the matrix-vector
  kernels of the failing case. MMQ's 40 prompt-batch cases of the four types (9, 33 and 128 tokens, the files' shapes
  among them) all pass natively, and pmix-q2-fast scored with the same build reproduces its earlier score.
- The check on a P106 itself, with MMQ's dp4a path, is part of C394. It replays a recorded set
  ([below](#the-check-set-for-a-p106)).

## The check set for a P106

test-backend-ops draws each case's tensors, quantizes them and computes the reference on the host's CPU before the
card runs anything. On the rig's two Celeron cores that is est. an hour for the three runs C394 needs, so they are
recorded once on a build machine and the rig replays them: the card computes, the stored references judge it
([replay checks](../2026-11-04_replay_checks/README.md)). The suite is `n2p` in
[scripts/record_checks.sh](../../../scripts/record_checks.sh), and [run/rig_tk.sh](run/rig_tk.sh) has a `replay` step
for it beside `check`, which stays for a machine with a real CPU.

- **The build is the v3.0 release build** (`1551f5fe1`). `1a0832375`, the `-n2p` branch's head, is an ancestor of it,
  so a rig with the release folder already holds the N types, and the timing runs on the build that ships. The set
  was recorded with that build's own test-backend-ops (md5 `2c7f4054190d271eae3696792d487dd2`; it is part of the rig
  tarball, and in the GitHub release it is the `tests` archive), so the references are what the rig's CPU would compute.
- **The set:** 3,655 cases, 37,416,765,440 bytes as one tar (md5 `0ce5cd1e19a628fec02051a8ac9e035d`), recorded in 35
  minutes on a 12-core CPU with 8 workers at low priority: 16 minutes for the N types' 812 cases (9.3 GB), 19 for the
  2,567 regression cases of Q2_0, Q6_K and the grid types (25.7 GB), 12 seconds for 276 more GET_ROWS cases. The
  largest product case is 473 MB (Q2_0, 512 experts), the largest of all 933 MB (GET_ROWS of F32). It holds the cases
  of all nine native runs of [run/gpu_checks.sh](run/gpu_checks.sh); [replay_runs.tsv](replay_runs.tsv) is its run list (C394 replays `np`,
  `np-mmq` and `reg-half`) and [replay_manifest.tsv.gz](replay_manifest.tsv.gz) names every case with the md5 of its
  tensors and of its reference.
- **Every run is expected to pass, the one-token case at 2,561 rows included.** That case failed on the `-n2p`
  build ([GPU checks](#gpu-checks), Q1181) and does not fail on the release build. Run live through the PTX (fresh
  draws and a fresh CPU reference each time), it passes 35 times of 35 on `1551f5fe1`: 20 of 20 with the fork's
  Pascal kernels on and 15 of 15 with them skipped, on an RTX 3090 and an RTX 3080. On `1a0832375`, on the same RTX
  3080 the same day, it fails 13 times of 20: N3P-128 5 of 5 with the Pascal kernels on (errors 0.999998177,
  0.964768327, 0.001410587, 0.000787041 and 0.001027811, N2P-128 passing 5 of 5) and 8 of 10 results with them
  skipped, one of them a NaN at output index 2563 ([case_m2561_live.tsv](case_m2561_live.tsv)).
- **The cause was already fixed in the release build's last commit** (`1551f5fe1`): llama.cpp's `mul_mat_vec_q`
  bounded its rows by the output's stride, which for an expert product of one token is the row count times the
  experts used. With a row count that is not a multiple of the block's rows, each expert's last block wrote rows past
  its own, computed from the weights behind its matrix, over the next expert's first rows, in a race with that
  expert's own block. The kernel now bounds its rows by the matrix's row count, and the commit's message names this
  case. So a failure of it on a P106 would be new, and the replay reports it as wrong.
- **A run list can still name cases that are known to fail** (a seventh field, a regex on the case string):
  [scripts/replay_checks.sh](../../../scripts/replay_checks.sh) then counts their failures apart, and the run is as
  expected when nothing else fails. No suite uses it today.
- **Checked through the compute 6.1 PTX before it goes to a rig** (one RTX 3080, the release rig build, the fork's
  Pascal kernels forced on at a P106's launch shapes; the set's `runs-ptx.tsv`, 1–8 tokens). Every run is in
  [replay_ptx.tsv](replay_ptx.tsv):

| Run | Switches | Passed | Cases |
|---|---|---:|---:|
| `np` | defaults | 760 | 760 |
| `np-mtok-off` | the earlier multi-token kernels | 760 | 760 |
| `np-cap1` | groups of one | 760 | 760 |
| `np-rb1` | one row block | 760 | 760 |
| `np-notile` | no multi-token tiles | 760 | 760 |
| `np-skip` | llama.cpp's matrix-vector kernel | 760 | 760 |
| `reg-half` | Q2_0, Q6_K, the grid types | 2,285 | 2,285 |
| `reg-rows` | GET_ROWS of every type | 276 | 276 |

  All eight runs are as expected: 3,289 of the set's 3,655 cases were replayed and each passes in every run that
  holds it (7,121 results), with no Xid on the card. The release build's test file holds more cases than the `-n2p`
  branch's, so the counts are above those of the [GPU checks](#gpu-checks) (760 against 718 in `np`).

- **What the PTX check cannot see:** the 334 products of 9 or more tokens (`np-mmq`'s 40 cases among them). MMQ
  takes the host card's tile configuration through the PTX, not Pascal's, so those records are replayed for the first
  time on the P106. Each replayed case checks the md5 of its tensors and of its reference first. The other 32 cases
  are GET_ROWS of four types the CUDA backend does not support (F8_E4M3, NVFP4, TQ1_0, TQ2_0): a card skips them.

## Files

| File | What |
|---|---|
| [run/build.sh](run/build.sh) | the builds on the quant machine: the rig tarball (sm_61 + compute_61 PTX), the CPU tools and test-quantize-fns, a compute 8.6 llama-perplexity and test-backend-ops for the KLD and the native MMQ check |
| [run/gpu_checks.sh](run/gpu_checks.sh) | test-backend-ops runs per switch setting (MODE=ptx on another card, MODE=native on a P106) |
| [run/checks_table.sh](run/checks_table.sh) | the checks' output folders to the two tables below |
| [gpu_checks.tsv](gpu_checks.tsv) | per check and run: the switch, cases passed, cases run, cases the backend does not support, failures, exit status |
| [gpu_check_failures.tsv](gpu_check_failures.tsv) | every failing case by name, per check and run, with its error and the limit |
| [run/kld.sh](run/kld.sh) | the KL divergence of the files one at a time, the cards' memory sampled beside each |
| [kld.tsv](kld.tsv) | per file and text: mean KLD and its error, 99th and 99.9th percentile, same top token and its error, RMS change of the top token's probability, perplexity of the file and of the reference (`n2p_dn2p128` is file 1, `n2p_dn3p128` file 2, `pmix-q2-fast-n2pbuild` the control) |
| [mem_peak.tsv](mem_peak.tsv) | per file and card: the peak memory in use during its three texts, MiB |
| [run/quantize.sh](run/quantize.sh) | the two files: llama-quantize of the experts, then gguf_compose onto pmix-q2-fast, `verify`ed |
| [run/fit.sh](run/fit.sh) | the files' fit on nine 6 GB cards from their headers ([bench/file_choice/fit.py](../../../bench/file_choice/fit.py)), each once built |
| [fit/](fit/) | fit.py's rows per file (every context, cache type, micro-batch and draft; whole layers and with experts moved) |
| [run/rig_tk.sh](run/rig_tk.sh) | on a rig: the native check (MMQ included) by replay of the recorded set or live, per-kernel perf at Flash-Next's expert shapes, T(k) of the files |
| [replay_runs.tsv](replay_runs.tsv) | the recorded set's run list: run, switches, test-backend-ops' `-o` and `-p`, what passing means, a wrapper (none here) |
| [replay_manifest.tsv.gz](replay_manifest.tsv.gz) | the recorded set's cases: id, case, bytes and md5 of the tensors, bytes and md5 of the CPU reference |
| [replay_ptx.tsv](replay_ptx.tsv) | the set replayed through the compute 6.1 PTX: per run the switches, cases passed, FAIL lines, exit status, verdict |
| [case_m2561_live.tsv](case_m2561_live.tsv) | the one-token expert case at 2,561 rows, run live through the PTX on the release build and on the `-n2p` build: card, build, the fork's Pascal kernels (on, or skipped for the four types), repetition, case, verdict, error |

## The two files

Swift-1.5 Qwen3.8-Flash-Next (`qwen4exp`, 1,224 tensors; routed experts `[2560, 640, 512]` gate and up and
`[640, 2560, 512]` down per layer), from the Swift-1.5 Q8_0 (the KLD reference, FN1) with FN4's importance matrix (the
Q8_0 on 200 chunks of 512 tokens of the agent training text, [flash_next_quality](../2026-11-03_flash_next_quality/README.md)):

| File | Gate/up | Down | Every other tensor | Est. on the cards |
|---|---|---|---|---|
| N2P-N2P128 | N2P (268.75 MiB a tensor) | N2P-128 (275 MiB) | pmix-q2-fast's, byte for byte | ~42.3 GiB (+6.6 against pmix-q2-fast) |
| N2P-N3P128 | N2P | N3P-128 (337.5 MiB) | the same | ~45.3 GiB (+9.6) |

For comparison, Q2_0 takes 225 MiB a tensor (Swift Q2_0, pmix-q2-fast: 35,735 MiB on the cards) and Swift's IQ3_XXS
mix 44,710 MiB. llama-quantize (`--pure`, every other tensor left at Q8_0, unused) writes the experts; gguf_compose
takes them onto pmix-q2-fast, so the dense spine, the makers' hyper-connections and output layer and the n-gram table
stay as in that file, and a KLD against it isolates the experts' format. The second file is the first with its down
projections taken from a second llama-quantize run. Requantizing replaces the makers' GSQ-RCO experts (quantized from
the BF16 with their own method) with N2P from the Q8_0: Q8_0's own error is about a sixteenth of a 4-bit type's, so the
double rounding costs little, but the makers' method is not in it.

## Commands

```bash
# build (a build machine with Docker; WORK/src a llama.cpp clone with the fork's branch as a remote)
WORK=<dir> REPO=<this repository> REF=fork/pascal-p106-v3-b12076-n2p JOBS=8 DOCKER_ARGS='--cpu-shares=256' \
  STEPS='rig extra kld' bash docs/data/2026-11-04_n2p_flash/run/build.sh
# the files (CPU, ~5 h for the first and ~2 h for the second at 8 threads under load; resumable)
REPO=<this repository> Q8=<Swift-1.5 Q8_0 shard 1> IMATRIX=<imatrix.gguf> BASE=<pmix-q2-fast shard 1> \
  QUANTIZE=<llama-quantize of the branch> WORK=<scratch> OUT1=<dir> OUT2=<dir> NAME1=<prefix> NAME2=<prefix> \
  MOUNTS='<dirs>' bash docs/data/2026-11-04_n2p_flash/run/quantize.sh
# fit on nine 6 GB cards from the headers, each file once built
REPO=<this repository> OUT=<dir> FILES='pmix-q2-fast=<shard 1> n2p-dn2p128=<shard 1>@<marker> n2p-dn3p128=<shard 1>@<marker>' \
  bash docs/data/2026-11-04_n2p_flash/run/fit.sh
# correctness through the PTX on a newer card (the rig tarball), then natively there (the compute 8.6 build: llama.cpp's
# kernels and MMQ)
docker run --rm --gpus '"device=<card>"' -v <dirs> -e MODE=ptx -e BIN=<rig tree>/bin -e OUT=<dir> -e DEV=0 \
  nvidia/cuda:12.9.2-devel-ubuntu24.04 nice -n 19 bash docs/data/2026-11-04_n2p_flash/run/gpu_checks.sh
docker run --rm --gpus '"device=<card>"' -v <dirs> -e MODE=native -e BIN=<compute 8.6 tree>/bin -e OUT=<dir> -e DEV=0 \
  -e RUNS='np reg-rows' nvidia/cuda:12.9.2-devel-ubuntu24.04 nice -n 19 bash docs/data/2026-11-04_n2p_flash/run/gpu_checks.sh
# the checks' tables from the two output folders
bash docs/data/2026-11-04_n2p_flash/run/checks_table.sh <folder> native86=<OUT of the native check> ptx=<OUT of the PTX check>
# KL divergence: FN2's kld.sh and split against FN1's bases, each file in place, the control last. ENV is the quality
# chain's environment file (../2026-11-03_flash_next_quality) with PPL=<this branch's compute 8.6 llama-perplexity>
# and GPUS_ALL=<the cards>; a build marker is the file quantize.sh writes when a file is verified
ENV=<environment file> QUALITY=docs/data/2026-11-03_flash_next_quality/run BASES=<its kld folder> OUT=<score folder> \
  HOT=<scratch> LINKS=<folder for the links> TABLE_FAST=<the n-gram table shard on a fast disk> SCORE_TAG=n2p DONE=<dir> \
  SCORE=docs/data/2026-11-04_pascal_mix/run/score_direct.sh SCORE_ALL=docs/data/2026-11-04_pascal_mix/run/score_all.sh \
  GPUS=<the cards> LIST='n2p_dn2p128 <file 1, shard 1> <its build marker>
n2p_dn3p128 <file 2, shard 1> <its build marker>
pmix-q2-fast-n2pbuild <pmix-q2-fast, shard 1>' \
  setsid nohup bash docs/data/2026-11-04_n2p_flash/run/kld.sh >> score.log 2>&1 < /dev/null &
#   each text is one call: llama-perplexity -m <file> -f <text> -c <c> -b <c> -ub 512 --chunks <n> -fa on -fit on \
#     --kl-divergence-base <the reference's logits> --kl-divergence   (agent: c 8192, n 12; wiki and code: c 512, n 80)
python3 -I docs/data/2026-11-03_flash_next_quality/run/table.py <score folder> n2p_dn2p128 n2p_dn3p128 pmix-q2-fast-n2pbuild > kld.tsv
# the check set for a P106: recorded on a build machine's CPU with the release rig build's own test-backend-ops
# (containers without a card, --cpu-shares=256, nice 19), tarred with its md5
OUT=<sets folder> SUITE=n2p BIN=<release rig tree>/bin JOBS=8 nice -n 19 bash scripts/record_checks.sh
# the set once through the compute 6.1 PTX on a newer card: its PTX run list, the fork's Pascal paths forced on
docker run --rm --gpus '"device=<card>"' -v <dirs> -e SET=<sets folder>/n2p -e RUNS_FILE=<sets folder>/n2p/runs-ptx.tsv \
  -e BIN=<release rig tree>/bin -e OUT=<dir> -e DEV=0 -e CUDA_CACHE_PATH=<dir>/nvcache -e CUDA_CACHE_MAXSIZE=4294967296 \
  -e ENV_ALL='GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5' \
  nvidia/cuda:12.9.2-devel-ubuntu24.04 nice -n 19 bash scripts/replay_checks.sh
# on a rig, idle (C394), the set unpacked there and scripts/replay_checks.sh beside rig_tk.sh: the native check with
# MMQ by replay, per-kernel perf at Flash-Next's expert shapes, T(k) of the files
BIN=<release rig tree>/bin SET=<the n2p set's folder> CPU_CHECK=200 FILES='n2p-dn2p128=<shard 1> pmix-q2-fast=<shard 1>' \
  VC=<folder with verify_cost.sh> OUT=<dir> TS=<the file's -ts> setsid nohup bash run/rig_tk.sh > rig_tk.out 2>&1 &
#   STEPS='check perf tk' instead computes the references live on the host's CPU (no SET): for a machine with a real CPU
```

Provenance: the Swift-1.5 Q8_0 is ukisai's (`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF`, folder `Q8_0/`); the levels
are ik_llama.cpp's IQ2_K and IQ3_K tables (MIT). Licence: the repository's.
