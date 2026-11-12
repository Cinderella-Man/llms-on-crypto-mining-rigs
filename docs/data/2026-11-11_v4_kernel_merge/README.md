# The v4 kernel merge: three read-once kernel branches in `pascal-p106-v4` (2026-11-11)

Every switch of the v4 line goes into `pascal-p106-v4`, off by default, whether or not it has proven a gain yet. After
[the late merge](../2026-11-11_v4_late_merge/README.md) (`08634ce00`) three kernel branches finished, each cut from the
day merge (`18d0fa309`) and each checked on a CPU by its own session; all three extend C722's read-once kernel for
Qwen3.8-Flash-Next's verify passes (`ggml/src/ggml-cuda/mmvq-pascal-dmt.cu`):

- [C727](../2026-11-11_c727_q6k_readonce/README.md), `GGML_CUDA_MMVQ_PASCAL_Q6K_MT`: Q6_K's products of 2-4 tokens
  (`attn_output` and `ssm_out`, 6,144 → 2,560) in the read-once kernel;
- [C728](../2026-11-11_c728_readonce_k6144/README.md), `GGML_CUDA_MMVQ_PASCAL_DMT_KT`: the activations staged one K tile
  at a time, so a block holds 6 warps at K = 6,144;
- [C729](../2026-11-11_c729_readonce_5to8/README.md), `GGML_CUDA_MMVQ_PASCAL_RO8`: IQ4_XS's, Q4_K's and Q5_K's products of
  5-8 tokens on the read-once kernels.

This folder merges them into `pascal-p106-v4`, builds the result once for the rig (compute 6.1) and checks it on a CPU:
each branch's own static checks on the merged build, compared with the tables each branch published; the host-side
kernel choices of all five trees (the base, the three branches, the merge) under 37 switch settings; `test-llama-archs`'
identity with the base with every switch off and with all of them on; llama-server's replies with every switch on; the
CPU test set; and one replay set holding the three branches' kernel cases for the rig. The registry row is Q1391 in
[quants/experiments.md](../../../quants/experiments.md); every switch is in [docs/switches.md](../../switches.md).

**Result:** the merged tree builds for the rig, and each switch alone does exactly what its branch did. Every GPU
function of the merged build is the base's or the code of the one branch that added it (8,540 functions: 8,483 the
base's, 9 C727's, 24 C728's, 24 C729's), each branch's own SASS tables on the merged objects are the ones it published,
and the host's kernel choices are each branch's for every product under 37 switch settings. With the three on together
each product runs the kernel of the branch that owns it, and Q6_K's products of 5-8 tokens stay on the k-column kernel
(the one combination no branch built). With every switch off, and with all of them on, `test-llama-archs`' logits are
the base's bit for bit on all 146 models; llama-server's MTP-drafted replies with every switch of the four v4 merges on
are the base's (the same tokens, probabilities, draft counts and 406 verify steps); the CPU test set passes (ctest 70 of
70, static and shared). Nothing ran on a GPU: exactness on the card and the speed are R120's.

## The branch

`pascal-p106-v4` at **`d097d6d88`**: one merge commit per branch on `08634ce00` (`git merge --no-ff`, no rebase), pushed
first as `pascal-p106-v4-b12246-kern` for the build machine to fetch, and `pascal-p106-v4` moved to the same commit once
every check below had passed. The order is the order the branches finished in (C727 at 17:40, C728 at 18:30, C729 at
18:45), and it is the order their merge notes assume: C728's tiles were given guards against C727's Q6_K and C729's
5-8 tokens (its commits `e4f8e2a65` and `aa8879e50`), and C729's notes name its merge points with both.

| Commit | Merge of | Brings (off by default) | Conflicts and their resolution |
|---|---|---|---|
| `91e9dc888` | `-c727` `0e113d1f3` | `GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1` (`=2` without the prefetch; `_FAULT=1` for the checks) | none: the merge's diff is exactly `-c727`'s against `18d0fa309` |
| `add77c4c1` | `-c728` `aa8879e50` | `GGML_CUDA_MMVQ_PASCAL_DMT_KT=1` (`=2` wherever the tiles fit; `_KT_M`, `_KT_NW`, `_KT_FAULT`) | the header's switch comments (C727's and C728's paragraphs at the same place): both kept. The merge's added and removed lines are exactly `-c728`'s |
| `d097d6d88` | `-c729` `3f166ab56` | `GGML_CUDA_MMVQ_PASCAL_RO8=1` (`=2` without the prefetch; `_RO8_FAULT`) | `mmvq-pascal-dmt.cu`: the file's top comment and defines, the switch functions (C727's `dmt_q6k_mode`, `dmt_type_mode`, `dmt_q6k_fault` and C729's `dmt_ro8`, `dmt_ro8_fault`), the switch test of `dmt_pick_pf` and of `ggml_cuda_mmvq_pascal_dmt_takes`, and the two faults' lines; comments in `mmvq.cu`, `mmvq-pascal-kcol.cu`, the header and `test-backend-ops.cpp`. All kept, with one helper added (below). `mmvq-pascal-q5k.cu` and its header are `-c729`'s exactly |

The one piece of new code: `dmt_nt_mode(type, nt)` gives the switch of a product by its token count: at 2-4 tokens
C727's `dmt_type_mode` (`_DMT` for IQ4_XS and Q4_K, `_Q6K_MT` for Q6_K), at 5-8 tokens `_RO8` for IQ4_XS and Q4_K and
nothing for Q6_K. `ggml_cuda_mmvq_pascal_dmt_takes` and `dmt_pick_pf` ask it where C727 asked `dmt_type_mode` and C729
asked `nt > 4 ? _RO8 : _DMT`; and `dmt_launch` compiles no Q6_K kernel of 5-8 tokens (an unreachable branch that would
abort). So each switch alone takes exactly the products its branch took, and where two meet:

- C727 with C729: Q6_K's products of 5-8 tokens stay with the k-column kernel (Q6_K at 5-8 tokens under `_RO8` is C795,
  Q1390, planned);
- C728 with C727 and C729: the tiles take only IQ4_XS and Q4_K at 2-4 tokens (C728's own guards), so Q6_K and 5-8 tokens
  keep the whole-staging kernel (the tiles for Q6_K are C785, Q1374; at 5-8 tokens C794, Q1389);
- all three on: every product runs the kernel its own branch runs (`pick` below shows it for every product shape).

## The checks

[run/checks.sh](run/checks.sh) ran every step on a build machine's CPU, in Docker containers at `--cpu-shares=256` and
`nice 19`, one build or test at a time, no GPU, on `d097d6d88` (2026-11-11 18:58–19:45, the rerun below 19:45–19:48,
`final` 19:49–19:50). The base is `08634ce00`: its rig tarball and its static CPU build are the late merge's. Every
verdict line is in [summary.tsv](summary.tsv), in the order the steps ran.

| Check | What ran | Result |
|---|---|---|
| builds | the rig tarball (CUDA 12.9, sm_61, no AVX, `test-backend-ops` and `test-llama-archs` in `bin/`) and a static native CPU build of every target | both built, 0 error lines; the tarball has 0 AVX instructions and needs GLIBC 2.38 at most |
| switch list | C770's list of the `LLAMA_` and `GGML_` names the sources read, in the merged build | 858 names: the late merge's 850 and the branches' eight (`_Q6K_MT`, `_Q6K_MT_FAULT`, `_DMT_KT`, `_DMT_KT_M`, `_DMT_KT_NW`, `_DMT_KT_FAULT`, `_RO8`, `_RO8_FAULT`), each in `llama-server` |
| GPU code by build | `cuobjdump -sass` and `-ptx` of `llama-server` from the merged build, the base's and each branch's own tarball, cut into functions (anonymous-namespace hashes masked) and attributed with the day merge's [sass_attrib.py](../2026-11-11_v4_day_merge/run/sass_attrib.py) | SASS: 8,540 functions, 8,483 the base's, 9 C727's, 24 C728's and 24 C729's, each with its branch's code; nothing missing, lost or merged. PTX: the same, but for 10 `flash_attn_tile` kernels whose PTX in C727's own tarball differs from the base's in one name only: a static shared variable's per-file suffix (`…_cu_<hash>_<number>`); with it masked their text is the base's, their SASS is the base's, and the merged build has the base's code for them |
| C727 | its [run/sass_check.sh](../2026-11-11_c727_q6k_readonce/run/sass_check.sh) on the merged `mmvq-pascal-dmt.cu.o` | the table of the 9 Q6_K instantiations (registers, stack, local memory, every instruction count of the kernel and of its main loop) is the one C727 published, line for line; no stack or local memory |
| C727 | its [run/q6k_emul.c](../2026-11-11_c727_q6k_readonce/run/q6k_emul.c): the one-token kernel's and the read-once kernel's Q6_K arithmetic emulated from the same bytes | 0 of 61,440 outputs differ over 30 products; both controls differ in 30 of 30; the exact conversions hold in 133,686,017 cases; both tables byte for byte C727's |
| C728 | its [run/sass_check.sh](../2026-11-11_c728_readonce_k6144/run/sass_check.sh) on the merged objects | every one of the 50 rows it published is in the merged table as it is (the 24 tiles' kernels and C722's 24 among them); the 25 rows more are C727's 9 Q6_K and C729's 16 IQ4_XS and Q4_K instantiations of 5-8 tokens; no tiles' kernel with stack or local memory |
| C729 | its [run/compare_sass.py](../2026-11-11_c729_readonce_5to8/run/compare_sass.py)'s kernel table on the merged objects | every one of the 64 rows it published (C722's and C690's kernels at 2-8 tokens, C729's 24 new among them) is in the merged table with the same code, registers, stack, local memory, spills and counts; the 9 rows more are C727's Q6_K (its script names Q6_K by its number, 14); none of the 73 with stack, local memory or a spill instruction |
| kernel choices | [run/kern_pick.cu](run/kern_pick.cu) built against each tree's own `mmvq-pascal-dmt.cu` (the base, the three branch tips, the merge) and run under 37 settings of the switches: for IQ4_XS, Q4_K and Q6_K, 2-8 tokens and rows of 1-64 super-blocks, whether the product is taken and with which kernel, R, prefetch, warps, shared memory and staging stride | under every setting, each of the merged tree's 49,728 product lines is the line of the branch that owns it in that branch's own tree (Q6_K at 2-4 tokens C727's, IQ4_XS and Q4_K at 2-4 C728's, at 5-8 C729's), and Q6_K at 5-8 tokens is off; under the 8 settings of the base's switches alone (none set among them) every line is the base's |
| bit identity | `bench/mmvq_pascal/verify/multitoken_bitid` built against the merged tree | it holds the modes `q6k` (C727), `dmtkt` (C728), `dmt8` and `q5k8` (C729), and the build's own takes functions |
| every switch off | `test-llama-archs -b CPU -s 1` on every architecture against `08634ce00`'s logits | 146 of 146 models the same bit for bit (151 of 151 tests pass); the same with each branch's switch, its fault, and the three on together (CUDA switches: they change nothing on a CPU) |
| every switch off | [C565's CPU test set](../2026-11-05_c565_ladder4fix/README.md), static and shared libraries | both built (0 error lines); ctest 70 of 70, `test-arg-parser` with network, `test-llama-archs` written and compared (146 files, none differ), the nine hand tests, in each |
| every switch on | `combo`: `test-llama-archs` with the 31 switches of the four merges on together | 146 of 146 models the same as `08634ce00` |
| every switch on, llama-server | `combo`: the late merge's [server_combo.sh](../2026-11-11_v4_late_merge/run/server_combo.sh) on a random qwen4exp model with an MTP block, arms `late` and `latn` (with and without the backend sampler) with the four kernel switches (`_DMT=1` and the three) added to every exact host switch of the night's, the day's and the late merges, against `08634ce00`'s llama-server | both arms the base's tokens, top probabilities, draft counts and 406 verify steps (the rows C744's cap cuts left out); C770's line names the 21 variables set, all read by this build (858 names); C744's cap leaves out 12 passes in 8 drafts and C742 398 hidden-row copies in 406 drafts. The script's own verdict reads FAIL by design: it wants no switch line from the base, and this base has C770's (`switches: 1 set`, `LLAMA_SPEC_DUMP`, 850 names); `combocheck` checks that this is its only failing line |
| replay set | `scripts/record_checks.sh SUITE=v4kern` with the merged tarball's `test-backend-ops` | 696 cases, 47 runs: the suites `q6kmt` (13 runs), `dmtkt` (12) and `ro8` (13) as their branches recorded them, and 9 runs with the three switches on together over all their cases (at the preset's settings, without the prefetch, the tiles wherever they fit, two row blocks a round, R = 2, without the preset; and each branch's fault with all three on, which must fail cases). `v4kern.tar` 4,074,956,800 bytes, md5 `902506fe67896f898f0f58c5674546d6` |

The first run's `off` step ran C727's switch and its fault as one arm (a missing space in the driver joined the two),
and its C729 table comparison read Q6_K's rows (named `14` by C729's script) as unexpected and the spill columns one
off; both were corrected in `run/checks.sh` and rerun from the same build and dumps (`STEPS="sasscmp off combocheck"`).
In `summary.tsv` the first run's `sass llama-server ptx by build` (CHECK: the 10 names), `c729 compare_sass.py…`
(DIFFERS) and `combo server_combo late latn` (FAIL: the base's line) are each followed by the line that settles them:
`sass ptx listed: names only` OK, the rerun's `c729` SAME and `combo server_combo with a base that has C770` OK.

What launches with all three switches on, at Flash-Next's widths (`pick/flashnext_all.tsv`, the merged tree):

| Products | 2 tokens | 3 tokens | 4 tokens | 5 tokens | 6 tokens | 7-8 tokens |
|---|---|---|---|---|---|---|
| IQ4_XS, K = 2,560 (`attn_qkv`, `attn_q`, `attn_gate`) | read-once, 6 warps | read-once, 6 | read-once, 6 | read-once, 6 | read-once, 6 | read-once, 6 and 5 |
| IQ4_XS, K = 6,144 (`ssm_out`) | read-once, 6 | read-once, 6 | **K tiles, 6** | read-once, 3 | read-once, 1 | k-column |
| Q4_K, K = 2,560 | read-once, 6 | read-once, 6 | read-once, 6 | read-once, 6 | read-once, 6 | read-once, 5 |
| Q4_K, K = 6,144 (`ssm_out`, `attn_output`) | read-once, 6 | **K tiles, 6** | **K tiles, 6** | read-once, 2 | k-column | k-column |
| Q6_K, K = 6,144 (`attn_output`, `ssm_out`) | read-once, 4 | read-once, 3 | read-once, 2 | k-column | k-column | k-column |

## The rig tarball

The checked build, `llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern.tar.gz`: 678,294,162 bytes, md5
`37a4c3731f76455dce14013210b2fac7` (`TESTS=1`, CUDA 12.9 image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, sm_61, 0 AVX
instructions, highest glibc symbol GLIBC_2.38, `test-backend-ops` in `bin/`, md5 `1b2e6a7b426c8062a2bdee9acb761148`).
Every check above ran on it; R120 takes it.

`pascal-p106-v4`'s own, built after the push from the same clone and ccache (`final`):
`llama.cpp-20261111-d097d6d88-pascal-p106-v4.tar.gz`, 678,294,914 bytes, md5 `d88056ecda0639f67cd279f38be2aaa3`. Nine
of its 11 files in `bin/` are the checked build's bytes; `llama-server` and `llama-cli` differ in the web UI they embed
and nowhere in the GPU code (their `.nv_fatbin` sections are the same bytes): the build downloads llama.cpp's prebuilt
UI assets from upstream's Hugging Face bucket (`scripts/ui-assets.cmake`, the build number's version or else
`latest`), and between the two builds, 48 minutes apart, upstream's assets changed (another bundle name and file
hashes), which moves the rest of those two binaries' code and data. So the two tarballs run the same engine; they
differ in the web page llama-server serves.

## The rig run (R120)

[run/rig_kern.sh](run/rig_kern.sh), on rig 1, from the three branches' rig scripts (R117, R118 and R116's first three
steps) in one run on the merged build:

- `replay`: the set `v4kern` through `scripts/replay_checks.sh` on one card, about one small case in 20 also against the
  rig's own CPU: every run "ok", and each of the six fault runs failing cases;
- `bitid`: `multitoken_bitid`, every product against its columns' one-token products bit for bit: each branch's own
  arms (`q6k`: C727's; `dmtkt`: C728's; `dmt8`, `q5k8`: C729's), then every mode (`q6k`, `dmtkt`, `dmt`, `dmt8`, `q5k`,
  `q5k8`) with the three switches on together, the tiles everywhere, and each fault with all three on; 0 comparisons may
  differ, and the faults and controls must;
- `perf`: `test-backend-ops perf` at Flash-Next's IQ4_XS, Q4_K, Q5_K and Q6_K shapes, 1-8 tokens, arms `off` (the preset's
  k-column switches), `mt` (+ `_DMT=1 _Q5K_MT=1`), `kern` (+ the three switches), `kernm2` (+ `_DMT_KT_M=2`), two passes;
- `served`: R49's profile of a served verify pass, `off on on off`, both arms with `_DMT=1 _Q5K_MT=1`, `on` with the
  three switches.

It decides, per switch and together: exact on the card or not (any difference is a fault to find first), gain,
negligible or loss per shape and token count against `mt`, each pass by value, and the products' time in a served
verify pass.

R120a ran the replay and the bit identity on rig 1 on 2026-11-12, with the bit-identity list extended to C726's `q3k`,
C757's `glu` and C754's `f32` modes and all six verify-pass switches on: [2026-11-12_r120a_kern_exact](../2026-11-12_r120a_kern_exact/README.md).
C727, C728, C729 and C754 are exact on the card, alone and together; the `q3k` and `glu` differences are the k-column
kernel's declined products, the same with every switch off (Q1411).

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the driver: every step above |
| [run/kern_pick.cu](run/kern_pick.cu) | the kernel choices of one tree's `mmvq-pascal-dmt.cu`, without a GPU |
| [run/rig_kern.sh](run/rig_kern.sh) | R120's run on a rig |
| [summary.tsv](summary.tsv) | every verdict line (row, check, verdict, detail) |
| [sass_attrib.txt](sass_attrib.txt) | `sass_attrib.py attrib`'s output for the SASS and the PTX, and the PTX functions it lists compared with the suffix masked |
| [sass_counts_c727.tsv](sass_counts_c727.tsv), [sass_counts_c728.tsv](sass_counts_c728.tsv), [sass_kernels_c729.tsv](sass_kernels_c729.tsv) | each branch's table on the merged objects |
| [pick_flashnext_all.tsv](pick_flashnext_all.tsv), [pick_all_kernels.tsv](pick_all_kernels.tsv), [pick_arms.txt](pick_arms.txt) | `kern_pick.cu`'s 37 switch settings, and with all three switches: Flash-Next's widths, and the count of each kernel per type and token range |
| [replay_cases.tsv](replay_cases.tsv), [replay_runs.tsv](replay_runs.tsv) | the set `v4kern`: its cases and its runs |

## Commands

On the build machine, with this repository at `$REPO`, a llama.cpp clone with the fork as remote `fork` at `SEED_SRC`,
the base's unpacked rig tarball and static CPU build, and the three branches' unpacked rig tarballs:

```bash
M=<work folder> REPO=$REPO LOCK=<flock file> BASE_TREE=<base rig tree> BASE_CPU=<base CPU bin/> \
BR_C727=<C727 rig tree> BR_C728=<C728 rig tree> BR_C729=<C729 rig tree> SEED_SRC=<clone> \
bash docs/data/2026-11-11_v4_kernel_merge/run/checks.sh
# once every check has passed and pascal-p106-v4 is moved to the checked commit:
M=<work folder> REPO=$REPO LOCK=<flock file> SEED_SRC=<clone> STEPS=final bash docs/data/2026-11-11_v4_kernel_merge/run/checks.sh
```

On the rig (R120), with the merged tarball unpacked, the set, the harness and the preset files:

```bash
BIN=<tarball>/bin OUT=<dir> SET=<v4kern folder> REPLAY=scripts/replay_checks.sh BITID=<multitoken_bitid> MODEL=<gguf> \
DRAFT=<draft gguf> PRESET=release/flash-next-v3/settings.env RIG_SERVER=bench/agent_check/rig_server.sh \
REQ=docs/data/2026-11-04_draft_one_card/run/greedy_request.json VERIFY_PROF=docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh \
setsid -f nohup bash docs/data/2026-11-11_v4_kernel_merge/run/rig_kern.sh > console.txt 2>&1 < /dev/null
```

## What it decides

- Every switch of the v4 line so far is in one branch, `pascal-p106-v4`, off by default: the night's 17 branches, the
  day's 17, the late merge's four and these three. With nothing set it runs the late merge's code.
- R120 checks the three kernel branches on a P106 in one run; it stands for R117 and R118 and for R116's replay, bit
  identity and perf steps (R116's run of R42's driver stays its own row).

Provenance and licence: the models are `test-llama-archs`' random-weight models and `test-backend-ops`' random tensors;
everything here is this repository's MIT licence.
