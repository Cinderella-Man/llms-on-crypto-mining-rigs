# The old v3 base's branches on the current line (C80, Q573), 2026-11-05

Five fork branches were built on the old v3 base (`pascal-p106-v3`, `37e717ff6`) and never moved to the current line.
Each is now ported onto `pascal-p106-v3-b12076-ladder4` (`e3b30303d`, every v3-b12076 feature in one tree) as its own
branch made of its own commits, with every change behind a switch that is off by default. With no switch set, each
port runs as `-ladder4`. They are not merged into `-ladder4`: the combined tree is `-ladder5` (Q1026). Registry:
[Q573](../../../quants/experiments.md) (the move to the new base), with the branches' own rows Q260, Q261 and Q263 (the
prompt path), Q211 (the multi-token kernels) and Q194 (the cheap guess; [docs/drafter.md](../../drafter.md), "A cheaper guess").
The switches are listed with every other one in [docs/switches.md](../../switches.md).

## The branches

| Port (fork) | Tip | From (old base) | Commits | Switches, default off |
|---|---|---|---|---|
| `pascal-p106-v3-b12076-textreuse` | `5b8a616df` | `pascal-p106-v3-textreuse` `60615dffd` | 1, clean | `LLAMA_SERVER_TEXT_REUSE=1`: keep the cached ids where the prompt spells the same text (no re-read of a reply whose tokens the model split its own way) |
| `pascal-p106-v3-b12076-decodemask` | `33f88566a` | `pascal-p106-v3-decodemask` `55143f751` | 1, one conflict | `LLAMA_GPU_KQ_MASK_MAX=<m>`: the card-built mask only for micro-batches of up to m tokens (with `LLAMA_GPU_KQ_MASK=1`) |
| `pascal-p106-v3-b12076-promptpath` | `1cb98250c` | `pascal-p106-v3-promptpath` `8130a649d` | textreuse, a merge of decodemask, then 1 | both of the above and `LLAMA_SERVER_TAIL_CKPT_MIN=<n>`: an end-of-prompt checkpoint only n tokens past the newest one |
| `pascal-p106-v3-b12076-cheapguess` | `046beb835` | `pascal-p106-v3-cheapguess` `366478ef6` | 4 + 1 (the switches) | staircase drafter (`--pipe-guess staircase`): `LLAMA_PIPE_STAIR_CONT=1`, `LLAMA_DRAFT_PRUNE=1`, `LLAMA_DRAFT_HEAD=q8_0\|f16`, `LLAMA_DRAFT_VOCAB=<k>`, `LLAMA_DRAFT_ANC=dense`, `LLAMA_DRAFT_SPLIT=<n>`, `LLAMA_PIPE_DRAFT_TAPS_ONLY=1`; CUDA: `GGML_CUDA_MMVQ_MAX_COLS=<n>`; `LLAMA_DRAFT_LOAD_CHUNK=<MiB>` (8 by default, see below) |
| `pascal-p106-v3-b12076-mtpk` | `7655db25b` | `pascal-p106-v3-mtpk` `f0f2be53b` (its own 2 commits; `-iqmmvq` and `-lowbit` are in `-ladder4` already) | 2 + 1 | `GGML_CUDA_MMVQ_PASCAL_MT=1`: `mul_mat_vec_q_pascal_mt` for IQ2_S, IQ3_XXS, N2P, N3P, Q8_0 and Q5_0 at 2–4 tokens; `_MT_RB`, `_MT_ACT_KB` tune it |

Each port's commits keep their author and message, with `(cherry picked from commit …)` and a note where the port
changed something.

## What the ports changed

- **`-decodemask`:** the base now lets M-RoPE text batches (Qwen3.5/3.6, Ornith, Qwen3.8) build the mask on the card;
  the bound sits beside that rule. Every caller goes through the one predicate, so the bound covers the hybrid models'
  attention too.
- **`-promptpath`:** the slot keeps its checkpoints as shared pointers since `--ctx-checkpoints-share`, so the newest
  one is read as `checkpoints.back()->n_tokens`; a checkpoint that `--ctx-checkpoints-spread` places near the prompt
  end is kept, as user-message checkpoints are.
- **`-cheapguess`:** three behaviours were unconditional or on by default on the old branch and are now switches off
  by default:
  - `LLAMA_PIPE_STAIR_CONT=1`: without a store, the request's continuations of 2-token or longer contexts go before
    the drafter. Unset, the copy lookup alone, then the drafter, as `-ladder4` does. With a store nothing changes.
  - `LLAMA_DRAFT_PRUNE=1`: the last block runs on the output row only. It was on unless set to 0, and is exact.
  - `LLAMA_DRAFT_LOAD_CHUNK=<MiB>`: the drafter file is read, and the head converted, in pieces (default 8; 0 reads
    each tensor at once). The weights are the same bytes either way, so this one stays on: a Q8_0 head converted at
    once held ~424 MB, which the rig's 4 GB host lacked next to the target.
  `GGML_CUDA_MMVQ_MAX_COLS` still takes effect: dense products still choose through `ggml_cuda_should_use_mmvq`.
- **`-mtpk`:**
  - The kernel sits next to `-kshape`'s launcher, which keeps its warp tiles. `mul_mat_vec_q_pascal_mt` runs only
    without tiles: tile staging covers shapes the old base declined.
  - On this line `-kcol` takes plain 1–8-token products and `-mtok` takes 2–8-token expert products of Q2_0, the grid
    types, IQ1_M and N2P/N3P first. So with the switch on, the kernel sees what those decline, chiefly Q8_0 and Q5_0
    experts and fused products.
  - Skipped slots (id −1, `LLAMA_MOE_TAIL`) are handled as the multi-column kernel handles them: they start no group,
    block 0 writes their zeros and no expert's rows are read. On the old branch such an id would have been taken as
    an expert. The test list gains the skipped-id cases of the six types at 2–4 tokens.
  - The default flips from on (unless 0) to off.

## The checks

On a build machine, CPU only (Docker, `--cpu-shares=256`, `-j 8`, one build at a time under a lock, no GPU), one branch
after the other ([run/checks.sh](run/checks.sh)):

| Step | What |
|---|---|
| build | the rig tarball (`scripts/build_llamacpp_docker.sh`, `TESTS=1`: sm_61 code and compute_61 PTX, no AVX) and `test-llama-archs` from its build tree |
| identity | `test-llama-archs -b CPU` on every architecture, the logits written by `-ladder4`'s rig build and compared by the port's, no switch set: bit for bit |
| cpubuild | a CPU build without llamafile GEMM (batch-invariant): `llama-server`, `test-llama-archs`, `llama-pipe-drafter-check` |
| own | each branch's published check with its switches on: [check_server.sh](../2026-10-28_reply_rereads/run/check_server.sh) (text reuse, and with `LLAMA_SERVER_TAIL_CKPT_MIN=4096`) and [mask_check_cpu.sh](../2026-10-28_reply_rereads/run/mask_check_cpu.sh) (the card-built mask for decode steps and for all, each mask compared with the host's) from [2026-10-28_reply_rereads](../2026-10-28_reply_rereads/README.md); `llama-pipe-drafter-check` against `drafter.Drafter.forward` on a random-weight head of the real shape for every option ([2026-10-27_cheap_guess](../2026-10-27_cheap_guess/README.md)'s `check.txt` case), the piecewise load against whole tensors, and the defaults against `-ladder4`'s drafter |

`-mtpk` has no CPU check beyond its build and identity: its kernels run on a GPU only (C540).

## Results (2026-11-05)

**Every port builds for the rig, equals `-ladder4` bit for bit with no switch set, and passes its own branch's check with
its switches on.** Per branch ([summary.tsv](summary.tsv); tarball md5s in [md5.txt](md5.txt)):

| Port | Rig build | Identity with `-ladder4` (no switch set) | Own checks, switches on |
|---|---|---|---|
| `-textreuse` | OK | 143 logits compared, none differ | text reuse, 6 of 6 trials: each 700-token reply's ids differ from its text's tokenization; without the switch the next request restores the 18-token checkpoint and processes 744–761 tokens, with it the cache covers the prompt and the whole reply (720–721 tokens, 6–8 processed), and every answer equals a cold server's ([textreuse_check_textreuse.tsv](textreuse_check_textreuse.tsv)); the same numbers as the original branch's check |
| `-promptpath` | OK | none differ | text reuse, 6 of 6, as `-textreuse` ([textreuse_check_promptpath.tsv](textreuse_check_promptpath.tsv)); with `LLAMA_SERVER_TAIL_CKPT_MIN=4096` added, 6 of 6, and the second request creates no end-of-prompt checkpoint ([textreuse_check_promptpath_tail.tsv](textreuse_check_promptpath_tail.tsv)); the mask check as `-decodemask` |
| `-decodemask` | OK | none differ | `test-llama-archs` 148 of 148 with the host mask, the decode-only setting and the all-batches setting; the last compared 1,557 card masks with the host's, all identical (micro-batches of 32, 64 and 128 tokens); the decode-only setting built none, since every micro-batch of the test is over 16 tokens ([mask_cpu_decodemask.txt](mask_cpu_decodemask.txt)) |
| `-cheapguess` | OK | none differ | `llama-pipe-drafter-check` on the CPU: 9 option sets, 0 of 8 steps outside the tolerance in each, worst 0.0025 standard deviations of the reference's logits (the Q8_0 head 0.039, its own rounding) ([drafter_options.tsv](drafter_options.tsv)); `LLAMA_DRAFT_LOAD_CHUNK=0` gives the default's step lines, with and without the Q8_0 head; with no option set the port's drafter gives `-ladder4`'s step lines exactly ([drafter_rig-ladder4.txt](drafter_rig-ladder4.txt), [drafter_rig-port.txt](drafter_rig-port.txt)) |
| `-mtpk` | OK (sm_61 and the PTX) | none differ | none on a CPU |

Identity: `test-llama-archs -b CPU` builds a random model of every architecture and compares the logits of each with
the file `-ladder4` wrote; the five ports' comparison lines are byte for byte the same ([identity_textreuse.txt](identity_textreuse.txt);
two architectures share a logits file and differ from each other in both builds). Both builds report the same
summary: 148 tests, the same 7 failing in `-ladder4` too.

Not covered here: `LLAMA_PIPE_STAIR_CONT` (it only picks the guesser kind in llama-server; the guesser it picks is the
one that runs with a store); any kernel on a GPU (`-mtpk`'s products, `-cheapguess` on a card,
`GGML_CUDA_MMVQ_MAX_COLS`, the card-built decode mask on cards); any rig run.


## Commands

```bash
# on the build machine, with -ladder4's rig tree unpacked (test-llama-archs in its bin/), the tiny random Flint
# (Flint's tokenizer and chat template) and the drafter-check case; WORK/src a llama.cpp clone with the fork as remote
WORK=$W REPO=$REPO BASE_BUILD=$L4/llama.cpp-20261105-e3b30303d-pascal-p106-v3-b12076-ladder4 TINY=$TINY CASE=$CASE \
  JOBS=8 LOCK=$LOCKFILE DOCKER_ARGS='--label pascal-quants=1' PY_IMAGE=python:3.12-slim-bookworm \
  bash docs/data/2026-11-05_c80_port/run/checks.sh
cat $W/summary.tsv
```

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the builds and checks above, per branch |
| [summary.tsv](summary.tsv) | branch, check, verdict, detail: the final verdict of every check |
| [md5.txt](md5.txt) | the five rig tarballs: md5, name, bytes |
| `identity_<port>.txt` | `test-llama-archs`' comparison lines of each port against `-ladder4`'s logits |
| `textreuse_check_*.tsv` | `textreuse_check.py`'s trials (per trial and server: prompt and reply tokens, cached tokens, tokens processed, whether the answer equals the cold server's) |
| `mask_cpu_*.txt` | `mask_check_cpu.sh`'s three arms: exit status, summary, card masks compared, by micro-batch size |
| [drafter_options.tsv](drafter_options.tsv), `drafter_*.txt` | the drafter check per option set; the step lines of the defaults (CPU build, `opt0`), of `LLAMA_DRAFT_PRUNE=1` (`opt1`) and of both rig builds |

## Provenance and licence

Code: the fork's branches (MIT, as llama.cpp). The tiny random Flint carries Flint's tokenizer and chat template and
random weights; the drafter-check case is a random-weight head with `drafter.Drafter.forward`'s logits
(`bench/staircase_drafter/check_export.py case`). No model output or third-party data is in this folder.
