# Ornith's 2–4 token passes on the read-once kernel, on three newer cards (C953, Q1687, 2026-11-23)

What this is: the GPU half of the check of [C899](../2026-11-21_c899_kcol_ornith/README.md) (registry
[Q1625](../../../quants/experiments.md)), point 7 of the GPU checks of the night's branches (registry
[Q1687](../../../quants/experiments.md)). C899's branch (`pascal-p106-v5-kcol-ornith` `37c05b76a`, one commit on
`pascal-p106-v5` `3dfad876f`) gives the existing switch `GGML_CUDA_MMVQ_PASCAL_MT` a new value, `2`: only the plain
Q5_0 and Q8_0 matrix-vector products of 2 to 4 tokens that a P106 ran faster on the read-once kernel
(`mul_mat_vec_q_pascal_mt`) go to it, and the k-column kernels leave those to it; everything else runs as with the
switch unset. Such passes come from speculative decoding (a verify step checks 2 to 4 tokens) and from the end of each
prompt, whose last 4 tokens the server reads as a pass of their own to take its checkpoint. The branch's processor
check passed; here it runs on a build machine's three RTX 3090s, through the rig build's compute 6.1 PTX (the same
binaries a P106 rig runs). Correctness only: nothing here is timed.

**The fork's Pascal kernels have to be forced on for this.** The fork's matrix-vector kernels for Pascal
(`mul_mat_vec_q_pascal_mc`, `_mt` and the k-column kernels) switch themselves on only on compute 6.1 cards; on any
other card they stay off unless `GGML_CUDA_MMVQ_PASCAL=2`, and the switch under test acts only inside them. Every run
here sets `GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2` and a P106's launch shapes (5 SMs, 96 KB of shared memory
per SM: `GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`), as [the k-column launch shapes' GPU check](../2026-11-04_kshape/README.md) did.

## Result

**In plain words:** it passes. Every matrix product the switch can touch computes correctly with it on and off. On
the cards the switch sends exactly the products its rule names to the read-once kernel and leaves everything else
alone. Ornith's replies do change with it, as expected, and in a useful way: with the switch on, Ornith's replies
while it drafts its own next tokens are exactly the replies it gives without drafting. With the switch off they are
not.

| Pass criterion (Q1687 point 7) | Result |
|---|---|
| (a) `test-backend-ops` `MUL_MAT`, Q5_0 and Q8_0 at 2, 3 and 4 tokens: `=2` passes as unset does | **yes**: 284 of 284 cases pass in each setting (204 built-in cases at 1 to 5 tokens, 80 of Ornith's shapes and the rule's bounds), live and replayed on the same inputs ([bops.tsv](bops.tsv)) |
| (b) Ornith on the three cards, greedy replies with `=2` against unset: the same, or the first difference named and sized | **differ, named and sized**: with its draft all three replies leave unset's at tokens 69, 54 and 17, without it at 132, 72 and 64, where the two most likely tokens sit 0.121, 0.459 and 0.003 apart in log-probability (drafting) and 0.013, 0.041 and 0.084 (not drafting) ([compare.tsv](compare.tsv)) |
| (added) the switch's unset value is the base's | **yes**: unset, the branch gives the base's tokens and draft counts in all 18 requests of both models |

**The rule on the cards.** Both settings computed the same recorded inputs (`--record`, then `--replay` per setting),
so each case's error against the processor is comparable across settings to the last digit. A "kernels" step replayed
them once more with each kernel family alone (`=1` with the k-column kernels off: every product of 2 to 4 tokens on
the read-once kernel; the k-column kernels off: `mul_mat_vec_q_pascal_mc`; `GGML_CUDA_MMVQ_PASCAL_KCOL=2`: every
product on the k-column kernels), all 284 cases passing in each ([bops-kernels.tsv](bops-kernels.tsv)). Matching the
errors shows which kernel computed each case ([bops-cases.tsv](bops-cases.tsv), [bops-verdict.tsv](bops-verdict.tsv)):

| Part | Cases | The rule takes | `=2` gives the read-once kernel's result | The rule leaves | `=2` gives unset's result | `=2` changes the result |
|---|---:|---:|---:|---:|---:|---:|
| built-in cases | 204 | 88 | 88 | 116 | 116 | 6 |
| Ornith's shapes and the rule's bounds | 80 | 27 | 27 | 53 | 53 | 3 |

- **The read-once kernel and `mul_mat_vec_q_pascal_mc` give the same bits** in all 178 cases of 2 to 4 tokens: both
  form each token's sums as the one-token kernel does.
- **So `=2` changes a result only where unset ran a k-column kernel whose sums differ:** the Q5_0 products wider than
  4,096 values (8,192 in: Ornith's `ssm_out` and `attn_output`; 6,144 in among the built-in cases) at 2, 3 and 4 tokens,
  9 cases in all. Q8_0 up to 4,096 values in (Ornith's `attn_k` and `attn_v`) the k-column kernels already sum as the
  one-token kernel, so moving them changes no bit; the narrower Q5_0 products run `mul_mat_vec_q_pascal_mc` unset.

**Ornith's replies.** Ornith-1.5-122B Pascal N2P GGUF with its v5 preset, split over the three cards (`-ts 1,1,1`),
two models: `ornmtp` drafting three tokens a step with the MTP layer in its own file (`--spec-type draft-mtp
--spec-draft-n-max 3`, so every step verifies 2 to 4 tokens), and `ornith`, the preset as it is (one token a step).
Three prompts (C854b's coding request and prose paragraph, and an Elixir program), 256 tokens each, temperature 0,
twice as the preset serves them and once with `n_probs` 2 (which samples on the host and returns the two most likely
tokens with their log-probabilities); arms: the branch unset, the branch `=2`, the base unset
([replies.tsv](replies.tsv)).

| Comparison | code | prose | prog |
|---|---|---|---|
| branch unset against base, drafting and not | same | same | same |
| each arm, round 2 and the `n_probs` request against round 1 | same | same | same |
| drafting, `=2` against unset: first differing token | 69 | 54 | 17 |
| not drafting, `=2` against unset: first differing token | 132 | 72 | 64 |
| **drafting against not drafting, unset** (branch and base) | differs at 69 | differs at 54 | differs at 17 |
| **drafting against not drafting, `=2`** | **same** | **same** | **same** |

- **Sized:** at the first difference the two most likely tokens sit 0.121 (code), 0.459 (prose) and 0.003 (prog)
  apart in log-probability in the unset drafting reply, and 0.013, 0.041 and 0.084 apart in the unset plain reply;
  before it, the chosen tokens' log-probabilities move by at most 0.076 (drafting) and 0.095 (plain) between the
  settings. Near-ties that a slightly different sum flips.
- **Why plain decoding moves too:** the server reads each prompt's last 4 tokens as a pass of its own (for the
  end-of-prompt checkpoint), and in that pass `=2` sends `ssm_out` and `attn_output` to the read-once kernel. The
  hidden state after the prompt differs in its last bits, and the reply follows a few dozen tokens later.
- **Why drafting matches plain decoding under `=2`:** a verify pass then forms every dense product's sums as a
  one-token step does; with the switch unset the two 8,192-wide products run the k-column kernel, whose sums differ,
  and the drafted reply leaves plain decoding's. Across these three prompts nothing else in a verify pass of 2 to 4
  tokens (the experts, the attention and the recurrent layers included) changes a token.
- Drafting with `=2`: 258 and 261 drafted with 168 and 167 accepted on the coding request in rounds 1 and 2 (the same
  tokens; the draft's own state carries between requests), 296/156 on prose, 300/154 on the program; unset 289/158,
  314/150, 298/155 in every round. All replies read as coherent text.

## Files

| File | What it holds |
|---|---|
| [bops.tsv](bops.tsv) | each `test-backend-ops` run: part, run (record, replays, live), setting, cases, passed, failed, exit status |
| [bops-cases.tsv](bops-cases.tsv) | every replayed case: its shape, whether the rule takes it, verdict and error unset and with `=2`, and which kernel family's run gives each error |
| [bops-verdict.tsv](bops-verdict.tsv) | per part: the rule's cases computed as the read-once kernel does, the others as unset |
| [bops-kernels.tsv](bops-kernels.tsv) | the kernels step's three replays per part |
| [replies.tsv](replies.tsv) | every request of every load: model, arm, round, prompt, tokens, md5 of the token ids, drafted, accepted, stop |
| [compare.tsv](compare.tsv) | per comparison and prompt: first differing token per round, the two candidates there, the largest log-probability change before it |
| [run/run.sh](run/run.sh) | the driver (steps `bops`, `load`, `kernels`) |
| [run/shapes.py](run/shapes.py) | writes the test file of Ornith's 9 dense Q5_0/Q8_0 shapes and 7 shapes at the rule's bounds, 1 to 5 tokens each |
| [run/replies.py](run/replies.py) | the requests of each load |
| [run/tables.py](run/tables.py) | the tables from the driver's output |

## Commands

On the build machine, with the branch's rig tarball (`llama.cpp-20261120-37c05b76a-pascal-p106-v5-kcol-ornith.tar.gz`,
md5 `ade8ec94ae8db18ca3a2ff28aaf0e6e1`, built by C899 with `TESTS=1`) and the base's (`3dfad876f`, md5
`0b5b983e22d7296ff44fe363d55f61bb`) unpacked, this repository copied to `$REPO`, and the cards checked with a short
CUDA allocation and copy on each first:

```bash
env WORK=$WORK NEW=$NEW_TREE/bin REF=$BASE_TREE/bin REPO=$REPO ORNITH=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
    MOUNTS="<the folders holding the builds and the model>" DOCKER_ARGS="--cpu-shares=256" \
    setsid nohup bash $REPO/docs/data/2026-11-23_c953_ornith_kcol_gpu/run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
# after it wrote $WORK/DONE (bops 15:33-15:35, load 15:35-15:46):
env WORK=$WORK NEW=$NEW_TREE/bin REF=$BASE_TREE/bin REPO=$REPO ORNITH=... STEPS=kernels \
    setsid nohup bash $REPO/docs/data/2026-11-23_c953_ornith_kcol_gpu/run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
# then, on the output copied back:
python3 -I run/tables.py $WORK .
```

Every container runs `nvidia/cuda:12.9.2-devel-ubuntu24.04` with the PTX JIT's cache in `$WORK/jitcache`. The
`test-backend-ops` runs use card 0; the loads all three.

## Provenance and licence

The model file is Ornith-1.5-122B Pascal N2P GGUF (valid1), the release file of
[release/ornith-v5](../../../release/ornith-v5/settings.env). The prompts are this project's own. The tables are
derived from the runs' logs and replies and published under this repository's licence.
