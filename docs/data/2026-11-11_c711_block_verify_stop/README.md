# C711 (Q1263): block verification stops at the end of a turn (2026-11-11, the night's code rows)

With `LLAMA_SPEC_BLOCK_VERIFY=1` ([Q797](../2026-11-04_block_verify/README.md)) llama-server verifies a sampled draft
as one block (`common_sampler_sample_and_accept_n_block` in `common/sampling.cpp`; Sun et al.,
[arXiv:2403.10444](https://arxiv.org/abs/2403.10444)). To weigh the block, the function walks every drafted row and
accepts each drafted token into the sampler on the way. So a drafted end-of-generation token inside the block is
followed by rows that are never output. With the tools' grammar the main sampler's token for the row after the end
fails the grammar and is drawn again from the whole vocabulary on the host: the cost that
[C675](../2026-11-11_c675_end_of_turn_stop/README.md) removed from token-by-token rejection, 389.65 ms each on a P106
rig ([R38](../2026-11-10_prob_verify_rig/README.md)).

This folder holds the change, behind C675's switch on Miner Llama's v4 line, and its CPU check. The registry row is
Q1263 in [quants/experiments.md](../../../quants/experiments.md); the switch is listed in
[docs/switches.md](../../switches.md).

**Result:** with `LLAMA_SPEC_REJECT_STOP=1` block verification now ends its block at the first drafted
end-of-generation token. On the CPU the extended unit test passes with the switch at its default and with it on from
the environment. In 4,000 random drafts the switch's result equals the default's on the cut draft, token for token,
without the next token when the cut block is accepted whole. 1,356 of the drafts were cut, and 2,644 had nothing to cut.
Under a grammar that only an end of generation completes, the switch samples one row instead of two, in both paths.
With the switch off nothing changes: the digest of the switch-off block results over the 4,000 drafts is the same as
from C675's tree. The exact enumeration gives the target's output distribution with the cut, to 1.17e-07. The rig
tarball builds (0 AVX instructions). The cut's effect on real loads is not measured yet (Q1291, FN69).

## The change

Branch `pascal-p106-v4-b12246-c711` at **`ca3c0b4fd`**, two commits on C675's `pascal-p106-v4-b12246-c675`
(`b2763cce7`, itself one commit on `pascal-p106-v4` `84f7bd3d2`):

| File | What |
|---|---|
| `common/sampling.cpp` | the block function's body becomes `spec_block_verify(..., next)`, unchanged but for one `if`: with `next` false, a block accepted whole takes no next token. The public function finds the first drafted end-of-generation token before the last drafted one when `LLAMA_SPEC_REJECT_STOP` is on, and verifies the draft cut after it, with `next` false. Otherwise it calls the body with the whole draft and `next` true |
| `common/sampling.h` | the switch's comment names the block path |
| `tools/server/server-context.cpp` | a second log line at the server's start when the switch and block verification are both on |
| `tests/test-spec-reject-stop.cpp` | C675's unit test, extended to the block path (below) |
| `tests/test-spec-block.cpp` | two more configurations of the exact enumeration, with the cut |

**C675's switch, not a new one.** `LLAMA_SPEC_REJECT_STOP=1` already means "verification of sampled drafts stops at an
accepted end of generation, as sample-and-match does". With block verification on, both sampled-draft paths serve the
same run: block verification for the requests it takes, token-by-token rejection for the requests with backend
sampling. One switch turns the stop on for both, and a preset needs one line, not two. The cost is that the block
cut cannot be measured apart from the rejection stop in a run where both paths are used; with block verification on
and no backend sampling only the block path runs, so the switch measures the cut alone. The Flash-Next preset runs
with backend sampling (`-bs`), so there block verification is never used and the switch acts through the
token-by-token stop alone: on real loads `LLAMA_SPEC_BLOCK_VERIFY=1` gives the same counts as without it
([2026-11-11_fn69_fn66_stops](../2026-11-11_fn69_fn66_stops/README.md), Q1291).

**With the switch off nothing changes:** the function calls the old body with the whole draft and `next` true, which
is the old function line for line.

**What the cut does:** for a draft `x1 … xg` whose first end-of-generation token is `xe` with `e < g`:

1. The walk covers `x1 … xe` only. It stops at the row of `xe` without accepting `xe` into the sampler, as it always
   stops at the last drafted row. No row behind the end is sampled, so none is checked against a grammar.
2. The weights and the backward scan for the accepted length run over the cut block. The draws from the sampler's
   random stream are those of a block of length `e`.
3. If the cut block is accepted whole (the reply ends at `xe`), no next token is sampled. Sample-and-match samples no
   row after a drafted end of generation either, and token-by-token rejection with C675's switch stops there too. If
   the block is rejected at some position before, the next token is drawn from the residual there, as before.

**A trailing end of generation** (the last drafted token) is not a cut: the block is verified whole and, when it is
accepted whole, the next row is sampled. The other two paths do the same, so that the three stop at the same tokens.

## Why the cut is exact

Block verification is exact for any draft whose length is a function of its prefix: the drafter may stop when it
likes, provided it decides from what it has drafted. Cutting a draft after its first end-of-generation token is such
a rule. What is verified is therefore a valid draft, and the accepted tokens and the next token follow the target's
distribution. Taking no next token after an accepted end of generation only outputs fewer tokens: whatever the target
would say after the end of a turn is never shown. `test-spec-block` checks both by exact enumeration: the distribution
of the first `g + 1` output tokens, the target's own continuation after a cut block included, equals the target's
(below).

## What is not taken over

1. **The screened-head stop** (`LLAMA_SCREEN_HEAD_GRAMMAR=1` with a model that has an `output_screen`), which the other
   two paths have: once a lazy grammar is triggered inside a screened batch, the later rows need the full head. The
   block walk goes on through those rows. It is a separate fault of block verification (the walk weighs rows the
   screen cut short), not an end-of-turn one, and does not occur on the Flash-Next preset, which has no
   `output_screen`: Q1292.
2. **The device-resample stop** of sample-and-match: block verification is never used with backend sampling (the
   server keeps token-by-token rejection for those requests), so no row of the walk was drawn on the device.
3. **The checkpoint replay**: a step that ends at the end of a turn and has to restore its checkpoint (a draft longer
   than the recurrent target's rollback snapshots) re-accepts its tokens through `server_accept_replay`, which samples
   the row after the end once more. This is C675's limit 3, unchanged.

## The CPU check

`run/cpu_check.sh` on a build machine, CPU only, every test in a container without network. It builds the branch and
C675's branch (static CPU builds, `GGML_NATIVE=ON`); C675's tree gets this branch's `tests/test-spec-reject-stop.cpp`
laid over it, so that both build the same test. Then:

- **ctest** on the branch: `test-spec-reject-stop` (the switch at its default), `test-spec-reject-stop-env`
  (`LLAMA_SPEC_REJECT_STOP=1` from the environment), `test-spec-block`, `test-speculative-rejection-sampling` and
  `test-sampling`.
- **identity, block path off:** the extended test on C675's tree and on this branch, 4,000 random drafts each. Every
  result of the block path with the switch off, in order, goes into one digest; one sampler carries its state from
  draft to draft, so the digest covers the sampler's state too. The two digests must be equal. (On C675's tree the
  checks with the switch on fail, as that tree has no cut; only the digest is read there.)
- **identity, enumeration:** `test-spec-block` of both trees prints the same line for each of C675's 11 configurations.

`test-spec-reject-stop` runs on the dummy llama model that `test-llama-archs` writes, with its EOS set to token 7 and a
logit bias that lifts token 7 to the top of the target's ten candidates (C675's set-up). The block path's checks:

| Check | Drafts | What must hold |
|---|---|---|
| fixed | C675's six drafts (q(x) = p(x)/2, every position of weight 1, so the block is accepted whole) | off: the whole draft and one more token; on: the draft up to its first end of generation before the last drafted token, no more |
| random | 4,000 drafts of four tokens from q = p/2 + noise/2, the same drafts as C675's rejection check | on = off on the cut draft without its next token, token for token; on = off where there is nothing to cut; where both end in a residual draw before the cut, the two samplers in the same state (a block verification and eight samples after it, from copies, give the same tokens); at least 5 % of the drafts cut and 5 % not, at least 2 % of them cut and accepted whole and 2 % cut and ending in a residual draw |
| grammar | the draft {EOG, a, b} under the grammar `root ::= ""` (only an end of generation completes it, the tools' grammar after a call), 50 times, both paths | off: EOG, EOG with two rows sampled under the grammar; on: EOG alone, one row; on's grammar rejections below off's |

And `test-spec-block`'s two new configurations: the toy drafter's block ends at token 0 (its end of generation), and a
cut block accepted whole takes no next token; once with dense distributions, once with target zeros, top-2 drafts,
point masses and prefix-dependent lengths together. The output distribution must equal the target's within 1e-6, and
block verification must accept no fewer tokens in expectation than token-by-token rejection.

### Results

| Step | Verdict | Detail |
|---|---|---|
| CPU build, branch | OK | 0 error lines; 3 warning lines, all `-Wmissing-field-initializers` on `common_sampler::stats` in code this change does not touch (C675's three) |
| ctest, branch | OK | 6 of 6: `test-sampling`, `test-spec-block`, `test-generate-speculative-target`, `test-speculative-rejection-sampling`, `test-spec-reject-stop`, `test-spec-reject-stop-env` (run with `LLAMA_SPEC_REJECT_STOP=1`, as its log shows) |
| CPU build, C675's tree with the new test | OK | 0 error lines, the same 3 warning lines |
| identity, block path off | SAME | digest `1404fbd5c5a6a29a` over 4,000 drafts from both trees. On C675's tree the checks with the switch on fail, 1,498 failure lines, as they must where there is no cut |
| identity, enumeration | SAME | C675's 11 configurations print the same lines from both trees; this branch prints 13 |
| rig tarball | OK | `llama.cpp-20261110-ca3c0b4fd-pascal-p106-v4-b12246-c711.tar.gz`, 647,592,506 bytes, md5 `59b233fc0319e6c70790c29cf9e9191c`; sm_61, `TESTS=1`, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most |

The fixed drafts on the block path ([cases.tsv](cases.tsv); token 7 is the end of generation, the last token of each
result with the switch off is the next token after the block). They are the rejection path's results, token for token:

| Case | Draft | Switch off | Switch on |
|---|---|---|---|
| EOG first | 7 22 119 | 7 22 119 22 | 7 |
| EOG second | 22 7 119 | 22 7 119 22 | 22 7 |
| EOG last | 22 119 7 | 22 119 7 22 | 22 119 7 22 |
| EOG alone | 7 | 7 22 | 7 22 |
| no EOG | 22 119 96 | 22 119 96 22 | 22 119 96 22 |
| EOG twice | 22 7 7 119 | 22 7 7 119 7 | 22 7 |

**The random drafts** (4,000, four tokens each, the same in both test runs; [counts.tsv](counts.tsv)): 1,356 drafted an
end of generation before the last position and were cut. Of these, 1,251 cut blocks were accepted whole and 105 ended
in a residual draw before the end; in those 105 the switch's sampler and the default's on the cut draft were left in the
same state. 2,644 drafts had nothing to cut. With the switch off the block results hold 3,728 tokens behind an end of
generation, from rows a reply never shows. The token-by-token checks give C675's counts unchanged (1,148 cut, 2,852
unchanged, 1,675 ending in a residual draw), so the drafts are C675's.

**The grammar case** (50 trials of the draft {EOG, a, b} under `root ::= ""`):

| Path | Rows sampled under the grammar, off / on | First draws the grammar rejected, off / on |
|---|---|---|
| token-by-token rejection | 100 / 50 | 89 / 44 |
| block verification | 100 / 50 | 81 / 41 |

With the switch off the row after the end is sampled in every trial and its first draw is rejected and drawn again
from the whole row in 45 and 40 of the 50 trials. With it on that row is not sampled.

**The enumeration** (`test-spec-block`, 200 instances each):

| Configuration | max \|P_out − P_target\| | E[accepted], block | E[accepted], token | Gap |
|---|---|---|---|---|
| cut at an EOG, g ≤ 4 | 1.02e-07 | 0.8487 | 0.7942 | +6.87 % |
| cut at an EOG + target zeros, top-2 drafts, point masses, prefix-dependent lengths, g ≤ 4 | 1.17e-07 | 0.5407 | 0.5278 | +2.43 % |

## What it decides

- **The cut is ready for the count on real loads** (Q1291, FN69): FN51's chains with block verification, the cut off
  and on. It is right if the grammar rejections after the end of a turn fall to the preset's count (0 in FN51) and
  tokens a step stay within the spread of two loads. The tarball holds C675 too, so the same loads can also take
  FN66's arms.
- **The cut goes with block verification if block verification goes into a preset**, as C675's stop goes with
  probabilistic verification. On its own it changes nothing a reply says.
- **The cut is exact** (above): every row up to the end of the turn is decided as by a draft that ends there, and
  nothing behind it is output.

## Limits

1. **No GPU and no model:** the unit test runs the sampler path on a dummy model, and the enumeration on toy
   distributions. That the cut removes the grammar rejections after the end of a turn on real loads, and that tokens a
   step do not change beyond the spread, is a count on the cards: Q1291.
2. **The screened-head stop is not taken over** (above): Q1292.
3. **The cut can change how often a step accepts the end of a turn.** Without the cut, block verification weighs the
   end-of-generation position with the rows behind it; with the cut, by itself. Both are exact, but the share of steps
   that accept the end may differ. In the random drafts the switch output 12,603 tokens up to the first end of
   generation and the default 12,617, in runs that draw different random numbers, so the two counts are not a paired
   comparison. Tokens a step on real loads are read in Q1291.
4. **Block verification is not in a preset.** It was written and checked on the CPU (Q797, C243), and its rig run
   (C243r) was written for DFlash2 and ran nothing. The cut matters only once block verification is measured on
   Flash-Next's MTP line.

## Files

| File | What |
|---|---|
| [run/cpu_check.sh](run/cpu_check.sh) | the CPU check: both builds, ctest, the two identity checks |
| [checks.tsv](checks.tsv) | the CPU check's verdicts, as `cpu_check.sh` wrote them |
| [cases.tsv](cases.tsv) | the unit test's fixed drafts on the block path and what each shows |
| [counts.tsv](counts.tsv) | the random drafts', the grammar case's and the enumeration's counts |

The tests themselves are in the fork: `tests/test-spec-reject-stop.cpp` and `tests/test-spec-block.cpp` on
`pascal-p106-v4-b12246-c711`.

## Commands

`WORK` is a work folder on a build machine with Docker, `REPO` a checkout of this repository, `REMOTE` the fork's git
remote name in `$WORK/src` (an https fetch URL).

```bash
# the rig tarball (CUDA 12.9 for sm_61, test-backend-ops included)
WORK=$WORK TESTS=1 JOBS=8 DOCKER_ARGS="--label pascal-quants=1 --cpu-shares 256" \
  nice -n 19 bash $REPO/scripts/build_llamacpp_docker.sh $REMOTE/pascal-p106-v4-b12246-c711

# the CPU check (both trees from the clone the build fetched into; BASE is C675's branch)
WORK=$WORK SRC=$WORK/src TIP=ca3c0b4fd BASE=b2763cce7 JOBS=8 THREADS=4 N_TRIALS=4000 \
  nice -n 19 bash $REPO/docs/data/2026-11-11_c711_block_verify_stop/run/cpu_check.sh
```

On the build machine both ran one after the other under a lock shared with the night's other builds
(`flock <lock> <command>`), at the lowest CPU priority, with no GPU. The tarball took 2 minutes (the compiler cache of
C675's build), the CPU check 38 seconds.

## Provenance

The change and the tests are Miner Llama's own (MIT, as llama.cpp). The test model is generated by the fork's
`test-llama-archs` with random weights; no model file, text or data is downloaded.
