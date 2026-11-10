# C675 (Q1187): probabilistic verification stops at the end of a turn (2026-11-10, the night's code rows)

With `--spec-draft-sampling probabilistic` and a temperature above 0, llama-server verifies a draft by token-by-token
rejection (`common_sampler_sample_and_accept_n_rejection` in `common/sampling.cpp`). That path has no stop at an
accepted end-of-generation token. The preset's path, sample-and-match (`common_sampler_sample_and_accept_n`), has had
one since upstream's [#29638](https://github.com/ggml-org/llama.cpp/pull/29638). So with probabilistic verification
the drafted tokens behind the end of a turn are still verified. They are never output, but the main sampler's token
for the row after the end is checked against the tools' grammar, fails it, and is drawn again from the whole
vocabulary on the host. In [FN51's loads](../2026-11-10_prob_verify_reply_length/README.md) 33 and 29 of 60 replies
had exactly one such rejection, and the preset's replies none. On a P106 rig each one costs 389.65 ms
([R38](../2026-11-10_prob_verify_rig/README.md)).

This folder holds the change, behind a new switch on Miner Llama's v4 line, and its CPU check. The registry row is Q1187
in [quants/experiments.md](../../../quants/experiments.md); the switch is listed in
[docs/switches.md](../../switches.md).

**Result:** the switch `LLAMA_SPEC_REJECT_STOP=1` gives the rejection path the match path's stop. On the CPU, the
unit test passes with the switch at its default and with it on from the environment. In 4,000 random drafts the
switch's result is the default's cut right after the first accepted end-of-generation token, token for token. 1,148 of
the drafts were cut and 2,852 unchanged. With the switch off nothing changes: upstream's own test of the rejection path
prints the same distribution, line for line, from the base build, from this one, and from this one with the switch on.
The rig tarball builds (0 AVX instructions). The stop's effect on real loads, the grammar rejections FN51 counted, is
not measured yet (Q1262).

## The change

Branch `pascal-p106-v4-b12246-c675` at **`b2763cce7`**, one commit on `pascal-p106-v4` (`84f7bd3d2`, the
`-ladder6` tree):

| File | What |
|---|---|
| `common/sampling.cpp`, `common/sampling.h` | `LLAMA_SPEC_REJECT_STOP`, read once (`common_spec_reject_stop()`; a setter for tests). With it at `1`, after a drafted token is accepted the rejection path stops if the token is an end-of-generation token and not the last drafted one, or if the screened head meets a grammar that has become active. These are sample-and-match's own conditions, copied |
| `tools/server/server-context.cpp` | one log line at the server's start when the switch is on |
| `tests/test-spec-reject-stop.cpp`, `tests/CMakeLists.txt` | the unit test, registered twice: the switch at its default, and on from the environment |

**With the switch off nothing changes:** the stop is one `if` after the accept, behind the switch. No other line of the
path moves.

**The three stops of sample-and-match, and the two taken over:**

1. **After an accepted end-of-generation token that is not the last drafted one.** This is the stop the row is about.
   The step then outputs fewer tokens than it verified. The server rolls back the rest, as it does when the match path
   stops at the same token: from the recurrent target's rollback snapshots when the draft fits them, from the step's
   checkpoint otherwise.
2. **Where the screened head meets an active grammar** (`LLAMA_SCREEN_HEAD_GRAMMAR=1` with a model that has an
   `output_screen`). The batch was computed with the screened head while the grammar could not reject a token. Once
   it can, the later rows need the full head, or the grammar's tokens outside the screen get no probability. This
   condition does not occur on the Flash-Next preset; it is taken over because the rejection path is otherwise wrong
   there.
3. **After a device resample: left out.** In the match path the token resampled on the host is output, and the device
   drew the later rows from the same stream positions, so the two tokens would share a random number. In the rejection
   path the resampled token is output only when the step ends at that row (a rejection whose residual has no mass).
   Every later row's output comes from the drafted token or the sampler's own generator, so stopping there would only
   throw accepted tokens away.

**A trailing end-of-generation token still takes the next row,** as in sample-and-match: when the last drafted token
is the end of the turn and is accepted, the bonus row is sampled. Sample-and-match keeps that row for its checkpoint
replay, where the last token comes from the target. The rejection path is never called on a replay (the server
re-accepts a replayed draft with `server_accept_replay`), so it could stop there too. It does not, so that the two
paths stop at the same tokens and the count on the cards can be read against the preset's.

## The CPU check

`run/cpu_check.sh` on a build machine, CPU only, every test in a container without network. It builds the branch and
the branch it was cut from (static CPU builds, `GGML_NATIVE=ON`), then:

- **ctest** on the branch: `test-spec-reject-stop` (the switch at its default), `test-spec-reject-stop-env`
  (`LLAMA_SPEC_REJECT_STOP=1`), `test-speculative-rejection-sampling` (upstream's test that the rejection path
  preserves the target distribution) and `test-sampling`.
- **identity:** `test-speculative-rejection-sampling` of the base and of the branch, the branch once more with the
  switch on, 20,000 trials each on the same model file: every printed probability the same. Its drafts hold one token,
  which the switch never cuts.

`test-spec-reject-stop` runs on the dummy llama model that `test-llama-archs` writes (`-a llama -s 999`, 128 tokens).
Its vocabulary has no end-of-generation token, so the test sets its EOS to token 7 with a metadata override, and a
logit bias lifts token 7 to the top of the target's ten candidates (top-k 10, temperature 1). Every check compares the
switch on and off from one sampler state:

| Check | Drafts | What must hold |
|---|---|---|
| fixed | six drafts the target accepts for certain (q(x) = p(x)/2): the EOG first, second, last, alone, twice, nowhere | off: the whole draft and one more token; on: cut right after the first EOG that is not the last drafted token, the same tokens up to there |
| random | 4,000 drafts of four tokens drawn from q = p/2 + noise/2 over the target's candidates | on = off cut after the first accepted EOG before the last drafted token, else on = off, token for token; at least 5 % of the trials cut, 5 % unchanged and 5 % ending in a residual draw |
| default | — | `common_spec_reject_stop()` is off with the variable unset and on with it at `1` |

### Results

| Step | Verdict | Detail |
|---|---|---|
| CPU build, branch | OK | 0 error lines; 3 warning lines, all `-Wmissing-field-initializers` on `common_sampler::stats` in code this change does not touch |
| ctest, branch | OK | 5 of 5: `test-sampling`, `test-generate-speculative-target`, `test-speculative-rejection-sampling`, `test-spec-reject-stop`, `test-spec-reject-stop-env` (run with `LLAMA_SPEC_REJECT_STOP=1`, as its log shows) |
| CPU build, base | OK | 0 error lines, the same 3 warning lines |
| identity | SAME | `test-speculative-rejection-sampling`, 20,000 trials: the 128 tokens' target and empirical probabilities the same in all three runs, and each run passed (every empirical probability within six standard errors of the target) |
| rig tarball | OK | `llama.cpp-20261110-b2763cce7-pascal-p106-v4-b12246-c675.tar.gz`, 647,374,693 bytes, md5 `534a6f2c49c165e6b057de224c32fcd7`; sm_61, `TESTS=1`, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most |

The fixed drafts ([cases.tsv](cases.tsv); token 7 is the end of generation, the last token of each result is the
row after the draft or the token after the stop):

| Case | Draft | Switch off | Switch on |
|---|---|---|---|
| EOG first | 7 22 119 | 7 22 119 22 | 7 |
| EOG second | 22 7 119 | 22 7 119 22 | 22 7 |
| EOG last | 22 119 7 | 22 119 7 22 | 22 119 7 22 |
| EOG alone | 7 | 7 22 | 7 22 |
| no EOG | 22 119 96 | 22 119 96 22 | 22 119 96 22 |
| EOG twice | 22 7 7 119 | 22 7 7 119 7 | 22 7 |

**The random drafts** (4,000, four tokens each), the same in both test runs: 1,148 cut after an accepted end of
generation, 2,852 unchanged, 1,675 of all 4,000 ending in a rejection and a residual draw. Behind the 1,148 cuts the
default's results hold 536 more end-of-generation tokens, from rows it verified that a reply never shows.

## What it decides

- **The switch is ready for the count on real loads** (Q1262): FN51's two chains with a third arm, probabilistic
  verification with the switch on. It is right if the grammar rejections fall to the preset's count (0 in FN51)
  and tokens a step stay within the spread of two loads.
- **If the count holds, the stop goes with probabilistic verification** into the v4 preset candidate (Q1254's five
  switches). The rig's estimate is +0.6 points on probabilistic verification's +2.97 % in generation tok/s
  ([R38](../2026-11-10_prob_verify_rig/README.md): 13 rejections a load at 389.65 ms each in about 911 s of
  generation). It is timed in R38's pair or in the agent check of the stacked line (Q1261).
- **The stop is exact:** every row up to the end of the turn is decided as before, and nothing behind it is output.
  The same result comes from a draft that ends at its first end-of-generation token.

## Limits

1. **No GPU and no model:** the unit test runs the sampler path on a dummy model. That the stop removes FN51's grammar
   rejections, and that tokens a step do not change, is a count on real loads: Q1262, the same chains as FN51 with a
   third arm.
2. **The screened-head stop is not exercised:** the test model has no `output_screen`. Its condition is sample-and-
   match's, copied.
3. **A step that stops at the end of a turn and has to restore its checkpoint** (a draft longer than the recurrent
   target's snapshots) re-accepts the step through `server_accept_replay`, which samples the row after the end once
   more. The Flash-Next preset's drafts fit the snapshots (no restore inside a step in FN43's 33,564 steps), so this does
   not occur there.
4. **Block verification** (`LLAMA_SPEC_BLOCK_VERIFY=1`) walks the whole block, the rows behind an end of turn
   included, and is not changed here: Q1263.

## Files

| File | What |
|---|---|
| [run/cpu_check.sh](run/cpu_check.sh) | the CPU check: both builds, ctest, the identity runs |
| [checks.tsv](checks.tsv) | the CPU check's verdicts, as `cpu_check.sh` wrote them |
| [cases.tsv](cases.tsv) | the unit test's fixed drafts and what each shows |

The test itself is in the fork: `tests/test-spec-reject-stop.cpp` on `pascal-p106-v4-b12246-c675`.

## Commands

`WORK` is a work folder on a build machine with Docker, `REPO` a checkout of this repository, `REMOTE` the fork's git
remote name in `$WORK/src` (an https fetch URL).

```bash
# the rig tarball (CUDA 12.9 for sm_61, test-backend-ops included)
WORK=$WORK TESTS=1 JOBS=8 DOCKER_ARGS="--label pascal-quants=1 --cpu-shares 256" \
  nice -n 19 bash $REPO/scripts/build_llamacpp_docker.sh $REMOTE/pascal-p106-v4-b12246-c675

# the CPU check (both trees from the clone the build fetched into)
WORK=$WORK SRC=$WORK/src TIP=b2763cce7 BASE=84f7bd3d2 JOBS=8 THREADS=4 N_TRIALS=20000 \
  nice -n 19 bash $REPO/docs/data/2026-11-11_c675_end_of_turn_stop/run/cpu_check.sh
```

On the build machine both ran one after the other under a lock shared with the night's other builds
(`flock <lock> <command>`), at the lowest CPU priority, with no GPU.

## Provenance

The change and the test are Miner Llama's own (MIT, as llama.cpp). The test model is generated by the fork's
`test-llama-archs` with random weights; no model file, text or data is downloaded.
