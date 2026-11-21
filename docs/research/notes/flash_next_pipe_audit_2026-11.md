# Flash-Next's pipelined results and the stale conv history (2026-11)

**In short.** From 2026-11-05 on, pipelined decoding on Flash-Next gave wrong text: every pipe slot read a stale copy
of the conv window's history (Q1670). This audit found no measured result for Flash-Next that rests on that fault.
The fault was spotted by reading on the day it came (Q1042), so nobody ran Flash-Next's pipe on a card again until it
was fixed. Every number for Flash-Next's pipe comes from 2026-11-03, on a build from before the fault, and stands.
One line of work opens again, because without the fix it could only have produced wrong text: guessing the pipe's
tokens with Flash-Next's own MTP head. It gets two new rows, Q1675 and Q1676.

Registry row: Q1674 (FN124), desk work, 2026-11-21.

## The fault

Swift Qwen3.8-Flash-Next is the `qwen4exp` architecture: Gated DeltaNet layers, each with a short conv window over
the last tokens' inputs. Pipelined decoding (`--pipe-depth`, Q550) gives each pipe slot its own group of recurrent
states, so a wrong guess can be dropped. The fused conv window (`LLAMA_GDN_FUSE=1`, the default since `-gdnfuse`,
Q438) copied the window's new history to state group 0 whatever the slot. The pipelining branch had put the slot's
group only into the unfused path. The two branches met at the merge `e3b30303d` (2026-11-05). From there on, v3.0,
v4.0 and all of v5 carry it.

On random test models the logits moved by at most 3.027e-05 and no greedy token changed. On the real file, Swift
GSQ-RCO Q2_0, the pipelined replies left plain decoding's within four words and turned to noise. The fix is six lines
on the fork branch `pascal-p106-v5-pipe-logits` (`80d83fa1a`). With it, every pipelined job on the real file gives
plain decoding's tokens on the CPU. The check on the cards is Q1672. Plain decoding, Flint, Ornith and the other
hybrid architectures were never affected
([C932's data](../../data/2026-11-21_c932_pipe_logits/README.md)).

## How the results were found

A result counts when it ran a Flash-Next-type model (any Flash-Next file, or a random `qwen4exp` test model) with pipe
slots (`--pipe-depth`, `llama-pipespec`, `llama-pipe-recurrent-check`, `--pipe-guess mtp`) and the fused conv window
on, from 2026-11-05 on. The search covered the registry, every data folder from 2026-11-04 on, the research notes,
the release notes and the book, plus the scripts of every rig and CPU run of that period. Most mentions of a
"pipeline" there are llama.cpp's own overlap of micro-batches across the cards, which uses no pipe slot. Flint's
`--pipe-depth 4` runs are another architecture.

## The results and their verdicts

| Result | When, build | What ran | Verdict | What the verdict becomes |
|---|---|---|---|---|
| Q550: the pipe for Flash-Next built and checked ([data](../../data/2026-11-03_flash_next_pipelined/README.md)) | 2026-11-03, `f377294d6` | 73 pipelined runs on a random `qwen4exp` (CPU, two virtual devices), all identical to plain decoding | not affected: before the merge | stands |
| Q551 step 1, the pipe in the agent check ([data](../../data/2026-11-03_flash_next_pipelined_rig/README.md)) | 2026-11-03, `f377294d6` | Swift Q2_0 on nine P106s: 13 exact pipelined runs, then depth 3 against depth 0 in the agent check: 4.9 against 7.9 tok/s, 29.7 % of the tokens right guesses and 39.9 % wrong, +372 ms a wrong guess, memory at depth 3 | not affected: before the merge | stands, and so do the rows priced on it: Q593, Q594, Q595, Q162 (parked until Q593), Q554, Q555, Q689, Q1109, Q562 |
| Q573: `-hybrid` on the b12076 base | 2026-11-03, `8c25952e4` | 29 pipelined runs on a random `qwen4exp`, identical | not affected: before the merge | stands |
| Q962: the cumulative build `e3b30303d` | 2026-11-05 | the merge where the fault entered; its checks ran no pipe | nothing measured on the pipe | the merge is now named as the fault's start |
| Q1042: the bug found by reading | 2026-11-05 | no run | the same fault | closed as Q1670 |
| v3.0's known issues ([notes](../../../release/notes-v3.0.md)); v4.0 says they stand | 2026-11-06, 2026-11-16 | no run | correct as written: pipe off in Flash-Next's preset, `LLAMA_GDN_FUSE=0` as the workaround | true until the fix lands |
| Pull request 7's review: `--pipe-guess mtp` (Q1610) | 2026-11-20, `d45edef2c` | `--spec-type draft-mtp --pipe-depth 2 --pipe-guess mtp` on a random `qwen4exp` with an MTP layer, fused conv on | **suspect for Flash-Next**: on random weights the fault changes no greedy token, so the MTP-guessed pipe was never shown exact on a real Flash-Next file | the review's other verdicts stand (head against base, the hang that pull request 3 fixes); a run on the real file on the fix: Q1675 |
| Pull request 7's and C908's pipe with `LLAMA_PIPE_GRAMMAR` (Q1612, Q1645) | 2026-11-20, 2026-11-21 | `--pipe-depth 3` on a random dense llama model | not affected: another architecture | stands |
| C880: the fork's tools on the new batch API (Q1589, [data](../../data/2026-11-21_c880_batch_api/README.md)) | 2026-11-21, `3dfad876f` and `aebff903e` | `check_identity.sh`, `check_streams.sh` and `llama-pipe-recurrent-check` on a random `qwen4exp`, old binaries against new | not affected: both builds carried the fault, and "the same before and after" compares them | stands; it is the run that found the fault |
| Flint's agent checks and checks with `--pipe-depth 4` (R159 to R216, C901, C903) | 2026-11-05 to 2026-11-21 | Flint's files | not affected: another architecture, the same at every point C932 checked | stand |
| The races (Q93, Q1195, Q1271) | 2026-11-10 to 2026-11-16 | several whole samples at once | not affected: no pipe slots | stand |
| Flash-Next's presets v3 to v5 | | the MTP draft, no pipe | not affected | stand |

Two pages repeat these numbers and stand as written: the [Qwen-V100 note](qwen_v100_llamacpp.md) (§6.1: "lost 1.6×")
and chapters [8](../../book/08-guessing-ahead.md) and [9](../../book/09-the-models.md) of the book. The
[Flash-Next review](flash_next_review_2026-11.md) now names the fault and its fix where it gives its reasons.

## What the verdicts become

Nothing parked on these numbers reopens because of the fault: they were measured before it. What changes is that the
pipe on Flash-Next gives the right text again with the fix. Any of those rows needed that to run on today's engine.
One idea was blocked outright: the MTP head as the pipe's guesser. Pull request 7 brought it as `--pipe-guess mtp`,
and its only check was on random weights.

## The reruns, ranked by what they could reopen for Flash-Next's agentic speed

Each runs after Q1672 has confirmed the fix on the P106s. That row also runs Swift GSQ-RCO Q2_0 pipelined at depths 1
and 3 as Q550 did, so it repeats Q551's exactness step on the fix.

1. **Q1675: the MTP head as the pipe's guesser on the real file, nothing timed.** C932's build holds pull requests 3
   and 7. Flash-Next's v5 preset plus `--pipe-guess mtp` at depths 1, 2 and 3, as far as they load at 65,536 tokens,
   then with the guess chain kept on a hit (`--pipe-mtp-walk`) and the pipe kept under a grammar
   (`LLAMA_PIPE_GRAMMAR=1`). It records each card's memory, whether the reply at temperature 0 is the preset's, the
   guesses right and wrong, and one reply's speed as a diagnostic. It ranks first because it is the only pipe line that
   could beat the MTP draft's verify. For it: Q554's event model gave est. 2.30× at depth 3 over one-token decoding for
   a guesser right 70 % of the time (2026-11-03), and the contributor's MTP guesses were right 1,092 of 1,904 on their
   plain text. Against it: a wrong guess cost 372 ms at depth 3 (Q593). And on v3.0's release line a 4-token verify
   took 100.2, 101.6, 105.5 and 106.3 ms in four loads, against 51.1 ms for a plain token
   ([review](flash_next_review_2026-11.md)).
2. **Q1676: its agent pair**, if Q1675 passes and is not slower: the v5 preset with Q1675's best line against the
   preset, two runs each, alternated. The preset's two control runs in
   [R227](../../data/2026-11-20_r227_flash_taper1400_agent/README.md) made 31.05 and 31.46 tok/s.
3. **Q593: where a wrong guess's time goes**, profiled on today's engine with the fix. Its answer decides whether any
   Flash-Next pipe pays, MTP-guessed or not.
4. **Q595 and Q594:** the `cont` guesser at depths 1 and 2, and the pipe under a tool-call grammar (now pull request
   7's `LLAMA_PIPE_GRAMMAR=1`, Q1612). These rank last: `cont` got 29.7 % of the tokens right in Q551, and the preset's
   draft already beats one-token steps.
