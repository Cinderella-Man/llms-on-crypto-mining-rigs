# C820: the last MTP drafting pass copies no hidden row, at any draft length

Registry: [Q1434](../../../quants/experiments.md) (C820), from Q1424 (C815). The rig check is Q1453 (R147), not run yet.
Also does what Q1377 (C788) planned: mark the last pass by C744's slot cap.

## What it is

Every MTP drafting pass copies its hidden row (40 KB for Flash-Next) from the draft card to the host, so that the next
pass can read it. The last pass's row is never read. C742's `LLAMA_MTP_SKIP_LAST_ROW=1` leaves that copy out, but only
for a pass after which every drafting sequence has `--spec-draft-n-max` tokens (est. 0.19 ms a step on a P106).

Under C815's `LLAMA_SPEC_ADAPT_ONLINE` most drafts stop earlier, so `=1` rarely applies. **`LLAMA_MTP_SKIP_LAST_ROW=2`**
(fork `pascal-p106-v4-b12246-c820`, off by default; `=1` is unchanged) leaves the copy out whenever every drafting
sequence stops after the pass, whatever the pass samples:

- at its `n_stop`: `n_max`, or the slot's cap under C744's `LLAMA_MTP_DRAFT_CAP` (a reply's end);
- under `LLAMA_SPEC_ADAPT_ONLINE`, at `n_last`, the last length the rule can keep within `n_max` and the slot's cap;
- under `LLAMA_SPEC_ADAPT_ONLINE`, where the rule stops after any next token (`common_spec_adapt_stops_next`, the
  look-ahead). It asks `common_spec_adapt_continue` with the next token at probability 1. With S fixed or run, every
  test and bound the rule makes after the drafted prefix only grows with that token's probability, so "stop at 1" means
  "stop at any probability". With `LLAMA_SPEC_ADAPT_S` per step (S_STEP) a passing test also raises the next
  threshold, so the look-ahead says no there.

A stop the rule decides from the pass's own token, when the look-ahead cannot settle it, is known only after the pass:
that pass copies its row as before. The graph, the drafts, the verify passes and the replies are the same as off. The
drafter's closing line (WARN) counts the result:

```
C820: <steps> drafting steps, <n> hidden rows copied to the host, <n> left out, <n> of them below n_max, <n> of those by the rule's look-ahead
```

Code: `common/speculative.cpp` and `.h` (the drafter's pass loop, `common_spec_adapt_stops_next`), and
`tests/test-spec-adapt.cpp`. Host code only.

## The checks (quant machine, CPU only)

Driver `run/checks.sh` (from C815's), every step in a container at `--cpu-shares=256`, `nice -n 19`, under the shared
build lock. Base: `pascal-p106-v4` `2144cf760`; tip: `b86f1b7fc`.

| Step | Result |
|---|---|
| CPU build of the tip (every target), base's llama-server | OK, OK (3 warnings, all in `common/sampling.cpp` as before) |
| `test-spec-adapt` ×2 | 673,134/673,134 both, the same output |
| server check, 24 arms (`run/server_check.sh`) | PASS (below) |
| test-llama-archs against the base's logits: no switch; `LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_ONLINE=2 LLAMA_MTP_SKIP_LAST_ROW=2` | 146/146, 146/146 |
| rig tarball (sm_61, no AVX) | `llama.cpp-20261113-b86f1b7fc-pascal-p106-v4-b12246-c820.tar.gz`, 678,636,571 B, md5 `285b98efec53c08f398d2ea20171df59` (0 AVX instructions, GLIBC_2.38; `llama-server` md5 `f5e75ef330a0cd7c1bf63a1b8fd7a66e`); the base `2144cf760`'s beside it, md5 `35456bc301c226cfcbbae68f84d66356` |
| SASS / PTX of `llama-server` against the base's | the same files (8,540 functions each) |
| ctest static / shared (C565's set) | 72/72, 72/72; the network test, archs (146 files written and compared) and the hand tests OK in both |
| switch list | 870 names, `LLAMA_MTP_SKIP_LAST_ROW` in it |

**`test-spec-adapt`** keeps C815's 20,000 random drafts (the same cases: C820's draws have their own random stream) and
adds, before every drafting pass of every draft at `online` 1 and 2: `stops_next` equals "continue is false at
probability 1" with S fixed or run, and false with S_STEP; when it says the pass is the last, the drafter does stop
right after it, and `continue` is false at seven probabilities (0, 1, the drafted one, a random one, a random one below
0.001, 7 and −1, which the rule clamps). Over the 20,000 drafts the last pass is known before it runs (`n_max`, the last
allowed length or the look-ahead) in 12,188 drafts at `online` 1 and 7,828 at `online` 2; the look-ahead says "last"
before 9,742 of 63,517 passes and 6,050 of 49,010.

**The server check** (`run/server_check.sh`) uses C815's random-weight model, requests and cost tables: G1 and G2
greedy, S1 and S2 at temperature 1, `--spec-draft-n-max 6`. Each group runs the skip off (its reference), `=1` and
`=2`; `dbase` is the base's llama-server with the skip off. Every arm gave the reference's tokens, top probabilities and
draft counts on all four requests, the same verify steps and the same rule steps with the drafts themselves (17 arms
equal; `server/verdict.txt`). The copies, from the drafter's closing lines (`server/copies.tsv`):

| Group | Settings | Drafts | `=1`: copied, left out | `=2`: copied, left out | below n_max | by the look-ahead |
|---|---|---|---|---|---|---|
| d | default widths, S fixed, `ONLINE=2` | 407 | 1,332, 130 | 1,318, 144 | 14 | 0 |
| w | every width, S fixed, `ONLINE=2` | 407 | 954, 0 | 876, 78 | 78 | 69 |
| r | every width, S run, `ONLINE=2` | 407 | 949, 0 | 872, 77 | 77 | 68 |
| b | as d, the backend sampler | 407 | 1,332, 130 | 1,318, 144 | 14 | 0 |
| p | as d, probabilistic drafting | 408 | 1,114, 70 | 1,100, 84 | 14 | 0 |
| v | C797's arm (FN82's est. v4 table, S run) | 408 | 812, 0 | 404, 408 | 408 | 396 |
| n | no rule: every draft at n_max | 407 | 2,035, 407 | 2,035, 407 | 0 | 0 |
| c | no rule, C744's cap | 407 | 1,995, 387 | 1,975, 407 | 20 | 0 |

In every group the two arms run the same passes (copied plus left out equal), and `=2` leaves out exactly its rows below
n_max more than `=1`. Without the rule (n) `=2` is `=1`; with C744's cap (c) it also leaves out the last row of the 20
capped drafts. On this model the drafter puts about 0.10 on each drafted token, so in d, b and p a token at probability
1 would pass the next test, and the look-ahead never settles a stop there: only the 14 drafts that reach a reply's end
gain. With every width (w, r) and in C797's arm (v, where the rule keeps nothing) it settles most stops. How often it
does on Flash-Next's drafts, whose probabilities are far higher, is what R147 measures.

## The merged tree

`pascal-p106-v4-b12246-c820` `b86f1b7fc` is merged `--no-ff` into `pascal-p106-v4` as `2a3d6bf0d` (on C793's
`eb204a388`). The merge adds exactly the branch's diff: the same 3 files, 122 lines in and 9 out. On a static CPU
build of `2a3d6bf0d`:

- `test-spec-adapt` passed 673,134 of 673,134 checks;
- test-llama-archs gave 146/146 the base's logits, with no switch and with the switch on;
- the server check's d, w, n and c groups passed, with 9 arms equal to their reference and the same copies as on the
  branch (`merged/`).

That server run is the second one. The first passed only the d, n and c arms, and lacked the `=1` arms of n and c. Its
look-ahead total was then 0, since that total comes only from w, r and v on this model, so the check failed on its own
coverage. Every equality passed and the counts were the same. The check now requires the look-ahead only when w, r or v
runs, and the merged run has w and every `=1` arm. In `checks.tsv` the second run's `pascal-p106-v4` row names
`80c94fe0d`, the branch's tip by then (C794 merged on top); its binaries were the `2a3d6bf0d` build.

## The rig check (R147, Q1453, not run)

One served pair on a P106, untimed: R86's loads (C615's `rig_loads.sh`) with the Flash-Next v3 preset plus C815's
`LLAMA_SPEC_ADAPT_ONLINE=2` on a draft of up to 6 tokens (`run/r147_preset.env`: FN82's est. v4 table,
`LLAMA_SPEC_ADAPT_DRAFT_MS=21.87`, `--spec-draft-p-min 0`), loads `trace traces2 prof s2 prof-2 s2-2`, the skip off in
`trace`, `prof` and `prof-2` and `=2` in the others. It must show: every load's replies the skip-off load's; the C820
line's rows left out, below n_max and by the look-ahead, by value; and ms a step of `s2` and `s2-2` against `prof` and
`prof-2` (S37's `arms.py`), and of `traces2` against `trace` stage by stage (R86's `step_gap.py`), each by value.

## Files

| File | What |
|---|---|
| `checks.tsv` | the chain's summary: every step, verdict and detail |
| `server/copies.tsv` | per server arm: steps, tokens drafted, kept and accepted (the rule's log); C742's and C820's closing lines; C815's stops |
| `server/verdict.txt` | the server check's verdict lines |
| `merged/copies.tsv`, `merged/verdict.txt` | the same on the merged tree `2a3d6bf0d` (d, w, n and c groups) || `run/checks.sh` | the chain (build, unit, model, server, archs, rig tarball, switch list, SASS and PTX, ctest, merged) |
| `run/server_check.sh` | the server check |
| `run/r147_preset.env` | R147's preset |

## Commands

On a build machine with Docker, the images `pq-c565-cpu:24.04` (CPU build) and `pq-c735-srv:24.04` (with curl and jq),
and a llama.cpp clone with the fork as remote `fork`:

```bash
M=<work dir> REPO=<this repository> LOCK=<a flock file> BASE_COMMIT=2144cf760 SEED_SRC=<the clone> \
  JOBS=8 THREADS=8 bash docs/data/2026-11-13_c820_online_skip_row/run/checks.sh
# after the merge into pascal-p106-v4:
STEPS=merged MERGED_STEPS="build off server" M=... REPO=... LOCK=... SEED_SRC=... bash docs/data/2026-11-13_c820_online_skip_row/run/checks.sh
```

Provenance: the random-weight model is test-llama-archs' own (`-o --mtp`), built in the run; no external data.
