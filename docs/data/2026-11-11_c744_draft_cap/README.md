# The MTP drafter stops at the slot's draft cap (C744, Q1314, 2026-11-11)

What this is: a switch in Miner Llama's MTP drafter, `LLAMA_MTP_DRAFT_CAP` (off by default), that stops drafting
where llama-server will cut the draft anyway. Near the end of a reply the server caps each draft at the tokens still
to generate, less one (`get_n_draft_max`: with 3 tokens left a draft keeps 2, with 2 left it keeps 1), and at the
context's end at the room left. The MTP drafter took that cap as a request field but stopped at its own length
(`--spec-draft-n-max`, 3 in the presets), so it ran every pass and the server cut the extra tokens afterwards: those
passes were wasted. [C405](../2026-11-11_c405_server_gap/README.md)'s step trace counted them (9 spare passes in 6 of
202 steps on a CPU, 2 in 179 steps on a P106 rig in [R86](../2026-11-11_r86_step_trace/README.md)). The registry row
is Q1314 in [quants/experiments.md](../../../quants/experiments.md).

Fork branch `pascal-p106-v4-b12246-c744` at `4b0e2730a`, one commit on `pascal-p106-v4-b12246-c748` (`5b329df16`). Not
merged into another branch. Everything here ran on a build machine's CPU; nothing ran on a card.

## Result

**With the switch every spare drafting pass is gone, and the replies, the kept drafts and the verify steps are the
same bytes.** On a build machine's CPU (llama-server, the tiny random qwen4exp model with one MTP block, the draft at
three tokens), four requests per arm: the base and the switch off run 1,218 drafting passes for 1,206 kept draft
tokens, 12 passes whose tokens the server cut, in 8 of 406 steps (the last two steps of each reply). With the switch
on the drafter runs 1,206 passes, none spare, and each arm gives its reference's tokens, top probabilities, draft
counts and all 406 verify steps ([passes.md](passes.md), [server_verdict.txt](server_verdict.txt)).

**On the rig it is small: est. 3.7 ms for each pass left out.** R86 timed a P106's drafting passes 2 and 3 at 3.859
ms (0.896 ms of host work, 2.941 ms waiting for the card, 0.022 ms reading the id) and 3.707 ms (0.862, 2.825, 0.020).
A reply that runs into the cap loses up to three passes: pass 3 with 3 tokens left, passes 2 and 3 with 2 left, est.
11.27 ms in all. R86's greedy request had 2 spare passes in its 179 steps, est. 7.57 ms at most, against est. 23.1 s
for its steps (179 at R86's 128.92 ms), 0.03 %. The cap binds only where a request limits its reply:
- the agent check's replay (`bench/agent_check/replay.py`) sets each request's `max_tokens` to the recorded reply's
  length, so every replayed reply ends in the cap;
- a live agent run sends `max_tokens` 32,768, so a reply that ends on its own end-of-generation token never reaches the
  cap, and the switch changes nothing there.

So it goes into the negligible group (C534): no rig row of its own.

## The change

In `common_speculative_impl_draft_mtp::draft` (`common/speculative.cpp`) each drafting sequence gets the length it
stops at, `n_stop`. Off, that is `--spec-draft-n-max`, as before. With `LLAMA_MTP_DRAFT_CAP=1` it is the slot's cap
(the request's `dp.n_max`) when that is lower, but only where leaving out the passes changes nothing else:
- **a greedy draft** (`--spec-draft-sampling greedy`, the default): its sampler is reset at the start of every draft,
  so passes left out change no later draw. A probabilistic draft keeps its sampler's random state from one step to the
  next, and every pass draws from it; leaving passes out would change the next steps' drafts, so the cap leaves it as
  it was (Q1363, parked);
- **a cap that keeps `--spec-draft-n-min`**: the drafter clears a draft shorter than `n_min` before the server cuts
  it, so with a cap below `n_min` the drafter's length check would see a shorter draft than without the cap. With the
  cap at or above `n_min` it sees the same answer.

Every pass the cap leaves out comes after the last token the server keeps, so the passes that run are the same ones in
the same order. The draft context's cells of every drafting pass are removed after the draft either way (the server's
`seq_rm` from the target's last position on), so the next catch-up finds the same cache.

What the switch also changes, by design:
- with `LLAMA_STEP_TRACE` ([C405](../2026-11-11_c405_server_gap/README.md)) a step's line counts the passes the cap
  left out (`"cap"`), and has fewer `mtp.dp` marks;
- with `LLAMA_SPEC_DUMP` a capped step's line holds the candidates of the passes that ran: the rows of tokens the
  server cut are not there;
- at its end the drafter logs `draft length cap: <n> drafting passes left out in <m> drafts`, and at its start
  `draft length: a greedy draft stops at its slot's cap` (llama-server prints both at its default verbosity).

## The check

Every build and test ran in a container on a build machine's CPU, at low priority, one at a time under a lock
([run/checks.sh](run/checks.sh)); the results are in [summary.tsv](summary.tsv).

| Check | Verdict |
|---|---|
| static native CPU build of `4b0e2730a`, every target | OK, 3 warning lines (the base's, in `common/sampling.cpp`) |
| llama-server, one slot, eleven arms ([run/server_check.sh](run/server_check.sh)) | PASS (below) |
| llama-server, two slots, three arms (C741's [two_slots.py](../2026-11-11_c741_server_yields/run/two_slots.py)) | PASS (below) |
| `test-llama-archs -b CPU -s 1`, the switch unset and `=1`, against `5b329df16`'s logits | 146 of 146 the same, both |
| ctest (without the tests that need a download, the network or torch) | 70 of 70 |
| rig tarball, `TESTS=1` | OK (below) |

**One slot.** `base` is `5b329df16`'s llama-server, the other arms this branch's: `off` (the switch unset), `cap`
(`LLAMA_MTP_DRAFT_CAP=1`), `capnt` (`cap` without the step trace, which every other arm writes); the same with the
backend sampler (`-bs`, as the release preset): `bbase`, `boff`, `bcap`; with a probabilistic draft
(`--spec-draft-sampling probabilistic`): `pbase`, `pcap`; with `--spec-draft-n-min 2`: `nbase`, `ncap`. Each answered
four requests: 200 and 600 prompt tokens, 96 or 128 tokens greedy (G1, G2) and 96 at temperature 1.0 (S1, S2; top-k 20,
top-p 0.95, two seeds), with the top 3 probabilities of every token. `off` and `boff` give their reference's tokens,
top probabilities, draft counts and every verify step of `LLAMA_SPEC_DUMP` (the draft, the drafter's candidates and
probabilities, the target's top tokens) as they are; the capped arms give the same with the dump's candidate rows cut
to the draft's length, since those rows are the passes the cap leaves out ([server.tsv](server.tsv)).

The passes ([passes.md](passes.md); spare passes are drafting passes beyond the kept draft in a step's trace line, cut
rows the candidate rows beyond the draft in a dump line):

| arm | generation steps | drafting passes | drafted (kept) | accepted | spare passes (steps) | left out by the cap | cut rows (steps) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base (`5b329df16`) | 406 | 1,218 | 1,206 | 2 | 12 (8) | 0 | 12 (8) |
| off | 406 | 1,218 | 1,206 | 2 | 12 (8) | 0 | 12 (8) |
| cap | 406 | 1,206 | 1,206 | 2 | 0 (0) | 12 | 0 (0) |
| capnt | no trace | | | | | | 0 (0) |
| bbase | 406 | 1,218 | 1,206 | 2 | 12 (8) | 0 | 12 (8) |
| boff | 406 | 1,218 | 1,206 | 2 | 12 (8) | 0 | 12 (8) |
| bcap | 406 | 1,206 | 1,206 | 2 | 0 (0) | 12 | 0 (0) |
| pbase | 396 | 1,188 | 1,177 | 12 | 11 (7) | 0 | 11 (7) |
| pcap | 396 | 1,182 | 1,177 | 12 | 5 (3) | 6 | 5 (3) |
| nbase | 406 | 1,218 | 1,206 | 2 | 12 (8) | 0 | 12 (8) |
| ncap | 406 | 1,214 | 1,206 | 2 | 8 (4) | 4 | 8 (4) |

- In every capped arm the passes left out are exactly the reference's spare passes less the arm's, and its cut rows
  the reference's less the arm's; the drafter's closing log line gives the same total (12 in 8 drafts in `cap` and
  `bcap`, 6 in 4 in `pcap`, 4 in 4 in `ncap`).
- `pcap`: the greedy requests G1 and G2 are capped (6 passes); S1 and S2 sample their drafts and keep their 5 spare
  passes, as designed.
- `ncap`: with 3 tokens left the cap (2) keeps `n_min` 2 and one pass goes in each reply (4); with 2 left the cap (1)
  is below `n_min` and the drafter runs its three passes, as off (8 spare).
- The model's MTP layer is random, so almost no draft is accepted (2 of 1,206 greedy draft tokens). A reply therefore
  takes one token a step and passes through both capped steps; with real acceptance a step can jump over one.

**Two slots** (`-np 2`, `--metrics`, the step trace). Request A (G2's: 600 prompt tokens, 128 greedy) is posted, and
request B (G1's: 200 prompt tokens, 96 greedy) once `/slots` shows A with 16 tokens; a client thread reads `/slots` and
`/metrics` in turn every 2 ms until both are answered. In all three arms (`p2base`, `p2off`, `p2cap`) A and B are the
same tokens, top probabilities and draft counts as the one-slot base's answers to the same requests, and no read
failed. B ends first, while A still drafts its three tokens: there the cap drops B's rows from the batched passes but
no pass; at A's end, alone, it drops whole passes.

| arm | generation steps | drafting passes (batched) | left out by the cap (rows) | cut rows (dump lines) | reads answered |
|---|---:|---:|---:|---:|---:|
| p2base | 126 | 378 | 0 | 6 (4) | 398 |
| p2off | 126 | 378 | 0 | 6 (4) | 333 |
| p2cap | 126 | 375 | 6 | 0 (0) | 380 |

**The rig tarball:** `llama.cpp-20261111-4b0e2730a-pascal-p106-v4-b12246-c744.tar.gz`, 656,347,475 B, md5
`d8d96795350e1be28d6506960f5d53cc`, `TESTS=1`, sm_61, image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, no AVX instruction
in `llama-bench`, highest glibc symbol GLIBC_2.38, `test-llama-archs` in `bin/`.

## Merging with the other v4 branches

`git merge-tree` of `-c744` with `-c738`, `-c741`, `-c735`, `-c770`, `-c771`, `-c762`, `-c761`, `-c703`, `-c693`,
`-c620`, `-c619`, `-c405`, `-c217` and `-night` is clean. With `-c615` (and the day merge that holds it) one hunk
conflicts at the end of the MTP drafter's constructor: keep both, `cap_init();` and then C615's `LLAMA_DRAFT_PREFETCH`
block. With the prefetch on, a capped step issues the n-gram rows of the kept tokens only.

With [C738](../2026-11-11_c738_verify_prepare/README.md) (`LLAMA_VERIFY_PREPARE`): its drafter hook fires when the
pass is the drafter's last by `--spec-draft-n-max`, and its server gate refuses a draft whose cap is below that length.
So at a capped step it prepares nothing with or without this switch, and every other step is the same.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict, as `run/checks.sh` wrote it (the server row is cut at 3,000 characters: the whole verdict is `server_verdict.txt`) |
| [server_verdict.txt](server_verdict.txt) | `run/server_check.sh`'s verdict, line by line |
| [server.tsv](server.tsv) | every arm's requests: prompt and predicted tokens, drafted and accepted tokens, the tokens |
| [passes.md](passes.md), [passes.tsv](passes.tsv) | `run/passes.py` over every arm's step trace and dump: passes, kept tokens, spare passes, passes left out, cut rows |
| [run/checks.sh](run/checks.sh) | the driver of every check (the CPU build, the servers, the logits, ctest, the rig tarball) |
| [run/server_check.sh](run/server_check.sh) | llama-server's arms, one slot and two, and the verdict |
| [run/passes.py](run/passes.py) | the passes of `LLAMA_STEP_TRACE` and `LLAMA_SPEC_DUMP` files |

The two-slot client is C741's [run/two_slots.py](../2026-11-11_c741_server_yields/run/two_slots.py); the rig build
uses C620's [run/checks.sh](../2026-11-11_c620_draft_kv_catchup/run/checks.sh) (its `build` step).

## Commands

On a build machine, CPU only (each build and test in a container at low CPU priority, one at a time under a lock):

```bash
# WORK: a work folder; SRC: a llama.cpp clone in $WORK/rig/src with the fork as remote "fork" (https fetch URL), which
# scripts/build_llamacpp_docker.sh checks out and cleans; BASE_CPU: bin/ of 5b329df16's static native CPU build with
# test-llama-archs; BASE_SRV: the folder of its llama-server; REPO: this repository
WORK=$WORK REPO=$REPO SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c744 BASE_CPU=$BASE_CPU BASE_SRV=$BASE_SRV \
  LOCK=$WORK/lock JOBS=8 bash docs/data/2026-11-11_c744_draft_cap/run/checks.sh
```

By hand, with a CPU build of the branch and a model with an MTP block:

```bash
LLAMA_MTP_DRAFT_CAP=1 LLAMA_STEP_TRACE=trace.jsonl LLAMA_SPEC_DUMP=dump.jsonl llama-server -m model.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0
jq -c 'select(.draft != null) | del(.run, .step, .task)' dump.jsonl > out/steps.jsonl
cp trace.jsonl out/step_trace.jsonl
python3 -I docs/data/2026-11-11_c744_draft_cap/run/passes.py cap=out
```

## Limits

- Every number here is from a build machine's CPU, with a random model whose drafts are almost never accepted; the
  rig's cost of a pass is R86's.
- The cap binds only in a reply's last steps when the request limits the reply (`n_predict`, `max_tokens`) and at the
  context's end; a reply that ends on its own end-of-generation token is not changed.
- A probabilistic draft and a cap below `n_min` keep their passes (above).
- The two-slot check posts B at one point of A's reply; where B lands only moves which steps hold both slots.

## Provenance and licence

The models are `test-llama-archs`'s random fixtures (this fork's test code). The engine is Miner Llama (MIT, as
llama.cpp). The tables are this project's measurements.
