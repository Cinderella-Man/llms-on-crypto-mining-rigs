# No hidden row back from the last drafting pass (C742, Q1311, 2026-11-11)

What this is: a switch in Miner Llama's MTP drafter, `LLAMA_MTP_SKIP_LAST_ROW` (off by default), that stops the last
drafting pass of a step from copying its hidden row to the host. Each drafting pass's decode copies the hidden row of
its output (`h_nextn`) back from the MTP layer's card, and the pass's `synchronize` waits for that copy; the next pass
reads it as its input. The last pass has no next pass, so its row was copied and never read.
[R86](../2026-11-11_r86_step_trace/README.md) put the last pass's way back from card 8 at 0.336 ms a step (median, the
drafted id and this row together). The registry row is Q1311 in [quants/experiments.md](../../../quants/experiments.md).

Fork branch `pascal-p106-v4-b12246-c742` at `08cc9bf95`, one commit on `pascal-p106-v4-b12246-c748` (`5b329df16`). Not
merged into another branch. Everything here ran on a build machine's CPU; nothing ran on a card.

## Result

**With the switch each draft copies one hidden row fewer to the host, and the replies, the drafts and the verify steps
are the same bytes.** On a build machine's CPU (llama-server, the tiny random qwen4exp model with one MTP block, the
draft at three tokens), four requests per arm: with the switch off the draft context copies 1,218 hidden rows to the
host in 406 drafts, 3.00 a draft, one per drafting pass; with it on 812, 2.00 a draft, and the 406 last passes copy
none. Every arm gives its reference's tokens, top probabilities, draft counts and all its verify steps
([copies.md](copies.md), [server_verdict.txt](server_verdict.txt)).

**On the rig it is small: est. 0.19 ms a step.** Flash-Next's hidden row is 10,240 F32 values (four hyper-connection
streams of 2,560), 40 KB, and a P106 copies to the host at ~213 MB/s ([mining_rig.md](../../mining_rig.md)): est.
0.19 ms, inside the 0.336 ms way back R86 measured. Against R86's greedy step of 128.92 ms (median, the v4 night
build) that is est. 0.15 %, and it applies to every step whose draft runs its three passes (the presets'
`--spec-draft-p-min 0` drafts three tokens every step). So it goes into the negligible group (C534): no rig row of its
own.

## The change

- **The draft context** (`src/llama-context.cpp`): `llama_skip_embeddings_nextn_next(ctx)` marks the next decode, which
  then leaves its `h_nextn` rows on the device instead of queueing their copy to the host. The mark lasts one decode
  (the decode takes it at its start, like the micro-batch taper's hints). The graph is the same, so a step's passes
  still reuse one graph: a graph without the `h_nextn` output would be built again for the last pass and again for the
  next step's first pass, and R86 timed a draft graph build at 0.689 and 1.890 ms. The context counts the micro-batches
  that copied their rows to the host and those that left them out for the mark
  (`llama_embeddings_nextn_host_stats`).
- **The drafter** (`common/speculative.cpp`, `common_speculative_impl_draft_mtp::draft`): with
  `LLAMA_MTP_SKIP_LAST_ROW=1`, before each drafting pass it checks whether every drafting sequence will have
  `--spec-draft-n-max` tokens after it; if so it marks the pass. A pass after which a `p_min` cut ends the draft is
  known only after the pass, so it still copies its row. The last pass still samples its token, so a probabilistic
  draft's random state moves as it did.
- **What else changes:** with `LLAMA_STEP_TRACE` ([C405](../2026-11-11_c405_server_gap/README.md)) every draft adds to
  its step's line the hidden rows the draft context copied to the host (`"h.host"`) and left out (`"h.skip"`), with the
  switch off as well. With the switch, the drafter logs `draft hidden rows: a step's last drafting pass copies none to
  the host` at its start and `draft hidden rows to the host: <n> copied, <m> left out (the last pass) in <k> drafts` at
  its end (llama-server prints both at its default verbosity).
- **With `LLAMA_DRAFT_H_CARRY=1`** no drafting pass copies its row to the host (the rows stay on the card between
  passes), so this switch has nothing to leave out there.

## The check

Every build and test ran in a container on a build machine's CPU, at low priority, one at a time under a lock
([run/checks.sh](run/checks.sh)); the results are in [summary.tsv](summary.tsv).

| Check | Verdict |
|---|---|
| static native CPU build of `08cc9bf95`, every target | OK, 3 warning lines (the base's, in `common/sampling.cpp`) |
| llama-server, one slot, eleven arms ([run/server_check.sh](run/server_check.sh)) | PASS (below) |
| llama-server, two slots, three arms (C741's [two_slots.py](../2026-11-11_c741_server_yields/run/two_slots.py)) | PASS (below) |
| `test-llama-archs -b CPU -s 1`, the switch unset and `=1`, against `5b329df16`'s logits | 146 of 146 the same, both |
| ctest (without the tests that need a download, the network or torch) | 70 of 70 |
| rig tarball, `TESTS=1` | OK (below) |

**One slot.** `base` is `5b329df16`'s llama-server, the other arms this branch's: `off` (the switch unset), `on`
(`LLAMA_MTP_SKIP_LAST_ROW=1`), `onnt` (`on` without the step trace, which every other arm writes); the same with the
backend sampler (`-bs`, as the release preset): `bbase`, `boff`, `bon`; with a probabilistic draft
(`--spec-draft-sampling probabilistic`): `pbase`, `pon`; with a `p_min` cut (`--spec-draft-p-min 0.1022`: the random
model's drafter puts about 0.102 on its top token, so some drafts end after one or two passes): `mbase`, `mon`. Each
answered four requests: 200 and 600 prompt tokens, 96 or 128 tokens greedy (G1, G2) and 96 at temperature 1.0 (S1, S2;
top-k 20, top-p 0.95, two seeds), with the top 3 probabilities of every token. Every arm of the branch gives its
reference's tokens, top probabilities, draft counts and every verify step of `LLAMA_SPEC_DUMP` (the draft, the
drafter's candidates and probabilities, the target's top tokens), compared in full ([server.tsv](server.tsv)).

The copies ([copies.md](copies.md); a draft is the trace's marks from `mtp.drf` to `mtp.drf.end`, its passes the
`mtp.dp` marks; the base's trace has no counts):

| arm | drafts | drafting passes | drafts at 3 passes | hidden rows copied to the host | left out | copies a draft | trace lines off the rule |
|---|---:|---:|---:|---:|---:|---:|---:|
| base (`5b329df16`) | 406 | 1,218 | 406 | not counted | | | |
| off | 406 | 1,218 | 406 | 1,218 | 0 | 3.00 | 0 |
| on | 406 | 1,218 | 406 | 812 | 406 | 2.00 | 0 |
| onnt | no trace | | | 812 (log) | 406 (log) | | |
| boff | 406 | 1,218 | 406 | 1,218 | 0 | 3.00 | 0 |
| bon | 406 | 1,218 | 406 | 812 | 406 | 2.00 | 0 |
| pon | 396 | 1,188 | 396 | 792 | 396 | 2.00 | 0 |
| mon | 407 | 893 | 189 | 704 | 189 | 1.73 | 0 |

- The rule, checked on every trace line of the branch's arms: the draft context copies one row batch per drafting pass,
  the left-out ones included (`h.host + h.skip` = passes); with the switch off none is left out, with it on exactly
  one per draft that reaches three passes. The drafter's closing log line gives the trace's sums in every arm with the
  switch, and `onnt`'s is `on`'s.
- `mon`: 218 of 407 drafts end after one or two passes on the `p_min` cut and copy every row; the 189 that reach three
  passes leave the last one out. Its replies and verify steps are `mbase`'s.
- `pon`: the last pass still samples (only its row's copy goes), and the replies are `pbase`'s.
- The model's MTP layer is random, so almost no draft is accepted; that changes how many tokens a step takes, not the
  passes a draft runs.

**Two slots** (`-np 2`, `--metrics`, the step trace). Request A (G2's: 600 prompt tokens, 128 greedy) is posted, and
request B (G1's: 200 prompt tokens, 96 greedy) once `/slots` shows A with 16 tokens; a client thread reads `/slots` and
`/metrics` in turn every 2 ms until both are answered. In all three arms (`p2base`, `p2off`, `p2on`) A and B are the
same tokens, top probabilities and draft counts as the one-slot base's answers to the same requests, and no read
failed. In 94 of the 126 drafts both slots draft in the same passes: there a pass's decode holds both slots' rows, and
the last pass leaves both out in one go.

| arm | drafts | drafting passes (batched) | drafts of both slots | hidden rows copied (batches) | left out | reads answered |
|---|---:|---:|---:|---:|---:|---:|
| p2base | 126 | 378 | 94 | not counted | | 230 |
| p2off | 126 | 378 | 94 | 378 | 0 | 222 |
| p2on | 126 | 378 | 94 | 252 | 126 | 224 |

**The rig tarball:** `llama.cpp-20261111-08cc9bf95-pascal-p106-v4-b12246-c742.tar.gz`, 656,353,567 B, md5
`fd4934e7e72417c0913c6236f77bb708`, `TESTS=1`, sm_61, image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, no AVX instruction
in `llama-bench`, highest glibc symbol GLIBC_2.38, `test-backend-ops` and `test-llama-archs` in `bin/`.

## Merging with the other v4 branches

`git merge-tree` of `-c742` with `-day` (the day merge, which holds `-c738` and `-c619`), `-night`, `-c744`, `-c738`,
`-c619`, `-c615`, `-c405`, `-c703`, `-c735`, `-c741`, `-c761`, `-c762`, `-c770` and `-c771` is clean.

- **With [C744](../2026-11-11_c744_draft_cap/README.md)** (`LLAMA_MTP_DRAFT_CAP`): this switch marks the pass that is
  last by `--spec-draft-n-max`. At a step that C744 caps below that, the pass that is last by the cap copies its row as
  before (nothing wrong, one row not saved). To leave that one out too, the merged check takes C744's `n_stop[seq_id]`
  in place of `params.n_max`.
- **With [C738](../2026-11-11_c738_verify_prepare/README.md)** (`LLAMA_VERIFY_PREPARE`): its hook fires at the same
  pass, after the pass's decode; the mark is on the draft context and C738 prepares the target's, so they do not meet.
- **With C619** (one draft graph a step): the graph is the same with and without the mark.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict, as `run/checks.sh` wrote it (the server row is cut at 3,000 characters: the whole verdict is `server_verdict.txt`) |
| [server_verdict.txt](server_verdict.txt) | `run/server_check.sh`'s verdict, line by line |
| [server.tsv](server.tsv) | every arm's requests: prompt and predicted tokens, drafted and accepted tokens, the tokens |
| [copies.md](copies.md), [copies.tsv](copies.tsv) | `run/copies.py` over every arm's step trace and dump: drafts, passes, hidden rows copied and left out, the rule's misses |
| [run/checks.sh](run/checks.sh) | the driver of every check (the CPU build, the servers, the logits, ctest, the rig tarball) |
| [run/server_check.sh](run/server_check.sh) | llama-server's arms, one slot and two, and the verdict |
| [run/copies.py](run/copies.py) | the drafts, passes and hidden-row copies of `LLAMA_STEP_TRACE` files |

The two-slot client is C741's [run/two_slots.py](../2026-11-11_c741_server_yields/run/two_slots.py); the rig build
uses C620's [run/checks.sh](../2026-11-11_c620_draft_kv_catchup/run/checks.sh) (its `build` step).

## Commands

On a build machine, CPU only (each build and test in a container at low CPU priority, one at a time under a lock):

```bash
# WORK: a work folder; SRC: a llama.cpp clone in $WORK/rig/src with the fork as remote "fork" (https fetch URL), which
# scripts/build_llamacpp_docker.sh checks out and cleans; BASE_CPU: bin/ of 5b329df16's static native CPU build with
# test-llama-archs; BASE_SRV: the folder of its llama-server; REPO: this repository
WORK=$WORK REPO=$REPO SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c742 BASE_CPU=$BASE_CPU BASE_SRV=$BASE_SRV \
  LOCK=$WORK/lock JOBS=8 bash docs/data/2026-11-11_c742_last_draft_row/run/checks.sh
```

By hand, with a CPU build of the branch and a model with an MTP block:

```bash
LLAMA_MTP_SKIP_LAST_ROW=1 LLAMA_STEP_TRACE=trace.jsonl LLAMA_SPEC_DUMP=dump.jsonl llama-server -m model.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0
jq -c 'select(.draft != null) | del(.run, .step, .task)' dump.jsonl > out/steps.jsonl
cp trace.jsonl out/step_trace.jsonl
python3 -I docs/data/2026-11-11_c742_last_draft_row/run/copies.py on=out
```

## Limits

- Every number here is from a build machine's CPU, where the "copy" is a memory copy; the rig's saving is the estimate
  above (the row's size at the card-to-host rate), not a measurement.
- A draft that a `p_min` cut ends before its last pass keeps every copy: whether a pass is the last is known before it
  runs only by length.
- The two-slot check posts B at one point of A's reply; where B lands only moves which drafts hold both slots.

## Provenance and licence

The models are `test-llama-archs`'s random fixtures (this fork's test code). The engine is Miner Llama (MIT, as
llama.cpp). The tables are this project's measurements.
