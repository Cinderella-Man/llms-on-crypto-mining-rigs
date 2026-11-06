# The v3 release's quality gates and release files (2026-11-05/06, build machine)

Three pieces of work for the v3.0 release (`pascal-p106-v3-b12076-ladder5` `1551f5fe1`), run in one detached script
on a build machine with RTX 30 cards, through the rig build's compute 6.1 PTX (speed is never measured there):

- **T5a, Ornith-1.5-122B valid1's greedy identity,** `1551f5fe1` against Q189's `-lowbit` `d7a75deb3`: the four
  prompts of [2026-11-05_release_regression](../2026-11-05_release_regression/README.md) at temperature 0, 256 tokens
  each, compared with its `compare_greedy.py`.
- **T5b, the quality gates for Flint's and Ornith's presets:** the KL divergence of the expert skip
  (`LLAMA_MOE_TAIL=p=0.9,renorm=1,hist=0`) against each model's reference logits, paired by chunk against the switch
  unset, with C200k's gate (within noise when d − 2 se ≤ 0) per text. Flint Q8_0 against Flint BF16 (wikitext-2
  test, 80 × 512) and Flint Q8_0 (held-out agent sessions, 12 × 8,192); Ornith valid1 against Ornith-1.5-122B Q8_0
  (wikitext-2, agent sessions, Exercism code). Ornith also runs on `d7a75deb3` with every switch unset (`t3a.tsv`: `1551f5fe1`'s change
  over it). On these cards (compute 8.6) neither build takes its Pascal kernels, so this and T5a compare only the
  code every card runs; the Pascal kernels' part of the difference is measured on the rig (T3a).
- **T5c, the release model files:** `pmix-q2-fast` shards 1–2, Ornith valid1 and Flint `p065` copied with
  `quantize.imatrix.file` and `quantize.imatrix.dataset` reduced to the files' own names (they held the folders the
  importance matrices were made in), every tensor byte for byte (verified), each with its sha256. `p065` is composed
  again from E3's recipe ([2026-10-27_flint_near_q8](../2026-10-27_flint_near_q8/README.md)) and must match the md5
  of the file the rig ran, `8387b2e585a7b9cc10f40cb6b4531694`.

**The card-hop cast has no arm.** `LLAMA_HOP_CAST` (the residual crosses each card boundary in Q8_0, a speed gain within noise
in quality on Flash-Next) is applied by `build_hop_cast`, and in `1551f5fe1` only Flash-Next's graph (`qwen4exp`) calls it.
On Flint (`flint`) and Ornith (`qwen35moe`) the switch changes nothing, so its arms would equal the switch unset and
it brings no speed either: it stays out of both presets for v3.0.

**`-lowbit` rebuilt for compute 6.1.** Its rig tarball of 2026-10-27 was built for compute 6.1 and 7.0
(`ARCHS = 610,700`, for the V100s the build machine then had). An RTX 30 card's driver compiles the 7.0 PTX in place
of the 6.1 PTX, and a kernel launch of that code fails there (`CUDA error: invalid argument` in
`ggml_cuda_kernel_launch`). Rebuilt by [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh),
which now builds for compute 6.1 only: `llama.cpp-20261105-d7a75deb3.tar.gz`, md5 `83aac1e8844d6ab5d6c961d2754de57b`.

## Results

The chain ran 2026-11-05 from 20:55 to 22:12; every step ended `ok`, and the kernel log's Xid count stayed 0 before and
after each ([summary.tsv](summary.tsv)). [steps.tsv](steps.tsv) also keeps the two first tries that failed in seconds,
Flint's KLD arms and the `p065` compose on a model path that was wrong, before their second runs, which are the rows
that count.

### T5b: the expert skip is above noise on every text, so it stays out of Flint's and Ornith's presets

`LLAMA_MOE_TAIL=p=0.9,renorm=1,hist=0` (arm `tail`) against the switch unset (arm `off`), both on `1551f5fe1`, each
file against its own reference logits. The change is paired by chunk; **the gate is C200k's: within noise when
d − 2 se ≤ 0**, otherwise above noise. [B-flint/kld.tsv](B-flint/kld.tsv), [B-flint/gate.tsv](B-flint/gate.tsv),
[B-ornith/kld.tsv](B-ornith/kld.tsv), [B-ornith/gate.tsv](B-ornith/gate.tsv).

| Model, text (reference) | `off` KLD | `tail` KLD | change ± se | same top token, `off` → `tail` | gate |
|---|---:|---:|---:|---:|---|
| Flint Q8_0, agent (Flint Q8_0) | 0.256538 ± 0.006110 | 0.344857 ± 0.007417 | +0.08832 ± 0.02050 | 93.535 → 91.807 % | **above noise** |
| Flint Q8_0, wiki (Flint BF16) | 0.051887 ± 0.002414 | 0.173874 ± 0.005339 | +0.12198 ± 0.00569 | 93.471 → 86.275 % | **above noise** |
| Ornith valid1, agent (Ornith Q8_0) | 0.052441 ± 0.001061 | 0.060000 ± 0.001159 | +0.00756 ± 0.00090 | 94.939 → 94.615 % | **above noise** |
| Ornith valid1, code (Ornith Q8_0) | 0.061123 ± 0.002564 | 0.071327 ± 0.002829 | +0.01021 ± 0.00178 | 96.588 → 96.387 % | **above noise** |
| Ornith valid1, wiki (Ornith Q8_0) | 0.176878 ± 0.002787 | 0.200708 ± 0.003103 | +0.02383 ± 0.00280 | 83.985 → 82.505 % | **above noise** |

- **All five tests are above noise, so the expert skip is a quality loss on both models at p = 0.9 and stays out of
  the Flint and Ornith presets in v3.0**, parked with these numbers. Flash-Next's own gate for the same cut is in
  [2026-11-04_moe_tail_kld](../2026-11-04_moe_tail_kld/README.md); this one is for the two models whose preset it was
  meant to join.
- The `experts_kept_mean` (10) and `routed_reads_pct` (100) columns of the two `kld.tsv` files are the table script's
  defaults, not measurements: the arm ran with `hist=0`, so no experts-kept histogram was taken (Flint keeps 6 of 256
  experts per token, Ornith 8). The tables say nothing about the reads the skip would save, only what it costs.
- The Flint agent text's paired error (0.02050) is the largest of the five, as the 12 chunks of 8,192 tokens are few
  (the 80 chunks of 512 on wiki and code resolve far more).

### T5b: Ornith `1551f5fe1` against `d7a75deb3` on the shared code: wiki is worse

The release build against Q189's `-lowbit` build (rebuilt for compute 6.1), Ornith valid1 with every switch unset in
both, on three RTX 3090s. On these cards (compute 8.6) neither build takes its Pascal kernels, so this compares only
the code every card runs. [B-ornith/t3a.tsv](B-ornith/t3a.tsv) pairs by chunk with `d7a75deb3` as the base, and
applies the same rule: **not worse when d − 2 se ≤ 0, otherwise WORSE**.

| Text | `d7a75deb3` KLD | `1551f5fe1` KLD | `1551f5fe1` minus `d7a75deb3` ± se | same top token, `d7a75deb3` → `1551f5fe1` | verdict |
|---|---:|---:|---:|---:|---|
| agent | 0.052554 ± 0.001077 | 0.052441 ± 0.001061 | −0.00011 ± 0.00037 | 94.982 → 94.939 % | not worse |
| code | 0.061739 ± 0.002605 | 0.061123 ± 0.002564 | −0.00062 ± 0.00067 | 96.637 → 96.588 % | not worse |
| wiki | 0.175210 ± 0.002738 | 0.176878 ± 0.002787 | **+0.00167 ± 0.00078** | 84.142 → 83.985 % | **WORSE** (d − 2 se = +0.00011) |

- **Wiki is worse by the rule, narrowly:** the change is 2.1 se above zero, one text of three, so chance is not ruled
  out. Agent and code are not worse.
- **`t3a.tsv` answers this question; `gate.tsv` does not.** `gate.tsv` lists the `lowbit` arm against `off`, which
  asks the other direction (is `d7a75deb3` worse than `1551f5fe1`?): its wiki row, −0.00167 ± 0.00078, is the same
  pair of numbers with the sign turned and reads "within noise" under that question.
- **This is not the release's verdict.** It covers the shared code only; the Pascal kernels' part is measured on the
  rig by T3a. If T3a confirms a loss, the shared-code change behind this wiki drift is bisected after the release
  (registry row Q1126).

### T5a: Ornith's greedy tokens differ between the builds on all 8 requests

Four prompts at temperature 0, 256 tokens each, through both builds' `llama-server` (pipelined and plain-decoding
requests), `d7a75deb3` as the reference ([A/identity.md](A/identity.md)). Within each build the pipelined and the
plain tokens are the same for every prompt; **across the builds they differ on every prompt** (8 requests of 8):

| Prompt (prompt tokens) | first difference at token | reference's top-two margin there | largest difference in the chosen token's log-probability before it |
|---|---:|---:|---:|
| `py-duration` (68) | 74 | **1.1332** nats | 0.5117 |
| `js-fib` (109) | 7 | 0.1654 nats | 0.0459 |
| `sh-review` (3,656) | 7 | **1.1965** nats | 0.1529 |
| `prose` (58) | 25 | 0.6856 nats | 0.4457 |

- **Two of the four first differences are not near ties.** At `py-duration` and `sh-review` the reference's first
  choice led its second by more than 1 nat (1.1332 and 1.1965), a margin a different order of additions does not
  normally flip; `prose` (0.6856) is in between and only `js-fib` (0.1654) is close to a tie. The identity check does
  not say what changed; the Pascal kernels' part is measured on the rig (T3a), the shared code's part by the
  chunk-paired KLD above.
- The rendered prompts are the same in both builds (`template = ref` is yes on all four rows): the difference is not
  in the chat template.

### T5c: the release model files' clean copies and checksums

Each copy has `quantize.imatrix.file` and `quantize.imatrix.dataset` reduced to the file's own names where they held
folders, and every tensor byte for byte; the audit flags no string in the copies that still looks like a path or a host
(all four `*.copy.flags` files are empty). Headers as written: [C/*.copy.kv](C/) (header keys only, with the tensor count last);
the checksums: [C/sha256sums.txt](C/sha256sums.txt).

| File | keys changed | sha256 |
|---|---|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf` | 2 (`quantize.imatrix.file`, `quantize.imatrix.dataset`) | `7d4b6cedfdbbefe3d016a0017f0874239f359589d44c2e4d0f211f5299004ce9` |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00002-of-00003.gguf` | 0 | `f20e58f28af29423a731897bc0b58abb7918b19d0628de614eae609e89604965` |
| `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf` | 1 (`quantize.imatrix.file`) | `ef28114d83df37a534d3b1667ba766de2c808a29d630c11318e75e0e69bbe76a` |
| `flint-near-q8-p065.gguf` | 1 (`quantize.imatrix.file`) | `da1c33ae6eac089c3d64146ce9e567422e80b78ee334abc9874e4203bab4c695` |

- **`p065` was composed again and is the file the rig ran:** the composed file's md5 is
  `8387b2e585a7b9cc10f40cb6b4531694`, equal to the rig file's, before its copy was made.
- These are the sha256 lines in the Hugging Face cards' `CHECKSUMS` (`release/*/huggingface/`), which match. The headers
  keep their makers' `general.name`, `general.license` and `general.quantized_by` (Unsloth's on `valid1` and `p065`,
  Swift's on `pmix-q2-fast`): rewriting them is a licence decision, left to each upload.

## Files

| File | What it is |
|---|---|
| [run/chain.sh](run/chain.sh) | the whole chain: four lanes (Ornith's cards, Flint's cards, the build, the files), a done marker per step, `summary.tsv` |
| [run/pyrun.py](run/pyrun.py) | runs a gguf-py script under `python3 -I` with a given gguf-py first on the path |
| [summary.tsv](summary.tsv), [steps.tsv](steps.tsv) | each step's status, times, Xid counts before and after, and its result line (`steps.tsv` keeps the failed first tries) |
| [B-flint/](B-flint/), [B-ornith/](B-ornith/) | T5b: `kld.tsv` (every arm and text), `gate.tsv` (the expert skip's gate), Ornith's `t3a.tsv` (`1551f5fe1` against `d7a75deb3`) |
| [A/identity.md](A/identity.md) | T5a: the greedy comparison's output |
| [C/](C/) | T5c: the copies' headers (`*.copy.kv`), the audit's flags (`*.copy.flags`, empty) and `sha256sums.txt` |

## How it ran

On the build machine, with its paths in an environment file (`chain.sh`'s header lists every variable):

```bash
setsid nohup bash docs/data/2026-11-05_night_quant_chain/run/chain.sh night.env > console.txt 2>&1 < /dev/null &
# a second start for lanes that had to be restarted beside the running ones (here: Flint's, after a model path fix)
LANES="flint files" FILES_AFTER=C-p065 setsid nohup bash docs/data/2026-11-05_night_quant_chain/run/chain.sh night.env > console2.txt 2>&1 < /dev/null &
```

Cards: three RTX 3090s for Ornith (`-ts 1,1,1`, every layer on the cards), an RTX 3080 12 GB and two RTX 3090s for
Flint (`-ts 1,2,2`). Every container carries a label and a CPU share of 256; the CPU and disk steps run at
`nice -n 19 ionice -c3`; the kernel log's Xid count is taken before and after every step.

The scripts it uses: [kld.sh](../2026-11-03_flash_next_quality/run/kld.sh) (unchanged, with the texts and bases of
[2026-10-27_ornith_kld](../2026-10-27_ornith_kld/)), [table.sh](../2026-11-04_moe_tail_kld/run/table.sh),
the release regression's `greedy_requests.py`, `compare_greedy.py` and `prompts.jsonl`,
[gguf_release_header.py](../../../scripts/gguf_release_header.py) and E3's `near_q8.py` with
[compose_gguf.py](../../../scripts/compose_gguf.py).
