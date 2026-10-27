# Q88 and Q160: the persistent MoE kernel, the unified v2 build and the release candidates on the rig (2026-10-24/25)

The rig runs that took the persistent MoE kernel into the fork and the unified v2 build to the release: the kernel on
against off in one binary (Q88 build-in, branch `pascal-p106-v2-persistent` `8de13900f`), the unified build
`pascal-p106-v2-next` `62fa5fc3b` against the combined build `b3052a4f8` in llama-bench and in pipelined decoding (Q160),
and the identity runs of the release candidates rc1–rc4 on the P106s, the last one on the v2.0 commit `37e717ff6`. Nine
P106-090 cards unless a row says six. Results: registry rows Q88 and Q160 in
[quants/experiments.md](../../../quants/experiments.md), [docs/engine_features.md](../../engine_features.md) and, for
the release defaults, [2026-10-25_release_rig](../2026-10-25_release_rig/README.md). Single-request numbers are
diagnostics; the release is chosen on agentic speed ([2026-10-26_release_setup](../2026-10-26_release_setup/README.md)).

| File | What |
|---|---|
| `llama_bench.csv` | 64 llama-bench rows: Q88 build-in (K2h-r16 and UD-Q4_K_XL, the persistent kernel off and on, 6 and 9 cards, two rounds) and Q160 (the combined build against v2-next, the same grid); pp512 and tg128 in one process, tok/s, standard deviation, start time |
| `v2next-k2h-9c/summary.tsv`, `v2next-ud-9c/summary.tsv` | Q160 step 2: [bench/pipespec/rig_run.sh](../../../bench/pipespec/README.md) on v2-next `62fa5fc3b`, K2h-r16 and UD-Q4_K_XL, 45 jobs of 256 tokens each (plain, depth 0, the `cont` guesser at depths 1–4, `copy` and `spec` at depth 3, oracle ceilings at depths 1–4): tok/s, speed-up, guesses, identity |
| `rc3-k2h-9c/summary.tsv`, `rc3-ud-9c/summary.tsv` | the same script on rc3 `7e6ec762f`: the warm-up and two jobs, then llama-pipespec crashed ([2026-10-25_regen_crash](../2026-10-25_regen_crash/README.md)) |
| `rc4-k2h-9c/summary.tsv`, `rc4-ud-9c/summary.tsv` | the same script on the v2.0 commit `37e717ff6` at its defaults, depths 1 and 3, 33 jobs and the warm-up; `tok_s` and `speedup` from the traces (`basis` = `trace`) |
| `traces.tar.gz` | the 136 per-token traces of these runs (time of each sampled token in µs, guesses in flight, outcome), in `<run>/<job>.trace`; unpacked here they sit next to the summaries |
| `run/rig_q88_build.sh` | Q88 build-in: the kernel's self-check on six cards, then llama-bench off and on alternating twice, K2h-r16 and UD, 6 and 9 cards |
| `run/rig_v2next.sh` | Q160: the self-check (step 0), llama-bench combined against v2-next (1), llama-pipespec at depths 1–4 (2), llama-server `--pipe-depth 3` against no guesses (3) |
| `run/rig_rc1.sh`, `run/rig_rc3.sh`, `run/rig_rc4.sh` | each release candidate at its defaults before any timing: which kernels run (first-use lines, the persistent kernel's self-check, logits against each kernel switched off, launch groups), llama-pipespec identity at depths 1 and 3, the server check; rc3 adds Q163's prompt kernel, rc4 the drafter's identity (step d) and Q163's prompt speed (step 4) |
| `run/rig_q150.sh`, `run/rig_q151.sh`, `run/rig_q151_ab.sh`, `run/rig_q152.sh` | rig kits for Q150 (host defaults), Q151 (decode attention) and Q152 (attention block) on their own builds; none ran as written: their switches were measured one at a time on the release candidate ([2026-10-25_release_rig](../2026-10-25_release_rig/README.md)) |
| `run/t1.sh`, `run/t2.sh`, `run/t5.sh` | correctness on CUDA GPUs other than the P106 (an RTX 3090, two V100s through the compute_61 PTX): the kernel's operator tests, its faults and grid sizes (t1); K2h-r16 and G3 end to end, kernel off against on (t2); the same on v2-next with llama-pipespec at depths 3 and 4 (t5) |

## What it shows

| Question | Result |
|---|---|
| Q88 build-in: persistent kernel off → on, tg128 tok/s (round 1 / round 2) | K2h-r16 six cards 37.26 / 37.61 → 40.54 / 40.92, nine cards 39.55 / 39.10 → 40.93 / 42.10; UD-Q4_K_XL six cards 35.78 / 35.87 → 38.26 / 38.15, nine cards 37.54 / 37.53 → 39.83 / 39.77; pp512 unchanged (~250 on six cards, ~231 on nine) |
| The kernel's self-check on the P106s (`GGML_CUDA_PERSISTENT_MOE_CHECK=1`, K2h-r16, six cards, one token per step) | 0 layers differed from the per-op path in every graph evaluation on every card (130 evaluations per card) |
| Q160: combined → v2-next, tg128 tok/s (round 1 / round 2) | K2h-r16 six cards 37.77 / 37.64 → 40.71 / 40.83, nine cards 39.48 / 39.17 → 42.11 / 41.42; UD six cards 35.77 / 35.81 → 38.24 / 38.54, nine cards 37.59 / 37.63 → 39.85 / 39.85; pp512 unchanged |
| Q160: pipelined decoding on v2-next | 45 of 45 jobs identical to plain decoding for each file, depth 4 included; per-depth speed-ups in the summaries, corrected for the timer artifact in [2026-10-24_rig_c7recheck/corr.txt](../2026-10-24_rig_c7recheck/corr.txt) |
| rc3 `7e6ec762f` | every kernel ran at its defaults (persistent kernel 0 differed; logits differ with the Pascal matrix-vector kernels, decode attention, the K-quant kernels and Q163's prompt kernel each turned off); llama-pipespec crashed after the warm-up and two jobs, llama-server on the third reuse of a cached prompt: the crash of [2026-10-25_regen_crash](../2026-10-25_regen_crash/README.md) |
| rc4 `460160e93`, the drafter's identity (step d) | nine cards, even split and light last card, greedy and at the makers' sampling: every run (plain, depth 0, oracle, copy, continuation and staircase drafter at depths 1 and 3) identical to plain decoding; the six-card arm did not load (out of memory on the first card with the drafter) |
| rc4 `2d53179aa` (the allocator fix), then `bff28aba6` and the v2.0 commit `37e717ff6` | llama-pipespec 33 of 33 jobs identical to plain for K2h-r16 and UD on each build; llama-server with K2h-r16: `--pipe-depth 3 --pipe-guess cont` 13 of 13 requests identical, the filler-only floor 13 of 13, no `--pipe-depth` at T 0 6 of 6 (the UD server checks have no recorded result) |
| v2.0 `37e717ff6` at its defaults, pipelined, K2h-r16 (plain → `cont` depth 3) | chat 36.2 → 46.2, code 36.1 → 49.2, edit 35.5 → 52.7, rewrite 40.1 → 108.6 tok/s; UD-Q4_K_XL 34.2 → 44.4, 34.4 → 44.7, 34.1 → 52.7, 38.9 → 85.2 |

The `tok_s` of the v2-next and rc3 summaries comes from llama-pipespec's own clock, which before fork commit
`99c3a7c00` also counted the wait for the prompt's tail (0.1–3.3 s per job); the rc4 summaries already report the
trace-based rate. `python3 -I bench/pipespec/trace_speedups.py v2next-k2h-9c v2next-ud-9c` (run here after unpacking
the traces) prints the corrected per-depth tables. The served-request medians of the Q160 row (K2h-r16 39.2 → 48.5 tok/s,
UD 36.6 → 47.3) come from `rig_v2next.sh` step 3; that step's per-request files were overwritten by the later
release-candidate checks, so only the registry row keeps them. Its filler-only arm did not start (`--pipe-floor` is newer
than `62fa5fc3b`).

The release candidates' Q163 prompt speed (`rig_rc4.sh` step 4) and rc3's GPU time per token and launch group
(`rig_rc3.sh` step 0c: 26.43 ms summed over the cards, 27.65 with the attention block off) are in
[2026-10-25_release_rig](../2026-10-25_release_rig/README.md) (`llama_bench.tsv` step `prefill-attention`,
`op_profile_rc3.tsv`).

## Checks on other GPUs

- **Q88 (t1, RTX 3090, natively and through the compute_61 PTX):** MOE_LAYER 9 of 9 against the CPU and bit for bit
  against the per-op path, also at
  grid sizes 1, 2, 3 and 5,248; each deliberate fault caught (routing weights × 1.0001 by the self-check, the shared
  expert left out against the CPU, a top-k never published by a timeout). End to end (t2, t5; two V100s through the PTX):
  K2h-r16 and G3 self-checks 11,997 layer evaluations each, every stage identical; logits of 2 × 128 one-token steps
  byte-identical with the kernel off and on; greedy text 4 × 128 tokens identical; llama-pipespec plain and depths 3
  and 4 identical to plain decoding with the kernel off; CUDA graphs captured and replayed with the kernel.
- **Q151, the Pascal decode attention kernel** (V100 through the compute_61 PTX): `FLASH_ATTN_EXT` 24 of 24 Flint-shape
  cases; dropping the sinks fails exactly the 6 sink cases the kernel takes and dropping the last tile all 18 cases it
  takes, while the 6 cases it leaves to the stock kernel pass throughout; output × 1.01 stays inside the 5e-4 NMSE
  limit. G3, 4 × 512 tokens at `-ub 1`, kernel on against off: mean KLD 0.052 ± 0.008, same top token 91.2 ± 0.9 %,
  PPL ratio 1.055 ± 0.029. The summation-order reference on the same setup (stock attention, the Pascal matrix-vector
  kernels against llama.cpp's): 0.047 ± 0.007, 92.7 ± 0.8 %, 1.049 ± 0.028. The kernel moves Flint's output no more than
  rounding does; on the rig it decoded slower and ships off ([engine_features.md](../../engine_features.md)).
- **Release candidates** (rc1, rc3 and rc4 `5215256bd`): with every feature off the G3 logits matched the base
  `d379da309` byte for byte; `check_identity.sh` at `-lv 4` 15 of 15 runs identical, with the persistent kernel's and
  Q163's first-use lines.

## How it was made

Rig side (2026-10-24 16:26–21:00 and 2026-10-25 09:00–23:00), each script run from any directory with the build's
tarball unpacked in `/opt/llama-builds` on the rig
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) with `TESTS=1`); `RIG` is the ssh alias,
`OUT` where the log goes:

```bash
T=llama.cpp-20261024-8de13900f-pascal-p106-v2-persistent docs/data/2026-10-24_q88_persistent/run/rig_q88_build.sh
T=llama.cpp-20261024-62fa5fc3b-pascal-p106-v2-next STEPS='0 1 2 3' docs/data/2026-10-24_q88_persistent/run/rig_v2next.sh
T=llama.cpp-20261025-7e6ec762f-pascal-p106-v2-rc3 STEPS='0 2 3' docs/data/2026-10-24_q88_persistent/run/rig_rc3.sh
T=llama.cpp-20261025-460160e93-pascal-p106-v2-rc4 HEAD=<drafter head .gguf> STEPS=d docs/data/2026-10-24_q88_persistent/run/rig_rc4.sh
T=llama.cpp-20261025-37e717ff6-pascal-p106-v2-rc4 STEPS='2 3' docs/data/2026-10-24_q88_persistent/run/rig_rc4.sh
```

The llama-bench tables of the logs were turned into `llama_bench.csv` one row per test; the summaries are
`rig_run.sh`'s own `summary.tsv`. The traces were packed with `tar --sort=name` and `gzip -9n`. Models: Flint
K2h-r16 (released as Flint-Pascal-Q5, recipe in [quants/experiments.md](../../../quants/experiments.md), Q113) and
Unsloth's Flint UD-Q4_K_XL; the prompts are bench/pipespec's (Q51's chat and code prompts, two edits of this
repository's scripts). Flint is Apache-2.0; the traces and tables are this repository's (MIT).
