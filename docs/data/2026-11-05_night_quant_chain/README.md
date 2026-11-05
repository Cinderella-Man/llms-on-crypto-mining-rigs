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

Filled in when the chain ends (`summary.tsv`, `gate_*.tsv`, `t3a.tsv`, `identity.md`, the copies' headers and
checksums).

## Files

| File | What it is |
|---|---|
| [run/chain.sh](run/chain.sh) | the whole chain: four lanes (Ornith's cards, Flint's cards, the build, the files), a done marker per step, `summary.tsv` |
| [run/pyrun.py](run/pyrun.py) | runs a gguf-py script under `python3 -I` with a given gguf-py first on the path |

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
