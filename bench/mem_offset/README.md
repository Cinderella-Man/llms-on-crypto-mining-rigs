# mem_offset: testing a memory overclock on one P106-090, with proof that the output stays bit-identical

A P106-090's GDDR5 runs at 8,008 MT/s (`nvidia-smi` shows 4,004 MHz). Decode speed on this rig depends partly on
memory bandwidth, so a memory clock offset can buy a few percent: est. +3–5 % decode at +10 % memory clock, because only
about half of a token's time streams weights ([hardware tuning note](../../docs/research/notes/hardware_tuning.md),
sections 1–2). The cards have no ECC, though. A memory clock past a card's margin can corrupt weights silently instead
of crashing, so "it didn't crash" proves nothing.

This kit tests one card at a time and accepts an offset only if the model's output is unchanged bit for bit. It follows
the protocol of the note's section 5.2:
1. find the units of the driver's offset;
2. record stock references;
3. raise the offset in +200 MT/s steps, with a correctness and bandwidth check at each step;
4. at the best step, run a memory pattern test, an exact output check and a one-hour soak.

The offset goes back to 0 at the end, on any failure and on Ctrl-C. Nothing persists across a reboot or driver reload.

## What is in it

| File | What it does |
|---|---|
| `nvml_memoc.py` | Reads and sets one card's memory (or, with `--core`, graphics) clock offset through NVML, from Python's `ctypes` on the driver's `libnvidia-ml.so.1`: no install, no Xorg or Coolbits change. `probe` is read-only. |
| `memoc_steps.sh` | The test, stage by stage or all at once (below). |
| `exact_check.sh` | The exact output check on Flint: one-token llama-perplexity statistics and a greedy-decoding hash against stock. |
| `build.sh` | Builds `mmvq_pascal` and `cuda_memtest` for compute capability 6.1 in Docker and packs the kit into a tarball with MD5 sums. |
| `test/dryrun.sh` | Runs `memoc_steps.sh` against stub programs on any Linux machine (no GPU, no root) to check its control flow: a clean run, a failed exact check (step back), an Xid, an interrupt, and a memory test that cannot run. |

The tarball adds `bin/mmvq_pascal` ([bench/mmvq_pascal](../mmvq_pascal/README.md): streaming-read bandwidth, plus
correctness checks and timing of the decode kernels) and `bin/cuda_memtest`
([cuda_memtest](https://github.com/ComputationalRadiationPhysics/cuda_memtest): walking ones on the address lines,
own-address, moving inversions, random patterns and a stress test). Both are built with sm_61 code, compute_61 PTX,
host code without AVX and the CUDA runtime linked statically, so they need only the NVIDIA driver.

## Building

On any x86-64 Linux machine with Docker and the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image (CUDA 13 can no longer
target Pascal):

```bash
bench/mem_offset/build.sh /path/to/workdir      # JOBS=8, DOCKER_ARGS="--label …" if the build machine wants them
```

It clones cuda_memtest at a pinned commit, builds both programs (a few minutes) and writes
`workdir/mem_offset-kit-<date>.tar.gz` with its `.md5`. `BUILDINFO.txt` in the tarball shows that neither binary
contains AVX instructions, the highest glibc symbol they need (2.38; Ubuntu 24.04 has 2.39) and their CUDA code.

## Running on the rig

Run as root, with nothing else on the card under test, and with someone near the rig's keyboard: a card that hangs
needs a reboot, and on this rig a reboot needs the disk passphrase typed at the console.

```bash
K=mem_offset-kit-20261024
scp $K.tar.gz rig:/tmp/ && ssh rig "mkdir -p /tmp/oc && tar xzf /tmp/$K.tar.gz -C /tmp/oc && cd /tmp/oc/$K && md5sum -c MD5SUMS"

# 1. read-only, ~1 min: do the offset calls exist on this card and driver, and in what range? Also the PCIe link and
#    the CPU frequency settings
ssh rig "python3 /tmp/oc/$K/nvml_memoc.py probe"

# 2. the test on GPU0 (~1 h 50 min), detached so that a dropped ssh connection does not stop it
ssh rig "cd /tmp/oc/$K && setsid nohup ./memoc_steps.sh all > console.log 2>&1 < /dev/null &"
ssh rig "tail -f /tmp/oc/$K/out/steps.log"
# stop early and cleanly (at the next check):  ssh rig "touch /tmp/oc/$K/out/STOP"
# stop at once (the trap still resets):       ssh rig "pkill -TERM -f memoc_steps.sh"

# 3. results, then clean up (the KLD base file is ~0.6 GB)
ssh rig "cat /tmp/oc/$K/out/summary.txt"
scp -r rig:/tmp/oc/$K/out ./mem_offset-gpu0 && ssh rig "rm -rf /tmp/oc /tmp/$K.tar.gz"
```

If `probe` reports `Not Supported` for both offset APIs on the card, route A is closed: stop there. The Coolbits
route (section 1.3 of the note) needs an Xorg reconfiguration.

### Stages

`memoc_steps.sh STAGE` runs one stage. The stages share their state in `out/`. `memoc_steps.sh all` runs them in
order:

| Stage | What | Pass | Time (est.) |
|---|---|---|---|
| `units` | NVML does not document the offset's unit. Sets a raw +100, samples the memory clock under a short mmvq load against offset 0, then sets 0 again. A clock 50 MHz higher means the raw unit is MT/s; 100 MHz higher means the reported MHz (2 MT/s each). | the clock moves by one of the two | 1 min |
| `base` | At offset 0: a `cuda_memtest` smoke test on 64 MB; writes the KLD base file (Flint UD-Q4_K_XL, 4,096 one-token steps on cards 0–5), reruns against it for the reference statistics, and hashes a 128-token greedy completion | the smoke test is clean; the stock rerun shows the same top token everywhere (the run is deterministic) | 15 min (3 model loads) |
| `sweep` | +200, +400, … up to `MAX` (1,200) MT/s; at each step the mmvq sweep (Q8_0, Q5_0, Q4_1, Q5_K and IQ4_XS kernels, each checked against a double-precision CPU reference, then timed; streaming-read GB/s) and an Xid scan of the kernel log | the mmvq checks all pass, no Xid, and the streaming read gains ≥ `MIN_RISE` (0.5 %) over the step before; the first step that fails ends the sweep, and the step before it is BEST | 1–1.5 min per step |
| `memtest` | `cuda_memtest` at BEST, tests 0–8 and 10, one pass or `MEMTEST_MIN` (10) minutes, whichever ends first | no block reports errors, no Xid; if the program fails without reporting a memory error, the run stops rather than stepping back | 10 min |
| `exact` | At BEST: the KLD run against the stock base file, and the greedy hash | statistics identical to the stock rerun, hash identical | 10 min |
| `soak` | At BEST: llama-bench repeating a 512-token prompt plus 128 generated tokens on cards 0–5 for `SOAK_MIN` (60) minutes, checking every minute for an Xid, the process and the card's temperature (`TEMP_MAX`, 70 °C); then the KLD run again | the hour completes, no Xid, temperature under the limit, statistics identical afterwards | 65 min |

In `all`, a failed `memtest` or `exact` at BEST steps back 200 MT/s and tests again (+20 min per step back).
`out/summary.txt` collects the results:
- `sweep.csv` holds GB/s and the memory clock under load at each step;
- `smi.csv` holds the card's clocks, power, temperature and clock event reasons every 500 ms;
- `smi-cards.csv` holds the same for all six cards every 2 s;
- the logs of every run sit next to them.

Every value can be changed through the environment: `GPU`, `STEP`, `MAX`, `MIN_RISE`, `MMVQ_ARGS`, `MEMTEST_MIN`,
`SOAK_MIN`, `TEMP_MAX`, `OUT`, and for the llama.cpp runs `B` (bin directory of the build), `M` (model), `WIKI`,
`DEVS` (the cards; must include the card under test and hold the model) and `E` (extra environment,
`LLAMA_NO_MOE_DOWN_RESCALE=tg` for UD-Q4_K_XL). The defaults are the v2 base build
`/opt/llama-builds/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin`, `/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf`
and wikitext-2 under `/opt/llama-builds/data/`.

### Reading the result

- **The units line** fixes how NVML's raw value maps to the clock on this card and driver. Every offset in the logs
  is in MT/s, the units of `nvidia-settings` and of the note: +800 is +10 %.
- **Streaming GB/s against the memory clock** gives the fraction of the clock gain the card turns into bandwidth.
  The kernels' GB/s at each step, divided by the clock ratio, gives the fraction f of section 2.1.
- **BEST is the last clean step, not the setting to keep.** Rare errors near the edge can take hours to show, so the
  setting for service is one step below BEST. Once every card is done, check it with the nine-card exact check and
  llama-bench against stock (stages 5 and 6 of the note).
- **The offset is gone after every stage and at the end.** Read it back with
  `python3 nvml_memoc.py get 0`; `python3 nvml_memoc.py reset 0` forces 0.

## The exact check

The fork's decode is deterministic: the same build, model, card set and input give byte-identical logits. With stable
memory, an overclocked run therefore gives exactly the output of a stock run.
- **Why not "KLD = 0".** llama-perplexity stores the base file's log-probabilities in 16 bits, so even identical
  logits show a small KLD: `Mean KLD` prints 0.000000, while the maximum is ~1e-5 (0.000043–0.000062 on two RTX 3090s
  with Flint G3 and bit-identical logits).
- **What the test is.** `exact_check.sh kld` requires the run's whole statistics block to equal a stock rerun's,
  digit for digit: perplexity, KLD mean, maximum and percentiles, the Δp statistics and "same top p" at 100 %.
  A changed logit anywhere in the 4,096 steps changes that block.
- **The greedy hash** is a second, coarser check through a different binary (`llama-completion`).

Each llama.cpp run loads the 25.1 GB model again (~2.5 min) and then takes ~2.5 min for its 4,096 one-token steps on
six cards. GPU0 holds the first layers of the six-card split, so its weights take part in every token.

## Safety notes

- **What a reset clears.** Offset 0 restores stock at once, and any reboot or driver reload clears an offset. The
  kit changes nothing persistent: no Xorg configuration, no files outside `/tmp/oc`, no service.
- **A hung card** (Xid 79, or a CUDA process that never returns) needs a reboot, after which every card is at stock.
- **Run on a quiet rig.** Nothing else should use the card under test. `memoc_steps.sh` refuses to start if a compute
  process is on it (`FORCE=1` overrides), and the soak uses cards 0–5.
- **Not verified before the first run on the rig:**
  - `set` and `units` on a P106-090: the kit only read the offset APIs on driver 580 on other cards;
  - whether an offset set by one process stays after it exits while Xorg holds the driver (every offset change in
    `steps.log` is followed by a read-back from a new process, and the memory clock under load shows it);
  - `cuda_memtest`'s run time for 6 GB (it stops after `MEMTEST_MIN` either way).
