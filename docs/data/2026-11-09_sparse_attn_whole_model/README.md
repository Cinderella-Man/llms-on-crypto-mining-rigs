# Q1171: the sparse Pascal attention on the whole model at depth, off the rig (2026-11-09)

Miner Llama v3.0's build holds a Pascal attention kernel that reads only the ~2,051 cells each query token selects
([Q230](../2026-11-04_qsa_sparse_attn/README.md), `GGML_CUDA_FATTN_SPARSE_PASCAL`, off by default). It had passed its
operator cases on a CPU and through the compute 6.1 PTX, and no real model had decoded through it. This folder is that
check ([Q1171](../../../quants/experiments.md)): Qwen3.8-Flash-Next decoding greedily after prompts of 8,192, 32,768
and 60,000 tokens with the switch off and on, alone and with the MTP draft, on three RTX 3090s that compile the
build's compute 6.1 PTX. Correctness only: no time was recorded.

**Result: no fault shows, and the check is blunter than a hash comparison suggests.**

- **The texts:** with the switch on the model wrote the same 128 tokens as with it off at 32,768 and 60,000 tokens,
  alone and with the draft, and at 8,192 with the draft. Alone at 8,192 the two texts part at token 122 of 128.
- **That token is decided by rounding.** There the dense arm had three candidates within 0.114 of each other in
  log-probability (`_` −1.4082, `_Str` −1.4728, `_Con` −1.5222). The sparse kernel with its lists cut into 32 splits
  instead of 10, the same sums in another order, writes the dense arm's 128 tokens; with 1 split it writes a third
  text.
- **Rounding moves this model far.** Two orders of the same sums in the one attention kernel move the next token's
  log-probabilities by 0.0945, 0.109 and 0.11 (three pairs of split counts); the sparse kernel against the dense
  kernels moves them by 0.0421, 0.0647 and 0.111 (32, 10 and 1 splits), and a deliberate 1 % error in the kernel's
  output by 0.139. A greedy text therefore shows a gross fault and hides a small one: the 1 % error left the text
  unchanged at 32,768 and 60,000 tokens, and the kernel's larger test fault (one list entry dropped per split)
  changed it at 32,768 and 60,000 in both sets and at 8,192 alone.
- **At 32,768 and 60,000 tokens two loads of the same arm do not repeat.** With the switch off both times, the
  log-probabilities of the first 16 generated tokens differ by up to 0.564 and 0.474 alone and 0.954 and 0.968 with
  the draft, and the draft line's second load at 60,000 wrote another text from token 118. At 8,192 three loads agree
  to the last digit. Only dense kernels run in those arms, so this is the engine's, not the sparse kernel's
  ([Q575](../../../quants/experiments.md)).

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v3-b12076-ladder5` `1551f5fe1`, the v3.0 release build: sm_61 code and compute 6.1 PTX |
| Cards | three RTX 3090s in `nvidia/cuda:12.9.2-devel-ubuntu24.04`; they cannot run sm_61 code, so the driver compiles the PTX |
| Model | Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0, three shards (the n-gram table in the third) |
| Draft | `mtp-Qwen3.8-Flash-Next-draft.gguf` (sha256 `00818378…`), for the draft set |
| Flags | the [Flash-Next preset](../../../release/flash-next-v3/settings.env)'s `PRESET_ARGS` without `-ts` (its nine-card split) and `-otd token_embd=CUDA4`; the plain set also without the three draft flags |
| Variables | the preset's `PRESET_ENV` without `GGML_CUDA_KEEPALIVE`, and `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5`, so the kernel cuts its lists as on a P106-090 (10 splits for one token) |
| Requests | per depth a fresh prompt of Python standard-library source, raw text through `/completion`, 128 tokens at temperature 0 with end of text ignored ([run/whole_requests.py](run/whole_requests.py), a changed copy of Q230's `long_requests.py`); then per depth a 16-token probe that also returns the top 10 log-probabilities of each token |

The prompts held 7,940, 32,767 and 59,933 tokens, the same in every arm (their md5 is in [texts.tsv](texts.tsv)).
One llama-server load per arm; the arms:

| Arm | Variables | What it is |
|---|---|---|
| `off` | `GGML_CUDA_FATTN_SPARSE_PASCAL=0` | the dense kernels read every cell |
| `on` | `=1` | the sparse kernel takes each attention call of at most 8 query tokens once the cache holds 1.5 × 2,051 cells per query token |
| `fault` | `=1`, `_FAULT=1` | the kernel's own test fault: its output × 1.01 |
| `offfault` | `=0`, `_FAULT=1` | the fault is the sparse kernel's, so this is a second `off` load |
| `off.r2`, `on.r2` | as `off`, `on` | one more load each at 8,192, with 128-token probes |
| `on.s1`, `on.s32` | `=1`, `_SPLITS=1` or `32` | the same kernel, its sums in another order (8,192 only, 128-token probes) |
| `on.f3` | `=1`, `_FAULT=3` | the kernel's test fault 3: every split's last list entry dropped (10 of a query token's 2,051 cells) |

Three things about this path:

- **`CUDA_FORCE_PTX_JIT=1` stops the run.** It puts cuBLAS on PTX as well, cuBLAS ships none, and the first prompt
  batch aborts in `cublasCreate`. A build that holds only sm_61 code and compute 6.1 PTX needs no forcing on a newer
  card.
- **Mode 1 runs here** because it asks whether the build holds Turing code, not which card it is on. The fork's other
  Pascal kernels (its dense attention, matrix-vector and expert kernels) switch on at compute capability 6.1 exactly
  unless forced, so they stay off on these cards: the dense arm is llama.cpp's own FP32 attention kernel compiled
  from the same PTX, and the preset's k-column variables do nothing here. The check is of the sparse kernel inside
  the real model, not of the P106's whole kernel set.
- **Three cards make two card boundaries**, where the rig's nine make eight (the preset casts the residual to Q8_0
  at each).

## The texts

128 greedy tokens per request. Full hashes and counts: [texts.tsv](texts.tsv).

| Set | Prompt tokens | `off` | `on` | on against off | Drafted, accepted (both arms) |
|---|---:|---|---|---|---|
| plain | 7,940 | `cb8efafe` | `0aba9e95` | **parts at token 122** | |
| plain | 32,767 | `1169fb75` | `1169fb75` | same | |
| plain | 59,933 | `14e76abd` | `14e76abd` | same | |
| draft | 7,940 | `0aba9e95` | `0aba9e95` | same | 118, 87 |
| draft | 32,767 | `1169fb75` | `1169fb75` | same | 114, 88 |
| draft | 59,933 | `14e76abd` | `14e76abd` | same | 117, 87 |

With dense kernels only, the draft set's text at 7,940 tokens is not the plain set's: it parts from it at the same
token 122 and equals the plain set's `on` text. Verifying four tokens in one pass orders the sums differently from
decoding one token at a time.

## The token where the plain texts part

At 7,940 tokens every arm repeats exactly: `off`, `offfault` and `off.r2` agree in all 160 log-probabilities of the
16-token probe, `on` and `on.r2` likewise. So what differs between arms there comes from the arms.

| Arm at 7,940 tokens | Text | Against `off` | Token 122: `_`, `_Str`, `_Con` |
|---|---|---|---|
| `off` (dense) | `cb8efafe` | | −1.4082, −1.4728, −1.5222: writes `_` |
| `on`, 10 splits | `0aba9e95` | parts at token 122 | −1.7028, −1.3271, −1.3718: writes `_Str` |
| `on.s32`, 32 splits | `cb8efafe` | the same 128 tokens | −1.4801, −1.5112, −1.5157: writes `_` |
| `on.s1`, 1 split | `6c244160` | parts at token 3 | (another text by then) |

Token 3 is the same kind of place: in the dense arm ` #` leads ` mat` by 0.0389, with 10 splits by 0.1352, and with
1 split ` mat` leads by 0.2954. [margins.txt](margins.txt) has every such point with its candidates.

How far the arms are apart from the first decoded token on ([steps.tsv](steps.tsv): the largest difference in
log-probability over the candidates both arms list, per generated token; token 1 comes from the prompt's last batch,
which the dense kernels compute in every arm):

| At 7,940 tokens | Token 1 | Token 2 | Token 3 |
|---|---:|---:|---:|
| `off.r2` against `off` | 0 | 0 | 0 |
| `on.r2` against `on` | 0 | 0 | 0 |
| `on` against `off` | 0 | 0.0647 | 0.088 |
| `on.s32` against `off` | 0 | 0.0421 | 0.039 |
| `on.s1` against `off` | 0 | 0.111 | 0.793 |
| `on.s32` against `on` | 0 | 0.0945 | 0.0737 |
| `on.s1` against `on` | 0 | 0.11 | 0.769 |
| `on.s1` against `on.s32` | 0 | 0.109 | 0.763 |
| `fault` (output × 1.01) against `on` | 0 | 0.139 | 0.535 |
| `on.f3` (entries dropped) against `on` | 0 | 0.779 | (texts apart) |

The sparse kernel differs from the dense kernels by no more than it differs from itself with another split count.
The kernel's operator cases, which compare one call with a CPU reference, bound its error per call; this table says
that on the whole model a change of that size is already worth 0.04 to 0.11 in log-probability one token later.

## Did the sparse kernel compute the attention in every `on` arm?

The build logs no counter for the kernel and its profiler labels operators, not kernels, so the proof is the
kernel's own test faults: they act only inside it.

| Set, prompt tokens | Evidence |
|---|---|
| plain, 7,940 | `on` differs from `off` in 144 of 154 probe values (the 10 equal ones are token 1's); `fault` parts from `on` at token 3, `on.f3` at token 2 |
| plain, 32,767 | `on.f3` parts from `on` at token 41 (`bfcc73e5`); `fault` writes `on`'s text |
| plain, 59,933 | `on.f3` parts from `on` at token 38 (`81a82619`); `fault` writes `on`'s text |
| draft, 7,940 | the kernel takes few calls here by its own rule: a verify pass of 3 or 4 tokens goes to it from 9,230 or 12,306 cells. `on` equals `off` in 140 of 160 probe values (the first 14 tokens), and no fault changes the text |
| draft, 32,767 | `on.f3` parts from `on` at token 41 (`7f6520aa`; 119 drafted against 114); `fault` writes `on`'s text |
| draft, 59,933 | `on.f3` parts from `on` at token 33 (`92956505`; 126 drafted against 117); `fault` writes `on`'s text |

At depth the probes add little to this, because two loads of one arm are already apart there (next section). The
largest log-probability difference in the 16-token probes ([probe.tsv](probe.tsv)):

| Set, prompt tokens | Second `off` against `off` | `fault` against `on` | `on.f3` against `on` |
|---|---:|---:|---:|
| plain, 32,767 | 0.564 | 0.533 | 2.43 |
| plain, 59,933 | 0.474 | 0.59 | 1.49 |
| draft, 32,767 | 0.954 | 0.838 | 2.38 |
| draft, 59,933 | 0.968 | 1.15 | 1.54 |

## Two loads of one arm at depth

`offfault` is `off` again: the fault variable is read only inside the sparse kernel, which is off.

| Set, prompt tokens | Second `off` load against the first: 16-token probe | 128-token text |
|---|---|---|
| plain, 7,940 | 160 of 160 values the same | same |
| plain, 32,767 | 1 of 158 the same, largest difference 0.564 | same |
| plain, 59,933 | 1 of 158 the same, largest difference 0.474 | same |
| draft, 7,940 | 160 of 160 the same | same |
| draft, 32,767 | 0 of 156 the same, largest difference 0.954 | same |
| draft, 59,933 | 2 of 156 the same, largest difference 0.968 | **parts at token 118** (`f550b7ed`, 119 drafted against 117) |

Token 1 differs too, so the prompt's own passes are not repeatable at these depths. The first 16 tokens are safe from
it (the smallest lead of a chosen token there is 3.06 at 32,767 and 5.59 at 59,933); a closer call further on is not.
Of the 16 texts written at depth without fault 3, 15 are their depth's text and one is not; with fault 3 all 4
differ, from tokens 41, 38, 41 and 33.

## What this means

- **For the kernel:** nothing here asks for a change. Its per-call accuracy is the operator cases' business (on a
  CPU, through the PTX, and natively on a P106 by replay, [Q931](../../../quants/experiments.md)); the whole model
  adds that the kernel takes the calls its rule says, at 60,000 cells as at 8,000, and that the model's output stays
  the text the dense kernels write except where rounding decides.
- **For any greedy comparison on this model:** equal hashes are evidence, unequal hashes are not a fault. A
  comparison needs the lead at the token where the texts part and a yardstick for rounding (here: the same kernel
  with another split count), and at 32,768 tokens and beyond a second load of the reference arm.
- **Open:** why two loads differ at depth ([Q575](../../../quants/experiments.md)), and which stage turns a
  rounding difference into 0.1 of log-probability within one token
  ([Q1177](../../../quants/experiments.md)).

## Files

| File | What |
|---|---|
| [texts.tsv](texts.tsv) | per set, depth and arm: prompt tokens, the prompt's md5, generated, drafted and accepted tokens, the text's md5, the arm it is compared with and the verdict |
| [probe.tsv](probe.tsv) | per set, depth and arm, from the probes: tokens the same from the start, candidates compared, how many log-probabilities are identical, the largest difference, the smallest lead of a chosen token |
| [steps.tsv](steps.tsv) | at 7,940 tokens, per pair of arms: the largest log-probability difference at each of the first 8 generated tokens |
| [margins.txt](margins.txt) | every point where two arms' texts part: the token, each arm's candidates and the lead |
| [run/whole_model.sh](run/whole_model.sh) | the driver: one container and llama-server per set and arm, the requests from the host |
| [run/whole_requests.py](run/whole_requests.py) | the requests and the probes |
| [run/compare.py](run/compare.py), [run/steps.py](run/steps.py), [run/margin.py](run/margin.py) | the three tables and the margins from the saved answers |

## Commands

On the machine with the cards, Docker and `python3` (the prompts are its `/usr/lib/python3*/*.py`), the build's tree
unpacked and the model's shards and the draft in place:

```bash
. release/flash-next-v3/settings.env
export BIN=<build tree>/bin MODEL=<models>/$PRESET_MODEL DRAFT=<models>/$PRESET_DRAFT
export GPUS=<three cards, comma-separated> MOUNTS="<folders the container reads>" DOCKER_ARGS="--cpu-shares=256"
export PRESET_ENV="${PRESET_ENV/ GGML_CUDA_KEEPALIVE=decode/}"
R=docs/data/2026-11-09_sparse_attn_whole_model/run
S=GGML_CUDA_FATTN_SPARSE_PASCAL

# the first pass: both sets, off and on, then the 1.01 fault and a second off
export OUT=<work>/out
ARMS="off on" bash $R/whole_model.sh
ARMS="fault offfault" bash $R/whole_model.sh

# the second pass: repeats and split counts at 8,192 with 128-token probes, fault 3 at every depth
export OUT=<work>/diag
SETS=plain DEPTHS=8192 PROBE_TOKENS=128 ARMS="off on" TAG=r2 bash $R/whole_model.sh
SETS=plain DEPTHS=8192 PROBE_TOKENS=128 ARMS=on TAG=s1 EXTRA_ENV=${S}_SPLITS=1 bash $R/whole_model.sh
SETS=plain DEPTHS=8192 PROBE_TOKENS=128 ARMS=on TAG=s32 EXTRA_ENV=${S}_SPLITS=32 bash $R/whole_model.sh
SETS="plain draft" ARMS=on TAG=f3 EXTRA_ENV=${S}_FAULT=3 bash $R/whole_model.sh

# the tables
mkdir -p <work>/all/resp <work>/all/tables && cp <work>/out/resp/*.json <work>/diag/resp/*.json <work>/all/resp/
python3 -I $R/compare.py <work>/all/resp <work>/all/tables off.r2=off on.r2=on on.s1=on on.s32=on on.f3=on
python3 -I $R/steps.py <work>/all/resp plain 8192 8 off.r2=off on.r2=on on=off on.s1=off on.s32=off on.s1=on \
    on.s32=on on.s1=on.s32 fault=on on.f3=on > <work>/all/tables/steps.tsv
python3 -I $R/margin.py <work>/all/resp plain 8192 on off on.r2 off.r2    # and the other pairs of margins.txt
```

`whole_model.sh` ran detached (`setsid -f nohup`, a marker file at the end). The first pass ran before the driver had
`PROBE_TOKENS`, `EXTRA_ENV` and `TAG`; their defaults leave its requests as they were. In the first run of the second
pass the draft set's `on.f3` arm did not start, because Docker was still removing the previous arm's container when
the next one asked for its name; the driver now waits for the name, and that arm ran again by itself.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), the v3.0 release build. The prompts are the machine's own Python 3.14
standard-library source (PSF licence), built there and not stored; the tables hold the model's token ids and short
token texts at the points where texts part, not the answers. Model: Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0
([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open License 1.0 and
qwen-community-1.0), re-sharded by [scripts/gguf_reshard.py](../../../scripts/gguf_reshard.py); the MTP draft is the
preset's.
