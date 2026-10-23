# Pipelined speculation: llama-pipespec checks and rig plan (Q87)

`llama-pipespec` (in the fork, branch `pascal-p106-v2-pipespec`, `examples/pipespec`) decodes one sequence either plainly
(one `llama_decode` per token) or pipelined: up to D guessed tokens run on the first cards while the later cards still
work on the current token, and each guess is checked by sampling the real token once and comparing (sample-and-match).
Design, prior work and results: [pipelined_speculation.md](../../docs/research/notes/pipelined_speculation.md).

| File | What it does |
|---|---|
| [check_identity.sh](check_identity.sh) | Plain run plus ~20 pipelined runs (oracle with 0/10/50 % wrong guesses, copy lookup, ngram-simple, ngram-map-k at depths 1–3, depth 0) in one process; compares tokens, per-token logits hashes, the KV state after generation and probe logits with the plain run |
| [rig_run.sh](rig_run.sh) | The timing matrix for the rig: plain, depth 0, copy lookup at depths 1–5 and three drafters at depth 3 on three prompts, oracle ceilings and the cost of wrong guesses; writes `summary.tsv` with speed-ups and an identity check per job |
| [drop_vs_cancel.py](drop_vs_cancel.py) | Event model of the cards: speed-up with every guess right, 70 % right and every guess wrong, when wrong guesses run to the end (this executor) or are skipped per card (ideal cancellation) |
| [traces.py](traces.py) | Per job: tokens/s and the median time from one sampled token to the next, split by whether a right guess (one pipeline cycle), a wrong one (a full pass plus the dropped work) or none (a full pass) was in flight behind it |
| [prompts/](prompts/) | `chat.txt` (an explanation, little to copy), `edit.txt` (add type hints to a 90-line script: partial copy), `rewrite.txt` (rename two variables in a 53-line script: almost all copy) |

## llama-pipespec in short

```
llama-pipespec -m <gguf> -ngl 99 -fa on -c 8192 [--temp 0 | --temp 1 --seed 42 ...] --pipe-chat --pipe-prompt prompts/rewrite.txt \
    -n 512 --pipe-mode pipe --pipe-depth 3 --pipe-guess copy --pipe-copy-min 2 [--pipe-out tokens.txt] [--pipe-trace t.tsv]
```

- `--pipe-mode plain|pipe`, `--pipe-depth D` (guesses in flight), `--pipe-guess none|copy|ngram-simple|spec|oracle`,
  `--pipe-spec <llama.cpp drafter>` for `spec`, `--pipe-oracle <tokens file>` and `--pipe-miss <share>` for `oracle`.
- `--pipe-jobs FILE`: one job per line as `key=value` pairs (`name mode guess depth copy_min miss oracle prompt n spec
  probe chat out state_out trace`), all with one model load; `oracle=@<name>` replays an earlier job's output.
- Every job prints a `PIPESPEC name=… tok_s=… hits=… misses=… wasted=… tokens=<hash> logits=<hash> state=<hash>
  probe=<hash>` line. `hits`/`misses` count guesses that were checked; `wasted` counts dropped tokens.
- `--pipe-trace` writes one line per sampled token: microseconds since the start, guesses in flight, and `h` (the guess
  behind it was right), `m` (wrong), `n` (no guess), `p` (plain).
- `--pipe-backend-greedy`: greedy sampling on the last card in both modes (llama.cpp's backend sampling), so only a token
  id comes back instead of the logits (Flint: 150,144 floats, 0.6 MB per token, ~3 ms on a PCIe 1.0 x1 link, queued on
  the last card behind every token's compute). `rig_run.sh` uses it unless `BG=0`.

## Rig plan

Run from the private repository's root (the `rig` SSH alias). Expected numbers come from the event model in the note;
the rig decides.

**0. Install.** The fork's commit 733093c58 is built as `llama.cpp-20261023-733093c58-pascal-p106-v2-pipespec.tar.gz`
(by `TESTS=1 llms-on-crypto-mining-rigs/scripts/build_llamacpp_docker.sh fork/pascal-p106-v2-pipespec`; rebuild the
same way after new commits, ~5 min with ccache):

```
scp ~/llama-builds/llama.cpp-20261023-733093c58-pascal-p106-v2-pipespec.tar.gz rig:/opt/llama-builds/
ssh rig 'cd /opt/llama-builds && tar xzf llama.cpp-20261023-733093c58-pascal-p106-v2-pipespec.tar.gz && ls llama.cpp-20261023-733093c58-pascal-p106-v2-pipespec/bin/llama-pipespec'
scp -r llms-on-crypto-mining-rigs/bench/pipespec rig:/tmp/
```

The commands below use `$B`; set it in your shell first: `B=/opt/llama-builds/llama.cpp-20261023-733093c58-pascal-p106-v2-pipespec/bin`.

**1. Identity on the P106 kernels** (six cards, S4-L0, ~5 min each plus the model load), sampling on the host and
on the last card. Every row must say `same`:

```
ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf \
    ARGS='-ngl 99 -fa on -c 4096' N=128 DEPTHS='1 3 5' MISSES='0 0.3' OUT=/tmp/pipespec/check-s4l0 \
    bash /tmp/pipespec/check_identity.sh"
ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf \
    ARGS='-ngl 99 -fa on -c 4096 --pipe-backend-greedy' N=128 DEPTHS='1 3 5' MISSES='0 0.3' \
    OUT=/tmp/pipespec/check-s4l0-bg bash /tmp/pipespec/check_identity.sh"
```

**2. Timing, S4-L0** (each run ~15 min plus the load; start detached, one at a time, nothing else on the cards):

```
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=s4l0-6c GPUS=0,1,2,3,4,5 \
    ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/s4l0-6c.out 2>&1 < /dev/null &"
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=s4l0-9c GPUS=0,1,2,3,4,5,6,7,8 \
    DEPTHS='1 2 3 4 5 6 8' ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_run.sh > /tmp/pipespec/s4l0-9c.out 2>&1 < /dev/null &"
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=s4l0-9c-light GPUS=0,1,2,3,4,5,6,7,8 \
    TS=6,6,6,6,6,6,6,5,2 DEPTHS='1 2 3 4 5 6 8' ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_run.sh \
    > /tmp/pipespec/s4l0-9c-light.out 2>&1 < /dev/null &"
```

Wait for each with `ssh rig 'tail -2 /tmp/pipespec/<tag>.out'` (the last line is the summary table once done).

**3. Timing, G3**: the same three commands with `MODEL=<G3 on a stick>` and `TAG=g3-…`, rescale off as in its KLD
scoring (`ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1`). G3 (29.6 GB) is not on the rig yet.

**4. The cost of host sampling** (six cards, logits to the host for every token, guesses included):

```
ssh rig "setsid nohup env BIN=$B MODEL=/mnt/models2/flint/flint-S4L0.gguf TAG=s4l0-6c-host BG=0 GPUS=0,1,2,3,4,5 \
    PROMPTS=rewrite DEPTHS='1 3 5' ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 bash /tmp/pipespec/rig_run.sh \
    > /tmp/pipespec/s4l0-6c-host.out 2>&1 < /dev/null &"
```

**5. Collect and clean up:**

```
mkdir -p results/2026-10-24_q87_pipespec && scp -r rig:/tmp/pipespec/out rig:/tmp/pipespec/*.out results/2026-10-24_q87_pipespec/
ssh rig 'rm -rf /tmp/pipespec'
python3 -I llms-on-crypto-mining-rigs/bench/pipespec/traces.py results/2026-10-24_q87_pipespec/out/s4l0-6c
```

**What to read from `summary.tsv`:**

| Rows | Question | Expected (event model) | If not |
|---|---|---|---|
| `same_as_plain` | Is every pipelined job's output plain decoding's output? | `yes` everywhere | a bug: keep the `run.log` and the tokens |
| `d0-*` | Does the pipelined API cost anything by itself? | speed-up 0.98–1.02 | host time per submit (traces) |
| `oracle-dD-m0` | The ceiling: one token per slowest stage | 6 cards 2.0 / 3.0 / 4.0 / 4.5 / 4.5 at D = 1–5; 9 cards up to 6.2 (even) and 8.9 (light last card) | per-stage overheads: CUDA graph launches per slot, mask copies, the 75 W clock under load |
| `oracle-d3-m1` | Wrong guesses cost nothing at D ≤ S_eff − 1 | ≥ 0.98 | the host or the last card's logits copy is the bottleneck |
| `oracle-d3-m0.3` | 70 % right guesses | ~1.9× (6 cards) | – |
| `copy2-dD-rewrite`, `-edit`, `-chat` | Real gain per kind of prompt | rewrite ≫ edit > chat ≈ 1.0 | compare hits/misses with the n-gram replay |

Then pick D per card count (the event model says D ≤ 3 on six cards, where the output head makes the last card the
slowest stage) and the guesser for Q53, and decide whether a per-stage launch or sampling on the GPU is worth building
first (see "What a production version needs" in the note).
