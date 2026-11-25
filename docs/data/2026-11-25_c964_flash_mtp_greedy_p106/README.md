# C964: Flash-Next's greedy replies with the MTP draft against plain decoding, on a P106

What this is: registry [Q1694](../../../quants/experiments.md) (C964), on the rig's nine P106-090 cards at stock
clocks, 2026-11-25. On three RTX 3090s running the rig build through its compute 6.1 PTX,
[C951](../2026-11-21_c951_flash_head_gpu/README.md) found that Flash-Next's v5 preset with an MTP draft leaves plain
decoding's greedy text on two of three requests (the prose paragraph at token 56, the deep prompt at token 32; the
coding request held). [Q971](../../../quants/experiments.md) had seen the same on a CPU. This row asks whether a P106,
with the fork's Pascal kernels, keeps plain decoding's text. Correctness only, nothing timed.

**In plain words:** no. On the P106 the draft's greedy replies leave plain decoding's text on all three requests,
the coding one too, each time at a token where the two most likely candidates are almost equally likely and the
draft's verify pass ranks them the other way round from a one-token step. So a check on real files cannot take plain
decoding as the reference for a drafting arm on a P106 either: it compares drafting arms with each other (Q971's rule).
The P106 also does not write the 3090's replies: on the deep prompt both of its arms start with another first token.

## The result

[compare.tsv](compare.tsv): the draft arm against plain decoding, token ids of the same request (counted from 0), and
the two most likely tokens at the first differing position with their probabilities, from each arm's own reply.

| Request | Same as plain decoding | First differing token | Draft arm's two candidates there | Plain decoding's two candidates there |
|---|---|---|---|---|
| code | no | 124 | "fraction" 0.4700, "with" 0.4681 | "with" 0.4791, "fraction" 0.4591 |
| prose | no | 67 | " index" 0.3816, " slot" 0.3212 | " slot" 0.3451, " index" 0.3206 |
| deep (8,189 prompt tokens) | no | 21 | " forward" 0.1959, " graph" 0.1948 | " graph" 0.2109, " forward" 0.1930 |

The tokens before each of these positions are the same in both arms, so each pair of candidates is read on the same
text: the verify pass of the draft arm (a pass over the accepted token and up to three drafted ones) and plain
decoding's one-token pass give the same two candidates in the other order. On the coding request the gap is 0.0019
against 0.0200; on the deep prompt 0.0011 against 0.0179; on prose the verify pass puts " index" 0.0604 ahead where
the one-token pass puts " slot" 0.0245 ahead. Where the texts go after that:

| Request | Text before | Plain decoding goes on | The draft arm goes on |
|---|---|---|---|
| code | `YYYY-MM-DDTHH:MM:SS.fff (` | `with fractional seconds)` | `fractional seconds)` |
| prose | `bucket holds a linked list of entries that hash to the same` | ` slot.` | ` index.` |
| deep | `` _FIRST=1`: the draft's embedding lookup runs before its `` | ` graph is captured, so the` | ` forward pass, not after.` |

**Against C951's 3090 replies** ([vs3090.tsv](vs3090.tsv); C951's arms on the same target file, round 1, the branch
build `c9c3134d4`):

| Request | P106 plain decoding against the 3090's | P106 draft arm against the 3090's draft arm |
|---|---|---|
| code | the same 256 token ids | differ from token 124 (the P106 draft arm's own flip above) |
| prose | differ from token 131 | differ from token 56 |
| deep | differ from token 0 | differ from token 0 |

- On the deep prompt the first token comes from the prompt's last pass, so both P106 arms give it the same
  probabilities: "-" 0.3869 and "LL" 0.3482. The 3090s chose "LL". Reading 8,189 prompt tokens on the P106 and on a
  3090 already ends on different rankings of two close candidates, as
  [FN93](../2026-11-16_fn93_ornith_greedy_split/README.md) found for Ornith: a newer card does not write the rig's
  replies.
- On prose the 3090's flip at token 56 (the draft arm " stores", plain decoding " holds") does not happen on the P106:
  both P106 arms write " holds" there (the draft arm 0.3556 against 0.3455, plain decoding 0.3502 against 0.3075). The
  P106's draft arm flips 11 tokens later instead.
- The draft is not the same file: C951 used Swift's own MTP head on the 3090s, this run the Gyro MTP draft the
  preset names. C951 found both of its heads leave plain decoding's text at the same tokens, so the head is not what
  makes a flip; the flips come where a verify pass and a one-token pass rank two close candidates differently.

**What this decides:** checks on real files on a P106 may not take plain decoding as the reference for a drafting arm's
greedy replies. They compare drafting arms with each other (a branch against its base, one head against another, both
with the draft), and a P106 check takes its references from a P106. Finding the first tensor where a verify pass and
one-token passes part stays Q971's work; how large that drift is on the P106's code paths, measured over whole replies,
is [Q1861](../../../quants/experiments.md).

**The second pass:** each load answered the three requests twice, once as C951 sent them (`p1`) and once asking for the
two most likely tokens at every generated position (`p2`, `n_probs` 2, which keeps the server's sampling on the host).
Both passes gave the same 256 token ids on every request in both arms ([replies.tsv](replies.tsv)), so the candidates
above are read from replies equal to the ones compared. The draft arm's drafted and accepted counts change from pass to
pass (code 242/174 and 236/176) while its replies stay the same, as they did between C951's two rounds.

**The loads** ([loads.tsv](loads.tsv)): both loaded and stopped cleanly (exit 0). The draft arm took 5,545 MiB on card 8
(the draft's card) after the load, plain decoding 2,359 MiB; cards 0–7 hold 26 to 46 MiB more with the draft.

## Files

| File | What |
|---|---|
| [compare.tsv](compare.tsv) | per request: the draft arm against plain decoding, the first differing token, both arms' two candidates there, each arm's two passes alike |
| [vs3090.tsv](vs3090.tsv) | per request and arm: the token ids against C951's 3090 arm (md5 equal, first differing token) |
| [replies.tsv](replies.tsv) | one line per arm, pass and request: prompt tokens, generated tokens, md5 of the token ids and of the text, drafted and accepted, stop |
| [tokens.tsv](tokens.tsv) | the generated token ids of every reply of pass `p1` |
| [loads.tsv](loads.tsv) | each load: seconds to `/health`, the server's exit, MiB per card after the load and after the requests |
| [run/c964.sh](run/c964.sh) | the driver, run on the rig: one server per arm, the requests, the memory |
| [run/session.py](run/session.py) | C951's three greedy requests, then the same with `n_probs` 2 |
| [run/c964_tables.py](run/c964_tables.py) | the tables above from the driver's output (and C951's replies, for the 3090 columns) |

## How it ran

- **Rig:** nine P106-090 cards at stock clocks (the memory offsets of the day were reset before the run), driver
  580.178.04, nothing else running.
- **Engine:** Miner Llama's `pascal-p106-v5-cand-b12674` at `eee615cdb` (the v5 engine with upstream tag `b12674`
  merged), the rig build `llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674`, `llama-server` md5
  `2727faf28b5b34dfe42399a9d431da67`.
- **Line:** [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) as it is (its
  `PRESET_ENV` and `PRESET_ARGS`, the layer split `-ts 6,6,6,6,5,6,5,6,3` on nine cards). Arm `mtp`: `-md` the Gyro MTP
  draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`, 2,697,439,648 bytes) with every `--spec-*` flag. Arm `plain`: the same
  line without `-md` and without the `--spec-*` flags. Target: Swift GSQ-RCO Q2_0 (UkisAI's Swift-1.5, its n-gram
  table cut into a third shard).
- **Requests:** C951's three greedy ones (temperature 0, top-k 1, 256 tokens, no prompt cache): C854b's coding request
  (27 prompt tokens), its prose paragraph (19) and the deep prompt, the first 170 lines of `docs/switches.md` as it
  stood at commit `6a9cf9d9` plus a question (8,189 prompt tokens; md5 `dea4956d2c52140e5ebfedf0dd46c770`, C951's).
- **Times:** the draft arm loaded in 307 s, plain decoding in 287 s; the whole run took 17 minutes.

```bash
# on the laptop: the run's files into one folder on the rig
git show 6a9cf9d9:docs/switches.md > switches-6a9cf9d9.md
scp docs/data/2026-11-25_c964_flash_mtp_greedy_p106/run/c964.sh docs/data/2026-11-25_c964_flash_mtp_greedy_p106/run/session.py \
    release/flash-next-v5/settings.env switches-6a9cf9d9.md "$RIG:$WORK-run/"
# on the rig, detached (BIN the build's bin/ folder, MODELS the folder with the target's shards and the draft)
ssh "$RIG" "cd $WORK-run && mkdir -p $WORK && WORK=$WORK BIN=$BIN \
  MODEL=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  MODEL_DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf \
  PRESET=$WORK-run/settings.env SESSION=$WORK-run/session.py TEXT=$WORK-run/switches-6a9cf9d9.md \
  setsid -f nohup bash c964.sh > $WORK.driver.log 2>&1 < /dev/null"
# when $WORK/DONE exists: copy $WORK back, then the tables (C951's raw replies are optional)
python3 -I docs/data/2026-11-25_c964_flash_mtp_greedy_p106/run/c964_tables.py <copy of WORK> \
    docs/data/2026-11-25_c964_flash_mtp_greedy_p106 docs/data/2026-11-21_c951_flash_head_gpu/replies.tsv [C951's out/]
```

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp). Model files: UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the
Gyro MTP draft, as the [v5 preset](../../../release/flash-next-v5/settings.env) names them. The tables hold token ids,
md5s and short pieces of the model's replies; the full replies and the server logs are not published.
