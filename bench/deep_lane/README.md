# The deep lane: speed at fixed context depths, arm against arm

A coding agent's context grows with every turn, and generation slows as it grows: each new token reads the whole
cache. The bug-fix tasks of the agent check ([agent_check](../agent_check/README.md)) end below 10,000 tokens of
context, so they do not show what a setting does at 30,000 or 100,000. This tool holds the depth fixed instead. It
loads llama-server on the line of a release preset, sends prompts of exactly 0, 8,192, 16,384, 32,768 and 49,152
tokens cut from recorded agent text, and generates 256 tokens after each, for two or more settings (arms) whose loads
alternate. Per depth it reports every load's speed by value, the spread between loads, and each arm's class against
the control: gain, negligible or loss.

Its numbers are single-request diagnostics at a known depth. They explain the agent check's results and do not
replace them. The registry rows it serves are Q1143 (the lane), Q1153 (the MTP draft by depth), Q1154 (a
131,072-token line), Q1219 (196,608 and 262,144 tokens) and Q1172 (draft acceptance by depth) in [quants/experiments.md](../../quants/experiments.md). The
method follows the depth sweep of
[docs/data/2026-11-06_flint_q5_depth](../../docs/data/2026-11-06_flint_q5_depth/README.md).

## Files

| File | What it does |
|---|---|
| [deep_lane.sh](deep_lane.sh) | The driver. Runs on the machine that holds the cards, detached: per load one llama-server, the quiet check, the RAM watchdog, the client; at the end the tables |
| [deep_client.py](deep_client.py) | One load's requests on the loopback: builds the prompts, sends them, writes one CSV line per request |
| [deep_table.py](deep_table.py) | The tables and classes from `requests.csv` |
| [remote.sh](remote.sh) | Starts the driver on a rig over ssh, waits for it, collects the results and cleans up |
| [selftest.sh](selftest.sh) | The whole tool against [mock_server.py](../agent_check/mock_server.py), no model and no GPU |

Bash, curl and Python 3.9 or newer (standard library only) on the machine that runs the driver.

## How a run goes

**The server line** comes from a preset file, by default
[release/flash-next-v3/settings.env](../../release/flash-next-v3/settings.env): its `PRESET_ENV`, `PRESET_ARGS`, model
and draft. Nothing about the machine is in the scripts. `BIN`, `MODELS`, `TEXT` and `OUT` name the files, and the line
is changed through the environment:

| Variable | Effect on the line |
|---|---|
| `TS` | the card split, replacing the preset's `-ts` (for example `1,1,1` on three equal cards) |
| `CTX`, `UB` | context size and micro-batch, replacing `-c` and `-ub` |
| `DRAFT=off` | the line without `-md`, the `--spec-*` flags and the draft's other flags |
| `ARGS_DROP`, `ARGS_SET` | flags removed with their values; `flag value` pairs replaced or added |
| `ENV_DROP`, `ENV_ADD` | variables removed from `PRESET_ENV`; variables added |
| `RUNNER`, `STOP_CMD` | a command the server line is handed to (a container), and one that runs after each stop |

**An arm is a named setting.** `ARMS="dense sparse"` with `ARM_ENV_dense=GGML_CUDA_FATTN_SPARSE_PASCAL=0` and
`ARM_ENV_sparse=GGML_CUDA_FATTN_SPARSE_PASCAL=1` compares the sparse Pascal attention off and on. Besides
`ARM_ENV_<arm>` an arm can set `ARM_DRAFT_<arm>` (on or off), `ARM_CTX_<arm>` and `ARM_UB_<arm>` (131,072 tokens need
`-ub 256` on nine 6 GB cards), `ARM_ARGS_<arm>` (flags, as `ARGS_SET`) and `ARM_BIN_<arm>` (its own llama-server, so that two
builds alternate in one run; its `--version` goes to `build-<arm>.txt`). The first arm is the control.

**Loads alternate.** One untimed warm-up load of the first arm comes first, because the first load on an idle rig
runs slower than later ones (registry row Q1055). Then every arm loads once per repetition (`REPS`, default 2):
`dense1 sparse1 dense2 sparse2`. Each load is a fresh server start. A finished load leaves a marker, so a run that was
cut short resumes with the same command; a results folder holding loads of other settings is refused.

**The quiet check** runs before every load: no llama process, at most 100 MiB on any card, with the cards'
temperature and clock and the host's free memory written to `quiet_<load>.txt`. A machine that is not idle stops the
run. `QUIET=0` skips the check for a machine where speed is not measured, and marks every request `valid=0`; the
tables of such a run print no speed. On a host under 8 GB the server runs under
[scripts/ram_watchdog.sh](../../scripts/ram_watchdog.sh), which stops it before the host runs out of memory.

**The prompts** are cut from recorded agent text passed by path (`TEXT`), never from text in this repository:

- Recorded chat requests (the agent check's `proxy.jsonl.bodies.gz`, optionally filtered by tag with `TAG`) count once
  per session, as the session's last request, rendered by the server's own chat template with its tools. Plain text is
  read as it is.
- The server tokenizes the sessions into one stream of ids. A prompt of depth D is the stream's first ids followed by a
  fixed tail of 64 ids, D ids together.
- The tail is the end of one recorded request: the end of its last message and the template's start of the assistant's
  turn. The model therefore writes the same turn at every depth, and only the context in front of it grows. The
  request chosen is the one whose recorded reply was the longest, and its session is left out of the stream.
- Depth 0 is the tail alone. The ids are kept in `prompts.json`, so every load sends the same prompts.

**The requests** of a load, depths ascending, each to `/completion` with token ids as the prompt:

1. `prime`: the prompt without its tail, no output asked. The server reads what its cache lacks, which is a chunk on
   top of the previous depth. Its prompt timings are the prompt reading at that depth.
2. `decode`, pass 1 and pass 2: the whole prompt, 256 tokens at temperature 0 with end of text ignored. Pass 1 is the
   first request at a new depth, as every turn of an agent is; pass 2 repeats it.
3. `sample` (with `SAMPLER_PASS=1`): once more with the server line's own sampler and a fixed seed.

`PAUSE=<seconds>` makes the client wait before every request. Nothing the server times changes, and a profiler's trace
(`GGML_CUDA_PROFILE` in an arm's `ARM_ENV_<arm>`) then falls into one piece per request at the idle gaps.

`CACHE=0` drops the prime and makes every request read its whole prompt fresh. That is the method of the Flint depth
sweep, and it costs the full prompt time for every request: at the 69.4 prompt tok/s the release preset ran on the
agent check, a 49,152-token prompt is est. 12 minutes.

A model whose state cannot be rolled back token by token reuses the cache through llama-server's context checkpoints,
which are taken four tokens before the end of each prompt. The prime puts one right in front of the tail, so the next
depth continues from there. The `cache_n` column and the table's last check show what each prime read. A prime that
read its whole prompt means the server found no checkpoint: the run is still correct, only longer.

## Running it

On a rig, from a machine that only runs ssh (`RIG` is the rig's ssh alias; `BIN` and `MODELS` are paths on the rig;
`TEXT_LOCAL` is copied over):

```bash
RIG=rig BIN=/opt/llama-builds/<build>/bin/llama-server MODELS=<model folder on the rig> \
  TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^<arm>/' WD_ENV=MIN_AVAIL_MB=60 \
  ARMS="dense sparse" ARM_ENV_dense=GGML_CUDA_FATTN_SPARSE_PASCAL=0 ARM_ENV_sparse=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  bash bench/deep_lane/remote.sh start
RIG=rig bash bench/deep_lane/remote.sh wait
RIG=rig OUT=<results folder> bash bench/deep_lane/remote.sh collect
```

`start` copies this folder, the preset and the watchdog to `/tmp/deep-lane` on the rig and starts the driver there in
its own session, so a dropped connection does not stop it. `wait` is one ssh call that returns when the run has ended.
`collect` brings the results back and removes the folder on the rig. `DRY=1` with `start` prints each arm's server
line and starts nothing.

On a machine with newer cards, through the build's compute 6.1 PTX, the server runs in a container and the driver on
the host. The times of such a run are not measurements (`QUIET=0`); the reply hashes and the draft counters are valid:

```bash
BIN=<build>/bin/llama-server MODELS=<model folder> TEXT=<recorded>/proxy.jsonl.bodies.gz OUT=<results folder> \
  QUIET=0 TS=1,1,1 ARGS_DROP="-otd" ENV_DROP=GGML_CUDA_KEEPALIVE ENV_ADD=CUDA_CACHE_PATH=<results folder>/jit \
  RUNNER='docker run --rm --init --name deep-lane --network host --gpus "device=0,1,2"
          -v <build>:<build> -v <model folder>:<model folder> -v <results folder>:<results folder>
          nvidia/cuda:12.9.2-devel-ubuntu24.04' \
  STOP_CMD="docker rm -f deep-lane" ARMS="draft nodraft" ARM_DRAFT_nodraft=off SAMPLER_PASS=1 \
  setsid -f nohup bash bench/deep_lane/deep_lane.sh > console.txt 2>&1 < /dev/null
```

- A build that holds only sm_61 code and compute 6.1 PTX needs no switch for this: a newer card cannot run the sm_61
  code, so its driver compiles the PTX. `CUDA_CACHE_PATH` keeps the compiled code between loads.
- The server's variables go on the command line the runner receives (`env NAME=value … llama-server …`), so they
  reach the server inside the container. With `--network host` the server's loopback port is the host's.
- `-otd token_embd=CUDA4` in the preset names the fifth card: drop it, or point it at a card that exists, when fewer
  cards are used. `ARGS_DROP="-ts -otd"` leaves the split to llama-server.

Every variable is listed at the top of [deep_lane.sh](deep_lane.sh).

## What it writes

| File in `OUT` | Content |
|---|---|
| `requests.csv` | one line per request, every load |
| `loads.tsv` | one line per load: arm, timed or warm-up, valid, draft, context, start, load seconds, the line as it ran, the binary |
| `table.md`, `classes.tsv` | the tables below; the classes as a table |
| `quiet_<load>.txt`, `fit_<load>.txt` | the quiet check before the load; memory per card and on the host after it |
| `server-<load>.log`, `ram_watchdog-<load>.tsv`, `gpu.csv` | the server's log, the watchdog's samples, the cards every 10 s |
| `<load>.jsonl`, `prompts.json` | each request's whole `timings` and the first 160 characters of its reply; the prompts' token ids |
| `done` or `failed` | how the run ended |

The columns of `requests.csv`: `arm`, `load`, `kind` (warmup, prime, decode, sample), `pass`, `depth`,
`prompt_tokens` (ids sent), `cache_n` (reused), `prompt_n` and `prompt_ms` (read), `gen_n` and `gen_ms` (generated),
`text_md5` (the reply's text), `ids_md5` (its token ids), `draft_n` and `draft_accepted` (the server's draft counters),
`tokens_per_step`, `prompt_md5`, `cache_src`, `valid`, `wall_s`, `time`.

`requests.csv`, `loads.tsv`, `table.md` and `classes.tsv` hold no recorded text. `prompts.json`, the `.jsonl` files
and the server logs hold token ids, reply text and paths: publish them only if the recorded text may be published.

## The tables

- **Generation, tok/s:** per depth and arm, each load as `pass 1, pass 2 → load`, where a load's speed is its two
  passes' tokens over their time, with the load's ms a token. Then the spread, the arm against the control in tok/s
  and percent, and the class.
- **Class:** the spread is the larger of the two arms' differences between their fastest and slowest load. An arm is a
  gain when its mean is above the control's by more than the spread, a loss when below by more than it, and negligible
  otherwise. With one load per arm there is no spread and no class. `--class-on pass1` classes on pass 1 alone.
- **The step on pass 1 and on pass 2, ms** (when a draft ran): the time of one step per load, with the same columns;
  for a time, less is the gain. At depth two loads of one arm can write different replies, and the draft's acceptance
  differs with the reply, so tok/s mixes the time of a step with the tokens a step yields. The step's time is the part
  a kernel changes. An arm whose draft is not in the control's state gets no class here: its steps yield other
  tokens.
- **Prompt reading, tok/s:** the tokens each prime read over their time, with the tokens read and reused.
- **The draft and the step:** draft tokens accepted of drafted, tokens a step and ms a step. A step is one pass of the
  main model. It yields one token of its own and the draft tokens it accepts, so a request's steps are its generated
  tokens minus the accepted ones.
- **Checks:** the same prompt ids on every load, the tokens out, and per depth whether the greedy reply is the same on
  every pass and load of every arm. Arms that change only speed must write the same reply. Where the control's own
  loads wrote different replies, the line counts the replies per arm instead and says whether every load wrote its
  own reply again on pass 2: a difference between arms cannot be told from one between loads there.

Every value is one load of one arm at one depth. A number quoted from a table names its arm, load and depth.

## Checked against the mock server

`bash selftest.sh <work folder>` runs the driver on the real release preset with
[mock_server.py](../agent_check/mock_server.py) in place of llama-server, on recorded text it writes itself, with
stand-ins for `nvidia-smi` and `ps`. For this the mock speaks `/tokenize`, `/apply-template` and `/completion`, and its
`timings` carry `cache_n`, `cache_src`, the per-second fields and, when the line has `-md`, `draft_n` and
`draft_n_accepted`. It rolls its cache back to checkpoints as a recurrent model's server does.

On 2026-11-10 all 36 checks passed, on a Linux machine's CPU under Python 3.14:

| Run | What it showed |
|---|---|
| five arms × 2 loads, quiet checks on | 164 requests; every prompt exactly its depth; the preset's nine-card line unchanged; the draft-off arm without `-md`, `--spec-*` and `-otd`; the context arm with `-c 131072 -ub 256`; an arm as fast as the control negligible, a faster one a gain, the one without the draft a loss, at every depth; the step's time classed the same way, with no class for the arm without the draft; the same greedy reply in every arm; each prime read only the chunk past the previous depth's checkpoint |
| the same command again | every load skipped, the same `requests.csv` |
| the same folder, another setting | refused, the folder untouched |
| a llama process running | refused before any load |
| quiet checks off, a runner, three equal cards | every request `valid=0`, no speed in the tables, `-ts 1,1,1`, `-otd` dropped, a `CUDA_CACHE_PATH` added and the keep-alive variable dropped |
| `CACHE=0` on plain text | no prime, every request read its whole prompt |
| a depth the text cannot fill | stopped in the warm-up load, before any timed load |
| a depth the context cannot hold | refused before any load |
| `deep_table.py` on hand-made speeds | inside the spread negligible, above it a gain, below it a loss; each load's ms a token; replies that differ between the control's own loads counted per arm |

Four more checks on the same day:

- The driver ran on requests a real agent recorded: the 15 sessions of the agent check's own self-test (pi against the
  mock, through the logging proxy), at depths 0, 2,048, 8,192 and 16,384.
- `remote.sh` started a run over ssh with the RAM watchdog, waited for it, collected it and removed its folder.
- With a `docker run` line as the `RUNNER` (the mock in a container on the host's network), the variables reached the
  server, each load stopped cleanly and no container was left.
- The agent check's self-test still passes with the extended mock (11 checks).

The mock has no model, so these checks cover the driver, not llama-server.

## Run against llama-server

The first run against llama-server counted the MTP draft's acceptance by depth on 2026-11-10
([docs/data/2026-11-10_draft_acceptance_by_depth](../../docs/data/2026-11-10_draft_acceptance_by_depth/README.md)):
the release preset through the build's PTX on three newer cards, quiet checks off, five loads, depths 0 to 49,152 on
the release check's recorded requests.

- The server's template rendered the recorded requests with their tools, and every prompt had exactly its depth in
  tokens.
- The primes continued from a context checkpoint. At 16,384, 32,768 and 49,152 they read 8,708, 16,388 and 16,900
  tokens on 7,612, 16,316 and 32,188 cached: the checkpoint the server restores lies 4 or 516 tokens before the end
  of the previous prime, so a prime can read 512 tokens more than its chunk.
- A request that asks for no output generates one token.
- Each arm wrote the same greedy reply on pass 1 and pass 2. The arm without the draft wrote another reply than the
  arm with it at every depth, so the check "the same greedy reply in every arm" fails between those two arms. The
  cause is not a fault of the draft ([docs/data/2026-11-10_draft_line_exactness](../../docs/data/2026-11-10_draft_line_exactness/README.md),
  registry row Q1175): a pass over two or more tokens is another arithmetic than a one-token pass, and from 8,192
  tokens on two loads of one arm write different replies too. Read that check as exact only at depth 0 and between
  arms whose passes have the same shape.
- **The first run on the rig** timed the sparse Pascal attention off and on at the same depths on nine P106-090
  cards on 2026-11-10 ([docs/data/2026-11-10_deep_lane](../../docs/data/2026-11-10_deep_lane/README.md)): 70
  minutes for a warm-up load and four timed loads, every load checked quiet, the RAM watchdog silent on the 4 GB
  host. From 8,192 tokens on each load wrote its own greedy reply and accepted another share of the draft, so two
  loads of one arm ran 16.82 and 19.97 tok/s at 8,192 while their steps took 153.00 and 152.97 ms. The step tables
  come from that run: class a kernel on the step's time, and read tok/s beside the draft's counters.
- **The second run on the rig** put the line with its MTP draft against the same line without it
  (`ARMS="draft nodraft" ARM_DRAFT_nodraft=off`), 62 minutes, same data page. The arm without the draft repeats to
  within 0.26 tok/s between two loads at every depth, since it yields one token a step whatever it writes, while
  the draft arm's two loads were up to 5.35 tok/s apart. Read such a pair on the break-even: the draft's step over
  the plain line's token is the tokens a step the draft must yield, and it does not depend on the reply
  ([run/draft_pair.py](../../docs/data/2026-11-10_deep_lane/run/draft_pair.py) prints it). The prime requests of the
  two arms also time what the draft adds to prompt reading.
- **The third run on the rig** held the same two lines at a 131,072-token context (`CTX=131072 UB=256`) at 49,152,
  65,536 and 120,000 tokens, one load an arm, 68 minutes, same data page. One arm's 12 recorded sessions hold too
  few tokens for 120,000, so the prompts came from two arms' (`TAG='^(t1swift|t1base)/'`, 24 sessions); every prime
  continued from a checkpoint, and the longest read 54,724 tokens on 65,212 cached. `LOADS` put the line without the
  draft first, so that its numbers would be in if the draft line ran out of host memory. Run the projection of
  [scripts/fit_probe.sh](../../scripts/fit_probe.sh) and a load check (`WARM=0 REPS=1 DEPTHS=0`) before such a run:
  together they take a few minutes and read every card's memory before an hour of loads.
- **A run can be extended later.** The seventh run on the rig gave the third one its second load of each arm, five
  and a half hours later, 68 minutes: the collected folder went back to the rig's `out` folder and the same command
  ran with a longer `LOADS` (`... nodraft2:nodraft draft2:draft`). The settings check (`settings.md5`, which leaves
  `LOADS` out) let the folder resume, the driver skipped the two finished loads, ran a warm-up load and the two new
  ones, and wrote the tables of all four. The new warm-up replaces the old one's per-load files.
- **The fourth run on the rig** read prompts only: four arms (`--spec-draft-n-ubatch` 128, 256 and 512 through
  `ARM_ARGS_<arm>`, and the line without the draft), two loads each, 108 minutes, same data page. With
  `N_PREDICT=16` a load with the draft took 14 minutes instead of the second pair's 16, and the primes' numbers
  repeated that pair's. `LOADS` put the arm most likely to be stopped by the RAM watchdog last in each round, since a
  stop ends the run; it was not stopped. [run/draft_ubatch.py](../../docs/data/2026-11-10_deep_lane/run/draft_ubatch.py)
  reads such a run: prompt reading of every arm against the control and the floor, and the host's memory per load.
- **The fifth run on the rig** profiled a step at depth: `LOADS="warm:sparse prof:prof"` with the profiler in
  `ARM_ENV_prof` and `PAUSE=4`, depths 16,384, 32,768 and 49,152, the unprofiled load first at the same depths
  (`WARM_DEPTHS`), 44 minutes
  ([docs/data/2026-11-10_step_parts_depth](../../docs/data/2026-11-10_step_parts_depth/README.md)). The trace held
  the whole load, prompts included, in 85.8 MB, and cut into the load and its 10 requests at the pauses.
- **A later run on the rig took the host profile at depth:** `GGML_CUDA_PROFILE_HOST` in `ARM_ENV_host` with
  `PAUSE=4`, depths 16,384 and 49,152, 33 minutes (same data page,
  [run/host_depth.py](../../docs/data/2026-11-10_step_parts_depth/run/host_depth.py)). The profile cuts at the pauses
  too, but the first decode at a depth leaves a second gap after its tail's pass, where the server saves a context
  checkpoint. Library log lines that a switch writes at the info level, such as `LLAMA_QSA_INPUT_STATS`, reach the
  server log only with `-lv 4` in the arm (`ARM_ARGS_<arm>="-lv 4"`).
- **The sixth run on the rig** read prompts in micro-batches of 128 through the sparse kernel: `-ub` per arm through
  `ARM_UB_<arm>`, the switch per arm through `ARM_ENV_<arm>` (no `ENV_ADD`, so that each server line holds the
  switch once), depths 0, 16,384, 32,768 and 49,152, `N_PREDICT=16`, two loads an arm, 80 minutes
  ([docs/data/2026-11-10_deep_lane](../../docs/data/2026-11-10_deep_lane/README.md)). A smaller `-ub` moves the
  checkpoint the next prime continues from (4 or `-ub` + 4 tokens before the end of the previous prime), so the
  primes past the first read slightly other chunks than the control's.
  [run/prompt_depth.py](../../docs/data/2026-11-10_deep_lane/run/prompt_depth.py) reads such a run: prompt reading
  per arm with each arm's own chunk, another run's loads beside where they read the same chunk, and the host's memory.
- **The eighth run on the rig** timed five settings of the sparse switch's threshold (`ARM_ENV_<arm>`, with
  `ENV_ADD` for the variable all five share) at 0, 4,096, 8,192 and 12,288 tokens, two loads an arm
  ([docs/data/2026-11-10_deep_lane](../../docs/data/2026-11-10_deep_lane/README.md)). The RAM watchdog stopped the
  fifth load, which ends a run. `collect` with `CLEAN=0` kept the rig's folder, and the same command with a `LOADS`
  that put the stopped arm last resumed it two minutes later: the four finished loads were skipped, a warm-up load ran
  again, and the stopped load ran whole. The resumed run appends to `loads.tsv`, so it lists the stopped start too;
  the tables take each load's last line. Depths 4,096 tokens apart restore an older checkpoint at the third prime
  (8,708 tokens read on 3,516 cached at 12,288). [run/threshold_table.py](../../docs/data/2026-11-10_deep_lane/run/threshold_table.py)
  reads such a run on the step, with tok/s and the share accepted beside it.
- **The ninth run on the rig** held a 196,608-token line (`CTX=196608 UB=256`) at 120,000 and 196,000 tokens, with
  and without the draft, one load an arm, 131 minutes
  ([docs/data/2026-11-10_deep_lane](../../docs/data/2026-11-10_deep_lane/README.md)). Three arms' recorded sessions
  (`TAG='^(t1swift|t1base|t1neg)/'`, 36 sessions) fill 196,000 tokens. A depth must leave the request's
  `N_PREDICT` tokens free in the context, so the deepest depth of a 196,608-token line is 196,352; 196,000 leaves
  room. With depths 120,000 and 196,000 the first prime reads 119,876 tokens on an empty cache in one request (30.5
  minutes with the draft); the warm-up load at depth 0 reads every card's memory before the deep loads, in place of a
  separate load check.
- **The tenth run on the rig** put a preset candidate against the release preset: five switches at once, two
  through `ARM_ENV_<arm>` and five flags through `ARM_ARGS_<arm>`, one of which replaces a value the preset holds
  (`--spec-draft-n-ubatch 128` became 512, checked with `DRY=1` first), and `SAMPLER_PASS=1`, since one switch
  (probabilistic verification) acts only on a request with a temperature above 0; 80 minutes for a warm-up and four
  timed loads ([docs/data/2026-11-10_v4_candidate](../../docs/data/2026-11-10_v4_candidate/README.md)).
  [run/v4c_table.py](../../docs/data/2026-11-10_v4_candidate/run/v4c_table.py) reads such a run per depth and load,
  decode, the sampler pass and prompts side by side, with a class for each.
- **The eleventh run on the rig** put two builds in one run: v4's day build with every gain and with no switch against
  the v3.0 release, `ARM_BIN_v3` naming v3.0's llama-server beside `BIN`, the loads in an order that answers the main
  question first (`LOADS="warm:v3 v31:v3 gains1:gains v32:v3 gains2:gains off1:off off2:off"`). The RAM watchdog
  stopped one load (MemAvailable 54 MiB at 32,768 tokens with the checkpoints in host memory); the folder copied back
  and `remote.sh start` again resumed the run, skipping the four finished loads
  ([docs/data/2026-11-11_r90_v4_vs_v3](../../docs/data/2026-11-11_r90_v4_vs_v3/README.md)). A launcher whose own
  `DRAFT` names the draft file must pass `DRAFT=on` to this driver: its `DRAFT` is on or off, and any other value
  drops `-md`.
- **The model notices the join.** A deep prompt is recorded sessions cut at a token count with the tail's 64 tokens
  behind them. In 10 of the 12 greedy replies at depths above 0 the model opened by saying that the context looked
  cut or garbled; at depth 0 it continued the tail's turn. Decode and prompt speed do not depend on what is written.
  A draft arm's speed does, through its acceptance, so its numbers at depth are for such replies (registry row
  Q1176 plans prompts that end in a whole recorded request).
