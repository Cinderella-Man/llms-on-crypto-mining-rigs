# Recall at depth: are answers still right deep into a long prompt?

A speed number says nothing about whether the answer is right. Upstream llama.cpp issue
[#32375](https://github.com/ggml-org/llama.cpp/issues/32375) (2026-11-16) reports a Tesla P100 (compute 6.0) that
answers a 60,240-token prompt with a short echo of the last small-talk line and none of six facts planted in it, with
no error and normal prompt speed, while the same binary on the CPU lists all six. The same prompt family passes at
56,460 tokens and again at 84,649 and 111,246. Hybrid models are affected there. A P106 is compute 6.1 and runs its own
kernels, so this tool checks the claim on the rig: registry row Q1546 in
[quants/experiments.md](../../quants/experiments.md).

It is this project's own version of the issue's test, with English text of its own and lengths counted in the
server's tokens.

## The test

- **The prompt:** small talk between a user and an assistant (16 exchanges drawn at random with a fixed seed), in
  which the user states six facts about themselves: a cat called Pumpernickel, growing up in Dundee, an old green
  Volvo, work as a lighthouse keeper, the film Metropolis, hibiscus tea instead of coffee. Each fact is planted a fixed
  distance from the end of the prompt: 48,000, 32,000, 16,000, 4,000, 1,000 and 300 tokens. The last message is the
  question (`QUESTION`):
  - `kinds` (the default): "Which pet, city, car, job, film and drink did I mention in our conversation? Answer in six
    short lines, one for each, without comments." It names what to look for, so the answer needs no list of the
    messages.
  - `list`: "Think back over our whole conversation: what have I told you about myself? List every fact as a short
    list, without comments." The first run's question (R183). On Flash-Next at 56,437 tokens its greedy reasoning
    began listing the messages one by one and looped before it answered.
- **Depths are fixed, lengths vary:** the conversation is built backward from the question, so a longer prompt only
  has more small talk at its start, and with the same seed every prompt longer than 48,000 tokens ends with the same
  48,000 tokens. A fact deeper than a short prompt goes to its start: the 8,000-token control holds all six.
- **Counted by the server:** each exchange's cost and the final length come from the server's own chat template
  (`/apply-template`) and tokenizer (`/tokenize`). Each fact's depth is measured on the final prompt, from the fact's
  first token to the end. The table gives the length the server reports for the request.
- **The request:** greedy (temperature 0, top-k 1), the whole prompt read (`cache_prompt: false`, so a repeat computes
  everything again), at most 8,192 reply tokens and never past the context.
- **Pass:** all six marker words (pumpernickel, dundee, volvo, lighthouse, metropolis, hibiscus) in the reply's answer,
  without its reasoning. What follows a miss is `ON_MISS`'s:
  - `repeat` (the default): the prompt is sent once more, the same request, to see whether the miss repeats, and the
    run stops there;
  - `sample`: an answer that misses a fact stops the run at once; a reply that hits the cap with no answer text (all
    of it reasoning) is sent once more with the makers' sampling (`SAMPLING`, default temperature 1.0, top-p 0.95,
    top-k 20, min-p 0, as the Flash-Next presets set it) and the seed, and the run stops there;
  - `go`: no repeat, the next prompt.

## Files

| File | What it does |
|---|---|
| [recall_depth.py](recall_depth.py) | Builds the prompts on the server's counts, sends them, scores the replies (`run`); the table (`table`); prints one prompt (`show`). Python 3, standard library |
| [serve.sh](serve.sh) | On the rig: the idle check, then llama-server on a release preset's line, unchanged, with the RAM watchdog ([scripts/ram_watchdog.sh](../../scripts/ram_watchdog.sh)); waits for it; stops it |
| [remote.sh](remote.sh) | From the client's machine: copies `serve.sh`, the preset and the watchdog to the rig, runs its steps over ssh, collects the rig's files and removes them |

## Run it

The server listens on the rig's loopback; the client reaches it through an ssh tunnel.

```bash
export RIG=rig BIN=<llama-server's folder on the rig> MODEL=<first shard on the rig> DRAFT=<MTP draft on the rig>
bash bench/recall_depth/remote.sh put && bash bench/recall_depth/remote.sh start && bash bench/recall_depth/remote.sh wait
ssh -N -L 18080:127.0.0.1:8080 "$RIG" &
URL=http://127.0.0.1:18080 OUT=out python3 bench/recall_depth/recall_depth.py run 8000 56460 60240
OUT=out python3 bench/recall_depth/recall_depth.py table > out/table.md
bash bench/recall_depth/remote.sh stop && OUT=out/rig bash bench/recall_depth/remote.sh collect
```

`PRESET` picks the preset (default [release/flash-next-v4/settings.env](../../release/flash-next-v4/settings.env));
`SEED` the small talk and every request's seed (default 1546), `CTX` the server's context (default 65536),
`MAX_TOKENS` the reply's cap (default 8192; a reply never goes past the context, so a 60,207-token prompt in a
65,536-token context gets at most 5,313), `QUESTION`, `ON_MISS` and `SAMPLING` as above, `NAME` the prompts' name
prefix.

## Results

| Date | Model, build | Question | Result | Folder |
|---|---|---|---|---|
| 2026-11-17 (R183) | Flash-Next (Swift Qwen3.8-Flash-Next Pascal Q2 GGUF), Miner Llama v4.0, the v4 preset | `list` | control 6/6; at 56,437 tokens no answer (the reasoning looped to the cap) | [2026-11-17_r183_recall_depth](../../docs/data/2026-11-17_r183_recall_depth/README.md) |
| 2026-11-17 (R190) | Flash-Next (UkisAI's Swift-1.5 GSQ-RCO Q2_0, the v4 preset's own file), Miner Llama v4.0, the v4 preset | `kinds` | 6/6 at 8,080, 56,443 and 60,207 tokens | [2026-11-17_r183_recall_depth](../../docs/data/2026-11-17_r183_recall_depth/README.md) |
