# The candidate stack's fit load (R161, 2026-11-14)

One load of `v4s` (the v4 preset on `v4-rc2` with every candidate stacked; [../README.md](../README.md)) before the
deep lane: no warm-up, one prompt of 49,152 tokens, then 256 tokens at temperature 0 twice and 256 with the line's
sampler. The question was whether the forced CUDA graphs and the checkpoints on the cards fit beside the 5,923 MiB that
Flint Pascal Q8 GGUF puts on two cards, and whether the host keeps at least 400 MiB available. Run on 2026-11-14,
10:01:48–10:15:26, with `FIT=1` ([../run/r161.sh](../run/r161.sh)).

## Each card's memory, MiB (of 6,144)

| Card | After loading | Highest over the load | Added by the prompt and the replies |
|---:|---:|---:|---:|
| 0 | 5,383 | 5,427 | 44 |
| 1 | 5,923 | 5,971 | 48 |
| 2 | 5,081 | 5,121 | 40 |
| 3 | 5,591 | 5,633 | 42 |
| 4 | 5,923 | 5,971 | 48 |
| 5 | 5,081 | 5,121 | 40 |
| 6 | 5,545 | 5,587 | 42 |
| 7 | 5,293 | 5,341 | 48 |
| 8 | 4,793 | 4,873 | 80 |

"After loading" is the lane's check when the server came up; "highest" is the largest of the monitor's samples, one
every 2 s.

## The host

- MemAvailable: 3,187 MiB at the server's start, 1,397 MiB after loading, **1,099 MiB at its lowest**; CmaFree 0
  throughout, so usable memory (MemAvailable less CmaFree) was the same. The RAM watchdog (default limits) did not act.
- The fit asked for at least 400 MiB: the stack fits, and no candidate was dropped.

## The load

The server was up after 587 s. The prime read 49,028 tokens on 60 cached at 271.8 tok/s (180.4 s); decode at 49,152
tokens ran 22.32 tok/s on pass 1 and 23.24 on pass 2, the sampled request 29.28; no graph reallocated. The server
turned `--ctx-checkpoints-incall` off at load (`not with sliding-window layers - off`); `--ctx-checkpoints-vram 4`
kept the checkpoints on the cards (a restored checkpoint of the 60-token tail: 4.102 MiB on the cards). One load, so
these speeds are not classed.

## Files

| File | Content |
|---|---|
| [fitmon_gpu.csv](fitmon_gpu.csv) | every card's memory in MiB every 2 s: time, card, MiB |
| [fitmon_mem.tsv](fitmon_mem.tsv) | the host's MemAvailable and CmaFree in MiB every second: epoch, available, CMA free |
| [loads.tsv](loads.tsv), [requests.csv](requests.csv), [table.md](table.md) | the lane's own files for the load: its line and binary, its requests, its table |
