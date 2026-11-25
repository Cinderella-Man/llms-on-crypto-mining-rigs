# p2p_check: does peer-to-peer between two cards deliver the right bytes?

A small CUDA program for the moment someone forces NVIDIA's peer-to-peer on cards or boards the driver refuses
(`NVreg_RegistryDwords="ForceP2P=0x11"`). The driver then says "yes", and that proves nothing: on some boards the data
never arrives, with no error anywhere. Every test here checks every word, prints before it starts (so a hang shows
where), and includes a control copy through host memory that must pass.

```bash
bench/p2p_check/build.sh                                 # in the CUDA 12.9 container: build/p2p_check (sm_61, sm_86)
p2p_check --pairs 0:1 --tests control,store --mb 16      # writes first: posted, they cannot hang the card waiting
p2p_check --pairs 0:1 --tests control,copy,store,load,time
```

| Test | What runs | Counted on |
|---|---|---|
| `control` | the pattern copied A → host → B (runs without peer access too) | B; must be 0 wrong |
| `copy` | `cudaMemcpyPeer` A → B | B |
| `store` | a kernel on A writes into B's memory through the peer pointer | B |
| `load` | a kernel on A reads B's memory through the peer pointer | A (a read that gets no answer can hang the kernel: run it last, under `timeout`) |
| `time` | host-timed `cudaMemcpyPeerAsync` against the copy staged through host memory, 10 KB to 16 MiB | — |
| `--host-cards C,...` | per listed card: host → card and card → host through pinned memory, 64 MiB checked word for word, then timed 10 KB to 64 MiB with `time` (no peer access needed; `--pairs none` skips the pairs) | the card, the host |

Each wrong count also says how many of the wrong words are still 0, the value the target had before the test: all of
them still 0 means nothing arrived; other values mean the data went somewhere, or came from somewhere, else.

Exit code 0 when every requested test delivered every word, 1 on any wrong word, 2 on a CUDA error.

On the P106 rig it showed that this board carries no peer traffic at all
([2026-11-25_p2p_force](../../docs/data/2026-11-25_p2p_force/README.md)).
