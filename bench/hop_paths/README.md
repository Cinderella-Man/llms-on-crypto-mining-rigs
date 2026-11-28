# hop_paths: two ways a card-to-card hop could move, measured on their own

On a board without peer-to-peer (the P106 rig: every card on PCIe 1.0 x1 behind ASM1184e switches or a chipset port)
a hop between two cards goes card → host memory → card. These two programs measure the pieces of that path that the
engine's hop code ([card hops](../../docs/research/notes/card_hops_x1.md)) could use instead of what it uses today.
Their results are in [docs/data/2026-11-28_hop_speedup](../../docs/data/2026-11-28_hop_speedup/).

| Program | Question | Registry |
|---|---|---|
| [vram_write.cpp](vram_write.cpp) | Can the host write a hop's bytes into the receiving card's video memory (through its BAR1 window, Vulkan's device-local host-visible memory) faster than the card reads them from pinned host memory? Times `cpu_memcpy` and `cpu_nt` (non-temporal stores) against `dma_h2d` and `dma_d2h` at one token's Q8_0 hop (10,880 B), a four-token verify pass (43,520 B), a 256-token prompt micro-batch (2,785,280 B) and 16 MiB, plus one uncached 4-byte read (`cpu_read4`) | Q1941 |
| [memops_check.cpp](memops_check.cpp) | Do stream memory operations (`cuStreamWriteValue32` / `cuStreamWaitValue32` on a mapped host word) run on the card, and how fast does a signal cross from card A's stream to card B's, against an event? Prints three attributes per card first | Q752 |

Both are standalone (no llama.cpp), run in under a minute per card, and need nothing but the NVIDIA driver, CUDA's
runtime (linked statically into `vram_write`) and, for `vram_write`, the Vulkan loader.

## Build

`bash build.sh` builds both in the CUDA 12.9 devel image (Ubuntu 24.04) into `./out`, with `MD5SUMS`; no GPU is
needed. `OUT`, `IMAGE` and `DOCKER_ARGS` change the defaults. The compile lines are at the top of each source file.

## Run

```bash
vram_write <gpu in PCI bus order, as nvidia-smi> [reps up to 64 KiB, 300] [reps above, 20] > vram_<gpu>.csv
CUDA_DEVICE_ORDER=PCI_BUS_ID memops_check <A> <B> [reps, 1000]
```

`vram_write` prints CSV (`gpu,bus,path,bytes,reps,median_us,p10_us,p90_us,mb_per_s`) and exits 3 if a word written
through the window reads back wrong. `memops_check` exits 4 if the driver refuses the stream memory operations. Run them
on an idle machine: nothing else on the cards or the links.

## What the numbers mean

- A `cpu_*` copy ends with an `sfence` and one read of the last word written. PCIe never lets a read pass earlier posted
  writes on the same path, so the read's return means every byte has landed; the time includes that one round trip,
  which `cpu_read4` gives alone.
- The host core is busy for the whole of a `cpu_*` copy: write-combined stores stall once the processor's buffers
  are full. On a two-core host that is a core taken from the server for as long as the copy runs.
- `dma_h2d` is the second leg of today's hop and `dma_d2h` the first; a hop through the host by the processor would
  replace only the second.
