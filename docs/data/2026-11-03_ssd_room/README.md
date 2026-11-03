# Q435: room for Flash-Next's n-gram table on the rig's internal SSD (2026-11-03)

[Q214](../2026-11-03_table_storage/README.md) found that Qwen3.8-Flash-Next's 28.8 GB n-gram table needs flash: the
rig's internal 64 GB SATA SSD serves 4 KiB random reads in 152 µs, its 4.5 TB USB hard disk in 10.4 ms. The SSD was
89 % full. This step moves the data off it, onto the USB disk, and works out whether the table fits. The registry row is
Q435 in [quants/experiments.md](../../../quants/experiments.md).

## Result so far: 31.0 GB free, room for the table on its own (28.8 GB)

| Step | Root free (bytes) |
|---|---:|
| Before | 6,211,215,360 |
| After moving the builds and model side files to the USB disk (verified copies, symlinks left) | 10,972,057,600 |
| After deleting the Qwen3.8-27B Q4_0 file (16.1 GB, the owner's decision; no copy kept) | 27,028,525,056 |
| After the debloat (Q506, Q507): 128 packages and all 13 snaps with snapd removed, 7 services off, apt timers masked | 30,974,889,984 |
| With the table's own shard on the SSD (planned) | est. 2,174,751,552 |

- **The table is in the first shard, not on its own.** Swift-1.5 GSQ-RCO's files keep `per_layer_token_embd` (the
  table, 28,800,138,432 bytes in IQ4_NL) in shard 1 together with blocks 0–13 and the output layer: shard 1 is 39.8 GB,
  shard 2 (26.8 GB) holds blocks 13–47 only. So the SSD would need room for 39.8 GB, not 28.8.
- **The table can get a shard of its own.** [scripts/gguf_reshard.py](../../../scripts/gguf_reshard.py) moves the
  tensors that match a pattern into one more shard, copying every byte unchanged, and can write that shard to another
  disk with a symlink beside the others. On Swift Q2_0 it gives 11.0 GB + 26.75 GB + 28.80 GB (the table alone).
  llama.cpp reads the 3-shard layout (`split.count` 3, all 1,224 tensors) and projects the same 43,772 MiB on the
  cards as with the original two shards (Q213's baseline line, header-only copies). Its `verify` step rebuilds the
  original shards' bytes from the new ones and prints their SHA-256; the self-test
  ([scripts/gguf_reshard_test.py](../../../scripts/gguf_reshard_test.py)) passes.
- **The table alone fits, with 2.2 GB to spare** (plus the 3.0 GB that ext4 reserves for root). What else could come
  off the SSD, left in place for the owner to decide:

| What | Bytes | Kind |
|---|---:|---|
| `/opt/nvidia/nsight-compute`, `/opt/nvidia/nsight-systems` | 2,252,488,719 | CUDA profiler GUIs, not used on the rig |
| `~/llama.cpp` (an upstream source tree with a build) | 850,375,112 | rebuildable |
| `~/cheap-guess` (two builds of the cheaper MTP-guess branch, 2026-10-28) | 357,758,136 | rebuildable from the quant machine's tarballs |
| `/var/log/journal` | 268,435,456 | logs |
| `/var/cache/apt` | 138,109,929 | package cache |
| `~/.cache` | 54,258,701 | per-user cache |
| `/swap.img` | 4,026,531,840 | kept |

## What moved where

| Old path | Now | Bytes |
|---|---|---:|
| `/opt/llama-builds` (every rig build, the q20 drafter heads, wikitext-2) | symlink to `<disk>/from-ssd/opt/llama-builds` | 3,726,708,490 |
| `~/mtp-head` (Flint's MTP head GGUFs) | symlink to `<disk>/from-ssd<home>/mtp-head` | 910,275,072 |
| `~/answer-store` (the agentic-run-1 answer store) | symlink to `<disk>/from-ssd<home>/answer-store` | 123,565,088 |
| `~/models/Qwen3.8-27B-Q4_0.gguf` | deleted (no copy, no symlink); `~/models` is empty | 16,056,478,688 |

Each folder was copied with `rsync -aHAX`, checked with a checksum dry run (`rsync -aHAX -c -n --itemize-changes`,
which printed nothing), and only then removed and replaced by a symlink. The builds run from the new place
(`/opt/llama-builds/<build>/bin/llama-server --version`, and Q213's probes after the move).

## The debloat

`scripts/rig_debloat.sh` (Q506, Q507, published with their own data) ran after the moves, with nothing on the cards:
128 packages (desktop apps, LibreOffice, printing, ModemManager, Avahi, Bluetooth, crash reporters, PackageKit,
fwupd, the update tools and the `ubuntu-desktop` metapackages) and all 13 snaps with snapd went, running services fell
from 34 to 27, and the apt timers are masked. The dry run stopped once on `python3-uno`, LibreOffice's Python bridge,
which the script's protected-name pattern (`^python3`) catches; it ran with that one package exempted. Root free
space rose by 3.95 GB; available RAM did not change (2,807 MiB before, 2,755 after). Afterwards all nine cards answer
`nvidia-smi`, `nvcc` 12.9 runs, and the display manager, NetworkManager and remote access are active.

## Files

| File | What |
|---|---|
| `run/move_to_disk.sh` | the move: copy, checksum dry run, remove, symlink, per folder |

## How it was made

On 2026-11-03, on the rig (Ubuntu 24.04, 56 GB root file system on the internal SSD, ext4 with 3.0 GB reserved for
root; the USB disk ext4), as root:

```bash
ssh $RIG "DEST=<disk>/from-ssd PATHS='/opt/llama-builds <home>/answer-store <home>/mtp-head' bash -s" \
  < docs/data/2026-11-03_ssd_room/run/move_to_disk.sh
ssh $RIG 'du -x -h --max-depth=2 /home /root /opt /var /usr | sort -rh | head -40'    # the inventory
python3 -I scripts/gguf_reshard.py split --headers-only --isolate '^per_layer_token_embd\.' --out-dir <dir> \
  <Q2_0 shard 1> <Q2_0 shard 2>                                                       # the layout check
python3 -I scripts/gguf_reshard_test.py <scratch dir>
```
