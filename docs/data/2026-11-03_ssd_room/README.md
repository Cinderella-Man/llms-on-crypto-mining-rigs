# Q435: Flash-Next's n-gram table on the rig's internal SSD (2026-11-03)

[Q214](../2026-11-03_table_storage/README.md) found that Qwen3.8-Flash-Next's 28.8 GB n-gram table needs flash: the
rig's internal 64 GB SATA SSD serves 4 KiB random reads in 152 µs, its 4.5 TB USB hard disk in 10.4 ms. The SSD was
89 % full. This step moved the data off it onto the USB disk, gave the table a GGUF shard of its own, and put that
shard on the SSD for both Swift-1.5 GSQ-RCO files. The registry row is Q435 in
[quants/experiments.md](../../../quants/experiments.md).

## Result: the table is on the SSD, shared by Swift Q2_0 and IQ3_XXS, with 7.0 GB to spare

| Step | Root free (bytes) |
|---|---:|
| Before | 6,211,215,360 |
| The builds and model side files moved to the USB disk (verified copies, symlinks left) | 10,972,057,600 |
| The Qwen3.8-27B Q4_0 file deleted (16.1 GB, the owner's decision; no copy kept) | 27,028,525,056 |
| The debloat (Q506, Q507): 128 packages and all 13 snaps with snapd removed, 7 services off, apt timers masked | 30,974,889,984 |
| The CUDA profilers (Nsight Compute and Systems) purged, ext4's root reserve cut from 5 % to 1 % | 35,779,481,600 |
| The n-gram table's shard on the SSD | 6,979,362,816 |

- **The table was not a shard of its own.** Swift-1.5 GSQ-RCO's files keep `per_layer_token_embd` (the table,
  28,800,138,432 bytes in IQ4_NL) in shard 1 together with blocks 0–13 and the output layer: shard 1 is 39.8 GB,
  shard 2 holds the rest of the blocks. [scripts/gguf_reshard.py](../../../scripts/gguf_reshard.py) moves it into a
  third shard, copying every byte unchanged, so only the table needs the SSD:

  | File | Shard 1 | Shard 2 | Shard 3 (the table) |
  |---|---:|---:|---:|
  | Q2_0 | 10,998,979,680 | 26,750,834,816 | 28,800,138,432 |
  | IQ3_XXS | 10,985,652,256 | 36,180,282,560 | 28,800,138,432 |

  `gguf_reshard.py verify` rebuilt the four original shards from the new files to the SHA-256 recorded when they were
  downloaded ([Q213's table](../2026-11-03_flash_next_fit/README.md#how-it-was-made): `79e2a387…`, `ef3bb04f…`;
  `3bddaa66…`, `b0b15f78…`), so no byte changed. Its self-test
  ([scripts/gguf_reshard_test.py](../../../scripts/gguf_reshard_test.py)) passes.
- **Both files share one table.** The two table shards are byte-identical (md5 `98ce140e641bc5d68af9f31df0f1c048`):
  the GSQ-RCO builds quantize the experts differently but keep the same IQ4_NL table. One copy on the SSD serves both,
  through a symlink in each model's folder.
- **llama.cpp reads the 3-shard files** (`split.count` 3, 1,224 tensors) and projects the same memory per card as the
  original files did in [Q213](../2026-11-03_flash_next_fit/README.md): 735 MiB left on the fullest card at 64k and
  `-ub 512`, 664 at 128k and `-ub 256`, 519 with the MTP draft, and 187 for IQ3_XXS at 48k and `-ub 128`.
- **The files went by drive, not network.** On 2026-11-03 data into the rigs ran at 21–28 kB/s (out of them at ~100
  MB/s; Q511), so the 4.5 TB drive was carried to the quality machine, filled there and checked by reading back, and
  checked again on the rig.

Left on the SSD for the owner to decide:

| What | Bytes | Kind |
|---|---:|---|
| `~/llama.cpp` (an upstream source tree with a build) | 850,375,112 | rebuildable |
| `~/cheap-guess` (two builds of the cheaper MTP-guess branch, 2026-10-28) | 357,758,136 | rebuildable from the quality machine's tarballs |
| `/var/log/journal` | 268,435,456 | logs |
| `/var/cache/apt` | 138,109,929 | package cache |
| `~/.cache` | 54,258,701 | per-user cache |
| `/swap.img` | 4,026,531,840 | kept |

## The layout

| Path | What |
|---|---|
| `<ssd>/<home>/models/flash-next/Swift-Qwen3.8-Flash-Next-GSQ-RCO-table-00003-of-00003.gguf` | the table shard, on the internal SSD |
| `<disk>/models/flash_next/swift_q2_0/…-Q2_0-0000{1,2}-of-00003.gguf` | Q2_0's other shards, on the USB disk |
| `<disk>/models/flash_next/swift_q2_0/…-Q2_0-00003-of-00003.gguf` | symlink to the table |
| `<disk>/models/flash_next/swift_iq3_xxs/…-IQ3_XXS-0000{1,2}-of-00003.gguf` | IQ3_XXS's other shards |
| `<disk>/models/flash_next/swift_iq3_xxs/…-IQ3_XXS-00003-of-00003.gguf` | symlink to the same table |
| `<disk>/models/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf` | the MTP draft (agentionai's Gyro repository) |
| `<disk>/models/flash_next/table-on-drive/` | the table's copy from the drive trip, kept aside |
| `/opt/llama-builds`, `~/mtp-head`, `~/answer-store` | symlinks to `<disk>/from-ssd/…` |

`llama-server -m <disk>/models/flash_next/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`
loads all three shards; llama.cpp maps the table read-only and reads its rows through the page cache.

## What moved where

| Old path | Now | Bytes |
|---|---|---:|
| `/opt/llama-builds` (every rig build, the q20 drafter heads, wikitext-2) | symlink to `<disk>/from-ssd/opt/llama-builds` | 3,726,708,490 |
| `~/mtp-head` (Flint's MTP head GGUFs) | symlink to `<disk>/from-ssd<home>/mtp-head` | 910,275,072 |
| `~/answer-store` (the agentic-run-1 answer store) | symlink to `<disk>/from-ssd<home>/answer-store` | 123,565,088 |
| `~/models/Qwen3.8-27B-Q4_0.gguf` | deleted (no copy, no symlink) | 16,056,478,688 |

Each folder was copied with `rsync -aHAX`, checked with a checksum dry run (`rsync -aHAX -c -n --itemize-changes`,
which printed nothing), and only then removed and replaced by a symlink. The builds run from the new place.

## The debloat

[scripts/rig_debloat.sh](../../../scripts/rig_debloat.sh) ([Q506, Q507](../2026-11-03_rig_debloat/README.md)) ran after the moves, with nothing on the cards:
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
| `run/table_to_ssd.sh` | the table: copy to the SSD with its md5, read back and check, the drive's copy aside, a symlink in each model folder |

## How it was made

On 2026-11-03, on the rig (Ubuntu 24.04, 56 GB root file system on the internal SSD; the USB disk ext4), as root,
and on the quality machine (which holds the downloaded files):

```bash
# the inventory and the moves (rig)
ssh $RIG 'du -x -h --max-depth=2 /home /root /opt /var /usr | sort -rh | head -40'
ssh $RIG "DEST=<disk>/from-ssd PATHS='/opt/llama-builds <home>/answer-store <home>/mtp-head' bash -s" \
  < docs/data/2026-11-03_ssd_room/run/move_to_disk.sh

# the table in its own shard (quality machine; header reads, then a copy of every byte; ~8 min per file)
python3 -I scripts/gguf_reshard_test.py <scratch dir>
python3 -I scripts/gguf_reshard.py split --isolate '^per_layer_token_embd\.' --out-dir <out>/swift_q2_0 \
  <Q2_0 shard 1> <Q2_0 shard 2>
python3 -I scripts/gguf_reshard.py verify <Q2_0 shard 1> <Q2_0 shard 2> --new <out>/swift_q2_0/*-of-00003.gguf
md5sum <out>/swift_q2_0/*-of-00003.gguf     # the same for IQ3_XXS

# the table on the SSD, linked into both model folders (rig, after the files arrived)
ssh $RIG "SRC=<disk>/models/flash_next/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf \
  MD5=98ce140e641bc5d68af9f31df0f1c048 SSD=<home>/models/flash-next/Swift-Qwen3.8-Flash-Next-GSQ-RCO-table-00003-of-00003.gguf \
  BACKUP=<disk>/models/flash_next/table-on-drive \
  LINKS='<disk>/models/flash_next/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf <disk>/models/flash_next/swift_iq3_xxs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00003-of-00003.gguf' \
  bash -s" < docs/data/2026-11-03_ssd_room/run/table_to_ssd.sh

# the memory projection of the real files (Q213's lines)
MODEL=<disk>/models/flash_next/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf BUILDS=<builds> \
  OUT=probe_real.md CONFIGS='q2real-f16-ub512-64k|<v2.0 build>|65536|-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6' \
  bash scripts/fit_probe.sh
```

The table copy ran at ~38 MB/s on the rig's two Celeron cores (USB read, SSD write and md5 at once); the md5 of the
SSD copy was taken again after dropping the page cache.
