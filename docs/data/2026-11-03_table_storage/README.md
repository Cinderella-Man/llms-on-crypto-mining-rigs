# Q214: where can Flash-Next's n-gram table live on the rig? (2026-11-03)

Qwen3.8-Flash-Next keeps a 28.8 GB n-gram embedding table on disk, and every generated token makes an estimated
11–14 random 4 KiB reads of it ([flash_next_variants.md](../../research/notes/flash_next_variants.md), section 5). The rig
has 4 GB of RAM, so the page cache cannot hold the table, and the read time lands on every token. This run measures 4 KiB
random reads on the rig's two disks: its internal 64 GB SATA SSD and the 4.5 TB USB 3 portable hard disk (2.5-inch)
that holds the model files.

## Result: the table needs an SSD; the USB hard disk would cost ~0.1 s a token

| Disk | Queue depth | Reads/s | MiB/s | Mean | Median | 99th percentile |
|---|---:|---:|---:|---:|---:|---:|
| internal SATA SSD | 1 | 6,039 | 23.6 | 152 µs | 138 µs | 249 µs |
| internal SATA SSD | 32 | 23,792 | 92.9 | 1.33 ms | 1.32 ms | 2.38 ms |
| USB 3 portable hard disk | 1 | 95 | 0.4 | 10.4 ms | 9.9 ms | 16.2 ms |
| USB 3 portable hard disk | 32 | 101 | 0.4 | 315 ms | 312 ms | 392 ms |
| USB 3 flash stick, SanDisk Ultra Fit 64 GB (second rig) | 1 | 2,243 | 8.8 | 398 µs | 408 µs | 569 µs |
| USB 3 flash stick, SanDisk Ultra Fit 64 GB (second rig) | 32 | 2,831 | 11.1 | 11.0 ms | 10.9 ms | 12.3 ms |
| USB 3 flash stick, Corsair Survivor 128 GB (second rig) | 1 | 1,363 | 5.3 | 685 µs | 668 µs | 1.07 ms |
| USB 3 flash stick, Corsair Survivor 128 GB (second rig) | 32 | 1,453 | 5.7 | 21.3 ms | 21.4 ms | 24.8 ms |

- **Per token, from the table alone (my arithmetic on the estimated 11–14 reads):**
  - **SSD:** 1.7–2.1 ms with one read at a time, or ~0.6 ms with all 14 issued together.
  - **Hard disk:** 115–146 ms either way. A single spindle gains nothing from a deeper queue: 95 reads/s at depth 1, 101
    at depth 32.
- **Against Flash-Next's estimated 66–86 ms of compute per token on the rig:** the SSD keeps est. 11.4–14.7 tok/s, while
  the hard disk would drop it to est. 4.3–5.5 tok/s. These speeds are estimates: no Flash-Next token has run on a P106 yet.
- **USB 3 flash sticks sit in between (Q504):** est. 4.4–5.6 ms a token on the SanDisk and 7.5–9.6 ms on the Corsair,
  at depth 1, and a deeper queue barely helps either. Against the same estimated compute that is est. 5–8 % and 9–14 %
  slower than with no table cost at all, so a stick works when the SSD has no room. The SSD costs est. 2–3 %.
- **What it means:** the table goes on flash. The internal SSD is fast enough, but it had 5.9 GB free on 2026-11-03,
  against the ~29 GB the table needs; freeing room there was the next task (Q435). On a rig without SSD room, the
  faster of two sticks is the fallback.

## Files

| File | What |
|---|---|
| `summary.tsv` | one row per disk and queue depth: reads/s, MiB/s, mean, median and 99th-percentile completion latency |
| `ssd-*.json`, `usbhdd-*.json`, `sandisk-*.json`, `corsair-*.json` (depth 1 and 32 each) | fio's full JSON output for each run (file paths replaced by `<SSD>`, `<USBDISK>`, `<SANDISK>` and `<CORSAIR>`) |
| `run/fio_table_reads.sh` | the run: 4 KiB random reads at depth 1 and 32, 30 s each after a 5 s ramp, `O_DIRECT`, `--readonly` |

## How it ran

The two sticks were measured the same way on a second, identical rig, with a 41.5 GiB and a 39.4 GiB file.

On the rig, as root, with fio 3.36 from Ubuntu 24.04 (`apt-get install --no-install-recommends fio`). Each test reads an
existing model file larger than the host's RAM: a 15.0 GiB file on the SSD and a 44.9 GiB file on the USB disk. Nothing
is written, and no other job ran on the rig during the 2.5-minute run.

```bash
FILES="ssd=<SSD>/Qwen3.8-27B-Q4_0.gguf usbhdd=<USBDISK>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf" \
  OUT=/tmp/table_storage bash run/fio_table_reads.sh
```
