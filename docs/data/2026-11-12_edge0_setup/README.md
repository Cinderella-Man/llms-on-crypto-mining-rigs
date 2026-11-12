# Q1438: Edge0 145B Q3_K set up on an RTX 3090 machine (2026-11-12)

How one Edge0 model folder was fetched, checked and made runnable on a machine with RTX 3090 cards, one day after the
weights appeared. What Edge0 is, the safety review and what it means for a P106 rig are in
[edge0.md](../../research/notes/edge0.md); the registry row is Q1438 in
[quants/experiments.md](../../../quants/experiments.md#rigs-and-other-models).

## Result

**Status on 2026-11-12, 21:46: the engine is built and sees the card; the folder is downloaded and its hash check is
running; no token has been generated yet.**

- **The folder:** `Edge0/Edge0-145B-q3k` at Hugging Face revision `main` on 2026-11-12, 76 files, 105.37 GB. Every
  file has the size the repository lists, and the 11 files it stores by hash have that SHA-256
  ([manifest_145b_q3k.tsv](manifest_145b_q3k.tsv) lists all 76 with the hash of each).
- **The engine:** Edge0 `b7900cbb322926ccebdcd0c0c5a1280497ecff22`, `edge0-strata` built for `sm_86` with CUDA 13.4
  and `edge0-infer` with its `cpu` preset, in an Ubuntu 24.04 container that mounts the host's CUDA toolkit read-only.
  ggml comes from llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The compile took 4 min 47 s at `-j 8`
  (179 steps for the engine, 289 for the launcher). In the container the engine links `libcudart.so.13` and
  `libcublas.so.13` and sees the card as compute capability 8.6.
- **Nothing from the model folder is executed.** Its `start.sh` and the launch fields of its `model.json` are not
  used: the launcher reads `model_name` only.

## What went wrong on the way, and the rule it gives

The first download wrote two files at once, unthrottled, to the machine's system disk, a 2 TB QLC NVMe SSD that was
82 % full. After about 26 GB in ten minutes the disk's fast cache was exhausted: it took writes at 6–41 MB/s, 5.6 GB
of unwritten data queued in memory, the kernel's I/O pressure stayed near 50 %, the load average rose from 8.5 to 30,
and the machine stopped answering new connections for seconds at a time. A server running on its GPUs kept running
(no disk waits in its threads), but everything that writes to that disk stalled. Throttling the download to
2 × 20 MB/s did not help, because the disk was already behind. The machine was normal again 4 min 30 s after the
download stopped.

**Rule: a model of this size does not go onto a QLC system disk in one fast write.** What worked:

- the two large files (`experts.bin`, `ple-fp8.gguf`) on two hard disks, with symlinks in the model folder
  ([relocate_big_files.sh](run/relocate_big_files.sh));
- the download inside a container with a 3 GB memory limit, so unwritten data cannot pile up beyond what the target
  disk takes, and a rate limit per file ([fetch_edge0.sh](run/fetch_edge0.sh), `RATE`);
- the n-gram table loaded into memory at start (`ple-io=ram`), since a hard disk cannot serve the engine's per-token
  reads.

## Files

| File | What it is |
|---|---|
| [manifest_145b_q3k.tsv](manifest_145b_q3k.tsv) | path, bytes and SHA-256 of all 76 files of the folder, and whether the hash was checked against the repository's (`lfs`) or computed here (`local`) |
| [run/fetch_edge0.sh](run/fetch_edge0.sh) | resumable download with curl, the file list and hashes from the Hugging Face API, hard links to a sibling folder for identical files, verification |
| [run/relocate_big_files.sh](run/relocate_big_files.sh) | moves named files to other disks and leaves symlinks |
| [run/build_edge0.sh](run/build_edge0.sh) | builds `edge0-strata` and `edge0-infer` from a pinned commit in a container; no GPU needed |
| [run/run_edge0.sh](run/run_edge0.sh) | serves one folder on one card, on the host's loopback address |

## How it ran

`WORK` is a work directory on the machine; the two `/mnt/...` paths stand for two other disks.

```bash
# 1. the folder: small files to $WORK, the two large files to other disks, 2 x 60 MB/s, 3 GB of page cache at most
docker run -d --rm --name edge0-fetch --memory=3g --memory-swap=3g --cpu-shares=256 \
  -v $WORK:$WORK -v /mnt/disk1:/mnt/disk1 -v /mnt/disk2:/mnt/disk2 \
  -e WORK=$WORK -e REPO=Edge0/Edge0-145B-q3k -e JOBS=2 -e RATE=60M -e MODEL_DIR=$WORK/models/Edge0-145B-q3k \
  -w $WORK/dl --entrypoint bash <an Ubuntu image with curl and python3> -c \
  'bash relocate_big_files.sh experts.bin=/mnt/disk1/edge0/Edge0-145B-q3k ple-fp8.gguf=/mnt/disk2/edge0/Edge0-145B-q3k \
   && bash fetch_edge0.sh'

# 2. the engine and the launcher, for RTX 30 cards, with the host's CUDA 13 toolkit
WORK=$WORK CUDA_DIR=/usr/local/cuda-13.4 ARCH=86 JOBS=8 bash build_edge0.sh

# 3. check the folder, then serve it on one card (the one with the widest PCIe link)
docker run --rm -v $WORK:$WORK -v /mnt/disk1:/mnt/disk1 -v /mnt/disk2:/mnt/disk2 --entrypoint bash edge0-build:24.04 -c \
  "$WORK/src/Edge0/edge0-infer/build/cpu/bin/edge0-infer inspect --model $WORK/models/Edge0-145B-q3k --json"
WORK=$WORK GPU=<GPU uuid> CUDA_DIR=/usr/local/cuda-13.4 PLE_IO=ram \
  EXTRA_MOUNTS="-v /mnt/disk1:/mnt/disk1 -v /mnt/disk2:/mnt/disk2" bash run_edge0.sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "edge0_145b", "messages": [{"role": "user", "content": "Hello"}]}'
```

A second folder reuses the files that are identical: `LINK_FROM=$WORK/models/Edge0-145B-q3k REPO=Edge0/Edge0-120B-q3k`
downloads 32.7 GB instead of 92.3 GB.

## Provenance and licence

The model files are Edge0's, under the Edge0 Community Model License 1.0 and the Qwen Community License 1.0; none is
republished here, only their names, sizes and hashes. The engine is built from
[Edge0-AI/Edge0](https://github.com/Edge0-AI/Edge0) (Apache-2.0; `edge0-strata` MIT, a fork of
[Strata](https://github.com/Niko1221/Strata)). The scripts in `run/` are this project's.
