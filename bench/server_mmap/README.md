# server_mmap: page faults on the model file in llama-server (Q119)

With every layer on the cards, llama-server keeps one tensor on the host: the token-embedding table, ~312 MiB for
Flint UD-Q4_K_XL. By default the model file is memory-mapped, so each embedding row a request needs is read from the
model stick on first use. Q118 measured ~0.23 major page faults per generated token on the first pass over new text and
~0.02 on a repeat. llama.cpp offers two ways around it:
- `--no-mmap` reads the table into host memory at load time;
- `--mlock` keeps the mapped pages in RAM.

Which helps depends on the host. This rig has 4 GB of RAM, so the scripts log memory as closely as the faults.

`run_q119.sh` runs on the rig as root (~15 min). For each mode (default mmap, `--no-mmap`, `--mlock`) it:
1. drops the page cache;
2. starts llama-server on all cards (UD-Q4_K_XL, backend sampling, `-t 2`, as in Q118);
3. sends the same chat request twice with [client.py](client.py): 1,000 new tokens, fixed seed. The first pass faults
   in the rows the text needs; the repeat shows the warm case;
4. records, per request, tok/s, major and minor faults per token (from `/proc/<pid>/stat`), MemAvailable before and
   after, and the server's resident and locked memory;
5. samples `/proc/meminfo` every 2 s through the whole mode: MemAvailable, free, cached, mlocked, swap free, dirty.

**What `--mlock` locks.** The loader locks the mapping from the start of the file to the end of the last tensor kept
on the host. In the UD-Q4_K_XL file `token_embd` is the third tensor and ends at 0.61 GiB. After loading, the loader
unmaps everything before it (the output head, which lives on a card), which leaves ~0.3 GiB locked. In a file whose
host tensors sat near the end, `--mlock` would try to pin almost the whole file. The script therefore skips the mode
when MemAvailable is below `MLOCK_MIN_MB` (1,200) before the start.

```bash
scp -r bench/server_mmap rig:/tmp/q119 && ssh rig "cd /tmp/q119 && setsid nohup ./run_q119.sh > console.log 2>&1 < /dev/null &"
ssh rig "tail -f /tmp/q119/out-q119/q119.log"
scp -r rig:/tmp/q119/out-q119 . && ssh rig "rm -rf /tmp/q119"
```

Env: `B` (build), `M` (model), `E`, `PORT`, `N` (tokens), `MODES`, `RUNS`, `T`, `MLOCK_MIN_MB`, `OUT`.
`out-q119/q119.csv` has one row per request; `mem_<mode>.csv` holds the memory samples and `server_<mode>.log` the
server logs. `test/dryrun.sh` checks the script against a stub server on any Linux machine.
