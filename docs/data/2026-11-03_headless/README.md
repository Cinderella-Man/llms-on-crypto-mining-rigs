# Q83: what stopping the desktop frees, and whether `-ub 256` helps at 262k (2026-11-03)

The rigs run Ubuntu's desktop session (GDM, GNOME Shell, Xorg). It holds a few MiB on every card and some of the
4 GB host, and it is what provides NV-CONTROL for the memory-clock route (Q133). The question: how much does stopping
it free, and does that let Flint Q8_0 run its 262,144-token context with `-ub 256` instead of the release's `-ub 128`,
which was chosen because of the 4 GB host? The test ran on the second, identical rig after its debloat (Q506), with
`p106_check.sh`'s long-context serve check: the release command line on nine cards and a 16,399-token prompt.

## Result: `-ub 256` fits even with the desktop, but prompt reading is ~3.8× slower; headless frees ~0.3 GB

| Run (Flint Q8_0, 262k context, 16,399-token prompt) | Prompt | Generation | Lowest host MemAvailable |
|---|---:|---:|---:|
| Desktop, `-ub 128` (the release; the setup check earlier that day) | 580 tok/s | 37.2 tok/s | 913–948 MiB |
| Desktop, `-ub 256` | 154 tok/s | 36.2 tok/s | 1,039 MiB |
| Headless, `-ub 128` | 579 tok/s | 37.3 tok/s | 1,115 MiB |
| Headless, `-ub 256` | 154 tok/s | 27.1 tok/s | 1,248 MiB |

| Idle rig | Desktop | Headless |
|---|---:|---:|
| Host MemAvailable | 2,718 MB | 2,994 MB |
| VRAM used per card | 5 MiB | 0 MiB |
| Desktop processes, resident | 290 MB | 0 |
| Running services | 29 | 15 |

- **`-ub 256` is no gain.** It runs at 262k without memory trouble, with or without the desktop, but reads prompts at
  154 tok/s against 579–580 tok/s with `-ub 128`, and generation is no faster. The release keeps `-ub 128`.
- **Headless frees about 280 MB of host RAM and 5 MiB per card,** and the lowest MemAvailable during a long prompt
  rises by ~170 MiB. Prompt and generation speed are unchanged. For Flash-Next that is a little more page cache for its
  n-gram table, whose reads already cost about 1 % of decode with the cache the rig has (Q228).
- **What it means:** the desktop stays, for now. Headless costs NV-CONTROL (Q133's memory clocks) and gains little
  speed. It comes back into play if host RAM becomes the limit for a model.
- Every number here is from one request, so it is a diagnostic in this project's terms (agentic speed is the measure).
  The differences above are far outside run-to-run noise.

## Files

| File | What |
|---|---|
| `summary.txt` | the three runs' exit codes and the restored default target |
| `snapshots.txt` | host RAM, VRAM per card, the desktop's resident memory and running services, before each phase |
| `check_desktop_ub256.log`, `check_headless_ub128.log`, `check_headless_ub256.log` | `p106_check.sh`'s lines for each run |
| `run/headless_check.sh` | the run: three long-context serve checks, then the desktop started again |

## How it ran

On the rig, as root, detached, with nothing else running. `scripts/p106_check.sh` gained a `UB` setting for this
(default 128, the release). Switching to headless and back is `systemctl isolate multi-user.target` and
`systemctl isolate graphical.target`; the default boot target is not changed.

```bash
BIN=<engine>/bin MODEL=<flint-Q8_0.gguf> CHECK=scripts/p106_check.sh OUT=/tmp/headless bash run/headless_check.sh
```
