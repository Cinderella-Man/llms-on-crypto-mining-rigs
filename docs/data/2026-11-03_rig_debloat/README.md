# Q506, Q507: the desktop install stripped down to what a rig runs (2026-11-03)

The rigs came with Ubuntu 24.04's full desktop install: mail, a browser, an office suite, media players, printing and
the software store, a dozen snaps with their runtimes, and idle services such as ModemManager, CUPS, Avahi, Bluetooth,
the crash reporters and PackageKit, plus automatic upgrades. A rig is an appliance that only runs the engine, so
[scripts/rig_debloat.sh](../../../scripts/rig_debloat.sh) removes all of that. It keeps the NVIDIA driver, CUDA, the
kernel, SSH, Tailscale, NetworkManager, Python and the desktop session (GDM, GNOME Shell, Xorg): NV-CONTROL needs X until
headless mode is measured (Q83). Two identical rigs (9 × P106-090, 2-core Celeron, 4 GB RAM) ran it the same day.

## Result: more disk and fewer services, no RAM

| | First rig | Second rig |
|---|---:|---:|
| Packages removed | 128 | 103 |
| Snaps | 13 → 0 | 10 → 0 |
| Running services | 34 → 27 | 34 → 33, then 30 once finished by hand (below) |
| Root disk free | 26 → 29 GB | 22 → 23 GB |
| RAM available | 2,807 → 2,755 MB | 2,875 → 2,734 MB |
| `p106_check.sh` (nine cards: chat, tools, a 16k-token prompt) | — | 0 FAIL before and after |

- **No RAM back:** the removed services were small, and the desktop session that holds most of the host's memory
  stays. Each "after" figure was read just after a large apt run, so a few tens of MB either way is cache.
- **Automatic upgrades off (Q507):** `unattended-upgrades` is purged and `apt-daily.timer` and `apt-daily-upgrade.timer`
  are masked. An unattended NVIDIA driver upgrade would break `nvidia-smi` until a reboot, and an apt run inside a timed
  run would skew it. Upgrades are now a manual step between runs.
- **`snapd` cannot come back** with a later install: `/etc/apt/preferences.d/no-snapd.pref` pins it to -10.

## What the first runs found, and the fixes in the script

- **The protected-package list was too broad.** It matched every `python3*` package, so the first dry run on the first
  rig stopped on `python3-uno`, LibreOffice's own Python bridge. The list now protects only the interpreter (`python3`,
  `python3.N`, `python3-minimal`, `libpython3*`).
- **Running daemons broke the purge.** On the second rig `avahi-daemon`, `cups-browsed` and `kerneloops` kept running
  while their packages were removed: their system users stayed in use, `cups-browsed`'s removal script failed and apt
  exited 100 with the package half removed. Stopping those daemons, purging `cups-browsed` again and running the
  pending triggers left `dpkg --audit` clean. The script now disables and stops the services, sockets and paths
  included, before apt runs, and checks `dpkg --audit` at the end.

## How it ran

On each rig, as root, with no model running: a dry run first, which lists everything apt would remove and stops if
any of it is protected, then the removal. Each package step is simulated before it runs, and nothing is autoremoved.

```bash
bash scripts/rig_debloat.sh
APPLY=1 bash scripts/rig_debloat.sh
```

The log goes to `/root/rig_debloat.log` on the rig; this page summarizes it.
