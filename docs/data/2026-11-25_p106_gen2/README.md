# Gen2 on a P106-090's link: first try on the rig (2026-11-25, Q994)

Can one of the rig's P106-090 cards run its PCIe link at 5 GT/s (Gen2) instead of 2.5 GT/s? The cards advertise Gen1
only, and the closed driver forces Gen1 on every CUDA start. This run followed the method that took a P104-100 (also
Pascal) to Gen2 on the same driver, 580.178.04. That project is
[WJNKAC/nvidia-p104-unlock-gen2-for-linux](https://github.com/WJNKAC/nvidia-p104-unlock-gen2-for-linux); I read its code
and notes and ran none of it. The test card was card 4, alone on a chipset port (`00:1c.3`).
Result page: [diy_card_interconnect.md](../../research/notes/diy_card_interconnect.md), section 3; registry Q994.

**Outcome: no Gen2, and card 4 fell off the bus.** The writes landed, but the card's Gen2 declaration never latched;
a second try broke the link, and nothing short of a reboot brings the card back. Don't repeat these steps on a card
the driver has already initialized since power-on (below).

## What was measured

**Card 4 at Gen1, before anything was changed** (stock driver; `bench/p2p_check --host-cards 4,0 --tests time`;
pinned memory; 64 MiB copies checked word for word; times are the median of 100):

| Card | Host → card | Card → host | 10 KB host → card | 10 KB card → host |
|---|---|---|---|---|
| 4 (own chipset port) | 200 MB/s | 213 MB/s | 62.3 µs | 57.8 µs |
| 0 (behind a switch) | 189 MB/s | 213 MB/s | 66.5 µs | 61.1 µs |

**The P104's register map holds on the P106-090.** BAR0's mirror of the PCIe link registers matched config space
exactly: `0x88084` = `LnkCap` `0x00453c41` and `0x88088` = `LnkCtl`/`LnkSta` `0x10110140`. The values read on card 4,
after the day's several driver loads:

| Register (BAR0) | Before | After the push |
|---|---|---|
| `0x880A4` `LnkCap2` mirror | `0x00000002` (Gen1 only) | `0x00000002` (no latch) |
| `0x8841C` `PRIV_MISC_1` | `0xc0b46500` | `0xc0b4ad00` (the P104's unlocked value) |
| `0x88610` `VSEC_HIERARCHY` | `0x00000000` | `0x00000001` |
| `0x8C040` `LINK_CONFIG_0` | `0x20085800` (speed field 2 = 2.5 GT/s) | `0x20045800` (speed field 1 = 5 GT/s) |
| `0x8C2C0` `CYA_0` | `0x068711b3` (Gen2-disable bit 2 already 0) | unchanged |
| `0x8C1C0` | `0x00110036` | unchanged (try 2 never ran) |

## The steps, in order

1. **The driver, patched in memory only.** The closed driver's link-speed function (`_nv017105rm`, mode =
   generation) is called with mode 1 on every CUDA start and forces Gen1. [run/patch_gen1_bail.py](run/patch_gen1_bail.py)
   shifts its range check so mode 1 takes the function's existing early return.
   - Its 32-byte sequence occurs once in the rig's `nvidia.ko`, at the function's start + 6 (file offset `0x12fe56`;
     `.text` at `0x2a90`, the function at `0x12d3c0`), and the disassembly confirms both changed bytes.
   - The copy has 2 bytes changed and the distribution's signature removed: Linux refuses a module whose signature
     fails even with signatures not enforced, and loads an unsigned one with a taint mark. Secure Boot was off.
   - It was loaded with `insmod` from `/var/tmp`; the installed file was never touched.
2. **The push** ([run/gen2_push.py](run/gen2_push.py)), with the display modules unloaded and card 4 not opened.
   - It refuses to write unless the register mirror matches config space.
   - Every write happened while the link was stable: the four XP3G overrides, the declaration bits in `0x8841C`,
     `0x88610`, and the speed field.
   - Then it wrote the training trigger (`0x8872C` = 6), stopped touching the card, and retrained from the port above.
   - Result: the link came back at 2.5 GT/s, and the declaration was still Gen1 only.
3. **The retry** ([run/retry_latch.py](run/retry_latch.py)). Try 1 wrote `0x8872C` = 0, then 6, and polled the
   card's registers for up to 10 s waiting for the declaration.
   - It never latched.
   - After the following retrain from the port, the card answered `ffffffff` to every config read, and the port showed
     link training that never ends (`LnkSta` `0x1811`, data link down).
   - **This step is the likely cause.** The P104 project never writes 0 there, and reports that user-space accesses
     while the link changes killed its card twice in two tries.
4. **Recovery attempts, all failed:**
   - a retrain from the port;
   - removing the device, a secondary bus reset and a rescan (a bus reset can't reach a card whose link is down);
   - a link disable and enable;
   - the port's target speed set to 2.5 GT/s with a retrain.
   
   The slot has no power control (`SltCap: PwrCtrl-`). Card 4 needs a reboot of the machine.
5. **Back to stock:** the port's target speed restored, the stock `nvidia.ko` loaded, and the eight other cards
   checked with a copy through host memory (every word right, no kernel errors).

## What it means

- **On a card already initialized by the driver since power-on, the declaration doesn't latch.** The P104 project
  calls this the stuck state: unlock bits written (`0x8841C & 0xA800` nonzero) but the declaration still Gen1. Its
  notes say every rescue fails in that state, and that the first push after a reboot works. The CMP 100-210 work
  ([mintoriakamoto](https://github.com/mintoriakamoto/CMP100-210xFULLTENSORSxGEN3x1)) says the same: the clamp is
  latched at the card's initialisation, so the clear has to come before it. Card 4 had been initialized by four driver
  loads that day.
- **The P106 starts from a different value:** `0x8841C` read `0xc0b46500`, with bits 13 and 14 set, where the P104
  notes give `0xc0b40500` or `0xc0b48d00`. Another P104 tool clears exactly those two bits
  ([skbskhud/p104-unlock-v3](https://github.com/skbskhud/p104-unlock-v3)), so the P106's clear may differ.
- **What a next try needs:**
  - a fresh boot with the NVIDIA modules kept from loading (a one-time `modprobe.blacklist=` on the kernel line,
    typed at the boot menu);
  - the push done before the driver first initializes the card;
  - the P104 sequence exactly, without the extra steps of step 3;
  - someone at the console to type the disk passphrase on the reboot that a dropped card needs.

## The second try: set up for a fresh boot

[run/gen2_fresh_boot.sh](run/gen2_fresh_boot.sh) prepares and runs it in steps, by hand. First, `install-entry`
writes a one-time GRUB entry (`/boot/grub/custom.cfg`, which GRUB reads without regenerating `grub.cfg`) and picks
it for the next boot only, with `grub-reboot`. The entry is the default one, plus:
- `modprobe.blacklist=nvidia`: the driver does not load itself at boot;
- `module_blacklist=nvidia_drm,nvidia_modeset`: the kernel refuses the two display modules, which the driver's udev
  rule loads by name as soon as the driver appears, and which initialize every card;
- `systemd.mask=nvidia-persistenced.service`: the persistence daemon stays off.

On that boot no card is initialized until the patched driver is loaded by hand. The steps then run in order:
`check`, `load`, `push` and `test`. The push is the P104 sequence exactly, once, without the extra steps of the
first try's step 3. `remove-entry` undoes the entry.

## Files

| File | What |
|---|---|
| [run/patch_gen1_bail.py](run/patch_gen1_bail.py) | the 2-byte copy of `nvidia.ko` 580.178.04, signature removed; refuses unless the sequence occurs exactly once |
| [run/gen2_push.py](run/gen2_push.py) | read-only check of the register mirror, then the push; refuses on any failed check |
| [run/retry_latch.py](run/retry_latch.py) | the retry that preceded the card's drop: **don't run it**; it is kept because it is what ran |
| [run/gen2_fresh_boot.sh](run/gen2_fresh_boot.sh) | the second try: the one-time boot entry, then `check`, `load`, `push`, `test` on the fresh boot |
| [../../../bench/p2p_check/](../../../bench/p2p_check/README.md) | `--host-cards`: host ↔ card copies checked word for word and timed (binary md5 `d392338dd555f251444c85d646c0ef23`) |

Commands (on the build machine, then the rig, as root; `$RIG` is your rig):

```bash
python3 -I run/patch_gen1_bail.py nvidia.ko nvidia.ko.gen1bail            # on the build machine, a copy of the rig's module
ssh "$RIG" 'insmod nvidia.ko.gen1bail NVreg_PreserveVideoMemoryAllocations=1 NVreg_TemporaryFilePath=/var; modprobe nvidia_uvm'
ssh "$RIG" 'python3 -I gen2_push.py 0000:07:00.0'                         # read-only checks
ssh "$RIG" 'python3 -I gen2_push.py 0000:07:00.0 push'
```

Provenance: the rig's own register reads and `lspci`; the method from the projects linked above, read only.
