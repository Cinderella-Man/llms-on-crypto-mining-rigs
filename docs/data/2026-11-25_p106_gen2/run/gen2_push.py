"""Ask one P106 card for a PCIe Gen2 link, for a test: unlock its Gen2 declaration, set its target speed to 5 GT/s,
start link training, then retrain from the port above it.

The register sequence is the one WJNKAC/nvidia-p104-unlock-gen2-for-linux found on a P104-100 (Pascal) with the same
driver, 580.178.04 (read, not run; their kernel module `gen2kr.c`, push_fn). This is an independent user-space version
for a P106-090:
  - every BAR0 write happens while the link is stable; the last one starts training, and after it the card's BAR0 is
    not touched again (their user-space writes hung a card only while the link was changing);
  - it refuses to write unless BAR0's mirror of the card's PCIe link registers (0x88084 LnkCap, 0x88088
    LnkCtl/LnkSta) equals the card's real config-space values, so wrong offsets for this chip stop it before any
    write, and unless the speed register can be trusted (not all ones, commit bit 0 clear).
Nothing is persistent: a power cycle returns the card to stock.

Usage (root, on the rig, with the NVIDIA driver loaded but the card not yet opened by anything):
  python3 -I gen2_push.py <bdf, e.g. 0000:07:00.0>            # read only: print the registers and the checks
  python3 -I gen2_push.py <bdf> push                          # the writes, the training trigger, the retrain
"""
import mmap
import os
import struct
import subprocess
import sys
import time

R_LNKCAP, R_LNKSTA, R_CAP2 = 0x88084, 0x88088, 0x880A4
R_CONFIG, R_VSEC, R_LTSSM = 0x8841C, 0x88610, 0x8872C
R_LINKCFG0, R_CYA = 0x8C040, 0x8C2C0
XP3G = ((0x8E110, 0x00000001), (0x8E11C, 0x00000004), (0x8E120, 0x00000000), (0x8E12C, 0x00200000))


def setpci(bdf, reg, val=None):
    arg = f"{reg}={val}" if val is not None else reg
    r = subprocess.run(["setpci", "-s", bdf, arg], capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        raise RuntimeError(f"setpci {bdf} {arg}: {r.stderr.strip()}")
    return int(r.stdout.strip(), 16) if val is None else None


def speed(lnksta_word):
    return {1: "2.5 GT/s", 2: "5 GT/s", 3: "8 GT/s"}.get(lnksta_word & 0xF, f"code {lnksta_word & 0xF}")


def main():
    bdf = sys.argv[1]
    push = len(sys.argv) > 2 and sys.argv[2] == "push"
    up = os.path.basename(os.path.realpath(f"/sys/bus/pci/devices/{bdf}/.."))
    print(f"card {bdf}, port above it {up}")

    cfg_lnkcap = setpci(bdf, "CAP_EXP+0x0c.l")
    cfg_lnkctl_sta = setpci(bdf, "CAP_EXP+0x10.l")
    up_lnkctl2 = setpci(up, "CAP_EXP+0x30.w")
    up_lnksta = setpci(up, "CAP_EXP+0x12.w")
    print(f"config: card LnkCap {cfg_lnkcap:#010x} (max speed code {cfg_lnkcap & 0xF}), "
          f"LnkCtl/LnkSta {cfg_lnkctl_sta:#010x} (link {speed(cfg_lnkctl_sta >> 16)}); "
          f"port LnkCtl2 {up_lnkctl2:#06x} (target code {up_lnkctl2 & 0xF}), LnkSta {up_lnksta:#06x} ({speed(up_lnksta)})")

    fd = os.open(f"/sys/bus/pci/devices/{bdf}/resource0", os.O_RDWR | os.O_SYNC)
    m = mmap.mmap(fd, 16 << 20, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0)
    rd = lambda o: struct.unpack_from("<I", m, o)[0]

    def wr(o, v):
        struct.pack_into("<I", m, o, v & 0xFFFFFFFF)

    regs = {name: rd(off) for name, off in (("LnkCap mirror", R_LNKCAP), ("LnkCtl/Sta mirror", R_LNKSTA),
                                             ("LnkCap2 mirror", R_CAP2), ("PRIV_MISC_1", R_CONFIG),
                                             ("VSEC_HIERARCHY", R_VSEC), ("LINK_CONFIG_0", R_LINKCFG0),
                                             ("CYA_0", R_CYA))}
    for name, v in regs.items():
        print(f"  {name:18s} {v:#010x}")
    lc0 = regs["LINK_CONFIG_0"]
    print(f"  LINK_CONFIG_0 SPEED field {(lc0 >> 18) & 3} (1 = 5 GT/s, 2 = 2.5 GT/s), commit bit {lc0 & 1}; "
          f"CYA_0 DIS_G2 bit {(regs['CYA_0'] >> 2) & 1}")

    checks = [
        ("BAR0 LnkCap mirror equals config LnkCap", regs["LnkCap mirror"] == cfg_lnkcap),
        ("BAR0 LnkCtl/Sta mirror equals config LnkCtl/LnkSta", regs["LnkCtl/Sta mirror"] == cfg_lnkctl_sta),
        ("no register reads all ones", all(v != 0xFFFFFFFF for v in regs.values())),
        ("LINK_CONFIG_0 commit bit clear", (lc0 & 1) == 0),
        ("the port above can target 5 GT/s", (up_lnkctl2 & 0xF) >= 2),
    ]
    for what, ok in checks:
        print(f"  check: {what}: {'yes' if ok else 'NO'}")
    if not push:
        m.close()
        os.close(fd)
        return 0
    if not all(ok for _, ok in checks):
        print("refusing to write: a check failed")
        m.close()
        os.close(fd)
        return 1

    # 1. the Gen2 declaration, 2. Gen2 not disabled, 3. target speed 5 GT/s, all while the link is stable
    for off, v in XP3G:
        wr(off, v)
    v = rd(R_CONFIG)
    wr(R_CONFIG, (v | (1 << 11) | (1 << 13) | (1 << 15)) & ~((1 << 12) | (1 << 14)))
    v = rd(R_VSEC)
    wr(R_VSEC, (v & ~(1 << 12)) | 1)
    v = rd(R_CYA)
    wr(R_CYA, v & ~(1 << 2))
    v = rd(R_LINKCFG0)
    if v == 0xFFFFFFFF or (v & 1):
        print(f"LINK_CONFIG_0 reads {v:#010x} now: not writing the speed, stopping before the training trigger")
        m.close()
        os.close(fd)
        return 1
    wr(R_LINKCFG0, (v & ~0x000C0000) | (1 << 18))
    print(f"written: PRIV_MISC_1 {rd(R_CONFIG):#010x}, VSEC_HIERARCHY {rd(R_VSEC):#010x}, "
          f"CYA_0 {rd(R_CYA):#010x}, LINK_CONFIG_0 {rd(R_LINKCFG0):#010x}")
    # 4. start training; the card's BAR0 is not touched after this
    wr(R_LTSSM, 6)
    m.close()
    os.close(fd)
    time.sleep(0.15)

    # 5. retrain from the port above (config space of the port only)
    ctl = setpci(up, "CAP_EXP+0x10.w")
    setpci(up, "CAP_EXP+0x10.w", f"{(ctl | 0x20) & 0xFFFF:04x}")
    for i in range(20):
        time.sleep(0.1)
        sta = setpci(up, "CAP_EXP+0x12.w")
        if (sta & 0xF) >= 2:
            break
    print(f"port LnkSta after retrain: {sta:#06x} ({speed(sta)}) after {(i + 1) * 100} ms")
    print(f"card config now: LnkCap {setpci(bdf, 'CAP_EXP+0x0c.l'):#010x}, "
          f"LnkSta {setpci(bdf, 'CAP_EXP+0x12.w'):#06x} ({speed(setpci(bdf, 'CAP_EXP+0x12.w'))})")
    return 0 if (sta & 0xF) >= 2 else 3


if __name__ == "__main__":
    sys.exit(main())
