"""After gen2_push.py left a card advertising Gen1 only (the declaration did not latch): two more tries.

  1. the training trigger again (0x8872C: 0, then 6) and up to 10 s for the declaration (LnkCap2 mirror 0x880A4) to
     read 0x6, then a retrain from the port above;
  2. the extra write of another P104-100 tool (skbskhud/p104-unlock-v3, read, not run): 0x8C1C0 bits 19:16 to 2,
     while the link is stable, then the same trigger, wait and retrain.

Usage (root): python3 -I retry_latch.py <card bdf> <port bdf>
"""
import mmap
import os
import struct
import subprocess
import sys
import time

bdf, up = sys.argv[1], sys.argv[2]


def setpci(dev, reg, val=None):
    a = f"{reg}={val}" if val is not None else reg
    r = subprocess.run(["setpci", "-s", dev, a], capture_output=True, text=True, timeout=20)
    return int(r.stdout.strip(), 16) if val is None else None


def bar():
    fd = os.open(f"/sys/bus/pci/devices/{bdf}/resource0", os.O_RDWR | os.O_SYNC)
    return fd, mmap.mmap(fd, 16 << 20, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0)


def rd(m, o):
    return struct.unpack_from("<I", m, o)[0]


def wr(m, o, v):
    struct.pack_into("<I", m, o, v & 0xFFFFFFFF)


def retrain(tag):
    ctl = setpci(up, "CAP_EXP+0x10.w")
    setpci(up, "CAP_EXP+0x10.w", f"{(ctl | 0x20) & 0xFFFF:04x}")
    for _ in range(20):
        time.sleep(0.1)
        sta = setpci(up, "CAP_EXP+0x12.w")
        if (sta & 0xF) >= 2:
            break
    print(f"  {tag}: port LnkSta {sta:#06x}, card LnkCap {setpci(bdf, 'CAP_EXP+0x0c.l'):#010x}")
    return (sta & 0xF) >= 2


def trigger_and_wait(m, secs=10):
    wr(m, 0x8872C, 0)
    wr(m, 0x8872C, 6)
    t0 = time.time()
    while time.time() - t0 < secs:
        if rd(m, 0x880A4) == 0x6:
            return time.time() - t0
        time.sleep(0.5)
    return None


fd, m = bar()
print(f"start: LnkCap2 mirror {rd(m, 0x880A4):#x}, PRIV_MISC_1 {rd(m, 0x8841C):#010x}, "
      f"LINK_CONFIG_0 {rd(m, 0x8C040):#010x}, 0x8C1C0 {rd(m, 0x8C1C0):#010x}")
t = trigger_and_wait(m)
print(f"try 1, trigger again: declaration {'latched after %.1f s' % t if t is not None else 'not latched'}")
m.close()
os.close(fd)
ok = retrain("try 1 retrain")

if not ok:
    fd, m = bar()
    v = rd(m, 0x8C1C0)
    if v == 0xFFFFFFFF:
        print("0x8C1C0 reads all ones: not writing")
    else:
        wr(m, 0x8C1C0, (v & ~0x000F0000) | (2 << 16))
        print(f"try 2: 0x8C1C0 {v:#010x} -> {rd(m, 0x8C1C0):#010x}")
        t = trigger_and_wait(m)
        print(f"try 2: declaration {'latched after %.1f s' % t if t is not None else 'not latched'}")
    m.close()
    os.close(fd)
    ok = retrain("try 2 retrain")
print("result:", "Gen2" if ok else "still Gen1")
sys.exit(0 if ok else 3)
