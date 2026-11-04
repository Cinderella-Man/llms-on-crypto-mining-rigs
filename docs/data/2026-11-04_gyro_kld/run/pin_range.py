#!/usr/bin/env python3
"""Holds one byte range of a file in the page cache: maps it read-only and mlock()s it (which reads it in, in order),
then sleeps, so another process that maps or reads the same file (a model's n-gram table read at random) finds it in
RAM. Exits after SECONDS or on SIGTERM; the pages are unlocked when it exits. Needs CAP_IPC_LOCK (root).
  python3 -I pin_range.py FILE OFFSET LENGTH SECONDS
"""
import ctypes, os, sys, time

path, start, length, seconds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
page = os.sysconf("SC_PAGE_SIZE")
off = start // page * page
size = length + (start - off)
fd = os.open(path, os.O_RDONLY)
addr = libc.mmap(None, size, 1, 1, fd, off)  # PROT_READ, MAP_SHARED
if addr in (None, ctypes.c_void_p(-1).value):
    sys.exit(f"mmap failed: {os.strerror(ctypes.get_errno())}")
t0 = time.time()
if libc.mlock(addr, size) != 0:
    sys.exit(f"mlock failed: {os.strerror(ctypes.get_errno())}")
print(f"{time.strftime('%F %T')} pinned {size} bytes of {os.path.basename(path)} in {time.time() - t0:.0f} s", flush=True)
time.sleep(seconds)
print(f"{time.strftime('%F %T')} released", flush=True)
