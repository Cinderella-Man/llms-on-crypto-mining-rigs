#!/usr/bin/env python3
"""load_lines.py: what each load's llama-server log says about C693's two switches, one row per load.

    python3 -I load_lines.py OUT > load_lines.tsv

OUT is a deep-lane results folder (bench/deep_lane), with server-<load>.log per load. Per log:
- h_nextn: the buffer for the MTP draft's rows (upstream #31927, LLAMA_MTP_H_DEV=1), "<device> <MiB>" per
  `h_nextn buffer size` line (printed at verbosity 4 only);
- host_rows: the lines that say the rows go through host memory instead: the switch's own (`LLAMA_MTP_H_DEV=0,
  extracting the h_nextn rows to host memory`, verbosity 4) and the fallback's warning (`<reason>, extracting ...`,
  any verbosity), each with its count;
- prefetch_first: how many `sched-trace: prefetch: split ...` lines (a scheduler's first early fetch of the next
  split's inputs, upstream #30966, with LLAMA_UBATCH_TRACE=1) and the first of them;
- prefetch_total: the `sched-trace: prefetch: N input(s) fetched early by this scheduler` lines (one per freed
  scheduler that fetched any), as their N;
- trace_lines: the number of `sched-trace:` lines of any kind (0 means the trace was not on);
- compute_buffers: each `compute buffer size` line's device and MiB, in log order (verbosity 4 only);
- buffer_warnings: llama.cpp's warnings that a compute buffer outgrew its reservation (`does not match expectation`).
Standard library only.
"""
import glob
import os
import re
import sys

src = sys.argv[1]
H = re.compile(r'(\S+) h_nextn buffer size = +([0-9.]+) MiB')
HOST = re.compile(r'(\S+): (.*), extracting the h_nextn rows to host memory')
PF1 = re.compile(r'sched-trace: prefetch: split .*')
PFN = re.compile(r'sched-trace: prefetch: (\d+) input\(s\) fetched early')
CB = re.compile(r'(\S+) compute buffer size = +([0-9.]+) MiB')
WARN = re.compile(r'(\S+) compute buffer size of ([0-9.]+) MiB, does not match expectation of ([0-9.]+) MiB')

print('\t'.join(['load', 'h_nextn', 'host_rows', 'prefetch_first', 'prefetch_total', 'trace_lines', 'compute_buffers',
                 'buffer_warnings']))
for path in sorted(glob.glob(os.path.join(src, 'server-*.log')), key=os.path.getmtime):
    load = os.path.basename(path)[len('server-'):-len('.log')]
    h, host, pf1, pfn, cb, warn = [], {}, [], [], [], []
    trace = 0
    with open(path, errors='replace') as f:
        for line in f:
            if 'sched-trace:' in line:
                trace += 1
                m = PFN.search(line)
                if m:
                    pfn.append(m.group(1))
                elif PF1.search(line):
                    pf1.append(PF1.search(line).group(0).strip())
                continue
            m = H.search(line)
            if m:
                h.append(f'{m.group(1)} {m.group(2)}')
                continue
            m = HOST.search(line)
            if m:
                key = f'{m.group(1)}: {m.group(2)}'
                host[key] = host.get(key, 0) + 1
                continue
            m = WARN.search(line)
            if m:
                warn.append(f'{m.group(1)} {m.group(2)}>{m.group(3)}')
                continue
            m = CB.search(line)
            if m:
                cb.append(f'{m.group(1)} {m.group(2)}')
    print('\t'.join([load,
                     '; '.join(h) or '-',
                     '; '.join(f'{k} (x{n})' for k, n in host.items()) or '-',
                     f'{len(pf1)}' + (f': {pf1[0]}' if pf1 else ''),
                     '+'.join(pfn) or '-',
                     str(trace),
                     '; '.join(cb) or '-',
                     '; '.join(warn) or '-']))
