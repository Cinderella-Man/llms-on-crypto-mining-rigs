#!/usr/bin/env python3
"""fold_updates.py: a host profile (GGML_CUDA_PROFILE_HOST) of a load with LLAMA_QSA_KPOOL_DEV=1, made readable by
host_depth.py (docs/data/2026-11-10_step_parts_depth/run/host_depth.py). Standard library only.

    python3 -I fold_updates.py IN.csv OUT.csv [--emb 4] [--last 8] [--max-gpu-us 200]

With the switch, every pass of the MTP draft evaluates once more on the draft's last card (card 8) before its
embedding card (card 4): the split that writes the k-pool tables' update into the card's copy (the stage's cast and
two SET_ROWS, 30-60 us of GPU time). host_depth.py ends a draft pass at each evaluation on the last card, so it would
count every pass twice. This script gives each such evaluation the embedding card's number, so that a draft pass is
again the evaluations up to one on the last card: the update, the embedding card, the last card. The pass's lead then
starts at the update's entry, its crossing is still the embedding card's exit to the last card's entry, and the update's
GPU time counts with the embedding card's (emb_gpu); no time is dropped. An evaluation is folded when it is on the last
card, the next evaluation in time is on the embedding card, and its GPU time is under --max-gpu-us (the draft's real
pass on the last card takes milliseconds). The count of folded evaluations goes to stderr. Rows keep their order and
values; only the device column of the folded ones changes."""
import argparse
import csv
import sys

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('dst')
ap.add_argument('--emb', type=int, default=4)
ap.add_argument('--last', type=int, default=8)
ap.add_argument('--max-gpu-us', type=float, default=200.0)
A = ap.parse_args()

with open(A.src, newline='') as f:
    rd = csv.reader(f)
    head = next(rd)
    rows = [r for r in rd]
iD, iE, iG = head.index('device'), head.index('host_enter_ms'), head.index('gpu_us')
ok = []
for k, r in enumerate(rows):
    try:
        ok.append((float(r[iE]), k))
    except (ValueError, IndexError):
        pass  # a line cut at the server's exit stays as it is
ok.sort()
folded = 0
for (t, k), (_, k2) in zip(ok, ok[1:]):
    r, n = rows[k], rows[k2]
    try:
        if int(r[iD]) == A.last and int(n[iD]) == A.emb and float(r[iG]) < A.max_gpu_us:
            r[iD] = str(A.emb)
            folded += 1
    except (ValueError, IndexError):
        continue
with open(A.dst, 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(head)
    w.writerows(rows)
print('fold_updates: %d of %d evaluations folded from card %d to card %d' % (folded, len(rows), A.last, A.emb),
      file=sys.stderr)
