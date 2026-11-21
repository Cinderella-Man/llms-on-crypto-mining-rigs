"""took_ms of a rig cards.csv by how many cards are busy (P0 and util > 0) and how many asleep (P8)."""
import csv, gzip, statistics, sys
for path in sys.argv[1:]:
    rows = [r for r in csv.DictReader(l for l in gzip.open(path, "rt") if not l.startswith("#"))]
    n = sum(1 for k in rows[0] if k.endswith("_pstate"))
    groups = {}
    for r in rows:
        p8 = sum(r[f"g{i}_pstate"] == "P8" for i in range(n))
        busy = sum(r[f"g{i}_util"] not in ("", "0") for i in range(n))
        key = "all asleep (P8)" if p8 == n else ("some busy" if busy else "awake, util 0")
        groups.setdefault(key, []).append(float(r["took_ms"]))
    print(path, len(rows), "rows,", n, "cards")
    for k, v in sorted(groups.items()):
        v.sort()
        print(f"  {k:18s} {len(v):6d} rows  median {statistics.median(v):6.1f} ms  p10 {v[len(v)//10]:6.1f}  p90 {v[len(v)*9//10]:6.1f}")
