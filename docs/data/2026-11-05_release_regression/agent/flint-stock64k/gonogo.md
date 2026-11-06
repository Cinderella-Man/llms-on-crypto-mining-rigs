- **harness: infrastructure: GO** — 12 of 12 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 67 requests logged; attempts with a row count different from pi's replies: rel/bugfix-python-affine-cipher/a1/t1 (4 rows / 3 replies), rel/bugfix-python-scale-generator/a1/t1 (5 rows / 2 replies); rows without timings: 2
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.20 s, p90 0.28 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 59 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: NO-GO** — turns ≥ 2: median cache hit 93.5 %, min 0.0 %; cache paths exact 47, older checkpoint 2, reply re-read 4, unknown 2
- **memory: NO-GO** — lowest MemAvailable 1277 MB; swap used 31 → 292 MB
- server log server-rel.log: backend sampling off 68, checkpoint restored 0, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
