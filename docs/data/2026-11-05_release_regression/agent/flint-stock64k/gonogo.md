- **harness: infrastructure: GO** — 8 of 8 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 44 requests logged; attempts with a row count different from pi's replies: rel/bugfix-python-affine-cipher/a1/t1 (4 rows / 3 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.20 s, p90 0.27 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 39 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 94.0 %, min 45.3 %; cache paths exact 34, older checkpoint 0, reply re-read 1, unknown 1
- **memory: NO-GO** — lowest MemAvailable 1277 MB; swap used 31 → 287 MB
- server log server-rel.log: backend sampling off 47, checkpoint restored 0, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
