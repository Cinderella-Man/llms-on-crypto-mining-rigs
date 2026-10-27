- **harness: infrastructure: GO** — 9 of 9 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 57 requests logged; attempts with a row count different from pi's replies: rel/bugfix-python-scale-generator/a1/t1 (5 rows / 4 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.20 s, p90 0.35 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 53 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 95.0 %, min 37.0 %; cache paths exact 44, older checkpoint 0, reply re-read 3, unknown 1
- **memory: NO-GO** — lowest MemAvailable 627 MB; swap used 355 → 614 MB
- server log server-rel.log: backend sampling off 0, checkpoint restored 11, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
