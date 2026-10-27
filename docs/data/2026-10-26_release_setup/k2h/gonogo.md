- **harness: infrastructure: GO** — 10 of 10 attempts finished without an infrastructure failure; 0 tries re-run (none), 0 attempts given up
- **harness: proxy rows: NO-GO** — 64 requests logged; attempts with a row count different from pi's replies: rel/bugfix-python-scale-generator/a1/t1 (15 rows / 14 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.19 s, p90 0.40 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 60 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 94.2 %, min 18.8 %; cache paths exact 49, older checkpoint 0, reply re-read 4, unknown 1
- **memory: NO-GO** — lowest MemAvailable 726 MB; swap used 376 → 633 MB
- server log server-rel.log: backend sampling off 0, checkpoint restored 13, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
