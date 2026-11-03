- **harness: infrastructure: GO** — 16 of 16 attempts finished without an infrastructure failure; 1 tries re-run (pipe3/bugfix-python-affine-cipher/a1-t1-infra), 0 attempts given up
- **harness: proxy rows: NO-GO** — 81 requests logged; attempts with a row count different from pi's replies: pipe0/bugfix-python-list-ops/a1/t1 (3 rows / 2 replies); rows without timings: 1
- **harness: first token vs prefill: GO** — proxy first token minus server prefill: median 0.07 s, p90 0.18 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 73 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 96.1 %, min 49.5 %; cache paths exact 63, older checkpoint 0, reply re-read 1, unknown 1
- **memory: NO-GO** — lowest MemAvailable 138 MB; swap used 356 → 1485 MB
- server log server-pipe0.log: backend sampling off 0, checkpoint restored 7, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-pipe3.log: backend sampling off 0, checkpoint restored 0, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 1
- server log server-pipe3b.log: backend sampling off 0, checkpoint restored 8, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
