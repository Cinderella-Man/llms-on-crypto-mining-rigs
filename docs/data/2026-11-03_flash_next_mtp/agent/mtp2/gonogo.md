- **harness: infrastructure: GO** — 20 of 21 attempts finished without an infrastructure failure; 2 tries re-run (mtp/bugfix-javascript-space-age/a1-t1-infra, mtp/bugfix-python-affine-cipher/a1-t1-infra), 0 attempts given up
- **harness: proxy rows: GO** — 123 requests logged; attempts with a row count different from pi's replies: none; rows without timings: 0
- **harness: first token vs prefill: NO-GO** — proxy first token minus server prefill: median 0.09 s, p90 1.45 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 116 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 96.4 %, min 43.2 %; cache paths exact 99, older checkpoint 0, reply re-read 3
- **memory: NO-GO** — lowest MemAvailable 195 MB; swap used 32 → 1475 MB
- server log server-mtp.log: backend sampling off 0, checkpoint restored 2, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-mtp2.log: backend sampling off 0, checkpoint restored 12, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-nodraft.log: backend sampling off 0, checkpoint restored 9, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
