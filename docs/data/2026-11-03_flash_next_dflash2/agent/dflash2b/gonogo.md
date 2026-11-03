- **harness: infrastructure: GO** — 31 of 32 attempts finished without an infrastructure failure; 3 tries re-run (dflash2/bugfix-python-affine-cipher/a1-t1-infra, mtp/bugfix-javascript-space-age/a1-t1-infra, mtp/bugfix-python-affine-cipher/a1-t1-infra), 0 attempts given up
- **harness: proxy rows: GO** — 183 requests logged; attempts with a row count different from pi's replies: none; rows without timings: 0
- **harness: first token vs prefill: NO-GO** — proxy first token minus server prefill: median 0.08 s, p90 1.19 s (limit 0.5 s + round trip 0.30 s; includes the per-request fixed cost before prefill)
- **tool calls: GO** — malformed 0 of 169 calls (0.0 %: unknown tool 0, invalid arguments 0, markup in text 0); calls inside the reasoning 0
- **cache: GO** — turns ≥ 2: median cache hit 96.3 %, min 43.2 %; cache paths exact 148, older checkpoint 0, reply re-read 3
- **memory: NO-GO** — lowest MemAvailable 177 MB; swap used 32 → 1475 MB
- server log server-dflash2.log: backend sampling off 0, checkpoint restored 0, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 1
- server log server-dflash2b.log: backend sampling off 0, checkpoint restored 10, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-mtp.log: backend sampling off 0, checkpoint restored 2, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-mtp2.log: backend sampling off 0, checkpoint restored 12, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0
- server log server-nodraft.log: backend sampling off 0, checkpoint restored 9, checkpoint created 0, checkpoint erased 0, full re-processing 0, screened head 0, NaN / inf 0, watchdog kill 0

**Overall: NO-GO**
