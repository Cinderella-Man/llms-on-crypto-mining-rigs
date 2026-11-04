# Q551 and Q561: the retyped Swift Q2_0 with the DFlash2 drafter (not run yet)

The next step of the Flash-Next speed ladder ([Q551](../../../quants/experiments.md)): the agent check
(`bench/agent_check`) of [Q218's retyped Swift-1.5 GSQ-RCO Q2_0](../2026-11-03_retype_agent/README.md) (the
hyper-connection projections in Q8_0, the router in F16) with Inco AI's DFlash2 drafter on the line
[Q561](../2026-11-03_flash_next_dflash2/README.md) found best (`dflash2b`: `--spec-draft-n-max 2`, `-b 512`,
`--no-host`, the drafter on the output head's card). Each change alone gave 9.0 and 9.5 tok/s against 7.8 without a
draft; this run measures the two together.

**Status: not run.** It was stopped on 2026-11-03 while the model file was still being copied to the rig, before
anything ran on the cards. It is queued again, and its results will land in this folder.

## Files

- [run/s28_tables.py](run/s28_tables.py): the tables for the smoke test and the timed arm (tokens per second, draft
  acceptance, host memory, the tasks every arm ran). It is Q561's `s18_tables.py` extended to smoke tests with any
  number of requests. The usage is in its docstring.
