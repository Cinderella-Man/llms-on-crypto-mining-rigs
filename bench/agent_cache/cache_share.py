#!/usr/bin/env python3
"""Prompt-token accounting for multi-turn agent sessions: replays session logs and counts the prompt tokens a cache
would serve under four policies: no cache; llama-server today (prefix to 4 tokens before the previous prompt's end);
plus exact reuse of the generated turn; plus the first request restored from a saved state. It made the table in
docs/research/notes/prefill_pascal.md section 4.3 (Q41's E2 pilot logs: 200 BFCL v4 multi-turn sessions on
UD-Q4_K_XL, 2,364 requests).
Usage: python3 -I cache_share.py <session log .jsonl> [...]
  the logs bench/bfcl_flint/multi_turn.py run writes (raw/mt-<name>.jsonl): one record per session, turns -> steps,
  each step with n_prompt (the whole prompt re-sent) and tokens_predicted (generated, reasoning included).
"""
import json, sys, statistics as st
# Prompt-token accounting for multi-turn agent sessions (BFCL v4 multi-turn logs of Q41's E2 lane).
# Each record: turns -> steps; each step has n_prompt (whole prompt re-sent) and tokens_predicted (generated, incl. reasoning).
tot = dict(nocache=0, server=0, live=0, live_xsess=0, gen=0)
steps_n = 0; sessions = 0; first_prompts = []; deltas = []; newonly = []
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        steps = [s for t in r['turns'] for s in t]
        if not steps: continue
        sessions += 1
        first_prompts.append(steps[0]['n_prompt'])
        prev_p = None; prev_g = 0
        for s in steps:
            p, g = s['n_prompt'], s['tokens_predicted']
            tot['nocache'] += p; tot['gen'] += g; steps_n += 1
            if prev_p is None:
                tot['server'] += p; tot['live'] += p; tot['live_xsess'] += 0  # system prompt + tools from a saved state
                # cross-session: count only tokens after the shared prefix; approximate shared prefix by min first prompt later
            else:
                d = p - prev_p                     # re-rendered previous assistant turn + tool result / user message
                n = max(d - prev_g, 0)             # only tool result / user message, if the assistant turn matches its generated tokens
                deltas.append(d); newonly.append(n)
                tot['server'] += d + 4             # llama-server today: checkpoint 4 tokens before the previous prompt end
                tot['live'] += n                   # live cache through the generated tokens (exact re-render)
                tot['live_xsess'] += n
            prev_p, prev_g = p, g
# cross-session: first prompt minus a common prefix (system prompt + tool docs differ per BFCL task, so use 0 as the lower bound)
print(f"sessions {sessions}, requests {steps_n}, requests per session {steps_n/sessions:.1f}")
print(f"first prompt tokens: median {st.median(first_prompts):.0f}, mean {st.mean(first_prompts):.0f}")
print(f"per-step prompt growth: median {st.median(deltas):.0f}, mean {st.mean(deltas):.0f}; of it new (non-generated): median {st.median(newonly):.0f}, mean {st.mean(newonly):.0f}")
print(f"generated tokens per request: mean {tot['gen']/steps_n:.0f}")
for k in ('nocache','server','live'):
    print(f"{k:8s} prompt tokens processed {tot[k]:9d}  share of no-cache {tot[k]/tot['nocache']:.3f}  per generated token {tot[k]/tot['gen']:.2f}")
x = tot['live'] - sum(first_prompts)
print(f"live cache + first prompt from a saved state: {x} ({x/tot['nocache']:.3f} of no-cache, {x/tot['gen']:.2f} per generated token)")
