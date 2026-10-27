#!/usr/bin/env bash
# Q106 (2026-10-24): G3 against the three Q8_0 references (agent text at 8k, wikitext-2 at 8k and at 512) with
# per-token dumps, MoE down-projection rescale off. The other candidates are the same call with their name, file and
# switch: ud (UD-Q4_K_XL, rescale on: ""), g3h, k2h, k1h (rescale off). Env as for common.sh.
set -e; . "$(dirname "$0")/common.sh"
score g3 $MODELS/flint-q73-G3.gguf "-e LLAMA_NO_MOE_DOWN_RESCALE=1"
