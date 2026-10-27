#!/usr/bin/env bash
# Q126 (2026-10-24, two V100s): K2-r16 (K2h-r16 with G3's Q8_0 head, F16 router) on Q106's 24 agent chunks at 8k
# against the Q8_0 reference, with dumps for the written/read/schema split. UD-Q4_K_XL first, as the paired baseline on
# these cards (rescale on), then K2-r16, then K2h, whose RTX 3090 dump shows what the card type alone moves (both
# rescale off). Q127's probes are the same call: agent g3-v100 / swa-v100 / g3a-v100 "$NR", swx-v100 and g3r-v100 (G3
# with the rescale on) "". Env: common.sh.
set -e; . "$(dirname "$0")/common.sh"
agent ud-v100 $UD ""
agent k2r16-v100 $MODELS/flint-q126-K2-r16.gguf "$NR"
agent k2h-v100 $MODELS/flint-q78k-K2h.gguf "$NR"
