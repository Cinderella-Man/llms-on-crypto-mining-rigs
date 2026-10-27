# Q36 (2026-10-23): prints the job for a combined file: S4 with the named 8-layer groups lowered to Q4_1 (each class's
# tensors taken from S1 or S2), the router retyped to F16, its bytes per token estimated, and scored rescale-off on the
# GPU set given as <lane>. S4-L0 = 'a5 a6', L1 = L0 + 'd4 d5 d6', L2 = L1 + 'u4 u5 u6' (the d and u groups are clipped
# to the layers that class covers). Env for the printed job as for q36_compose_score.sh.
#   q36_final.py <name> <lane> <class><group>...      e.g. q36_final.py flint-S4L0 q36d a5 a6 > job.sh
import sys
name, lane, items = sys.argv[1], sys.argv[2], sys.argv[3:]
groups = {1: range(0, 9), 2: range(9, 17), 3: range(17, 25), 4: range(25, 33), 5: range(33, 41), 6: range(41, 49)}
tens = {'a': ('attn_output\\.weight', 'flint-mixS1.gguf', lambda l: True),
        'd': ('ffn_down_exps\\.weight', 'flint-mixS2.gguf', lambda l: 2 <= l <= 45),
        'u': ('ffn_(gate|up)_exps\\.weight', 'flint-mixS2.gguf', lambda l: l >= 2)}
M = '$MODELS/q36'
layers = {c: [] for c in tens}
for it in items:
    c, g = it[0], int(it[1:])
    layers[c] += [l for l in groups[g] if tens[c][2](l)]
rules = [f"'blk\\.({'|'.join(map(str, sorted(ls)))})\\.{tens[c][0]}='\"$MODELS/{tens[c][1]}\""
         for c, ls in layers.items() if ls]
print(f"""set -euo pipefail; mkdir -p {M}
python3 scripts/compose_gguf.py "$MODELS/flint-mixS4.gguf" "{M}/tmp-{name}.gguf" {' '.join(rules)}
python3 scripts/retype_tensors.py "{M}/tmp-{name}.gguf" "{M}/{name}.gguf" 'ffn_gate_inp=F16' | tail -1
rm -f "{M}/tmp-{name}.gguf"
python3 scripts/token_cost.py --fixed-ms 8.4 "{M}/{name}.gguf" | head -2
ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 docs/data/2026-10-23_q36_q45/run/kld_gpu.sh "{M}/{name}.gguf" _nr_{lane}
""")
