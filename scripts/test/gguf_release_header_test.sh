#!/usr/bin/env bash
# gguf_release_header_test.sh: checks scripts/gguf_release_header.py on a small GGUF it writes itself, with the header
# Unsloth's Flint files carry (license other / flint-preview, quantized_by Unsloth, an importance-matrix record, a
# string array). Needs python3 with numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py); no GPU, no model file.
# Usage: scripts/test/gguf_release_header_test.sh   (PASS/FAIL per check; exit 0 when all pass)
set -u
here=$(cd "$(dirname "$0")" && pwd)
TOOL=$(cd "$here/.." && pwd)/gguf_release_header.py
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; fails=$((fails + 1)); fi; }

python3 - "$T/in.gguf" <<'EOF'
import sys
import numpy as np
import gguf
w = gguf.GGUFWriter(sys.argv[1], arch="llama")
w.add_name("Flint Preview")
w.add_quantized_by("Unsloth")
w.add_license("other")
w.add_license_name("flint-preview")
w.add_license_link("https://example.invalid/flint-preview")
w.add_string("quantize.imatrix.file", "flint-imatrix-unsloth.gguf")
w.add_int32("quantize.imatrix.entries_count", 337)
w.add_array("tokenizer.ggml.tokens", ["<s>", "</s>", "hello", "world"])
w.add_uint32("llama.context_length", 262144)
rng = np.random.default_rng(1)
w.add_tensor("token_embd.weight", rng.standard_normal((16, 64), dtype=np.float32))
w.add_tensor("blk.0.attn_q.weight", rng.standard_normal((64, 64)).astype(np.float16))
w.add_tensor("output.weight", rng.standard_normal((16, 64), dtype=np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
EOF
check "test file written" '[ -s $T/in.gguf ]'

python3 "$TOOL" "$T/in.gguf" "$T/dry.gguf" --preset flint --name "Flint K2h-r16" --dry-run > "$T/dry.out" 2>&1
check "dry run: lists the changes, writes nothing" '[ ! -e $T/dry.gguf ] && grep -q "^~ general.license = .other. -> .apache-2.0." $T/dry.out && grep -q "^- general.license.name" $T/dry.out && grep -q "^+ general.sampling.min_p = 0.0 (FLOAT32)" $T/dry.out'

python3 "$TOOL" "$T/in.gguf" "$T/out.gguf" --preset flint --name "Flint K2h-r16" --quantized-by "a test" \
    --set general.url=https://example.invalid/repo > "$T/run.out" 2>&1
echo $? > "$T/run.rc"
check "rewrite: exit 0, verified" '[ "$(cat $T/run.rc)" = 0 ] && grep -q "^verified" $T/run.out'

python3 - "$T/out.gguf" > "$T/fields.txt" <<'EOF'
import sys
import gguf
r = gguf.GGUFReader(sys.argv[1])
for f in r.fields.values():
    v = f.contents()
    print(f"{f.name}\t{f.types[0].name}\t{len(v) if isinstance(v, list) else v}")
EOF
field() { awk -F'\t' -v k="$1" '$1 == k { print $2 "\t" $3 }' "$T/fields.txt"; }
check "license apache-2.0, name and link gone" '[ "$(field general.license)" = "$(printf "STRING\tapache-2.0")" ] && [ -z "$(field general.license.name)" ] && [ -z "$(field general.license.link)" ]'
check "name, quantized_by and url set" '[ "$(field general.name | cut -f2)" = "Flint K2h-r16" ] && [ "$(field general.quantized_by | cut -f2)" = "a test" ] && [ "$(field general.url | cut -f2)" = "https://example.invalid/repo" ]'
check "base model Flint by RecursiveSI" '[ "$(field general.base_model.count)" = "$(printf "UINT32\t1")" ] && [ "$(field general.base_model.0.organization | cut -f2)" = RecursiveSI ]'
check "the makers sampling with llama.cpp key types" '[ "$(field general.sampling.top_k)" = "$(printf "INT32\t20")" ] && [ "$(field general.sampling.min_p)" = "$(printf "FLOAT32\t0.0")" ] && field general.sampling.top_p | grep -qP "^FLOAT32\t0\.9499"'
check "imatrix record, token array and context length kept" '[ "$(field quantize.imatrix.file | cut -f2)" = flint-imatrix-unsloth.gguf ] && [ "$(field tokenizer.ggml.tokens)" = "$(printf "ARRAY\t4")" ] && [ "$(field llama.context_length | cut -f2)" = 262144 ]'

python3 "$TOOL" "$T/in.gguf" "$T/bad.gguf" --set general.architecture=qwen3 > "$T/bad.out" 2>&1
echo $? > "$T/bad.rc"
check "architecture change refused" '[ "$(cat $T/bad.rc)" != 0 ] && [ ! -e $T/bad.gguf ]'
python3 "$TOOL" "$T/out.gguf" > "$T/show.out" 2>&1
check "print mode shows the new header" 'grep -q "^general.license = .apache-2.0." $T/show.out && grep -q "^3 tensors" $T/show.out'
echo "$fails failed"
[ "$fails" = 0 ]
