"""Print the chat template embedded in a GGUF (stdout) and its tokenizer facts (stderr).

usage: python gguf_meta.py MODEL.gguf > template.jinja
Needs llama.cpp's gguf-py: set GGUF_PY to its folder (default /src/llama.cpp/gguf-py) and have numpy installed.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.environ.get("GGUF_PY", "/src/llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

KEYS = ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.add_bos_token", "tokenizer.ggml.padding_token_id", "general.name", "general.architecture"]
MARKERS = ("<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<think>",
           "</think>", "<tool_response>", "</tool_response>")

r = GGUFReader(sys.argv[1])
meta = {k: (r.fields[k].contents() if k in r.fields else None) for k in KEYS}
template = r.fields["tokenizer.chat_template"].contents()
sys.stdout.write(template)
print(f"template sha256 {hashlib.sha256(template.encode()).hexdigest()} ({len(template)} chars)", meta, file=sys.stderr)
toks = r.fields["tokenizer.ggml.tokens"].contents()
print("special tokens:", [(i, t) for i, t in enumerate(toks) if t.startswith("<|") or t in MARKERS][:40],
      file=sys.stderr)
