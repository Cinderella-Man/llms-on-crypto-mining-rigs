import sys, hashlib
sys.path.insert(0, "/src/llama.cpp/gguf-py")
from gguf import GGUFReader
for p in sys.argv[1:]:
    r = GGUFReader(p)
    out = {}
    for k in ["tokenizer.chat_template","tokenizer.ggml.eos_token_id","tokenizer.ggml.eot_token_id","tokenizer.ggml.bos_token_id","tokenizer.ggml.add_bos_token","tokenizer.ggml.padding_token_id","general.name","general.architecture"]:
        f = r.fields.get(k)
        if f is None: out[k]=None; continue
        v = f.contents()
        out[k]=v
    t = out.pop("tokenizer.chat_template") or ""
    print(p, hashlib.sha256(t.encode()).hexdigest()[:16], len(t), out)
    if "--dump" in p or len(sys.argv)==2:
        open("/bfcl/template_from_gguf.jinja","w").write(t)
    toks = r.fields["tokenizer.ggml.tokens"].contents()
    for i in [out.get("tokenizer.ggml.eos_token_id"), out.get("tokenizer.ggml.eot_token_id")]:
        if i is not None: print("  tok", i, repr(toks[i]))
    print("  specials:", [ (i,t) for i,t in enumerate(toks) if t.startswith("<|") or t in ("<tool_call>","</tool_call>","<arg_key>","</arg_key>","<arg_value>","</arg_value>","<think>","</think>","<tool_response>","</tool_response>")][:40])
