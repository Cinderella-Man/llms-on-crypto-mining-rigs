"""Flint's chat template and a llama-server client with the makers' sampling, shared by the task harnesses
(bfcl_flint, livecodebench_flint).

Prompts are rendered on the client with the template embedded in the GGUF (bfcl_flint/gguf_meta.py extracts it) and
sent as raw text to llama-server's native /completion endpoint: no --jinja, no server-side reasoning or tool parsing,
no grammar. The output is the raw text, including the <think> block.
"""
import datetime
import hashlib
import json
import os
import subprocess

import requests

# The makers' sampling (Flint's model card, generation_config.json, rec's defaults), sent with every request. The
# other llama-server samplers are set to their neutral values so that server defaults cannot change the
# distribution (llama-server's own default min-p is 0.05). The chain order is pinned; rec (mlx-lm's make_sampler)
# also applies top-k before top-p.
SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "typical_p": 1.0,
    "top_n_sigma": -1.0,
    "xtc_probability": 0.0,
    "dry_multiplier": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mirostat": 0,
    "samplers": ["top_k", "top_p", "min_p", "temperature"],
}


def parse_attempts(spec):
    """'1-5' -> [1, 2, 3, 4, 5]; '4,5' -> [4, 5]. Attempt n samples with seed n (plus any offset)."""
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def sha256_of(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def make_template(template_str):
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise RuntimeError(message)

    # Same settings as transformers' _compile_jinja_template
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template_str)


def server_props(url):
    p = requests.get(f"{url}/props", timeout=60, headers={"Connection": "close"}).json()
    out = {k: p.get(k) for k in ("total_slots", "model_path", "build_info")}
    out["n_ctx_slot"] = p["default_generation_settings"]["n_ctx"]
    if p.get("chat_template"):
        out["chat_template_sha256"] = hashlib.sha256(p["chat_template"].encode()).hexdigest()
    return out


def count_tokens(url, prompt):
    r = requests.post(f"{url}/tokenize", json={"content": prompt}, timeout=600, headers={"Connection": "close"})
    r.raise_for_status()
    return len(r.json()["tokens"])


def complete(url, prompt, n_predict, seed, cache_prompt):
    """One sampled completion of a pre-rendered prompt on /completion. A new connection per request (keep-alive
    gives spurious HTTP 400s against llama-server, gorilla #1339). Server timings are not kept."""
    body = {"prompt": prompt, "n_predict": n_predict, "seed": seed, "cache_prompt": cache_prompt,
            "stream": False, **SAMPLING}
    r = requests.post(f"{url}/completion", json=body, timeout=14400, headers={"Connection": "close"})
    r.raise_for_status()
    j = r.json()
    return {
        "content": j["content"],
        "tokens_predicted": j.get("tokens_predicted"),
        "tokens_evaluated": j.get("tokens_evaluated"),
        "stop_type": j.get("stop_type"),
        "truncated": j.get("truncated"),
    }


def manifest(url, extra):
    """What a run needs to be reproducible; written next to the raw output."""
    try:
        commit = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    return {"date": datetime.datetime.now().isoformat(timespec="seconds"), "harness_commit": commit,
            "sampling": SAMPLING, "server": server_props(url), **extra}
