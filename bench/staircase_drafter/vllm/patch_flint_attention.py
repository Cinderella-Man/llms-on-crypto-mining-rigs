"""Let vLLM's Flint model run on GPUs below SM100 (RTX 3090: compute 8.6) with an attention backend that serves sinks.

vLLM's Flint model (PR #62938, nightly from 899a1689) asserts FlashInfer attention on every device, and vLLM's
FlashInfer serves attention sinks only on compute 9.0, 10.x and 12.x. TRITON_ATTN serves sinks on any compute
capability. This patch makes FlintAttention fall back to the standard Attention layer whenever the backend is not
FlashInfer: Flint's own Triton kernel still applies the per-head q/k RMSNorm and RoPE (with its cache write skipped),
then `self.attn(q, k, v)` writes the KV cache and runs attention. The FlashInfer path is unchanged.

usage: python patch_flint_attention.py IMAGE_MODEL_PY OUT_MODEL_PY
Mount OUT_MODEL_PY read-only over <site-packages>/vllm/models/flint/model.py and serve with
--attention-backend TRITON_ATTN. Every anchor must occur exactly once, so a changed upstream file fails loudly.
"""
import sys

EDITS = [
    (  # __init__: stop before the FlashInfer imports and asserts
        "        # The fused kernel writes the KV cache itself (in the FlashInfer layout)\n",
        "        # Below SM100 any backend that serves sinks works (TRITON_ATTN on\n"
        "        # Ampere): the standard Attention layer writes its own KV cache.\n"
        "        self.stock_attention = type(self.attn.impl).__name__ != \"FlashInferImpl\"\n"
        "        if self.stock_attention:\n"
        "            assert get_current_vllm_config().model_config.dtype == torch.bfloat16\n"
        "            assert self.attn.kv_cache_dtype in (\"auto\", \"bfloat16\")\n"
        "            self.flint_decode = False\n"
        "            return\n"
        "        # The fused kernel writes the KV cache itself (in the FlashInfer layout)\n",
    ),
    (  # forward: norm and RoPE in Flint's kernel without the cache write, then the standard layer
        "        qkv, _ = self.qkv_proj(hidden_states)\n"
        "        k_cache, v_cache, layer_slots = self._cache_views()\n",
        "        qkv, _ = self.qkv_proj(hidden_states)\n"
        "        if self.stock_attention:\n"
        "            # Returns normalized, rotated q; normalizes and rotates k in place.\n"
        "            q = qk_norm_rope_cache(\n"
        "                qkv,\n"
        "                positions,\n"
        "                self.q_norm.weight,\n"
        "                self.k_norm.weight,\n"
        "                self.q_norm.variance_epsilon,\n"
        "                self._cos_sin(),\n"
        "                None,\n"
        "                None,\n"
        "                None,\n"
        "                self.num_heads,\n"
        "                self.num_kv_heads,\n"
        "                self.head_dim,\n"
        "            )\n"
        "            _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)\n"
        "            output, _ = self.o_proj(self.attn(q, k, v))\n"
        "            return output\n"
        "        k_cache, v_cache, layer_slots = self._cache_views()\n",
    ),
]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src).read()
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"anchor found {n} times, expected once:\n{old}")
        text = text.replace(old, new)
    compile(text, dst, "exec")
    with open(dst, "w") as f:
        f.write(text)
    print(f"patched {src} -> {dst} ({len(EDITS)} edits)")


if __name__ == "__main__":
    main()
