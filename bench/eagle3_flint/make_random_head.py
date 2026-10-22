#!/usr/bin/env python3
"""Write a random-weight EAGLE-3 draft head with Flint's shapes as a Hugging Face checkpoint.

The checkpoint uses the SpecForge / EAGLE `LlamaForCausalLMEagle3` layout (config.json + model.safetensors with
`fc.*`, `midlayer.*`, `norm.*`, `lm_head.*`, `d2t`, `t2d`), which llama.cpp's converter turns into an `eagle3` GGUF:

    python convert_hf_to_gguf.py <out_dir> --target-model-dir <dir with Flint's config.json and tokenizer files> \
        --outtype q8_0 --outfile head.gguf

The weights are random, so its drafts are never accepted; it exists to time the draft step on the target hardware.
Shapes come from the target's config.json (hidden size, heads, head dim, vocabulary), so the same script works for
other targets.

Draft vocabulary:
  --draft-vocab 0       full target vocabulary, own LM head, no d2t mapping
  --draft-vocab N       N-row LM head plus d2t/t2d. The rows are the target's special/added tokens plus the lowest
                        BPE ids (BPE assigns low ids to early, frequent merges, so this approximates a frequency-ranked
                        subset; a trained head would use real token counts from the target's own outputs)
  --no-lm-head          no LM head at all: llama.cpp then borrows the target model's output matrix

Example (Flint, 32k draft vocabulary, FFN 6144):
    uv run --with torch --with safetensors python make_random_head.py --target-dir flint-target \
        --out flint-eagle3-rand-v32k --draft-vocab 32000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-dir", required=True, type=Path, help="folder with the target's config.json and tokenizer.json")
    ap.add_argument("--out", required=True, type=Path, help="output folder for config.json + model.safetensors")
    ap.add_argument("--draft-vocab", type=int, default=32000, help="draft LM head rows; 0 = full target vocabulary")
    ap.add_argument("--no-lm-head", action="store_true", help="omit the LM head (borrow the target's at run time)")
    ap.add_argument("--intermediate-size", type=int, default=None, help="FFN width (default: the target's intermediate_size)")
    ap.add_argument("--kv-heads", type=int, default=None, help="KV heads (default: the target's num_key_value_heads)")
    ap.add_argument("--aux-layers", type=str, default=None,
                    help="comma-separated target layer ids whose *inputs* feed the head (llama.cpp convention); "
                         "default: llama.cpp's [2, L//2, L-3]")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tcfg = json.loads((args.target_dir / "config.json").read_text())
    tcfg = {**tcfg, **tcfg.get("text_config", {})}
    hidden = tcfg["hidden_size"]
    n_head = tcfg["num_attention_heads"]
    head_dim = tcfg.get("head_dim") or hidden // n_head
    n_kv = args.kv_heads or tcfg["num_key_value_heads"]
    ffn = args.intermediate_size or tcfg["intermediate_size"]
    vocab = tcfg["vocab_size"]
    n_layer_tgt = tcfg["num_hidden_layers"]
    eos = tcfg.get("eos_token_id")

    if args.no_lm_head:
        draft_vocab = vocab
    else:
        draft_vocab = args.draft_vocab or vocab

    cfg = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "hidden_size": hidden,
        "intermediate_size": ffn,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "num_hidden_layers": 1,
        "hidden_act": "silu",
        "rms_norm_eps": tcfg.get("rms_norm_eps", 1e-6),
        "max_position_embeddings": tcfg.get("max_position_embeddings", 4096),
        "rope_theta": tcfg.get("rope_theta", 10000.0),
        "rope_scaling": None,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "vocab_size": vocab,
        "draft_vocab_size": draft_vocab,
        "target_hidden_size": hidden,
        "bos_token_id": tcfg.get("bos_token_id"),
        "eos_token_id": eos[0] if isinstance(eos, list) else eos,
        "pad_token_id": tcfg.get("pad_token_id"),
        "torch_dtype": "bfloat16",
    }
    if args.aux_layers:
        cfg["eagle_aux_hidden_state_layer_ids"] = [int(x) for x in args.aux_layers.split(",")]

    g = torch.Generator().manual_seed(args.seed)

    def lin(out_f: int, in_f: int) -> torch.Tensor:
        return (torch.randn(out_f, in_f, generator=g) * 0.02).to(torch.bfloat16)

    def norm(n: int) -> torch.Tensor:
        return torch.ones(n, dtype=torch.bfloat16)

    q_dim, kv_dim = n_head * head_dim, n_kv * head_dim
    t: dict[str, torch.Tensor] = {
        # encoder: three target layers (3 x hidden) -> draft hidden
        "fc.weight": lin(hidden, 3 * hidden),
        # decoder layer: attention sees [norm(token embedding), norm(fused features)] = 2 x hidden
        "midlayer.input_layernorm.weight": norm(hidden),
        "midlayer.hidden_norm.weight": norm(hidden),
        "midlayer.self_attn.q_proj.weight": lin(q_dim, 2 * hidden),
        "midlayer.self_attn.k_proj.weight": lin(kv_dim, 2 * hidden),
        "midlayer.self_attn.v_proj.weight": lin(kv_dim, 2 * hidden),
        "midlayer.self_attn.o_proj.weight": lin(hidden, q_dim),
        "midlayer.post_attention_layernorm.weight": norm(hidden),
        "midlayer.mlp.gate_proj.weight": lin(ffn, hidden),
        "midlayer.mlp.up_proj.weight": lin(ffn, hidden),
        "midlayer.mlp.down_proj.weight": lin(hidden, ffn),
        "norm.weight": norm(hidden),
    }

    if not args.no_lm_head:
        t["lm_head.weight"] = lin(draft_vocab, hidden)

    if draft_vocab < vocab:
        tok = json.loads((args.target_dir / "tokenizer.json").read_text())
        special = sorted({a["id"] for a in tok.get("added_tokens", []) if a["id"] < vocab})
        if len(special) >= draft_vocab:
            raise SystemExit(f"--draft-vocab {draft_vocab} is smaller than the {len(special)} special tokens")
        chosen = set(special)
        i = 0
        while len(chosen) < draft_vocab:
            chosen.add(i)
            i += 1
        ids = torch.tensor(sorted(chosen), dtype=torch.int64)
        # EAGLE/SpecForge convention: d2t holds offsets, target_id = draft_id + d2t[draft_id]
        t["d2t"] = ids - torch.arange(draft_vocab, dtype=torch.int64)
        t2d = torch.zeros(vocab, dtype=torch.bool)
        t2d[ids] = True
        t["t2d"] = t2d

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    save_file(t, str(args.out / "model.safetensors"), metadata={"format": "pt"})

    n_params = sum(v.numel() for k, v in t.items() if v.is_floating_point())
    print(f"wrote {args.out}: {n_params / 1e6:.1f} M parameters, draft vocab {draft_vocab}"
          f"{' (target head borrowed)' if args.no_lm_head else ''}, FFN {ffn}, KV heads {n_kv}, "
          f"target layers {cfg.get('eagle_aux_hidden_state_layer_ids', [2, n_layer_tgt // 2, n_layer_tgt - 3])}")


if __name__ == "__main__":
    main()
