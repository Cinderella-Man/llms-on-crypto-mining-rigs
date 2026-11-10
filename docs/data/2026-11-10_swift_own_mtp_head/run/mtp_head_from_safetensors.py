#!/usr/bin/env python3
"""mtp_head_from_safetensors.py: the MTP head of a Qwen3.8-Flash-Next checkpoint as a GGUF draft, without PyTorch.

    PYTHONPATH=<llama.cpp>/gguf-py python3 mtp_head_from_safetensors.py --model-dir <safetensors folder> \
        --like <an MTP draft GGUF of the same architecture> --out <new.gguf> --name "<general.name>" \
        [--tsv <file>] [--types <file>]

What `convert_hf_to_gguf.py --mtp` does for this architecture (conversion/qwen4exp.py and conversion/qwen.py of
llama.cpp), written out for the 33 tensors an MTP-only file holds, so it runs with numpy and gguf-py alone:

  - the token embeddings and the output head are kept as they are;
  - `mtp.layers.0.*` becomes block <n_layer> (48), `mtp.pre_fc_norm_embedding` / `mtp.pre_fc_norm_hidden` its
    `nextn.enorm` / `nextn.hnorm`, `mtp.hyper_connection_mixer` its `nextn.hc_head_*`;
  - `mtp.fc_embedding` and `mtp.fc_hidden` are joined along the input into `nextn.eh_proj`;
  - the experts' `gate_up_proj` is cut in half along its rows into `ffn_gate_exps` and `ffn_up_exps`;
  - the indexer's `index_qk_proj` is cut into `indexer.q_proj` (the first heads x head size rows) and
    `indexer.k_proj`;
  - every `*norm.weight` is stored plus one (the checkpoint keeps the gains centred on zero);
  - a tensor the reference keeps in F32 (the norms, the router, the shared expert's gate) is written in F32, every
    other one in BF16, the checkpoint's own bytes.

The reference draft (--like) gives the tensor names, their order, their shapes and every key/value pair; each new
tensor must have the reference's shape or the script stops. `general.name` and `general.file_type` are set anew.
llama-quantize then gives the tensors the reference's types (see make_head.sh).
Reads the checkpoint through memory maps; needs about 7 GB of memory for the expert halves.
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np

import gguf

BF16 = gguf.GGMLQuantizationType.BF16
F32 = gguf.GGMLQuantizationType.F32


class Safetensors:
    """The tensors of a sharded safetensors checkpoint as read-only memory maps of their BF16 words."""

    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "model.safetensors.index.json")) as f:
            self.where = json.load(f)["weight_map"]
        self.heads = {}

    def get(self, name):
        shard = self.where[name]
        path = os.path.join(self.folder, shard)
        if shard not in self.heads:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.heads[shard] = (8 + n, json.loads(f.read(n)))
        base, head = self.heads[shard]
        t = head[name]
        if t["dtype"] != "BF16":
            raise SystemExit(f"{name}: dtype {t['dtype']}, this script reads BF16 only")
        a, b = t["data_offsets"]
        m = np.memmap(path, dtype="<u2", mode="r", offset=base + a, shape=tuple(t["shape"]))
        if m.nbytes != b - a:
            raise SystemExit(f"{name}: {b - a} bytes for shape {t['shape']}")
        return m


def rules(blk, hp):
    """GGUF name -> (operation, source tensors). Operations: copy; cat (join along the last axis); rows a b (the
    rows a..b of the first axis); mid a b (a..b of the axis before the last); every *norm.weight gets plus one."""
    layer = "mtp.layers.0."
    n_ff = hp["moe_intermediate_size"]
    n_q = hp["indexer_n_heads"] * hp["indexer_head_dim"]
    b = f"blk.{blk}."
    r = {
        "token_embd.weight": ("copy", ["model.language_model.embed_tokens.weight"]),
        "output.weight": ("copy", ["lm_head.weight"]),
        b + "nextn.eh_proj.weight": ("cat", ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"]),
        b + "nextn.enorm.weight": ("copy", ["mtp.pre_fc_norm_embedding.weight"]),
        b + "nextn.hnorm.weight": ("copy", ["mtp.pre_fc_norm_hidden.weight"]),
        b + "ffn_down_exps.weight": ("copy", [layer + "mlp.experts.down_proj"]),
        b + "ffn_gate_exps.weight": (f"mid 0 {n_ff}", [layer + "mlp.experts.gate_up_proj"]),
        b + "ffn_up_exps.weight": (f"mid {n_ff} {2 * n_ff}", [layer + "mlp.experts.gate_up_proj"]),
        b + "ffn_gate_inp.weight": ("copy", [layer + "mlp.gate.weight"]),
        b + "ffn_gate_inp_shexp.weight": ("copy", [layer + "mlp.shared_expert_gate.weight"]),
        b + "indexer.q_proj.weight": (f"rows 0 {n_q}", [layer + "self_attn.indexer.index_qk_proj.weight"]),
        b + "indexer.k_proj.weight": (f"rows {n_q} -1", [layer + "self_attn.indexer.index_qk_proj.weight"]),
        b + "indexer.q_norm.weight": ("copy", [layer + "self_attn.indexer.q_layernorm.weight"]),
        b + "indexer.k_norm.weight": ("copy", [layer + "self_attn.indexer.k_layernorm.weight"]),
        b + "attn_output.weight": ("copy", [layer + "self_attn.o_proj.weight"]),
    }
    for g, s in (("down", "down_proj"), ("gate", "gate_proj"), ("up", "up_proj")):
        r[b + f"ffn_{g}_shexp.weight"] = ("copy", [layer + f"mlp.shared_expert.{s}.weight"])
    for x in "qkv":
        r[b + f"attn_{x}.weight"] = ("copy", [layer + f"self_attn.{x}_proj.weight"])
    for x in "qk":
        r[b + f"attn_{x}_norm.weight"] = ("copy", [layer + f"self_attn.{x}_norm.weight"])
    for g, s in (("nextn.hc_head", "mtp.hyper_connection_mixer"), ("hc_attn", layer + "attn_hyper_connection"),
                 ("hc_ffn", layer + "mlp_hyper_connection")):
        r[b + g + "_norm.weight"] = ("copy", [s + ".hc_norm.weight"])
        r[b + g + "_down.weight"] = ("copy", [s + ".input_mix_weight_down.weight"])
        r[b + g + "_up.weight"] = ("copy", [s + ".input_mix_weight_up.weight"])
        if not g.startswith("nextn"):
            r[b + g + "_inject.weight"] = ("copy", [s + ".block_inject_weight.weight"])
    return r


def build(op, parts):
    """The BF16 words of a new tensor from its source tensors."""
    w = op.split()
    if w[0] == "copy":
        return parts[0]
    if w[0] == "cat":
        return np.concatenate(parts, axis=-1)
    a, b = int(w[1]), int(w[2])
    if w[0] == "rows":
        return parts[0][a:] if b < 0 else parts[0][a:b]
    if w[0] == "mid":
        return np.ascontiguousarray(parts[0][..., a:b, :])
    raise SystemExit(f"unknown operation {op}")


def to_f32(words):
    """BF16 words as float32, exactly."""
    return (np.asarray(words).astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--like", required=True, help="an MTP draft GGUF of the same architecture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="the new file's general.name")
    ap.add_argument("--tsv", help="write the list of tensors here")
    ap.add_argument("--types", help="write llama-quantize's --tensor-type-file here: the reference's type per tensor")
    a = ap.parse_args()

    with open(os.path.join(a.model_dir, "config.json")) as f:
        cfg = json.load(f)
    hp = {**cfg, **cfg.get("text_config", {})}
    if hp.get("mtp_num_hidden_layers") != 1:
        raise SystemExit(f"mtp_num_hidden_layers is {hp.get('mtp_num_hidden_layers')}, this script handles 1")
    blk = hp["num_hidden_layers"]
    st = Safetensors(a.model_dir)
    todo = rules(blk, hp)
    used = {s for _op, src in todo.values() for s in src}
    kept = {n for n in st.where if n.startswith("mtp.")} | {"lm_head.weight", "model.language_model.embed_tokens.weight"}
    if used != kept:
        raise SystemExit(f"tensors without a rule: {sorted(kept - used)}; rules without a tensor: {sorted(used - kept)}")

    ref = gguf.GGUFReader(a.like)
    names = [t.name for t in ref.tensors]
    if sorted(names) != sorted(todo):
        raise SystemExit(f"the reference and the rules name other tensors: {sorted(set(names) ^ set(todo))}")
    arch = ref.fields[gguf.Keys.General.ARCHITECTURE].contents()
    out = gguf.GGUFWriter(a.out, arch)
    for field in ref.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == gguf.Keys.General.NAME:
            val = a.name
        elif field.name == gguf.Keys.General.FILE_TYPE:
            val = int(gguf.LlamaFileType.MOSTLY_BF16)
        out.add_key_value(field.name, val, vtype, sub_type=sub)

    lines = ["tensor\ttype\tshape\toperation\tplus_one\tfrom"]
    for t in ref.tensors:
        op, src = todo[t.name]
        data = build(op, [st.get(s) for s in src])
        want = tuple(int(x) for x in reversed(t.shape))  # the reference's shape in the checkpoint's axis order
        if data.shape != want:
            if tuple(x for x in data.shape if x != 1) != tuple(x for x in want if x != 1):
                raise SystemExit(f"{t.name}: shape {data.shape} from {src}, the reference has {want}")
            data = data.reshape(want)
        plus = t.name.endswith("norm.weight")
        if t.tensor_type == F32:
            data = to_f32(data)
            if plus:
                data = data + np.float32(1)
            out.add_tensor(t.name, data, raw_dtype=F32)
        else:
            if plus:
                raise SystemExit(f"{t.name}: a norm the reference does not keep in F32")
            out.add_tensor(t.name, np.ascontiguousarray(data), raw_dtype=BF16)
        lines.append(f"{t.name}\t{'F32' if t.tensor_type == F32 else 'BF16'}\t{list(int(x) for x in t.shape)}\t{op}\t"
                     f"{int(plus)}\t{' + '.join(src)}")
    out.write_header_to_file()
    out.write_kv_data_to_file()
    out.write_tensors_to_file()
    out.close()
    if a.types:
        with open(a.types, "w") as f:
            for t in ref.tensors:
                if t.tensor_type != F32:
                    f.write(f"^{re.escape(t.name)}$={t.tensor_type.name.lower()}\n")
    text = "\n".join(lines) + "\n"
    if a.tsv:
        with open(a.tsv, "w") as f:
            f.write(text)
    sys.stdout.write(text)
    print(f"wrote {a.out}: {len(names)} tensors, {os.path.getsize(a.out):,} bytes")


if __name__ == "__main__":
    main()
