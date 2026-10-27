"""A one-tensor GGUF holding Ornith-1.5-122B's output layer in Q5_0, made from the Q8_0 reference's output layer
(dequantized, then gguf-py's Q5_0 quantizer, no importance matrix), for compose_gguf.py to swap into a rig file
(uzu lesson 3: the output layer is read every token; Q5_0 is a fast format on Pascal).

usage: python head_q5_0.py REF_Q8_0.gguf OUT.gguf      (needs numpy and gguf-py on PYTHONPATH)
"""
import sys

import numpy as np
import gguf
from gguf.quants import dequantize, quantize


def main():
    src, out = sys.argv[1:]
    r = gguf.GGUFReader(src)
    t = next(x for x in r.tensors if x.name == "output.weight")
    w = dequantize(t.data, t.tensor_type).astype(np.float32)
    shape = [int(s) for s in reversed(t.shape)]          # rows, columns
    w = w.reshape(shape)
    q = quantize(w, gguf.GGMLQuantizationType.Q5_0)
    back = dequantize(q, gguf.GGMLQuantizationType.Q5_0).reshape(shape)
    rel = float(np.sqrt(((back - w) ** 2).mean() / (w ** 2).mean()))
    wr = gguf.GGUFWriter(out, arch=r.get_field(gguf.Keys.General.ARCHITECTURE).contents())
    wr.add_tensor("output.weight", q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.Q5_0)
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    print(f"output.weight {shape} {t.tensor_type.name} -> Q5_0: {q.nbytes / 1e6:.0f} MB (was {t.n_bytes / 1e6:.0f}), "
          f"relative rms error {rel:.4f}")


if __name__ == "__main__":
    main()
