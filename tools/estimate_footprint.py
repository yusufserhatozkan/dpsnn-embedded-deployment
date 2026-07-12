"""Estimate Flash and RAM footprint of an ONNX model for STM32 deployment.

Analyses the ONNX graph to count parameters (Flash) and estimate peak
activation buffer (RAM).  These are approximations — X-CUBE-AI's Analyse
step gives the authoritative numbers, but this script is useful for a
quick sanity check before generating C code.

STM32 B-U585I-IOT02A limits:
  Flash : 2,048 KB  (used for weights + code)
  RAM   :   786 KB  (used for activations + stack)

Usage (from repo root):
    python tools/estimate_footprint.py export/dpsnn_scnn128.onnx
    python tools/estimate_footprint.py export/dpsnn_scnn128_int8.onnx
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


STM32_FLASH_KB = 2048
STM32_RAM_KB   = 786


def analyse_onnx(onnx_path: str) -> None:
    import onnx
    from onnx import numpy_helper, TensorProto

    proto = onnx.load(onnx_path)
    file_kb = os.path.getsize(onnx_path) / 1024

    # ---- Parameter count & weight bytes --------------------------------
    weight_bytes = 0
    n_params = 0
    dtype_names = {
        TensorProto.FLOAT:  ("float32", 4),
        TensorProto.INT8:   ("int8",    1),
        TensorProto.UINT8:  ("uint8",   1),
        TensorProto.INT32:  ("int32",   4),
        TensorProto.INT64:  ("int64",   8),
        TensorProto.FLOAT16:("float16", 2),
    }
    dtype_counts: dict[str, int] = {}

    for init in proto.graph.initializer:
        t = numpy_helper.to_array(init)
        n_params += t.size
        nbytes = t.size * t.itemsize
        weight_bytes += nbytes
        dname = t.dtype.name
        dtype_counts[dname] = dtype_counts.get(dname, 0) + nbytes

    # ---- Input / output shapes ----------------------------------------
    def _dims(type_proto):
        s = type_proto.tensor_type.shape
        return [d.dim_value if d.dim_value else d.dim_param for d in s.dim]

    inp_shape = _dims(proto.graph.input[0].type)
    out_shape = _dims(proto.graph.output[0].type)

    # ---- Peak activation RAM estimate ---------------------------------
    # Conservative upper bound: sum of largest tensor shape seen in the
    # graph's value_info (intermediate activations).  X-CUBE-AI will
    # overlap buffers, so the actual footprint is lower, but this tells
    # you the order of magnitude.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])
        input_dim = sess.get_inputs()[0].shape[1]
        if isinstance(input_dim, str):
            input_dim = inp_shape[1]   # fall back to proto value
        dummy = np.zeros((1, input_dim), dtype=np.float32)
        out = sess.run(None, {"noisy_audio": dummy})[0]
        output_size = out.shape[1]
    except Exception:
        input_dim  = inp_shape[1] if len(inp_shape) > 1 else "?"
        output_size = "?"

    # Heuristic activation RAM: input + output + 2× largest intermediate
    # (rough; correct only for linear graphs without branching)
    if isinstance(input_dim, int) and isinstance(output_size, int):
        act_bytes = (input_dim + output_size) * 4   # FP32 io tensors
        # Add a rough intermediate buffer estimate based on hidden dim from
        # the first linear layer weight shape, if available
        act_kb_estimate = act_bytes / 1024
    else:
        act_kb_estimate = None

    # ---- Report --------------------------------------------------------
    print(f"\n{'='*58}")
    print(f"  ONNX Footprint Estimate:  {os.path.basename(onnx_path)}")
    print(f"{'='*58}")
    print(f"  File size           : {file_kb:>8.1f} KB")
    print(f"  Parameters          : {n_params:>8,}")
    print(f"  Weight bytes total  : {weight_bytes:>8,} B  ({weight_bytes/1024:.1f} KB)")
    print(f"  Dtype breakdown     :")
    for dname, nbytes in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {dname:<12} {nbytes:>8,} B  ({nbytes/1024:.1f} KB)")
    print(f"\n  Input shape         : {inp_shape}")
    print(f"  Output shape        : {out_shape}")
    if isinstance(input_dim, int):
        print(f"  Input samples       : {input_dim}")
        print(f"  Output samples      : {output_size}")
    if act_kb_estimate is not None:
        print(f"\n  I/O activation est. : {act_kb_estimate:.1f} KB  (IO tensors only)")
        print(f"  (peak activation RAM incl. intermediates is higher; use")
        print(f"   X-CUBE-AI Analyse for the authoritative figure)")

    print(f"\n  STM32 B-U585I-IOT02A limits:")
    print(f"    Flash  : {STM32_FLASH_KB} KB")
    print(f"    RAM    : {STM32_RAM_KB} KB")

    weight_kb = weight_bytes / 1024
    flash_ok = weight_kb < STM32_FLASH_KB * 0.8   # leave 20% for code
    ram_note = "(add ~200 KB code/stack budget on top)"

    flash_flag = "OK " if flash_ok else "EXCEEDS LIMIT"
    print(f"\n  Weight fit in Flash : {weight_kb:>6.1f} KB  [{flash_flag}]")
    if act_kb_estimate:
        print(f"  IO tensors in RAM   : {act_kb_estimate:>6.1f} KB  {ram_note}")
    print(f"{'='*58}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate STM32 Flash/RAM footprint from ONNX model")
    parser.add_argument("onnx_path", help="ONNX model file to analyse")
    args = parser.parse_args()
    analyse_onnx(args.onnx_path)


if __name__ == "__main__":
    main()
