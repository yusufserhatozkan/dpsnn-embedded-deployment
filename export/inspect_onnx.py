"""Inspect an ONNX graph to plan quantization.

Prints:
    1. Op-type histogram (how many of each op are in the graph)
    2. Nodes likely to be part of spike-neuron logic (Sigmoid / Greater /
       Where / Cast / Sub) — these are the candidates to keep in FP32
    3. A suggested CLI snippet for `quantize_int8.py`

Usage:
    python export/inspect_onnx.py --onnx_path export/dpsnn_pretrained.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import onnx


# Op types that carry the threshold / spike logic (the fragile part for INT8).
SPIKE_OP_TYPES = {"Sigmoid", "Greater", "GreaterOrEqual", "Less", "Where", "Cast"}

# Op types that do the heavy multiply-accumulate (safe to quantize).
HEAVY_OP_TYPES = {"Conv", "ConvTranspose", "MatMul", "Gemm"}


def histogram(graph) -> Counter:
    return Counter(node.op_type for node in graph.node)


def list_nodes_by_op(graph, op_types: set[str]):
    return [n for n in graph.node if n.op_type in op_types]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--show_names", action="store_true",
                        help="Print every spike-related node name (long output)")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx_path):
        sys.exit(f"ONNX file not found: {args.onnx_path}")

    proto = onnx.load(args.onnx_path)
    g = proto.graph
    fp32_kb = os.path.getsize(args.onnx_path) / 1024

    print(f"\n=== {args.onnx_path} ({fp32_kb:.1f} KB) ===")
    print(f"Inputs:  {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input]}")
    print(f"Outputs: {[o.name for o in g.output]}")
    print(f"Total nodes: {len(g.node)}")

    hist = histogram(g)
    print("\nOp-type histogram (top 20):")
    for op, count in hist.most_common(20):
        marker = ""
        if op in HEAVY_OP_TYPES:
            marker = "  [HEAVY -- quantize]"
        elif op in SPIKE_OP_TYPES:
            marker = "  [SPIKE -- keep FP32]"
        print(f"    {op:<22} {count:>6d}{marker}")

    heavy_nodes = list_nodes_by_op(g, HEAVY_OP_TYPES)
    spike_nodes = list_nodes_by_op(g, SPIKE_OP_TYPES)

    print(f"\nHeavy ops (quantization targets):  {len(heavy_nodes)}")
    print(f"Spike-logic ops (skip):            {len(spike_nodes)}")

    if args.show_names and spike_nodes:
        print("\nFirst 30 spike-logic node names:")
        for n in spike_nodes[:30]:
            print(f"    {n.op_type:<10} {n.name}")
        if len(spike_nodes) > 30:
            print(f"    ... and {len(spike_nodes) - 30} more")

    print("\n--- Suggested mixed-precision quantization command ---")
    print("python export/quantize_int8.py \\")
    print(f"    --onnx_path {args.onnx_path} \\")
    print("    --hdf5_path data/results/save/test.hdf5 \\")
    int8_out = args.onnx_path.replace(".onnx", "_int8_mixed.onnx")
    print(f"    --output_path {int8_out} \\")
    print("    --op_types_to_quantize Conv MatMul Gemm \\")
    print("    --calibrate_method Entropy \\")
    print("    --n_calib 200")
    print()


if __name__ == "__main__":
    main()
