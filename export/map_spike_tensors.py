"""Map ONNX graph nodes to their quantization role.

For DPSNN / SCNN-only models, nodes fall into three categories:
  SAFE     -- Conv/Gemm whose output feeds a standard activation (ReLU/Sigmoid).
              Fully quantize: weights AND activation outputs.
  SPIKE    -- Conv/Gemm whose output feeds a spike threshold (Greater node) within
              MAX_HOPS graph steps.  These outputs are membrane potentials.
              Strategy 1 (default): exclude entirely (weights stay FP32).
  UNKNOWN  -- Could not classify within MAX_HOPS.  Treat as SPIKE conservatively.

Outputs
-------
  <onnx_path>.spike_map.json   -- JSON with keys:
      safe_nodes  : list of node names to fully quantize
      spike_nodes : list of node names to exclude
      unknown_nodes: list of node names treated conservatively as spike
      stats       : summary counts

Usage
-----
  python export/map_spike_tensors.py --onnx_path export/dpsnn_scnn128.onnx
  python export/map_spike_tensors.py --onnx_path export/dpsnn_scnn128.onnx --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

import onnx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Op types that carry weights -- candidates for quantization
WEIGHT_OP_TYPES = {"Conv", "ConvTranspose", "MatMul", "Gemm"}

# A conv output reaching one of these = it's a membrane potential / spike path.
# Greater/GreaterOrEqual is the spike threshold comparator.
SPIKE_THRESHOLD_OPS = {"Greater", "GreaterOrEqual", "Less", "LessOrEqual"}

# A conv output reaching one of these = standard ANN activation path.
SAFE_ACTIVATION_OPS = {"Relu", "Sigmoid", "Tanh", "PRelu", "LeakyRelu"}

# Ops we "pass through" during BFS (intermediate computation that doesn't
# change the classification of the data flow).
PASSTHROUGH_OPS = {
    "Add", "Sub", "Mul", "Div",
    "Reshape", "Transpose", "Flatten", "Squeeze", "Unsqueeze",
    "Gather", "GatherElements", "GatherND", "Slice", "Concat", "Expand",
    "LayerNormalization", "BatchNormalization", "InstanceNormalization",
    "Clip", "Abs", "Pow", "Sqrt", "Exp", "Log",
    "Identity", "Cast",
    # Overlap-add path in decoder: shape-inference and control-flow ops
    "Shape", "Equal", "If", "Pad", "ScatterElements", "ScatterND",
    "Range", "NonZero", "Compress",
}

# Max BFS hops before giving up and calling it UNKNOWN.
# The decoder goes through ~404 overlap-add steps before reaching the graph
# output, so we need a large limit.  Visited-tensor deduplication keeps this
# from being slow despite the high limit.
MAX_HOPS = 600


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_consumer_map(graph):
    """tensor_name -> list of nodes that consume it as input."""
    consumers: dict[str, list] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def classify_conv(node, consumer_map: dict, graph_output_tensors: set[str]) -> str:
    """Return 'safe', 'spike', or 'unknown' for a single Conv/Gemm node.

    BFS forward from the node's primary output tensor.  Stops when it finds:
      - SPIKE_THRESHOLD_OPS     -> 'spike'  (membrane potential path)
      - SAFE_ACTIVATION_OPS     -> 'safe'   (standard ANN activation path)
      - graph output tensor     -> 'safe'   (final output, no activation needed)

    If BFS exhausts MAX_HOPS without a clear signal -> 'unknown'.
    """
    if not node.output:
        return "unknown"

    start_tensor = node.output[0]

    # Fast-path: if this node's output IS a graph output, it's safe (decoder)
    if start_tensor in graph_output_tensors:
        return "safe"

    queue = deque()
    visited_tensors: set[str] = set()
    queue.append((start_tensor, 0))
    visited_tensors.add(start_tensor)

    while queue:
        tensor, hop = queue.popleft()
        if hop > MAX_HOPS:
            continue

        # If this tensor reaches the graph output -> safe (no spike in path)
        if tensor in graph_output_tensors:
            return "safe"

        for consumer in consumer_map.get(tensor, []):
            op = consumer.op_type

            if op in SPIKE_THRESHOLD_OPS:
                return "spike"

            if op in SAFE_ACTIVATION_OPS:
                return "safe"

            # Where appears after Greater in spike reset gates -> spike path
            if op == "Where":
                return "spike"

            # Pass through: follow all outputs
            if op in PASSTHROUGH_OPS or op in WEIGHT_OP_TYPES:
                for out in consumer.output:
                    if out and out not in visited_tensors:
                        visited_tensors.add(out)
                        queue.append((out, hop + 1))

    return "unknown"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(onnx_path: str, verbose: bool = False) -> dict:
    proto = onnx.load(onnx_path)
    g = proto.graph

    consumer_map = build_consumer_map(g)

    # Set of tensor names that are graph outputs (final outputs, e.g. decoder)
    graph_output_tensors = {o.name for o in g.output}

    weight_nodes = [n for n in g.node if n.op_type in WEIGHT_OP_TYPES]

    safe_nodes: list[str] = []
    spike_nodes: list[str] = []
    unknown_nodes: list[str] = []

    for node in weight_nodes:
        label = classify_conv(node, consumer_map, graph_output_tensors)
        name = node.name or f"<unnamed_{node.op_type}>"
        if label == "safe":
            safe_nodes.append(name)
        elif label == "spike":
            spike_nodes.append(name)
        else:
            unknown_nodes.append(name)

    stats = {
        "total_weight_ops": len(weight_nodes),
        "safe": len(safe_nodes),
        "spike": len(spike_nodes),
        "unknown": len(unknown_nodes),
    }

    result = {
        "onnx_path": onnx_path,
        "safe_nodes": safe_nodes,
        "spike_nodes": spike_nodes,
        "unknown_nodes": unknown_nodes,
        "stats": stats,
    }

    return result


def print_report(result: dict, verbose: bool = False) -> None:
    s = result["stats"]
    print(f"\n=== Spike tensor map: {result['onnx_path']} ===")
    print(f"Total weight-bearing nodes : {s['total_weight_ops']}")
    print(f"  SAFE  (fully quantize)   : {s['safe']}")
    print(f"  SPIKE (exclude / FP32)   : {s['spike']}")
    print(f"  UNKNOWN (treat as spike) : {s['unknown']}")

    exclude_count = s["spike"] + s["unknown"]
    print(f"\nNodes to exclude from quantization: {exclude_count}")
    print(f"Nodes to fully quantize:            {s['safe']}")

    if verbose:
        if result["safe_nodes"]:
            print("\nSAFE nodes (fully quantize):")
            for n in result["safe_nodes"]:
                print(f"    {n}")
        if result["spike_nodes"]:
            print("\nSPIKE nodes (exclude):")
            for n in result["spike_nodes"][:50]:
                print(f"    {n}")
            if len(result["spike_nodes"]) > 50:
                print(f"    ... and {len(result['spike_nodes']) - 50} more")
        if result["unknown_nodes"]:
            print("\nUNKNOWN nodes (treated as spike/excluded):")
            for n in result["unknown_nodes"][:20]:
                print(f"    {n}")
            if len(result["unknown_nodes"]) > 20:
                print(f"    ... and {len(result['unknown_nodes']) - 20} more")

    # Suggested command
    out_path = result["onnx_path"].replace(".onnx", "_int8_corrected.onnx")
    map_path = result["onnx_path"] + ".spike_map.json"
    print(f"\n--- Next step ---")
    print(f"Spike map saved to: {map_path}")
    print(f"Run corrected quantization:")
    print(f"  python export/quantize_int8.py \\")
    print(f"      --onnx_path {result['onnx_path']} \\")
    print(f"      --hdf5_path data/results/save/test.hdf5 \\")
    print(f"      --output_path {out_path} \\")
    print(f"      --spike_map {map_path} \\")
    print(f"      --n_calib 50")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx_path", required=True,
                        help="Path to FP32 ONNX model")
    parser.add_argument("--verbose", action="store_true",
                        help="Print all node names")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx_path):
        sys.exit(f"ONNX file not found: {args.onnx_path}")

    result = analyse(args.onnx_path, verbose=args.verbose)

    # Save JSON
    map_path = args.onnx_path + ".spike_map.json"
    with open(map_path, "w") as f:
        json.dump(result, f, indent=2)

    print_report(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
