"""Correct spike-aware INT8 quantization for DPSNN.

Implements the supervisor's method exactly:
  - INT8 weights for ALL Conv/Gemm/ConvTranspose
  - INT8 activations ONLY after ReLU (encoder) and Sigmoid (mask)
  - Membrane potentials (partial sums between Conv and spike threshold) stay FP32
  - Spike outputs (0/1) stay FP32

Key difference from v1/v2 (why those failed):
  onnxruntime quantize_static places output QDQ *between* Conv and ReLU
  (quantizing the partial sum / membrane potential equivalent — WRONG).
  This script places QDQ *after* ReLU/Sigmoid (CORRECT).

Standard INT8 math (IoT lecture Week 5 Part 2, slide 14-16):
    q_A2 = (S_A2 / (S_W2 * S_A1)) * ReLU(q_w2 * q_A1)
    The INT32 partial sum (q_w2 * q_A1) is NOT clipped to INT8.
    Only the ReLU output q_A2 is quantized.

Usage
-----
  python export/quantize_spike_aware_correct.py \\
      --onnx_path   export/dpsnn_scnn128.onnx \\
      --spike_map   export/dpsnn_scnn128.onnx.spike_map.json \\
      --hdf5_path   data/results/save/test.hdf5 \\
      --output_path export/dpsnn_scnn128_int8_correct_v3.onnx \\
      --n_calib 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.preprocess import quant_pre_process
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def build_consumer_map(graph):
    consumers: dict[str, list] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return consumers


def build_producer_map(graph):
    producers: dict[str, object] = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producers[out] = node
    return producers


def find_activation_output_tensors(
    graph, safe_node_names: set[str]
) -> tuple[list[str], list[str]]:
    """Find output tensors of Relu/Sigmoid nodes that directly follow a SAFE Conv.

    Returns (relu_tensors, sigmoid_tensors) as separate lists so each group can
    receive its own quantization strategy:
      - ReLU outputs  → one shared scale computed from global min/max across all time steps
      - Sigmoid outputs → fixed scale 1/255, zp=-128 (output is always in [0, 1])
    """
    producer_map = build_producer_map(graph)
    relu_tensors: list[str] = []
    sigmoid_tensors: list[str] = []

    for node in graph.node:
        if node.op_type not in ('Relu', 'Sigmoid'):
            continue

        if not node.input:
            continue
        inp = node.input[0]
        if inp not in producer_map:
            continue

        producer = producer_map[inp]
        is_safe = False

        # Direct: Conv → Relu/Sigmoid
        if producer.op_type in ('Conv', 'ConvTranspose', 'Gemm'):
            if (producer.name or '') in safe_node_names:
                is_safe = True

        # One hop: Conv → Add (bias) → Relu/Sigmoid
        elif producer.op_type == 'Add':
            for add_inp in producer.input:
                if not add_inp or add_inp not in producer_map:
                    continue
                grandparent = producer_map[add_inp]
                if grandparent.op_type in ('Conv', 'ConvTranspose', 'Gemm'):
                    if (grandparent.name or '') in safe_node_names:
                        is_safe = True
                        break

        if is_safe and node.output:
            if node.op_type == 'Relu':
                relu_tensors.append(node.output[0])
            else:
                sigmoid_tensors.append(node.output[0])

    return list(set(relu_tensors)), list(set(sigmoid_tensors))


def find_unique_weight_names(graph, node_names_to_quantize: set[str]) -> dict[str, str]:
    """Return {weight_initializer_name: op_type} for Conv nodes in the given set.

    Uses a set to deduplicate — shared weights (used across all 399 unrolled time
    steps) appear only once even though 399+ nodes reference them.
    """
    weight_map: dict[str, str] = {}
    for node in graph.node:
        if node.op_type not in ('Conv', 'ConvTranspose', 'Gemm'):
            continue
        if (node.name or '') not in node_names_to_quantize:
            continue
        if len(node.input) >= 2 and node.input[1]:
            weight_map[node.input[1]] = node.op_type
    return weight_map


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(model_path: str, target_tensors: list[str],
              hdf5_path: str, n_samples: int) -> dict[str, tuple[float, float]]:
    """Run inference on n_samples and collect min/max for each target tensor.

    Returns {tensor_name: (min_val, max_val)}.
    """
    # Add target tensors as model outputs for calibration
    model = onnx.load(model_path)
    existing_outputs = {o.name for o in model.graph.output}
    for t in target_tensors:
        if t not in existing_outputs:
            model.graph.output.append(
                helper.make_tensor_value_info(t, TensorProto.FLOAT, None)
            )

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        tmp_path = f.name
    onnx.save(model, tmp_path)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(tmp_path, sess_options=opts)
    output_names = [o.name for o in sess.get_outputs()]
    input_name = sess.get_inputs()[0].name

    # Read input_dim from model
    input_shape = model.graph.input[0].type.tensor_type.shape
    input_dim = input_shape.dim[1].dim_value

    stats: dict[str, list] = {t: [] for t in target_tensors}

    with h5py.File(hdf5_path, 'r') as f_hdf5:
        for i in range(n_samples):
            audio = f_hdf5[str(i)]['noisy'][()].astype(np.float32).squeeze()
            chunk = audio[:input_dim][np.newaxis, :]
            outputs = sess.run(None, {input_name: chunk})
            for name, val in zip(output_names, outputs):
                if name in stats:
                    stats[name].append((float(val.min()), float(val.max())))

    os.unlink(tmp_path)

    result = {}
    for t in target_tensors:
        if stats[t]:
            min_val = min(s[0] for s in stats[t])
            max_val = max(s[1] for s in stats[t])
            result[t] = (min_val, max_val)
        else:
            result[t] = (0.0, 1.0)

    return result


# ---------------------------------------------------------------------------
# Quantization scale helpers
# ---------------------------------------------------------------------------

def compute_asymmetric_scale_zp(min_val: float, max_val: float,
                                 qmin: int = -128, qmax: int = 127):
    """Compute INT8 asymmetric scale and zero_point from observed range."""
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    scale = (max_val - min_val) / (qmax - qmin)
    if scale == 0:
        scale = 1e-8
    zero_point = int(round(qmin - min_val / scale))
    zero_point = int(np.clip(zero_point, qmin, qmax))
    return np.float32(scale), np.int8(zero_point)


def compute_per_channel_scale_zp(weight: np.ndarray, axis: int = 0):
    """Per-channel symmetric INT8 quantization for Conv weights."""
    n_channels = weight.shape[axis]
    scales = np.zeros(n_channels, dtype=np.float32)
    zero_points = np.zeros(n_channels, dtype=np.int8)

    for i in range(n_channels):
        ch = np.take(weight, i, axis=axis)
        max_abs = np.max(np.abs(ch))
        scales[i] = max_abs / 127.0 if max_abs > 0 else 1e-8

    q_weight = np.round(weight / scales.reshape(
        [-1] + [1] * (weight.ndim - 1)
    )).clip(-128, 127).astype(np.int8)

    return q_weight, scales, zero_points


# ---------------------------------------------------------------------------
# ONNX graph modification
# ---------------------------------------------------------------------------

def insert_int16_sim_after_tensor(graph, tensor_name: str, scale: float,
                                   suffix: str) -> str:
    """Simulate INT16 activation quantization via Div → Round → Clip → Mul.

    Compatible with ONNX opset 13 (QuantizeLinear only supports int8 in opset 13;
    INT16 native support requires opset 21). Produces identical quantization noise
    to true INT16 with scale = max_val / 32767 and zero_point = 0.

    Clip range [0, 32767] is appropriate for non-negative activations (ReLU, Sigmoid).
    """
    div_out   = f'{tensor_name}_div16_{suffix}'
    round_out = f'{tensor_name}_rnd16_{suffix}'
    clip_out  = f'{tensor_name}_clp16_{suffix}'
    mul_out   = f'{tensor_name}_dq16_{suffix}'
    scale_name = f'{tensor_name}_s16_{suffix}'
    cmin_name  = f'{tensor_name}_cmin_{suffix}'
    cmax_name  = f'{tensor_name}_cmax_{suffix}'

    graph.initializer.extend([
        numpy_helper.from_array(np.array(scale,      dtype=np.float32), name=scale_name),
        numpy_helper.from_array(np.array(0.0,        dtype=np.float32), name=cmin_name),
        numpy_helper.from_array(np.array(32767.0,    dtype=np.float32), name=cmax_name),
    ])

    div_node = helper.make_node('Div',   [tensor_name, scale_name],     [div_out],
                                name=f'Div16_{tensor_name}_{suffix}')
    graph.node.extend([
        div_node,
        helper.make_node('Round', [div_out],                             [round_out],
                         name=f'Round16_{tensor_name}_{suffix}'),
        helper.make_node('Clip',  [round_out, cmin_name, cmax_name],    [clip_out],
                         name=f'Clip16_{tensor_name}_{suffix}'),
        helper.make_node('Mul',   [clip_out, scale_name],               [mul_out],
                         name=f'Mul16_{tensor_name}_{suffix}'),
    ])

    div_node_name = div_node.name
    for node in graph.node:
        if node.name == div_node_name:
            continue
        for j, inp in enumerate(node.input):
            if inp == tensor_name:
                node.input[j] = mul_out

    return mul_out


def insert_qdq_after_tensor(graph, tensor_name: str, scale: float, zp: int,
                             suffix: str) -> str:
    """Insert a Q→DQ pair after `tensor_name`.

    All nodes that consumed `tensor_name` now consume the DQ output instead.
    Returns the new DQ output tensor name.
    """
    q_output = f'{tensor_name}_quantized_{suffix}'
    dq_output = f'{tensor_name}_dequantized_{suffix}'
    scale_name = f'{tensor_name}_scale_{suffix}'
    zp_name = f'{tensor_name}_zp_{suffix}'

    # Add scale and zero_point as initializers
    scale_tensor = numpy_helper.from_array(
        np.array(scale, dtype=np.float32), name=scale_name)
    zp_tensor = numpy_helper.from_array(
        np.array(zp, dtype=np.int8), name=zp_name)
    graph.initializer.extend([scale_tensor, zp_tensor])

    # QuantizeLinear node
    q_node = helper.make_node(
        'QuantizeLinear',
        inputs=[tensor_name, scale_name, zp_name],
        outputs=[q_output],
        name=f'Q_{tensor_name}_{suffix}',
        axis=None,
    )

    # DequantizeLinear node
    dq_node = helper.make_node(
        'DequantizeLinear',
        inputs=[q_output, scale_name, zp_name],
        outputs=[dq_output],
        name=f'DQ_{tensor_name}_{suffix}',
        axis=None,
    )

    graph.node.extend([q_node, dq_node])

    # Redirect consumers: replace tensor_name → dq_output
    # (skip the Q node itself which has tensor_name as input)
    q_node_name = q_node.name
    for node in graph.node:
        if node.name == q_node_name:
            continue
        for j, inp in enumerate(node.input):
            if inp == tensor_name:
                node.input[j] = dq_output

    return dq_output


def quantize_conv_weight_in_graph(graph, weight_name: str, op_type: str) -> bool:
    """Replace Conv weight initializer with INT8 + DequantizeLinear.

    The DequantizeLinear outputs the ORIGINAL weight name, so all Conv nodes
    that share this weight (e.g., all 399 unrolled encoder time steps) are
    updated automatically — no per-node input patching needed.
    """
    init_by_name = {init.name: init for init in graph.initializer}
    if weight_name not in init_by_name:
        return False  # not a static initializer (e.g. dynamic/external)

    weight = numpy_helper.to_array(init_by_name[weight_name]).copy()

    # Determine per-channel axis (0 for Conv/Gemm, 1 for ConvTranspose)
    axis = 1 if op_type == 'ConvTranspose' else 0
    q_weight, scales, zero_points = compute_per_channel_scale_zp(weight, axis=axis)

    int8_name = weight_name + '_int8'
    scale_name = weight_name + '_w_scale'
    zp_name    = weight_name + '_w_zp'

    # Remove original FP32 initializer
    to_remove = [i for i, init in enumerate(graph.initializer) if init.name == weight_name]
    for idx in sorted(to_remove, reverse=True):
        del graph.initializer[idx]

    graph.initializer.extend([
        numpy_helper.from_array(q_weight,     name=int8_name),
        numpy_helper.from_array(scales,        name=scale_name),
        numpy_helper.from_array(zero_points,   name=zp_name),
    ])

    # DequantizeLinear outputs the ORIGINAL weight name — all referencing Conv
    # nodes still see the same input name (no node edits required).
    dq_node = helper.make_node(
        'DequantizeLinear',
        inputs=[int8_name, scale_name, zp_name],
        outputs=[weight_name],
        name=f'DQ_weight_{weight_name.replace("/", "_").replace(".", "_")}',
        axis=axis,
    )
    graph.node.append(dq_node)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def quantize_spike_aware_correct(
    onnx_path: str,
    spike_map_path: str,
    hdf5_path: str,
    output_path: str,
    n_calib: int = 50,
    quantize_spike_weights: bool = True,
    relu_percentile: float = 100.0,
    activation_bits: int = 8,
) -> None:
    """Apply the supervisor's correct spike-aware quantization.

    Strategy:
    - INT8 weights for ALL Conv (SAFE + SPIKE paths if quantize_spike_weights=True)
    - Activations ONLY at ReLU output (encoder) and Sigmoid output (mask)
    - Partial sums (between Conv and ReLU/Sigmoid) stay FP32
    - Membrane potentials and spike outputs stay FP32

    activation_bits=8  → INT8 QDQ (QuantizeLinear/DequantizeLinear, opset 13)
    activation_bits=16 → INT16-simulated via Div→Round→Clip→Mul (opset 13 compatible)
                         scale = calibrated_max / 32767, zero_point = 0

    relu_percentile controls the shared ReLU scale upper bound.
    100.0 = absolute max (default, no clipping).
    Lower values (e.g. 95.0) clip extreme outlier time steps so that typical
    frames get finer quantization resolution, at the cost of saturating rare peaks.
    """
    # ── Load spike map ──────────────────────────────────────────────────────
    with open(spike_map_path) as f:
        spike_data = json.load(f)
    safe_node_names = set(spike_data['safe_nodes'])
    spike_node_names = set(spike_data['spike_nodes'] + spike_data.get('unknown_nodes', []))
    all_weight_nodes = safe_node_names | spike_node_names

    print(f"SAFE nodes  : {len(safe_node_names)}")
    print(f"SPIKE nodes : {len(spike_node_names)}")

    # ── Pre-process ─────────────────────────────────────────────────────────
    prep_path = onnx_path + '.v3_prep.onnx'
    print("Pre-processing ONNX model...")
    quant_pre_process(onnx_path, prep_path, skip_optimization=False)

    # ── Load model ──────────────────────────────────────────────────────────
    model = onnx.load(prep_path)
    graph = model.graph

    # ── Find ReLU/Sigmoid output tensors of SAFE Conv nodes ─────────────────
    print("Finding ReLU/Sigmoid activation outputs of SAFE Conv nodes...")
    relu_tensors, sigmoid_tensors = find_activation_output_tensors(graph, safe_node_names)
    print(f"  Found {len(relu_tensors)} ReLU tensors, {len(sigmoid_tensors)} Sigmoid tensors")

    if not relu_tensors and not sigmoid_tensors:
        print("WARNING: No activation tensors found. Check that safe_node_names match graph node names.")

    # ── Calibrate ReLU outputs; Sigmoid uses a fixed scale ──────────────────
    # Sigmoid output is always in [0, 1] by definition, so no calibration needed.
    # ReLU outputs are calibrated but share ONE global scale across all 401 time
    # steps — this matches real hardware (X-CUBE-AI uses one scale per layer).
    relu_scale = np.float32(1e-8)
    relu_zp = np.int8(0)
    global_max = 1.0
    if relu_tensors:
        print(f"Calibrating {len(relu_tensors)} ReLU tensors on {n_calib} samples...")
        relu_stats = calibrate(prep_path, relu_tensors, hdf5_path, n_calib)
        all_max_vals = [v[1] for v in relu_stats.values()]
        global_min = min(v[0] for v in relu_stats.values())
        global_max = float(np.percentile(all_max_vals, relu_percentile))
        relu_scale, relu_zp = compute_asymmetric_scale_zp(global_min, global_max)
        abs_max = max(all_max_vals)
        print(f"  Absolute max across all time steps: {abs_max:.4f}")
        print(f"  {relu_percentile}th-percentile max used for scale: {global_max:.4f}")
        print(f"  Shared ReLU scale: {relu_scale:.6f}, zp: {relu_zp}")

    # Fixed scale for Sigmoid (output always in [0, 1])
    # INT8:  maps [-128, 127] → [0, 1], scale = 1/255, zp = -128
    # INT16: maps [0, 32767]  → [0, 1], scale = 1/32767, zp = 0
    sigmoid_scale_int8 = np.float32(1.0 / 255.0)
    sigmoid_zp_int8 = np.int8(-128)
    sigmoid_scale_int16 = float(1.0 / 32767.0)
    if sigmoid_tensors:
        if activation_bits == 16:
            print(f"  Fixed Sigmoid INT16 scale: {sigmoid_scale_int16:.8f}")
        else:
            print(f"  Fixed Sigmoid INT8 scale: {sigmoid_scale_int8:.6f}, zp: {sigmoid_zp_int8}")

    # ── Insert activation quantization nodes ────────────────────────────────
    if activation_bits == 16:
        relu_scale_int16 = float(global_max / 32767.0) if global_max > 0 else 1e-8
        print(f"Inserting INT16-simulated Div→Round→Clip→Mul nodes (scale={relu_scale_int16:.8f})...")
        for i, tensor_name in enumerate(relu_tensors):
            insert_int16_sim_after_tensor(graph, tensor_name, relu_scale_int16,
                                          suffix=f'relu_{i}')
        for i, tensor_name in enumerate(sigmoid_tensors):
            insert_int16_sim_after_tensor(graph, tensor_name, sigmoid_scale_int16,
                                          suffix=f'sig_{i}')
        print(f"  Inserted {len(relu_tensors)} ReLU INT16 sim nodes (shared scale)")
        print(f"  Inserted {len(sigmoid_tensors)} Sigmoid INT16 sim nodes (fixed scale)")
    else:
        print(f"Inserting INT8 Q->DQ nodes...")
        for i, tensor_name in enumerate(relu_tensors):
            insert_qdq_after_tensor(graph, tensor_name, float(relu_scale), int(relu_zp),
                                    suffix=f'relu_{i}')
        for i, tensor_name in enumerate(sigmoid_tensors):
            insert_qdq_after_tensor(graph, tensor_name, float(sigmoid_scale_int8),
                                    int(sigmoid_zp_int8), suffix=f'sig_{i}')
        print(f"  Inserted {len(relu_tensors)} ReLU Q->DQ pairs (shared scale)")
        print(f"  Inserted {len(sigmoid_tensors)} Sigmoid Q->DQ pairs (fixed scale)")

    # ── Quantize Conv weights to INT8 ────────────────────────────────────────
    nodes_to_quantize_weights = all_weight_nodes if quantize_spike_weights else safe_node_names
    print(f"Finding unique weight tensors for {len(nodes_to_quantize_weights)} Conv nodes...")
    weight_map = find_unique_weight_names(graph, nodes_to_quantize_weights)
    print(f"  {len(weight_map)} unique weight initializers found")

    quantized_count = 0
    for weight_name, op_type in weight_map.items():
        try:
            if quantize_conv_weight_in_graph(graph, weight_name, op_type):
                quantized_count += 1
        except Exception as e:
            print(f"  Skipping {weight_name}: {e}")
    print(f"  Quantized {quantized_count} weight tensors")

    # ── Save model ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    onnx.save(model, output_path)

    fp32_kb = os.path.getsize(onnx_path) / 1024
    out_kb = os.path.getsize(output_path) / 1024
    print(f"\nFP32: {fp32_kb:.1f} KB  ->  Corrected INT8: {out_kb:.1f} KB")
    print(f"Saved -> {output_path}")

    # Cleanup
    if os.path.exists(prep_path):
        os.remove(prep_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--onnx_path',   required=True)
    parser.add_argument('--spike_map',   required=True)
    parser.add_argument('--hdf5_path',   required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--n_calib',     type=int, default=50)
    parser.add_argument('--relu_percentile', type=float, default=100.0,
                        help='Percentile of per-time-step max values used to set the shared '
                             'ReLU scale. 100=absolute max (no clipping). Lower values '
                             '(e.g. 95.0) give finer resolution for typical frames at the '
                             'cost of saturating rare peaks.')
    parser.add_argument('--activation_bits', type=int, default=8, choices=[8, 16],
                        help='Bit-width for activation quantization. '
                             '8=INT8 QDQ (QuantizeLinear/DequantizeLinear). '
                             '16=INT16-simulated via Div→Round→Clip→Mul (opset-13 compatible, '
                             'scale=calibrated_max/32767). Weights always INT8.')
    parser.add_argument('--safe_weights_only', action='store_true',
                        help='Only quantize weights of SAFE Conv nodes (spike path stays FP32)')
    args = parser.parse_args()

    quantize_spike_aware_correct(
        onnx_path=args.onnx_path,
        spike_map_path=args.spike_map,
        hdf5_path=args.hdf5_path,
        output_path=args.output_path,
        n_calib=args.n_calib,
        quantize_spike_weights=not args.safe_weights_only,
        relu_percentile=args.relu_percentile,
        activation_bits=args.activation_bits,
    )


if __name__ == '__main__':
    main()
