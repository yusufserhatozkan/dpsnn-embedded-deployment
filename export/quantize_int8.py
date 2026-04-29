"""INT8 static quantization of a speech-enhancement ONNX model.

Uses onnxruntime.quantization.quantize_static with QDQ format and a
calibration dataset drawn from the VoiceBank-DEMAND test HDF5 file.

Default config (matches the "naive INT8" baseline that failed at 6.45 dB):
    per-channel, Percentile (99.999), asymmetric activations, INT8 weights+acts.

To run a mixed-precision experiment (skip spike-neuron ops):
    python export/quantize_int8.py \\
        --onnx_path   export/dpsnn_pretrained.onnx \\
        --hdf5_path   data/results/save/test.hdf5 \\
        --output_path export/dpsnn_pretrained_int8_mixed.onnx \\
        --op_types_to_quantize Conv MatMul Gemm \\
        --calibrate_method Entropy --n_calib 200

To try entropy calibration on the full graph:
    python export/quantize_int8.py ... --calibrate_method Entropy

To do weight-only quantization (acts stay FP32):
    python export/quantize_int8.py ... --weight_only

Output files:
    <output_path>             - INT8 QDQ ONNX model
    <output_path>.metrics.txt - calibration-set quality snapshot
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------

class _CalibrationReader:
    """Reads noisy audio chunks from a VoiceBank-DEMAND HDF5 file.

    Yields numpy arrays of shape (1, input_dim) for use as ONNX calibration
    inputs.  Each audio file is processed as one chunk (zero-padded to
    input_dim if needed).
    """

    def __init__(self, hdf5_path: str, input_dim: int, n_samples: int):
        self.hdf5_path = hdf5_path
        self.input_dim = input_dim
        self.n_samples = n_samples

    def __iter__(self):
        yielded = 0
        with h5py.File(self.hdf5_path, "r") as f:
            for idx in range(len(f)):
                if yielded >= self.n_samples:
                    break
                audio = f[str(idx)]["noisy"][()].astype(np.float32).squeeze()
                # Take first input_dim samples; pad if shorter
                if len(audio) >= self.input_dim:
                    chunk = audio[:self.input_dim]
                else:
                    chunk = np.pad(audio, (0, self.input_dim - len(audio)))
                yield {"noisy_audio": chunk[np.newaxis, :]}   # (1, input_dim)
                yielded += 1


# ---------------------------------------------------------------------------
# ORT-based calibration data provider
# ---------------------------------------------------------------------------

def _make_calibration_data_reader(hdf5_path: str, input_dim: int, n_samples: int):
    """Return an onnxruntime CalibrationDataReader for the given dataset."""
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._iter = iter(_CalibrationReader(hdf5_path, input_dim, n_samples))

        def get_next(self):
            try:
                return next(self._iter)
            except StopIteration:
                return None

    return _Reader()


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

_CALIBRATION_METHODS = {
    "MinMax":       "MinMax",
    "Percentile":   "Percentile",
    "Entropy":      "Entropy",
    "Distribution": "Distribution",
}

_QUANT_TYPES = {
    "QInt8":  "QInt8",
    "QUInt8": "QUInt8",
    "QInt16": "QInt16",
    "QUInt16": "QUInt16",
}


def quantize(
    onnx_path: str,
    hdf5_path: str,
    output_path: str,
    n_calib: int = 100,
    calibrate_method: str = "Percentile",
    per_channel: bool = True,
    activation_symmetric: bool = False,
    activation_type: str = "QInt8",
    weight_type: str = "QInt8",
    percentile: float = 99.999,
    op_types_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
    weight_only: bool = False,
) -> None:
    """Statically quantize an FP32 ONNX model to a low-bit QDQ format.

    Parameters
    ----------
    calibrate_method
        One of {MinMax, Percentile, Entropy, Distribution}. Entropy minimizes
        KL divergence between FP32 and INT8 distributions and is recommended
        for skewed activations (e.g., membrane potentials).
    per_channel
        Per-channel weight quantization. Almost always better than per-tensor.
    activation_symmetric
        If True, activation zero_point=0 (symmetric range). Better for ReLU /
        post-spike outputs that are non-negative. Default False (asymmetric).
    activation_type / weight_type
        QInt8, QUInt8, QInt16, QUInt16. INT16 has 256x more headroom than
        INT8 and is useful as a "less aggressive" mid-point experiment.
    percentile
        Used only when calibrate_method=Percentile. Clips extreme outliers.
    op_types_to_quantize
        If provided, ONLY these op types are quantized. Use this for mixed
        precision: ['Conv', 'MatMul', 'Gemm'] keeps spike-neuron ops in FP32.
    nodes_to_exclude
        Specific node names to skip. Combine with op_types_to_quantize for
        fine-grained control.
    weight_only
        If True, quantize weights only; activations stay FP32. Smallest model
        with full-precision compute.
    """
    from onnxruntime.quantization import (
        quantize_static,
        quantize_dynamic,
        QuantFormat,
        QuantType,
        CalibrationMethod,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Read input_dim from the ONNX model
    import onnx
    proto = onnx.load(onnx_path)
    input_shape = proto.graph.input[0].type.tensor_type.shape
    input_dim = input_shape.dim[1].dim_value
    print(f"ONNX model input_dim: {input_dim}")

    # Step 1: pre-process (shape inference + node fusion)
    pre_path = onnx_path + ".preprocessed.onnx"
    print(f"Pre-processing ONNX model -> {pre_path}")
    quant_pre_process(onnx_path, pre_path, skip_symbolic_shape=True)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Resolve enums from string args
    calib_enum = getattr(CalibrationMethod, calibrate_method)
    weight_enum = getattr(QuantType, weight_type)
    act_enum = getattr(QuantType, activation_type)

    # ---- Weight-only (no calibration needed) ----
    if weight_only:
        print(f"Weight-only quantization (activations stay FP32) -> {output_path} ...")
        quantize_dynamic(
            model_input=pre_path,
            model_output=output_path,
            weight_type=weight_enum,
            per_channel=per_channel,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
        )
    else:
        # ---- Static quantization (with calibration) ----
        print(f"Loading calibration data from {hdf5_path} ({n_calib} samples) ...")
        calib_reader = _make_calibration_data_reader(hdf5_path, input_dim, n_calib)

        extra_options = {"ActivationSymmetric": activation_symmetric}
        if calibrate_method == "Percentile":
            extra_options["percentile"] = percentile

        print(
            f"Quantizing -> {output_path}\n"
            f"  method={calibrate_method}  per_channel={per_channel}  "
            f"act_sym={activation_symmetric}\n"
            f"  weight={weight_type}  activation={activation_type}\n"
            f"  op_types_to_quantize={op_types_to_quantize}\n"
            f"  nodes_to_exclude={'(' + str(len(nodes_to_exclude)) + ' nodes)' if nodes_to_exclude else None}"
        )
        quantize_static(
            model_input=pre_path,
            model_output=output_path,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            activation_type=act_enum,
            weight_type=weight_enum,
            calibrate_method=calib_enum,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            extra_options=extra_options,
        )

    # Cleanup temp file
    if os.path.exists(pre_path):
        os.remove(pre_path)

    fp32_kb = os.path.getsize(onnx_path) / 1024
    int8_kb = os.path.getsize(output_path) / 1024
    ratio = fp32_kb / int8_kb if int8_kb > 0 else float("nan")
    print(f"\nFP32: {fp32_kb:.1f} KB  ->  Quantized: {int8_kb:.1f} KB  "
          f"(compression: {ratio:.2f}x)")


# ---------------------------------------------------------------------------
# Quality snapshot (SI-SNR on calibration set)
# ---------------------------------------------------------------------------

def quality_snapshot(
    fp32_onnx: str,
    int8_onnx: str,
    hdf5_path: str,
    input_dim: int,
    n_samples: int = 50,
) -> None:
    """Print SI-SNR comparison between FP32 and INT8 on a small sample set."""
    import onnxruntime as ort

    def _run(session, audio_chunk):
        return session.run(None, {"noisy_audio": audio_chunk})[0]

    def _sisnr(est, ref):
        est = est - est.mean()
        ref = ref - ref.mean()
        dot = np.sum(est * ref)
        proj = dot / (np.sum(ref ** 2) + 1e-8) * ref
        noise = est - proj
        return 10 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + 1e-8))

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    fp32_sess = ort.InferenceSession(fp32_onnx, sess_options=opts)
    int8_sess = ort.InferenceSession(int8_onnx, sess_options=opts)

    fp32_sisnrs, int8_sisnrs, diff_sisnrs = [], [], []
    for sample in list(_CalibrationReader(hdf5_path, input_dim, n_samples)):
        chunk = sample["noisy_audio"]   # (1, input_dim)
        noisy = chunk[0, -input_dim:]   # reference: raw noisy input

        fp32_out = _run(fp32_sess, chunk)[0]
        int8_out = _run(int8_sess, chunk)[0]

        # Trim to same length as output
        out_len = fp32_out.shape[-1]
        fp32_sisnrs.append(_sisnr(fp32_out, noisy[:out_len]))
        int8_sisnrs.append(_sisnr(int8_out, noisy[:out_len]))
        diff_sisnrs.append(fp32_sisnrs[-1] - int8_sisnrs[-1])

    print(f"\n--- Quality Snapshot ({n_samples} samples) ---")
    print(f"FP32 avg SI-SNR:  {np.mean(fp32_sisnrs):.2f} dB")
    print(f"INT8 avg SI-SNR:  {np.mean(int8_sisnrs):.2f} dB")
    print(f"SI-SNR drop:      {np.mean(diff_sisnrs):.2f} dB  "
          f"(std {np.std(diff_sisnrs):.2f})")
    print(f"Max drop:         {np.max(diff_sisnrs):.2f} dB")

    metrics = (
        f"fp32_sisnr_mean={np.mean(fp32_sisnrs):.4f}\n"
        f"int8_sisnr_mean={np.mean(int8_sisnrs):.4f}\n"
        f"sisnr_drop_mean={np.mean(diff_sisnrs):.4f}\n"
        f"sisnr_drop_std={np.std(diff_sisnrs):.4f}\n"
        f"sisnr_drop_max={np.max(diff_sisnrs):.4f}\n"
        f"n_samples={n_samples}\n"
    )
    metrics_path = int8_onnx + ".metrics.txt"
    with open(metrics_path, "w") as fh:
        fh.write(metrics)
    print(f"Metrics saved -> {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static / weight-only quantization of a speech-enhancement ONNX model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    parser.add_argument("--onnx_path",   required=True, help="Input FP32 ONNX model")
    parser.add_argument("--hdf5_path",   required=False,
                        help="VoiceBank-DEMAND test HDF5 (required unless --weight_only)")
    parser.add_argument("--output_path", required=True, help="Output quantized ONNX model")
    parser.add_argument("--n_calib",     type=int, default=100,
                        help="Number of calibration samples (static only)")
    parser.add_argument("--no_snapshot", action="store_true",
                        help="Skip the post-quantization SI-SNR snapshot")

    # Calibration / number-format knobs
    parser.add_argument("--calibrate_method", default="Percentile",
                        choices=list(_CALIBRATION_METHODS.keys()),
                        help="Calibration strategy")
    parser.add_argument("--percentile", type=float, default=99.999,
                        help="Percentile clip (only used when --calibrate_method=Percentile)")
    parser.add_argument("--no_per_channel", action="store_true",
                        help="Disable per-channel weight quantization (use per-tensor)")
    parser.add_argument("--activation_symmetric", action="store_true",
                        help="Use symmetric activation quantization (zero_point=0)")
    parser.add_argument("--activation_type", default="QInt8",
                        choices=list(_QUANT_TYPES.keys()),
                        help="Activation number format")
    parser.add_argument("--weight_type", default="QInt8",
                        choices=list(_QUANT_TYPES.keys()),
                        help="Weight number format")

    # Spike-aware quantization (Tao's corrected method)
    parser.add_argument("--spike_map", default=None,
                        help="Path to JSON produced by map_spike_tensors.py.  "
                             "Automatically populates --nodes_to_exclude with "
                             "spike-path Conv/Gemm nodes (membrane potential "
                             "inputs and spike outputs stay FP32).")

    # Mixed precision
    parser.add_argument("--op_types_to_quantize", nargs="+", default=None,
                        help="If set, ONLY these op types are quantized "
                             "(e.g. 'Conv MatMul Gemm' for mixed precision)")
    parser.add_argument("--nodes_to_exclude", nargs="+", default=None,
                        help="Specific node names to keep in FP32")

    # Weight-only mode
    parser.add_argument("--weight_only", action="store_true",
                        help="Quantize weights only; activations stay FP32 (no calibration)")

    args = parser.parse_args()

    if not args.weight_only and not args.hdf5_path:
        parser.error("--hdf5_path is required for static quantization "
                     "(omit --hdf5_path only with --weight_only)")

    # Load spike map and build exclusion list
    nodes_to_exclude = args.nodes_to_exclude or []
    if args.spike_map:
        import json
        with open(args.spike_map) as f:
            spike_data = json.load(f)
        spike_exclude = (spike_data.get("spike_nodes", [])
                         + spike_data.get("unknown_nodes", []))
        nodes_to_exclude = list(set(nodes_to_exclude + spike_exclude))
        print(f"Spike map loaded: {len(spike_exclude)} nodes excluded "
              f"(spike-path Conv/Gemm/ConvTranspose stay FP32)")
        print(f"  SAFE nodes to quantize: {spike_data['stats']['safe']}")
        print(f"  SPIKE nodes excluded:   {len(spike_exclude)}")

    quantize(
        onnx_path=args.onnx_path,
        hdf5_path=args.hdf5_path,
        output_path=args.output_path,
        n_calib=args.n_calib,
        calibrate_method=args.calibrate_method,
        per_channel=not args.no_per_channel,
        activation_symmetric=args.activation_symmetric,
        activation_type=args.activation_type,
        weight_type=args.weight_type,
        percentile=args.percentile,
        op_types_to_quantize=args.op_types_to_quantize,
        nodes_to_exclude=nodes_to_exclude if nodes_to_exclude else None,
        weight_only=args.weight_only,
    )

    if not args.no_snapshot and args.hdf5_path:
        import onnx
        proto = onnx.load(args.onnx_path)
        input_dim = proto.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        quality_snapshot(
            args.onnx_path, args.output_path,
            args.hdf5_path, input_dim, n_samples=50)


if __name__ == "__main__":
    main()
