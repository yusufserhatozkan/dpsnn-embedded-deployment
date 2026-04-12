"""INT8 static quantization of a speech-enhancement ONNX model.

Uses onnxruntime.quantization.quantize_static with QDQ format and a
calibration dataset drawn from the VoiceBank-DEMAND test HDF5 file.

Usage (from repo root):
    python export/quantize_int8.py \\
        --onnx_path  export/dpsnn_scnn128.onnx \\
        --hdf5_path  data/results/save/test.hdf5 \\
        --output_path export/dpsnn_scnn128_int8.onnx \\
        --n_calib 100

Output files:
    <output_path>              — INT8 QDQ ONNX model
    <output_path>.metrics.txt — calibration-set quality snapshot
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

def quantize(
    onnx_path: str,
    hdf5_path: str,
    output_path: str,
    n_calib: int = 100,
) -> None:
    """Statically quantize an FP32 ONNX model to INT8 QDQ format.

    Steps
    -----
    1. Pre-process the model to insert fake-quant nodes (required by ORT).
    2. Run calibration on n_calib samples to determine quantization ranges.
    3. Write INT8 QDQ model to output_path.
    """
    from onnxruntime.quantization import (
        quantize_static,
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

    # Step 2: calibration data reader
    print(f"Loading calibration data from {hdf5_path} ({n_calib} samples) ...")
    calib_reader = _make_calibration_data_reader(hdf5_path, input_dim, n_calib)

    # Step 3: static quantization
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    print(f"Quantizing -> {output_path} ...")
    quantize_static(
        model_input=pre_path,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=False,          # per-tensor: simpler, more portable
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True},
    )

    # Cleanup temp file
    if os.path.exists(pre_path):
        os.remove(pre_path)

    fp32_kb = os.path.getsize(onnx_path) / 1024
    int8_kb = os.path.getsize(output_path) / 1024
    print(f"\nFP32: {fp32_kb:.1f} KB  →  INT8: {int8_kb:.1f} KB  "
          f"(compression: {fp32_kb/int8_kb:.1f}×)")


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
        description="INT8 static quantization of a speech-enhancement ONNX model")
    parser.add_argument("--onnx_path",   required=True,
                        help="Input FP32 ONNX model")
    parser.add_argument("--hdf5_path",   required=True,
                        help="VoiceBank-DEMAND test HDF5 (data/results/save/test.hdf5)")
    parser.add_argument("--output_path", required=True,
                        help="Output INT8 ONNX model path")
    parser.add_argument("--n_calib",     type=int, default=100,
                        help="Number of calibration samples (default: 100)")
    parser.add_argument("--no_snapshot", action="store_true",
                        help="Skip quality snapshot after quantization")
    args = parser.parse_args()

    quantize(args.onnx_path, args.hdf5_path, args.output_path, args.n_calib)

    if not args.no_snapshot:
        import onnx
        proto = onnx.load(args.onnx_path)
        input_dim = proto.graph.input[0].type.tensor_type.shape.dim[1].dim_value
        quality_snapshot(
            args.onnx_path, args.output_path,
            args.hdf5_path, input_dim, n_samples=50)


if __name__ == "__main__":
    main()
