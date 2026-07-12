"""Full-test-set evaluation of a speech-enhancement ONNX model.

Reproduces the vctk_trainer TestCallback metrics on the VoiceBank-DEMAND
test HDF5 file using ONNX Runtime — no PyTorch required at inference time.

Computes: SI-SNR, PESQ (wb), STOI, composite scores (PESQ, OVRL, SIG, BAK)
for noisy input and enhanced output, reported as averages over all utterances.

Usage (from repo root):
    python evaluation/eval_onnx.py \\
        --onnx_path export/dpsnn_scnn128.onnx \\
        --hdf5_path data/results/save/test.hdf5 \\
        --output_path results/eval_dpsnn_scnn128.txt

    # INT8 comparison:
    python evaluation/eval_onnx.py \\
        --onnx_path export/dpsnn_scnn128_int8.onnx \\
        --hdf5_path data/results/save/test.hdf5 \\
        --output_path results/eval_dpsnn_scnn128_int8.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sisnr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SNR (dB) between 1-D arrays."""
    est = est - est.mean()
    ref = ref - ref.mean()
    dot = np.sum(est * ref)
    proj = dot / (np.sum(ref ** 2) + 1e-8) * ref
    noise = est - proj
    return float(10 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + 1e-8)))


def _normalize(est: np.ndarray) -> np.ndarray:
    """Peak-normalize a 1-D signal to unit amplitude."""
    peak = np.max(np.abs(est))
    return est / peak if peak > 0 else est


def _build_chunks(audio: np.ndarray, input_dim: int, output_size: int) -> np.ndarray:
    """Replicate EvaluationDataset chunking for one utterance.

    Pads audio to a multiple of output_size, prepends context zeros, then
    slices into overlapping windows of length input_dim with hop output_size.

    Returns ndarray of shape (n_chunks, input_dim).
    """
    context_size = input_dim - output_size  # start context (end context = 0)
    audio_length = len(audio)

    # Pad to multiple of output_size
    remainder = audio_length % output_size
    if remainder:
        audio = np.pad(audio, (0, output_size - remainder))

    target_outputs = len(audio)

    # Prepend context zeros (end_context_size = 0 in our config)
    padded = np.pad(audio, (context_size, 0))

    chunks = [
        padded[t:t + input_dim]
        for t in range(0, target_outputs, output_size)
    ]
    return np.stack(chunks).astype(np.float32)   # (n_chunks, input_dim)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    onnx_path: str,
    hdf5_path: str,
    sr: int = 16000,
) -> dict:
    """Run full test-set evaluation and return averaged metrics dict."""
    import onnxruntime as ort
    from pesq import pesq as eval_pesq
    from pystoi import stoi as eval_stoi
    from dpsnn.data.metrics import eval_composite

    # Load ONNX session
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])

    # Infer input_dim and output_size from the model
    input_dim = sess.get_inputs()[0].shape[1]
    # Run one dummy forward to get output_size
    dummy = np.zeros((1, input_dim), dtype=np.float32)
    output_size = sess.run(None, {"noisy_audio": dummy})[0].shape[1]
    print(f"Model: input_dim={input_dim}, output_size={output_size}")

    # Metric accumulators
    n = 0
    noisy_sisnrs, enh_sisnrs = [], []
    noisy_pesqs, enh_pesqs = [], []
    noisy_stois, enh_stois = [], []
    noisy_comp = np.zeros(4)
    enh_comp   = np.zeros(4)

    with h5py.File(hdf5_path, "r") as f:
        total = len(f)
        for idx in range(total):
            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"  [{idx+1}/{total}]", flush=True)

            audio = f[str(idx)]["noisy"][()].astype(np.float32).squeeze()
            clean = f[str(idx)]["clean"][()].astype(np.float32).squeeze()
            audio_length = int(f[str(idx)].attrs["length"])

            # Build chunks and run ONNX inference
            chunks = _build_chunks(audio, input_dim, output_size)     # (n_chunks, input_dim)
            enhanced_chunks = sess.run(None, {"noisy_audio": chunks})[0]  # (n_chunks, output_size)

            # Reconstruct and trim to original length
            enhanced = enhanced_chunks.flatten()[:audio_length]
            noisy_trim = audio[:audio_length]
            clean_trim = clean[:audio_length]

            # Normalize enhanced
            enhanced = _normalize(enhanced)

            # SI-SNR
            noisy_sisnrs.append(_sisnr(noisy_trim, clean_trim))
            enh_sisnrs.append(_sisnr(enhanced, clean_trim))

            # PESQ (wideband)
            try:
                noisy_pesqs.append(eval_pesq(sr, clean_trim, noisy_trim, "wb"))
                enh_pesqs.append(eval_pesq(sr, clean_trim, enhanced, "wb"))
            except Exception:
                pass  # skip files where PESQ can't run (too short etc.)

            # STOI
            try:
                noisy_stois.append(eval_stoi(clean_trim, noisy_trim, sr, extended=False))
                enh_stois.append(eval_stoi(clean_trim, enhanced, sr, extended=False))
            except Exception:
                pass

            # Composite (PESQ, OVRL, SIG, BAK)
            try:
                noisy_comp += np.array(eval_composite(clean_trim, noisy_trim, sr))
                enh_comp   += np.array(eval_composite(clean_trim, enhanced, sr))
            except Exception:
                pass

            n += 1

    noisy_comp /= n
    enh_comp   /= n

    results = {
        "n_utterances": n,
        "noisy_sisnr":  float(np.mean(noisy_sisnrs)),
        "enh_sisnr":    float(np.mean(enh_sisnrs)),
        "noisy_pesq":   float(np.mean(noisy_pesqs)) if noisy_pesqs else float("nan"),
        "enh_pesq":     float(np.mean(enh_pesqs))   if enh_pesqs   else float("nan"),
        "noisy_stoi":   float(np.mean(noisy_stois)) if noisy_stois else float("nan"),
        "enh_stoi":     float(np.mean(enh_stois))   if enh_stois   else float("nan"),
        # composite [pesq, ovrl, sig, bak]
        "noisy_comp_pesq": noisy_comp[0], "noisy_comp_ovrl": noisy_comp[1],
        "noisy_comp_sig":  noisy_comp[2], "noisy_comp_bak":  noisy_comp[3],
        "enh_comp_pesq":   enh_comp[0],   "enh_comp_ovrl":   enh_comp[1],
        "enh_comp_sig":    enh_comp[2],   "enh_comp_bak":    enh_comp[3],
    }
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    n = results["n_utterances"]
    print(f"\n{'='*55}")
    print(f"  Evaluation results  ({n} utterances)")
    print(f"{'='*55}")
    print(f"{'Metric':<22} {'Noisy':>10} {'Enhanced':>10}")
    print(f"{'-'*44}")
    print(f"{'SI-SNR (dB)':<22} {results['noisy_sisnr']:>10.2f} {results['enh_sisnr']:>10.2f}")
    print(f"{'PESQ (wb)':<22} {results['noisy_pesq']:>10.3f} {results['enh_pesq']:>10.3f}")
    print(f"{'STOI':<22} {results['noisy_stoi']:>10.3f} {results['enh_stoi']:>10.3f}")
    print(f"{'Composite PESQ':<22} {results['noisy_comp_pesq']:>10.3f} {results['enh_comp_pesq']:>10.3f}")
    print(f"{'Composite OVRL':<22} {results['noisy_comp_ovrl']:>10.3f} {results['enh_comp_ovrl']:>10.3f}")
    print(f"{'Composite SIG':<22} {results['noisy_comp_sig']:>10.3f} {results['enh_comp_sig']:>10.3f}")
    print(f"{'Composite BAK':<22} {results['noisy_comp_bak']:>10.3f} {results['enh_comp_bak']:>10.3f}")
    print(f"{'='*55}")


def _save_results(results: dict, output_path: str) -> None:
    lines = [f"{k}={v}" for k, v in results.items()]
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Metrics saved -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full test-set ONNX evaluation on VoiceBank-DEMAND")
    parser.add_argument("--onnx_path",   required=True,
                        help="ONNX model to evaluate (FP32 or INT8)")
    parser.add_argument("--hdf5_path",   required=True,
                        help="VoiceBank-DEMAND test HDF5 (data/results/save/test.hdf5)")
    parser.add_argument("--output_path", required=True,
                        help="Path to save metrics text file")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    args = parser.parse_args()

    print(f"Evaluating {args.onnx_path}")
    print(f"Test set:   {args.hdf5_path}")

    results = evaluate(args.onnx_path, args.hdf5_path, sr=args.sr)
    _print_results(results)

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _save_results(results, args.output_path)


if __name__ == "__main__":
    main()
