"""Evaluate the streaming StreamSpikeNet on the full VoiceBank-DEMAND test set.

Simulates exactly what the MCU does: processes one 80-sample frame at a time,
carries recurrent state across frames, assembles 40-sample output chunks.

Usage:
    python evaluation/eval_streaming.py \\
        --ckpt_path egs/voicebank/lightning_logs/version_2/checkpoints/<best>.ckpt \\
        --hdf5_path data/results/save/test.hdf5 \\
        --output_path results/eval_exp7_streaming.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export.export_to_onnx import load_from_checkpoint
from export.export_streaming import StreamingWrapper


def _sisnr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    dot = np.sum(est * ref)
    proj = dot / (np.sum(ref ** 2) + 1e-8) * ref
    noise = est - proj
    return float(10 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + 1e-8)))


def _normalize(est: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(est))
    return est / peak if peak > 0 else est


def run_streaming_chunk(wrapper: StreamingWrapper, chunk: np.ndarray) -> np.ndarray:
    """Run streaming inference on one fixed-length chunk (input_dim samples).

    Matches the batch model exactly: processes feature_steps frames with warmup
    reset at frame context_step, produces (time_steps * stride + stride) = 16000
    enhanced samples.  State is reset between chunks (same as batch model).
    """
    L = wrapper.L
    stride = wrapper.stride
    context_step = wrapper.context_step
    B = wrapper.srnn_readout.output_dim
    feature_steps = (len(chunk) - L) // stride + 1

    audio_t = torch.from_numpy(chunk).float()

    context_win = torch.zeros(1, B, context_step)
    v_plif      = torch.zeros(1, B, 1)
    mem_readout = torch.zeros(1, B)
    ola_tail    = torch.zeros(1, stride)

    out_chunks = []
    with torch.no_grad():
        for t in range(feature_steps):
            frame = audio_t[t * stride: t * stride + L].unsqueeze(0)
            enhanced, context_win, v_plif, mem_readout, ola_tail = wrapper(
                frame, context_win, v_plif, mem_readout, ola_tail)

            # After context_step warmup frames, reset membrane + OLA state.
            # Matches the batch model which initialises those to zero at local_t=0.
            if t == context_step - 1:
                v_plif      = torch.zeros(1, B, 1)
                mem_readout = torch.zeros(1, B)
                ola_tail    = torch.zeros(1, stride)

            if t >= context_step:
                out_chunks.append(enhanced.squeeze(0).numpy())

        # Flush the final OLA tail (last frame's non-overlapping half).
        out_chunks.append(ola_tail.squeeze(0).numpy())

    return np.concatenate(out_chunks)  # matches batch model output size


def _build_chunks(audio: np.ndarray, input_dim: int, output_size: int) -> np.ndarray:
    """Identical to eval_onnx._build_chunks — splits variable-length utterances."""
    context_size = input_dim - output_size
    remainder = len(audio) % output_size
    if remainder:
        audio = np.pad(audio, (0, output_size - remainder))
    target_outputs = len(audio)
    padded = np.pad(audio, (context_size, 0))
    chunks = [padded[t:t + input_dim] for t in range(0, target_outputs, output_size)]
    return np.stack(chunks).astype(np.float32)


def evaluate(ckpt_path: str, hdf5_path: str, sr: int = 16000) -> dict:
    from pesq import pesq as eval_pesq
    from pystoi import stoi as eval_stoi
    from dpsnn.data.metrics import eval_composite

    print(f"Loading checkpoint: {ckpt_path}")
    model = load_from_checkpoint(ckpt_path)
    model.eval()
    if getattr(model, "no_pointwise", False):
        # Exp 12 checkpoints have no proj/mask modules; use the conf wrapper
        # in the architecture-matching configuration.
        from export.export_streaming_conf import ConfStreamingWrapper
        wrapper = ConfStreamingWrapper(model, no_binarization=True,
                                       no_pointwise=True)
    else:
        wrapper = StreamingWrapper(model)
    wrapper.eval()

    input_dim   = model.hparams["input_dim"]                           # 16160
    output_size = (model.time_steps - 1) * model.stride + model.L     # 16000

    print(f"Model: input_dim={input_dim}, L={model.L}, stride={model.stride}, "
          f"context_step={model.context_step}, time_steps={model.time_steps}")
    print(f"Evaluating on {hdf5_path} ...")

    n = 0
    noisy_sisnrs, enh_sisnrs = [], []
    noisy_pesqs,  enh_pesqs  = [], []
    noisy_stois,  enh_stois  = [], []
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

            # Split utterance into fixed 1-second chunks — identical to batch eval.
            chunks = _build_chunks(audio, input_dim, output_size)  # (n_chunks, input_dim)
            enhanced_chunks = np.stack([
                run_streaming_chunk(wrapper, c) for c in chunks
            ])  # (n_chunks, output_size)

            enhanced = enhanced_chunks.flatten()[:audio_length]
            noisy_trim = audio[:audio_length]
            clean_trim = clean[:audio_length]

            if len(enhanced) < audio_length:
                enhanced = np.pad(enhanced, (0, audio_length - len(enhanced)))

            enhanced = _normalize(enhanced)

            noisy_sisnrs.append(_sisnr(noisy_trim, clean_trim))
            enh_sisnrs.append(_sisnr(enhanced, clean_trim))

            try:
                noisy_pesqs.append(eval_pesq(sr, clean_trim, noisy_trim, "wb"))
                enh_pesqs.append(eval_pesq(sr, clean_trim, enhanced, "wb"))
            except Exception:
                pass

            try:
                noisy_stois.append(eval_stoi(clean_trim, noisy_trim, sr, extended=False))
                enh_stois.append(eval_stoi(clean_trim, enhanced, sr, extended=False))
            except Exception:
                pass

            try:
                noisy_comp += np.array(eval_composite(clean_trim, noisy_trim, sr))
                enh_comp   += np.array(eval_composite(clean_trim, enhanced, sr))
            except Exception:
                pass

            n += 1

    noisy_comp /= n
    enh_comp   /= n

    return {
        "n_utterances": n,
        "noisy_sisnr":  float(np.mean(noisy_sisnrs)),
        "enh_sisnr":    float(np.mean(enh_sisnrs)),
        "noisy_pesq":   float(np.mean(noisy_pesqs)) if noisy_pesqs else float("nan"),
        "enh_pesq":     float(np.mean(enh_pesqs))   if enh_pesqs   else float("nan"),
        "noisy_stoi":   float(np.mean(noisy_stois)) if noisy_stois else float("nan"),
        "enh_stoi":     float(np.mean(enh_stois))   if enh_stois   else float("nan"),
        "noisy_comp_pesq": noisy_comp[0], "noisy_comp_ovrl": noisy_comp[1],
        "noisy_comp_sig":  noisy_comp[2], "noisy_comp_bak":  noisy_comp[3],
        "enh_comp_pesq":   enh_comp[0],   "enh_comp_ovrl":   enh_comp[1],
        "enh_comp_sig":    enh_comp[2],   "enh_comp_bak":    enh_comp[3],
    }


def _print_results(results: dict) -> None:
    n = results["n_utterances"]
    print(f"\n{'='*55}")
    print(f"  Streaming eval  ({n} utterances)")
    print(f"{'='*55}")
    print(f"{'Metric':<22} {'Noisy':>10} {'Enhanced':>10}")
    print(f"{'-'*44}")
    print(f"{'SI-SNR (dB)':<22} {results['noisy_sisnr']:>10.2f} {results['enh_sisnr']:>10.2f}")
    print(f"{'PESQ (wb)':<22} {results['noisy_pesq']:>10.3f} {results['enh_pesq']:>10.3f}")
    print(f"{'STOI':<22} {results['noisy_stoi']:>10.3f} {results['enh_stoi']:>10.3f}")
    print(f"{'Comp PESQ':<22} {results['noisy_comp_pesq']:>10.3f} {results['enh_comp_pesq']:>10.3f}")
    print(f"{'Comp OVRL':<22} {results['noisy_comp_ovrl']:>10.3f} {results['enh_comp_ovrl']:>10.3f}")
    print(f"{'Comp SIG':<22} {results['noisy_comp_sig']:>10.3f} {results['enh_comp_sig']:>10.3f}")
    print(f"{'Comp BAK':<22} {results['noisy_comp_bak']:>10.3f} {results['enh_comp_bak']:>10.3f}")
    print(f"{'='*55}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--hdf5_path",
                        default="data/results/save/test.hdf5")
    parser.add_argument("--output_path",
                        default="results/eval_exp7_streaming.txt")
    args = parser.parse_args()

    results = evaluate(args.ckpt_path, args.hdf5_path)
    _print_results(results)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    lines = [f"{k}={v}" for k, v in results.items()]
    with open(args.output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nMetrics saved -> {args.output_path}")


if __name__ == "__main__":
    main()
