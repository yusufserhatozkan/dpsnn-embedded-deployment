"""Extract test utterances from VoiceBank-DEMAND HDF5 for STM32 deployment.

Creates float32 binary files and/or WAV files for a selection of test
utterances.  The binary files (.bin) can be flashed to STM32 or streamed
over UART for on-device inference testing.

For STM32 inference you need one 'input chunk' at a time: a window of
input_dim float32 samples (= output_size + context_size).  For full-file
testing, use the full-utterance files and chunk them on the host.

Usage (from repo root):
    # Extract first 5 utterances as float32 binary + WAV
    python tools/extract_test_audio.py \\
        --hdf5_path data/results/save/test.hdf5 \\
        --output_dir deploy/test_audio \\
        --n_files 5

    # Extract a specific file by index
    python tools/extract_test_audio.py \\
        --hdf5_path data/results/save/test.hdf5 \\
        --output_dir deploy/test_audio \\
        --indices 0 42 100
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import h5py
import numpy as np


def save_float32_bin(path: str, audio: np.ndarray) -> None:
    """Save 1-D float32 array to a raw binary file."""
    audio.astype(np.float32).tofile(path)


def save_wav(path: str, audio: np.ndarray, sr: int = 16000) -> None:
    """Save 1-D float32 array as a 16-bit WAV file."""
    import wave
    # Clip and convert float32 → int16
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def extract(
    hdf5_path: str,
    output_dir: str,
    indices: list[int],
    sr: int = 16000,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    manifest_lines = []

    with h5py.File(hdf5_path, "r") as f:
        for idx in indices:
            key = str(idx)
            if key not in f:
                print(f"  WARNING: index {idx} not in HDF5, skipping")
                continue

            noisy = f[key]["noisy"][()].astype(np.float32).squeeze()
            clean = f[key]["clean"][()].astype(np.float32).squeeze()
            file_id = f[key].attrs.get("ID", f"idx{idx}")
            length  = int(f[key].attrs["length"])

            # Trim to original length
            noisy = noisy[:length]
            clean = clean[:length]

            # Peak-normalize noisy (same as what the eval script does to enhanced)
            noisy_norm = noisy / (np.max(np.abs(noisy)) + 1e-8)

            prefix = os.path.join(output_dir, f"{idx:04d}")
            save_float32_bin(f"{prefix}_noisy.bin", noisy_norm)
            save_float32_bin(f"{prefix}_clean.bin", clean)
            save_wav(f"{prefix}_noisy.wav", noisy_norm, sr)
            save_wav(f"{prefix}_clean.wav", clean, sr)

            manifest_lines.append(
                f"{idx}\t{file_id}\t{length}\t"
                f"{prefix}_noisy.bin\t{prefix}_clean.bin"
            )

            print(f"  [{idx}] {file_id}  {length} samples ({length/sr:.2f}s)  "
                  f"→ {prefix}_*.bin/.wav")

    manifest_path = os.path.join(output_dir, "manifest.tsv")
    with open(manifest_path, "w") as fh:
        fh.write("idx\tfile_id\tlength\tnoisy_bin\tclean_bin\n")
        fh.write("\n".join(manifest_lines) + "\n")
    print(f"\nManifest saved → {manifest_path}")
    print(f"Binary format: float32, native endian, 1 channel, {sr} Hz")
    print(f"  Load on STM32: cast buffer to float* and pass directly to inference")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract test audio from HDF5 to float32 binary + WAV")
    parser.add_argument("--hdf5_path",  required=True,
                        help="VoiceBank-DEMAND test HDF5")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for extracted files")
    parser.add_argument("--n_files",    type=int, default=0,
                        help="Extract first N files (0 = use --indices)")
    parser.add_argument("--indices",    type=int, nargs="+",
                        help="Specific HDF5 indices to extract")
    parser.add_argument("--sr",         type=int, default=16000)
    args = parser.parse_args()

    if args.n_files > 0:
        indices = list(range(args.n_files))
    elif args.indices:
        indices = args.indices
    else:
        print("Error: specify --n_files or --indices")
        sys.exit(1)

    print(f"Extracting {len(indices)} utterances from {args.hdf5_path}")
    print(f"Output dir: {args.output_dir}")
    extract(args.hdf5_path, args.output_dir, indices, sr=args.sr)


if __name__ == "__main__":
    main()
