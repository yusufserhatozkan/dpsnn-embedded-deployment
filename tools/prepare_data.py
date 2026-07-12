"""One-shot VoiceBank-DEMAND data prep for thesis deployment work.

The on-disk dataset (data/) ships as 48 kHz wavs without transcripts. Tao's
trainer expects 16 kHz wavs in *_16k folders plus per-utterance .txt
transcripts. The transcripts are only consumed by `create_csv` to populate a
`char` field that no downstream consumer ever reads (verified: dpsnn.data.
hdf5_prepare and dpsnn.data.wave_dataset2 both ignore it).

This script bridges the gap by:
  1. Resampling each of the four wav folders 48 kHz -> 16 kHz into *_16k siblings
  2. Writing train.csv / valid.csv / test.csv with absolute wav paths and an
     empty `char` column, splitting validation speakers as TRAIN_SPEAKERS[:2]
     to match Tao's `valid_speaker_count=2` default
  3. Building HDF5 caches via `dpsnn.data.hdf5_prepare.create_hdf5`

Run from any cwd:
    python tools/prepare_data.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The dpsnn package is installed via `pip install --editable .` but only resolves
# as a namespace package via cwd-style sys.path. Running this script via
# `python tools/prepare_data.py` puts tools/ (not the repo root) on sys.path[0],
# which hides dpsnn. Insert the repo root explicitly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

from dpsnn.data.voicebank_prepare import TRAIN_SPEAKERS
from dpsnn.data.hdf5_prepare import create_hdf5

DATA_DIR = REPO_ROOT / "data"
SAVE_DIR = DATA_DIR / "results" / "save"

SAMPLE_RATE = 16000
ORIG_SR = 48000
VALID_SPEAKER_COUNT = 2

WAV_FOLDERS = {
    "clean_trainset_28spk_wav": "clean_trainset_28spk_wav_16k",
    "noisy_trainset_28spk_wav": "noisy_trainset_28spk_wav_16k",
    "clean_testset_wav":        "clean_testset_wav_16k",
    "noisy_testset_wav":        "noisy_testset_wav_16k",
}

CSV_HEADER = [
    "ID", "duration",
    "noisy_wav", "noisy_wav_format", "noisy_wav_opts",
    "clean_wav", "clean_wav_format", "clean_wav_opts",
    "char", "char_format", "char_opts",
]


def resample_folder(src: Path, dst: Path, default_resampler: Resample) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    wavs = sorted(src.glob("*.wav"))
    print(f"  {src.name} -> {dst.name}: {len(wavs)} files")
    skipped = 0
    extra_resamplers: dict[int, Resample] = {}
    for wav_path in tqdm(wavs, leave=False):
        out_path = dst / wav_path.name
        if out_path.exists():
            skipped += 1
            continue
        signal, sr = torchaudio.load(str(wav_path))
        if sr == SAMPLE_RATE:
            torchaudio.save(str(out_path), signal, sample_rate=SAMPLE_RATE)
            continue
        if sr == ORIG_SR:
            signal = default_resampler(signal)
        else:
            local = extra_resamplers.get(sr)
            if local is None:
                local = Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                extra_resamplers[sr] = local
            signal = local(signal)
        torchaudio.save(str(out_path), signal, sample_rate=SAMPLE_RATE)
    if skipped:
        print(f"    ({skipped} already present, skipped)")


def write_csv(
    csv_path: Path,
    noisy_dir: Path,
    clean_dir: Path,
    *,
    exclude_speakers: list[str] | None = None,
    include_only_speakers: list[str] | None = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    wavs = sorted(noisy_dir.glob("*.wav"))
    if exclude_speakers is not None:
        wavs = [w for w in wavs if not any(w.name.startswith(spk + "_") for spk in exclude_speakers)]
    if include_only_speakers is not None:
        wavs = [w for w in wavs if any(w.name.startswith(spk + "_") for spk in include_only_speakers)]

    print(f"  {csv_path.name}: {len(wavs)} entries")
    rows = [CSV_HEADER]
    missing = 0
    for wav_path in tqdm(wavs, leave=False):
        snt_id = wav_path.stem
        clean_path = clean_dir / wav_path.name
        if not clean_path.exists():
            missing += 1
            continue
        signal, sr = torchaudio.load(str(wav_path))
        duration = signal.shape[1] / sr
        rows.append([
            snt_id, f"{duration:.6f}",
            str(wav_path.resolve()), "wav", "",
            str(clean_path.resolve()), "wav", "",
            "", "string", "",
        ])
    if missing:
        print(f"    WARN: {missing} noisy files had no matching clean file, skipped", file=sys.stderr)
    with csv_path.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def main() -> None:
    print(f"Repo root: {REPO_ROOT}")
    print(f"Data dir:  {DATA_DIR}")
    print(f"Save dir:  {SAVE_DIR}")
    if not DATA_DIR.exists():
        sys.exit(f"ERROR: data directory not found at {DATA_DIR}")

    # 1. Resample all four wav folders
    print("\n[1/3] Resampling 48 kHz -> 16 kHz")
    default_resampler = Resample(orig_freq=ORIG_SR, new_freq=SAMPLE_RATE)
    for src_name, dst_name in WAV_FOLDERS.items():
        src = DATA_DIR / src_name
        dst = DATA_DIR / dst_name
        if not src.exists():
            print(f"  WARN: source folder missing: {src}", file=sys.stderr)
            continue
        resample_folder(src, dst, default_resampler)

    # 2. CSV manifests
    print("\n[2/3] Writing CSV manifests")
    train_noisy = DATA_DIR / "noisy_trainset_28spk_wav_16k"
    train_clean = DATA_DIR / "clean_trainset_28spk_wav_16k"
    test_noisy  = DATA_DIR / "noisy_testset_wav_16k"
    test_clean  = DATA_DIR / "clean_testset_wav_16k"

    valid_speakers = TRAIN_SPEAKERS[:VALID_SPEAKER_COUNT]
    print(f"  validation speakers: {valid_speakers}")

    write_csv(SAVE_DIR / "train.csv", train_noisy, train_clean, exclude_speakers=valid_speakers)
    write_csv(SAVE_DIR / "valid.csv", train_noisy, train_clean, include_only_speakers=valid_speakers)
    write_csv(SAVE_DIR / "test.csv",  test_noisy,  test_clean)

    # 3. HDF5 caches
    print("\n[3/3] Building HDF5 caches")
    for split in ("train", "valid", "test"):
        csv_path = SAVE_DIR / f"{split}.csv"
        hdf_path = SAVE_DIR / f"{split}.hdf5"
        if hdf_path.exists():
            print(f"  {hdf_path.name} already exists, skipping")
            continue
        print(f"  building {hdf_path.name} from {csv_path.name}")
        create_hdf5(str(csv_path), str(hdf_path), SAMPLE_RATE, channels=1)

    print("\nDone.")


if __name__ == "__main__":
    main()
