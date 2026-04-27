# DPSNN Embedded Deployment

> **Fork notice.** This repository is a fork of
> [`tao-sun/dpsnn`](https://github.com/tao-sun/dpsnn) (Sun & Bohté 2024,
> *Neuromorphic Computing and Engineering*). The upstream README is preserved
> below. Everything in this repo beyond the upstream content is part of a
> master's thesis on hardware-aware deployment of speech enhancement models
> to embedded IoT hardware.

---

## Thesis Context

**Title:** *Hardware-Aware Optimization And Benchmarking Of Real-Time Speech
Enhancement Models On Embedded IoT Platforms*

**Research question:** Do efficiency claims from desktop/GPU evaluations of
speech enhancement neural networks transfer to bare-metal embedded
deployment? Measured via on-device latency, RAM usage, and audio quality
(SI-SNR, PESQ, STOI).

**Target hardware:** STMicroelectronics **B-U585I-IOT02A** Discovery Kit
(Cortex-M33, 2 MB Flash, 786 KB RAM)

**Deployment pipeline:** PyTorch → ONNX → INT8 (onnxruntime QDQ) →
X-CUBE-AI → STM32 firmware

**Dataset:** VoiceBank-DEMAND (28-speaker, Edinburgh DataShare), 16 kHz

---

## What This Fork Adds Over `tao-sun/dpsnn`

The upstream repo provides the DPSNN architecture, training script, and a
pretrained 256-channel checkpoint. To take that work to embedded hardware,
this fork adds:

### 1. A simplified DPSNN variant for embedded targets
- New `scnn_only` flag on `StreamSpikeNet`
  ([`dpsnn/models/dp_binary_net.py`](dpsnn/models/dp_binary_net.py)) that
  removes the SRNN dense+recurrent path while keeping the SCNN
  (depthwise SpikeConv1d + PLIFNode neurons).
- Justified by Table 5 of the DPSNN paper: SCNN contributes more to SI-SNR
  than SRNN.
- Trained at **N = B = H = 128** for 200 epochs on VoiceBank-DEMAND.
  **57,476 parameters**, best **SI-SNR = 9.52 dB** at epoch 163.

### 2. A Conv-TasNet baseline
- [`convtasnet/model.py`](convtasnet/model.py) — Conv-TasNet
  (Luo & Mesgarani 2018) sized to match the SCNN-only model
  (N=48, B=48, H=96, P=3, tcn_depth=3, tcn_repeats=1, **56,839 params**).
- Acts as the non-spiking, guaranteed-deployable comparison point.

### 3. ONNX export with two upstream bug fixes
- [`export/export_to_onnx.py`](export/export_to_onnx.py) — exports either
  DPSNN or Conv-TasNet to ONNX (opset 13).
- Fixes baked into `dpsnn/models/dp_binary_net.py`:
  1. **`aten::col2im` unsupported** in opset 13 → replaced `nn.Fold`
     overlap-add with a `F.pad + sum` loop (mathematically equivalent).
  2. **`Where` node type mismatch** — `torch.where(x > thr, x, 0.0)` has a
     Python float64 literal that ONNX rejects against a float32 tensor.
     Fixed to `torch.where(..., torch.zeros_like(x))`.
- Validated end-to-end with
  [`export/validate_onnx.py`](export/validate_onnx.py): max abs diff between
  PyTorch and ONNX Runtime is 4.01e-05 for the pretrained model and
  9.16e-05 for our SCNN-only variant (both far below the 1e-3 threshold).

### 4. INT8 static quantization
- [`export/quantize_int8.py`](export/quantize_int8.py) — uses
  `onnxruntime.quantization.quantize_static` (QDQ format, per-tensor MinMax)
  with calibration samples streamed from the VoiceBank test HDF5.
- Includes a quality snapshot that compares FP32 vs INT8 SI-SNR on a few
  utterances and writes `<output>.metrics.txt`.

### 5. Full ONNX evaluation pipeline
- [`evaluation/eval_onnx.py`](evaluation/eval_onnx.py) — runs an ONNX model
  over the entire 824-sample VoiceBank test set and reports SI-SNR, PESQ,
  STOI, and composite (DNSMOS-style) scores.
- Replicates `EvaluationDataset.__getitem__` chunking exactly (prepend
  context-size zeros, pad to multiple of output_size, slice with
  output_size hop) so ONNX outputs match PyTorch outputs.

### 6. Embedded-deployment tooling
- [`tools/estimate_footprint.py`](tools/estimate_footprint.py) — reads an
  ONNX proto, counts parameters and weight bytes by dtype, and reports
  Flash/RAM usage against the 2 MB / 786 KB STM32 limits.
- [`tools/extract_test_audio.py`](tools/extract_test_audio.py) — extracts
  noisy/clean utterances from the HDF5 cache to float32 `.bin` and `.wav`
  files for on-board playback / inference.
- [`tools/prepare_data.py`](tools/prepare_data.py) — resamples the
  VoiceBank wav files (originally 48 kHz) down to 16 kHz, writes
  train/valid/test CSVs and HDF5 caches.

### 7. Windows + single-GPU patches
- `vctk.yaml`: `accelerator: auto`, `strategy: auto`,
  `data_folder: ../../data`, `num_workers: 0` (h5py handles aren't
  picklable on Windows worker spawn).
- `vctk_trainer.py`: hardcoded `num_workers=8` → `0`.
- Use `--device_num 1` (integer) instead of the upstream
  `--devices 0` (list, invalid for the CPU accelerator).

### 8. Experiment log
- [`results/experiment_log.md`](results/experiment_log.md) — a complete
  written record of every experiment in the project (pretrained baseline,
  ONNX export, SCNN-only training trajectory through all 200 epochs,
  Conv-TasNet plan, footprint analysis).

---

## Current Status

| Phase | Status |
|---|---|
| 1 — Environment & data prep | done |
| 2 — Pretrained ONNX export + validation | done (8.2 MB FP32, diff 4.01e-05) |
| 3 — SCNN-only N=128 training | done (200 epochs, **9.52 dB SI-SNR** @ epoch 163) |
| 4 — Conv-TasNet baseline | implemented, training pending |
| 5 — SCNN-only ONNX export + validation | done (2.9 MB FP32, diff 9.16e-05) |
| 6 — INT8 quantization | script ready, run pending |
| 7 — STM32 deployment | footprint analyzed, X-CUBE-AI work pending |

### Pretrained N=256 baseline (full 824-sample VoiceBank test set)

| Metric | Noisy input | Enhanced | Clean ref |
|---|---|---|---|
| SI-SNR (dB) | 8.44 | **18.08** | — |
| PESQ (wb) | 1.97 | **2.26** | 4.64 |
| STOI | 0.921 | **0.925** | 1.00 |

### SCNN-only N=128 (our trained variant)

| Property | Value |
|---|---|
| Parameters | 57,476 |
| Best val SI-SNR | **9.52 dB** (epoch 163) |
| FP32 ONNX size | 2,943 KB |
| Estimated INT8 weights | ~56 KB (well under 2 MB Flash) |
| Estimated peak RAM | ~463 KB (under 786 KB) |

Full per-epoch trajectory: see
[`results/experiment_log.md`](results/experiment_log.md).

---

## Repository Layout

```
dpsnn-embedded-deployment/
├── dpsnn/                       # Core package (upstream + scnn_only flag)
│   ├── data/                    # Datasets, augmentation, metrics, DNSMOS
│   ├── layers/                  # Spiking neurons, surrogates, SI-SDR loss
│   └── models/dp_binary_net.py  # StreamSpikeNet (with our scnn_only patch)
├── egs/voicebank/               # Training script + YAML config
│   ├── vctk_trainer.py          # PyTorch Lightning trainer (Windows-patched)
│   ├── vctk.yaml                # Config (accelerator/strategy/num_workers patched)
│   └── *.ckpt                   # Saved checkpoints (gitignored)
├── tools/
│   ├── prepare_data.py          # 48 kHz → 16 kHz resample + HDF5 cache
│   ├── estimate_footprint.py    # ONNX → Flash/RAM estimate vs STM32 limits
│   └── extract_test_audio.py    # HDF5 → .bin / .wav for STM32
├── export/                      # ONNX export + INT8 quantization
│   ├── export_to_onnx.py
│   ├── validate_onnx.py
│   └── quantize_int8.py
├── evaluation/
│   └── eval_onnx.py             # Full 824-utterance test (SI-SNR/PESQ/STOI)
├── convtasnet/
│   └── model.py                 # Conv-TasNet baseline
├── results/
│   └── experiment_log.md        # Detailed write-up of every experiment
├── audio_demos/                 # Upstream demo outputs
├── figures/                     # Architecture diagrams
└── data/                        # Dataset (gitignored — never committed)
```

---

## Setup

```bash
conda create --name dpsnn python=3.11.5
conda activate dpsnn
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    pytorch-cuda==11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install --editable .          # required (dpsnn is a namespace package)

# Thesis-specific deps
pip install onnx onnxruntime pesq pystoi scipy soundfile librosa
```

Verify:
```bash
python -c "import torch, dpsnn; print('OK')"
```

---

## Reproducing the Thesis Pipeline

### 1. Prepare the dataset

Download the VoiceBank-DEMAND clean+noisy train and test wav folders from
the [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791)
into `data/`, then:

```bash
python tools/prepare_data.py
```

This resamples to 16 kHz and writes `data/results/save/{train,valid,test}.{csv,hdf5}`.

### 2. Train the SCNN-only variant (N=128)

```bash
cd egs/voicebank
PYTHONPATH=../../ python -u vctk_trainer.py --config vctk.yaml \
    -L 80 --stride 40 -N 128 -B 128 -H 128 \
    --context_dur 0.01 --frame_dur 0.5 --max_epochs 200 -X 1 --lr 1e-2 \
    --device_num 1 --scnn_only --batch_size 64
```

Top-3 checkpoints by val_loss are saved under
`egs/voicebank/lightning_logs/version_N/checkpoints/`.

### 3. Export and validate ONNX

```bash
python export/export_to_onnx.py \
    --ckpt_path egs/voicebank/lightning_logs/version_9/checkpoints/epoch=163-val_loss=90.5477-val_sisnr=-9.5152.ckpt \
    --output_path export/dpsnn_scnn128.onnx

python export/validate_onnx.py \
    --ckpt_path egs/voicebank/lightning_logs/version_9/checkpoints/epoch=163-val_loss=90.5477-val_sisnr=-9.5152.ckpt \
    --onnx_path export/dpsnn_scnn128.onnx \
    --model dpsnn
```

### 4. INT8-quantize

```bash
python export/quantize_int8.py \
    --onnx_path export/dpsnn_scnn128.onnx \
    --hdf5_path data/results/save/test.hdf5 \
    --output_path export/dpsnn_scnn128_int8.onnx \
    --n_calib 50
```

### 5. Evaluate FP32 vs INT8 on the full test set

```bash
python evaluation/eval_onnx.py --onnx_path export/dpsnn_scnn128.onnx
python evaluation/eval_onnx.py --onnx_path export/dpsnn_scnn128_int8.onnx
```

### 6. Check embedded footprint

```bash
python tools/estimate_footprint.py export/dpsnn_scnn128_int8.onnx
```

### 7. Generate STM32 input data

```bash
python tools/extract_test_audio.py
```

### 8. Deploy via X-CUBE-AI

Run the X-CUBE-AI **Analyse** step on the INT8 ONNX, then **Generate**
to produce C code, integrate with the STM32 firmware skeleton, build, and
flash. On-device latency is measured with TIM16 (Prescaler = 15999 →
100 µs tick); printf is routed via `_write` over USART1 at 115200 baud
(needs `-u_printf_float` linker flag).

---

## Citing the Upstream Work

If you build on the DPSNN architecture itself, please cite the original
paper:

```bibtex
@article{sun2024dpsnn,
  title={DPSNN: spiking neural network for low-latency streaming speech enhancement},
  author={Sun, Tao and Boht{\'e}, Sander},
  journal={Neuromorphic Computing and Engineering},
  volume={4},
  number={4},
  pages={044008},
  year={2024},
  publisher={IOP Publishing}
}
```

---

## Upstream README (preserved)

> The text below is the original `README.md` from
> [`tao-sun/dpsnn`](https://github.com/tao-sun/dpsnn).

# DPSNN: Spiking Neural Network for Low-Latency Streaming Speech Enhancement

Inspired by the Dual-Path Recurrent Neural Network (DPRNN) in deep neural
networks (DNNs), we develop a two-phase time-domain streaming SNN framework
for speech enhancement, named [Dual-Path Spiking Neural Network
(DPSNN)](https://iopscience.iop.org/article/10.1088/2634-4386/ad93f9/pdf).
DPSNNs achieve low latency by replacing the STFT and inverse STFT (iSTFT) in
traditional frequency-domain models with a learned convolutional encoder and
decoder. In the DPSNN, the first phase uses Spiking Convolutional Neural
Networks (SCNNs) to capture temporal contextual information, while the
second phase uses Spiking Recurrent Neural Networks (SRNNs) to focus on
frequency-related features. Evaluating on the Voice Cloning Toolkit (VCTK)
Corpus and Intel N-DNS Challenge dataset, our approach demonstrates
excellent performance in speech objective metrics, along with the very low
latency (approximately 5 ms) required for applications like hearing aids.

## Architecture
The proposed DPSNN adopts the encoder-separator-decoder architecture. The
encoder uses convolutions to convert waveform signals into 2D feature maps,
effectively replacing the function of STFT. In the separator, a 2D mask is
calculated, primarily relying on the SCNN and SRNN modules that capture the
temporal and frequency contextual information of the encoded feature maps,
respectively. In addition, threshold-based activation suppression, along
with L1 regularization loss, is applied to specific non-spiking layers in
DPSNNs to further improve their energy efficiency. After applying the
calculated mask to the feature maps from the encoder, the decoder
transforms the masked feature maps back to enhanced waveform signals.

<img src="./figures/arch.jpeg" />

## Installation
Follow the steps in installation.txt.

## Training and Inference
```bash
cd egs/voicebank
# Training and testing
python -u vctk_trainer.py --config vctk.yaml -L 80 --stride 40 -N 256 -B 256 -H 256 --context_dur 0.01 --max_epochs 500 -X 1 --lr 1e-2
# Inference only
python -u vctk_trainer.py --config vctk.yaml -L 80 --stride 40 -N 256 -B 256 -H 256 --context_dur 0.01 --max_epochs 500 -X 1 --lr 1e-2 --test_ckpt_path ./epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt
```
The model file for this run can be found in
`egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt`. Note
that the model itself may differ slightly across different versions of
PyTorch.

## Demos
Demos are in the `audio_demos/` folder.
