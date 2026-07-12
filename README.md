# DPSNN Embedded Deployment

> **Fork notice.** This repository is a fork of
> [`tao-sun/dpsnn`](https://github.com/tao-sun/dpsnn) (Sun & Bohté 2024,
> *Neuromorphic Computing and Engineering*). The upstream README is preserved
> below. Everything in this repo beyond the upstream content is part of a
> bachelor's thesis on hardware-aware deployment of speech enhancement models
> to embedded IoT hardware.

---

## Thesis Context

**Title:** *Hardware-Aware Optimization and Benchmarking of Low-Latency Speech
Enhancement Models on Embedded Internet-of-Things Platforms*

**Research question:** Do efficiency claims from desktop/GPU evaluations of
speech enhancement neural networks transfer to bare-metal embedded
deployment? Measured via on-device latency, RAM usage, and audio quality
(SI-SNR, PESQ, STOI).

**Target hardware:** STMicroelectronics **B-U585I-IOT02A** Discovery Kit
(Cortex-M33, 2 MB Flash, 786 KB RAM)

**Deployment pipeline:** PyTorch → ONNX (streaming, custom nn.Fold) →
X-CUBE-AI (10-step postprocess) → STM32 firmware

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
- Trained at **N = B = H = 128** on VoiceBank-DEMAND. **71,299 parameters**
  (vs 104,579 for the full SCNN+SRNN at the same width). The final deployed
  checkpoint (Exp 7) reaches **17.42 dB test SI-SNR** (streaming, as deployed). An initial
  `frame_dur=0.5` run plateaued at 9.52 dB and was discarded for too little
  temporal context — see [`results/log_training.md`](results/log_training.md).

### 2. ONNX export with two upstream bug fixes
- [`export/export_to_onnx.py`](export/export_to_onnx.py) — exports the DPSNN
  streaming model to ONNX (opset 13) and runs the 10-step X-CUBE-AI postprocess.
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

### 3. INT8 static quantization (exploratory — not deployed)
- [`export/quantize_int8.py`](export/quantize_int8.py) — uses
  `onnxruntime.quantization.quantize_static` (QDQ format, per-tensor MinMax)
  with calibration samples streamed from the VoiceBank test HDF5.
- Includes a quality snapshot that compares FP32 vs INT8 SI-SNR on a few
  utterances and writes `<output>.metrics.txt`.
- Investigated but not used on-device: INT8 does not reduce the activation
  footprint (the bottleneck is graph topology, not weights) and the spike path
  stays FP32, so X-CUBE-AI reports `model_fmt: float`.

### 4. Full ONNX evaluation pipeline
- [`evaluation/eval_onnx.py`](evaluation/eval_onnx.py) — runs an ONNX model
  over the entire 824-sample VoiceBank test set and reports SI-SNR, PESQ,
  STOI, and composite (DNSMOS-style) scores.
- Replicates `EvaluationDataset.__getitem__` chunking exactly (prepend
  context-size zeros, pad to multiple of output_size, slice with
  output_size hop) so ONNX outputs match PyTorch outputs.

### 5. Embedded-deployment tooling
- [`tools/estimate_footprint.py`](tools/estimate_footprint.py) — reads an
  ONNX proto, counts parameters and weight bytes by dtype, and reports
  Flash/RAM usage against the 2 MB / 786 KB STM32 limits.
- [`tools/extract_test_audio.py`](tools/extract_test_audio.py) — extracts
  noisy/clean utterances from the HDF5 cache to float32 `.bin` and `.wav`
  files for on-board playback / inference.
- [`tools/prepare_data.py`](tools/prepare_data.py) — resamples the
  VoiceBank wav files (originally 48 kHz) down to 16 kHz, writes
  train/valid/test CSVs and HDF5 caches.

### 6. Windows + single-GPU patches
- `vctk.yaml`: `accelerator: auto`, `strategy: auto`,
  `data_folder: ../../data`, `num_workers: 0` (h5py handles aren't
  picklable on Windows worker spawn).
- `vctk_trainer.py`: hardcoded `num_workers=8` → `0`.
- Use `--device_num 1` (integer) instead of the upstream
  `--devices 0` (list, invalid for the CPU accelerator).

### 7. Experiment log
- [`results/experiment_log.md`](results/experiment_log.md) — a complete
  written record of every experiment in the project (pretrained baseline,
  ONNX export, SCNN-only training trajectory, footprint analysis).

---

## Current Status

| Phase | Status |
|---|---|
| 1 — Environment & data prep | done |
| 2 — Pretrained ONNX export + validation | done (8.2 MB FP32, diff 4.01e-05) |
| 3 — SCNN-only N=128 training | done — 71,299 params, test FP32 SI-SNR **17.42 dB** streaming / 17.41 dB whole-utterance (Exp 7, bf16 retrain) |
| 3b — SCNN-only N=64 training | done — 23,430 params, test FP32 SI-SNR **16.70 dB** (Exp 10), RTF **1.086×** on STM32 |
| 5 — INT8 quantization (explored, not deployed) | INT8 does not address the memory bottleneck (graph topology, not weights); spike-path FP32 islands force `model_fmt: float`. FP32 is deployed. See [`results/log_quantization.md`](results/log_quantization.md). |
| 6 — RAM bottleneck investigation | done — 1.37 MB activation floor traced to 399-step unrolled OLA chain; resolved by streaming export (Exp 7) and by Exp 9 (BatchNorm + stride=kernel=80) |
| 7 — STM32 deployment | **done** — Exp 7 (N=128) and Exp 10 (N=64) streaming FP32 validated on B-U585I-IOT02A. N=128: 46 KB activations, 6.145 ms/frame, RTF 2.46×. N=64: 23 KB activations, 2.715 ms/frame, RTF 1.086×. Both match Python ONNX to 0.00 dB on real speech (p232_009, 4.16 s). |

### Deployed model — Exp 7 streaming FP32 on STM32U585

Final on-device measurements (full record in [`results/log_xcube_ai.md`](results/log_xcube_ai.md)):

| Metric | Value | Budget | Status |
|---|---|---|---|
| Activation SRAM (peak) | 46 KB | 786 KB | ✓ 5.8 % |
| Model weights (Flash) | 280 KB | 2 MB | ✓ 14 % |
| Total Flash (weights + embedded test utterance) | ~540 KB | 2 MB | ✓ 26 % |
| Per-frame latency | 6.145 ms | — | — |
| Real-Time Factor (RTF) | 2.46× | < 1× | ✗ not real-time |
| MCU SI-SNR (p232_009) | 15.66 dB | — | matches Python ref to 0.00 dB |
| aiValidation NSE (all 5 outputs) | 1.000 | — | numerically identical to ONNX |

RTF > 1 is a **measured negative finding**, not a project failure: the unrolled 399-step
SCNN at 160 MHz Cortex-M33 cannot keep up with 16 kHz audio at N=128. This is one axis
of the quality-latency trade-off characterised in the thesis.

### Channel-reduced variant — Exp 10 N=64 streaming FP32 on STM32U585

| Metric | Value | Budget | Status |
|---|---|---|---|
| Activation SRAM (peak) | 23 KB | 786 KB | ✓ 2.9 % |
| Model weights (Flash) | 91 KB | 2 MB | ✓ 4.4 % |
| Per-frame latency | 2.715 ms | — | — |
| Real-Time Factor (RTF) | 1.086× | < 1× | ✗ 8.6% over |
| MCU SI-SNR (p232_009) | 15.98 dB | — | matches Python ref to 0.00 dB |
| Test-set SI-SNR | 16.70 dB | — | −0.72 dB vs N=128 |

### INT8 quantization — explored, not deployed

Consistent with the paper, INT8 is **not** part of the deployed pipeline, for two reasons:

- **It does not address the memory bottleneck.** The 1.37 MiB overflow comes from the
  399-step overlap-add chain (graph topology), not weight storage — INT8 weight
  quantization leaves peak activations within 10 KB of the FP32 number.
- **The spike path stays FP32.** The spike-aware quantizer deliberately keeps membrane
  potentials in FP32; those islands prevent X-CUBE-AI from propagating INT8 across the
  heavy convolutions, so the compiler reports `model_fmt: float` and the generated
  kernels are essentially identical to the FP32 build.

Both deployed models therefore use **FP32 streaming** throughout. The exploratory
quantization study (spike-aware QDQ, percentile calibration, W8A16 ceiling) is recorded
in [`results/log_quantization.md`](results/log_quantization.md).

---

## Repository Layout

```
dpsnn-embedded-deployment/
├── dpsnn/                       # Core package (upstream + scnn_only flag)
│   ├── data/                    # Datasets, augmentation, metrics, DNSMOS
│   ├── layers/                  # Spiking neurons (PLIFNode, ALIFNode), surrogates, SI-SDR loss
│   └── models/dp_binary_net.py  # StreamSpikeNet (with scnn_only + bnorm patches)
├── egs/voicebank/               # Training script + YAML config
│   ├── vctk_trainer.py          # PyTorch Lightning trainer (Windows-patched)
│   ├── vctk.yaml                # Config (accelerator/strategy/num_workers/bf16 patched)
│   └── lightning_logs/          # version_0..version_9 — checkpoints (gitignored)
├── models/                      # Pre-built deployable models — ready to run (see models/README.md)
│   ├── dpsnn_n128.ckpt          # N=128 checkpoint (Exp 7, 17.42 dB); dpsnn_n64.ckpt = N=64 (Exp 10, 16.70 dB)
│   └── dpsnn_n{128,64}_streaming[_xcubeai].onnx  # Streaming ONNX (raw + X-CUBE-AI-ready)
├── tools/
│   ├── prepare_data.py          # 48 kHz → 16 kHz resample + HDF5 cache
│   ├── estimate_footprint.py    # ONNX → Flash/RAM estimate vs STM32 limits
│   ├── extract_test_audio.py    # HDF5 → .bin / .wav for STM32
│   ├── inline_if_nodes.py       # Pipeline step 1 (legacy stand-alone; now run by export_to_onnx.py)
│   ├── fix_shapes.py            # Pipeline step 2 helper (legacy stand-alone)
│   ├── bake_shapes.py           # Pipeline step 10 helper (legacy stand-alone)
│   ├── wav_to_c_array.py        # Embed a WAV utterance as float32 C array (test_utterance.c)
│   └── mcu_audio_receiver.py    # USART1 receiver: collects MCU-enhanced frames, computes SI-SNR
├── export/                      # ONNX export + INT8 quantization + streaming
│   ├── export_to_onnx.py        # Batch export + postprocess_for_xcubeai() 10-step pipeline
│   ├── export_streaming.py      # Single-frame streaming export with recurrent state I/O
│   ├── validate_onnx.py
│   ├── quantize_int8.py                   # Standard ORT (kept for reference, catastrophic on SNNs)
│   ├── quantize_spike_aware_correct.py    # Custom spike-aware QDQ (the thesis quantizer)
│   └── map_spike_tensors.py               # BFS classifier: SAFE vs SPIKE Conv outputs
├── evaluation/
│   ├── eval_onnx.py             # Full 824-utterance batch test (SI-SNR/PESQ/STOI/DNSMOS)
│   └── eval_streaming.py        # Streaming-wrapper evaluation (matches batch numerically)
├── results/                     # Live experiment logs (also mirrored to report/ for LaTeX bundle)
│   ├── experiment_log.md        # Index + X-CUBE-AI compatibility-fix history (9 rounds)
│   ├── log_baseline.md          # Exp 0–2 (pretrained + ONNX validation)
│   ├── log_training.md          # Exp 3, 5, 7, 9, 10 (training runs)
│   ├── log_quantization.md      # Exp 1b/1c, 6, 7-pct, 8 (W8A16 ceiling)
│   ├── log_xcube_ai.md          # Stedgeai analyze runs, RAM root cause, on-device validation
│   └── exp{7,8,9,10}_*_epoch_log.md
├── deploy/                      # MCU receiver outputs + reference clips
│   ├── mcu_enhanced.wav         # Output reconstructed from USART1 stream
│   ├── mcu_test_results.txt     # SI-SNR comparison vs Python reference
│   └── test_audio/              # HDF5 → .bin / .wav extracts for embedding
├── st_ai_output/, st_ai_ws/     # X-CUBE-AI analyze reports + generated C workspace
└── data/                        # Dataset (gitignored — never committed)
```

The MCU firmware C project (CubeIDE workspace, X-CUBE-AI-generated C code,
`app_x-cube-ai.c` streaming loop) lives **outside this repo** at
a sibling `Stm_deployment/` folder.

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
    --context_dur 0.01 --frame_dur 1.0 --max_epochs 100 -X 1 --lr 1e-2 \
    --device_num 1 --scnn_only --batch_size 64
```

Top-3 checkpoints by val_loss are saved under
`egs/voicebank/lightning_logs/version_N/checkpoints/`. Training runs for 100
epochs without early stopping; the best checkpoint by val_loss is used. If OOM,
reduce `--batch_size` to 32.

### 3. Export and validate ONNX

Best checkpoint from Exp 7 (bf16 retrain, 17.42 dB streaming):
`egs/voicebank/lightning_logs/version_2/checkpoints/epoch=87-val_loss=82.2283-val_sisnr=-17.7750.ckpt`

```bash
python export/export_to_onnx.py \
    --ckpt_path egs/voicebank/lightning_logs/version_2/checkpoints/epoch=87-val_loss=82.2283-val_sisnr=-17.7750.ckpt \
    --output_path export/dpsnn_scnn128_exp7.onnx

python export/validate_onnx.py \
    --ckpt_path egs/voicebank/lightning_logs/version_2/checkpoints/epoch=87-val_loss=82.2283-val_sisnr=-17.7750.ckpt \
    --onnx_path export/dpsnn_scnn128_exp7.onnx
```

### 4. INT8-quantize (spike-aware, pct95) — exploratory, not in the deployed pipeline

Uses `export/quantize_spike_aware_correct.py` — custom QDQ placement after
ReLU/Sigmoid only, shared scale per layer, 95th-percentile calibration.
Standard `quantize_int8.py` destroys quality on SNNs (see `results/log_quantization.md`).
The deployed firmware uses FP32 streaming; this step reproduces the quantization study only.

```bash
python export/quantize_spike_aware_correct.py \
    --onnx_path export/dpsnn_scnn128_exp7.onnx \
    --spike_map export/dpsnn_scnn128.onnx.spike_map.json \
    --hdf5_path data/results/save/test.hdf5 \
    --output_path export/dpsnn_scnn128_exp7_int8_pct95.onnx \
    --n_calib 50 --relu_percentile 95.0
```

### 4b. Preprocess ONNX for X-CUBE-AI

The raw exported ONNX has ten distinct incompatibilities with X-CUBE-AI
(v10.2.0 / ST Edge AI Core 2.2.0-20266). All of them are now fixed by
`postprocess_for_xcubeai()` inside `export/export_to_onnx.py`, which runs
automatically after each ONNX export and writes `<stem>_xcubeai.onnx`:

1. Inline 798 degenerate `If` nodes (PyTorch tracer control-flow artefacts)
2. Remove 3990 dead `Equal` / `Shape` / `Gather` / `Constant` nodes
3. Strip 399 empty `""` trailing inputs from `Pad` nodes
4. ORT `ORT_ENABLE_BASIC` constant-folding (eliminates `ConstantOfShape`, residual `Shape`/`Gather`)
5. Strip 7 foreign opset imports (`ai.onnx.ml`, `com.microsoft`, …) injected by ORT
6. Bundled `onnxsim` pass — removes 400 no-op `Reshape` and 399 redundant `Concat`
7. Inflate `Pad` and `ConvTranspose` to rank-4 NCHW (X-CUBE-AI rejects rank-2 Pad and rank-3 ConvTranspose)
8. ConvTranspose **time-as-H** layout (kernel `(K,1)`, not `(1,K)`)
9. Eliminate BOOL tensors — replace `Greater + Cast` → `Sub → Sign → Relu`; replace `Where(bool,A,B)` with arithmetic
10. ONNX shape inference + ORT-driven concrete-shape baking (~5 min on the full 399-step graph)

Reproducible workflow is now just:

```bash
python export/export_to_onnx.py \
    --ckpt_path egs/voicebank/lightning_logs/version_2/checkpoints/epoch=87-val_loss=82.2283-val_sisnr=-17.7750.ckpt \
    --output_path export/dpsnn_scnn128_exp7.onnx
# Produces both dpsnn_scnn128_exp7.onnx (raw) and dpsnn_scnn128_exp7_xcubeai.onnx (X-CUBE-AI-ready).
```

See [`results/experiment_log.md`](results/experiment_log.md) §"X-CUBE-AI Compatibility Fix"
for the full incremental debugging history (nine rounds) that produced this pipeline.

### 4c. Streaming export (the actual deployment path)

The batch model has a 399-step unrolled OLA chain (1,196 × 64 KB buffers) that
forces ≥ 1.37 MB activation RAM even after maximal slot reuse — over the
786 KB SRAM budget. The deployed path is a **single-frame streaming model**
with explicit recurrent state I/O. Export via:

```bash
python export/export_streaming.py \
    --ckpt_path egs/voicebank/lightning_logs/version_2/checkpoints/epoch=87-val_loss=82.2283-val_sisnr=-17.7750.ckpt \
    --output_path export/dpsnn_streaming.onnx
```

The streaming wrapper takes `frame (1,80)` + state tensors (`context_win (1,128,4)`,
`v_plif (1,128,1)`, `mem_readout (1,128)`, `ola_tail (1,40)`, total state 3.2 KB) and
produces `enhanced (1,40)` + updated state. The MCU runs this 399× per utterance.
Numerically identical to the batch model (max abs diff = 0.0); X-CUBE-AI reports
46 KB activations.

### 5. Evaluate FP32 vs INT8 on the full test set

```bash
python evaluation/eval_onnx.py \
    --onnx_path export/dpsnn_scnn128.onnx \
    --hdf5_path data/results/save/test.hdf5 \
    --output_path results/eval_scnn128_fp32.txt

python evaluation/eval_onnx.py \
    --onnx_path export/dpsnn_scnn128_int8.onnx \
    --hdf5_path data/results/save/test.hdf5 \
    --output_path results/eval_scnn128_int8.txt
```

### 6. Check embedded footprint

```bash
python tools/estimate_footprint.py export/dpsnn_scnn128_int8.onnx
```

### 7. Generate STM32 input data

```bash
python tools/extract_test_audio.py
```

### 8. Deploy via X-CUBE-AI (Exp 7 streaming FP32 — actually deployed)

```bash
# Analyse
stedgeai analyze --model export/dpsnn_streaming_xcubeai.onnx \
    --target stm32u5 --optimization balanced --compression none \
    --name dpsnn_streaming

# Generate C code
stedgeai generate --model export/dpsnn_streaming_xcubeai.onnx \
    --target stm32u5 --name dpsnn_streaming \
    --output Stm_deployment/X-CUBE-AI/App/

# On-target numerical validation (X-CUBE-AI aiValidation UART mode)
stedgeai validate --model export/dpsnn_streaming_xcubeai.onnx \
    --target stm32 --mode target --desc serial:COM3 --name dpsnn_streaming
```

The MCU firmware lives in a separate workspace
(a sibling `Stm_deployment/` folder). Latency is measured with the
DWT cycle counter at 160 MHz sysclk; per-frame audio is streamed to PC over
USART1 at 115200 baud and reassembled by [`tools/mcu_audio_receiver.py`](tools/mcu_audio_receiver.py).

**Measured on-device result** (Exp 7 streaming FP32, full utterance p232_009 /
VoiceBank-DEMAND, 4.16 s, 1662 frames):

- MCU SI-SNR = **15.66 dB** (input 6.77 dB → +8.89 dB improvement)
- Python ONNX reference = 15.66 dB → MCU vs reference **delta = 0.00 dB**
- Per-frame latency 6.145 ms → **RTF 2.46×** (not real-time on this MCU)
- Activation RAM 46 KB / 786 KB; total Flash ~540 KB / 2048 KB
  (model weights 280 KB + embedded test utterance ~260 KB)

See [`results/log_xcube_ai.md`](results/log_xcube_ai.md) §10 for the full trace.

---

## License

This repository is released under the [MIT License](LICENSE). It builds on
external work — the upstream DPSNN code by Sun & Bohté, speech-quality metric
code from `facebookresearch/denoiser`, Microsoft's DNSMOS, and the
VoiceBank-DEMAND corpus — credited in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## Citing the Upstream Work

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
