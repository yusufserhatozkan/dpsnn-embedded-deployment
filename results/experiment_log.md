# Experiment Log

## Setup
- Repo: dpsnn-embedded-deployment
- Strategy: Validate-First — test ONNX export before training simplified models

---

## Experiment 0: Pretrained Model Inference
- Date: 2026-04-12
- Checkpoint: egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt
- Params: N=256, B=256, H=256, L=80, stride=40, context_dur=0.01, X=1
- Status: **COMPLETED** (CPU, 28 min, 824/824 samples)

### Command Used
```bash
cd egs/voicebank
PYTHONPATH=../../ python -u vctk_trainer.py --config vctk.yaml \
    -L 80 --stride 40 -N 256 -B 256 -H 256 \
    --context_dur 0.01 --max_epochs 500 -X 1 --lr 1e-2 \
    --device_num 1 \
    --test_ckpt_path ./epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt
```

### Desktop Baseline Metrics (Pretrained, full 824-sample test set)
| Metric | Noisy (input) | Enhanced | Clean (reference) |
|--------|--------------|---------|------------------|
| SI-SNR (dB) | 8.44 | **18.08** | — |
| PESQ (wb) | 1.97 | **2.26** | 4.64 |
| STOI | 0.921 | **0.925** | 1.00 |

### Composite Scores (full test set)
| | PESQ | OVRL | SIG | BAK |
|---|---|---|---|---|
| Noisy | 1.97 | 2.64 | 3.36 | 2.45 |
| Enhanced | 2.26 | 2.80 | 3.37 | 2.25 |

### Spiking Efficiency
- Total FLOPs: 296,915,968
- Spike event fire rates: 0.463 / 0.169 / 0.138 / 0.079 (by module)
- Effective SynOPS: 61.5M ops/s (vs 296.9M FLOPs = 79% savings from sparsity)

### Issues Fixed
- `data_folder` in vctk.yaml: `./voicebank` → `../../data`
- `accelerator` in vctk.yaml: `gpu` → `auto` (CPU fallback)
- `strategy` in vctk.yaml: `ddp_find_unused_parameters_true` → `auto`
- `num_workers` hardcoded as 8 in DataLoader → 0 (h5py not picklable on Windows)
- Run from `egs/voicebank/` with `PYTHONPATH` set to repo root; use `--device_num 1` not `--devices 0`

---

## Experiment 1: ONNX Export of Pretrained Model
- Date: 2026-04-12
- Script: `python export/export_to_onnx.py --ckpt_path egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt --output_path export/dpsnn_pretrained.onnx`
- Status: **SUCCEEDED**

### Export Result
- [x] **SUCCEEDED** — proceed with DPSNN as primary target

### Bugs Fixed to Enable Export
1. **`aten::col2im` unsupported** (ONNX opset 13): `nn.Fold` in the overlap-add step of
   `dp_binary_net.py` replaced with a loop of `F.pad + sum`. Mathematically equivalent.
2. **`Where` node type mismatch**: `torch.where(x > threshold, x, 0.0)` — the literal `0.0`
   is Python float64 → ONNX double. Fixed to `torch.zeros_like(x)` (float32).

### TracerWarnings (non-blocking)
- `torch.tensor(noisy_x.shape[-1])` — constant; safe to ignore.
- `event_rates` float conversion — event rates baked as constants at this input_dim;
  safe to ignore since input_dim is fixed at deployment time.
- `onnx::Slice` constant folding — cosmetic; does not affect correctness.

### File Size
- FP32 ONNX: **8,229.6 KB (~8.0 MB)**
- Input shape: (1, 16160) — 1s @ 16kHz + 10ms context prefix
- Graph: 399 time steps unrolled (static graph, fixed input length)

---

## Experiment 2: ONNX Numerical Validation
- Date: 2026-04-12
- Script: `python export/validate_onnx.py --ckpt_path egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt --onnx_path export/dpsnn_pretrained.onnx`
- Status: **PASS**

### Validation Result
- Max absolute difference (PyTorch vs ONNX Runtime): **4.01e-05**
- [x] **PASS** (diff < 1e-3)

---

## Decision After Phase 1
- [x] **DPSNN path** — ONNX export works, numerical validation passed (diff=4.01e-05)
- [ ] ~~Conv-TasNet fallback~~ — not needed; ONNX export succeeded

**Summary:** Both blocking risks resolved. Spiking neurons (PLIFNode, ALIFNode) export cleanly.
The 8MB FP32 model at N=B=H=256 is too large for STM32 (2MB Flash), but confirms the
pipeline works end-to-end. Proceed to Phase 2: simplified DPSNN (SCNN-only, N=B=H=128).

- Next step: Phase 3 — train SCNN-only variant at N=B=H=128

---

## Phase 3: Simplified DPSNN Training

### Model Variant: SCNN-only, N=B=H=128

| Property | Value |
|---|---|
| Architecture | StreamSpikeNet, scnn_only=True |
| N / B / H | 128 / 128 / 128 |
| X (blocks) | 1 |
| L / stride | 80 / 40 (5ms frame, 2.5ms hop) |
| context_dur | 0.01 s |
| Parameters | **57,476** |
| vs. pretrained | 90,756 full → 57,476 SCNN-only → 13,644 (pretrained N=B=H=256 = ~1M+) |

### Parameter Breakdown (N=B=H=128)
- Full model (SCNN+SRNN): 90,756 params
- SCNN-only: 57,476 params (~37% fewer than full 128-channel variant)

---

## Experiment 3: SCNN-only N=B=H=128 Training
- Date: 2026-04-12
- Config: N=128, B=128, H=128, L=80, stride=40, context_dur=0.01, X=1, scnn_only=True
- Status: **IN PROGRESS**

### Command
```bash
cd egs/voicebank
PYTHONPATH=../../ python -u vctk_trainer.py --config vctk.yaml \
    -L 80 --stride 40 -N 128 -B 128 -H 128 \
    --context_dur 0.01 --frame_dur 0.5 --max_epochs 200 -X 1 --lr 1e-2 \
    --device_num 1 --scnn_only --batch_size 64
```

### Notes
- Default batch_size=1024 OOM'd (8GB GPU): 399 unrolled steps → padded_frames = 399×(1024,16000)×4B ≈ 26 GB.
- Reduced frame_dur 1.0→0.5 (199 time steps vs 399) and batch_size→64.
- Epochs 0-1: ~27 min each. Epochs 2+: ~12 min each (CUDA warmup). Total: ~42 hours.
- Input_dim=8160 (0.5s + 10ms context); 57,472 samples/epoch (random-hop ×5 per file).
- EarlyStopping is commented out in vctk_trainer.py — runs all 200 epochs.
- Best checkpoint (as of epoch 38): `lightning_logs/version_8/checkpoints/epoch=31-val_loss=91.5398-val_sisnr=-8.5645.ckpt`

### Epoch Progress
| Epoch | val_loss | val_sisnr | Notes |
|---|---|---|---|
| 0 | 94.40 | -5.61 dB | random init |
| 1 | 94.20 | -5.81 dB | |
| 2 | 95.00 | -5.04 dB | |
| 3 | 93.20 | -6.77 dB | |
| 4 | 97.60 | -2.45 dB | spike/instability |
| 5 | 93.10 | -6.97 dB | |
| 6 | 92.80 | -7.21 dB | |
| 7 | 95.30 | -4.72 dB | spike |
| 8 | 92.70 | -7.35 dB | |
| 9–11 | ~92.7 | ~-7.3 dB | steady improvement |
| 12 | 92.50 | -7.57 dB | |
| 13 | 93.50 | -6.60 dB | |
| 14–17 | ~92.1 | ~-7.9 dB | |
| 18 | 92.00 | -8.09 dB | |
| 19–25 | ~92.3 | ~-7.7 dB | some fluctuation |
| 26 | 91.90 | -8.21 dB | |
| 27–29 | ~92–93 | ~-7 dB | fluctuation |
| 30 | 91.78 | -8.33 dB | |
| 31 | 91.54 | **-8.56 dB** | **best checkpoint ↑** |
| 32–33 | ~91.7 | ~-8.4 dB | |
| 34–38 | ~92–93 | ~-7.8 dB | fluctuation |

### Results
*Pending training completion.*

| Metric | Noisy (input) | Enhanced | Clean (reference) |
|--------|--------------|---------|------------------|
| SI-SNR (dB) | — | — | — |
| PESQ (wb) | — | — | 4.64 |
| STOI | — | — | 1.00 |

---

## Phase 4: Conv-TasNet Baseline

### Model Variant: N=48, B=48, H=96, P=3, tcn_depth=3, tcn_repeats=1

| Property | Value |
|---|---|
| Architecture | Conv-TasNet (Luo & Mesgarani 2018) |
| N / B / H | 48 / 48 / 96 |
| TCN blocks | depth=3, repeats=1 (3 blocks, dilations 1,2,4) |
| L / stride | 80 / 40 |
| Parameters | **56,839** (matched to SCNN-only 57,476) |
| Implementation | `convtasnet/model.py` |

### Training Command
```bash
cd egs/voicebank
PYTHONPATH=../../ python -u vctk_trainer.py --config vctk.yaml \
    -L 80 --stride 40 -N 48 -B 48 -H 96 --P 3 --tcn_depth 3 --tcn_repeats 1 \
    --context_dur 0.01 --frame_dur 0.5 --max_epochs 200 -X 1 --lr 1e-3 \
    --device_num 1 --model convtasnet --batch_size 64
```

### Results
*Pending training.*

| Metric | Noisy | Enhanced | Clean |
|--------|-------|---------|-------|
| SI-SNR (dB) | — | — | — |
| PESQ (wb) | — | — | 4.64 |
| STOI | — | — | 1.00 |

---

## Phase 5/6/7: Deployment Footprint Analysis

Measured on pretrained FP32 model; SCNN-only estimates are projections.

### Model Size vs STM32 B-U585I-IOT02A Limits (2 MB Flash, 786 KB RAM)

| Model | Params | FP32 weights | INT8 weights | ONNX file |
|-------|--------|-------------|-------------|-----------|
| Pretrained (N=256) | 372,996 | 1,457 KB | ~364 KB | 8,230 KB |
| SCNN-only (N=128) | 57,476 | ~225 KB | ~56 KB | ~TBD |
| Conv-TasNet (N=48) | 56,839 | ~222 KB | ~56 KB | ~TBD |

**Key finding:** Weights alone fit comfortably in Flash for all variants.
The ONNX file is large due to graph unrolling (199 time steps), but
X-CUBE-AI generates compact C code from the model — file size is not
the deployment constraint.

### Estimated RAM Budget (SCNN-only, INT8)
| Item | KB |
|------|----|
| Model activations (estimated) | ~200 |
| Input buffer (8160 float32) | 32 |
| Output buffer (8000 float32) | 31 |
| Code + stack | ~200 |
| **Total** | **~463 KB** |

Within the 786 KB limit. Use `python tools/estimate_footprint.py <onnx>` to
update when quantized models are available.

### Key Deployment Risk
The ONNX graph has 199 unrolled time steps. X-CUBE-AI handles standard
Conv/Linear layers well but may not support all ONNX ops generated by
the spiking model. Run X-CUBE-AI Analyse step before committing to this
path. Conv-TasNet is the fallback (standard CNN ops, guaranteed support).

---
