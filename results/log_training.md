# SCNN-only N=128 Training

Covers the design of the SCNN-only variant, the first failed training run,
the supervisor reset, and the successful retrain that produced the deployment model.

---

## Model Variant: SCNN-only, N=B=H=128

| Property | Value |
|---|---|
| Architecture | StreamSpikeNet, scnn_only=True |
| N / B / H | 128 / 128 / 128 |
| X (blocks) | 1 |
| L / stride | 80 / 40 (5 ms frame, 2.5 ms hop) |
| context_dur | 0.01 s |
| Parameters | **57,476** |
| Full model (SCNN+SRNN) | 90,756 params |
| Reduction vs full | ~37% fewer params by removing SRNN path |

SCNN-only flag removes the SRNN dense+recurrent path while keeping the SCNN
(depthwise SpikeConv1d + PLIFNode neurons). Justified by Table 5 of the DPSNN paper:
SCNN contributes more to SI-SNR than SRNN.

---

## Experiment 3: First Training Run — FAILED (frame_dur=0.5)

- Date: 2026-04-12 → 2026-04-15
- Config: N=128, B=128, H=128, L=80, stride=40, context_dur=0.01, **frame_dur=0.5**, X=1, scnn_only=True
- batch_size=64, lr=1e-2, max_epochs=200 (EarlyStopping commented out)
- Status: **COMPLETED but DISCARDED** — root cause identified, checkpoints deleted

### Command

```bash
cd egs/voicebank
PYTHONPATH=../../ python -u vctk_trainer.py --config vctk.yaml \
    -L 80 --stride 40 -N 128 -B 128 -H 128 \
    --context_dur 0.01 --frame_dur 0.5 --max_epochs 200 -X 1 --lr 1e-2 \
    --device_num 1 --scnn_only --batch_size 64
```

### Notes

- Default batch_size=1024 OOM'd (8 GB GPU): 399 unrolled steps × (1024, 16000) × 4B ≈ 26 GB.
  Reduced frame_dur 1.0→0.5 (199 time steps) and batch_size→64.
- Epochs 0–1: ~27 min each. Epochs 2+: ~12 min each (CUDA warmup). Total: ~42 hours.
- Training resumed into version_9 after crash at epoch 156 batch 282.

### Epoch Progress (selected)

| Epoch | val_loss | val_sisnr | Notes |
|---|---|---|---|
| 0 | 94.40 | -5.61 dB | random init |
| 31 | 91.54 | -8.56 dB | first notable checkpoint |
| 59 | 91.07 | -9.03 dB | new best |
| 88 | 91.00 | -9.08 dB | new best |
| 105 | 90.85 | -9.24 dB | new best |
| 132 | 90.82 | -9.25 dB | new best |
| 139 | 90.61 | -9.47 dB | new best |
| 163 | **90.55** | **-9.52 dB** | **final best** (ckpt: 90.5477, v9) |
| 199 | 90.70 | -9.40 dB | TRAINING COMPLETE |

Best checkpoint: `lightning_logs/version_9/checkpoints/epoch=163-val_loss=90.5477-val_sisnr=-9.5152.ckpt`

### Root Cause of Failure

`frame_dur=0.5` gives the model only **199 time steps** of context per forward pass.
The upstream paper and pretrained model use `frame_dur=1.0` (399 steps). The model
simply does not have enough temporal context. The 9.52 dB ceiling after 200 epochs is
a direct consequence of halved temporal context, not a channel-width or dataset problem.

Confirmed by Experiment 5: with frame_dur=1.0, epoch 0 alone reaches 15.4 dB.

---

## Experiment 4: SCNN-only ONNX Export & Validation (of discarded checkpoint)

- Date: 2026-04-15
- Checkpoint: `lightning_logs/version_9/checkpoints/epoch=163-...-9.5152.ckpt`
- Status: **PASS** — validated the export pipeline works for SCNN-only graphs

### Results

- ONNX file: 2,943 KB (FP32), input shape (1, 8160) — 0.5s @ 16 kHz + 10ms context
- Max abs diff (PyTorch vs ORT): **9.16e-05** (PASS < 1e-3)
- Graph: 199 time steps unrolled

Note: this export was for the discarded 9.52 dB checkpoint. The final deployment ONNX
(`export/dpsnn_scnn128.onnx`) was produced after Experiment 5 with the 17.60 dB checkpoint.

---

## Supervisor Reset (2026-04-27)

Meeting outcome:
- **9.52 dB run discarded.** Training artifacts deleted.
- **New plan:** retrain with `frame_dur=1.0` (399 steps) and plateau-based EarlyStopping.
  Target: ≥ 15 dB SI-SNR. Scale to N=256 only if not reached.
- **Conv-TasNet dropped** from scope.
- **Priority while retraining:** INT8-quantize Tao's pretrained N=256 model to validate
  the deployment pipeline (see `log_quantization.md`).

---

## Experiment 5: Successful Retrain — frame_dur=1.0

- Date: 2026-04-28
- Config: N=128, B=128, H=128, L=80, stride=40, context_dur=0.01, **frame_dur=1.0**, X=1, scnn_only=True
- batch_size=32, lr=1e-2, max_epochs=200, EarlyStopping(patience=10)
- Input_dim=16160 (1s @ 16 kHz + 10ms context), 815 batches/epoch

### Key finding

Previous run (9.52 dB best across 200 epochs) used frame_dur=0.5 (199 time steps).
This run uses frame_dur=1.0 (399 steps). **Epoch 0 alone (15.4 dB) already exceeds the entire previous run.**

### Phase 1: Without gradient clipping (version_0, epochs 0–19)

| Epoch | val_loss | val_sisnr | Notes |
|---|---|---|---|
| 0 | 84.60 | -15.4 dB | already above 15 dB target |
| 3 | **83.20** | **-16.8 dB** | new best |
| 6 | 82.90 | -17.1 dB | new best |
| 7 | **82.80** | **-17.2 dB** | new best |
| 10 | **82.70** | **-17.3 dB** | new best |
| 11 | 84.00 | -16.0 dB | mse spike (−1.3 dB), patience 1/10 |
| 15 | **82.60** | **-17.4 dB** | new best |
| 18 | 84.60 | -15.4 dB | mse=10.0 (largest spike), patience 3/10 |
| 19 | — | — | mse=52.5 mid-epoch — **training diverged, stopped manually** |

### Fix: gradient clipping added

Added `gradient_clip_val: 1.0` to vctk.yaml. Restarted from `epoch=15` checkpoint —
Lightning restored full optimizer/scheduler state. `gn=1.000` on every batch confirms
clipping is active; gradients were routinely >1.0 before.

### Phase 2: With gradient clipping (version_1, resumed from epoch 15)

| Epoch | val_loss | val_sisnr | Notes |
|---|---|---|---|
| 16 | 82.6315 | -17.37 dB | gn=1.000, mse settling |
| 18 | **82.5947** | **-17.41 dB** | new overall best |
| 20 | **82.50** | **-17.5 dB** | new best |
| 29 | 82.51 | -17.50 dB | new best |
| 30 | **82.48** | **-17.52 dB** | new best |
| 35 | **82.46** | **-17.55 dB** | new best |
| 39 | **82.41** | **-17.60 dB** | **new best — final checkpoint** |
| 48 | 82.4473 | -17.56 dB | top-3, patience 9/10 |
| 49 | 82.50 | -17.5 dB | patience 10/10 → **EarlyStopping fired** |

### Final Result

- **Best checkpoint:** `lightning_logs/version_1/checkpoints/epoch=39-val_loss=82.4127-val_sisnr=-17.5976.ckpt`
- **Best val SI-SNR: 17.60 dB**
- Total epochs trained: 49 (15 without clipping + 34 with clipping)
- Top-3 checkpoints:

| Checkpoint | val_loss | SI-SNR |
|---|---|---|
| epoch=39 | 82.4127 | **17.60 dB** |
| epoch=48 | 82.4473 | 17.56 dB |
| epoch=35 | 82.4632 | 17.55 dB |

- vs Tao pretrained N=256: 17.60 dB vs 18.08 dB — **gap of only 0.48 dB** with 5× fewer params (71.3 K vs ~373 K)
- Supervisor target (≥15–16 dB): **ACHIEVED** ✓

---

## Post-training Pipeline (2026-04-29)

### ONNX Export

`export/dpsnn_scnn128.onnx` — 5630.9 KB, max diff vs PyTorch: **6.68e-05** ✓

```bash
python export/export_to_onnx.py \
    --ckpt_path egs/voicebank/lightning_logs/version_1/checkpoints/epoch=39-val_loss=82.4127-val_sisnr=-17.5976.ckpt \
    --output_path export/dpsnn_scnn128.onnx
```

### Footprint (STM32 B-U585I-IOT02A)

| Resource | Used | Limit | Status |
|---|---|---|---|
| Flash (weights) | 278.5 KB | 2048 KB | ✓ 13.6% |
| RAM (I/O tensors) | 125.6 KB | 786 KB | ✓ 16.0% |
| Peak RAM (w/ intermediates) | TBD | 786 KB | needs X-CUBE-AI Analyse |

### Full Test Set Evaluation (824 utterances, FP32 ONNX)

| Metric | Noisy | SCNN-only N=128 | Pretrained N=256 |
|---|---|---|---|
| SI-SNR (dB) | 8.44 | **17.23** | 18.08 |
| PESQ (wb) | 1.971 | **2.089** | 2.264 |
| STOI | 0.921 | 0.920 | 0.925 |
| Composite OVRL | 2.637 | 2.480 | 2.798 |
| Composite SIG | 3.357 | 2.935 | 3.366 |
| Composite BAK | 2.445 | **2.909** | 2.246 |

**Key observations:**
- SI-SNR gap vs pretrained: **0.85 dB** on test set (val showed 0.48 dB — 0.37 dB generalisation gap)
- STOI flat (0.921→0.920): intelligibility essentially unchanged
- BAK improved vs pretrained (2.909 vs 2.246): stronger background suppression
- Model is 5.2× smaller (71.3 K vs ~373 K params) with only 0.85 dB SI-SNR penalty
