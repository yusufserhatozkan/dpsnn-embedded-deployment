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
| Parameters | **71,299** |
| Full model (SCNN+SRNN) | 104,579 params |
| Reduction vs full | ~32% fewer params by removing SRNN path |

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

SI-SNR gap vs pretrained is 0.85 dB on the test set (val showed 0.48 dB — 0.37 dB generalisation gap). STOI is flat (0.921→0.920), intelligibility essentially unchanged. BAK improved vs pretrained (2.909 vs 2.246): stronger background suppression. The model is 5.2× smaller (71.3 K vs ~373 K params) with only 0.85 dB SI-SNR penalty.

---

## Supervisor Feedback: Training Speed-up (2026-05-08)

- Commit: `cbc0e20` (thesis branch)
- Status: **RETRAIN PENDING** — changes pushed, awaiting run on RTX 4060
- Expected per-epoch time: **~7–10 min** (was ~20–30 min before these changes)

### Changes Made

#### `dpsnn/models/dp_binary_net.py`

1. **Removed `readout_threshold`** — deleted the learnable `nn.Parameter` scalar that suppressed readout outputs below a learned threshold.
2. **Removed `torch.where(x > self.readout_threshold, ...)`** in `forward()` — readout membrane potential now flows unsuppressed.
3. **Removed `proj_loss` and `readout_loss`** L1 regularisation terms from `training_step`. These penalised spike event rates to encourage sparsity for neuromorphic targets (Loihi). The STM32 deployment target runs dense INT8 GEMM via X-CUBE-AI and gets zero benefit from spike sparsity, so the penalty only slowed convergence without reward.
4. **New training loss:** `loss = 0.001 * mse_loss + 100 + sisnr_loss` (dropped `+ 0.001*proj_loss + 0.001*readout_loss`).
5. Removed `proj` and `read` TensorBoard log entries.

#### `egs/voicebank/vctk.yaml`

- Changed `# precision: 32` (commented out) → `precision: bf16-mixed` — enables mixed-precision training for ~2× speedup on RTX 40-series (bf16 compute, FP32 weights in checkpoint).

#### `README.md`

- Updated example training command batch_size from 32 → 64.

### Deployment Safety

These changes do **not** affect the exported model. The deployed ONNX uses fixed input shape `(1, 16160)`, the same 6 weight tensors, and the same architecture. Flash, RAM, and latency on STM32 are identical.

### Training Command for Retrain

```bash
cd egs/voicebank
PYTHONPATH=../../ python -u vctk_trainer.py --config vctk.yaml \
    -L 80 --stride 40 -N 128 -B 128 -H 128 \
    --context_dur 0.01 --frame_dur 1.0 --max_epochs 200 -X 1 --lr 1e-2 \
    --device_num 1 --scnn_only --batch_size 64
```

Fallbacks: drop to `--batch_size 48` if OOM; change `bf16-mixed` → `16-mixed` in `vctk.yaml` if NaN gradients appear.

### Expected Outcome

Target: **≥ 17.23 dB test SI-SNR** (matching Experiment 5). ±0.3 dB variation from bf16 numeric noise is normal. EarlyStopping patience=10 will fire automatically.

---

## Experiment 7: bf16 Retrain — 100 Epochs, No Sparsity Losses

- Date: 2026-05-08 23:36 → 2026-05-09 ~19:15
- Config: N=128, B=128, H=128, L=80, stride=40, context_dur=0.01, frame_dur=1.0, X=1, scnn_only=True
- batch_size=64, lr=1e-2, max_epochs=100, **no EarlyStopping**, precision=bf16-mixed
- GPU: RTX 4060, ~11 min 45 sec/epoch
- Checkpoint dir: `egs/voicebank/lightning_logs/version_2/checkpoints/`

### Post-training crash (fixed)

Training completed all 100 epochs. The post-training test call crashed with `TypeError: Got unsupported ScalarType BFloat16` — DNSMOS metric called `.numpy()` on a bf16 tensor. Fixed by adding `.float()` cast to `enhanced_tensor`, `noisy_tensor`, and `clean` after normalization in `on_test_batch_end`. Test re-run succeeded on the same best checkpoint.

### Epoch Log (all 100 epochs)

See `results/exp7_epoch_log.md` for the full per-epoch table. Key milestones:

| Epoch | val_loss | val_sisnr | notes |
|---|---|---|---|
| 0 | 85.1660 | −14.83 dB | start |
| 7 | 82.8171 | −17.18 dB | first >17 dB |
| 30 | 82.3773 | −17.62 dB | surpassed Exp 5 val best |
| 51 | 82.2768 | −17.73 dB | new best |
| 70 | 82.2567 | −17.75 dB | new best |
| 82 | 82.2402 | −17.76 dB | new best |
| **87** | **82.2283** | **−17.775 dB** | **best checkpoint** |
| 99 | 82.2813 | −17.72 dB | final epoch |

### Best Checkpoint

`lightning_logs/version_2/checkpoints/epoch=87-val_loss=82.2283-val_sisnr=-17.7750.ckpt`

### Full Test Set Results (824 utterances)

| Metric | Noisy | **Exp 7 (this run)** | Exp 5 (previous) | Pretrained N=256 |
|---|---|---|---|---|
| SI-SNR (dB) | 8.44 | **17.41** | 17.23 | 18.08 |
| PESQ (wb) | 1.971 | **2.115** | 2.089 | 2.264 |
| STOI | 0.921 | **0.923** | 0.920 | 0.925 |
| Composite OVRL | 2.637 | **2.591** | 2.480 | 2.798 |
| Composite SIG | 3.357 | **3.115** | 2.935 | 3.366 |
| Composite BAK | 2.445 | **2.947** | 2.909 | 2.246 |
| DNSMOS OVRL | 2.684 | **2.712** | — | — |
| DNSMOS SIG | 3.324 | **3.186** | — | — |
| DNSMOS BAK | 3.111 | **3.460** | — | — |

> **Composite-score provenance.** The "Composite" OVRL/SIG/BAK above are from
> the in-training PyTorch test callback. The standalone ONNX eval of the
> deployed *streaming* model ([`results/eval_exp7_streaming.txt`](eval_exp7_streaming.txt))
> reports slightly different composites (OVRL 2.647 / SIG 3.194 / BAK 2.963)
> while SI-SNR (17.42), PESQ (2.149) and STOI (0.923) match to the digit. The
> deterministic metrics agree; the composite gap is a measurement-pipeline
> artefact (PyTorch test callback vs `eval_onnx`/`eval_streaming` normalisation),
> not a difference in the audio — batch and streaming outputs are bit-identical
> (max abs diff = 0.0).

**vs Experiment 5:**

| SI-SNR | PESQ | STOI | OVRL | SIG | BAK |
|---|---|---|---|---|---|
| +0.18 dB | +0.026 | +0.003 | +0.111 | +0.180 | +0.038 |

All metrics improved over Experiment 5 — bf16 training + longer run (100 vs 49 epochs) both contributed. SI-SNR gap vs pretrained N=256 narrowed from 0.85 dB (Exp 5) to **0.67 dB** (Exp 7) with the same 71.3 K params. SIG improved significantly (+0.18) and BAK held up (+0.038). STOI is near-identical — intelligibility unchanged. Epoch pace stable at ~11:45/epoch, bottlenecked by the 399 sequential SNN time steps rather than GPU compute.

---

## Post-training Pipeline — Experiment 7 (2026-05-13)

### ONNX Export

- Date: 2026-05-13
- Script: `python export/export_to_onnx.py --ckpt_path egs/voicebank/lightning_logs/version_2/checkpoints/epoch=87-val_loss=82.2283-val_sisnr=-17.7750.ckpt --output_path export/dpsnn_scnn128_exp7.onnx`
- Status: **SUCCEEDED**
- Output: `export/dpsnn_scnn128_exp7.onnx` — **5437.1 KB**
- Input shape: (1, 16160) — 1s @ 16 kHz + 10ms context prefix
- Graph: 399 time steps unrolled
- TracerWarnings: same as Exp 5 — all safe to ignore (constant folding, float conversion)

### INT8 Quantization (spike-aware pct95)

- Date: 2026-05-13
- Script: `python export/quantize_spike_aware_correct.py` with `--relu_percentile 95.0 --n_calib 50`
- Output: `export/dpsnn_scnn128_exp7_int8_pct95.onnx`
- Spike map reused from Exp 5 (`dpsnn_scnn128.onnx.spike_map.json`) — valid since architecture is identical

**Calibration results:**
- SAFE nodes: 1201 | SPIKE nodes: 1201
- ReLU tensors: 403 | Sigmoid tensors: 399
- Absolute max across all time steps: 10.7642
- 95th-percentile max (used for scale): **9.2446**
- Shared ReLU scale: 0.036253, zp: −128
- Fixed Sigmoid INT8 scale: 0.003922, zp: −128
- 6 unique weight tensors quantized to INT8

Note: ONNX file size FP32→INT8 appears to grow (5437 KB → 6318 KB) because QDQ format
adds Q/DQ nodes to the graph. Actual weight Flash is smaller — see footprint below.

### Footprint (STM32 B-U585I-IOT02A)

| Resource | Value | Limit | Status |
|---|---|---|---|
| Weight bytes (Flash) | **95.9 KB** | 2048 KB | ✓ 4.7% |
| RAM (I/O tensors) | **125.6 KB** | 786 KB | ✓ 16.0% |
| Peak RAM (w/ intermediates) | TBD | 786 KB | needs X-CUBE-AI Analyse |

Dtype breakdown: 70.0 KB int8 weights + 15.7 KB int64 constants + 10.2 KB float32 biases/scales.

Note: Exp 7 reports 95.9 KB vs Exp 5's 140.9 KB — difference is in how scale params are counted
by `estimate_footprint.py`. X-CUBE-AI Analyse will give the authoritative figure.

### Test Audio Extraction

- Date: 2026-05-13
- Command: `python tools/extract_test_audio.py --hdf5_path data/results/save/test.hdf5 --output_dir deploy/test_audio --n_files 3`
- Output dir: `deploy/test_audio/`
- Format: float32, native endian, 1 channel, 16000 Hz
- Manifest: `deploy/test_audio/manifest.tsv`

| Index | Speaker | Samples | Duration | Files |
|---|---|---|---|---|
| 0 | p232_001 | 27,861 | 1.74s | `0000_noisy.bin`, `0000_clean.bin`, `.wav` pairs |
| 1 | p232_002 | 43,444 | 2.72s | `0001_noisy.bin`, `0001_clean.bin`, `.wav` pairs |
| 2 | p232_003 | 114,958 | 7.18s | `0002_noisy.bin`, `0002_clean.bin`, `.wav` pairs |

For STM32 firmware: cast buffer to `float*` and pass directly to inference.
File 0 (1.74s, ~109 KB) is the most practical for initial embedding — fits in Flash alongside model weights.

### Outcome of the deployment path (added 2026-05-26)

The Exp 7 INT8 batch ONNX was successfully analysed by X-CUBE-AI but the resulting
activation footprint (1.37 MB) overshoots the 786 KB SRAM budget. Root cause was
graph structure (1,196 × 64 KB OLA buffers from the 399-step unroll), not dtype —
INT8 saved < 0.1 % on activation RAM because the SPIKE-path FP32 islands are
unquantizable. See `log_xcube_ai.md` §4 for the per-tensor breakdown.

Two solutions were built on top of Exp 7:

1. **Streaming export** (`export/export_streaming.py`) — single-frame model + explicit
   recurrent state I/O, 46 KB activations, **numerically identical to batch** (max abs
   diff = 0.0, 824-set SI-SNR 17.42 dB vs batch 17.41 dB).
2. **Exp 9 retrain** — BatchNorm + stride=kernel=80 → 501 KiB batch activations,
   16.486 dB (−0.93 dB).

The deployed firmware uses Exp 7 streaming FP32. On-device MCU SI-SNR matches the
Python ONNX reference to 0.00 dB on real speech (p232_009, 4.16 s); RTF = 2.459×.
Full hardware trace in `log_xcube_ai.md` §10.

---

## Experiment 9 — BatchNorm + Non-Overlapping Decoder

**Date:** 2026-05-14 to 2026-05-15
**Goal:** Fix the two structural barriers blocking Exp7 from fitting in 786 KB SRAM:
1. Replace `ChannelWiseLayerNorm` → `BatchNorm1d` (folds at INT8 quant, eliminates FP32 island)
2. Set `stride=kernel=80` (no overlap → no OLA chain → eliminates the 1,196 × 64 KB OLA buffers)

### Config

| Parameter | Value |
| --- | --- |
| Architecture | SCNN-only, X=1 |
| N, B, H | 128 |
| L (kernel) | 80 |
| stride | **80** (was 40 in Exp7) |
| frame_dur | 1.0 s |
| norm_type | **bnorm** (was lnorm) |
| time_steps | **200** (was 399) |
| batch_size | 64 |
| precision | bf16-mixed |
| epochs | 100 |

### Training Curve (val_loss = 100 − val_sisnr)

- Epoch 0: val_loss=86.21 → Epoch 99: **val_loss=83.18** (best)
- Convergence was steady; best checkpoint at final epoch, suggesting model could benefit from more epochs
- Per-epoch log: `results/exp9_bn_nooverlap_epoch_log.md`
- Checkpoint: `egs/voicebank/lightning_logs/version_4/checkpoints/epoch=99-val_loss=83.1804-val_sisnr=-16.8212.ckpt`

### Test-Set Evaluation (824-sample VoiceBank-DEMAND test set)

| Metric | Noisy | Exp9 enhanced | Exp7 (reference) |
| --- | --- | --- | --- |
| SI-SNR (dB) | 8.44 | **16.486** | 17.41 |
| PESQ (wb) | 1.971 | 1.987 | 2.149 |
| STOI | 0.921 | 0.917 | 0.923 |
| OVRL | 2.637 | 2.509 | 2.584 |
| SIG | 3.357 | 3.158 | — |
| BAK | 2.445 | 2.718 | — |

**SI-SNR drop vs Exp7: −0.93 dB** — within the ≤1 dB deployment budget. Success criterion (≥16.0 dB) met.

### Deployment Footprint (stedgeai analyze, stm32u5, `-O balanced -c none`)

| Model | weights | activations | Status |
| --- | --- | --- | --- |
| Exp9 FP32 | 280 KiB | **501 KiB** | ✓ FITS (285 KiB headroom) |
| Exp9 true-INT8 | 280 KiB | **501 KiB** | ✓ FITS (spike-path FP32 limits savings) |

**This is the first batch-mode model to fit within the 786 KiB SRAM budget.** Exp7's floor was 1.37 MB due to the OLA chain; Exp9 eliminates that entirely by making stride=kernel.

### Next Step

Run `stedgeai generate` on `C:/ai/dpsnn_exp9_bn_nooverlap.onnx` to produce C code for STM32U585.
