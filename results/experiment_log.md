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
    --context_dur 0.01 --max_epochs 300 -X 1 --lr 1e-2 \
    --device_num 1 --scnn_only
```

### Results
*Pending training completion.*

| Metric | Noisy (input) | Enhanced | Clean (reference) |
|--------|--------------|---------|------------------|
| SI-SNR (dB) | — | — | — |
| PESQ (wb) | — | — | 4.64 |
| STOI | — | — | 1.00 |

---
