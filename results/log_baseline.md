# Baseline & Pipeline Validation

Covers the pretrained N=256 model: inference, ONNX export, numerical validation,
and the initial decision to proceed with DPSNN over Conv-TasNet.

---

## Setup

- Repo: dpsnn-embedded-deployment
- Strategy: Validate-First — test ONNX export on the pretrained model before training simplified variants

---

## Experiment 0: Pretrained Model Inference

- Date: 2026-04-12
- Checkpoint: `egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt`
- Params: N=256, B=256, H=256, L=80, stride=40, context_dur=0.01, X=1
- Status: **COMPLETED** (CPU, 28 min, 824/824 samples)

### Command

```bash
cd egs/voicebank
PYTHONPATH=../../ python -u vctk_trainer.py --config vctk.yaml \
    -L 80 --stride 40 -N 256 -B 256 -H 256 \
    --context_dur 0.01 --max_epochs 500 -X 1 --lr 1e-2 \
    --device_num 1 \
    --test_ckpt_path ./epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt
```

### Desktop Baseline Metrics (full 824-sample test set, re-confirmed 2026-04-28)

| Metric | Noisy (input) | Enhanced | Clean (reference) |
|--------|--------------|---------|------------------|
| SI-SNR (dB) | 8.44 | **18.08** | — |
| PESQ (wb) | 1.971 | **2.264** | 4.64 |
| STOI | 0.921 | **0.925** | 1.00 |
| Composite OVRL | 2.637 | **2.798** | — |
| Composite SIG | 3.357 | **3.366** | — |
| Composite BAK | 2.445 | **2.246** | — |

### Spiking Efficiency

- Total FLOPs: 296,915,968
- Spike event fire rates: 0.463 / 0.169 / 0.138 / 0.079 (by module)
- Effective SynOPS: 61.5M ops/s (vs 296.9M FLOPs = 79% savings from sparsity)

### Issues Fixed (Windows / environment)

- `data_folder` in vctk.yaml: `./voicebank` → `../../data`
- `accelerator` in vctk.yaml: `gpu` → `auto` (CPU fallback)
- `strategy` in vctk.yaml: `ddp_find_unused_parameters_true` → `auto`
- `num_workers` hardcoded as 8 in DataLoader → 0 (h5py not picklable on Windows)
- Run from `egs/voicebank/` with `PYTHONPATH` set to repo root; use `--device_num 1` not `--devices 0`

---

## Experiment 1: ONNX Export of Pretrained Model

- Date: 2026-04-28 (re-run on fresh machine; original was 2026-04-12)
- Script: `python export/export_to_onnx.py --ckpt_path egs/voicebank/epoch=478-... --output_path export/dpsnn_pretrained.onnx`
- Status: **SUCCEEDED**

### Bugs Fixed to Enable Export

1. **`aten::col2im` unsupported** (ONNX opset 13): `nn.Fold` in the overlap-add step of
   `dp_binary_net.py` replaced with a loop of `F.pad + sum`. Mathematically equivalent.
2. **`Where` node type mismatch**: `torch.where(x > threshold, x, 0.0)` — the literal `0.0`
   is Python float64 → ONNX double. Fixed to `torch.zeros_like(x)` (float32).

### TracerWarnings (non-blocking)

- `torch.tensor(noisy_x.shape[-1])` — constant; safe to ignore.
- `event_rates` float conversion — baked as constants at this input_dim; safe since input_dim is fixed at deployment time.
- `onnx::Slice` constant folding — cosmetic, does not affect correctness.

### File

- FP32 ONNX: **8,229.6 KB (~8.0 MB)**
- Input shape: (1, 16160) — 1s @ 16 kHz + 10ms context prefix
- Graph: 399 time steps unrolled (static graph, fixed input length)

---

## Experiment 2: ONNX Numerical Validation

- Date: 2026-04-12
- Script: `python export/validate_onnx.py --ckpt_path ... --onnx_path export/dpsnn_pretrained.onnx`
- Status: **PASS**

### Result

- Max absolute difference (PyTorch vs ONNX Runtime): **4.01e-05**
- PASS (threshold < 1e-3) ✓

---

## Decision After Phase 1

- **DPSNN path chosen** — ONNX export works, numerical validation passed (diff=4.01e-05)
- **Conv-TasNet fallback dropped** — not needed; ONNX export succeeded

The 8 MB FP32 model at N=B=H=256 is too large for STM32 (2 MB Flash), but confirms the
pipeline works end-to-end. Spiking neurons (PLIFNode, ALIFNode) export cleanly.

Proceed to: train SCNN-only variant at N=B=H=128.

---

## Phase 4: Conv-TasNet Baseline (Dropped)

Considered as a fallback if DPSNN ONNX export failed. Dropped after Experiment 1 succeeded.
Kept in `convtasnet/model.py` for reference.

### Planned variant: N=48, B=48, H=96, P=3, tcn_depth=3, tcn_repeats=1

| Property | Value |
|---|---|
| Architecture | Conv-TasNet (Luo & Mesgarani 2018) |
| Parameters | **56,839** (matched to SCNN-only 57,476) |
| Status | **Dropped** — DPSNN pipeline validated; Conv-TasNet not needed |
