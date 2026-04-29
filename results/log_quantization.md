# INT8 Quantization Study

Covers all INT8 quantization experiments: the initial failures on the pretrained N=256 model,
the root cause investigation, and the final quantization of the SCNN-only N=128 deployment model.

---

## Experiment 1b: Standard INT8 on Pretrained N=256 — CATASTROPHIC FAILURE

- Date: 2026-04-28
- Method: onnxruntime `quantize_static`, QDQ format, per-tensor MinMax, 50 calibration samples
- Model: pretrained N=256 (`export/dpsnn_pretrained.onnx`)

### Results (full 824-sample test set)

| Metric | Noisy | FP32 | INT8 | Drop |
|---|---|---|---|---|
| SI-SNR | 8.44 dB | 18.08 dB | **6.53 dB** | −11.55 dB |
| PESQ (wb) | 1.971 | 2.264 | **1.321** | −0.943 |
| STOI | 0.921 | 0.925 | **0.831** | −0.094 |

The INT8 model (6.53 dB) is **worse than the raw noisy input** (8.44 dB).

### INT8 v2: per-channel + Percentile (2026-04-28)

- Method: per_channel=True, CalibrationMethod.Percentile (99.999), ActivationSymmetric=False
- Result: SI-SNR=6.45 dB, PESQ=1.331, STOI=0.828 — **equally catastrophic**
- Conclusion: calibration strategy is irrelevant; the failure is architectural

### Root Cause

DPSNN spike activations are binary (0 or 1). Standard INT8 quantization assumes smooth,
approximately-Gaussian activations. The 0/1 distribution maps to a tiny subset of INT8's
256 levels — any numerical noise from quantization completely destroys spike patterns.

The deeper mechanism:
1. Spiking neuron: `membrane += conv_output; spike = (membrane > threshold)`
2. Even a tiny INT8 rounding error in `conv_output` shifts `membrane` across the threshold
   boundary → spike pattern flips.
3. Wrong spike feeds back into the next Conv → errors compound over 399 time steps.

This is a fundamental incompatibility between SNNs and standard INT8 PTQ, not a
calibration problem. **This is a key thesis finding.**

---

## Experiment 1c: Extended INT8 Strategies — Pretrained N=256

- Date: 2026-04-29
- Goal: systematically locate where the failure occurs

Snapshot metric = SI-SNR of enhanced vs noisy input (NOT vs clean reference;
use only as a relative comparison within this table).

### Results

| Strategy | Calibration | Ops quantized | Avg drop | Std | Max drop |
|---|---|---|---|---|---|
| v1: MinMax per-tensor | MinMax | All | −11.55 dB (full eval) | — | — |
| v2: Percentile per-channel | Percentile 99.999 | All | −11.63 dB (full eval) | — | — |
| A: Entropy, full graph | Entropy | All | −0.61 dB | 3.66 | 5.68 dB |
| B: Mixed precision | Entropy | Conv+Gemm only, spike ops FP32 | −0.43 dB | 5.64 | 18.56 dB |
| **C: Weight-only** | None | Weights only, acts FP32 | **−0.30 dB** | 0.42 | 0.95 dB |

### Findings

**Weight-only INT8 (Strategy C):** Near-lossless. Max drop 0.95 dB. Quantizing weights alone
is safe — membrane potential computation stays in FP32, so threshold-boundary errors never occur.
This is the viable deployment path.

**Entropy calibration (Strategy A):** Better average than MinMax/Percentile but std=3.66 and
max drop=5.68 dB. Calibration strategy does not fix the architectural problem.

**Mixed precision (Strategy B):** Average −0.43 dB but std=5.64 and max drop=18.56 dB.
The Conv layers feeding spike neurons still cause threshold-flipping cascades even when
only weights are quantized, because the `conv_output` fed to `membrane` still has INT8 error.

### Revised root cause (sharpened)

The SNN-INT8 incompatibility is specifically located at the **activation precision of the
membrane potential**, not weight precision. Weight-only INT8 avoids this entirely.

---

## Experiment 6: Corrected INT8 Quantization of SCNN-only N=128

- Date: 2026-04-29
- Model: `export/dpsnn_scnn128.onnx` (17.23 dB FP32)

### Supervisor's Correct Rule (meeting 2026-04-29)

Based on IoT lecture Week 5 Part 2, slide 14-16:
- **Quantize:** all Conv/Gemm weights + activation outputs **only at ReLU (encoder) and Sigmoid (mask)**
- **Do NOT quantize:** membrane potentials (conv output before spike threshold) or spike outputs (already 0/1)
- Formula: `q_A2 = (S_A2 / (S_W2 · S_A1)) · ReLU(q_W2 · q_A1)`
  The INT32 partial sum is NOT clipped to INT8.

### Step 1 — Graph Analysis (`export/map_spike_tensors.py`)

BFS from each Conv/Gemm output, classifying as SAFE (feeds ReLU/Sigmoid) or SPIKE (feeds threshold).

Result on `dpsnn_scnn128.onnx`:
- **1201 SAFE:** encoder_1d (401×), mask (401×), decoder_1d (399×)
- **1201 SPIKE:** proj/BinaryConv (403×), dconv (399×), dense (399×)
- **0 UNKNOWN**

Map saved: `export/dpsnn_scnn128.onnx.spike_map.json`

### Attempt v1 — spike_map exclusion, ORT default ops (FAILED)

| Metric | FP32 | INT8 v1 | Drop |
|---|---|---|---|
| SI-SNR | 17.23 dB | 7.64 dB | **−9.59 dB** |

Root cause: `quant_pre_process` decomposes `LayerNormalization` into primitives (ReduceMean,
Sub, Mul, Add). `quantize_static` then inserts QDQ pairs around these decomposed ops, corrupting
the LayerNorm output that feeds the spike threshold.

### Attempt v2 — spike_map + op_types=['Conv','Gemm','ConvTranspose'] (FAILED)

```bash
python export/quantize_int8.py \
    --onnx_path export/dpsnn_scnn128.onnx \
    --output_path export/dpsnn_scnn128_int8_corrected_v2.onnx \
    --hdf5_path data/results/save/test.hdf5 \
    --spike_map export/dpsnn_scnn128.onnx.spike_map.json \
    --n_calib 50 --calibrate_method MinMax \
    --op_types_to_quantize Conv Gemm ConvTranspose
```

| Metric | FP32 | INT8 v2 | Drop |
|---|---|---|---|
| SI-SNR | 17.23 dB | 7.92 dB | **−9.31 dB** |

Root cause (confirmed by binary search): ORT places the output QDQ node **between Conv and ReLU**
(quantizing the partial sum / membrane potential equivalent). Each SAFE component alone tolerates
this (encoder: 0.00 dB, decoder: 0.09 dB, mask: 0.00 dB drop), but combined across 399 recurrent
steps the errors cascade catastrophically.

### Attempt v3 (weight-only) — SUCCEEDED

Strategy: quantize all Conv/Gemm/ConvTranspose weights to INT8, activations stay FP32.

```bash
python export/quantize_int8.py \
    --onnx_path export/dpsnn_scnn128.onnx \
    --output_path export/dpsnn_scnn128_int8_weightonly.onnx \
    --weight_only --no_snapshot
```

- Actual weight bytes: **140.9 KB** (58.8 KB INT8 + 44.5 KB FP32 biases + 37.6 KB scale params)
- Flash: 278.5 KB → 140.9 KB (~2× compression)

| Metric | Noisy | FP32 | Weight-only INT8 | Drop |
|---|---|---|---|---|
| SI-SNR (dB) | 8.44 | 17.23 | **16.92** | −0.31 dB |
| PESQ (wb) | 1.971 | 2.089 | 1.757 | −0.332 |
| STOI | 0.921 | 0.920 | 0.916 | −0.004 |
| Composite OVRL | 2.637 | 2.480 | 1.825 | −0.655 |
| Composite SIG | 3.357 | 2.935 | 2.007 | −0.928 |
| Composite BAK | 2.445 | 2.909 | 2.655 | −0.254 |

SI-SNR preserved (−0.31 dB) but perceptual metrics degrade more (PESQ −0.332, SIG −0.928),
indicating subtle spectral artifacts that SI-SNR misses.

### Attempt v3-correct (custom QDQ placement) — `export/quantize_spike_aware_correct.py`

Implements the supervisor's rule exactly by directly manipulating the ONNX graph:
1. Identifies ReLU/Sigmoid nodes that follow SAFE Conv nodes (BFS, one-hop-through-Add)
2. Calibrates the **ReLU/Sigmoid output tensors** (not Conv outputs)
3. Inserts Q→DQ **after** the activation: `Conv → ReLU → [Q→DQ]`
4. Quantizes all 6 unique Conv weight initializers to INT8 per-channel

Output: 802 Q→DQ pairs (401 encoder ReLU + 401 mask Sigmoid), 6 weight tensors quantized.

```bash
python export/quantize_spike_aware_correct.py \
    --onnx_path export/dpsnn_scnn128.onnx \
    --spike_map export/dpsnn_scnn128.onnx.spike_map.json \
    --hdf5_path data/results/save/test.hdf5 \
    --output_path export/dpsnn_scnn128_int8_v3_correct.onnx \
    --n_calib 50
```

| Metric | Noisy | FP32 | Weight-only | v3-correct | v3 Drop |
|---|---|---|---|---|---|
| SI-SNR (dB) | 8.44 | 17.23 | 16.92 | **15.73** | −1.50 dB |
| PESQ (wb) | 1.971 | 2.089 | 1.757 | **1.801** | −0.288 |
| STOI | 0.921 | 0.920 | 0.916 | 0.916 | −0.004 |
| Composite OVRL | 2.637 | 2.480 | 1.825 | **2.011** | −0.469 |
| Composite SIG | 3.357 | 2.935 | 2.007 | **2.311** | −0.624 |
| Composite BAK | 2.445 | 2.909 | 2.655 | **2.689** | −0.220 |

**v3-correct tradeoff:**
- Worse SI-SNR (−1.50 dB vs −0.31 dB): each of the 401 unrolled encoder time steps has
  its own independently calibrated scale. Real hardware (X-CUBE-AI) uses a single scale per
  layer — per-step calibration introduces variance that inflates the full-eval drop.
  The 20-sample quick check showed only −0.10 dB, confirming it is a generalisation problem.
- Better perceptual metrics (PESQ +0.044, SIG +0.304): INT8 clipping after ReLU/Sigmoid
  acts as a regulariser, reducing spectral artifacts that PESQ/composite penalize.
- STOI unchanged: intelligibility preserved in both approaches.

**Known fix not yet implemented:** use a single shared scale across all 401 encoder ReLU
outputs (global min/max over all time steps × calibration samples), and a fixed scale
for mask Sigmoid (output always in [0,1], so scale=1/255). This would match real hardware
behavior and is expected to significantly reduce the SI-SNR gap.

---

## Summary — All Quantization Results

| Model | SI-SNR | PESQ | Flash (weights) | Notes |
|---|---|---|---|---|
| FP32 ONNX | 17.23 dB | 2.089 | 278.5 KB | Baseline |
| Standard INT8 (ORT) | ~7–8 dB | ~1.3 | — | Catastrophic — membrane potential quantized |
| **Weight-only INT8** | **16.92 dB** | 1.757 | **140.9 KB** | **Recommended for deployment** |
| v3-correct INT8 | 15.73 dB | **1.801** | 140.9 KB | Better perceptual; SI-SNR fixable with shared scale |

**Thesis contribution:** Standard PTQ tools (onnxruntime `quantize_static`) incorrectly place
activation QDQ between Conv and ReLU, quantizing the partial sum (membrane potential equivalent).
This destroys SNN output quality. Weight-only INT8 is the robust deployment solution.
The correct activation INT8 (v3-correct) is theoretically sound but needs shared-scale calibration.

---

## Footprint — SCNN-only N=128 on STM32 B-U585I-IOT02A

| Resource | FP32 | Weight-only INT8 | Limit |
|---|---|---|---|
| Flash (weights) | 278.5 KB | **140.9 KB** | 2048 KB |
| RAM (I/O tensors) | 125.6 KB | 125.6 KB | 786 KB |
| Peak RAM (w/ intermediates) | TBD | TBD | 786 KB — needs X-CUBE-AI Analyse |
