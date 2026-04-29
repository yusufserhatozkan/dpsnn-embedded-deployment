# Experiment Log

Full records are split across three files. Open the relevant one for details.

| File | Contents |
|---|---|
| [log_baseline.md](log_baseline.md) | Pretrained N=256 inference, ONNX export, numerical validation, pipeline decision |
| [log_training.md](log_training.md) | SCNN-only N=128 model design, both training runs, final test-set evaluation |
| [log_quantization.md](log_quantization.md) | All INT8 experiments, root cause analysis, footprint numbers |

---

## Experiment Outcomes at a Glance

| Experiment | What | Outcome |
|---|---|---|
| 0 | Pretrained N=256 inference | 18.08 dB SI-SNR, PESQ 2.264 — desktop ceiling |
| 1 | ONNX export of pretrained model | PASS, diff 4.01e-05; two upstream bugs fixed |
| 2 | ONNX numerical validation | PASS < 1e-3 |
| 1b | Standard INT8 on pretrained N=256 | FAILED — 6.53 dB (worse than noisy input) |
| 1c | Extended INT8 strategies on pretrained N=256 | Weight-only INT8 viable (−0.30 dB); all activation INT8 fails |
| 3 | SCNN-only N=128 first run (frame_dur=0.5) | DISCARDED — 9.52 dB ceiling; root cause: too few time steps |
| 4 | SCNN-only ONNX export of discarded checkpoint | PASS — confirms export pipeline works for SCNN-only |
| 5 | SCNN-only N=128 retrain (frame_dur=1.0, gradient clipping) | **17.60 dB val / 17.23 dB test** — deployment model |
| 6 | INT8 quantization of SCNN-only N=128 | v4 shared-scale: **16.94 dB, 140.9 KB Flash** — recommended |

---

## Full Quantization Comparison (824-utterance VoiceBank-DEMAND test set)

All numbers are for the **SCNN-only N=128** model. Noisy and FP32 rows are the reference.
Flash = weight bytes only (not ONNX file size).

| Model | Approach | SI-SNR (dB) | PESQ (wb) | STOI | OVRL | SIG | BAK | Flash |
|---|---|---|---|---|---|---|---|---|
| Noisy input | — | 8.44 | 1.971 | 0.921 | 2.637 | 3.357 | 2.445 | — |
| FP32 ONNX | baseline | 17.23 | 2.089 | 0.920 | 2.480 | 2.935 | 2.909 | 278.5 KB |
| Naive INT8 v1 | ORT quantize_static, spike_map exclusion only | 7.64 | 1.328 | 0.846 | 1.341 | 1.456 | 2.197 | — |
| Naive INT8 v2 | ORT quantize_static, op_types restricted | 7.92 | 1.309 | 0.840 | 1.340 | 1.490 | 2.167 | — |
| Weight-only INT8 | All weights INT8, all activations FP32 | 16.92 | 1.757 | 0.916 | 1.825 | 2.007 | 2.655 | 140.9 KB |
| v3 corrected | QDQ after ReLU/Sigmoid, per-step scale | 15.73 | 1.801 | 0.916 | 2.011 | 2.311 | 2.689 | 140.9 KB |
| **v4 corrected** | **QDQ after ReLU/Sigmoid, shared scale** | **16.94** | **1.759** | **0.915** | **1.928** | **2.180** | **2.700** | **140.9 KB** |

**Drop vs FP32 (v4 recommended model):**

| SI-SNR | PESQ | STOI | OVRL | SIG | BAK |
|---|---|---|---|---|---|
| −0.29 dB | −0.330 | −0.005 | −0.552 | −0.755 | −0.209 |
