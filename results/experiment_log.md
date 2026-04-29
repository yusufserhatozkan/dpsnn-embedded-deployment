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
| 6 | INT8 quantization of SCNN-only N=128 | Weight-only: **16.92 dB, 140.9 KB Flash** — recommended |

---

## Key Numbers for the Thesis

| Model | SI-SNR | PESQ | STOI | Flash (weights) |
|---|---|---|---|---|
| Noisy input | 8.44 dB | 1.971 | 0.921 | — |
| Pretrained N=256 (FP32) | 18.08 dB | 2.264 | 0.925 | 1,457 KB |
| SCNN-only N=128 (FP32) | 17.23 dB | 2.089 | 0.920 | 278.5 KB |
| SCNN-only N=128 (weight-only INT8) | **16.92 dB** | 1.757 | 0.916 | **140.9 KB** |
| Standard INT8 (any variant) | ~6–8 dB | ~1.3 | ~0.83 | — |
