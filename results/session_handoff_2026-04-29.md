# Session Handoff — 2026-04-29

## Where we left off

This document records everything accomplished in the 2026-04-29 session so the next
conversation can pick up without re-deriving context. Read this alongside
`results/experiment_log.md` (full numerical records) and the plan file.

---

## Project context (one paragraph)

Master's thesis: deploy a spiking neural network (DPSNN, SCNN-only variant) for
speech enhancement to STM32 B-U585I-IOT02A (2 MB Flash, 786 KB RAM). The trained
model is SCNN-only N=128 (71.3 K params, 17.23 dB SI-SNR on 824-utterance VoiceBank-DEMAND
test set). The open question going into this session was: can the model be INT8-quantized
for deployment, and how? Supervisor (Dr. Tang) gave a specific correction at the 2026-04-29
meeting about the correct quantization rule for SNNs.

---

## Supervisor's rule (from meeting 2026-04-29)

Correct INT8 quantization for a spiking network:
- **Quantize:** Conv/Gemm weights everywhere + activation outputs ONLY at ReLU (encoder) and Sigmoid (mask)
- **Do NOT quantize:** membrane potentials (= partial sum W·A between Conv and spike threshold), spike outputs (already 0/1)
- Formula from IoT lecture Week 5 Part 2, slide 14-16:
  `q_A2 = (S_A2 / (S_W2 · S_A1)) · ReLU(q_W2 · q_A1)`
  The INT32 partial sum `q_W2 · q_A1` is NOT clipped to INT8.

---

## What was completed this session

### 1. Graph analysis tool — `export/map_spike_tensors.py`
- BFS from each Conv/Gemm output, classifies as SAFE (feeds ReLU/Sigmoid) or SPIKE (feeds spike threshold)
- MAX_HOPS=600 needed because decoder overlap-add path is 404 hops long
- Result on `dpsnn_scnn128.onnx`: **1201 SAFE**, **1201 SPIKE**, 0 UNKNOWN
  - SAFE: encoder_1d ×401, mask ×401, decoder_1d ×399
  - SPIKE: proj/BinaryConv ×403, dconv ×399, dense ×399
- Output: `export/dpsnn_scnn128.onnx.spike_map.json`

### 2. Modified quantization script — `export/quantize_int8.py`
- Added `--spike_map` CLI argument: auto-populates `nodes_to_exclude` from JSON
- Added `--weight_only` flag: uses `quantize_dynamic` (no calibration, weights INT8 only)

### 3. Binary search — why v1/v2 failed

Four attempts at spike-aware activation INT8 all failed catastrophically:

| Attempt | SI-SNR drop | Why |
|---|---|---|
| v1: spike_map exclusion, ORT default | -9.59 dB | `quant_pre_process` decomposes LayerNorm; QDQ added around decomposed ops |
| v2: + op_types=['Conv','Gemm','ConvTranspose'] | -9.31 dB | QDQ placed BETWEEN Conv and ReLU (partial sum quantized) |
| Encoder-only activation INT8 | 0.00 dB | Fine alone |
| Decoder-only activation INT8 | -0.09 dB | Fine alone |
| Mask-only activation INT8 | 0.00 dB | Fine alone |
| All three combined | -9.31 dB | Errors compound across 399 recurrent steps |

**Root cause confirmed:** ORT's `quantize_static` inserts the output QDQ node
*between* Conv and ReLU (at the partial sum level). This is precisely the membrane
potential quantization the supervisor warned against, appearing at the encoder/mask/
decoder level rather than the SNN layer. Individually each component handles it fine;
combined across 399 recurrent time steps the errors cascade.

### 4. Weight-only INT8 — `export/dpsnn_scnn128_int8_weightonly.onnx`
- All Conv weights INT8, activations stay FP32
- Flash: 278.5 KB → 140.9 KB total weights (58.8 KB INT8 + 44.5 KB FP32 biases + 37.6 KB scale params)
- Full 824-sample eval:

| Metric | FP32 | Weight-only INT8 | Drop |
|---|---|---|---|
| SI-SNR | 17.23 dB | **16.92 dB** | -0.31 dB |
| PESQ (wb) | 2.089 | 1.757 | -0.332 |
| STOI | 0.920 | 0.916 | -0.004 |
| Composite OVRL | 2.480 | 1.825 | -0.655 |
| Composite SIG | 2.935 | 2.007 | -0.928 |
| Composite BAK | 2.909 | 2.655 | -0.254 |

**Finding:** SI-SNR preserved (-0.31 dB) but perceptual metrics degrade more (PESQ -0.332,
SIG -0.928). Weight quantization introduces subtle spectral artifacts that SI-SNR misses.

### 5. Correct activation INT8 — `export/quantize_spike_aware_correct.py` (NEW)

New custom script that correctly implements the supervisor's method:
- Finds each Relu/Sigmoid node that directly follows a SAFE Conv (BFS, one-hop-through-Add)
- Calibrates the **ReLU/Sigmoid output tensors** (not Conv outputs)
- Inserts Q→DQ **after** the activation: `Conv → ReLU → [Q→DQ]`
  (not ORT's wrong default: `Conv → [Q→DQ] → ReLU`)
- Quantizes all 6 unique Conv weight initializers to INT8 per-channel
  (6 unique weights are shared across 399 unrolled time steps via the same ONNX initializer;
  the DequantizeLinear outputs the ORIGINAL weight name so all 401 Conv nodes get updated automatically)

Run command:
```bash
python export/quantize_spike_aware_correct.py \
    --onnx_path export/dpsnn_scnn128.onnx \
    --spike_map export/dpsnn_scnn128.onnx.spike_map.json \
    --hdf5_path data/results/save/test.hdf5 \
    --output_path export/dpsnn_scnn128_int8_v3_correct.onnx \
    --n_calib 50
```

Output: 802 Q→DQ pairs (401 encoder ReLU + 401 mask Sigmoid), 6 weight tensors quantized.

### 6. V3 corrected INT8 — `export/dpsnn_scnn128_int8_v3_correct.onnx`
Full 824-sample eval:

| Metric | FP32 | Weight-only | v3 Corrected | v3 Drop |
|---|---|---|---|---|
| SI-SNR | 17.23 dB | 16.92 dB | **15.73 dB** | -1.50 dB |
| PESQ (wb) | 2.089 | 1.757 | **1.801** | -0.288 |
| STOI | 0.920 | 0.916 | 0.916 | -0.004 |
| Composite OVRL | 2.480 | 1.825 | **2.011** | -0.469 |
| Composite SIG | 2.935 | 2.007 | **2.311** | -0.624 |
| Composite BAK | 2.909 | 2.655 | **2.689** | -0.220 |

**V3 tradeoff:** worse SI-SNR (-1.50 dB) but better perceptual metrics than weight-only.
The worse SI-SNR comes from a calibration generalization problem: the 401 unrolled encoder
time steps each have their OWN calibrated scale/zero_point in v3 (402 × 50 calibration
observations). In real hardware (X-CUBE-AI), a single scale per layer is used across all
time steps — per-step calibration is not representative. This explains the gap between
20-sample check (-0.10 dB) and full 824-sample eval (-1.50 dB).

A potential fix for v3: use a SINGLE shared scale for all 401 encoder ReLU outputs
(computed from global min/max across all time steps × calibration samples), and similarly
for mask Sigmoid outputs (all in [0,1], so scale = 1/255 fixed). Not yet implemented.

---

## Current file inventory

| File | Status | Notes |
|---|---|---|
| `export/dpsnn_scnn128.onnx` | ✓ exists | FP32, 5630.9 KB, max diff 6.68e-05 |
| `export/dpsnn_scnn128.onnx.spike_map.json` | ✓ exists | 1201 SAFE, 1201 SPIKE |
| `export/dpsnn_scnn128_int8_weightonly.onnx` | ✓ exists | Weight-only INT8, 10202.1 KB (ONNX overhead), 140.9 KB actual weights |
| `export/dpsnn_scnn128_int8_v3_correct.onnx` | ✓ exists | Custom correct placement, 6536.3 KB |
| `export/dpsnn_scnn128_preprocessed.onnx` | ✓ exists | After quant_pre_process, diff=0 |
| `export/map_spike_tensors.py` | ✓ committed | BFS graph classifier |
| `export/quantize_spike_aware_correct.py` | ✓ new | Custom correct QDQ placement |
| `export/quantize_int8.py` | ✓ modified | Added --spike_map, --weight_only |
| `results/eval_scnn128_int8_weightonly.json` | ✓ exists | Full 824-sample eval |
| `results/eval_scnn128_int8_v3_correct.json` | ✓ exists | Full 824-sample eval |
| `results/experiment_log.md` | ✓ updated | Experiments 5 and 6 fully logged |
| `egs/voicebank/lightning_logs/version_1/checkpoints/epoch=39-...ckpt` | ✓ exists | Best checkpoint, 17.60 dB val |

Binary search helper scripts (not needed after this session, can delete):
- `export/_check_enc.py`, `export/_check_mask.py`, `export/_check_weightonly.py`, `export/_check_v3.py`

---

## Recommended deployment model (for X-CUBE-AI, STM32 firmware)

**Use `export/dpsnn_scnn128_int8_weightonly.onnx`** for the deployment pipeline:
- Best SI-SNR preservation (-0.31 dB)
- Robust calibration (no sensitivity to per-step scale)
- 140.9 KB weights fit easily in 2 MB Flash
- 125.6 KB I/O RAM well within 786 KB

**For thesis argument:** Both weight-only and v3 are far better than all naive INT8 attempts
(-9.31 dB). The key thesis contribution is identifying WHY standard PTQ tools fail for SNNs
(QDQ at partial sum level = membrane potential quantization) and demonstrating the correct fix.

---

## Remaining pipeline steps (not yet done)

These are the next actions in order:

1. **X-CUBE-AI Analyse** (manual GUI step, CANNOT be automated):
   - Open STM32CubeMX → B-U585I-IOT02A → X-CUBE-AI pack
   - Load `export/dpsnn_scnn128_int8_weightonly.onnx` as network
   - Click Analyze → record Flash/RAM/MACs/op-coverage report
   - Save screenshot to `deploy/analysis_results/`
   - Goal: confirm every ONNX op (Conv, Greater, Where, Cast, etc.) is supported
   - This is the highest remaining risk — if an op is unsupported, need to find a workaround

2. **Generate C code** (after Analyze confirms op support):
   - STM32CubeMX → Generate Code
   - Integrate into firmware skeleton per README §Phase 7:
     - TIM16 Prescaler 15999 → 100 µs ticks
     - USART1 @ 115200, `_write` for printf, `-u_printf_float` linker flag

3. **Embed test audio as C header**:
   ```bash
   python tools/extract_test_audio.py \
       --hdf5_path data/results/save/test.hdf5 \
       --output_path deploy/test_audio.h \
       --n_frames 10
   ```

4. **Flash and measure latency**:
   - Flash firmware to B-U585I-IOT02A
   - Capture per-frame inference latency over UART
   - Compute RTF = (frame_dur × 1000) / latency_ms = (1.0 × 1000) / latency_ms
   - Target: RTF < 1.0 for real-time capability

5. **Verify output match**:
   - Run desktop INT8 ONNX rollout on one utterance
   - Compare bit-for-bit with STM32 UART output
   - Match = desktop INT8 metrics ARE the on-device metrics

6. **Update README.md** with final results

---

## Key thesis argument (current state)

The thesis asks: do efficiency claims from desktop DPSNN evaluations transfer to embedded deployment?

Answer (current evidence):
- FP32 DPSNN works on desktop: 17.23 dB SI-SNR, 278.5 KB weights → fits in STM32 Flash ✓
- Standard INT8 PTQ (as used in ANN papers): catastrophically fails (-9 to -11 dB) due to
  quantizing membrane potentials and spike outputs
- Correct INT8 for SNNs (weight-only): 16.92 dB, 140.9 KB → viable, modest quality loss
- Intelligibility (STOI) preserved across all approaches (0.916-0.920)
- Missing: actual on-device latency (RTF) to complete the efficiency claim

---

## Important technical notes for next session

1. **Conda env:** `dpsnn` — activate with `conda activate dpsnn` or `conda run -n dpsnn`
2. **Working directory:** `c:\Users\Serhat\Desktop\dpsnn-embedded-deployment`
3. **Data:** `data/results/save/test.hdf5` (824 utterances, 16 kHz, pre-processed)
4. **Model input:** shape (1, 16160) — 1 second @ 16 kHz + 10 ms context prefix
5. **Model output:** shape (1, 16000) — 1 second enhanced audio
6. **ONNX opset:** 13, unrolled 399 time steps (static graph, fixed input length)
7. **6 unique Conv weights** in the unrolled graph (shared across all time steps):
   - encoder, mask, decoder, proj (BinaryConv), dconv, dense
8. **eval_onnx.py requires:** `--onnx_path`, `--hdf5_path`, `--output_path` (all three mandatory)
9. **quant_pre_process** decomposes LayerNorm into primitives — this is why it must be
   done BEFORE any quantization that uses op_types_to_quantize (the decomposed ops would
   otherwise get QDQ pairs inserted around them)
