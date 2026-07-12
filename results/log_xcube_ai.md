# X-CUBE-AI Deployment Log

Companion to `log_quantization.md` and `experiment_log.md`. This file tracks our
end-to-end experience getting the DPSNN SCNN-only N=128 (Exp 7, 17.41 dB test
SI-SNR) model through the ST Edge AI toolchain (X-CUBE-AI v10.2.0,
ST Edge AI Core v2.2.0-20266) and onto the STM32U585 (B-U585I-IOT02A).

> **Single source of truth.** Every analyze/generate/validate attempt, every
> error, every flag we tried — log it here. Future me / supervisor should be
> able to reconstruct what was tried without re-running 3-hour analyses.

---

## 0. Target hardware budget

- MCU: STM32U585AIIxQ (Cortex-M33 @ 160 MHz)
- Internal SRAM: 786 KB total (this is the budget for activations)
- Internal Flash: 2 MB (room for weights + spillover candidates)
- External PSRAM: present on B-U585I-IOT02A (8 MB Hyperbus) but **forbidden**
  by user constraint until everything else fails.

---

## 1. Toolchain layout

- Installed via STM32CubeMX → X-CUBE-AI pack v10.2.0 at
  `~/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/`
- CLI binary: `Utilities/windows/stedgeai.exe` (we invoke this directly; the
  STM32CubeMX GUI crashes on this graph).
- Bundled Python 3.9 + TensorFlow + ONNXRuntime in
  `Utilities/windows/Lib/site-packages/` (handles QDQ ONNX, .tflite, .h5).
- Workspaces:
  - Input ONNX files: `C:/ai/`
  - stedgeai workspace (intermediates): `~/AppData/Local/Temp/mxAI_<name>/`
  - Output reports/headers: `~/.stm32cubemx/<name>/`

### Useful CLI flags discovered (from `stedgeai analyze --help`)

| Flag | Meaning | Status |
|------|---------|--------|
| `--optimization balanced\|time\|ram` | global objective | tried `balanced`; `ram` queued |
| `--compression none\|lossless\|low\|medium\|high` | weight compression | tried `none` and `high` (weights only — irrelevant to our RAM issue) |
| `--memory-pool FILE` | multi-heap activation placement | NOT yet tried — candidate for spilling to internal Flash |
| `--split-weights` | C-array per tensor (helps linker placement) | NOT yet tried |
| `--cut-input-tensors / --cut-output-tensors` | sub-graph carving | NOT yet tried — enables chunked inference |
| `--allocate-activations / --allocate-states` | multi-heap controls | NOT yet tried |
| `--no-onnx-optimizer` / `--use-onnx-simplifier` | preprocessor knobs | leaving defaults so we don't trigger new errors |
| `--verbosity 3` | verbose per-op log | useful for catching unsupported-op fallbacks |
| `--quantize [FILE]` | tensor-format JSON, NOT calibration PTQ | confirmed misleading — does NOT replace our spike-aware quantizer |

---

## 2. Model pedigree

Throughout this whole effort we have been using **Experiment 7** SCNN-only
(N=B=H=128, 71,299 trained params, 17.41 dB test SI-SNR). Earlier Exp 5 v4
numbers cited in `log_quantization.md` are historical — they are NOT what is
flowing through stedgeai.

Files in `C:/ai/` that matter:

| File | Source | Role |
|------|--------|------|
| `dpsnn_fp32_final_v3.onnx` | Exp 7 FP32 → 10-step pipeline | FP32 reference for stedgeai |
| `dpsnn_int8_final_v3.onnx` | Exp 7 spike-aware INT8 (post-act QDQ) → 10-step pipeline | "v3" INT8, the supervisor-style quantizer |
| `dpsnn_true_int8_final.onnx` | Exp 7 pre-act QDQ → 10-step pipeline | New "true-INT8" variant (this log focuses on it) |

The 10-step pipeline lives in `export/export_to_onnx.py::postprocess_for_xcubeai()`.
See `experiment_log.md` for the 9 rounds of compatibility fixes baked into it.

Evaluation on 824-sample VoiceBank-DEMAND test set:
- FP32 reference: **17.41 dB**
- post-act INT8 (`dpsnn_int8_final_v3`): **~16.7 dB**
- pre-act true-INT8 (`dpsnn_true_int8_final`): **16.71 dB** (-0.70 dB) — see
  `results/eval_exp7_true_int8.txt`

---

## 3. Analyze runs done so far

| Run | Model | Flags | Wall-time | weights | act (RAM) | Notes |
|-----|-------|-------|-----------|---------|-----------|-------|
| 1 | `dpsnn_fp32_final_v3.onnx` | `-O balanced -c none` | ~3 h | 87.3 MB | 1.47 MB | Passes 167/167. RAM dominated by FP32 OLA chain. |
| 2 | `dpsnn_int8_final_v3.onnx` | `-O balanced -c none` | ~3 h | 282 KiB | 1.47 MB | Weights drop 99.7%; RAM nearly unchanged. |
| 3 | `dpsnn_int8_final_v3.onnx` | `-O ram -c high` | ~3 h | 282 KiB | **1.47 MB** | `-c high` only affects weights; `-O ram` saved <1% on this graph. |
| 4 | `dpsnn_true_int8_final.onnx` | `-O balanced -c none` | 3 h 15 min | 282 KiB | **1.37 MB** | Pre-activation QDQ saved ~100 KB. Still 2× the 786 KB budget. |
| 5 | `dpsnn_true_int8_final.onnx` | `-O ram -c none` | 4 h 19 min | 282 KiB | **1.37 MB** | **Identical** to `-O balanced`. Confirms planner is already optimal; bottleneck is graph structure. |
| 6 | `dpsnn_fp32_fused_ola.onnx` (Solution A) | `-O balanced -c none` | 4 h 21 min | 410 KiB | **8.30 MB** | **Worse than per-frame.** X-CUBE-AI implements ConvTranspose1d as upsample-then-conv; the upsample tensor (1, 128, ~15921) is ~8 MB and unsplittable. |
| 7 | `dpsnn_exp9_bn_nooverlap.onnx` (Exp9 FP32) | `-O balanced -c none` | ~36 min | 280 KiB | **501 KiB ✓** | **FITS!** BN+stride=kernel=80 eliminates OLA chain. 285 KiB headroom. |
| 8 | `dpsnn_exp9_bn_nooverlap_true_int8.onnx` (Exp9 INT8) | `-O balanced -c none` | ~37 min | 280 KiB | **501 KiB ✓** | Spike-path FP32 islands remain → nearly identical to FP32. Both fit. |

Report files live under `~/.stm32cubemx/<name>/<name>_analyze_report.txt`.
Run-1..4 logs are `C:/ai/analyze_*.log`.

---

## 4. Root-cause analysis of the 1.37 MB floor

Source: `~/.stm32cubemx/dpsnn_true_int8/dpsnn_true_int8_analyze_report.txt`
(18,037 C-array entries, 164,217 lines).

### 4.1 Tensor size class histogram (activations pool only)

| Count | Items each | Bytes each | Role |
|------:|-----------:|-----------:|------|
| 11,602 | 128 | 512 B | per-frame 128-channel SCNN state vectors (transient) |
| 1,196 | 16,000 | **64,000 B** | **OLA buffers** (Pad/Add `_model_(Pad\|Add)_*_output_0`) |
| 1,208 | 640 | 2,560 B | Pad helpers / small intermediates |
| 2,003 | 80 | 320 B | per-frame slice windows |
| 2,015 | 1 | 4 B | scalar membrane potentials, thresholds |
| 1 | 16,160 | 64,640 B | input audio |

The dominant class by total volume is the **1,196 × 64 KB OLA tensors**. Even
after stedgeai's memory planner reuses addresses, ~22 of these slots are
simultaneously live → **~1.4 MB peak**.

### 4.2 Where the 1.37 MB comes from at peak

Looking at the per-op live-set bars in the report (right column = % of total RAM):

- `_model_mask_*_Conv_output_0`: peaks at 22.9% live = **~314 KB**
- `_model_srnn_readout_*_output_0`: peaks at 22.9% live = **~314 KB** (same address class, alternating)
- `_model_decoder_1d_*_ConvTr*_inflated_*`: peaks at 14.2% live = **~196 KB**
- Plus 2 × 64 KB OLA Add buffers live concurrently
- Plus encoder/LayerNorm FP32 island

Said differently: the planner *already* rotates the 64 KB OLA slot — without
that rotation it would be 1,196 × 64 KB = 76 MB. The floor we see is what
remains after maximal reuse.

### 4.3 Why INT8 quantization does not shrink RAM further

The encoder LayerNorm and the SPIKE-path membrane potentials (proj, dconv,
srnn_readout) are deliberately kept in FP32 to preserve accuracy. They form
FP32 islands that force adjacent tensors (the mask Conv output, the decoder
ConvTranspose output) to be FP32 too. INT8-tensor-axis savings only happen on
ops with INT8 in AND INT8 out — most of our heavy tensors don't qualify.

### 4.4 Decision

The bottleneck is **graph structure** (399-step unroll + LayerNorm island),
not tensor dtype. RAM-shrinking solutions must change the graph, not the
quantizer.

---

## 5. Solution brainstorming (within user constraints)

Constraints: keep INT8, keep input window = 1.01 s, keep N=B=H=128, no external PSRAM.

| # | Idea | Expected RAM impact | Effort | Accuracy risk |
|---|------|---------------------|--------|---------------|
| A | Replace 399-step unrolled OLA in `export_to_onnx.py` with a single `ConvTranspose1d(stride=hop)` | Probably big (collapses 1,196 × 64 KB class). Need confirmation. | 1–2 days | None (bit-equivalent) |
| B | Full per-frame streaming inference (one frame in/one frame out + state) | Largest — drops peak to ~one-frame working set | 3–5 days | Small (~0.1–0.3 dB) from boundary handling |
| C | Chunked inference (split into K subgraphs via `--cut-*-tensors`, MCU loops K times) | Linear in K | 2–3 days | None to small |
| D | `--memory-pool` mapping activations to internal Flash | Maybe 100–300 KB; Flash is slow → latency hit | 0.5 day to test | None |
| E | `--optimization ram` (untried on the `true_int8` graph) | Unknown; was useless on v3 graph | 0.5 day | None |
| F | INT8 LayerNorm bridge (collapse FP32 island) | Modest (frees a few tensors) | 1 day; needs op support check | Small |

### 5.1 Self-review of the brainstorm

- (A) depends on whether stedgeai's planner currently fails to reuse the OLA
  slot because of name/lifetime tracking. If it already reuses 22 of them
  down to ~1.4 MB, replacing with a single ConvTranspose1d would compress that
  to 1 buffer of ~64 KB → big save. Worst-case stedgeai allocates one full
  16,000-sample output (~64 KB) plus a per-frame work buffer; that still ≤
  budget by a wide margin. **Risk = low.**
- (B) is the canonical solution for streaming MCU speech enhancement. The
  PyTorch SCNN is already recurrent; the artificial unroll is a side effect of
  ONNX export. Risk = some boundary handling for Conv1d "same" padding and
  LayerNorm running stats.
- (C) is a stopgap of (B). If (A) lands us at ~600 KB but not 786 KB clean, (C)
  buys us another factor of 2 with little code.
- (D) is a fallback; running activations from internal Flash is slow but it's
  internal Flash, not PSRAM, so it doesn't violate the constraint.
- (E) is cheap to verify; run it in the background.
- (F) needs verifying v10.2 supports a quantized LayerNorm pattern. Likely
  marginal.

**Plan**: do (E) in background as a sanity check. Pursue (A) in foreground
(no retrain needed — pure ONNX export change). If (A) fits → ship. Else add
(C). (B) is the safety net if both fall short.

---

## 6. Next steps log

- [x] Read `dpsnn_true_int8_analyze_report.txt`, confirm OLA chain is the floor.
- [x] Tried `-O ram` on `dpsnn_true_int8_final.onnx` — no change (1.37 MB).
- [x] Tried fused ConvTranspose1d OLA (Solution A) — made things WORSE (8.30 MB).
- [x] Implemented streaming inference (Solution B) — **46 KB activations**. SUCCESS.
- [x] Trained Exp9 (BN + stride=kernel=80) — 16.49 dB SI-SNR, FP32 activations 501 KB. FITS.
- [x] Quantized Exp9 to true-INT8, INT8 analyze running.
- [ ] Record INT8 analyze result for Exp9, pick best deployment target (streaming vs Exp9 batch).
- [ ] Generate C code (`stedgeai generate`) for chosen model.
- [ ] Write MCU wrapper (streaming: 399× loop; Exp9 batch: single call).
- [ ] Commit all code changes.

---

## 7. Findings log (append as we go)

(Each entry: date, what we tried, what happened, file pointers.)

### 2026-05-14 — Root-cause confirmed

- Identified 1,196 OLA buffers of 64 KB each as the dominant tensor class.
- Memory planner already rotates the slot; 1.37 MB is the post-reuse floor.
- Confirms graph-structure change (solution A) is required; pure quantization
  cannot reach 786 KB.
- Report: `~/.stm32cubemx/dpsnn_true_int8/dpsnn_true_int8_analyze_report.txt`

### 2026-05-14 — Solution B (streaming inference): **46 KB activations — FITS**

Root of the problem: the full 399-frame graph can never fit in 786 KB SRAM because
the memory planner's floor is structurally determined by the unrolled OLA chain.
Solution: export a single-frame model with explicit recurrent state I/O.

**StreamingWrapper design** (`export/export_streaming.py`):
- Input: `frame (1,80)` + state tensors `context_win (1,128,4)`, `v_plif (1,128,1)`,
  `mem_readout (1,128)`, `ola_tail (1,40)` — total state = **3.2 KB**
- Output: `enhanced (1,40)` + updated state tensors
- MCU runs this 399× per utterance, feeds state_out as state_in

**PLIFNode math in streaming** (explicit state, no `self.v` mutation):
```
v_new  = v_plif + (dconv_out - (v_plif - v_reset)) * w.sigmoid()
spike  = heaviside(v_new - threshold)          # BOOL → eliminated by pipeline step 9
v_next = v_new * (1.0 - spike.detach())        # hard reset, no masked_fill needed
```

**ALIFNode (no_spiking=True) math**:
```
new_mem = mem * alpha.sigmoid() + (1 - alpha.sigmoid()) * R_m * dense(x)
```

**OLA in streaming** (stride=40, kernel=80):
```
enhanced   = ola_tail + decoded[:, :40]   # finalize previous overlap region
new_tail   = decoded[:, 40:]              # first half of this frame, pending next
```

**10-step pipeline changes**:
- `fix_output_shape` now only touches the first (main audio) output
- `onnxsim` now dynamically queries primary input name instead of hardcoding "noisy_audio"
- Both fixes in `export_to_onnx.py` and `tools/fix_shapes.py`

**stedgeai analyze result**:
- Analysis time: **~16 seconds** (vs 3-4 hours for full model — 44 nodes vs 17,236)
- weights: **285 KB** (Flash, fine)
- activations: **47,072 B = 46 KB** — **under 786 KB budget by 17×**
- MACC per frame: 886K (× 399 frames = 354M total, comparable to full model)
- Report: `~/.stm32cubemx/dpsnn_streaming/dpsnn_streaming_analyze_report.txt`

**Files produced**:
- `export/export_streaming.py` — StreamingWrapper + export script
- `export/dpsnn_streaming.onnx` (286 KB raw) + `export/dpsnn_streaming_xcubeai.onnx` (same)
- `C:/ai/dpsnn_streaming.onnx` — input for stedgeai

**Quality validation** (`evaluation/eval_streaming.py` on 824-sample VoiceBank-DEMAND test set):

| Metric | Noisy | Batch Exp 7 | Streaming Exp 7 |
|--------|-------|-------------|-----------------|
| SI-SNR (dB) | 8.44 | 17.41 | **17.42** |
| PESQ (wb) | 1.971 | 2.149 | **2.149** |
| STOI | 0.921 | 0.923 | **0.923** |

Zero quality loss — the streaming wrapper is numerically identical to the batch model
(max abs diff = 0.0 confirmed in unit test). Results in `results/eval_exp7_streaming.txt`.

**Warmup handling**: the batch model skips SCNN/decoder for the first `context_step=4`
frames. The streaming eval matches this by running those frames to build `context_win`
but resetting `v_plif`, `mem_readout`, and `ola_tail` to zeros afterwards. The MCU
wrapper must do the same (first 4 calls are warmup — output is valid from call 5).

---

### 2026-05-14 — Solution A (fused-OLA) implemented in `export_to_onnx.py`

- Added `enable_fused_ola(model)` helper that monkey-patches `forward()` to:
  - collect 399 per-frame mask outputs (b, N, 1) into a list,
  - stack them into a (b, N, 399) tensor,
  - run **one** `F.conv_transpose1d(weight=decoder.weight, bias=None,
    stride=hop)`, producing (b, 1, 16000) directly,
  - add a static `bias × M(p)` correction buffer where M(p)=1 at boundaries
    (positions [0,39] and [15960,15999]) and M(p)=2 in the overlap region.
    M is precomputed once, registered as a buffer, baked into the ONNX as a
    constant initializer.
- Trained `decoder_1d.bias = 0.040029` (Exp 7 epoch=87 checkpoint).
- Sanity check (`export/_test_fused_ola.py`):
  - reference per-frame OLA vs fused: **max-abs-diff = 1.14e-5**, mean = 3.3e-7.
  - PASS — within fp32 reduction-order tolerance.
- Re-exported via `python export/export_to_onnx.py --fused_ola ...`
  - Raw ONNX: `export/dpsnn_scnn128_exp7_fused.onnx` (3.79 MB)
  - Pipeline output: `export/dpsnn_scnn128_exp7_fused_xcubeai.onnx` (4.06 MB)
  - Copy for stedgeai: `C:/ai/dpsnn_fp32_fused_ola.onnx`
- 10-step pipeline diagnostics:
  - Step 8 reports **"Inflated 0 Pad and 1 ConvTranspose nodes"** vs
    previously **399 Pads + 1 ConvTranspose** — confirms the structural
    change is captured.
  - Final node count: 17,236 vs ~20,022 before (saved ~2.8K nodes from the
    OLA chain).
- Kicked off `stedgeai analyze ... -O balanced` on the fused graph in
  background (`bjeyjr8au` for `-O ram` on true_int8, `b89p4g9iw` for fused
  FP32 balanced). Logs at `C:/ai/analyze_true_int8_oram.log` and
  `C:/ai/analyze_fp32_fused.log`.

---

### 2026-05-15 — Exp9 (BN + no-overlap): **501 KiB FP32 activations — FITS**

**Training** (Variant B, `lightning_logs/version_4`):
- Command: `python vctk_trainer.py --scnn_only -L 80 --stride 80 -N 128 -B 128 -H 128 -X 1 --frame_dur 1.0 --norm_type bnorm --exp_name exp9_bn_nooverlap --batch_size 64 --max_epochs 100 --precision bf16-mixed`
- Best checkpoint: `epoch=99-val_loss=83.1804-val_sisnr=-16.8212.ckpt`
- Test SI-SNR: **16.486 dB** (success criterion ≥ 16.0 dB — PASSED; -0.93 dB vs Exp7)
- PESQ enhanced: 1.987 | STOI enhanced: 0.917
- time_steps: 200 (down from 399 — stride=kernel=80, no overlap)

**ONNX export** (`export/export_to_onnx.py`):
- Raw: `export/dpsnn_exp9_bn_nooverlap.onnx` (2230.2 KB)
- X-CUBE-AI ready: `export/dpsnn_exp9_bn_nooverlap_xcubeai.onnx` (2075.7 KB)
- Key pipeline diagnostics: **0 If nodes** (eliminated by stride=kernel — no unrolled conditional branches), 200 Pad nodes stripped of empty inputs, 7608 nodes after bool elimination
- Copy for stedgeai: `C:/ai/dpsnn_exp9_bn_nooverlap.onnx`

**stedgeai analyze** (Run 7, FP32, `-O balanced -c none`, ~36 min, 167/167 ops pass):
```
weights (ro)   :   286,732 B (280.01 KiB)  — Flash, fine
activations(rw):   512,832 B (500.81 KiB)  — RAM  ← FITS (786 KiB budget)
ram (total)    :   512,832 B (500.81 KiB)
macc           :   181,519,375
```
Report: `<repo>\st_ai_output\dpsnn_exp9_bn_nooverlap_analyze_report.txt`
Log: `C:/ai/analyze_exp9_bn_nooverlap.log`

**Why it fits:** stride=kernel=80 means each frame decodes to exactly 80 output samples with no overlap accumulation. The 1,196 × 64 KB OLA buffers that dominated Exp7 (floor: 1.37 MB) are completely absent. The remaining 501 KiB is the SCNN state traffic (128-channel per-frame tensors) plus encoder/decoder I/O.

**Quantization** (true-INT8 spike-aware):
- Spike map: `export/dpsnn_exp9_bn_nooverlap.onnx.spike_map.json` — 802 SAFE, 402 SPIKE nodes
- Shared Conv→ReLU scale: 0.1076, zp: -5
- Shared Conv→Sigmoid scale: 0.1642, zp: -15
- INT8 ONNX: `export/dpsnn_exp9_bn_nooverlap_true_int8.onnx` (2290.6 KB)
- X-CUBE-AI ready INT8: `export/dpsnn_exp9_bn_nooverlap_true_int8_xcubeai.onnx` (2189.6 KB)
- Copy for stedgeai: `C:/ai/dpsnn_exp9_bn_nooverlap_true_int8.onnx`

**stedgeai analyze** (Run 8, INT8, `-O balanced -c none`, ~37 min, 167/167 ops pass):
```
weights (ro)   :   286,732 B (280.01 KiB)  — Flash, fine
activations(rw):   512,640 B (500.62 KiB)  — RAM  ← FITS (786 KiB budget)
ram (total)    :   512,640 B (500.62 KiB)
macc           :   181,519,375
model_fmt      :   float  (spike-path FP32 islands prevent full INT8 propagation)
```
Report: `<repo>\st_ai_output\dpsnn_exp9_bn_nooverlap_int8_analyze_report.txt`
Log: `C:/ai/analyze_exp9_bn_nooverlap_int8.log`

**Why INT8 ≈ FP32 RAM:** The spike-aware quantizer intentionally keeps membrane potentials and spike comparator outputs in FP32. These FP32 islands (proj→dconv→neuron path) prevent X-CUBE-AI from propagating INT8 across most heavy activations, so `model_fmt` reports "float". The Conv→ReLU / Conv→Sigmoid QDQ pairs only quantize those specific tensors; the surrounding spike-path tensors stay FP32. Net effect: ~200 bytes saved vs FP32.

**Deployment decision:** Both Exp9 variants (FP32 and INT8) fit the 786 KiB budget with ~285 KiB headroom. Three viable deployment paths now exist:

| Option | Model | activations | Quality | MCU complexity |
|--------|-------|------------|---------|----------------|
| A | Exp9 FP32 batch | 501 KiB | 16.49 dB | single inference call |
| B | Exp9 INT8 batch | 501 KiB | ~16.2 dB (est.) | single inference call |
| C | Exp7 streaming INT8 | 46 KiB | 17.42 dB | 399× loop + state handoff |

Option A is the simplest deployment path. Option C gives best quality at lowest RAM but requires firmware loop. **Recommendation: start with Option A (Exp9 FP32 batch) for first firmware integration, then try C if latency budget allows.**

---

### 2026-05-24 — Generated C code review: PLIFNode and ALIFNode verified correct

Deployed model: Exp7 streaming FP32 (`dpsnn_streaming`), 43 nodes, 46 KB activations.
C file: `Stm_deployment/X-CUBE-AI/App/dpsnn_streaming.c` (2264 lines).

**PLIFNode verification** (layers 23-32):

| Layer | Name | Operation | Maps to |
|-------|------|-----------|---------|
| 23 | `_Sub_1` | `dconv_out - v_plif` | `(x - v)` |
| 24 | `_Mul_0` | `_Sub_1 × α` (ScaleBias, bias=0, scale=sigmoid(w)) | `(x-v) × α` |
| 25 | `_Add_0` | `v_plif + _Mul_0` | `v_new = v + (x-v)×α` ✅ |
| 26 | `_Sub_2` | `v_new - 1.0` (threshold constant) | `v_new - θ` |
| 27-28 | Sign → ReLU | spike ∈ {0,1} | BOOL-eliminated spike path ✅ |
| 30 | `_Sub_3` | `1.0 - spike` | hard-reset multiplier |
| 32 | Mul | `v_new × (1 - spike)` | `v_next` (hard reset) ✅ |

Hard reset formula `v_next = v_new × (1 - spike)` is correctly implemented — when spike=1, v_next=0; when spike=0, v_next=v_new. Identical to Python streaming wrapper.

**ALIFNode verification** (no_spiking=True, readout accumulator):
- Layer: `_Mul_2_output_0` ScaleBias (scale=`alpha_readout`, bias=0) applied to `mem_readout` input.
- Followed by: `(1 - alpha_readout) × R_m × dense(x)` branch, then Add.
- Result: `new_mem = alpha × mem + (1-alpha) × R_m × dense(x)` ✅

**OLA verification**: `enhanced = ola_tail + decoded[:40]`, `new_tail = decoded[40:]` — correct slice-and-add pattern confirmed.

**Context window**: concat `[context_win, proj_spike]` along frame axis, slice oldest column out. Confirmed.

Network hash from C header: `0xcaef2e2ed1259677faef1f5c5ec10c87`

---

### 2026-05-24 — syscalls.c fix: linker warnings miscounted as errors

**Problem**: STM32CubeIDE reported "Build Failed. 7 errors, 7 warnings" despite producing a valid ELF. The 7 "errors" were linker warnings from `--specs=nosys.specs` default stubs for `_close`, `_fstat`, `_getpid`, `_isatty`, `_kill`, `_lseek`, `_read`. CubeIDE counts all stderr output from arm-none-eabi-ld as errors.

**Root cause**: `syscalls.c` only had Keil/ARMCC stubs (`#if defined(__CC_ARM)`). The GCC branch had no definitions, so the linker fell back to nosys.specs stubs which emit warnings at link time.

**Fix**: Added GCC stubs block to `Stm_deployment/X-CUBE-AI/App/syscalls.c`:
```c
#if defined (__GNUC__) && !defined(__CC_ARM) && !defined(__ARMCC_VERSION)
#include <errno.h>
#include <sys/stat.h>
int _close(int fd)  { (void)fd; errno = ENOSYS; return -1; }
/* ... 6 more stubs ... */
#elif defined (__CC_ARM) || defined(__ARMCC_VERSION)
```

After fix: Build succeeded with 0 errors, 0 warnings. ELF: 362,160 B text + 11,064 B data + 76,160 B BSS = 449,384 B total (Flash: 373 KB / 2048 KB = 18%).

---

### 2026-05-24 — On-target UART validation: **PASS — all outputs match**

**Setup:**
- MCU flashed via STM32CubeProgrammer (drag-and-drop from Debug/Stm_deployment.elf)
- Firmware wired to `aiValidation` mode — X-CUBE-AI protobuf UART harness on UART4 @ 115200 baud, COM3
- Command: `stedgeai validate --model C:\ai\dpsnn_streaming.onnx --target stm32 --mode target --desc serial:COM3 --name dpsnn_streaming`
- stedgeai version: ST Edge AI Core v2.2.0 / ST.AI runtime lib v10.1.0

**Troubleshooting log** (for future reference):
1. `--onnx` flag → `error: --model is required` — fixed: use `--model`
2. `--desc COM3` → `invalid/unsupported "lib:COM3" descriptor` — fixed: use `--desc serial:COM3`
3. `PermissionError(13) on COM3` → CubeIDE had port locked — fixed: close CubeIDE entirely before running
4. `Unable to bind ST.AI runtime with "network"` → default c_name mismatch — fixed: add `--name dpsnn_streaming`

**Device confirmed**: STM32U575/585 @ 160/160 MHz, FPU+ICache, UART4

**Per-frame latency** (10 random samples, 399 frames each):

| Metric | Value |
|--------|-------|
| Mean latency | 6.185 ms / frame |
| Min / Max | 6.181 / 6.187 ms |
| Std | 0.002 ms |
| CPU cycles | 989,572 cycles / frame |
| Cycles/MACC | 1.12 |
| Total per 1-second utterance | 399 × 6.185 ms = **2.468 s** |
| Real-Time Factor (RTF) | **2.47×** (not real-time; MCU needs 2.47 s per 1 s audio) |

**Cross-accuracy (all 5 tensor outputs)**:

| Output tensor | Shape | RMSE | NSE | cos |
|---------------|-------|------|-----|-----|
| enhanced | (40,) | **3.481e-6** | 1.000000 | 1.000000 |
| new_v_plif | (128,1) | 1.2e-7 | 1.000000 | 1.000000 |
| new_mem_readout | (128,) | 8.4e-8 | 1.000000 | 1.000000 |
| new_context_win | (128,4) | 0.0 | 1.000000 | 1.000000 |
| new_ola_tail | (40,) | 2.1e-7 | 1.000000 | 1.000000 |

All errors are pure FP32/FP64 rounding (MCU uses FP32; PC ONNX reference uses FP64 internally). No systematic error → **C code for PLIFNode/ALIFNode/OLA is mathematically correct**.

Network hash confirmed: `0xcaef2e2ed1259677faef1f5c5ec10c87`

Full report: `~\st_ai_output\dpsnn_streaming_val_io.npz` + `dpsnn_streaming_validate_report.txt`

**Deployment status: COMPLETE ✅**

---

## 8. Supervisor + Examiner Meeting (2026-05-24)

### Key findings from meeting

**Hardware validation is the critical open risk.**

1. **Silent C code failure risk** — X-CUBE-AI generates C code for PLIF and ALIF spiking
   neurons automatically without errors, but the generated logic may be mathematically
   incorrect (e.g., no spike reset, wrong membrane update formula). This is a "silent
   failure" — the model compiles and runs but produces wrong audio.
   - **Action item:** Manually inspect the generated `.c` files for the neuron ops.
     Key sections to check: PLIFNode update rule, spike detection, hard reset logic,
     ALIFNode (no_spiking) accumulation.
   - Files to inspect: `st_ai_ws/generated_dpsnn_exp9_bn_nooverlap/dpsnn_exp9_bn_nooverlap.c`

2. **UART validation required** — Because output is a continuous audio stream, the only
   reliable way to verify MCU math matches PyTorch is to pipe the MCU's output back to
   PC via UART and compare sample-by-sample.
   - Validation protocol:
     a. Run a known noisy test utterance through the PyTorch model → save reference
        float32 output (40 or 80 samples per frame).
     b. Flash same utterance to MCU, run model, stream output via UART.
     c. Compare PC-side reference vs UART-received values. Expect small FP32 rounding;
        any systematic error → C code bug in a neuron layer.

3. **Frame size vs total input clarified** — examiners confirmed our understanding:
   - L=80 = encoder kernel = one processing window = 5 ms (frame)
   - 1 second = 16,000 samples = 399 overlapping frames (stride=40) for Exp7
   - 1 second = 16,000 samples = 200 non-overlapping frames (stride=80) for Exp9
   - The streaming wrapper already handles this correctly.

---

## 10. Real audio end-to-end test (Option A, 2026-05-25)

### Motivation
`aiValidation` proved per-call numerical correctness (NSE=1.0, RMSE=3.48e-6) but used random zero-state inputs — not a real speech signal. To close the gap ("did the deployed model actually enhance speech?"), we implemented a firmware-side 399-frame streaming inference loop over a real utterance.

### Utterance: p232_009 (VoiceBank-DEMAND, female speaker) — FINAL
- Source file: `data/noisy_testset_wav_16k/p232_009.wav` (66,522 samples = 4.16 s, full utterance used)
- Input SI-SNR (noisy vs clean): **6.77 dB** (challenging noise condition)
- Python ONNX reference SI-SNR (enhanced vs clean): **15.66 dB** (+8.89 dB improvement)
- MCU SI-SNR: **15.66 dB**, delta = **0.00 dB** — PASS

*Note: p232_006 was attempted first but contained only the word "when" in the first second. Switched to p232_009 full utterance for a meaningful demo.*

*Note: model output has inherent gain ~20× (decoder scaling); WAVs are peak-normalised before saving. SI-SNR metric is scale-invariant so this does not affect results. Residual hush in enhanced audio is a model-level characteristic, not a deployment artifact — MCU and Python produce identical output.*

### Firmware design (`app_x-cube-ai.c` rewrite)
- `test_utterance.c/h`: 66,522 float32 samples = 266,088 B (~260 KiB) in `.rodata` Flash (fits in 2 MB Flash). The `.c` source text is ~919 KB, but the on-Flash data footprint is 260 KiB.
- 6 KB ping-pong state buffers: ctx_a/b (512f), vp_a/b (128f), mem_a/b (128f), tail_a/b (40f)
- Loop: N_FRAMES=1662 frames × stride=40; frame count read from `TEST_UTTERANCE_N_FRAMES` in header (auto-adapts to any utterance length)
- Each frame: 80-sample window → `ai_dpsnn_streaming_run` → 40 float32 enhanced samples streamed over USART1 @ 115200 baud
- Total UART transfer: 1662×160 = 265,920 bytes ≈ 23 s at 115200 baud
- Timing: DWT cycle counter, 160 MHz sysclk, reports ms/frame and RTF after loop

### PC receiver (`tools/mcu_audio_receiver.py`)
- Reads expected N_FRAMES from `deploy/reference_sisnr.txt` (auto-adapts to utterance length)
- Waits for `AUDIO_START\r\n` marker, receives N_FRAMES×160 bytes, parses timing + `AUDIO_END\r\n`
- Computes MCU SI-SNR vs clean reference, diffs against Python ONNX reference
- Saves peak-normalised WAV to `deploy/mcu_enhanced.wav` and `deploy/mcu_test_results.txt`
- Pass criterion: |MCU SI-SNR − Python ref SI-SNR| < 0.1 dB

### Final confirmed results (2026-05-25)

| Metric | Value |
|--------|-------|
| Utterance | p232_009 (4.16 s, VoiceBank-DEMAND) |
| Input SI-SNR | 6.77 dB |
| MCU enhanced SI-SNR | **15.66 dB** (+8.89 dB) |
| Python ONNX reference | 15.66 dB |
| MCU vs Python delta | **0.00 dB** |
| Per-frame latency | **6.147 ms** |
| RTF | **2.459×** |
| Activation SRAM | **46 KB** / 786 KB |
| aiValidation NSE | **1.0** (all 5 outputs) |

---

### Thesis deadlines (hard)

| Date | Milestone |
|------|-----------|
| 2026-05-26 | First thesis draft to supervisors |
| 2026-06-05 | Final thesis submission deadline |
| 2026-06-15 09:00 | Final defense |

---

## 9. Remaining deployment pipeline (updated 2026-05-24)

- [x] RAM constraint solved (streaming: 46 KB; Exp9 batch: 501 KB — both fit)
- [x] stedgeai analyze confirmed for both deployment options
- [x] stedgeai generate run for Exp7 streaming FP32 (C files in `Stm_deployment/X-CUBE-AI/App/`)
- [x] **Inspect generated C code** — PLIFNode and ALIFNode math verified correct (see §7 entry 2026-05-24 C code review)
- [x] **MCU wrapper** — used X-CUBE-AI `aiValidation` UART mode (wired in `app_x-cube-ai.c`; no custom wrapper needed)
- [x] **UART validation** — all 5 outputs NSE=1.0, cos=1.0, RMSE_enhanced=3.481e-6 (see §7 entry 2026-05-24 UART validation)
- [x] **Measure latency** — 6.185 ms/frame, 989,572 cycles/frame, RTF=2.47× (from validation report)
- [x] **Phase 1 complete** — `tools/wav_to_c_array.py` written; p232_006 utterance (81656→16000 samples) embedded as `test_utterance.c` (16,000 float32 = 64,000 B / ~62.5 KiB in `.rodata`; ~221 KB `.c` source text, in Stm_deployment/X-CUBE-AI/App/). Python ONNX reference SI-SNR = **20.50 dB** for this utterance. Reference binary saved to `deploy/test_enhanced_ref.bin`.
- [x] **Phase 2 complete** — `app_x-cube-ai.c` rewritten: replaced aiValidation with 399-frame streaming loop. Ping-pong state buffers (ctx_a/b, vp_a/b, mem_a/b, tail_a/b = ~6 KB). Frame-by-frame UART streaming (160 bytes/frame). DWT cycle counter for timing.
- [x] **Phase 3 complete** — `tools/mcu_audio_receiver.py` written: waits for AUDIO_START, receives 63840 bytes, computes MCU SI-SNR vs clean reference, compares to Python reference.
- [x] **Phase 4 COMPLETE (2026-05-25)** — Built, flashed, ran on full p232_009 (4.16 s, 1662 frames). MCU SI-SNR = **15.66 dB**, Python ref = 15.66 dB, delta = **0.00 dB** — **PASS**. Latency: 6.147 ms/frame, RTF = 2.459×. Audio output verified audibly. Fixed WAV clipping bug (peak-normalise enhanced output before saving).
- [x] **Deployment fully complete** — all claims evidenced: memory (46 KB), latency (6.1 ms/frame), quality preserved (0.00 dB delta vs Python ONNX reference), real speech enhanced on hardware.
- [ ] Start thesis writing (deadline: 2026-05-26 draft)
